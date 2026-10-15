"""ComfyUI model selection and native IMAGE, VIDEO and AUDIO outputs."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .catalog import CATALOG
from .model import ModelHandle, generation_options, resolve_model
from .worker import RESULT_PREFIX

WORKER_SCRIPT = Path(__file__).with_name("worker.py")


class OrbitQuantModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(CATALOG),),
                "model_path": ("STRING", {"default": ""}),
                "repo_id": ("STRING", {"default": ""}),
                "revision": ("STRING", {"default": "main"}),
                "python_executable": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("ORBITQUANT_MODEL", "STRING")
    RETURN_NAMES = ("model", "summary_json")
    FUNCTION = "load"
    CATEGORY = "OrbitQuant"
    DESCRIPTION = (
        "Select a published model recipe. Leave model_path and repo_id empty to download it. "
        "Use repo_id for a compatible variant, or model_path for a local release. "
        "python_executable selects an existing model environment; no packages are installed."
    )

    def load(self, model, model_path="", repo_id="", revision="main", python_executable=""):
        handle = resolve_model(model, model_path, repo_id, revision, python_executable)
        return handle, json.dumps(handle.to_dict(), indent=2)


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def _worker_environment() -> dict:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    # Do not inject ComfyUI's package paths into a separate model environment.
    environment.pop("PYTHONPATH", None)
    return environment


def _log_tail(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])


class _Progress:
    """Forward the worker's progress.json to ComfyUI's progress bar."""

    def __init__(self, destination: Path, progress):
        self.path, self.progress, self.last = destination / "progress.json", progress, None

    def poll(self):
        if self.progress is None or not self.path.exists():
            return
        update = json.loads(self.path.read_text())
        current = (update["step"], update["total"])
        if current != self.last:
            self.progress(*current)
            self.last = current


class _ServingWorker:
    """One generation process per model environment that keeps its last model loaded."""

    def __init__(self, python_executable: str):
        self.process = subprocess.Popen(
            [
                python_executable or sys.executable,
                "-u",
                str(WORKER_SCRIPT),
                "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_worker_environment(),
            start_new_session=os.name == "posix",
        )
        self.results: queue.Queue = queue.Queue()
        self._log = None
        self._lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            if line.startswith(RESULT_PREFIX):
                self.results.put(json.loads(line[len(RESULT_PREFIX) :]))
                continue
            with self._lock:
                if self._log is not None:
                    self._log.write(line)
                    self._log.flush()
        self.results.put(None)

    def submit(self, request_path: Path, log_path: Path):
        with self._lock:
            if self._log is not None:
                self._log.close()
            self._log = log_path.open("a", encoding="utf-8")
        self.process.stdin.write(f"{request_path}\n")
        self.process.stdin.flush()

    def stop(self):
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        _stop_process(self.process)
        with self._lock:
            if self._log is not None:
                self._log.close()
                self._log = None


_SERVING: dict[str, _ServingWorker] = {}


def stop_serving():
    """Stop every kept worker and free the GPU memory its model holds."""
    while _SERVING:
        _SERVING.popitem()[1].stop()


atexit.register(stop_serving)


def _run_once(request_path: Path, log_path: Path, python_executable, interrupt_check, progress):
    command = [
        python_executable or sys.executable,
        "-u",
        str(WORKER_SCRIPT),
        str(request_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_worker_environment(),
            start_new_session=os.name == "posix",
        )
        last_status = time.monotonic()
        try:
            while process.poll() is None:
                interrupt_check()
                progress.poll()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.25)
                if time.monotonic() - last_status >= 60:
                    print(f"[OrbitQuant] Running PID {process.pid}; log: {log_path}", flush=True)
                    last_status = time.monotonic()
        except BaseException:
            _stop_process(process)
            raise
    if process.returncode:
        raise RuntimeError(
            f"OrbitQuant generation failed ({process.returncode}). Log: {log_path}\n"
            f"{_log_tail(log_path)}"
        )


def _run_kept(request_path: Path, log_path: Path, python_executable, interrupt_check, progress):
    key = python_executable or sys.executable
    worker = _SERVING.get(key)
    if worker is not None and worker.process.poll() is not None:
        _SERVING.pop(key).stop()
        worker = None
    if worker is None:
        stop_serving()  # one kept model at a time
        worker = _SERVING[key] = _ServingWorker(python_executable)
    worker.submit(request_path, log_path)
    last_status = time.monotonic()
    while True:
        try:
            interrupt_check()
        except BaseException:
            stop_serving()  # the interrupted generation is still running in the worker
            raise
        progress.poll()
        try:
            status = worker.results.get(timeout=0.25)
        except queue.Empty:
            if time.monotonic() - last_status >= 60:
                print(f"[OrbitQuant] Running PID {worker.process.pid}; log: {log_path}", flush=True)
                last_status = time.monotonic()
            continue
        if status is None:
            _SERVING.pop(key, None)
            worker.stop()
            raise RuntimeError(
                f"OrbitQuant worker exited ({worker.process.returncode}). Log: {log_path}\n"
                f"{_log_tail(log_path)}"
            )
        if status["request"] != str(request_path):
            continue
        if not status["ok"]:
            raise RuntimeError(
                f"OrbitQuant generation failed: {status['error']}. Log: {log_path}\n"
                f"{_log_tail(log_path)}"
            )
        return


def run_worker(
    request: dict, python_executable: str, interrupt_check, progress=None, keep_loaded=False
) -> dict:
    """Run one request in a worker process. ``keep_loaded`` reuses a process that keeps the
    model in memory for the next request; otherwise a kept worker is stopped first."""
    destination = Path(request["destination"])
    request_path = destination / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    log_path = destination / "generation.log"
    reporter = _Progress(destination, progress)
    if keep_loaded:
        _run_kept(request_path, log_path, python_executable, interrupt_check, reporter)
    else:
        stop_serving()
        _run_once(request_path, log_path, python_executable, interrupt_check, reporter)
    result_path = Path(request["result"])
    if not result_path.is_file():
        raise RuntimeError(f"OrbitQuant worker did not return a result. Log: {log_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _generate(
    model,
    media_type,
    prompt,
    seed,
    width=0,
    height=0,
    steps=0,
    frames=0,
    profile="default",
    allocator_cap_gib=0.0,
    reference_images=None,
    lyrics="",
    speakers=None,
    keep_loaded=False,
):
    if not isinstance(model, ModelHandle):
        raise ValueError("Connect OrbitQuant Model Loader to this generator")
    # Reject invalid requests before creating files, unloading models or starting a worker.
    options = generation_options(
        model,
        media_type,
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        frames=frames,
        profile=profile,
        allocator_cap_gib=allocator_cap_gib,
        references=["pending"] * len(reference_images) if reference_images is not None else [],
        lyrics=lyrics,
        speakers=[{"path": "pending", "text": text} for _, text in speakers or []],
    )
    import comfy.model_management as mm
    import folder_paths

    root = Path(folder_paths.get_output_directory()) / "orbitquant"
    root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"{model.recipe.adapter}-", dir=root))
    options["references"] = []
    if reference_images is not None:
        import numpy as np
        from PIL import Image

        for index, tensor in enumerate(reference_images):
            path = destination / f"reference-{index}.png"
            pixels = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels).convert("RGB").save(path)
            options["references"].append(str(path))
    options["speakers"] = []
    for index, (audio, text) in enumerate(speakers or []):
        path = destination / f"speaker-{index}.wav"
        _write_wav(path, audio)
        options["speakers"].append({"path": str(path), "text": text})
    mm.throw_exception_if_processing_interrupted()
    mm.unload_all_models()
    mm.soft_empty_cache()
    from comfy.utils import ProgressBar

    progress = ProgressBar(options["steps"] or 1)
    request = {
        "model": model.to_dict(),
        "options": options,
        "destination": str(destination),
        "result": str(destination / "result.json"),
    }
    result = run_worker(
        request,
        model.python_executable,
        mm.throw_exception_if_processing_interrupted,
        progress.update_absolute,
        keep_loaded=keep_loaded,
    )
    output = Path(result[media_type]).resolve()
    if not output.is_relative_to(destination.resolve()) or not output.is_file():
        raise RuntimeError("OrbitQuant worker did not produce the expected media file")
    return output, json.dumps(result, indent=2)


SPEAKER_SLOTS = 4


def _write_wav(path, audio):
    """Write a ComfyUI AUDIO value as 16-bit PCM WAV."""
    import wave

    samples = audio["waveform"][0].detach().cpu().float().clamp(-1, 1).numpy()
    pcm = (samples.T * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(int(audio["sample_rate"]))
        handle.writeframes(pcm.tobytes())


def _collect_speakers(slots):
    """Pair each connected speaker audio with the text spoken in it."""
    speakers = []
    for index in range(1, SPEAKER_SLOTS + 1):
        audio = slots.get(f"speaker_audio_{index}")
        if audio is not None:
            speakers.append((audio, slots.get(f"speaker_text_{index}", "")))
    return speakers


def _inputs(media_type):
    required = {
        "model": ("ORBITQUANT_MODEL", {"forceInput": True}),
        "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
        "seed": (
            "INT",
            {"default": 42, "min": 0, "max": 2**63 - 1, "control_after_generate": True},
        ),
        "steps": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 1000,
                "tooltip": "Zero uses the model's released step count.",
            },
        ),
    }
    if media_type != "audio":
        for dimension in ("width", "height"):
            required[dimension] = (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 16,
                    "tooltip": "Zero uses the model's released dimension.",
                },
            )
    if media_type == "video":
        required.update(
            frames=("INT", {"default": 0, "min": 0, "max": 1000}),
            profile=(
                ["default", "fast", "low-memory", "exact", "balanced", "speed", "minimum_vram"],
            ),
            allocator_cap_gib=("FLOAT", {"default": 0.0, "min": 0.0, "max": 192.0, "step": 0.25}),
        )
    if media_type == "audio":
        required.update(
            lyrics=("STRING", {"default": "", "multiline": True, "tooltip": "YuE2 song lyrics."}),
            frames=(
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Drama 3 frame limit. Zero uses 500.",
                },
            ),
            profile=(["default", "exact", "fast", "lowmem", "turbo", "eager", "cpu-offload"],),
        )
    result = {"required": required}
    if media_type == "image":
        result["optional"] = {
            "reference_images": ("IMAGE",),
            "profile": (
                ["default", "model-offload", "int8-attention"],
                {
                    "tooltip": "FLUX.3 Klein: model-offload keeps its components in host memory; "
                    "int8-attention is faster and changes the image slightly."
                },
            ),
            "keep_loaded": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Keep the model loaded in its worker for the next run. It holds GPU "
                    "memory until another OrbitQuant model, a video or audio run, or an "
                    "interrupt stops the worker.",
                },
            ),
        }
    elif media_type != "audio":
        result["optional"] = {"reference_images": ("IMAGE",)}
    else:
        optional = {}
        for index in range(1, SPEAKER_SLOTS + 1):
            optional[f"speaker_audio_{index}"] = (
                "AUDIO",
                {"tooltip": f"Drama 3 voice reference for <|speaker:{index - 1}|>."},
            )
            optional[f"speaker_text_{index}"] = (
                "STRING",
                {"default": "", "tooltip": "The words spoken in the matching reference audio."},
            )
        result["optional"] = optional
    return result


class OrbitQuantGenerateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return _inputs("image")

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report_json")
    FUNCTION = "generate"
    CATEGORY = "OrbitQuant"

    def generate(
        self,
        model,
        prompt,
        seed,
        steps=0,
        width=0,
        height=0,
        reference_images=None,
        profile="default",
        keep_loaded=True,
    ):
        path, report = _generate(
            model,
            "image",
            prompt,
            seed,
            width,
            height,
            steps,
            profile=profile,
            reference_images=reference_images,
            keep_loaded=keep_loaded,
        )
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels)[None], report


class OrbitQuantGenerateModelVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return _inputs("video")

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report_json")
    FUNCTION = "generate"
    CATEGORY = "OrbitQuant"
    OUTPUT_NODE = True

    def generate(
        self,
        model,
        prompt,
        seed,
        steps=0,
        width=0,
        height=0,
        frames=0,
        profile="default",
        allocator_cap_gib=0.0,
        reference_images=None,
    ):
        path, report = _generate(
            model,
            "video",
            prompt,
            seed,
            width,
            height,
            steps,
            frames,
            profile,
            allocator_cap_gib,
            reference_images,
        )
        import folder_paths
        from comfy_api.latest import InputImpl

        relative = path.relative_to(Path(folder_paths.get_output_directory()).resolve())
        preview = {"filename": path.name, "subfolder": str(relative.parent), "type": "output"}
        return {
            "ui": {"images": [preview], "animated": (True,)},
            "result": (InputImpl.VideoFromFile(str(path)), report),
        }


class OrbitQuantGenerateAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return _inputs("audio")

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report_json")
    FUNCTION = "generate"
    CATEGORY = "OrbitQuant"

    def generate(
        self, model, prompt, seed, lyrics="", steps=0, frames=0, profile="default", **slots
    ):
        path, report = _generate(
            model,
            "audio",
            prompt,
            seed,
            steps=steps,
            frames=frames,
            profile=profile,
            lyrics=lyrics,
            speakers=_collect_speakers(slots),
        )
        from comfy_extras.nodes_audio import load

        waveform, sample_rate = load(str(path))
        return {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate,
        }, report


MEDIA_NODE_CLASSES = {
    cls.__name__: cls
    for cls in (
        OrbitQuantModelLoader,
        OrbitQuantGenerateImage,
        OrbitQuantGenerateModelVideo,
        OrbitQuantGenerateAudio,
    )
}
MEDIA_NODE_NAMES = {
    "OrbitQuantModelLoader": "OrbitQuant Model Loader",
    "OrbitQuantGenerateImage": "OrbitQuant Generate Image",
    "OrbitQuantGenerateModelVideo": "OrbitQuant Generate Model Video",
    "OrbitQuantGenerateAudio": "OrbitQuant Generate Audio",
}
