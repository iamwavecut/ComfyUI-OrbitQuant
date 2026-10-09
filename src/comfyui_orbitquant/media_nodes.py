"""ComfyUI model selection and native IMAGE, VIDEO and AUDIO outputs."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .catalog import CATALOG
from .model import ModelHandle, generation_options, resolve_model


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


def run_worker(request: dict, python_executable: str, interrupt_check, progress=None) -> dict:
    destination = Path(request["destination"])
    request_path = destination / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    log_path = destination / "generation.log"
    command = [
        python_executable or sys.executable,
        "-u",
        str(Path(__file__).with_name("worker.py")),
        str(request_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    # Do not inject ComfyUI's package paths into a separate model environment.
    environment.pop("PYTHONPATH", None)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=os.name == "posix",
        )
        last_status = time.monotonic()
        last_progress = None
        try:
            while process.poll() is None:
                interrupt_check()
                progress_path = destination / "progress.json"
                if progress is not None and progress_path.exists():
                    update = json.loads(progress_path.read_text())
                    current_progress = (update["step"], update["total"])
                    if current_progress != last_progress:
                        progress(*current_progress)
                        last_progress = current_progress
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.25)
                if time.monotonic() - last_status >= 60:
                    print(f"[OrbitQuant] Running PID {process.pid}; log: {log_path}", flush=True)
                    last_status = time.monotonic()
        except BaseException:
            _stop_process(process)
            raise
    if process.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
        raise RuntimeError(
            f"OrbitQuant generation failed ({process.returncode}). Log: {log_path}\n{tail}"
        )
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
    mm.throw_exception_if_processing_interrupted()
    mm.unload_all_models()
    mm.soft_empty_cache()
    from comfy.utils import ProgressBar

    progress = ProgressBar(options["steps"])
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
    )
    output = Path(result[media_type]).resolve()
    if not output.is_relative_to(destination.resolve()) or not output.is_file():
        raise RuntimeError("OrbitQuant worker did not produce the expected media file")
    return output, json.dumps(result, indent=2)


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
            lyrics=("STRING", {"default": "", "multiline": True}),
            profile=(["default", "exact", "fast", "lowmem", "turbo"],),
        )
    result = {"required": required}
    if media_type != "audio":
        result["optional"] = {"reference_images": ("IMAGE",)}
    return result


class OrbitQuantGenerateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return _inputs("image")

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report_json")
    FUNCTION = "generate"
    CATEGORY = "OrbitQuant"

    def generate(self, model, prompt, seed, steps=0, width=0, height=0, reference_images=None):
        path, report = _generate(
            model, "image", prompt, seed, width, height, steps, reference_images=reference_images
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

    def generate(self, model, prompt, seed, lyrics, steps=0, profile="default"):
        path, report = _generate(
            model, "audio", prompt, seed, steps=steps, profile=profile, lyrics=lyrics
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
