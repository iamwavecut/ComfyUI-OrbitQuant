from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


W4A4_COMPONENTS = frozenset({"transformer", "transformer_ref", "text_encoder"})
SOURCE_PRECISION_COMPONENTS = frozenset({"vae", "audio_vae"})
REQUIRED_SCRIPTS = {
    "generator": Path("scripts/run_quantized_example.py"),
    "decoder": Path("scripts/decode_h3_latents.py"),
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"MiniMax H3 {label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"MiniMax H3 {label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"MiniMax H3 {label} must contain a JSON object: {path}")
    return payload


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class MiniMaxH3Release:
    path: Path
    manifest: dict[str, Any]

    @classmethod
    def from_path(cls, model_path: str | Path) -> MiniMaxH3Release:
        if not str(model_path).strip():
            raise ValueError("MiniMax H3 model path must not be empty")
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"MiniMax H3 release directory does not exist: {path}")

        manifest = _read_json(path / "quantization_manifest.json", "manifest")
        if manifest.get("artifact_format") != "orbitquant-multicomponent-v1":
            raise ValueError(
                "MiniMax H3 manifest artifact_format must be "
                f"'orbitquant-multicomponent-v1', got {manifest.get('artifact_format')!r}"
            )
        if (manifest.get("weight_bits"), manifest.get("activation_bits")) != (4, 4):
            raise ValueError("MiniMax H3 release must use the W4A4 recipe")

        w4a4_components = set(manifest.get("w4a4_components", []))
        if w4a4_components != W4A4_COMPONENTS:
            raise ValueError(
                "MiniMax H3 manifest w4a4_components must be exactly "
                f"{sorted(W4A4_COMPONENTS)}, got {sorted(w4a4_components)}"
            )
        source_components = set(manifest.get("source_precision_components", []))
        if source_components != SOURCE_PRECISION_COMPONENTS:
            raise ValueError(
                "MiniMax H3 manifest source_precision_components must be exactly "
                f"{sorted(SOURCE_PRECISION_COMPONENTS)}, got {sorted(source_components)}"
            )

        for component in sorted(W4A4_COMPONENTS | SOURCE_PRECISION_COMPONENTS):
            if not (path / component).is_dir():
                raise ValueError(f"MiniMax H3 release component is missing: {component}")
        for label, relative_path in REQUIRED_SCRIPTS.items():
            if not (path / relative_path).is_file():
                raise ValueError(f"MiniMax H3 {label} script is missing: {relative_path}")

        return cls(path=path, manifest=manifest)

    @property
    def repo_id(self) -> str:
        return str(self.manifest.get("repo_id", "unknown"))

    @property
    def bits(self) -> str:
        return f"W{self.manifest['weight_bits']}A{self.manifest['activation_bits']}"

    @property
    def w4a4_components(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest["w4a4_components"]))

    @property
    def source_precision_components(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest["source_precision_components"]))

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "valid": True,
            "path": str(self.path),
            "repo_id": self.repo_id,
            "source_model_id": self.manifest.get("source_model_id", "unknown"),
            "source_revision": self.manifest.get("source_revision", "unknown"),
            "bits": self.bits,
            "w4a4_components": list(self.w4a4_components),
            "source_precision_components": list(self.source_precision_components),
            "offload_policy": "text_encoder_cuda_then_cpu_transformer_cpu_then_cuda",
            "vae_policy": "source_precision_fp32_decode",
        }


@dataclass(frozen=True)
class MiniMaxH3RunResult:
    output_path: Path
    latents_path: Path
    checkpoint_dir: Path
    generation_log_path: Path
    decode_log_path: Path
    generation_metrics_path: Path
    decode_metrics_path: Path
    report_path: Path
    generation_metrics: dict[str, Any]
    decode_metrics: dict[str, Any]

    @property
    def report(self) -> dict[str, Any]:
        return _read_json(self.report_path, "ComfyUI run report")


class MiniMaxH3Runner:
    def __init__(self, release: MiniMaxH3Release):
        self.release = release

    @staticmethod
    def _run_stage(command: list[str], *, stage: str, log_path: Path, cwd: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                subprocess.run(
                    command,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=os.environ.copy(),
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"MiniMax H3 {stage} failed with exit code {exc.returncode}; "
                    f"see {log_path}"
                ) from exc

    def run(
        self,
        *,
        output_dir: str | Path,
        filename: str,
        prompt: str,
        task: str,
        reference_path: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
    ) -> MiniMaxH3RunResult:
        if not prompt.strip():
            raise ValueError("MiniMax H3 prompt must not be empty")
        if task not in {"t2va", "ref2va"}:
            raise ValueError(f"MiniMax H3 task must be t2va or ref2va, got {task!r}")
        if width <= 0 or height <= 0:
            raise ValueError("MiniMax H3 width and height must be positive")
        if num_frames < 4:
            raise ValueError("MiniMax H3 num_frames must be at least 4")
        if steps < 2:
            raise ValueError("MiniMax H3 steps must include at least two sigma points")
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".mp4":
            raise ValueError("MiniMax H3 filename must be a plain .mp4 filename")

        reference: Path | None = None
        if task == "ref2va":
            if not reference_path.strip():
                raise ValueError("MiniMax H3 ref2va requires a reference image path")
            reference = Path(reference_path).expanduser().resolve()
            if not reference.is_file():
                raise ValueError(f"MiniMax H3 reference image does not exist: {reference}")

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / filename
        stem = output_path.with_suffix("")
        latents_path = stem.with_suffix(".latents.pt")
        checkpoint_dir = destination / f"{stem.name}.checkpoints"
        generation_log_path = stem.with_suffix(".generate.log")
        decode_log_path = stem.with_suffix(".decode.log")
        generation_metrics_path = output_path.with_suffix(".metrics.json")
        decode_metrics_path = output_path.with_suffix(".decode.metrics.json")
        report_path = output_path.with_suffix(".comfyui.json")

        generation_command = [
            sys.executable,
            str(self.release.path / REQUIRED_SCRIPTS["generator"]),
            "--release",
            str(self.release.path),
            "--output",
            str(output_path),
            "--save-latents",
            str(latents_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--prompt",
            prompt,
            "--task",
            task,
            "--seed",
            str(seed),
            "--width",
            str(width),
            "--height",
            str(height),
            "--num-frames",
            str(num_frames),
            "--steps",
            str(steps),
            "--manual-stage-offload",
            "--transformer-runtime-mode",
            "auto_fused",
        ]
        if reference is not None:
            generation_command.extend(["--reference", str(reference)])

        self._run_stage(
            generation_command,
            stage="generation",
            log_path=generation_log_path,
            cwd=self.release.path / "scripts",
        )
        if not latents_path.is_file():
            raise RuntimeError(f"MiniMax H3 generation did not persist latents: {latents_path}")
        generation_metrics = _read_json(generation_metrics_path, "generation metrics")

        decode_command = [
            sys.executable,
            str(self.release.path / REQUIRED_SCRIPTS["decoder"]),
            "--latents",
            str(latents_path),
            "--vae",
            str(self.release.path / "vae"),
            "--vae-dtype",
            "fp32",
            "--audio-vae",
            str(self.release.path / "audio_vae"),
            "--audio-vae-dtype",
            "fp32",
            "--output",
            str(output_path),
        ]
        self._run_stage(
            decode_command,
            stage="decode",
            log_path=decode_log_path,
            cwd=self.release.path / "scripts",
        )
        if not output_path.is_file():
            raise RuntimeError(f"MiniMax H3 decoder did not produce video: {output_path}")
        decode_metrics = _read_json(decode_metrics_path, "decode metrics")

        report = {
            "status": "pass",
            "release": self.release.summary,
            "task": task,
            "prompt": prompt,
            "reference_path": str(reference) if reference is not None else None,
            "seed": seed,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "steps": steps,
            "output_path": str(output_path),
            "latents_path": str(latents_path),
            "checkpoint_dir": str(checkpoint_dir),
            "generation_log_path": str(generation_log_path),
            "decode_log_path": str(decode_log_path),
            "generation_metrics": generation_metrics,
            "decode_metrics": decode_metrics,
        }
        _atomic_json_write(report_path, report)
        return MiniMaxH3RunResult(
            output_path=output_path,
            latents_path=latents_path,
            checkpoint_dir=checkpoint_dir,
            generation_log_path=generation_log_path,
            decode_log_path=decode_log_path,
            generation_metrics_path=generation_metrics_path,
            decode_metrics_path=decode_metrics_path,
            report_path=report_path,
            generation_metrics=generation_metrics,
            decode_metrics=decode_metrics,
        )


class OrbitQuantMiniMaxH3ReleaseLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("ORBITQUANT_H3_RELEASE", "STRING")
    RETURN_NAMES = ("release", "summary_json")
    FUNCTION = "load"
    CATEGORY = "OrbitQuant/MiniMax H3"

    def load(self, model_path: str):
        release = MiniMaxH3Release.from_path(model_path)
        return release, json.dumps(release.summary, indent=2, sort_keys=True)


class OrbitQuantMiniMaxH3GenerateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "release": ("ORBITQUANT_H3_RELEASE", {"forceInput": True}),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "task": (["t2va", "ref2va"], {"default": "t2va"}),
                "reference_path": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
                "width": ("INT", {"default": 608, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 32}),
                "num_frames": ("INT", {"default": 44, "min": 4, "max": 4096, "step": 4}),
                "steps": ("INT", {"default": 50, "min": 2, "max": 1000}),
                "filename_prefix": (
                    "STRING",
                    {"default": "orbitquant/minimax-h3", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report_json")
    FUNCTION = "generate"
    CATEGORY = "OrbitQuant/MiniMax H3"
    OUTPUT_NODE = True

    def run_for_comfy(
        self,
        release: MiniMaxH3Release,
        prompt: str,
        task: str,
        reference_path: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        filename_prefix: str,
    ) -> tuple[Any, str, dict[str, str]]:
        if not isinstance(release, MiniMaxH3Release):
            raise ValueError("release must come from OrbitQuant MiniMax H3 Release Loader")

        import folder_paths
        from comfy_api.latest import InputImpl

        output_root = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix,
                output_root,
                width,
                height,
            )
        )
        destination = Path(full_output_folder)
        destination.mkdir(parents=True, exist_ok=True)
        output_filename = f"{filename}_{counter:05}_.mp4"
        result = MiniMaxH3Runner(release).run(
            output_dir=destination,
            filename=output_filename,
            prompt=prompt,
            task=task,
            reference_path=reference_path,
            seed=int(seed),
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            steps=int(steps),
        )
        video = InputImpl.VideoFromFile(str(result.output_path))
        report_json = json.dumps(result.report, indent=2, sort_keys=True)
        preview = {
            "filename": output_filename,
            "subfolder": subfolder,
            "type": "output",
        }
        return video, report_json, preview

    def generate(
        self,
        release: MiniMaxH3Release,
        prompt: str,
        task: str,
        reference_path: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        filename_prefix: str,
    ) -> dict[str, object]:
        video, report_json, preview = self.run_for_comfy(
            release,
            prompt,
            task,
            reference_path,
            seed,
            width,
            height,
            num_frames,
            steps,
            filename_prefix,
        )
        return {
            "ui": {"images": [preview], "animated": (True,)},
            "result": (video, report_json),
        }
