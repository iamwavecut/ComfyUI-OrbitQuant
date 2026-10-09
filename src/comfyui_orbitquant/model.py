"""Resolve model revisions and validate requests before allocating GPU memory."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import CATALOG, TURBO_SIGMAS, Recipe


@dataclass(frozen=True)
class ModelHandle:
    recipe: Recipe
    location: str
    revision: str | None
    python_executable: str = ""

    def to_dict(self):
        return {**asdict(self), "recipe": self.recipe.to_dict()}

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "recipe": Recipe(**data["recipe"])})


def resolve_model(
    model: str,
    model_path: str = "",
    repo_id: str = "",
    revision: str = "main",
    python_executable: str = "",
) -> ModelHandle:
    if model not in CATALOG:
        raise ValueError(f"Unknown model recipe: {model}")
    recipe = CATALOG[model]
    if model_path.strip():
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Model directory does not exist: {path}")
        validate_layout(path, recipe.adapter)
        return ModelHandle(recipe, str(path), None, python_executable.strip())
    from huggingface_hub import HfApi

    location = repo_id.strip() or model
    info = HfApi().model_info(location, revision=revision.strip() or "main")
    return ModelHandle(recipe, location, info.sha, python_executable.strip())


def read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_layout(path: Path, adapter: str):
    if adapter == "kandinsky6":
        config = read_json(path / "config.json")
        encoder = read_json(path / "text_encoder/config.json")
        if config.get("_class_name") != "Kandinsky6Transformer3DModel":
            raise ValueError("Selected recipe requires a Kandinsky 6 transformer")
        if encoder.get("model_type") != "qwen2_5_vl":
            raise ValueError("Kandinsky release requires the Qwen2.5-VL text_encoder")
        read_json(path / "artifact_metadata.json")["source_model"]["revision"]
        quantized = [config, encoder]
    elif adapter.endswith("_component") or adapter == "ideogram4":
        index = read_json(path / "model_index.json")
        expected = {
            "flux_component": "flux",
            "flux2_component": "flux2",
            "wan_component": "wan",
            "z_image_component": "z_image",
            "ideogram4": "universal",
        }[adapter]
        if index.get("_class_name") != "OrbitQuantComponentArtifact":
            raise ValueError("Selected recipe requires an OrbitQuant component artifact")
        if index.get("target_policy") != expected:
            raise ValueError(f"Selected recipe requires target_policy={expected}")
        if adapter == "ideogram4" and index.get("source_model_id") != "fal/ideogram-v4-instant":
            raise ValueError("Selected recipe requires the Ideogram Instant source")
        return
    elif adapter == "minimax_h3":
        from .minimax_h3 import MiniMaxH3Release

        MiniMaxH3Release.from_path(path)
        return
    elif adapter == "yue2":
        if read_json(path / "config.json").get("model_type") != "yue2":
            raise ValueError("Selected recipe requires a YuE2 release")
        if not (path / "src/run.py").is_file():
            raise ValueError("YuE2 release is missing src/run.py")
        return
    elif adapter == "boogu":
        config = read_json(path / "transformer/config.json")
        if config.get("_class_name") != "BooguImageTransformer2DModel":
            raise ValueError("Selected recipe requires a Boogu transformer")
        quantized = [config]
    else:
        expected = {
            "flux2": "Flux2KleinPipeline",
            "krea2": "Krea2Pipeline",
            "qwen_image21": "QwenImage21Pipeline",
        }[adapter]
        if read_json(path / "model_index.json").get("_class_name") != expected:
            raise ValueError(f"Selected recipe requires {expected}")
        quantized = [
            read_json(path / f"{component}/config.json")
            for component in ("transformer", "text_encoder")
        ]
    for config in quantized:
        if config.get("quantization_config", {}).get("quant_method") != "orbitquant":
            raise ValueError("The selected component is not an OrbitQuant checkpoint")


def generation_options(
    handle: ModelHandle,
    media_type: str,
    *,
    prompt: str,
    seed: int,
    width: int = 0,
    height: int = 0,
    steps: int = 0,
    frames: int = 0,
    profile: str = "default",
    allocator_cap_gib: float = 0,
    references: list[str] | None = None,
    lyrics: str = "",
) -> dict:
    recipe = handle.recipe
    if recipe.media_type != media_type:
        raise ValueError(
            f"This model produces {recipe.media_type}; use the matching generator node"
        )
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if not 0 <= seed < 2**63:
        raise ValueError("Seed must be between 0 and 2**63 - 1")
    if any(value < 0 for value in (width, height, steps, frames)):
        raise ValueError(
            "Dimensions, steps and frames cannot be negative; zero uses model defaults"
        )
    width, height, steps, frames = (
        width or recipe.width,
        height or recipe.height,
        steps or recipe.steps,
        frames or recipe.frames,
    )
    multiple = 32 if recipe.adapter == "minimax_h3" else 16
    if media_type != "audio" and (width % multiple or height % multiple):
        raise ValueError(f"Width and height must be positive multiples of {multiple}")
    if recipe.adapter in {"kandinsky6", "wan_component"} and frames % 4 != 1:
        raise ValueError("Frame count must be 4n + 1, for example 121 or 81")
    if recipe.adapter == "minimax_h3" and not 120 <= frames <= 360:
        raise ValueError("MiniMax H3 requires 120 to 360 frames")
    if recipe.adapter == "minimax_h3" and steps < 2:
        raise ValueError("MiniMax H3 requires at least two sigma points")
    if recipe.adapter == "qwen_image21" and steps not in TURBO_SIGMAS:
        raise ValueError("Turbo Image supports 4, 5, 6, 7 or 8 steps")
    if recipe.adapter == "kandinsky6" and steps != 10:
        raise ValueError("The released Kandinsky distilled profiles use 10 steps")
    if references and not recipe.reference_images:
        raise ValueError(f"Reference images are not supported by the {recipe.adapter} adapter")
    if recipe.adapter == "minimax_h3" and len(references or []) > 1:
        raise ValueError("MiniMax H3 accepts one reference image")
    profiles = {
        "kandinsky6": ("fast", "low-memory", "exact"),
        "minimax_h3": ("balanced", "speed", "minimum_vram"),
        "yue2": ("exact", "fast", "lowmem", "turbo"),
    }.get(recipe.adapter, ("default",))
    profile = profiles[0] if profile == "default" else profile
    if profile not in profiles:
        raise ValueError(f"{recipe.adapter} profiles: {', '.join(profiles)}")
    if not math.isfinite(allocator_cap_gib) or allocator_cap_gib < 0:
        raise ValueError("Allocator cap must be finite and nonnegative")
    if allocator_cap_gib and recipe.adapter != "kandinsky6":
        raise ValueError("The allocator cap is available for Kandinsky profiles only")
    if recipe.adapter == "yue2" and not lyrics.strip():
        raise ValueError("YuE2 requires lyrics")
    return dict(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        frames=frames,
        profile=profile,
        allocator_cap_gib=allocator_cap_gib,
        references=references or [],
        lyrics=lyrics,
    )
