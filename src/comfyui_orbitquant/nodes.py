from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orbitquant.artifacts import OrbitQuantManifest, validate_orbitquant_artifact
from orbitquant.pipeline import load_quantized_pipeline_component


def read_manifest(artifact_path: str | Path) -> OrbitQuantManifest:
    path = Path(artifact_path)
    return OrbitQuantManifest.from_dict(
        json.loads((path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )


def _manifest_payload(manifest: Any) -> dict[str, Any]:
    return {
        "source_model_id": manifest.source_model_id,
        "source_revision": getattr(manifest, "source_revision", "unknown"),
        "source_license": getattr(manifest, "source_license", "unknown"),
        "bits": f"W{manifest.weight_bits}A{manifest.activation_bits}",
        "target_policy": getattr(manifest, "target_policy", "unknown"),
        "runtime_mode": getattr(manifest, "runtime_mode", "unknown"),
        "activation_kernel_backend": getattr(manifest, "activation_kernel_backend", "auto"),
        "quantized_module_count": len(getattr(manifest, "quantized_modules", [])),
        "adaln_module_count": len(getattr(manifest, "adaln_modules", [])),
        "skipped_module_count": len(getattr(manifest, "skipped_modules", [])),
    }


def _summary_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Source: {payload['source_model_id']}",
            f"Revision: {payload['source_revision']}",
            f"Bits: {payload['bits']}",
            f"Policy: {payload['target_policy']}",
            f"Runtime: {payload['runtime_mode']}",
            f"Activation backend: {payload['activation_kernel_backend']}",
            f"Quantized modules: {payload['quantized_module_count']}",
            f"AdaLN INT4 modules: {payload['adaln_module_count']}",
            f"Skipped modules: {payload['skipped_module_count']}",
        ]
    )


class OrbitQuantArtifactInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artifact_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                )
            }
        }

    RETURN_TYPES = ("STRING", "ORBITQUANT_INFO")
    RETURN_NAMES = ("summary", "info")
    FUNCTION = "inspect"
    CATEGORY = "OrbitQuant"

    def inspect(self, artifact_path: str):
        if not artifact_path:
            raise ValueError("artifact_path must not be empty")
        path = Path(artifact_path)
        manifest = read_manifest(path)
        validation = validate_orbitquant_artifact(path)
        payload = _manifest_payload(manifest)
        payload.update(validation)
        return (_summary_text(payload), payload)


class OrbitQuantPipelineComponentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPELINE", {"forceInput": True}),
                "artifact_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "component": (
                    ["transformer", "model", "diffusion_model"],
                    {"default": "transformer"},
                ),
                "strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PIPELINE", "ORBITQUANT_INFO")
    RETURN_NAMES = ("pipeline", "info")
    FUNCTION = "load"
    CATEGORY = "OrbitQuant"

    def load(self, pipeline: Any, artifact_path: str, component: str, strict: bool):
        if not artifact_path:
            raise ValueError("artifact_path must not be empty")
        manifest = load_quantized_pipeline_component(
            pipeline,
            artifact_path,
            component=component,
            strict=bool(strict),
        )
        return (pipeline, _manifest_payload(manifest))


class _OrbitQuantTransformerLoader:
    loader_target = "generic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPELINE", {"forceInput": True}),
                "artifact_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PIPELINE", "ORBITQUANT_INFO")
    RETURN_NAMES = ("pipeline", "info")
    FUNCTION = "load"
    CATEGORY = "OrbitQuant"

    def load(self, pipeline: Any, artifact_path: str, strict: bool):
        if not artifact_path:
            raise ValueError("artifact_path must not be empty")
        manifest = load_quantized_pipeline_component(
            pipeline,
            artifact_path,
            component="transformer",
            strict=bool(strict),
        )
        payload = _manifest_payload(manifest)
        payload["loader_target"] = self.loader_target
        return (pipeline, payload)


class OrbitQuantFluxLoader(_OrbitQuantTransformerLoader):
    loader_target = "flux"


class OrbitQuantZImageLoader(_OrbitQuantTransformerLoader):
    loader_target = "z_image"


class OrbitQuantWanLoader(_OrbitQuantTransformerLoader):
    loader_target = "wan"


NODE_CLASS_MAPPINGS = {
    "OrbitQuantArtifactInspector": OrbitQuantArtifactInspector,
    "OrbitQuantPipelineComponentLoader": OrbitQuantPipelineComponentLoader,
    "OrbitQuantFluxLoader": OrbitQuantFluxLoader,
    "OrbitQuantZImageLoader": OrbitQuantZImageLoader,
    "OrbitQuantWanLoader": OrbitQuantWanLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OrbitQuantArtifactInspector": "OrbitQuant Inspect Artifact",
    "OrbitQuantPipelineComponentLoader": "OrbitQuant Pipeline Component Loader",
    "OrbitQuantFluxLoader": "OrbitQuant FLUX Loader",
    "OrbitQuantZImageLoader": "OrbitQuant Z-Image Loader",
    "OrbitQuantWanLoader": "OrbitQuant Wan Loader",
}
