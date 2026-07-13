from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUNTIME_MODE_OPTIONS = (
    "auto_fused",
    "dequant_bf16",
    "debug_no_quant",
    "debug_no_activation_quant",
    "triton_packed_matmul",
    "native_packed_matmul",
)

ACTIVATION_KERNEL_BACKEND_OPTIONS = (
    "auto",
    "cpu",
    "mps",
    "triton_cuda",
    "triton_rocm",
    "triton_xpu",
)


def _missing_orbitquant_error() -> RuntimeError:
    return RuntimeError(
        "ComfyUI-OrbitQuant requires the orbitquant package. Install OrbitQuant "
        "into the Python environment used by ComfyUI before using this node."
    )


def _orbitquant_manifest_cls() -> Any:
    try:
        from orbitquant.artifacts import OrbitQuantManifest
    except ImportError as exc:
        raise _missing_orbitquant_error() from exc
    return OrbitQuantManifest


def validate_orbitquant_artifact(artifact_path: str | Path) -> dict[str, Any]:
    try:
        from orbitquant.artifacts import validate_orbitquant_artifact as validate
    except ImportError as exc:
        raise _missing_orbitquant_error() from exc
    return validate(artifact_path)


def load_quantized_pipeline_component(
    pipeline: Any,
    artifact_path: str | Path,
    *,
    component: str,
    strict: bool,
    runtime_mode: str | None = None,
    activation_kernel_backend: str | None = None,
) -> Any:
    try:
        from orbitquant.pipeline import load_quantized_pipeline_component as load_component
    except ImportError as exc:
        raise _missing_orbitquant_error() from exc
    return load_component(
        pipeline,
        artifact_path,
        component=component,
        strict=strict,
        runtime_mode=runtime_mode,
        activation_kernel_backend=activation_kernel_backend,
    )


def _validate_runtime_options(runtime_mode: str, activation_kernel_backend: str) -> None:
    if runtime_mode not in RUNTIME_MODE_OPTIONS:
        accepted = ", ".join(RUNTIME_MODE_OPTIONS)
        raise ValueError(f"runtime_mode must be one of [{accepted}], got {runtime_mode!r}")
    if activation_kernel_backend not in ACTIVATION_KERNEL_BACKEND_OPTIONS:
        accepted = ", ".join(ACTIVATION_KERNEL_BACKEND_OPTIONS)
        raise ValueError(
            "activation_kernel_backend must be one of "
            f"[{accepted}], got {activation_kernel_backend!r}"
        )


def read_manifest(artifact_path: str | Path) -> Any:
    path = Path(artifact_path)
    return _orbitquant_manifest_cls().from_dict(
        json.loads((path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )


def read_model_index(artifact_path: str | Path) -> dict[str, Any]:
    path = Path(artifact_path) / "model_index.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_payload(manifest: Any) -> dict[str, Any]:
    quantized_modules = list(getattr(manifest, "quantized_modules", []))
    adaln_modules = list(getattr(manifest, "adaln_modules", []))
    skipped_modules = list(getattr(manifest, "skipped_modules", []))
    return {
        "source_model_id": manifest.source_model_id,
        "source_revision": getattr(manifest, "source_revision", "unknown"),
        "source_license": getattr(manifest, "source_license", "unknown"),
        "bits": f"W{manifest.weight_bits}A{manifest.activation_bits}",
        "target_policy": getattr(manifest, "target_policy", "unknown"),
        "runtime_mode": getattr(manifest, "runtime_mode", "unknown"),
        "activation_kernel_backend": getattr(manifest, "activation_kernel_backend", "auto"),
        "quantized_module_count": len(quantized_modules),
        "adaln_module_count": len(adaln_modules),
        "skipped_module_count": len(skipped_modules),
        "quantized_modules": quantized_modules,
        "adaln_modules": adaln_modules,
        "skipped_modules": skipped_modules,
    }


def _payload_with_model_index(manifest: Any, artifact_path: str | Path) -> dict[str, Any]:
    payload = _manifest_payload(manifest)
    model_index = read_model_index(artifact_path)
    if model_index:
        payload["artifact_component"] = model_index.get("component", "unknown")
        payload["artifact_weight_name"] = model_index.get("weight_name", "unknown")
    return payload


def _summary_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Source: {payload['source_model_id']}",
        f"Revision: {payload['source_revision']}",
        f"Bits: {payload['bits']}",
    ]
    if "artifact_component" in payload:
        lines.append(f"Component: {payload['artifact_component']}")
    lines.extend(
        [
            f"Policy: {payload['target_policy']}",
            f"Runtime: {payload['runtime_mode']}",
            f"Activation backend: {payload['activation_kernel_backend']}",
            f"Quantized modules: {payload['quantized_module_count']}",
            f"AdaLN INT4 modules: {payload['adaln_module_count']}",
            f"Skipped modules: {payload['skipped_module_count']}",
        ]
    )
    return "\n".join(lines)


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
        payload = _payload_with_model_index(manifest, path)
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
                "runtime_mode": (list(RUNTIME_MODE_OPTIONS), {"default": "auto_fused"}),
                "activation_kernel_backend": (
                    list(ACTIVATION_KERNEL_BACKEND_OPTIONS),
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("PIPELINE", "ORBITQUANT_INFO")
    RETURN_NAMES = ("pipeline", "info")
    FUNCTION = "load"
    CATEGORY = "OrbitQuant"

    def load(
        self,
        pipeline: Any,
        artifact_path: str,
        component: str,
        strict: bool,
        runtime_mode: str = "auto_fused",
        activation_kernel_backend: str = "auto",
    ):
        if not artifact_path:
            raise ValueError("artifact_path must not be empty")
        _validate_runtime_options(runtime_mode, activation_kernel_backend)
        manifest = load_quantized_pipeline_component(
            pipeline,
            artifact_path,
            component=component,
            strict=bool(strict),
            runtime_mode=runtime_mode,
            activation_kernel_backend=activation_kernel_backend,
        )
        payload = _payload_with_model_index(manifest, artifact_path)
        payload["requested_runtime_mode"] = runtime_mode
        payload["requested_activation_kernel_backend"] = activation_kernel_backend
        return (pipeline, payload)


class _OrbitQuantTransformerLoader:
    loader_target = "generic"
    display_name = "OrbitQuant Transformer Loader"
    accepted_target_policies: tuple[str, ...] = ()

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
                "runtime_mode": (list(RUNTIME_MODE_OPTIONS), {"default": "auto_fused"}),
                "activation_kernel_backend": (
                    list(ACTIVATION_KERNEL_BACKEND_OPTIONS),
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("PIPELINE", "ORBITQUANT_INFO")
    RETURN_NAMES = ("pipeline", "info")
    FUNCTION = "load"
    CATEGORY = "OrbitQuant"

    def load(
        self,
        pipeline: Any,
        artifact_path: str,
        strict: bool,
        runtime_mode: str = "auto_fused",
        activation_kernel_backend: str = "auto",
    ):
        if not artifact_path:
            raise ValueError("artifact_path must not be empty")
        _validate_runtime_options(runtime_mode, activation_kernel_backend)
        preflight_payload = _payload_with_model_index(read_manifest(artifact_path), artifact_path)
        self._validate_target_policy(preflight_payload)
        manifest = load_quantized_pipeline_component(
            pipeline,
            artifact_path,
            component="transformer",
            strict=bool(strict),
            runtime_mode=runtime_mode,
            activation_kernel_backend=activation_kernel_backend,
        )
        payload = _payload_with_model_index(manifest, artifact_path)
        self._validate_target_policy(payload)
        payload["loader_target"] = self.loader_target
        payload["requested_runtime_mode"] = runtime_mode
        payload["requested_activation_kernel_backend"] = activation_kernel_backend
        return (pipeline, payload)

    def _validate_target_policy(self, payload: dict[str, Any]) -> None:
        if not self.accepted_target_policies:
            return
        target_policy = payload.get("target_policy")
        if target_policy in self.accepted_target_policies:
            return
        accepted = ", ".join(self.accepted_target_policies)
        raise ValueError(
            f"{self.display_name} expected an artifact with target_policy in "
            f"[{accepted}], got {target_policy!r} from {payload.get('source_model_id')!r}."
        )


class OrbitQuantFluxLoader(_OrbitQuantTransformerLoader):
    loader_target = "flux"
    display_name = "OrbitQuant FLUX Loader"
    accepted_target_policies = ("flux", "flux2")


class OrbitQuantZImageLoader(_OrbitQuantTransformerLoader):
    loader_target = "z_image"
    display_name = "OrbitQuant Z-Image Loader"
    accepted_target_policies = ("z_image",)


class OrbitQuantWanLoader(_OrbitQuantTransformerLoader):
    loader_target = "wan"
    display_name = "OrbitQuant Wan Loader"
    accepted_target_policies = ("wan",)


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
