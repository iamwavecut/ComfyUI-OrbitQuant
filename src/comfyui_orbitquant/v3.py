from __future__ import annotations

from typing import Any

from comfyui_orbitquant import nodes
from comfyui_orbitquant.minimax_h3 import INFERENCE_PROFILES

try:
    from comfy_api.latest import ComfyExtension, io, ui
except Exception as exc:  # pragma: no cover - exercised through lazy import tests.
    raise ImportError("ComfyUI V3 API is not available") from exc


_CATEGORY = "OrbitQuant"
_INFO_TYPE = "ORBITQUANT_INFO"
_PIPELINE_TYPE = "PIPELINE"
_RELEASE_TYPE = "ORBITQUANT_RELEASE"


def _pipeline_input() -> Any:
    return io.Custom(_PIPELINE_TYPE).Input("pipeline")


def _pipeline_output() -> Any:
    return io.Custom(_PIPELINE_TYPE).Output("pipeline", display_name="pipeline")


def _info_output() -> Any:
    return io.Custom(_INFO_TYPE).Output("info", display_name="info")


def _artifact_path_input() -> Any:
    return io.String.Input(
        "artifact_path",
        default="",
        multiline=False,
        tooltip="Path to an OrbitQuant component artifact directory.",
    )


def _strict_input() -> Any:
    return io.Boolean.Input(
        "strict",
        default=True,
        tooltip="Use strict OrbitQuant artifact state-dict loading.",
    )


def _runtime_mode_input() -> Any:
    return io.Combo.Input(
        "runtime_mode",
        options=list(nodes.RUNTIME_MODE_OPTIONS),
        default="auto_fused",
        tooltip="OrbitQuant runtime mode used when loading the artifact.",
    )


def _activation_kernel_backend_input() -> Any:
    return io.Combo.Input(
        "activation_kernel_backend",
        options=list(nodes.ACTIVATION_KERNEL_BACKEND_OPTIONS),
        default="auto",
        tooltip="Activation quantization kernel backend requested from OrbitQuant.",
    )


class OrbitQuantArtifactInspectorV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OrbitQuantArtifactInspector",
            display_name="OrbitQuant Inspect Artifact",
            category=_CATEGORY,
            description="Validate an OrbitQuant artifact and summarize its metadata.",
            inputs=[_artifact_path_input()],
            outputs=[
                io.String.Output("summary", display_name="summary"),
                _info_output(),
            ],
        )

    @classmethod
    def execute(cls, artifact_path: str) -> io.NodeOutput:
        summary, info = nodes.OrbitQuantArtifactInspector().inspect(artifact_path)
        return io.NodeOutput(summary, info)


class OrbitQuantPipelineComponentLoaderV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OrbitQuantPipelineComponentLoader",
            display_name="OrbitQuant Pipeline Component Loader",
            category=_CATEGORY,
            description="Attach an OrbitQuant component artifact to a pipeline object.",
            inputs=[
                _pipeline_input(),
                _artifact_path_input(),
                io.Combo.Input(
                    "component",
                    options=["transformer", "model", "diffusion_model"],
                    default="transformer",
                    tooltip="Pipeline attribute that receives the quantized component.",
                ),
                _strict_input(),
                _runtime_mode_input(),
                _activation_kernel_backend_input(),
            ],
            outputs=[
                _pipeline_output(),
                _info_output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: Any,
        artifact_path: str,
        component: str,
        strict: bool,
        runtime_mode: str,
        activation_kernel_backend: str,
    ) -> io.NodeOutput:
        loaded_pipeline, info = nodes.OrbitQuantPipelineComponentLoader().load(
            pipeline,
            artifact_path,
            component,
            strict,
            runtime_mode,
            activation_kernel_backend,
        )
        return io.NodeOutput(loaded_pipeline, info)


class _OrbitQuantTransformerLoaderV3(io.ComfyNode):
    legacy_loader_cls: type[nodes._OrbitQuantTransformerLoader]
    node_id: str
    display_name: str
    description: str

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=_CATEGORY,
            description=cls.description,
            inputs=[
                _pipeline_input(),
                _artifact_path_input(),
                _strict_input(),
                _runtime_mode_input(),
                _activation_kernel_backend_input(),
            ],
            outputs=[
                _pipeline_output(),
                _info_output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: Any,
        artifact_path: str,
        strict: bool,
        runtime_mode: str,
        activation_kernel_backend: str,
    ) -> io.NodeOutput:
        loaded_pipeline, info = cls.legacy_loader_cls().load(
            pipeline,
            artifact_path,
            strict,
            runtime_mode,
            activation_kernel_backend,
        )
        return io.NodeOutput(loaded_pipeline, info)


class OrbitQuantFluxLoaderV3(_OrbitQuantTransformerLoaderV3):
    legacy_loader_cls = nodes.OrbitQuantFluxLoader
    node_id = "OrbitQuantFluxLoader"
    display_name = "OrbitQuant FLUX Loader"
    description = "Attach a FLUX or FLUX.2 OrbitQuant transformer artifact to a pipeline."


class OrbitQuantZImageLoaderV3(_OrbitQuantTransformerLoaderV3):
    legacy_loader_cls = nodes.OrbitQuantZImageLoader
    node_id = "OrbitQuantZImageLoader"
    display_name = "OrbitQuant Z-Image Loader"
    description = "Attach a Z-Image OrbitQuant transformer artifact to a pipeline."


class OrbitQuantWanLoaderV3(_OrbitQuantTransformerLoaderV3):
    legacy_loader_cls = nodes.OrbitQuantWanLoader
    node_id = "OrbitQuantWanLoader"
    display_name = "OrbitQuant Wan Loader"
    description = "Attach a Wan OrbitQuant transformer artifact to a pipeline."


class OrbitQuantReleaseLoaderV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OrbitQuantReleaseLoader",
            display_name="OrbitQuant Release Loader",
            category=_CATEGORY,
            description="Validate and route a local OrbitQuant multicomponent release.",
            inputs=[
                io.String.Input(
                    "model_path",
                    default="",
                    multiline=False,
                    tooltip="Local directory downloaded from an OrbitQuant model repo.",
                )
            ],
            outputs=[
                io.Custom(_RELEASE_TYPE).Output("release", display_name="release"),
                io.String.Output("summary_json", display_name="summary_json"),
            ],
        )

    @classmethod
    def execute(cls, model_path: str) -> io.NodeOutput:
        release, summary_json = nodes.OrbitQuantReleaseLoader().load(model_path)
        return io.NodeOutput(release, summary_json)


class OrbitQuantGenerateVideoV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OrbitQuantGenerateVideo",
            display_name="OrbitQuant Generate Video",
            category=_CATEGORY,
            description=(
                "Generate MiniMax H3 audio-video with durable latents, stage offload, "
                "and source FP32 VAE decode."
            ),
            is_output_node=True,
            inputs=[
                io.Custom(_RELEASE_TYPE).Input("release"),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.Combo.Input("task", options=["t2va", "ref2va"], default="t2va"),
                io.Combo.Input(
                    "inference_profile",
                    options=list(INFERENCE_PROFILES),
                    default="balanced",
                ),
                io.String.Input("reference_path", default="", multiline=False),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2**63 - 1,
                    control_after_generate=False,
                ),
                io.Int.Input("width", default=608, min=64, max=4096, step=32),
                io.Int.Input("height", default=480, min=64, max=4096, step=32),
                io.Int.Input("num_frames", default=124, min=120, max=360, step=4),
                io.Int.Input("steps", default=24, min=2, max=1000),
                io.String.Input(
                    "filename_prefix",
                    default="orbitquant/minimax-h3",
                    multiline=False,
                ),
            ],
            outputs=[
                io.Video.Output("video", display_name="video"),
                io.String.Output("report_json", display_name="report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        release: Any,
        prompt: str,
        task: str,
        inference_profile: str,
        reference_path: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        filename_prefix: str,
    ) -> io.NodeOutput:
        video, report_json, preview = (
            nodes.OrbitQuantGenerateVideo().run_for_comfy(
                release,
                prompt,
                task,
                inference_profile,
                reference_path,
                seed,
                width,
                height,
                num_frames,
                steps,
                filename_prefix,
            )
        )
        return io.NodeOutput(video, report_json, ui=ui.PreviewVideo([preview]))


class OrbitQuantExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OrbitQuantArtifactInspectorV3,
            OrbitQuantPipelineComponentLoaderV3,
            OrbitQuantFluxLoaderV3,
            OrbitQuantZImageLoaderV3,
            OrbitQuantWanLoaderV3,
            OrbitQuantReleaseLoaderV3,
            OrbitQuantGenerateVideoV3,
        ]


async def comfy_entrypoint() -> OrbitQuantExtension:
    return OrbitQuantExtension()
