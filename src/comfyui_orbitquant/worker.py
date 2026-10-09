"""Isolated generation process used by the ComfyUI model nodes."""

from __future__ import annotations

import argparse
import contextlib
import errno
import importlib
import inspect
import json
import os
import runpy
import shutil
import sys
import threading
import time
from pathlib import Path

# File invocation also works with a separate model environment and an installed wheel.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comfyui_orbitquant.catalog import BOOGU_SOURCE, IDEOGRAM_COMPONENTS, TURBO_SIGMAS
from comfyui_orbitquant.model import ModelHandle, read_json, validate_layout


def check_environment(adapter: str):
    """Reject incompatible environments before downloading model weights."""
    import torch

    if adapter == "yue2":
        if (
            sys.version_info[:2] != (3, 12)
            or torch.__version__.split("+")[0] != "2.10.0"
            or torch.version.cuda != "12.8"
        ):
            raise RuntimeError(
                "YuE2 requires its separate Python 3.12 / Torch 2.10.0 CUDA 12.8 environment. "
                "Set python_executable in OrbitQuant Model Loader; see the YuE2 model card."
            )
        return
    import diffusers

    if adapter == "minimax_h3":
        import diffusers.modular_pipelines as modular

        if not hasattr(modular, "MiniMaxH3Ref2VABlocks"):
            raise RuntimeError(
                "MiniMax H3 requires the Diffusers revision pinned in its model card. "
                "Select the MiniMax model environment with python_executable."
            )
        return

    names = {
        "kandinsky6": "Kandinsky6TI2VAPipeline",
        "flux_component": "FluxPipeline",
        "flux2_component": "Flux2KleinPipeline",
        "flux2": "Flux2KleinPipeline",
        "z_image_component": "ZImagePipeline",
        "wan_component": "WanPipeline",
        "ideogram4": "Ideogram4Pipeline",
        "krea2": "Krea2Pipeline",
        "qwen_image21": "QwenImage21Pipeline",
    }
    if adapter in names and not hasattr(diffusers, names[adapter]):
        raise RuntimeError(
            f"This Diffusers installation does not provide {names[adapter]}. "
            "Install the model environment listed in the ComfyUI-OrbitQuant README, "
            "then select its python_executable in OrbitQuant Model Loader."
        )
    if adapter == "boogu" and importlib.util.find_spec("boogu") is None:
        raise RuntimeError("Install the Boogu package in the selected model environment")


def resolve_files(handle: ModelHandle) -> Path:
    if handle.revision is None:
        path = Path(handle.location)
    else:
        from huggingface_hub import constants, snapshot_download

        # Legacy checksums cover every published file, including card assets.
        legacy = (
            handle.recipe.adapter.endswith("_component") or handle.recipe.adapter == "ideogram4"
        )
        ignore = (
            None
            if legacy
            else [
                "assets/*",
                "benchmark/*",
                "evaluation/*",
                "evidence/*",
                "examples/*",
                "src/evaluation_harness/*",
                "src/kernel-source/*",
                ".git*",
            ]
        )
        path = Path(
            snapshot_download(
                handle.location,
                revision=handle.revision,
                ignore_patterns=ignore,
            )
        )
        if handle.recipe.adapter in {"yue2", "minimax_h3"}:
            # These runtimes resolve sibling modules and kernels relative to __file__.
            # Hub blob symlinks lose that structure. Hard links preserve it without
            # duplicating checkpoint data on the same filesystem.
            destination = (
                Path(constants.HF_HOME)
                / "comfyui-orbitquant-runtime"
                / handle.location.replace("/", "--")
                / handle.revision
            )
            materialize_runtime(path, destination)
            path = destination
    validate_layout(path, handle.recipe.adapter)
    return path


def materialize_runtime(source: Path, destination: Path):
    for file in source.rglob("*"):
        if not file.is_file():
            continue
        target = destination / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(file.resolve(), target)
        except FileExistsError:
            continue
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                raise
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            shutil.copyfile(file, temporary)
            temporary.replace(target)


def configure_kandinsky(pipe, profile: str):
    from orbitquant.runtime.kandinsky6 import (
        install_kandinsky6_attention,
        install_kandinsky6_norm_fusion,
        install_kandinsky6_pointwise,
        install_kandinsky6_projection_reuse,
        install_kandinsky6_qk_norm,
    )
    from orbitquant.runtime.kandinsky6_video import (
        install_kandinsky6_video_io,
        install_kandinsky6_video_norm,
    )

    if profile != "exact":
        install_kandinsky6_attention(pipe.transformer)
    install_kandinsky6_pointwise(pipe.transformer)
    install_kandinsky6_projection_reuse(pipe.transformer)
    if profile != "exact":
        install_kandinsky6_norm_fusion(pipe.transformer)
        install_kandinsky6_qk_norm(pipe.transformer)
    install_kandinsky6_video_io(pipe)
    if profile == "exact":
        pipe.vae.enable_tiling()
    else:
        install_kandinsky6_video_norm(pipe)
        pipe.vae.enable_tiling(
            tile_sample_min_height=272,
            tile_sample_min_width=320,
            tile_sample_stride_height=240,
            tile_sample_stride_width=288,
        )
    if profile == "low-memory":
        decode = pipe.vae.decode

        def decode_after_reclaim(*args, **kwargs):
            pipe.maybe_free_model_hooks()
            return decode(*args, **kwargs)

        pipe.vae.decode = decode_after_reclaim
    pipe.enable_model_cpu_offload()
    return pipe


def _ideogram_pipeline_class():
    module = importlib.import_module("diffusers.pipelines.ideogram4.pipeline_ideogram4")
    cls = module.Ideogram4Pipeline
    if "unconditional_transformer" in cls._optional_components:
        return cls
    from comfyui_orbitquant._ideogram_patch import patch_source

    # Apply the released compatibility patch in this process only. Never edit site-packages.
    namespace = dict(vars(module))
    source = patch_source(inspect.getsource(module))
    exec(compile(source, module.__file__, "exec"), namespace)
    return namespace["Ideogram4Pipeline"]


def stage_ideogram_text_encoder(pipe):
    # Ideogram calls encoder submodules directly, bypassing model offload hooks.
    # Place the encoder explicitly for this stage, then release it before denoising.
    encode_prompt = pipe.encode_prompt

    def encode_on_cuda(*args, **kwargs):
        pipe.text_encoder.to("cuda")
        try:
            return encode_prompt(*args, **kwargs)
        finally:
            pipe.text_encoder.to("cpu")

    pipe.encode_prompt = encode_on_cuda


def load_pipeline(path: Path, adapter: str, profile: str):
    import diffusers
    import orbitquant  # noqa: F401 -- registers the quantizers
    import torch

    dtype = torch.float16 if adapter == "qwen_image21" else torch.bfloat16
    if adapter == "kandinsky6":
        from transformers import Qwen2_5_VLForConditionalGeneration

        source = read_json(path / "artifact_metadata.json")["source_model"]
        transformer = diffusers.Kandinsky6Transformer3DModel.from_pretrained(
            path, torch_dtype=dtype
        )
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            subfolder="text_encoder",
            dtype=dtype,
        )
        pipe = diffusers.Kandinsky6TI2VAPipeline.from_pretrained(
            source["repo_id"],
            revision=source["revision"],
            transformer=transformer,
            text_encoder=encoder,
            torch_dtype=dtype,
        )
        return configure_kandinsky(pipe, profile)
    if adapter.endswith("_component"):
        from orbitquant.pipeline import load_quantized_pipeline_from_artifact

        cls = {
            "flux_component": diffusers.FluxPipeline,
            "flux2_component": diffusers.Flux2KleinPipeline,
            "z_image_component": diffusers.ZImagePipeline,
            "wan_component": diffusers.WanPipeline,
        }[adapter]
        pipe = load_quantized_pipeline_from_artifact(
            path,
            pipeline_cls=cls,
            torch_dtype=dtype,
            runtime_mode="auto_fused",
        )
    elif adapter == "ideogram4":
        from orbitquant.artifacts import load_orbitquant_artifact

        index = read_json(path / "model_index.json")
        cls = diffusers.Ideogram4Transformer2DModel
        config = cls.load_config(
            index["source_model_id"], subfolder="transformer", revision=index["source_revision"]
        )
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            transformer = cls.from_config(config)
        finally:
            torch.set_default_dtype(old_dtype)
        load_orbitquant_artifact(transformer, path, runtime_mode="auto_fused")
        pipe = _ideogram_pipeline_class().from_pretrained(
            IDEOGRAM_COMPONENTS[0],
            revision=IDEOGRAM_COMPONENTS[1],
            transformer=transformer,
            unconditional_transformer=None,
            torch_dtype=dtype,
        )
    elif adapter == "boogu":
        from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
        from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
        from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        source = Path(
            snapshot_download(
                BOOGU_SOURCE[0],
                revision=BOOGU_SOURCE[1],
                allow_patterns=[
                    "mllm/*",
                    "processor/*",
                    "scheduler/*",
                    "vae/*",
                    "model_index.json",
                ],
            )
        )
        pipe = BooguImageTurboPipeline(
            transformer=BooguImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            ),
            vae=diffusers.AutoencoderKL.from_pretrained(source / "vae", torch_dtype=dtype),
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(source / "scheduler"),
            mllm=AutoModelForImageTextToText.from_pretrained(
                source / "mllm", torch_dtype=dtype, trust_remote_code=True
            ),
            processor=AutoProcessor.from_pretrained(source / "processor", trust_remote_code=True),
        )
    else:
        name = {
            "flux2": "Flux2KleinPipeline",
            "krea2": "Krea2Pipeline",
            "qwen_image21": "QwenImage21Pipeline",
        }[adapter]
        pipe = getattr(diffusers, name).from_pretrained(path, torch_dtype=dtype)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
    if adapter == "ideogram4":
        stage_ideogram_text_encoder(pipe)
    return pipe


def pipeline_arguments(handle: ModelHandle, options: dict) -> dict:
    recipe = handle.recipe
    args = dict(
        prompt=options["prompt"],
        width=options["width"],
        height=options["height"],
        num_inference_steps=options["steps"],
    )
    if recipe.adapter == "ideogram4":
        # Ideogram's structured captions are optional; wrap plain text without inventing details.
        try:
            caption = json.loads(options["prompt"])
        except json.JSONDecodeError:
            caption = {"high_level_description": options["prompt"]}
        if not isinstance(caption, dict) or "high_level_description" not in caption:
            raise ValueError("Ideogram JSON prompts need high_level_description")
        args.update(prompt=json.dumps(caption, ensure_ascii=False), mu=0.0, std=1.75)
    elif recipe.adapter == "qwen_image21":
        args.update(
            sigmas=TURBO_SIGMAS[options["steps"]],
            output_resolution=max(options["width"], options["height"]),
        )
    elif recipe.adapter == "boogu":
        args.pop("prompt")
        args.update(
            instruction=[options["prompt"]],
            negative_instruction="",
            empty_instruction="",
            text_guidance_scale=1.0,
            image_guidance_scale=1.0,
            empty_instruction_guidance_scale=0.0,
            use_dmd_student_inference=True,
            dmd_conditioning_sigma=0.001,
            align_res=False,
        )
    else:
        args["guidance_scale"] = 0.0 if recipe.adapter == "krea2" else recipe.guidance
    if recipe.media_type == "video":
        args["num_frames"] = options["frames"]
    if options["references"]:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in options["references"]]
        if recipe.adapter == "boogu":
            args.update(input_images=images, input_image_paths=options["references"])
        else:
            args["image"] = images
    return args


def _run_yue2(path: Path, options: dict, destination: Path):
    check_environment("yue2")
    prompt_path = destination / "yue2-request.json"
    prompt_path.write_text(
        json.dumps(
            {"style": options["prompt"], "lyrics": options["lyrics"], "seed": options["seed"]}
        ),
        encoding="utf-8",
    )
    old_argv = sys.argv
    try:
        sys.argv = [
            str(path / "src/run.py"),
            "--prompt",
            str(prompt_path),
            "--output",
            str(destination),
            "--profile",
            "fast",
            "--fuse-rms-quant",
            "--kv-cache-dtype",
            "int8",
            "--mode",
            options["profile"],
            "--ode-steps",
            str(options["steps"]),
        ]
        runpy.run_path(str(path / "src/run.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
    return {"audio": str(destination / "audio.wav")}


def run(request: dict) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Published OrbitQuant media recipes require an NVIDIA CUDA GPU")
    handle = ModelHandle.from_dict(request["model"])
    options = request["options"]
    adapter = handle.recipe.adapter
    check_environment(adapter)
    destination = Path(request["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    if adapter == "kandinsky6":
        cap = options["allocator_cap_gib"] or (7.25 if options["profile"] == "low-memory" else 0)
        if cap:
            total = torch.cuda.get_device_properties(0).total_memory
            fraction = cap * 2**30 / total
            if fraction > 1:
                raise ValueError("Allocator cap exceeds physical GPU memory")
            torch.cuda.set_per_process_memory_fraction(fraction)
        torch.backends.cudnn.benchmark = False
    path = resolve_files(handle)
    if adapter == "yue2":
        return _run_yue2(path, options, destination)
    if adapter == "minimax_h3":
        from comfyui_orbitquant.minimax_h3 import MiniMaxH3Release, MiniMaxH3Runner

        refs = options["references"]
        result = MiniMaxH3Runner(MiniMaxH3Release.from_path(path)).run(
            output_dir=destination,
            filename="video.mp4",
            prompt=options["prompt"],
            task="ref2va" if refs else "t2va",
            reference_path=refs[0] if refs else "",
            inference_profile=options["profile"],
            seed=options["seed"],
            width=options["width"],
            height=options["height"],
            num_frames=options["frames"],
            steps=options["steps"],
        )
        return {"video": str(result.output_path)}
    pipe = load_pipeline(path, adapter, options["profile"])
    kwargs = pipeline_arguments(handle, options)
    kwargs["generator"] = torch.Generator(device="cuda").manual_seed(options["seed"])

    def progress(step, total):
        temporary = destination / "progress.tmp"
        temporary.write_text(json.dumps({"step": step + 1, "total": total}))
        temporary.replace(destination / "progress.json")

    def callback(pipeline, step, timestep, callback_kwargs):
        progress(step, pipeline.num_timesteps)
        return callback_kwargs

    if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
        kwargs["callback_on_step_end"] = callback
    elif adapter == "boogu":
        kwargs["step_func"] = progress
    torch.cuda.synchronize()
    start = time.perf_counter()
    # Krea's text cache checks tensor version counters, which inference tensors lack.
    context = torch.no_grad if adapter == "krea2" else torch.inference_mode
    with context():
        output = pipe(**kwargs)
    torch.cuda.synchronize()
    result = {"pipeline_seconds": time.perf_counter() - start}
    if handle.recipe.media_type == "image":
        image_path = destination / "image.png"
        output.images[0].save(image_path)
        result["image"] = str(image_path)
    else:
        from diffusers.utils import encode_video

        video_path = destination / "video.mp4"
        audio_options = {}
        if adapter == "kandinsky6":
            audio_options = dict(
                audio=output.audio[0][None],
                audio_sample_rate=pipe.audio_sample_rate,
            )
        encode_video(
            output.frames[0], fps=handle.recipe.fps, output_path=str(video_path), **audio_options
        )
        result["video"] = str(video_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = read_json(args.request)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            print("OrbitQuant generation is running", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    os.environ.setdefault("ORBITQUANT_STRICT_PACKED", "1")
    start = time.perf_counter()
    try:
        result = run(request)
        result.update(
            model=request["model"],
            options=request["options"],
            process_seconds=time.perf_counter() - start,
        )
        Path(request["result"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        stop.set()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
