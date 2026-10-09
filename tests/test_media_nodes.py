import importlib
import json
from types import SimpleNamespace

import pytest

from comfyui_orbitquant import media_nodes, model, worker
from comfyui_orbitquant.catalog import CATALOG, TURBO_SIGMAS
from comfyui_orbitquant.model import ModelHandle, generation_options


def handle(adapter="kandinsky6"):
    recipe = next(r for r in CATALOG.values() if r.adapter == adapter)
    return ModelHandle(recipe, recipe.repo_id, "immutable-sha")


def options(adapter="kandinsky6", **kwargs):
    h = handle(adapter)
    return generation_options(
        h,
        h.recipe.media_type,
        prompt="A fox in snow",
        seed=42,
        lyrics="[verse]\nA small test song",
        **kwargs,
    )


@pytest.mark.parametrize("recipe", CATALOG.values(), ids=CATALOG.keys())
def test_all_published_recipes_have_valid_defaults(recipe):
    h = ModelHandle(recipe, recipe.repo_id, "sha")
    result = generation_options(h, recipe.media_type, prompt="test", seed=0, lyrics="test")
    assert result["steps"] == recipe.steps
    assert result["width"] == recipe.width
    assert ModelHandle.from_dict(h.to_dict()) == h


def test_catalog_covers_all_public_model_families():
    assert len(CATALOG) == 26
    assert {r.adapter for r in CATALOG.values()} == {
        "kandinsky6",
        "flux_component",
        "flux2_component",
        "z_image_component",
        "wan_component",
        "flux2",
        "ideogram4",
        "krea2",
        "qwen_image21",
        "boogu",
        "minimax_h3",
        "yue2",
    }
    assert not any("lab" in name or "packed-matmul" in name for name in CATALOG)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"width": 319}, "multiples"),
        ({"width": -16}, "negative"),
        ({"frames": 120}, "4n"),
        ({"steps": 8}, "10 steps"),
        ({"allocator_cap_gib": float("nan")}, "finite"),
        ({"allocator_cap_gib": float("inf")}, "finite"),
        ({"profile": "speed"}, "profiles"),
        ({"references": ["ref.png"]}, "Reference"),
    ],
)
def test_kandinsky_rejects_invalid_requests_before_loading(kwargs, message):
    with pytest.raises(ValueError, match=message):
        options(**kwargs)


def test_kandinsky_320p_and_low_memory():
    result = options(width=576, height=320, profile="low-memory", allocator_cap_gib=7)
    assert (result["width"], result["height"], result["frames"]) == (576, 320, 121)
    assert result["allocator_cap_gib"] == 7


def test_mismatched_generator_fails_before_network_or_comfy_import():
    with pytest.raises(ValueError, match="produces video"):
        media_nodes.OrbitQuantGenerateImage().generate(handle(), "test", 1)


def test_private_variant_uses_selected_recipe_and_pinned_revision(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    class Api:
        def model_info(self, repo, revision):
            calls.append((repo, revision))
            return SimpleNamespace(sha="resolved-revision")

    monkeypatch.setattr(hub, "HfApi", Api)
    recipe = handle("qwen_image21").recipe.repo_id
    h = model.resolve_model(recipe, repo_id="example/private-variant", revision="v1")
    assert calls == [("example/private-variant", "v1")]
    assert h.recipe.adapter == "qwen_image21"
    assert h.revision == "resolved-revision"


def test_local_model_does_not_contact_hub(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model, "validate_layout", lambda p, a: calls.append((p, a)))
    h = model.resolve_model(handle().recipe.repo_id, str(tmp_path))
    assert h.revision is None
    assert calls == [(tmp_path, "kandinsky6")]


def test_kandinsky_layout_checks_both_components(tmp_path):
    (tmp_path / "text_encoder").mkdir()
    config = {
        "_class_name": "Kandinsky6Transformer3DModel",
        "quantization_config": {"quant_method": "orbitquant"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "artifact_metadata.json").write_text(
        json.dumps({"source_model": {"revision": "x"}})
    )
    encoder = {"model_type": "qwen2_5_vl", "quantization_config": {"quant_method": "orbitquant"}}
    (tmp_path / "text_encoder/config.json").write_text(json.dumps(encoder))
    model.validate_layout(tmp_path, "kandinsky6")
    encoder["quantization_config"]["quant_method"] = "other"
    (tmp_path / "text_encoder/config.json").write_text(json.dumps(encoder))
    with pytest.raises(ValueError, match="not an OrbitQuant"):
        model.validate_layout(tmp_path, "kandinsky6")


@pytest.mark.parametrize("adapter", ["flux_component", "z_image_component", "wan_component"])
def test_component_recipe_rejects_wrong_policy(adapter, tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "OrbitQuantComponentArtifact",
                "target_policy": "other",
            }
        )
    )
    with pytest.raises(ValueError, match="target_policy"):
        model.validate_layout(tmp_path, adapter)


@pytest.mark.parametrize("count", [4, 5, 6, 7, 8])
def test_turbo_image_uses_distilled_sigmas(count):
    args = worker.pipeline_arguments(handle("qwen_image21"), options("qwen_image21", steps=count))
    assert args["sigmas"] == TURBO_SIGMAS[count]
    assert "guidance_scale" not in args


def test_invalid_turbo_schedule_and_video_frame_count():
    with pytest.raises(ValueError, match="4, 5, 6"):
        options("qwen_image21", steps=3)
    with pytest.raises(ValueError, match="4n"):
        options("wan_component", frames=80)
    with pytest.raises(ValueError, match="120 to 360"):
        options("minimax_h3", frames=81)


def test_ideogram_plain_caption_and_no_cfg():
    args = worker.pipeline_arguments(handle("ideogram4"), options("ideogram4"))
    assert json.loads(args["prompt"]) == {"high_level_description": "A fox in snow"}
    assert args["mu"] == 0 and args["std"] == 1.75
    assert "guidance_scale" not in args and "guidance_schedule" not in args


def test_boogu_uses_instruction_and_dmd_contract():
    args = worker.pipeline_arguments(handle("boogu"), options("boogu"))
    assert args["instruction"] == ["A fox in snow"]
    assert "prompt" not in args
    assert args["use_dmd_student_inference"] is True
    assert args["dmd_conditioning_sigma"] == 0.001


@pytest.mark.parametrize("profile", ["fast", "low-memory", "exact"])
def test_kandinsky_installs_runtime_before_offload(monkeypatch, profile):
    pytest.importorskip("orbitquant")
    runtime = importlib.import_module("orbitquant.runtime.kandinsky6")
    video = importlib.import_module("orbitquant.runtime.kandinsky6_video")
    calls = []
    for mod in (runtime, video):
        for name in dir(mod):
            if name.startswith("install_kandinsky6_"):
                monkeypatch.setattr(mod, name, lambda *a, _name=name, **kw: calls.append(_name))
    pipe = SimpleNamespace(
        transformer=object(),
        vae=SimpleNamespace(
            enable_tiling=lambda **kw: calls.append(("tiles", kw)),
            decode=lambda x: calls.append(("decode", x)),
        ),
        enable_model_cpu_offload=lambda: calls.append("offload"),
        maybe_free_model_hooks=lambda: calls.append("reclaim"),
    )
    worker.configure_kandinsky(pipe, profile)
    assert calls[-1] == "offload"
    pipe.vae.decode("latent")
    if profile == "low-memory":
        assert calls[-2:] == ["reclaim", ("decode", "latent")]
    if profile == "exact":
        assert ("tiles", {}) in calls
        assert "install_kandinsky6_attention" not in calls
        assert "install_kandinsky6_qk_norm" not in calls
    else:
        assert "install_kandinsky6_attention" in calls


def test_video_export_keeps_kandinsky_audio(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "Generator", lambda **kw: SimpleNamespace(manual_seed=lambda x: x))
    monkeypatch.setattr(worker, "resolve_files", lambda h: tmp_path)
    monkeypatch.setattr(worker, "check_environment", lambda a: None)
    audio = torch.zeros(1, 44100)
    output = SimpleNamespace(frames=[["frame"]], audio=audio)

    class Pipe:
        audio_sample_rate = 44100

        def __call__(self, **kw):
            assert torch.is_inference_mode_enabled()
            calls.append(kw)
            return output

    monkeypatch.setattr(worker, "load_pipeline", lambda *a: Pipe())
    monkeypatch.setattr(
        diffusers.utils, "encode_video", lambda *a, **kw: calls.append(kw), raising=False
    )
    worker.run({"model": handle().to_dict(), "options": options(), "destination": str(tmp_path)})
    assert calls[-1]["audio_sample_rate"] == 44100
    assert calls[-1]["audio"].shape == (1, 44100)
    assert calls[0]["num_frames"] == 121


def test_image_output_is_comfy_float_bhwc(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "out.png"
    image.new("RGB", (32, 16), (255, 0, 128)).save(path)
    monkeypatch.setattr(media_nodes, "_generate", lambda *a, **kw: (path, "report"))
    result, report = media_nodes.OrbitQuantGenerateImage().generate(handle("flux2"), "x", 0)
    assert result.shape == (1, 16, 32, 3) and result.dtype == torch.float32
    assert result.max() == 1 and result.min() == 0 and report == "report"


def test_audio_output_is_comfy_bct(monkeypatch, tmp_path):
    import sys

    torch = pytest.importorskip("torch")
    path = tmp_path / "out.wav"
    decoded = torch.linspace(-0.5, 0.5, 9600).reshape(2, 4800)

    def load(filename):
        assert filename == str(path)
        return decoded, 48000

    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", SimpleNamespace(load=load))
    monkeypatch.setattr(media_nodes, "_generate", lambda *a, **kw: (path, "report"))
    audio, _ = media_nodes.OrbitQuantGenerateAudio().generate(handle("yue2"), "x", 0, "lyrics")
    assert audio["waveform"].shape == (1, 2, 4800)
    assert audio["sample_rate"] == 48000
    assert torch.equal(audio["waveform"][0], decoded)


def test_interrupt_stops_worker_and_does_not_report_success(monkeypatch, tmp_path):
    stopped = []

    class Process:
        pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(media_nodes.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(media_nodes, "_stop_process", lambda p: stopped.append(p.pid))

    def interrupt():
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        media_nodes.run_worker({"destination": str(tmp_path)}, "", interrupt)
    assert stopped == [12345]


def test_download_pins_all_components_to_resolved_revision(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    calls = []
    monkeypatch.setattr(
        hub, "snapshot_download", lambda *a, **kw: calls.append((a, kw)) or tmp_path
    )
    monkeypatch.setattr(worker, "validate_layout", lambda *a: None)
    worker.resolve_files(handle())
    assert calls[0][1]["revision"] == "immutable-sha"
    assert "assets/*" in calls[0][1]["ignore_patterns"]


def test_legacy_download_preserves_checksum_coverage(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda *a, **kw: calls.append(kw) or tmp_path)
    monkeypatch.setattr(worker, "validate_layout", lambda *a: None)
    worker.resolve_files(handle("ideogram4"))
    assert calls[0]["ignore_patterns"] is None


@pytest.mark.parametrize("adapter", ["yue2", "minimax_h3"])
def test_bundled_runtime_download_preserves_paths(monkeypatch, tmp_path, adapter):
    hub = pytest.importorskip("huggingface_hub")
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(worker, "materialize_runtime", lambda *a: calls.append(a))
    monkeypatch.setattr(worker, "validate_layout", lambda *a: None)
    worker.resolve_files(handle(adapter))
    assert calls[0][0] == tmp_path
    assert calls[0][1].name == "immutable-sha"
    assert calls[0][1].parent.name == handle(adapter).location.replace("/", "--")


def test_materialize_runtime_resolves_siblings_without_copying_weights(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    blob = tmp_path / "blob"
    blob.write_text("data")
    (source / "run.py").symlink_to(blob)
    destination = tmp_path / "runtime"
    worker.materialize_runtime(source, destination)
    worker.materialize_runtime(source, destination)
    result = destination / "run.py"
    assert result.resolve().parent == destination
    assert result.stat().st_ino == blob.stat().st_ino
    assert result.read_text() == "data"


def test_preflight_rejects_old_diffusers_before_model_download(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace())
    with pytest.raises(RuntimeError, match="Kandinsky6TI2VAPipeline"):
        worker.check_environment("kandinsky6")


def test_yue_preflight_checks_cuda_build(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.10.0", version=SimpleNamespace(cuda="13.0")),
    )
    with pytest.raises(RuntimeError, match="CUDA 12.8"):
        worker.check_environment("yue2")


@pytest.mark.parametrize("fails", [False, True])
def test_ideogram_encoder_runs_on_gpu_and_offloads_on_error(fails):
    calls = []

    def encode(*args, **kwargs):
        assert calls[-1] == "cuda"
        if fails:
            raise ValueError("bad caption")
        return "embeddings"

    pipe = SimpleNamespace(
        text_encoder=SimpleNamespace(to=lambda device: calls.append(device)),
        encode_prompt=encode,
    )
    worker.stage_ideogram_text_encoder(pipe)
    if fails:
        with pytest.raises(ValueError, match="bad caption"):
            pipe.encode_prompt("caption")
    else:
        assert pipe.encode_prompt("caption") == "embeddings"
    assert calls == ["cuda", "cpu"]


def test_krea_generation_keeps_version_counters_for_text_cache(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    seen = []

    class Pipeline:
        def __call__(self, **kwargs):
            text = torch.ones(2)
            version = text._version
            text.add_(1)
            assert text._version > version
            assert not torch.is_grad_enabled()
            seen.append(True)
            return SimpleNamespace(images=[SimpleNamespace(save=lambda p: p.touch())])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "Generator", lambda **kw: SimpleNamespace(manual_seed=lambda s: s))
    monkeypatch.setattr(worker, "check_environment", lambda a: None)
    monkeypatch.setattr(worker, "resolve_files", lambda h: tmp_path)
    monkeypatch.setattr(worker, "load_pipeline", lambda *a: Pipeline())
    result = worker.run(
        {
            "model": handle("krea2").to_dict(),
            "options": options("krea2"),
            "destination": str(tmp_path),
        }
    )
    assert seen and result["image"] == str(tmp_path / "image.png")
