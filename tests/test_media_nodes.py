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
    assert len(CATALOG) == 30
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
        "flux3",
        "drama3",
        "moss_image",
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


def test_flux3_edit_keeps_first_reference_size_unless_dimensions_are_set():
    follow = options("flux3", references=["a.png"])
    assert (follow["width"], follow["height"], follow["steps"]) == (0, 0, 16)
    fixed = options("flux3", references=["a.png"], width=768, height=512)
    assert (fixed["width"], fixed["height"]) == (768, 512)
    text_to_image = options("flux3")
    assert (text_to_image["width"], text_to_image["height"]) == (1024, 1024)
    with pytest.raises(ValueError, match="at most 10"):
        options("flux3", references=["a.png"] * 11)
    with pytest.raises(ValueError, match="multiples"):
        options("flux3", width=1000, height=1024)


def test_flux3_layout_checks_the_release_manifest_and_both_components(tmp_path):
    for name in ("flux3_orbitquant.py", "ae.safetensors", "dit/model.safetensors"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "text_encoder").mkdir()
    quantized = {"quant_method": "orbitquant"}
    (tmp_path / "dit/quantization_config.json").write_text(json.dumps(quantized))
    encoder = {"quantization_config": quantized}
    (tmp_path / "text_encoder/config.json").write_text(json.dumps(encoder))
    with pytest.raises(FileNotFoundError):
        model.validate_layout(tmp_path, "flux3")
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="FLUX.3 Klein"):
        model.validate_layout(tmp_path, "flux3")
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "flux3-orbitquant"}))
    model.validate_layout(tmp_path, "flux3")
    (tmp_path / "text_encoder/config.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="not an OrbitQuant"):
        model.validate_layout(tmp_path, "flux3")


def test_moss_image_defaults_profiles_and_text_only_input():
    result = options("moss_image")
    assert (result["width"], result["height"], result["steps"]) == (1024, 1024, 50)
    assert CATALOG["WaveCut/MOSS-Image-SFT-OrbitQuant-W4A8"].guidance == 3.0
    assert options("moss_image", profile="model-offload")["profile"] == "model-offload"
    assert options("moss_image", profile="int8-attention")["profile"] == "int8-attention"
    with pytest.raises(ValueError, match="Reference"):
        options("moss_image", references=["a.png"])
    with pytest.raises(ValueError, match="multiples"):
        options("moss_image", width=1000)


def test_moss_image_layout_checks_the_release_manifest_and_both_components(tmp_path):
    for name in ("moss_orbitquant.py", "pipeline_moss_image.py", "transformer/model.safetensors"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "text_encoder").mkdir()
    quantized = {"quant_method": "orbitquant"}
    (tmp_path / "transformer/quantization_config.json").write_text(json.dumps(quantized))
    text_encoder = {"quantization_config": quantized}
    (tmp_path / "text_encoder/config.json").write_text(json.dumps(text_encoder))
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "flux3-orbitquant"}))
    with pytest.raises(ValueError, match="MOSS-Image"):
        model.validate_layout(tmp_path, "moss_image")
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "moss-image-orbitquant"}))
    model.validate_layout(tmp_path, "moss_image")
    (tmp_path / "pipeline_moss_image.py").unlink()
    with pytest.raises(ValueError, match="missing pipeline_moss_image.py"):
        model.validate_layout(tmp_path, "moss_image")


def test_drama3_defaults_and_speaker_rules():
    result = options("drama3", speakers=[{"path": "voice.wav", "text": "Hello."}])
    assert (result["frames"], result["steps"], result["profile"]) == (500, 0, "default")
    with pytest.raises(ValueError, match="text spoken"):
        options("drama3", speakers=[{"path": "voice.wav", "text": " "}])
    with pytest.raises(ValueError, match="at most 16"):
        options("drama3", speakers=[{"path": "v.wav", "text": "x"}] * 17)
    with pytest.raises(ValueError, match="profiles"):
        options("drama3", profile="turbo")
    with pytest.raises(ValueError, match="not supported"):
        options("yue2", speakers=[{"path": "v.wav", "text": "x"}])


def test_drama3_layout_requires_runtime_files(tmp_path):
    (tmp_path / "orbitquant_manifest.json").write_text(json.dumps({"format": "drama3-orbitquant"}))
    with pytest.raises(ValueError, match="missing run_tts.py"):
        model.validate_layout(tmp_path, "drama3")
    for name in ("run_tts.py", "drama3_orbitquant.py", "codec.pth", "model.safetensors.index.json"):
        (tmp_path / name).write_text("x")
    model.validate_layout(tmp_path, "drama3")


def test_drama3_command_line_selects_runtime_flags(tmp_path):
    base = options("drama3", speakers=[{"path": "ref.wav", "text": "Hello."}])
    args = worker.drama3_arguments(tmp_path, base, tmp_path / "out")
    assert args[:2] == [str(tmp_path / "run_tts.py"), "--model"]
    assert args[args.index("--reference-audio") + 1] == "ref.wav"
    assert args[args.index("--reference-text") + 1] == "Hello."
    assert args[args.index("--max-new-tokens") + 1] == "500"
    assert "--eager" not in args and "--cpu-offload" not in args
    dialogue = dict(base, prompt="<|speaker:0|> Hi. <|speaker:1|> Hello.")
    assert "--eager" in worker.drama3_arguments(tmp_path, dialogue, tmp_path / "out")
    offload = dict(base, profile="cpu-offload")
    assert "--cpu-offload" in worker.drama3_arguments(tmp_path, offload, tmp_path / "out")


def test_audio_node_pairs_connected_speakers_with_their_text():
    inputs = media_nodes._inputs("audio")
    assert [k for k in inputs["optional"] if k.startswith("speaker_audio_")] == [
        f"speaker_audio_{i}" for i in range(1, media_nodes.SPEAKER_SLOTS + 1)
    ]
    slots = {"speaker_audio_1": "a", "speaker_text_1": "one", "speaker_text_2": "orphan",
             "speaker_audio_3": "c", "speaker_text_3": "three"}
    assert media_nodes._collect_speakers(slots) == [("a", "one"), ("c", "three")]


def test_reference_audio_is_written_as_pcm_wav(tmp_path):
    torch = pytest.importorskip("torch")
    import wave

    audio = {"waveform": torch.tensor([[[0.0, 0.5, -0.5, 1.5]]]), "sample_rate": 24000}
    media_nodes._write_wav(tmp_path / "voice.wav", audio)
    with wave.open(str(tmp_path / "voice.wav")) as handle:
        assert (handle.getnchannels(), handle.getframerate(), handle.getnframes()) == (1, 24000, 4)
        frames = handle.readframes(4)
    assert list(memoryview(frames).cast("h")) == [0, 16384, -16384, 32767]


def test_flux3_offload_profile_is_limited_to_flux3():
    assert options("flux3", profile="model-offload")["profile"] == "model-offload"
    assert options("flux3", profile="int8-attention")["profile"] == "int8-attention"
    with pytest.raises(ValueError, match="profiles"):
        options("flux3", profile="eager")
    with pytest.raises(ValueError, match="profiles"):
        options("krea2", profile="model-offload")
    assert "profile" in media_nodes._inputs("image")["optional"]


FAKE_SERVER = """
import json, os, sys
PREFIX = "ORBITQUANT-RESULT "
for line in sys.stdin:
    path = line.strip()
    request = json.load(open(path))
    print("log line for", os.path.basename(os.path.dirname(path)), flush=True)
    ok = request["options"].get("fail") is None
    if ok:
        json.dump({"image": "x", "pid": os.getpid()}, open(request["result"], "w"))
    status = {"request": path, "ok": ok}
    if not ok:
        status["error"] = "ValueError: bad request"
    print(PREFIX + json.dumps(status), flush=True)
"""


def _kept_request(tmp_path, name, **options):
    destination = tmp_path / name
    destination.mkdir()
    return {"destination": str(destination), "result": str(destination / "result.json"),
            "options": options}


def _kept(tmp_path, name, interrupt=lambda: None, **options):
    request = _kept_request(tmp_path, name, **options)
    return media_nodes.run_worker(request, "", interrupt, keep_loaded=True)


@pytest.fixture
def fake_server(monkeypatch, tmp_path):
    script = tmp_path / "server.py"
    script.write_text(FAKE_SERVER)
    monkeypatch.setattr(media_nodes, "WORKER_SCRIPT", script)
    yield
    media_nodes.stop_serving()


def test_kept_worker_serves_consecutive_requests_in_one_process(fake_server, tmp_path):
    first = _kept(tmp_path, "a")
    second = _kept(tmp_path, "b")
    assert first["pid"] == second["pid"]
    assert "log line for b" in (tmp_path / "b" / "generation.log").read_text()


def test_kept_worker_failure_raises_and_keeps_the_process(fake_server, tmp_path):
    first = _kept(tmp_path, "a")
    with pytest.raises(RuntimeError, match="bad request"):
        _kept(tmp_path, "b", fail=True)
    third = _kept(tmp_path, "c")
    assert first["pid"] == third["pid"]


def test_interrupt_stops_the_kept_worker(fake_server, tmp_path):
    _kept(tmp_path, "a")
    worker = next(iter(media_nodes._SERVING.values()))

    def interrupt():
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        _kept(tmp_path, "b", interrupt)
    assert not media_nodes._SERVING
    assert worker.process.poll() is not None


def test_a_run_without_keep_loaded_stops_the_kept_worker(fake_server, monkeypatch, tmp_path):
    _kept(tmp_path, "a")
    worker = next(iter(media_nodes._SERVING.values()))
    monkeypatch.setattr(media_nodes, "_run_once", lambda *a: None)
    with pytest.raises(RuntimeError, match="did not return a result"):
        media_nodes.run_worker(_kept_request(tmp_path, "b"), "", lambda: None)
    assert not media_nodes._SERVING
    assert worker.process.poll() is not None


def test_serving_cache_loads_once_per_key_and_releases_on_change(monkeypatch):
    monkeypatch.setattr(worker, "_SERVING", True)
    monkeypatch.setattr(worker, "_LOADED", {})
    loads = []
    monkeypatch.setattr(worker, "release_cached", lambda: loads.append("release"))
    assert worker.cached("a", lambda: loads.append("a") or "A") == "A"
    assert worker.cached("a", lambda: loads.append("a") or "A") == "A"
    assert worker.cached("b", lambda: loads.append("b") or "B") == "B"
    assert loads == ["release", "a", "release", "b"]


def test_one_shot_worker_does_not_cache(monkeypatch):
    monkeypatch.setattr(worker, "_SERVING", False)
    loads = []
    worker.cached("a", lambda: loads.append(1))
    worker.cached("a", lambda: loads.append(1))
    assert loads == [1, 1]
