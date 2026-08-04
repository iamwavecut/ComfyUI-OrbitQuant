from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from comfyui_orbitquant import minimax_h3, nodes


def _make_release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    (release / "quantization_manifest.json").write_text(
        json.dumps(
            {
                "artifact_format": "orbitquant-multicomponent-v1",
                "repo_id": "WaveCut/MiniMax-H3-OrbitQuant-W4A4",
                "source_model_id": "MiniMaxAI/MiniMax-H3",
                "source_revision": "73372e6",
                "weight_bits": 4,
                "activation_bits": 4,
                "w4a4_components": ["transformer", "transformer_ref", "text_encoder"],
                "source_precision_components": ["vae", "audio_vae"],
            }
        ),
        encoding="utf-8",
    )
    for component in ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"):
        (release / component).mkdir()
    scripts = release / "scripts"
    scripts.mkdir()
    for name in ("run_quantized_example.py", "decode_h3_latents.py"):
        (scripts / name).write_text("# fixture\n", encoding="utf-8")
    return release


def test_release_descriptor_accepts_exact_h3_component_policy(tmp_path):
    release_path = _make_release(tmp_path)

    release = minimax_h3.MiniMaxH3Release.from_path(release_path)

    assert release.path == release_path.resolve()
    assert release.repo_id == "WaveCut/MiniMax-H3-OrbitQuant-W4A4"
    assert release.bits == "W4A4"
    assert release.w4a4_components == ("text_encoder", "transformer", "transformer_ref")
    assert release.source_precision_components == ("audio_vae", "vae")
    assert release.summary["valid"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda release: (release / "quantization_manifest.json").unlink(), "manifest"),
        (
            lambda release: _rewrite_manifest(release, weight_bits=8),
            "W4A4",
        ),
        (
            lambda release: _rewrite_manifest(release, w4a4_components=["transformer"]),
            "w4a4_components",
        ),
        (
            lambda release: _rewrite_manifest(release, source_precision_components=["vae"]),
            "source_precision_components",
        ),
        (lambda release: (release / "scripts" / "decode_h3_latents.py").unlink(), "decoder"),
        (lambda release: (release / "transformer_ref").rmdir(), "transformer_ref"),
    ],
)
def test_release_descriptor_rejects_incomplete_or_wrong_release(tmp_path, mutation, message):
    release_path = _make_release(tmp_path)
    mutation(release_path)

    with pytest.raises(ValueError, match=message):
        minimax_h3.MiniMaxH3Release.from_path(release_path)


def _rewrite_manifest(release: Path, **updates) -> None:
    manifest_path = release / "quantization_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(updates)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_release_loader_returns_descriptor_and_json_summary(tmp_path):
    release_path = _make_release(tmp_path)

    release, summary_json = minimax_h3.OrbitQuantMiniMaxH3ReleaseLoader().load(
        str(release_path)
    )

    assert isinstance(release, minimax_h3.MiniMaxH3Release)
    summary = json.loads(summary_json)
    assert summary["repo_id"] == "WaveCut/MiniMax-H3-OrbitQuant-W4A4"
    assert summary["bits"] == "W4A4"
    assert summary["vae_policy"] == "source_precision_fp32_decode"


def test_runner_persists_latents_then_decodes_with_source_fp32_vaes(tmp_path, monkeypatch):
    release = minimax_h3.MiniMaxH3Release.from_path(_make_release(tmp_path))
    output_dir = tmp_path / "output"
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        if "run_quantized_example.py" in command[1]:
            latents = Path(command[command.index("--save-latents") + 1])
            latents.parent.mkdir(parents=True, exist_ok=True)
            latents.write_bytes(b"latents")
            output = Path(command[command.index("--output") + 1])
            output.with_suffix(".metrics.json").write_text(
                json.dumps({"status": "pass", "generation_seconds": 12.5}),
                encoding="utf-8",
            )
        else:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"mp4")
            output.with_suffix(".decode.metrics.json").write_text(
                json.dumps({"status": "pass", "decode_seconds": 1.5}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(minimax_h3.subprocess, "run", fake_run)

    result = minimax_h3.MiniMaxH3Runner(release).run(
        output_dir=output_dir,
        filename="proof.mp4",
        prompt="dynamic dolly zoom",
        task="t2va",
        reference_path="",
        seed=42,
        width=608,
        height=480,
        num_frames=44,
        steps=50,
    )

    assert len(commands) == 2
    assert [environment["PYTHONUNBUFFERED"] for environment in environments] == ["1", "1"]
    generation, decode = commands
    assert generation[0] == minimax_h3.sys.executable
    assert "--manual-stage-offload" in generation
    assert generation[generation.index("--transformer-runtime-mode") + 1] == "auto_fused"
    assert generation[generation.index("--width") + 1] == "608"
    assert generation[generation.index("--height") + 1] == "480"
    assert generation[generation.index("--steps") + 1] == "50"
    assert generation[generation.index("--num-frames") + 1] == "44"
    assert generation.index("--save-latents") < generation.index("--checkpoint-dir")
    assert decode[decode.index("--vae") + 1] == str(release.path / "vae")
    assert decode[decode.index("--vae-dtype") + 1] == "fp32"
    assert decode[decode.index("--audio-vae") + 1] == str(release.path / "audio_vae")
    assert decode[decode.index("--audio-vae-dtype") + 1] == "fp32"
    assert result.output_path == output_dir / "proof.mp4"
    assert result.latents_path.is_file()
    assert result.generation_metrics["generation_seconds"] == 12.5
    assert result.decode_metrics["decode_seconds"] == 1.5
    assert result.report_path.is_file()


def test_runner_adds_ref2va_reference_and_rejects_missing_reference(tmp_path, monkeypatch):
    release = minimax_h3.MiniMaxH3Release.from_path(_make_release(tmp_path))
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"png")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "run_quantized_example.py" in command[1]:
            Path(command[command.index("--save-latents") + 1]).write_bytes(b"latents")
            output = Path(command[command.index("--output") + 1])
            output.with_suffix(".metrics.json").write_text('{"status":"pass"}')
        else:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"mp4")
            output.with_suffix(".decode.metrics.json").write_text('{"status":"pass"}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(minimax_h3.subprocess, "run", fake_run)
    runner = minimax_h3.MiniMaxH3Runner(release)

    runner.run(
        output_dir=tmp_path / "out",
        filename="ref.mp4",
        prompt="zoom",
        task="ref2va",
        reference_path=str(reference),
        seed=7,
        width=608,
        height=480,
        num_frames=44,
        steps=50,
    )

    generation = commands[0]
    assert generation[generation.index("--task") + 1] == "ref2va"
    assert generation[generation.index("--reference") + 1] == str(reference.resolve())

    with pytest.raises(ValueError, match="reference"):
        runner.run(
            output_dir=tmp_path / "out2",
            filename="bad.mp4",
            prompt="zoom",
            task="ref2va",
            reference_path="",
            seed=7,
            width=608,
            height=480,
            num_frames=44,
            steps=50,
        )


def test_runner_surfaces_failed_stage_and_log_path(tmp_path, monkeypatch):
    release = minimax_h3.MiniMaxH3Release.from_path(_make_release(tmp_path))

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(minimax_h3.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match=r"generation.*\.generate\.log"):
        minimax_h3.MiniMaxH3Runner(release).run(
            output_dir=tmp_path / "out",
            filename="failure.mp4",
            prompt="zoom",
            task="t2va",
            reference_path="",
            seed=1,
            width=608,
            height=480,
            num_frames=44,
            steps=50,
        )


def test_legacy_node_mappings_expose_h3_loader_and_generator():
    assert nodes.NODE_CLASS_MAPPINGS["OrbitQuantMiniMaxH3ReleaseLoader"] is (
        minimax_h3.OrbitQuantMiniMaxH3ReleaseLoader
    )
    assert nodes.NODE_CLASS_MAPPINGS["OrbitQuantMiniMaxH3GenerateVideo"] is (
        minimax_h3.OrbitQuantMiniMaxH3GenerateVideo
    )
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["OrbitQuantMiniMaxH3GenerateVideo"] == (
        "OrbitQuant MiniMax H3 Generate Video"
    )


def test_generator_node_returns_standard_video_and_preview(tmp_path, monkeypatch):
    release = minimax_h3.MiniMaxH3Release.from_path(_make_release(tmp_path))
    output_root = tmp_path / "comfy-output"
    output_root.mkdir()
    video_marker = object()

    fake_folder_paths = ModuleType("folder_paths")
    fake_folder_paths.get_output_directory = lambda: str(output_root)
    fake_folder_paths.get_save_image_path = (
        lambda prefix, output_dir, width, height: (
            str(output_root / "orbitquant"),
            "minimax-h3",
            3,
            "orbitquant",
            prefix,
        )
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)

    class FakeInputImpl:
        @staticmethod
        def VideoFromFile(path):
            assert Path(path).name == "minimax-h3_00003_.mp4"
            return video_marker

    fake_latest = ModuleType("comfy_api.latest")
    fake_latest.InputImpl = FakeInputImpl
    fake_comfy_api = ModuleType("comfy_api")
    fake_comfy_api.latest = fake_latest
    monkeypatch.setitem(sys.modules, "comfy_api", fake_comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", fake_latest)

    calls = []

    def fake_run(self, **kwargs):
        calls.append(kwargs)
        output_path = Path(kwargs["output_dir"]) / kwargs["filename"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")
        report_path = output_path.with_suffix(".comfyui.json")
        report_path.write_text('{"status":"pass"}', encoding="utf-8")
        return SimpleNamespace(output_path=output_path, report={"status": "pass"})

    monkeypatch.setattr(minimax_h3.MiniMaxH3Runner, "run", fake_run)

    response = minimax_h3.OrbitQuantMiniMaxH3GenerateVideo().generate(
        release,
        "dynamic dolly zoom",
        "t2va",
        "",
        42,
        608,
        480,
        44,
        50,
        "orbitquant/minimax-h3",
    )

    assert calls[0]["width"] == 608
    assert calls[0]["height"] == 480
    assert calls[0]["steps"] == 50
    assert response["result"][0] is video_marker
    assert json.loads(response["result"][1])["status"] == "pass"
    assert response["ui"]["images"] == [
        {"filename": "minimax-h3_00003_.mp4", "subfolder": "orbitquant", "type": "output"}
    ]
    assert response["ui"]["animated"] == (True,)
