import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import comfyui_orbitquant.nodes as nodes


def test_node_mappings_expose_loader_and_inspector():
    assert (
        nodes.NODE_CLASS_MAPPINGS["OrbitQuantArtifactInspector"]
        is nodes.OrbitQuantArtifactInspector
    )
    assert nodes.NODE_CLASS_MAPPINGS["OrbitQuantPipelineComponentLoader"] is (
        nodes.OrbitQuantPipelineComponentLoader
    )
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["OrbitQuantArtifactInspector"] == (
        "OrbitQuant Inspect Artifact"
    )


def test_root_init_exposes_comfyui_node_mappings():
    root_init = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("comfyui_orbitquant_root", root_init)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert "OrbitQuantArtifactInspector" in module.NODE_CLASS_MAPPINGS


def test_inspector_reports_manifest_and_validation(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    manifest = SimpleNamespace(
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        weight_bits=4,
        activation_bits=4,
        target_policy="flux2",
        runtime_mode="dequant_bf16",
        activation_kernel_backend="auto",
        quantized_modules=["a", "b"],
        adaln_modules=["c"],
        skipped_modules=["d"],
    )

    monkeypatch.setattr(nodes, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(
        nodes,
        "validate_orbitquant_artifact",
        lambda path: {"valid": True, "tensor_count": 7},
    )

    text, payload = nodes.OrbitQuantArtifactInspector().inspect(str(artifact_dir))

    assert "example/model" in text
    assert "W4A4" in text
    assert payload["valid"] is True
    assert payload["quantized_module_count"] == 2
    assert payload["adaln_module_count"] == 1
    assert payload["skipped_module_count"] == 1


def test_loader_calls_orbitquant_pipeline_component_loader(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pipeline = object()
    manifest = SimpleNamespace(source_model_id="example/model", weight_bits=4, activation_bits=4)
    calls = []

    def fake_loader(pipeline_arg, artifact_arg, *, component, strict):
        calls.append((pipeline_arg, Path(artifact_arg), component, strict))
        return manifest

    monkeypatch.setattr(nodes, "load_quantized_pipeline_component", fake_loader)

    returned_pipeline, payload = nodes.OrbitQuantPipelineComponentLoader().load(
        pipeline,
        str(artifact_dir),
        "transformer",
        True,
    )

    assert returned_pipeline is pipeline
    assert calls == [(pipeline, artifact_dir, "transformer", True)]
    assert payload["source_model_id"] == "example/model"
    assert payload["bits"] == "W4A4"


def test_loader_rejects_empty_artifact_path():
    with pytest.raises(ValueError, match="artifact_path"):
        nodes.OrbitQuantPipelineComponentLoader().load(object(), "", "transformer", True)
