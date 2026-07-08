import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    assert nodes.NODE_CLASS_MAPPINGS["OrbitQuantFluxLoader"] is nodes.OrbitQuantFluxLoader
    assert nodes.NODE_CLASS_MAPPINGS["OrbitQuantZImageLoader"] is nodes.OrbitQuantZImageLoader
    assert nodes.NODE_CLASS_MAPPINGS["OrbitQuantWanLoader"] is nodes.OrbitQuantWanLoader
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["OrbitQuantFluxLoader"] == (
        "OrbitQuant FLUX Loader"
    )


def test_root_init_exposes_comfyui_node_mappings():
    root_init = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("comfyui_orbitquant_root", root_init)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert "OrbitQuantArtifactInspector" in module.NODE_CLASS_MAPPINGS
    assert callable(module.comfy_entrypoint)


def _install_fake_comfy_api(monkeypatch):
    class FakeComfyExtension:
        pass

    class FakeComfyNode:
        pass

    class FakeSchema:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeNodeOutput:
        def __init__(self, *values, ui=None):
            self.values = values
            self.ui = ui

    class FakeType:
        def __init__(self, type_name):
            self.type_name = type_name

        def Input(self, name, **kwargs):
            return {"kind": "input", "type": self.type_name, "name": name, **kwargs}

        def Output(self, **kwargs):
            return {"kind": "output", "type": self.type_name, **kwargs}

    fake_io = SimpleNamespace(
        ComfyNode=FakeComfyNode,
        Schema=FakeSchema,
        NodeOutput=FakeNodeOutput,
        String=FakeType("STRING"),
        Boolean=FakeType("BOOLEAN"),
        Combo=FakeType("COMBO"),
        Custom=lambda type_name: FakeType(type_name),
    )
    fake_latest = ModuleType("comfy_api.latest")
    fake_latest.ComfyExtension = FakeComfyExtension
    fake_latest.io = fake_io
    fake_api = ModuleType("comfy_api")
    fake_api.latest = fake_latest
    monkeypatch.setitem(sys.modules, "comfy_api", fake_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", fake_latest)
    sys.modules.pop("comfyui_orbitquant.v3", None)


def test_v3_entrypoint_exposes_modern_comfyui_nodes(monkeypatch):
    _install_fake_comfy_api(monkeypatch)
    v3 = importlib.import_module("comfyui_orbitquant.v3")

    extension = asyncio.run(v3.comfy_entrypoint())
    node_list = asyncio.run(extension.get_node_list())
    schema = v3.OrbitQuantPipelineComponentLoaderV3.define_schema()

    assert [node.__name__ for node in node_list] == [
        "OrbitQuantArtifactInspectorV3",
        "OrbitQuantPipelineComponentLoaderV3",
        "OrbitQuantFluxLoaderV3",
        "OrbitQuantZImageLoaderV3",
        "OrbitQuantWanLoaderV3",
    ]
    assert schema.kwargs["node_id"] == "OrbitQuantPipelineComponentLoader"
    assert schema.kwargs["display_name"] == "OrbitQuant Pipeline Component Loader"
    assert schema.kwargs["category"] == "OrbitQuant"
    assert [output["type"] for output in schema.kwargs["outputs"]] == [
        "PIPELINE",
        "ORBITQUANT_INFO",
    ]


def test_v3_nodes_delegate_to_legacy_implementations(monkeypatch):
    _install_fake_comfy_api(monkeypatch)
    v3 = importlib.import_module("comfyui_orbitquant.v3")
    pipeline = object()

    monkeypatch.setattr(
        nodes.OrbitQuantArtifactInspector,
        "inspect",
        lambda self, artifact_path: (f"summary:{artifact_path}", {"artifact_path": artifact_path}),
    )
    monkeypatch.setattr(
        nodes.OrbitQuantFluxLoader,
        "load",
        lambda self, pipeline_arg, artifact_path, strict: (
            pipeline_arg,
            {"artifact_path": artifact_path, "strict": strict},
        ),
    )

    inspect_output = v3.OrbitQuantArtifactInspectorV3.execute("/tmp/artifact")
    load_output = v3.OrbitQuantFluxLoaderV3.execute(pipeline, "/tmp/flux", True)

    assert inspect_output.values == ("summary:/tmp/artifact", {"artifact_path": "/tmp/artifact"})
    assert load_output.values == (pipeline, {"artifact_path": "/tmp/flux", "strict": True})


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
    monkeypatch.setattr(
        nodes,
        "read_model_index",
        lambda path: {"component": "denoiser", "weight_name": "model.safetensors"},
    )

    text, payload = nodes.OrbitQuantArtifactInspector().inspect(str(artifact_dir))

    assert "example/model" in text
    assert "W4A4" in text
    assert "Component: denoiser" in text
    assert payload["valid"] is True
    assert payload["artifact_component"] == "denoiser"
    assert payload["artifact_weight_name"] == "model.safetensors"
    assert payload["quantized_module_count"] == 2
    assert payload["adaln_module_count"] == 1
    assert payload["skipped_module_count"] == 1
    assert payload["quantized_modules"] == ["a", "b"]
    assert payload["adaln_modules"] == ["c"]
    assert payload["skipped_modules"] == ["d"]


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


@pytest.mark.parametrize(
    ("loader_cls", "target"),
    [
        (lambda: nodes.OrbitQuantFluxLoader, "flux"),
        (lambda: nodes.OrbitQuantZImageLoader, "z_image"),
        (lambda: nodes.OrbitQuantWanLoader, "wan"),
    ],
)
def test_specialized_loaders_load_transformer_component(monkeypatch, tmp_path, loader_cls, target):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pipeline = object()
    manifest = SimpleNamespace(
        source_model_id=f"example/{target}",
        weight_bits=4,
        activation_bits=4,
        target_policy=target,
    )
    calls = []

    def fake_loader(pipeline_arg, artifact_arg, *, component, strict):
        calls.append((pipeline_arg, Path(artifact_arg), component, strict))
        return manifest

    monkeypatch.setattr(nodes, "read_manifest", lambda artifact_arg: manifest)
    monkeypatch.setattr(nodes, "load_quantized_pipeline_component", fake_loader)

    returned_pipeline, payload = loader_cls()().load(pipeline, str(artifact_dir), True)

    assert returned_pipeline is pipeline
    assert calls == [(pipeline, artifact_dir, "transformer", True)]
    assert payload["loader_target"] == target
    assert payload["bits"] == "W4A4"


def test_flux_loader_accepts_flux2_artifacts(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pipeline = object()
    manifest = SimpleNamespace(
        source_model_id="black-forest-labs/FLUX.2-klein-4B",
        weight_bits=4,
        activation_bits=4,
        target_policy="flux2",
    )

    monkeypatch.setattr(
        nodes,
        "read_manifest",
        lambda artifact_arg: manifest,
    )
    monkeypatch.setattr(
        nodes,
        "load_quantized_pipeline_component",
        lambda pipeline_arg, artifact_arg, *, component, strict: manifest,
    )

    returned_pipeline, payload = nodes.OrbitQuantFluxLoader().load(
        pipeline, str(artifact_dir), True
    )

    assert returned_pipeline is pipeline
    assert payload["loader_target"] == "flux"
    assert payload["target_policy"] == "flux2"


def test_specialized_loader_rejects_mismatched_target_policy(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pipeline = object()
    manifest = SimpleNamespace(
        source_model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        weight_bits=4,
        activation_bits=4,
        target_policy="wan",
    )

    calls = []

    def fail_loader(pipeline_arg, artifact_arg, *, component, strict):
        calls.append((pipeline_arg, artifact_arg, component, strict))
        raise AssertionError("mismatched artifact should be rejected before loading")

    monkeypatch.setattr(nodes, "read_manifest", lambda artifact_arg: manifest)
    monkeypatch.setattr(nodes, "load_quantized_pipeline_component", fail_loader)

    with pytest.raises(ValueError, match="OrbitQuant FLUX Loader"):
        nodes.OrbitQuantFluxLoader().load(pipeline, str(artifact_dir), True)
    assert calls == []


def test_loader_rejects_empty_artifact_path():
    with pytest.raises(ValueError, match="artifact_path"):
        nodes.OrbitQuantPipelineComponentLoader().load(object(), "", "transformer", True)
