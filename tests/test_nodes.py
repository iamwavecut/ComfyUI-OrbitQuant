import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import comfyui_orbitquant.nodes as nodes


def _orbitquant_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("orbitquant")
    from orbitquant import OrbitQuantConfig
    from orbitquant.layers import OrbitQuantLinear
    from orbitquant.pipeline import quantize_pipeline, save_quantized_pipeline_component

    return (
        torch,
        OrbitQuantConfig,
        OrbitQuantLinear,
        quantize_pipeline,
        save_quantized_pipeline_component,
    )


def _tiny_pipeline(torch):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

    return TinyPipeline()


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


def test_readme_documents_kernel_extra_for_auto_fused_runtime():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert 'runtime_mode="auto_fused"' in readme
    assert "git clone https://github.com/iamwavecut/ComfyUI-OrbitQuant.git" in readme
    assert "git@github.com:iamwavecut/ComfyUI-OrbitQuant.git" not in readme
    assert 'python -m pip install "orbitquant[kernels]>=0.1.2"' in readme
    assert 'python -m pip install -e "/path/to/OrbitQuant[kernels]"' in readme
    assert 'runtime_mode="dequant_bf16"' in readme
    assert "packed kernels are not installed" in readme


def test_pyproject_depends_on_public_orbitquant_release():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"orbitquant>=0.1.2"' in pyproject
    assert "git+ssh://git@github.com/iamwavecut/OrbitQuant.git" not in pyproject


def test_missing_orbitquant_dependency_message_is_actionable(monkeypatch):
    def raise_missing_dependency():
        raise nodes._missing_orbitquant_error()

    monkeypatch.setattr(nodes, "_orbitquant_manifest_cls", raise_missing_dependency)

    with pytest.raises(RuntimeError, match="Install OrbitQuant"):
        nodes.read_manifest("/tmp/orbitquant-artifact")


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
    input_names = [input_spec["name"] for input_spec in schema.kwargs["inputs"]]
    assert input_names == [
        "pipeline",
        "artifact_path",
        "component",
        "strict",
        "runtime_mode",
        "activation_kernel_backend",
    ]
    runtime_input = schema.kwargs["inputs"][4]
    activation_input = schema.kwargs["inputs"][5]
    assert runtime_input["options"][0] == "auto_fused"
    assert runtime_input["default"] == "auto_fused"
    assert activation_input["options"][0] == "auto"
    assert activation_input["default"] == "auto"


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
        lambda self,
        pipeline_arg,
        artifact_path,
        strict,
        runtime_mode,
        activation_kernel_backend: (
            pipeline_arg,
            {
                "artifact_path": artifact_path,
                "strict": strict,
                "runtime_mode": runtime_mode,
                "activation_kernel_backend": activation_kernel_backend,
            },
        ),
    )

    inspect_output = v3.OrbitQuantArtifactInspectorV3.execute("/tmp/artifact")
    load_output = v3.OrbitQuantFluxLoaderV3.execute(
        pipeline,
        "/tmp/flux",
        True,
        "auto_fused",
        "auto",
    )

    assert inspect_output.values == ("summary:/tmp/artifact", {"artifact_path": "/tmp/artifact"})
    assert load_output.values == (
        pipeline,
        {
            "artifact_path": "/tmp/flux",
            "strict": True,
            "runtime_mode": "auto_fused",
            "activation_kernel_backend": "auto",
        },
    )


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

    def fake_loader(
        pipeline_arg,
        artifact_arg,
        *,
        component,
        strict,
        runtime_mode,
        activation_kernel_backend,
    ):
        calls.append(
            (
                pipeline_arg,
                Path(artifact_arg),
                component,
                strict,
                runtime_mode,
                activation_kernel_backend,
            )
        )
        return manifest

    monkeypatch.setattr(nodes, "load_quantized_pipeline_component", fake_loader)

    returned_pipeline, payload = nodes.OrbitQuantPipelineComponentLoader().load(
        pipeline,
        str(artifact_dir),
        "transformer",
        True,
    )

    assert returned_pipeline is pipeline
    assert calls == [(pipeline, artifact_dir, "transformer", True, "auto_fused", "auto")]
    assert payload["source_model_id"] == "example/model"
    assert payload["bits"] == "W4A4"
    assert payload["requested_runtime_mode"] == "auto_fused"
    assert payload["requested_activation_kernel_backend"] == "auto"


def test_loader_attaches_real_orbitquant_component_artifact(tmp_path):
    (
        torch,
        OrbitQuantConfig,
        OrbitQuantLinear,
        quantize_pipeline,
        save_quantized_pipeline_component,
    ) = _orbitquant_runtime()
    source_pipeline = _tiny_pipeline(torch)
    config = OrbitQuantConfig(
        block_size=4,
        target_policy="generic_dit",
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    restored_pipeline = _tiny_pipeline(torch)
    returned_pipeline, payload = nodes.OrbitQuantPipelineComponentLoader().load(
        restored_pipeline,
        str(tmp_path),
        "transformer",
        True,
        "debug_no_activation_quant",
        "auto",
    )

    restored_layer = returned_pipeline.transformer.transformer_blocks[0]["attn"]["to_q"]
    assert returned_pipeline is restored_pipeline
    assert isinstance(restored_layer, OrbitQuantLinear)
    assert restored_layer.runtime_mode == "debug_no_activation_quant"
    assert restored_layer.activation_kernel_backend == "auto"
    assert torch.isfinite(restored_layer(torch.randn(1, 2, 8))).all()
    assert payload["source_model_id"] == "example/model"
    assert payload["artifact_component"] == "transformer"
    assert payload["requested_runtime_mode"] == "debug_no_activation_quant"
    assert payload["requested_activation_kernel_backend"] == "auto"


def test_node_graph_smoke_inspects_then_loads_real_orbitquant_artifact(tmp_path):
    (
        torch,
        OrbitQuantConfig,
        OrbitQuantLinear,
        quantize_pipeline,
        save_quantized_pipeline_component,
    ) = _orbitquant_runtime()
    source_pipeline = _tiny_pipeline(torch)
    config = OrbitQuantConfig(
        block_size=4,
        target_policy="generic_dit",
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    inspect_summary, inspect_info = nodes.OrbitQuantArtifactInspector().inspect(str(tmp_path))
    restored_pipeline = _tiny_pipeline(torch)
    loaded_pipeline, load_info = nodes.OrbitQuantPipelineComponentLoader().load(
        restored_pipeline,
        str(tmp_path),
        "transformer",
        True,
        "debug_no_activation_quant",
        "auto",
    )

    restored_layer = loaded_pipeline.transformer.transformer_blocks[0]["attn"]["to_q"]
    assert "example/model" in inspect_summary
    assert inspect_info["valid"] is True
    assert inspect_info["artifact_component"] == "transformer"
    assert inspect_info["quantized_module_count"] == 1
    assert load_info["source_model_id"] == inspect_info["source_model_id"]
    assert load_info["artifact_component"] == inspect_info["artifact_component"]
    assert load_info["quantized_module_count"] == inspect_info["quantized_module_count"]
    assert load_info["requested_runtime_mode"] == "debug_no_activation_quant"
    assert isinstance(restored_layer, OrbitQuantLinear)
    assert torch.isfinite(restored_layer(torch.randn(1, 2, 8))).all()


def test_loader_input_types_default_to_auto_fused_runtime():
    pipeline_inputs = nodes.OrbitQuantPipelineComponentLoader.INPUT_TYPES()["required"]
    flux_inputs = nodes.OrbitQuantFluxLoader.INPUT_TYPES()["required"]

    assert pipeline_inputs["runtime_mode"][1]["default"] == "auto_fused"
    assert pipeline_inputs["runtime_mode"][0][0] == "auto_fused"
    assert pipeline_inputs["activation_kernel_backend"][1]["default"] == "auto"
    assert pipeline_inputs["activation_kernel_backend"][0][0] == "auto"
    assert flux_inputs["runtime_mode"][1]["default"] == "auto_fused"
    assert flux_inputs["activation_kernel_backend"][1]["default"] == "auto"


def test_loader_rejects_invalid_runtime_options(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    with pytest.raises(ValueError, match="runtime_mode"):
        nodes.OrbitQuantPipelineComponentLoader().load(
            object(),
            str(artifact_dir),
            "transformer",
            True,
            "unknown",
            "auto",
        )

    with pytest.raises(ValueError, match="activation_kernel_backend"):
        nodes.OrbitQuantPipelineComponentLoader().load(
            object(),
            str(artifact_dir),
            "transformer",
            True,
            "auto_fused",
            "unknown",
        )


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

    def fake_loader(
        pipeline_arg,
        artifact_arg,
        *,
        component,
        strict,
        runtime_mode,
        activation_kernel_backend,
    ):
        calls.append(
            (
                pipeline_arg,
                Path(artifact_arg),
                component,
                strict,
                runtime_mode,
                activation_kernel_backend,
            )
        )
        return manifest

    monkeypatch.setattr(nodes, "read_manifest", lambda artifact_arg: manifest)
    monkeypatch.setattr(nodes, "load_quantized_pipeline_component", fake_loader)

    returned_pipeline, payload = loader_cls()().load(
        pipeline,
        str(artifact_dir),
        True,
        "native_packed_matmul",
        "mps",
    )

    assert returned_pipeline is pipeline
    assert calls == [(pipeline, artifact_dir, "transformer", True, "native_packed_matmul", "mps")]
    assert payload["loader_target"] == target
    assert payload["bits"] == "W4A4"
    assert payload["requested_runtime_mode"] == "native_packed_matmul"
    assert payload["requested_activation_kernel_backend"] == "mps"


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
        lambda pipeline_arg,
        artifact_arg,
        *,
        component,
        strict,
        runtime_mode,
        activation_kernel_backend: manifest,
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

    def fail_loader(
        pipeline_arg,
        artifact_arg,
        *,
        component,
        strict,
        runtime_mode,
        activation_kernel_backend,
    ):
        calls.append(
            (
                pipeline_arg,
                artifact_arg,
                component,
                strict,
                runtime_mode,
                activation_kernel_backend,
            )
        )
        raise AssertionError("mismatched artifact should be rejected before loading")

    monkeypatch.setattr(nodes, "read_manifest", lambda artifact_arg: manifest)
    monkeypatch.setattr(nodes, "load_quantized_pipeline_component", fail_loader)

    with pytest.raises(ValueError, match="OrbitQuant FLUX Loader"):
        nodes.OrbitQuantFluxLoader().load(pipeline, str(artifact_dir), True)
    assert calls == []


def test_loader_rejects_empty_artifact_path():
    with pytest.raises(ValueError, match="artifact_path"):
        nodes.OrbitQuantPipelineComponentLoader().load(object(), "", "transformer", True)
