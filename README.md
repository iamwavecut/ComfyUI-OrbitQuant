# ComfyUI-OrbitQuant

ComfyUI custom nodes for inspecting OrbitQuant artifacts and attaching a
quantized transformer component to an existing pipeline object.

The quantization implementation lives in the `orbitquant` Python package. This
node pack only validates artifacts, reports metadata, and calls OrbitQuant's
component-loading API.

## Nodes

| Node | Purpose |
| --- | --- |
| `OrbitQuant Inspect Artifact` | Validate an OrbitQuant artifact directory and return a text summary plus structured metadata. |
| `OrbitQuant Pipeline Component Loader` | Attach an OrbitQuant component artifact to a pipeline attribute such as `transformer`. |
| `OrbitQuant FLUX Loader` | Attach a FLUX or FLUX.2 transformer artifact and reject non-FLUX policies. |
| `OrbitQuant Z-Image Loader` | Attach a Z-Image transformer artifact and reject other target policies. |
| `OrbitQuant Wan Loader` | Attach a Wan transformer artifact and reject other target policies. |

The same nodes are exposed through the legacy `NODE_CLASS_MAPPINGS` interface
and the modern ComfyUI V3 `comfy_entrypoint` interface when `comfy_api` is
available.

## Install

Clone this repository into ComfyUI's custom node directory:

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:iamwavecut/ComfyUI-OrbitQuant.git
```

Install `orbitquant` into the Python environment used by ComfyUI:

```bash
python -m pip install "orbitquant @ git+ssh://git@github.com/iamwavecut/OrbitQuant.git@orbitquant-mvp"
```

Restart ComfyUI after installation.

## Usage

Use an OrbitQuant artifact directory produced by the OrbitQuant package or
downloaded from Hugging Face.

1. Load or create the source Diffusers pipeline in your workflow.
2. Add the matching OrbitQuant loader node.
3. Set `artifact_path` to the local artifact directory.
4. Connect the pipeline object into the loader node.
5. Use the returned pipeline object for the downstream generation nodes.

For model-specific loaders, the artifact `target_policy` is checked before the
component is attached:

| Loader | Accepted `target_policy` |
| --- | --- |
| `OrbitQuant FLUX Loader` | `flux`, `flux2` |
| `OrbitQuant Z-Image Loader` | `z_image` |
| `OrbitQuant Wan Loader` | `wan` |

## Artifact Requirements

The loader expects the standard OrbitQuant component artifact layout:

```text
artifact/
  README.md
  SHA256SUMS
  model_index.json
  model.safetensors
  quantization_config.json
  orbitquant_manifest.json
  orbitquant_codebooks.safetensors
  orbitquant_rotations.safetensors
  prompts.json
  benchmark/summary.json
```

`OrbitQuant Inspect Artifact` validates required files, checksums, tensor
shapes, source model metadata, bit settings, runtime mode, target policy, and
module counts.

## Python API

The node classes can also be called from Python when building a custom ComfyUI
workflow wrapper:

```python
from comfyui_orbitquant.nodes import OrbitQuantFluxLoader

pipeline, info = OrbitQuantFluxLoader().load(
    pipeline,
    "/models/orbitquant/flux2-klein-w4a4",
    True,
)
```

The nodes delegate artifact parsing and component loading to OrbitQuant:

```python
from orbitquant.artifacts import OrbitQuantManifest, validate_orbitquant_artifact
from orbitquant.pipeline import load_quantized_pipeline_component
```

No quantization math or artifact parsing logic is duplicated in this repository.
