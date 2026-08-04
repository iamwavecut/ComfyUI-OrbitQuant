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
| `OrbitQuant Pipeline Component Loader` | Attach any compatible `universal` or model-specific OrbitQuant component artifact to a pipeline attribute such as `transformer`. |
| `OrbitQuant FLUX Loader` | Attach a FLUX or FLUX.2 transformer artifact and reject non-FLUX policies. |
| `OrbitQuant Z-Image Loader` | Attach a Z-Image transformer artifact and reject other target policies. |
| `OrbitQuant Wan Loader` | Attach a Wan transformer artifact and reject other target policies. |
| `OrbitQuant MiniMax H3 Release Loader` | Validate the published multicomponent H3 W4A4 release and its component precision policy. |
| `OrbitQuant MiniMax H3 Generate Video` | Run T2VA or Ref2VA with durable latents/checkpoints, manual stage offload, source FP32 VAEs, and a standard ComfyUI video preview. |

The same nodes are exposed through the legacy `NODE_CLASS_MAPPINGS` interface
and the modern ComfyUI V3 `comfy_entrypoint` interface when `comfy_api` is
available.

## Install

Install through ComfyUI-Manager, or clone this repository into ComfyUI's
custom node directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iamwavecut/ComfyUI-OrbitQuant.git
```

ComfyUI-Manager installs `requirements.txt` (the `orbitquant` package) and then
runs `install.py`, which provisions the optimized native kernel package for
the current runtime by downloading the matching prebuilt variant wheel from
the OrbitQuant GitHub release. Provisioning is best effort: when no variant
matches the runtime, packed runtime modes fall back to OrbitQuant's Triton or
dequantized paths and the node pack keeps working.

For a manual clone, install the `orbitquant` package into the Python
environment used by ComfyUI and provision the native kernels explicitly:

```bash
python -m pip install "orbitquant>=0.9.1,<0.10"
python -m orbitquant.cli.main kernels-install
```

For the default optimized `runtime_mode="auto_fused"` path on CUDA, install
OrbitQuant with its kernel runtime extra. This provides the Triton fallback
used when no native variant matches:

```bash
python -m pip install "orbitquant[hf,kernels]>=0.9.1,<0.10"
```

If you install this node pack from PyPI, the same kernel runtime dependencies
are available through the node pack extra:

```bash
python -m pip install "comfyui-orbitquant[kernels]"
```

For a source checkout, install the package from the local OrbitQuant repository:

```bash
python -m pip install -e /path/to/OrbitQuant
```

For a source checkout with the kernel runtime dependencies:

```bash
python -m pip install -e "/path/to/OrbitQuant[kernels]"
```

Restart ComfyUI after installation.

## Usage

Use an OrbitQuant artifact directory produced by the OrbitQuant package or
downloaded from Hugging Face.

1. Load or create the source Diffusers pipeline in your workflow.
2. Add the matching OrbitQuant loader node.
3. Set `artifact_path` to the local artifact directory.
4. Connect the pipeline object into the loader node.
5. Keep `runtime_mode` at `auto_fused` for optimized packed-weight inference.
6. Use the returned pipeline object for the downstream generation nodes.

For model-specific loaders, the artifact `target_policy` is checked before the
component is attached:

| Loader | Accepted `target_policy` |
| --- | --- |
| `OrbitQuant FLUX Loader` | `flux`, `flux2` |
| `OrbitQuant Z-Image Loader` | `z_image` |
| `OrbitQuant Wan Loader` | `wan` |

Use `OrbitQuant Pipeline Component Loader` for artifacts with
`target_policy="universal"` or for future transformer components that do not
have a specialized node. This loader validates the artifact schema without
restricting the source architecture name.

### Runtime Modes

`runtime_mode` defaults to `auto_fused`. On supported devices, OrbitQuant will
use packed low-bit matmul kernels instead of materializing a full BF16/FP16
weight matrix. `activation_kernel_backend` defaults to `auto`; the
`triton_rocm` and `triton_xpu` backends are experimental in OrbitQuant.

Use `runtime_mode="dequant_bf16"` only as an explicit compatibility or debug
path when packed kernels are not installed in the ComfyUI Python environment.

## MiniMax H3 W4A4 video

The H3 nodes consume the Diffusers-native multicomponent release instead of the
older single-component artifact layout described below. Download the private
model into a local directory using the same environment as ComfyUI:

```bash
hf download WaveCut/MiniMax-H3-OrbitQuant-W4A4 \
  --revision e434bbea523349576e7c3d2f6090744aa4597123 \
  --local-dir /models/MiniMax-H3-OrbitQuant-W4A4
python -m pip install "orbitquant[hf,kernels]>=0.9.1,<0.10"
python -m pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc" \
  "transformers>=5.13,<6" accelerate av soundfile
```

Build this two-node graph:

1. Set `OrbitQuant MiniMax H3 Release Loader.model_path` to the downloaded
   directory.
2. Connect its `release` output to `OrbitQuant MiniMax H3 Generate Video`.
3. For T2VA keep `task=t2va`. For Ref2VA choose `ref2va` and set
   `reference_path` to a local image.
4. Use `width=608`, `height=480`, and `steps=50` for the verified 480p recipe.

The official schedule has 50 sigma points and 49 denoiser forwards. The default
44-frame smoke is economical; raise `num_frames` to 124 for the full model-card
duration. The text encoder enters GPU memory for conditioning and is then moved
back to RAM before the selected transformer enters GPU memory.

The node saves generation logs, per-step checkpoints, and the latent bundle as
soon as each exists. Only after denoising succeeds does it decode with the
untouched source FP32 VAEs. The output is a standard ComfyUI `VIDEO`, so the
core preview and downstream video nodes work without VideoHelperSuite.

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

The node classes can also be called directly from Python when building a custom
ComfyUI workflow wrapper.

Inspect an artifact:

```python
from comfyui_orbitquant.nodes import OrbitQuantArtifactInspector

summary, info = OrbitQuantArtifactInspector().inspect(
    "/models/orbitquant/flux1-schnell-w4a4"
)
print(summary)
print(info["target_policy"])
```

Attach a FLUX-family transformer artifact to an existing pipeline object:

```python
from comfyui_orbitquant.nodes import OrbitQuantFluxLoader

pipeline, info = OrbitQuantFluxLoader().load(
    pipeline,
    "/models/orbitquant/flux1-schnell-w4a4",
    strict=True,
    runtime_mode="auto_fused",
    activation_kernel_backend="auto",
)
```

The nodes delegate artifact parsing and component loading to OrbitQuant:

```python
from orbitquant.artifacts import OrbitQuantManifest, validate_orbitquant_artifact
from orbitquant.pipeline import load_quantized_pipeline_component
```

No quantization math or artifact parsing logic is duplicated in this repository.
