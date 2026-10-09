# ComfyUI-OrbitQuant

Generate images, video with audio, and music with WaveCut's OrbitQuant models.
The model loader downloads an immutable Hugging Face revision and selects the matching runtime.
Outputs connect to ComfyUI's standard `SaveImage`, `SaveVideo`, and `SaveAudio` nodes.

## Quick start

1. Install this repository in `ComfyUI/custom_nodes` and install `requirements.txt` with ComfyUI's Python.
2. Prepare the model environment below. Restart ComfyUI.
3. Open **Templates → ComfyUI-OrbitQuant** and select an image, video, or music workflow.
4. Choose a model in **OrbitQuant Model Loader**. Empty `model_path` and `repo_id` use the selected public release.
5. Connect the loader to the matching generator and run the workflow.

The [example workflows](example_workflows) include Kandinsky at 320p and 480p, FLUX.2 klein, MiniMax H3, and YuE2.
Zero dimensions, frames, or steps use that model's release defaults.

## Models

| Family | Published quantizations | Output | Reference images |
| --- | --- | --- | --- |
| Kandinsky 6 Lite distilled 5s | W4A4 transformer + W4A6 text encoder | Video + audio | — |
| FLUX.2 klein 4B | W4A4, W3A3, W2A4, W2A3 | Image | Yes |
| FLUX.2 klein 9B | W4A4 | Image | Yes |
| FLUX.1 schnell | W4A4, W3A3, W2A4, W2A3 | Image | — |
| Z-Image Turbo | W4A4, W3A3, W2A4, W2A3 | Image | — |
| Wan 2.1 T2V 1.3B | W4A6, W4A4 | Video | — |
| Ideogram v4 Instant | W4A4, W3A3, W2A4, W2A3, W4A6 | Image | — |
| MiniMax H3 | W4A4 | Video + audio | One |
| Krea 2 Turbo | W4A4 | Image | — |
| Turbo Image 2.1 | W4A4 | Image | Yes |
| Boogu Image 0.1 Turbo | W4A8 | Image | Yes |
| YuE2 3B | W4A4 | 48 kHz stereo audio | — |

The catalog contains 26 public repositories. For a compatible private variant, select its base recipe and set `repo_id`.
Use `hf auth login` in the worker environment for private or gated repositories.
Alternatively, set `model_path` to a complete local release. The loader validates its layout before generation.

## Model environments

The new generators run in a separate process. `python_executable` selects an existing Python environment.
An empty value uses ComfyUI's Python. The nodes never install or upgrade packages during generation.
An isolated environment avoids changing dependencies used by other custom nodes.

The common CUDA environment covers the image models, Kandinsky, and Wan. MiniMax H3 and YuE2 use the separate environments described below.

Use Python 3.12 and the following packages for the common environment:

```bash
python3.12 -m venv /path/to/orbitquant-env
source /path/to/orbitquant-env/bin/activate
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install "orbitquant[hf]==0.12.0" \
  "diffusers @ git+https://github.com/huggingface/diffusers.git@d961a388fd02e4db38d17350c8dd9b8abe642e05" \
  "transformers>=5.17,<6" accelerate av imageio-ffmpeg pillow librosa soundfile einops kernels bitsandbytes
python -m orbitquant.cli.main kernels-install
```

Set `python_executable=/path/to/orbitquant-env/bin/python` in the loader.
The published media recipes require NVIDIA CUDA. This release does not claim CPU, MPS, AMD, or Intel generation support.
GPU and host RAM requirements depend on the model. An 8 GB Kandinsky profile does not imply that every model fits 8 GB.

YuE2 uses its bundled OrbitQuant and native kernels in a separate Python 3.12 / Torch 2.10.0 CUDA 12.8 environment.
Follow the [YuE2 setup](https://huggingface.co/WaveCut/YuE2-3B-OrbitQuant-W4A4#run), then select that environment's Python.
Its published kernels support SM89 and SM120.

MiniMax H3 requires Diffusers revision `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`.
Use the same Torch and OrbitQuant versions in a separate environment, with that Diffusers revision instead of the common revision.
The model loader's `python_executable` selects it. Newer Diffusers releases removed APIs used by the published MiniMax runner.

For Boogu, add the pinned official package to the common environment:

```bash
python -m pip install --no-deps \
  "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git@25f8f888298224a94e5ec2abafb98abea9031a0d"
```

The explicit dependency installation above supplies this adapter's runtime. The upstream package's optional optimization paths are outside this recipe.

## Generation controls

- **Kandinsky:** `fast`, `low-memory`, and `exact` use the released OrbitQuant runtime profiles. The distilled recipe uses 10 steps and 121 frames. Dimensions must be multiples of 16. The 320p template uses 576 × 320. `allocator_cap_gib` limits the PyTorch allocator, which excludes CUDA runtime allocations.
- **Turbo Image:** 4–8 steps select the corresponding distilled sigma schedule. The pipeline uses FP16 as specified by its release.
- **Ideogram:** enter plain text or a JSON caption with `high_level_description`. The adapter applies the published compatibility patch inside the worker process. It moves the text encoder to CUDA explicitly because this pipeline bypasses the normal offload hook.
- **Krea:** the adapter uses `no_grad` so its text cache can track tensor versions.
- **MiniMax H3:** use `balanced`, `speed`, or `minimum_vram`. Connect one reference image for Ref2VA. Without a reference, the node selects T2VA.
- **YuE2:** enter style in `prompt` and song text in `lyrics`. Profiles are `exact`, `fast`, `lowmem`, and `turbo`.

Each request releases the worker's CUDA allocations when it finishes. ComfyUI unloads its managed models before the worker starts.
Each request also loads the model again, so end-to-end latency includes process startup and loading.
The first request can include downloads and kernel compilation. Published warm pipeline timings exclude those costs.

Cancellation stops the worker process tree. Failed runs report the generation log path.
Logs, request metadata, and intermediate outputs remain in `ComfyUI/output/orbitquant` for diagnosis.
Use the standard save nodes to retain workflow metadata in the final media.

## ComfyUI integration

The current interface uses V3 `ComfyExtension`, typed schemas, asynchronous execution, and native media outputs.
On older ComfyUI versions, the package exposes the legacy mappings. Existing node identifiers remain unchanged.
The model handle contains only metadata, so ComfyUI does not cache CUDA tensors from an external process.

The new nodes are **OrbitQuant Model Loader**, **OrbitQuant Generate Image**, **OrbitQuant Generate Model Video**, and **OrbitQuant Generate Audio**.
The earlier **OrbitQuant Generate Video** node remains available for existing MiniMax workflows.

This integration follows the current [V3 node API](https://docs.comfy.org/custom-nodes/v3_migration),
[native data types](https://docs.comfy.org/custom-nodes/backend/datatypes), and
[workflow template format](https://docs.comfy.org/custom-nodes/workflow_templates).

## Verified environment

All 26 public catalog entries completed generation through ComfyUI's queue and standard save nodes on Linux with an RTX 4090.
The tested host used ComfyUI 0.39.0, frontend 1.55.14, Python 3.12.3, and Torch 2.12.1 CUDA 13.0.
YuE2 used its separate Torch 2.10.0 CUDA 12.8 worker. MiniMax used the pinned Diffusers environment above.

Image checks used 512 × 512 outputs. Kandinsky produced 576 × 320 video with 121 frames and stereo audio.
MiniMax produced 608 × 480 video with 124 frames and stereo audio. YuE2 produced 48 kHz stereo audio.
These are integration checks, not a new model-quality benchmark. Windows and physical 8 GB GPUs were not tested for this node release.

## Nodes

| Node | Purpose |
| --- | --- |
| `OrbitQuant Inspect Artifact` | Validate an OrbitQuant artifact directory and return a text summary plus structured metadata. |
| `OrbitQuant Pipeline Component Loader` | Attach any compatible `universal` or model-specific OrbitQuant component artifact to a pipeline attribute such as `transformer`. |
| `OrbitQuant FLUX Loader` | Attach a FLUX or FLUX.2 transformer artifact and reject non-FLUX policies. |
| `OrbitQuant Z-Image Loader` | Attach a Z-Image transformer artifact and reject other target policies. |
| `OrbitQuant Wan Loader` | Attach a Wan transformer artifact and reject other target policies. |
| `OrbitQuant Release Loader` | Validate any supported multicomponent release and select its allowlisted adapter from `comfyui_orbitquant.json`. |
| `OrbitQuant Generate Video` | Run a supported video release with durable intermediate artifacts and a standard ComfyUI video preview. |

Current ComfyUI uses the V3 entrypoint. Legacy mappings remain available when V3 is absent.

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
python -m pip install "orbitquant>=0.12.0,<1"
python -m orbitquant.cli.main kernels-install
```

For the default optimized `runtime_mode="auto_fused"` path on CUDA, install
OrbitQuant with its kernel runtime extra. This provides the Triton fallback
used when no native variant matches:

```bash
python -m pip install "orbitquant[hf,kernels]>=0.12.0,<1"
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

The generic release nodes consume the Diffusers-native multicomponent release
instead of the older single-component artifact layout described below. Download
the public model into a local directory using the same environment as ComfyUI:

```bash
hf download WaveCut/MiniMax-H3-OrbitQuant-W4A4 \
  --local-dir /models/MiniMax-H3-OrbitQuant-W4A4
python -m pip install "orbitquant[hf,kernels]>=0.12.0,<1"
python -m pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc" \
  "transformers>=5.13,<6" accelerate av soundfile
```

On the RunPod ComfyUI image, start ComfyUI with its global allocator and offload
layers disabled. The OrbitQuant generator subprocess then owns the bounded
memory policy instead of competing with ComfyUI's DynamicVRAM and async-offload
hooks:

```bash
python main.py --listen 0.0.0.0 --port 8188 \
  --disable-cuda-malloc \
  --disable-dynamic-vram \
  --disable-async-offload
```

Load the [public workflow](evidence/minimax-h3/workflows/minimax-h3-t2va-ui.json)
or build the same graph with `OrbitQuant Release Loader` and `OrbitQuant
Generate Video`. The public node types are model-agnostic; H3-specific component
and execution rules live in the release config and an internal allowlisted
adapter, so another model family does not require another pair of nodes.

1. Set `OrbitQuant Release Loader.model_path` to the downloaded
   directory.
2. Connect its `release` output to `OrbitQuant Generate Video`.
3. For T2VA keep `task=t2va`. For Ref2VA choose `ref2va` and set
   `reference_path` to a local image.
4. Use `width=608`, `height=480`, and `steps=24` for the verified 480p recipe.
5. Start with `inference_profile=balanced`; choose another profile only for its
   documented memory/latency tradeoff.

The release schedule uses 24 sigma points and 23 denoiser forwards. MiniMax H3
requires 5–15 seconds at 24 FPS; `num_frames=124` is the shortest verified VAE
packing sequence and is therefore the default smoke. The text encoder enters
GPU memory layer-by-layer for conditioning and is then moved back to RAM before
the selected transformer runs.

### Inference profiles

All measurements below use W4A4 native packed kernels, no exact INT8 weight
cache, 608×480, 124 frames, 24 sigma points, and source FP32 VAEs. Process peaks
include CUDA allocations outside PyTorch's own accounting.

| Profile | Placement | Hardware | Task | Process peak | Denoise / generation |
| --- | --- | --- | --- | ---: | ---: |
| `balanced` (default) | streamed leaf offload, 12 GiB allocator cap | RTX PRO 6000 | T2VA | 6.36 GiB child; 6.90 GiB with idle ComfyUI | 46.68 s denoise |
| `speed` | resident transformer | RTX PRO 6000 | T2VA | 21.14 GiB | 46.84 s denoise; 51.10 s generation |
| `minimum_vram` | low-CPU-memory streamed leaf offload, 8 GiB cap | RTX 4090 | T2VA | 4.07 GiB | 154.25 s denoise; 188.70 s generation |
| `speed` | resident `transformer_ref` | RTX PRO 6000 | Ref2VA | 24.06 GiB | 118.48 s denoise; 155.42 s generation |

`balanced` is the Pareto default on the tested PRO 6000: streamed transfers
overlap compute closely enough to match resident denoising while cutting the
child's physical peak by about 70%. `minimum_vram` is the verified absolute
minimum endpoint. `speed` removes transformer transfers when VRAM is available.
Native-auto Torch Flash SDPA was the fastest supported attention path on the
tested CUDA 13 / SM120 stack; SageAttention2's available binary did not contain
SM120 code, and cuDNN was slower. These unsupported branches are not part of the
public recipe.

T2VA and Ref2VA both use sequential CUDA text conditioning. Ref2VA also encodes
the reference through the untouched source FP32 visual VAE before loading the
quantized `transformer_ref`; with a release runner from OrbitQuant 0.11 the
runner keeps the VAEs on the GPU for that encode when the device and the
profile's memory cap leave room, and streams them tiled otherwise. Visual decode
uses tiled source FP32 VAE offload; the source FP32 audio VAE enters GPU only for
its audio stage. Neither VAE is quantized.

The node saves generation logs, per-step checkpoints, and the latent bundle as
soon as each exists. Only after denoising succeeds does it decode with the
untouched source FP32 VAEs. The output is a standard ComfyUI `VIDEO`, so the
core preview and downstream video nodes work without VideoHelperSuite. Decode
also retains a high-quality CRF 1 yuv444p master next to the preview; the
published HEVC example is derived from that master at CRF 10.

The [proof bundle](evidence/minimax-h3/README.md) includes the official ComfyUI
workflow-image export with embedded JSON, live `/prompt` results, CRF 1 and HEVC
media, frame timelines, audio spectrum, exact revisions, and machine-readable
measurements.

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

Quantization math and weight loading remain in OrbitQuant.
