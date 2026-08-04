# MiniMax H3 OrbitQuant — ComfyUI proof

The W4A4 multicomponent release works through current ComfyUI on an RTX PRO
6000. The public surface is model-agnostic: `OrbitQuant Release Loader` routes
an allowlisted adapter from the release config, and `OrbitQuant Generate Video`
returns ComfyUI's standard `VIDEO`. No MiniMax-specific node classes are exposed.

![Public MiniMax H3 OrbitQuant workflow](workflow-export.png)

This 6120×2620 PNG was produced by ComfyUI's `Workflow Image → Export → png`
action, not by taking a browser screenshot. Its `tEXtworkflow` chunk contains
the same six-node workflow, two links, and verified 608×480 generation values.

## Public workflow

- [Loadable ComfyUI workflow](workflows/minimax-h3-t2va-ui.json)
- [API prompt exported from the live graph](workflows/minimax-h3-public-ui-api.json)
- [T2VA API proof workflow](workflows/minimax-h3-t2va-api.json)
- [Ref2VA API proof workflow](workflows/minimax-h3-ref2va-api.json)
- [Machine-readable report](report.json)
- [Publication audit](publication.json)

The layout and documentation structure are derived from Comfy-Org's bundled
[`video_minimax_h3_t2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/7653f1cdef1d92394b6ef9946018c0a8aa4136b8/templates/video_minimax_h3_t2v.json):
model links and usage notes, a compact generation node, and core `SaveVideo`.
The imported graph contained six nodes, two links, and no missing nodes. The
live values were independently read back as 608×480, 124 frames, seed 42, and
50 sigma points.

## RTX PRO 6000 measurements

Both jobs used 124 frames at 24 FPS and the full 50-point schedule (49 denoiser
forwards). All 300 eligible modules dispatched through native W4A4. Decode used
the untouched source FP32 visual and audio VAEs.

| task | transformer | generation | generation peak | FP32 decode | decode peak |
|---|---|---:|---:|---:|---:|
| T2VA | `transformer` | 137.14 s | 18.91 GiB | 5.48 s | 11.29 GiB |
| Ref2VA | `transformer_ref` | 318.86 s | 30.46 GiB | 5.50 s | 11.29 GiB |

T2VA used manual stage offload. Ref2VA used H3's component manager with a 64 GB
reserve margin, fixing the earlier CPU-VAE/CUDA-reference mismatch. In both
runs the text encoder was on CPU after conditioning.

## Reviewed media

- T2VA: [H.265/HEVC](media/hevc/comfyui-t2va-608x480.mp4) · [H.264](media/h264/comfyui-t2va-608x480.mp4) · [timeline](media/timelines/comfyui-t2va-608x480.png)
- Ref2VA: [H.265/HEVC](media/hevc/comfyui-ref2va-608x480.mp4) · [H.264](media/h264/comfyui-ref2va-608x480.mp4) · [timeline](media/timelines/comfyui-ref2va-608x480.png)

The H.264 and HEVC variants each probe as 608×480, 124 frames, and AAC stereo
at 32 kHz. Container duration is 5.175 seconds for H.264 and 5.184 seconds for
HEVC. Full-resolution temporal samples were reviewed for face geometry, eyes,
lips, motion ghosting, texture breakup, and grid artifacts. Both runs passed.

## Exact environment

- ComfyUI `9a9fdb10ed144ce760d9682cb247526ea23cc525`
- ComfyUI frontend `1.47.12`
- ComfyUI-OrbitQuant candidate `51d5473bc2bd4f14572724d362770c8ffacb10ed`
- OrbitQuant `0.9.1`
- Diffusers `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
- Model release `e434bbea523349576e7c3d2f6090744aa4597123`
- RunPod `runpod-torch-v280`, NVIDIA RTX PRO 6000 Blackwell Server Edition

The exact public graph also reached terminal `pass` through `/prompt`
(`823b3407-ae85-43fd-b51a-129ea7a11f4d`) and core `SaveVideo` produced a
second valid 608×480 H.264/AAC file.

## Published release

The public Hugging Face release is revision
[`fa2d87b221910e0d9cb457151e5e14153362f1e9`](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/commit/fa2d87b221910e0d9cb457151e5e14153362f1e9).
It has one commit, 113 files, no `.cache`, `__pycache__`, or `.pyc` files, and
the independently audited byte total is 67,542,922,354. Anonymous requests
returned HTTP 200 for the model page and HTTP 206 for a video range request.
