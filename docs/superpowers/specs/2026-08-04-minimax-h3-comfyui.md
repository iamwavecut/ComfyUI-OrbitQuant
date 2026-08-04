# MiniMax H3 ComfyUI Support Design

## Outcome

ComfyUI-OrbitQuant can execute the published private
`WaveCut/MiniMax-H3-OrbitQuant-W4A4` release as a native ComfyUI graph and
return a previewable `VIDEO`. The proof must come from a clean current ComfyUI
checkout on an RTX PRO 6000, submitted through ComfyUI's `/prompt` API.

## Architecture

The node pack adds a model-agnostic release loader and video generation output
node. The loader reads an allowlisted adapter id from `comfyui_orbitquant.json`,
then the internal H3 adapter validates the multicomponent manifest and exact component policy:
W4A4 `transformer`, `transformer_ref`, and `text_encoder`; source-precision
`vae` and `audio_vae`. It returns an immutable local-release descriptor without
loading weights.

The generation node delegates execution to the scripts shipped in the model
release. T2VA runs the published denoising recipe with manual stage offload
and atomically saved latents/checkpoints. It then runs the published decoder
with explicit FP32 visual and audio VAEs. The output is wrapped as ComfyUI's
standard `VIDEO` type and exposed through the standard video preview UI.
Ref2VA uses H3's component manager to place the source FP32 visual VAE for
reference encoding while preserving the text-encoder conditioning offload.

This boundary avoids copying model math into the node pack, makes partial
artifacts durable, and keeps the text encoder on GPU only while conditioning.

## Inputs and outputs

`OrbitQuant Release Loader` accepts a local release directory and returns
`ORBITQUANT_RELEASE` plus a JSON summary.

`OrbitQuant Generate Video` accepts the release descriptor, prompt,
task (`t2va` or `ref2va`), optional reference path, seed, dimensions, frame
count, step count, and filename prefix. It returns standard `VIDEO` plus a JSON
run report and is an output node.

## Validation and failure behavior

- Reject empty or non-directory model paths.
- Reject absent or malformed `quantization_manifest.json`.
- Reject releases whose W4A4 or source-precision component sets differ from the
  required policy.
- Reject absent runner/decoder scripts and component directories before GPU
  execution.
- Require a reference image for `ref2va`.
- Require positive dimensions, 120–360 requested frames (the H3 5–15 second
  range before VAE packing), and at least two sigma points.
- Preserve stage logs, metrics, checkpoints, and latents next to the output.
- Surface subprocess failure with the stage name and log path.

## Compatibility

Both legacy `NODE_CLASS_MAPPINGS` and the modern V3 `comfy_entrypoint` expose
the same model-agnostic node ids. Model-specific H3 ids are deliberately not
exposed. The V3 implementation uses current official ComfyUI
`InputImpl.VideoFromFile` and `ui.PreviewVideo` APIs. Existing generic loaders
remain unchanged.

## Proof package

The published evidence includes the exact ComfyUI revision, node-pack revision,
OrbitQuant version, GPU/system report, `/object_info` excerpts, API workflow,
UI workflow derived from the official bundled MiniMax H3 preset, `/prompt`
response, terminal `/history` response, server log,
stage metrics, media probe, frame contact sheet, full output video, and remote
artifact audit. At least one T2VA workflow and one Ref2VA workflow exercise the
two denoisers at 608 x 480 and the official 50-point schedule.
