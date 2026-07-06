# ComfyUI-OrbitQuant

ComfyUI custom nodes for loading and inspecting OrbitQuant artifacts.

This repository is private and experimental. Quantization logic lives in the
`orbitquant` package; this extension only provides ComfyUI integration.

## Nodes

- `OrbitQuant Inspect Artifact`: validates an OrbitQuant artifact and returns a
  text summary plus structured metadata.
- `OrbitQuant Pipeline Component Loader`: loads a quantized OrbitQuant component
  artifact into a Diffusers-like pipeline object. The default component is
  `transformer`.

## Install

Clone this repository into `ComfyUI/custom_nodes/ComfyUI-OrbitQuant`, then make
sure the `orbitquant` Python package is importable by the ComfyUI environment.

During development the package dependency points at the private
`iamwavecut/OrbitQuant` `orbitquant-mvp` branch.

## Current Scope

- Thin wrapper only; no quantization implementation is duplicated here.
- First supported path is component artifact loading through
  `orbitquant.load_quantized_pipeline_component`.
- Image workflow loaders for FLUX/Z-Image and video workflow loaders for Wan
  still need ComfyUI runtime-specific adapters after the base nodes are tested
  inside a real ComfyUI checkout.
