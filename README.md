# ComfyUI-OrbitQuant

ComfyUI custom nodes for loading and inspecting OrbitQuant artifacts.

This repository is private and experimental. Quantization logic lives in the
`orbitquant` package; this extension only provides ComfyUI integration.

## Nodes

- `OrbitQuant Inspect Artifact`: validates an OrbitQuant artifact and returns a
  text summary plus structured metadata, including the saved artifact component
  from `model_index.json` when present.
- `OrbitQuant Pipeline Component Loader`: loads a quantized OrbitQuant component
  artifact into a Diffusers-like pipeline object. The default component is
  `transformer`.
- `OrbitQuant FLUX Loader`: loads an OrbitQuant transformer artifact into a
  FLUX pipeline object.
- `OrbitQuant Z-Image Loader`: loads an OrbitQuant transformer artifact into a
  Z-Image pipeline object.
- `OrbitQuant Wan Loader`: loads an OrbitQuant transformer artifact into a Wan
  pipeline object.

## Install

Clone this repository into `ComfyUI/custom_nodes/ComfyUI-OrbitQuant`, then make
sure the `orbitquant` Python package is importable by the ComfyUI environment.

During development the package dependency points at the private
`iamwavecut/OrbitQuant` `orbitquant-mvp` branch.

## Current Scope

- Thin wrapper only; no quantization implementation is duplicated here.
- First supported path is component artifact loading through
  `orbitquant.load_quantized_pipeline_component`.
- Loader nodes use OrbitQuant's artifact component validation, so a transformer
  loader fails loudly if the artifact was saved for a different component.
- FLUX, Z-Image, and Wan loaders currently attach the quantized transformer
  component to an existing pipeline object. Runtime-specific ComfyUI workflow
  adapters can be added after these base nodes are tested inside a real ComfyUI
  checkout.
