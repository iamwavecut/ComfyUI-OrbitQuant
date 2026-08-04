# MiniMax H3 ComfyUI Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable MiniMax H3 W4A4 generation nodes and publish end-to-end ComfyUI proof from an RTX PRO 6000.

**Architecture:** A model-agnostic config/manifest loader selects an allowlisted internal adapter. A generic video output node launches the release's pinned latent-generation and FP32-VAE decode scripts, then returns ComfyUI's standard video object and preview metadata.

**Tech Stack:** Python 3.11+, OrbitQuant 0.9.1, current ComfyUI V3 API, Diffusers MiniMax H3 modular pipeline, pytest, RunPod RTX PRO 6000.

## Global Constraints

- Use `WaveCut/MiniMax-H3-OrbitQuant-W4A4` from its private pinned revision.
- W4A4 components are exactly `transformer`, `transformer_ref`, and `text_encoder`.
- `vae` and `audio_vae` stay as original source-precision copies and decode in FP32.
- Text encoding happens on GPU, then the encoder is offloaded before denoising.
- Evidence workflows use 608 x 480 and 50 sigma points.
- Persist latent/checkpoint/log artifacts as they appear.
- Test both T2VA and Ref2VA through ComfyUI's `/prompt` API.

---

### Task 1: Release descriptor and validation

**Files:**
- Create: `src/comfyui_orbitquant/minimax_h3.py`
- Modify: `src/comfyui_orbitquant/nodes.py`
- Test: `tests/test_minimax_h3.py`

**Interfaces:**
- Produces: `MiniMaxH3Release.from_path(path: str | Path) -> MiniMaxH3Release`
- Produces: `OrbitQuantReleaseLoader.load(model_path: str) -> tuple[OrbitQuantRelease, str]`

- [ ] Write failing tests for valid manifests and every preflight rejection.
- [ ] Run `pytest tests/test_minimax_h3.py -q` and confirm failures identify missing interfaces.
- [ ] Implement the immutable descriptor, manifest policy checks, required-path checks, and legacy loader node.
- [ ] Run `pytest tests/test_minimax_h3.py -q` and confirm the descriptor tests pass.
- [ ] Commit the tested descriptor slice.

### Task 2: Durable generation runner

**Files:**
- Modify: `src/comfyui_orbitquant/minimax_h3.py`
- Modify: `src/comfyui_orbitquant/nodes.py`
- Test: `tests/test_minimax_h3.py`

**Interfaces:**
- Consumes: `MiniMaxH3Release`
- Produces: `MiniMaxH3Runner.run(...) -> MiniMaxH3RunResult`
- Produces: `OrbitQuantGenerateVideo.generate(...) -> dict[str, object]`

- [ ] Write failing tests that assert exact generator and decoder commands, FP32 VAE flags, stage-offload flags, reference validation, durable paths, and failure reporting.
- [ ] Run the focused tests and confirm they fail before implementation.
- [ ] Implement explicit argument lists with `subprocess.run`, stage logs, immediate latent/checkpoint paths, metrics collection, and output validation.
- [ ] Implement the legacy output node returning `VIDEO` and a JSON report.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit the durable runner slice.

### Task 3: Current ComfyUI V3 surface and documentation

**Files:**
- Modify: `src/comfyui_orbitquant/v3.py`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Test: `tests/test_nodes.py`
- Test: `tests/test_minimax_h3.py`

**Interfaces:**
- Consumes: legacy H3 node implementations.
- Produces: V3 nodes with ids `OrbitQuantReleaseLoader` and `OrbitQuantGenerateVideo`.

- [ ] Add failing V3 schema/delegation tests and dependency/documentation assertions.
- [ ] Run the focused tests and confirm the new expectations fail.
- [ ] Add V3 schemas using standard `VIDEO`, `InputImpl.VideoFromFile`, and `ui.PreviewVideo`; expose both nodes from the extension.
- [ ] Document download, graph construction, T2VA, Ref2VA, offload, output artifacts, and pinned runtime requirements; raise OrbitQuant's floor to 0.9.1.
- [ ] Run `pytest -q`, `ruff check .`, and build the wheel.
- [ ] Commit the V3 and documentation slice.

### Task 4: Clean ComfyUI GPU proof

**Files:**
- Create: `evidence/minimax-h3/workflows/minimax-h3-t2va-api.json`
- Create: `evidence/minimax-h3/workflows/minimax-h3-ref2va-api.json`
- Create: `evidence/minimax-h3/report.json`

**Interfaces:**
- Consumes: published node branch and private H3 release.
- Produces: terminal ComfyUI histories and media artifacts for both tasks.

- [ ] Push the tested feature branch and record its commit SHA.
- [ ] Start an official-template RTX PRO 6000 pod without secrets in pod environment variables.
- [ ] Install a clean pinned ComfyUI checkout, the feature branch, OrbitQuant 0.9.1, and the H3 runtime; copy the HF token securely after pod readiness.
- [ ] Download the private model at its pinned revision and start ComfyUI with logs captured.
- [ ] Verify both node ids in `/object_info`, submit the T2VA API workflow to `/prompt`, and poll to terminal `/history`.
- [ ] Inspect an early and late full-resolution frame before accepting the T2VA result.
- [ ] Submit the Ref2VA workflow, poll to terminal history, and inspect early/late frames.
- [ ] Copy all proof artifacts locally as they appear, generate probes/contact sheets, and write the machine-readable report.

### Task 5: Publish and independently audit proof

**Files:**
- Modify: `README.md`
- Create: `evidence/minimax-h3/README.md`
- Create: `evidence/minimax-h3/media/*`
- Create: `evidence/minimax-h3/logs/*`

**Interfaces:**
- Consumes: local proof package.
- Produces: remotely viewable code branch and model-card evidence with reproducible instructions.

- [ ] Add concise evidence links and measured results to the node-pack README.
- [ ] Commit and push the final proof package.
- [ ] Upload the workflow, reports, contact sheets, and compressed videos to the H3 model repository while preserving its single-commit history.
- [ ] Independently download remote manifests and media, compare hashes, probe streams, and verify Git/HF revisions.
- [ ] Delete only the owned RunPod pod and verify no owned pods remain.
- [ ] Run final local tests and report exact revisions, measurements, artifact links, and remaining risks.
