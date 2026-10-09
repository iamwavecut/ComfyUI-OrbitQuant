#!/usr/bin/env python3
# ruff: noqa: E501
"""Patch Diffusers 0.39 Ideogram4Pipeline for fal's single-branch Instant model.

The fal checkpoint was published before the matching Diffusers changes landed.
This deterministic patch keeps the dual-branch path intact and changes behavior
only when ``unconditional_transformer is None``.
"""

from __future__ import annotations

import math

MARKER = "# FAL_IDEOGRAM4_INSTANT_SINGLE_BRANCH_PATCH_V1"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def patch_source(source: str) -> str:
    if MARKER in source:
        return source

    source = replace_once(
        source,
        '    _optional_components = ["prompt_enhancer_head"]\n',
        f'    {MARKER}\n    _optional_components = ["prompt_enhancer_head", "unconditional_transformer"]\n',
        "optional component list",
    )
    source = replace_once(
        source,
        "        unconditional_transformer: Ideogram4Transformer2DModel,\n",
        "        unconditional_transformer: Ideogram4Transformer2DModel | None = None,\n",
        "constructor annotation",
    )
    source = replace_once(
        source,
        "        guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45 + (3.0,) * 3,\n",
        "        guidance_schedule: list[float] | torch.Tensor | None = None,\n",
        "single-branch guidance default",
    )

    old_guidance_validation = """        # Guidance is controlled by either a constant `guidance_scale` or a per-step `guidance_schedule`; exactly
        # one must be set (the `guidance_schedule` default makes the no-arg call use the recommended schedule).
        if guidance_scale is not None and guidance_schedule is not None:
            raise ValueError("Only one of `guidance_scale` and `guidance_schedule` may be set.")
        if guidance_scale is None and guidance_schedule is None:
            raise ValueError("One of `guidance_scale` and `guidance_schedule` must be set.")
        if guidance_schedule is not None and len(guidance_schedule) != num_inference_steps:
            raise ValueError(
                f"`guidance_schedule` must have length `num_inference_steps` ({num_inference_steps}), "
                f"got {len(guidance_schedule)}."
            )
"""
    new_guidance_validation = """        if self.unconditional_transformer is None:
            if guidance_scale is not None or guidance_schedule is not None:
                raise ValueError(
                    "`guidance_scale` and `guidance_schedule` must be omitted for a single-branch "
                    "Ideogram 4 Instant pipeline."
                )
        else:
            # Guidance is controlled by either a constant scale or a per-step schedule.
            if guidance_scale is not None and guidance_schedule is not None:
                raise ValueError("Only one of `guidance_scale` and `guidance_schedule` may be set.")
            if guidance_scale is None and guidance_schedule is None:
                raise ValueError("One of `guidance_scale` and `guidance_schedule` must be set.")
            if guidance_schedule is not None and len(guidance_schedule) != num_inference_steps:
                raise ValueError(
                    f"`guidance_schedule` must have length `num_inference_steps` ({num_inference_steps}), "
                    f"got {len(guidance_schedule)}."
                )
"""
    source = replace_once(
        source,
        old_guidance_validation,
        new_guidance_validation,
        "guidance validation",
    )

    old_neg_inputs = """        # 4. Unconditional (image-only) branch, derived from the conditioning: zeroed text features and the
        # image-region slices of the layout.
        neg_llm_features = torch.zeros(
            batch_size * num_images_per_prompt,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=device,
        )
        neg_position_ids = position_ids[:, max_sequence_length:]
        neg_segment_ids = segment_ids[:, max_sequence_length:]
        neg_indicator = indicator[:, max_sequence_length:]
"""
    new_neg_inputs = """        # 4. The distilled Instant model has no unconditional branch. Build image-only
        # guidance inputs only for the original dual-branch pipeline.
        if self.unconditional_transformer is not None:
            neg_llm_features = torch.zeros(
                batch_size * num_images_per_prompt,
                num_image_tokens,
                llm_features.shape[-1],
                dtype=llm_features.dtype,
                device=device,
            )
            neg_position_ids = position_ids[:, max_sequence_length:]
            neg_segment_ids = segment_ids[:, max_sequence_length:]
            neg_indicator = indicator[:, max_sequence_length:]
        else:
            neg_llm_features = None
            neg_position_ids = None
            neg_segment_ids = None
            neg_indicator = None
"""
    source = replace_once(source, old_neg_inputs, new_neg_inputs, "negative branch inputs")

    old_schedule = """        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 5. Resolve the per-step guidance schedule (a constant `guidance_scale` broadcasts to every step, otherwise
        # use the provided `guidance_schedule`, validated by `check_inputs`) and the tensor of per-step weights `gw`.
        if guidance_scale is not None:
            guidance_schedule = [float(guidance_scale)] * num_inference_steps
        gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
"""
    terminal_expression = 1.0 - (1.0 / (1.0 + math.exp(0.5 * -15.0)))
    new_schedule = f"""        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        if self.unconditional_transformer is None:
            # The native schedule terminates at a small nonzero sigma. Diffusers
            # normally appends zero; preserving the native terminal is required by
            # the distilled Instant checkpoint.
            self.scheduler.sigmas[-1] = {terminal_expression!r}
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 5. Guidance exists only for the original dual-branch model.
        gw = None
        if self.unconditional_transformer is not None:
            if guidance_scale is not None:
                guidance_schedule = [float(guidance_scale)] * num_inference_steps
            gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
"""
    source = replace_once(source, old_schedule, new_schedule, "schedule and guidance setup")

    source = replace_once(
        source,
        """        llm_features = llm_features.to(self.transformer.dtype)
        neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)
""",
        """        llm_features = llm_features.to(self.transformer.dtype)
        if self.unconditional_transformer is not None:
            neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)
""",
        "negative feature dtype",
    )

    old_neg_forward = """                # Unconditional pass uses image-only positions with zeroed text features.
                neg_v = self.unconditional_transformer(
                    hidden_states=latents.to(self.unconditional_transformer.dtype),
                    timestep=t_model,
                    encoder_hidden_states=neg_llm_features,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0].to(torch.float32)

                # Expose the current step's guidance weight via `self.guidance_scale` so callbacks can read it.
                self._guidance_scale = guidance_schedule[i]
                gw_i = gw[i]
                v = gw_i * pos_v + (1.0 - gw_i) * neg_v
"""
    new_neg_forward = """                if self.unconditional_transformer is None:
                    # CFG was distilled into the conditional branch.
                    self._guidance_scale = None
                    v = pos_v
                else:
                    # Unconditional pass uses image-only positions with zeroed text features.
                    neg_v = self.unconditional_transformer(
                        hidden_states=latents.to(self.unconditional_transformer.dtype),
                        timestep=t_model,
                        encoder_hidden_states=neg_llm_features,
                        position_ids=neg_position_ids,
                        segment_ids=neg_segment_ids,
                        indicator=neg_indicator,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0].to(torch.float32)

                    # Expose the current step's guidance weight so callbacks can read it.
                    self._guidance_scale = guidance_schedule[i]
                    gw_i = gw[i]
                    v = gw_i * pos_v + (1.0 - gw_i) * neg_v
"""
    source = replace_once(source, old_neg_forward, new_neg_forward, "negative branch forward")
    return source
