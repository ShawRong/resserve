# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Opaque MXFP4 linear op preserving quant/GEMM dispatch under compile."""
from __future__ import annotations

import os

import torch


@torch.library.custom_op("residue::mxfp4_linear", mutates_args=())
def mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    indices: torch.Tensor,
    masks: torch.Tensor,
    mode: str,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    mext_max_m: int = 0,
    weight_scale_base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use full-channel MExt at small M, including short prefill."""
    from nvfp4.mxfp4 import quantize, quantize_selective

    if mode not in ("standard", "selective", "mext_r1"):
        raise ValueError(f"unknown MXFP4 linear mode {mode}")
    if mext_max_m < 0:
        raise ValueError("mext_max_m must be nonnegative")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("linear input must have a positive feature dimension")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (10, 0):
        raise ValueError("MXFP4 linear currently requires SM100")

    output_shape = (*x.shape[:-1], weight.shape[0])
    rows = x.numel() // x.shape[-1]
    if rows == 0:
        return torch.empty(output_shape, device=x.device, dtype=output_dtype)
    source = x.reshape(rows, x.shape[-1])
    use_fold = mode == "mext_r1" or rows <= mext_max_m
    warm_fold = False
    if not use_fold and mext_max_m > 0:
        from flashinfer.autotuner import AutoTuner

        # vLLM profiles only its maximum token count before Graph capture.
        warm_fold = AutoTuner.get().is_tuning_mode
    if use_fold or warm_fold:
        from quant.flashinfer_integration import mxfp4_linear_adapter as fold_tuner

        fold_weight, fold_scale = weight, weight_scale
        if mode == "selective":
            if weight_scale_base is None:
                raise ValueError("selective MExt requires prepared base-K scales")
            fold_weight = weight[:, :source.shape[-1] // 2]
            fold_scale = weight_scale_base
        tune = fold_tuner.tuner_enabled()
        if use_fold or tune:
            output = fold_tuner.tuned_mxfp4_fold(
                source if use_fold else source[:mext_max_m],
                fold_weight, fold_scale, out_dtype=output_dtype,
                pdl_mode=os.environ.get("MXFP4_MEXT_R1_PDL", "early"), tune=tune,
            )
    if not use_fold:
        if mode == "selective":
            activation, activation_scale = quantize_selective(
                source, indices, masks
            )
        else:
            activation, activation_scale = quantize(source)
        from flashinfer import mm_fp4

        activation_scale = activation_scale.reshape((rows + 127) // 128 * 128, -1)
        weight_scale = weight_scale.reshape((weight.shape[0] + 127) // 128 * 128, -1)
        output = mm_fp4(
            activation, weight.T, activation_scale, weight_scale.T,
            None, output_dtype, backend="cute-dsl", block_size=32, use_nvfp4=False,
        )
    if bias is not None:
        output = (output + bias).to(output_dtype)
    return output.reshape(output_shape)


@mxfp4_linear.register_fake
def _fake_mxfp4_linear(
    x,
    weight,
    weight_scale,
    indices,
    masks,
    mode,
    output_dtype,
    bias=None,
    mext_max_m=0,
    weight_scale_base=None,
):
    del weight_scale, indices, masks, mode, bias, mext_max_m, weight_scale_base
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), device=x.device, dtype=output_dtype
    )
