# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Prepare and launch paged NVFP4 decode attention."""

from __future__ import annotations

import functools
import math
from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .kernel_paged_decode import GroupedQueryAttentionDecodePagedFp4
from .kv_layout import (
    PAGE_SIZE,
    SF_VEC,
    NvFp4PagedKVPool,
)


from .plan import select_decode_plan, static_compile_plans, validate_decode_plan

N_PAD = 128

Qmul4ReductionWorkspace = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Qmul4GmemReductionWorkspace = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


RESIDUE_GQA_GROUPS = (4, 6, 8)


def _residue_padded_group_size(
    num_qo_heads: int, num_kv_heads: int
) -> int:
    """Physical base+residue rows per KV head for the residue tactic."""
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    g = num_qo_heads // num_kv_heads
    if g == 4:
        return 8
    if g in (6, 8):
        return 16
    raise NotImplementedError(
        f"residue Q supports logical GQA ratios {RESIDUE_GQA_GROUPS}, got {g}"
    )


def _pick_tile_shape(group_size: int, prediction: int) -> Tuple[int, int]:
    """Return (grouped_head_tile, prediction_tile) for a power-of-two padded group."""
    if group_size not in (8, 16) or prediction != 1:
        raise NotImplementedError("production ResAttn requires single-token Q8/Q16")
    return group_size, 1


def check_shape_supported(num_qo_heads: int, num_kv_heads: int, head_dim: int,
                          prediction: int = 1) -> None:
    """Reject unsupported shapes before compiling the kernel."""
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError(
            f"num_qo_heads({num_qo_heads}) must be a multiple of num_kv_heads({num_kv_heads})"
        )
    if head_dim not in (128, 256):
        raise NotImplementedError(f"ResAttn supports head_dim=128 or 256, got {head_dim}")
    _residue_padded_group_size(num_qo_heads, num_kv_heads)
    if prediction != 1:
        raise NotImplementedError("production ResAttn supports single-token decode only")


@functools.lru_cache(maxsize=8)
def _sm_count(device_str: str) -> int:
    return torch.cuda.get_device_properties(device_str).multi_processor_count


def plan_qmul4_decode(batch, num_kv_heads, max_kv_len, device, *,
                       grouped_head_tile=16, residue_fold=True, head_dim=128):
    """BF16 output wrapper over the shared split/reduction policy."""
    if not residue_fold:
        raise NotImplementedError("ordinary QK is not supported")
    return select_decode_plan(
        batch, num_kv_heads, max_kv_len, _sm_count(str(device)),
        grouped_head_tile=grouped_head_tile, residue_fold=residue_fold, head_dim=head_dim,
    )


def _use_single_softmax_scheduler(
    grouped_head_tile: int,
    reduction: str,
    v_mode: str,
    kv_splits: int,
) -> bool:
    """Select the softmax scheduler."""
    return (
        grouped_head_tile in (8, 16)
        and v_mode == "qmul4_tmem"
        and (
            (reduction == "none" and kv_splits == 1)
            or (reduction == "gmem_fused" and kv_splits in (4, 8))
        )
    )


def allocate_qmul4_reduction_workspace(
    splits: int,
    batch: int,
    num_qo_heads: int,
    head_dim: int,
    device,
    *,
    partial_o_dtype: torch.dtype = torch.float32,
) -> Qmul4ReductionWorkspace:
    """Allocate split-K reduction workspace."""
    if partial_o_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "partial_o_dtype must be torch.float32 or torch.bfloat16, "
            f"got {partial_o_dtype}"
        )
    o_partial = torch.empty(
        splits,
        batch,
        1,
        num_qo_heads,
        head_dim,
        dtype=partial_o_dtype,
        device=device,
    )
    stats_shape = (splits, batch, 1, num_qo_heads)
    l_partial = torch.empty(stats_shape, dtype=torch.float32, device=device)
    m_partial = torch.empty_like(l_partial)
    return o_partial, l_partial, m_partial


def allocate_qmul4_gmem_reduction_workspace(
    splits: int,
    batch: int,
    num_qo_heads: int,
    head_dim: int,
    device,
    *,
    partial_o_dtype: torch.dtype = torch.bfloat16,
) -> Qmul4GmemReductionWorkspace:
    """Allocate fused global-memory reduction workspace."""
    partials = allocate_qmul4_reduction_workspace(
        splits,
        batch,
        num_qo_heads,
        head_dim,
        device,
        partial_o_dtype=partial_o_dtype,
    )
    completion = torch.zeros(
        batch * num_qo_heads,
        dtype=torch.int32,
        device=device,
    )
    return (*partials, completion)


def _reduction_workspace_partial_o_dtype(
    reduction_workspace: (
        Qmul4ReductionWorkspace | Qmul4GmemReductionWorkspace | None
    ),
) -> str:
    """Return the CuTe specialization key for a caller-owned workspace."""
    if reduction_workspace is None:
        return "fp32"
    dtype = reduction_workspace[0].dtype
    if dtype is torch.float32:
        return "fp32"
    if dtype is torch.bfloat16:
        return "bf16"
    raise ValueError(
        "reduction workspace partial O must be float32 or bfloat16, "
        f"got {dtype}"
    )


@functools.cache
def _compiled(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prediction: int,
    reduction: str,
    v_mode: str = "qmul4_tmem",
    k_sf_mode: str = "linear",
    output_nvfp4: bool = False,
    output_num_qo_heads: int | None = None,
    single_softmax_wg: bool = False,
    partial_o_dtype: str = "fp32",
    smem_correction_broadcast: bool = False,
    residue_fold: bool = True,
    page_size: int = 32,
):
    """Compile or retrieve a decode kernel."""
    if not residue_fold:
        raise NotImplementedError("ordinary QK is not supported")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if reduction not in ("none", "kernel", "gmem_fused", "cga_sharded"):
        raise ValueError(f"unsupported production reducer: {reduction}")
    if v_mode != "qmul4_tmem":
        raise ValueError("production decode supports qmul4_tmem only")
    grouped_head_tile, prediction_tile = _pick_tile_shape(
        num_qo_heads // num_kv_heads, prediction
    )
    if output_num_qo_heads is None:
        output_num_qo_heads = num_qo_heads
    if (
        output_num_qo_heads <= 0
        or output_num_qo_heads > num_qo_heads
        or output_num_qo_heads % num_kv_heads
    ):
        raise ValueError(
            "output_num_qo_heads must be a positive multiple of H_kv no "
            f"larger than padded H_q; got {output_num_qo_heads}, "
            f"H_q={num_qo_heads}, H_kv={num_kv_heads}"
        )
    if page_size != 32:
        raise ValueError("ResAttn requires page32")
    if page_size == 32 and (output_nvfp4
                            or reduction not in ("none", "kernel", "cga_sharded")):
        raise NotImplementedError("page32 supports BF16 and none/kernel/CGA reduction")
    fmha = GroupedQueryAttentionDecodePagedFp4(
        head_dim,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=256,
        reduction_mode=reduction,  # type: ignore[arg-type]
        page_size=page_size,
        logical_grouped_heads=(
            output_num_qo_heads // num_kv_heads
            if output_nvfp4 or output_num_qo_heads != num_qo_heads
            else None
        ),
        single_softmax_wg=single_softmax_wg,
        smem_correction_broadcast=smem_correction_broadcast,
    )
    in_dtype = cutlass.Float4E2M1FN
    if partial_o_dtype not in ("fp32", "bf16"):
        raise ValueError(
            "partial_o_dtype must be 'fp32' or 'bf16', "
            f"got {partial_o_dtype!r}"
        )
    sym_batch = cute.sym_int()
    sym_pages = cute.sym_int()
    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype, (sym_batch, 1, num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0), assumed_align=16)


    _sd2 = lambda: cute.sym_int(divisibility=2)
    k_fake = cute.runtime.make_fake_tensor(
        in_dtype, (sym_pages, page_size, num_kv_heads, head_dim),
        stride=(_sd2(), _sd2(), _sd2(), 1), assumed_align=16)
    sf_flat = lambda: cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (cute.sym_int(),), stride_order=(0,), assumed_align=16)
    k_sf_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (sym_pages, num_kv_heads, page_size, head_dim // SF_VEC),
        stride=(
            cute.sym_int(divisibility=16),
            page_size * (head_dim // SF_VEC),
            head_dim // SF_VEC,
            1,
        ),
        assumed_align=16,
    )

    v_fake = cute.runtime.make_fake_tensor(
        in_dtype, (sym_pages, page_size, num_kv_heads, head_dim),
        stride=(_sd2(), head_dim, _sd2(), 1), assumed_align=16)

    v_sf_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (sym_pages, num_kv_heads, page_size, head_dim // SF_VEC),
        stride=(
            cute.sym_int(divisibility=16),
            page_size * (head_dim // SF_VEC),
            head_dim // SF_VEC,
            1,
        ),
        assumed_align=16,
    )
    if output_nvfp4 and reduction not in ("none", "kernel", "gmem_fused"):
        raise ValueError(
            "fused NVFP4 output requires qmul4_tmem and no reduction, the "
            "deterministic kernel reducer, or the fused last-CTA reducer"
        )
    o_dtype = cutlass.Float4E2M1FN if output_nvfp4 else cutlass.BFloat16
    o_fake = cute.runtime.make_fake_compact_tensor(
        o_dtype,
        (sym_batch, 1, output_num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0), assumed_align=16)
    o_sf_fake = o_global_scale_fake = None
    if output_nvfp4:
        o_sf_fake = sf_flat()
        o_global_scale_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), stride_order=(0,), assumed_align=4
        )


    o_partial_fake = l_partial_fake = m_partial_fake = None
    completion_fake = None
    if reduction in ("kernel", "gmem_fused"):
        sym_splits = cute.sym_int()
        o_partial_fake = cute.runtime.make_fake_compact_tensor(
            (
                cutlass.Float32
                if partial_o_dtype == "fp32"
                else cutlass.BFloat16
            ),
            (
                sym_splits,
                sym_batch,
                1,
                num_qo_heads,
                head_dim,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        partial_shape = (
            sym_splits,
            sym_batch,
            1,
            num_qo_heads,
        )
        l_partial_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            partial_shape,
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            partial_shape,
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        if reduction == "gmem_fused":
            completion_fake = cute.runtime.make_fake_compact_tensor(
                Int32,
                (cute.sym_int(),),
                stride_order=(0,),
                assumed_align=16,
            )
    seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_batch,), assumed_align=4,
    )
    block_tables_fake = cute.runtime.make_fake_tensor(
        Int32, (sym_batch, cute.sym_int()),
        stride=(cute.sym_int(), 1), assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_options = "--enable-tvm-ffi --opt-level 3"
    return cute.compile(
        fmha, Int32(1), q_fake, k_fake, v_fake,
        sf_flat(), k_sf_fake, v_sf_fake,
        o_fake, o_sf_fake, o_global_scale_fake,
        o_partial_fake, l_partial_fake, m_partial_fake,
        completion_fake,
        Float32(1.0), Float32(1.0), seq_lens_fake, block_tables_fake,
        stream_fake, True,
        options=compile_options,
    )


_Q_RESIDUE_SCRATCH: dict = {}


def _q_residue_scratch(b: int, h_kv: int, g: int, d: int, device):
    """Allocate or retrieve grouped Q quantization buffers."""
    physical_g = _residue_padded_group_size(h_kv * g, h_kv)
    key = (b, h_kv, g, physical_g, d, str(device))
    hit = _Q_RESIDUE_SCRATCH.get(key)
    if hit is None:
        q_pack = torch.zeros(
            b, h_kv, physical_g, d // 2, dtype=torch.uint8, device=device
        )
        q_sf = torch.zeros(
            b * h_kv * N_PAD * (d // SF_VEC),
            dtype=torch.uint8,
            device=device,
        )
        hit = (q_pack, q_sf)
        _Q_RESIDUE_SCRATCH[key] = hit
    return hit


def _quantize_q_residue_cuda(q: torch.Tensor, num_kv_heads: int):
    """Quantize Q and its residue for grouped attention."""
    if not (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and hasattr(torch.ops, "nvfp4")
    ):
        return None
    B, HQ, D = q.shape
    if HQ % num_kv_heads:
        raise ValueError(f"H_q={HQ} must be divisible by H_kv={num_kv_heads}")
    g = HQ // num_kv_heads
    if g not in RESIDUE_GQA_GROUPS:
        raise NotImplementedError(
            "decode residue Q8/Q16 supports logical GQA ratios "
            f"{RESIDUE_GQA_GROUPS}, got {g}"
        )
    q4 = q.reshape(B, num_kv_heads, g, D).contiguous()
    q_pack, q_sf = _q_residue_scratch(
        B, num_kv_heads, g, D, q.device
    )
    physical_g = q_pack.shape[2]
    if not hasattr(torch.ops.nvfp4, "scaled_fp4_quant_mext_r1_grouped_q"):
        return None
    torch.ops.nvfp4.scaled_fp4_quant_mext_r1_grouped_q(q4, q_pack, q_sf)
    return (
        q_pack.reshape(B, 1, num_kv_heads * physical_g, D // 2),
        q_sf.view(torch.float8_e4m3fn),
    )


def _check_decode_metadata(block_tables, seq_lens, batch, device):
    """Validate paged decode metadata."""
    if (
        block_tables.dtype != torch.int32
        or block_tables.device != device
        or block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.shape[1] < 1
        or block_tables.stride(1) != 1
        or block_tables.stride(0) < block_tables.shape[1]
    ):
        raise ValueError(
            "block_tables must be int32 [batch, max_pages_per_seq] on the Q "
            "device, with unit column stride and non-overlapping rows"
        )
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.device != device
        or tuple(seq_lens.shape) != (batch,)
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous int32 [batch] on the Q device")


def run_paged_fp4_decode_qmul4_tmem(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    v_data_hnd: torch.Tensor,
    v_sf_bthd: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    max_seq_len: int = 0,
    k_sf_mode: str = "linear",
    value_scale: float = 1.0,
    reduction_workspace: Qmul4ReductionWorkspace | None = None,
    residue_fold: bool = True,
    direct_output: bool = True,
    plan: tuple[int, str] | None = None,
) -> torch.Tensor:
    """Run paged NVFP4 attention with BF16 output."""
    if not residue_fold:
        raise NotImplementedError("ordinary QK is not supported")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if direct_output is False:
        raise NotImplementedError("padded output is not supported")
    B, HQ, D = q.shape
    H = pool.num_kv_heads
    check_shape_supported(HQ, H, D, prediction=1)
    _check_decode_metadata(block_tables, seq_lens, B, q.device)
    if v_data_hnd.shape != (pool.k_data.shape[0], H, pool.page_size, D // 2):
        raise ValueError(
            "v_data_hnd must be [page, head, page_size, D/2], got "
            f"{tuple(v_data_hnd.shape)}"
        )
    if v_sf_bthd.shape != (pool.k_data.shape[0], H, pool.page_size, D // SF_VEC):
        raise ValueError(
            "v_sf_bthd must be [page, head, page_size, D/16] (trtllm contract), got "
            f"{tuple(v_sf_bthd.shape)}"
        )

    g_pad = _residue_padded_group_size(HQ, H)
    if (tuple(out.shape) != (B, HQ, D) or out.dtype != torch.bfloat16
            or out.device != q.device or not out.is_contiguous()):
        raise ValueError("direct_output requires contiguous BF16 [B, H_q, D]")


    if plan is None:
        max_len = max_seq_len or int(seq_lens.max().item())
        splits, reduction = plan_qmul4_decode(
            B, H, max_len, q.device, grouped_head_tile=g_pad, head_dim=D,
            residue_fold=residue_fold,
        )
    else:
        splits, reduction = plan
    f = compile_decode_variant(decode_compile_spec(
        HQ, H, D, (splits, reduction),
        output_nvfp4=False, residue_fold=residue_fold, page_size=pool.page_size,
        k_sf_mode=k_sf_mode, direct_output=direct_output,
        partial_o_dtype=_reduction_workspace_partial_o_dtype(reduction_workspace),
    ))
    fused = _quantize_q_residue_cuda(q, H)
    if fused is None:
        raise RuntimeError(
            "decode QK residue requires the nvfp4 MExt CUDA extension"
        )
    q_pack, q_sf = fused
    o_partial = l_partial = m_partial = None
    if reduction == "kernel":
        if reduction_workspace is None:
            reduction_workspace = allocate_qmul4_reduction_workspace(
                splits, B, H * g_pad, D, q.device
            )
        o_partial, l_partial, m_partial = reduction_workspace
        expected_shapes = (
            (splits, B, 1, H * g_pad, D),
            (splits, B, 1, H * g_pad),
            (splits, B, 1, H * g_pad),
        )
        for index, (name, tensor, shape) in enumerate(zip(
            ("o_partial", "l_partial", "m_partial"),
            reduction_workspace,
            expected_shapes,
        )):
            expected_dtypes = (
                (torch.float32, torch.bfloat16)
                if index == 0
                else (torch.float32,)
            )
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype not in expected_dtypes
                or tensor.device != q.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous with dtype in "
                    f"{expected_dtypes}, shape {shape}, on "
                    f"{q.device}, got shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype} device={tensor.device}"
                )
    k_arg = pool.k_data.view(torch.float4_e2m1fn_x2).permute(0, 2, 1, 3)
    v_arg = v_data_hnd.view(torch.float4_e2m1fn_x2).permute(0, 2, 1, 3)
    f(
        Int32(splits),
        q_pack.view(torch.float4_e2m1fn_x2),
        k_arg,
        v_arg,
        q_sf,
        pool.k_sf,
        v_sf_bthd,
        out.view(B, 1, HQ, D),
        None, None, o_partial, l_partial, m_partial, None,
        Float32(sm_scale),
        Float32(value_scale),
        seq_lens,
        block_tables,
        False,
    )
    return out


def decode_compile_spec(num_qo_heads, num_kv_heads, head_dim, plan, *,
                        output_nvfp4=False, residue_fold=True,
                        k_sf_mode="linear", direct_output=True,
                        partial_o_dtype=None, page_size=32):
    """Build the decode kernel compilation specification."""
    if not residue_fold:
        raise NotImplementedError("ordinary QK is not supported")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if direct_output is False:
        raise NotImplementedError("padded output is not supported")
    validate_decode_plan(plan, output_nvfp4=output_nvfp4, head_dim=head_dim)
    splits, reduction = plan
    tile = _residue_padded_group_size(num_qo_heads, num_kv_heads)
    ssw = _use_single_softmax_scheduler(tile, reduction, "qmul4_tmem", splits)
    return dict(
        args=(num_kv_heads * tile, num_kv_heads, head_dim, 1, reduction,
              "qmul4_tmem", k_sf_mode, output_nvfp4,
              num_qo_heads),
        kwargs=dict(page_size=page_size,
                    single_softmax_wg=ssw, smem_correction_broadcast=ssw,
                    residue_fold=residue_fold,
                    partial_o_dtype=partial_o_dtype or (
                        "bf16" if reduction == "gmem_fused" else "fp32")),
    )


def decode_compile_variants(num_qo_heads, num_kv_heads, head_dim, *,
                             residue_fold, output_nvfp4, page_size=32):
    if not residue_fold:
        raise NotImplementedError("ordinary QK is not supported")
    check_shape_supported(num_qo_heads, num_kv_heads, head_dim)
    tile = _residue_padded_group_size(num_qo_heads, num_kv_heads)
    specs, seen = [], set()
    for plan in static_compile_plans(output_nvfp4=output_nvfp4,
                                     grouped_head_tile=tile, residue_fold=residue_fold,
                                     head_dim=head_dim):
        spec = decode_compile_spec(num_qo_heads, num_kv_heads, head_dim, plan,
                                    output_nvfp4=output_nvfp4, residue_fold=residue_fold, page_size=page_size)
        key = (spec["args"], tuple(sorted(spec["kwargs"].items())))
        if key not in seen:
            seen.add(key)
            specs.append(dict(spec, label=f"{plan[1]} ssw={spec['kwargs']['single_softmax_wg']} "
                                         f"nvfp4={output_nvfp4}"))
    return specs


def compile_decode_variant(plan: dict):
    """Compile one specialization from `decode_compile_variants` (cached)."""
    return _compiled(*plan["args"], **plan["kwargs"])
