# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResAttn decode with FlashInfer prefill and KV-cache writes."""
from dataclasses import replace
import os

import torch

from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
    FlashInferTrtllmAPIDecode,
    TRTLLMPrefill,
)

from nvfp4.cute_attn.kv_layout import NvFp4PagedKVPool

from .decode import ResAttnDecodeMixin
from .warmup import _register_decode_warmup


def _require_trtllm_decode(metadata):
    decode = metadata.decode
    if (not isinstance(decode, FlashInferTrtllmAPIDecode)
            or decode.kernel is not FlashInferDecodeKernel.TRTLLM_GEN):
        raise NotImplementedError(
            "ResAttn requires stock FlashInfer TRTLLM_GEN decode metadata "
            "for NVFP4 KV"
        )
    return decode


class ResAttnFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if os.environ.get("RESIDUE_ATTN_CUDAGRAPH", "0") == "1":
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("ResAttn decode does not support DCP")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self._cg_capture_seq_len = int(
            os.environ.get("RESIDUE_ATTN_CG_CAPTURE_LEN", "0")
        ) or vllm_config.model_config.max_model_len

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build=False) -> FlashInferMetadata:
        stock = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if stock.use_cascade or not stock.causal:
            raise NotImplementedError("ResAttn requires causal, non-cascade attention")
        if stock.num_prefill_tokens and not isinstance(stock.prefill, TRTLLMPrefill):
            raise NotImplementedError("This adapter requires stock TRTLLM prefill")
        n = stock.num_decodes
        if n:
            if n != stock.num_decode_tokens:
                raise NotImplementedError("ResAttn requires one token per decode")
            decode = _require_trtllm_decode(stock)
            table = decode.block_tables
            if (
                table.dtype != torch.int32 or table.ndim != 2
                or table.shape[0] != n or table.shape[1] < 1
                or table.stride(1) != 1 or table.stride(0) < table.shape[1]
            ):
                raise ValueError("Expected int32 block table with unit column stride")
        return stock

    def build_for_cudagraph_capture(self, common_attn_metadata):
        md = self.build(0, common_attn_metadata)
        md._resinfer_graph_capture = True
        if md.num_decodes:
            decode = _require_trtllm_decode(md)
            decode.max_seq_len = max(
                decode.max_seq_len, self._cg_capture_seq_len
            )
        return md


class ResAttnFlashInferImpl(ResAttnDecodeMixin, FlashInferImpl):
    can_return_lse_for_decode = False

    decode_backend = "resattn"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gqa = (
            self.num_heads // self.num_kv_heads
            if self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
            else 0
        )
        if (
            not self.is_kvcache_nvfp4
            or self.head_size not in (128, 256)
            or gqa not in (4, 6, 8)
        ):
            raise NotImplementedError(
                "ResAttn candidate requires NVFP4, D128/D256, and GQA4/6/8"
            )
        if (self.alibi_slopes is not None or self.window_left != -1
                or self.logits_soft_cap not in (None, 0.0)
                or self.sinks is not None
                or self.kv_sharing_target_layer_name is not None):
            raise NotImplementedError("Unsupported ResAttn decode attention variant")
        # Preserve BF16 Q for decode; stock converts only the prefill slice.
        # Set after super(): FlashInfer's constructor sets an instance value.
        self.supports_quant_query_input = False
        self._decode_reduction_workspaces = {}
        self._residue_pool_key = None
        self._residue_pool = None
        if self.decode_backend == "resattn":
            _register_decode_warmup(self.num_heads, self.num_kv_heads, self.head_size)

    def fused_output_quant_supported(self, quant_key):
        return False

    def _pool_for(self, kv_cache, layer):
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape), kv_cache.stride(),
               float(layer._k_scale_float), float(layer._v_scale_float))
        if key != self._residue_pool_key:
            k, v = kv_cache.split(self.num_kv_heads, dim=1)
            order = FlashInferBackend.get_kv_cache_stride_order()
            k = canonicalize_singleton_dim_strides(k.permute(*order))
            v = canonicalize_singleton_dim_strides(v.permute(*order))
            kd, ks = nvfp4_split_data_scale(k)
            vd, vs = nvfp4_split_data_scale(v)
            page_size = kd.shape[2]
            if page_size != 32 or tuple(kd.shape[1:]) != (self.num_kv_heads, page_size, self.head_size // 2):
                raise ValueError("ResAttn requires stock HND page32 with matching head dimension")
            self._residue_pool = NvFp4PagedKVPool(
                num_kv_heads=self.num_kv_heads, head_dim=self.head_size,
                k_data=kd, v_data=vd, k_sf=ks, v_sf=vs,
                page_size=page_size,
                k_global_scale=key[-2], v_global_scale=key[-1],
            )
            self._residue_pool_key = key
        return self._residue_pool

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("page32 attention returns BF16 output only")
        if attn_metadata is None:
            return output.fill_(0)
        if query.dtype != torch.bfloat16:
            raise ValueError("ResAttn needs original BF16 Q, not FP8 Q")
        if output.dtype != torch.bfloat16:
            raise ValueError("ResAttn output dtype does not match fusion mode")
        md = attn_metadata
        if md.num_prefill_tokens:
            # vLLM already wrote KV once before this forward.
            prefill_md = md
            q_pre, k_pre, v_pre = query, key, value
            bf16_output = output
            prefill_output = bf16_output
            if md.num_decodes:
                n, end = md.num_decode_tokens, md.num_actual_tokens
                prefill_md = replace(
                    md, num_actual_tokens=md.num_prefill_tokens,
                    num_decodes=0, num_decode_tokens=0, decode=None,
                    slot_mapping=md.slot_mapping[n:end])
                q_pre, k_pre, v_pre = query[n:end], key[n:end], value[n:end]
                prefill_output = bf16_output[n:end]
            pool = self._pool_for(kv_cache, layer)
            self._prefill_bf16(layer, q_pre, pool, prefill_md, prefill_output)
            if md.num_decodes:
                _require_trtllm_decode(md)
                self._decode_forward(query, self._pool_for(kv_cache, layer),
                                     md, bf16_output)
            return output
        if md.num_decodes:
            _require_trtllm_decode(md)
            pool = self._pool_for(kv_cache, layer)
            self._decode_forward(query, pool, md, output)
        return output


    def _prefill_bf16(self, layer, query, pool, md, output):
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache
        from vllm import _custom_ops
        from vllm.v1.attention.backends.flashinfer import _get_trtllm_workspace_buffer
        from quant.flashinfer_integration.trt_attn import backend_info
        backend_info()
        n = md.num_prefill_tokens
        q = query.reshape(-1, self.num_heads * self.head_size)[:n]
        q8, _ = _custom_ops.scaled_fp8_quant(q, scale=layer._q_scale)
        q8 = canonicalize_singleton_dim_strides(
            q8.reshape(n, self.num_heads, self.head_size))
        prefill = md.prefill
        trtllm_batch_context_with_kv_cache(
            query=q8, kv_cache=(pool.k_data, pool.v_data),
            workspace_buffer=_get_trtllm_workspace_buffer(),
            block_tables=prefill.block_tables.contiguous(),
            seq_lens=prefill.seq_lens, max_q_len=prefill.max_q_len,
            max_kv_len=prefill.max_seq_len, batch_size=md.num_prefills,
            cum_seq_lens_q=prefill.cum_seq_lens_q,
            cum_seq_lens_kv=prefill.cum_seq_lens_kv,
            bmm1_scale=self.scale * layer._q_scale_float * pool.k_global_scale,
            bmm2_scale=pool.v_global_scale,
            out=output.reshape(-1, self.num_heads, self.head_size)[:n],
            kv_cache_sf=(pool.k_sf, pool.v_sf), enable_pdl=False)
        self._native_bf16_prefill_calls = getattr(self, '_native_bf16_prefill_calls', 0) + 1
class ResAttnFlashInferBackend(FlashInferBackend):
    """Stock prefill/cache contract with ResAttn decode, including mixed batches."""
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["nvfp4"]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return ResAttnFlashInferImpl

    @staticmethod
    def get_builder_cls():
        return ResAttnFlashInferMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        # vLLM splits larger hybrid-cache allocation blocks into kernel pages.
        return [32]

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 256]

    @classmethod
    def supports_sink(cls):
        return False
class TrtBf16FlashInferImpl(ResAttnFlashInferImpl):
    decode_backend = "trtllm"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trt_decode_buffers = {}
    def _decode_forward(self, query, pool, md, output):
        from quant.flashinfer_integration.trt_attn import TrtDecode
        from vllm.v1.attention.backends.flashinfer import _get_trtllm_workspace_buffer
        if pool.page_size != 32:
            raise ValueError("Native-BF16 TRT decode requires --block-size 32")
        n = md.num_decode_tokens
        q = query.reshape(-1, self.num_heads, self.head_size)[:n]
        out = output.reshape(-1, self.num_heads, self.head_size)[:n]
        graph = getattr(md, "_resinfer_graph_capture", False)
        key = n if graph else 0
        buffer = self._trt_decode_buffers.get(key)
        if buffer is None or buffer.q8.shape[0] != n:
            buffer = self._trt_decode_buffers[key] = TrtDecode(q, _get_trtllm_workspace_buffer())
        decode = md.decode
        bound = decode.block_tables.shape[1] * pool.page_size
        length = bound if graph else max(1, decode.max_seq_len)
        buffer.run(
            q, (pool.k_data, pool.k_sf, pool.v_data, pool.v_sf),
            decode.block_tables, decode.seq_lens, out, length,
            sm_scale=self.scale, k_scale=pool.k_global_scale, v_scale=pool.v_global_scale)
class TrtBf16FlashInferBackend(ResAttnFlashInferBackend):
    @staticmethod
    def get_impl_cls():
        return TrtBf16FlashInferImpl
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32]
