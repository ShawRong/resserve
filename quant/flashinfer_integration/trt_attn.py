"""BF16 query/output adapter for FlashInfer's native TRT page32 decode."""
from functools import lru_cache


@lru_cache(maxsize=1)
def backend_info():
    import flashinfer
    from flashinfer.artifacts import ArtifactPath, CheckSumHash

    if (ArtifactPath.TRTLLM_GEN_FMHA != "resinfer-page32-bf16-v1/fmha/trtllm-gen/"
            or CheckSumHash.TRTLLM_GEN_FMHA != "0c2adfa0f7450809abea29bfb208802ac6fcd675ba1c258bc38234c203e3536f"):
        raise RuntimeError("Native BF16 decode requires the patched FlashInfer build; see README.md (Installation)")
    return dict(backend="flashinfer_trtllm_native_bf16", dispatch="flashinfer_native_default",
                flashinfer_version=flashinfer.__version__,
                flashinfer_revision=getattr(flashinfer, "__git_commit__", None),
                artifact=ArtifactPath.TRTLLM_GEN_FMHA,
                artifact_checksum=CheckSumHash.TRTLLM_GEN_FMHA)


class TrtDecode:
    """Reuse Q/counter buffers; FlashInfer owns kernel selection and launch."""

    def __init__(self, query, workspace):
        import torch
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

        self.metadata = backend_info()
        self.decode = trtllm_batch_decode_with_kv_cache
        self.workspace = workspace
        sms = torch.cuda.get_device_properties(query.device).multi_processor_count
        size = get_trtllm_gen_multi_ctas_kv_counter_bytes(query.shape[0], query.shape[1], sms)
        self.q8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
        self.counter = torch.zeros(size, dtype=torch.uint8, device=query.device)

    def run(self, query, kv, table, lengths, output, max_length, *,
            sm_scale=None, k_scale=1.0, v_scale=1.0):
        kd, ks, vd, vs = kv
        self.q8.copy_(query)
        self.decode(
            self.q8, (kd, vd), self.workspace, table, lengths, max_length,
            bmm1_scale=(query.shape[-1] ** -.5 if sm_scale is None else sm_scale) * k_scale,
            bmm2_scale=v_scale, out=output, kv_cache_sf=(ks, vs),
            multi_ctas_kv_counter_buffer=self.counter,
            backend="trtllm-gen", enable_pdl=False)
        return output
