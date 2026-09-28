# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidate-only launch adapter; workload, oracle, comparator and timer stay fixed."""

import math

from rocke.helpers import compile_kernel
from rocke.runtime.packing import pack_args
from kernels.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    wmma_fmha_fwd_grid,
)

class UnsupportedCase(Exception):
    """A documented absence of an implementation, not an execution failure."""


class RockeKernels:
    def __init__(self, rt):
        self.rt = rt
        self.cache = {}

    def prepare(self, case, buffers):
        # These gates describe the existing adapter, not hardware limitations.
        # Unsupported rows remain mandatory in the frozen coverage corpus.
        if case.layout != "dense":
            raise UnsupportedCase("gfx1151 has no paged/packed attention adapter")
        if case.dtype not in ("fp16", "bf16") or case.kv_dtype:
            raise UnsupportedCase("WMMA adapter requires matching fp16/bf16 Q/K/V storage")
        spec = WmmaFmhaFwdSpec(
            head_size=case.head_dim, num_query_heads=case.heads_q,
            num_kv_heads=case.heads_kv,
            mask_mode="none" if case.mask == "none" else "causal",
            dtype=case.dtype,
            causal_bottom_right=case.mask == "causal_bottomright",
            query_tail=case.seqlen_q % 16 != 0,
            kv_tail=case.seqlen_k % 16 != 0,
            sliding_window=case.window,
            use_softcap=case.softcap > 0,
            use_sinks=case.sinks,
            use_alibi=case.alibi,
            use_qq_bias=case.qq_bias,
        )
        key = spec.kernel_name()
        if key not in self.cache:
            kernel = build_wmma_fmha_fwd(spec, arch="gfx1151")
            art = compile_kernel(
                kernel, arch="gfx1151", backend="python", capture_ir_text=False,
            )
            module = self.rt.load_module(art.hsaco)
            signature = tuple({"name": param.name, "type": param.type.name} for param in kernel.params)
            self.cache[key] = (module, module.get_function(art.kernel_name), signature)
        _, fn, signature = self.cache[key]
        q, k, v = (buffers.arrays[name] for name in ("q", "k", "v"))
        values = {
            "Q": buffers.ptrs["q"], "K": buffers.ptrs["k"], "V": buffers.ptrs["v"],
            "O": buffers.ptrs["rocke_out"], "scale_log2": case.scale * math.log2(math.e),
            "seqlen_q": case.seqlen_q, "seqlen_k": case.seqlen_k,
            "stride_q_token": q.strides[-3] // q.itemsize,
            "stride_q_head": q.strides[-2] // q.itemsize,
            "stride_k_token": k.strides[-3] // k.itemsize,
            "stride_k_head": k.strides[-2] // k.itemsize,
            "stride_v_token": v.strides[-3] // v.itemsize,
            "stride_v_head": v.strides[-2] // v.itemsize,
            "stride_o_token": q.strides[-3] // q.itemsize,
            "stride_o_head": q.strides[-2] // q.itemsize,
        }
        if spec.use_softcap:
            values["softcap"] = case.softcap
        if spec.use_sinks:
            values["sink_ptr"] = buffers.ptrs["sinks"]
        if spec.use_alibi:
            values["alibi_slopes_ptr"] = buffers.ptrs["alibi_slopes"]
        if spec.use_qq_bias:
            bias = buffers.arrays["qq_bias"]
            values.update(
                qq_bias_ptr=buffers.ptrs["qq_bias"],
                qq_bias_rows=bias.shape[0], qq_bias_cols=bias.shape[1],
                qq_bias_stride=bias.strides[0] // bias.itemsize,
            )
        packed = pack_args(signature, values)
        grid = wmma_fmha_fwd_grid(spec, seqlen_q=case.seqlen_q, batch=case.batch)

        def launch(stream):
            self.rt.launch(fn, grid, (spec.block_size, 1, 1), packed,
                           stream=stream, record_event=False)

        return launch, key

    def close(self):
        self.rt.sync()
        for module, _, _ in self.cache.values():
            module.unload()
        self.cache.clear()


