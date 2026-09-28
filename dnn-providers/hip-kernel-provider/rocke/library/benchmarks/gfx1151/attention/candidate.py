# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidate-only launch adapter; workload, oracle, comparator and timer stay fixed."""

import math
import struct

from rocke.helpers import compile_kernel
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
        if case.window or case.softcap or case.sinks or case.alibi or case.qq_bias:
            raise UnsupportedCase("existing WMMA adapter lacks this score/mask feature")
        spec = WmmaFmhaFwdSpec(
            head_size=case.head_dim, num_query_heads=case.heads_q,
            num_kv_heads=case.heads_kv,
            mask_mode="none" if case.mask == "none" else "causal",
            dtype=case.dtype,
            causal_bottom_right=case.mask == "causal_bottomright",
            query_tail=case.seqlen_q % 16 != 0,
            kv_tail=case.seqlen_k % 16 != 0,
        )
        key = spec.kernel_name()
        if key not in self.cache:
            art = compile_kernel(
                build_wmma_fmha_fwd(spec, arch="gfx1151"), arch="gfx1151",
                backend="python", capture_ir_text=False,
            )
            module = self.rt.load_module(art.hsaco)
            self.cache[key] = (module, module.get_function(art.kernel_name))
        _, fn = self.cache[key]
        q, k, v = (buffers.arrays[name] for name in ("q", "k", "v"))
        packed = struct.pack(
            "<QQQQfiiiiiiiiii",
            buffers.ptrs["q"], buffers.ptrs["k"], buffers.ptrs["v"],
            buffers.ptrs["rocke_out"], case.scale * math.log2(math.e),
            case.seqlen_q, case.seqlen_k,
            q.strides[-3] // q.itemsize, q.strides[-2] // q.itemsize,
            k.strides[-3] // k.itemsize, k.strides[-2] // k.itemsize,
            v.strides[-3] // v.itemsize, v.strides[-2] // v.itemsize,
            q.strides[-3] // q.itemsize, q.strides[-2] // q.itemsize,
        )
        grid = wmma_fmha_fwd_grid(spec, seqlen_q=case.seqlen_q, batch=case.batch)

        def launch(stream):
            self.rt.launch(fn, grid, (spec.block_size, 1, 1), packed,
                           stream=stream, record_event=False)

        return launch, key

    def close(self):
        self.rt.sync()
        for module, _ in self.cache.values():
            module.unload()
        self.cache.clear()


