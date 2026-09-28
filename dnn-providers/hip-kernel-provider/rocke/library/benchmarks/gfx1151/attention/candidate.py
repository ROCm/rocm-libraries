# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidate-only launch adapter; workload, oracle, comparator and timer stay fixed."""

from types import SimpleNamespace

from dispatch.attention import AttentionMaskType, AttentionRequest, dispatch_attention
from rocke.runtime.launcher import release_retained_for_stream

class UnsupportedCase(Exception):
    """A documented absence of an implementation, not an execution failure."""


class _HipTensor:
    """Device-tensor protocol over a real, caller-owned HIP allocation."""

    is_cuda = True
    device = SimpleNamespace(type="cuda", index=0)

    def __init__(self, owner, name):
        self._owner = owner
        self._array = owner.arrays[name]
        self._pointer = owner.ptrs[name]
        self.shape = self._array.shape
        self.ndim = self._array.ndim
        self.dtype = self._array.dtype
        self.strides = tuple(stride // self._array.itemsize for stride in self._array.strides)

    def data_ptr(self):
        return self._pointer

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def element_size(self):
        return self._array.itemsize

    def numel(self):
        return self._array.size


class RockeKernels:
    def __init__(self, rt):
        self.rt = rt
        self.cache = {}
        self._streams = set()

    def prepare(self, case, buffers):
        masks = {
            "none": AttentionMaskType.NO_MASK,
            "causal_topleft": AttentionMaskType.TOP_LEFT_CAUSAL,
            "causal_bottomright": AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        }
        request = AttentionRequest(
            batch=case.batch, nhead_q=case.heads_q, nhead_k=case.heads_kv,
            seqlen_q=max(case.q_lengths, default=case.seqlen_q),
            seqlen_k=max(case.k_lengths, default=case.seqlen_k),
            hdim_q=case.head_dim, hdim_v=case.head_dim,
            arch="gfx1151", dtype=case.dtype, mask_type=masks[case.mask],
            layout=case.layout, kv_block_size=case.block_size or 16,
            sliding_window=case.window, use_softcap=case.softcap > 0,
            use_sinks=case.sinks, use_alibi=case.alibi, use_qq_bias=case.qq_bias,
            use_fp8=bool(case.kv_dtype),
        )
        if request not in self.cache:
            self.cache[request] = dispatch_attention(request)
        selection = self.cache[request]
        tensors = {
            name: _HipTensor(buffers, name)
            for name in (
                "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "seqused_k",
                "block_table", "sinks", "alibi_slopes", "qq_bias",
            )
            if name in buffers.ptrs
        }
        tensors["out"] = _HipTensor(buffers, "rocke_out")
        scalars = {"softmax_scale": case.scale, "softcap": case.softcap}
        if case.kv_dtype:
            scalars.update(
                k_scale=float(buffers.arrays["k_scale"].item()),
                v_scale=float(buffers.arrays["v_scale"].item()),
            )
        binding = selection.bind_torch(tensors, fence=False, **scalars)

        def launch(stream):
            self._streams.add(stream)
            binding.launch(stream=stream)

        return launch, selection.spec.kernel_name()

    def release(self, stream):
        """Release launch owners after the benchmark destroys its synced graphs."""
        release_retained_for_stream(stream)
        self._streams.discard(stream)

    def close(self):
        self.rt.sync()
        for stream in tuple(self._streams):
            self.release(stream)
        self.cache.clear()


