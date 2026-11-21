# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""PyTorch wrapper of the gfx125 CK fp8 decode kernel over the SHUFFLE 5D paged KV cache.

pa_decode takes the arguments of aiter's FlyDSL pa_decode for the case it covers: one query
token, bf16 or fp16 Q/O, head dim 128, fp8 e4m3 K/V with per-tensor scales, pages of 16, 64 or
128 tokens, at most 16 query heads per kv head. Load libck_fmha_decode_paged5d.so (target
ck_fmha_decode_paged5d) from CK_FMHA_DECODE_PAGED5D_LIB or the library search path.
"""

import ctypes
import os

import torch

HDIM = 128
PAGE_SIZES = (16, 64, 128)

_lib = None
_buffers = {}
_ones = {}


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(
            os.environ.get("CK_FMHA_DECODE_PAGED5D_LIB", "libck_fmha_decode_paged5d.so")
        )
        lib.ck_fmha_decode_paged5d_workspace_bytes.restype = ctypes.c_size_t
        lib.ck_fmha_decode_paged5d_workspace_bytes.argtypes = [ctypes.c_int] * 5
        lib.ck_fmha_decode_paged5d_num_counters.restype = ctypes.c_size_t
        lib.ck_fmha_decode_paged5d_num_counters.argtypes = [ctypes.c_int] * 2
        lib.ck_fmha_decode_paged5d.restype = ctypes.c_int
        lib.ck_fmha_decode_paged5d.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int]
            + [ctypes.c_void_p] * 6
            + [ctypes.c_int] * 7
            + [ctypes.c_float, ctypes.c_void_p]
        )
        _lib = lib
    return _lib


def _buffer(kind, nbytes, device):
    """Scratch workspace or zeroed counters, shared by all calls on a device; grows on demand.

    Calls must not run concurrently (the kernel leaves the counters at zero for the next call).
    Make the first call at the largest shape eagerly: growing is not possible during graph
    capture. Outgrown buffers are kept since captured graphs may still use them.
    """
    buffers = _buffers.setdefault((kind, device), [])
    if not buffers or buffers[-1].numel() < nbytes:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CK pa_decode buffers must be allocated before graph capture"
            )
        if kind == "counters":
            # built on the host and copied: needs no device kernel of torch
            buffers.append(torch.zeros(nbytes, dtype=torch.uint8).to(device))
        else:
            buffers.append(
                torch.empty(max(nbytes, 1), dtype=torch.uint8, device=device)
            )
    return buffers[-1]


def _scale_ptr(scale, device):
    if scale is None:
        if device not in _ones:
            _ones[device] = torch.ones(1, dtype=torch.float32).to(device)
        scale = _ones[device]
    if (
        not isinstance(scale, torch.Tensor)
        or scale.numel() != 1
        or scale.dtype != torch.float32
    ):
        raise ValueError(
            "key_scale / value_scale must be one float32 element (per-tensor)"
        )
    return scale.data_ptr()


def pa_decode(
    output,  # [num_seqs, num_query_heads, 128], the type of query
    query,  # [num_seqs, num_query_heads, 128] bf16 or fp16
    key_cache,  # [num_blocks, num_kv_heads, 128 // 16, page_size, 16] fp8 e4m3
    value_cache,  # [num_blocks, num_kv_heads, page_size // 16, 128, 16] fp8 e4m3
    context_lengths,  # [num_seqs] int32
    block_tables,  # [num_seqs, max_num_blocks_per_seq] int32
    softmax_scale,
    query_length=1,
    *,
    key_scale=None,  # [1] float32 on the device
    value_scale=None,  # [1] float32 on the device
    sliding_window=0,  # keys attended, the query's own included; 0 = no window
    max_context_length=None,
    work_plan=None,  # accepted for call compatibility; this kernel needs no plan
    **unsupported,
):
    """Decode attention of one query token per sequence.

    The launch configuration depends on the batch, the kv heads, sliding_window and
    max_context_length (default: block_tables.shape[1] * page_size), not on context_lengths, so the
    call can be captured in a HIP graph. max_context_length must bound every context length.
    """
    for name, value in unsupported.items():
        if isinstance(value, torch.Tensor) or value not in (None, False, 0):
            raise NotImplementedError(f"CK pa_decode does not support {name}")
    if query_length != 1:
        raise NotImplementedError("CK pa_decode supports query_length 1 only")
    batch, nhead_q, hdim = query.shape
    num_blocks, nhead_k = key_cache.shape[:2]
    page = key_cache.shape[3] if key_cache.dim() == 5 else 0
    if hdim != HDIM or output.shape != query.shape:
        raise ValueError("query and output must be [num_seqs, num_query_heads, 128]")
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or output.dtype != query.dtype
    ):
        raise ValueError("query must be bf16 or fp16 and output of the same type")
    if key_cache.element_size() != 1 or value_cache.element_size() != 1:
        raise ValueError("key_cache and value_cache must be fp8 e4m3")
    if page not in PAGE_SIZES or tuple(key_cache.shape) != (
        num_blocks,
        nhead_k,
        HDIM // 16,
        page,
        16,
    ):
        raise ValueError(
            "key_cache must be [num_blocks, num_kv_heads, 8, page_size, 16], page_size 16, 64 or 128"
        )
    if tuple(value_cache.shape) != (num_blocks, nhead_k, page // 16, HDIM, 16):
        raise ValueError(
            "value_cache must be [num_blocks, num_kv_heads, page_size // 16, 128, 16]"
        )
    if nhead_q % nhead_k != 0 or nhead_q // nhead_k > 16:
        raise ValueError(
            "num_query_heads must be a multiple of num_kv_heads, at most 16 times"
        )
    for t in (output, query, key_cache, value_cache, context_lengths):
        if not t.is_contiguous():
            raise ValueError(
                "output, query, the caches and context_lengths must be contiguous"
            )
    if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
        raise ValueError("block_tables and context_lengths must be int32")
    if block_tables.stride(1) != 1 or block_tables.shape[0] < batch:
        raise ValueError("block_tables must have unit column stride and num_seqs rows")
    if context_lengths.numel() < batch:
        raise ValueError("context_lengths must have num_seqs entries")
    tensors = [output, query, key_cache, value_cache, context_lengths, block_tables]
    tensors += [s for s in (key_scale, value_scale) if isinstance(s, torch.Tensor)]
    if any(t.device != query.device for t in tensors):
        raise ValueError("all tensors must be on the query's device")
    if max_context_length is None:
        max_context_length = block_tables.shape[1] * page

    lib = _load()
    device = query.device
    stream = torch.cuda.current_stream(device).cuda_stream
    nbytes = lib.ck_fmha_decode_paged5d_workspace_bytes(
        batch, nhead_k, max_context_length, sliding_window, page
    )
    ws = _buffer("workspace", nbytes, device)
    counters = _buffer(
        "counters", lib.ck_fmha_decode_paged5d_num_counters(batch, nhead_k) * 4, device
    )
    err = lib.ck_fmha_decode_paged5d(
        query.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        block_tables.data_ptr(),
        block_tables.stride(0),
        context_lengths.data_ptr(),
        _scale_ptr(key_scale, device),
        _scale_ptr(value_scale, device),
        output.data_ptr(),
        ws.data_ptr(),
        counters.data_ptr(),
        batch,
        nhead_q,
        nhead_k,
        max_context_length,
        sliding_window,
        page,
        1 if query.dtype == torch.float16 else 0,
        float(softmax_scale),
        stream,
    )
    if err != 0:
        raise RuntimeError(f"ck_fmha_decode_paged5d failed: hipError_t {err}")
