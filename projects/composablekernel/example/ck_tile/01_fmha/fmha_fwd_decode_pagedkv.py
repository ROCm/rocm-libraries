# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""PyTorch wrapper of CK fmha_fwd_decode_pagedkv: gfx125 decode over a paged fp8 KV cache.

pa_decode takes the arguments of aiter's FlyDSL pa_decode for the case it covers: one query
token, bf16 or fp16 Q/O, head dim 128, fp8 e4m3 K/V with per-tensor scales, pages of 16, 64 or
128 tokens. Load libck_fmha_fwd_decode_pagedkv.so (target
ck_fmha_fwd_decode_pagedkv) from CK_FMHA_FWD_DECODE_PAGEDKV_LIB or the library search path.
"""

import ctypes
import math
import os

import torch

HDIM = 128
PAGE_SIZES = (16, 64, 128)

_lib = None
_buffers = {}
_scales = {}
# distinct scale numbers kept per device (captured graphs may hold their tensors)
_MAX_SCALE_NUMBERS = 64

# aiter pa_decode arguments this kernel has no use for (its own scratch and tuning)
_IGNORED = ("context_partition_size", "exp_sums", "max_logits", "temporary_output")
# aiter pa_decode arguments accepted at their defaults only
_DEFAULTS = {"mask_unowned_kv": True, "store_output": True, "quantized_pad_rows": 0}


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(
            os.environ.get(
                "CK_FMHA_FWD_DECODE_PAGEDKV_LIB", "libck_fmha_fwd_decode_pagedkv.so"
            )
        )
        lib.ck_fmha_fwd_decode_pagedkv_max_workspace_bytes.restype = ctypes.c_size_t
        lib.ck_fmha_fwd_decode_pagedkv_max_workspace_bytes.argtypes = [ctypes.c_int] * 2
        lib.ck_fmha_fwd_decode_pagedkv_num_counters.restype = ctypes.c_size_t
        lib.ck_fmha_fwd_decode_pagedkv_num_counters.argtypes = [ctypes.c_int] * 2
        lib.ck_fmha_fwd_decode_pagedkv.restype = ctypes.c_int
        lib.ck_fmha_fwd_decode_pagedkv.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int]
            + [ctypes.c_void_p] * 6
            + [ctypes.c_int] * 7
            + [ctypes.c_float]
            + [ctypes.POINTER(ctypes.c_int)] * 2
            + [ctypes.c_void_p]
        )
        _lib = lib
    return _lib


def _buffer(kind, nbytes, device):
    """Scratch workspace or zeroed counters, shared by all calls on a device; grows on demand.

    Each call leaves the counters at zero for the next. A call that finds them non-zero leaves
    output rows unwritten without an error: calls on one device must not run concurrently (two
    streams, two graphs replayed at once), and after a failed call (a device fault) use
    reset_buffers(). The buffers are per device and not per stream because a graph is captured
    on a stream of its own and replayed on another.
    Both sizes depend only on num_seqs * num_kv_heads: make the first call at the largest batch
    eagerly, since growing is not possible during graph capture. Outgrown buffers are kept since
    captured graphs may still use them.
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


def reset_buffers():
    """Zeroes the counters of every device again; call after a failed launch, not during capture."""
    for (kind, _), buffers in _buffers.items():
        if kind == "counters":
            for b in buffers:
                b.copy_(torch.zeros(b.numel(), dtype=torch.uint8))


def _scale_ptr(scale, device):
    if scale is None:
        scale = 1.0
    if isinstance(scale, (int, float)):
        # a host number: kept on the device, one tensor per value
        value = float(scale)
        if (device, value) not in _scales:
            if sum(d == device for d, _ in _scales) >= _MAX_SCALE_NUMBERS:
                raise ValueError(
                    "too many distinct key_scale / value_scale numbers: pass a float32 tensor"
                )
            _scales[(device, value)] = torch.tensor([value], dtype=torch.float32).to(
                device
            )
        scale = _scales[(device, value)]
    if (
        not isinstance(scale, torch.Tensor)
        or scale.numel() != 1
        or scale.dtype != torch.float32
    ):
        raise ValueError(
            "key_scale / value_scale must be a number or one float32 element (per-tensor)"
        )
    return scale.data_ptr()


def pa_decode(
    output,  # [num_seqs, num_query_heads, 128], the type of query
    query,  # [num_seqs, num_query_heads, 128] bf16 or fp16; both may be strided over seqs and heads
    key_cache,  # [num_blocks, num_kv_heads, 128 // 16, page_size, 16] fp8 e4m3
    value_cache,  # [num_blocks, num_kv_heads, page_size // 16, 128, 16] fp8 e4m3
    context_lengths,  # [num_seqs] int32
    block_tables,  # [num_seqs, max_num_blocks_per_seq] int32
    softmax_scale,
    query_length=1,
    *,
    key_scale=None,  # a number or [1] float32 on the device; both or neither
    value_scale=None,
    sliding_window=0,  # keys attended, the query's own included; 0 = no window
    max_context_length=None,
    work_plan=None,  # accepted for call compatibility; this kernel needs no plan
    compute_type=None,  # None or the cache dtype
    **unsupported,
):
    """Decode attention of one query token per sequence.

    The launch configuration depends on the batch, the kv heads, sliding_window and
    max_context_length (default: block_tables.shape[1] * page_size), not on context_lengths, so the
    call can be captured in a HIP graph. max_context_length must bound every context length.
    """
    for name, value in unsupported.items():
        if name in _IGNORED:
            continue
        if name in _DEFAULTS:
            if value != _DEFAULTS[name]:
                raise NotImplementedError(
                    f"CK pa_decode supports {name}={_DEFAULTS[name]} only"
                )
            continue
        if isinstance(value, torch.Tensor) or value not in (None, False, 0):
            raise NotImplementedError(f"CK pa_decode does not support {name}")
    if compute_type is not None and compute_type != key_cache.dtype:
        raise NotImplementedError("CK pa_decode computes in the fp8 type of the caches")
    if (key_scale is None) != (value_scale is None):
        raise ValueError(
            "key_scale and value_scale must either both be provided or both be None"
        )
    if not (math.isfinite(softmax_scale) and softmax_scale > 0):
        raise ValueError(
            f"softmax_scale must be finite and positive, got {softmax_scale}"
        )
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
    if nhead_q % nhead_k != 0:
        raise ValueError("num_query_heads must be a multiple of num_kv_heads")
    for t in (key_cache, value_cache, context_lengths):
        if not t.is_contiguous():
            raise ValueError("the caches and context_lengths must be contiguous")
    if query.stride(2) != 1 or output.stride(2) != 1:
        raise ValueError("query and output must have unit stride along the head dim")
    o_batch, o_head = _strides(output)
    if o_head < HDIM or o_batch < (nhead_q - 1) * o_head + HDIM:
        raise ValueError("output rows must not overlap")
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
    # the library launches on, and looks kernels up for, the current device
    with torch.cuda.device(device):
        _launch(
            lib,
            device,
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            key_scale,
            value_scale,
            batch,
            nhead_q,
            nhead_k,
            max_context_length,
            sliding_window,
            page,
        )


def _strides(t):
    """(batch, head) strides in elements; those of a contiguous tensor for axes of size 1."""
    _, nhead, hdim = t.shape
    head = t.stride(1) if nhead > 1 else hdim
    batch = t.stride(0) if t.shape[0] > 1 else (nhead - 1) * head + hdim
    return batch, head


def _launch(
    lib,
    device,
    output,
    query,
    key_cache,
    value_cache,
    context_lengths,
    block_tables,
    softmax_scale,
    key_scale,
    value_scale,
    batch,
    nhead_q,
    nhead_k,
    max_context_length,
    sliding_window,
    page,
):
    stream = torch.cuda.current_stream(device).cuda_stream
    # a kv head takes one row of workgroups per 16 of its query heads
    grid_rows = nhead_k * max(1, -(-(nhead_q // nhead_k) // 16))
    ws = _buffer(
        "workspace",
        lib.ck_fmha_fwd_decode_pagedkv_max_workspace_bytes(batch, grid_rows),
        device,
    )
    counters = _buffer(
        "counters",
        lib.ck_fmha_fwd_decode_pagedkv_num_counters(batch, grid_rows) * 4,
        device,
    )
    err = lib.ck_fmha_fwd_decode_pagedkv(
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
        (ctypes.c_int * 2)(*_strides(query)),
        (ctypes.c_int * 2)(*_strides(output)),
        stream,
    )
    if err != 0:
        raise RuntimeError(f"ck_fmha_fwd_decode_pagedkv failed: hipError_t {err}")
