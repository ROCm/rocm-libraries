# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from: https://github.com/ROCm/aiter
#   commit:   a9918248830460694b50982be606bb062228193d
#   path:     aiter/ops/flydsl/kernels/causal_conv1d_prefill.py
#   compiler: flydsl 0.3.4 (pip wheel)   toolchain: ROCm 7.1.0
#
# The per-sequence path takes the decode structure of the same commit's
# aiter/ops/flydsl/kernels/causal_conv1d_update.py (build_causal_conv1d_update_module,
# is_varlen decode): history from the state, roll the window into the state, every
# load issued before the guarded stores, scalar (SMEM) loads of the per-sequence
# offsets through kernels.common.buffer_ops.
#
# Modifications are listed in the module docstring.
#
# Upstream-first: prefer landing these changes in AITER; this copy exists so the
# provider is not blocked on that. Re-diff with tools/diff_upstream.py.

"""Depthwise causal 1-D convolution over token-packed sequences, hipDNN semantics.

For sequence ``n`` (tokens ``cu[n] .. cu[n+1]`` of ``x [T, C]``) and channel ``c``,
let ``xs`` be the ``W - 1`` state columns (oldest first) followed by the sequence's
tokens::

    y[t, c]              = act(bias[c] + sum_{j < W} w[c, j] * xs[t + j, c])
    final[row(n), c, :]  = the last W - 1 entries of xs

``row(n)`` is ``state_indices[n]`` when a pool is addressed, else ``n``. An absent
initial state is zeros; ``final`` may alias ``initial``.

**Work split.** Grid x is ``N`` sequence blocks followed by ``ceil(T / TILE)`` token
tiles; grid y covers the channels, ``block_threads * lanes`` per block, ``lanes``
contiguous channels per thread (the upstream prefill's channel-to-lane mapping).

* A **sequence block** owns everything that touches the state: the first
  ``min(len, W - 1)`` outputs of its sequence, whose window reaches into the initial
  state, and the final state. It reads the whole initial row before it writes the
  final row, and no other block reads that row, which is what makes the in-place
  update (``final`` bound to ``initial``'s buffer) race-free. A decode step is one
  sequence block per sequence.
* A **token tile** convolves ``TILE`` consecutive packed tokens with the upstream
  prefill's register-resident sliding window, but over *global* token positions: a
  token at position ``p >= W - 1`` of its sequence has its whole window inside the
  sequence, so its taps are just the previous ``W - 1`` rows of ``x``. Tokens at
  ``p < W - 1`` belong to the sequence blocks and are skipped; the tile finds them
  from the sequence starts in its window (two binary searches over ``cu_seqlens``).

The grid depends only on ``T``, ``N`` and ``C`` -- never on the sequence lengths,
which live on the device -- and every distribution of lengths, including empty
sequences, is load-balanced.

Modifications from the upstream prefill kernel:

1. Restructured into sequence blocks and token tiles (above). Upstream launches a
   ``(chunk, sequence)`` grid sized by a host-side ``max_len``, skips empty
   sequences entirely and reads and writes one in-place ``conv_states`` tensor.
2. Separate initial and final states, each optional, with their own strides; the
   per-sequence ``has_initial_state`` flags and ``pad_slot_id`` are gone (hipDNN has
   neither: an absent initial state is zeros, and every row is in range). A row
   outside ``[0, state_rows)`` is not read or written, so a malformed index stays
   in bounds.
3. Every operand is a raw 64-bit address plus runtime element strides, so the
   kernarg segment is plain scalars; x and y have a unit channel stride. ``T``,
   ``N``, ``C``, the pool size and the optional-operand flags are runtime.
4. ``cu_seqlens`` and ``state_indices`` are read as 32-bit words with a runtime
   step of 1 (int32) or 2 (int64, the low word: offsets and rows are non-negative
   and below 2^31).
5. Width, dtype, activation, ``lanes``, ``block_threads`` and ``TILE`` are the only
   build-time values. The upstream ``torch`` dtype argument is a string.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr

from kernels.common import buffer_ops

KERNEL_NAME = "causal_conv1d_kernel"

# The optional operands, one bit each in the `flags` kernarg.
FLAG_BIAS = 1
FLAG_INITIAL_STATE = 2
FLAG_FINAL_STATE = 4
FLAG_STATE_INDICES = 8

# Kernel widths the builder accepts: the head-token mask below needs
# tile_tokens + 2 * W - 5 < 64.
MIN_WIDTH = 2
MAX_WIDTH = 6

# Tokens loaded ahead of the convolution in a tile, as upstream's `prefetch`.
PREFETCH = 8

_ELEMENT = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}


def build_causal_conv1d_module(
    width: int,
    dtype_str: str,
    silu: bool,
    lanes: int,
    block_threads: int,
    tile_tokens: int,
):
    """The launcher for one (width, dtype, activation, schedule) object."""
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise ValueError(f"width {width} outside [{MIN_WIDTH}, {MAX_WIDTH}]")
    if tile_tokens < 1 or tile_tokens + 2 * width - 5 >= 64:
        raise ValueError(f"tile_tokens {tile_tokens} too large for width {width}")
    if lanes < 1 or block_threads < 1:
        raise ValueError("lanes and block_threads must be positive")
    element = _ELEMENT[dtype_str]
    W = width
    TILE = tile_tokens
    # One bit per output a sequence start hands to its sequence block.
    head_bits = (1 << (W - 1)) - 1

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def causal_conv1d_kernel(
        x_ptr: fx.Int64,
        weight_ptr: fx.Int64,
        bias_ptr: fx.Int64,
        initial_ptr: fx.Int64,
        final_ptr: fx.Int64,
        cu_seqlens_ptr: fx.Int64,
        state_indices_ptr: fx.Int64,
        y_ptr: fx.Int64,
        x_stride: fx.Int64,
        y_stride: fx.Int64,
        weight_stride_c: fx.Int64,
        weight_stride_k: fx.Int64,
        bias_stride: fx.Int64,
        initial_stride_n: fx.Int64,
        initial_stride_c: fx.Int64,
        initial_stride_k: fx.Int64,
        final_stride_n: fx.Int64,
        final_stride_c: fx.Int64,
        final_stride_k: fx.Int64,
        tokens: fx.Int32,
        seqs: fx.Int32,
        channels: fx.Int32,
        state_rows: fx.Int32,
        cu_seqlens_step: fx.Int32,
        state_indices_step: fx.Int32,
        flags: fx.Int32,
    ):
        ptr_ty = fx.Pointer[element, fx.AddressSpace.Global].ir_type
        vec_ty = ir.VectorType.get([lanes], element.ir_type)
        x = fx.inttoptr(ptr_ty, x_ptr)
        w = fx.inttoptr(ptr_ty, weight_ptr)
        bias = fx.inttoptr(ptr_ty, bias_ptr)
        initial = fx.inttoptr(ptr_ty, initial_ptr)
        final = fx.inttoptr(ptr_ty, final_ptr)
        y = fx.inttoptr(ptr_ty, y_ptr)
        cu_rsrc = buffer_ops.create_buffer_resource_from_addr(cu_seqlens_ptr)
        index_rsrc = buffer_ops.create_buffer_resource_from_addr(state_indices_ptr)

        def load_word(rsrc, index, step):
            # Uniform: lands in SGPRs through the scalar cache.
            return fx.Int32(
                buffer_ops.buffer_load(
                    rsrc, index * step, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )

        def seq_offset(m):
            return load_word(cu_rsrc, m, cu_seqlens_step)

        def lower_bound(target):
            # First m in [0, N] with cu[m] >= target; cu is non-decreasing.
            lo = fx.Int32(0)
            hi = seqs + 1
            while lo < hi:
                mid = (lo + hi) // 2
                right = seq_offset(mid) < target
                lo = right.select(mid + 1, lo)
                hi = right.select(hi, mid)
            return lo

        def load_row(ptr, offset):
            # `lanes` contiguous channels of one token row.
            return fx.Vector(fx.ptr_load(ptr + offset, vec_ty)).to(fx.Float32)

        def load_columns(ptr, offset, stride_c, stride_k, cols):
            # [lanes, cols] -> cols vectors over the lanes. Packed (the cols of one
            # channel contiguous, channels adjacent) is one vector load per thread:
            # per-element loads at a channel stride cost a TA pass per element.
            columns = [
                fx.Vector.filled(lanes, 0.0, fx.Float32) for _ in range_constexpr(cols)
            ]
            if (stride_k == 1) & (stride_c == cols):
                block_ty = ir.VectorType.get([lanes * cols], element.ir_type)
                block = fx.Vector(fx.ptr_load(ptr + offset, block_ty)).to(fx.Float32)
                columns = [
                    fx.Vector.from_elements(
                        [block[lane * cols + k] for lane in range_constexpr(lanes)]
                    )
                    for k in range_constexpr(cols)
                ]
            else:
                columns = [
                    fx.Vector.from_elements(
                        [
                            ptr[offset + lane * stride_c + k * stride_k]
                            for lane in range_constexpr(lanes)
                        ]
                    ).to(fx.Float32)
                    for k in range_constexpr(cols)
                ]
            return columns

        def store_row(ptr, offset, value):
            fx.ptr_store(value.to(element), ptr + offset)

        def store_columns(ptr, offset, stride_c, stride_k, columns):
            cols = len(columns)
            narrow = [column.to(element) for column in columns]
            if (stride_k == 1) & (stride_c == cols):
                block = fx.Vector.from_elements(
                    [
                        narrow[k][lane]
                        for lane in range_constexpr(lanes)
                        for k in range_constexpr(cols)
                    ]
                )
                fx.ptr_store(block, ptr + offset)
            else:
                for lane in range_constexpr(lanes):
                    for k in range_constexpr(cols):
                        fx.ptr_store(
                            narrow[k][lane],
                            ptr + (offset + lane * stride_c + k * stride_k),
                        )

        def convolve(window, value, weights, base):
            acc = base
            for j in range_constexpr(W - 1):
                acc = acc + window[j] * weights[j]
            acc = acc + value * weights[W - 1]
            if const_expr(silu):
                acc = acc / (fx.Vector.filled(lanes, 1.0, fx.Float32) + fx.exp(-acc))
            return acc

        def sequence_block(n, c, weights, base):
            start = seq_offset(n)
            length = seq_offset(n + 1) - start
            row = n
            if (flags & FLAG_STATE_INDICES) != 0:
                row = load_word(index_rsrc, n, state_indices_step)
            row_ok = (row >= 0) & (row < state_rows)
            row64 = fx.Int64(row)

            history = [
                fx.Vector.filled(lanes, 0.0, fx.Float32) for _ in range_constexpr(W - 1)
            ]
            if ((flags & FLAG_INITIAL_STATE) != 0) & row_ok:
                history = load_columns(
                    initial,
                    row64 * initial_stride_n + c * initial_stride_c,
                    initial_stride_c,
                    initial_stride_k,
                    W - 1,
                )

            # The outputs whose window reaches into the state. Loads past the
            # sequence aim at row 0, which exists (T >= 1); their stores are skipped.
            window = list(history)
            for p in range_constexpr(W - 1):
                valid = fx.Int32(p) < length
                position = valid.select(start + p, fx.Int32(0))
                value = load_row(x, fx.Int64(position) * x_stride + c)
                acc = convolve(window, value, weights, base)
                if valid:
                    store_row(y, fx.Int64(start + p) * y_stride + c, acc)
                window = window[1:] + [value]

            # final[k] = xs[length + k]: a token when length + k >= W - 1, else the
            # state column length + k (a sequence shorter than W - 1 shifts it).
            if ((flags & FLAG_FINAL_STATE) != 0) & row_ok:
                rolled = []
                for k in range_constexpr(W - 1):
                    source = length + k
                    from_x = source >= W - 1
                    position = from_x.select(start + source - (W - 1), fx.Int32(0))
                    value = load_row(x, fx.Int64(position) * x_stride + c)
                    for h in range_constexpr(W - 1):
                        value = (source == h).select(history[h], value)
                    rolled.append(value)
                store_columns(
                    final,
                    row64 * final_stride_n + c * final_stride_c,
                    final_stride_c,
                    final_stride_k,
                    rolled,
                )

        def token_tile(tile, c, weights, base):
            first = tile * TILE
            # Bit u of `heads` marks tile token u - (W - 2) as one of the first W - 1
            # of its sequence: a start s marks tokens s .. s + W - 2.
            lo = lower_bound(first - (W - 2))
            hi = lower_bound(first + TILE)
            hi = (hi < seqs).select(hi, seqs)
            heads = fx.Int64(0)
            for m, state in range(lo, hi, 1, init=[heads]):
                shift = fx.Int64(seq_offset(fx.Int32(m)) - first + (W - 2))
                heads = yield [state[0] | (fx.Int64(head_bits) << shift)]

            # Rows before the first token only feed skipped (head) outputs.
            window = []
            for j in range_constexpr(W - 1):
                position = first - (W - 1) + j
                position = (position < 0).select(fx.Int32(0), position)
                window.append(load_row(x, fx.Int64(position) * x_stride + c))
            last = tokens - 1
            prefetched = []
            for t in range_constexpr(TILE):
                if const_expr(t % PREFETCH == 0):
                    prefetched = []
                    for q in range_constexpr(min(PREFETCH, TILE - t)):
                        position = first + (t + q)
                        position = (position < tokens).select(position, last)
                        prefetched.append(
                            load_row(x, fx.Int64(position) * x_stride + c)
                        )
                value = prefetched[t % PREFETCH]
                acc = convolve(window, value, weights, base)
                position = first + t
                body = ((heads >> (t + W - 2)) & 1) == 0
                if (position < tokens) & body:
                    store_row(y, fx.Int64(position) * y_stride + c, acc)
                window = window[1:] + [value]

        c32 = (
            fx.Int32(fx.block_idx.y) * block_threads + fx.Int32(fx.thread_idx.x)
        ) * lanes
        if c32 < channels:
            c = fx.Int64(c32)
            weights = load_columns(
                w, c * weight_stride_c, weight_stride_c, weight_stride_k, W
            )
            base = fx.Vector.filled(lanes, 0.0, fx.Float32)
            if (flags & FLAG_BIAS) != 0:
                base = load_columns(bias, c * bias_stride, bias_stride, fx.Int64(1), 1)[
                    0
                ]
            role = fx.Int32(fx.block_idx.x)
            if role < seqs:
                sequence_block(role, c, weights, base)
            else:
                token_tile(role - seqs, c, weights, base)

    @flyc.jit
    def launch(
        x_ptr: fx.Int64,
        weight_ptr: fx.Int64,
        bias_ptr: fx.Int64,
        initial_ptr: fx.Int64,
        final_ptr: fx.Int64,
        cu_seqlens_ptr: fx.Int64,
        state_indices_ptr: fx.Int64,
        y_ptr: fx.Int64,
        x_stride: fx.Int64,
        y_stride: fx.Int64,
        weight_stride_c: fx.Int64,
        weight_stride_k: fx.Int64,
        bias_stride: fx.Int64,
        initial_stride_n: fx.Int64,
        initial_stride_c: fx.Int64,
        initial_stride_k: fx.Int64,
        final_stride_n: fx.Int64,
        final_stride_c: fx.Int64,
        final_stride_k: fx.Int64,
        tokens: fx.Int32,
        seqs: fx.Int32,
        channels: fx.Int32,
        state_rows: fx.Int32,
        cu_seqlens_step: fx.Int32,
        state_indices_step: fx.Int32,
        flags: fx.Int32,
        grid_x: fx.Int32,
        grid_y: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        causal_conv1d_kernel(
            x_ptr,
            weight_ptr,
            bias_ptr,
            initial_ptr,
            final_ptr,
            cu_seqlens_ptr,
            state_indices_ptr,
            y_ptr,
            x_stride,
            y_stride,
            weight_stride_c,
            weight_stride_k,
            bias_stride,
            initial_stride_n,
            initial_stride_c,
            initial_stride_k,
            final_stride_n,
            final_stride_c,
            final_stride_k,
            tokens,
            seqs,
            channels,
            state_rows,
            cu_seqlens_step,
            state_indices_step,
            flags,
        ).launch(grid=(grid_x, grid_y, 1), block=(block_threads, 1, 1), stream=stream)

    return launch
