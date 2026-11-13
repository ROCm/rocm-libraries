# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime-layout pieces shared by the gfx942 and gfx950 tiled 2D attention builders.

``runtime_layout`` (hipDNN's ``AttentionPagedUnified`` engine) appends kernel
arguments after the fixed ones and the runtime shape kernargs
(``RUNTIME_KERNARG_FIELDS``: num_query_heads, num_kv_heads, sliding_window), in
the order of ``RUNTIME_LAYOUT_EXTRA_ARGS``. Both architectures take the same
list, so the ABI and the sink-logit read live here once; a divergence between the
two builders would be a silent ABI break, and the C++ twin
(``instance_attention_tiled_2d_rtl_internal.h``) mirrors this list.
"""

from __future__ import annotations

from rocke.core.ir import F32, I16, I32, PtrType

# (name, type) in ABI order. ``block_tables_v_ptr`` is V's block table; it
# shares ``block_table_stride`` with ``block_tables_ptr`` (K's table) and may be
# the same pointer. ``q_uniform_len`` is L > 0 for a dense Q (L rows per
# sequence) and 0 for a ragged one, whose offsets must be non-decreasing with
# off[b+1] - off[b] >= seq_lens_q[b] and off[B] the end of the last block.
# Strides are in elements; ``kv_block_span_bytes`` is the per-block
# buffer-resource size. ``sink_is_f32`` nonzero: ``sink_ptr`` holds Hq fp32
# logits, else Hq logits in the Q dtype (ignored without sinks).
RUNTIME_LAYOUT_EXTRA_ARGS = (
    ("block_tables_v_ptr", "ptr<i32>"),
    ("seq_lens_q_ptr", "ptr<i32>"),
    ("has_seq_lens_q", "i32"),
    ("q_uniform_len", "i32"),
    ("q_seq_stride", "i32"),
    ("q_tok_stride", "i32"),
    ("q_head_stride", "i32"),
    ("o_seq_stride", "i32"),
    ("o_tok_stride", "i32"),
    ("o_head_stride", "i32"),
    ("kv_block_stride", "i32"),
    ("kv_tok_stride", "i32"),
    ("kv_head_stride", "i32"),
    ("kv_block_span_bytes", "i32"),
    ("q_block_grid", "i32"),
    ("sink_is_f32", "i32"),
)


def declare_runtime_layout_params(b):
    """Declare the appended kernel arguments on ``b``; ``{name: Value}`` in ABI order."""
    params = {}
    for name, ty in RUNTIME_LAYOUT_EXTRA_ARGS:
        if ty == "ptr<i32>":
            params[name] = b.param(name, PtrType(I32, "global"), readonly=True, align=4)
        else:
            params[name] = b.param(name, I32)
    return params


def sink_logit(b, sinks, dtype, qh, qh_in, sink_is_f32=None):
    """Query head ``qh``'s sink logit in f32.

    ``sink_is_f32`` is the runtime-layout kernarg (``None`` outside that mode,
    where ``sinks`` holds Hq logits in the Q dtype). With it, nonzero means
    ``sinks`` holds Hq fp32 logits, else Hq logits in the Q dtype. The fp32 form
    is read as two 16-bit halves and both forms mask on their own flag, so neither
    reads past the other's buffer (2*Hq halves vs Hq).
    """
    if sink_is_f32 is None:
        return b.cast_to_f32(b.global_load(sinks, qh, dtype, align=2))
    is_f32 = b.cmp_ne(sink_is_f32, b.const_i32(0))
    is_q = b.cmp_eq(sink_is_f32, b.const_i32(0))
    f32_in = b.land(qh_in, is_f32)
    qh2 = b.mul(qh, b.const_i32(2))
    lo_idx = b.select(f32_in, qh2, b.const_i32(0))
    lo = b.global_load(sinks, lo_idx, I16, align=2)
    hi_idx = b.select(f32_in, b.add(qh2, b.const_i32(1)), b.const_i32(0))
    hi = b.global_load(sinks, hi_idx, I16, align=2)
    lo32 = b.zext(lo, I32)
    hi32 = b.zext(hi, I32)
    word = b.add(lo32, b.shl(hi32, b.const_i32(16)))
    sink_w = b.bitcast(word, F32)
    q_idx = b.select(b.land(qh_in, is_q), qh, b.const_i32(0))
    sink_h = b.global_load(sinks, q_idx, dtype, align=2)
    return b.select(is_f32, sink_w, b.cast_to_f32(sink_h))
