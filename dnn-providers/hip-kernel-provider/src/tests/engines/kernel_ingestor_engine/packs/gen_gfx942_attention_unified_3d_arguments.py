#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Print Gfx942AttentionUnified3dArgumentsTable.inc.

Every row is one strided decode case (K and V layouts, batch, Hq, Hkv, scale, and for a
bias kernel the attn_mask's strides) and
the segment kernel's full ordered argument list rocKE would pass for it:

* names and types, in order: ``kernels.common.attention_unified
  ._tiled_3d_signatures("gfx942", dtype, strided_kv=True)`` (the segment half, with
  the spec class's runtime fields num_query_heads, num_kv_heads, sliding_window);
* the runtime head counts and sliding_window (0):
  ``_runtime_kernarg_values(problem)``;
* for a bias row (``include_additive_bias=True``), the attn_mask's three strides:
  ``_attention_additive_bias.additive_bias_launch_values``;
* the 8 strided K/V values: ``kernels.common.attention_kv_cache
  .StridedKvCacheLayout(KvTensorLayout(...), KvTensorLayout(...)).arguments()``
  (constructing the layouts also runs rocKE's admission checks);
* the other values: what the strided entry of ``run_unified_attention_torch``
  hands ``_run_3d_tiled_on_stream`` for a no-softcap, no-fp8 decode
  (``block_table=None``, ``sinks=None``, ``bt_stride=0``, ``softcap=0``,
  ``k_scale = v_scale = 1``, ``num_seqs = B``), with each pointer named by the
  buffer ``seg_vals`` binds it to (``null`` for an absent tensor).

TestGfx942AttentionUnified3dArguments.cpp asserts gfx942StridedKvArguments() and
gfx942Unified3dSegmentArguments() reproduce every row value-by-value, in order.
Before printing, the script also asserts rocKE's reduce signature is the 6
arguments the engine's reduceSignature() lists (no LSE, num_query_heads i32).

The build generates the table into the build tree through
gen_gfx942_attention_unified_3d_table.py (kernel_ingestor_engine/CMakeLists.txt);
it is never committed. To print it by hand, from the provider root:

    cd rocke/library && PYTHONPATH=.:../platform/python python3 \
        ../../src/tests/engines/kernel_ingestor_engine/packs/gen_gfx942_attention_unified_3d_arguments.py
"""

import struct
from types import SimpleNamespace

from kernels.common._attention_additive_bias import additive_bias_launch_values
from kernels.common.attention_kv_cache import KvTensorLayout, StridedKvCacheLayout
from kernels.common.attention_unified import (
    _runtime_kernarg_values,
    _tiled_3d_signatures,
)

# The buffer each pointer argument takes in _run_3d_tiled_on_stream's seg_vals
# for a strided decode call.
POINTER_BUFFERS = {
    "segm_output_ptr": "segm_output",
    "segm_max_ptr": "segm_max",
    "segm_expsum_ptr": "segm_expsum",
    "query_ptr": "query",
    "key_cache_ptr": "key",
    "value_cache_ptr": "value",
    "sink_ptr": "null",
    "block_tables_ptr": "null",
    "seq_lens_ptr": "seqused_k",
    "alibi_slopes_ptr": "null",
    "qq_bias_ptr": "null",
    "query_start_len_ptr": "cu_seqlens_q",
    "additive_bias_ptr": "additive_bias",
}


def bhsd(b, h, s, d, row=None):
    """Logical [B,H,S,D] shape and element strides of BHSD storage (row = padded D)."""
    row = row or d
    return (b, h, s, d), (h * s * row, s * row, row, 1)


def bshd(b, h, s, d, row=None):
    """BSHD storage viewed as logical [B,H,S,D]."""
    row = row or d
    return (b, h, s, d), (s * h * row, row, h * row, 1)


def padded_capacity(b, h, s, d, cap):
    """BHSD storage of capacity ``cap`` tokens, Skv = s of them used."""
    return (b, h, s, d), (h * cap * d, cap * d, d, 1)


# (label, batch, Hq, Hkv, scale, K layout, V layout). Scales are exact in f32. Hkv is the
# layouts' H; Hq spans every supported GQA ratio (1, 2, 4, 8, 16), since the kernel reads
# both head counts at runtime.
CASES = (
    ("bhsd_b1_h1_s1_d64", 1, 1, 1, 0.125, bhsd(1, 1, 1, 64), bhsd(1, 1, 1, 64)),
    (
        "bhsd_b4_h8_s4096_d128",
        4,
        32,
        8,
        0.08838834764831845,
        bhsd(4, 8, 4096, 128),
        bhsd(4, 8, 4096, 128),
    ),
    ("bhsd_b2_h8_s17_d256", 2, 16, 8, 0.0625, bhsd(2, 8, 17, 256), bhsd(2, 8, 17, 256)),
    (
        "bshd_b4_h8_s4096_d128",
        4,
        64,
        8,
        1.0,
        bshd(4, 8, 4096, 128),
        bshd(4, 8, 4096, 128),
    ),
    (
        "bshd_b16_h1_s513_d64",
        16,
        16,
        1,
        0.125,
        bshd(16, 1, 513, 64),
        bshd(16, 1, 513, 64),
    ),
    (
        "bshd_b1_h8_s8192_d256",
        1,
        8,
        8,
        0.0625,
        bshd(1, 8, 8192, 256),
        bshd(1, 8, 8192, 256),
    ),
    (
        "padrow_bhsd_b2_h8_s100_d128",
        2,
        32,
        8,
        0.5,
        bhsd(2, 8, 100, 128, row=136),
        bhsd(2, 8, 100, 128, row=136),
    ),
    (
        "padrow_bshd_b3_h4_s33_d64",
        3,
        8,
        4,
        0.125,
        bshd(3, 4, 33, 64, row=72),
        bshd(3, 4, 33, 64, row=72),
    ),
    (
        "padcap_bhsd_b2_h8_s1000_d128",
        2,
        128,
        8,
        0.25,
        padded_capacity(2, 8, 1000, 128, 1024),
        padded_capacity(2, 8, 1000, 128, 1024),
    ),
    (
        "mixed_kbhsd_vbshd_b4_h8_s2048_d128",
        4,
        32,
        8,
        0.08838834764831845,
        bhsd(4, 8, 2048, 128),
        bshd(4, 8, 2048, 128),
    ),
    (
        "mixed_kbshd_vpadrow_b2_h1_s300_d64",
        2,
        4,
        1,
        0.125,
        bshd(2, 1, 300, 64),
        bhsd(2, 1, 300, 64, row=80),
    ),
    (
        "mixed_kpadcap_vbshd_b1_h8_s7_d256",
        1,
        16,
        8,
        0.0625,
        padded_capacity(1, 8, 7, 256, 64),
        bshd(1, 8, 7, 256),
    ),
)


# (label, batch, Hq, Hkv, scale, K layout, V layout, attn_mask (batch, head, sq) element
# strides): the bias kernel's rows. Sq is 1, so the sq stride is 0 (a broadcast axis).
BIAS_CASES = (
    (
        "bias_full_bhsd_b4_h8_s4096_d128",
        4,
        32,
        8,
        0.08838834764831845,
        bhsd(4, 8, 4096, 128),
        bhsd(4, 8, 4096, 128),
        (32 * 4096, 4096, 0),
    ),
    (
        "bias_per_head_bshd_b2_h1_s300_d64",
        2,
        4,
        1,
        0.125,
        bshd(2, 1, 300, 64),
        bshd(2, 1, 300, 64),
        (0, 304, 0),
    ),
    (
        "bias_key_row_padcap_b1_h8_s7_d256",
        1,
        16,
        8,
        0.0625,
        padded_capacity(1, 8, 7, 256, 64),
        bshd(1, 8, 7, 256),
        (0, 0, 0),
    ),
)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def c_float(x):
    return f"{f32(x).hex()}F"


def c_array(values):
    return "{" + ", ".join(str(v) for v in values) + "}"


def _signatures(bias=False):
    """rocKE's gfx942 strided segment and reduce signatures (runtime fields included)."""
    return _tiled_3d_signatures(
        "gfx942", "fp16", strided_kv=True, include_additive_bias=bias
    )


# hipDNN's reduce launch: no LSE, num_query_heads read at runtime.
REDUCE_ARGUMENTS = (
    ("output_ptr", "ptr"),
    ("segm_output_ptr", "ptr"),
    ("segm_max_ptr", "ptr"),
    ("segm_expsum_ptr", "ptr"),
    ("seq_lens_ptr", "ptr"),
    ("num_query_heads", "i32"),
)


def check_reduce():
    """The reduce launch Gfx942AttentionUnified3dNative.cpp passes is rocKE's."""
    _, reduce = _signatures()
    got = tuple(
        (p["name"], "ptr" if p["type"].startswith("ptr<") else p["type"])
        for p in reduce
    )
    assert got == REDUCE_ARGUMENTS, f"rocKE reduce signature changed: {got}"


def arguments(batch, hq, hkv, scale, k_layout, v_layout, bias=None):
    assert k_layout[0][1] == hkv and v_layout[0][1] == hkv and hq % hkv == 0
    kv = StridedKvCacheLayout(
        KvTensorLayout(*k_layout), KvTensorLayout(*v_layout)
    ).arguments()
    problem = SimpleNamespace(num_query_heads=hq, num_kv_heads=hkv, sliding_window=0)
    scalars = {
        "scale": f32(scale),
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": batch,
        "block_table_stride": 0,
        "qq_bias_stride_0": 0,
        **_runtime_kernarg_values(problem),
        **kv,
    }
    if bias is not None:
        scalars.update(additive_bias_launch_values(None, bias))
        del scalars["additive_bias_ptr"]
    segment, _ = _signatures(bias is not None)
    out = []
    for param in segment:
        name, ty = param["name"], param["type"]
        if ty.startswith("ptr<"):
            out.append(f'{{"{name}", "ptr", "{POINTER_BUFFERS[name]}", 0, 0.0F}}')
        elif ty == "f32":
            out.append(f'{{"{name}", "f32", "", 0, {c_float(scalars[name])}}}')
        else:
            out.append(f'{{"{name}", "{ty}", "", {int(scalars[name])}, 0.0F}}')
    assert set(scalars) <= {p["name"] for p in segment}
    return out


def main():
    check_reduce()
    rows = [(*case, None) for case in CASES] + list(BIAS_CASES)
    for label, batch, hq, hkv, scale, k_layout, v_layout, bias in rows:
        args = arguments(batch, hq, hkv, scale, k_layout, v_layout, bias)
        has_bias = "true" if bias is not None else "false"
        print(
            f'{{"{label}", {batch}, {hq}, {hkv}, {c_float(scale)},\n'
            f" {c_array(k_layout[0])}, {c_array(k_layout[1])},\n"
            f" {c_array(v_layout[0])}, {c_array(v_layout[1])},\n"
            f" {has_bias}, {c_array(bias or (0, 0, 0))}, {len(args)},\n"
            f" {{{{\n  " + ",\n  ".join(args) + "}}},"
        )


if __name__ == "__main__":
    main()
