#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Regenerate Gfx942AttentionUnified3dArgumentsTable.inc.

Every row is one strided decode case (K and V layouts, batch, scale) and the
segment kernel's full ordered argument list rocKE would pass for it:

* names and types, in order: ``kernels.common.attention_unified._3d_signature
  (dtype, strided_kv=True)``;
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

Regenerate (from the provider root, dnn-providers/hip-kernel-provider):

    cd rocke/library && PYTHONPATH=.:../platform/python python3 \
        ../../src/tests/engines/kernel_ingestor_engine/packs/gen_gfx942_attention_unified_3d_arguments.py \
        > ../../src/tests/engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dArgumentsTable.inc
"""

import struct

from kernels.common.attention_kv_cache import KvTensorLayout, StridedKvCacheLayout
from kernels.common.attention_unified import _3d_signature

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


# (label, batch, scale, K layout, V layout). Scales are exact in f32.
CASES = (
    ("bhsd_b1_h1_s1_d64", 1, 0.125, bhsd(1, 1, 1, 64), bhsd(1, 1, 1, 64)),
    ("bhsd_b4_h8_s4096_d128", 4, 0.08838834764831845, bhsd(4, 8, 4096, 128), bhsd(4, 8, 4096, 128)),
    ("bhsd_b2_h8_s17_d256", 2, 0.0625, bhsd(2, 8, 17, 256), bhsd(2, 8, 17, 256)),
    ("bshd_b4_h8_s4096_d128", 4, 1.0, bshd(4, 8, 4096, 128), bshd(4, 8, 4096, 128)),
    ("bshd_b16_h1_s513_d64", 16, 0.125, bshd(16, 1, 513, 64), bshd(16, 1, 513, 64)),
    ("bshd_b1_h8_s8192_d256", 1, 0.0625, bshd(1, 8, 8192, 256), bshd(1, 8, 8192, 256)),
    ("padrow_bhsd_b2_h8_s100_d128", 2, 0.5, bhsd(2, 8, 100, 128, row=136), bhsd(2, 8, 100, 128, row=136)),
    ("padrow_bshd_b3_h4_s33_d64", 3, 0.125, bshd(3, 4, 33, 64, row=72), bshd(3, 4, 33, 64, row=72)),
    ("padcap_bhsd_b2_h8_s1000_d128", 2, 0.25, padded_capacity(2, 8, 1000, 128, 1024), padded_capacity(2, 8, 1000, 128, 1024)),
    ("mixed_kbhsd_vbshd_b4_h8_s2048_d128", 4, 0.08838834764831845, bhsd(4, 8, 2048, 128), bshd(4, 8, 2048, 128)),
    ("mixed_kbshd_vpadrow_b2_h1_s300_d64", 2, 0.125, bshd(2, 1, 300, 64), bhsd(2, 1, 300, 64, row=80)),
    ("mixed_kpadcap_vbshd_b1_h8_s7_d256", 1, 0.0625, padded_capacity(1, 8, 7, 256, 64), bshd(1, 8, 7, 256)),
)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def c_float(x):
    return f"{f32(x).hex()}F"


def c_array(values):
    return "{" + ", ".join(str(v) for v in values) + "}"


def arguments(batch, scale, k_layout, v_layout):
    kv = StridedKvCacheLayout(KvTensorLayout(*k_layout), KvTensorLayout(*v_layout)).arguments()
    scalars = {
        "scale": f32(scale),
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": batch,
        "block_table_stride": 0,
        "qq_bias_stride_0": 0,
        **kv,
    }
    out = []
    for param in _3d_signature("fp16", strided_kv=True):
        name, ty = param["name"], param["type"]
        if ty.startswith("ptr<"):
            out.append(f'{{"{name}", "ptr", "{POINTER_BUFFERS[name]}", 0, 0.0F}}')
        elif ty == "f32":
            out.append(f'{{"{name}", "f32", "", 0, {c_float(scalars[name])}}}')
        else:
            out.append(f'{{"{name}", "{ty}", "", {int(scalars[name])}, 0.0F}}')
    assert set(scalars) <= {p["name"] for p in _3d_signature("fp16", strided_kv=True)}
    return out


def main():
    for label, batch, scale, k_layout, v_layout in CASES:
        args = ",\n  ".join(arguments(batch, scale, k_layout, v_layout))
        print(
            f'{{"{label}", {batch}, {c_float(scale)},\n'
            f" {c_array(k_layout[0])}, {c_array(k_layout[1])},\n"
            f" {c_array(v_layout[0])}, {c_array(v_layout[1])},\n"
            f" {{{{\n  {args}}}}}}},"
        )


if __name__ == "__main__":
    main()
