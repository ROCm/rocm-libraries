#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_fmha_decode_emit.py -- Python reference emitter for
# the RDNA (gfx11 / gfx12) split-KV GQA-packed WMMA decode and its merge
# kernel (kernels/gfx1151/wmma_fmha_decode.py). The config table must match
# gfx1151_wmma_fmha_decode_emit.c and prove_parity_binding.py
# cfgs_gfx1151_wmma_fmha_decode index for index:
#
#   0..35    decode: gfx11-generic, gfx12-generic x fp16, bf16 x buckets
#            64, 128, 256 x bias none, q, f32 (the per-family auto knobs)
#   36..43   merge: the same archs x dtypes x layout dense, packed (ragged)
#   44..83   FEATURES x (gfx11-generic, gfx12-generic): each feature alone
#            (softcap with both tanh forms, sinks, ALiBi, QQ-bias, FP8 K/V at
#            each bucket, ragged, paged per-tile and per-row lookups) and model
#            combos with the additive bias
#   84..121  KNOBS on bf16 d128 (or the stated bucket) x archs: num_waves
#            1/2/8, waves_per_eu, scheduler default / max-ilp, dv_split, Q in
#            LDS / registers, V store early / late, prefetch on / off,
#            lockstep on / off, FP8 conversion forms, softcap tanh forms
#   122..127 processor replays (gfx1151, gfx1201) of generic configs
#
# The batch, lengths, head counts (Hq / Hkv / Hv), head widths, strides,
# window bounds, split count and every feature operand are kernel arguments,
# so each config covers GQA packing, Hv != Hkv and every split mode.
from kernels.gfx1151.wmma_fmha_decode import (
    WmmaFmhaDecodeSpec,
    build_wmma_fmha_decode,
    build_wmma_fmha_decode_merge,
)
from _emit_common import run_emit

ARCHS = ("gfx11-generic", "gfx12-generic")
DTYPES = ("fp16", "bf16")
BUCKETS = (64, 128, 256)
BIASES = ("", "q", "f32")
# (dtype, head_size, bias, flags, layout, page_block_size); flags: c softcap,
# s sinks, a ALiBi, q QQ-bias, 8 FP8 (OCP E4M3FN) K/V.
FEATURES = (
    ("fp16", 128, "", "c", "dense", 0),
    ("bf16", 256, "", "c", "dense", 0),
    ("bf16", 64, "", "s", "dense", 0),
    ("fp16", 128, "", "a", "dense", 0),
    ("bf16", 64, "", "q", "dense", 0),
    ("bf16", 64, "", "8", "dense", 0),
    ("fp16", 128, "", "8", "dense", 0),
    ("bf16", 256, "", "8", "dense", 0),
    ("bf16", 128, "", "", "ragged", 0),
    ("fp16", 64, "", "", "paged", 16),
    ("bf16", 128, "", "", "paged", 8),
    ("bf16", 256, "", "", "paged", 64),
    ("bf16", 64, "", "s8", "paged", 16),
    ("fp16", 128, "", "ca", "dense", 0),
    ("bf16", 128, "", "aq", "ragged", 0),
    ("bf16", 256, "", "cs8", "ragged", 0),
    ("bf16", 64, "q", "csaq8", "paged", 16),
    ("fp16", 128, "f32", "s", "ragged", 0),
    ("bf16", 64, "f32", "ac8", "paged", 32),
    ("fp16", 256, "q", "q", "paged", 4),
)
KNOBS = (
    dict(num_waves=2),
    dict(num_waves=8, head_size=64),
    dict(num_waves=1),
    dict(waves_per_eu=1),
    dict(scheduler_strategy="default"),
    dict(scheduler_strategy="max-ilp", head_size=256),
    dict(dv_split=2),
    dict(q_mode="lds"),
    dict(q_mode="reg", head_size=256),
    dict(v_store_early=True),
    dict(v_store_early=False),
    dict(prefetch=True),
    dict(prefetch=False, head_size=64),
    dict(lockstep=True),
    dict(lockstep=False),
    dict(fp8_cvt=True, kv_dtype="fp8e4m3"),
    dict(fp8_cvt=False, kv_dtype="fp8e4m3", head_size=64),
    dict(tanh_fast=True, softcap=True, head_size=256),
    dict(tanh_fast=False, softcap=True),
)


def _feature_fields(row):
    dtype, hs, bias, flags, layout, ps = row
    return dict(
        head_size=hs,
        dtype=dtype,
        bias=bias,
        softcap="c" in flags,
        sinks="s" in flags,
        alibi="a" in flags,
        qq_bias="q" in flags,
        kv_dtype="fp8e4m3" if "8" in flags else "",
        layout=layout,
        page_block_size=ps,
    )


def _configs():
    out = []
    for arch in ARCHS:
        for dtype in DTYPES:
            for hs in BUCKETS:
                for bias in BIASES:
                    out.append(
                        (arch, "decode", dict(head_size=hs, dtype=dtype, bias=bias))
                    )
    for arch in ARCHS:
        for dtype in DTYPES:
            for layout in ("dense", "ragged"):
                out.append(
                    (arch, "merge", dict(head_size=64, dtype=dtype, layout=layout))
                )
    for arch in ARCHS:
        for row in FEATURES:
            out.append((arch, "decode", _feature_fields(row)))
    for arch in ARCHS:
        for knob in KNOBS:
            fields = dict(head_size=128, dtype="bf16")
            fields.update(knob)
            out.append((arch, "decode", fields))
    out.append(("gfx1151", "decode", dict(head_size=128, dtype="bf16")))
    out.append(("gfx1151", "merge", dict(head_size=64, dtype="fp16", layout="ragged")))
    out.append(("gfx1151", "decode", _feature_fields(FEATURES[12])))
    out.append(("gfx1201", "decode", dict(head_size=64, dtype="fp16", bias="q")))
    out.append(("gfx1201", "decode", _feature_fields(FEATURES[15])))
    out.append(("gfx1201", "merge", dict(head_size=64, dtype="bf16", layout="dense")))
    return out


CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, kind, fields = CONFIGS[idx]
    return (kind, WmmaFmhaDecodeSpec(**fields)), arch


def _build(selected, *, arch):
    kind, spec = selected
    if kind == "merge":
        return build_wmma_fmha_decode_merge(spec, arch=arch)
    return build_wmma_fmha_decode(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage=f"usage: gfx1151_wmma_fmha_decode_emit.py <config_index 0..{len(CONFIGS) - 1}>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
