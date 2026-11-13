#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_fmha_fwd_v2_emit.py -- Python reference emitter for
# the RDNA (gfx11 / gfx12) multi-wave WMMA attention prefill
# (kernels/gfx1151/wmma_fmha_fwd_v2.py). The config table must match
# gfx1151_wmma_fmha_fwd_v2_emit.c and prove_parity_binding.py
# cfgs_gfx1151_wmma_fmha_fwd_v2 index for index:
#
#   0..47    gfx11-generic, gfx12-generic x fp16, bf16 x buckets
#            64, 96, 128, 192v128, 256v128, 256 x mask none, causal
#   48..71   the same archs x dtypes x (d64 causal, d128 none, d256 causal)
#            x bias f32, q
#   72..81   gfx12-generic unsplit Q (wave_pair=False): fp16, bf16 x
#            192v128, 256v128 x none, causal; then bf16 256v128 causal bias q
#            and fp16 192v128 none bias f32
#   82..101  tuning knobs on bf16 d128 causal, per arch: block_n 64 (on
#            d64), block_n 16, block_m 64, waves_per_eu 0, max-ilp, eager
#            rescale, pads 0/16, cta_order hbq, forced wave pair, and
#            block_m 32 on d256
#   102..105 processor replays (gfx1151, gfx1201) of generic configs
#   106..165 FEATURES x (gfx11-generic, gfx12-generic): each feature alone
#            and in model combos (gpt-oss sinks + SWA D64, Gemma-2 softcap
#            D256, FP8 KV, paged 8/16/32/64 incl. per-row and per-tile
#            lookups, ragged, everything at once)
#   166..167 gfx12-generic unsplit wide Q with features
#   168..169 processor replays (gfx1151, gfx1201) of feature configs
#
# The head counts (Hq / Hkv / Hv), head widths, strides and window bounds
# are kernel arguments, so every config covers Hv != Hkv by construction.
from kernels.gfx1151.wmma_fmha_fwd_v2 import WmmaFmhaFwdV2Spec, build_wmma_fmha_fwd_v2
from _emit_common import run_emit

ARCHS = ("gfx11-generic", "gfx12-generic")
DTYPES = ("fp16", "bf16")
BUCKETS = ((64, 64), (96, 96), (128, 128), (192, 128), (256, 128), (256, 256))
MASKS = ("none", "causal")
BIAS_SHAPES = ((64, 64, "causal"), (128, 128, "none"), (256, 256, "causal"))
KNOBS = (
    dict(head_dim=64, head_dim_v=64, block_n=64),
    dict(block_n=16),
    dict(block_m=64),
    dict(waves_per_eu=0),
    dict(scheduler_strategy="max-ilp"),
    dict(lazy_rescale=False),
    dict(k_pad=0, v_pad=16),
    dict(cta_order="hbq"),
    dict(wave_pair=True),
    dict(head_dim=256, head_dim_v=256, block_m=32),
)
# (dtype, head_dim, head_dim_v, mask, bias, flags, layout, page_block_size,
#  block_n, wave_pair); flags: c softcap, s sinks, a ALiBi, q QQ-bias,
#  8 FP8 (OCP E4M3FN) K/V.
FEATURES = (
    ("fp16", 128, 128, "none", None, "c", "dense", 0, 32, None),
    ("bf16", 256, 256, "causal", None, "c", "dense", 0, 32, None),
    ("bf16", 64, 64, "causal", None, "s", "dense", 0, 32, None),
    ("fp16", 128, 128, "none", None, "s", "dense", 0, 32, None),
    ("bf16", 128, 128, "causal", None, "a", "dense", 0, 32, None),
    ("fp16", 128, 128, "none", None, "a", "dense", 0, 32, None),
    ("fp16", 128, 128, "causal", None, "q", "dense", 0, 32, None),
    ("bf16", 64, 64, "none", None, "q", "dense", 0, 32, None),
    ("bf16", 128, 128, "none", None, "8", "dense", 0, 32, None),
    ("fp16", 64, 64, "causal", None, "8", "dense", 0, 32, None),
    ("bf16", 256, 256, "causal", None, "8", "dense", 0, 32, None),
    ("bf16", 128, 128, "causal", None, "", "ragged", 0, 32, None),
    ("fp16", 64, 64, "none", None, "", "ragged", 0, 32, None),
    ("bf16", 128, 128, "causal", None, "", "paged", 16, 32, None),
    ("bf16", 128, 128, "causal", None, "", "paged", 32, 32, None),
    ("fp16", 64, 64, "none", None, "", "paged", 64, 32, None),
    ("bf16", 64, 64, "causal", None, "", "paged", 8, 32, None),
    ("bf16", 256, 256, "causal", None, "", "paged", 16, 32, None),
    ("bf16", 64, 64, "causal", None, "s", "paged", 16, 32, None),
    ("bf16", 128, 128, "causal", None, "8", "ragged", 0, 32, None),
    ("fp16", 128, 128, "causal", None, "8", "paged", 32, 32, None),
    ("bf16", 128, 128, "causal", None, "sa", "paged", 16, 32, None),
    ("fp16", 128, 128, "causal", None, "q", "ragged", 0, 32, None),
    ("bf16", 128, 128, "none", None, "ca", "dense", 0, 32, None),
    ("fp16", 64, 64, "causal", "f32", "s", "dense", 0, 32, None),
    ("bf16", 128, 128, "causal", "q", "csaq8", "paged", 16, 32, None),
    ("bf16", 256, 256, "causal", None, "cs", "dense", 0, 32, None),
    ("bf16", 64, 64, "causal", None, "", "paged", 32, 64, None),
    ("bf16", 64, 64, "causal", None, "s", "paged", 64, 64, None),
    ("bf16", 128, 128, "causal", None, "8", "ragged", 0, 16, None),
)
FEATURES_GFX12 = (
    ("bf16", 192, 128, "causal", None, "c8", "dense", 0, 32, False),
    ("fp16", 256, 128, "none", None, "s", "paged", 32, 32, False),
)


def _feature_fields(row):
    dtype, hd, hdv, mask, bias, flags, layout, ps, bn, pair = row
    return dict(
        dtype=dtype,
        head_dim=hd,
        head_dim_v=hdv,
        mask=mask,
        bias=bias,
        use_softcap="c" in flags,
        use_sinks="s" in flags,
        use_alibi="a" in flags,
        use_qq_bias="q" in flags,
        layout=layout,
        page_block_size=ps,
        kv_dtype="fp8e4m3" if "8" in flags else "",
        block_n=bn,
        wave_pair=pair,
    )


def _configs():
    out = []
    for arch in ARCHS:
        for dtype in DTYPES:
            for hd, hdv in BUCKETS:
                for mask in MASKS:
                    out.append(
                        (
                            arch,
                            dict(dtype=dtype, head_dim=hd, head_dim_v=hdv, mask=mask),
                        )
                    )
    for arch in ARCHS:
        for dtype in DTYPES:
            for hd, hdv, mask in BIAS_SHAPES:
                for bias in ("f32", "q"):
                    out.append(
                        (
                            arch,
                            dict(
                                dtype=dtype,
                                head_dim=hd,
                                head_dim_v=hdv,
                                mask=mask,
                                bias=bias,
                            ),
                        )
                    )
    for dtype in DTYPES:
        for hd in (192, 256):
            for mask in MASKS:
                out.append(
                    (
                        "gfx12-generic",
                        dict(
                            dtype=dtype,
                            head_dim=hd,
                            head_dim_v=128,
                            mask=mask,
                            wave_pair=False,
                        ),
                    )
                )
    out.append(
        (
            "gfx12-generic",
            dict(
                dtype="bf16",
                head_dim=256,
                head_dim_v=128,
                mask="causal",
                bias="q",
                wave_pair=False,
            ),
        )
    )
    out.append(
        (
            "gfx12-generic",
            dict(
                dtype="fp16",
                head_dim=192,
                head_dim_v=128,
                mask="none",
                bias="f32",
                wave_pair=False,
            ),
        )
    )
    for arch in ARCHS:
        for knob in KNOBS:
            base = dict(dtype="bf16", head_dim=128, head_dim_v=128, mask="causal")
            base.update(knob)
            out.append((arch, base))
    out.append(("gfx1151", dict(dtype="fp16", head_dim=64, head_dim_v=64, mask="none")))
    out.append(
        (
            "gfx1151",
            dict(dtype="bf16", head_dim=128, head_dim_v=128, mask="causal", bias="q"),
        )
    )
    out.append(("gfx1201", dict(dtype="fp16", head_dim=64, head_dim_v=64, mask="none")))
    out.append(
        (
            "gfx1201",
            dict(
                dtype="bf16",
                head_dim=256,
                head_dim_v=128,
                mask="causal",
                wave_pair=False,
            ),
        )
    )
    for arch in ARCHS:
        for row in FEATURES:
            out.append((arch, _feature_fields(row)))
    for row in FEATURES_GFX12:
        out.append(("gfx12-generic", _feature_fields(row)))
    out.append(("gfx1151", _feature_fields(FEATURES[2])))
    out.append(("gfx1201", _feature_fields(FEATURES[8])))
    return out


CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, fields = CONFIGS[idx]
    return WmmaFmhaFwdV2Spec(**fields), arch


def main() -> int:
    return run_emit(
        _spec,
        build_wmma_fmha_fwd_v2,
        usage=f"usage: gfx1151_wmma_fmha_fwd_v2_emit.py <config_index 0..{len(CONFIGS) - 1}>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
