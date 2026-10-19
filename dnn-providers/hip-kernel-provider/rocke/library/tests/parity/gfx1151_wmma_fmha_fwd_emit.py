#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_fmha_fwd_emit.py -- Python reference emitter for the
# gfx1151 (RDNA3.5 / Strix Halo) WMMA FMHA forward instance parity harness.
# Selects one of 157 sampled configurations by argv[1] (0..156), builds it
# via build_wmma_fmha_fwd(arch=<cfg arch>) and prints
# lower_kernel_to_llvm(kernel, arch=<cfg arch>) to stdout so it can be
# byte-compared with the C emitter gfx1151_wmma_fmha_fwd_emit.c. Configs 0..135
# use gfx1151; 136..142 replay representative configs at gfx11-generic, whose
# output must equal the replayed config's (one kernel set for every gfx11).
# Configs 143..156 exercise runtime_head_dims on gfx1151, gfx11-generic and
# gfx12-generic.
from dataclasses import replace

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, build_wmma_fmha_fwd
from _emit_common import run_emit


_V_HEAD_CASES = (
    # head_size, v_head_size, value_tile_size, mask_mode, v_lds_stage
    (128, 64, 0, "none", False),
    (64, 128, 0, "causal", True),
    (192, 128, 0, "none", True),
    (128, 256, 64, "none", False),
)


_ATTN_BIAS_CASES = (
    # layout, page, head_size, mask, tails, sinks, bias_dtype,
    # extras (s=softcap a=alibi q=qq_bias), value_tile, dtype
    ("dense", 0, 64, "none", False, False, "f32", "", 0, "fp16"),
    ("dense", 0, 64, "causal", True, True, "q", "", 0, "fp16"),
    ("dense", 0, 128, "none", False, False, "q", "", 0, "fp16"),
    ("ragged", 0, 64, "causal", True, True, "f32", "", 0, "fp16"),
    ("paged", 16, 64, "causal", True, False, "q", "", 0, "fp16"),
    ("dense", 0, 64, "causal", True, False, "f32", "saq", 0, "fp16"),
    ("dense", 0, 256, "none", False, False, "f32", "", 64, "fp16"),
    ("dense", 0, 128, "causal", True, True, "q", "saq", 0, "fp16"),
    ("dense", 0, 64, "none", False, False, "q", "", 0, "bf16"),
)

# Standard dense, transposed QK (D64 and D128 bottom-right), FP8 packed KV,
# distinct V width, additive bias, and D256 output-column tiling.
_GENERIC_REPLAY = (0, 70, 85, 115, 123, 128, 133)

_RTDIM_BASE = 136 + len(_GENERIC_REPLAY)
_RTDIM_CASES = (
    # arch, dtype, head_size, v_head_size, mask, flags, layout, page, bias dtype
    # ("" = none). Flags: t=query/kv tails, l=v_lds_stage, 8=fp8 KV,
    # c=causal_tile_skip, s=softcap, k=sinks, a=alibi, q=qq_bias.
    ("gfx1151", "fp16", 64, 0, "none", "", "dense", 0, ""),
    ("gfx1151", "bf16", 128, 0, "causal", "tlc", "dense", 0, ""),
    ("gfx1151", "fp16", 256, 0, "none", "l", "dense", 0, ""),
    ("gfx1151", "fp16", 128, 64, "window", "t", "dense", 0, ""),
    ("gfx1151", "fp16", 128, 0, "window", "tl", "dense", 0, "f32"),
    ("gfx1151", "fp16", 128, 0, "causal", "t8", "paged", 16, ""),
    ("gfx1151", "bf16", 64, 0, "window", "tlskaq", "ragged", 0, "q"),
    ("gfx11-generic", "fp16", 128, 0, "causal", "tl", "dense", 0, ""),
    ("gfx11-generic", "bf16", 256, 128, "none", "", "dense", 0, "f32"),
    ("gfx12-generic", "fp16", 128, 0, "causal", "tl", "dense", 0, ""),
    ("gfx12-generic", "bf16", 128, 64, "window", "t", "dense", 0, "f32"),
    ("gfx12-generic", "fp16", 256, 0, "causal", "tl8", "paged", 32, ""),
    ("gfx12-generic", "fp16", 64, 0, "none", "", "dense", 0, ""),
    ("gfx12-generic", "bf16", 192, 0, "window", "tska", "ragged", 0, ""),
)


def _rtdim_spec(case) -> WmmaFmhaFwdSpec:
    _arch, dtype, head, v_head, mask, flags, layout, page, bias = case
    return WmmaFmhaFwdSpec(
        head_size=head,
        v_head_size=v_head,
        dtype=dtype,
        mask_mode=mask,
        query_tail="t" in flags,
        kv_tail="t" in flags,
        v_lds_stage="l" in flags,
        layout=layout,
        page_block_size=page,
        kv_dtype="fp8e4m3" if "8" in flags else "",
        use_attn_bias=bool(bias),
        bias_dtype=bias or "f32",
        use_softcap="s" in flags,
        use_sinks="k" in flags,
        use_alibi="a" in flags,
        use_qq_bias="q" in flags,
        causal_tile_skip="c" in flags,
        runtime_head_dims=True,
    )


def _spec_and_arch(idx: int):
    if 136 <= idx < _RTDIM_BASE:
        return _spec(_GENERIC_REPLAY[idx - 136]), "gfx11-generic"
    if _RTDIM_BASE <= idx < _RTDIM_BASE + len(_RTDIM_CASES):
        case = _RTDIM_CASES[idx - _RTDIM_BASE]
        return _rtdim_spec(case), case[0]
    return _spec(idx), "gfx1151"


def _spec(idx: int) -> WmmaFmhaFwdSpec:
    if 127 <= idx < 127 + len(_ATTN_BIAS_CASES):
        (
            layout,
            page,
            head,
            mask,
            tails,
            sinks,
            bias_dtype,
            extras,
            tile,
            dtype,
        ) = _ATTN_BIAS_CASES[idx - 127]
        return WmmaFmhaFwdSpec(
            head_size=head,
            dtype=dtype,
            mask_mode=mask,
            query_tail=tails,
            kv_tail=tails,
            v_lds_stage=tails,
            use_sinks=sinks,
            use_softcap="s" in extras,
            use_alibi="a" in extras,
            use_qq_bias="q" in extras,
            layout=layout,
            page_block_size=page,
            value_tile_size=tile,
            use_attn_bias=True,
            bias_dtype=bias_dtype,
        )
    if 123 <= idx < 127:
        head, v_head, tile, mask, vlds = _V_HEAD_CASES[idx - 123]
        return WmmaFmhaFwdSpec(
            head_size=head,
            mask_mode=mask,
            v_lds_stage=vlds,
            v_head_size=v_head,
            value_tile_size=tile,
        )
    if 119 <= idx < 123:
        # Standard-path causal tile skip: two head widths, with/without tails.
        variant = idx - 119
        tails = bool(variant % 2)
        return WmmaFmhaFwdSpec(
            head_size=64 if variant < 2 else 128,
            mask_mode="causal",
            query_tail=tails,
            kv_tail=tails,
            v_lds_stage=tails,
            causal_tile_skip=True,
        )
    if 99 <= idx < 115:
        variant = idx - 99
        vlds = bool(variant % 2)
        return WmmaFmhaFwdSpec(
            head_size=256,
            dtype="fp16" if variant < 8 else "bf16",
            mask_mode="causal" if vlds else "none",
            query_tail=vlds,
            kv_tail=vlds,
            v_lds_stage=vlds,
            value_tile_size=(16, 32, 64, 128)[(variant % 8) // 2],
        )
    if 115 <= idx < 119:
        variant = idx - 115
        paged = bool(variant % 2)
        return WmmaFmhaFwdSpec(
            head_size=256,
            dtype="fp16" if variant < 2 else "bf16",
            mask_mode="causal",
            query_tail=True,
            kv_tail=True,
            layout="paged" if paged else "ragged",
            page_block_size=32 if paged else 0,
            kv_dtype="fp8e4m3",
            value_tile_size=128,
            v_lds_stage=paged,
        )
    if 94 <= idx < 99:
        strategies = (
            "max-ilp",
            "max-memory-clause",
            "iterative-ilp",
            "iterative-minreg",
            "iterative-maxocc",
        )
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="window",
            layout="ragged",
            query_tail=True,
            kv_tail=True,
            v_lds_stage=True,
            scheduler_strategy=strategies[idx - 94],
        )
    if 86 <= idx < 94:
        variant = idx - 86
        return WmmaFmhaFwdSpec(
            head_size=96,
            dtype="fp16" if variant < 4 else "bf16",
            mask_mode=("none", "causal", "window", "window")[variant % 4],
            query_tail=bool(variant % 2),
            kv_tail=bool(variant % 2),
            v_lds_stage=bool(variant % 2),
        )
    if 70 <= idx < 86:
        variant = idx - 70
        return WmmaFmhaFwdSpec(
            head_size=64 if (variant % 8) < 4 else 128,
            dtype="fp16" if variant < 8 else "bf16",
            mask_mode="none" if variant % 4 < 2 else "causal",
            transposed_qk=True,
            block_n=32 if variant % 2 == 0 else 64,
            num_waves=1 if variant % 2 == 0 else 2,
        )
    if 58 <= idx < 70:
        bases = (0, 7, 14, 15, 22, 23, 24, 25, 44, 47, 51, 57)
        return replace(_spec(bases[idx - 58]), kv_dtype="fp8e4m3")
    if 42 <= idx < 58:
        bases = (0, 7, 12, 15, 16, 39, 0, 7, 12, 15, 16, 17, 26, 33, 38, 39)
        pages = (0, 0, 0, 0, 0, 0, 16, 64, 16, 32, 32, 64, 64, 32, 16, 64)
        page = pages[idx - 42]
        return replace(
            _spec(bases[idx - 42]),
            layout="paged" if page else "ragged",
            page_block_size=page,
        )
    if 26 <= idx < 42:
        feature_cases = (
            (12, dict(mask_mode="window")),
            (13, dict(mask_mode="window")),
            (2, dict(use_softcap=True)),
            (8, dict(use_softcap=True)),
            (2, dict(use_sinks=True)),
            (8, dict(use_sinks=True)),
            (12, dict(use_alibi=True)),
            (13, dict(use_alibi=True)),
            (0, dict(use_qq_bias=True)),
            (6, dict(use_qq_bias=True)),
            (18, dict(mask_mode="window", use_sinks=True)),
            (19, dict(mask_mode="window", use_sinks=True)),
            (
                24,
                dict(
                    mask_mode="window",
                    use_softcap=True,
                    use_sinks=True,
                    use_alibi=True,
                    use_qq_bias=True,
                ),
            ),
            (
                25,
                dict(
                    mask_mode="window",
                    use_softcap=True,
                    use_sinks=True,
                    use_alibi=True,
                    use_qq_bias=True,
                ),
            ),
            (2, dict(use_softcap=True, use_alibi=True)),
            (8, dict(use_softcap=True, use_alibi=True)),
        )
        base, features = feature_cases[idx - 26]
        return replace(_spec(base), **features)
    if 18 <= idx < 26:
        bases = (12, 13, 3, 9, 12, 13, 16, 17)
        return replace(
            _spec(bases[idx - 18]),
            query_tail=idx < 20 or idx >= 22,
            kv_tail=idx >= 20,
        )
    if 12 <= idx < 18:
        return _spec((2, 8, 3, 9, 5, 11)[idx - 12])
    if 6 <= idx < 12:
        return replace(_spec(idx - 6), dtype="bf16")
    if idx == 0:
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 1:
        return WmmaFmhaFwdSpec(
            head_size=128,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 2:
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="causal",
            v_lds_stage=False,
        )
    if idx == 3:
        return WmmaFmhaFwdSpec(
            head_size=256,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 4:
        return WmmaFmhaFwdSpec(
            head_size=128,
            mask_mode="causal",
            v_lds_stage=False,
        )
    if idx == 5:
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="none",
            v_lds_stage=True,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec_and_arch,
        build_wmma_fmha_fwd,
        usage="usage: gfx1151_wmma_fmha_fwd_emit.py <config_index 0..156>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
