# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.dsv41_mono.wide_gemm`` / ``router_topk`` (W13 WP6 wide path). CPU only: knob checks, names, the
host scale packing, the emitted LLVM text and that every mutant changes it.

The GPU proofs (numerics vs a float64 MXFP8 reference and vs ATOM's GEMM, quantization parity with ATOM's
``quantize_fp8``, every mutant caught, timing vs ATOM's ``native_quant_linear`` / routing in the same process) run on
gfx950: ``hackathon-prep/dsv41/wp6/wp6_bench.py`` and ``wp6_rbench.py``. Results: ``experiments/dsv41-mono/wps/WP6.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
from kernels.gfx950.dsv41_mono import router_topk as RT
from kernels.gfx950.dsv41_mono import wide_gemm as W
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _ir(k):
    return lower_kernel_to_llvm(W.build_wide_kernel(k), arch="gfx950")


def test_knob_checks():
    bad = [
        dict(k=1000),  # not a multiple of 128
        dict(k=5120, splits=3),  # 40 chunks
        dict(waves=3),
        dict(xin="fp16"),
        dict(order="x"),
        dict(fold="l2"),  # needs order=xcd
        dict(xsrc="glb"),  # needs xin=fp8
        dict(pipe=1, bra=2),
        dict(mutant="fold"),  # needs splits > 1
        dict(k=5120, tiles=24, splits=1),  # LDS
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            W.WideKnobs(**kw).check()
    W.WideKnobs(k=5120, tiles=6, splits=8, order="xcd", fold="l2", mutant="fold").check()
    W.WideKnobs(k=5120, tiles=6, xin="fp8", xsrc="glb", waves=1, mt=4, splits=5).check()


def test_names_list_deviations():
    assert W.kernel_name(W.WideKnobs()) == "rk41w_k5120_t6_bf16"
    k = W.WideKnobs(k=1280, tiles=3, splits=2, order="xcd", fold="l2", waves=4)
    assert W.kernel_name(k) == "rk41w_k1280_t3_bf16_waves4_splits2_orderxcd_foldl2"
    assert W.quant_name(W.WideKnobs(k=1152, tiles=6, splits=3, xin="fp8", xsrc="glb")) == "rk41q_k1152_t6_s3"


def test_grid():
    assert W.grid_of(W.WideKnobs(k=5120, splits=8), 1792) == 14 * 8
    assert W.grid_of(W.WideKnobs(k=5120, splits=8, order="xcd", fold="l2"), 1792) == 16 * 8  # 14 row groups -> 16
    assert W.grid_of(W.WideKnobs(k=1280, waves=4), 16384) == 256


def test_pack_wscale_bytes():
    rng = np.random.default_rng(0)
    k = W.WideKnobs(k=1152, splits=3)  # kc 3: one partial group per split
    ws = rng.integers(100, 140, size=(4, 36), dtype=np.uint8)
    p = W.pack_wscale(ws, k).view(np.uint32)
    assert p.shape == (4, 3, 1, 4)
    for r in range(4):
        for s in range(3):
            for g in range(4):
                d = int(p[r, s, 0, g])
                for j in range(4):
                    want = ws[r, 4 * (s * 3 + j) + g] if j < 3 else 127
                    assert (d >> (8 * j)) & 255 == want


def test_glb_quant_kernel_lowers():
    k = W.WideKnobs(k=1280, tiles=6, splits=2, xin="fp8", xsrc="glb")
    t = lower_kernel_to_llvm(W.build_quant_kernel(k), arch="gfx950")
    assert "cvt.scalef32.pk.fp8.f32" in t
    t2 = lower_kernel_to_llvm(W.build_quant_kernel(k, "xq"), arch="gfx950")
    assert t2 != t


@pytest.mark.parametrize("base", [
    dict(k=5120, tiles=3, splits=8, order="xcd", fold="l2", waves=4),
    dict(k=1280, tiles=6, waves=4, pipe=1, gch=2),
    dict(k=5120, tiles=6, splits=5, xin="fp8", xsrc="glb", waves=2, mt=2, order="xcd", fold="l2"),
])
def test_every_mutant_changes_the_ir(base):
    ref = _ir(W.WideKnobs(**base))
    assert "mfma.scale.f32.16x16x128.f8f6f4" in ref
    for m in ("kskip", "scale", "fold", "tok") + (("xq",) if base.get("xsrc") != "glb" else ()):
        kn = W.WideKnobs(**base, mutant=m)
        try:
            kn.check()
        except ValueError:
            continue
        assert _ir(kn) != ref, m


def test_fold_l2_drops_sc1():
    sc1 = _ir(W.WideKnobs(k=5120, tiles=3, splits=8, order="xcd", fold="sc1", waves=4))
    l2 = _ir(W.WideKnobs(k=5120, tiles=3, splits=8, order="xcd", fold="l2", waves=4))
    # partial stores and fold loads carry cache policy 16 (sc1) only in the sc1 fold
    assert sc1.count("i32 16)") > l2.count("i32 16)")


def test_router_knobs_and_mutants():
    with pytest.raises(ValueError):
        RT.RouterKnobs(splits=7).check()
    with pytest.raises(ValueError):
        RT.RouterKnobs(splits=1, mutant="drop").check()
    ref = lower_kernel_to_llvm(RT.build_router_kernel(RT.RouterKnobs(tiles=6, splits=16)), arch="gfx950")
    assert "mfma.f32.16x16x32bf16" in ref.replace(".bf16", "bf16") or "mfma" in ref
    for m in ("tie", "bias", "drop", "renorm"):
        t = lower_kernel_to_llvm(RT.build_router_kernel(RT.RouterKnobs(tiles=6, splits=16, mutant=m)), arch="gfx950")
        assert t != ref, m
    assert RT.grid_of(RT.RouterKnobs(tiles=6, splits=16)) == 96
