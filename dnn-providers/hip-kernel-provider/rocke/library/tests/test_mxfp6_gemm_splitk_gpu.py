# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 numerics of the MXFP6 GEMM split-K / stream-K units (``splitk="fused"``; needs a gfx950 GPU and the rocKE
direct compiler, ``ROCKE_RUST_COMPILE``; skipped elsewhere).

Each case runs ``skbench.run_case`` without timing: every arm on the exact-integer set (bit for bit), Gaussian and
uniform-code data (scaled-MFMA tolerance), three launches each with C NaN-filled first and all three bitwise
identical (the counters are reset by the kernel itself); split arms with s >= 3 also on the order set (the fix-up's
split order, bit for bit); the must-fail mutants (dropped partial, reversed order, missing signal, missing counter
reset, a dropped K step) must each fail.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import pytest


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime / device
        return None


pytestmark = [pytest.mark.skipif(_gpu_arch() != "gfx950", reason="needs a gfx950 GPU"),
              pytest.mark.skipif(not os.environ.get("ROCKE_RUST_COMPILE"),
                                 reason="needs the rocKE direct compiler (ROCKE_RUST_COMPILE)")]


def _sk(s, **kw):
    return {"splitk": "fused", "sk_split": s, **kw}


CASES = [
    # decode-like: few tiles, long K; pure split, write-through release, a short (exact-K) build
    {"id": "dec-ak", "tile": "256x32x256", "fam": "ak", "base": "family", "m": 32, "n": 1024, "k": 8192,
     "arms": {"s4": _sk(4), "s4wt": _sk(4, sk_release="wt"), "s8short": _sk(8, stages=2, short=True),
              "s4x": _sk(4, sk_static=True, sk_release="wt"), "s4xw1": _sk(4, sk_static=True, sk_release="wt1"),
              "s4own": _sk(4, sk_static=True, sk_release="wt", sk_fix="owner"),
              "s4pre": _sk(4, sk_static=True, sk_release="wt", kernarg_preload=True)}, "stress": 20},
    # deep ring, exact-K data-parallel units (s = 1) and a short split
    {"id": "dec-deep", "tile": "128x32x256", "fam": "ak", "base": "family", "m": 7, "n": 640, "k": 4096,
     "arms": {"s1d8": _sk(1, m_mask=True, stages=8, occupancy=1, short=True),
              "s2d8": _sk(2, m_mask=True, stages=8, occupancy=1, short=True, sk_static=True, sk_release="wt")}},
    # the shipped choice (library.dispatch -> splitk.select): decode with a K tail (K % 128 != 0, k_tail) and m_mask;
    # a short grid (36 tiles) split along K
    {"id": "dispatch-ktail", "tile": "128x32x256", "fam": "ak", "base": "dispatch", "m": 6, "n": 640, "k": 2880,
     "arms": {"ship": {}}},
    {"id": "dispatch-short", "tile": "128x128x256", "fam": "ak", "base": "dispatch", "m": 768, "n": 768, "k": 8192,
     "arms": {"ship": {}}},
    # any M (m_mask): rows past M are neither read nor stored
    {"id": "dec-mask", "tile": "256x32x256", "fam": "ak", "base": "family", "m": 5, "n": 768, "k": 4096,
     "arms": {"s4m": _sk(4, m_mask=True), "s1m": _sk(1, m_mask=True)}},
    {"id": "dec-am", "tile": "256x64x256", "fam": "am", "base": "family", "m": 48, "n": 512, "k": 8192,
     "arms": {"s4m": _sk(4, m_mask=True)}},
    # am with M % 16 != 0 / M < tile_m (the M-padded producer layout, gemm.mcontig_rows): decode M = 1, an odd M
    # above one tile (two row tiles, the last one partial), and the shipped dispatch choice
    {"id": "dec-am-odd", "tile": "256x64x256", "fam": "am", "base": "family", "m": 1, "n": 512, "k": 8192,
     "arms": {"s4m": _sk(4, m_mask=True), "s1m": _sk(1, m_mask=True)}},
    {"id": "am-odd-2t", "tile": "128x64x256", "fam": "am", "base": "family", "m": 83, "n": 384, "k": 4096,
     "arms": {"s2m": _sk(2, m_mask=True), "s1m": _sk(1, m_mask=True)}},
    {"id": "dispatch-am-odd", "tile": "128x32x256", "fam": "am", "base": "dispatch", "m": 17, "n": 1024, "k": 4096,
     "arms": {"ship": {}}},
    # edge tiles along N, batch, E3M2
    {"id": "edge-batch", "tile": "128x64x256", "fam": "ak", "base": "config", "batch": 3, "m": 128, "n": 456,
     "k": 4096, "fmt": "e3m2", "arms": {"s2": _sk(2)}},
    # large tile, accumulators in AGPRs: the reducer reads every slot
    {"id": "large", "tile": "256x256x256", "fam": "ak", "base": "config", "m": 512, "n": 512, "k": 6144,
     "arms": {"s3st": _sk(3, ring="stage")}},
    # DP + SK hybrid: 6 tiles on a 4-tile wave -> 4 data-parallel tiles + 2 tiles split 2 ways
    {"id": "hybrid", "tile": "128x128x256", "fam": "ak", "base": "family", "m": 256, "n": 384, "k": 6144,
     "arms": {"h2": _sk(2, ring="stage", sk_hybrid=True, sk_wave=4)}},
]


@pytest.fixture(scope="module")
def env():
    from builders.gfx950.mxfp6_gemm import libbench as LB
    from builders.gfx950.mxfp6_gemm import tune as TU
    from benchmarks.gfx950.mxfp6_gemm import bench as B
    from rocke.runtime.hip_module import Runtime

    td = Path(tempfile.mkdtemp(prefix="sk_gpu_"))
    rt = Runtime()
    cs = LB.CompileServer(os.environ["ROCKE_RUST_COMPILE"], td / "objs", jobs=8, batch=16)
    return rt, cs, LB.Data(td / "data", 42, 8), TU.Clock(B.device_bdf())


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_splitk_case(env, case):
    from builders.gfx950.mxfp6_gemm import skbench as S
    from kernels.gfx950.mxfp6_gemm import library as LIB

    rt, cs, data, clk = env
    res = S.run_case(rt, cs, LIB.LEVERS, data, clk, {**case, "time": False}, random.Random(0))
    assert not res["skipped"], res["skipped"]
    for name in case["arms"]:
        arm = res["arms"][name]
        assert arm["ok"], (name, arm.get("verify"), arm.get("mutants"))
        for mu, v in arm.get("mutants", {}).items():
            if v["required"]:
                assert v["failed_as_required"], (name, mu)
