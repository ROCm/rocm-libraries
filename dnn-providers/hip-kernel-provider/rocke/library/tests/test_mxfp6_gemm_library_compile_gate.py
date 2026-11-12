"""Compile gate of the shipped MXFP6 tile library: every ``library.dispatch`` kernel (each config x family, at its
benchmark S and G problems, both element formats) builds with the direct compiler and the library levers without VGPR spills or scratch.

Needs the rocKE direct compiler (``ROCKE_RUST_COMPILE``); no GPU. About 560 compiles.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from builders.gfx950.mxfp6_gemm import libtiles as LT
from kernels.gfx950.mxfp6_gemm import gemm as G
from kernels.gfx950.mxfp6_gemm import library as LB

pytestmark = pytest.mark.skipif(not os.environ.get("ROCKE_RUST_COMPILE"),
                                reason="needs the rocKE direct compiler (ROCKE_RUST_COMPILE)")

CASES = [(mt0, mt1, du, fam, st, fmt) for (mt0, mt1, du) in LB.LIBRARY for fam in ("ak", "am") for st in ("S", "G")
         for fmt in ("e2m3", "e3m2")]


@pytest.mark.parametrize("cfg", CASES, ids=[f"{a}x{b}x{c}_{f}_{s}_{t}" for a, b, c, f, s, t in CASES])
def test_dispatch_kernel_has_no_spills(cfg):
    from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust, code_object_notes

    mt0, mt1, du, fam, st, fmt = cfg
    p = LT.problems(LT.Tile(mt0, mt1, du))[{"S": 0, "G": 1}[st]]
    spec = LB.dispatch(mt0, mt1, du, fam, fmt, p.m, p.n, p.batch, k_target=p.k)
    old = os.environ.get("ROCKE_RUST_LEVERS")
    os.environ["ROCKE_RUST_LEVERS"] = LB.LEVERS
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "k.hsaco"
            _compile_rust(G.build_mxfp6_gemm(spec, arch="gfx950"), out)
            notes = code_object_notes(out.read_bytes())
    finally:
        if old is None:
            os.environ.pop("ROCKE_RUST_LEVERS", None)
        else:
            os.environ["ROCKE_RUST_LEVERS"] = old
    assert not notes.get("vgpr_spill_count") and not notes.get("sgpr_spill_count") \
        and not notes.get("private_segment_fixed_size"), (spec.schedule, spec.tile_m, spec.tile_n, notes)
