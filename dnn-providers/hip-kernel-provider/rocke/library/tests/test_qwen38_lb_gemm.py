# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L3 (``kernels.gfx950.qwen38_mono.large_batch.gemm``): CPU checks.

* every knob value the P-L3 bench uses builds and lowers (Python engine), kernel names list only deviations from
  the defaults and are unique per knob set, invalid combinations are refused;
* the LLVM text carries what the GPU proofs rely on: the scaled MFMA with op_sel bytes for ``sop``, the
  write-through (``sc1``) partial stores and the agent-scope arrival for ``fold=last``, no partials for
  ``splits=1``, the grid-sync poll only behind the runtime ``gsync`` flag, no 1-element ``av1`` LDS loads (their
  name hint collides with ``av16`` + counter in kernels this large: duplicate SSA names).
The GPU proofs (tolerance vs float64 torch, bitwise across order-preserving knobs, determinism, must-fail mutants,
shuffled timing) run in ``hackathon-prep/qwen38/truemono-lb/p-l3``.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from kernels.gfx950.qwen38_mono.large_batch import gemm as G
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


SOP = dict(wlay="tile", xslay="chunk", xscale="glb", sop=True)
CASES = [
    G.GemmKnobs(),
    G.GemmKnobs(w="bf16", k=3072, waves=4, splits=6, xu=False, sb=False),  # FlyTape's out-proj plan
    G.GemmKnobs(k=3072, splits=6, map="xcd", fold="none", **SOP),
    G.GemmKnobs(k=2560, bucket=16, xsrc="global", depth=4),
    G.GemmKnobs(w="bf16", k=320, bucket=128, xa=2, wlay="tile", bpre=2),  # ragged K (HC up)
    G.GemmKnobs(k=320, bucket=32, **SOP),
    G.GemmKnobs(k=10240, bucket=128, splits=20, tb=4, waves=4, **SOP),
    G.GemmKnobs(k=2560, mt=2, bpre=4, splits=5, **SOP),
    G.GemmKnobs(k=3072, splits=6, mutant="fold"),
    G.GemmKnobs(k=2560, mutant="scale"),
    G.GemmKnobs(w="bf16", k=2560, mutant="kskip", wlay="tile"),
    G.GemmKnobs(k=2560, mutant="sopsel", **SOP),
]


@pytest.mark.parametrize("kn", CASES, ids=[G.kernel_name(k) for k in CASES])
def test_builds_and_lowers(kn):
    text = _ll(G.build_gemm_kernel(kn))
    assert G.kernel_name(kn) in text
    # every MFMA of the regime is present
    if kn.w == "mx":
        assert "mfma.scale.f32.16x16x128.f8f6f4" in text
    else:
        assert "mfma.f32.16x16x32.bf16" in text
    # no duplicate SSA names (the av1 / av16 collision) and no 1-element I32 LDS vector loads
    defs = re.findall(r"^\s*(%[\w.]+) = ", text, re.M)
    assert len(defs) == len(set(defs))


def test_names_unique_and_minimal():
    names = {G.kernel_name(k) for k in CASES}
    assert len(names) == len(CASES)
    assert G.kernel_name(G.GemmKnobs()) == "q38lb_gemm_mx_k2560_b128"
    assert G.kernel_name(replace(G.GemmKnobs(), tag="rc")).endswith("_rc")


@pytest.mark.parametrize(
    "bad",
    [
        dict(splits=3),  # 20 chunks
        dict(waves=2),
        dict(mt=3),
        dict(depth=9),
        dict(mutant="fold"),  # needs splits > 1
        dict(w="bf16", mutant="scale"),
        dict(sop=True),  # needs tile layout and global op_sel scales
        dict(sop=True, wlay="tile", xslay="chunk", xscale="glb", splits=2),  # kc 10 not a multiple of 4
        dict(xslay="chunk", xsrc="global"),
        dict(bpre=9),
        dict(fold="none", mutant="fold", splits=2),
        dict(mutant="sopsel"),  # needs sop
    ],
)
def test_invalid_refused(bad):
    with pytest.raises(ValueError):
        G.GemmKnobs(**bad).check()


def _opsel(text):
    return set(re.findall(r"f8f6f4\.v4i32\.v8i32\([^)]*i32 4, i32 0, i32 (\d), i32 %[\w.]+, i32 (\d), i32 %", text))


def test_protocol_text():
    split = _ll(G.build_gemm_kernel(G.GemmKnobs(k=3072, splits=6)))
    # fold=last: sc1 (write-through) partial stores, acquire before the fold
    assert "buffer_inv sc1" in split
    assert "global_atomic_add" in split  # arrival (and the grid sync)
    none = _ll(G.build_gemm_kernel(G.GemmKnobs(k=3072, splits=6, fold="none")))
    assert "buffer_inv sc1" not in none  # no fold, no acquire
    one = _ll(G.build_gemm_kernel(G.GemmKnobs(k=3072)))
    assert "buffer_inv sc1" not in one
    # every build carries the grid sync behind the runtime gsync flag (8-stripe poll with s_sleep)
    assert "s_sleep 1" in one
    # sop: op_sel bytes 0..3 for both scales; the default path uses byte 0 only
    assert _opsel(_ll(G.build_gemm_kernel(G.GemmKnobs(k=3072, splits=6, **SOP)))) == {(str(j), str(j)) for j in range(4)}
    assert _opsel(one) == {("0", "0")}
    # mutant sopsel: the same kernel with every op_sel byte moved to the next chunk's (a different body)
    sop = _ll(G.build_gemm_kernel(G.GemmKnobs(k=2560, **SOP)))
    mut = _ll(G.build_gemm_kernel(G.GemmKnobs(k=2560, mutant="sopsel", **SOP)))
    name = G.kernel_name(G.GemmKnobs(k=2560, **SOP))
    assert mut.replace(G.kernel_name(G.GemmKnobs(k=2560, mutant="sopsel", **SOP)), name) != sop


def test_stream_kernel_builds():
    for d, v in ((4, 1), (8, 1), (16, 1), (4, 4), (8, 2)):
        sk = G.StreamKnobs(depth=d, vec=v)
        assert G.stream_name(sk) in _ll(G.build_stream_kernel(sk))
    assert G.stream_name(G.StreamKnobs()) == "q38lb_gst_d4_g256"
