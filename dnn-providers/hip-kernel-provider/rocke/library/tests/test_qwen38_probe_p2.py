# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the true-mono probe P2 (``qwen38_mono/probes/p2_head.py``).

The GPU gate (bitwise logits / argmax vs ``mx_decode_gemv`` + ATOM's sampler tie
rule, planted ties, NaN / inf rows, stress with mutants that must fail) runs in
``hackathon-prep/qwen38/truemono/P2/bench_p2.py check``."""

import pytest
from kernels.gfx950 import mx_decode_gemv as M
from kernels.gfx950.qwen38_mono.probes import p2_head as P
from rocke.core.ir import uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

N, K = 248320, 2560


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")


@pytest.mark.parametrize("ep", ["fused", "logits", "keys"])
@pytest.mark.parametrize("rm", [4, 16])
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"waves_per_eu": 1},
        {"waves": 8},
        {"pro": "w_first"},
        {"ksplit": 2, "waves": 8},
        {"korder": "lane"},
        {"fsync": "epoch"},
        {"sload": "bcast", "wload": "x1"},
        {"act": "bf16q8"},
    ],
)
def test_head_clone_is_byte_identical(ep, rm, kw):
    """The forked emitter reproduces ``build_mx_decode_gemv`` exactly."""
    sp = M.MxDecodeGemvSpec(n=N, k=K, rows_max=rm, epilogue=ep, **kw)
    assert _ll(P.build_head_clone(sp)) == _ll(M.build_mx_decode_gemv(sp))


@pytest.mark.parametrize("ep", ["fused", "logits"])
@pytest.mark.parametrize("xmode", P.XMODES)
@pytest.mark.parametrize("threads,waves", [(256, 4), (512, 8), (512, 4)])
@pytest.mark.parametrize("pre", P.PRES)
def test_p2_builds_lower(ep, xmode, threads, waves, pre):
    h = M.MxDecodeGemvSpec(n=N, k=K, rows_max=16, epilogue=ep, waves=waves)
    k = P.build_p2(P.P2Spec(head=h, threads=threads, xmode=xmode, pre=pre))
    assert k.name.endswith("_pyonly") and uses_python_only_ops(k)
    ll = _ll(k)
    if xmode == "pairs":
        assert "sc1" in ll or "i32 16)" in ll  # sc1 mailbox loads / stores
    if xmode == "bulk":
        assert "buffer_wbl2" in ll and "buffer_inv" in ll


def test_p2_names_unique_and_mutants_named():
    h = M.MxDecodeGemvSpec(n=N, k=K, epilogue="fused")
    specs = [
        P.P2Spec(head=h, xmode=x, pre=p, mutant=mu)
        for x in ("pairs", "bulk")
        for p in P.PRES
        for mu in ("none", "stale_tag")
    ]
    specs.append(P.P2Spec(head=h, mutant="tie_high"))
    names = [s.kernel_name() for s in specs]
    assert len(set(names)) == len(names)
    assert all(("MUT" in n) == (s.mutant != "none") for n, s in zip(names, specs))


@pytest.mark.parametrize(
    "kw,why",
    [
        ({"xmode": "nope"}, "xmode"),
        ({"threads": 128}, "threads"),
        ({"grid": 200}, "grid"),
        ({"prod_ctas": 7}, "k must split"),
        ({"prod_first": 250}, "producers inside"),
        ({"xmode": "none", "mutant": "stale_tag"}, "stale_tag"),
    ],
)
def test_p2_invalid(kw, why):
    h = M.MxDecodeGemvSpec(n=N, k=K, epilogue="fused")
    ok, msg = P.is_valid_p2(P.P2Spec(head=h, **kw))
    assert not ok and why in msg


def test_p2_invalid_logits_tie_mutant():
    h = M.MxDecodeGemvSpec(n=N, k=K, epilogue="logits")
    ok, msg = P.is_valid_p2(P.P2Spec(head=h, mutant="tie_high"))
    assert not ok and "tie_high" in msg


def test_producer_builds():
    k = P.build_p2_producer(k=K, rows_max=16, threads=512)
    assert "p2prod" in k.name
    _ll(k)
    with pytest.raises(ValueError):
        P.build_p2_producer(k=K, rows_max=4, prod_ctas=7)


def test_p2_nanrule_and_pimpl_knobs():
    h = M.MxDecodeGemvSpec(n=N, k=K, epilogue="fused")
    a = P.P2Spec(head=h, nanrule="sampler")
    bb = P.P2Spec(head=h, pimpl="ir")
    assert "nansampler" in a.kernel_name() and "piir" in bb.kernel_name()
    assert _ll(P.build_p2(a)) != _ll(P.build_p2(P.P2Spec(head=h)))
    _ll(P.build_p2(bb))
    ok, msg = P.is_valid_p2(P.P2Spec(head=M.MxDecodeGemvSpec(n=N, k=K, epilogue="logits"), nanrule="sampler"))
    assert not ok and "nanrule" in msg


@pytest.mark.parametrize("rm", [4, 16])
def test_p2_l2pf_builds(rm):
    h = M.MxDecodeGemvSpec(n=N, k=K, rows_max=rm, epilogue="fused")
    sp = P.P2Spec(head=h, threads=512, l2pf=4)
    assert "l2pf4" in sp.kernel_name()
    ll = _ll(P.build_p2(sp))
    assert "global_load_dwordx4" in ll
    ok, msg = P.is_valid_p2(P.P2Spec(head=h, threads=256, l2pf=4))
    assert not ok and "spare wave" in msg
