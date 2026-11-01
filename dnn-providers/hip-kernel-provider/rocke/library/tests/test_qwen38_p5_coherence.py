# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P5 kernels (``kernels.gfx950.qwen38_mono.probes.p5_coherence``): CPU checks.

Every builder variant lowers with the Python engine (the probe family's
exemption: ``q38p5_`` names, ``backend="python"``), and the emitted LLVM text
carries exactly the cache maintenance each mode names: the bulk rule's
``buffer_wbl2 sc1`` / ``buffer_inv sc1``, ``sc1`` (aux 16) stores and loads,
``s_dcache_inv``, ``glc`` scalar loads, ``s_buffer_load_dword``, and nothing
extra in the negative controls. The GPU proofs (positives pass, negative
controls and mutants fail, timing) run in ``hackathon-prep/qwen38/truemono/P5``.
"""

from __future__ import annotations

import re

import pytest
from kernels.gfx950.qwen38_mono.probes import p5_coherence as P
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


def _common(ll):
    assert "ds_write_b32" in ll  # LDS pad kept (one CTA per CU)
    assert "s_memrealtime" in ll


def _buffer_ops(ll, op):
    """aux (cache-policy) values of every ``raw.ptr.buffer.<op>`` call."""
    pat = rf"call [^\n]*@llvm\.amdgcn\.raw\.ptr\.buffer\.{op}\.[^\n]*?i32 (\d+)\)"
    return [int(x) for x in re.findall(pat, ll)]


@pytest.mark.parametrize("wmode", P.KV_WMODES)
@pytest.mark.parametrize("rmode", P.KV_RMODES)
@pytest.mark.parametrize("tail", [1, 4])
def test_kv_kernel(wmode, rmode, tail):
    k = P.build_kv_kernel(P.KvSpec(wmode=wmode, rmode=rmode, tail=tail))
    assert k.name.startswith("q38p5_kv")
    ll = _ll(k)
    _common(ll)
    assert ("buffer_wbl2 sc1" in ll) == (wmode == "wbl2")
    assert ("buffer_inv sc1" in ll) == (rmode == "inv")
    assert ("buffer_inv sc0" in ll) == (rmode == "inv0")
    stores = _buffer_ops(ll, "store")
    loads = _buffer_ops(ll, "load")
    assert stores and all(a == (16 if wmode == "sc1" else 0) for a in stores)
    # tail rows by rmode, the 32 stable rows always plain
    assert sorted(loads).count(16) == (tail if rmode == "sc1" else 0)
    assert len(loads) == tail + 32
    assert "global_load_dwordx2 $0, $4, $5 sc1" in ll  # bounded ready-pair poll


def test_kv_mutant_and_names():
    names = {P.build_kv_kernel(P.KvSpec(wmode=w, rmode=r)).name for w in P.KV_WMODES for r in P.KV_RMODES}
    assert len(names) == len(P.KV_WMODES) * len(P.KV_RMODES)
    mut = P.build_kv_kernel(P.KvSpec(mutant="early"))
    assert mut.name.endswith("_mutearly")
    _common(_ll(mut))
    ns = P.build_kv_kernel(P.KvSpec(stamps=False, tag="rc"))
    assert ns.name.endswith("_ns_rc")
    assert P.KvSpec().positive and P.KvSpec(wmode="sc1", rmode="sc1").positive
    assert not P.KvSpec(rmode="none").positive and not P.KvSpec(wmode="plain").positive
    assert not P.KvSpec(mutant="early").positive


def test_kv_rejects():
    with pytest.raises(ValueError):
        P.build_kv_kernel(P.KvSpec(rmode="bogus"))
    with pytest.raises(ValueError):
        P.build_kv_kernel(P.KvSpec(tail=9))


@pytest.mark.parametrize("wmode", P.SC_WMODES)
@pytest.mark.parametrize("rmode", P.SC_RMODES)
def test_scalar_kernel(wmode, rmode):
    k = P.build_scalar_kernel(P.ScSpec(wmode=wmode, rmode=rmode))
    ll = _ll(k)
    _common(ll)
    assert ("buffer_wbl2 sc1" in ll) == (wmode == "wbl2")
    smem = rmode.startswith(("smem", "sbuf"))
    assert ("s_load_dword $0, $1, 0x0" in ll) == rmode.startswith("smem")
    assert ("s_buffer_load_dword" in ll) == rmode.startswith("sbuf")
    assert ("s_dcache_inv" in ll) == ("_dinv" in rmode)
    assert (" glc" in ll) == ("_glc" in rmode)
    inv = rmode in ("vec_inv",) or (smem and rmode.endswith("_inv"))
    assert ("buffer_inv sc1" in ll) == inv
    loads = _buffer_ops(ll, "load")
    if rmode == "vec_sc1":
        assert loads == [16]
    elif rmode in ("vec_inv", "vec_inv0", "vec_plain"):
        assert loads == [0]
    else:
        assert loads == []
    assert bool(re.search(r"buffer_inv sc0(?! sc1)", ll)) == (rmode == "vec_inv0")


@pytest.mark.parametrize("rmode", P.HOST_RMODES)
def test_host_kernel(rmode):
    k = P.build_host_kernel(P.HostSpec(rmode=rmode))
    ll = _ll(k)
    _common(ll)
    assert "global_load_dword $0, $3, $4 sc0 sc1" in ll  # round-word poll, system scope
    assert "global_store_dword $0, $1, $2 sc0 sc1" in ll  # per-CTA ack, system scope
    assert ("buffer_inv sc0 sc1" in ll) == rmode.endswith("invsys")
    assert ("s_dcache_inv" in ll) == ("_dinv" in rmode)
    assert bool(re.search(r"buffer_inv sc0(?! sc1)", ll)) == (rmode == "vec_inv0")
    assert (" glc" in ll) == (rmode == "smem_glc")
    assert ll.count("s_load_dword $0, $1, 0x0") == (2 if rmode == "smem_pre" else int(rmode.startswith("smem")))
    if rmode.startswith("vec_"):
        want = {"vec_sc1": 16, "vec_sys": 17}.get(rmode, 0)
        assert _buffer_ops(ll, "load") == [want, want]


def test_host_writer_kernel():
    k = P.build_host_writer_kernel()
    assert k.name == "q38p5_hostw"
    ll = _ll(k)
    assert "store" in ll and "s_memrealtime" not in ll


def test_positive_sets():
    assert {"reg", "vec_sc1", "smem_dinv_inv"} <= P.SC_POSITIVE
    assert "smem" not in P.SC_POSITIVE and "sbuf" not in P.SC_POSITIVE
    assert P.HostSpec().positive and not P.HostSpec(rmode="smem").positive
