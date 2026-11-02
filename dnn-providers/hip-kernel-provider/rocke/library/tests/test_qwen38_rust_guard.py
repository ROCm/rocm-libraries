# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PLAN decision 18 (a): the serving guard of the Rust direct code-object backend (CPU only).

* a kernel family / variant compiles direct only when the tuning DB holds a passing paired record on BOTH rocke
  nodes (WIN, or TIE with ratio <= 1) with one lever set; every regressing family (HC, fused MoE rows 4, the head,
  Q2) falls back to COMGR;
* the rule's must-fail cases (one node only, a tie above 1, a loss, different lever sets, no record);
* the guard never touches the IR, its serialization or the LLVM text, so every COMGR object is byte-identical;
* the existing D0.6 behaviour is unchanged with the guard off (``ROCKE_DIRECT_GUARD`` unset).
"""

from __future__ import annotations

import json

import pytest
from kernels.gfx950.qwen38_mono.framework import FrameworkKnobs
from kernels.gfx950.qwen38_mono.persist import knobs as K
from kernels.gfx950.qwen38_mono.persist import ptoy as T
from kernels.gfx950.qwen38_mono.persist import rust_guard as G
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import compile as C

B3 = "+".join(K.RUST_LEVER_SETS["b3"])
M1 = "+".join(K.RUST_LEVER_SETS["moe_r1"])


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")


def _recs(rows):
    out = {}
    for key, node, ratio, verdict, levers in rows:
        r = G.make_record(key, node, ratio=ratio, verdict=verdict, levers=levers, base="test", job="t")
        out[(key, G.node_of(node))] = r
    return out


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in ("ROCKE_DIRECT_GUARD", "ROCKE_DIRECT_GUARD_DB", "ROCKE_CODEOBJ_BACKEND", "ROCKE_DIRECT_LEVERS"):
        monkeypatch.delenv(v, raising=False)


# ------------------------------------------------------------------ the shipped DB


@pytest.mark.parametrize(
    "key",
    ["hc.delta_ple.attn.r4s", "hc.qsa.mlp.r2s", "hc.delta.mlp.r4s", "hc.delta_ple.mlp.r4s", "hc.qsa.mlp.r4s",
     "head.fused.r4", "head.fused.r16", "head.logits.r16", "fw.q2.r1", "fw.q2.r4s", "probe.p5.kv", "moe_lb2.tp1.fp8.r16",
     "no.such.kernel"],
)
def test_regressing_families_fall_back_to_comgr(key):
    s = G.select(key)
    assert s.backend == "comgr" and s.levers == "", s


@pytest.mark.parametrize(
    "key,levers",
    [("b3.r1.tuned", B3), ("b3.r4s.tuned", B3), ("b3.r4s.mono", B3), ("b3.r2s.tuned", B3), ("b3.r4g2.tuned", B3),
     ("moe_fused.r1", M1), ("fw.b3.r1", B3), ("fw.moe.r1", M1), ("toy.d8w0", B3),
     ("hc.delta.attn.r2s", B3 + "+dsfold+addrfold+dppcombinei+minmax3+ldspairsfar+fmac+vmlat60")],
)
def test_both_node_winners_go_direct_with_the_timed_set(key, levers):
    s = G.select(key)
    assert s.backend == "direct" and s.levers == levers, s
    assert {n for n, _r, _v in s.ratios} == set(G.NODES) and all(r <= 1.0 for _n, r, _v in s.ratios)


def test_shipped_db_is_a_tuning_db_of_ratios_only():
    from rocke.dispatch.tuning.search.db import TuningDB

    data = json.loads(G.db_path().read_text(encoding="utf-8"))
    assert data["version"] in (1, 2) and data["records"]
    for d in data["records"]:
        assert d["family"] == G.FAMILY and d["unit"] == "ratio" and 0.5 < d["value"] < 2.0
        assert d["knobs"]["verdict"] in G.VERDICTS
    assert len(TuningDB(None).records()) == 0  # the class reads the same format (no lock needed in tests)


# --------------------------------------------------------------------- the rule


def test_rule_must_fail_cases():
    win = [("k", "n01-09", 0.98, "WIN", "a+b"), ("k", "n07-17", 0.99, "WIN", "a+b")]
    s = G.select("k", _recs(win))
    assert (s.backend, s.levers) == ("direct", "a+b")
    # one node only
    assert G.select("k", _recs(win[:1])).backend == "comgr"
    # a tie above 1 on one node
    assert G.select("k", _recs([win[0], ("k", "n07-17", 1.002, "TIE", "a+b")])).backend == "comgr"
    # a tie at or below 1 passes ("TIE-or-better")
    assert G.select("k", _recs([win[0], ("k", "n07-17", 0.999, "TIE", "a+b")])).backend == "direct"
    # a loss, even below 1 by point estimate (the harness's verdict wins)
    assert G.select("k", _recs([win[0], ("k", "n07-17", 0.999, "LOSS", "a+b")])).backend == "comgr"
    # measured but not booked (screen and confirmation disagree)
    assert G.select("k", _recs([win[0], ("k", "n07-17", 0.99, "HOLD", "a+b")])).backend == "comgr"
    # the two nodes timed different objects
    assert G.select("k", _recs([win[0], ("k", "n07-17", 0.99, "WIN", "a")])).backend == "comgr"
    with pytest.raises(ValueError):
        G.make_record("k", "n03-05", ratio=1.0, verdict="WIN", base="b", job="j")
    with pytest.raises(ValueError):
        G.make_record("k", "n01-09", ratio=1.0, verdict="MAYBE", base="b", job="j")


def test_family_selection_needs_every_key():
    rows = [("b3.r1", n, 0.98, "WIN", "a") for n in G.NODES] + [("b3.r4", n, 0.97, "WIN", "a") for n in G.NODES]
    assert G.select_family("b3", _recs(rows)).backend == "direct"
    bad = rows + [("b3.r9", "n01-09", 1.03, "LOSS", "a"), ("b3.r9", "n07-17", 1.02, "LOSS", "a")]
    assert G.select_family("b3", _recs(bad)).backend == "comgr"
    assert G.select_family("b3x", _recs(rows)).backend == "comgr"  # no records
    assert G.select_family("b3", _recs(rows + [("b3.r8", n, 0.97, "WIN", "z") for n in G.NODES])).backend == "comgr"


# ------------------------------------------------------- kernels: marks, bytes


def test_guard_kernel_marks_and_never_touches_ir():
    base = T.build_static()
    ser, ll = serialize(base), _ll(base)
    k = T.build_static()
    k, s = G.guard_kernel(k, "hc.delta_ple.attn.r4s")
    assert s.backend == "comgr" and C.requested_code_object_backend(k) == "comgr" and "direct_levers" not in k.attrs
    assert serialize(k) == ser and _ll(k) == ll
    d = T.build_static()
    d, s = G.guard_kernel(d, "b3.r1.tuned")
    assert C.requested_code_object_backend(d) == "direct" and d.attrs["direct_levers"] == B3
    assert serialize(C.direct_source(d)) == ser and _ll(d) == ll and d.name == base.name


def test_guarded_comgr_kernel_ignores_the_env_direct_switch(monkeypatch):
    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "direct")
    k, _ = G.guard_kernel(T.build_static(), "moe_lb2.tp1.fp8.r16")
    assert C._selected_code_object_backend(None, k) == "comgr"
    u = T.build_static()
    assert C._selected_code_object_backend(None, u) == "direct"  # unguarded kernels keep the experiment switch


def test_rust_family_with_a_key_uses_the_db():
    k = K.rust_family(T.build_static(), "b3", key="b3.r4s.mono")
    assert C.requested_code_object_backend(k) == "direct" and k.attrs["direct_levers"] == B3
    m = K.rust_family(T.build_static(), "head", key="head.fused.r4")
    assert C.requested_code_object_backend(m) == "comgr" and "direct_levers" not in m.attrs
    # fused MoE rows 4: the RNR set (no nofence) is booked on both nodes (the key decides, not the family table)
    r4 = K.rust_family(T.build_static(), "moe_r4", key="moe_fused.r4")
    assert C.requested_code_object_backend(r4) == "direct" and "nofence" not in r4.attrs["direct_levers"]


def test_guard_off_keeps_d06_behaviour_and_guard_on_gates_families(monkeypatch, tmp_path):
    k = K.rust_family(T.build_static(), "b3")
    assert C.requested_code_object_backend(k) == "direct"  # unchanged (guard off)
    monkeypatch.setenv("ROCKE_DIRECT_GUARD", "1")
    k = K.rust_family(T.build_static(), "b3")
    assert C.requested_code_object_backend(k) == "direct"  # every b3 key passes in the shipped DB
    # a DB in which one B3 variant loses keeps the whole family off direct (mutant of the shipped DB)
    from rocke.dispatch.tuning.search.db import TuningDB

    db = TuningDB(tmp_path / "db.json")
    for (key, node), r in G.load_records().items():
        db.put(r, force=True)
    db.put(G.make_record("b3.r9.tuned", "n01-09", ratio=1.02, verdict="LOSS", levers=B3, base="m", job="m"), force=True)
    db.put(G.make_record("b3.r9.tuned", "n07-17", ratio=1.01, verdict="LOSS", levers=B3, base="m", job="m"), force=True)
    monkeypatch.setenv("ROCKE_DIRECT_GUARD_DB", str(tmp_path / "db.json"))
    k = K.rust_family(T.build_static(), "b3")
    assert C.requested_code_object_backend(k) is None and "direct_levers" not in k.attrs


def test_scope_backend_guard_keys():
    sc = K.RustScope.of({"g": "b3", "r": "moe_r1"})
    w = K.RUST_DIRECT_FAMILIES
    assert K.rust_scope_backend(sc, ("g", "r"), w) == "direct"  # D0.6 rule unchanged
    assert K.rust_scope_backend(sc, ("g", "r"), w, guard_keys={"g": "fw.b3.r1", "r": "fw.moe.r1"}) == "direct"
    assert K.rust_scope_backend(sc, ("g", "r"), w, guard_keys={"g": "fw.b3.r1", "r": "fw.q2.r1"}) == "comgr"
    assert K.rust_scope_backend(sc, ("g", "r"), guard_keys={"g": "fw.b3.r1", "r": "fw.moe.r1"}) == "comgr"  # D0.6 first


def test_framework_auto_is_unchanged_by_the_guard_module():
    a = T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3", "r": "b3"}), code_object_backend="auto"))
    assert C.requested_code_object_backend(a) == "comgr"


def test_comgr_objects_byte_identical_with_the_guard():
    """With a COMGR available: a guarded-to-COMGR kernel compiles to the unguarded kernel's code object."""
    try:
        base = C.compile_kernel(T.build_static(), arch="gfx950", backend="python", code_object_backend="comgr")
    except Exception as e:  # no COMGR on this host
        pytest.skip(f"no COMGR: {type(e).__name__}")
    k, _ = G.guard_kernel(T.build_static(), "hc.delta_ple.attn.r4s")
    art = C.compile_kernel(k, arch="gfx950", backend="python")
    assert art.code_object_backend == "comgr" and bytes(art.hsaco) == bytes(base.hsaco)
