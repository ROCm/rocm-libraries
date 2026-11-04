# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the backward sweep driver: release masks, the candidate -> spec
adapter, legality by the family validator, deduplication of identical
kernels, stage construction and chaining across records, representative
cells, counts, the job plan, scoring and the stage driver (with a stubbed
compiler and timing harness).

CPU only; nothing is compiled or timed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.common import attention_bwd_sweep as sw
from kernels.common.attention_bwd import (
    AGPR_ALLOCS,
    NOT_YET_EFFECTIVE_KNOBS,
    AttnBwdSpec,
    _not_built,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_dq import validate_attn_bwd_dq_spec

from ._attention_bwd_test_utils import fake_bench_ok

REL1 = sw.Release(frozenset({1}))
BUILT = sw.Release(frozenset({1, 2, 4, 5, 6, 7}))


def _sl(arch="gfx942", d=128):
    return sw.primary_slice(arch, "general", d)


# --------------------------------------------------------------------------- release
def test_release_masks_and_pinned_knobs():
    sl = _sl()
    base = sw.default_knobs(sl)
    assert REL1.values(sl, "transpose_source", "lds_plain") == ("lds_plain",)
    assert BUILT.values(sl, "ws_layout", "head_major") == ("head_major", "token_major")
    for k in NOT_YET_EFFECTIVE_KNOBS:
        assert not sw.Release(sw.ALL_RELEASED).has(k)  # pinned while inert
    assert sw.Release.parse("1,2").families == {1, 2}
    assert sw.Release.parse("all").families == set(sw.LEVER_FAMILIES)
    assert sw.Release.parse(None).families == {1}
    with pytest.raises(ValueError):
        sw.Release.parse("9")
    assert set(sw.KNOB_FAMILY) >= set(sw.SWEPT_KNOBS) - {"head_pack"}
    assert set(base) == set(sw.SWEPT_KNOBS)


def test_default_knobs_are_the_shipped_start_point():
    for arch in ("gfx942", "gfx950", "gfx1151"):
        for d in (32, 64, 128):
            sl = _sl(arch, d)
            c = sw.Candidate.make(sl, sw.default_knobs(sl))
            shipped = validate_attn_bwd_spec(
                AttnBwdSpec(head_size=d, dtype="bf16"), arch
            )
            assert sw.legality(c).resolved.kernel_name() == shipped.kernel_name()


# --------------------------------------------------------------------------- adapter
def test_candidate_spec_adapter_and_record_roundtrip():
    sl = _sl("gfx950", 64)
    c = sw.geometry_stage(sl, REL1).candidates[0]
    spec = c.spec()  # AttnBwdSpec names (scheduler_strategy, tuples)
    assert isinstance(spec, AttnBwdSpec)
    fields = c.spec_fields()
    assert "codegen_strategy" not in fields and "scheduler_strategy" in fields
    assert not set(fields) & set(sw.RUNTIME_KNOBS)
    json.dumps(fields)
    back = sw.Candidate.from_record(json.loads(json.dumps(c.record())))
    assert back == c and back.key() == c.key()
    r = c.with_runtime(g_split=2)
    assert r.key() != c.key() and r.spec().dkv_mode == "atomic"
    assert c.config().label == c.key()
    assert dict(c.config().knobs) == {k: v for k, v in c.k.items() if k != "head_pack"}


# --------------------------------------------------------------------------- legality
def test_legality_is_the_validator_plus_the_emission_check():
    sl = _sl("gfx950", 128)
    g = sw.geometry_stage(sl, REL1).candidates
    n = 0
    for c in g[:6]:
        for x in sw._structure_cands(c, sw.Release(sw.ALL_RELEASED))[::5]:
            lg = sw.legality(x)
            try:
                s = validate_attn_bwd_spec(x.spec(), sl.arch)
            except ValueError:
                assert not lg.ok and lg.rule not in (None, "not_built")
                continue
            if not _not_built(s) and s.dq_mode == "split":
                # split points: the dQ kernel the plan builds is validated too
                try:
                    validate_attn_bwd_dq_spec(s.dq_spec(**x.dq_knobs()), sl.arch)
                except ValueError:
                    assert not lg.ok and lg.rule.startswith("dq_"), lg
                    continue
            assert lg.ok == (not _not_built(s))
            assert (lg.rule == "not_built") == bool(_not_built(s))
            n += 1
    assert n > 10


@pytest.mark.parametrize(
    "msg,tok",
    [
        ("LDS plan 70000 B exceeds 65536 B on gfx942", "lds"),
        (
            "resource gate: this resolved spec fails on gfx942 (sgpr_spill: set "
            "waves_per_eu_2 64 direct; RESOURCE_REFUSED)",
            "resource_refused",
        ),
        (
            "... spills arch VGPRs on gfx1151 (LAYOUT_SPILLING_COMBOS)",
            "resource_refused",
        ),
        ("agpr_alloc=(0, 0) needs arch-VGPR + AGPR estimate 300 <= 256", "vgpr_form"),
        ("waves_per_eu=2 needs arch-VGPR + AGPR estimate 300 <= 256", "waves_per_eu"),
        ("arch-VGPR estimate 300 exceeds 256 (+32)", "arch_vgpr"),
        ("AGPR estimate 300 exceeds 256 (+32)", "agpr"),
        ("agpr_alloc=(128, 128) below the AGPR estimate 192", "agpr_alloc"),
        ("agpr_alloc=(256, 256) leaves 0 of the 256 registers per wave", "x"),
        ("kt_source='reg' requires kv_residency='reg'", "coupling"),
    ],
)
def test_rule_tokens(msg, tok):
    want = "agpr_alloc_share" if tok == "x" else tok
    assert sw.rule_token(msg) == want


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
@pytest.mark.parametrize("d", [64, 128])
def test_geometry_stage_builds_every_candidate(arch, d):
    """Every geometry-stage candidate is legal and built (portable paths)."""
    gs = sw.geometry_stage(_sl(arch, d), REL1)
    assert len(gs.candidates) > 1
    for c in gs.candidates:
        lg = sw.legality(c)
        assert lg.ok, (c.k, lg.rule, lg.detail)
        assert not _not_built(lg.resolved)
        assert c.k["sched"] == "none" and c.k["ring_depth"] == 1
        assert c.k["global_path"] == "vgpr"
    assert len({c.key() for c in gs.candidates}) == len(gs.candidates)


def test_structures_enumerate_grids_and_untied_atoms():
    sl = _sl("gfx950", 64)
    c = next(
        x
        for x in sw.geometry_stage(sl, REL1).candidates
        if x.k["waves"] == 4 and x.k["block_m"] == 32 and x.k["block_n"] == 64
    )
    ss = sw.structures(c, sw.Release(frozenset({1, 2, 3, 4})))
    assert any(s["warp_grid_g02"] == (2, 2) for s in ss)
    assert all(
        s["pt_route"] == "lds"
        for s in ss
        if (s["warp_grid_g02"], s["warp_grid_g13"]) != ((1, 4), (4, 1))
    )
    assert any(s["atom_g4"] != s["atom_g02"] for s in ss)
    assert not any(s["global_path"] == "dma" and s["transpose_source"] == "xt_lds"
                   for s in ss)  # fmt: skip
    # unreleased families keep the candidate's values
    one = sw.structures(c, REL1)
    assert len(one) == 1 and all(one[0][k] == c.k[k] for k in one[0])
    # gfx1201: never relabel (the validator rejects it on WMMA)
    sl12 = _sl("gfx1201", 64)
    for g in sw.geometry_stage(sl12, REL1).candidates[:3]:
        for s in sw.structures(g, sw.Release(sw.ALL_RELEASED)):
            assert s["pt_route"] == "lds" and s["kv_residency"] == "lds"


# --------------------------------------------------------------------------- dedupe
def test_dedupe_collapses_resolved_duplicates_only():
    sl = _sl("gfx942", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    assert sw.legality(c).resolved.agpr_alloc == (0, 0)
    same = [c, c.with_knobs(agpr_alloc=(0, 0)), c.with_knobs(waves_per_eu=None)]
    reps, dup = sw.dedupe(same)
    assert len(reps) == 1 and len(dup) == 2
    assert reps[0].key() == min(x.key() for x in same)
    other = c.with_knobs(edge_tiles=True)
    assert len(sw.dedupe(same + [other])[0]) == 2


def test_block_k4_is_inert_under_step_emission():
    """kK4 no longer changes the resolved agpr_alloc (it used to on this tile,
    through the register estimate), so the candidates are one kernel; the
    backend-chosen AGPR form that kK4 64 reached by accident is now reached by
    ``agpr_alloc = "backend"``."""
    sl = _sl("gfx942", 128)
    base = sw.default_knobs(sl)
    base.update(waves=4, block_m=32, block_n=64, warp_grid_g4=(1, 4))
    base.update(warp_grid_g02=(1, 4), warp_grid_g13=(4, 1), agpr_alloc=None)
    a = sw.Candidate.make(sl, {**base, "block_k4": 32})
    b = sw.Candidate.make(sl, {**base, "block_k4": 64})
    la, lb = sw.legality(a), sw.legality(b)
    assert la.ok and lb.ok
    assert la.resolved.agpr_alloc == lb.resolved.agpr_alloc == (0, 0)
    assert (la.arch_vgpr, la.agpr) == (lb.arch_vgpr, lb.agpr)
    assert len(sw.dedupe([a, b])[0]) == 1
    be = a.with_knobs(agpr_alloc="backend")
    assert sw.legality(be).ok
    assert len(sw.dedupe([a, b, be])[0]) == 2


def test_g4_slicing_is_enumerated_with_block_k4_in_stage_h():
    """Stage H enumerates (g4_slicing, block_k4): the candidate's own pair,
    then every slicing value at every kK4 that is a multiple of the G4 atom K
    dividing kN0. A one-step ``slice`` and a one-slice ``slice_db`` resolve to
    the code they emit, so dedupe merges them; distinct slice widths stay
    distinct kernels; the geometry stage keeps kK4 pinned."""
    allrel = sw.Release(sw.ALL_RELEASED)
    for arch, k4a in (("gfx942", 16), ("gfx950", 32)):
        sl = _sl(arch, 64)
        c = sw.Candidate.make(sl, sw.default_knobs(sl))
        pts = sw.g4_slicing_values(c, allrel)
        assert pts[0] == {"g4_slicing": "step", "block_k4": c.k["block_k4"]}
        n = c.k["block_n"]
        k4s = {k for k in (16, 32, 64) if n % k == 0 and k % k4a == 0}
        for v in ("slice", "slice_db", "slice_db_g3"):
            assert {p["block_k4"] for p in pts if p["g4_slicing"] == v} == k4s
        assert sw.g4_slicing_values(c, REL1) == pts[:1]  # family 5 unreleased
        cands = [c.with_knobs(**p) for p in pts]
        assert all(sw.legality(x).ok for x in cands)
        reps, dup = sw.dedupe(cands)
        one_step = c.with_knobs(g4_slicing="slice", block_k4=k4a)
        assert (
            sw.effective_key(one_step)
            == sw.effective_key(c.with_knobs(block_k4=k4a))
            == sw.effective_key(c)
        )
        assert len(reps) == len(cands) - 1  # only the one-step slice merges
        db = {
            sw.effective_key(c.with_knobs(g4_slicing="slice_db", block_k4=k))
            for k in k4s
        }
        assert len(db) == len(k4s)
        h = sw.schedule_values(c, sw.Release(sw.ALL_RELEASED, "production"))
        assert {p.get("g4_slicing", "step") for p in h} == {
            "step",
            "slice",
            "slice_db",
            "slice_db_g3",
        }
        full = sw.schedule_values(c, sw.Release(sw.ALL_RELEASED, "full"))
        assert {(p["g4_slicing"], p["block_k4"]) for p in full} == {
            (p["g4_slicing"], p["block_k4"]) for p in pts
        }
        geo = sw.geometries(sl, allrel)
        assert len({g["block_k4"] for g in geo if g["block_n"] == n}) == 1
    # MFMA LDS accumulators, the per-fragment dQ issue and WMMA: step only
    sl = _sl("gfx942", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    for knobs in (dict(acc_in_lds=True, agpr_alloc=(0, 0)), dict(dq_issue="fragment")):
        assert len(sw.g4_slicing_values(c.with_knobs(**knobs), allrel)) == 1
    rdna = _sl("gfx1151", 64)
    cr = sw.Candidate.make(rdna, sw.default_knobs(rdna))
    assert len(sw.g4_slicing_values(cr, allrel)) == 1
    # the default pair keeps the earlier candidate keys
    assert c.with_knobs(g4_slicing="step").key() == c.key()


def test_backend_alloc_dedupes_with_a_policy_none_resolution():
    """``"backend"`` emits what a policy result of ``None`` emits, so the two
    collapse where the policy resolves to ``None``."""
    sl = _sl("gfx950", 128)
    for c in sw.geometry_stage(sl, sw.Release(sw.ALL_RELEASED)).candidates:
        lg = sw.legality(c)
        if lg.resolved.agpr_alloc is None:
            be = c.with_knobs(agpr_alloc="backend")
            assert sw.effective_key(be) == sw.effective_key(c)
            break
    else:
        pytest.skip("no geometry resolves to the backend form here")


def test_schedule_stage_offers_the_backend_alloc():
    sl = _sl("gfx942", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    allocs = {
        v["agpr_alloc"] for v in sw.schedule_values(c, sw.Release(sw.ALL_RELEASED))
    }
    assert {None, "backend", (0, 0)} <= allocs


def test_workgroup_share_rule_has_its_own_token():
    assert (
        sw.rule_token(
            "workgroup share: waves=8 puts 2 waves on each SIMD; arch-VGPR + AGPR "
            "estimate 300 exceeds 256 per wave"
        )
        == "workgroup_share"
    )


def test_tuning_knobs_leave_head_pack_to_the_plan():
    """A configuration from any slice but ``head_pack`` must not pin
    ``head_pack`` (a copied ``False`` switched packing off for the class's
    short-query requests); the ``head_pack`` slice forces it on."""
    sl = _sl("gfx950", 128)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    assert "head_pack" in c.k  # the compile key keeps it
    assert "head_pack" not in c.tuning_knobs()
    assert "head_pack" not in c.config().knobs
    hp = sw.secondary_slice(sl, "head_pack")
    c2 = sw.Candidate.make(hp, sw.default_knobs(hp))
    assert c2.tuning_knobs()["head_pack"] is True


def test_fallback_structure_order_is_meaningful_before_the_key():
    """Fewest arch VGPRs, then AGPRs, then the structure nearest the default,
    then LDS; the key only breaks exact ties."""
    checked = 0
    for sl in (_sl(a, d) for a in ("gfx942", "gfx950") for d in (64, 128)):
        checked += _check_fallbacks(sl)
    assert checked >= 1


def _check_fallbacks(sl):
    gs = sw.geometry_stage(sl, REL1)
    base = sw.default_knobs(sl)
    checked = 0
    for g in gs.candidates:
        geo = {k: g.k[k] for k in sw.GEOMETRY_KNOBS}
        knobs = {**base, **geo, **sw._default_atoms(sl, geo["block_k4"])}
        knobs["warp_grid_g02"] = (1, geo["waves"])
        knobs["warp_grid_g13"] = (geo["waves"], 1)
        default = sw.Candidate.make(sl, knobs)
        if sw.legality(default).ok:
            continue  # not a fallback geometry
        legal = [
            x
            for x in sw._structure_cands(default, sw.Release(sw.ALL_RELEASED))
            if sw.legality(x).ok
        ]
        best = min(legal, key=lambda x: sw.fallback_order(x, default))
        assert best.key() == g.key()
        key = sw.fallback_order(best, default)
        for x in legal:
            assert sw.fallback_order(x, default)[:4] >= key[:4]
        checked += 1
    return checked


# --------------------------------------------------------------------------- stages
def _timed(c, score, stage):
    return {**c.record(), "stage": stage, "status": "timed", "score": score}


def _write(out, sl, stage, recs, shard=(0, 1)):
    p = sw.shard_path(out, sl, stage, shard)
    for r in recs:
        sw.append_jsonl(p, r)


def test_stage_chaining_from_shard_records(tmp_path):
    sl = _sl("gfx942", 64)
    rel = sw.Release(frozenset({1, 2, 4, 6, 7}))
    geo = sw.geometry_stage(sl, rel).candidates
    # two shard files; ranked merges them (lower score first)
    _write(
        tmp_path,
        sl,
        "geometry",
        [_timed(c, 1.0 + i, "geometry") for i, c in enumerate(geo[::2])],
        (0, 2),
    )
    _write(tmp_path, sl, "geometry", [_timed(c, 1.5 + i, "geometry")
                                      for i, c in enumerate(geo[1::2])], (1, 2))  # fmt: skip
    ranked = sw.ranked(tmp_path, sl, "geometry")
    assert len(ranked) == len(geo) and ranked[0] == geo[0] and ranked[1] == geo[1]
    prior = {"geometry": ranked}
    st = sw.stage_candidates("structure", sl, rel=rel, prior=prior)
    geos = {tuple(sorted((k, c.k[k]) for k in sw.GEOMETRY_KNOBS)) for c in st}
    assert len(geos) == sw.STRUCTURE_FROM_GEOMETRIES
    reps, _ = sw.dedupe(st)
    prior["structure"] = reps
    sch = sw.stage_candidates("schedule", sl, rel=rel, prior=prior)
    allocs = {c.k["agpr_alloc"] for c in sch}
    assert allocs <= set(AGPR_ALLOCS)
    # at most the smallest legal reservation besides None / "backend" / (0, 0)
    assert len({a for a in allocs if isinstance(a, tuple) and a != (0, 0)}) <= 1
    prior["schedule"] = sw.dedupe(sch)[0]
    cg = sw.stage_candidates("codegen", sl, rel=rel, prior=prior)
    assert {c.k["scheduler_strategy"] for c in cg} == set(sw.CODEGEN_STRATEGIES)
    prior["codegen"] = cg
    sz = sw.stage_candidates("swizzle", sl, rel=rel, prior=prior)
    top = cg[: sw.KEEP["codegen"]]
    # no predictor tie on these buffers: each kept candidate, then its row-pad
    # variants (one buffer at a time), each with and without its lds_order
    # variant
    assert [c for c in sz if c in top] == top and sz[0] == top[0]
    for c in sz:
        if c in top:
            continue
        base = next(t for t in top if {k: v for k, v in t.k.items()
                                        if k not in ("lds_swizzle", "lds_order")}
                    == {k: v for k, v in c.k.items()
                        if k not in ("lds_swizzle", "lds_order")})  # fmt: skip
        moved = {k for k in set(c.k) | set(base.k) if c.k.get(k) != base.k.get(k)}
        assert moved and moved <= {"lds_swizzle", "lds_order"}, moved
        if "lds_swizzle" in moved:
            assert c.k["lds_swizzle"].count("=") == 1
        if "lds_order" in moved:
            assert c.k["lds_order"] == "phase_low"
    prior["swizzle"] = sz
    alt = sw.stage_candidates("alternate", sl, rel=rel, prior=prior)
    assert len(alt) == min(sw.ALTERNATE_STRUCTURES, len(reps))
    w = sz[0].k
    for c in alt:
        assert all(c.k[k] == w[k] for k in sw.SCHEDULE_KNOBS + sw.MINOR_KNOBS)
    prior["alternate"] = alt
    cells = sw.representative_cells(sl)
    rt = sw.stage_candidates("runtime", sl, rel=rel, prior=prior, cells=cells)
    assert {c.r["g_split"] for c in rt} == {1, 2, 4}  # divisors of G = 4
    assert {c.r["lb_order"] for c in rt} == {
        "natural",
        "reverse",
        "mirror",
        "interleave",
    }
    # mirror pairing: the paired decode, the device CU count as request hint
    for c in rt:
        paired = c.r["lb_order"] == "mirror"
        assert c.spec().lb_mirror is paired
        assert (c.r.get("num_cus") == sw.DEVICE_NUM_CUS) is paired
    assert {c.r["scale_placement"] for c in rt} == {
        "fold_ds",
        "dq_before_atomic",
        "convert",
    }
    assert all(c.spec().dkv_mode == ("atomic" if c.r["g_split"] > 1 else "direct")
               for c in rt)  # fmt: skip
    prior["runtime"] = rt
    conf = sw.stage_candidates("confirm", sl, rel=rel, prior=prior)
    assert conf == rt[: sw.KEEP["runtime"]]


def test_swizzle_ties_expand_with_the_predictor(monkeypatch):
    sl = _sl("gfx942", 64)
    c = sw.geometry_stage(sl, REL1).candidates[0]
    assert sw.swizzle_ties(c) == []
    monkeypatch.setattr(
        sw,
        "swizzle_ties",
        lambda x: [("dot", ("pad8", "xor")), ("kt_res", ("pad16", "xor")),
                   ("qt", ("none", "xor")), ("zz", ("a", "b"))],
    )  # fmt: skip
    out = sw.stage_candidates("swizzle", sl, rel=REL1, prior={"codegen": [c]})
    assert len(out) == 2**sw.MAX_TIED_BUFFERS
    assert {x.k["lds_swizzle"].count("=") for x in out} == {3}


def test_alternate_changed(tmp_path):
    sl = _sl("gfx942", 64)
    g = sw.geometry_stage(sl, REL1).candidates
    _write(tmp_path, sl, "swizzle", [_timed(g[0], 1.0, "swizzle")])
    _write(tmp_path, sl, "alternate", [_timed(g[0], 1.0, "alternate")])
    assert not sw.alternate_changed(tmp_path, sl)
    _write(tmp_path, sl, "alternate", [_timed(g[1], 0.5, "alternate")])
    assert sw.alternate_changed(tmp_path, sl)


def test_secondary_slice_recompiles_the_primary_top():
    prim = _sl("gfx942", 128)
    sec = sw.secondary_slice(prim, "fp16")
    assert sec.dtype == "fp16" and sec.tag().endswith("-fp16")
    top = sw.geometry_stage(prim, REL1).candidates[:3]
    out = sw.stage_candidates("schedule", sec, rel=REL1, prior={"primary": top})
    assert {c.slice for c in out} == {sec} and len(out) == 3
    with pytest.raises(ValueError):
        sw.secondary_slice(prim, "bf8")


# --------------------------------------------------------------------------- cells
def test_representative_cells_of_the_primary_slices():
    d128 = [c.id for c in sw.representative_cells(_sl("gfx942", 128))]
    assert d128 == [
        "d128_h32_kv8_b8_q1024_k1024_causal_tl_bf16_bshd",
        "d128_h32_kv8_b2_q4096_k4096_causal_tl_bf16_bshd",
        "d128_h32_kv8_b1_q8192_k8192_causal_tl_bf16_bshd",
    ]
    d64 = [c.id for c in sw.representative_cells(_sl("gfx950", 64))]
    assert d64 == ["d64_h32_kv8_b4_q2048_k2048_causal_tl_bf16_bshd"]
    assert len(sw.cohort_cells(_sl("gfx942", 128))) == 59
    assert len(sw.cohort_cells(_sl("gfx942", 64))) == 25


# --------------------------------------------------------------------------- counts / plan
def test_count_slice_bounds():
    sc = sw.count_slice(_sl("gfx942", 128), REL1)
    assert sc.geometries > 0 and sc.product >= sc.geometries - sc.fallback
    assert sc.stages["runtime"] == sc.stages["confirm"] == 0
    assert sc.stages["structure"] <= sw.STRUCTURE_FROM_GEOMETRIES * sc.structures_max


W = sw.WallInputs(
    t_compile=6.0, compile_speedup=10.0, t_run=0.5, t_attempt=2.0,
    t_reference=10.0, t_gate=1.0,
)  # fmt: skip


def test_plan_jobs_formula():
    # compile 1000 * 6 / 10 = 600 s; timing: 10 arms in one group on 3 cells x 6
    # attempts: (2 + 11 * 0.5 + 10 * 1) * 18 = 315 s, + 3 references = 30 s
    assert sw.plan_jobs(1000, 10, W, n_cells=3, attempts=6, job_cap_s=1200,
                        margin_s=200) == 1  # fmt: skip
    assert sw.plan_jobs(4000, 0, W, n_cells=3, attempts=6, job_cap_s=1200,
                        margin_s=200) == 3  # fmt: skip
    many = sw.plan_jobs(0, 120, W, n_cells=3, attempts=6, arms_per_attempt=12)
    assert many == 4  # 10 groups x 18 attempts x 19.5 s + 30 s = 3540 s


def test_wall_inputs_by_name():
    m = {"t_compile_serial": 2, "t_run": 0.1, "t_reference": 3}
    w = sw.WallInputs.from_mapping(m)
    assert w.compile_speedup == 1.0 and w.t_gate == 0.0
    with pytest.raises(KeyError):
        sw.WallInputs.from_mapping({"t_run": 1})


# --------------------------------------------------------------------------- scoring
def _summary(cid, rocke, bar, *, status="ok", key="k"):
    return {
        "id": cid,
        "bar_arm": "default|native",
        "arms": {
            f"rocke|{key}": {"status": status, "median_ms": rocke, "spread": 0.01,
                             "attempt_medians_ms": [rocke] * 5},
            "default|native": {"status": "ok", "median_ms": bar, "spread": 0.01,
                               "attempt_medians_ms": [bar] * 5},
        },  # fmt: skip
        "verdicts": {f"rocke|{key}": "pass"},
    }


def test_score_is_the_geometric_mean_ratio():
    s = sw.score_from_summaries("k", [_summary("a", 2.0, 4.0), _summary("b", 2.0, 1.0)])
    assert s["score"] == pytest.approx(1.0) and s["worst"] is None
    bad = sw.score_from_summaries(
        "k",
        [_summary("a", 2.0, 4.0), _summary("b", 2.0, 1.0, status="correctness_fail")],
    )
    assert bad["score"] is None and bad["worst"] == "correctness_fail"


# --------------------------------------------------------------------------- driver
class _Entry:
    def __init__(self, n):
        self.kernel_name, self.hsaco_sha, self.key = n, "sha", "key"
        self.hsaco = b"\x7fELF"


class _Res:
    def __init__(self, ok, err=None):
        self.ok, self.error, self.detail, self.cached = ok, err, "", False
        self.entry = _Entry("k") if ok else None


def test_driver_stage_records_tokens_and_resumes(tmp_path, monkeypatch):
    from benchmarks.common import attention_bwd_bench as bb
    from benchmarks.common import attention_bwd_compile as bc

    sl = _sl("gfx942", 128)
    cands = sw.geometry_stage(sl, REL1).candidates
    prune_key, fail_key = cands[0].key(), cands[1].key()

    def fake_batch(items, **kw):
        return [_Res(True) for _ in items]

    class Gate:
        def static_prune(self, arch, spec, hsaco):
            k = next(c.key() for c in cands if sw.legality(c).resolved == spec)
            return ("agpr_copy_traffic" if k == prune_key else None), {"r": 1}

    calls = []

    def fake_bench(argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        cfgs = json.loads(Path(argv[argv.index("--rocke-configs") + 1]).read_text())
        ids = argv[argv.index("--ids") + 1].split(",")
        calls.append(len(cfgs["configs"]))
        for cid in ids:
            arms = {"default|native": {"status": "ok", "median_ms": 2.0, "spread": 0.0,
                                       "attempt_medians_ms": [2.0] * 5}}  # fmt: skip
            for i, c in enumerate(cfgs["configs"]):
                st = "correctness_fail" if c["label"] == fail_key else "ok"
                arms[f"rocke|{c['label']}"] = {
                    "status": st, "median_ms": 1.0 + i, "spread": 0.0,
                    "attempt_medians_ms": [1.0 + i] * 5,
                }  # fmt: skip
            bb.append_jsonl(out / "cells.jsonl", {"id": cid, "complete": True,
                            "bar_arm": "default|native", "arms": arms})  # fmt: skip
        bb.append_jsonl(out / "env.jsonl", {"host": "n", "cu_count": 1})
        return 0

    monkeypatch.setattr(bc, "compile_batch", fake_batch)
    monkeypatch.setattr(bb, "main", fake_bench)

    class _TC:
        comgr, flavor = "c", "llvm22"

        def record(self):
            return {"comgr": "c"}

    drv = sw.Driver(sw.Options(out=tmp_path, arch="gfx942", release=REL1,
                               arms_per_attempt=5, gate="x:y"))  # fmt: skip
    drv._gate, drv._toolchain = Gate(), _TC()
    drv._cache = type("C", (), {"root": tmp_path / "cache"})()
    counts = drv.run_stage(sl, "geometry")
    assert counts["pruned"] == 1 and counts["correctness_fail"] == 1
    # geometries whose cheapest legal structure is split dQ (no G4) emit the
    # same kernel for every G4 grid: those duplicates are recorded, not timed
    _reps, dup = sw.dedupe(cands)
    assert counts["duplicate"] == len(dup)
    assert counts["timed"] == len(cands) - 2 - len(dup)
    assert calls and max(calls) <= 5
    recs = sw.read_records(tmp_path, sl, "geometry")
    assert recs[prune_key]["status"] == "pruned:agpr_copy_traffic"
    assert recs[fail_key]["status"] == "correctness_fail"
    timed = [r for r in recs.values() if r["status"] == "timed"]
    assert all(r["released"] == [1] and r["keep"] == 8 and r["env"] for r in timed)
    assert all(set(r["cells"]) == {c.id for c in sw.representative_cells(sl)}
               for r in timed)  # fmt: skip
    n = len(calls)
    again = drv.run_stage(sl, "geometry")
    assert len(calls) == n and again["skipped_done"] == len(cands)
    assert len(sw.ranked(tmp_path, sl, "geometry", 8)) == 8


def test_dry_run_stage_and_shards_partition(tmp_path):
    sl = _sl("gfx950", 64)
    seen = []
    for i in range(3):
        d = sw.Driver(sw.Options(out=tmp_path, arch="gfx950", release=REL1,
                                 shard=(i, 3), dry_run=True))  # fmt: skip
        d.run_stage(sl, "geometry")
        seen += [r["key"] for r in sw.read_records(tmp_path, sl, "geometry").values()]
    keys = {c.key() for c in sw.geometry_stage(sl, REL1).candidates}
    assert set(seen) == keys
    files = list(sw.stage_dir(tmp_path, sl, "geometry").glob("shard-*.jsonl"))
    assert len(files) == 3


# --------------------------------------------------------------------------- cli
def test_cli_count_prints_counts_only(capsys):
    argv = ["--count", "--arch", "gfx1151", "--family", "general", "--d", "128"]
    assert sw.main(argv) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith(
        "count arch=gfx1151 family=general d=128 released=1 "
    )
    assert out[0].endswith("not_yet_effective=")  # every knob's emission is built
    assert not re.search(r"\d+\.\d+", out[0])
    assert (
        sw.main(["--count", "--arch", "gfx942", "--family", "dense", "--d", "64"]) == 0
    )
    line = capsys.readouterr().out.strip()
    assert line.startswith(
        "count arch=gfx942 family=dense d=64 status=seeded_from_general"
    )
    assert line.endswith("stages=runtime,confirm")
    assert (
        sw.main(["--count", "--arch", "gfx942", "--family", "unified", "--d", "64"])
        == 0
    )
    assert capsys.readouterr().out.strip().endswith("status=no_validator")


def test_sweep_level_is_parsed_recorded_and_counted(tmp_path, capsys):
    assert sw.Release.parse("all", "full") == sw.Release(sw.ALL_RELEASED, "full")
    assert sw.Release.parse(None).level == "production"
    with pytest.raises(ValueError):
        sw.Release(frozenset({1}), "fast")
    argv = ["--count", "--arch", "gfx1151", "--family", "general", "--d", "64"]
    assert sw.main(argv + ["--level", "full"]) == 0
    line = capsys.readouterr().out.strip()
    assert " released=1 level=full " in line and " product_scope=legacy " in line
    sl = _sl("gfx950", 64)
    d = sw.Driver(sw.Options(out=tmp_path, arch="gfx950", dry_run=True,
                             release=sw.Release(frozenset({1}), "full")))  # fmt: skip
    d.run_stage(sl, "geometry")
    recs = sw.read_records(tmp_path, sl, "geometry").values()
    assert recs and all(r["level"] == "full" for r in recs)


def test_cli_count_plan_reads_constants_by_arch(tmp_path, capsys):
    consts = tmp_path / "c.json"
    consts.write_text(json.dumps({"per_arch": {"gfx942": {
        "t_compile_serial": 2.0, "compile_speedup": 8.0, "t_run": 0.2,
        "t_attempt": 3.0, "t_reference": 5.0, "t_gate": 0.5}}}))  # fmt: skip
    argv = ["--count", "--plan", "--constants", str(consts), "--arch", "gfx942",
            "--family", "general", "--d", "128"]  # fmt: skip
    assert sw.main(argv) == 0
    line = capsys.readouterr().out.strip()
    assert re.search(r" jobs_geometry=\d+ .* jobs_upper_bound=\d+$", line)
    assert not re.search(r"\d+\.\d+", line)
    assert "jobs_expected" not in line
    data = json.loads(consts.read_text())
    data["per_arch"]["gfx942"]["survival"] = 0.5
    consts.write_text(json.dumps(data))
    assert sw.main(argv) == 0
    line = capsys.readouterr().out.strip()
    up = int(re.search(r"jobs_upper_bound=(\d+)", line).group(1))
    exp = int(re.search(r"jobs_expected=(\d+)$", line).group(1))
    assert exp <= up
    consts.write_text(json.dumps({"per_arch": {}}))
    assert sw.main(argv) == 0
    assert capsys.readouterr().out.strip().endswith("jobs=unknown")


def test_cli_guards(tmp_path):
    inside = Path(sw.__file__).resolve().parent
    with pytest.raises(SystemExit):
        sw.main(["--stage", "geometry", "--dry-run", "--out-dir", str(inside)])
    with pytest.raises(SystemExit):  # a timed stage needs the gate
        sw.main(["--stage", "geometry", "--out-dir", str(tmp_path), "--arch",
                 "gfx942", "--d", "64"])  # fmt: skip
    with pytest.raises(SystemExit):
        sw.main(["--stage", "geometry", "--dry-run", "--out-dir", str(tmp_path),
                 "--shard", "3/3"])  # fmt: skip


def test_gate_has_no_environment_default(monkeypatch, tmp_path):
    """``--gate`` is explicit: an exported variable selects no gate, so a
    timed stage still refuses to run without it on the command line."""
    monkeypatch.setenv("ROCKE_BWD_PRUNE_GATE", "benchmarks.common.x:GATE")
    assert sw.make_parser().parse_args([]).gate is None
    args = sw.make_parser().parse_args(["--gate", "m:f"])
    assert args.gate == "m:f"
    with pytest.raises(SystemExit, match="pruning-set gate"):
        sw.main(["--stage", "geometry", "--out-dir", str(tmp_path), "--arch",
                 "gfx942", "--d", "64"])  # fmt: skip


def _fake_env(monkeypatch, compiled):
    """Stub compiler (records what it compiles) and timing harness (every
    configuration twice as fast as the bar on every cell)."""
    from benchmarks.common import attention_bwd_bench as bb
    from benchmarks.common import attention_bwd_compile as bc

    def fake_batch(items, **kw):
        compiled.extend(repr(s) for _, s in items)
        return [_Res(True) for _ in items]

    fake_bench = fake_bench_ok(bb)

    monkeypatch.setattr(bc, "compile_batch", fake_batch)
    monkeypatch.setattr(bb, "main", fake_bench)


class _NoPrune:
    def static_prune(self, arch, spec, hsaco):
        return None, {}


def _driver(tmp_path, shard=(0, 1), stage_rel=REL1):
    class _TC:
        def record(self):
            return {"comgr": "c"}

    d = sw.Driver(sw.Options(out=tmp_path, arch="x", release=stage_rel, shard=shard,
                             gate="x:y"))  # fmt: skip
    d._gate, d._toolchain = _NoPrune(), _TC()
    d._cache = type("C", (), {"root": tmp_path / "cache"})()
    return d


def test_duplicates_compile_once_across_shards(tmp_path, monkeypatch):
    sl = _sl("gfx942", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    group = [c, c.with_knobs(agpr_alloc=(0, 0)), c.with_knobs(waves_per_eu=None)]
    monkeypatch.setattr(sw, "stage_candidates", lambda *a, **k: list(group))
    compiled = []
    _fake_env(monkeypatch, compiled)
    for i in range(3):
        _driver(tmp_path, shard=(i, 3)).run_stage(sl, "geometry")
    assert len(compiled) == 1
    recs = sw.read_records(tmp_path, sl, "geometry")
    rep = min(x.key() for x in group)
    assert recs[rep]["status"] == "timed"
    assert all(recs[x.key()]["status"] == f"duplicate:{rep}" for x in group
               if x.key() != rep)  # fmt: skip


def test_compiler_crash_prone_candidates_compile_in_child_processes(
    tmp_path, monkeypatch
):
    """A candidate with a machine-scheduler strategy (the gfx942 comgr
    segfaults on one) compiles in process mode whatever the driver's compile
    mode, so a crash is its own record; the other candidates keep the
    driver's mode."""
    from benchmarks.common import attention_bwd_compile as bc

    sl = _sl("gfx942", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    group = [c, c.with_knobs(scheduler_strategy="iterative-ilp")]
    assert [sw.crash_isolated(sw.legality(x).resolved) for x in group] == [
        False,
        True,
    ]
    monkeypatch.setattr(sw, "stage_candidates", lambda *a, **k: list(group))
    compiled = []
    _fake_env(monkeypatch, compiled)
    fake, modes = bc.compile_batch, []

    def by_mode(items, **kw):
        modes.append((kw["mode"], [sw.crash_isolated(s) for _, s in items]))
        return fake(items, **kw)

    monkeypatch.setattr(bc, "compile_batch", by_mode)
    _driver(tmp_path).run_stage(sl, "geometry")
    assert sorted(modes) == [("process", [True]), ("thread", [False])]
    recs = sw.read_records(tmp_path, sl, "geometry")
    assert all(recs[x.key()]["status"] == "timed" for x in group)


def test_confirm_shards_the_cohort_cells(tmp_path, monkeypatch):
    sl = _sl("gfx942", 64)
    fin = sw.geometry_stage(sl, REL1).candidates[:2]
    _write(tmp_path, sl, "runtime", [_timed(c, 1.0 + i, "runtime")
                                     for i, c in enumerate(fin)])  # fmt: skip
    _fake_env(monkeypatch, [])
    for i in range(2):
        counts = _driver(tmp_path, shard=(i, 2)).run_stage(sl, "confirm")
        assert counts["timed"] == 2
    cohort = sw.cohort_cells(sl)
    res = sw.confirm_verdicts(tmp_path, sl)
    assert set(res) == {c.key() for c in fin}
    assert all(r["cells"] == len(cohort) and r["verdicts"] == {"pass": len(cohort)}
               for r in res.values())  # fmt: skip
    again = _driver(tmp_path, shard=(0, 2)).run_stage(sl, "confirm")
    assert again["skipped_done"] == 2 and again["timed"] == 0


def test_timing_removes_the_reference_cache(tmp_path, monkeypatch):
    sl = _sl("gfx942", 64)
    _fake_env(monkeypatch, [])
    d = _driver(tmp_path)
    ref = d._ref_dir(sl, "geometry")
    ref.mkdir(parents=True)
    (ref / "ref-x.pt").write_bytes(b"x")
    d.run_stage(sl, "geometry")
    assert not ref.exists()
    assert sw.read_records(tmp_path, sl, "geometry")


def test_declaration_twins_compile_once_and_keep_the_declaration():
    """Under the attr route ``s_in_agpr`` only declares the plan: the two twins
    are one kernel, and the representative is the declaring twin (its rule-6
    budget admits the S / dP / dQ reads the backend issues)."""
    sl = _sl("gfx950", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl)).with_knobs(
        waves=4, block_m=32, block_n=256, warp_grid_g4=(1, 4),
        warp_grid_g02=(1, 4), warp_grid_g13=(4, 1), agpr_alloc=(256, 256),
    )  # fmt: skip
    t = c.with_knobs(s_in_agpr=True)
    f = c.with_knobs(s_in_agpr=False)
    assert sw.legality(t).ok and sw.legality(f).ok, (sw.legality(t), sw.legality(f))
    assert sw.effective_key(t) == sw.effective_key(f)
    reps, dup = sw.dedupe([f, t])
    assert [r.key() for r in reps] == [t.key()] and dup == {f.key(): t.key()}


# --------------------------------------------------------------------------- dense family
ALL = sw.Release(sw.ALL_RELEASED)


def _dsl(arch="gfx950", d=128):
    return sw.primary_slice(arch, "dense", d)


def _general_seeds(arch="gfx950", d=128):
    """Two general runtime-stage survivors: the default tile on the register
    ring, and the same with a head split."""
    gsl = _sl(arch, d)
    k = dict(sw.default_knobs(gsl), ring_depth=2)
    return [
        sw.Candidate.make(gsl, k, {"g_split": 1, "lb_order": "natural"}),
        sw.Candidate.make(gsl, k, {"g_split": 2, "lb_order": "reverse"}),
    ]


def test_dense_is_sweepable_and_seeds_from_the_general_survivors():
    assert "dense" in sw.SWEEPABLE_FAMILIES
    sl = _dsl()
    seeds = _general_seeds()
    mirror = seeds[0].with_runtime(lb_order="mirror", num_cus=sw.DEVICE_NUM_CUS)
    prior = {"general:runtime": seeds + [mirror]}
    cands = sw.stage_candidates("runtime", sl, rel=ALL, prior=prior)
    assert cands and all(c.slice == sl for c in cands)
    assert all(c.r["route"] == "dense" and "head_pack" not in c.k for c in cands)
    # the mirror seed has no dense counterpart
    assert not any(c.r.get("lb_order") == "mirror" for c in cands)
    # the body knobs and runtime values are the seeds'
    body = {k: v for k, v in seeds[0].k.items() if k != "head_pack"}
    for c in cands:
        assert {k: c.k[k] for k in body if k not in sw.DENSE_RUNTIME_AXES} == {
            k: v for k, v in body.items() if k not in sw.DENSE_RUNTIME_AXES
        }
        assert c.r["g_split"] in (1, 2)
    # every dense-only knob value appears; persistent points carry the CU hint
    for name in ("oob", "epilogue", "edge_tiles", "ws_layout", "persistent"):
        vals = {c.k.get(name) for c in cands}
        assert len(vals) >= 2, name
    assert {c.k.get("persist_decode") for c in cands} >= {"kv_tile_major", "hkv_major"}
    assert all(
        c.r.get("num_cus") == sw.DEVICE_NUM_CUS for c in cands if c.k.get("persistent")
    )
    # the dense grid remap rides on xcd_chunk (non-persistent points only)
    xcd = [c for c in cands if c.r.get("xcd_chunk")]
    assert xcd and all(not c.k.get("persistent") and c.r["num_xcds"] for c in xcd)
    # other stages of a dense slice have no candidates (no body funnel)
    assert sw.stage_candidates("geometry", sl, rel=BUILT, prior=prior) == []
    # before the general runtime stage: its pre-runtime top x runtime values
    pre = sw.stage_candidates(
        "runtime", sl, rel=BUILT, prior={"general:codegen": [seeds[0]]},
        cells=sw.representative_cells(_sl()),
    )  # fmt: skip
    assert len({c.r.get("g_split") for c in pre}) > 1


def test_dense_legality_keys_and_refusals():
    sl = _dsl()
    seed = _general_seeds()[0]
    d = sw.Candidate.make(sl, seed.k, {**seed.r, "route": "dense"})
    assert sw.legality(d).ok
    # a dense candidate is keyed apart from its general twin and by its
    # dense-only knobs
    assert d.key() != seed.key()
    assert sw.effective_key(d) != sw.effective_key(seed)
    clamp = d.with_knobs(oob="clamp")
    assert sw.legality(clamp).ok
    assert sw.effective_key(clamp) != sw.effective_key(d)
    # not built on the dense shell: the mirror pass (split dQ is built)
    assert sw.legality(d.with_runtime(lb_order="mirror")).rule == "not_built"
    assert sw.legality(d.with_knobs(dq_mode="split")).ok
    # head_pack = False rides along and is dropped from the plan's knobs
    assert "head_pack" not in d.tuning_knobs()
    # dense-only knob couplings come from the dense spec
    bad = d.with_knobs(persist_decode="hkv_major")
    assert not sw.legality(bad).ok
    assert sw.legality(d.with_knobs(oob="none")).rule == "catalog"
    # the general family keeps its keys (no dense field leaks into them)
    assert sw.Candidate.make(_sl("gfx950"), seed.k, seed.r).key() == seed.key()


def test_dense_cells_skip_head_packed_and_thd_cells():
    from benchmarks.common import attention_bwd_bench as bb

    sl = _dsl("gfx950", 128)
    cells = sw.cohort_cells(sl)
    got = sw.dense_cells(sl, cells, 64)
    assert got and len(got) < len(cells)
    assert not any(c.is_thd or sw.head_packed(c, 64) for c in got)
    one = bb.Cell("x", 128, 32, 8, 4, 1, 4096, "causal_tl", "bf16", "bshd")
    assert sw.head_packed(one, 64) and not sw.dense_cells(sl, [one], 64)
    mha = bb.Cell("y", 128, 8, 8, 4, 1, 4096, "causal_tl", "bf16", "bshd")
    assert not sw.head_packed(mha, 64)


def test_dense_plan_of_a_candidate_on_a_cell():
    sl = _dsl("gfx950", 128)
    seed = _general_seeds()[1]  # g_split = 2
    d = sw.Candidate.make(sl, seed.k, {**seed.r, "route": "dense"})
    cell = sw.dense_cells(sl, sw.representative_cells(sl), 128)[0]
    plan = sw.dense_plan(d, cell)
    assert plan.route == "dense" and plan.specs["dense"].dkv_mode == "atomic"
    pers = d.with_knobs(persistent=True).with_runtime(num_cus=sw.DEVICE_NUM_CUS)
    assert sw.dense_plan(pers, cell).specs["dense"].persistent
    v = sw.dense_tuned_value(pers)
    assert v["persistent"] and v["g_split"] == 2 and "route" not in v
    assert "num_cus" not in v and "head_pack" not in v


def _consts(tmp_path):
    p = tmp_path / "gc.json"
    p.write_text(json.dumps({"BAND_K": 2.0, "TIE_EPS": 0.01, "RERUN_MAX": 1,
                             "NOISY_SPREAD": 0.1, "ATTEMPTS_NOISY": 9,
                             "CLIFF_BAND": 0.2}))  # fmt: skip
    return p


def _arm(t, spread=0.0):
    return {"status": "ok", "median_ms": t, "spread": spread,
            "attempt_medians_ms": [t] * 5}  # fmt: skip


def test_dense_vs_general_is_a_paired_two_way_verdict(tmp_path):
    from benchmarks.common import attention_bwd_bench as bb

    consts = bb.load_gate_constants(_consts(tmp_path))
    s = {
        "arms": {"rocke|d": _arm(0.9), "rocke|g": _arm(1.0)},
        "aa_control": "ok",
        "aa_controls": {"rocke|d": "ok", "rocke|g": "ok"},
    }
    r = sw.dense_vs_general(s, "d", "g", consts, gated=True)
    assert r["win"] and r["verdict"] == "pass" and r["reverse"] == "fail"
    assert r["aa"] == "ok" and r["gated"]
    # the A/A of both arms: one arm's twin does not bound the other's noise
    for per, want in (
        ({"rocke|d": "fail", "rocke|g": "ok"}, "fail"),
        ({"rocke|d": "ok", "rocke|g": "incomplete"}, "incomplete"),
        ({"rocke|g": "ok"}, None),  # the dense arm had no twin
    ):
        got = sw.dense_vs_general({**s, "aa_controls": per}, "d", "g", consts,
                                  gated=True)  # fmt: skip
        assert got["aa"] == want, per
    old = {k: v for k, v in s.items() if k != "aa_controls"}  # one twin only
    assert sw.dense_vs_general(old, "d", "g", consts, gated=True)["aa"] is None
    tie = {"arms": {"rocke|d": _arm(1.0), "rocke|g": _arm(1.0)}}
    assert not sw.dense_vs_general(tie, "d", "g", consts, gated=True)["win"]
    slow = {"arms": {"rocke|d": _arm(1.1), "rocke|g": _arm(1.0)}}
    assert sw.dense_vs_general(slow, "d", "g", consts, gated=True)["verdict"] == "fail"
    gone = {"arms": {"rocke|g": _arm(1.0)}}
    assert (
        sw.dense_vs_general(gone, "d", "g", consts, gated=True)["verdict"] == "untimed"
    )
    assert (
        sw.dense_vs_general(s, "d", "g", None, gated=True)["verdict"] == "no_constants"
    )


class _Seen(list):
    """The stages compiled (and ``aa``: the --aa-control all flag of every
    bench call)."""


def _dense_bench(monkeypatch, dense_ms, aa="ok"):
    """Stub compiler and timing harness: every dense arm at ``dense_ms``, every
    general arm at 1.0, the A/A control ``aa``."""
    from benchmarks.common import attention_bwd_bench as bb
    from benchmarks.common import attention_bwd_compile as bc

    seen = _Seen()
    seen_aa = []

    def fake_batch(items, **kw):
        seen.extend(st for st, _ in items)
        return [_Res(True) for _ in items]

    def fake_bench(argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        cfgs = json.loads(Path(argv[argv.index("--rocke-configs") + 1]).read_text())
        all_twins = "--aa-control" in argv and (
            argv[argv.index("--aa-control") + 1] == "all"
        )
        seen_aa.append(all_twins)
        for cid in argv[argv.index("--ids") + 1].split(","):
            arms = {"default|native": _arm(2.0)}
            for c in cfgs["configs"]:
                dense = c["runtime"].get("route") == "dense"
                arms[f"rocke|{c['label']}"] = _arm(dense_ms if dense else 1.0)
            labels = [c["label"] for c in cfgs["configs"]]
            per = {f"rocke|{x}": aa for x in (labels if all_twins else labels[:1])}
            bb.append_jsonl(out / "cells.jsonl", {
                "id": cid, "complete": True, "bar_arm": "default|native", "arms": arms,
                "aa_control": aa, "aa_controls": per,
                "verdicts": {f"rocke|{c['label']}": "pass" for c in cfgs["configs"]},
            })  # fmt: skip
        bb.append_jsonl(out / "env.jsonl", {"host": "n"})
        return 0

    monkeypatch.setattr(bc, "compile_batch", fake_batch)
    monkeypatch.setattr(bb, "main", fake_bench)
    seen.aa = seen_aa  # the --aa-control all flag of every bench call
    return seen


def _dense_driver(tmp_path):
    d = _driver(tmp_path)
    d.o = sw.Options(out=tmp_path, arch="gfx950", release=BUILT, gate="x:y",
                     gate_constants=str(_consts(tmp_path)))  # fmt: skip
    return d


def _seed_general(tmp_path, arch="gfx950", d=128):
    gsl = _sl(arch, d)
    seeds = _general_seeds(arch, d)
    _write(tmp_path, gsl, "runtime", [_timed(c, 0.5 + i, "runtime")
                                      for i, c in enumerate(seeds)])  # fmt: skip
    return seeds


def test_dense_runtime_stage_compiles_dense_kernels_per_cell(tmp_path, monkeypatch):
    sl = _dsl()
    _seed_general(tmp_path)
    seen = _dense_bench(monkeypatch, 0.8)
    drv = _dense_driver(tmp_path)
    assert drv.run_stage(sl, "geometry") == {"skipped": "dense_seeds_from_general"}
    counts = drv.run_stage(sl, "runtime")
    assert counts["timed"] > 0 and set(seen) == {"dense"}
    recs = [r for r in sw.read_records(tmp_path, sl, "runtime").values()
            if r["status"] == "timed"]  # fmt: skip
    cells = sw.dense_cells(sl, sw.representative_cells(sl), 128)
    assert recs and all(len(r["kernel"]) <= len(cells) for r in recs)
    assert all(r["runtime"]["route"] == "dense" for r in recs)


@pytest.mark.parametrize(
    "dense_ms,aa,gated,want",
    [(0.8, "ok", True, True), (1.0, "ok", True, False), (0.8, "fail", True, False),
     (0.8, "ok", False, False)],
)  # fmt: skip
def test_dense_confirm_and_table_entries(tmp_path, monkeypatch, dense_ms, aa,
                                         gated, want):  # fmt: skip
    """confirm times the best dense candidate against the general winner
    (general first, one group); a key enters the table only on a paired win
    on every cell with a clean A/A and the gate on."""
    sl = _dsl()
    seeds = _seed_general(tmp_path)
    best = sw.Candidate.make(sl, seeds[0].k, {**seeds[0].r, "route": "dense"})
    _write(tmp_path, sl, "runtime", [_timed(best, 0.4, "runtime")])
    cands = sw.stage_candidates("confirm", sl, rel=BUILT, prior=_dense_driver(
        tmp_path).prior(sl))  # fmt: skip
    assert [c.slice.family for c in cands] == ["general", "dense"]
    assert cands[0] == seeds[0] and cands[1] == best
    seen = _dense_bench(monkeypatch, dense_ms, aa)
    drv = _dense_driver(tmp_path)
    if not gated:
        drv.o = replace_opts(drv.o, gate=None)
    drv.run_stage(sl, "confirm")
    assert seen.aa and all(seen.aa)  # confirm: a twin of every arm
    recs = sw.read_records(tmp_path, sl, "confirm")
    dense = [r for r in recs.values() if r["slice"]["family"] == "dense"]
    assert dense and all("vs_general" in r for r in dense)
    entries = sw.dense_table_entries(tmp_path, sl)
    if not want:
        assert entries == []
        return
    assert entries and all(e["arch"] == "gfx950" for e in entries)
    for e in entries:
        assert e["knobs"]["ring_depth"] == 2 and "route" not in e["knobs"]
        assert len(e["key"]) == 9
    # CLI: the entries are written outside the source tree
    dest = tmp_path / "table.json"
    argv = ["--arch", "gfx950", "--family", "dense", "--d", "128", "--out-dir",
            str(tmp_path), "--emit-dense-table", str(dest)]  # fmt: skip
    assert sw.main(argv) == 0
    assert json.loads(dest.read_text())["entries"] == json.loads(json.dumps(entries))


def replace_opts(o, **kw):
    import dataclasses

    return dataclasses.replace(o, **kw)
