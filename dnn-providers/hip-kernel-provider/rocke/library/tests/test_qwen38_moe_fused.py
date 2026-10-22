# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fused Qwen3.8 MoE (``kernels.gfx950.qwen38_mono.moe_fused``) and its stage
functions (``stage_moe_fused``). CPU only.

* plan: every gate_up unit (slot x 32-column block) and every down unit
  (slot x 16-column n-block) of every slot, routed and shared, is assigned to
  exactly one CTA for every union size U (rows 1: 10; rows 2..4: 10..10 R),
  every IOD split and both maps; the routed slots split by IOD stay whole; the
  ``line`` map's busiest CTA carries the least a CTA can (``max(ceil(W / N),
  16 ceil(G / N))`` down units: a CTA holding ceil(G / N) gate_up units
  cannot carry less) and, when that granularity does not bind, per-CTA bytes
  differ by at most one down unit; the static bounds (chains and down units a
  wave, slots a CTA) hold every assignment.
* LDS: the fused arena fits the mono arena of every variant.
* builds: every variant and the knob values lower at llvm22 / llvm23; the m4 /
  m5 functions are empty; the sidecars pair the three (ATOM refuses a partial
  link); knob validation refuses bad values.
* the split port (``moe``) is byte-identical after the fused track's refactors
  (``gu_epilogue_entry``, ``combine_tile``): sha256 of its tuned knob sets'
  IR, pinned before the refactor.

GPU (bitwise vs the mono, in situ): ``hackathon-prep/qwen38/moe-fused/fused_ab.py``.
"""

import hashlib
import json
import os

import pytest
from kernels.gfx950.qwen38_mono import abi as A
from kernels.gfx950.qwen38_mono import moe as MO
from kernels.gfx950.qwen38_mono import moe_fused as MF
from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVORS = ("llvm22", "llvm23")
# the mono arena's int32 words per (rows, seq) (ATOM layout.for_rows, pinned)
MONO_LDS = {(1, False): 22640, (2, False): 32256, (2, True): 32256, (3, False): 32384, (3, True): 32384,
            (4, False): 38912, (4, True): 38912}  # fmt: skip

PLAN_KNOBS = [
    {},
    {"iod_split": 0},
    {"map": "groups"},
    {"map": "groups", "iod_split": 0},
    {"sh_ctas": 8, "sh_base": 160},
    {"sh_ctas": 24, "sh_base": 150},
    {"sh_ctas": 13, "sh_base": 3},
    {"line_order": "router_dn"},
    {"line_order": "router_dn", "map": "groups"},
    {"line_order": "router_dn", "iod_split": 0, "sh_ctas": 13, "sh_base": 3},
    {"sh_join": 4},
    {"sh_join": 8, "map": "groups"},
    {"sh_join": 1, "iod_split": 0, "sh_ctas": 24, "sh_base": 150},
    {"sh_join": 6, "line_order": "router_dn", "sh_ctas": 13, "sh_base": 3},
]


def _u_values(rows):
    return [A.TOP_K] if rows == 1 else list(range(A.TOP_K, A.TOP_K * rows + 1))


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("knobs", PLAN_KNOBS, ids=lambda k: json.dumps(k, sort_keys=True))
def test_plan_covers_every_unit_once(rows, knobs):
    plan = MF.FusedPlan(rows, A.EXPERTS, MF.FusedKnobs(**knobs))
    plan.validate()
    bd = plan.bounds()
    for U in _u_values(rows):
        asg = plan.assignment(U)
        gu, dn = [], []
        for _kind, g, d, jn in asg.values():
            gu += g
            dn += d + jn
            assert len(g) <= bd["gu"] and len(d) <= bd["dn"] and len(jn) <= bd["jn"]
            assert len({s for s, _ in d}) <= max(bd["sl"], 1)
            assert len({s for s, _ in jn}) <= max(bd["jsl"], 1)
            assert not jn or plan.knobs.sh_join
        want_gu = [(s, cb) for s in range(-1, U) for cb in range(MF.GU_UNITS)]
        want_dn = [(s, n) for s in range(-1, U) for n in range(MF.DN_UNITS)]
        assert sorted(gu) == sorted(want_gu), U
        assert sorted(dn) == sorted(want_dn), U
        # the shared expert only on the shared group, routed slots only on routed CTAs (or joined)
        for bid, (kind, g, d, jn) in asg.items():
            assert all((s == -1) == (kind == "sh") for s, _ in g + d), bid
            assert all(s >= 0 for s, _ in jn) and (not jn or kind == "sh"), bid
    # a CTA's static wave slots hold its units
    assert -(-bd["gu"] // 2) == plan.j <= 2
    assert plan.dnj * A.WAVES >= bd["dn"] + 4
    assert plan.jnj * A.WAVES >= bd["jn"]


@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("split", [0, 1])
def test_line_map_balance(rows, split):
    plan = MF.FusedPlan(rows, A.EXPERTS, MF.FusedKnobs(iod_split=split))
    for U in _u_values(rows):
        asg = plan.assignment(U)
        for li, (lo, ul) in enumerate(plan.line_slots(U)):
            ctas = [b for b in range(A.GRID) if plan.role(b)[0] == "rt" and plan.role(b)[1] == li]
            n, g, w = len(ctas), MF.GU_UNITS * ul, (MF.GU_WEIGHT * MF.GU_UNITS + MF.DN_UNITS) * ul
            load = [MF.GU_WEIGHT * len(asg[b][1]) + len(asg[b][2]) for b in ctas]
            assert max(load) <= max(-(-w // n), MF.GU_WEIGHT * (-(-g // n))), (U, li)
            if MF.GU_WEIGHT * (-(-g // n)) <= w // n:
                assert max(load) - min(load) <= 1, (U, li)
            # the line's slots: whole slots, contiguous
            slots = {s for b in ctas for s, _ in asg[b][1] + asg[b][2]}
            assert slots == set(range(lo, lo + ul))
            if split:
                assert all((b % 8) // 4 == li for b in ctas)


@pytest.mark.parametrize("rows", [2, 4])
def test_router_dn_order(rows):
    """``line_order=router_dn``: the router (M3) CTAs sit at the end of their line and carry no more gate_up
    units than any other CTA of the line; the per-CTA load bound of the line map still holds."""
    plan = MF.FusedPlan(rows, A.EXPERTS, MF.FusedKnobs(line_order="router_dn"))
    mp = plan.moe
    routers = {(mp.base("m3") + i) % A.GRID for i in range(mp.m3_tasks)}
    for U in _u_values(rows):
        asg = plan.assignment(U)
        for li, (_lo, ul) in enumerate(plan.line_slots(U)):
            ctas = [b for b in range(A.GRID) if plan.role(b)[0] == "rt" and plan.role(b)[1] == li]
            n = len(ctas)
            pos = {b: plan.role(b)[2] for b in ctas}
            assert sorted(pos.values()) == list(range(n))
            rpos = sorted(pos[b] for b in ctas if b in routers)
            assert rpos == list(range(n - len(rpos), n)), (U, li)
            gmax_r = max((len(asg[b][1]) for b in ctas if b in routers), default=0)
            gmin_o = min(len(asg[b][1]) for b in ctas if b not in routers)
            assert gmax_r <= gmin_o, (U, li)
            g, w = MF.GU_UNITS * ul, (MF.GU_WEIGHT * MF.GU_UNITS + MF.DN_UNITS) * ul
            load = [MF.GU_WEIGHT * len(asg[b][1]) + len(asg[b][2]) for b in ctas]
            assert max(load) <= max(-(-w // n), MF.GU_WEIGHT * (-(-g // n))), (U, li)


@pytest.mark.parametrize("w", [1, 4, 8])
def test_sh_join_weights(w):
    """``sh_join``: a joined shared CTA's routed down units are ~ w / 8 of a routed CTA's byte share."""
    plan = MF.FusedPlan(4, A.EXPERTS, MF.FusedKnobs(sh_join=w))
    for U in (10, 28, 40):
        asg = plan.assignment(U)
        rt = [MF.GU_WEIGHT * len(g) + len(d) for k, g, d, _j in asg.values() if k == "rt"]
        jn = [len(j) for k, _g, _d, j in asg.values() if k == "sh"]
        share = sum(rt) / len(rt)
        assert all(abs(x - share * w / 8) <= 2 + MF.GU_WEIGHT for x in jn), (U, share, jn)


def test_iod_split_deals_by_cta_count():
    plan = MF.FusedPlan(4, A.EXPERTS, MF.FusedKnobs())
    assert plan.lines_n == (120, 120)
    for U in range(10, 41):
        (lo0, u0), (lo1, u1) = plan.line_slots(U)
        assert lo0 == 0 and lo1 == u0 and u0 + u1 == U and abs(u0 - u1) <= 1
    odd = MF.FusedPlan(4, A.EXPERTS, MF.FusedKnobs(sh_ctas=13, sh_base=3))
    assert sum(odd.lines_n) == A.GRID - 13 and odd.lines_n[0] != odd.lines_n[1]


@pytest.mark.parametrize("rows,seq", sorted(MONO_LDS))
def test_lds_fits_mono_arena(rows, seq):
    plan = MF.FusedPlan(rows, A.EXPERTS, MF.FusedKnobs())
    _off, stride, words = MF.lds_layout(plan)
    assert words <= MONO_LDS[(rows, seq)]
    # every wave's chain slots write gred at unit k < 2 J (k * 64 + 64 words), valid or not
    assert stride["gred"] >= 2 * plan.j * 64
    assert stride["dnA"] >= max(plan.bounds()["sl"], 1) * A.DN_ROW
    assert words <= MO.LDS_LIMIT_WORDS


def test_spec_history_in_unused_tails():
    """spec=prev histories: 64 bytes a layer, inside the unused tail of a small region's 256-byte rounding of
    rows 0..3, disjoint, 16-byte aligned, never overlapping any region's used pairs."""
    used = []
    for row in range(A.MAX_ROWS):
        for name, pairs in A.REGIONS.items():
            o = A.SC[name] + row * A.SCRATCH_BYTES
            used.append((o, o + pairs * A.PAIR))
    spans = []
    for layer in range(MF.HIST_LAYERS):
        h = MF.hist_offset(layer)
        assert h % 16 == 0
        spans.append((h, h + 64))
        assert not any(a < h + 64 and h < b for a, b in used), layer
    spans.sort()
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert max(b for _a, b in spans) <= A.MAX_ROWS * A.SCRATCH_BYTES
    with pytest.raises(ValueError):
        MF.hist_offset(MF.HIST_LAYERS)


def test_combine_counters_in_unused_tails():
    """combine=last counters: one int32 per hidden tile, in the unused tail of a region's 256-byte rounding,
    disjoint from every region's used pairs and from the spec=prev histories."""
    used = []
    for row in range(A.MAX_ROWS):
        for name, pairs in A.REGIONS.items():
            o = A.SC[name] + row * A.SCRATCH_BYTES
            used.append((o, o + pairs * A.PAIR))
    hist = [(MF.hist_offset(x), MF.hist_offset(x) + 64) for x in range(MF.HIST_LAYERS)]
    offs = [MF.counter_offset(t) for t in range(MF.DN_UNITS)]
    assert len(set(offs)) == MF.DN_UNITS and all(o % 4 == 0 for o in offs)
    for o in offs:
        assert not any(a <= o < b for a, b in used + hist), o
        r = A.SC[MF.CTR_REGION] + (o - A.SC[MF.CTR_REGION]) // A.SCRATCH_BYTES * A.SCRATCH_BYTES
        assert o + 4 <= r + (A.REGIONS[MF.CTR_REGION] * A.PAIR + 255) // 256 * 256
    with pytest.raises(ValueError):
        MF.counter_offset(MF.DN_UNITS)


def test_numa_header_matches_atom():
    """The NUMA header offset the fused body assumes is ATOM's (``numa.header_offset``)."""
    pytest.importorskip("flydsl")
    numa = pytest.importorskip("atom.models.qwen4_exp_mono.kernels.numa")
    assert MF.NUMA_HDR == numa.header_offset()
    assert MF.NUMA_PAGE == numa.PAGE and MF.NUMA_CHUNK == numa.CHUNK


def test_private_tag_is_xor():
    """ys / stamps tag: launch tag ^ 2^31, never the launch tag (any step count); the xtag_or mutant keeps the
    old OR, which equals the launch tag once bit 31 is set (step >= 2^25)."""

    def ir(knobs):
        return _lower_kernel_to_llvm_python(MF.build_fused_suffix(4, MF.FusedKnobs(**knobs)), llvm_flavor="llvm23",
                                            arch="gfx950")  # fmt: skip

    x, o = ir({}), ir({"mutate": "xtag_or"})
    assert "xor i32" in x and x != o
    for step in (0, 1, 2**25 - 1, 2**25, 2**26 - 1, 2**26 + 3):
        for layer in (0, 47):
            tag = (step * A.LAYER_SLOTS + layer + 1) & 0xFFFFFFFF
            assert tag ^ 0x80000000 != tag
            assert (tag | 0x80000000 == tag) == (step % 2**26 >= 2**25)


def test_private_regions_fit_ss():
    ss_bytes = A.REGIONS["ss"] * A.PAIR
    assert MF.YS_BYTES <= ss_bytes
    assert MF.STAMP_OFF + A.GRID * MF.STAMP_PTS * 8 <= ss_bytes
    assert len(MF.PHASES) <= MF.STAMP_PTS


def test_knob_refusals():
    for bad in ({"map": "x"}, {"iod_split": 2}, {"sh_ctas": 0}, {"sh_base": 250}, {"dn_issue": "x"},
                {"dn_inflight": 9}, {"extras": "x"}, {"tw_at": "x"}, {"slots": "x"}, {"gu_interleave": False},
                {"mutate": "x"}, {"w_cpol": 4}, {"route_gate": "x"}, {"line_impl": "x"}, {"guard": "x"},
                {"xchg_pre": "x"}, {"spec": "x"}, {"spec_loads": 0}, {"spec_keep": 11},
                {"combine": "x"}, {"combine_base": -2}, {"numa": 2}, {"lg_copy": "x"}, {"xchg_home": "x"},
                {"numa_hdr": 5}, {"line_order": "x"}, {"sh_join": 9}, {"sh_join": 2, "combine": "sh"},
                {"dn_lds": 0}, {"dn_issue": "x"}, {"slots": "x"}, {"ys_home": "x"}):
        with pytest.raises(ValueError):
            MF.FusedKnobs(**bad)
    with pytest.raises(ValueError, match="arith"):
        MF.FusedPlan(4, A.EXPERTS, MF.FusedKnobs(slots="arith")).validate()


BUILD_KNOBS = [
    {},
    {"iod_split": 0, "map": "groups"},
    {"dn_issue": "after_epi", "tw_at": "route", "extras": "late"},
    {"w_null": True, "phase_stamps": True},
    {"mutate": "combine_swap"},
    {"gu_apipe": 0, "gu_interleave": False, "a_pre": False, "w_cpol": 0, "mailbox": {"impl": "asm"}},
    {"route_gate": "pre_poll", "route_pause": "s_sleep1", "dn_inflight": 0},
    {"dn_issue": "after_gu", "tw_at": "epi", "line_impl": "const"},
    {"dn_issue": "pipe", "dn_inflight": 0},
    {"guard": "loads", "xchg_pre": "none", "line_impl": "const", "map": "groups"},
    {"guard": "loads", "dn_inflight": 2},
    {"spec": "prev", "guard": "loads", "xchg_pre": "none"},
    {"spec": "prev", "spec_keep": 4, "spec_loads": 8},
    {"combine": "sh", "guard": "loads"},
    {"combine_base": -1},
    {"combine": "sh", "mutate": "combine_swap", "sh_ctas": 13, "sh_base": 3},
    {"line_impl": "const", "sh_ctas": 13, "sh_base": 3},
    {"numa": 1},
    {"lg_copy": "numa2", "xchg_home": "numa2", "route_gate": "pre_poll"},
    {"numa": 1, "lg_copy": "numa2", "xchg_home": "numa2", "router_knobs": "mono", "numa_hdr": 1 << 24},
    {"line_order": "router_dn", "line_impl": "const"},
    {"sh_join": 4, "line_order": "router_dn", "guard": "loads"},
    {"sh_join": 8, "map": "groups", "line_impl": "const", "sh_ctas": 13, "sh_base": 3},
    {"combine": "last"},
    {"combine": "last", "sh_join": 3, "slots": "wave", "mutate": "combine_swap"},
    {"slots": "wave", "route_gate": "pre_poll"},
    {"dn_issue": "lds", "dn_lds": 1},
    {"dn_issue": "lds", "dn_lds": 1, "guard": "loads", "w_null": True},
    {"mutate": "xtag_or"},
    {"ys_home": "numa2", "combine": "last"},
    {"ys_home": "numa2", "combine": "sh", "numa": 1},
]


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("knobs", BUILD_KNOBS, ids=lambda k: json.dumps(k, sort_keys=True))
def test_suffix_lowers(flavor, rows, knobs):
    kn = SMF.knobs_from_json(knobs)
    for router in (True, False):
        k = MF.build_fused_suffix(rows, MF.FusedKnobs(**{**kn.__dict__, "router": router}))
        ir = _lower_kernel_to_llvm_python(k, llvm_flavor=flavor, arch="gfx950")
        assert "define amdgpu_kernel" in ir
        assert ("llvm.amdgcn.s.memrealtime" in ir or "s_memrealtime" in ir) == kn.phase_stamps


def test_new_knobs_change_ir_only_when_set():
    """Each new knob value changes the body (it is built, not ignored)."""
    base = SMF.best_knobs(4, "best_plain")
    ref = _lower_kernel_to_llvm_python(MF.build_fused_suffix(4, base), llvm_flavor="llvm23", arch="gfx950")
    for kn in ({"numa": 1}, {"lg_copy": "numa2"}, {"xchg_home": "numa2"}, {"line_order": "router_dn"},
               {"sh_join": 4}, {"combine": "last"}, {"slots": "wave"}, {"dn_issue": "lds"}, {"ys_home": "numa2"}):
        k = MF.FusedKnobs(**{**base.__dict__, **kn})
        ir = _lower_kernel_to_llvm_python(MF.build_fused_suffix(4, k), llvm_flavor="llvm23", arch="gfx950")
        assert ir != ref, kn


def test_slots_arith_rows1_lowers():
    kn = MF.FusedKnobs(slots="arith")
    ir = _lower_kernel_to_llvm_python(MF.build_fused_suffix(1, kn), llvm_flavor="llvm23", arch="gfx950")
    assert "define amdgpu_kernel" in ir


def test_stage_functions(tmp_path):
    written = SMF.write_functions(str(tmp_path), ["r1", "r4s"], MF.FusedKnobs(), arena_words=None, flavor="llvm23")
    assert {(w["stage"], w["variant"]) for w in written} == {(s, v) for s in SMF.STAGES for v in ("r1", "r4s")}
    pairs = set()
    for w in written:
        with open(w["json"]) as f:
            meta = json.load(f)
        assert meta["symbol"] == f"rocke_q38_{w['stage']}_{w['variant']}"
        assert meta["requires"] == {s: meta["pair"] for s in SMF.STAGES if s != w["stage"]}
        assert meta["fused"] is True
        pairs.add((w["variant"], meta["pair"]))
        with open(w["ll"]) as f:
            ir = f.read()
        body = ir[ir.index("define void"):]
        if w["stage"] == "m3":
            assert "mfma" in body and w["lds_bytes"] > 0
            assert "alwaysinline" in ir
        else:  # empty: entry block returns
            assert "mfma" not in body and "barrier" not in body and w["lds_bytes"] == 0
    assert len(pairs) == 2  # one pair id per variant generation


@pytest.mark.parametrize("which", sorted(SMF.BEST_SETS))
def test_best_sets(tmp_path, which):
    """The serving sets (``--knobs best`` / ``best_plain``) build for every narrow variant and fit the mono arena;
    the NUMA sets' sidecars carry the layout ATOM checks."""
    for rows in (1, 2, 3, 4):
        kn = SMF.best_knobs(rows, which)
        MF.FusedPlan(rows, A.EXPERTS, kn).validate()
        for seq in ((False,) if rows == 1 else (False, True)):
            SMF.build_function("m3", rows, seq, kn, arena_words=MONO_LDS[(rows, seq)])
    assert SMF.main(["--out", str(tmp_path), "--variants", "r1,r4s", "--knobs", which, "--no-arena-check",
                     "--flavor", "llvm23"]) == 0
    with open(tmp_path / "m3_r1.json") as f:
        m3 = json.load(f)
    assert m3["moe_knobs"] == "fused:" + SMF.best_knobs(1, which).suffix()
    assert (m3.get("mailboxes") == "numa1") == (which == "best")


def test_generator_defaults_to_best(tmp_path):
    """No --knobs: the tuned sets, never the spilling first-light defaults."""
    assert SMF.main(["--out", str(tmp_path), "--variants", "r4s", "--no-arena-check", "--flavor", "llvm23"]) == 0
    with open(tmp_path / "m3_r4s.json") as f:
        assert json.load(f)["moe_knobs"] == "fused:" + SMF.best_knobs(4).suffix()


def test_stage_function_refuses_lds_past_arena():
    with pytest.raises(ValueError, match="LDS arena"):
        SMF.build_function("m3", 4, True, MF.FusedKnobs(), arena_words=1000)
    with pytest.raises(ValueError):
        SMF.build_function("m3", 1, True, MF.FusedKnobs(), arena_words=None)  # seq needs 2 rows
    with pytest.raises(ValueError, match="router"):
        SMF.build_function("m3", 1, False, MF.FusedKnobs(router=False), arena_words=None)


def test_numa_sidecars(tmp_path):
    """numa=1 bodies say so (``mailboxes: numa1``) and every body that uses the NUMA area names the header."""
    SMF.write_functions(str(tmp_path / "n1"), ["r4s"], MF.FusedKnobs(numa=1), arena_words=None, flavor="llvm23")
    SMF.write_functions(str(tmp_path / "lg"), ["r4s"], MF.FusedKnobs(lg_copy="numa2"), arena_words=None,
                        flavor="llvm23")  # fmt: skip
    SMF.write_functions(str(tmp_path / "pl"), ["r4s"], MF.FusedKnobs(), arena_words=None, flavor="llvm23")
    for st in SMF.STAGES:
        with open(tmp_path / "n1" / f"{st}_r4s.json") as f:
            m = json.load(f)
        assert m["mailboxes"] == "numa1" and m["numa_hdr"] == MF.NUMA_HDR
        with open(tmp_path / "lg" / f"{st}_r4s.json") as f:
            m = json.load(f)
        assert m["mailboxes"] == "plain" and m["numa_hdr"] == MF.NUMA_HDR
        with open(tmp_path / "pl" / f"{st}_r4s.json") as f:
            m = json.load(f)
        assert "mailboxes" not in m and "numa_hdr" not in m


def test_atom_numa_link_rules(tmp_path, monkeypatch):
    """ATOM (when importable): a numa=1 generation keeps the numa=1 lever and refuses a build without it; a body
    using the NUMA area needs the area; plain bodies drop the lever (``extern_stages.numa_kept``)."""
    pytest.importorskip("flydsl")
    XS = pytest.importorskip("atom.models.qwen4_exp_mono.kernels.extern_stages")
    from atom.models.qwen4_exp_mono.kernels import kernel as K
    from atom.models.qwen4_exp_mono.kernels import sched

    SMF.write_functions(str(tmp_path / "n1"), ["r4s"], MF.FusedKnobs(numa=1), arena_words=None, flavor="llvm23")
    SMF.write_functions(str(tmp_path / "pl"), ["r4s"], MF.FusedKnobs(), arena_words=None, flavor="llvm23")

    def spec(d):
        return {s: f"{tmp_path}/{d}/{s}_{{variant}}.ll" for s in SMF.STAGES}

    monkeypatch.delenv(sched.ENV_SCHED, raising=False)
    with pytest.raises(ValueError, match="numa lever"):  # the area does not exist
        XS.resolve(K.STAGES["delta"], 4, True, 0, K.LAYER_ARGS, K._ALL_STAGES, spec=spec("n1"))
    monkeypatch.setenv(sched.ENV_SCHED, "numa=1")
    got = XS.resolve(K.STAGES["delta"], 4, True, 0, K.LAYER_ARGS, K._ALL_STAGES, spec=spec("n1"))
    assert {e.mailboxes for e in got} == {"numa1"}
    assert XS.numa_kept(got, {"numa": 1})
    with pytest.raises(ValueError, match="numa=1 mailbox layout"):
        XS.numa_kept(got, {})
    with pytest.raises(ValueError, match="numa=1 mailbox layout"):
        XS.numa_kept(got, {"numa": 3})
    plain = XS.resolve(K.STAGES["delta"], 4, True, 0, K.LAYER_ARGS, K._ALL_STAGES, spec=spec("pl"))
    assert not XS.numa_kept(plain, {"numa": 1})
    with pytest.raises(ValueError, match="numa=1 mailbox layout"):
        XS.numa_kept(got[:1] + plain[1:], {"numa": 1})


def test_atom_refuses_partial_link(tmp_path):
    """ATOM's resolve (when importable) refuses m3 without the fused m4 / m5 of its generation."""
    pytest.importorskip("atom.models.qwen4_exp_mono.kernels.extern_stages")
    from atom.models.qwen4_exp_mono.kernels import extern_stages as XS
    from atom.models.qwen4_exp_mono.kernels import kernel as K

    SMF.write_functions(str(tmp_path), ["r1"], MF.FusedKnobs(), arena_words=None, flavor="llvm23")
    spec = {"m3": f"{tmp_path}/m3_{{variant}}.ll"}
    with pytest.raises(ValueError, match="needs"):
        XS.resolve(K.STAGES["delta"], 1, False, 0, K.LAYER_ARGS, K._ALL_STAGES, spec=spec)
    spec.update({s: f"{tmp_path}/{s}_{{variant}}.ll" for s in ("m4", "m5")})
    got = XS.resolve(K.STAGES["delta"], 1, False, 0, K.LAYER_ARGS, K._ALL_STAGES, spec=spec)
    assert {e.stage for e in got} == {"m3", "m4", "m5"}


# ----------------------------------------------------------------- split port byte identity

# sha256[:16] of the split port's LLVM text (gfx950; llvm23 keys plain, llvm22 keys prefixed ``llvm22/``) for the
# default knobs, the tuned sets (stage_moe.BEST_KNOBS as of the refactor, pinned literally here so that retuning
# stage_moe does not move this golden) and a mixed set; suffix kernels rows 1..4, stage functions r1 r2s r3s r4s.
# Pinned with the moe.py from BEFORE the fused track's refactors (gu_epilogue_entry, combine_tile; hackathon-prep
# qwen38/moe-fused/split_golden.py --moe-file): the refactors must not change one byte.
SPLIT_SETS = {
    "default": {},
    "best1": {"m5_ranks": "pre", "a_pre": True, "slots": "arith", "m5_extras": "first", "shared_early": True,
              "mailbox": {"impl": "ir"}},
    "bestN": {"dedupe": True, "m4_waves": 4, "m4_map": "balanced", "m5_ranks": "pre", "a_pre": True,
              "m4_guard": "none", "m4_interleave": True, "m4_apipe": 1, "m5_extras": "m4", "m4_weights": "late",
              "mailbox": {"impl": "ir"}, "w_cpol": 18},
    "misc": {"m5_ranks": "pre", "slots": "table", "m5_extras": "after", "route_gate": "pre_poll",
             "route_pause": "s_sleep1", "shared_first": True, "sched_issue": True},
}  # fmt: skip


def _split_hashes():
    from kernels.gfx950.qwen38_mono import stage_moe as SM
    from kernels.gfx950.qwen38_mono.device_function import (
        DeviceFunctionKnobs,
        lower_device_function,
    )
    from kernels.gfx950.qwen38_mono.stage_fn import parse_variant

    out = {}
    for flavor in ("llvm23", "llvm22"):
        pre = "" if flavor == "llvm23" else flavor + "/"
        for name, kn in SPLIT_SETS.items():
            knobs = SM.knobs_from_json(kn)
            for rows in (1, 2, 3, 4):
                if (name == "best1" and rows != 1) or (name == "bestN" and rows == 1):
                    continue
                ir = _lower_kernel_to_llvm_python(MO.build_moe_suffix(MO.MoeSuffixSpec(rows=rows, knobs=knobs)),
                                                  llvm_flavor=flavor, arch="gfx950")  # fmt: skip
                out[f"{pre}suffix/{name}/r{rows}"] = hashlib.sha256(ir.encode()).hexdigest()[:16]
            for v in ("r1", "r2s", "r3s", "r4s"):
                rows, seq, _g = parse_variant(v)
                if name in ("best1", "bestN") and (name == "best1") != (rows == 1):
                    continue
                for st in ("m3", "m4", "m5"):
                    kern, _p = SM.build_function(st, rows, seq, knobs, m5_reuse=True, arena_words=None)
                    fn = lower_device_function(kern, arch="gfx950", symbol=f"rocke_q38_{st}_{v}", llvm_flavor=flavor,
                                               knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip
                    out[f"{pre}fn/{name}/{st}_{v}"] = fn.sha256()[:16]
    return out


def test_split_port_ir_unchanged():
    path = os.path.join(os.path.dirname(__file__), "golden", "qwen38_moe_split_ir_sha256.json")
    with open(path) as f:
        want = json.load(f)
    assert _split_hashes() == want


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--print-split-golden"]:
        print(json.dumps(_split_hashes(), indent=1, sort_keys=True))
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-q"]))
