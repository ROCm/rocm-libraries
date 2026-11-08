# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the pack / GEMM overlap plans (``kernels/gfx950/mxfp6_gemm/overlap.py``) and its harness helpers.

* chunk plans: every C tile computed exactly once, the chunk GEMM specs are legal, the per-XCD operand panel count;
* schedules: the host happens-before proof is clean for the shipped schedules and flags each must-fail mutant
  (RAW for a GEMM that does not wait for its pack, WAR for a pack that does not wait for the previous GEMM), and
  the checker itself catches hand-built hazards;
* the co-resident pack kernel: builder validation, no LDS, cache hints, names; the bench's arm parsing / CI math.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import overlap as O
from kernels.gfx950.mxfp6_gemm.gemm import PRESETS, Mxfp6GemmSpec, validate_spec


# ----------------------------------------------------------------- chunk plans


@pytest.mark.parametrize("chunks,blk", sorted(O.CHUNK_XCD.items()) + sorted(O.CHUNK_XCD_ALT.items()))
def test_chunk_plans_cover_every_tile_once(chunks, blk):
    a, b, ctas = O.xcd_panels(4096, 4096, chunks, blk)
    tiles = [(g, t) for _, g, t in ctas]
    assert len(tiles) == 256 and len(set(tiles)) == 256
    rows = 4096 // chunks
    for c, g, _ in ctas:  # a chunk's CTAs compute its own rows only
        assert c * rows <= g * 256 < (c + 1) * rows
    panels = [len(x) + len(y) for x, y in zip(a, b)]
    assert panels == [12] * 8 if chunks <= 8 else panels == [18] * 8
    assert all(len(x) == len(set(x)) for x in a)


@pytest.mark.parametrize("chunks", sorted(O.CHUNK_XCD))
def test_chunk_gemm_specs_are_legal(chunks):
    for ch in O.chunk_plan(4096, chunks):
        over = O.chunk_spec_overrides({**PRESETS["best_v6"]}, ch)
        ok, why = validate_spec(replace(Mxfp6GemmSpec(), **over))
        assert ok, why
        assert over["m"] == 4096 // chunks


def test_chunk_plan_rejects_bad_splits():
    with pytest.raises(ValueError):
        O.chunk_plan(4096, 3)
    with pytest.raises(ValueError):
        O.chunk_plan(4096, 32)  # 128-row chunks: not whole 256-row tiles
    with pytest.raises(ValueError):
        O.chunk_plan(4096, 32, tile_m=128)  # legal split, but no default XCD block


# ----------------------------------------------------------------- schedules


SAFE = [
    O.schedule_serial(),
    O.schedule_chunked(2), O.schedule_chunked(4), O.schedule_chunked(8), O.schedule_chunked(16),
    O.schedule_chunked(4, mode="pipelined"), O.schedule_chunked(8, mode="pipelined"),
    O.schedule_chunked(4, gemm_only=True), O.schedule_chunked(4, per_chunk_streams=False),
    O.schedule_next_call(), O.schedule_next_call(gemm_first=False),
]


@pytest.mark.parametrize("s", SAFE, ids=lambda s: s.label)
def test_schedules_are_hazard_free(s):
    for calls in (1, 2, 5):
        assert O.check_schedule(s, calls) == []


@pytest.mark.parametrize("make,kind", [
    (lambda: O.schedule_chunked(4, mutant="nowait"), "RAW"),
    (lambda: O.schedule_chunked(4, mutant="nowar"), "WAR"),
    (lambda: O.schedule_chunked(4, mode="pipelined", mutant="nowait"), "RAW"),
    (lambda: O.schedule_chunked(4, mode="pipelined", mutant="nowar"), "WAR"),
    (lambda: O.schedule_next_call(mutant="nowait"), "RAW"),
    (lambda: O.schedule_next_call(mutant="nowar"), "WAR"),
])
def test_schedule_mutants_are_flagged(make, kind):
    bad = O.check_schedule(make(), 3)
    assert bad and any(x.startswith(kind) for x in bad), bad


def test_single_stream_chunk_gemms_serialize():
    """per_chunk_streams=False puts every chunk GEMM on one stream: safe, but no overlap between chunk GEMMs."""
    s = O.schedule_chunked(4, per_chunk_streams=False)
    assert [x for x in s.streams if x.startswith("G")] == ["G"]


def _op(kind, stream, what, reads=(), writes=()):
    return O.StreamOp(kind, stream, what, (), tuple(reads), tuple(writes))


def test_checker_catches_hand_built_hazards():
    w = _op("launch", "P", "pack", writes=["x"])
    r = _op("launch", "G", "gemm", reads=["x"])
    # no event: RAW
    s = O.Schedule([], [w, r], ("G", "P"), "t")
    assert any(x.startswith("RAW") for x in O.check_schedule(s, 1))
    # wait before the record exists (host order): still a hazard
    s = O.Schedule([], [_op("wait", "G", "e"), r, w, _op("record", "P", "e")], ("G", "P"), "t")
    assert O.check_schedule(s, 2)
    # proper record / wait: clean for one call; two calls need the WAR edge too
    s = O.Schedule([], [w, _op("record", "P", "e"), _op("wait", "G", "e"), r], ("G", "P"), "t")
    assert O.check_schedule(s, 1) == []
    assert any(x.startswith("WAR") for x in O.check_schedule(s, 2))
    s = O.Schedule([_op("record", "G", "g")], [_op("wait", "P", "g"), w, _op("record", "P", "e"),
                                               _op("wait", "G", "e"), r, _op("record", "G", "g")], ("G", "P"), "t")
    assert O.check_schedule(s, 4) == []
    # same stream: ordered without events
    s = O.Schedule([], [w, _op("launch", "P", "gemm", reads=["x"])], ("P",), "t")
    assert O.check_schedule(s, 3) == []


def test_transitive_order():
    a = _op("launch", "A", "k1", writes=["x"])
    b = _op("launch", "B", "k2")
    c = _op("launch", "C", "k3", reads=["x"])
    s = O.Schedule([], [a, _op("record", "A", "ea"), _op("wait", "B", "ea"), b, _op("record", "B", "eb"),
                        _op("wait", "C", "eb"), c], ("A", "B", "C"), "t")
    assert O.check_schedule(s, 1) == []


# ----------------------------------------------------------------- co-resident pack kernel


def _ops(kernel):
    out = []

    def walk(region):
        for op in region.ops:
            out.append(op)
            for r in op.regions:
                walk(r)

    walk(kernel.body)
    return out


@pytest.mark.parametrize("kw", [{}, {"order": "row"}, {"nt_store": True}, {"n_wg": 7}, {"scales": False}])
def test_pack_lite_has_no_lds_and_names_its_knobs(kw):
    k = O.build_pack_lite(1024, 8192, **kw)
    names = [op.name for op in _ops(k)]
    assert not any("smem_alloc" in x or "smem" in x.split(".")[-1] for x in names), names
    assert k.attrs.get("waves_per_eu") == (8, 8)
    st = [op for op in _ops(k) if op.name == "memref.global_store_vN"]
    n_sc = 0 if kw.get("scales") is False else 4
    assert len(st) == 2 + n_sc + (2 + n_sc if kw.get("n_wg") == 7 else 0), names  # + the guarded remainder copy
    assert all(bool(op.attrs.get("nontemporal")) == bool(kw.get("nt_store")) for op in st)


def test_pack_lite_names_are_unique():
    cfgs = [{}, {"order": "row"}, {"nt_store": True}, {"n_wg": 512}, {"scales": False}, {"mutant": "kg"},
            {"mutant": "scale"}, {"waves_per_eu": 0}, {"nt_load": True}, {"stage": "lds16"},
            {"stage": "lds16", "nt_load": True}, {"sc1_store": True}]
    names = [O.build_pack_lite(512, 4096, **c).name for c in cfgs]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("bad", [dict(rows=100, k=4096), dict(rows=512, k=4000), dict(rows=512, k=4096, order="x"),
                                 dict(rows=512, k=4096, mutant="x"), dict(rows=512, k=4096, n_wg=0)])
def test_pack_lite_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        O.build_pack_lite(**bad)


# ----------------------------------------------------------------- harness helpers


def test_paired_ci_brackets_the_median_ratio():
    from benchmarks.gfx950.mxfp6_gemm.overlap_bench import paired_ci

    rng = np.random.default_rng(0)
    ref = 1.0 + 0.01 * rng.standard_normal(40)
    arm = ref / 1.05 * (1 + 0.002 * rng.standard_normal(40))
    med, lo, hi = paired_ci(list(ref), list(arm))
    assert lo <= med <= hi and lo > 1.03 and hi < 1.07
    med, lo, hi = paired_ci(list(ref), list(ref))
    assert med == lo == hi == 1.0


def test_bench_arm_builds_follow_the_chunk_plan():
    from benchmarks.gfx950.mxfp6_gemm import overlap_bench as OB

    b = OB.gemm_builds({"kind": "chunked", "chunks": 4}, 4096, 4096, 65536)
    assert sorted(b) == [0, 1, 2, 3] and len({key for key, _ in b.values()}) == 1
    over = b[0][1]
    assert over["m"] == 1024 and over["xcd_block"] == O.CHUNK_XCD[4]
    assert {x: over[x] for x in PRESETS["best_v6"]} == PRESETS["best_v6"]
    full = OB.gemm_builds({"kind": "gemm"}, 4096, 4096, 65536)["all"]
    assert full[1]["m"] == 4096
    assert OB.gemm_levers({"kind": "gemm"}).endswith("sreassoc,m0share,ldsgroup")
    mut = OB.gemm_builds({"kind": "gemm", "gemm_mutant": "k_order"}, 4096, 4096, 65536)["all"]
    assert mut[0] != full[0] and mut[1]["mutant"] == "k_order"
    assert OB.schedule_of({"kind": "nextcall"}).scenario == "next-call"
    assert OB.schedule_of({"kind": "chunked", "chunks": 4}).scenario == "single-call"
    assert OB.schedule_of({"kind": "chunked", "chunks": 4, "mode": "pipelined"}).scenario == "next-call"


# ----------------------------------------------------------------- fused K-flag path


@pytest.mark.parametrize("mode,dep", [("strict", "event"), ("strict", "value"), ("strict", "spin"),
                                      ("next", "event")])
def test_kflag_schedules(mode, dep):
    s = O.schedule_kflag(mode=mode, dep=dep)
    for calls in (1, 2, 5):
        assert O.check_schedule(s, calls) == []
    bad = O.check_schedule(O.schedule_kflag(mode=mode, dep=dep, mutant="nowar"), 4)
    assert bad and all(x.startswith(("WAR", "WAW")) for x in bad)
    # without the flag guard the same schedule is a RAW hazard: the flags are what orders the GEMM after its pack
    s.meta["flag_guarded"] = ()
    assert any(x.startswith("RAW") for x in O.check_schedule(s, 2))
    sched = O.schedule_kflag(mode=mode, dep=dep)
    gemms = [op for op in sched.call if op.kind == "launch" and op.what == "gemm"]
    assert all(op.stream == "G" for op in gemms)
    assert not [op for op in sched.call if op.kind == "wait" and op.stream == "G"]
    if dep != "event":  # no event packet at all between consecutive GEMMs
        assert not [op for op in sched.call if op.kind in ("record", "wait")]


def test_kflag_parity_flips_per_slot_use():
    assert [O.kflag_parity(c, 1) for c in range(4)] == [1, 0, 1, 0]
    assert [O.kflag_parity(c, 2) for c in range(8)] == [1, 1, 0, 0, 1, 1, 0, 0]
    # the four-call next-call list: a slot is used twice per list, so repeating the list keeps the parity sequence
    s = O.schedule_kflag(mode="next")
    assert s.meta["calls_per_list"] == 4 and s.meta["slots"] == 2


def _names(k):
    return [op.name for op in _ops(k)]


def test_pack_lite_flag_mode_publishes_each_chunk():
    k = O.build_pack_lite(512, 4096, flag_tiles=8)
    names = _names(k)
    assert [x.name for x in k.params][-5:] == ["counters", "flags", "gdone", "timeouts", "parity"]
    assert names.count("memref.global_atomic_add") == 1 and names.count("memref.global_flag_store") == 2
    assert "memref.fence" in names
    first_store = names.index("memref.global_store_vN")
    assert names.index("memref.global_atomic_add") > first_store  # arrival after the chunk's stores
    early = _names(O.build_pack_lite(512, 4096, flag_tiles=8, mutant="early"))
    assert early.index("memref.global_atomic_add") < early.index("memref.global_store_vN")
    assert "memref.fence" not in early
    sc1 = O.build_pack_lite(512, 4096, flag_tiles=8, release="sc1")
    asm = [op.attrs["template"] for op in _ops(sc1) if op.name == "tile.inline_asm"]
    assert asm and all("sc0 sc1" in t for t in asm) and "memref.fence" not in _names(sc1)


@pytest.mark.parametrize("bad", [dict(flag_tiles=3), dict(flag_tiles=8, order="row"), dict(flag_tiles=8, scales=False),
                                 dict(mutant="early"), dict(flag_tiles=8, release="x")])
def test_pack_lite_flag_mode_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        O.build_pack_lite(512, 4096, **bad)


def _asm(k):
    return [op.attrs["template"] for op in _ops(k) if op.name == "tile.inline_asm"]


def test_kflag_gemm_polls_every_loop_body():
    spec = replace(Mxfp6GemmSpec(), **PRESETS["best_v6"])
    k = O.build_gemm_kflag(spec, 16)
    assert [x.name for x in k.params][-5:] == ["flags", "done_cnt", "done_sig", "timeouts", "parity"]
    asm = _asm(k)
    polls = [t for t in asm if "s_sleep" in t]
    loads = [t for t in asm if t.startswith("s_load_dword") and "s_sleep" not in t]
    # prologue wait, loop-top check, static-tail wait; prologue prefetch + one prefetch per loop body
    assert len(polls) == 3 and len(loads) == 4
    assert all("glc" in t for t in asm if "s_load" in t)
    assert O.build_gemm_kflag(spec, 16, mutant="noflag").name != k.name
    assert not [t for t in _asm(O.build_gemm_kflag(spec, 16, mutant="noflag")) if "s_load" in t]
    with pytest.raises(ValueError):
        O.build_gemm_kflag(spec, 7)  # does not divide the K tiles
    with pytest.raises(ValueError):
        O.build_gemm_kflag(replace(spec, tail_mode="clamp"), 16)


def test_kflag_gemm_without_flags_matches_the_plain_ir():
    """Polls aside the K-flag GEMM is the plain kernel: same MFMA / DTL / LDS-read counts."""
    from kernels.gfx950.mxfp6_gemm.gemm import build_mxfp6_gemm

    spec = replace(Mxfp6GemmSpec(), **PRESETS["best_v6"])
    plain, kf = _names(build_mxfp6_gemm(spec)), _names(O.build_gemm_kflag(spec, 16, mutant="noflag"))
    for op in set(plain) | set(kf):
        if op not in ("tile.inline_asm", "tile.global_ptr_add", "arith.smin", "arith.div", "arith.muli", "arith.addi",
                      "arith.constant", "arith.xori"):
            assert plain.count(op) == kf.count(op), op


def test_pack_lite_wait_prev_polls_first_and_counts_timeouts():
    k = O.build_pack_lite(512, 4096, flag_tiles=8, release="sc1", wait_prev=True)
    names = _names(k)
    ops = _ops(k)
    spin = [i for i, op in enumerate(ops) if op.name == "tile.inline_asm" and "s_sleep" in op.attrs["template"]]
    stores = [i for i, op in enumerate(ops) if op.name == "tile.inline_asm" and "global_store" in op.attrs["template"]]
    assert spin and stores and spin[0] < stores[0]  # waits before any (write-through) store
    assert names.count("memref.global_atomic_add") == 2  # the timeout counter + the chunk arrival
    with pytest.raises(ValueError):
        O.build_pack_lite(512, 4096, wait_prev=True)  # needs flag mode


@pytest.mark.parametrize("kw", [{}, {"stage": "lds16"}, {"stage": "lds16", "flag_tiles": 8, "release": "sc1"}])
def test_pack_lite_block_size_matches_the_kernel(kw):
    """Launching with another block size than the kernel's flat work-group size fails (hipError 719)."""
    k = O.build_pack_lite(512, 4096, **kw)
    assert k.attrs["max_workgroup_size"] == O.pack_lite_block(kw.get("stage", "none"))
    assert O.pack_lite_block("lds16") == 128 and O.pack_lite_block() == 256


def test_pack_lite_lds16_stage_fits_next_to_the_gemm_cta():
    from kernels.gfx950.mxfp6_gemm.gemm import lds_bytes

    k = O.build_pack_lite(4096, 65536, stage="lds16")
    smem = [op for op in _ops(k) if op.name == "tile.smem_alloc" or "smem_alloc" in op.name]
    assert smem
    spec = replace(Mxfp6GemmSpec(), **PRESETS["best_v6"])
    assert lds_bytes(spec) + 16 * (4 * 96 + 16) <= 160 * 1024
    with pytest.raises(ValueError):
        O.build_pack_lite(512, 4096, stage="lds16", flag_tiles=6)
    with pytest.raises(ValueError):
        O.build_pack_lite(512, 4096, stage="lds16", order="row")
