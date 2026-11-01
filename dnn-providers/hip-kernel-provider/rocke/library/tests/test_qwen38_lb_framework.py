# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L1 (large-batch framework): CPU checks of ``large_batch.lbf_plan``, ``lbf``, ``lbf_hazard``, ``lbf_lint`` and
``lbf_stale``.

The host payload model equals an independent word-level simulation of the kernel's arithmetic (masking, several layers,
the cross-layer edge) and P-L2's closed form on one layer; LB validate finds planted write-after-read violations and
cycles; the IR index-width lint and the host formulas agree on a planted overflow; every kernel variant lowers with
the pieces the GPU bench relies on; the knob refusals hold; the ISA hazard scanner flags a planted hazard and not its
padded twin. The hardware proofs run in ``hackathon-prep/qwen38/truemono-lb/wp-l1``.
"""

from __future__ import annotations

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono.large_batch import lbf as F
from kernels.gfx950.qwen38_mono.large_batch import lbf_hazard as Z
from kernels.gfx950.qwen38_mono.large_batch import lbf_lint as LL
from kernels.gfx950.qwen38_mono.large_batch import lbf_plan as P
from kernels.gfx950.qwen38_mono.large_batch import lbf_stale as ST
from kernels.gfx950.qwen38_mono.large_batch import sync_plan as SP
from kernels.gfx950.qwen38_mono.persist.knobs import PersistKnobs
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(kd):
    return _lower_kernel_to_llvm_python(kd, arch="gfx950", llvm_flavor=FLAVOR)


# ------------------------------------------------------------------------------------------------ graphs / tables


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("kind", ["ft", "lb"])
@pytest.mark.parametrize("bucket", list(P.BUCKETS))
def test_graph_buckets_and_tables(tp, kind, bucket):
    g = P.build_graph(tp, bucket, kind)
    assert g.S == {("ft", 1): 16, ("ft", 2): 14, ("lb", 1): 11, ("lb", 2): 10}[(kind, tp)]
    nt = -(-bucket // 16)
    flat = [t for st in g.tt for t in st]
    assert all(-1 <= t < nt for t in flat)
    t = P.tables(g, P.EdgeSync.default(g.plan))
    assert t["stab"].shape == (g.S, P.SCOLS) and t["utab"].shape == (g.N, P.UCOLS)
    # deps words decode to the plan's read sets and the producers' token tiles
    u0 = g.u0()
    for i, s in enumerate(g.plan.stages):
        pi = g.prev(i)
        for u in (0, s.units - 1):
            row = t["utab"][u0[i] + u]
            w = t["deps"][row[3]:row[3] + row[4]].astype(np.int64)
            assert list(w & ((1 << P.TT_SHIFT) - 1)) == [u0[pi] + d for d in s.deps[u]]
            assert list((w >> P.TT_SHIFT) - 1) == [flat[u0[pi] + d] for d in s.deps[u]]
    # the grid prefix column counts grid stages before each stage
    gs = sorted(P.EdgeSync.default(g.plan).grid_stages(g.S))
    assert [int(x) for x in t["stab"][:, 3]] == [sum(1 for q in gs if q < i) for i in range(g.S)]
    assert t["gpl"] == len(gs)


def test_bucket_for_and_masking():
    assert [P.bucket_for(r) for r in (1, 5, 8, 9, 16, 17, 100, 128)] == [8, 8, 8, 16, 16, 32, 128, 128]
    with pytest.raises(ValueError):
        P.bucket_for(129)
    g = P.build_graph(1, 128, "lb")
    full, part = g.live(128), g.live(17)
    assert full.all() and 0 < (~part).sum() < g.N
    # recurrence units of rows >= 32 are masked at 17 live rows (tiles 0, 1 live)
    rec = g.names.index("gdn_recur")
    u0 = g.u0()[rec]
    Hv = 48
    assert part[u0 + 7 * Hv] and not part[u0 + 8 * Hv]  # quad 7 = rows 28..31 (tile 1) live; quad 8 = rows 32.. masked


def test_edge_sync_default_matches_pl2():
    for tp, rows in ((1, 16), (2, 128)):
        p = SP.make_plan(tp, rows, "lb")
        assert P.EdgeSync.default(p).gmask == tuple(SP.gmask_full(p, 0.4))


# ------------------------------------------------------------------------------------------------ payload model


def _simulate(g, ltot, layers, E, rows):
    """Word-level simulation of the kernel: payload words, read sets of live producers, masked units silent."""
    p = g.plan
    live = g.live(rows)
    u0 = g.u0()
    pay = {}
    for lay in layers:
        tag = P.layer_tag(E, ltot, lay)
        for i, s in enumerate(p.stages):
            pi = g.prev(i)
            src = lay - 1 if i == 0 else lay
            for u in range(s.units):
                gu = u0[i] + u
                if not live[gu]:
                    continue
                chk = 0
                if not (lay == 0 and i == 0):
                    for d in s.deps[u]:
                        gd = u0[pi] + d
                        if not live[gd]:
                            continue
                        words = pay[(src, gd)]
                        chk += int(sum(int(words[(gd & 7) * 64 + q]) for q in range(64)))
                h = SP.h_unit(tag, lay, i, gu)
                pay[(lay, gu)] = np.asarray([(h + 3 * chk + q * SP.W_STEP) & SP.M32 for q in range(P.PAY_WORDS)],
                                            dtype=np.uint64)
    return pay, live


@pytest.mark.parametrize("rows", [8, 5])
def test_payload_model_equals_simulation(rows):
    g = P.build_graph(2, 8, "lb", seed=3)
    ltot, E = 3, 2
    base, live = P.expected_payload(g, ltot=ltot, layers=range(ltot), E=E, rows=rows)
    pay, live2 = _simulate(g, ltot, range(ltot), E, rows)
    assert (live == live2).all()
    for (lay, gu), words in pay.items():
        assert int(words[0]) == int(base[lay, gu]) and int(words[17]) == (int(base[lay, gu]) + 17 * SP.W_STEP) & SP.M32


def test_payload_model_masking_simulation_multi_tile():
    g = P.build_graph(1, 32, "lb", seed=5)
    ltot, E = 2, 0
    base, live = P.expected_payload(g, ltot=ltot, layers=range(ltot), E=E, rows=13)
    pay, _ = _simulate(g, ltot, range(ltot), E, 13)
    assert (~live).sum() > 0
    for (lay, gu), words in pay.items():
        assert int(words[0]) == int(base[lay, gu])
    assert all(base[:, ~live].ravel() == 0)


def test_payload_model_one_layer_equals_pl2_closed_form():
    g = P.build_graph(1, 16, "ft", seed=1)
    E = 4
    base, _ = P.expected_payload(g, ltot=1, layers=range(1), E=E)
    H, C = SP.expected_checksums(g.plan, 1, E + 1)
    assert ((H + 3 * C) & SP.M32 == base).all()


# ------------------------------------------------------------------------------------------------ LB validate


def _toy_plan(units=4, shift=1):
    """Two stages; stage 1 unit u reads stage 0 units {u, u + shift}; stage 0 unit u reads stage 1 unit u (previous
    layer). Slot reuse at 1 slot is unordered across CTAs (a planted WAR violation)."""
    s0 = SP.Stage("a", [1] * units, [[u] for u in range(units)])
    s1 = SP.Stage("b", [1] * units, [sorted({u, (u + shift) % units}) for u in range(units)])
    p = SP.Plan(1, 16, "ft", False, [s0, s1])
    p.check()
    return P.LBGraph(p, [[-1] * units, [-1] * units])


def test_validate_detects_planted_war():
    g = _toy_plan()
    bad = P.validate(g, P.EdgeSync("flags"), slots=1, strict=False)
    assert bad and "overwrites slot" in bad[0]
    # a grid barrier on every edge orders every reuse; so do launch boundaries per layer
    assert P.validate(g, P.EdgeSync("grid"), slots=1, strict=False) == []
    assert P.validate(g, P.EdgeSync("flags"), slots=1, walk="launch", strict=False) == []
    # dropping the wait of a stage is caught
    with pytest.raises(ValueError):
        P.validate(g, P.EdgeSync("mix", (0,), nowait=(1,)), slots=1)


def test_validate_exchange_slots():
    g = _toy_plan(shift=0)
    assert P.validate(g, P.EdgeSync("flags"), slots=2, strict=False) == []
    # the exchange stage's lines go to every rank: one exchange slot is reused before the other rank read it
    bad = P.validate(g, P.EdgeSync("flags"), slots=2, ranks=2, exchange=("b",), xslots=1, strict=False)
    assert bad and "xslot" in bad[0]
    assert P.validate(g, P.EdgeSync("flags"), slots=2, ranks=2, exchange=("b",), xslots=2, strict=False) == []
    # a grid barrier is per rank: it does not order a read of the other rank's exchange lines (they must be polled)
    bad = P.validate(g, P.EdgeSync("grid"), slots=2, ranks=2, exchange=("b",), xslots=2, strict=False)
    assert bad and "(RAW)" in bad[0]


def test_validate_cycle_is_deadlock():
    with pytest.raises(ValueError, match="deadlock"):
        P._ancestors([[1], [0]])


@pytest.mark.parametrize("tp,rows", [(1, 32), (2, 128)])
def test_validate_lb_default_parity(tp, rows):
    g = P.build_graph(tp, rows, "lb")
    assert P.validate(g, P.EdgeSync.default(g.plan), slots=2) == []
    with pytest.raises(ValueError):
        P.validate(g, P.EdgeSync("none"), slots=2)


def test_ccl_war_hook():
    from kernels.gfx950.qwen38_mono.ccl import war

    sch = war.oneshot_schedule(2, 4, 8, 3, 2, 2)
    assert war.check(sch) == []
    sch1 = war.oneshot_schedule(2, 4, 8, 3, 2, 2, consumer_shift=1)
    assert war.check(sch1)


# ------------------------------------------------------------------------------------------------ index widths


def _lint(kd, g, L, slots):
    t = P.tables(g, P.EdgeSync.default(g.plan))
    u = t["utab"]
    ab = dict(nunits=(g.N, g.N), gpl=(0, g.S), dl=(0, 2**30), maxit=(0, 2**30), blk_layer=(t["blk_layer"],) * 2,
              gridn=(1, 512), l0=(0, L - 1), nl=(1, L), ltot=(L, L), ntiles=(1, 8), slots=(2, slots))
    tb = dict(stab={"cols": 4, 0: (0, g.N), 1: (0, g.N), 2: (0, 15), 3: (0, g.S)},
              utab={"cols": 8, **{c: (int(u[:, c].min()), int(u[:, c].max())) for c in range(8)}},
              deps=(0, int(t["deps"].max())), dly=(0, 1 << 20), ctl=(0, 1 << 30))
    return LL.index_lint(kd, arg_bounds=ab, table_bounds=tb)


def test_index_lint_and_host_bounds_agree():
    g = P.build_graph(1, 128, "lb")
    kd = F.build_lb_kernel(F.LBKnobs(), g.S)
    assert _lint(kd, g, 6, 6) == []
    assert P.index_bounds(g, ltot=6, slots=6, pf=8).check() == []
    # planted: 300 layers of full slots overflow the payload offset in both
    f = _lint(kd, g, 300, 300)
    assert any(x.kind == "overflow" for x in f)
    assert any(s.startswith("pay_byte") for s in P.index_bounds(g, ltot=300, slots=300, pf=8).check())
    # without table annotations the unit-table loads are unbounded and reported
    f2 = LL.index_lint(kd, arg_bounds={})
    assert any(x.kind == "unbounded" for x in f2)


# ------------------------------------------------------------------------------------------------ kernel variants


ARMS = {
    "mix": F.LBKnobs(),
    "grid": F.LBKnobs(sync="grid", pub="drain", lazy=False),
    "tile": F.LBKnobs(sync="tile", pub="drain", lazy=False),
    "tile_w": F.LBKnobs(sync="tile", pub="wave", lazy=False),
    "a2a": F.LBKnobs(sync="a2a", pub="drain", lazy=False),
    "mixd": F.LBKnobs(sync="mix", pub="drain", lazy=False),
    "none": F.LBKnobs(sync="none", lazy=False),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_variants_lower(arm):
    kn = ARMS[arm]
    ll = _ll(F.build_lb_kernel(kn, 3))
    name = F.kernel_name(kn, 3)
    assert name.startswith("q38lbf_st_s3_") and name in ll
    has_poll = "v_cmp_ne_u32_e64" in ll and "s_memrealtime" in ll
    assert has_poll == kn.flags
    assert ("global_atomic_add" in ll) and ("s_nop 4" in ll)  # ticket / barrier / done counter; hz pad


def test_mutants_differ_and_hz():
    base = _ll(F.build_lb_kernel(F.LBKnobs(tag="x"), 3)).split("\n", 3)[-1]
    for m in ("early", "tag", "noacq"):
        ll = _ll(F.build_lb_kernel(F.LBKnobs(mutant=m, tag="x"), 3)).split("\n", 3)[-1]
        assert ll != base
    kd0 = F.build_lb_kernel(F.LBKnobs(hz=False), 3)
    assert Z.pad_asm_vmem(kd0) > 0
    assert Z.pad_asm_vmem(kd0) == 0  # idempotent
    kd1 = F.build_lb_kernel(F.LBKnobs(), 3)
    assert Z.pad_asm_vmem(kd1) == 0


def test_knob_refusals():
    with pytest.raises(ValueError):
        F.LBKnobs(pub="counted").check()
    with pytest.raises(ValueError):
        F.LBKnobs(sync="grid", lazy=True).check()
    with pytest.raises(NotImplementedError):
        F.LBKnobs(persist=PersistKnobs(fresh="ids")).check()
    with pytest.raises(NotImplementedError):
        F.LBKnobs(persist=PersistKnobs(arrivals="xcd")).check()
    with pytest.raises(ValueError):
        F.LBKnobs(persist=PersistKnobs(ticket="off")).check()  # ticket knobs disagree
    F.LBKnobs(persist=PersistKnobs(ticket="off"), ticket=False).check()
    F.LBKnobs(persist=PersistKnobs(machine_licm="on")).check()
    assert "mlon" in F.kernel_name(F.LBKnobs(persist=PersistKnobs(machine_licm="on")), 3)


def test_machine_licm_attr():
    kd = F.build_lb_kernel(F.LBKnobs(), 2)
    assert kd.attrs.get("machine_licm") == "off"
    kd = F.build_lb_kernel(F.LBKnobs(persist=PersistKnobs(machine_licm="on")), 2)
    assert "machine_licm" not in kd.attrs


def test_iv_names_unique():
    ll = _ll(F.build_lb_kernel(F.LBKnobs(), 4))
    for i in range(4):
        assert ll.count(f"%lbu{i} = phi") <= 1


# ------------------------------------------------------------------------------------------------ hazard scanner

_HIT = """0000000000001000 <k>:
\tv_readlane_b32 s4, v1, 0                                  // 000000001000: D2890004 00010101
\tglobal_store_dword v0, v2, s[4:5]                         // 000000001008: DC700000 00040200
"""
_PADDED = """0000000000001000 <k>:
\tv_readlane_b32 s4, v1, 0                                  // 000000001000: D2890004 00010101
\ts_nop 4                                                   // 000000001008: BF800004
\tglobal_store_dword v0, v2, s[4:5]                         // 00000000100C: DC700000 00040200
"""
_BRANCH = """0000000000001000 <k>:
\tv_readfirstlane_b32 s6, v1                                // 000000001000: 7E0C0501
\ts_branch 1                                                // 000000001004: BF820001
\ts_nop 0                                                   // 000000001008: BF800000
\tbuffer_load_dword v3, v0, s[4:7], 0 offen                 // 00000000100C: E0501000 80010300
"""


def test_hazard_scanner():
    h, nv = Z.scan_text(_HIT)
    assert nv == 1 and len(h) == 1 and h[0]["regs"] == [4]
    assert Z.scan_text(_PADDED)[0] == []
    h, _ = Z.scan_text(_BRANCH)  # the branch skips the s_nop: still a hit through the taken path
    assert len(h) == 1


# ------------------------------------------------------------------------------------------------ stale harness


@pytest.mark.parametrize("proto", ST.PROTOS)
@pytest.mark.parametrize("mut", ST.MUTANTS)
def test_stale_variants_lower(proto, mut):
    for sig in ST.SIGS:
        ll = _ll(ST.build_stale_kernel(ST.StaleKnobs(proto, mut, sig)))
        assert ST.stale_name(ST.StaleKnobs(proto, mut, sig)) in ll
        assert ("buffer_inv sc1" in ll) == (proto == "inv" and mut != "early")


def test_knob_defaults_and_bounds():
    # the measured defaults (P-L2 5.1, WP-L1): mix, per-wave, lazy, s_nop, static stage code, hz on, depth 8
    d = F.LBKnobs()
    assert (d.sync, d.pub, d.lazy, d.pause, d.stagecode, d.hz, d.depth, d.ticket) == \
        ("mix", "wave", True, "s_nop", "static", True, 8, True)
    F.LBKnobs(depth=1).check()
    F.LBKnobs(depth=16).check()
    for dep in (0, 17):
        with pytest.raises(ValueError):
            F.LBKnobs(depth=dep).check()
    with pytest.raises(NotImplementedError):
        F.LBKnobs(sync="a2a", pub="drain", lazy=False, stagecode="loop").check()
    F.LBKnobs(sync="tile", pub="drain", lazy=False, stagecode="loop").check()


# ------------------------------------------------------------------------------------------------ runtime stage loop


def test_stagecode_loop_lowers_and_refuses_mixed_emitters():
    kn = F.LBKnobs(stagecode="loop")
    ll = _ll(F.build_lb_kernel(kn, 11))
    assert "stagecodeloop" in F.kernel_name(kn, 11) and "%lbsi" in ll
    # one body: the per-stage loop names exist once (static code has one per stage)
    assert ll.count("lbu0") == 0 and "lbur" in ll
    with pytest.raises(ValueError):
        F.build_lb_kernel(kn, [F.StageSpec("a", F.StreamEmitter()), F.StageSpec("b", F.StreamEmitter())])
    with pytest.raises(ValueError):
        F.LBKnobs(stagecode="x").check()
    # the default (static) build is not affected by the knob's existence (names unchanged)
    assert F.kernel_name(F.LBKnobs(), 11) == "q38lbf_st_s11_mix_wave"


# ------------------------------------------------------------------------------------------------ mutation-driven tests
# (survivors of the first CPU mutation run, hackathon-prep/qwen38/truemono-lb/wp-l1/mut)

from kernels.gfx950.qwen38_mono.persist import sync as _PS  # noqa: E402
from rocke.core.ir import I32 as _I32  # noqa: E402
from rocke.core.ir import I64 as _I64  # noqa: E402
from rocke.core.ir import IRBuilder as _IRB  # noqa: E402
from rocke.core.ir import PtrType as _Ptr  # noqa: E402


def _lint_kernel(fn):
    """A one-store kernel: ``fn(b, x, base)`` returns the store index (or None) with x = readfirstlane(param x)."""
    b = _IRB("lintk")
    x = b.readfirstlane(b.param("x", _I32))
    base = b.readfirstlane(b.param("base", _I64))
    idx = fn(b, x, base)
    if idx is not None:
        b.store(b.const_i32(1), b.int_to_ptr(base, _Ptr(_I32, "global")), idx)
    b.ret()
    return b.kernel, idx


def _rng_of(fn, lo=0, hi=10):
    kd, v = _lint_kernel(fn)
    rng, _d, _a = LL.ranges(kd, arg_bounds={"x": (lo, hi)})
    return rng[v.name]


_EXPRS = [
    (lambda b, x: b.sub(x, b.const_i32(20)), (-20, -10)),
    (lambda b, x: b.mul(x, b.const_i32(-3)), (-30, 0)),
    (lambda b, x: b.div(x, b.const_i32(3)), (0, 3)),
    (lambda b, x: b.mod(x, b.const_i32(4)), (0, 3)),
    (lambda b, x: b.land(x, b.const_i32(7)), (0, 7)),
    (lambda b, x: b.lshr(x, b.const_i32(1)), (0, 5)),
    (lambda b, x: b.shl(x, b.const_i32(2)), (0, 40)),
    (lambda b, x: b.smax(x, b.const_i32(5)), (5, 10)),
    (lambda b, x: b.smin(x, b.const_i32(5)), (0, 5)),
    (lambda b, x: b.add(x, b.const_i32(5)), (5, 15)),
    (lambda b, x: b.select(b.cmp_lt(x, b.const_i32(3)), x, b.const_i32(100)), (0, 100)),
    (lambda b, x: b.thread_id_x(), (0, 511)),
    (lambda b, x: b.lane_id(), (0, 63)),
]


@pytest.mark.parametrize("k", range(len(_EXPRS)))
def test_lint_intervals(k):
    expr, want = _EXPRS[k]
    assert _rng_of(lambda b, x, base: expr(b, x)) == want


def test_lint_hw_ids_and_loop_iv():
    kd, v = _lint_kernel(lambda b, x, base: b.block_id_x())
    assert LL.ranges(kd, arg_bounds={}, grid=300)[0][v.name] == (0, 299)
    b = _IRB("lintloop")
    x = b.readfirstlane(b.param("x", _I32))
    base = b.readfirstlane(b.param("base", _I64))
    with b.scf_for(b.const_i32(2), x, b.const_i32(1), iv_name="liv") as i:
        b.store(b.const_i32(1), b.int_to_ptr(base, _Ptr(_I32, "global")), i)
    b.ret()
    assert LL.ranges(b.kernel, arg_bounds={"x": (5, 9)})[0]["%liv"] == (2, 8)


def test_lint_sinks_and_masking():
    big = 1 << 30
    # a store index that overflows
    kd, _ = _lint_kernel(lambda b, x, base: b.mul(x, b.const_i32(big)))
    f = LL.index_lint(kd, arg_bounds={"x": (0, 4)})
    assert [g.kind for g in f] == ["overflow"] and "global_store" in f[0].sink
    assert "overflow" in str(f[0])

    # an overflowing offset that reaches memory only through an inline-asm VMEM vaddr
    def asm_only(b, x, base):
        _PS.st_sc1(b, base, b.mul(x, b.const_i32(big)), b.const_i32(1))
        return None

    kd, _ = _lint_kernel(asm_only)
    f = LL.index_lint(kd, arg_bounds={"x": (0, 4)})
    assert len(f) == 1 and f[0].sink.startswith("tile.inline_asm")

    # an i32 offset widened into a 64-bit address is a sink; the same zext stored as data is not
    def zext_addr(b, x, base):
        a = b.add(base, b.zext(b.mul(x, b.const_i32(big)), _I64))
        b.store(b.const_i32(1), b.int_to_ptr(a, _Ptr(_I32, "global")), b.const_i32(0))
        return None

    kd, _ = _lint_kernel(zext_addr)
    assert any(g.sink == "arith.zext" for g in LL.index_lint(kd, arg_bounds={"x": (0, 4)}))

    def zext_data(b, x, base):
        b.store(b.zext(b.mul(x, b.const_i32(big)), _I64), b.int_to_ptr(base, _Ptr(_I64, "global")), b.const_i32(0))
        return None

    kd, _ = _lint_kernel(zext_data)
    assert LL.index_lint(kd, arg_bounds={"x": (0, 4)}) == []

    # an unknown (asm) value is unbounded unless a mask bounds it
    def raw(b, x, base):
        return b.inline_asm("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", [], _I32)

    kd, _ = _lint_kernel(raw)
    assert [g.kind for g in LL.index_lint(kd, arg_bounds={})] == ["unbounded"]

    def masked(b, x, base):
        return b.land(b.inline_asm("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", [], _I32), b.const_i32(15))

    kd, _ = _lint_kernel(masked)
    assert LL.index_lint(kd, arg_bounds={}) == []


def test_lint_asm_vaddr_positions():
    from types import SimpleNamespace as NS

    def op(t, c):
        return NS(attrs={"template": t, "constraints": c})

    assert LL._asm_vaddr(op("global_store_dword $0, $1, $2 sc1", "v,v,s")) == [0]
    assert LL._asm_vaddr(op("global_load_dword $0, $1, $2 sc1\ns_waitcnt vmcnt(0)", "=&v,v,s")) == [0]
    assert LL._asm_vaddr(op("global_atomic_add $0, $1, $2, $3 sc0", "=v,v,v,s")) == [0]
    assert LL._asm_vaddr(op("global_atomic_add $0, $1, $2", "v,v,s")) == [0]
    assert LL._asm_vaddr(op("buffer_load_dword $0, $1, $2, 0 offen", "=v,v,s")) == [0]
    assert LL._asm_vaddr(op("s_nop 4\nglobal_store_dword $1, $2, $3", "=s,v,v,s")) == [0]
    assert LL._asm_vaddr(op("ds_write_b32 $0, $1", "v,v")) == []


def test_lint_table_columns():
    b = _IRB("lintcol")
    tab = b.readfirstlane(b.param("tab", _I64))
    u = b.readfirstlane(b.param("u", _I32))
    base = b.readfirstlane(b.param("base", _I64))
    p = b.int_to_ptr(tab, _Ptr(_I32, "global"))
    row = b.mul(u, b.const_i32(8))
    c0 = b.global_load_i32(p, row)
    c3 = b.global_load_i32(p, b.add(row, b.const_i32(3)))
    q = b.int_to_ptr(base, _Ptr(_I32, "global"))
    b.store(b.const_i32(1), q, c0)
    b.store(b.const_i32(1), q, c3)
    b.ret()
    tb = {"tab": {"cols": 8, 0: (0, 7), 3: (100, 200)}}
    rng = LL.ranges(b.kernel, arg_bounds={"u": (0, 4)}, table_bounds=tb)[0]
    assert rng[c0.name] == (0, 7) and rng[c3.name] == (100, 200)
    rng = LL.ranges(b.kernel, arg_bounds={"u": (0, 4)}, table_bounds={"tab": (5, 6)})[0]
    assert rng[c0.name] == (5, 6)


def test_hazard_wait_states_and_vop3b():
    lst = "0000000000001000 <k>:\n\tv_readlane_b32 s4, v1, 0  // 000000001000: 0\n"
    for n, hit in ((3, True), (4, False)):
        t = lst + f"\ts_nop {n}  // 000000001008: 0\n\tglobal_store_dword v0, v2, s[4:5]  // 00000000100C: 0\n"
        assert bool(Z.scan_text(t)[0]) == hit
    # a VOP3B carry-out SGPR destination is a write of that SGPR
    t = ("0000000000001000 <k>:\n\tv_add_co_u32 v1, s[4:5], v2, v3  // 000000001000: 0\n"
         "\tglobal_store_dword v0, v2, s[4:5]  // 000000001008: 0\n")
    h, _ = Z.scan_text(t)
    assert h and h[0]["regs"] == [4, 5]


def test_objdump_env(tmp_path, monkeypatch):
    f = tmp_path / "llvm-objdump"
    f.write_text("")
    monkeypatch.setenv("LLVM_OBJDUMP", str(f))
    assert Z.objdump() == str(f)


def test_unit_tiles_values():
    g = P.build_graph(1, 128, "lb")
    tt = dict(zip(g.names, g.tt))
    # attn_mix_down at 128 rows: 2 token blocks of 64 rows -> first tiles 0 and 4, alternating
    assert sorted(set(tt["attn_mix_down"])) == [0, 4] and tt["attn_mix_down"][:2] == [0, 4]
    # recurrence (4 rows, value head) units: quad q -> tile 4q // 16
    Hv = 48
    assert tt["gdn_recur"][4 * Hv] == 1 and tt["gdn_recur"][4 * Hv - 1] == 0 and tt["gdn_recur"][-1] == 7
    assert tt["moe_combine"][10] == 1 and set(tt["gdn_proj"]) == {-1}
    g2 = P.build_graph(1, 64, "ft")
    t2 = dict(zip(g2.names, g2.tt))
    assert set(t2["attn_mix_down"]) == {-1}  # one token block below 128 rows
    assert t2["attn_mix_combine"][8] == 1  # unit k = rows 2k, 2k + 1
    assert t2["attn_mix_reduce"][-1] == 3 and t2["gdn_out_reduce"][-1] == 3


def test_tables_columns_and_constants():
    g = P.build_graph(2, 16, "lb")
    t = P.tables(g, P.EdgeSync.default(g.plan))
    u = t["utab"]
    blocks = [b_ for s in g.plan.stages for b_ in s.blocks]
    assert list(u[:, 0]) == list(np.cumsum([0] + blocks[:-1])) and list(u[:, 1]) == blocks
    assert t["blk_layer"] == sum(blocks)
    readers = np.zeros(g.N, dtype=int)
    for w in t["deps"]:
        readers[int(w) & ((1 << P.TT_SHIFT) - 1)] += 1
    assert list(u[:, 6]) == list(readers) and (u[:, 7] == 0).all()
    assert t["s0g"] == int(0 in P.EdgeSync.default(g.plan).grid_stages(g.S))
    assert P.PAY_WORDS == F.PAY_WORDS == 512 and P.TT_SHIFT == F.TT_SHIFT
    assert max(P.build_graph(tp, b_, k).N for tp in (1, 2) for b_ in P.BUCKETS for k in ("ft", "lb")) < 1 << P.TT_SHIFT


def test_index_bounds_formulas():
    g = P.build_graph(1, 16, "lb")
    t = P.tables(g, P.EdgeSync("grid"))
    N = g.N
    bd = P.index_bounds(g, ltot=6, slots=2, pf=8).items
    assert bd["pay_byte"] == (2 * N * 512) * 4 - 1
    assert bd["flg_byte"] == (2 * N * 8) * 4 - 1
    assert bd["dly_elem"] == 6 * N and bd["utab_elem"] == N * P.UCOLS and bd["unit_id"] == N
    assert bd["deps_elem"] == int((t["utab"][:, 3] + t["utab"][:, 4]).max())
    assert bd["wblk_bytes"] == int(t["utab"][:, 1].max()) * 8192
    assert P.index_bounds(g, ltot=6, slots=2, pf=8, dly_layers=3).items["dly_elem"] == 3 * N


def test_lint_exact_i32_boundary():
    # an index whose largest value is exactly 2^31 - 1 is fine; one more overflows
    top = (1 << 31) - 1
    kd, _ = _lint_kernel(lambda b, x, base: b.add(x, b.const_i32(top - 10)))
    assert LL.index_lint(kd, arg_bounds={"x": (0, 10)}) == []
    assert [g.kind for g in LL.index_lint(kd, arg_bounds={"x": (0, 11)})] == ["overflow"]
    kd, _ = _lint_kernel(lambda b, x, base: b.sub(x, b.const_i32(10)))
    assert LL.index_lint(kd, arg_bounds={"x": (-(1 << 31) + 10, 0)}) == []
    assert [g.kind for g in LL.index_lint(kd, arg_bounds={"x": (-(1 << 31) + 9, 0)})] == ["overflow"]
    assert LL.LO == -(1 << 31) and LL.HI == (1 << 31) - 1


def test_bounds_check_boundary():
    b = P.Bounds({"a": (1 << 31) - 1, "b": 1 << 31})
    assert b.check() == [f"b: {1 << 31} > {(1 << 31) - 1}"]
    assert b.check(limit=1 << 31) == []
    assert P.EdgeSync().thr == 0.4 and P.EdgeSync().mode == "mix"


def test_hazard_sgpr_source_is_not_a_write():
    # a VOP2 with an SGPR source operand reads (not writes) it: no hazard
    t = ("0000000000001000 <k>:\n\tv_add_u32_e32 v0, s4, v1  // 000000001000: 0\n"
         "\tglobal_store_dword v0, v2, s[4:5]  // 000000001004: 0\n")
    assert Z.scan_text(t)[0] == []
    # a VOP3 compare with an SGPR destination writes it
    t = ("0000000000001000 <k>:\n\tv_cmp_eq_u32_e64 s[4:5], v1, v2  // 000000001000: 0\n"
         "\tglobal_store_dword v0, v2, s[4:5]  // 000000001008: 0\n")
    assert Z.scan_text(t)[0]
