# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the in-kernel CCL (``kernels.gfx950.qwen38_mono.ccl``):
layout, knobs, the slot-reuse checker and its mutants, the peer-address lint
(real builds clean, synthetic violations caught), the CPU references, and
the LLVM text of every build. The GPU proofs (bitwise rank agreement against
these references, negative controls, stress, timing) run in the
multi-process suite ``rocke.examples.gfx950.ccl``."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono.ccl import collectives as C
from kernels.gfx950.qwen38_mono.ccl import layout as L
from kernels.gfx950.qwen38_mono.ccl import lint as LI
from kernels.gfx950.qwen38_mono.ccl import ref as RF
from kernels.gfx950.qwen38_mono.ccl import war as WA
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"
K = L.CclKnobs


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


# ------------------------------------------------------------------ layout


@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 5, 128])
@pytest.mark.parametrize("cols,dtype", [(2560, "bf16"), (10240, "f32")])
@pytest.mark.parametrize("kn", [K(), K(form="pair"), K(ar="twoshot"), K(homing="dual", slots=3)])
def test_layout_validates(ranks, rows, cols, dtype, kn):
    sp = L.AllReduceSpec(ranks, rows, cols, dtype, kn)
    lay = L.make_layout([sp])
    lay.validate()
    assert lay.nbytes % L.ARENA_PAGE == 0
    for e in lay.exchanges:
        assert e.units == sp.units
        last = e.payload_off(e.slots - 1, e.srcs - 1, e.units - 1) + L.UNIT_BYTES
        assert last <= e.payload + e.payload_bytes
        if e.flags >= 0:
            seen = set()
            for s, src, u in itertools.product(range(e.slots), range(e.srcs), [0, e.units - 1]):
                for cp in range(e.copies):
                    o = e.flag_off(s, src, u, cp)
                    assert o not in seen and e.flags <= o < e.flags + e.flag_bytes
                    seen.add(o)
            if e.copies == 2:
                o0, o1 = e.flag_off(1, 0, 3 % e.units, 0), e.flag_off(1, 0, 3 % e.units, 1)
                assert o1 - o0 == 8192 and ((o0 ^ o1) >> 13) & 1


def test_layout_rejects_overlap_and_bad_shapes():
    with pytest.raises(ValueError):
        L.AllReduceSpec(2, 1, 100, "bf16")  # not whole units
    with pytest.raises(ValueError):
        L.CclKnobs(homing="dual", form="pair")
    with pytest.raises(ValueError):
        L.CclKnobs(slots=1)
    lay = L.make_layout([L.AllReduceSpec(2, 4, 2560)])
    e = lay.exchanges[0]
    bad = L.Layout((e, L.Exchange("x", e.payload, 2, 2, 1)), lay.nbytes)
    with pytest.raises(ValueError):
        bad.validate()


def test_multi_spec_layout_and_names_unique():
    specs = [L.AllReduceSpec(2, 8, 2560), L.AllReduceSpec(2, 8, 2560, knobs=K(ar="twoshot")), L.AllGatherSpec(2, 8, 1280), L.ArgmaxSpec(2, 8)]
    lay = L.make_layout(specs)
    assert {e.name for e in lay.exchanges} == {"0.ar", "1.rs", "1.ag", "2.gather", "3.amax"}
    names = set()
    for kn in [K(), K(form="pair"), K(ar="twoshot"), K(slots=3), K(homing="dual"), K(push_cpol="sc0sc1nt"), K(pause="none"),
               K(header=False), K(hooks=True), K(mutant="ge"), K(waves=4), K(carry="mem"), K(poll_cpol="sc1"), K(poll_cpol="none"),
               K(mutant="any"), K(mutant="deadpush"), K(sync="grid")]:
        names.add(L.AllReduceSpec(2, 8, 2560, knobs=kn).name())
    assert len(names) == 17


# --------------------------------------------------------------------- war


@pytest.mark.parametrize("ranks,waves,units", [(2, 2, 5), (4, 3, 8), (2, 4, 4), (3, 2, 6)])
def test_war_real_schedules_pass(ranks, waves, units):
    for uses in (1, 2, 3):
        assert WA.check(WA.oneshot_schedule(ranks, waves, units, uses, 3, 2)) == []
        assert WA.check(WA.twoshot_schedule(ranks, waves, units, uses, 3, 2)) == []


def test_war_spec_schedules_pass():
    for sp in [L.AllReduceSpec(2, 1, 2560), L.AllReduceSpec(4, 2, 2560, knobs=K(ar="twoshot")), L.AllGatherSpec(4, 2, 1280)]:
        assert WA.check(WA.spec_schedule(sp, waves_total=4)) == []


def test_war_mutants_fail():
    # a consumer moved off the producer's wave
    assert WA.check(WA.oneshot_schedule(2, 3, 6, 3, 2, 2, consumer_shift=1))
    # a ring of one slot
    assert WA.check(WA.oneshot_schedule(2, 2, 4, 3, 2, 1))
    # slots restarting at every launch (no slot-base advance) with an odd use count
    assert WA.check(WA.oneshot_schedule(2, 2, 4, 3, 3, 2, continuous=False))


# -------------------------------------------------------------------- lint


def _all_builds():
    out = []
    for kn in [K(), K(form="pair"), K(ar="twoshot"), K(form="pair", ar="twoshot"), K(homing="dual", hooks=True), K(slots=3),
               K(header=False), K(sync="grid"), K(form="pair", ar="twoshot", sync="grid", header=False)]:
        for n in (1, 2, 4, 8):
            out.append(C.build_allreduce_kernel(L.AllReduceSpec(n, 4, 2560, "bf16", kn)))
    out.append(C.build_allreduce_kernel(L.AllReduceSpec(4, 3, 10240, "f32", K(ar="twoshot", homing="dual"))))
    out.append(C.build_allgather_kernel(L.AllGatherSpec(2, 4, 1280, "bf16")))
    out.append(C.build_allgather_kernel(L.AllGatherSpec(4, 4, 1280, "f32", K(form="pair"))))
    out.append(C.build_argmax_kernel(L.ArgmaxSpec(4, 100)))
    out.append(C.build_allgather_kernel(L.AllGatherSpec(2, 4, 1280, "bf16", K(header=False))))
    out.append(C.build_argmax_kernel(L.ArgmaxSpec(2, 64, K(form="pair", waves=2, header=False))))
    return out


def test_lint_real_builds_clean():
    for k in _all_builds():
        assert LI.lint_peer_access(k) == [], k.name


def _synthetic(kind):
    b = IRBuilder(f"lint_{kind}")
    p = b.param("peer1", I64)
    off = b.const_i32(0)
    if kind == "fatomic":
        ptr = b.int_to_ptr(b.add(p, b.const_i64(64)), PtrType(F32, "global"))
        b.global_atomic_add_f32(ptr, off, b.const_f32(1.0))
    elif kind == "rsrc":
        b.buffer_rsrc(b.int_to_ptr(p, PtrType(I32, "global")), b.const_i32(1024))
    elif kind == "sload":
        b.inline_asm("s_load_dword $0, $1, 0x0\ns_waitcnt lgkmcnt(0)", "=s,s", [p], I32)
    elif kind == "nosc":
        b.inline_asm("global_store_dword $0, $1, $2 sc1", "v,v,s", [off, off, p], None)
    elif kind == "asmatomic":
        b.inline_asm("global_atomic_add $0, $1, $2", "v,v,s", [off, off, p], None)
    elif kind == "irload":
        b.global_load_i32(b.int_to_ptr(p, PtrType(I32, "global")), off)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("kind", ["fatomic", "rsrc", "sload", "nosc", "asmatomic", "irload"])
def test_lint_catches_violations(kind):
    probs = LI.lint_peer_access(_synthetic(kind))
    assert probs, kind
    with pytest.raises(LI.PeerLintError):
        LI.check(_synthetic(kind))


def _synthetic_local(kind):
    """Accesses of the receive arena (kernel param ``arena``)."""
    b = IRBuilder(f"lintl_{kind}")
    a = b.param("arena", I64)
    dst = b.param("dst", PtrType(I32, "global"))
    off = b.const_i32(16)
    addr = b.add(a, b.const_i64(64))
    if kind == "asm_nobits":
        b.inline_asm("global_load_dword $0, $1, off\ns_waitcnt vmcnt(0)", "=&v,v", [addr], I32)
    elif kind == "asm_sc1":
        b.inline_asm("global_load_dword $0, $1, off sc1\ns_waitcnt vmcnt(0)", "=&v,v", [addr], I32)
    elif kind == "irload":
        b.global_load_i32(b.int_to_ptr(addr, PtrType(I32, "global")), off)
    elif kind == "ok_sys":
        v = b.inline_asm("global_load_dword $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)", "=&v,v", [addr], I32)
        b.store(v, dst, off)  # loaded DATA does not taint: storing it with plain IR is fine
    elif kind == "ok_atomic":
        b.inline_asm("global_atomic_add $0, $1, $2", "v,v,s", [off, off, a], None)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("kind,bad,waivable", [("asm_nobits", True, True), ("asm_sc1", True, True), ("irload", True, False),
                                               ("ok_sys", False, False), ("ok_atomic", False, False)])
def test_lint_local_arena(kind, bad, waivable):
    probs = LI.lint_peer_access(_synthetic_local(kind))
    assert bool(probs) == bad, (kind, probs)
    waived = LI.lint_peer_access(_synthetic_local(kind), allow_local_scope=True)
    assert bool(waived) == (bad and not waivable), (kind, waived)


@pytest.mark.parametrize("form", ["line", "pair"])
@pytest.mark.parametrize("pc", ["sc1", "none"])
def test_lint_rejects_unscoped_poll_builds(form, pc):
    """The ``poll_cpol`` lever builds (``none`` is shown broken on hardware)
    fail the lint unless explicitly waived; the default passes."""
    k = C.build_allreduce_kernel(L.AllReduceSpec(2, 8, 2560, "bf16", K(form=form, poll_cpol=pc)))
    assert any("system-scope" in p for p in LI.lint_peer_access(k))
    assert LI.lint_peer_access(k, allow_local_scope=True) == []
    with pytest.raises(LI.PeerLintError):
        LI.check(k)
    assert LI.lint_peer_access(C.build_allreduce_kernel(L.AllReduceSpec(2, 8, 2560, "bf16", K(form=form)))) == []


# --------------------------------------------------------------- references


def test_bf16_rounding():
    x = np.array([1.0, -2.5, 1.00390625, 1.01171875, 3.0e38, 1e-40], np.float32)
    h = RF.f32_to_bf16(x)
    assert h[0] == 0x3F80 and h[1] == 0xC020
    assert h[2] == 0x3F80  # tie (1 + 2^-8) rounds to even
    assert h[3] == 0x3F82  # tie (1 + 3 * 2^-8) rounds up to even
    assert np.isfinite(RF.bf16_to_f32(h)).all()
    assert np.array_equal(RF.f32_to_bf16(RF.bf16_to_f32(h)), h)


def test_rank_order_sum_is_order_sensitive():
    rng = np.random.default_rng(1)
    parts = [rng.standard_normal(4096).astype(np.float32) * (10.0 ** k) for k in range(4)]
    fwd, rev = RF.rank_sum(parts), RF.rank_sum(parts, reverse=True)
    assert not np.array_equal(fwd, rev)  # the revsum mutant must be detectable at N >= 3
    manual = ((parts[0] + parts[1]).astype(np.float32) + parts[2]).astype(np.float32) + parts[3]
    assert np.array_equal(fwd, manual.astype(np.float32))


def test_allreduce_chain_bounded():
    rng = np.random.default_rng(2)
    for dtype in ("bf16", "f32"):
        xs = [RF.f32_to_wire(rng.standard_normal(2560).astype(np.float32), dtype) for _ in range(4)]
        for two in (False, True):
            out, nxt = RF.allreduce_chain(xs, dtype, 96, RF.chain_scales(4), twoshot=two)
            assert np.isfinite(out).all() and np.abs(out).max() < 1e3


def test_argmax_merge_ties_take_lower_global_id():
    vals = np.array([[1.0, 5.0, 2.0], [1.0, 4.0, 3.0], [0.5, 5.0, 3.0]], np.float32)
    ids = np.array([[10, 20, 30], [5, 25, 35], [1, 15, 40]], np.int32)
    v, i = RF.argmax_merge(vals, ids)
    assert list(v) == [1.0, 5.0, 3.0] and list(i) == [5, 15, 35]


def test_lse_and_topk_merge():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 3, 64)).astype(np.float32)
    m = x.max(axis=2)
    s = np.exp(x - m[..., None]).sum(axis=2).astype(np.float32)
    M, S = RF.lse_merge(m, s)
    full = x.transpose(1, 0, 2).reshape(3, -1)
    assert np.allclose(M + np.log(S), np.log(np.exp(full).sum(axis=1)), rtol=1e-5)
    ids = np.arange(4 * 64).reshape(4, 1, 64).repeat(3, axis=1)
    vv, ii = RF.topk_merge(x, ids, 5)
    for row in range(3):
        order = np.lexsort((ids.transpose(1, 0, 2).reshape(3, -1)[row], -full[row]))[:5]
        assert list(ii[row]) == list(ids.transpose(1, 0, 2).reshape(3, -1)[row][order])


def test_delay_hash_host_twin_in_range():
    vals = {RF.delay_units(12345, gw, u, p, 40) for gw in range(16) for u in range(8) for p in (0, 1)}
    assert max(vals) <= 40 and len(vals) > 10


# ------------------------------------------------------------------ builds


@pytest.mark.parametrize("kn", [K(), K(form="pair"), K(ar="twoshot"), K(form="pair", ar="twoshot"), K(homing="dual"),
                                K(header=True, hooks=True), K(mutant="ge"), K(mutant="slot"), K(mutant="revsum"), K(push_cpol="sc0sc1nt"),
                                K(poll_cpol="none"), K(form="pair", poll_cpol="sc1"), K(mutant="any"), K(form="pair", mutant="deadpush"),
                                K(sync="grid"), K(header=False)])
def test_allreduce_llvm_text(kn):
    sp = L.AllReduceSpec(4, 3, 2560, "bf16", kn)
    k = C.build_allreduce_kernel(sp)
    assert C.backend_for(k) == "python"
    ll = _ll(k)
    assert "s_memrealtime" in ll and "s_addc_u32" in ll  # deadlines, tag wrap
    assert "atomicrmw" not in ll  # no IR atomics at all (local counters are asm, agent scope)
    names = [p.name for p in k.params]
    assert names[:6] == ["peer0", "peer1", "peer2", "peer3", "ctrl", "arena"]
    if kn.form == "line":
        assert "global_store_dwordx4 $0, $1, $2 " in ll and "global_store_dword $0, $1, off sc0 sc1" in ll
        assert ("sc0 sc1 nt" in ll) == (kn.push_cpol == "sc0sc1nt")
    else:
        assert "global_load_dwordx2" in ll and "v_lshl_add_u64" in ll
    if kn.mutant == "ge":
        assert "v_cmp_lt_u" in ll
    if kn.poll_cpol == "none":
        assert "global_load_dword $0, $6, off \\0A" in ll  # the flag poll without cache bits
    if kn.poll_cpol == "sc1":
        assert "offset:8 sc1\\0A" in ll
    # every wide asm store is followed by the store-data hazard pad
    assert "global_store_dwordx4" in ll and "\\0As_nop 1" in ll
    if kn.hooks:
        assert "htab" in names
    assert ("i32 -5" in ll) == kn.header  # CCL_MISMATCH is recorded only by the header check
    assert ("smem_pool." in ll) and ("store i32 1, ptr addrspace(3)" in ll)  # the per-wave dead flag
    assert ("load i32, ptr addrspace(3)" in ll) == (kn.mutant != "deadpush")  # pushes guarded by it


def test_header_default_on():
    """DESIGN-CCL 4.10: the launch header is on by default (off only in
    microbenchmarks, named ``nohdr``) for every collective."""
    assert K().header and L.select_knobs(2, 4096).header
    assert "hdr" not in L.AllReduceSpec(2, 4, 2560).name()
    assert L.AllReduceSpec(2, 4, 2560, knobs=K(header=False)).name().endswith("_nohdr")
    for on, off in [
        (L.AllReduceSpec(2, 4, 2560), L.AllReduceSpec(2, 4, 2560, knobs=K(header=False))),
        (L.AllGatherSpec(2, 4, 1280), L.AllGatherSpec(2, 4, 1280, knobs=K(header=False))),
        (L.ArgmaxSpec(2, 64), L.ArgmaxSpec(2, 64, K(form="pair", waves=2, header=False))),
    ]:
        build = {L.AllReduceSpec: C.build_allreduce_kernel, L.AllGatherSpec: C.build_allgather_kernel, L.ArgmaxSpec: C.build_argmax_kernel}[type(on)]
        assert "i32 -5" in _ll(build(on)) and "i32 -5" not in _ll(build(off)), type(on).__name__
    assert on.header_hash() != L.ArgmaxSpec(2, 65).header_hash()


@pytest.mark.parametrize("form", ["line", "pair"])
def test_acceptance_mutants_have_no_ahead_check(form):
    """``ge`` / ``any`` accept without the tag-ahead check, so the stale-tag
    suite's must-fail outcome is deterministic (wrong data, no error)."""
    ahead = "v_add_u32_e32 $4, -1, $4" if form == "line" else "v_lshl_add_u64"
    base = _ll(C.build_allreduce_kernel(L.AllReduceSpec(2, 4, 2560, "bf16", K(form=form, header=False))))
    assert ahead in base
    for mut in ("ge", "any"):
        ll = _ll(C.build_allreduce_kernel(L.AllReduceSpec(2, 4, 2560, "bf16", K(form=form, header=False, mutant=mut))))
        assert ahead not in ll, mut
    any_line = _ll(C.build_allreduce_kernel(L.AllReduceSpec(2, 4, 2560, "bf16", K(header=False, mutant="any"))))
    assert "v_cmp_eq_u32_e64 $1, 0, $0" in any_line


def test_grid_sync_build():
    sp = L.AllReduceSpec(2, 4, 2560, "bf16", K(sync="grid"))
    assert not sp.reg_carry  # the grid-synced chain carries through xbuf
    ll = _ll(C.build_allreduce_kernel(sp))
    assert "global_atomic_add $0, $1, $2, $3 sc0 sc1" in ll  # system-scope counter add, waited
    assert "v_cmp_lt_u32_e64 $1, $0, $7" in ll  # counter wait
    assert "i32 -2" in ll  # GRID_SYNC_TIMEOUT
    with pytest.raises(ValueError):
        L.AllGatherSpec(2, 4, 1280, knobs=K(sync="grid"))


def test_pack_kernargs_alignment():
    k = C.build_allreduce_kernel(L.AllReduceSpec(2, 1, 2560))
    vals = {p.name: 0 for p in k.params}
    vals["scale"] = 0.5
    buf = C.pack_kernargs(k, vals)
    # 2 peers + ctrl + arena + 4 pointers (8 B) + 6 i32 + f32
    assert len(buf) == 8 * 8 + 7 * 4


def test_ccl_import_leaves_helper_lowering_unchanged():
    """Importing the CCL family does not change a mirrored mailbox build."""
    import importlib

    from rocke.helpers import mailbox as M

    def toy():
        b = IRBuilder("toy_mb")
        box = b.param("box", PtrType(I32, "global"))
        mb = M.mailbox_from_ptr(b, box, b.const_i32(4096), M.MailboxKnobs())
        M.put(b, mb, b.const_i32(0), b.const_i32(5), b.const_i32(1))
        b.ret()
        return b.kernel

    before = _ll(toy())
    importlib.reload(C)
    after = _ll(toy())
    assert before == after


def test_register_carry_selection():
    small = L.AllReduceSpec(2, 4, 2560, "bf16")
    assert small.reg_carry and small.units <= small.grid() * small.knobs.waves
    big = L.AllReduceSpec(2, 128, 10240, "f32")
    assert not big.reg_carry  # more units than waves: the xbuf chain
    assert not L.AllReduceSpec(2, 4, 2560, "bf16", K(carry="mem")).reg_carry
    for sp in (small, big, L.AllReduceSpec(2, 4, 2560, "bf16", K(carry="mem", form="pair", ar="twoshot"))):
        ll = _ll(C.build_allreduce_kernel(sp))
        assert ("cclxv" in ll) == sp.reg_carry


def test_auto_selection_follows_measured_crossover():
    # shape="chain": the best-case persistent chain (each wave waits for its own unit)
    ch = {"shape": "chain"}
    assert L.select_knobs(2, 5 << 10, **ch) == K(form="pair")
    assert L.select_knobs(2, 160 << 10, **ch) == K(form="line")
    assert L.select_knobs(4, 5 << 10, **ch) == K(form="pair", ar="twoshot")
    assert L.select_knobs(4, 320 << 10, **ch) == K(form="line", ar="twoshot")
    assert L.select_knobs(3, 40 << 10, **ch) == K(form="pair")
    assert L.select_knobs(3, 160 << 10, **ch) == K(form="line")
    assert L.select_knobs(3, 640 << 10, **ch) == K(form="line", ar="twoshot")
    assert L.select_knobs(8, 1 << 20, **ch) == K(form="line", ar="twoshot")
    assert L.select_knobs(2, 160 << 10, calibrated=True, **ch) == K(form="line", homing="dual")
    assert L.select_knobs(2, 640 << 10, calibrated=True, **ch) == K(form="line")
    assert L.select_knobs(2, 5 << 10, calibrated=True, **ch) == K(form="pair")
    sp = L.auto_allreduce_spec(2, 128, 10240, "bf16", shape="chain")
    assert sp.knobs.form == "line" and sp.knobs.ar == "oneshot"
    # shape="layer" (default): the grid-synced chain (a consumer that needs the whole vector)
    assert L.select_knobs(2, 80 << 10) == K(form="pair")
    assert L.select_knobs(2, 320 << 10) == K(form="line")
    assert L.select_knobs(2, 640 << 10) == K(form="line", ar="twoshot")
    assert L.select_knobs(3, 80 << 10) == K(form="pair")
    assert L.select_knobs(3, 160 << 10) == K(form="pair", ar="twoshot")
    assert L.select_knobs(3, 320 << 10) == K(form="line", ar="twoshot")
    assert L.select_knobs(4, 20 << 10) == K(form="pair")
    assert L.select_knobs(4, 40 << 10) == K(form="pair", ar="twoshot")
    assert L.select_knobs(4, 160 << 10) == K(form="pair", ar="twoshot")
    assert L.select_knobs(4, 320 << 10) == K(form="line", ar="twoshot")
    assert L.select_knobs(8, 10 << 10) == K(form="pair")
    assert L.select_knobs(2, 160 << 10, calibrated=True) == K(form="line", homing="dual")
    assert L.auto_allreduce_spec(2, 128, 10240, "bf16").knobs == K(form="line", ar="twoshot")
    with pytest.raises(ValueError):
        L.select_knobs(2, 4096, shape="best")
