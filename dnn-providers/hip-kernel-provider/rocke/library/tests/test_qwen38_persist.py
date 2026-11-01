# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The true-mono persistent platform (``kernels.gfx950.qwen38_mono.persist``,
WP1): CPU checks.

* knobs (``PersistKnobs``, ``EdgeKnobs`` / ``EdgePlan``, ``RustScope``):
  defaults, refusals, suffixes;
* the ``tagged_poll`` reload-fix op variant (``reload="launder"``) and the
  ``irf`` mailbox: the original op's IR is unchanged, the variant re-derives
  the load address per retry;
* parity: every P4 / P5 probe kernel rebuilt on the platform primitives
  lowers to byte-identical LLVM IR (and the patching is not vacuous);
* the static checks: KV acquire (P5's positive / negative classification),
  spins that never reload (P2), exec-join, fused staging over a polled mailbox
  (P8: the fused MoE's ``ys`` over ``ss``);
* framework hooks: defaults byte-identical, R4 / R5 / R2, edge plans
  (deadline polls, refusal of unconsumed emitter-level knobs), Rust scopes
  invisible to the LLVM lowering;
* the loop driver and the test chain (``ptoy``) lower in every mode; the
  microbench kernels lower.

The hardware gates run in ``hackathon-prep/qwen38/truemono/WP1``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono import mailbox_ir  # noqa: F401  (registers "ir" / "irf")
from kernels.gfx950.qwen38_mono.framework import FrameworkKnobs
from kernels.gfx950.qwen38_mono.persist import knobs as K
from kernels.gfx950.qwen38_mono.persist import kv as KV
from kernels.gfx950.qwen38_mono.persist import lint as L
from kernels.gfx950.qwen38_mono.persist import micro as MI
from kernels.gfx950.qwen38_mono.persist import parity as PP
from kernels.gfx950.qwen38_mono.persist import ptoy as T
from kernels.gfx950.qwen38_mono.persist import sync as PS
from rocke.core.ir import CPOL_SC1, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import mailbox as M

FLAVOR = "llvm22"


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


# --------------------------------------------------------------------- knobs


def test_persist_knob_defaults_are_d0():
    d = K.PersistKnobs()
    assert (d.inline, d.args, d.fresh, d.machine_licm, d.wgid) == ("always", "const", "all", "off", "lds")
    assert (d.arrivals, d.stripes, d.ticket, d.kv_bulk, d.xhold) == ("cta", 8, "xcd", "sc1", "xpre")
    assert d.suffix() == "" and K.D0_DEFAULTS == d
    assert K.LEGACY.suffix() == "arkernarg_frids_mlon_wgsgpr"


@pytest.mark.parametrize(
    "kw",
    [dict(args="smem"), dict(stripes=3), dict(unroll=2), dict(deadline=-1), dict(arrivals="lane"), dict(kv_bulk="x")],
)
def test_persist_knobs_refuse(kw):
    with pytest.raises(ValueError):
        K.PersistKnobs(**kw)


def test_inline_never_is_refused_explicitly():
    with pytest.raises(NotImplementedError, match="P6"):
        K.PersistKnobs(inline="never").check()


def test_edge_knobs_and_plan():
    with pytest.raises(ValueError):
        K.EdgeKnobs(protocol="xcd_local")  # needs an XCD-affine placement
    with pytest.raises(ValueError):
        K.EdgeKnobs(poll="deadline")  # needs a bound
    with pytest.raises(ValueError):
        K.EdgeKnobs(placement="island")  # needs CUs
    e = K.EdgeKnobs(protocol="xcd_local", placement="xcd_affine")
    assert not e.needs_release and K.EdgeKnobs(protocol="bulk").needs_release
    assert K.EdgeKnobs(poll="irf").mailbox_impl == "irf" and K.EdgeKnobs(poll="deadline", deadline=9).mailbox_impl == "asm"
    plan = K.EdgePlan.of({"m5>a1": K.EdgeKnobs(poll="irf"), "*>c1": K.EdgeKnobs(pause="s_sleep1")})
    assert plan.edge("m5", "a1").poll == "irf"
    assert plan.edge("b3", "c1").pause == "s_sleep1"
    assert plan.edge("a1", "a2") == K.EdgeKnobs()
    plan.validate({"m5", "a1", "c1", "b3"})
    with pytest.raises(ValueError, match="unknown"):
        plan.validate({"m5", "a1"})
    assert K.EdgePlan.of().is_default and not plan.is_default
    with pytest.raises(TypeError):
        K.EdgePlan.of({"a>b": "pairs"})


def test_rust_scope_sets_and_forbidden_levers():
    sc = K.RustScope.of({"b3": "b3", "m3": "moe_r1"})
    assert "nofence" in sc.levers("m3") and "loadorder" in sc.levers("b3")
    assert sc.levers("a1") is None
    with pytest.raises(ValueError, match="measured losses"):
        K.RustScope.of({"head": "nofence+minmax"}, {"head": "head"})
    assert K.RustScope.of({"x": "minmax+selxor"}).levers("x") == ("minmax", "selxor")


# ------------------------------------------------------ tagged_poll reload fix


def _poll_kernel(reload: str):
    b = IRBuilder(f"tp_{reload or 'orig'}")
    p = b.param("p", PtrType(I32, "global"))
    tag = b.readfirstlane(b.param("tag", I32))
    rs = b.buffer_rsrc(p, b.const_i32(1 << 16))
    (v,) = b.tagged_poll(tag, [(rs, b.mul(b.lane_id(), b.const_i32(8)), 2)], reload=reload)
    b.global_store(p, b.lane_id(), b.vec_extract(v, 0))
    b.ret()
    return b.kernel


def test_tagged_poll_original_op_is_unchanged():
    k = _poll_kernel("")
    op = [o for o in L.ops_in_order(k) if o.name == "tile.tagged_poll"][0]
    assert "reload" not in op.attrs  # the original op carries no new attribute
    ll = _ll(k)
    assert 'call void asm sideeffect "s_nop 0", ""()' in ll
    assert "tagged_poll reload" not in ll


def test_tagged_poll_launder_variant():
    ll = _ll(_poll_kernel("launder"))
    assert '"; tagged_poll reload $0", "=v,0"' in ll
    assert 'call void asm sideeffect "s_nop 0", "~{memory}"()' in ll
    # the load uses the laundered offset, inside the loop block
    loop = ll[ll.index("poll.loop") :]
    loop = loop[: loop.index("poll.retry")]
    assert "tagged_poll reload" in loop and "raw.ptr.buffer.load" in loop
    with pytest.raises(ValueError):
        _poll_kernel("volatile")


def test_irf_mailbox_impl():
    kn = M.MailboxKnobs(impl="irf")
    assert kn.suffix() == "mbirf"
    b = IRBuilder("irf_poll")
    p = b.param("p", PtrType(I32, "global"))
    mb = M.mailbox_from_ptr(b, p, b.const_i32(1 << 16), kn)
    (vs,) = M.poll(b, [(mb, b.lane_id(), 1)], b.const_i32(5), kn)
    b.global_store(p, b.lane_id(), vs[0])
    b.ret()
    assert "tagged_poll reload" in _ll(b.kernel)
    assert mailbox_ir.backend_for(b.kernel) == "python"


# -------------------------------------------------------------------- parity


def _p4_specs():
    S = PP.p4_module()
    kn2 = M.MailboxKnobs(impl="ir", numa_copies=2)
    out = [
        ("build_counter_kernel", S.CounterSpec()),
        ("build_counter_kernel", S.CounterSpec(pause="s_sleep1")),
        ("build_a2a_kernel", S.A2ASpec(knobs=kn2)),
        ("build_a2a_kernel", S.A2ASpec(knobs=kn2, producers=256)),
        ("build_a2a_kernel", S.A2ASpec(poll="deadline")),
        ("build_a2a_kernel", S.A2ASpec(knobs=kn2, mutant="skiplast")),
        ("build_ticket_kernel", S.TicketSpec()),
        ("build_ticket_kernel", S.TicketSpec(start="xcd")),
        ("build_boundary_kernel", S.BoundarySpec(rows=4)),
    ]
    for r in (1, 4, 16):
        out += [("build_xl_kernel", S.XlSpec(rows=r, knobs=kn2)),
                ("build_xl_kernel", S.XlSpec(rows=r, mode="bulk", knobs=kn2)),
                ("build_xl_kernel", S.XlSpec(rows=r, mutant="tagoff", knobs=kn2))]  # fmt: skip
    return out


@pytest.mark.parametrize("fn,spec", _p4_specs())
def test_p4_parity_llvm_identical(fn, spec):
    S = PP.p4_module()
    ref = _ll(getattr(S, fn)(spec))
    with PP.platform_p4() as m:
        got = _ll(getattr(m, fn)(spec))
    assert got == ref


def test_p4_parity_patch_is_not_vacuous():
    """A different primitive in the patch map changes the IR: the builders
    really use the patched module attributes."""
    S = PP.p4_module()
    ref = _ll(S.build_counter_kernel(S.CounterSpec()))

    def other(b, voff, base, expected, maxit, pause="s_nop"):
        return PS.poll_counter(b, voff, base, expected, maxit, "s_sleep1")

    with PP.patched(S, {"poll_counter": other}):
        got = _ll(S.build_counter_kernel(S.CounterSpec()))
    assert got != ref
    assert S.poll_counter is not other  # restored


@pytest.mark.parametrize("w", ["wbl2", "sc1", "plain"])
@pytest.mark.parametrize("r", ["inv", "inv0", "sc1", "none"])
def test_p5_kv_parity_llvm_identical(w, r):
    p5 = PP.p5_module()
    sp = p5.KvSpec(wmode=w, rmode=r)
    p5._DELAYS[0] = 0
    ref = _ll(p5.build_kv_kernel(sp))
    p5._DELAYS[0] = 0
    assert _ll(PP.build_kv_kernel(sp)) == ref


@pytest.mark.parametrize("r", ["reg", "smem", "smem_dinv", "smem_dinv_inv", "vec_inv", "sbuf_dinv_inv"])
def test_p5_scalar_parity_llvm_identical(r):
    p5 = PP.p5_module()
    sp = p5.ScSpec(rmode=r)
    p5._DELAYS[0] = 0
    ref = _ll(p5.build_scalar_kernel(sp))
    p5._DELAYS[0] = 0
    assert _ll(PP.build_scalar_kernel(sp)) == ref
    assert p5._scalar_read.__name__ == "_scalar_read"  # restored after the context


# ------------------------------------------------------------- static checks


@pytest.mark.parametrize("w", ["wbl2", "sc1", "plain"])
@pytest.mark.parametrize("r", ["inv", "inv0", "sc1", "none"])
def test_kv_lint_matches_p5_classification(w, r):
    """Clean exactly for P5's memory-model positives; every negative control
    is flagged (writer: plain; reader: none / inv0)."""
    p5 = PP.p5_module()
    sp = p5.KvSpec(wmode=w, rmode=r)
    f = L.check_kv_acquire(PP.build_kv_kernel(sp))
    assert (not f) == sp.positive, [str(x) for x in f]
    rules = {x.rule for x in f}
    assert ("kv_store_unreleased" in rules) == (w == "plain")
    assert ("kv_load_unacquired" in rules) == (r in ("none", "inv0"))


@pytest.mark.parametrize(
    "r,rules",
    [("reg", set()), ("smem", {"sread_stale_scalar"}), ("smem_dinv", {"sread_l2_unacquired"}), ("smem_dinv_inv", set())],
)
def test_kv_lint_scalar_rereads(r, rules):
    p5 = PP.p5_module()
    f = L.check_kv_acquire(PP.build_scalar_kernel(p5.ScSpec(rmode=r)))
    assert {x.rule for x in f} == rules
    if r == "smem_dinv":
        assert not L.check_kv_acquire(PP.build_scalar_kernel(p5.ScSpec(rmode=r)), strict=False)


def test_kv_lint_unmarked_kernel():
    p5 = PP.p5_module()
    f = L.check_kv_acquire(p5.build_kv_kernel(p5.KvSpec()))
    assert [x.rule for x in f] == ["unmarked"]


def test_kv_markers_are_invisible_to_lowering():
    b = IRBuilder("mk")
    p = b.param("p", PtrType(I32, "global"))
    rs = b.buffer_rsrc(p, b.const_i32(4096))
    with KV.marked(b, "launch"):
        v = b.buffer_load_cp(rs, b.const_i32(0), b.const_i32(0), dwords=1, aux=CPOL_SC1)
    b.global_store(p, b.const_i32(1), v)
    b.ret()
    b2 = IRBuilder("mk")
    p2 = b2.param("p", PtrType(I32, "global"))
    rs2 = b2.buffer_rsrc(p2, b2.const_i32(4096))
    v2 = b2.buffer_load_cp(rs2, b2.const_i32(0), b2.const_i32(0), dwords=1, aux=CPOL_SC1)
    b2.global_store(p2, b2.const_i32(1), v2)
    b2.ret()
    assert _ll(b.kernel) == _ll(b2.kernel)
    assert "_op" not in b.__dict__  # the marker context restored the builder


_SPIN = """
0000000000001900 <k>:
\ts_load_dword s4, s[0:1], 0x0                              // 000000001900: C0020100 00000000
\tv_cmp_ne_u32_e32 vcc, s4, v1                               // 000000001908: 7D9A0204
\ts_and_saveexec_b64 s[4:5], vcc                             // 00000000190C: BE84206A
\ts_and_b64 vcc, exec, 0                                     // 000000001910: 86EA807E
\ts_nop 0                                                    // 000000001914: BF800000
\ts_mov_b64 vcc, vcc                                         // 000000001918: BEEA016A
\ts_cbranch_vccz 65533                                       // 00000000191C: BF86FFFD
\ts_endpgm                                                   // 000000001920: BF810000
"""
_POLL = """
0000000000001900 <k>:
\tbuffer_load_dword v42, v42, s[28:31], 0 offen offset:4 sc1 // 000000001900: E0505004 80072A2A
\ts_waitcnt vmcnt(0)                                         // 000000001908: BF8C0F70
\tv_cmp_eq_u32_e32 vcc, s8, v42                              // 00000000190C: 7D945408
\ts_andn2_b64 exec, exec, s[6:7]                             // 000000001910: 89FE067E
\ts_cbranch_execnz 65531                                     // 000000001914: BF89FFFB
\ts_endpgm                                                   // 000000001918: BF810000
"""
_DELAY = """
0000000000001900 <k>:
\ts_add_i32 s9, s9, -1                                       // 000000001900: 8109C109
\ts_cmp_lg_u32 s9, 0                                         // 000000001904: BF078009
\ts_sleep 8                                                  // 000000001908: BF8E0008
\ts_cbranch_scc1 65532                                       // 00000000190C: BF85FFFC
\ts_endpgm                                                   // 000000001910: BF810000
"""


def test_spin_lint():
    f = L.spin_without_reload(_SPIN)
    assert len(f) == 1 and "0x1914..0x191c" in f[0].detail
    assert not L.spin_without_reload(_POLL)
    assert not L.spin_without_reload(_DELAY)  # a counted delay loop terminates


def test_staging_lint_flags_p8_and_clears_a_private_region():
    from kernels.gfx950.qwen38_mono import atom_layout as AL

    regions = dict(AL.SC)
    rbytes = {n: (p * AL.PAIR + 255) // 256 * 256 for n, p in AL.REGIONS.items()}
    polled = L.polled_regions(AL._FLOW)
    assert "ss" in polled
    f = L.check_staging_overlap(L.fused_moe_staging(1), regions, rbytes, AL.SCRATCH_BYTES, AL.MAX_ROWS, polled)
    assert f and all("'ss'" in x.detail for x in f)  # P8: the ys staging aliases ss
    moved = [L.Staging("ys@tail", AL.SCRATCH_BYTES * AL.MAX_ROWS, 4096)]
    assert not L.check_staging_overlap(moved, regions, rbytes, AL.SCRATCH_BYTES, AL.MAX_ROWS, polled)
    # an unpolled region is not a hazard
    assert not L.check_staging_overlap(L.fused_moe_staging(1), regions, rbytes, AL.SCRATCH_BYTES, AL.MAX_ROWS, {"x8"})


def test_resource_gate_meta_scan():
    # msgpack: fixstr key + positive fixint / uint8 values
    blob = (bytes([0xA0 | 17]) + b".vgpr_spill_count" + bytes([0]) + bytes([0xA0 | 27])
            + b".private_segment_fixed_size" + bytes([0xCC, 200]))  # fmt: skip
    ok, rep = L.resource_gate(blob)
    assert rep["meta"]["private_segment_fixed_size"] == 200 and not ok


# --------------------------------------------------------- framework hooks


def test_framework_default_knobs_unchanged():
    k = FrameworkKnobs()
    assert k.persist is None and k.edges is None and k.rust_scope is None and k.err_arg == ""
    assert k.suffix() == ""


def test_static_build_hooks():
    base = _ll(T.build_static())
    # an empty edge plan is the default build (same name, same bytes)
    assert _ll(T.build_static(FrameworkKnobs(edges=K.EdgePlan.of()))) == base
    # rust scopes are invisible to the LLVM lowering
    ks = T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3"})))
    assert _ll(ks) == base
    assert ks.attrs["direct_scopes"].startswith("g:dppcombine")
    assert any(op.attrs.get("rocke.scope") == "g" for op in L.ops_in_order(ks))
    # persist: R4 attribute -> compile option, R5 LDS word, R2 fresh ids
    kp = T.build_static(FrameworkKnobs(persist=K.PersistKnobs(args="kernarg")))
    assert kp.attrs["machine_licm"] == "off"
    from rocke.helpers.compile import _comgr_options_for_kernel

    assert _comgr_options_for_kernel(kp)[-2:] == ["-mllvm", "-disable-machine-licm"]
    assert "-disable-machine-licm" not in _comgr_options_for_kernel(T.build_static())
    llp = _ll(kp)
    assert llp != base and "ds_read_b32" in llp and "ds_read_b32" not in base  # R5: wgid via LDS
    with pytest.raises(ValueError, match="kernarg"):
        T.build_static(FrameworkKnobs(persist=K.PersistKnobs()))  # args='const' is the loop driver's


def test_edge_plan_deadline_and_irf():
    dl = K.EdgePlan.of(default=K.EdgeKnobs(poll="deadline", deadline=1 << 20))
    ll = _ll(T.build_static(FrameworkKnobs(edges=dl, err_arg="err")))
    assert "global_atomic_or" in ll  # the launch-fatal word on the timeout path
    with pytest.raises(ValueError, match="err_arg"):
        T.build_static(FrameworkKnobs(edges=dl))
    lli = _ll(T.build_static(FrameworkKnobs(edges=K.EdgePlan.of(default=K.EdgeKnobs(poll="irf")))))
    assert "tagged_poll reload" in lli


def test_edge_plan_refuses_unconsumed_emitter_knobs():
    plan = K.EdgePlan.of({"g>r": K.EdgeKnobs(placement="xcd_affine")})
    with pytest.raises(ValueError, match="not consumed"):
        T.build_static(FrameworkKnobs(edges=plan))
    with pytest.raises(ValueError, match="unknown"):
        T.build_static(FrameworkKnobs(edges=K.EdgePlan.of({"g>q9": K.EdgeKnobs()})))


# ------------------------------------------------------------- loop driver


@pytest.mark.parametrize(
    "pk",
    [K.D0_DEFAULTS, K.PersistKnobs(args="lds"), K.PersistKnobs(fresh="ids", machine_licm="on", wgid="sgpr"),
     K.PersistKnobs(loop_carry=False)],  # fmt: skip
)
def test_loop_driver_lowers(pk):
    k = T.build_loop(FrameworkKnobs(persist=pk))
    ll = _ll(k)
    assert "_loop" in k.name and "s_load_dwordx2" in ll or pk.args == "lds"
    assert ("machine_licm" in k.attrs) == (pk.machine_licm == "off")


def test_loop_driver_refusals():
    with pytest.raises(ValueError):
        T.build_loop(FrameworkKnobs(persist=K.PersistKnobs(args="kernarg")))
    with pytest.raises(NotImplementedError):
        T.build_loop(FrameworkKnobs(persist=K.D0_DEFAULTS, stamps=True))


def test_loop_and_static_mutants_lower():
    dl = K.EdgePlan.of(default=K.EdgeKnobs(poll="deadline", deadline=1 << 20))
    for k in (T.build_static(FrameworkKnobs(edges=dl, err_arg="err"), mutant="prevtag"),
              T.build_loop(FrameworkKnobs(persist=K.D0_DEFAULTS, edges=dl, err_arg="err"), mutant="prevtag")):  # fmt: skip
        assert "mutprevtag" in k.name and "global_atomic_or" in _ll(k)


def test_ptoy_reference_is_deterministic_and_layer_dependent():
    rng = np.random.default_rng(0)
    x0 = rng.integers(-1000, 1000, size=T.HX).astype(np.int32)
    ws = [rng.integers(-50, 50, size=(T.G_TASKS, T.HX)).astype(np.int32) for _ in range(3)]
    a, b = T.reference(x0, ws), T.reference(x0, ws)
    assert all((u == v).all() for u, v in zip(a, b))
    assert not (a[0] == a[1]).all()
    assert T.launch_tag(5, 0) == 5 * M.LAYER_SLOTS + 1


def test_ptoy_wrongw_mutant_changes_only_the_weight_row():
    """``wrongw`` (WP1 fix round): the numerics mutant. The clean builds are
    unchanged by the option, the mutant differs in the weight index only, and
    the host twin of the mutant differs from the reference on every layer."""
    clean_s, clean_l = _ll(T.build_static()), _ll(T.build_loop(FrameworkKnobs(persist=K.D0_DEFAULTS)))
    assert _ll(T.build_static(mutant="none")) == clean_s
    for clean, k in ((clean_s, T.build_static(mutant="wrongw")),
                     (clean_l, T.build_loop(FrameworkKnobs(persist=K.D0_DEFAULTS), mutant="wrongw"))):  # fmt: skip
        ll = _ll(k)
        assert "mutwrongw" in k.name and ll != clean
        assert ("srem" in ll or "urem" in ll) and not ("srem" in clean or "urem" in clean)
    with pytest.raises(ValueError):
        T.build_static(mutant="nope")
    rng = np.random.default_rng(1)
    x0 = rng.integers(-1000, 1000, size=T.HX).astype(np.int32)
    ws = [rng.integers(-50, 50, size=(T.G_TASKS, T.HX)).astype(np.int32) for _ in range(4)]
    ref, mut = T.reference(x0, ws), T.reference(x0, [np.roll(w, -1, axis=0) for w in ws])
    assert all(not (a == b).all() for a, b in zip(ref, mut))


def test_check_launch_raises_on_the_fatal_word_only():
    e = np.zeros(PS.ERR_WORDS, np.int32)
    PS.check_launch(e)
    e[PS.ERR_TIMEOUT] = 3  # timeouts without the fatal bit: counted, not fatal by themselves
    PS.check_launch(e)
    e[PS.ERR_FATAL] = PS.FATAL_BITS["poll_timeout"]
    with pytest.raises(PS.LaunchFatal, match="poll_timeout") as ei:
        PS.check_launch(e, "layer 3")
    assert ei.value.reasons == ["poll_timeout"] and ei.value.timeouts == 3 and "layer 3" in str(ei.value)
    e[PS.ERR_FATAL] = PS.FATAL_BITS["aborted"]
    with pytest.raises(PS.LaunchFatal, match="aborted"):
        PS.check_launch(e)


def test_require_launch_fatal_path_b_rule():
    dl = K.EdgePlan.of(default=K.EdgeKnobs(poll="deadline", deadline=1 << 16))
    K.require_launch_fatal(FrameworkKnobs(edges=dl, err_arg="err"))
    with pytest.raises(ValueError, match="err_arg"):
        K.require_launch_fatal(FrameworkKnobs(edges=dl))
    with pytest.raises(ValueError, match="edges"):
        K.require_launch_fatal(FrameworkKnobs(err_arg="err"))
    with pytest.raises(ValueError, match="default"):
        K.require_launch_fatal(FrameworkKnobs(edges=K.EdgePlan.of(default=K.EdgeKnobs(poll="irf")), err_arg="err"))
    mixed = K.EdgePlan.of({"g>r": K.EdgeKnobs(poll="asm")}, default=K.EdgeKnobs(poll="deadline", deadline=9))
    with pytest.raises(ValueError, match="g>r"):
        K.require_launch_fatal(FrameworkKnobs(edges=mixed, err_arg="err"))


# ---------------------------------------------------------------- micro


@pytest.mark.parametrize(
    "spec", [MI.GsSpec(), MI.GsSpec(arrivals="wave"), MI.GsSpec(stripes=1), MI.GsSpec(mutant="early")]
)
def test_gs_kernel_lowers(spec):
    ll = _ll(MI.build_gs_kernel(spec))
    assert "global_atomic_or" in ll and "global_atomic_add" in ll and "ds_write_b32" in ll


def test_grid_sync_layout():
    gs = PS.GridSync()
    assert gs.words == 64 and gs.count == 4 and gs.nbytes == 64 * PS.LINE
    assert PS.GridSync(arrivals="wave").count == 32
    with pytest.raises(ValueError):
        PS.GridSync(grid=100)
    assert PS.fatal_reasons(PS.FATAL_BITS["sync_timeout"] | PS.FATAL_BITS["poll_timeout"]) == [
        "poll_timeout", "sync_timeout"]  # fmt: skip


def test_ticket_fatal_kernel_lowers():
    ll = _ll(MI.build_ticket_fatal_kernel())
    assert "global_atomic_or" in ll and "v_readlane_b32" in ll


def _consuming_build(knobs, *, edge=None, persist=None):
    """The ptoy static unit with an emitter that asks for an edge / a persist knob."""
    from kernels.gfx950.qwen38_mono.framework import StageSpec, build_layer_kernel

    tb = T.table(False)
    g_emit = T._emit_g("none")

    def emit_g(b, ctx, p, t):
        if edge is not None:
            ctx.edge(*edge)
        if persist is not None:
            ctx.persist_knob(persist)
        g_emit(b, ctx, p, t)

    specs = [StageSpec(tb.stage("g"), emit_g), StageSpec(tb.stage("r"), T._emit_r)]
    return build_layer_kernel("q38pt_consume", specs, tb, T.layout(1), knobs=knobs, args=T.ARGS)


def test_emitter_consumption_of_edge_and_persist_knobs():
    plan = K.EdgePlan.of({"r>g": K.EdgeKnobs(placement="xcd_affine")})
    with pytest.raises(ValueError, match="not consumed"):
        _consuming_build(FrameworkKnobs(edges=plan))
    _consuming_build(FrameworkKnobs(edges=plan), edge=("r", "g"))  # the emitter asked: accepted
    pk = K.PersistKnobs(args="kernarg", kv_bulk="bulk")
    with pytest.raises(ValueError, match="kv_bulk"):
        _consuming_build(FrameworkKnobs(persist=pk))
    _consuming_build(FrameworkKnobs(persist=pk), persist="kv_bulk")


def test_round_plan_validation():
    from kernels.gfx950.qwen38_mono.persist import driver as DR
    from rocke.helpers.task_schedule import ValidationError, validate_stack

    assert DR.validate_round_plan(["delta", "delta_ple", "delta", "qsa"] * 12) == []
    # negative control: a layer whose first stage does not wait on the previous layer rewrites y while the
    # previous layer's r may still read it (cross-layer write-after-read): refused
    assert validate_stack([T.table(False)] * 2, strict=False)
    with pytest.raises(ValidationError):
        validate_stack([T.table(False)] * 2)
    assert validate_stack([T.table(True)] * 2, strict=False) == []


# ------------------------------------------------- mutation-driven additions


def _elf(sections):
    """A minimal ELF64 LE object with ``sections`` = [(name, type, bytes)] plus .shstrtab."""
    import struct as st

    names = b"\0" + b"".join(n.encode() + b"\0" for n, _, _ in sections) + b".shstrtab\0"
    body, hdrs, off, noff = b"", [], 64, 1
    for n, typ, data in sections:
        hdrs.append((noff, typ, off, len(data)))
        noff += len(n) + 1
        if typ != 8:
            body += data
            off += len(data)
    hdrs.append((noff, 3, off, len(names)))
    body += names
    off += len(names)
    shoff = off
    head = bytearray(64)
    head[:6] = b"\x7fELF\x02\x01"
    st.pack_into("<Q", head, 0x28, shoff)
    st.pack_into("<HHH", head, 0x3A, 64, len(hdrs) + 1, len(hdrs))
    sh = bytes(64)  # the null section header
    for name, typ, o, size in hdrs:
        sh += st.pack("<IIQQQQ", name, typ, 0, 0, o, size) + bytes(24)
    return bytes(head) + body + sh


def test_isa_gate_elf_and_records(tmp_path, monkeypatch):
    import hashlib
    import json

    import rocke.helpers.compile as RC
    from kernels.gfx950.qwen38_mono.persist import isa_gate as IG

    elf = _elf([(".text", 1, b"\x01\x02\x03\x04"), (".bss", 8, b"\x00" * 16), (".note", 7, b"abc")])
    secs = IG.elf_sections(elf)
    assert secs[".text"] == b"\x01\x02\x03\x04" and secs[".bss"] == b"" and secs[".note"] == b"abc"
    assert IG.text_sha(elf) == hashlib.sha256(b"\x01\x02\x03\x04").hexdigest()
    with pytest.raises(AssertionError):
        IG.elf_sections(b"\x7fELF\x01\x01" + bytes(64))  # ELF32 refused
    with pytest.raises(AssertionError):
        IG.elf_sections(b"\x7fELF\x02\x02" + bytes(64))  # big-endian refused

    def boom(*a, **k):
        raise RuntimeError("no comgr")

    monkeypatch.setattr(RC, "compile_kernel", boom)
    r = IG.record("ptoy_static", T.build_static())
    assert r == {"ok": False, "kind": "ptoy_static", "error": "RuntimeError: no comgr"}
    fake = {"ok": True, "kind": "x", "keyed_sha": "k", "objects": [{"text_sha": "t" * 12}]}
    monkeypatch.setattr(IG, "record", lambda kind, k: dict(fake, kind=kind))
    n = IG.write_pool("kernels.gfx950.qwen38_mono.persist.ptoy:isa_builds_ref", "H", "base", 2, str(tmp_path),
                      solo=False)  # fmt: skip
    assert n == 2
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["H__base__fix-0__0.json", "H__base__fix-0__1.json"]
    d = json.loads((tmp_path / files[0]).read_text())
    assert set(d["builds"]) == {"static", "loop"} and d["builds"]["loop"]["kind"] == "ptoy_loop"


_EJ = """
0000000000001900 <k>:
\ts_cbranch_execz 1                                          // 000000001900: BF880001
\ts_nop 0                                                    // 000000001904: BF800000
\tv_mov_b32_e32 v1, v2                                       // 000000001908: 7E020302
\ts_or_b64 exec, exec, s[4:5]                                // 00000000190C: 87FE047E
\ts_endpgm                                                   // 000000001910: BF810000
"""


def test_lint_more():
    fwd = _SPIN.replace("s_cbranch_vccz 65533", "s_cbranch_vccz 1    ")
    assert not L.spin_without_reload(fwd)  # a forward branch is no loop
    f = L.exec_join_findings(_EJ)
    assert len(f) == 1 and "v_mov_b32_e32" in f[0].detail
    assert not L.exec_join_findings(_EJ.replace("v_mov_b32_e32 v1, v2", "v_cmp_eq_u32_e32 vcc, v1, v2"))
    blob = (bytes([0xA0 | 11]) + b".vgpr_count" + bytes([0xCD, 1, 2]) + bytes([0xA0 | 11]) + b".sgpr_count"
            + bytes([0xCE, 0, 1, 0, 0]) + bytes([0xA0 | 17]) + b".sgpr_spill_count" + bytes([0x00]))  # fmt: skip
    m = L.code_object_meta(blob)
    assert m.get("vgpr_count") == 258 and m.get("sgpr_count") == 65536 and m.get("sgpr_spill_count") == 0
    ok, rep = L.resource_gate(blob, _SPIN)
    assert not ok and rep["spin"] and rep["ok"] is False
    ok2, rep2 = L.resource_gate(blob, _POLL)
    assert ok2 and rep2["exec_join"] == [] and rep2["spin"] == []
    ok3, rep3 = L.resource_gate(blob, _EJ)
    assert ok3 and len(rep3["exec_join"]) == 1  # reported, not gating by default
    assert not L.resource_gate(blob, _EJ, strict_exec_join=True)[0]


def test_kv_lint_acquire_must_follow_the_poll():
    """An acquire issued BEFORE the poll does not cover the loads after it."""
    b = IRBuilder("kv_order")
    p = b.param("p", PtrType(I32, "global"))
    rs = b.buffer_rsrc(p, b.const_i32(4096))
    mb = M.mailbox_from_ptr(b, p, b.const_i32(4096), M.MailboxKnobs())
    KV.acquire(b, "inv")
    KV.poll_ready(b, mb, b.const_i32(0), b.const_i32(1), b.const_i32(100))
    v = KV.kv_load(b, rs, b.const_i32(0), launch=True, sc1=False)
    b.global_store(p, b.const_i32(1), v)
    b.ret()
    assert [f.rule for f in L.check_kv_acquire(b.kernel)] == ["kv_load_unacquired"]
    b2 = IRBuilder("kv_noorder")
    p2 = b2.param("p", PtrType(I32, "global"))
    rs2 = b2.buffer_rsrc(p2, b2.const_i32(4096))
    v2 = KV.kv_load(b2, rs2, b2.const_i32(0), launch=True, sc1=False)
    b2.global_store(p2, b2.const_i32(1), v2)
    b2.ret()
    assert [f.rule for f in L.check_kv_acquire(b2.kernel)] == ["kv_load_unordered"]
    from kernels.gfx950.qwen38_mono import atom_layout as AL

    st = L.fused_moe_staging(2)
    assert st[0].offset == AL.SC["ss"] and st[1].offset - st[0].offset == AL.SCRATCH_BYTES


def _gs_ll(maxit, fatal=True):
    b = IRBuilder(f"gs_{maxit}_{fatal}")
    s = b.param("s", PtrType(I32, "global"))
    e = b.param("e", PtrType(I32, "global"))
    lds0 = PS.pad_lds(b)
    sa = b.readfirstlane(b.global_ptr_to_i64(s))
    ea = b.readfirstlane(b.global_ptr_to_i64(e))
    PS.grid_sync(b, PS.GridSync(maxit=maxit), sa, b.readfirstlane(b.block_id_x()), b.const_i32(1), lds0, ea,
                 fatal=fatal)  # fmt: skip
    b.ret()
    return _ll(b.kernel)


def _lds_kernel(fn):
    b = IRBuilder("ldsk")
    s = b.param("s", PtrType(I32, "global"))
    sa = b.readfirstlane(b.global_ptr_to_i64(s))
    fn(b, sa)
    b.ret()
    return _ll(b.kernel)


def test_pad_lds_arrive_word_zeroes_the_cta_arrive_counter_only_when_asked():
    plain = _lds_kernel(lambda b, sa: PS.pad_lds(b))
    arr = _lds_kernel(lambda b, sa: PS.pad_lds(b, arrive_word=True))
    assert plain == _lds_kernel(lambda b, sa: PS.pad_lds(b, arrive_word=False))  # P4's bytes by default
    assert arr.count("ds_write_b32") == plain.count("ds_write_b32") + 1
    pat = rf"add nsw i32 %\w+, {4 * PS.LDS_ARRIVE}\n"
    assert re.search(pat, arr) and not re.search(pat, plain)


def test_cta_arrive_last_wave_rule():
    def body(b, sa):
        lds0 = PS.pad_lds(b)  # no arrive word here: the only +8 is cta_arrive's counter address
        PS.cta_arrive(b, lds0, sa, b.const_i32(PS.LINE * 3), 8, value=1)

    ll = _lds_kernel(body)
    assert "ds_add_rtn_u32" in ll and "s_waitcnt vmcnt(0)" in ll and "global_atomic_add" in ll
    pat = rf"add nsw i32 %\w+, {4 * PS.LDS_ARRIVE}\n"
    assert re.search(pat, ll) and ("srem i32" in ll or "urem i32" in ll)
    with pytest.raises(ValueError):
        _lds_kernel(lambda b, sa: PS.cta_arrive(b, PS.pad_lds(b), sa, b.const_i32(0), 0))
    assert "ds_add_rtn_u32" in _lds_kernel(lambda b, sa: PS.cta_arrive(b, PS.pad_lds(b), sa, b.const_i32(0), 1))


def test_grid_sync_last_arrivals():
    gs = PS.GridSync(arrivals="last")
    assert gs.count == PS.GridSync().count  # one arrival per CTA per epoch, like "cta"
    with pytest.raises(ValueError):
        PS.GridSync(arrivals="lane")
    ll = _ll(MI.build_gs_kernel(MI.GsSpec(arrivals="last")))
    assert "ds_add_rtn_u32" in ll and "alast" in MI.GsSpec(arrivals="last").name()
    assert "ds_add_rtn_u32" not in _ll(MI.build_gs_kernel(MI.GsSpec()))

    def no_lds(b, sa):
        PS.grid_sync_arrive(b, gs, sa, b.const_i32(0))

    with pytest.raises(ValueError, match="lds0"):
        _lds_kernel(no_lds)


def test_aborted_reads_the_abort_word():
    seen = {}

    def body(b, sa):
        lds0 = PS.pad_lds(b)
        v = PS.aborted(b, lds0)
        seen["v"] = v
        with b.scf_if(v):
            PS.lds_write(b, lds0, b.const_i32(2))

    ll = _lds_kernel(body)
    assert seen["v"] is not None and "icmp ne" in ll and "ds_read_b32" in ll


def test_grid_sync_defaults_are_exact_and_fatal():
    def mk(**kw):
        def body(b, sa):
            lds0 = PS.pad_lds(b)
            PS.grid_sync(b, PS.GridSync(maxit=100), sa, b.readfirstlane(b.block_id_x()), b.const_i32(1), lds0, sa, **kw)

        return _lds_kernel(body)

    d = mk()
    assert d == mk(bias=0, fatal=True)
    assert d != mk(bias=1) and d != mk(fatal=False)


def test_table_params_cache_per_stage():
    from kernels.gfx950.qwen38_mono.persist.driver import TableParams
    from rocke.core.ir import I64

    b = IRBuilder("tp")
    tab = b.readfirstlane(b.param("tab", I64))
    tp = TableParams(b, ("a", "b"), tab, "const", None)
    tp.set_layer(b.const_i32(0))
    a1 = tp.addr("a")
    assert a1 is not None and tp.addr("a") is a1 and tp.addr("b") is not a1
    tp.new_stage()
    assert tp.addr("a") is not a1


def test_loop_per_cta_stage_may_fill_the_grid():
    from kernels.gfx950.qwen38_mono.framework import ScratchLayout, StageSpec
    from kernels.gfx950.qwen38_mono.persist.driver import build_persistent_kernel
    from rocke.helpers.task_schedule import Stage, TaskTable

    seen = {}

    def emit(b, ctx, p, t):
        seen["rows"] = ctx.rows

    def mk(n):
        tb = TaskTable((Stage("a", n, base=0, writes=("hx",)),), grid=256, kind="t")
        return build_persistent_kernel("q38pt_pc", [StageSpec(tb.stage("a"), emit, per_cta=True)], tb,
                                       ScratchLayout.of({"hx": 64}, parity_slots=2),
                                       knobs=FrameworkKnobs(persist=K.D0_DEFAULTS), args=("scratch",))  # fmt: skip

    assert mk(256).name.startswith("q38pt_pc")
    assert seen["rows"] == 1  # the default decode rows


def test_launch_tag_host_twin_wraps_like_i32():
    big = (1 << 31) // M.LAYER_SLOTS + 1
    v = (big * M.LAYER_SLOTS + 0 + 1) & 0xFFFFFFFF
    assert T.launch_tag(big, 0) == v - (1 << 32) < 0
    assert T.launch_tag(0, (1 << 31) - 2) == (1 << 31) - 1  # the largest positive i32 stays positive
    assert T.launch_tag(0, (1 << 31) - 1) == -(1 << 31)  # exactly 2**31 wraps


def test_grid_sync_wait_defaults_and_gs_maxit_bound():
    def mk(gs, **kw):
        def body(b, sa):
            lds0 = PS.pad_lds(b)
            PS.grid_sync_wait(b, gs, sa, b.const_i32(1), lds0, sa, **kw)

        return _lds_kernel(body)

    d = mk(PS.GridSync(maxit=100))
    assert d == mk(PS.GridSync(maxit=100), bias=0, fatal=True)
    assert d != mk(PS.GridSync(maxit=100), bias=1) and d != mk(PS.GridSync(maxit=100), fatal=False)
    assert "global_atomic_or" in mk(PS.GridSync(maxit=1))  # maxit=1 is bounded too
    assert "global_atomic_or" not in mk(PS.GridSync(maxit=0))


def test_noinit_mutant_and_lds_poison_kernel():
    good, mut = (_ll(MI.build_gs_kernel(MI.GsSpec(arrivals="last", mutant=m))) for m in ("none", "noinit"))
    pat = rf"add nsw i32 %\w+, {4 * PS.LDS_ARRIVE}\n"
    assert good != mut and good.count("ds_write_b32") > mut.count("ds_write_b32")
    with pytest.raises(ValueError):
        MI.build_gs_kernel(MI.GsSpec(mutant="noinit"))
    poison = _ll(MI.build_lds_poison_kernel())
    assert "ds_write_b32" in poison and "smem_pool" in poison
    assert re.search(pat, good)
    st = _ll(MI.build_gs_kernel(MI.GsSpec(arrivals="last", stagger=300)))
    assert st != good and "st300" in MI.GsSpec(arrivals="last", stagger=300).name()
    defs = re.findall(r"^\s*(%[\w.]+) = ", st, flags=re.M)
    assert len(defs) == len(set(defs))  # unique SSA names (a repeated scf_for iv_name does not lower)
    with pytest.raises(ValueError):
        MI.build_gs_kernel(MI.GsSpec(stagger=300))
    # last-wave arrivals carry no CTA barrier before the arrival: one barrier fewer per sync than the cta form
    assert good.count("s.barrier") < _ll(MI.build_gs_kernel(MI.GsSpec())).count("s.barrier")


def test_sync_constants_and_bounded_waits():
    assert (PS.ERR_MISMATCH, PS.ERR_TIMEOUT, PS.ERR_FATAL, PS.ERR_WORDS) == (0, 1, 2, 4)
    bits = list(PS.FATAL_BITS.values())
    assert len(set(bits)) == len(bits) and all(x & (x - 1) == 0 for x in bits)
    assert (PS.ST_OK, PS.ST_TIMEOUT, PS.ST_ABORTED) == (1, 2, 3) and PS.LINE == 256
    for s in (1, 2, 4, 8):
        gs = PS.GridSync(stripes=s)
        assert gs.words == 8 * s and gs.count == 256 // (8 * s) and gs.nbytes == gs.words * 256
    assert PS.GridSync(arrivals="wave", waves=4).count == 4 * 4
    unb, bnd, nofw = _gs_ll(0), _gs_ll(1000), _gs_ll(1000, fatal=False)
    assert "global_atomic_or" not in unb and "2147483647" in unb  # unbounded: no timeout path
    assert "global_atomic_or" in bnd and "1000" in bnd
    assert "global_atomic_or" not in nofw and "global_atomic_add" in nofw


def test_driver_param_reloads_and_barriers():
    from kernels.gfx950.qwen38_mono.framework import StageSpec
    from kernels.gfx950.qwen38_mono.persist import driver as DR
    from rocke.helpers.task_schedule import Stage

    llc = _ll(T.build_loop(FrameworkKnobs(persist=K.D0_DEFAULTS)))
    # R1: one scalar reload per argument per stage: g reads w, r reads h_out, the layer reads scratch
    assert llc.count("s_load_dwordx2 $0, $1, $2") == 3
    lll = _ll(T.build_loop(FrameworkKnobs(persist=K.PersistKnobs(args="lds"))))
    assert "s_load_dwordx2 $0, $1, $2" not in lll and lll.count("ds_read_b32") >= 6
    a = _ll(T.build_loop(FrameworkKnobs(persist=K.PersistKnobs(loop_carry=False))))
    assert a != llc
    s1 = StageSpec(Stage("a", 1, lds_phase="p1"), lambda *x: None)
    s2 = StageSpec(Stage("b", 1, lds_phase="p2"), lambda *x: None)
    assert DR.steady_barriers([s1, s2]) == {0, 1}  # the wrap p2 -> p1 and p1 -> p2
    assert DR.steady_barriers([s1]) == set()
    tp = DR.TableParams(IRBuilder("tp"), ("scratch", "w"), None, "const", None)
    assert tp.names() == ("scratch", "w")


def test_knob_suffixes_and_lookups():
    assert K.PersistKnobs(loop_carry=False).suffix() == "lc0"
    assert K.PersistKnobs(fatal_word=False, deadline=5).suffix() == "dl5_fw0"
    plan = K.EdgePlan.of({"m5": K.EdgeKnobs(poll="irf")}, default=K.EdgeKnobs(pause="s_sleep1"))
    assert plan.get("m5").poll == "irf" and plan.get("x").pause == "s_sleep1"
    assert plan.suffix().startswith("edpas_sleep1")
    assert K.EdgeKnobs(island=4, placement="island").suffix() == "plislandis4"


def _pool_bytes(ll):
    import re

    m = re.search(r"@smem_pool\S* = internal unnamed_addr addrspace\(3\) global \[(\d+) x i8\]", ll)
    return int(m.group(1)) if m else 0


def test_address_taken_lds_words_are_never_overlaid():
    """The R5 workgroup-id word and the loop driver's argument rows are used
    through inline asm on their integer address, which the lowering's LDS
    liveness does not see; without the pin before the return the pool packer
    overlaid them (hardware: the args=lds loop hung). Both must have their
    own bytes."""
    nargs = len(T.ARGS)
    assert _pool_bytes(_ll(T.build_loop(FrameworkKnobs(persist=K.PersistKnobs(args="lds"))))) >= 4 * (4 * nargs) + 4
    assert _pool_bytes(_ll(T.build_loop(FrameworkKnobs(persist=K.D0_DEFAULTS)))) >= 4
    # with a stage's own live LDS buffer next to the R5 word (both used inside the stage)
    from kernels.gfx950.qwen38_mono.framework import StageSpec, build_layer_kernel

    tb = T.table(False)
    g_emit = T._emit_g("none")

    def emit_g(b, ctx, p, t):
        buf = b.smem_alloc(I32, [64], name_hint="stagebuf")
        b.smem_store_vN(buf, [ctx.lane], ctx.lane, 1)
        v = b.smem_load_vN(buf, ctx.lane, dtype=I32, n=1)
        b.global_store(p.ptr(b, "h_out"), b.add(ctx.lane, b.const_i32(64)), v)
        g_emit(b, ctx, p, t)

    k = build_layer_kernel("q38pt_buf", [StageSpec(tb.stage("g"), emit_g), StageSpec(tb.stage("r"), T._emit_r)], tb,
                           T.layout(1), args=T.ARGS, knobs=FrameworkKnobs(persist=K.PersistKnobs(args="kernarg")))  # fmt: skip
    assert _pool_bytes(_ll(k)) >= 4 * 64 + 4
