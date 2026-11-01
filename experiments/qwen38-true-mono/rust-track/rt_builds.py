"""Rust track (RT): every true-mono stage kernel the direct compiler must cover, by label.

    from rt_builds import BUILDS, build
    k = build("b3_r4s_tuned")          # a KernelDef

Runs with PYTHONPATH on the rust tree (``copy-rocm-lib-rust``: platform/python + library). Labels:

- ``head_<epi>_r<rows>[_ir]``: ``mx_decode_gemv`` lm_head (n 248320, k 2560), epilogue ``logits`` / ``fused``
  (the hybrid's greedy argmax fold, ``waves_per_eu=1``), ``rows_max`` 1 / 4 / 8 / 16; ``_ir``: ``irops=True``.
- ``b3_<variant>_{mono,tuned}[_ir]``: ``stage_b3.build_b3_kernel``; ``tuned`` = ``TUNED_SPECS``; ``_ir``: every
  hand-off and device op as IR ops (mailbox ``impl=ir``, ``fence_impl=ir``, ``cvt=intrinsic``).
- ``fmoe_r<rows>[_ir]``: ``moe_fused.build_fused_suffix`` at ``best_knobs(rows, "best_plain")``.
- ``q2_<variant>[_ir]``: ``stage_fn.build_stage_kernel`` of the Q2 stage function (test layout).
- ``toy_<shape>[_ir]``: the toy chain at ``best_spec`` (d4w8l2, d8w0).
- ``p2_*``, ``p4_*``, ``p5_*``: the W2 probe kernels; ``ccl_*``: the in-kernel collectives.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable

HEAD = dict(n=248320, k=2560)


def _mb_ir(mb):
    import kernels.gfx950.qwen38_mono.mailbox_ir  # noqa: F401  (registers "ir")

    return dataclasses.replace(mb, impl="ir", fence_impl="ir")


def head(epi: str, rows: int, ir: bool):
    from kernels.gfx950 import mx_decode_gemv as M

    kw = dict(HEAD, rows_max=rows, epilogue=epi)
    if epi == "fused":
        kw["waves_per_eu"] = 1
    return M.build_mx_decode_gemv(M.MxDecodeGemvSpec(irops=ir, **kw))


def b3(variant: str, which: str, ir: bool):
    from kernels.gfx950.qwen38_mono import stage_b3 as SB
    from kernels.gfx950.qwen38_mono import stage_fn as SF

    rows, seq, group = SF.parse_variant(variant)
    shape = SB.B3Shape(rows, seq, group) if group else SB.B3Shape(rows, seq)
    knobs = SB.knobs_from_spec(SB.TUNED_SPECS[variant]) if which == "tuned" else SB.B3Knobs()
    if ir:
        knobs = dataclasses.replace(knobs, mailbox=_mb_ir(knobs.mailbox), cvt="intrinsic")
    return SB.build_b3_kernel(shape, knobs)


def fmoe(rows: int, ir: bool, stamps: bool = False):
    from kernels.gfx950.qwen38_mono import moe_fused as MF
    from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF

    knobs = SMF.best_knobs(rows, "best_plain")
    if ir:
        knobs = dataclasses.replace(knobs, mailbox=_mb_ir(knobs.mailbox))
    if os.environ.get("RT_PHASE_LEVERS"):  # direct only: per-phase lever scopes (an arm's ``@`` part)
        knobs = dataclasses.replace(knobs, phase_levers=os.environ["RT_PHASE_LEVERS"])
    if stamps:  # measurement only: thread 0 of every CTA stamps each phase (rt_fmoe.py --stamps)
        knobs = dataclasses.replace(knobs, phase_stamps=True)
    return MF.build_fused_suffix(rows, knobs)


Q2_LAYOUT = None


def q2(variant: str, ir: bool):
    from kernels.gfx950.qwen38_mono import stage_fn as SF
    from kernels.gfx950.qwen38_mono.mono_layout import MonoLayout

    lay = MonoLayout(
        grid=256, threads=512, max_rows=4, row_pairs=159360,
        sc={"qp": 227584, "qn": 283392}, tasks={"q2": 26},
        qsa_state=("pos", "pages", "key", "value", "index_key", "index_cmp", "cos", "sin",
                   "q_norm", "k_norm", "iq_norm", "ik_norm"),
    )  # fmt: skip
    rows, seq, group = SF.parse_variant(variant)
    knobs = SF.StageKnobs()
    if ir:
        import kernels.gfx950.qwen38_mono.mailbox_ir  # noqa: F401

        knobs = dataclasses.replace(knobs, mailbox_impl="ir")
    return SF.build_stage_kernel(SF.StageSpec("q2", rows, seq, group, knobs), lay)


def toy(shape: str, ir: bool):
    from kernels.gfx950.qwen38_mono import toy_chain as T

    spec = T.best_spec(4, 8, layers=2) if shape == "d4w8l2" else T.best_spec(8, 0)
    if ir:
        spec = dataclasses.replace(spec, knobs=dataclasses.replace(spec.knobs, irops=True))
    return T.build_toy_chain(spec)


def p2(**kw):
    from kernels.gfx950 import mx_decode_gemv as M
    from kernels.gfx950.qwen38_mono.probes import p2_head as P

    return P.build_p2(P.P2Spec(head=M.MxDecodeGemvSpec(**HEAD, rows_max=4, epilogue="fused"), **kw))


def p4(kind: str, **kw):
    from kernels.gfx950.qwen38_mono.probes import p4_sync as P

    if "knobs_ir" in kw:
        kw.pop("knobs_ir")
        from rocke.helpers import mailbox as MB

        kw["knobs"] = _mb_ir(MB.MailboxKnobs())
    spec = {"counter": P.CounterSpec, "a2a": P.A2ASpec, "ticket": P.TicketSpec, "xl": P.XlSpec,
            "boundary": P.BoundarySpec}[kind](**kw)  # fmt: skip
    fn = {"counter": P.build_counter_kernel, "a2a": P.build_a2a_kernel, "ticket": P.build_ticket_kernel,
          "xl": P.build_xl_kernel, "boundary": P.build_boundary_kernel}[kind]  # fmt: skip
    return fn(spec)


def p5(kind: str, **kw):
    from kernels.gfx950.qwen38_mono.probes import p5_coherence as P

    spec = {"kv": P.KvSpec, "sc": P.ScSpec, "host": P.HostSpec, "hostw": P.HostWriterSpec}[kind](**kw)
    fn = {"kv": P.build_kv_kernel, "sc": P.build_scalar_kernel, "host": P.build_host_kernel,
          "hostw": P.build_host_writer_kernel}[kind]  # fmt: skip
    return fn(spec)


def ccl(kind: str, **kw):
    from kernels.gfx950.qwen38_mono.ccl import collectives as C
    from kernels.gfx950.qwen38_mono.ccl import layout as L
    from kernels.gfx950.qwen38_mono.ccl import lm_probe as LP

    if kind == "ar":
        return C.build_allreduce_kernel(L.AllReduceSpec(**kw))
    if kind == "ag":
        return C.build_allgather_kernel(L.AllGatherSpec(**kw))
    if kind == "argmax":
        return C.build_argmax_kernel(L.ArgmaxSpec(**kw))
    return LP.build_linkbw_kernel(LP.LinkBwSpec(**kw))


def _fw_layout():
    """The mono layout from the ATOM tree (``Q38_ATOM_DIR``, else ``import atom``)."""
    import os

    from kernels.gfx950.qwen38_mono.mono_layout import from_atom

    return from_atom(os.environ.get("Q38_ATOM_DIR") or None)


def _fw_dummy(kind, shape, st):
    """``test_qwen38_stage_adapt._dummy``: a stand-in stage (polls one pair of every region it reads, puts one
    pair into every region it writes)."""
    from kernels.gfx950.qwen38_mono import atom_layout as A
    from kernels.gfx950.qwen38_mono.framework import StageSpec
    from rocke.helpers.lds_arena import LdsEntry

    def emit(b, ctx, p, t):
        with b.scf_if(b.cmp_eq(ctx.wave, b.const_i32(0))):
            specs = [(ctx.mb(r, local=True), 0, 1) for r in st.reads if r in A.REGIONS]
            if specs:
                ctx.poll(specs)
            with b.scf_if(b.cmp_eq(ctx.lane, b.const_i32(0))):
                for r in st.writes:
                    if r in A.REGIONS:
                        ctx.put(ctx.mb(r), t, t)

    return StageSpec(st, emit, lds=(LdsEntry(f"lds_{st.name}", 64),))


def fw(what: str, variant: str):
    """Framework (true-mono) layer kernels with the tracks' real stage bodies: ``q2`` (qsa layer, the Q2 stage
    function through ``stage_adapt``), ``b3`` (delta layer, the native framework B3 ``b3_stage_spec``), ``moe``
    (delta layer, M3..M5 through the adapter), ``layer_<kind>`` (a whole resident layer: those bodies plus
    stand-ins for the stages no track has written, as ``test_whole_layer_from_track_emitters_is_resident``)."""
    from functools import partial

    from kernels.gfx950.qwen38_mono import stage_adapt as SA
    from kernels.gfx950.qwen38_mono import stage_b3 as S
    from kernels.gfx950.qwen38_mono import stage_fn as SF
    from kernels.gfx950.qwen38_mono.atom_layout import ORDERS
    from kernels.gfx950.qwen38_mono.framework import FrameworkKnobs, LayerShape, build_layer

    rows, seq, _group = SF.parse_variant(variant)
    shape = LayerShape(rows=rows, seq=seq)
    lay = _fw_layout()
    b3f = lambda k, sh, st: S.b3_stage_spec(S.B3Shape(sh.rows, sh.seq, sh.group), base=st.base, phase=st.lds_phase)  # noqa: E731
    if what == "q2":
        return build_layer("qsa", shape, stages=("q2",), factories={"q2": SA.stage_fn_factory("q2", lay)},
                           knobs=SA.adapted_knobs())  # fmt: skip
    if what == "b3":
        return build_layer("delta", shape, stages=("b3",), factories={"b3": b3f}, knobs=SA.adapted_knobs())
    if what == "moe":
        facts = {n: SA.stage_fn_factory(n, lay) for n in ("m3", "m4", "m5")}
        return build_layer("delta", shape, stages=("m3", "m4", "m5"), factories=facts, knobs=SA.adapted_knobs())
    kind = what[len("layer_"):]
    have = set(SA.available_stage_fns())
    facts = {n: _fw_dummy for n in ORDERS[kind]}
    for n in ("q2", "m3", "m4", "m5"):
        if n in have and n in ORDERS[kind]:
            facts[n] = SA.stage_fn_factory(n, lay)
    if "b3" in ORDERS[kind]:
        facts["b3"] = b3f
    build = partial(build_layer, kind, shape, factories=facts)
    k, _w = SA.resident_build(lambda kn: build(knobs=kn), FrameworkKnobs())
    return k


def _builds() -> dict[str, Callable]:
    out: dict[str, Callable] = {}
    for epi in ("logits", "fused"):
        for rows in (1, 4, 8, 16):
            for ir in (False, True):
                out[f"head_{epi}_r{rows}{'_ir' * ir}"] = lambda e=epi, r=rows, i=ir: head(e, r, i)
    for v in ("r1", "r2s", "r3s", "r4s", "r4", "r4g2"):
        for which in ("mono", "tuned"):
            for ir in (False, True):
                out[f"b3_{v}_{which}{'_ir' * ir}"] = lambda v=v, w=which, i=ir: b3(v, w, i)
    for rows in (1, 4):
        for ir in (False, True):
            out[f"fmoe_r{rows}{'_ir' * ir}"] = lambda r=rows, i=ir: fmoe(r, i)
        out[f"fmoe_r{rows}_st"] = lambda r=rows: fmoe(r, False, stamps=True)
    for v in ("r1", "r2", "r4", "r4s"):
        for ir in (False, True):
            out[f"q2_{v}{'_ir' * ir}"] = lambda v=v, i=ir: q2(v, i)
    for shape in ("d4w8l2", "d8w0"):
        for ir in (False, True):
            out[f"toy_{shape}{'_ir' * ir}"] = lambda s=shape, i=ir: toy(s, i)
    out["p2_default"] = lambda: p2()
    out["p2_pimpl_ir"] = lambda: p2(pimpl="ir")
    out["p4_counter"] = lambda: p4("counter")
    out["p4_a2a"] = lambda: p4("a2a")
    out["p4_a2a_ir"] = lambda: p4("a2a", knobs_ir=True)
    out["p4_ticket"] = lambda: p4("ticket")
    out["p4_xl"] = lambda: p4("xl")
    out["p4_xl_ir"] = lambda: p4("xl", knobs_ir=True)
    out["p4_boundary"] = lambda: p4("boundary")
    out["p5_kv"] = lambda: p5("kv")
    out["p5_sc"] = lambda: p5("sc")
    out["p5_host"] = lambda: p5("host")
    out["p5_hostw"] = lambda: p5("hostw")
    for what in ("q2", "b3", "moe", "layer_delta", "layer_delta_ple", "layer_qsa"):
        for v in ("r1", "r4s"):
            out[f"fw_{what}_{v}"] = lambda w=what, v=v: fw(w, v)
    for v in ("r2", "r3", "r4", "r2s", "r3s"):  # rt_q2.py: every Q2 variant in the framework layer kernel
        out[f"fw_q2_{v}"] = lambda v=v: fw("q2", v)
    out["ccl_ar_tp2"] = lambda: ccl("ar", ranks=2, rows=1, cols=2560)
    out["ccl_ar_tp4"] = lambda: ccl("ar", ranks=4, rows=4, cols=2560)
    out["ccl_ag_tp2"] = lambda: ccl("ag", ranks=2, rows=1, cols=2560)
    out["ccl_argmax_tp2"] = lambda: ccl("argmax", ranks=2, rows=4)
    out["ccl_linkbw"] = lambda: ccl("linkbw")
    return out


BUILDS = _builds()


def build(label: str):
    return BUILDS[label]()
