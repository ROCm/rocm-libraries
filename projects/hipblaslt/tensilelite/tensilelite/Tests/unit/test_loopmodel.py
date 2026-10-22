# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The LoopModel: theta construction, the LoopIR it decodes to, and the registers it derives.

Covers `Tensile.LoopModel` (theta, decoder, loopir, traversal, adapter, checks) and
`Tensile.Components.LoopModel` (Order, Registers).  A test may lower to GIR to read a derived
fact back, but what it asserts belongs to the model; `test_gir.py` owns the lowering itself.
"""

import pytest
from Tensile.LoopModel import build_S, derive_S, render
from Tensile.LoopModel import checks
from Tensile.LoopModel import adapter
from Tensile.LoopModel.decoder import decode_theta
from Tensile.LoopModel.loopir import GLOBAL, REGISTER, SHARED, Axis
from Tensile.LoopModel.theta import Global, Shared, Trajectory
from Tensile.Lowering import lower_to_gir
from Tensile.Lowering.gir.passes import pipeline, run_pipeline
from Tensile.LoopModel import theta as theta_mod# COPY/READ -- the operand roles `off` keys on
from Tensile.LoopModel.traversal import presence, presence_axes, summation_names
from dataclasses import fields
import inspect
import subprocess
import sys
import Tensile.LoopModel.loopir as loopir
from Tensile.LoopModel import adapter as _adapter
import ast
from pathlib import Path
from types import SimpleNamespace


# Real MXFP8 16x16x128 geometry (fp8, 1B): wave M/N-fan = 8 MI, TDMSplit region 2 -> in-region fan
# 4, substep = DepthU/MI_K = 2.
MXFP8_GEOM = {
    "MatrixInstruction": [16, 16, 128, 1], "DepthU": 256, "ElemBytes": 1,
    "MIWaveTile": [8, 8],
    "GlobalReadVectorWidthA": 64, "GlobalReadVectorWidthB": 64, "LocalReadVectorWidth": 16,
}


SCENARIOS = {
    "baseline": ("Baseline (all defaults: TDM A+B, PGR2, PLR1, no split)", {}),
    # Reduced MT-region split ONLY (no register partition): A/B each split into 2 storage-
    # disjoint regions.  NOT the two-rate [4,2] -- that needs the real geometry (mxfp8_derive).
    "our_split": ("Reduced region split (A0/A1 x B0/B1) -- MT-region only, no register partition",
                  {"TDMSplit": True, "MIWaveTile": [2, 2]}),
    # Groups split each eight-tile M unit 4+4; lo pipelines at W=2 and hi reuses W=1.
    "mxfp8_derive": ("MXFP8 16x16x128 split M-unit groups: S_reg(A)=[lo:2,hi:1] "
                     "(fp A=192,B=256)",
                     {**MXFP8_GEOM, "TDMSplit": [2, 2],
                      "VgprPartitionA": 2, "VgprReuseStrategyA": 1}),
    # --- LoopOrder: axis-word over {K,M,N} ---------------------------------------
    "order_knm": ("LoopOrder='KNM' -- swap M/N blocks",
                  {"TDMSplit": [2, 2], "MIWaveTile": [2, 2], "LoopOrder": "KNM"}),
    "order_mnk": ("LoopOrder='MNK' -- M-outer, K-inner",
                  {"TDMSplit": [2, 2], "MIWaveTile": [2, 2], "LoopOrder": "MNK"}),
    "order_interleave": ("LoopOrder='KMKNMN' -- interleaved regions (split not contiguous with inner)",
                         {"TDMSplit": [2, 2], "MIWaveTile": [2, 2], "LoopOrder": "KMKNMN"}),
    # --- TDMSplit DU (shared K) axis: symmetric + asymmetric residual ------------
    "du_sym": ("TDMSplit DU symmetric [1,1,2,2] -> shared K_Split=2",
               {"TDMSplit": [1, 1, 2, 2]}),
    "du_asym": ("TDMSplit DU asymmetric [1,1,2,1] (2|1) -> K residual on A's copy hop",
                {"TDMSplit": [1, 1, 2, 1]}),
    # --- VGPR reuse: partition + strategy on real geometry -----------------------
    "reg_min_vgpr": ("VgprPartition=2 strategy=1 (min-VGPR: lo unroll, hi in-place)",
                     {**MXFP8_GEOM, "TDMSplit": [2, 2], "VgprPartitionA": 2, "VgprReuseStrategyA": 1}),
    "reg_max_sched": ("VgprPartition=2 strategy=0 (max-sched: both parts unroll W=R)",
                      {**MXFP8_GEOM, "TDMSplit": [2, 2], "VgprPartitionA": 2, "VgprReuseStrategyA": 0}),
    "plr2": ("Deeper local prefetch (PLR=2)", {"PrefetchLocalRead": 2}),
    "halfplr_a": ("HalfPLR on A (masked fan depth)",
                  {"TDMSplit": True, "MIWaveTile": [2, 2], "HalfPLR": 1}),
    "dtv": ("DirectToVgpr A+B (one-hop, VMEM path)", {"DirectToVgprA": True, "DirectToVgprB": True}),
    "padded": ("LDS padding on A",
               {"TDMSplit": True, "MIWaveTile": [2, 2],
                "LdsPadA": 8, "LdsBlockSizePerPad": 128}),
    "1lds": ("Single LDS buffer", {"TDMSplit": True, "MIWaveTile": [2, 2], "1LDSBuffer": 1}),
    "waveseparate": ("Wave-separated global read",
                     {"TDMSplit": True, "MIWaveTile": [2, 2],
                      "WaveSeparateGlobalReadA": 1}),
    # --- MX scenarios: MXSA/MXSB paths + the TDMFuse enum patterns ---------
    "mx_nofuse": ("MX scales present, no fusion (4 separate TDM streams: A,B,MXSA,MXSB) -- one "
                  "wave, since a fuse IS the wave selection",
                  {"TDMSplit": True, "MIWaveTile": [2, 2],
                   "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 0, "NumWaves": 1}),
    "mx_fuse_ab": ("MX + fuse A/B data (scales separate) [TDMFuse=0]",
                   {"TDMSplit": True, "MIWaveTile": [2, 2],
                    "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 0, "NumWaves": 4}),
    "mx_fuse_amx": ("MX + fuse {A,MXSA,MXSB} whole (A unsplit), B split [TDMFuse=2, TDMSplit=[1,2]]",
                    {"TDMSplit": [1, 2], "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 2, "NumWaves": 4}),
    # THE HETEROGENEOUS Phi : "a partition may mix a per-region-paired group with a whole-operand
    # (unsplit) group in the same Phi".
    "mx_fuse_ab_split": ("MX_AB + TDMSplit[2,2]: {A0,B0} and {A1,B1} paired by region index, "
                         "{MXSA,MXSB} whole [TDMFuse=0]",
                         {"TDMSplit": [2, 2], "MIWaveTile": [2, 2],
                          "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 0, "NumWaves": 4}),
    "mx_fuse_paired": ("MX + paired {A,MXSA},{MXSB,B} [TDMFuse=1]",
                       {"TDMSplit": True, "MIWaveTile": [2, 2],
                        "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 1, "NumWaves": 4}),
    "mx_fuse_nomx": ("TDMFuse=2 but NO MX present -> should warn + fall back",
                     {"TDMSplit": True, "MIWaveTile": [2, 2], "TDMFuse": 2, "NumWaves": 4}),
}


# ------------------------------------------------------------------------------------------------
# Theta: what the schedule derives, and the LoopIR it decodes to
# ------------------------------------------------------------------------------------------------



_RAW_PARAMS_TO_THETA = adapter.params_to_theta


def _calculated_theta(params, *, validate_register_depths=True):
    from gir_fixtures import calculate_registers

    if validate_register_depths:
        params = calculate_registers(params)
    return _RAW_PARAMS_TO_THETA(
        params,
        validate_register_depths=validate_register_depths,
    )


@pytest.fixture(autouse=True)
def _calculated_adapter(monkeypatch):
    monkeypatch.setattr(adapter, "params_to_theta", _calculated_theta)


class Space:
    GLOBAL = GLOBAL
    SHARED = SHARED
    REGISTER = REGISTER


def build_gir(theta):
    decoded = decode_theta(theta)
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=pipeline())
    return program


# --------------------------------------------------------------------------- helpers
# A bf16 NT KMN kernel matching Tests/common/gemm/gfx12/loopmodel_bf16_gfx1250.yaml:
# MI 16x16x32, DepthU 64 => S=2 K-substeps, PGR2 (copy off=2), PLR1 (read off=1).
BF16_NT_KMN = {
    "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1],
    "MIWaveTile": [2, 2], "MIWaveGroup": [1, 1],
    "MIBlock": [16, 16, 32, 1, 1, 1, 1, 1],
    "DepthU": 64, "WavefrontSize": 32,
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "HalfPLR": 0,
    "ProblemType": {"DataType": "b", "DestDataType": "b", "ComputeDataType": "s",
                    "TransposeA": False, "TransposeB": True},
    "LoopOrder": "KMN", "NumLdsBlk": 2, "1LDSBuffer": -1,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "LocalReadVectorWidth": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
    "InnerUnroll": 1, "VgprPartition": 1,
}


# Same kernel with PrefetchLocalRead=0: NO register read-ahead (dr=0).  This collapses the
# software-pipeline read stage and exercises the single-buffer drain path -- the shape that
# regressed repeatedly during Phase 3h (register depth, sigma_c copy order, drain read-ptr swaps).
BF16_NT_KMN_PLR0 = {**BF16_NT_KMN, "PrefetchLocalRead": 0}


def _theta(params):
    """`params_to_theta`, plus a compatibility shim for the scenarios' `VgprPartition*` /
    `VgprReuseStrategy*` keys.
    """
    import dataclasses as _dc
    from Tensile.LoopModel.theta import group_labels
    params = dict(params)
    parts = {nm: int(params.pop("VgprPartition" + nm, 1) or 1) for nm in ("A", "B")}
    strat = {nm: int(params.pop("VgprReuseStrategy" + nm, 0) or 0) for nm in ("A", "B")}
    th = _calculated_theta(params)
    for nm in ("A", "B"):
        n = parts[nm]
        if n <= 1:
            continue
        labels = group_labels(n)
        st = strat[nm]
        pol = {g: ("unroll" if st == 0 else
                   "inplace" if st == 2 else
                   ("overlap" if i == 0 else "inplace"))
               for i, g in enumerate(labels)}
        op = th.op(nm)
        op.fragment = _dc.replace(
            op.fragment, parts=n, labels=labels, group_policy=pol, ring_depths=None)
    return th


def _region_loops_over(nodes, out, loops=()):
    """Find the enclosing region loop for each shared copy."""
    for nd in nodes:
        if isinstance(nd, Loop):
            for b in (nd.bodies or []):
                _region_loops_over(b, out, loops + (nd,))
        elif isinstance(nd, Peel) or isinstance(nd, Bind):
            _region_loops_over(nd.body, out, loops)
        elif isinstance(nd, Cond):
            _region_loops_over(nd.then, out, loops)
            _region_loops_over(nd.els, out, loops)
        elif isinstance(nd, Inst) and isinstance(nd.op, Load) and nd.op.dst == Space.SHARED:
            axes = {axis for axis, _value in nd.op.coord}
            owner = next((loop for loop in reversed(loops) if loop.axis in axes), None)
            if owner is not None:
                out.append((owner, nd))


def _walk(nodes, env, out):
    """Flatten the rolled IR to a concrete Inst stream (mirrors render_unrolled + the
    TensileLite walker): unroll int-extent Loops, bind pinned Branch residues, descend
    Peels.  `out` receives (kind, inst, env) with kind in {read, copy, wmma}."""
    for nd in nodes:
        if isinstance(nd, Loop):
            # each sub-body over its own half-open range; a single-body loop
            # reports one range covering the whole trip.
            for lo, hi, b in nd.ranged_bodies():
                end = hi if hi is not None else lo + 1
                for v in range(lo, end):
                    _walk(b, {**env, nd.axis: v}, out)
        elif isinstance(nd, Branch):
            if nd.axis in env:
                _walk(nd.arms.get(env[nd.axis] % nd.modulus, []), env, out)
            else:                                  # pinned-residue branch: arm key IS the value
                for r, arm in nd.arms.items():
                    _walk(arm, {**env, nd.axis: r}, out)
        elif isinstance(nd, Peel):
            _walk(nd.body, env, out)
        elif isinstance(nd, Bind):
            # a peel-step iter binding: push the bound value (concrete for prologue/short; a
            # symbolic drain `T-M+t` stays unbound, body walked symbolically).
            _walk(nd.body, nd.bound_env(env), out)
        elif isinstance(nd, Cond):
            # honor a RESOLVED guard (a pinned drain substep NLL `Pred(k < n_s-dr)`: the reduction
            # Loop has bound k, so the OOB read-ahead is suppressed for real) -- this is what makes
            # the drain STAGGER.
            v = nd.pred.eval(env)
            if v is None or v:
                _walk(nd.then, env, out)
            else:
                _walk(nd.els, env, out)
        elif isinstance(nd, Inst):
            op = nd.op
            if isinstance(op, Mma):
                out.append(("wmma", nd, env))
            elif isinstance(op, Load) and op.dst == Space.REGISTER:
                out.append(("read", nd, env))
            elif isinstance(op, Load) and op.dst == Space.SHARED:
                out.append(("copy", nd, env))


def _find_peel(nodes, kind):
    """Find the prologue/drain Peel, descending the peel-validity Cond that now wraps the whole
    pipeline (prologue/steady/drain live in its `then`)."""
    for n in nodes:
        if isinstance(n, Peel) and n.kind == kind:
            return n
        if isinstance(n, Cond):
            p = _find_peel(n.then, kind)
            if p is not None:
                return p
    return None


@pytest.mark.parametrize("pgr", [1, 2, 3])
def test_counted_loop_exit_translates_the_drain_frame_once(pgr):
    """The exit converts each T-relative drain frame without losing its final steady copy."""
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.analyses.frame_hazards import FrameHazards, RAW
    from Tensile.Lowering.gir.analyses.frame_map import FrameMap
    from Tensile.Lowering.gir.emit_plan import plan_program
    from Tensile.Lowering.gir.frame_contract import build_contract

    program = build_gir(_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=pgr)))
    frames = AnalysisManager().get(FrameMap(), program)
    rings = {phi.gen.id: phi.gen.ring for phi in program.block("steady").phis}
    advances = {xfer.gen.id: xfer.adv for xfer in program.block("steady").xfers}
    drain_rel = program.block("drain0").gen_rel
    exit_delta = {
        gen_id: (advances[gen_id] - drain_rel) % ring
        for gen_id, ring in rings.items()
    }

    exits = 0
    for frame in frames.frames("steady"):
        for label, successor in frames.successors(("steady", frame)):
            if label != "drain0":
                continue
            exits += 1
            for gen_id, ring in rings.items():
                assert successor.of(gen_id) == (frame.of(gen_id) + exit_delta[gen_id]) % ring
    assert exits == len(frames.frames("steady"))

    plans = plan_program(program)
    steady = plans["steady"][0].action_id
    drain = plans["drain0"][0].action_id
    transfers = {(edge.dst, edge.src, edge.gen, edge.value)
                 for edge in build_contract(program).edges if edge.relative}
    assert {(steady, steady, gen_id, advances[gen_id] % ring)
            for gen_id, ring in rings.items()} <= transfers
    assert {(drain, steady, gen_id, exit_delta[gen_id])
            for gen_id in rings} <= transfers

    drain_raw = [
        hazard for hazard in AnalysisManager().get(FrameHazards(), program)
        if hazard.kind == RAW
        and hazard.producer.block == "steady"
        and hazard.consumer.block == "drain0"
    ]
    assert drain_raw
    nearest = [hazard for hazard in drain_raw if hazard.gap == 1]
    assert {hazard.producer.operand for hazard in nearest} == {"A", "B"}


def test_readahead_prologue_empty_without_dr():
    """dr=0 (no PLR) -> no read-ahead prologue for any operand."""
    from Tensile.LoopModel.schedule import preloaded_tiles
    th = _theta(BF16_NT_KMN_PLR0)
    assert preloaded_tiles(th, th.op("A"), 0) == []


def test_plr0_reuses_one_rotation_unit():
    """Without read-ahead, each unit is consumed before the same W=1 slot is refilled."""
    th = _theta(BF16_NT_KMN_PLR0)
    r = derive_S(th)
    S = r[0] if isinstance(r, tuple) else r
    assert S.get("A", "g0") == 1
    assert S.get("B", "g0") == 1


def _walk_bodies(node):
    """Every MUTABLE instruction list in the tree, for fixtures that reorder instructions.
    Yields the lists themselves, not their owners: a `Loop` may be MULTI-BODY, and its
    `.body` property deliberately raises rather than silently returning `bodies[0]`.
    """
    if isinstance(node, list):
        yield node
        for n in node:
            yield from _walk_bodies(n)
        return
    for sub in getattr(node, "bodies", None) or ():
        yield from _walk_bodies(sub)
    for attr in ("then", "els"):
        sub = getattr(node, attr, None)
        if sub is not None:
            yield from _walk_bodies(sub)


def _loops_with_ranges(nodes, out):
    for nd in nodes:
        if isinstance(nd, Loop):
            if len(nd.bodies) > 1:
                out.append(nd)
            for b in nd.bodies:
                _loops_with_ranges(b, out)
        elif isinstance(nd, (Peel, Bind)):
            _loops_with_ranges(nd.body, out)
        elif isinstance(nd, Cond):
            _loops_with_ranges(nd.then, out)
            _loops_with_ranges(nd.els, out)


def _count_first_touch(nodes, out):
    for nd in nodes:
        if isinstance(nd, Cond):
            if nd.kind == "first_touch":
                out.append(nd)
            _count_first_touch(nd.then, out)
            _count_first_touch(nd.els, out)
        elif isinstance(nd, Loop):
            for b in nd.bodies:
                _count_first_touch(b, out)
        elif isinstance(nd, (Peel, Bind)):
            _count_first_touch(nd.body, out)


def test_dr_g_anchors_from_the_floor():
    """Anchor the derivation itself, so a future rewrite cannot silently change WHICH operand
    keeps its look-ahead.
    """
    from Tensile.LoopModel.schedule import prefetch_steps_for
    def dr_of(order, dr=1, wt=(2, 2)):
        th = _theta({"MIWaveTile": list(wt), "LoopOrder": order, "PrefetchLocalRead": dr})
        S, _f = build_S(th)
        return {op.name: prefetch_steps_for(th, op, op.fragment.groups()[0], dr, S)
                for op in th.operands if op.movements and op.movements[-1].dst == Space.REGISTER}
    for order in ("KMN", "KNM", "MNK", "MKN", "NMK", "NKM"):
        assert dr_of(order) == {"A": 1, "B": 1}, f"{order}: PLR must be honoured for both operands"
    # W=2 stores two complete rotation units, so PLR2 is the exact capacity of this point.
    assert dr_of("KMN", dr=2) == {"A": 2, "B": 2}
    # PLR=0 means no read-ahead was requested at all
    assert dr_of("KMN", dr=0) == {"A": 0, "B": 0}


def _find_steady(nodes):
    for n in nodes:
        if isinstance(n, Loop) and n.outer:
            return n
        kids = (list(n.then) + list(n.els) if isinstance(n, Cond) else
                list(n.body) if isinstance(n, (Peel, Bind)) else
                [x for b in n.bodies for x in b] if isinstance(n, Loop) else [])
        r = _find_steady(kids)
        if r is not None:
            return r
    return None


def test_broadcast_fan_ignores_a_degenerate_mode():
    """the term-(ii) predicate says the qualifying outer mode must have extent > 1. No
 PARAMETER config can exercise that clause -- `translate` drops degenerate modes, so a theta built
 from params never carries an extent-1 inner mode -- which is exactly why it needs a direct
 test: a mutation deleting the clause passes the whole matrix otherwise.

 A hand-built theta can carry one, so the clause is a real guard, not decoration: without it a
 broadcast mode that re-traverses NOTHING would zero `dr_g` and cost the operand its
 look-ahead."""
    from Tensile.LoopModel.traversal import broadcast_width
    import dataclasses
    th = _theta({"MIWaveTile": [2, 2], "LoopOrder": "KMN"})
    A = th.op("A")
    assert broadcast_width(th, A) == 1, "KMN baseline should have no term (ii)"
    outermost = th.inner_axes()[0]
    # splice in a mode OUTSIDE the whole the axes it varies over set: a new loop order entry, declared on A's fragment
    # as an axis A is constant over (that declaration is what `geometry.broadcast_axes` reads).
    deg = dataclasses.replace(outermost, name="degenerate", extent=1)
    A.fragment = dataclasses.replace(
        A.fragment, broadcast_axes=tuple(A.fragment.broadcast_axes) + ("degenerate",))
    # theta's nest is fixed at construction, so a different nest means a different theta
    th1 = dataclasses.replace(th, ord=(deg,) + tuple(th.ord))
    assert broadcast_width(th1, A) == 1, \
        "an extent-1 broadcast mode re-traverses nothing and must NOT fire term (ii)"
    th2 = dataclasses.replace(th, ord=(dataclasses.replace(deg, extent=2),) + tuple(th.ord))
    assert broadcast_width(th2, A) == 2, \
        "an extent-2 broadcast mode outer to all of pres(A) MUST fire term (ii)"


def _prologue_peels(nodes):
    """Every `Peel(kind='prologue')` in the tree (both Cond arms)."""
    out = []
    for n in nodes:
        if isinstance(n, Peel) and n.kind == "prologue":
            out.append(n)
        kids = (list(n.then) + list(n.els) if isinstance(n, Cond) else
                list(n.body) if isinstance(n, (Peel, Bind)) else
                [x for b in n.bodies for x in b] if isinstance(n, Loop) else [])
        out += _prologue_peels(kids)
    return out


# ===========================================================================
# Per-operand S_shared / off  (LDSBufferA|B, PrefetchGlobalReadA|B, PrefetchLocalReadA|B)
# ===========================================================================
def test_shared_depth_is_the_buffer_count_not_the_prefetch_depth():
    """S_shared must come from the operand's LDS BUFFER COUNT, never from `delta`."""
    from Tensile.LoopModel.traversal import lds_buffers
    for pgr in (1, 2):
        th = _theta({"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": pgr,
                     "LoopOrder": "KMN"})
        for nm in ("A", "B"):
            assert lds_buffers(th, th.op(nm)) == 2, \
                f"PGR={pgr}: S_shared must be the NumLdsBlk=2 preset, not delta={pgr}"


@pytest.mark.parametrize("key,base", [("LDSBuffer", None),
                                      ("PrefetchGlobalRead", None),
                                      ("PrefetchLocalRead", None)])
def test_per_operand_override_defaults_to_the_preset(key, base):
    """`-1` (the default) must reproduce the whole-kernel preset exactly, so adding these knobs
    changes nothing for a caller that does not set them."""
    from Tensile.LoopModel.traversal import lds_buffers
    p = {"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
         "LoopOrder": "KMN"}
    ref = _theta(p)
    got = _theta(dict(p, **{key + "A": -1, key + "B": -1}))
    assert [lds_buffers(got, got.op(n)) for n in "AB"] == \
           [lds_buffers(ref, ref.op(n)) for n in "AB"]
    assert got.movement_offsets() == ref.movement_offsets()


def test_lds_buffers_above_the_allocation_is_rejected():
    """Unlike the VGPR side, the LDS allocation is
    sized by Solution.py against MaxLDS -- asking for a deeper ring than it reserved is an
    out-of-bounds LDS write, so it must be refused rather than silently emitted."""
    with pytest.raises(RuntimeError, match="exceeds the kernel's allocated NumLdsBlk"):
        _theta({"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": 2,
                "LoopOrder": "KMN", "NumLdsBlk": 2, "LDSBufferA": 3})


def test_dtv_operand_has_no_shared_ring_depth():
    """A direct global-to-register trajectory has no shared depth."""
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel.traversal import lds_buffers
    th = _theta({"MIWaveTile": [2, 2], "DepthU": 64, "LoopOrder": "KMN", "DirectToVgprA": True})
    A = th.op("A")
    assert A.trajectory.shared is None
    with pytest.raises(RuntimeError, match="no shared placement"):
        lds_buffers(th, A)
    depths, _minimum = build_S(th)
    assert depths.shared("A") is None
    assert depths.shared("B") == 2


def test_chunk_reach_is_the_distance_over_the_span():
    """`r = ceil((span - 1 + distance) / span)`, over the ONE distance the level defines.

    `distance = internal_PLR * group_prefetch_unit`; capacity/staging cap the selected PLR rather
    than silently saturating this physical conversion.
    """
    from Tensile.LoopModel import schedule, traversal
    th = adapter.params_to_theta(dict(
        MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64, ElemBytes=2,
        LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    A = next(o for o in th.operands if o.name == "A")
    level = traversal.readahead_level(th)
    assert level[0] == "K_inner", level         # KMN: the outermost non-region inner axis
    n_s = level[1]                              # the level's own extent, 2 here
    assert traversal.chunks_crossed(th, A, 0) == 0, "no read-ahead -> no crossing"
    for dr in range(1, n_s + 1):
        assert traversal.chunks_crossed(th, A, dr) == 1, f"dr={dr} must reach exactly one chunk"
    assert traversal.prefetch_distance_for(th, A, n_s + 1) \
        == (n_s + 1) * traversal.group_prefetch_unit_tiles(th, A)
    assert traversal.chunks_crossed(th, A, n_s + 1) == 2, \
        "the raw physical distance crosses two chunks; Schedule caps it before emission"


def _p212_theta(name="baseline"):
    _d, params = SCENARIOS[name]
    return adapter.params_to_theta(params)


def test_presence_is_a_PER_HOP_fact_and_bulk_ness_IS_presence():
    """line 71 defines bulk-ness AS the axes it varies over: an inner axis a coarse copy spans in ONE
 instruction is not one the copy varies over. So the same operand has two the axes it varies over sets -- the
 bulk copy's and the per-lane read's -- and asking without a hop must keep answering the DATA
 question the derived families (`summation_axes`, `free_axes`) are defined on."""
    th = _p212_theta()
    for op in th.operands:
        if not op.movements:
            continue
        copy_hop = op.movements[0]
        assert copy_hop.is_bulk, "this fixture's first hop should be the bulk global->shared copy"
        # the BULK copy spans every inner axis in one instruction => present on none of them
        assert presence_axes(th, op, copy_hop) == [], op.name
        # the per-lane read varies over exactly what the DATA does
        read_hop = op.movements[-1]
        assert not read_hop.is_bulk
        assert ([m.name for m in presence_axes(th, op, read_hop)]
                == [m.name for m in presence_axes(th, op)] != []), op.name


def test_a_SPLIT_tile_brings_the_region_axis_BACK_into_the_copy_hops_presence():
    """One instruction PER REGION, so the copy is present on the region mode and
 sits one level deeper. This is the fact `emit._region_mode_of` now reads instead of testing
 `op.split` and scanning `op.region_modes` itself."""
    th = _p212_theta("our_split")
    seen = 0
    for op in th.operands:
        if not op.movements or not op.region_axes:
            continue
        seen += 1
        assert op.split > 1
        assert [m.name for m in presence_axes(th, op, op.movements[0])] == [op.region_axes[0]], op.name
    assert seen >= 2, "the split scenario should split at least two operands"


def test_buffer_depths_are_owned_by_placements():
    th = _p212_theta()
    depths, _minimum = build_S(th)
    for operand in th.operands:
        if operand.is_output:
            continue
        assert depths.groups_of(operand.name) == operand.fragment.ring_depths
        assert depths.shared(operand.name) == operand.trajectory.shared.ring_depth


def test_shared_and_register_depths_have_separate_accessors():
    th = _p212_theta()
    depths, _minimum = build_S(th)
    operand = next(item for item in th.operands if not item.is_output)
    group = operand.fragment.groups()[0]
    assert depths.get(operand.name, group) == operand.fragment.ring_depths[group]
    assert depths.shared(operand.name) == operand.trajectory.shared.ring_depth
    assert "shared" not in depths.groups_of(operand.name)


def test_an_operand_BROADCAST_over_the_reduction_does_not_pay_for_it():
    """`free_tiles x k_tiles` multiplies in the THETA-level reduction extent unconditionally, so
 the accumulator -- absent from the reduction by definition -- was counted
 `k_tiles` times over. Presence-derived, with no role test, it is not."""
    from Tensile.LoopModel import traversal as geometry
    th = _p212_theta()
    k = geometry.k_tiles(th)
    assert k > 1, "this fixture must have >1 reduction substep or the bug is invisible"
    for op in th.operands:
        pt = geometry.presence_tiles(th, op)
        assert pt == geometry.free_tiles(th, op) * geometry.summation_tiles(th, op), op.name
        assert geometry.operand_buffer_regs(th, op) == pt * geometry.frag_regs(th, op), op.name
        if op.is_output:                       # broadcast over the reduction: no k factor
            assert geometry.summation_tiles(th, op) == 1
            assert pt == geometry.free_tiles(th, op)
            assert geometry.operand_buffer_regs(th, op) * k \
                == geometry.free_tiles(th, op) * k * geometry.frag_regs(th, op), \
                "the old formula should be exactly k_tiles too big"
        else:                                  # present on the reduction: unchanged
            assert geometry.summation_tiles(th, op) == k
            assert geometry.operand_buffer_regs(th, op) \
                == geometry.free_tiles(th, op) * k * geometry.frag_regs(th, op)


def test_a_sense_that_CONTRADICTS_the_path_is_an_error_not_a_silent_pick():
    """NEGATIVE CONTROL for the cross-check: `is_output` and the hop direction are two statements
    of the same fact, and disagreement is the confusion this function exists to end."""
    import copy as _copy
    rev = _copy.deepcopy(_p212_theta())
    out = next(o for o in rev.operands if o.is_output)
    out.trajectory = Trajectory(Global(), Shared())
    with pytest.raises(RuntimeError, match="runs the other way"):
        rev.__post_init__()


def test_per_hop_presence_REPRODUCES_the_two_hand_rolled_expressions_it_replaced():
    """The equivalence that makes axis 1 a refactor rather than a change, over EVERY scenario."""

    def bulk(op):
        return next((h for h in (op.movements or ()) if h.is_bulk), None)

    def old_mode(th, ops):
        inner = {m.name for m in th.inner_axes()}
        return next((rm for o in ops for rm in o.region_axes if rm in inner), None)

    def new_mode(th, ops):
        for o in ops:
            h = bulk(o)
            m = next((x.name for x in presence_axes(th, o, h)), None) if h else None
            if m:
                return m
        return None

    checked = 0
    for key, (_desc, params) in SCENARIOS.items():
        th = adapter.params_to_theta(params)
        for op in th.operands:
            if not op.movements:
                continue
            checked += 1
            h = bulk(op)
            note = next((m.name for m in presence_axes(th, op, h)), None) if h else None
            assert note == (None if op.split <= 1 else old_mode(th, [op])), \
                f"{key}/{op.name}: the per-region NOTE must be unchanged"
        byname = {o.name: o for o in th.operands}
        fused = {t for g in th.fused_copy_groups for t in g}
        groups = [tuple(g) for g in th.fused_copy_groups]
        groups += [(o.name,) for o in th.operands if bulk(o) and o.name not in fused]
        for g in groups:
            ops = [byname[n] for n in g if n in byname and bulk(byname[n])]
            if not ops:
                continue
            nregions = max((max(1, o.split) for o in ops), default=1)
            if nregions > 1:                       # the ONLY case the caller uses the mode in
                assert new_mode(th, ops) == old_mode(th, ops), \
                    f"{key}/{g}: a USED region mode changed -- this is not a refactor any more"
    assert checked > 40, f"only {checked} op-classes checked; the scenario sweep is not running"


def test_TDMSplit_gate_accepts_the_MT_half_and_refuses_the_DU_half():
    """The MT half splits a FREE axis and is orthogonal to the tail; the DU half splits the
    shared K axis, which is the axis the tail loop also shrinks.  So the gate is on the DU
    factors, not on TDMSplit being set at all.
    """
    def du_of(ts):
        v = [2, 2] if ts is True else list(ts)
        v = v + [1] * (4 - len(v))
        return max(1, int(v[2])), max(1, int(v[3]))

    for accepted in (True, [2, 2], [2, 2, 1, 1], [4, 2, 1, 1], [1, 1], [1, 1, 1, 1]):
        assert du_of(accepted) == (1, 1), f"{accepted} should be MT-only"
    for refused in ([1, 1, 2, 1], [1, 1, 1, 2], [2, 2, 2, 2], [1, 1, 4, 2]):
        assert du_of(refused) != (1, 1), f"{refused} splits DU and must be refused"

    # and the legacy bool the user asked to test with really is the MT-only [2,2,1,1]
    th = adapter.params_to_theta({**BF16_NT_KMN, "TDMSplit": True})
    assert [o.split for o in th.operands if o.movements] == [2, 2]
    assert all(len(o.region_axes) == 1 for o in th.operands if o.movements), \
        "MT split puts ONE region axis on each operand -- its own free axis"


class _Bytes2:
    def numBytes(self):
        return 2


def _kernel(**over):
    k = {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1], "DepthU": 64,
         "MIWaveTileA": 2, "MIWaveTileB": 2, "ProblemType": {"DataType": _Bytes2()},
         "PrefetchGlobalRead": 1, "PrefetchLocalRead": 1, "NumWaves": 1,
         "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": "KMN"}
    k.update(over)
    return k


def _regions_from_kernel(**over):
    """kernel dict -> theta -> GIR, the way kernel generation actually goes."""
    from Tensile.LoopModel import adapter
    prog = build_gir(adapter.params_to_theta(adapter.kernel_to_params(_kernel(**over))))
    return prog.meta.get("unit_regions", {})


def test_the_bridge_CARRIES_TDMSplit_through_to_theta():
    """A split Solution must produce a split theta."""
    assert _regions_from_kernel(TDMSplitA=1, TDMSplitB=1) == {("A",): 2, ("B",): 2}
    assert _regions_from_kernel(TDMSplitA=0, TDMSplitB=0) == {("A",): 1, ("B",): 1}
    assert _regions_from_kernel() == {("A",): 1, ("B",): 1}      # absent == unsplit


def test_a_split_kernel_reaches_EMIT_as_one_copy_per_region_with_a_walk():
    """The end-to-end region-walk property, entered from a KERNEL and not from a params dict."""
    from Tensile.LoopModel import adapter
    from Tensile.Lowering.gir.emit_plan import plan_block
    prog = build_gir(adapter.params_to_theta(
        adapter.kernel_to_params(_kernel(TDMSplitA=1, TDMSplitB=1))))
    acts = plan_block(prog, "steady")
    copies = [a for a in acts if a.kind == "tensor_load"]
    incs = [a for a in acts if a.kind == "region_inc"]
    assert len(copies) == 4, "2 operands x 2 regions, one copy Move each"
    assert all(a.at["regions"] == 2 for a in copies)
    assert sorted(a.at["region"] for a in copies) == [0, 0, 1, 1]
    assert len(incs) == 4, "one step out and one step back, per operand"
    assert sum(a.at["steps"] for a in incs) == 0, "the walk must close"


def test_the_CHUNK_advance_is_INVARIANT_under_region_splitting():
    """Splitting a tile changes how many LOADS deliver a chunk, never how the chunk pointer moves.
    `GrIncrementRegions` derived the peel's chunk from the POSITIONAL INDEX of the copy among that
    operand's peel copies -- sound while one copy is one chunk, off by `nregions` once it is not.
    """
    from Tensile.LoopModel import adapter
    from Tensile.Lowering.gir.emit_plan import plan_block

    def advances(split, **over):
        prog = build_gir(adapter.params_to_theta(
            adapter.kernel_to_params(_kernel(TDMSplit=split, **over))))
        return {lab: [(a.at["unit"], a.at["chunks"], a.at["to_chunk"])
                      for a in plan_block(prog, lab) if a.kind == "gr_inc"]
                for lab in prog.blocks}

    for pgr in (1, 2):
        for plr in (0, 1):
            split, plain = (advances(True, PrefetchGlobalRead=pgr, PrefetchLocalRead=plr),
                            advances(False, PrefetchGlobalRead=pgr, PrefetchLocalRead=plr))
            # COMPARED AS A MULTISET PER BLOCK, not as a sequence.
            for lab in set(split) | set(plain):
                assert sorted(split.get(lab, []), key=str) == sorted(plain.get(lab, []), key=str), \
                    f"PGR{pgr} PLR{plr} {lab}: split changed the chunk walk\n{split}\n{plain}"
            assert all(c == 1 for acts in split.values() for _u, c, _t in acts), \
                "every advance is exactly one DepthU stride -- the only thing tdmIncrementGir emits"


class _Bytes1:
    def numBytes(self):
        return 1


def _mxf8_kernel(order, split):
    """The triage Solution, as a kernel dict: MT64x64x256, MI 16x16x128, MXBlock 32, fused TDM.
    MIWaveGroup IS [2,2], MATCHING `NumWaves: 4` AND THE DOCSTRING ().
    """
    return {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": 256,
            "MIWaveTileA": 2, "MIWaveTileB": 2, "MatrixInstK": 128,
            "ProblemType": {"DataType": _Bytes1(), "MXBlockA": 32, "MXBlockB": 32},
            "PrefetchGlobalRead": 1, "PrefetchLocalRead": 1, "NumWaves": 4,
            "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": order,
            "TDMSplitA": split, "TDMSplitB": split, "TDMFuse": 0,
            "VectorWidthA": 2, "VectorWidthB": 2,
            "LocalReadVectorWidthMXS": 8,
            "UnrollMajorLDSMXSA": 1, "UnrollMajorLDSMXSB": 1}


def _two_tile_fold():
    """The scale read's supplied coverage: `carrier(t) = t // 2`, `slot(t) = t % 2`."""
    from Tensile.LoopModel.traversal import TransferCoverage
    tile = Axis("tile", 0)
    return TransferCoverage(
        carrier=tile // 2,
        slot=tile % 2,
    )


def _quantum_violations(order, split):
    from Tensile.LoopModel import adapter
    from Tensile.Lowering.gir.emit_plan import plan_block
    from Tensile.Lowering.gir.coverage import plan_coverage
    q = {"MXSA": _two_tile_fold(), "MXSB": _two_tile_fold()}
    prog = build_gir(adapter.params_to_theta(
        adapter.kernel_to_params(_mxf8_kernel(order, split),
                                 target={"ReadQuantum": q})))
    out = []
    for name in prog.blocks:                       # `blocks` is keyed BY NAME; iterating gives keys
        acts = plan_block(prog, name)
        if acts:
            out += ["%s: %s" % (name, v)
                    for v in plan_coverage(acts, lambda op: q.get(op)).violations]
    return out


def test_the_quantum_check_still_reports_a_REGISTER_straddle():
    """The half of a straddle that is NOT repairable by a union, and so must still be a violation.
    the union answers the LDS half -- one instruction, several source buffers, name them all.
    """
    from Tensile.Lowering.gir.emit_plan import EmitAction
    from Tensile.Lowering.gir.coverage import plan_coverage

    def _acts(reg_bufs):
        return [EmitAction(kind="ds_load",
                           at={"tc": "MXSA", "tile": t, "tile_flat": t, "k": 0, "k_flat": 0,
                               "reg_buf": reg_bufs[t], "token_ids": (1,),
                               "region": 0, "regions": 1})
                for t in range(2)]

    q = lambda op: _two_tile_fold() if op == "MXSA" else None
    assert not plan_coverage(_acts([0, 0]), q).violations, \
        "one register generation is a legal merge; the check must stay quiet"
    v = plan_coverage(_acts([0, 1]), q).violations
    assert v, "two register generations under one instruction is ill-formed; the check must report it"
    assert any("REGISTER" in str(x) for x in v), f"the violation must name the register half: {v}"


def test_the_quantum_check_is_clean_on_every_cell():
    """Generation-uniformity holds across the sweep, which is what the check buys."""
    # This order puts the MXS carrier axis on the rotation-unit boundary, so its 2-tile fold must
    # be narrowed before L3 can emit it. Both MT split spellings expose the same missing narrowing.
    bad = {}
    for order in ("KMN", "KMKNMN", "KMNKMN", "KMNMNK"):
        for split in (1, 2):
            v = _quantum_violations(order, split)
            if v:
                bad[(order, split)] = v
    assert not bad, "the merged movement must be generation-uniform; got %r" % (bad,)


def test_absorbing_the_spanned_axis_removes_the_straddle_on_the_shipping_path():
    """the actual repair: the axis one instruction SPANS leaves the read hop's traversal."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel import adapter
    from Tensile.Lowering.gir.emit_plan import plan_block
    target = {"ReadVectorElems": {"MXSA": 8, "MXSB": 8},
              "ReadQuantum": {"MXSA": _two_tile_fold(), "MXSB": _two_tile_fold()}}
    for order in ("KMN", "KMKNMN", "KMNKMN", "KMNMNK"):
        for split in (1, 2):
            k = _mxf8_kernel(order, split)
            th = adapter.params_to_theta(adapter.kernel_to_params(k, target=target))
            mxsa = next(o for o in th.operands if o.name == "MXSA")
            hop = next(h for h in mxsa.movements if not h.is_bulk)
            if split == 1:                     # an MT split is what makes M_split exist at all
                assert geometry.coverage_axes(th, mxsa, hop.coverage) == {"M_split"}, \
                    f"{order}: the 2-tile load spans the real factorized M_split axis"
                assert "M_split" not in {
                    axis.name for axis in presence_axes(th, mxsa, hop=hop)
                }, \
                    "a spanned axis is ABSENT from that hop's presence "
            prog = build_gir(th)
            n = {}
            for name in prog.blocks:
                for a in plan_block(prog, name):
                    if a.kind == "ds_load":
                        n[a.at["tc"]] = n.get(a.at["tc"], 0) + 1
            # one act per instruction: MXSA's acts are now HALF A's, matching the b64 coverage
            assert n.get("MXSA", 0) * 2 == n.get("A", 0) or split == 2, \
                f"{order} split{split}: MXSA acts {n} should be half A's once the axis is absorbed"


def test_an_explicit_W_outside_the_band_is_CLAMPED_not_rejected():
    """/ the repair rule : `multibuffer` REPAIRS an out-of-band width, it does not refuse
    the point.
    """
    from Tensile.LoopModel import build_S
    for want, why in ((1, "below the floor -> clamped UP to L_war"),
                      (99, "above R -> clamped DOWN to R")):
        th = _theta(dict(BF16_NT_KMN))
        A = th.op("A")
        g = A.fragment.groups()[0]
        A.fragment.group_policy = {"*": want}
        S, _floor = build_S(th)                      # must not raise
        got = S.get("A", g)
        assert 1 <= got <= 2, f"W={want} ({why}): clamped to {got}, outside the derived band"
    # and the clamp really is a clamp, not a pass-through
    th = _theta(dict(BF16_NT_KMN))
    th.op("A").fragment.group_policy = {"*": 99}
    S, _ = build_S(th)
    assert S.get("A", th.op("A").fragment.groups()[0]) != 99, \
        "an out-of-band width must be repaired, not honoured verbatim"


# =========================================================================== rho as the agent map
def _rho_sweep_configs():
    """(params, label) over the axes rho is derived from, RESTRICTED TO REALIZABLE POINTS."""
    import itertools
    base, out = {}, []                                  # {} == the `baseline` scenario's defaults
    for o, s, w, wr, ph, rh in itertools.product(
            ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"],
            [[1, 1, 1, 1], [2, 1, 1, 1], [1, 2, 1, 1], [2, 2, 1, 1]],
            [1, 2, 4], [[0, 0], [1, 1], [2, 2]],
            [{}, {"A": 2}, {"B": 2}], [{}, {"A": 2}, {"B": 2}]):
        need = (s[0] // max(1, wr[0] or s[0])) * (s[1] // max(1, wr[1] or s[1]))
        if need > w:
            continue
        p = dict(base)
        p.update(LoopOrder=o, TDMSplit=s, NumWaves=w,
                 TDMSplitWaveRegions=wr, ReadPhi=ph, ReadRho=rh)
        out.append((p, f"{o} split={s} waves={w} wr={wr} phi={ph} rho={rh}"))
    return out


def test_agent_assignment_is_consistent_across_the_matrix():
    from Tensile.LoopModel import checks
    cfgs = _rho_sweep_configs()
    assert len(cfgs) > 500, "sweep collapsed to %d configs -- the filter is too strong" % len(cfgs)
    bad = []
    for params, label in cfgs:
        try:
            th = adapter.params_to_theta(params)
        except Exception:
            continue
        for v in checks.agent_assignment_consistency(th):
            bad.append("%s: %s" % (label, v))
    assert not bad, "agent assignment disagrees on %d point(s):\n  %s" % (
        len(bad), "\n  ".join(bad[:8]))


def test_subwave_distribution_is_not_a_synchronization_scope():
    cfgs = _rho_sweep_configs()
    checked = 0
    for params, label in cfgs:
        if not params.get("ReadRho"):
            continue
        try:
            th = adapter.params_to_theta(params)
        except Exception:
            continue
        sub = th.agent_assignment.assignments_at("subwave")
        if not sub:
            continue
        checked += 1
        served = th.agent_assignment.served_axes()
        for r in sub:
            assert r.axis not in served or any(
                w.axis == r.axis for w in th.agent_assignment.coarser_than("wave")), (
                "%s: sub-wave mode %r leaked into served_modes()" % (label, r.axis))
        assert served == set(th.wave_served_axes()), label
    assert checked, "no config produced a subwave assignment"


def _expected_agent_facts(params):
    """Independent wave-region alias and sub-wave coverage facts from raw params."""
    ts = params.get("TDMSplit", [1, 1, 1, 1])
    if isinstance(ts, bool):
        ts = [2, 2] if ts else [1, 1]
    ts = list(ts) + [1] * (4 - len(ts))
    aMT, bMT = max(1, ts[0]), max(1, ts[1])
    wr = list(params.get("TDMSplitWaveRegions", []) or []) + [0, 0]
    waveA = max(1, min(aMT, int(wr[0]) or aMT))
    waveB = max(1, min(bMT, int(wr[1]) or bMT))
    relative = set()
    if waveA < aMT:
        relative.update(("A", "MXSA"))
    if waveB < bMT:
        relative.update(("B", "MXSB"))
    spans = {n: int(v or 0) for n, v in (params.get("ReadRho", {}) or {}).items() if int(v or 0) > 1}
    return relative, spans


def test_tdmsplit_wave_layout_marks_region_aliasing_without_specializing_the_axis():
    """Wave placement affects region aliasing, while TDMSplit axes remain coordinates."""
    cfgs = _rho_sweep_configs()
    saw_relative = saw_span = 0
    bad = []
    for params, label in cfgs:
        try:
            th = adapter.params_to_theta(params)
        except Exception:
            continue
        want_relative, want_spans = _expected_agent_facts(params)
        if want_relative:
            saw_relative += 1
        if want_spans:
            saw_span += 1
        if th.wave_served_axes():
            bad.append("%s: TDMSplit axis was wave-specialized: %s"
                       % (label, sorted(th.wave_served_axes())))
        for op in th.operands:
            want = want_spans.get(op.name, 0)
            shared = op.trajectory.shared
            read = op.trajectory.fragment_fill
            if shared is not None and shared.regions.wave_relative != (op.name in want_relative):
                bad.append("%s: %s wave_relative %s != oracle %s"
                           % (label, op.name, shared.regions.wave_relative,
                              op.name in want_relative))
            if read is not None and int(read.agent_group_size or 0) != want:
                bad.append("%s: %s agent_group_size %s != oracle %s"
                           % (label, op.name, read.agent_group_size, want))
    assert saw_relative > 50 and saw_span > 50, \
        "oracle never exercised: relative=%d span=%d" % (saw_relative, saw_span)
    assert not bad, "agent facts diverge from the oracle on %d point(s):\n  %s" % (
        len(bad), "\n  ".join(bad[:8]))


def test_agent_assignment_deduplicates_subwave_coverage_without_absorbing_split_axes():
    """Shared sub-wave coverage is deduplicated without absorbing TDMSplit axes."""
    th = adapter.params_to_theta({"MIWaveTile": [2, 2],
                                  "TDMSplit": [2, 2, 1, 1], "NumWaves": 4,
                                  "TDMSplitWaveRegions": [1, 1],
                                  "ReadRho": {"A": 2, "MXSA": 2}})
    axes = [r.axis for r in th.agent_assignment.axis_assignments if r.role == "read"]
    assert len(axes) == len(set(axes)), "axis assigned twice: %r" % (axes,)
    assert "M_split" not in axes and "N_split" not in axes, \
        "TDMSplit is coordinate geometry, not an agent assignment"


def test_agent_assignment_carries_copy_shares_for_every_fusion():
    from Tensile.LoopModel import adapter

    class _B1:
        def numBytes(self): return 1

    def shares(fuse, num_waves=4):
        k = {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2], "DepthU": 64,
             "MIWaveTileA": 2, "MIWaveTileB": 2, "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
             "ProblemType": {"DataType": _B1(), "MXBlockA": 32, "MXBlockB": 32},
             "NumWaves": num_waves, "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2,
             "LoopOrder": "KMN", "TDMFuse": fuse}
        th = adapter.params_to_theta(adapter.kernel_to_params(k))
        return {n: th.agent_assignment.copy_agent_count(n) for n in ("A", "B", "MXSA", "MXSB")}

    # one wave: no set to select between, so every operand keeps the only agent
    assert shares(0, num_waves=1) == {"A": 1, "B": 1, "MXSA": 1, "MXSB": 1}
    assert shares(0) == {"A": 2, "B": 2, "MXSA": 2, "MXSB": 2}   # parity, even
    assert shares(1) == {"A": 2, "B": 2, "MXSA": 2, "MXSB": 2}   # parity, even, crossed
    # the two that miscompared: shares [2,1,1] on the group, every wave on the lone operand
    assert shares(2) == {"A": 2, "B": 4, "MXSA": 1, "MXSB": 1}
    assert shares(3) == {"A": 4, "B": 2, "MXSA": 1, "MXSB": 1}

    # single agent: no partition to state, and the accessor's default is the unspecialized answer
    assert not [r for r in adapter.params_to_theta(adapter.kernel_to_params(
        {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2], "DepthU": 64, "MIWaveTileA": 2,
         "MIWaveTileB": 2, "ProblemType": {"DataType": _B1()}, "PrefetchGlobalRead": 2,
         "PrefetchLocalRead": 1, "NumWaves": 1, "enableTDMA": 1, "enableTDMB": 1,
         "NumLdsBlk": 2, "LoopOrder": "KMN", "TDMFuse": 0})).agent_assignment.axis_assignments if r.role == "copy"]


def test_read_assignment_queries_exclude_copy_entries():
    from Tensile.LoopModel.theta import AgentAssignment, AgentAxisAssignment
    assignment = AgentAssignment(axis_assignments=(
        AgentAxisAssignment("M_inner", "wave", 2, ("A", "read")),
        AgentAxisAssignment("M_inner", "wave", 2, ("A", "copy"), role="copy"),
        AgentAxisAssignment("M_inner", "wave", 1, ("MXSA", "copy"), role="copy"),
    ))
    assert assignment.group_size_over({"M_inner"}, "wave") == 2
    assert [item.extent for item in assignment.assignments_at("wave")] == [2]
    assert assignment.served_axes() == {"M_inner"}
    assert (assignment.copy_agent_count("A"), assignment.copy_agent_count("MXSA")) == (2, 1)
    assert assignment.copy_agent_count("nobody") == 1


def test_subwave_assignment_and_wave_relative_regions_are_independent():
    th = adapter.params_to_theta({"MIWaveTile": [2, 2],
                                  "TDMSplit": [2, 2, 1, 1], "NumWaves": 4,
                                  "TDMSplitWaveRegions": [1, 1], "ReadRho": {"A": 2}})
    a = th.op("A")
    read = a.trajectory.fragment_fill
    assert read.agent_group_size == \
        th.agent_assignment.group_size_over({a.free_mode}, level="subwave") == 2
    assert a.trajectory.shared.regions.wave_relative
    assert not (set(a.region_axes) & th.agent_assignment.served_axes())
    assert set(a.region_axes) <= {axis.name for axis in th.inner_axes()}


@pytest.mark.parametrize("order,plr,expect", [
    # K-innermost: the shipping surface.  A tile's own sequence is k0,k0,..,k1,k1,.. -- sequential
    # runs, so the group is refillable in place and the floor is 1.  Unchanged by.
    ("KMN", 0, {"A": (1, 1), "B": (1, 1)}),
    ("KMN", 1, {"A": (2, 1), "B": (2, 1)}),
    # Whole-set residency is represented by the unit's tile size, not by inflating W.
    ("MKN", 0, {"A": (1, 1), "B": (1, 1)}),
])
def test_the_live_peak_is_measured_per_name_not_per_collapsed_ring(order, plr, expect):
    """requires the peak "per **name**, not per rotation ring" -- "a direct live-range walk on
 the joint `(free-tile, rotation-slot)` name space". Projecting the free tile modes away makes
 two values that live in DIFFERENT names look like they compete for one."""
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": plr,
                                    "PrefetchGlobalRead": 2, "LoopOrder": order})
    steps = geometry._inner_steps(th)
    for name, (l_pf, l_war) in expect.items():
        op = th.op(name)
        g = op.fragment.groups()[0]
        assert geometry.group_live_peak(th, op, g, post_war=False) == l_pf, (order, name)
        assert geometry.group_live_peak(th, op, g) == l_war, (order, name)


def test_the_prefetch_lead_in_is_applied_in_the_names_own_sequence():
    """The trap that made the first attempt at the joint-space walk wrong, pinned."""
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": 1,
                                    "PrefetchGlobalRead": 2, "LoopOrder": "KMN"})
    steps = geometry._inner_steps(th)
    op = th.op("A")
    g = op.fragment.groups()[0]
    l_pf = geometry.group_live_peak(th, op, g, post_war=False)
    l_war = geometry.group_live_peak(th, op, g)
    assert (l_pf, l_war) == (2, 1), "prefetch lead-in lost: L_pf must be dr_g+1 above L_war"


def test_the_per_name_floor_is_never_above_the_collapsed_one():
    """SAFETY ARGUMENT for the joint-space walk, kept executable.  It may only REMOVE pessimism: if the
    per-name floor were ever ABOVE the old collapsed one, some already-shipped kernel would have
    been running below its true floor -- a the register-width floor race, not a footprint win.  Swept 630 groups
    when the change landed; this pins a representative slice so a future edit to the walk cannot
    quietly cross that line."""
    from Tensile.LoopModel import traversal as geometry
    import itertools
    for order, plr, mt in itertools.product(("KMN", "KNM", "MNK", "NMK", "MKN", "NKM"),
                                            (0, 1, 2), ([2, 2], [4, 4], [4, 2])):
        try:
            th = adapter.params_to_theta({"MIWaveTile": mt, "PrefetchLocalRead": plr,
                                          "PrefetchGlobalRead": 2, "LoopOrder": order})
        except RuntimeError as e:
            if "gives S" not in str(e):
                raise
            continue                      # no schedulable S here; there is no floor to compare
        steps = geometry._inner_steps(th)
        for name in ("A", "B"):
            op = th.op(name)
            for g in op.fragment.groups():
                l_war = geometry.group_live_peak(th, op, g)
                l_pf = geometry.group_live_peak(th, op, g, post_war=False)
                assert 1 <= l_war <= l_pf, (order, plr, mt, name, g, l_war, l_pf)


@pytest.mark.parametrize("mi_k,kext", [(32, 2), (16, 4)])
def test_whole_set_residency_grows_the_unit_not_the_width_floor(mi_k, kext):
    """A full inner K sweep is represented in unit size while L_floor remains one unit."""
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "DepthU": 64,
                                    "MatrixInstruction": [16, 16, mi_k, 1, 1, 4, 4, 2, 2],
                                    "PrefetchLocalRead": 0, "PrefetchGlobalRead": 2,
                                    "LoopOrder": "MNK"})
    assert next(
        int(m.extent.constant_value)
        for m in th.inner_axes()
        if m.name == "K_inner"
    ) == kext
    steps = geometry._inner_steps(th)
    a, b = th.op("A"), th.op("B")
    assert geometry.group_live_peak(th, a, a.fragment.groups()[0]) == 1
    assert geometry.group_live_peak(th, b, b.fragment.groups()[0]) == 1
    assert geometry.group_unit_tiles(th, a, a.fragment.groups()[0]) == kext
    assert geometry.group_unit_tiles(th, b, b.fragment.groups()[0]) == 4 * kext


def test_mkn_whole_set_operand_uses_a_larger_unit_not_a_larger_floor():
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": 0,
                                    "PrefetchGlobalRead": 2, "LoopOrder": "MKN"})
    steps = geometry._inner_steps(th)
    la = geometry.group_live_peak(th, th.op("A"), th.op("A").fragment.groups()[0])
    lb = geometry.group_live_peak(th, th.op("B"), th.op("B").fragment.groups()[0])
    assert (la, lb) == (1, 1)
    assert geometry.group_unit_tiles(th, th.op("B"), th.op("B").fragment.groups()[0]) \
        > geometry.group_unit_tiles(th, th.op("A"), th.op("A").fragment.groups()[0])


def test_the_walk_satisfies_every_closed_fact_the_paper_commits_to():
    """Unit widths are monotonic in read-ahead and have a one-unit no-prefetch floor."""
    from Tensile.LoopModel import traversal as geometry
    import itertools
    checked = dr0 = 0
    for order, plr, mt, du, mik in itertools.product(
            ("KMN", "KNM", "MNK", "NMK", "MKN", "NKM"), (0, 1, 2),
            ([2, 2], [4, 4], [4, 2]), (64, 128), (16, 32)):
        try:
            th = adapter.params_to_theta({
                "MIWaveTile": mt, "DepthU": du, "PrefetchLocalRead": plr, "PrefetchGlobalRead": 2,
                "MatrixInstruction": [16, 16, mik, 1, 1, mt[0], mt[1], 2, 2], "LoopOrder": order})
        except RuntimeError as e:
            if "gives S" not in str(e):
                raise
            continue                      # no schedulable S here; there is no walk to check
        steps = geometry._inner_steps(th)
        for nm in ("A", "B"):
            op = th.op(nm)
            dr = geometry.requested_read_ahead(th, op)
            for g in op.fragment.groups():
                l_pf = geometry.group_live_peak(th, op, g, post_war=False)
                checked += 1
                prev = geometry.group_live_peak(
                    th, op, g, post_war=False, off=max(0, dr - 1)
                )
                assert l_pf >= max(1, prev), ("read-ahead folds", order, plr, nm, l_pf, prev, dr)
                if dr == 0:
                    dr0 += 1
                    assert l_pf == 1, ("dr_g=0 floor", order, mt, nm, l_pf)
    assert checked > 400 and dr0 > 100, (checked, dr0)


@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("wave_regions", [1, 2])
def test_tdmsplit_is_an_exact_wave_tile_axis_factorization(split, wave_regions):
    wt = 8
    th = adapter.params_to_theta({
        "MIWaveTile": [wt, wt], "TDMSplit": [split, split, 1, 1],
        "TDMSplitWaveRegions": [min(split, wave_regions)] * 2,
    })
    extents = {
        axis.name: int(axis.extent.constant_value)
        for axis in th.inner_axes()
    }
    assert extents.get("M_split", 1) * extents.get("M_inner", 1) == wt
    assert extents.get("N_split", 1) * extents.get("N_inner", 1) == wt


@pytest.mark.parametrize("order,broadcast_outer_op", [
    ("KMN", None),   # K outermost of the inner nest -> neither operand is broadcast-outer
    ("KNM", None),
    ("NKM", "A"),    # A is broadcast over N_inner, the OUTERMOST inner mode: every A name is
    ("NMK", "A"),    # consumed once per N pass, so it is live for the WHOLE trip
    ("MKN", "B"),    # mirror: B broadcast over M_inner
    ("MNK", "B"),
])
def test_plr_is_honoured_on_every_loop_order(order, broadcast_outer_op):
    """`PrefetchLocalRead` is a PARAMETER and the register width is DERIVED, so the two must be
    made to agree by deriving the schedule -- never by quietly lowering PLR.
    """
    from Tensile.LoopModel.traversal import reloads_whole_set
    from Tensile.LoopModel.schedule import prefetch_refusals
    from Tensile.LoopModel.schedule import derive_S

    th = _theta({**BF16_NT_KMN, "LoopOrder": order})
    assert prefetch_refusals(th, derive_S(th)) == (), \
        f"{order}: PLR must be honoured on every order now"

    got = {op.name for op in th.operands
           if not op.is_output and reloads_whole_set(th, op)}
    want = {broadcast_outer_op} if broadcast_outer_op else set()
    assert got == want, f"{order}: broadcast-outer operands {got}, expected {want}"


def test_the_plr_state_is_named_in_the_geometry_dump():
    """The dump exposes P, U and requested/cap/realized internal PLR."""
    from Tensile.LoopModel import render

    for order in ("KMN", "NKM", "MKN"):
        txt = render.render_geometry(_theta({**BF16_NT_KMN, "LoopOrder": order}))
        assert "prefetch_shortfall" not in txt, f"{order}: unexpected downgrade"
        assert "is satisfied" in txt, f"{order}: the dump must state the request is satisfied"
        assert "P(original/region/group)" in txt
        assert "PLR(request/cap/realized)" in txt


# ------------------------------------------------------------------------------------------------
# LoopIR core: spaces, transfers, presence, and the canonical nest
# ------------------------------------------------------------------------------------------------


def _transfer(local_tile, source, destination, where=loopir.TRUE, effect=None):
    return loopir.Transfer(
        local_tile,
        source,
        destination,
        loopir.Effect(
            local_tile if effect is None else effect,
            where,
        ),
    )


def _lower_theta(theta):
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
    from Tensile.Lowering.gir.passes import run_pipeline

    decoded = decode_theta(theta)
    assert not validate_loopir(decoded.program, decoded.profile)
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=gir_pipeline())
    return decoded, program


def test_loopir_import_is_theta_independent():
    script = """
import sys
import Tensile.LoopModel.loopir
forbidden = (
    'Tensile.LoopModel.theta',
    'Tensile.LoopModel.decoder',
    'Tensile.LoopModel.emit',
    'Tensile.LoopModel.ir',
    'Tensile.LoopModel.schedule',
)
loaded = tuple(name for name in forbidden if name in sys.modules)
raise SystemExit(repr(loaded) if loaded else 0)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_functional_api_contains_primitives_not_decoder_policy():
    for name in (
        "rewrite",
        "replace",
        "insert",
        "append",
        "substitute_operation",
        "rewrite_operation",
        "defer",
        "hoist",
        "unroll",
        "pipeline",
    ):
        assert hasattr(loopir, name)
    for name in (
        "resolve_constant_loop_conditions",
        "move_guarded_operations_to_end",
        "unroll_and_sort_operation_runs",
        "rewrite_last_let",
        "append_region",
        "simplify_guards",
    ):
        assert not hasattr(loopir, name)
    assert not any(name.startswith("_") for name in loopir.__all__)


def test_scheduling_primitives_target_explicit_ast_nodes():
    program, _wave, load0, load1, reads = _g3_program()
    root = program.body
    assert loopir.defer(program, root, (load0,)).body == (
        root[0],
        load1,
        *reads,
        load0,
    )

    program, _wave, load0, load1, reads = _g3_program()
    root = program.body
    assert loopir.hoist(program, root, (reads[-1],)).body == (
        reads[-1],
        root[0],
        load0,
        load1,
        *reads[:-1],
    )

    axis = loopir.Axis("shape", 2)
    loop = loopir.Loop(axis, 0, 2, (reads[0],))
    program = loopir.Program(
        (),
        (*program.axes, axis),
        program.operands,
        (loop,),
        program.agents,
    )
    loopir.unroll(program, loop)
    assert tuple(
        node.value.constant_value
        for node in program.body
        if isinstance(node, loopir.Let)
    ) == (0, 1)


def _g3_program():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 3)
    a = loopir.Operand("A", (tile,))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    read = loopir.CompletionClass("read", loopir.WAVE)

    load0 = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            2 * tile + wave < 3,
            effect=a[wave, tile],
        ),),
        copy,
    )
    load1 = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            2 * tile + wave >= 3,
            effect=a[wave, tile],
        ),),
        copy,
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(
                a[index],
                loopir.SHARED,
                loopir.REGISTER,
                effect=a[wave, index],
            ),),
            read,
        )
        for index in range(3)
    )
    program = loopir.Program(
        (),
        (tile,),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            load0,
            load1,
            *reads,
        ),
        ((wave, loopir.WAVE),),
    )
    return program, wave, load0, load1, reads


def _raw_points(program, complete_operation, issue_operation):
    relation = next((
        dependence
        for dependence in loopir.dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is complete_operation
        and dependence.issue.operation is issue_operation
        and dependence.complete.level is loopir.SHARED
        and dependence.issue.level is loopir.SHARED
    ), None)
    return loopir.dependence_points(relation) if relation is not None else ()


def test_expression_analysis_is_stored_on_immutable_values():
    from copy import deepcopy
    import pickle
    from Tensile.LoopModel.loopir.semantics import _affine

    x = loopir.Axis("x", 4)
    y = loopir.Axis("y", 2)
    trip = loopir.Param("T")
    expression = 2 * x + y + trip
    predicate = (expression >= 1) & (x < y + 2)

    assert expression.free_axes() is expression._free_axes
    assert expression.free_params() is expression._free_params
    assert predicate.free_axes() is predicate._free_axes
    assert predicate.free_params() is predicate._free_params
    assert _affine(expression) is None
    assert expression._affine_computed
    affine = 2 * x + y + 3
    assert loopir.expr_equivalent(affine, y + 3 + 2 * x)
    assert affine._equivalence_key is not None
    copied_x, copied_y, copied_affine = deepcopy((x, y, affine))
    assert loopir.expr_equivalent(
        copied_affine,
        2 * copied_x + copied_y + 3,
    )
    pickled_x, pickled_y, pickled_affine = pickle.loads(
        pickle.dumps((x, y, affine))
    )
    assert loopir.expr_equivalent(
        pickled_affine,
        2 * pickled_x + pickled_y + 3,
    )


def test_g3_tensor_load_dependencies_are_exact_per_wave():
    program, wave, load0, load1, reads = _g3_program()

    assert {
        value for complete, _issue in _raw_points(program, load0, reads[0])
        for axis, value in complete.agents if axis is wave
    } == {0, 1}
    assert not _raw_points(program, load1, reads[0])

    assert {
        value for complete, _issue in _raw_points(program, load0, reads[1])
        for axis, value in complete.agents if axis is wave
    } == {0}
    assert {
        value for complete, _issue in _raw_points(program, load1, reads[1])
        for axis, value in complete.agents if axis is wave
    } == {1}

    assert not _raw_points(program, load0, reads[2])
    assert {
        value for complete, _issue in _raw_points(program, load1, reads[2])
        for axis, value in complete.agents if axis is wave
    } == {0, 1}


def test_dependence_repr_does_not_embed_program_subtrees():
    program, _wave, _load0, _load1, _reads = _g3_program()
    assert max(
        len(repr(dependence))
        for dependence in loopir.dependences(program)
    ) < 4096


def test_loopir_render_is_local_first_effect_last_with_named_axes():
    program, _wave, _load0, _load1, _reads = _g3_program()
    text = loopir.render(program)
    assert text.index("local=A[tile=tile]") < text.index(
        "effect=A[wave=wave, tile=tile]"
    )
    assert text.index("register=A[tile=0]") < text.index(
        "effect=A[wave=wave, tile=0]"
    )
    assert "case " not in text


def test_ordinary_schedule_keeps_guards_without_switches():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": "KMN",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
    }))
    nodes = tuple(loopir.walk(decoded.program))
    assert any(isinstance(node, loopir.Cond) for node in nodes)
    assert not any(isinstance(node, loopir.Switch) for node in nodes)


def test_loopir_dependence_vocabulary_is_raw_and_war_only():
    assert tuple(kind.name for kind in loopir.DependenceKind) == ("RAW", "WAR")
    assert not hasattr(loopir, "WAW")


def test_new_loopir_checker_rejects_reordered_phase_markers():
    from Tensile.LoopModel.checks import validate_loopir

    tile = loopir.Axis("tile", 1)
    operand = loopir.Operand("A", (tile,))
    move = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(operand[tile], loopir.GLOBAL, loopir.SHARED),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    program = loopir.Program(
        (),
        (tile,),
        (operand,),
        (
            loopir.Region(loopir.STEADY),
            move,
            loopir.Region(loopir.PROLOGUE),
            move,
        ),
    )
    assert any(
        "out of order" in error
        for error in validate_loopir(program)
    )


def test_loopir_checker_rejects_noncovering_tensorload_effects():
    from Tensile.LoopModel.checks import validate_loopir

    tile = loopir.Axis("tile", 2)
    operand = loopir.Operand("A", (tile,))
    copies = tuple(
        loopir.Movement(
            loopir.TENSOR_LOAD,
            (_transfer(
                operand[tile],
                loopir.GLOBAL,
                loopir.SHARED,
                where=tile == 0,
            ),),
            loopir.CompletionClass(f"copy{region}", loopir.WAVE),
        )
        for region in range(2)
    )
    program = loopir.Program((), (tile,), (operand,), copies)
    assert any(
        "not a disjoint cover" in error
        for error in validate_loopir(program)
    )


def test_wave_sensitive_effect_has_sixteen_points_but_local_tile_has_eight():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 8)
    a = loopir.Operand("A", (tile,))
    movement = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            effect=a[wave, tile],
        ),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    program = loopir.Program(
        (), (tile,), (a,), (movement,), ((wave, loopir.WAVE),)
    )
    write = next(
        effect for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
    )
    assert len(loopir.effect_points(write)) == 16
    assert len({
        point.tile[-1] for point in loopir.effect_points(write)
    }) == 8
    assert len(movement.transfers[0].local_tile.indices) == 1
    assert len(movement.effect_for(0).tile.indices) == 2


def test_ds_load_reads_effect_tile_and_writes_wave_local_tile():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 8)
    a = loopir.Operand("A", (tile,))
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            a[tile],
            loopir.SHARED,
            loopir.REGISTER,
            effect=a[wave, tile],
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program(
        (), (tile,), (a,), (read,), ((wave, loopir.WAVE),)
    )
    source, destination = loopir.effects(program)
    assert source.kind is loopir.READ
    assert destination.kind is loopir.WRITE
    assert len(loopir.effect_points(source)) == 16
    assert len(loopir.effect_points(destination)) == 8
    assert loopir.effect_domain(source).agent_axes == ((wave, loopir.WAVE),)
    assert loopir.effect_domain(destination).agent_axes == ()


def test_declared_wave_is_not_implicitly_added_to_wave_invariant_effect():
    wave = loopir.Axis("wave", 4)
    tile = loopir.Axis("tile", 8)
    a = loopir.Operand("A", (tile,))
    movement = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[tile], loopir.GLOBAL, loopir.SHARED),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    program = loopir.Program(
        (), (tile,), (a,), (movement,), ((wave, loopir.WAVE),)
    )
    write = next(
        effect for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
    )
    assert loopir.effect_domain(write).agent_axes == ()
    assert len(loopir.effect_points(write)) == 8


def test_g3_finite_wave_tile_intersection_remains_exact_with_runtime_loop():
    trip = loopir.Param("T")
    iteration = loopir.Axis("iter", trip)
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 3)
    a = loopir.Operand("A", (iteration, tile))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    read = loopir.CompletionClass("read", loopir.WAVE)
    loads = tuple(
        loopir.Movement(
            loopir.TENSOR_LOAD,
            (_transfer(
                a[iteration, tile],
                loopir.GLOBAL,
                loopir.SHARED,
                (2 * tile + wave < 3) if region == 0
                else (2 * tile + wave >= 3),
                effect=a[wave, iteration, tile],
            ),),
            copy,
        )
        for region in range(2)
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(
                a[iteration, index],
                loopir.SHARED,
                loopir.REGISTER,
                effect=a[wave, iteration, index],
            ),),
            read,
        )
        for index in range(3)
    )
    program = loopir.Program(
        (trip,),
        (iteration, tile),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            loopir.Loop(iteration, 0, trip, (*loads, *reads)),
        ),
        ((wave, loopir.WAVE),),
    )
    shared_raw = tuple(
        dependence for dependence in loopir.dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.level is loopir.SHARED
        and dependence.issue.level is loopir.SHARED
    )
    assert {
        (dependence.complete.operation, dependence.issue.operation)
        for dependence in shared_raw
    } == {
        (loads[0], reads[0]),
        (loads[0], reads[1]),
        (loads[1], reads[1]),
        (loads[1], reads[2]),
    }


def test_free_axis_is_a_bulk_local_effect_domain_not_an_issue_loop():
    program, _wave, load0, _load1, _reads = _g3_program()
    write = next(
        effect for effect in loopir.effects(program)
        if effect.operation is load0
        and effect.kind is loopir.WRITE
        and effect.level is loopir.SHARED
    )
    domain = loopir.effect_domain(write)
    assert tuple(axis.name for axis in domain.local_axes) == ("tile",)
    assert len(loopir.effect_points(write)) == 3


def test_dependence_is_minimal_and_kind_is_derived():
    assert tuple(field.name for field in fields(loopir.Effect)) == (
        "tile",
        "where",
    )
    assert tuple(field.name for field in fields(loopir.Dependence)) == (
        "complete",
        "issue",
        "domain",
    )
    program, _wave, load0, _load1, reads = _g3_program()
    assert all(
        isinstance(effect, loopir.EffectInstance)
        for effect in loopir.effects(program)
    )
    dependence = next(
        value for value in loopir.dependences(program)
        if value.complete.operation is load0
        and value.issue.operation is reads[0]
        and value.complete.level is loopir.SHARED
        and value.issue.level is loopir.SHARED
    )
    assert dependence.kind is loopir.RAW


def test_completion_wait_is_coalesced_across_ordered_serial_switch_cases():
    index = loopir.Axis("index", 2)
    tile = loopir.Axis("tile", 1)
    a = loopir.Operand("A", (tile,))
    copy_completion = loopir.CompletionClass("copy", loopir.WAVE)
    read_completion = loopir.CompletionClass("read", loopir.WAVE)
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[0], loopir.GLOBAL, loopir.SHARED),),
        copy_completion,
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[0], loopir.SHARED, loopir.REGISTER),),
            read_completion,
        )
        for _case in range(2)
    )
    program = loopir.Program(
        (),
        (index, tile),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            copy,
            loopir.Loop(
                index,
                0,
                index.extent,
                (loopir.Switch(
                    index,
                    ((0, (reads[0],)), (1, (reads[1],))),
                ),),
            ),
        ),
    )

    required = tuple(
        dependence
        for dependence in loopir.required_dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is copy
    )
    assert len(required) == 1
    assert required[0].issue.operation is reads[0]


def test_completion_wait_is_not_coalesced_across_conditional_arms():
    selector = loopir.Param("selector")
    index = loopir.Axis("index", 1)
    a = loopir.Operand("A", (index,))
    copy_completion = loopir.CompletionClass("copy", loopir.WAVE)
    read_completion = loopir.CompletionClass("read", loopir.WAVE)
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[index], loopir.GLOBAL, loopir.SHARED),),
        copy_completion,
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[index], loopir.SHARED, loopir.REGISTER),),
            read_completion,
        )
        for _arm in range(2)
    )
    program = loopir.Program(
        (selector,),
        (index,),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            copy,
            loopir.Cond(selector == 0, (reads[0],), (reads[1],)),
        ),
    )

    required = tuple(
        dependence
        for dependence in loopir.required_dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is copy
    )
    assert {dependence.issue.operation for dependence in required} == set(reads)


def test_program_bodies_hold_loopir_nodes_and_region_is_only_a_marker():
    program, _wave, _load0, _load1, _reads = _g3_program()
    assert all(isinstance(node, loopir.LoopIR) for node in program.body)
    marker = program.body[0]
    assert isinstance(marker, loopir.Region)
    assert tuple(field.name for field in fields(marker)) == ("tag",)


def test_movement_transfer_has_explicit_effect_and_local_tiles():
    _program, _wave, load0, _load1, _reads = _g3_program()
    transfer = load0.transfers[0]
    assert loopir.expr_equivalent(
        transfer.local_tile.indices[0],
        loopir.as_expr(_program.axes[0]),
    )
    assert load0.effect_for(0).tile.indices[0].free_axes() == frozenset((_wave,))
    assert not hasattr(transfer, "source_tile")
    assert not hasattr(transfer, "destination_tile")


def test_operation_api_has_checked_ds_load_and_implicit_serial_loops():
    tile = loopir.Axis("tile", 2)
    a = loopir.Operand("A", (tile,))
    b = loopir.Operand("B", (tile,))
    completion = loopir.CompletionClass("read", loopir.WAVE)
    assert issubclass(loopir.DS_LOAD, loopir.MovementOp)
    assert issubclass(loopir.WMMA, loopir.ComputeOp)

    with pytest.raises(loopir.IRTypeError, match="SHARED->REGISTER"):
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[tile], loopir.GLOBAL, loopir.SHARED),),
            completion,
        )
    independent = loopir.Movement(
        loopir.TENSOR_LOAD,
        (
            _transfer(a[tile], loopir.GLOBAL, loopir.SHARED),
            _transfer(
                b[tile],
                loopir.GLOBAL,
                loopir.SHARED,
                tile == 0,
            ),
        ),
        completion,
    )
    assert len(independent.transfers) == 2

    common_effect = loopir.Effect(a[tile], tile >= 0)
    common = loopir.Movement(
        loopir.DS_LOAD,
        (loopir.Transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
        completion,
        effect=common_effect,
    )
    assert common.effect_for(0) is common_effect
    assert common.tile_for_endpoint(0, source=True) is common_effect.tile
    assert loopir.expr_equivalent(
        common.tile_for_endpoint(0, source=False).indices[0],
        a[tile].indices[0],
    )

    with pytest.raises(loopir.IRTypeError, match="needs an Effect"):
        loopir.Movement(
            loopir.DS_LOAD,
            (loopir.Transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
            completion,
        )
    with pytest.raises(loopir.IRTypeError, match="not both"):
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
            completion,
            effect=loopir.Effect(a[tile]),
        )

    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
        completion,
    )
    loop = loopir.Loop(tile, (read,))
    assert not hasattr(loop, "binding")
    for removed in ("LOCAL_READ", "MATRIX", "ELEMENTWISE", "SERIAL"):
        assert not hasattr(loopir, removed)

    wave = loopir.Axis("wave", 2)
    with pytest.raises(loopir.IRTypeError, match="cannot be bound by Loop"):
        loopir.Program(
            (),
            (tile,),
            (a,),
            (loopir.Loop(wave, (read,)),),
            ((wave, loopir.WAVE),),
        )


def test_rewrite_mutates_the_program_in_place():
    program, _wave, load0, _load1, _reads = _g3_program()
    rewritten = loopir.rewrite(
        program,
        lambda node: () if node is load0 else node,
    )
    assert rewritten is program
    assert load0 not in tuple(loopir.walk(rewritten))
    assert rewritten.check() is rewritten


def test_program_body_check_can_be_deferred_explicitly():
    program, _wave, _load0, _load1, _reads = _g3_program()
    tile = program.axes[0]
    undeclared = loopir.Operand("undeclared", (tile,))
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            undeclared[tile],
            loopir.SHARED,
            loopir.REGISTER,
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )

    with pytest.raises(loopir.IRTypeError, match="undeclared operand"):
        program.with_body((read,))
    unchecked = program.with_body((read,), check=False)
    assert unchecked.body == (read,)
    with pytest.raises(loopir.IRTypeError, match="undeclared operand"):
        unchecked.check()


def test_tensorload_partition_mutates_generic_structure():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 3)
    a = loopir.Operand("A", (tile,))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    load = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            effect=a[wave, tile],
        ),),
        copy,
    )
    program = loopir.Program(
        (),
        (tile,),
        (a,),
        (load,),
        ((wave, loopir.WAVE),),
    )
    partitioned = loopir.partition_movement(
        program,
        load,
        (2 * tile + wave < 3, 2 * tile + wave >= 3),
        (copy, copy),
    )
    assert partitioned is program
    parts = tuple(
        node for node in loopir.walk(partitioned)
        if isinstance(node, loopir.Movement)
    )
    assert len(parts) == 2
    points = []
    for part in parts:
        write = next(
            effect for effect in loopir.effects(partitioned)
            if effect.operation is part
            and effect.kind is loopir.WRITE
            and effect.level is loopir.SHARED
        )
        points.append({
            (tuple(value for _axis, value in point.agents), point.tile)
            for point in loopir.effect_points(write)
        })
    assert not points[0] & points[1]
    assert len(points[0] | points[1]) == 6


def test_fuse_combines_transfers_but_not_domains():
    tile_a = loopir.Axis("tile_a", 2)
    tile_b = loopir.Axis("tile_b", 1)
    a = loopir.Operand("A", (tile_a,))
    b = loopir.Operand("B", (tile_b,))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    load_a = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[tile_a], loopir.GLOBAL, loopir.SHARED),),
        copy,
    )
    load_b = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(b[tile_b], loopir.GLOBAL, loopir.SHARED),),
        copy,
    )
    program = loopir.Program((), (tile_a, tile_b), (a, b), (load_a, load_b))
    fused = loopir.fuse_movements(program, (load_a, load_b), copy)
    operation = tuple(loopir.walk(fused))[0]
    assert isinstance(operation, loopir.Movement)
    assert tuple(
        transfer.local_tile.operand.name for transfer in operation.transfers
    ) == (
        "A",
        "B",
    )
    write_domains = tuple(
        tuple(axis.name for axis in loopir.effect_domain(effect).local_axes)
        for effect in loopir.effects(fused)
        if effect.kind is loopir.WRITE
    )
    assert write_domains == (("tile_a",), ("tile_b",))


def test_pipeline_mutates_composition_of_ordinary_nodes_and_markers():
    trip = loopir.Param("T")
    iteration = loopir.Axis("iter", trip)
    tile = loopir.Axis("tile", 1)
    a = loopir.Operand("A", (iteration, tile))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    load = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[iteration, tile], loopir.GLOBAL, loopir.SHARED),),
        copy,
    )
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(a[iteration, tile], loopir.SHARED, loopir.REGISTER),),
        loopir.IMMEDIATE,
    )
    main = loopir.Loop(iteration, 0, trip, (load, read))
    program = loopir.Program((trip,), (iteration, tile), (a,), (main,))
    pipelined = loopir.pipeline(program, main, ((load, 1),))

    assert pipelined is program
    guard = pipelined.body[0]
    assert isinstance(guard, loopir.Cond)
    assert tuple(
        node.tag for node in guard.then_body if isinstance(node, loopir.Region)
    ) == (loopir.PROLOGUE, loopir.STEADY, loopir.DRAIN)
    assert isinstance(guard.else_body[0], loopir.Region)
    assert guard.else_body[0].tag is loopir.SHORT
    assert not hasattr(loopir, "Peel")


def test_removed_first_attempt_values_are_not_public():
    for name in (
        "Placement",
        "Ref",
        "AgentPoint",
        "AgentSet",
        "Wait",
        "Sync",
        "TileFootprint",
        "TileAccess",
        "EffectDomain",
        "EffectPoint",
        "ScheduleIR",
    ):
        assert not hasattr(loopir, name)


def test_lowering_directly_converts_order_without_schedule_policy():
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move

    program, _wave, load0, load1, reads = _g3_program()
    gir = lower_to_gir(program)
    operations = [
        operation
        for block in gir.blocks.values()
        for operation in block.body
    ]
    assert all(isinstance(operation, Move) for operation in operations)
    assert tuple(operation.unit for operation in operations) == (
        ("A",),
        ("A",),
        (),
        (),
        (),
    )

    source = inspect.getsource(__import__(
        "Tensile.Lowering.loopir_to_gir",
        fromlist=("lower_to_gir",),
    ))
    assert "LoopModel.traversal" not in source
    assert "LoopModel.schedule" not in source
    assert "TDMSplit" not in source
    assert "TDMFuse" not in source


def test_stored_effect_change_removes_derived_edge_and_lowered_dependency():
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move

    tile = loopir.Axis("tile", 2)
    a = loopir.Operand("A", (tile,))
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[0], loopir.GLOBAL, loopir.SHARED),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(a[0], loopir.SHARED, loopir.REGISTER),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program((), (tile,), (a,), (copy, read))
    assert any(
        edge.kind is loopir.RAW
        and edge.complete.operation is copy
        and edge.issue.operation is read
        for edge in loopir.dependences(program)
    )
    lowered = lower_to_gir(program)
    moves = tuple(
        operation
        for block in lowered.blocks.values()
        for operation in block.body
        if isinstance(operation, Move)
    )
    assert moves[1].deps

    unrelated = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            a[0],
            loopir.SHARED,
            loopir.REGISTER,
            effect=a[1],
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    changed = loopir.Program((), (tile,), (a,), (copy, unrelated))
    assert not any(
        edge.kind is loopir.RAW
        and edge.complete.operation is copy
        and edge.issue.operation is unrelated
        for edge in loopir.dependences(changed)
    )
    changed_moves = tuple(
        operation
        for block in lower_to_gir(changed).blocks.values()
        for operation in block.body
        if isinstance(operation, Move)
    )
    assert not changed_moves[1].deps


def test_theta_pipeline_builds_valid_loopir_gir():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir import check_plan
    from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
    from Tensile.Lowering.gir.passes import run_pipeline

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
    }
    theta = adapter.params_to_theta(params)
    decoded = decode_theta(theta)
    assert not validate_loopir(decoded.program, decoded.profile)
    derived = loopir.dependences(decoded.program)
    required = loopir.required_dependences(decoded.program)
    assert derived and required
    actual = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(actual, passes=gir_pipeline())
    assert not check_plan(actual)


def test_dependency_free_lowering_preserves_emitted_gir():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
    from Tensile.Lowering.gir.passes import run_pipeline
    from Tensile.Lowering.gir.render import render_gir
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "LoopOrder": "KMN",
    }
    unchecked = decode_theta(adapter.params_to_theta(params))
    without_dependencies = lower_to_gir(
        unchecked.program,
        unchecked.profile,
        include_dependencies=False,
    )

    checked = decode_theta(adapter.params_to_theta(params))
    with_dependencies = lower_to_gir(checked.program, checked.profile)
    for block in with_dependencies.blocks.values():
        for operation in block.body:
            if hasattr(operation, "deps"):
                operation.deps = ()

    run_pipeline(
        with_dependencies,
        passes=gir_pipeline(),
        verify=False,
    )
    run_pipeline(
        without_dependencies,
        passes=gir_pipeline(),
        verify=False,
    )
    assert render_gir(with_dependencies) == render_gir(
        without_dependencies
    )


def test_target_lowering_uses_direct_ledger_not_symbolic_oracle(monkeypatch):
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    import Tensile.Lowering.loopir_to_gir as lowering

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "LoopOrder": "KMN",
    }))
    monkeypatch.setattr(
        lowering,
        "dependences",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("symbolic dependence oracle ran")
        ),
    )
    ledger = lowering._build_dependency_ledger(
        decoded.program,
        decoded.profile,
    )
    ledger_rows = sum(
        len(records)
        for records in ledger.values()
    )
    assert 0 < ledger_rows < 32
    assert lowering.lower_to_gir(decoded.program, decoded.profile).blocks


def test_wave_is_a_program_agent_not_a_loop_or_program_axis():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "NumWaves": 4,
    }))
    assert tuple(
        (axis.name, scope)
        for axis, scope in decoded.program.agents
    ) == (("wave", loopir.WAVE),)
    assert all(axis.name != "wave" for axis in decoded.program.axes)
    assert all(
        node.axis.name != "wave"
        for node in loopir.walk(decoded.program)
        if isinstance(node, loopir.Loop)
    )


def test_runtime_iter_is_projected_to_generation_phase_not_discarded():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
    }
    decoded = decode_theta(adapter.params_to_theta(params))
    gir = lower_to_gir(decoded.program, decoded.profile)
    steady_copies = [
        operation
        for operation in gir.blocks["steady"].body
        if isinstance(operation, Move)
        and operation.dsts
        and operation.dsts[0].tile.space == "shared"
    ]
    assert steady_copies
    assert all(operation.dsts[0].gdelta == 1 for operation in steady_copies)
    assert all(
        axis != "iter"
        for operation in steady_copies
        for axis, _value in operation.dsts[0].tile.coord
    )


def test_post_finalize_read_retiming_requires_profile_refinalization():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "LoopOrder": "KMN",
    }))
    read = next(
        operation for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.DS_LOAD
    )
    retimed = loopir.rewrite_operation(
        decoded.program,
        read,
        lambda operation: loopir.substitute_operation(
            operation,
            {
                axis: 0
                for axis in operation.transfers[0].local_tile.operand.axes
            },
        ),
    )
    errors = validate_loopir(retimed, decoded.profile)
    assert any("re-finalize" in error for error in errors)
    with pytest.raises(RuntimeError, match="re-finalize"):
        lower_to_gir(retimed, decoded.profile)


@pytest.mark.parametrize("order", ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"))
def test_all_basic_loop_orders_lower_to_valid_gir(order):
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": order,
    }
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_tdmsplit_tdmfuse_multiwave_program_lowers_to_valid_gir():
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 2, 1, 1],
        "TDMSplitWaveRegions": [1, 1],
        "NumWaves": 4,
        "TDMFuse": 0,
    }
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_decoder_wires_odd_read_groups_into_tensorload_effects():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import _Decoder

    params = {
        "MIWaveTile": [6, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 6, 2, 2, 1],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 1, 1, 1],
        "TDMSplitWaveRegions": [2, 1],
        "NumWaves": 2,
        "VectorWidthA": 2,
        "VectorWidthB": 2,
    }
    decoder = _Decoder(adapter.params_to_theta(params))
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        decoder._copy_nodes(),
        decoder.agents,
    )
    writes = tuple(
        effect
        for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
        and effect.level is loopir.SHARED
        and effect.tile.operand.name == "A"
    )
    assert len(writes) == 2
    points = [
        {
            point.tile
            for point in loopir.effect_points(
                effect,
                {program.params[0]: 1},
            )
        }
        for effect in writes
    ]
    assert all(len(region) == 6 for region in points)
    assert points[0].isdisjoint(points[1])
    assert (0, 0, 0, 2) in points[0]
    assert (1, 0, 0, 2) in points[1]
    assert (0, 0, 1, 0) in points[0]
    assert (1, 0, 1, 0) in points[1]


def test_whole_vector_single_wave_keeps_every_tdmsplit_region_live():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import _Decoder

    params = {
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 1, 1],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 1, 2, 1],
        "NumWaves": 1,
        "VectorWidthA": 4,
        "VectorWidthB": 4,
    }
    decoder = _Decoder(adapter.params_to_theta(params))
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        decoder._copy_nodes(),
        decoder.agents,
    )
    assert not validate_loopir(program)
    writes = tuple(
        effect
        for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
        and effect.level is loopir.SHARED
        and effect.tile.operand.name == "A"
    )
    assert len(writes) == 4
    assert all(
        loopir.effect_points(effect, {program.params[0]: 1})
        for effect in writes
    )
    finalized = _Decoder(adapter.params_to_theta(params)).build().program
    copies = {
        operation._identity
        for operation in loopir.operations(finalized)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
        and any(
            transfer.local_tile.operand.name == "A"
            for transfer in operation.transfers
        )
    }
    served = {
        dependence.complete.operation._identity
        for dependence in loopir.required_dependences(finalized)
        if dependence.kind is loopir.RAW
        and dependence.complete.tile.operand.name == "A"
        and isinstance(dependence.complete.operation, loopir.Movement)
        and dependence.complete.operation.op is loopir.TENSOR_LOAD
    }
    assert copies <= served


@pytest.mark.parametrize(
    "width,expected",
    (
        (8, {(0, 0): 0, (1, 0): 1}),
        (4, {(0, 0): 0, (1, 0): 0, (0, 1): 1, (1, 1): 1}),
    ),
)
def test_vector_width_changes_the_explicit_tdmsplit_effect(width, expected):
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import _Decoder

    decoder = _Decoder(adapter.params_to_theta({
        "MIWaveTile": [8, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 8, 2, 2, 1],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 1, 1, 1],
        "NumWaves": 2,
        "VectorWidthA": width,
        "VectorWidthB": 2,
    }))
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        decoder._copy_nodes(),
        decoder.agents,
    )
    copies = tuple(
        operation for operation in loopir.operations(program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
        and operation.transfers[0].local_tile.operand.name == "A"
    )
    expression = copies[0].effect_for(0).where.lhs
    axes = {axis.name: axis for axis in (*program.axes, *(
        axis for axis, _scope in program.agents
    ))}
    for (wave, split), region in expected.items():
        assert expression.evaluate({
            axes["wave"]: wave,
            axes["iter"]: 0,
            axes["K_inner"]: 0,
            axes["M_split"]: split,
            axes["M_inner"]: 0,
        }) == region


def test_cached_loopir_text_keeps_target_prefetch_and_vgpr_header():
    from types import SimpleNamespace

    from Tensile.Components.LoopModel.Program import loopModelIrTextCached
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S

    theta = adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "LoopOrder": "KMN",
    })
    depths, _minimum = build_S(theta)
    decoded = decode_theta(theta)
    writer = SimpleNamespace(
        states=SimpleNamespace(kernelName="diagnostic"),
        _loopModelThetaCache=("diagnostic", (theta, depths)),
        _loopModelMainloopCache=("diagnostic", decoded.program),
    )
    text = loopModelIrTextCached(writer)
    assert "# LoopModel geometry" in text
    assert "operand role" in text
    assert "register_depths:" in text
    assert "vgpr read_mode" in text
    assert "derived_dependences:" in text
    assert "\nprogram params(" in text


def test_two_chunk_global_prefetch_lowers_to_valid_gir():
    from Tensile.Tests.unit.gir_fixtures import BF16_NT_KMN_PLR0, theta

    point = theta(dict(BF16_NT_KMN_PLR0))
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


def test_nested_local_and_global_prefetch_lower_to_valid_gir():
    from Tensile.Tests.unit.gir_fixtures import BF16_NT_KMN, theta

    point = theta(dict(BF16_NT_KMN))
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


@pytest.mark.parametrize("split,waves", ((False, 1), (True, 4)))
@pytest.mark.parametrize("order", ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"))
@pytest.mark.parametrize("pgr", (1, 2))
@pytest.mark.parametrize("plr", (0, 1))
def test_prefetch_split_and_loop_order_gir_matrix(
    split, waves, order, pgr, plr
):
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": order,
        "PrefetchGlobalRead": pgr,
        "PrefetchLocalRead": plr,
        "NumWaves": waves,
    }
    if split:
        params.update({
            "TDMSplit": [2, 2, 1, 1],
            "TDMSplitWaveRegions": [1, 1],
            "TDMFuse": 0,
        })
    point = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


@pytest.mark.parametrize(
    "ring_overrides,expected_copies",
    (
        ({"VgprGroupA": 2, "VgprAllocA": 3}, 3),
        ({
            "VgprGroupB": 1,
            "VgprAllocB": 6,
            "PrefetchLocalReadA": 1,
                "PrefetchLocalReadB": 2,
        }, 6),
    ),
)
def test_physical_register_ring_projection_is_materialized_in_loopir(
    ring_overrides, expected_copies
):
    from Tensile.LoopModel import adapter, traversal
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": "KMN",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 2,
        "NumWaves": 1,
        **ring_overrides,
    }
    point = adapter.params_to_theta(params)
    depths, _minimum = build_S(point)
    copies = traversal.ring_trip_copies(point, depths)
    assert copies == expected_copies

    decoded = decode_theta(point)
    copy_steps = tuple(
        node for node in loopir.walk(decoded.program)
        if isinstance(node, loopir.Let)
        and node.axis.name == "_steady_copy"
    )
    assert len(copy_steps) == expected_copies
    assert tuple(step.value.constant_value for step in copy_steps) == tuple(
        range(expected_copies)
    )
    assert "loop_copies" not in inspect.signature(lower_to_gir).parameters
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


def test_ds_load_coverage_is_a_local_transfer_set():
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    iteration = loopir.Param("T")
    tile = loopir.Axis("tile", 4)
    operand = loopir.Operand("A", (tile,))
    read = loopir.Movement(
        loopir.DS_LOAD,
        (
            _transfer(
                operand[tile - tile % 2],
                loopir.SHARED,
                loopir.REGISTER,
            ),
            _transfer(
                operand[tile - tile % 2 + 1],
                loopir.SHARED,
                loopir.REGISTER,
            ),
        ),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program(
        (iteration,),
        (tile,),
        (operand,),
        (
            loopir.Region(loopir.STEADY),
            loopir.Loop(
                tile,
                0,
                tile.extent,
                (loopir.Cond(tile % 2 == 0, (read,), ()),),
            ),
        ),
    )
    writes = tuple(
        effect for effect in loopir.effects(program)
        if effect.operation is read
        and effect.kind is loopir.WRITE
    )
    assert len(writes) == 2
    assert all(not loopir.effect_domain(write).local_axes for write in writes)
    assert {
        point.tile
        for write in writes
        for point in loopir.effect_points(write)
    } == {(0,), (1,), (2,), (3,)}
    lowered = lower_to_gir(program)
    assert lowered.meta.get("read_fold", {}) == {}
    assert lowered.meta.get("coverage_axes", {}) == {}


def test_mx_read_order_is_explicit_in_loopir_and_lowers_to_valid_gir():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "LoopOrder": "MKN",
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMFuse": 0,
    }
    theta = adapter.params_to_theta(params)
    decoded = decode_theta(theta)
    compute = next(
        operation for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Compute)
    )
    assert tuple(
        tile.operand.name for tile in compute.reads
    ) == ("A", "B", "MXSA", "MXSB")

    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_mixed_tdmsplit_fused_copy_covers_the_peer_axis_logically():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "LoopOrder": "MKN",
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMSplit": (1, 2, 2, 1),
        "TDMSplitWaveRegions": (1, 1),
        "TDMFuse": 0,
    }
    theta = adapter.params_to_theta(params)
    decoded = decode_theta(theta)
    fused = next(
        operation for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
        and tuple(dict.fromkeys(
            transfer.local_tile.operand.name
            for transfer in operation.transfers
        )) == ("A", "B")
    )
    b_transfers = tuple(
        transfer for transfer in fused.transfers
        if transfer.local_tile.operand.name == "B"
    )
    transfer_b = b_transfers[0]
    k_split = next(
        index for index, axis in enumerate(transfer_b.local_tile.operand.axes)
        if axis.name == "K_split"
    )
    assert {
        transfer.local_tile.indices[k_split].constant_value
        for transfer in b_transfers
    } == {0, 1}
    assert all("bulk" not in axis.name for axis in decoded.program.axes)

    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


@pytest.mark.parametrize(
    "order,prefetch,split",
    (
        ("MKN", 1, None),
        ("KMN", 2, None),
        ("KMNMKN", 1, (2, 1, 1, 2)),
    ),
)
def test_target_coverage_inplace_read_order_lowers_to_valid_gir(
    order, prefetch, split
):
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": prefetch,
        "LoopOrder": order,
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMFuse": 0,
        "ReadVectorElems": {"MXSA": 8, "MXSB": 8},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    if split is not None:
        params.update({
            "TDMSplit": split,
            "TDMSplitWaveRegions": (1, 1),
        })
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_fused_mixed_coverage_drain_order_lowers_to_valid_gir():
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "LoopOrder": "MNKMNK",
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMFuse": 2,
        "TDMSplit": (2, 2, 1, 1),
        "TDMSplitWaveRegions": (2, 2),
        "ReadVectorElems": {"MXSA": 4, "MXSB": 4},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 2, "MXSB": 2},
    }
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_multi_axis_tensorload_regions_are_explicit_effect_partitions():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import _Decoder

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMNMKN",
        "NumWaves": 4,
        "TDMSplit": (2, 2, 2, 2),
        "TDMSplitWaveRegions": (1, 1),
    }
    decoder = _Decoder(adapter.params_to_theta(params))
    body = decoder._copy_nodes()
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        body,
        decoder.agents,
    )
    copies = tuple(
        operation for operation in loopir.operations(program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
    )
    assert len(copies) == 4
    assert not any(
        isinstance(node, loopir.Switch)
        for node in loopir.walk(program)
    )
    for region, copy in enumerate(copies):
        assert {
            copy.effect_for(index).where.rhs.constant_value
            for index in range(len(copy.transfers))
        } == {region}


def test_split_axis_without_retiming_lowers_as_an_expression():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMNMKN",
        "NumWaves": 4,
        "TDMSplit": (2, 2, 2, 2),
        "TDMSplitWaveRegions": (1, 1),
    }))
    assert not validate_loopir(decoded.program, decoded.profile)
    assert lower_to_gir(decoded.program, decoded.profile).blocks


def test_backward_same_iteration_relation_is_kept_conservatively():
    iteration = loopir.Axis("iteration", 2)
    operand = loopir.Operand("A", (iteration,))
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            operand[iteration],
            loopir.GLOBAL,
            loopir.SHARED,
        ),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            operand[iteration],
            loopir.SHARED,
            loopir.REGISTER,
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program(
        (),
        (iteration,),
        (operand,),
        (loopir.Loop(
            iteration,
            0,
            iteration.extent,
            (read, copy),
        ),),
    )
    relation = next(
        dependence for dependence in loopir.dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is copy
        and dependence.issue.operation is read
    )
    assert relation.domain.conservative


def test_no_load_loop_is_an_explicit_program_transform():
    from Tensile.LoopModel import adapter, traversal
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": "KMN",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 2,
        "VgprGroupA": 2,
        "VgprAllocA": 3,
        "SuppressNoLoadLoop": True,
    }
    point = adapter.params_to_theta(params)
    depths, _minimum = build_S(point)
    copies = traversal.ring_trip_copies(point, depths)
    decoded = decode_theta(point)
    assert loopir.DRAIN not in {
        region.tag for region in loopir.regions(decoded.program)
    }
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


# ------------------------------------------------------------------------------------------------
# Registers: ValuAlloc / VgprGroup and the footprint the writer emits
# ------------------------------------------------------------------------------------------------


pytestmark = pytest.mark.unit


_LEAVES = Path(__file__).resolve().parents[2] / "Lowering" / "leaves.py"


_KERNEL_WRITER_ASSEMBLY = Path(__file__).resolve().parents[2] / "KernelWriterAssembly.py"


_LOOPMODEL_REGISTERS = (Path(__file__).resolve().parents[2]
                        / "Components" / "LoopModel" / "Registers.py")


def _function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(namespace or {})
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


def _apply_method():
    return _function(_LOOPMODEL_REGISTERS, "applyLoopModelValuRegs")


def _writer(counts):
    return SimpleNamespace(
        states=SimpleNamespace(
            a=SimpleNamespace(numVgprValu=11),
            b=SimpleNamespace(numVgprValu=12),
            mxsa=SimpleNamespace(numVgprValu=13),
            mxsb=SimpleNamespace(numVgprValu=14),
            loopModelValuVgprs=dict(counts),
        ),
    )


def _alloc_kernel(use_loop_model=True, mx=True):
    return {
        "UseLoopModel": use_loop_model,
        "ProblemType": {
            "MXBlockA": 32 if mx else 0,
            "MXBlockB": 32 if mx else 0,
        },
    }


def test_ulm1_overwrites_scaffold_value_ring_sizes():
    writer = _writer({"A": 32, "B": 48, "MXSA": 8, "MXSB": 16})

    _apply_method()(writer, _alloc_kernel())

    assert writer.states.a.numVgprValu == 32
    assert writer.states.b.numVgprValu == 48
    assert writer.states.mxsa.numVgprValu == 8
    assert writer.states.mxsb.numVgprValu == 16


def test_ulm0_keeps_scaffold_value_ring_sizes():
    writer = _writer({"A": 32, "B": 48})

    _apply_method()(writer, _alloc_kernel(use_loop_model=False, mx=False))

    assert writer.states.a.numVgprValu == 11
    assert writer.states.b.numVgprValu == 12


def test_ulm1_requires_a_count_for_every_present_operand():
    writer = _writer({"A": 32})

    with pytest.raises(RuntimeError, match="B"):
        _apply_method()(writer, _alloc_kernel(mx=False))


def _buffer_counts(**counts):
    return SimpleNamespace(states=SimpleNamespace(**counts))


def test_ulm1_set_table_uses_each_operands_rotation_width():
    # vgprAllocationImplClassic stores one depth per operand.  The .set table
    # reads those fields; it does not take the max across operands.
    depth = _function(_LOOPMODEL_REGISTERS, "operandNumVgprBuffer")
    writer = _buffer_counts(numVgprBuffer=7, numVgprBufferA=2, numVgprBufferB=3,
                            numVgprBufferMXSA=0, numVgprBufferMXSB=0)
    count = _function(
        _KERNEL_WRITER_ASSEMBLY,
        "valuBufferCount",
        {"self": writer, "kernel": {"UseLoopModel": True}, "numBi": 7,
         "operandNumVgprBuffer": depth},
    )

    assert count("A") == 2
    assert count("B") == 3

    legacy = _buffer_counts(numVgprBuffer=7, numVgprBufferA=0, numVgprBufferB=0,
                            numVgprBufferMXSA=0, numVgprBufferMXSB=0)
    legacy_count = _function(
        _KERNEL_WRITER_ASSEMBLY,
        "valuBufferCount",
        {"self": legacy, "kernel": {"UseLoopModel": False}, "numBi": 7,
         "operandNumVgprBuffer": depth},
    )
    assert legacy_count("A") == 7


def _valu_alloc(kernel, tc, bpe):
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.Components.LoopModel.Registers import operandValuPuVaVg

    kernel = dict(kernel)
    kernel.setdefault("UseLoopModel", True)
    if "LoopOrder" not in kernel:
        kernel.setdefault("WmmaOuterOrder", 0)
    kernel.setdefault("PrefetchLocalRead", 0)
    kernel.setdefault("ClusterLocalRead", 0)
    kernel.setdefault(
        "LoopIters",
        int(kernel["DepthU"]) // int(kernel["MatrixInstK"]),
    )
    kernel.setdefault(
        "MIWaveTile",
        [kernel["MIWaveTileA"], kernel["MIWaveTileB"]],
    )
    kernel.setdefault(
        "MatrixInstruction",
        [
            16,
            16,
            kernel["MatrixInstK"],
            1,
            1,
            kernel["MIWaveTileA"],
            kernel["MIWaveTileB"],
            1,
            1,
        ],
    )
    Order.calculate(kernel)
    Registers.calculate(kernel)
    from Tensile.LoopModel import adapter
    theta = adapter.params_to_theta(
        adapter.kernel_to_params(kernel),
        validate_register_depths=False,
    )
    from Tensile.LoopModel.schedule import BufferDepths
    depths = BufferDepths(theta, include_shared=False)
    writer = SimpleNamespace(states=SimpleNamespace(
        bpr=4, packDTVA=False, convDTVA=False, packDTVB=False, convDTVB=False))
    writer._loopModelThetaCache = (id(kernel), (theta, depths))
    return operandValuPuVaVg(writer, kernel, tc, bpe)


def _valu_size(kernel, tc, bpe):
    """`(perBuffer, va)` -- the two terms most tests assert on."""
    alloc = _valu_alloc(kernel, tc, bpe)
    return alloc.perBuffer, alloc.va


def _nmk_kernel():
    # WIO5 = NMK, so both operands' prefetch unit is K.  DepthU 256 / MI K 128 = 2.
    return {
        "WmmaInnerOrder": 5,
        "DepthU": 256,
        "MatrixInstK": 128,
        "MIWaveTileA": 8,
        "MIWaveTileB": 8,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "VgprGroupA": 1,
        "VgprGroupB": 1,
        "VgprAllocA": 8,
        "VgprAllocB": 2,
        "ClusterLocalReadA": 1,
        "ClusterLocalReadB": 0,
        "PrefetchLocalReadA": 2,
        "PrefetchLocalReadB": 1,
        "TDMSplitA": 0,
        "TDMSplitB": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0},
    }


def test_nmk_va8_is_one_prefetch_unit_times_va():
    # One K-tile is 64 elems * 1 byte / 4 = 16 regs.  PU is 2, VG is 1, VA is 8.
    # 32 * 8 = 256.  The wave-tile block (8 * 16) times VA8 is 1024 and is not used.
    per, va = _valu_size(_nmk_kernel(), "A", 1)

    assert (per, va) == (32, 8)
    assert per * va == 2 * 8 // 1 * 16


def test_nmk_b_clr0_uses_stated_va_not_the_outer_loop():
    per, va = _valu_size(_nmk_kernel(), "B", 1)

    assert (per, va) == (32, 2)
    assert per * va == 64


def test_kmn_a_prefetch_unit_is_m_and_clr0_va_is_plr_plus_one():
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=1, VgprAllocA=-1, ClusterLocalReadA=0, PrefetchLocalReadA=1)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (128, 2)
    assert per * va == 8 * 2 // 1 * 16


def test_clr1_without_stated_va_uses_calculated_rotation_depth():
    kernel = _nmk_kernel()
    kernel["VgprAllocA"] = -1

    per, va = _valu_size(kernel, "A", 1)

    assert va == 8
    assert per * va == 256


def test_nmk_clr_split_keeps_k_sized_unit_instead_of_the_whole_rotation_set():
    """MT128x128 MIWT8x2 has A PU=K=2; M is counted by VA, not inside every slot."""
    kernel = _nmk_kernel()
    kernel.update(
        MIWaveTileB=2,
        MIWaveTile=[8, 2],
        MatrixInstruction=[16, 16, 128, 1, 1, 8, 2, 1, 4],
        TDMSplitA=1,
        TDMSplitB=1,
        VgprAllocA=-1,
        VgprAllocB=-1,
        VgprGroupA=-1,
        VgprGroupB=-1,
        ClusterLocalReadA=1,
        ClusterLocalReadB=1,
    )

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (32, 8)
    assert per * va == 256


def test_mkn_tdmsplit_mt_halves_b_prefetch_unit_and_keeps_vg1():
    # MKN: B's prefetch unit is N.  TDMSplit MT already halves it, so HalfPLR's
    # shape is VG1 VA3 PLR2 rather than VG2.
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=4, TDMSplitB=1, VgprGroupB=1, VgprAllocB=3,
                  ClusterLocalReadB=0, PrefetchLocalReadB=2)

    per, va = _valu_size(kernel, "B", 1)

    assert (per, va) == (64, 3)
    assert per * va == (8 // 2) * 3 // 1 * 16


def test_lowering_folds_the_layout_address_into_the_allocation():
    # The GIR layout address is a rotation block per ring slot.  When the ring
    # repeats the resident set over several prefetch generations it walks past
    # PU * VA / VG, and the generation it repeats is the buffer it reuses.
    fold = _function(_LEAVES, "allocated_valu_offset")

    # NMK VA8: slot stride 128, 8 slots.  The allocation is 8 * 32 = 256, so
    # slot 2 unit 0 is the next generation of buffer 0.
    assert fold(256, 7 * 128 + 7 * 16) == 240
    assert fold(256, 2 * 128 + 0 * 16) == 0
    # NKM: slot stride 128, 2 slots, allocation 4 * 64.  Nothing to fold, and
    # the two M_split halves keep their own registers.
    assert fold(256, 0 * 128 + 7 * 16) == 112
    assert fold(256, 1 * 128 + 7 * 16) == 240
    # A zero span is the non-UseLoopModel path.
    assert fold(0, 7 * 128 + 7 * 16) == 7 * 128 + 7 * 16


def test_lowering_spells_physical_valu_addresses_through_the_x_ring():
    fold = _function(_LEAVES, "allocated_valu_offset")
    symbol = _function(
        _LEAVES,
        "allocated_valu_symbol",
        {"allocated_valu_offset": fold},
    )

    assert symbol("A", 192, 64, 1, 0) == "ValuA_X0_I0"
    assert symbol("A", 192, 64, 1, 64 + 16) == "ValuA_X1_I0+16"
    assert symbol("A", 192, 64, 1, 128 + 48) == "ValuA_X2_I0+48"
    assert symbol("A", 256, 32, 1, 2 * 128) == "ValuA_X0_I0"


def test_halfplr_shape_is_vg2_va3():
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=1, VgprGroupA=2, VgprAllocA=3, ClusterLocalReadA=0,
                  PrefetchLocalReadA=2)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (64, 3)
    assert per * va == 8 * 3 // 2 * 16


def test_degenerate_free_axis_uses_the_same_k_unit_for_model_and_allocation():
    kernel = {
        "WmmaInnerOrder": 6,
        "WmmaOuterOrder": 2,
        "DepthU": 256,
        "MatrixInstK": 128,
        "MIWaveTileA": 1,
        "MIWaveTileB": 4,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "MIInputPerThreadMXSA": 4,
        "MIInputPerThreadMXSB": 4,
        "PrefetchLocalRead": 1,
        "ClusterLocalRead": 1,
        "TDMSplitA": 0,
        "TDMSplitB": 1,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0, "MXBlockA": 32, "MXBlockB": 32},
    }

    assert _valu_alloc(kernel, "A", 1).perBuffer == 32
    assert _valu_alloc(kernel, "MXSA", None).total == 2


def test_split_mxs_allocation_uses_its_model_footprint():
    kernel = {
        "LoopOrder": "KMKNMN",
        "DepthU": 256,
        "MatrixInstK": 128,
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "MIInputPerThreadMXSA": 4,
        "MIInputPerThreadMXSB": 4,
        "PrefetchLocalRead": 1,
        "ClusterLocalRead": 0,
        "TDMSplitA": 1,
        "TDMSplitB": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0, "MXBlockA": 32, "MXBlockB": 32},
    }

    # M_split advances the ring offset; it does not duplicate the physical PU.
    assert _valu_alloc(kernel, "MXSA", None).total == 4


def test_mab3_distributed_mxsb_uses_covered_unit_and_reduction_ring():
    kernel = {
        "UseLoopModel": True,
        "WmmaInnerOrder": 6,
        "WmmaOuterOrder": 2,
        "DepthU": 256,
        "MatrixInstK": 128,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstruction": [16, 16, 128, 1, 1, 1, 4, 1, 4],
        "MIWaveTile": [1, 4],
        "MIWaveTileA": 1,
        "MIWaveTileB": 4,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "MIInputPerThreadMXSA": 4,
        "MIInputPerThreadMXSB": 4,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "PrefetchLocalReadA": 1,
        "PrefetchLocalReadB": 1,
        "ClusterLocalRead": 1,
        "ClusterLocalReadA": 1,
        "ClusterLocalReadB": 1,
        "VgprGroupA": 1,
        "VgprGroupB": 1,
        "VgprAllocA": 1,
        "VgprAllocB": 4,
        "TDMSplitA": 0,
        "TDMSplitB": 1,
        "TDMFuse": 2,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "VectorWidthMXSA": 1,
        "VectorWidthMXSB": 1,
        "LocalReadVectorWidthMXS": 16,
        "ISA": [12, 5, 0],
        "MXScaleFormat": "InMemorySwizzle",
        "WavefrontSize": 32,
        "NumWaves": 4,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0, "MXBlockA": 32, "MXBlockB": 32},
    }

    alloc = _valu_alloc(kernel, "MXSB", None)
    assert (alloc.perBuffer, alloc.va, alloc.total) == (2, 4, 8)


def _f8_split_register_kernel(va_b=3):
    return {
        "UseLoopModel": True,
        "MatrixInstruction": [16, 16, 128, 1, 1, 8, 8, 2, 2],
        "MatrixInstK": 128,
        "DepthU": 256,
        "LoopIters": 2,
        "MIWaveTile": [8, 8],
        "MIWaveTileA": 8,
        "MIWaveTileB": 8,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "WmmaInnerOrder": 2,
        "WmmaOuterOrder": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 2,
        "ClusterLocalRead": 0,
        "ClusterLocalReadA": 0,
        "ClusterLocalReadB": 1,
        "VgprGroupA": 1,
        "VgprGroupB": 1,
        "VgprAllocA": 4,
        "VgprAllocB": va_b,
        "TDMSplitA": 1,
        "TDMSplitB": 1,
        "NumWaves": 4,
        "ElemBytes": 1,
        "ProblemType": {"Sparse": 0},
    }


def _f8_split_nkm_register_kernel():
    """The VA3/VA3 NKM shape whose split regions must not alias live inputs."""
    kernel = _f8_split_register_kernel()
    kernel.update(
        MatrixInstruction=[16, 16, 128, 1, 1, 8, 10, 2, 2],
        MIWaveTile=[8, 10],
        MIWaveTileB=10,
        WmmaInnerOrder=6,
        WmmaOuterOrder=2,
        PrefetchLocalRead=3,
        PrefetchLocalReadA=2,
        PrefetchLocalReadB=3,
        ClusterLocalReadA=0,
        ClusterLocalReadB=0,
        VgprAllocA=3,
        VgprAllocB=3,
    )
    return kernel


def test_calculated_f8_split_ring_matches_allocation():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import BufferDepths

    kernel = _f8_split_register_kernel()
    Order.calculate(kernel)
    Registers.calculate(kernel)
    theta = adapter.params_to_theta(kernel)
    depths = BufferDepths(theta, include_shared=False)

    assert kernel["VgprAllocA"] == geometry.group_ring_depth(
        theta, theta.op("A"), "g0", depths) == 4
    assert kernel["VgprAllocB"] == geometry.group_ring_depth(
        theta, theta.op("B"), "g0", depths) == 3


def test_split_regions_are_flattened_offsets_with_loop_split_refills():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import BufferDepths, Schedule, prefetch_refusals
    from Tensile.Lowering import lower_to_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    from Tensile.Lowering.gir.passes import pipeline, run_pipeline
    from Tensile.Lowering.gir.verify_dataflow import check_plan

    kernel = _f8_split_nkm_register_kernel()
    Order.calculate(kernel)
    Registers.calculate(kernel)
    theta = adapter.params_to_theta(kernel)
    depths = BufferDepths(theta, include_shared=False)
    schedule = Schedule(theta, depths)

    expected = {
        "A": (2, 4),
        "B": (3, 10),
    }
    for name, (plr, period) in expected.items():
        operand = theta.op(name)
        group = operand.fragment.groups()[0]
        width = geometry.group_ring_depth(theta, operand, group, depths)
        assert not hasattr(operand.fragment, "va_includes_regions")
        assert not hasattr(operand.fragment, "va_time_shares_regions")
        assert geometry.requested_read_ahead(theta, operand, depths) == plr
        assert schedule.want(operand) == plr
        assert schedule.steps(operand, group, plr) == plr
        assert geometry.ring_positions(theta, operand, group, width) == period

    a_slot = geometry.ring_slot(theta, theta.op("A"), "g0", 3)
    assert a_slot.evaluate({"M_split": 1, "K_inner": 0, "M_inner": 0}) == 2
    assert a_slot.evaluate({"M_split": 1, "K_inner": 1, "M_inner": 0}) == 0
    assert prefetch_refusals(theta, depths) == ()

    decoded = decode_theta(theta)
    split_axes = {
        axis.name
        for bindings in decoded.profile.read_issue_bindings.values()
        for axis, _value in bindings
    }
    assert split_axes == set()
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=pipeline())
    assert check_plan(program) == []
    reads = [action for action in plan_block(program, "steady") if action.kind == "read"]
    keys = [(action.at["tc"], action.at["tile_flat"], action.at["k_flat"],
             tuple(action.at.get("issue") or ()))
            for action in reads]
    assert len(keys) == len(set(keys))


def test_register_owner_calculates_all_eight_operand_parameters():
    from Tensile.Components.LoopModel import Order, Registers

    kernel = _f8_split_register_kernel()
    kernel.update({
        "PrefetchLocalReadA": -1,
        "PrefetchLocalReadB": -1,
        "ClusterLocalReadA": -1,
        "ClusterLocalReadB": -1,
        "VgprGroupA": -1,
        "VgprGroupB": -1,
        "VgprAllocA": -1,
        "VgprAllocB": -1,
    })
    Order.calculate(kernel)
    Registers.calculate(kernel)

    assert all(
        int(kernel[base + side]) >= 0
        for base in (
            "PrefetchLocalRead",
            "ClusterLocalRead",
            "VgprGroup",
            "VgprAlloc",
        )
        for side in ("A", "B")
    )


def test_adapter_rejects_calculated_ring_smaller_than_required_depth():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter

    kernel = _f8_split_register_kernel(va_b=1)
    Order.calculate(kernel)
    Registers.calculate(kernel)
    with pytest.raises(RuntimeError, match="VgprAllocB=1.*smaller.*rotation"):
        adapter.params_to_theta(kernel)


def test_wio1_clr1_va1_holds_its_complete_two_tile_prefetch_unit():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter

    kernel = _f8_split_register_kernel()
    kernel.update(
        MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2],
        MIWaveTile=[2, 2],
        MIWaveTileA=2,
        MIWaveTileB=2,
        DepthU=128,
        LoopIters=1,
        WmmaInnerOrder=1,
        WmmaOuterOrder=0,
        PrefetchLocalRead=2,
        ClusterLocalRead=1,
        ClusterLocalReadA=-1,
        ClusterLocalReadB=-1,
        VgprGroupA=-1,
        VgprGroupB=-1,
        VgprAllocA=-1,
        VgprAllocB=-1,
        TDMSplitA=0,
        TDMSplitB=0,
    )
    Order.calculate(kernel)
    Registers.calculate(kernel)

    assert kernel["VgprAllocA"] == 1
    adapter.params_to_theta(kernel)
