# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GIR lowering: the program built from a decoded theta, and the passes over it.

Covers `tensilelite.Lowering` -- `lower_to_gir`, the pass pipeline, the analyses (value placement,
region increment), the frame contract, and the dataflow checker.  The theta
each test starts from is a means; `test_loopmodel.py` owns what the model derives.
"""

import inspect
import itertools
import pytest
from test_loopmodel import BF16_NT_KMN, calculate_registers, mi_inputs, theta
from tensilelite.LoopModel import adapter
from tensilelite.LoopModel.adapter import kernel_to_params as _kernel_to_params
from tensilelite.LoopModel.decoder import decode_theta
from tensilelite.Lowering.loopir_to_gir import lower_to_gir
from tensilelite.Lowering.gir.passes import pipeline as gir_pipeline
from tensilelite.Lowering.gir.passes import run_pipeline
from tensilelite.Lowering.gir.verify_dataflow import check_plan
from tensilelite.Lowering.gir.analyses.region_increment import walk_violations
from tensilelite.Lowering.gir.analyses.value_placement import (
    RequiredValue, ValuePlacementSolver, BOTTOM, TOP, meet)
from tensilelite.Lowering.gir.nodes import BLOCK_ENTRY, BLOCK_EXIT
from tensilelite.Lowering.gir.nodes import (Program, Block, Mark, Pred, Bound, Goto, CondGoto,
                                        CondChain)
from tensilelite.Lowering.gir.analysis import AnalysisManager
from tensilelite.Lowering.gir.analyses.trip_domains import TripDomains
from tensilelite.Lowering.gir.emit_plan import plan_program
from tensilelite.Lowering.gir.frame_contract import build_contract
from tensilelite.Lowering.gir.nodes import CondChain


# ------------------------------------------------------------------------------------------------
# Decode, lower, verify: the configuration sweep over the whole path
# ------------------------------------------------------------------------------------------------



def loopOrder(inner, outer, tdmSplitA, tdmSplitB):
    """The LoopOrder word a TDMSplit pair gives: the WmmaFactors it pins, in WIO/WOO."""
    from tensilelite.Components.LoopModel.Order import forcedFactors, loopOrder as word
    return word(inner, outer,
                forcedFactors({"TDMSplitA": tdmSplitA, "TDMSplitB": tdmSplitB}))


def kernel_to_params(kernel):
    """Run the same register calculation as Solution before adapting the kernel."""

    return _kernel_to_params(calculate_registers(kernel))


def build_gir(theta, loop_copies=None):
    decoded = decode_theta(theta)
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=gir_pipeline())
    if loop_copies is not None:
        from tensilelite.Lowering import steady_chain
        assert len(steady_chain(program)) == max(1, int(loop_copies))
    return program


class _Bytes:
    """A DataType stand-in answering only what the bridge asks."""
    def __init__(self, n): self.n = n
    def numBytes(self): return self.n


_BASE = {
    "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1],
    "MIWaveTile": [2, 2], "MIWaveGroup": [1, 1],
    "MIBlock": [16, 16, 32, 1, 1, 1, 1, 1],
    "DepthU": 64, "WavefrontSize": 32,
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "HalfPLR": 0,
    "ProblemType": {"DataType": _Bytes(2), "DestDataType": _Bytes(2),
                    "ComputeDataType": _Bytes(4), "TransposeA": False, "TransposeB": True},
    "LoopOrder": "KMN", "NumLdsBlk": 2, "1LDSBuffer": -1,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "LocalReadVectorWidth": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
    "InnerUnroll": 1, "VgprPartition": 1,
    "enableTDMA": True, "enableTDMB": True, "NumWaves": 1, "TDMFuse": 0,
}


ORDERS3 = ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"]


def _mxf8(blk=None):
    o = {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1], "DepthU": 256,
         "LocalReadVectorWidth": 16, "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
         "ProblemType": {"DataType": _Bytes(1), "DestDataType": _Bytes(1),
                         "ComputeDataType": _Bytes(4), "TransposeA": True, "TransposeB": False}}
    if blk:
        o["ProblemType"] = {**o["ProblemType"], "MXBlockA": blk, "MXBlockB": blk, "Sparse": 0}
    return o


def test_k_axis_read_ahead_cap_counts_vg_buffers():
    from tensilelite.Components.LoopModel import Registers

    def cap(kernel, side):
        kernel = dict(kernel)
        kernel["PrefetchLocalRead"] = 999
        kernel["PrefetchLocalReadA"] = -1
        kernel["PrefetchLocalReadB"] = -1
        kernel["UseLoopModel"] = True
        Registers.calculate(kernel)
        return kernel["PrefetchLocalRead" + side]

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (1, 1)), 2, 1, "KMN", 0, 0, 1)
    k.update(LoopIters=2, WmmaInnerOrder=1, WmmaOuterOrder=3, VgprGroupA=1)
    k["LoopOrder"] = loopOrder(1, 3, 0, 0)
    vg1 = cap(k, "A")
    k["VgprGroupA"] = 2
    assert cap(k, "A") == 2 * vg1

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, 8, loopOrder(6, 2, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=6, WmmaOuterOrder=2, VgprGroupB=2)
    assert cap(k, "B") == 8


def _kernel(dover, mi, pgr, plr, order, sa, sb, vw):
    nw = mi[7] * mi[8]
    out = {**_BASE, **dover, "MatrixInstruction": list(mi),
            "MIWaveTile": [mi[5], mi[6]], "MIWaveGroup": [mi[7], mi[8]],
            "MIWaveTileA": mi[5], "MIWaveTileB": mi[6],
            "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr, "LoopOrder": order,
            "TDMSplitA": sa, "TDMSplitB": sb, "VectorWidthA": vw, "VectorWidthB": vw,
            "NumLdsBlk": max(2, pgr), "NumWaves": nw, "TDMFuse": 0}
    # after the overrides: a cell that changes MatrixInstruction must not inherit a stale count
    return mi_inputs(out)


def _mi(dlabel, wt, wg):
    k = [16, 16, 128, 1, 1] if dlabel != "bf16" else [16, 16, 32, 1, 1]
    return tuple(k + [wt[0], wt[1], wg[0], wg[1]])


@pytest.mark.parametrize(
    "order,pgr,split_a,split_b",
    (
        ("NMK", 1, 0, 0),
        ("MKN", 2, 1, 0),
        ("MKN", 2, 0, 2),
        ("KMNMKN", 2, 0, 2),
    ),
)
def test_folded_mxsb_refill_follows_its_last_broadcast_consumer(
    order, pgr, split_a, split_b
):
    """One folded scale read must not refill its slot between two M consumers."""
    k = _kernel(_mxf8(32), _mi("fp8", (2, 2), (2, 2)),
                pgr, 1, order, split_a, split_b, 2)
    k.update(
        ClusterLocalReadA=1,
        ClusterLocalReadB=1,
        VgprGroupA=1,
        VgprGroupB=1,
        VgprAllocA=2,
        VgprAllocB=2,
        ISA=[12, 5, 0],
        MXScaleFormat="InMemorySwizzle",
        LocalReadVectorWidthMXS=16,
        MatrixInstM=16,
        MatrixInstN=16,
    )
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    theta = adapter.params_to_theta(
        adapter.kernel_to_params(calculate_registers(k), target),
        validate_register_depths=False,
    )

    assert check_plan(build_gir(theta)) == []


def test_pgr3_decode_is_structurally_supported():
    """The decoder itself carries PGR3; Solution policy may still reject a concrete kernel."""
    k = {**_BASE, "PrefetchGlobalRead": 3, "NumLdsBlk": 3}
    assert build_gir(adapter.params_to_theta(kernel_to_params(k)))


def test_pgr3_decode_plan_is_clean():
    k = {**_BASE, "PrefetchGlobalRead": 3, "NumLdsBlk": 3}
    assert not check_plan(build_gir(adapter.params_to_theta(kernel_to_params(k))))


# ---------------------------------------------------------------------------
# loop_copies: GIR unrolls the steady body into a chain, and the emitter walks
# it. A count the emitter is not told about would be built and dropped.
# ---------------------------------------------------------------------------
def test_loop_copy_count_is_not_a_lowering_input():
    assert "loop_copies" not in inspect.signature(lower_to_gir).parameters


@pytest.mark.parametrize("vg,va,copies", [(-1, -1, 1), (2, 4, 1), (2, 2, 1), (2, 3, 3)])
def test_loop_copies_are_read_off_the_rings_not_off_a_parameter(vg, va, copies):
    """A ring the trip closes needs one block; `VG=2, VA=3` -- 1.5 buffers -- needs three."""
    from tensilelite.LoopModel.decoder import decode_theta
    from tensilelite.LoopModel.loopir import Let, walk
    k = _kernel(_mxf8(), _mi("fp8", (4, 4), (1, 1)), 1, 0, "KMN", 0, 0, 1)
    k["ClusterLocalRead"] = 0
    for side in ("A", "B"):
        k["ClusterLocalRead" + side] = 0
        k["VgprGroup" + side] = vg if side == "A" else -1
        k["VgprAlloc" + side] = va if side == "A" else -1
    theta = adapter.params_to_theta(kernel_to_params(k))
    program = decode_theta(theta).program
    steps = tuple(
        node for node in walk(program)
        if isinstance(node, Let) and node.axis.name == "_steady_copy"
    )
    assert (len(steps) or 1) == copies


def test_three_copy_frame_contract_has_backedge_incoming_and_guard_requirements():
    from tensilelite.LoopModel.schedule import BufferDepths
    from tensilelite.LoopModel import traversal as geometry
    from tensilelite.Lowering.gir.emit_plan import plan_program
    from tensilelite.Lowering.gir.frame_contract import build_contract

    k = _half_plr_kernel("NKM", 1)
    k["SuppressNoLoadLoop"] = True
    theta = adapter.params_to_theta(kernel_to_params(k))
    copies = geometry.ring_trip_copies(
        theta, BufferDepths(theta, include_shared=False))
    prog = build_gir(theta, loop_copies=copies)
    assert [prog.blocks[label].gen_rel for label in ("steady", "steady1", "steady2")] == [0, 1, 2]

    plans = plan_program(prog)
    anchors = {label: actions[0].action_id for label, actions in plans.items() if actions}
    contract = build_contract(prog)
    backedge = [edge for edge in contract.edges
                if edge.relative
                and edge.src == anchors["steady2"] and edge.dst == anchors["steady"]]
    assert backedge
    assert all((contract.generations[edge.gen].ring,
                contract.generations[edge.gen].entry,
                contract.generations[edge.gen].advance) == (2, 0, 1)
               for edge in backedge)
    # Every unrolled-copy transition advances the buffer frame by one.
    for source, destination in (("steady", "steady1"), ("steady1", "steady2")):
        incoming = [edge for edge in contract.edges
                    if edge.relative and edge.src == anchors[source]
                    and edge.dst == anchors[destination]]
        assert incoming
        assert all(edge.value == 1 for edge in incoming)

    # A separate control generation makes the copy chain one-way. Relative incoming alone is not
    # enough: without these singleton requirements ST can revisit a later copy through copy 0.
    copy_gens = {gen_id for gen_id, generation in contract.generations.items()
                 if generation.ring == 4}
    assert len(copy_gens) == 1
    copy_gen = next(iter(copy_gens))
    expected = {
        (anchors["prologue_join"], anchors["steady"]): ((0,), 0),
        (anchors["steady"], anchors["steady1"]): ((0, 1), 2),
        (anchors["steady1"], anchors["steady2"]): ((2,), 3),
        (anchors["steady2"], anchors["steady"]): ((3,), 1),
    }
    for key, (required, incoming_value) in expected.items():
        source, destination = key
        assert any(need.gen == copy_gen and need.src == source and need.dst == destination
                   and need.values == required for need in contract.requires)
        assert any(edge.gen == copy_gen and edge.src == source and edge.dst == destination
                   and not edge.relative and edge.value == incoming_value
                   for edge in contract.edges)

    # The shadow/skip arm owns its own full drain. Only the peeled prefetch arm enters the
    # modeled steady pipeline, otherwise its one-time rank-0 RAW forces tensorcnt(0) at the
    # recurring loop header.
    prefetch_gens = {gen_id for gen_id, generation in contract.generations.items()
                     if generation.ring == 3}
    assert len(prefetch_gens) == 1
    prefetch_gen = next(iter(prefetch_gens))
    assert any(need.gen == prefetch_gen
               and need.src == anchors["prologue_join"]
               and need.dst == anchors["steady"]
               and need.values == (2,)
               for need in contract.requires)


@pytest.mark.parametrize("vg,va,plr", [(1, 6, 4), (2, 12, 8)])
def test_deep_b_ring_keeps_full_plr_and_a_complete_three_copy_frame(vg, va, plr):
    """A 1.5-ring refills only slots whose old logical value has finished.

    The VA6 and VA12 spellings are the same shape at different VG granularity.  Both need three
    steady copies, no drain join, and a frame contract covering every copy edge.
    """
    from tensilelite.LoopModel import traversal as geometry
    from tensilelite.LoopModel.schedule import BufferDepths, Schedule
    from tensilelite.Lowering import steady_chain
    from tensilelite.Lowering.gir.emit_plan import plan_program
    from tensilelite.Lowering.gir.frame_contract import build_contract

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, plr, loopOrder(6, 2, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=6, WmmaOuterOrder=2,
             PrefetchLocalReadA=1, PrefetchLocalReadB=plr,
             ClusterLocalRead=0, ClusterLocalReadA=1, ClusterLocalReadB=0,
             VgprGroupB=vg, VgprAllocB=va, SuppressNoLoadLoop=True)
    theta = adapter.params_to_theta(kernel_to_params(k))
    depths = BufferDepths(theta, include_shared=False)
    operand = theta.op("B")
    plans = Schedule(theta, depths)
    assert geometry.requested_read_ahead(theta, operand, depths) == plr
    assert all(plans.plan(operand, group).prefetch_steps == plr
               for group in operand.fragment.groups())

    copies = geometry.ring_trip_copies(theta, depths)
    assert copies == 3
    prog = build_gir(theta, loop_copies=copies)
    assert not check_plan(prog)
    chain = steady_chain(prog)
    assert chain == ["steady", "steady1", "steady2"]
    assert not any(label.startswith("drain") for label in prog.blocks)
    assert [(prog.blocks[label].gen_rel, prog.blocks[label].chunk_base)
            for label in chain] == [(0, 0), (1, 1), (2, 2)]

    emitted = plan_program(prog)
    anchors = {label: emitted[label][0].action_id for label in chain}
    contract = build_contract(prog)
    backedge = [edge for edge in contract.edges
                if edge.relative
                and edge.src == anchors["steady2"] and edge.dst == anchors["steady"]]
    incoming = [edge for edge in contract.edges
                if not edge.relative and edge.dst == anchors["steady"]
                and edge.gen in {item.gen for item in backedge}]
    assert backedge and len(incoming) == len(backedge)
    required_edges = {(need.src, need.dst) for need in contract.requires}
    assert {(anchors["steady"], anchors["steady1"]),
            (anchors["steady1"], anchors["steady2"]),
            (anchors["steady2"], anchors["steady"])} <= required_edges


def test_the_covering_invariant_uses_the_derived_steady_period():
    """The calculated steady period covers exactly T."""
    from tensilelite.Lowering.gir.analysis import AnalysisManager
    from tensilelite.Lowering.gir.analyses.loop_shape import (
        Linear, LoopShape, reduction_coverage_violations)

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (2, 2)), 2, 1, "KMN", 0, 0, 1)
    k["SuppressNoLoadLoop"] = True
    th = adapter.params_to_theta(kernel_to_params(k))
    prog = build_gir(th)
    loops = AnalysisManager().get(LoopShape(), prog)
    steady = [lp for lp in loops if lp.header == "steady"]
    assert steady
    lp = steady[0]
    assert lp.per_trip == 1
    assert lp.trips == Linear(1, 0)
    assert lp.covers == lp.trips
    assert not reduction_coverage_violations(prog, loops)


# ---------------------------------------------------------------------------
# HalfPLR shape: VG=2 halves one buffer and VA=3 allocates three halves.
# ---------------------------------------------------------------------------
def _half_plr_kernel(order, half_plr):
    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (2, 2)), 2, 2, order, 0, 0, 1)
    k.update(HalfPLR=0, ClusterLocalRead=0,
             ClusterLocalReadA=0, ClusterLocalReadB=0)
    for side, bit in (("A", 1), ("B", 2)):
        stated = bool(half_plr & bit)
        k["VgprGroup" + side] = 2 if stated else -1
        k["VgprAlloc" + side] = 3 if stated else -1
    return k


@pytest.mark.parametrize("half_plr, halved", [(1, "A"), (2, "B"), (3, "AB")])
def test_halfplr_costs_one_and_a_half_buffers_on_the_named_operand(half_plr, halved):
    """VG2/VA3 is three halves: 1.5 PU, or 3/4 of a VG1/VA2 allocation.

    And only for the operand the bitmask names: HalfPLR=1 leaves B alone.
    """
    from tensilelite.LoopModel.schedule import BufferDepths
    from tensilelite.LoopModel import traversal as g

    def regs(k):
        th = adapter.params_to_theta(kernel_to_params(k))
        d = BufferDepths(th, include_shared=False)
        return th, {n: g.operand_emitted_regs(th, th.op(n), d) for n in ("A", "B")}

    half_kernel = _half_plr_kernel("KMN", half_plr)
    th, half = regs(half_kernel)
    for name in ("A", "B"):
        if name in halved:
            full_kernel = dict(half_kernel)
            full_kernel["VgprGroup" + name] = 1
            full_kernel["VgprAlloc" + name] = 2
            _, full = regs(full_kernel)
            assert half[name] * 4 == full[name] * 3, (name, half, full)
            assert th.op(name).fragment.vgpr_group == 2
            assert th.op(name).fragment.ring_depths == {"g0": 3}
        else:
            assert th.op(name).fragment.vgpr_group == 1


def test_clr_and_explicit_va_precedence_is_calculated_in_registers():
    k = _half_plr_kernel("KMN", 1)

    # CLR0: explicit VA replaces PLR+1.
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}

    # CLR0 with VA=-1: derive PLR2+1.
    k["VgprAllocA"] = -1
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}

    # CLR0 with VG=-1: explicit VA alone overrides PLR+1 at the default VG1.
    k["VgprGroupA"] = -1
    k["VgprAllocA"] = 5
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.vgpr_group == 1
    assert theta.op("A").fragment.ring_depths == {"g0": 5}

    # Explicit VA remains authoritative under CLR1.
    k["VgprGroupA"] = 2
    k["VgprAllocA"] = 3
    k.update(ClusterLocalRead=1, ClusterLocalReadA=1, ClusterLocalReadB=1)
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}


def test_all_inner_forced_clr1_retains_a_large_enough_explicit_va():
    from tensilelite.LoopModel import traversal as geometry
    from tensilelite.LoopModel.schedule import BufferDepths, Schedule

    def assert_prefetch_is_kept(theta):
        depths = BufferDepths(theta, include_shared=False)
        operand = theta.op("A")
        requested = geometry.requested_read_ahead(theta, operand, depths)
        assert requested > 0
        assert Schedule(theta, depths).plan(operand, "g0").prefetch_steps == requested

    k = _half_plr_kernel("NKM", 1)
    # One all-inner buffer is a complete rotation unit; VG2 therefore needs two such halves.
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}
    assert_prefetch_is_kept(theta)

    k["VgprAllocA"] = 1
    k["ClusterLocalReadA"] = 1
    with pytest.raises(RuntimeError, match="smaller.*rotation"):
        adapter.params_to_theta(kernel_to_params(k))


def test_forced_all_inner_rejects_explicit_va_below_required_rotation():
    """Registers owns the floor; adapter rejects an explicit smaller allocation."""

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (1, 1)),
                2, 2, loopOrder(5, 0, 1, 0), 1, 0, 2)
    k.update(WmmaInnerOrder=5, WmmaOuterOrder=0,
             ClusterLocalRead=0, ClusterLocalReadA=1, ClusterLocalReadB=0,
             VgprGroupA=2, VgprAllocA=3,
             VgprGroupB=-1, VgprAllocB=-1,
             SuppressNoLoadLoop=True, SourceSwap=False)
    with pytest.raises(RuntimeError, match="VgprAllocA=3.*smaller"):
        adapter.params_to_theta(kernel_to_params(k))


def test_f8maf2_clra_allocates_the_complete_resident_set():
    """CLR1 A needs four prefetch-unit buffers, not two region slots."""
    from tensilelite.LoopModel import traversal as geometry
    from tensilelite.LoopModel.schedule import BufferDepths

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, 4, loopOrder(6, 2, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=6, WmmaOuterOrder=2,
             PrefetchLocalReadA=1, PrefetchLocalReadB=4,
             ClusterLocalRead=0, ClusterLocalReadA=1, ClusterLocalReadB=0,
             VgprGroupA=-1, VgprAllocA=-1,
             VgprGroupB=1, VgprAllocB=6,
             SuppressNoLoadLoop=True, SourceSwap=True)

    params = kernel_to_params(k)
    theta = adapter.params_to_theta(params)
    depths = BufferDepths(theta, include_shared=False)

    assert (params["VgprAllocA"], params["VgprAllocB"]) == (4, 6)
    assert geometry.register_layout(theta, depths)["A"]["total"] == 256
    assert geometry.register_layout(theta, depths)["B"]["total"] == 192


@pytest.mark.parametrize("order", ["NKM"])
@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_halfplr_is_clean_or_refused_never_broken(order, half_plr):
    """The YAML's canonical NKM cell lowers cleanly at a three-copy chain."""
    from tensilelite.Lowering import steady_chain
    k = _half_plr_kernel(order, half_plr)
    k["SuppressNoLoadLoop"] = True
    prog = build_gir(adapter.params_to_theta(kernel_to_params(k)), loop_copies=3)
    assert len(steady_chain(prog)) == 3
    assert not check_plan(prog)


# ---------------------------------------------------------------------------
# SuppressNoLoadLoop: the scaffold folds the ramp-out into the loop and clamps the
# overrun loads, so the model says the same -- no drain, and the loop owns all T.
# ---------------------------------------------------------------------------
def _drain_shape(kernel, loop_copies=1):
    """(drain block labels, steady `covers`, meta peel/drain) for one kernel."""
    from tensilelite.Lowering.gir import analysis as _an
    from tensilelite.Lowering.gir.analyses.loop_shape import LoopShape, reduction_coverage_violations
    prog = build_gir(adapter.params_to_theta(kernel_to_params(kernel)), loop_copies=loop_copies)
    loops = _an.AnalysisManager().get(LoopShape(), prog)
    steady = next(lp for lp in loops if lp.header == "steady")
    return (sorted(l for l in prog.blocks if l.startswith("drain")),
            steady, prog.meta["peel_depth"], prog.meta["drain_steps"],
            reduction_coverage_violations(prog, loops), check_plan(prog))


@pytest.mark.parametrize("order", ORDERS3)
def test_default_keeps_the_drain(order):
    """CONTROL ARM. Without the flag nothing moves: M drain blocks, loop covers `T - M`."""
    k = _half_plr_kernel(order, 0)
    drains, steady, peel, steps, cov, plan = _drain_shape(k)
    assert len(drains) == peel == steps == 2
    assert (steady.covers.coeff, steady.covers.const) == (1, -2)
    assert steady.exit_to == "drain0"
    assert not cov and not plan


@pytest.mark.parametrize("order", ORDERS3)
def test_suppressed_no_load_loop_has_no_drain_and_the_loop_owns_every_chunk(order):
    """The prologue keeps its lead; the ramp-out goes; the loop's count grows by exactly M."""
    k = _half_plr_kernel(order, 0)
    k["SuppressNoLoadLoop"] = True
    drains, steady, peel, steps, cov, plan = _drain_shape(k)
    assert drains == []
    assert peel == 2 and steps == 0            # the lead survives, the ramp-out does not
    assert (steady.covers.coeff, steady.covers.const) == (1, 0)
    assert steady.exit_to == "end"
    assert not cov and not plan


@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_halfplr_lowers_drain_free_over_its_three_copies(half_plr):
    """VG2/VA3 uses per-copy exits, so a partial final unroll still covers all T chunks."""
    from tensilelite.Lowering import steady_chain
    k = _half_plr_kernel("NKM", half_plr)
    k["SuppressNoLoadLoop"] = True
    prog = build_gir(adapter.params_to_theta(kernel_to_params(k)), loop_copies=3)
    assert len(steady_chain(prog)) == 3
    drains, steady, _peel, steps, cov, plan = _drain_shape(k, loop_copies=3)
    assert drains == [] and steps == 0
    assert (steady.covers.coeff, steady.covers.const) == (1, 0)
    assert steady.trips.div == 1               # copy exits preserve T % 3
    assert steady.per_trip == 1
    assert not cov and not plan


# ---------------------------------------------------------------------------
# A region is an ordinary digit of the PU offset.  A short ring may defer a refill, but it never
# manufactures another read on an invariant consumer pass.
# ---------------------------------------------------------------------------
def _split_half_plr_theta(order, half_plr, sa, sb, wt=(2, 2), clr=0, pgr=2):
    k = _kernel(_mxf8(), _mi("fp8", wt, (1, 1)), pgr, 2, order, sa, sb, 1)
    k.update(HalfPLR=0, ClusterLocalRead=clr,
             ClusterLocalReadA=clr, ClusterLocalReadB=clr)
    for side, bit in (("A", 1), ("B", 2)):
        stated = bool(half_plr & bit)
        walks = ("M", "K") if side == "A" else ("N", "K")
        pu_axis = next(axis for axis in reversed(order) if axis in walks)
        split = sa if side == "A" else sb
        split_axis = (("M" if side == "A" else "N") if split == 1
                      else "K" if split == 2 else None)
        # TDMSplit has already supplied the half when it cuts this operand's PU.
        vg = 1 if split_axis == pu_axis else 2
        k["VgprGroup" + side] = vg if stated else -1
        k["VgprAlloc" + side] = 3 if stated else -1
    k["SuppressNoLoadLoop"] = True
    return adapter.params_to_theta(kernel_to_params(k))


def test_the_shallow_half_keeps_a_ring_digit_when_the_region_leads():
    """A TDMSplit-provided half is VG1/VA3 and still retains a physical ring digit."""
    from tensilelite.LoopModel import traversal as g
    from tensilelite.LoopModel.schedule import BufferDepths

    th = _split_half_plr_theta(loopOrder(1, 1, 1, 0), 1, sa=1, sb=0)
    op, depths = th.op("A"), BufferDepths(th, include_shared=False)
    assert op.fragment.vgpr_group == 1
    assert op.fragment.ring_depths == {"g0": 3}
    assert g.split_unit_divisor(th, op) == 2
    slot = g.ring_slot(th, op, "g0", g.group_ring_depth(th, op, "g0", depths))
    assert slot is not None
    assert "M_split" in {axis.name for axis in slot.free_axes()}


@pytest.mark.parametrize("wio", [1, 2])
@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_tdmsplit_consuming_vg_still_builds_the_three_copy_ring(wio, half_plr):
    """A split that halves the PU uses VG1 and still builds the HalfPLR ring.

    The ring must have three copies: one steady block would return to the wrong
    half-buffer.
    """
    from tensilelite.LoopModel.schedule import BufferDepths
    from tensilelite.LoopModel import traversal as geometry
    from tensilelite.Lowering import steady_chain
    from tensilelite.Lowering.gir.emit_plan import plan_program
    from tensilelite.Lowering.gir.frame_contract import build_contract

    theta = _split_half_plr_theta(loopOrder(wio, 1, 1, 1), half_plr, 1, 1)
    shaped = [theta.op(side) for side, bit in (("A", 1), ("B", 2)) if half_plr & bit]
    assert shaped
    assert all(geometry.vgpr_group(theta, op) == 1 for op in shaped)
    assert all(geometry.split_unit_divisor(theta, op) == 2 for op in shaped)
    depths = BufferDepths(theta, include_shared=False)
    assert all(geometry.vgpr_ring_carries(
        theta, op, op.fragment.groups()[0],
        geometry.group_ring_depth(theta, op, op.fragment.groups()[0], depths))
        for op in shaped)

    copies = geometry.ring_trip_copies(theta, depths)
    prog = build_gir(theta, loop_copies=copies)
    assert copies == 3
    assert steady_chain(prog) == ["steady", "steady1", "steady2"]
    assert not check_plan(prog)

    plans = plan_program(prog)
    anchors = {label: actions[0].action_id
               for label, actions in plans.items() if actions}
    contract = build_contract(prog)
    assert any(edge.relative
               and edge.src == anchors["steady2"]
               and edge.dst == anchors["steady"]
               for edge in contract.edges)
    assert contract.requires


def test_a_loopmodel_refusal_names_the_solution_exactly_once():
    """A bare refusal costs a bisect to place, and saying the name twice is noise."""
    from tensilelite.Components.LoopModel.Theta import loopModelTheta, namingSolution

    class _Writer:
        class states:
            kernelName = "Cijk_Alik_Bljk_F8SS_NAMED"

    k = _kernel(_mxf8(), _mi("fp8", (4, 4), (1, 1)), 1, 2, "KMN", 0, 0, 1)
    k["ClusterLocalRead"] = 0
    for side in ("A", "B"):
        k["ClusterLocalRead" + side] = 0
        k["VgprGroup" + side] = 3 if side == "A" else -1
        k["VgprAlloc" + side] = -1
    with pytest.raises((ValueError, RuntimeError)) as refusal:
        loopModelTheta(_Writer(), k)
    assert str(refusal.value).count("solution:") == 1
    assert "Cijk_Alik_Bljk_F8SS_NAMED" in str(refusal.value)

    writer = _Writer()
    with pytest.raises(ValueError) as nested:
        with namingSolution(writer):
            with namingSolution(writer):
                raise ValueError("inner failure")
    assert str(nested.value).count("solution:") == 1


# ------------------------------------------------------------------------------------------------
# ValuePlacementSolver: the shared def-placement fixpoints
# ------------------------------------------------------------------------------------------------



def _raise(kind, detail):
    raise AssertionError("%s: %r" % (kind, detail))


class N:
    """A stand-in body node; identity is what the anchors compare."""
    def __init__(self, tag):
        self.tag = tag
    def __repr__(self):
        return "N(%s)" % self.tag


def _pipeline(mod=2, adv=1):
    """prologue -> steady -> drain, steady self-loop (the back edge).
    Every block copies once.  Frames: the back edge advances the chunk by `adv`; prologue and drain
    sit in the same frame as steady for simplicity, so the only delta is the back edge.
    """
    p, s, d = N("copy@prologue"), N("copy@steady"), N("copy@drain")
    acc = {"prologue": [RequiredValue("prologue", p, 0)],
           "steady":   [RequiredValue("steady",   s, 0)],
           "drain":    [RequiredValue("drain",    d, 0)]}
    succs = {"prologue": ["steady"], "steady": ["steady", "drain"], "drain": []}
    preds = {"prologue": [], "steady": ["prologue", "steady"], "drain": ["steady"]}
    # The latch's per-trip def runs before the exit branch too, so the drain is in the POST-advance
    # frame — exactly what `Block.chunk_base` encodes in the real program.
    delta = lambda a, b: adv if a == "steady" and b in ("steady", "drain") else 0
    return acc, preds, succs, delta, (p, s, d)


def test_meet_is_a_lattice_not_a_preference():
    assert meet(BOTTOM, 3) == 3 and meet(3, BOTTOM) == 3
    assert meet(3, 3) == 3
    assert meet(3, 4) is TOP          # disagreement is reported, never resolved to a side
    assert meet(TOP, 3) is TOP


def test_selfloop_pointer_advances_once_per_trip_on_the_back_edge():
    """A rotating pointer whose every access names the same slot in its own frame still has to
    rotate once per trip — the whole rotation IS the back edge.  The def must therefore land at the
    END of the body (after that trip's access), not before it."""
    acc, preds, succs, delta, (p, s, d) = _pipeline(mod=2, adv=1)
    solver = ValuePlacementSolver(acc, preds, succs, delta, "prologue", 0, modulus=2)
    places = solver.solve(_raise)
    back = [pl for pl in places if pl.block == "steady" and pl.at_exit]
    assert len(back) == 1, places
    assert back[0].block == "steady"
    # after the body's own access, before the block ends: the trip bottom.
    assert back[0].after is s and back[0].before == BLOCK_EXIT


def test_prologue_handoff_becomes_a_def_at_the_end_of_the_prologue():
    """The pipeline shape that motivated this module.  The prologue leaves the descriptor at the
    chunk it just fetched; the steady body's copy needs the NEXT one.  That transition is an edge
    def, and the edge's own block is the prologue — so it lands AFTER the prologue's copy, which is
    exactly the hand-written scaffold's `prologue: copy(0); advance->1`."""
    p, s = N("copy@prologue"), N("copy@steady")
    acc = {"prologue": [RequiredValue("prologue", p, 0)], "steady": [RequiredValue("steady", s, 1)]}
    succs = {"prologue": ["steady"], "steady": []}
    preds = {"prologue": [], "steady": ["prologue"]}
    places = ValuePlacementSolver(acc, preds, succs, lambda a, b: 0,
                                  "prologue", 0).solve(_raise)
    edge = [pl for pl in places if pl.block == "prologue" and pl.at_exit]
    assert len(edge) == 1, places
    assert edge[0].block == "prologue"          # NOT inside the loop body
    assert edge[0].after is p                   # immediately after the copy it supersedes
    assert edge[0].before == BLOCK_EXIT
    assert (edge[0].from_value, edge[0].to_value) == (0, 1)


def test_intra_block_change_yields_a_window_between_the_two_accesses():
    a, b = N("read@0"), N("read@1")
    acc = {"steady": [RequiredValue("steady", a, 0), RequiredValue("steady", b, 1)]}
    places = ValuePlacementSolver(acc, {"steady": []}, {"steady": []},
                                  lambda x, y: 0, "steady", 0, modulus=2).solve(_raise)
    inner = [pl for pl in places if not pl.at_exit and pl.after is a]
    assert len(inner) == 1, places
    # the WINDOW is both accesses; picking a point inside it is PlacementPass's policy, not ours.
    assert inner[0].before is b and (inner[0].from_value, inner[0].to_value) == (0, 1)


def test_no_change_no_def():
    p, s = N("c0"), N("c1")
    acc = {"a": [RequiredValue("a", p, 0)], "b": [RequiredValue("b", s, 0)]}
    places = ValuePlacementSolver(acc, {"a": [], "b": ["a"]}, {"a": ["b"], "b": []},
                                  lambda x, y: 0, "a", 0).solve(_raise)
    assert places == []


def test_accessless_block_passes_the_value_through_both_ways():
    """A block with no access must not absorb the transition: the def belongs on the edge where
    supply and demand actually differ, not on every edge of the chain."""
    p, s = N("copy@a"), N("copy@c")
    acc = {"a": [RequiredValue("a", p, 0)], "b": [], "c": [RequiredValue("c", s, 0)]}
    places = ValuePlacementSolver(acc, {"a": [], "b": ["a"], "c": ["b"]},
                                  {"a": ["b"], "b": ["c"], "c": []},
                                  lambda x, y: 0, "a", 0).solve(_raise)
    assert places == [], places


def test_conflicting_join_is_reported_not_resolved():
    """Two predecessors leaving different values: there is no single correct def, and preferring a
    predecessor is the silent-wrong-answer this whole layer exists to avoid."""
    x, y, z, u = N("c@x"), N("c@y"), N("c@z"), N("c@u")
    acc = {"entry": [], "x": [RequiredValue("x", x, 0)],
           "y": [RequiredValue("y", y, 1)], "z": [RequiredValue("z", z, 0)],
           "u": [RequiredValue("u", u, 1)]}
    # y cannot change at exit: z needs 0 while u still needs 1.  Therefore z genuinely receives
    # different values from x and y.
    succs = {"entry": ["x", "y"], "x": ["z"], "y": ["z", "u"], "z": [], "u": []}
    preds = {"entry": [], "x": ["entry"], "y": ["entry"], "z": ["x", "y"], "u": ["y"]}
    seen = []
    ValuePlacementSolver(acc, preds, succs, lambda a, b: 0, "entry", 0).solve(
        lambda kind, detail: seen.append(kind))
    assert seen == ["join"]


def test_agreed_join_places_one_definition_at_the_common_block_entry():
    """CFG fan-in is not a value conflict when every reaching predecessor supplies the same value."""
    # `u` keeps p's old value live, so the change cannot be hoisted to p's exit.
    a, b, c = N("c@p"), N("c@t"), N("c@u")
    acc = {"p": [RequiredValue("p", a, 0)], "q": [], "t": [RequiredValue("t", b, 1)],
           "u": [RequiredValue("u", c, 0)]}
    succs = {"p": ["t", "u"], "q": ["t"], "t": [], "u": []}
    preds = {"p": [], "q": [], "t": ["p", "q"], "u": ["p"]}
    places = ValuePlacementSolver(
        acc, preds, succs, lambda x, y: 0, "p", 0).solve(_raise)
    head = [place for place in places if place.block == "t"]
    assert len(head) == 1, places
    assert head[0].after == BLOCK_ENTRY and head[0].before is b
    assert (head[0].from_value, head[0].to_value) == (0, 1)
    assert not any(place.block == "p" and place.at_exit for place in places)


def test_frame_delta_makes_cross_frame_blocks_comparable():
    """Same physical slot, different chunk frames: without the delta the solver would invent a def
    that the hardware does not need."""
    s, d = N("copy@steady"), N("copy@drain")
    acc = {"steady": [RequiredValue("steady", s, 0)], "drain": [RequiredValue("drain", d, 1)]}
    succs, preds = {"steady": ["drain"], "drain": []}, {"steady": [], "drain": ["steady"]}
    # drain sits one chunk further along, so demand 1 in drain IS supply 0 in steady.
    places = ValuePlacementSolver(acc, preds, succs,
                                  lambda a, b: -1 if (a, b) == ("steady", "drain") else 0,
                                  "steady", 0).solve(_raise)
    assert places == [], places


def test_demand_propagates_back_through_accessless_blocks_to_the_earliest_point():
    """The backward fixpoint is what makes EARLIEST earliest.
    a(uses 0) -> b(no access) -> c(uses 1).  The transition must be hoisted all the way back to the
    end of `a` — the last point the old value is still live — NOT left at b's exit next to the use.
    """
    a, c = N("copy@a"), N("copy@c")
    acc = {"a": [RequiredValue("a", a, 0)], "b": [], "c": [RequiredValue("c", c, 1)]}
    places = ValuePlacementSolver(acc, {"a": [], "b": ["a"], "c": ["b"]},
                                  {"a": ["b"], "b": ["c"], "c": []},
                                  lambda x, y: 0, "a", 0).solve(_raise)
    assert len(places) == 1, places
    assert places[0].block == "a", "demand did not reach back past the accessless block"
    assert places[0].after is a and places[0].at_exit
    assert (places[0].from_value, places[0].to_value) == (0, 1)


def test_demand_from_two_successors_must_agree_to_share_one_exit_def():
    """One register, one exit: two successors demanding different values cannot share a def at the
    predecessor's exit.  If the fixpoint quietly picked one, the other path would run with the
    wrong buffer — the silent-wrong-answer this layer exists to prevent."""
    a, s, t = N("copy@a"), N("copy@s"), N("copy@t")
    acc = {"a": [RequiredValue("a", a, 0)], "s": [RequiredValue("s", s, 1)], "t": [RequiredValue("t", t, 2)]}
    succs, preds = {"a": ["s", "t"], "s": [], "t": []}, {"a": [], "s": ["a"], "t": ["a"]}
    seen = []
    places = ValuePlacementSolver(acc, preds, succs, lambda x, y: 0, "a", 0, modulus=4).solve(
        lambda kind, detail: seen.append(kind))
    assert seen == [], seen
    # s and t each have ONE predecessor, so each transition is legally realized at that successor's
    # HEAD -- two distinct defs, never one shared exit def at `a`.
    assert not any(pl.at_exit for pl in places), places
    by_block = {pl.block: pl for pl in places}
    assert set(by_block) == {"s", "t"}, places
    assert by_block["s"].to_value == 1 and by_block["t"].to_value == 2
    assert by_block["s"].from_value == 0 and by_block["t"].from_value == 0
    assert by_block["s"].after == BLOCK_ENTRY and by_block["s"].before is s


# ------------------------------------------------------------------------------------------------
# The physical main-loop scaffold, as the frame contract represents it
# ------------------------------------------------------------------------------------------------



def _pgr2_program():
    return build_gir(theta({
        **BF16_NT_KMN,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
    }))


def test_scaffold_map_names_all_three_folded_main_loop_exits():
    """T==1, direct T==2, and the eventual steady exit are three physical scaffold paths."""
    prog = _pgr2_program()

    term = prog.blocks["prologue_join"].term
    assert isinstance(term, CondChain)
    assert [(pred.label, target) for pred, target in term.arms] == [
        ("toPGR1", "drain1"),
        ("", "steady"),
    ]
    assert (term.default_label, term.default) == ("LoopEndL", "drain0")
    assert prog.blocks["steady"].term.label == "LoopEndL"


def test_trip_domains_include_direct_and_counted_main_loop_exits():
    """One steady trip exits directly; repeated trips take both the back edge and final exit."""
    prog = _pgr2_program()
    domains = AnalysisManager().get(TripDomains(), prog)

    assert domains.admits("prologue_join", "drain1") == {1}
    assert domains.admits("prologue_join", "drain0") == {2}
    assert 3 in domains.admits("prologue_join", "steady")
    assert 3 in domains.admits("steady", "drain0")
    assert 3 not in domains.admits("steady", "steady")
    assert 4 in domains.admits("steady", "steady")


def test_frame_contract_keeps_three_folded_main_loop_exit_frames():
    """Each scaffold exit gets a guard value; counted-loop edges accept only the steady arm."""
    prog = _pgr2_program()
    plans = plan_program(prog)
    anchors = {label: actions[0].action_id for label, actions in plans.items() if actions}
    source = anchors["prologue_join"]
    destinations = {
        anchors["drain1"]: 1,
        anchors["steady"]: 2,
        anchors["drain0"]: 3,
    }

    contract = build_contract(prog)
    assignments = {}
    for edge in contract.edges:
        if edge.relative or edge.src != source or edge.dst not in destinations:
            continue
        assignments.setdefault(edge.gen, {})[edge.dst] = edge.value
    guards = [(gen, values) for gen, values in assignments.items() if values == destinations]

    assert len(guards) == 1, assignments
    guard_gen = guards[0][0]
    generation = contract.generations[guard_gen]
    assert (generation.ring, generation.entry, generation.advance) == (4, 0, 0)

    required = {(need.dst, need.src, need.gen): need.values for need in contract.requires}
    assert required[(anchors["drain1"], anchors["prologue_join"], guard_gen)] == (0, 1)
    assert required[(anchors["steady"], anchors["prologue_join"], guard_gen)] == (0, 2)
    assert required[(anchors["drain0"], anchors["prologue_join"], guard_gen)] == (0, 3)
    assert required[(anchors["steady"], anchors["steady"], guard_gen)] == (0, 2)
    assert required[(anchors["drain0"], anchors["steady"], guard_gen)] == (0, 2)
