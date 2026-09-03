# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Phase D: the Level-0 pipeline BLOCK DIAGRAM -- the selection-flow entry point.

Pure CPU (record CRC + inspect the extracted blocks; render a PNG). Asserts the STRUCTURE the diagram
reflects -- phase split, loop membership (from the scf.for IV), operand lanes, labels -- so the map stays
correct as the recorder/kernel evolve, and that the committed render is invoked the same way every time
(``pipe.block_diagram`` == the free function).
"""
from __future__ import annotations

from rocke.helpers.tiling import tiling_recorder as tr
from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import build_crc_gemm
from rocke.helpers.tiling.visualization import block_diagram as bd

# CRC defaults (256/256/32, 4x4 waves, atom 16, double-buffered, no swizzle); K=64 -> 2 K-tiles.
CRC_ARGS = (256, 256, 64)


def _record():
    (_kernel, _mma), pipe = tr.record_build(build_crc_gemm, *CRC_ARGS)
    return pipe


def _by_seq(blocks):
    return {b.seq: b for b in blocks}


def test_extract_blocks_splits_prologue_loop_epilogue():
    """The K-loop body is exactly the IV-dependent nodes (prefetch load, cur read, MMA, oth store);
    everything before is prologue, everything after is epilogue."""
    blocks, lo, hi = bd.extract_blocks(_record())
    assert (lo, hi) == (5, 11)                       # seq 5..11 = the scf.for body
    ph = {b.seq: b.phase for b in blocks}
    assert ph[0] == "prologue" and ph[4] == "prologue"          # fill + prologue coop load/store
    assert all(ph[s] == "loop" for s in range(5, 12))           # incl. the MMA op at seq 9
    assert all(ph[s] == "epilogue" for s in range(12, 17))      # last read, MMA, reorder, C store


def test_blocks_carry_operand_lane_and_labels():
    blocks, _lo, _hi = bd.extract_blocks(_record())
    b = _by_seq(blocks)
    assert (b[1].lane, b[1].label) == ("A", "global load A")
    assert (b[2].lane, b[2].label) == ("B", "global load B")
    assert b[3].label == "LDS store A" and b[3].sublabel == "-> buf 0"      # prologue -> buffer 0
    assert b[7].label == "LDS read A" and b[7].sublabel == "<- cur buf"     # loop reads cur
    assert b[10].sublabel == "-> other buf"                                 # loop stores prefetch -> oth
    assert b[12].sublabel == "<- last buf"                                  # epilogue reads the last tile
    assert b[9].lane == "C" and b[9].label.startswith("MMA")                # the op rides the C lane
    assert b[15].label.startswith("reorder")                                # C epilogue transform
    assert b[16].label == "global store C"


def test_loop_membership_needs_the_iv():
    """A node is 'in loop' iff its origin references the scf.for IV -- the prologue store (constant
    origin) is NOT, the loop store (cur/oth off the IV) IS."""
    pipe = _record()
    a_txns = [t for t in pipe.transactions if t.space == "lds" and t.space_name.startswith("%lds_a")]
    prologue_store = next(t for t in a_txns if t.kind == "store" and t.seq < 5)
    loop_store = next(t for t in a_txns if t.kind == "store" and t.seq > 5)
    assert not bd._origin_uses_iv(prologue_store.origin)
    assert bd._origin_uses_iv(loop_store.origin)


def test_block_diagram_renders(tmp_path):
    out = tmp_path / "block_diagram.png"
    pipe = _record()
    got = pipe.block_diagram(str(out), title="crc")          # the convenience method == the committed fn
    assert got == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_convenience_method_matches_free_function(tmp_path):
    pipe = _record()
    a = pipe.block_diagram(str(tmp_path / "a.png"))
    b = bd.block_diagram(pipe, str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").stat().st_size > 0
    assert (tmp_path / "b.png").stat().st_size > 0
    assert a.endswith("a.png") and b.endswith("b.png")


def test_edges_and_lanes_are_derived_from_the_value_graph():
    """Edges are the recorded SSA producer->consumer graph (no kind heuristics); the loop lanes fall out
    of reachability to the MMA. The prefetch load feeds the STORE, NEVER the read/MMA."""
    blocks, _lo, _hi = bd.extract_blocks(_record())
    edges = set(bd._dataflow_edges(blocks))
    # every real dependency, by Value identity
    assert {(1, 3), (2, 4), (5, 10), (6, 11), (7, 9), (8, 9),
            (12, 14), (13, 14), (14, 15), (15, 16)} <= edges
    assert (5, 7) not in edges and (5, 9) not in edges     # load does NOT feed the read or the MMA
    lanes = bd._loop_lanes(blocks, list(edges))
    assert lanes[5] == lanes[6] == lanes[10] == lanes[11] == "prefetch"
    assert lanes[7] == lanes[8] == lanes[9] == "compute"


def test_accumulator_carry_is_bridged_not_value_chained():
    """The accumulator crosses the scf.for iter-arg boundary (the SSA Value is rebound), so it is NOT a
    Value edge; _acc_bridge reconstructs fill -> loop-MMA -> epilogue-MMA from the dangling reg-path
    values."""
    blocks, _lo, _hi = bd.extract_blocks(_record())
    assert bd._acc_bridge(blocks) == [(0, 9), (9, 14)]
    assert (0, 9) not in set(bd._dataflow_edges(blocks))    # bridged, not a Value edge (no double-draw)
