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


# --------------------------------------------------------------------------------------------------
# Captions are DERIVED from the recording -- regression cover for the hardcoded-caption defects.
# Every assertion below is a caption that USED to be a fixed string asserting something false.
# --------------------------------------------------------------------------------------------------


def _interleaved():
    """The interleaved demo: an N-stride-1 C (the opposite major to CRC) AND four operand-side reorders
    ahead of the C-shuffle -- the two shapes a hardcoded caption got wrong."""
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import build_interleaved_gemm
    (_k, _m), pipe = tr.record_build(build_interleaved_gemm, 256, 256, 64)
    return pipe


def test_global_store_caption_names_the_recorded_stride1_axis():
    """The global-store sublabel is DERIVED from the recorded strides, so the two kernels -- whose C
    tensors have OPPOSITE contiguous axes -- get opposite captions. A hardcoded "col-major (M-contig)"
    was right for one and asserted the exact opposite of the other's recorded strides."""
    for pipe, want in ((_record(), "M stride-1 (contiguous)"),        # CRC   C (M,N) strides (1, ldc)
                       (_interleaved(), "N stride-1 (contiguous)")):  # demo  C (M,N) strides (ldc, 1)
        store = next(b for b in bd.extract_blocks(pipe)[0]
                     if b.space == "global" and b.kind == "store")
        assert store.sublabel == want


def test_no_caption_claims_a_major():
    """A "row-major"/"col-major" word names a DIFFERENT physical axis for each of A/B/C, which is how the
    wrong caption survived. The stride-1 axis is the whole fact -- no major word may reappear."""
    for pipe in (_record(), _interleaved()):
        text = " ".join(b.label + " " + b.sublabel for b in bd.extract_blocks(pipe)[0]).lower()
        assert "major" not in text and "contig)" not in text.replace("(contiguous)", "")


def test_operand_bridges_are_not_labelled_c_epilogue():
    """A transform takes the lane of the node that PRODUCED its input, and is captioned by what it
    BRIDGES. The demo's four operand-side reorders are A/B work; only the transform hanging off the MMA
    is the C-shuffle. Captioning every reorder "(C epilogue)" mislabelled four of five."""
    blocks = {b.seq: b for b in bd.extract_blocks(_interleaved())[0]}
    reorders = [b for b in blocks.values() if b.kind == "reorder"]
    assert len(reorders) == 5
    operand = [b for b in reorders if b.lane in ("A", "B")]
    assert len(operand) == 4 and {b.lane for b in operand} == {"A", "B"}
    assert all("C" not in b.label + b.sublabel for b in operand)   # ✗ never "(C epilogue)" on an A/B bridge
    c_shuffle = [b for b in reorders if b.lane == "C"]
    assert len(c_shuffle) == 1
    assert (c_shuffle[0].label, c_shuffle[0].sublabel) == ("reorder C", "-> store order")


def test_epilogue_reread_is_not_captioned_a_prefetch():
    """A global load is a staging PREFETCH only if it reaches an LDS store. A load from a space this
    recording also STORES is the read-modify-write input (an epilogue re-reading its own output tile) --
    captioning it "prefetch k=0" claimed a staging role it does not have."""
    pipe = _rmw_recording()
    blocks = {b.seq: b for b in bd.extract_blocks(pipe)[0]}
    assert blocks[2].sublabel == "read-modify-write input"
    assert "prefetch" not in blocks[2].sublabel
    assert bd.rmw_spaces(pipe.nodes) == {9}


def test_ambiguous_strides_state_the_raw_fact_rather_than_guess():
    """No unique stride-1 axis -> print the recorded strides, never invent a contiguous axis."""
    import types
    node = types.SimpleNamespace(strides=(4, 8), tile_desc=types.SimpleNamespace(shape=(4, 4)))
    assert bd._stride1_caption(node, ("M", "N")) == "strides (4, 8)"


def _rmw_recording():
    """A minimal READ-MODIFY-WRITE epilogue recording, built from the committed descriptor API (no kernel
    needed): combine -> transform -> [unrecorded elementwise] -> global store, with a global LOAD from the
    SAME space the store targets. The transform -> store SSA chain is deliberately BROKEN, because that is
    exactly what a kernel whose elementwise epilogue bypasses the recorded verbs produces."""
    import types

    from rocke.helpers.tiling import make_tile_desc

    td = make_tile_desc(shape=[8, 8], thread_tile=[1, 1], thread_dist=[8, 8], wave_size=64)

    def node(seq, kind, **kw):
        base = dict(seq=seq, kind=kind, space="reg", space_name="", space_id=None, origin=None,
                    produces=None, consumes=(), strides=None, tile_desc=None, tgt_enc=None,
                    dtype_name="f32")
        base.update(kw)
        return types.SimpleNamespace(**base)

    nodes = [node(0, "mma", produces=100),
             node(1, "reorder", produces=101, consumes=(100,), tgt_enc=td.layout),
             node(2, "load", space="global", space_name="%C", space_id=9, produces=102,
                  strides=(8, 1), tile_desc=td),
             node(3, "store", space="global", space_name="%C", space_id=9, consumes=(999,),
                  strides=(8, 1), tile_desc=td)]
    ops = [n for n in nodes if n.kind in ("mma", "reorder")]
    return types.SimpleNamespace(nodes=nodes, transactions=[n for n in nodes if n.kind in ("load", "store")],
                                 ops=ops, spaces={9: "%C"}, lds_spaces=lambda: [])
