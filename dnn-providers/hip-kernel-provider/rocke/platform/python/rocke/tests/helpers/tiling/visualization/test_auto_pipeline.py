# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CRC MVP: the Phase-C driver -- symbolic-origin resolver + the addressing round-trip (gate 1).

Pure CPU (build IR + inspect). Target: build_crc_gemm at a small real config (double-buffered,
cooperative, swizzle-parameterized) -- the symbolic origins / cooperative scope / swizzle the toy
first-light deliberately avoided.
"""
from __future__ import annotations

import dataclasses

import pytest

from rocke.helpers.tiling import tiling_recorder as tr
from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import (
    b32_swizzle,
    b64_swizzle,
    build_crc_gemm,
)
from rocke.helpers.tiling.visualization import auto_pipeline as ap

CFG = dict(tile_m=64, tile_n=64, tile_k=32, waves_m=2, waves_n=2)  # 4 waves, warp 32x32, tile_k 32


def _record(sw=False):
    (kernel, mma), pipe = tr.record_build(build_crc_gemm, 64, 64, 64, lds_swizzle=sw, **CFG)
    return kernel, mma, pipe


def _first_read(pipe, space_prefix="%lds_a"):
    return next(t for t in pipe.transactions
                if t.space_name.startswith(space_prefix) and t.kind == "load")


def test_resolver_matches_the_read_origin_formula():
    """The read origin is cur*tile_m + wm*warp_m: cur=(k//tile_k)%2, wm=(tid//64)//waves_n."""
    _k, _m, pipe = _record()
    origin = _first_read(pipe).origin
    assert ap.resolve_origin(origin, {"k": 0, "tid": 0})[1] == 0     # cur 0, wave 0
    assert ap.resolve_origin(origin, {"k": 0, "tid": 128})[1] == 32  # cur 0, wave 2 -> wm 1
    assert ap.resolve_origin(origin, {"k": 32, "tid": 0})[1] == 64   # cur 1 -> half 1
    assert ap.resolve_origin(origin, {"k": 32, "tid": 192})[1] == 96  # cur 1, wm 1


def test_resolver_passes_ints_and_flags_unknown():
    assert ap.resolve_value(7, {}) == 7
    with pytest.raises(ap.OriginResolutionError):
        ap.resolve_value(_first_read(_record()[2]).origin[1], {})  # scf.for IV with no 'k' binding


@pytest.mark.parametrize("sw,label", [(False, "none"), (b32_swizzle, "b32"), (b64_swizzle, "b64")])
def test_crc_lds_roundtrip_passes_through_swizzle(sw, label):
    """Addressing round-trip closes for BOTH double-buffered halves of lds_a and lds_b, through the
    b32/b64 swizzle -- keyed by (smem identity + resolved buffer-half)."""
    _k, _m, pipe = _record(sw)
    spaces = pipe.lds_spaces()
    assert len(spaces) == 2  # lds_a, lds_b
    for sid in spaces:
        assert ap.verify_lds_roundtrip(pipe, sid, tile_k=32) == [0, 1], f"{label} {pipe.spaces[sid]}"


def test_roundtrip_catches_a_misplaced_read():
    """A read whose resolved origin lands in the wrong buffer half is caught (the cur/oth confusion)."""
    _k, _m, pipe = _record()
    sid = pipe.lds_spaces()[0]
    read = _first_read(pipe)
    bad = dataclasses.replace(read, origin=(0, 999))  # free 999 -> no such buffer half
    pipe.nodes[pipe.nodes.index(read)] = bad
    with pytest.raises(ap.RoundTripError):
        ap.verify_lds_roundtrip(pipe, sid, tile_k=32)


def test_crc_mma_soundness_passes():
    """Gate 2: CRC's own interleaved operands are sound against the canonical machine ref
    (mma.a_layout) and K-aligned -- the operand-correctness the addressing round-trip can't see."""
    _k, _m, pipe = _record()
    assert ap.verify_mma_soundness(pipe) == 2  # two K-tiles -> two recorded MMA ops


def test_mma_soundness_catches_unsound_operand():
    """Feeding the C accumulator layout as the A operand is not a sound (M,K) operand -> caught."""
    _k, _m, pipe = _record()
    mma_op = next(o for o in pipe.ops if o.kind == "mma")
    bad = dataclasses.replace(mma_op, a_enc=mma_op.c_enc)  # C (M,N) is not a valid A (M,K) operand
    pipe.nodes[pipe.nodes.index(mma_op)] = bad
    with pytest.raises(ap.MmaSoundnessError):
        ap.verify_mma_soundness(pipe)


def test_roundtrip_catches_swizzle_mismatch():
    """Record WITH the b64 swizzle, then claim the read is un-swizzled: its addresses no longer match
    the swizzled store -> the COVERAGE gate fires (not just the half guard)."""
    _k, _m, pipe = _record(b64_swizzle)
    sid = pipe.lds_spaces()[0]
    read = _first_read(pipe)
    bad = dataclasses.replace(read, swizzle=False)  # read now disagrees with the swizzled store
    pipe.nodes[pipe.nodes.index(read)] = bad
    with pytest.raises(ap.RoundTripError, match="never written"):
        ap.verify_lds_roundtrip(pipe, sid, tile_k=32)


def test_view_prefetch_flow_and_lds_store_block(tmp_path):
    """view() takes only SELECTION (block XOR flow + scope/operand/buffer/wave); it DERIVES every physical
    fact from the recording. The PREFETCH flow (global load -> LDS store) renders at macro + wave scope; the
    single LDS-store block is wave-scoped; the dtype is derived + reflected in the filename."""
    import pathlib

    _k, _m, pipe = _record()
    assert pipe.arch == "gfx90a" and pipe.wave_size == 64          # captured from the TileMma, not assumed
    store = next(t for t in pipe.transactions if t.space == "lds" and t.kind == "store")
    assert ap.is_cooperative(store.tile_desc.layout, pipe.wave_size)        # coop store spans > 1 wave
    assert not ap.is_cooperative(_first_read(pipe).tile_desc.layout, pipe.wave_size)  # MMA read = 1 wave

    macro = ap.view(pipe, flow="prefetch", scope="macro", operand="B", out_path=str(tmp_path / "pf.png"))
    assert macro.endswith("pf.png") and pathlib.Path(macro).exists()       # filename == out_path (dtype lives in folder/title, not the name)
    wave0 = ap.view(pipe, flow="prefetch", scope="wave", operand="A", wave=0,
                    out_path=str(tmp_path / "pfw.png"))
    blk = ap.view(pipe, block="lds_store", operand="A", wave=0, out_path=str(tmp_path / "blk.png"))
    assert all(pathlib.Path(p).stat().st_size > 0 for p in (macro, wave0, blk))
    with pytest.raises(ValueError, match="EXACTLY ONE"):           # must pick block XOR flow
        ap.view(pipe, block="lds_store", flow="prefetch", out_path=str(tmp_path / "x.png"))


def test_view_compute_flow_builds_tee_from_the_recording(tmp_path):
    """flow='compute' builds the MMA tee from the RECORDED op alone (a_enc/b_enc/c_enc + canonical refs +
    derived dtypes -- no TileMma object) and renders A x B -> derived C; filename == the given out_path."""
    import pathlib

    _k, _m, pipe = _record()
    got = ap.view(pipe, flow="compute", out_path=str(tmp_path / "compute.png"))
    assert got.endswith("compute.png") and pathlib.Path(got).exists() and pathlib.Path(got).stat().st_size > 0
    # PHYSICAL LDS view: the recorded (round-trip-validated) read desc -> the MMA-operand registers
    phys = ap.view(pipe, flow="compute", lds_view="physical", operand="A", wave=0,
                   out_path=str(tmp_path / "phys.png"))
    assert phys.endswith("phys.png") and pathlib.Path(phys).exists() and pathlib.Path(phys).stat().st_size > 0


def test_view_epilogue_flow_classifies_and_renders(tmp_path):
    """flow='epilogue' (small, once at the end): C native -> {reorder} -> global store C, built recording-
    only (c_native from the tee, C-store dist from the recorded reorder op), f32 in the name."""
    import pathlib

    _k, _m, pipe = _record()
    got = ap.view(pipe, flow="epilogue", out_path=str(tmp_path / "epi.png"))
    assert got.endswith("epi.png") and pathlib.Path(got).exists() and pathlib.Path(got).stat().st_size > 0


@pytest.mark.parametrize("operand,free", [("A", "M"), ("B", "N")])
def test_lds_store_is_a_reposition_with_invariant_labels(tmp_path, operand, free):
    """The store is a REPOSITION into LDS (register slot -> LDS address), the datum's logical (free,K) LABEL
    invariant across the register file and the LDS view. On CRC there is NO transpose -- the free-contiguous
    coop load lands on the free-stride-1 banks (symmetric for A and B), so the store carries the operand's own
    (M,K)/(N,K) dims and states no axis swap. render_lds_store asserts edge_kind==reposition and
    reg.fwd_map==lds.flow_map; rendering the block exercises those."""
    import pathlib
    from rocke.helpers.tiling.transforms import describe_edge

    _k, _m, pipe = _record()
    dims = (free, "K")
    kind, why = describe_edge({(0, 0): (0, 0)}, None, src_dims=dims, to_space="lds")
    assert kind == "reposition" and "invariant" in why and "transpose" not in why

    got = ap.view(pipe, block="lds_store", operand=operand, wave=0, out_path=str(tmp_path / f"store_{operand}.png"))
    assert pathlib.Path(got).exists() and pathlib.Path(got).stat().st_size > 0  # internal reposition asserts held


def test_pipeline_label_gate_blocks_position_derived_labels():
    """The render-time label gate makes it IMPOSSIBLE to ship a label derived from a POSITION: a downstream
    stage whose label universe differs from upstream (a rectangular (K,M) transpose) raises -- unless the edge
    is an EXPLICIT relabel. Labels flow invariant; only FlowStage(relabel=True) may change them."""
    from rocke.helpers.tiling.visualization.layout_render import (
        FlowStage, LdsBankView, LabelMutationError, Pipeline, RegisterFileComponent)

    # rectangular tile: M in 0..3, K in 0..1 -> the (M,K) set differs from the transposed (K,M) set
    reg_fwd = {(m, k): (m, k) for m in range(4) for k in range(2)}
    reg = RegisterFileComponent(fwd_map=reg_fwd, dims=("M", "K"), dtype_bits=16)
    transposed = {(m, k): (k, m) for m in range(4) for k in range(2)}   # label DERIVED FROM POSITION (the bug)
    lds_bad = LdsBankView(mp=reg_fwd, addr_fn=lambda r, c: r * 2 + c, flow_map=transposed,
                          nbanks=32, elem_bytes=2, dims=("M", "K"))
    with pytest.raises(LabelMutationError, match="DERIVED FROM A POSITION"):
        Pipeline(stages=(FlowStage("regs", reg, source="reg_fwd"),
                         FlowStage("LDS", lds_bad, source="transposed"))).check_label_invariance()
    # an EXPLICIT relabel is the one sanctioned escape
    assert Pipeline(stages=(FlowStage("regs", reg, source="reg_fwd"),
                            FlowStage("LDS", lds_bad, source="transposed", relabel=True))).check_label_invariance()
    # carrying the INVARIANT label passes
    lds_ok = LdsBankView(mp=reg_fwd, addr_fn=lambda r, c: r * 2 + c, flow_map=reg_fwd,
                         nbanks=32, elem_bytes=2, dims=("M", "K"))
    assert Pipeline(stages=(FlowStage("regs", reg, source="reg_fwd"),
                            FlowStage("LDS", lds_ok, source="reg_fwd"))).check_label_invariance()


def test_ab_swap_C_is_derived_not_hand_relabeled(tmp_path):
    """AB-swap is the EXPLICIT input-relabel exemplar (A<->B, M<->N). C is NOT relabeled by hand -- it falls
    out of the fixed machine via derive_c_distribution. Recording the crossed build and building the tee from
    the recorded operands reproduces exactly that derived C."""
    import pathlib
    from rocke.helpers.tiling.transforms import derive_c_distribution
    from rocke.helpers.tiling.visualization.layout_render import MmaTee

    (_kernel, _mma), pipe = tr.record_build(build_crc_gemm, 64, 64, 64, lds_swizzle=False, ab_swap=True, **CFG)
    mma_op = next(o for o in pipe.ops if o.kind == "mma")
    tee = MmaTee(a_enc=mma_op.a_enc, b_enc=mma_op.b_enc, c_enc=mma_op.c_enc, atom_shape=mma_op.atom_shape,
                 a_canon=mma_op.a_canon, b_canon=mma_op.b_canon, c_canon=mma_op.c_canon,
                 a_dtype_bits=16, b_dtype_bits=16, c_dtype_bits=32,
                 dims_a=("M", "K"), dims_b=("N", "K"), dims_c=("M", "N"))
    derived = derive_c_distribution(mma_op.a_enc, mma_op.b_enc, a_canon=mma_op.a_canon,
                                    b_canon=mma_op.b_canon, c_canon=mma_op.c_canon)
    assert tee.c_mapping() == derived   # C is the machine fall-out of the swapped inputs, not hand-set
    got = ap.view(pipe, flow="compute", out_path=str(tmp_path / "abswap.png"))
    assert pathlib.Path(got).exists() and pathlib.Path(got).stat().st_size > 0


# --------------------------------------------------------------------------------------------------
# Generic flow segmentation + sweep enumeration (any kernel; each transition drawn once)
# --------------------------------------------------------------------------------------------------


def test_segment_flows_crc_each_transition_once():
    """The recorded CRC pipeline segments into prefetch(A,B) + lds_read(A,B) + compute + epilogue -- and the
    seqs are a DISJOINT COVER of every recorded node (each transition in exactly one flow)."""
    _k, _m, pipe = _record()
    flows = ap.segment_flows(pipe)
    roles = sorted({f.role for f in flows})
    assert roles == ["compute", "epilogue", "lds_read", "prefetch"]   # no "copy"/"standalone" for a GEMM
    assert {(f.role, f.lane) for f in flows} >= {("prefetch", "A"), ("prefetch", "B"),
                                                 ("lds_read", "A"), ("lds_read", "B"),
                                                 ("compute", "C"), ("epilogue", "C")}
    covered = sorted(s for f in flows for s in f.seqs)
    assert covered == sorted(n.seq for n in pipe.nodes)              # disjoint, full cover
    assert len(covered) == len(set(covered))


def test_plan_flows_crc_no_standalone_lds_store():
    """The prefetch flow SUBSUMES its LDS-store leg, so the sweep never emits a standalone lds_store L1
    diagram. L2 (coalescing, bank-conflict) are REDIRECTS (flow is None), not rendered."""
    _k, _m, pipe = _record()
    specs = ap.plan_flows(pipe)
    l1 = [s.name for s in specs if s.level == 1]
    assert "prefetch_A" in l1 and "prefetch_B" in l1 and "compute" in l1 and "epilogue" in l1
    assert not any("lds_store" in n for n in l1)                     # covered by prefetch, drawn once
    assert all(s.flow is None and s.redirect is not None for s in specs if s.level == 2)
    assert any(s.redirect[0] == "/bank-conflict" for s in specs if s.level == 2)


def _copy_pipeline():
    """A minimal NON-COMBINING recording (a stage-through copy: global load -> LDS store -> LDS read ->
    global store, NO mma) -- the genericity fixture: the segmenter must invent no compute flow."""
    import types

    def node(seq, kind, space, produces, consumes, sp_name="%x"):
        return types.SimpleNamespace(seq=seq, kind=kind, space=space, produces=produces,
                                     consumes=tuple(consumes), origin=None, space_name=sp_name)
    nodes = [node(0, "load", "global", 100, (), "%src"),
             node(1, "store", "lds", None, (100,), "%lds_c"),
             node(2, "load", "lds", 200, (), "%lds_c"),
             node(3, "store", "global", None, (200,), "%dst")]
    return types.SimpleNamespace(nodes=nodes, transactions=nodes, ops=[],
                                 spaces={1: "%lds_c"}, lds_spaces=lambda: [1])


def test_segment_flows_generic_no_compute_invented():
    """GENERICITY: a kernel with no combining op yields copy flows and NO compute flow -- and still covers
    every transition exactly once. Proves detection is not GEMM-bound."""
    pipe = _copy_pipeline()
    flows = ap.segment_flows(pipe)
    assert all(f.role != "compute" for f in flows)                  # nothing invents an MMA/compute stage
    covered = sorted(s for f in flows for s in f.seqs)
    assert covered == [0, 1, 2, 3] and len(covered) == len(set(covered))
    specs = ap.plan_flows(pipe)
    assert not any(s.level == 1 and s.name == "compute" for s in specs)
    assert not any(s.level == 0 and s.name == "localization" for s in specs)  # needs compute+coop; absent


def test_lds_read_flow_gains_a_no_box_reorder_panel():
    """The interleaved A-read coalesces M-inner then v_perms into the K-packed MMA order: the flow must
    draw that reorder as its OWN reg->reg stage (no distribution box) with a DERIVED interleave_idx arrow,
    and the finally-requested panel carries the one box. All generic (reorder_between), no CRC constants."""
    from rocke.helpers.tiling.visualization.kernel_stages import flow_lds_load_placement
    _k, _m, pipe = _record()
    sid = next(s for s in pipe.lds_spaces() if ap._operand_of(pipe.spaces[s]) == "A")
    read = next(t for t in pipe.transactions if t.space_id == sid and t.kind == "load")
    store = next(t for t in pipe.transactions if t.space_id == sid and t.kind == "store")
    mma = next(o for o in pipe.ops if o.kind == "mma")
    flow = flow_lds_load_placement(
        read_desc=read.tile_desc, flow_desc=mma.a_enc, dims=("M", "K"), nbanks=32, elem_bytes=2,
        n_waves=1, wave_size=64, cooperative=False, wave=0, stride=int(read.strides[0]),
        swizzle=read.swizzle or None, store_desc=store.tile_desc, store_stride=int(store.strides[0]),
        store_swizzle=store.swizzle or None)
    flow.check_label_invariance()                              # a reorder keeps the label SET -> passes
    reorder_stages = [s for s in flow.stages if s.reorder]
    assert len(reorder_stages) == 1                            # the coalesced landing, marked no-box
    assert reorder_stages[0].box_lines() == []                # reorder intermediary carries NO box
    final = flow.stages[-1]
    assert "interleave_idx(" in final.transform               # DERIVED reorder arrow, not a bare string
    assert final.box_lines() and final.box_lines()[0].startswith("src:")   # the ONE box, on the requested panel
    assert final.info and "v_perm" in final.info[0]           # the cost is stated on the panel


# --------------------------------------------------------------------------------------------------
# Selection + routing are DERIVED from the recorded graph, never from POSITION or node kind.
# --------------------------------------------------------------------------------------------------


def _interleaved():
    """The interleaved demo: four operand-side reorders recorded BEFORE the C-shuffle, so 'the first
    reorder' and 'the C-shuffle' are different ops -- the shape that made a position-based pick wrong."""
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import build_interleaved_gemm
    (_k, _m), pipe = tr.record_build(build_interleaved_gemm, 256, 256, 64)
    return pipe


def test_select_c_shuffle_is_not_the_first_recorded_reorder():
    """Taking the FIRST reorder picks an operand-side bridge, then captions the epilogue panel from it.
    Selection is by reachability: the transform hanging off the combining op's OUTPUT."""
    pipe = _interleaved()
    reorders = [o for o in pipe.ops if o.kind in ("reorder", "cross_lane")]
    assert len(reorders) == 5
    chosen = ap.select_c_shuffle(pipe)
    assert chosen.seq != reorders[0].seq            # ✗ position
    assert chosen.seq == reorders[-1].seq           # the one produced by the MMA
    # CRC has a single reorder, so the two agree there -- the old code passed by luck, not by rule.
    assert ap.select_c_shuffle(_record()[2]).seq == [o.seq for o in _record()[2].ops
                                                     if o.kind == "reorder"][0]


def test_select_c_shuffle_survives_a_broken_store_chain():
    """When the epilogue's elementwise combine is not a recorded verb the SSA chain reorder -> store is
    ABSENT, so forward reach finds nothing. BACKWARD reach (produced by a combining op) still identifies
    it -- that recorded mma -> reorder edge is the definition of an epilogue transform."""
    from rocke.tests.helpers.tiling.visualization.test_block_diagram import _rmw_recording
    pipe = _rmw_recording()
    assert ap.select_c_shuffle(pipe).seq == 1


def test_select_c_shuffle_refuses_rather_than_guesses():
    """No candidate at all -> raise, naming what was searched. Never fall through to a panel."""
    import types
    empty = types.SimpleNamespace(nodes=[], transactions=[], ops=[], spaces={}, lds_spaces=lambda: [])
    with pytest.raises(ap.EpilogueSelectionError, match="cannot identify the C-shuffle"):
        ap.select_c_shuffle(empty)


def test_epilogue_refuses_an_unknown_classification(monkeypatch, tmp_path):
    """The `/layout-viz` rule: an `unknown` classification must STOP AND ASK. It must NEVER collapse into
    the DIRECT branch, whose caption CLAIMS 'native == store order'."""
    from rocke.helpers.tiling.visualization import kernel_stages as ks
    monkeypatch.setattr(ap, "classify_epilogue", None, raising=False)
    monkeypatch.setattr(ks, "classify_epilogue", lambda *_a, **_k: ("unknown", "forced"))
    with pytest.raises(ap.EpilogueSelectionError, match="unknown"):
        ap.view(_record()[2], flow="epilogue", out_path=str(tmp_path / "e.png"))


def test_operand_bridge_joins_its_operand_flow_not_the_epilogue():
    """A transform INHERITS THE ROLE OF ITS PRODUCER -- it is the price of the hop that produced its
    input. The demo's global-load-side bridges join `prefetch`; only the MMA-fed one is `epilogue`.
    Routing every transform to `epilogue` by node kind swallowed four of five."""
    flows = {(f.role, f.lane): f for f in ap.segment_flows(_interleaved())}
    assert set(flows[("prefetch", "A")].seqs) >= {3}      # the A coop-store bridge, with its own operand
    assert set(flows[("prefetch", "B")].seqs) >= {4}
    epi = flows[("epilogue", "C")]
    assert len(epi.seqs) == 2                             # the C-shuffle + the global store, nothing else


def test_rmw_read_is_folded_into_the_epilogue_not_left_an_orphan_copy():
    """A global load from a space this recording also STORES is the epilogue's read-modify-write input.
    Left as a `copy` it segmented into a lone orphan flow with its own panel."""
    from rocke.tests.helpers.tiling.visualization.test_block_diagram import _rmw_recording
    flows = ap.segment_flows(_rmw_recording())
    assert all(f.role != "copy" for f in flows)
    assert set(next(f for f in flows if f.role == "epilogue").seqs) == {1, 2, 3}


# --------------------------------------------------------------------------------------------------
# Cooperative partition: WHICH axis the waves split is DERIVED, never assumed to be the free dim.
# --------------------------------------------------------------------------------------------------


def _coop(split, tile_free=256, tile_k=32, n_waves=8, vw=8):
    """A cooperative (free, K) macro-load descriptor whose ``n_waves`` waves split ``split`` ("K" or the
    free dim). Each lane still takes ``vw`` contiguous free elements; only which axis carries `wave_dist`
    differs, so the two cases are otherwise identical."""
    from rocke.helpers.tiling import make_tile_desc
    free_lanes = tile_free // vw if split == "K" else tile_free // (vw * n_waves)
    k_lanes = 64 // free_lanes
    wave_dist = [1, n_waves] if split == "K" else [n_waves, 1]
    return make_tile_desc(
        shape=[tile_free, tile_k], thread_tile=[vw, 1], thread_dist=[free_lanes, k_lanes],
        thread_order=[1, 0], block_repeat=[1, tile_k // (k_lanes * (n_waves if split == "K" else 1))],
        wave_dist=wave_dist, wave_size=64)


def test_coop_partition_derives_the_split_axis_both_ways():
    """`wave_dist=[1, n_waves]` splits K with every wave covering the FULL free extent; `[n_waves, 1]`
    splits the free dim. The old rule -- 'the free axis is the one with the smaller per-wave span' --
    silently assumed the second, so the first came out with its two axes swapped."""
    k_split = ap.coop_partition(_coop("K"), n_waves=8, wave_size=64, axis_names=("M", "K"))
    assert k_split["split"] == ["K"] and k_split["covered"] == ["M"]
    assert k_split["axes"]["M"]["extent"] == 256 and k_split["axes"]["K"]["extent"] == 32
    free_split = ap.coop_partition(_coop("free"), n_waves=8, wave_size=64, axis_names=("M", "K"))
    assert free_split["split"] == ["M"] and free_split["covered"] == ["K"]


def test_wave_footprint_is_the_real_coordinate_set():
    """Wave w of a K-splitting fetch touches a NON-CONTIGUOUS K set and the whole free extent."""
    part = ap.coop_partition(_coop("K"), n_waves=8, wave_size=64, axis_names=("M", "K"))
    fp = ap.wave_footprint(part, 0)
    assert fp["K"] == (0, 1, 16, 17)                       # ✗ not the range 0..17
    assert fp["M"] == tuple(range(256))
    note = ap._coop_note(part, 0)
    assert note == "K{0,1,16,17} x full M[0:256]"
    assert "[0:18]" not in note                            # the extent that never existed


def test_fmt_footprint_never_ranges_a_noncontiguous_set():
    """A range over a gapped set invents coordinates the wave never touches."""
    assert ap._fmt_footprint("K", (0, 1, 2, 3)) == "K[0:4]"          # contiguous -> a range
    assert ap._fmt_footprint("K", (0, 1, 16, 17)) == "K{0,1,16,17}"  # gapped -> the set
    assert ap._fmt_footprint("M", tuple(range(12))) == "M[0:12]"
