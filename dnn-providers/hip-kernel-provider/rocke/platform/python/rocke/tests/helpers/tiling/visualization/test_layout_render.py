# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Smoke + invariant tests for the layout-viz renderer and its `_canvas` primitive layer.

These lock the refactor that split the matplotlib-facing primitives into `_canvas.py`: every public
view must still build and render headless without error, the primitive layer must keep its conventions
(first-8 shade gating, edge/register ruler anchoring), and the LDS-conflict tooling's model self-test
must stay green. No GPU required.
"""
from __future__ import annotations

import os

from rocke.helpers.tiling.mma.mma_operation import TileMma
from rocke.helpers.tiling.visualization import _canvas as cv
from rocke.helpers.tiling.visualization import layout_render as lr
from rocke.helpers.tiling import lds_conflict as lc


def _mma():
    return TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")


# ---------------------------------------------------------------- primitive-layer invariants


def test_colour_model_reexported_and_identical():
    # layout_render re-exports the colour model that now lives in _canvas (back-compat for consumers).
    assert lr.ACCENTS is cv.ACCENTS and lr.NACC == cv.NACC == 8
    assert lr.cell_rgb(0, 0, 1) == cv.cell_rgb(0, 0, 1)
def test_shade_first8_gating():
    # first8: lanes 0..7 coloured, the rest neutral grey; full: every lane%8 coloured.
    assert cv.shade(0, 0, 1, color_mode="first8") == cv.cell_rgb(0, 0, 1)
    assert cv.shade(8, 0, 1, color_mode="first8") == cv.GREY
    assert cv.shade(8, 0, 1, color_mode="full") == cv.cell_rgb(8, 0, 1)
def test_reg_ticks_pack_and_edge_ticks_anchor():
    # f16 packs 2 elements/register -> one tick per 2-cell run; ascending anchors on the low edge.
    assert cv.reg_ticks([0, 1, 2, 3], 16) == [(0, "0"), (2, "1")]
    assert cv.reg_ticks([0, 1], 32) == [(0, "0"), (1, "1")]
    # descending edge-ticks anchor on the high edge.
    assert cv.edge_ticks([0, 1, 2]) == [(0, "0"), (1, "1"), (2, "2")]
    assert cv.edge_ticks([2, 1, 0]) == [(1, "2"), (2, "1"), (3, "0")]


# ---------------------------------------------------------------- view render smoke
def test_render_views_writes_all_panels(tmp_path):
    paths = lr.render_views(_mma().a_layout, axes=("M", "K"),
                            views=("logical", "macro", "register", "lds"),
                            nbanks=32, dtype_bits=16, out_dir=str(tmp_path), name="v", combined=True)
    assert paths and all(os.path.getsize(p) > 0 for p in paths)
def test_register_and_logical_components_render(tmp_path):
    mma = _mma()
    rp = lr.RegisterFileComponent(dist=mma.a_layout, dims=("M", "K"), dtype_bits=16).render(
        str(tmp_path / "rf.png"))
    lp = lr.LogicalTileComponent(dist=mma.c_layout, dims=("M", "N"), row_coord=0).render(
        str(tmp_path / "lt.png"))
    assert os.path.getsize(rp) > 0 and os.path.getsize(lp) > 0
def test_mma_tee_renders(tmp_path):
    p = lr.MmaTee.from_mma(_mma()).render(str(tmp_path), name="tee")
    assert os.path.getsize(p) > 0
def test_logical_view_modes(tmp_path):
    mma = _mma()
    base = dict(dist=mma.a_layout, dims=("M", "K"), row_coord=0, label_coords="logical")
    addr = lambda m, k: m * 16 + k
    a = lr.LogicalTileComponent(**base, mode="layout", atom=(8, 8))
    # a memory-transaction shade needs the memory order (addr_fn) AND dtype_bits (the b128 cap) -- no assumption.
    b = lr.LogicalTileComponent(**base, mode="thread_tile", addr_fn=addr, dtype_bits=16)
    # layout defaults preserve current behaviour (no mode) -> single default group, no auto shade.
    plain = lr.LogicalTileComponent(**base)
    assert plain.groups == () and plain.shade_map is None
    # thread_tile auto-derives one group per owning lane + a vectorization shade map.
    assert len(b.groups) == 64 and b.shade_map is not None
    for i, comp in enumerate((a, b)):
        p = comp.render(str(tmp_path / f"mode_{i}.png"))
        assert os.path.getsize(p) > 0
def test_render_coalescing_address_space(tmp_path):
    # the ADDRESS-SPACE coalescing view is a PURE consumer of an analysis CoalescingReport (fused + scattered);
    # both a single panel and the two-panel compare must render headless.
    from rocke.helpers.tiling.analysis import analyze_coalescing
    n = analyze_coalescing({(L, r): (0, 4 * L + r) for L in range(16) for r in range(4)},
                           ("M", "N"), (64, 1), 32, direction="store", line_bytes=128)   # fused
    m = analyze_coalescing({(L, r): (4 * L + r, L) for L in range(16) for r in range(4)},
                           ("M", "N"), (1, 256), 32, direction="store", line_bytes=128)  # scattered
    assert n.fully_coalesced and not m.fully_coalesced
    p1 = lr.render_coalescing(m, str(tmp_path / "coal_single.png"), lane_group=16)
    p2 = lr.render_coalescing_compare([n, m], str(tmp_path / "coal_compare.png"),
                                      titles=["N", "M"], lane_group=16)
    assert os.path.getsize(p1) > 0 and os.path.getsize(p2) > 0
def test_workflow_recipes_render(tmp_path):
    from rocke.helpers.tiling.register_mapper import RegisterMapper
    mma = _mma()
    dist = mma.a_layout
    rm = RegisterMapper(dist)
    mp = {(l, r): rm.matrix_coordinates(l, r)
          for l in range(rm.num_lanes) for r in range(rm.num_vector_items)}
    addr = lambda m, k: m * 16 + k
    w1 = lr.flow_mem_to_register(dist, dims=("M", "K"), dtype_bits=16)
    w1.trace((3, 5))
    w2 = lr.flow_lds_to_register(mp, addr, read_dist=dist, dims=("M", "K"), nbanks=32, elem_bytes=4)
    w2.trace((3, 5))
    w3 = lr.flow_wave_mma(mma)                            # == MmaTee
    p1 = w1.render(str(tmp_path / "w1.png"))
    p2 = w2.render(str(tmp_path / "w2.png"))
    p3 = w3.render(str(tmp_path), name="w3")
    assert all(os.path.getsize(p) > 0 for p in (p1, p2, p3))
def test_kloop_operand_strip_traces_end_to_end(tmp_path):
    from rocke.helpers.tiling.register_mapper import RegisterMapper
    mma = _mma()
    dist = mma.a_layout
    rm = RegisterMapper(dist)
    mp = {(l, r): rm.matrix_coordinates(l, r)
          for l in range(rm.num_lanes) for r in range(rm.num_vector_items)}
    addr = lambda m, k: m * 16 + k
    pipe = lr.flow_kloop_operand(load_dist=dist, store_mp=mp, store_addr=addr, read_dist=dist,
                                 dims=("M", "K"), name="A", nbanks=32, elem_bytes=4)
    assert len(pipe.stages) == 4
    # the original datum resolves in every staging step (global, load-regs, LDS, MMA-operand regs)
    for s in pipe.stages:
        assert s.cells_for((3, 5)), f"M3K5 lost at stage {s.name}"
    pipe.trace((3, 5))
    p = pipe.render(str(tmp_path / "kloop_a.png"))
    assert os.path.getsize(p) > 0
def test_classify_epilogue():
    from rocke.helpers.tiling.visualization import kernel_stages as ks
    mma = _mma()
    c_native = mma.c_layout
    # native == store -> direct; None -> unknown (caller must ask).
    assert ks.classify_epilogue(c_native, c_native)[0] == "direct"
    assert ks.classify_epilogue(c_native, None)[0] == "unknown"
def test_lds_conflict_served_model_general():
    # General (NOT config-specific) guards for the write-port served model in lds_conflict.
    a = lc.GFX90A
    # 1) COMBINE cap: a NARROW-DEEP pile (banks_used < COMBINE) cannot combine across idle banks, so it
    #    drains at banks*depth/min(banks,COMBINE) -- the old uncapped /COMBINE under-counted it.
    assert lc.served_phase(2, 16, a) == 2 * 16 / min(2, a.COMBINE)          # capped -> 16 (was 8)
    assert lc.served_phase(1, 8, a) == 1 * 8 / 1                            # single bank -> depth
    # 2) WIDE stores (banks_used >= COMBINE) are UNCHANGED by the cap (so the corpus is untouched).
    for banks, depth in ((4, 8), (8, 8), (16, 2), (8, 4)):
        assert lc.served_phase(banks, depth, a) == min(banks, a.PORT_BANKS) * depth / a.COMBINE
    # 3) End-to-end: a synthetic narrow-deep pile (2 banks x depth 16, 4 phases, footprint 256) -> cpa 3.0.
    r = lc.simulate_hist(lc._uniform(2, 16, 2, 4), 256, a)
    assert r["IDX"] == 32 and r["BC"] == 24 and r["BC"] / r["productive"] == 3.0
    # 4) Stripe rule is geometry-general: the conflict-free unit is set by the pad0 alias DEPTH read off the
    #    address map (unit = NB*W/depth), no per-config constant. Default (depth NB/4) == legacy 4*W.
    assert lc.predict_pad_sweep(8, "b128", a) == lc.predict_pad_sweep(8, "b128", a, pad0_depth=a.NB // 4)
    assert lc.recommend_pad(256, "b128", a) == 32                           # depth 8 (default) -> +32
    assert lc.recommend_pad(256, "b128", a, pad0_depth=16) == 16            # depth 16 (deeper alias) -> +16
def test_gate_is_scale_invariant():
    import pytest
    # The gate must reconcile PER-SERVED-GROUP sim vs WHOLE-RUN measured counters -- the invariant is
    # conflicts/access (a ratio), never the absolute BC/IDX (different scales).
    sim = {"BC": 24, "IDX": 32, "conflicts_per_access": 3.0}          # per-served-group
    hw = {"BC": 811008, "IDX": 1081344, "conflicts_per_access": 3.0}  # whole-run, SAME ratio
    assert lc.gate(sim, hw)                                           # passes: cpa matches across scale
    # a real model error IS caught (ratio disagrees)
    with pytest.raises(lc.ConflictModelError):
        lc.gate(sim, {**hw, "conflicts_per_access": 1.0})
    # a degenerate counter (IDX==BC -> NaN cpa) is never a silent pass
    with pytest.raises(lc.ConflictModelError):
        lc.gate(sim, {"conflicts_per_access": float("nan")})
    # absolute=True additionally compares BC/IDX (matched-scale corpus-style check)
    assert lc.gate(sim, dict(sim), absolute=True)
    with pytest.raises(lc.ConflictModelError):
        lc.gate(sim, hw, absolute=True)                              # cpa ok but BC scale differs
    # nothing comparable -> refuse a vacuous pass
    with pytest.raises(lc.ConflictModelError):
        lc.gate({"BC": 1}, {"IDX": 1})
def test_pipeline_traces_origin_across_stages(tmp_path):
    from rocke.helpers.tiling.register_mapper import RegisterMapper
    mma = _mma()
    dist = mma.a_layout
    rm = RegisterMapper(dist)
    mp = {(l, r): rm.matrix_coordinates(l, r)
          for l in range(rm.num_lanes) for r in range(rm.num_vector_items)}
    addr = lambda m, k: m * 16 + k
    pipe = lr.Pipeline(stages=(
        lr.FlowStage("global", lr.LogicalTileComponent(dist=dist, dims=("M", "K"),
                                                       label_coords="logical"), source="dist"),
        lr.FlowStage("registers", lr.RegisterFileComponent(dist=dist, dims=("M", "K"), dtype_bits=16),
                     source="dist", transform="global load"),
        lr.FlowStage("lds", lr.LdsBankView(mp=mp, addr_fn=addr, dims=("M", "K"), nbanks=32, elem_bytes=4),
                     source="mp", transform="LDS store"),
    ), title="A dataflow")
    # M3K5 must resolve to exactly one cell in each stage, and be the machine-correct location.
    assert pipe.stages[0].cells_for((3, 5)) == {(3, 5)}          # global coord
    assert pipe.stages[1].cells_for((3, 5)) == {(19, 1)}         # register (lane,reg) owning M3K5
    assert pipe.stages[2].cells_for((3, 5)) == {(1, 21)}         # LDS (depth,bank): addr 53 -> (1,21)
    pipe.trace((3, 5))
    p = pipe.render(str(tmp_path / "flow.png"))
    assert os.path.getsize(p) > 0
def test_lds_bank_view_renders_flow_and_thread(tmp_path):
    from rocke.helpers.tiling.register_mapper import RegisterMapper
    mma = _mma()
    rm = RegisterMapper(mma.a_layout)
    mp = {(l, r): rm.matrix_coordinates(l, r)
          for l in range(rm.num_lanes) for r in range(rm.num_vector_items)}
    addr = lambda r, c: r * 16 + c                       # K-contiguous LDS store
    flow = lr.LdsBankView(mp=mp, addr_fn=addr, nbanks=32, elem_bytes=4, label_by="flow", dims=("M", "K"))
    thread = lr.LdsBankView(mp=mp, addr_fn=addr, nbanks=32, elem_bytes=4, label_by="thread")
    assert flow.grid_size()[0] == 32                    # banks on x, origin (depth0,bank0) top-left
    fp = flow.render(str(tmp_path / "lds_flow.png"))
    tp = thread.render(str(tmp_path / "lds_thread.png"))
    assert os.path.getsize(fp) > 0 and os.path.getsize(tp) > 0


# ---------------------------------------------------------------- lds-conflict tooling
def test_lds_conflict_selftest_green():
    assert lc.selftest(verbose=False)


def test_ge2_axis_key_rule(tmp_path):
    """The >=2-axis KEY rule: a COMPACT per-thread group whose data spans >1 on >=2 axes (e.g. N0-3 K0-7,
    NOT N0-3 K0) must render unambiguously -- the FIRST such thread is promoted to DETAILED (the key), and a
    panel NOTE flags any thread whose internal (grid-pos -> coord) ordering differs from it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fwd(diverge=False):
        m = {}
        for l in range(64):                                    # DENSE (64 > dense_rows) -> per-row block groups
            b0, b1 = (l % 8) * 2, (l // 8) * 2                 # each lane's row = a 2x2 (2-AXIS) patch
            order = [(0, 0), (0, 1), (1, 0), (1, 1)]
            if diverge and l == 5:
                order = [(0, 0), (1, 0), (0, 1), (1, 1)]       # lane 5: a DIFFERENT internal ordering
            for r, (d0, d1) in enumerate(order):
                m[(l, r)] = (b0 + d0, b1 + d1)
        return m

    def run(diverge):
        c = lr.RegisterFileComponent(fwd_map=fwd(diverge), dims=("M", "K"), dtype_bits=16)
        fig, ax = plt.subplots()
        c.draw(ax)                                             # draw() applies the rule + sets _ambig_note
        plt.close(fig)
        g = c._effective_groups()
        g = c._disambiguate_ge2_axes(g)
        return sum(1 for x in g if x.detail == "detailed"), c._ambig_note

    det_u, note_u = run(False)
    assert det_u == 1 and note_u == "", "uniform: exactly T0 detailed as the key, no divergence note"
    det_d, note_d = run(True)
    assert det_d == 1 and "T0" in note_d and "T5" in note_d, "divergent lane must be flagged in the note"

    # a 1-AXIS strip (N0-3 K0) is unambiguous -> NO promotion, NO note.
    one_axis = {(l, r): (r, 0) for l in range(64) for r in range(4)}   # each row varies only in dim0
    c1 = lr.RegisterFileComponent(fwd_map=one_axis, dims=("M", "K"), dtype_bits=16)
    fig, ax = plt.subplots(); c1.draw(ax); plt.close(fig)
    assert c1._ambig_note == "", "a single-axis strip is unambiguous -- no note"
