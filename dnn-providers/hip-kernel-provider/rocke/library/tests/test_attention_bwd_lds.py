# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the backward LDS planner (``_attention_bwd_lds``).

* Phase planner: no two buffers that are live in the same phase overlap
  (checked byte by byte, independently of ``LdsPlan.check``), resident buffers
  overlap nothing, ring slots are distinct, every start-point tile fits the LDS
  capacity of its arch, and the phase aliasing is what makes the large gfx942
  tiles fit.
* XT layout: the address map is a bijection inside the buffer, vectors stay
  inside one swizzle chunk, the writer covers the image exactly once, and the
  conflict predictor sees the XOR image as conflict-free on gfx942 where the
  plain image conflicts.
* DMA slab layout: the source-side permutation is the inverse of the read map
  (the bytes a lane-contiguous DMA writes land where the reads look), and the
  chunk XOR is ``chunk16 ^ (row % 8)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from kernels.common._attention_bwd_lds import (
    BUFFER_ALIGN_BYTES,
    PHASES,
    BwdLdsRequest,
    DmaSlabLayout,
    XtLayout,
    XtWriter,
    choose_xt_swizzle,
    default_xt_writer,
    dma_slab_read_conflicts,
    lds_capacity_bytes,
    plan_bwd_lds,
    xt_conflicts,
    xt_read_accesses,
    xt_write_accesses,
)

# Per-arch start-point tiles (gfx942, gfx950, RDNA), as planner
# requests. Each entry: (arch, request kwargs).
START_POINTS = [
    # gfx942 "XT" body, K, K^T, V register resident.
    ("gfx942", {"head_size": 32, "k_m0": 32, "k_n0": 128, "waves": 4}),
    ("gfx942", {"head_size": 64, "k_m0": 32, "k_n0": 128, "waves": 4}),
    ("gfx942", {"head_size": 128, "k_m0": 16, "k_n0": 128, "waves": 4}),
    (
        "gfx942",
        {"head_size": 128, "k_m0": 16, "k_n0": 128, "waves": 4, "kt_source": "lds"},
    ),
    (
        "gfx942",
        {"head_size": 128, "k_m0": 16, "k_n0": 128, "waves": 4, "ring_depth": 2},
    ),
    (
        "gfx942",
        {"head_size": 128, "k_m0": 16, "k_n0": 64, "waves": 4},
    ),  # stage_vec = 1 row
    (
        "gfx942",
        {
            "head_size": 128,
            "k_m0": 16,
            "k_n0": 16,
            "waves": 1,
            "kv_residency": "lds",
            "kt_source": "lds",
            "transpose_source": "lds_plain",
        },
    ),
    # gfx950 "trload" body: DMA, tr reads, two-deep ring.
    ("gfx950", {"head_size": 32, "k_m0": 32, "k_n0": 128, "waves": 4}),
    (
        "gfx950",
        {
            "head_size": 64,
            "k_m0": 32,
            "k_n0": 256,
            "waves": 4,
            "kt_source": "lds",
            "transpose_source": "tr_read",
            "global_path": "dma",
            "ring_depth": 2,
        },
    ),
    (
        "gfx950",
        {
            "head_size": 128,
            "k_m0": 16,
            "k_n0": 192,
            "waves": 4,
            "kt_source": "lds",
            "transpose_source": "tr_read",
            "global_path": "dma",
            "ring_depth": 2,
        },
    ),
    (
        "gfx950",
        {
            "head_size": 128,
            "k_m0": 32,
            "k_n0": 128,
            "waves": 4,
            "kt_source": "lds",
            "transpose_source": "tr_read",
            "global_path": "dma",
            "ring_depth": 2,
        },
    ),
    (
        "gfx950",
        {
            "head_size": 128,
            "k_m0": 16,
            "k_n0": 192,
            "waves": 4,
            "kt_source": "lds",
            "transpose_source": "tr_read",
            "global_path": "dma",
            "ring_depth": 3,
        },
    ),
    (
        "gfx950",
        {
            "head_size": 128,
            "k_m0": 16,
            "k_n0": 64,
            "waves": 4,
            "transpose_source": "lds_plain",
        },
    ),
    # RDNA: the one-wave WMMA schedule (LDS K/V, LDS accumulators).
    *[
        (
            arch,
            {
                "head_size": d,
                "k_m0": 16,
                "k_n0": 16,
                "waves": 1,
                "wave_size": 32,
                "kv_residency": "lds",
                "kt_source": "lds",
                "transpose_source": "wmma_lds",
                "acc_in_lds": True,
            },
        )
        for arch in ("gfx1151", "gfx1201")
        for d in (32, 64, 128)
    ],
]


def _ids(cases):
    return [f"{a}-" + "-".join(f"{k}{v}" for k, v in kw.items()) for a, kw in cases]


def _byte_owner_check(plan) -> None:
    """Independent overlap check: claim every byte of every live buffer."""
    for phase in PHASES:
        owner = {}
        for buf in plan.live(phase):
            for byte in range(buf.offset, buf.end):
                prev = owner.setdefault(byte, (buf.name, buf.slot))
                assert prev == (
                    buf.name,
                    buf.slot,
                ), f"{phase}: byte {byte} owned by {prev} and {(buf.name, buf.slot)}"


@pytest.mark.parametrize("arch,kw", START_POINTS, ids=_ids(START_POINTS))
def test_start_points_fit_and_never_overlap(arch, kw):
    plan = plan_bwd_lds(BwdLdsRequest(**kw), arch=arch)
    assert plan.fits
    assert plan.capacity_bytes == lds_capacity_bytes(arch)
    assert plan.total_bytes <= plan.capacity_bytes
    _byte_owner_check(plan)
    for buf in plan.buffers:
        assert buf.offset % BUFFER_ALIGN_BYTES == 0
        assert buf.end <= plan.total_bytes
    # A resident buffer shares no byte with any other buffer, in any phase.
    for res in (b for b in plan.buffers if b.resident):
        for other in plan.buffers:
            if other is res:
                continue
            assert other.end <= res.offset or res.end <= other.offset
    # Footprint is resident + the largest phase.
    assert plan.total_bytes == plan.resident_bytes + max(s for _, s in plan.phase_bytes)


def test_per_arch_capacity_values():
    assert lds_capacity_bytes("gfx942") == 65536
    assert lds_capacity_bytes("gfx950") == 163840
    assert lds_capacity_bytes("gfx1151") == 65536
    assert lds_capacity_bytes("gfx1201") == 65536


def test_gfx942_d128_stage0_uses_exactly_the_capacity():
    # K + K^T = 2 * kN0 * D * 2 bytes is the stage-0 peak; the loop aliases it.
    plan = plan_bwd_lds(
        BwdLdsRequest(head_size=128, k_m0=16, k_n0=128, waves=4), arch="gfx942"
    )
    assert dict(plan.phase_bytes)["stage0_k"] == 2 * 128 * 128 * 2
    assert plan.total_bytes == 65536
    # Loop footprint: 4 * kM0 * D * 2 + kM0 * kN0 * 2 (Q, dO, Q^T, dO^T, dS;
    # the lse2 / Dsum stats are read from the workspace, not from LDS).
    assert dict(plan.phase_bytes)["loop"] == 4 * 16 * 128 * 2 + 16 * 128 * 2


def test_aliasing_is_what_makes_the_ring_fit():
    req = BwdLdsRequest(head_size=128, k_m0=16, k_n0=128, waves=4, ring_depth=2)
    plan = plan_bwd_lds(req, arch="gfx942")
    assert plan.fits
    assert plan.unaliased_bytes > plan.capacity_bytes
    # Stage 0 and loop buffers really share bytes.
    k_stage = plan.buffer("k_stage")
    q1 = plan.buffer("q", 1)
    assert k_stage.offset <= q1.offset < k_stage.end
    assert set(k_stage.phases).isdisjoint(q1.phases)


def test_ring_slots_are_distinct_and_complete():
    req = BwdLdsRequest(
        head_size=128,
        k_m0=16,
        k_n0=192,
        waves=4,
        kt_source="lds",
        transpose_source="tr_read",
        global_path="dma",
        ring_depth=3,
    )
    plan = plan_bwd_lds(req, arch="gfx950")
    # the ring stages each slot's Q / dO and its lse2 / Dsum rows
    for name in ("q", "do", "lse2", "dsum"):
        offs = plan.ring_offsets(name)
        assert len(offs) == 3 and len(set(offs)) == 3
    # the synchronous staging reads the stats from the workspace: no LDS copy
    sync = plan_bwd_lds(replace(req, global_path="vgpr", ring_depth=1), arch="gfx950")
    assert not sync.has("lse2") and not sync.has("dsum")
    assert not plan.has("qt") and not plan.has("dot")  # tr_read: no XT copies
    assert plan.layout("q").__class__ is DmaSlabLayout  # DMA buffers use slab XOR
    # A resident K copy feeds both K^T reads and the register staging of K.
    assert plan.has("k_res") and not plan.has("k_stage")


@pytest.mark.parametrize(
    "arch,kw",
    [
        ("gfx942", {"head_size": 128, "k_m0": 16, "k_n0": 256, "waves": 4}),
        (
            "gfx942",
            {
                "head_size": 128,
                "k_m0": 16,
                "k_n0": 128,
                "waves": 4,
                "kv_residency": "lds",
                "kt_source": "lds",
            },
        ),
        (
            "gfx1151",
            {
                "head_size": 128,
                "k_m0": 16,
                "k_n0": 64,
                "waves": 1,
                "wave_size": 32,
                "kv_residency": "lds",
                "kt_source": "lds",
                "transpose_source": "wmma_lds",
                "acc_in_lds": True,
            },
        ),
    ],
)
def test_over_capacity_raises_only_when_strict(arch, kw):
    req = BwdLdsRequest(**kw)
    with pytest.raises(ValueError, match="exceeds"):
        plan_bwd_lds(req, arch=arch)
    plan = plan_bwd_lds(req, arch=arch, strict=False)
    assert not plan.fits
    _byte_owner_check(plan)


def test_capacity_is_per_arch():
    kw = {
        "head_size": 128,
        "k_m0": 16,
        "k_n0": 192,
        "waves": 4,
        "kt_source": "lds",
        "transpose_source": "tr_read",
        "global_path": "dma",
        "ring_depth": 2,
    }
    assert plan_bwd_lds(BwdLdsRequest(**kw), arch="gfx950").fits
    with pytest.raises(ValueError):
        plan_bwd_lds(BwdLdsRequest(**kw), arch="gfx942")


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"kv_residency": "lds", "kt_source": "reg"}, "kt_source"),
        ({"global_path": "dma", "transpose_source": "xt_lds"}, "dma"),
        ({"ring_depth": 4}, "ring_depth"),
        ({"transpose_source": "bogus"}, "transpose_source"),
        ({"lds_swizzle": (("q", "pad7"),)}, "swizzle"),
    ],
)
def test_request_couplings_raise(kw, match):
    base = {"head_size": 64, "k_m0": 32, "k_n0": 128, "waves": 4}
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        BwdLdsRequest(**base)


def test_from_geometry_reads_a_geometry_like_object():
    # Local stand-in for BwdTileGeometry (owned by the body module).
    @dataclass(frozen=True)
    class _Geom:
        waves: int = 4
        k_m0: int = 16
        k_n0: int = 128
        k_k4: int = 32
        head_size: int = 128
        kv_residency: str = "reg"
        kt_source: str = "reg"
        transpose_source: str = "xt_lds"
        ring_depth: int = 1
        global_path: str = "vgpr"
        pt_route: str = "relabel"
        lds_swizzle: str = "qt=pad8,dot=xor"

    req = BwdLdsRequest.from_geometry(_Geom())
    assert req.lds_swizzle == (("dot", "xor"), ("qt", "pad8"))
    plan = plan_bwd_lds(req, arch="gfx942", strict=False)
    assert plan.layout("qt").swizzle == "pad8"
    assert plan.layout("dot").swizzle == "xor"
    assert plan.buffer("qt").nbytes == 128 * (16 * 2 + 8)


def test_pt_route_lds_and_split_dq_buffers():
    base = {"head_size": 64, "k_m0": 32, "k_n0": 128, "waves": 4}
    plan = plan_bwd_lds(BwdLdsRequest(pt_route="lds", **base), arch="gfx942")
    assert plan.has("pt") and plan.has("ds")
    plan = plan_bwd_lds(BwdLdsRequest(dq_mode="split", **base), arch="gfx942")
    assert not plan.has("ds") and not plan.has("pt")


# ---------------------------------------------------------------------------
# XT layout
# ---------------------------------------------------------------------------

XT_SHAPES = [(d, s) for d in (32, 64, 128) for s in (16, 32, 64, 128, 256)]


@pytest.mark.parametrize("rows,cols", XT_SHAPES)
@pytest.mark.parametrize("swizzle", ["xor", "none", "pad8", "pad16", "pad32"])
def test_xt_layout_is_a_bijection(rows, cols, swizzle):
    lay = XtLayout(rows, cols, 2, swizzle)
    offs = {lay.offset(r, c) for r in range(rows) for c in range(cols)}
    assert len(offs) == rows * cols
    assert min(offs) >= 0 and max(offs) + 2 <= lay.size_bytes
    if swizzle == "xor":
        assert lay.size_bytes == rows * cols * 2
        # A 4-element (8-byte) group stays contiguous: the b64 read is legal.
        for r in range(rows):
            for c in range(0, cols, 4):
                base = lay.offset(r, c)
                assert [lay.offset(r, c + i) for i in range(4)] == [
                    base + 2 * i for i in range(4)
                ]


@pytest.mark.parametrize(
    "seq,depth",
    [(16, 128), (32, 64), (32, 32), (128, 128), (128, 64), (128, 32), (64, 128)],
)
def test_xt_writer_covers_the_image_once(seq, depth):
    w = default_xt_writer(seq, depth, 256)
    lay = XtLayout(depth, seq, 2, "xor")
    seen = {}
    for it in range(w.iterations):
        for tid in range(w.threads):
            coords = w.unit_coords(tid, it)
            if coords is None:
                continue
            s0, d0 = coords
            for j in range(w.vec_d):
                for r in range(w.k_per_thread):
                    key = (d0 + j, s0 + r)
                    assert key not in seen
                    seen[key] = tid
    assert len(seen) == seq * depth
    # Global side: d_lanes consecutive lanes read one contiguous row segment.
    s_first, d_first = w.unit_coords(0, 0)
    segment = [w.unit_coords(t, 0) for t in range(w.d_lanes_eff)]
    assert all(s == s_first for s, _ in segment)
    assert [d for _, d in segment] == [
        d_first + i * w.vec_d for i in range(w.d_lanes_eff)
    ]
    # The writer's stores land on the expected element offsets.
    acc = xt_write_accesses(lay, w, wave=0, iteration=0, j=0)
    assert all(width == w.write_bytes for _, _, width in acc)


def test_xt_writer_validation():
    with pytest.raises(ValueError):
        XtWriter(seq=16, depth=100, threads=256, vec_d=8)
    with pytest.raises(ValueError):
        XtWriter(seq=15, depth=128, threads=256, k_per_thread=2)


# Shapes of the gfx942 start points: Q^T / dO^T (D x kM0) and K^T (D x kN0).
GFX942_XT = [
    ("qt", 128, 16),
    ("qt", 64, 32),
    ("qt", 32, 32),
    ("kt", 128, 128),
    ("kt", 64, 128),
    ("kt", 32, 128),
]


@pytest.mark.parametrize("kind,rows,cols", GFX942_XT)
def test_xt_xor_reads_conflict_free_on_gfx942(kind, rows, cols):
    lay = XtLayout(rows, cols, 2, "xor")
    w = default_xt_writer(cols, rows, 256)
    pred = xt_conflicts(lay, w, "gfx942")
    assert pred["read"] == 1
    plain = xt_conflicts(XtLayout(rows, cols, 2, "none"), w, "gfx942")
    assert plain["read"] > 1  # the predictor does see conflicts in the plain image
    if kind == "qt":
        # Per-step images: the writer is conflict-free as well.
        assert pred["write"] == 1


def test_xt_read_accesses_follow_the_mfma_b_layout():
    lay = XtLayout(64, 32, 2, "none")
    acc = xt_read_accesses(lay, n0=16, k0=16)
    assert len(acc) == 64
    for lane, addr, width in acc:
        assert width == 8
        row, col = 16 + lane % 16, 16 + (lane // 16) * 4
        assert addr == row * 64 + col * 2


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("rows,cols", [(128, 16), (64, 32), (128, 128)])
def test_choose_xt_swizzle_is_predictor_optimal(arch, rows, cols):
    w = default_xt_writer(cols, rows, 256)
    best, table = choose_xt_swizzle(rows, cols, w, arch)
    preds = {k: v["prediction"] for k, v in table.items() if "prediction" in v}
    assert preds[best]["read"] == min(p["read"] for p in preds.values())
    if arch == "gfx942":
        assert best == "xor"


def test_choose_xt_swizzle_without_predictor_profile():
    best, table = choose_xt_swizzle(64, 16, None, "gfx1151")
    assert best == "pad8"
    assert table["xor"]["prediction"] is None


# ---------------------------------------------------------------------------
# DMA slab layout
# ---------------------------------------------------------------------------

SLAB_SHAPES = [
    (16, 128),
    (32, 128),
    (32, 64),
    (16, 32),
    (64, 32),
    (192, 128),
    (256, 64),
]


@pytest.mark.parametrize("rows,cols", SLAB_SHAPES)
def test_dma_slab_source_permutation_inverts_the_read_map(rows, cols):
    lay = DmaSlabLayout(rows, cols, 2)
    slots = lay.size_bytes // 16
    sources = [lay.dma_source(s) for s in range(slots)]
    assert len(set(sources)) == slots  # every 16-byte source vector fetched once
    for s, (r, c) in enumerate(sources):
        assert c % 8 == 0
        assert lay.offset(r, c) == 16 * s  # the DMA'd bytes land where reads look
        assert [lay.offset(r, c + i) for i in range(8)] == [
            16 * s + 2 * i for i in range(8)
        ]


@pytest.mark.parametrize("rows,cols", SLAB_SHAPES)
def test_dma_slab_chunk_xor_is_row_mod_8(rows, cols):
    lay = DmaSlabLayout(rows, cols, 2)
    if lay.row_bytes < 128:
        pytest.skip("folded rows: the XOR key is the line index (checked below)")
    for r in range(rows):
        for c in range(0, cols, 8):
            chunk16 = (c * 2 % 128) // 16
            phys = (lay.offset(r, c) % 128) // 16
            assert phys == chunk16 ^ (r % 8)


def test_dma_slab_folded_rows_xor_by_line():
    lay = DmaSlabLayout(32, 32, 2)  # 64-byte rows, two rows per line
    for r in range(32):
        line = r // 2
        for c in range(0, 32, 8):
            logical_chunk = ((r % 2) * 64 + c * 2) // 16
            assert (lay.offset(r, c) % 128) // 16 == logical_chunk ^ (line % 8)
            assert lay.offset(r, c) // 128 == line


def test_dma_slab_validation():
    with pytest.raises(ValueError):
        DmaSlabLayout(16, 100, 2)  # 200-byte rows are not whole lines
    with pytest.raises(ValueError):
        DmaSlabLayout(3, 32, 2)  # folded rows must fill whole lines


def test_dma_slab_straight_reads_conflict_free_on_gfx950():
    # 16x16x32 A-operand reads (one ds_read_b128 per lane) of Q / dO tiles.
    for rows, cols in [(16, 128), (32, 128), (32, 64)]:
        assert dma_slab_read_conflicts(DmaSlabLayout(rows, cols, 2), "gfx950") == 1


def test_epilogue_phase_aliases_the_loop_and_is_absent_with_lds_accumulators():
    req = BwdLdsRequest(head_size=128, k_m0=16, k_n0=128, waves=4)
    plan = plan_bwd_lds(req, arch="gfx942")
    out = plan.buffer("dkv_out")
    assert out.phases == ("epilogue",) and out.nbytes == 128 * 128 * 2
    assert out.offset == plan.resident_bytes  # phase-aliased above the residents
    assert dict(plan.phase_bytes)["epilogue"] == 128 * 128 * 2
    _byte_owner_check(plan)
    acc = BwdLdsRequest(
        head_size=64, k_m0=16, k_n0=16, waves=1, wave_size=32,
        kv_residency="lds", kt_source="lds", transpose_source="wmma_lds",
        pt_route="lds", acc_in_lds=True,
    )  # fmt: skip
    assert not plan_bwd_lds(acc, arch="gfx1151").has("dkv_out")


def test_epilogue_variants_and_padded_epilogue_buffer():
    base = dict(head_size=128, k_m0=16, k_n0=128, waves=4)
    ref = plan_bwd_lds(BwdLdsRequest(**base), arch="gfx950")
    direct = plan_bwd_lds(BwdLdsRequest(dkv_epilogue="direct", **base), arch="gfx950")
    assert not direct.has("dkv_out") and dict(direct.phase_bytes)["epilogue"] == 0
    pair = plan_bwd_lds(
        BwdLdsRequest(dkv_epilogue="lds_staged_pair", **base), arch="gfx950"
    )
    assert pair.ring_offsets("dkv_out") == (0, 128 * 128 * 2)
    assert dict(pair.phase_bytes)["epilogue"] == 2 * 128 * 128 * 2
    _byte_owner_check(pair)
    # a global DMA path does not make the epilogue buffer a DMA slab
    dma = dict(base, kt_source="lds", transpose_source="tr_read", global_path="dma")
    assert plan_bwd_lds(BwdLdsRequest(**dma), arch="gfx950").buffer(
        "dkv_out"
    ).layout == ("rowmajor:none")
    for pad in (8, 16, 32):
        req = BwdLdsRequest(lds_swizzle=(("dkv_out", f"pad{pad}"),), **base)
        out = plan_bwd_lds(req, arch="gfx950").buffer("dkv_out")
        assert out.layout == f"rowmajor:pad{pad}"
        assert out.nbytes == 128 * (128 * 2 + pad)
    # the default request is the plan of the defaults
    assert ref.buffer("dkv_out").layout == "rowmajor:none"


@pytest.mark.parametrize("arch,kw", START_POINTS, ids=_ids(START_POINTS))
def test_phase_low_order_moves_only_the_resident_region(arch, kw):
    ref = plan_bwd_lds(BwdLdsRequest(**kw), arch=arch)
    low = plan_bwd_lds(BwdLdsRequest(lds_order="phase_low", **kw), arch=arch)
    _byte_owner_check(low)
    assert (low.total_bytes, low.resident_bytes) == (
        ref.total_bytes,
        ref.resident_bytes,
    )
    top = max(s for _, s in ref.phase_bytes)
    by_key = {(b.name, b.slot): b for b in low.buffers}
    for b in ref.buffers:
        moved = by_key[(b.name, b.slot)]
        assert (moved.nbytes, moved.layout, moved.phases) == (
            b.nbytes,
            b.layout,
            b.phases,
        )
        if b.resident:
            assert moved.offset == b.offset + top
        else:
            assert moved.offset == b.offset - ref.resident_bytes
    assert ref.phase_offset == ref.resident_bytes and low.phase_offset == 0


@pytest.mark.parametrize(
    "kw,match",
    [({"dkv_epilogue": "staged"}, "dkv_epilogue"), ({"lds_order": "x"}, "lds_order")],
)
def test_epilogue_and_order_values_are_checked(kw, match):
    with pytest.raises(ValueError, match=match):
        BwdLdsRequest(head_size=64, k_m0=32, k_n0=128, waves=4, **kw)
