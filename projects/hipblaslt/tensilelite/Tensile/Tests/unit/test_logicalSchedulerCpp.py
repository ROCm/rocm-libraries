# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the optional C++ (nanobind) LogicalScheduler primitives.

These tests compare the pure-Python data/config primitives in
``Tensile.Components.Subtile.LogicalScheduler`` against the compiled
``tensile_writer.subtile.logical_scheduler`` extension. They run only when the
extension is importable; otherwise they skip, so the default (Python-only)
TensileLite build is unaffected.

Scope: this covers the value/config layer (Pass, fmt_mt, MFMATileRange,
ReadGranularity, SchedulerConfig — including partition normalization and
candidate generation — plus the placement / op value types), the value types
(Dep, SubIterKSlot, EmittedModule, InlineModuleOp), AND the writer-free pass
pipeline (place_LRs through emit/build) exposed by the C++ ``LogicalScheduler``.
The pass-pipeline tests compare the C++ schedule against the Python one
pass-by-pass via the byte-identical print_* helpers.

Still NOT ported (and exercised only on the Python side): populate_instructions,
InstructionEmitter dispatch, writer VGPR-pool allocation, and rocisa Module /
Kernel.mainLoop control-flow emission. The C++ pipeline operates purely on the
data-only logical schedule.

The config cases below mirror those exercised by
``test_SubtileBasedLogicalScheduler`` and ``test_SubtileBasedSchedulerRef``.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import contextlib

import pytest

cppls = pytest.importorskip("tensile_writer.subtile.logical_scheduler")

from Tensile.Components.Subtile import LogicalScheduler as ls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_RG_KEYS = ("lrA", "lrB", "grA", "grB", "lrSA", "lrSB", "grSA", "grSB")


def _build(mod, spec):
    """Build a SchedulerConfig of module ``mod`` from a kwargs ``spec``.

    ReadGranularity fields are given as ``(mn, k)`` tuples (or ``None``).
    """
    kw = dict(spec)
    for key in _RG_KEYS:
        val = kw.get(key)
        if val is not None:
            kw[key] = mod.ReadGranularity(*val)
    return mod.SchedulerConfig(**kw)


class _FakeTileInfo:
    """Minimal stand-in exposing ``localMMATileGrid[0]`` for candidate tests."""

    def __init__(self, m):
        self.localMMATileGrid = (m, 0)


@contextlib.contextmanager
def cpp_delegation():
    """Temporarily enable C++ delegation in LogicalScheduler."""
    saved_use, saved_cpp = ls._USE_CPP, ls._CPP
    ls._CPP = cppls
    ls._USE_CPP = True
    try:
        yield
    finally:
        ls._USE_CPP = saved_use
        ls._CPP = saved_cpp


# Representative config cases mirroring the existing scheduler tests. Each maps
# a name to (kwargs, expected) where ``expected`` pins partitionSizesN (or M)
# to lock correctness, not just Python/C++ self-consistency.
CONFIGS = {
    # No scales, k=1 (test_no_scale_k1): single full partition each dim.
    "no_scale_k1": (
        dict(numMFMATilesM=2, numMFMATilesN=2, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2)),
        dict(partitionSizesM=[2], partitionSizesN=[2], numPartitions=1,
             hasScale=False),
    ),
    # FP4 2x2 with scales (test_2x2_*): tilesM=tilesN=8, 4x4 partitions.
    "fp4_2x2": (
        dict(numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
             lrSA=(2, 2), lrSB=(2, 2), grSA=(2, 2), grSB=(2, 2),
             partitionSizeM=4, partitionSizeN=4),
        dict(partitionSizesM=[4, 4], partitionSizesN=[4, 4], numPartitions=4,
             hasScale=True),
    ),
    # 10x1 BF16, no scale (test_10x1_k1_bf16): 10 partitions along N.
    "bf16_10x1": (
        dict(numMFMATilesM=10, numMFMATilesN=10, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
             partitionSizeM=1, partitionSizeN=10),
        dict(numPartitions=10, hasScale=False),
    ),
    # BF16 256x384 even N (test_bf16_partition_256x384): tilesN=12.
    "bf16_256x384_n6": (
        dict(numMFMATilesM=8, numMFMATilesN=12, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=6),
        dict(partitionSizesN=[6, 6]),
    ),
    "bf16_256x384_n4": (
        dict(numMFMATilesM=8, numMFMATilesN=12, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=4),
        dict(partitionSizesN=[4, 4, 4]),
    ),
    "bf16_256x384_n5": (
        dict(numMFMATilesM=8, numMFMATilesN=12, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=5),
        dict(partitionSizesN=[5, 2, 5]),
    ),
    # BF16 256x352 odd N (test_bf16_partition_256x352): tilesN=11.
    "bf16_256x352_n4": (
        dict(numMFMATilesM=8, numMFMATilesN=11, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=4),
        dict(partitionSizesN=[4, 3, 4]),
    ),
    "bf16_256x352_n3": (
        dict(numMFMATilesM=8, numMFMATilesN=11, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=3),
        dict(partitionSizesN=[3, 2, 3, 3]),
    ),
    # BF16 256x368 miWaveGroup[4,1] (test_bf16_partition_256x368): tilesN=23.
    "bf16_256x368_n4": (
        dict(numMFMATilesM=4, numMFMATilesN=23, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=4),
        dict(partitionSizesN=[4, 4, 3, 4, 4, 4]),
    ),
    "bf16_256x368_n8": (
        dict(numMFMATilesM=4, numMFMATilesN=23, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), partitionSizeN=8),
        dict(partitionSizesN=[8, 7, 8]),
    ),
    # Explicit list partition spec.
    "explicit_list_N": (
        dict(numMFMATilesM=2, numMFMATilesN=12, numSubIterK=2,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
             partitionSizeN=[5, 2, 5]),
        dict(partitionSizesN=[5, 2, 5], numPartitions=3),
    ),
    # pgr=0 single partition (plr/offsetPartition edge).
    "pgr0_single": (
        dict(numMFMATilesM=2, numMFMATilesN=2, numSubIterK=1,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), pgr=0),
        dict(numPartitions=1),
    ),
    # pgr=1 single partition.
    "pgr1_single": (
        dict(numMFMATilesM=2, numMFMATilesN=2, numSubIterK=1,
             lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), pgr=1),
        dict(numPartitions=1),
    ),
}


# ---------------------------------------------------------------------------
# Pass enum
# ---------------------------------------------------------------------------
def test_pass_enum_values_match():
    for member in ls.Pass:
        cpp = getattr(cppls.Pass, member.name)
        assert int(cpp.value) == int(member.value), member.name


# ---------------------------------------------------------------------------
# fmt_mt
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mt", [0, 1, 2, 3, 7, 15])
def test_fmt_mt_match(mt):
    assert cppls.fmt_mt(mt) == ls.fmt_mt(mt)


def test_fmt_mt_delegation():
    with cpp_delegation():
        assert ls.fmt_mt(0) == "n"
        assert ls.fmt_mt(2) == "n+2"


# ---------------------------------------------------------------------------
# MFMATileRange
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rng", [
    (0, 1, 0, 4),
    (0, 2, 0, 8),
    (1, 2, 4, 8),
    (2, 4, 0, 16),
])
def test_mfma_tile_range(rng):
    py = ls.MFMATileRange(*rng)
    cpp = cppls.MFMATileRange(*rng)
    assert list(cpp.subIterK_list) == list(py.subIterK_list)
    assert list(cpp.tileId_list) == list(py.tileId_list)
    assert cpp.fmt_k() == py.fmt_k()
    assert cpp.fmt_tiles() == py.fmt_tiles()
    assert (cpp.subIterK_start, cpp.subIterK_end, cpp.tileId_start,
            cpp.tileId_end) == (py.subIterK_start, py.subIterK_end,
                                py.tileId_start, py.tileId_end)


# ---------------------------------------------------------------------------
# ReadGranularity.tile_range
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mn,k", [(1, 1), (1, 2), (2, 2), (2, 1), (4, 2)])
@pytest.mark.parametrize("kk,t0,t1", [
    (0, 0, 8), (1, 0, 8), (3, 1, 7), (2, 3, 5), (0, 5, 5),
])
def test_read_granularity_tile_range(mn, k, kk, t0, t1):
    py = ls.ReadGranularity(mn, k).tile_range(kk, t0, t1)
    cpp = cppls.ReadGranularity(mn, k).tile_range(kk, t0, t1)
    assert (cpp.subIterK_start, cpp.subIterK_end, cpp.tileId_start,
            cpp.tileId_end) == (py.subIterK_start, py.subIterK_end,
                                py.tileId_start, py.tileId_end)


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CONFIGS))
def test_scheduler_config_parity(name):
    spec, expected = CONFIGS[name]
    py = _build(ls, spec)
    cpp = _build(cppls, spec)

    # Raw input specs are preserved verbatim (int stays int, list stays list)
    # for parity with the Python dataclass fields.
    def _norm_spec(v):
        return list(v) if isinstance(v, (list, tuple)) else v

    assert _norm_spec(cpp.partitionSizeM) == _norm_spec(py.partitionSizeM)
    assert _norm_spec(cpp.partitionSizeN) == _norm_spec(py.partitionSizeN)

    assert list(cpp.partitionSizesM) == list(py.partitionSizesM)
    assert list(cpp.partitionSizesN) == list(py.partitionSizesN)
    assert cpp.numPartitionsM == py.numPartitionsM
    assert cpp.numPartitionsN == py.numPartitionsN
    assert cpp.numPartitions == py.numPartitions
    assert cpp.hasScale == py.hasScale
    assert cpp.plr == py.plr
    assert cpp.offsetPartition == py.offsetPartition
    assert cpp.pgr == py.pgr
    assert list(cpp.prefixM) == list(py._prefixM)
    assert list(cpp.prefixN) == list(py._prefixN)

    # Absolute pins from the original scheduler tests.
    for key, val in expected.items():
        assert getattr(py, key) == val or list(getattr(py, key)) == val, \
            f"python {key}"
        cpp_val = getattr(cpp, key)
        if isinstance(val, list):
            assert list(cpp_val) == val, f"cpp {key}"
        else:
            assert cpp_val == val, f"cpp {key}"


def test_scheduler_config_errors_parity():
    """Invalid configs must raise in both implementations."""
    # pgr=0 with >1 partition.
    bad = dict(numMFMATilesM=4, numMFMATilesN=4, numSubIterK=1,
               lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
               partitionSizeM=2, partitionSizeN=2, pgr=0)
    with pytest.raises(Exception):
        _build(ls, bad)
    with pytest.raises(Exception):
        _build(cppls, bad)

    # Explicit list that does not sum to total.
    bad_sum = dict(numMFMATilesM=2, numMFMATilesN=12, numSubIterK=2,
                   lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
                   partitionSizeN=[5, 2, 4])
    with pytest.raises(Exception):
        _build(ls, bad_sum)
    with pytest.raises(Exception):
        _build(cppls, bad_sum)


# ---------------------------------------------------------------------------
# _normalize_partition_sizes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec,total,mn", [
    (0, 8, 1),
    (4, 8, 1),
    (3, 12, 1),
    (5, 12, 1),
    (4, 11, 1),
    (8, 23, 1),
    (4, 8, 2),
    (3, 9, 1),
    (6, 6, 1),
    (0, 10, 1),
    ([5, 2, 5], 12, 1),
    ([4, 4, 4], 12, 1),
    ([6, 6], 12, 2),
])
def test_normalize_partition_sizes(spec, total, mn):
    py = ls.SchedulerConfig._normalize_partition_sizes(spec, total, "X", mn)
    cpp = cppls.SchedulerConfig._normalize_partition_sizes(spec, total, "X", mn)
    assert list(cpp) == list(py)


# ---------------------------------------------------------------------------
# get_partition_candidates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N", [
    (4, 8), (8, 4), (8, 8), (1, 10), (10, 1), (16, 4), (3, 7),
])
def test_get_partition_candidates(M, N):
    py = ls.SchedulerConfig.get_partition_candidates(
        _FakeTileInfo(M), _FakeTileInfo(N))
    cpp = [tuple(c) for c in cppls.SchedulerConfig.get_partition_candidates(M, N)]
    assert cpp == [tuple(c) for c in py]


@pytest.mark.parametrize("M,N", [(4, 8), (8, 4), (10, 1)])
def test_get_partition_candidates_shim_and_delegation(M, N):
    tiA, tiB = _FakeTileInfo(M), _FakeTileInfo(N)
    py = [tuple(c) for c in
          ls.SchedulerConfig.get_partition_candidates(tiA, tiB)]
    shim = cppls.get_partition_candidates(tiA, tiB)
    assert shim == py
    with cpp_delegation():
        delegated = [tuple(c) for c in
                     ls.SchedulerConfig.get_partition_candidates(tiA, tiB)]
    assert delegated == py


# ---------------------------------------------------------------------------
# Placement / op value type string formatting
# ---------------------------------------------------------------------------
def test_placement_str_parity():
    tr = (0, 2, 0, 8)
    py_tr = ls.MFMATileRange(*tr)
    cpp_tr = cppls.MFMATileRange(*tr)

    assert str(ls.MFMAPlacement(1, ls.MFMATileRange(0, 2, 0, 4),
                                ls.MFMATileRange(0, 2, 4, 8))) == \
        str(cppls.MFMAPlacement(1, cppls.MFMATileRange(0, 2, 0, 4),
                                cppls.MFMATileRange(0, 2, 4, 8)))

    for tensor in ("A", "B", "SA", "SB"):
        for mt in (0, 1, 2):
            assert str(ls.LRPlacement(tensor, mt, py_tr, 3, 1)) == \
                str(cppls.LRPlacement(tensor, mt, cpp_tr, 3, 1))
            assert str(ls.GRPlacement(tensor, mt, py_tr, 3, 1)) == \
                str(cppls.GRPlacement(tensor, mt, cpp_tr, 3, 1))


def test_op_str_parity():
    assert str(ls.WaitGRCounts(1, 0, 2, 0)) == str(cppls.WaitGRCounts(1, 0, 2, 0))
    assert str(ls.WaitGRCounts()) == str(cppls.WaitGRCounts())
    assert str(ls.WaitGRCounts(0, 3, 0, 4)) == str(cppls.WaitGRCounts(0, 3, 0, 4))

    counts_py = ls.WaitGRCounts(1, 2, 0, 0)
    counts_cpp = cppls.WaitGRCounts(1, 2, 0, 0)
    assert str(ls.WaitGROp(counts_py, True, True)) == \
        str(cppls.WaitGROp(counts_cpp, True, True))
    assert str(ls.WaitGROp(None, False, True)) == \
        str(cppls.WaitGROp(None, False, True))

    assert str(ls.WaitLROp(True)) == str(cppls.WaitLROp(True))
    assert str(ls.WaitLROp(False)) == str(cppls.WaitLROp(False))
    assert str(ls.SyncOp()) == str(cppls.SyncOp())
    assert str(ls.MaskKOp(3)) == str(cppls.MaskKOp(3))
    assert str(ls.LRIncOp("A")) == str(cppls.LRIncOp("A"))
    assert str(ls.GRIncOp("SB")) == str(cppls.GRIncOp("SB"))

    py_skip = ls.SkipOp("LoopCounter", 2, "NLL", False, "")
    cpp_skip = cppls.SkipOp("LoopCounter", 2, "NLL", False, "")
    assert str(py_skip) == str(cpp_skip)
    assert cpp_skip.tensor == py_skip.tensor


def test_op_kind_parity():
    pairs = [
        (ls.MFMAPlacement(0, ls.MFMATileRange(0, 1, 0, 1),
                          ls.MFMATileRange(0, 1, 0, 1)),
         cppls.MFMAPlacement(0, cppls.MFMATileRange(0, 1, 0, 1),
                             cppls.MFMATileRange(0, 1, 0, 1))),
        (ls.LRPlacement("A", 0, ls.MFMATileRange(0, 1, 0, 1), 0),
         cppls.LRPlacement("A", 0, cppls.MFMATileRange(0, 1, 0, 1), 0)),
        (ls.GRPlacement("A", 0, ls.MFMATileRange(0, 1, 0, 1), 0),
         cppls.GRPlacement("A", 0, cppls.MFMATileRange(0, 1, 0, 1), 0)),
        (ls.WaitGROp(), cppls.WaitGROp()),
        (ls.WaitLROp(), cppls.WaitLROp()),
        (ls.SyncOp(), cppls.SyncOp()),
        (ls.MaskKOp(), cppls.MaskKOp()),
        (ls.LRIncOp("A"), cppls.LRIncOp("A")),
        (ls.GRIncOp("A"), cppls.GRIncOp("A")),
        (ls.SkipOp(), cppls.SkipOp()),
    ]
    for py, cpp in pairs:
        assert cpp.kind == py.kind


# ---------------------------------------------------------------------------
# InlineModuleOp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["inline", "preloop", "tail"])
def test_inline_module_op_parity(label):
    assert str(ls.InlineModuleOp(label=label)) == \
        str(cppls.InlineModuleOp(label=label))
    assert cppls.InlineModuleOp(label=label).kind == \
        ls.InlineModuleOp(label=label).kind
    assert cppls.InlineModuleOp(label=label).label == \
        ls.InlineModuleOp(label=label).label


def test_inline_module_op_default_label():
    assert str(cppls.InlineModuleOp()) == str(ls.InlineModuleOp())
    assert cppls.InlineModuleOp().label == ls.InlineModuleOp().label


# ---------------------------------------------------------------------------
# Placement constructors used by the value-type tests below
# ---------------------------------------------------------------------------
def _lr(mod, tensor="A", part=0):
    return mod.LRPlacement(tensor, 0, mod.MFMATileRange(0, 1, 0, 1), 0, part)


def _gr(mod, tensor="A", part=0):
    return mod.GRPlacement(tensor, 0, mod.MFMATileRange(0, 1, 0, 1), 0, part)


def _mfma(mod):
    return mod.MFMAPlacement(0, mod.MFMATileRange(0, 1, 0, 1),
                             mod.MFMATileRange(0, 1, 0, 1))


# ---------------------------------------------------------------------------
# Pass-populated placement fields: default empty + round-trip
# ---------------------------------------------------------------------------
def test_placement_pass_fields_default_empty():
    for make in (_mfma, _lr, _gr):
        cpp, py = make(cppls), make(ls)
        assert list(cpp.deps) == list(py.deps) == []
        assert list(cpp.preOps) == list(py.preOps) == []
        assert list(cpp.postOps) == list(py.postOps) == []
    assert dict(_mfma(cppls).vgpr_tile_maps) == \
        dict(_mfma(ls).vgpr_tile_maps) == {}
    assert list(_lr(cppls).vgpr_tile_map) == list(_lr(ls).vgpr_tile_map) == []
    assert dict(cppls.MaskKOp(3).vgpr_tile_map) == \
        dict(ls.MaskKOp(3).vgpr_tile_map) == {}


def test_placement_pre_post_ops_roundtrip():
    lr = _lr(cppls)
    lr.preOps = [cppls.WaitGROp(cppls.WaitGRCounts(1, 0, 0, 0), False, True),
                 cppls.SyncOp()]
    lr.postOps = [cppls.LRIncOp("A")]
    assert [str(o) for o in lr.preOps] == ["wait_gr(A=1)", "sync"]
    assert [o.kind for o in lr.preOps] == ["wait_gr", "sync"]
    assert [str(o) for o in lr.postOps] == ["lr_inc(A)"]


def test_lr_vgpr_tile_map_roundtrip():
    lr = _lr(cppls)
    lr.vgpr_tile_map = [{0: 4, 1: 5}, {2: 6}]
    assert [dict(d) for d in lr.vgpr_tile_map] == [{0: 4, 1: 5}, {2: 6}]


def test_mfma_vgpr_tile_maps_roundtrip():
    mfma = _mfma(cppls)
    mfma.vgpr_tile_maps = {"A": [{0: 1}], "B": [{0: 2}, {1: 3}]}
    got = {k: [dict(d) for d in v] for k, v in mfma.vgpr_tile_maps.items()}
    assert got == {"A": [{0: 1}], "B": [{0: 2}, {1: 3}]}


# ---------------------------------------------------------------------------
# Dep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mt_offset", [0, -1, -2])
@pytest.mark.parametrize("kind", ["lr", "gr"])
def test_dep_parity(kind, mt_offset):
    make = _lr if kind == "lr" else _gr
    py = ls.Dep(make(ls, "B", 1), mt_offset)
    cpp = cppls.Dep(make(cppls, "B", 1), mt_offset)
    assert cpp.mt_offset == py.mt_offset == mt_offset
    assert cpp.ref.kind == py.ref.kind == kind
    assert cpp.ref.tensor == py.ref.tensor == "B"
    assert cpp.ref.partition == py.ref.partition == 1


def test_dep_default_mt_offset():
    assert cppls.Dep(_lr(cppls)).mt_offset == ls.Dep(_lr(ls)).mt_offset == 0


def test_placement_deps_roundtrip():
    gr = _gr(cppls)
    gr.deps = [cppls.Dep(_lr(cppls, "A"), -1)]
    assert len(gr.deps) == 1
    assert gr.deps[0].ref.kind == "lr"
    assert gr.deps[0].ref.tensor == "A"
    assert gr.deps[0].mt_offset == -1


# ---------------------------------------------------------------------------
# SubIterKSlot
# ---------------------------------------------------------------------------
def test_subiterk_slot_parity():
    py, cpp = ls.SubIterKSlot(2), cppls.SubIterKSlot(2)
    assert cpp.subIterK == py.subIterK == 2
    assert cpp.mfma is None and py.mfma is None
    assert list(cpp.lrs) == list(py.lrs) == []
    assert list(cpp.grs) == list(py.grs) == []

    cpp.mfma = _mfma(cppls)
    cpp.lrs = [_lr(cppls, "A"), _lr(cppls, "B")]
    cpp.grs = [_gr(cppls, "A")]
    assert cpp.mfma.kind == "mfma"
    assert [o.tensor for o in cpp.lrs] == ["A", "B"]
    assert [o.tensor for o in cpp.grs] == ["A"]


# ---------------------------------------------------------------------------
# EmittedModule
# ---------------------------------------------------------------------------
def test_emitted_module_optype_parity():
    sources = [
        (_mfma(ls), _mfma(cppls), "mfma"),
        (_lr(ls), _lr(cppls), "lr"),
        (_gr(ls), _gr(cppls), "gr"),
        (ls.WaitGROp(), cppls.WaitGROp(), "wait_gr"),
        (ls.SyncOp(), cppls.SyncOp(), "sync"),
        (ls.LRIncOp("A"), cppls.LRIncOp("A"), "lr_inc"),
        (ls.InlineModuleOp(label="x"), cppls.InlineModuleOp(label="x"),
         "inline"),
    ]
    for py_src, cpp_src, kind in sources:
        py = ls.EmittedModule(moduleId=7, before=3, source=py_src)
        cpp = cppls.EmittedModule(moduleId=7, before=3, source=cpp_src)
        assert cpp.opType == py.opType == kind
        assert cpp.moduleId == py.moduleId == 7
        assert cpp.before == py.before == 3
        assert cpp.source.kind == kind


def test_emitted_module_empty_source():
    py, cpp = ls.EmittedModule(), cppls.EmittedModule()
    assert cpp.opType == py.opType == ""
    assert cpp.moduleId == py.moduleId == -1
    assert cpp.before is None and py.before is None
    assert cpp.source is None


# ---------------------------------------------------------------------------
# Default-off behavior
# ---------------------------------------------------------------------------
def test_default_path_is_python_only():
    """With the env flag unset, delegation must be disabled by default."""
    import os
    if os.environ.get("TENSILE_WRITER_CPP", "").strip().lower() not in (
            "", "0", "false", "no", "off"):
        pytest.skip("TENSILE_WRITER_CPP is set; default-off behavior not under test")
    assert ls._USE_CPP is False
    assert ls._CPP is None


# ===========================================================================
# Writer-free pass pipeline (place_LRs through emit/build)
# ---------------------------------------------------------------------------
# The C++ ``LogicalScheduler`` ports the pure, data-only pass pipeline. It does
# NOT populate rocisa instructions, allocate writer VGPR pools, or emit
# Kernel.mainLoop control flow — those remain Python-only. The print_* helpers
# emit byte-identical output to the Python LogicalScheduler, so we compare the
# two implementations pass-by-pass on representative BF16/fp8 gfx950 reference
# configs across PGR=0/1/2.
# ===========================================================================

# Additional pass-pipeline configs exercising PGR=0/1/2 and the scale (fp8/fp4)
# path beyond the value-layer CONFIGS above. These mirror cases from
# test_SubtileBasedLogicalScheduler / test_SubtileBasedSchedulerRef.
_PASS_EXTRA_CONFIGS = {
    # fp8-style 256x256 single full partition (8x8), PGR=2 then PGR=1.
    "fp8_8x8_pgr2": dict(numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2,
                         lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), pgr=2),
    "fp8_8x8_pgr1": dict(numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2,
                         lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2), pgr=1),
    # Multi-partition BF16 256x384 with PGR=1 (offsetPartition=0).
    "bf16_256x384_n6_pgr1": dict(numMFMATilesM=8, numMFMATilesN=12, numSubIterK=2,
                                 lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
                                 partitionSizeN=6, pgr=1),
    # FP4 2x2 with scales, PGR=1.
    "fp4_2x2_pgr1": dict(numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2,
                         lrA=(1, 1), lrB=(1, 1), grA=(1, 2), grB=(1, 2),
                         lrSA=(2, 2), lrSB=(2, 2), grSA=(2, 2), grSB=(2, 2),
                         partitionSizeM=4, partitionSizeN=4, pgr=1),
}

# name -> kwargs spec for the pass-pipeline parity sweep.
_PASS_CONFIGS = {name: spec for name, (spec, _exp) in CONFIGS.items()}
_PASS_CONFIGS.update(_PASS_EXTRA_CONFIGS)

# print_* method -> terminal pass that must run first. Each pass auto-runs its
# prerequisites in both implementations, so running the terminal pass exercises
# the whole chain up to that point.
_PRINT_TO_PASS = {
    "print_lr": "place_LRs",
    "print_vgpr": "assign_vgpr_tiles",
    "print_gr": "place_GRs",
    "print_deps": "annotate_deps",
    "print_remove_deps": "remove_cross_deps",
    "print_group_lr_gr": "group_lr_gr",
    "print_emit": "emit",
}


def test_shim_reexports_pass_pipeline():
    """The Python shim must expose the C++ pass-pipeline scheduler class."""
    assert hasattr(cppls, "LogicalScheduler")
    assert "LogicalScheduler" in cppls.__all__


@pytest.mark.parametrize("name", list(_PASS_CONFIGS))
@pytest.mark.parametrize("print_method", list(_PRINT_TO_PASS))
def test_pass_pipeline_print_parity(name, print_method):
    """C++ pass output must match Python byte-for-byte, pass-by-pass."""
    spec = _PASS_CONFIGS[name]
    py_sched = ls.LogicalScheduler(_build(ls, spec))
    cpp_sched = cppls.LogicalScheduler(_build(cppls, spec))

    run = _PRINT_TO_PASS[print_method]
    getattr(py_sched, run)()
    getattr(cpp_sched, run)()

    py_out = getattr(py_sched, print_method)()
    cpp_out = getattr(cpp_sched, print_method)()
    assert cpp_out == py_out, (
        f"{name} / {print_method} mismatch\n"
        f"--- Python ---\n{py_out}\n--- C++ ---\n{cpp_out}"
    )


@pytest.mark.parametrize("name", list(_PASS_CONFIGS))
def test_pass_pipeline_vgpr_metadata_parity(name):
    """assign_vgpr_tiles scalar outputs (unroll factor / tile peaks) match."""
    spec = _PASS_CONFIGS[name]
    py_sched = ls.LogicalScheduler(_build(ls, spec))
    cpp_sched = cppls.LogicalScheduler(_build(cppls, spec))
    py_sched.assign_vgpr_tiles()
    cpp_sched.assign_vgpr_tiles()
    assert cpp_sched.needs_unrolling == py_sched.needs_unrolling
    assert cpp_sched.unroll_factor == py_sched.unroll_factor
    assert dict(cpp_sched.tile_peaks) == dict(py_sched.tile_peaks)


@pytest.mark.parametrize("name", list(_PASS_CONFIGS))
def test_pass_pipeline_build_matches_emit(name):
    """build() runs the full pipeline and yields the same emit output."""
    spec = _PASS_CONFIGS[name]
    py_sched = ls.LogicalScheduler(_build(ls, spec))
    cpp_sched = cppls.LogicalScheduler(_build(cppls, spec))
    py_sched.build()
    cpp_sched.build()
    assert cpp_sched.print_emit() == py_sched.print_emit()
