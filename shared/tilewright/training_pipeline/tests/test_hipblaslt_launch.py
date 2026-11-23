# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random
from collections import Counter

import pytest

from bench_fakes import LIBRARY_STEM, solution, write_logic
from lib import dat
from lib import evaluate as ev
from lib import features as fs
from lib import hipblaslt_launch as hl


def _plan(m, n, k, occupancy, attrs, context, arch="gfx950", batch=1, version=None):
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=occupancy)
    attrs = dict({"work_group_x": 256, "work_group_y": 1, "work_group_z": 1}, **attrs)
    p = hl.launch_plan(prob, cfg, attrs, context, 256, arch, version=version)
    return (p.dynamic_queue, p.grid_wgs, p.partial_reduction)


HYBRID = {
    "tile_processing_strategy": 2,
    "work_assignment": 2,
    "workspace_size_per_elem_c": 4,
}


def test_hybrid_thresholds():
    sm224 = fs.ExecutionContext(cu_budget=224)
    assert _plan(128 * 480, 128, 512, 2, HYBRID, sm224) == (False, 320, "tree")
    assert _plan(128 * 481, 128, 512, 2, HYBRID, sm224) == (True, 481, "none")
    assert _plan(4096, 4096, 512, 3, HYBRID, sm224) == (True, 768, "none")
    assert _plan(4096, 4096, 512, 4, HYBRID, sm224) == (False, 1024, "none")
    assert _plan(128 * 1883, 128, 512, 4, HYBRID, sm224) == (False, 1883, "none")
    assert _plan(128 * 1884, 128, 512, 4, HYBRID, sm224) == (True, 768, "none")


def test_sm_target_and_schedules():
    def plan(budget, schedule="default", arch="gfx950"):
        ctx = fs.ExecutionContext(cu_budget=budget, schedule=schedule)
        return _plan(4096, 4096, 512, 2, HYBRID, ctx, arch)

    assert plan(256) == (False, 512, "none")
    assert plan(224) == plan(240) == plan(255) == plan(31) == (True, 512, "none")
    assert plan(0) == (False, 512, "none")
    assert plan(224, "auto") == (True, 512, "none")
    assert plan(0, "auto") == (False, 512, "none")
    assert plan(0, "dynamic") == (True, 512, "none")
    assert plan(224, arch="gfx942") == (False, 1024, "none")
    assert plan(224, "dynamic", "gfx942") == (True, 512, "none")
    assert hl.hybrid_runtime_choice("gfx950:xnack-")
    assert not hl.hybrid_runtime_choice("gfx9500")


def test_the_sm_target_is_the_cu_budget_as_it_is():
    """hipBLASLt takes a CU budget as its SM target as it is, every CU it
    names: 300 tiles of 300 K iterations spread over 100 CUs whole, and
    over 96 CUs on a streaming grid with partial tiles."""
    assert hl.SM_TARGET_GRANULE == 1
    prob = (128 * 300, 128, 64 * 300, 1, HYBRID)
    assert _plan(*prob, fs.ExecutionContext(cu_budget=100)) == (False, 100, "none")
    assert _plan(*prob, fs.ExecutionContext(cu_budget=96)) == (False, 86, "tree")


def test_stream_k_grids():
    excl = fs.EXCLUSIVE
    assert _plan(256, 256, 64 * 64, 1, HYBRID, excl) == (False, 32, "parallel")
    assert _plan(256, 256, 64 * 32, 1, HYBRID, excl) == (False, 16, "tree")
    atomic = dict(HYBRID, stream_k_atomic=1)
    assert _plan(256, 256, 64 * 32, 1, atomic, excl) == (False, 16, "atomic")
    dp = {"tile_processing_strategy": 1, "work_assignment": 0}
    assert _plan(4096, 4096, 512, 2, dp, excl) == (False, 512, "none")
    queue = dict(HYBRID, work_assignment=1)
    assert _plan(4096, 4096, 512, 5, queue, excl) == (True, 768, "none")
    assert _plan(4096, 4096, 512, 2, {}, excl) == (False, 1024, "none")
    huge = dict(HYBRID, work_group_x=1024, work_group_y=1024)
    assert _plan(128 * 300, 128, 64 * 300, 1, HYBRID, excl) == (False, 200, "tree")
    assert _plan(128 * 300, 128, 64 * 300, 1, huge, excl) == (False, 300, "none")
    assert _plan(128 * 4500, 128, 512, 9, huge, excl) == (False, 2048, "tree")
    assert _plan(1024, 1024, 512, 2, HYBRID, excl, batch=16) == (False, 512, "none")


# (m, n, k, batch, mt_m, mt_n, mt_k, work_group, cu_budget, schedule) of
# occupancy-8 Hybrid gfx950 kernels whose dynamic launch overflows the tree
# fix-up and is clamped to the flags after the per-XCD queue counters.
QUEUE_CLAMPED = [
    (36450, 36788, 31014, 16, 16, 16, 32, (16, 4, 1), 224, "dynamic"),
    (14498, 19072, 8121, 32, 16, 16, 64, (16, 4, 2), 96, "auto"),
    (30965, 13895, 63840, 64, 16, 16, 64, (16, 4, 2), 96, "default"),
    (12387, 60136, 977, 32, 16, 32, 64, (16, 8, 1), 224, "dynamic"),
    (2237, 60189, 2130, 64, 16, 16, 64, (16, 4, 2), 96, "default"),
    (16845, 22929, 6, 64, 16, 32, 64, (16, 8, 1), 64, "auto"),
    (56979, 7497, 10, 64, 16, 64, 32, (16, 16, 1), 224, "auto"),
]


@pytest.mark.parametrize("row", QUEUE_CLAMPED)
def test_dynamic_queues_take_the_head_of_the_flag_region(row):
    m, n, k, batch, mt_m, mt_n, mt_k, wg, budget, schedule = row
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=mt_m, mt_n=mt_n, mt_k=mt_k, occupancy=8)
    attrs = dict(HYBRID, work_group_x=wg[0], work_group_y=wg[1], work_group_z=wg[2])
    ctx = fs.ExecutionContext(cu_budget=budget, schedule=schedule)
    flags = hl.STREAM_K_FLAG_ELEMENTS - 8 * hl.CACHE_LINE_BYTES // 4
    assert hl.launch_plan(prob, cfg, attrs, ctx, 256, "gfx950") == (
        fs.LaunchPlan(True, flags, "none")
    )


def test_flag_region_clamp():
    excl = fs.EXCLUSIVE
    dyn = fs.ExecutionContext(schedule="dynamic")
    wg = {"work_group_x": 1024, "work_group_y": 1024}
    hybrid, queue = dict(HYBRID, **wg), dict(HYBRID, work_assignment=1, **wg)
    assert _plan(128 * 4500, 128, 512, 9, hybrid, excl) == (False, 2048, "tree")
    assert _plan(128 * 4500, 128, 512, 9, hybrid, dyn) == (True, 1792, "none")
    assert _plan(128 * 4500, 128, 512, 9, queue, excl) == (True, 1792, "none")
    assert _plan(128 * 4500, 128, 512, 9, queue, excl, "gfx942") == (
        True,
        1792,
        "none",
    )
    assert _plan(128 * 4500, 128, 512, 9, queue, excl, "gfx90a") == (
        True,
        2048 - 32,
        "none",
    )
    assert _plan(128 * 4500, 128, 512, 9, hybrid, excl, "gfx1250") == (
        False,
        2048,
        "tree",
    )
    small = dict(HYBRID, work_assignment=1)
    assert _plan(4096, 4096, 512, 2, small, excl, "gfx1250") == (True, 512, "none")
    with pytest.raises(ValueError, match="gfx1250"):
        _plan(4096, 4096, 512, 5, small, excl, "gfx1250")


# An architecture without a fixed queue count, made up for these tests; the
# XCD counts given with it are made up too.
UNLISTED = "gfx9990"


def test_the_device_xcd_count_sets_the_queues_where_the_arch_keeps_none():
    assert hl.queue_count("gfx950") == hl.queue_count("gfx950:xnack-", 3) == 8
    for arch in ("gfx1250", "gfx1250-strict", UNLISTED):
        assert hl.queue_count(arch) is None
    assert hl.queue_count(UNLISTED, 3) == hl.queue_count(UNLISTED + ":xnack-", 3) == 3
    prob = dict(m=128 * 4500, n=128, k=512, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=9)
    attrs = dict(HYBRID, work_assignment=1, work_group_x=1024, work_group_y=1024)
    attrs["work_group_z"] = 1
    excl = fs.EXCLUSIVE
    for n in (1, 3):
        got = hl.launch_plan(prob, cfg, attrs, excl, 256, UNLISTED, n)
        flags = hl.STREAM_K_FLAG_ELEMENTS - n * hl.CACHE_LINE_BYTES // 4
        assert got == fs.LaunchPlan(True, flags, "none")
    assert hl.launch_plan(prob, cfg, attrs, excl, 256, "gfx90a", 3) == (
        hl.launch_plan(prob, cfg, attrs, excl, 256, UNLISTED, 1)
    )
    for arch in (UNLISTED, "gfx1250-strict"):
        with pytest.raises(ValueError, match="hardware.num_xcd"):
            hl.launch_plan(prob, cfg, attrs, excl, 256, arch)
    source = hl.HipblasltLaunch(UNLISTED, num_xcd=3)
    assert repr(source) == f"HipblasltLaunch('{UNLISTED}', num_xcd=3)"
    assert source.id == hl.HipblasltLaunch(UNLISTED).id
    assert source(prob, cfg, attrs, excl, 256) == (
        hl.launch_plan(prob, cfg, attrs, excl, 256, UNLISTED, 3)
    )
    stem = f"TensileLibrary_X_Contraction_l_Alik_Bljk_Cijk_Dijk_{UNLISTED}"
    assert hl.HipblasltLaunch.for_library(stem, 3).num_xcd == 3
    assert hl.HipblasltLaunch.for_library(stem).num_xcd is None


QUEUE = dict(HYBRID, work_assignment=1)


@pytest.mark.parametrize(
    "attrs, arch, schedules, reduced, takes",
    [
        ({}, "gfx1250", ("default", "dynamic"), True, False),
        ({"tile_processing_strategy": 1}, "gfx1250", ("dynamic",), True, False),
        (dict(QUEUE, work_assignment=0), "gfx1250", ("dynamic",), True, False),
        (QUEUE, "gfx1250", ("default",), False, True),
        (dict(QUEUE, stream_k_atomic=1), "gfx1250", ("default",), True, False),
        (HYBRID, "gfx1250", ("default",), True, False),
        (HYBRID, "gfx1250", ("default", "auto"), True, False),
        (HYBRID, "gfx1250", ("default", "dynamic"), False, True),
        (HYBRID, "gfx950", ("default",), False, False),
        (HYBRID, "gfx950", ("default",), True, True),
    ],
)
def test_which_kernels_take_a_dynamic_queue(attrs, arch, schedules, reduced, takes):
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    assert hl.takes_dynamic_queue(cfg, attrs, arch, schedules, reduced) is takes


# A CU count no device of these tests has.
MADE_UP_CUS = 300


def test_a_kernel_that_takes_no_dynamic_queue_needs_no_queue_count():
    rng = random.Random(7)
    kernels = [
        dict(HYBRID, stream_k_atomic=a, work_assignment=w, work_group_x=wg)
        for a in (0, 1)
        for w in (0, 1, 2)
        for wg in (64, 256, 1024)
    ]
    for schedules, reduced in (
        (("default",), False),
        (("default",), True),
        (("default", "auto"), True),
        (("default", "dynamic"), False),
    ):
        budgets = (0, 64, 128, 224) if reduced else (0,)
        for attrs in kernels:
            attrs = dict(attrs, work_group_y=1, work_group_z=1)
            for occupancy in (1, 2, 4, 8):
                cfg = dict(mt_m=16, mt_n=16, mt_k=64, occupancy=occupancy)
                if hl.takes_dynamic_queue(cfg, attrs, "gfx1250", schedules, reduced):
                    continue
                for _ in range(40):
                    prob = dict(
                        m=rng.randint(1, 1 << 16),
                        n=rng.randint(1, 1 << 16),
                        k=rng.randint(1, 1 << 16),
                        batch=rng.choice((1, 1, 4)),
                        mi_dtype="bfloat16",
                    )
                    ctx = fs.ExecutionContext(
                        cu_budget=rng.choice(budgets), schedule=rng.choice(schedules)
                    )
                    hl.launch_plan(prob, cfg, attrs, ctx, MADE_UP_CUS, "gfx1250")


@pytest.mark.parametrize("assign", [1, 2])
def test_data_parallel_kernels_take_a_static_grid_only(assign):
    dp = {"tile_processing_strategy": 1, "work_assignment": assign}
    with pytest.raises(ValueError, match="StaticGrid"):
        _plan(4096, 4096, 512, 2, dp, fs.EXCLUSIVE)


def _clustered(strategy, x=None, y=None, **attrs):
    attrs["tile_processing_strategy"] = strategy
    for name, value in (("cluster_dim_x", x), ("cluster_dim_y", y)):
        if value is not None:
            attrs[name] = value
    return attrs


@pytest.mark.parametrize(
    "attrs, clustered",
    [
        ({}, False),
        (_clustered(2), False),
        (_clustered(2, 1, 1), False),
        (_clustered(2, 2, 1), True),
        (_clustered(2, 1, 4), True),
        (_clustered(2, 4, work_assignment=1), True),
        (_clustered(2, 2, work_assignment=2), True),
        (_clustered(1), False),
        (_clustered(1, 1, 1), False),
        (_clustered(1, 2, 1), True),
        (_clustered(1, 2, 4), True),
        (_clustered(1, 1, 2), True),
        (_clustered(0, 4, 1), False),
        (_clustered(0, 2, 4), False),
    ],
)
def test_persistent_kernels_launched_in_clusters_are_outside_the_rule(attrs, clustered):
    """TensileLite plans a persistent kernel, DataParallel or Stream-K,
    whose ClusterDim spans more than one workgroup with its own cluster
    rules, whatever its assignment; a non-persistent kernel launches one
    workgroup per tile with or without a cluster."""
    assert hl.persistent_cluster_kernel(attrs) is clustered
    assert hl.persistent_cluster_kernel(None) is False


def test_cluster_attributes_are_kernel_attributes():
    assert set(hl.CLUSTER_ATTRIBUTES) <= set(dat.ATTRIBUTE_NAMES)
    for strategy in ("StreamK", "DataParallel"):
        sol = solution(1, 0, tileProcessingStrategy=strategy, clusterDim=[2, 1, 1])
        assert hl.persistent_cluster_kernel(dat.kernel_dat_info(sol)["attributes"])
        sol = solution(2, 0, tileProcessingStrategy=strategy)
        assert not hl.persistent_cluster_kernel(dat.kernel_dat_info(sol)["attributes"])
    sol = solution(3, 0, clusterDim=[2, 1, 1])
    assert not hl.persistent_cluster_kernel(dat.kernel_dat_info(sol)["attributes"])


def test_launch_arch_is_the_compiler_target_of_the_library_suffix():
    stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk"
    assert hl.launch_arch(stem + "_gfx950") == "gfx950"
    assert hl.launch_arch(stem + "_CU256_ID75a8_gfx950") == "gfx950"
    assert hl.launch_arch(f"/lib/gfx950/{stem}_gfx950.dat.zlib") == "gfx950"
    assert hl.launch_arch(stem + "_gfx1250") == "gfx1250"
    assert hl.launch_arch(stem + "_gfx1250-strict") == "gfx1250"
    for bad in ("TensileLibrary", "TensileLibrary_"):
        with pytest.raises(ValueError):
            hl.launch_arch(bad)
    ids = {
        hl.HipblasltLaunch.for_library(stem + "_gfx1250").id,
        hl.HipblasltLaunch.for_library(stem + "_gfx1250-strict").id,
        hl.HipblasltLaunch("gfx1250:xnack+").id,
    }
    assert ids == {f"hipblaslt:gfx1250:{hl.rule_version('gfx1250')}"}


def test_each_arch_trains_on_its_rule_version():
    """Every architecture trains on RULE_VERSION but gfx1250, whose pools hold
    kernels of the spatial-cluster layout and which trains on version 4:
    RULE_1_ARCHS holds none. The id names the version."""
    assert hl.RULE_VERSION == 3 and hl.RULE_VERSIONS == (1, 2, 3, 4)
    assert hl.RULE_1_ARCHS == () and hl.RULE_4_ARCHS == ("gfx1250",)
    for arch in ("gfx950", "gfx950:xnack-", "gfx942", "gfx90a", "gfx1201"):
        assert hl.rule_version(arch) == hl.RULE_VERSION, arch
    for arch in ("gfx1250", "gfx1250-strict", "gfx1250:xnack+"):
        assert hl.rule_version(arch) == 4, arch
    stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk"
    gfx950 = hl.HipblasltLaunch.for_library(stem + "_Cijk_Dijk_ID75a0_gfx950")
    assert (gfx950.version, gfx950.id) == (3, "hipblaslt:gfx950:3")
    assert repr(gfx950) == "HipblasltLaunch('gfx950')"
    old = hl.HipblasltLaunch("gfx950", 8, version=1)
    assert old.id == "hipblaslt:gfx950:1"
    assert repr(old) == "HipblasltLaunch('gfx950', num_xcd=8, version=1)"
    gfx1250 = hl.HipblasltLaunch.for_library(stem + "_Cijk_Dijk_CU96_gfx1250")
    assert (gfx1250.version, gfx1250.id) == (4, "hipblaslt:gfx1250:4")
    assert repr(gfx1250) == "HipblasltLaunch('gfx1250')"
    assert hl.HipblasltLaunch("gfx1250", version=1).id == "hipblaslt:gfx1250:1"
    assert hl.HipblasltLaunch("gfx1250", version=3).id == "hipblaslt:gfx1250:3"
    assert hl.HipblasltLaunch("gfx950", version=4).id == "hipblaslt:gfx950:4"
    for bad in (0, 5):
        with pytest.raises(ValueError, match="rule version"):
            hl.HipblasltLaunch("gfx950", version=bad)
        with pytest.raises(ValueError, match="rule version"):
            _plan(4096, 4096, 512, 2, HYBRID, fs.EXCLUSIVE, version=bad)


@pytest.mark.parametrize(
    "arch, recorded, version",
    [
        ("gfx950", "hipblaslt:gfx950:1", 1),
        ("gfx950", "hipblaslt:gfx950:2", 2),
        ("gfx950", "", 3),
        ("gfx950", None, 3),
        ("gfx950", "hipblaslt:gfx950:3", 3),
        ("gfx950", "hipblaslt:gfx950:4", 4),
        ("gfx950", "hipblaslt:gfx950:5", 3),
        ("gfx950", "hipblaslt:gfx950:03", 3),
        ("gfx950", "hipblaslt:gfx942:1", 3),
        ("gfx950", "other:gfx950:1", 3),
        ("gfx1250", "hipblaslt:gfx1250:1", 1),
        ("gfx1250", "hipblaslt:gfx1250:2", 2),
        ("gfx1250", "hipblaslt:gfx1250:3", 3),
        ("gfx1250", "hipblaslt:gfx1250:4", 4),
        ("gfx1250", "hipblaslt:gfx1250:5", 4),
        ("gfx1250", "", 4),
        ("gfx1250-strict", "hipblaslt:gfx1250:1", 1),
        ("gfx942", "hipblaslt:gfx942:1", 1),
    ],
)
def test_a_model_ranks_with_the_rule_version_it_records(arch, recorded, version):
    """A model ranks with the rule version it records when that is one of
    RULE_VERSIONS, on any architecture; any other model ranks with its
    architecture's version, under which the engine refuses one that records
    another launch model."""
    assert hl.served_rule_version(arch, recorded) == version
    launch = hl.HipblasltLaunch.for_model(arch, recorded, 8)
    assert (launch.version, launch.num_xcd) == (version, 8)
    assert launch.id == f"hipblaslt:{hl.canonical_arch(arch)}:{version}"
    if recorded and recorded.rsplit(":", 1)[1] == str(version):
        assert launch.id == recorded


SK_STATIC = dict(HYBRID, work_assignment=0)


@pytest.mark.parametrize(
    "m, n, k, occupancy, budget, attrs, v1, v2",
    [
        # 1600 tiles on 256 CUs: version 1 takes one workgroup per tile in its
        # window of 2 to 32 tiles per CU, version 2 keeps the streaming grid
        # of up to two workgroups per CU, which splits no tile.
        (5120, 5120, 64, 1, 0, SK_STATIC, (False, 1600, "none"), (False, 246, "none")),
        # 2400 tiles on 64 CUs at occupancy 4: a streaming grid of more than
        # two workgroups per CU becomes one workgroup per tile in version 2;
        # 2400 tiles are past version 1's window.
        (7680, 5120, 64, 4, 64, SK_STATIC, (False, 253, "tree"), (False, 2400, "none")),
        # One grid in both, which does not divide the tiles: the tiles of one
        # K iteration have no partial results to reduce in version 2.
        (7680, 5120, 64, 1, 64, SK_STATIC, (False, 64, "tree"), (False, 64, "none")),
        (7680, 5120, 17, 1, 64, dict(SK_STATIC, stream_k_atomic=1),
         (False, 64, "atomic"), (False, 64, "none")),
        # Two K iterations: the rule is unchanged.
        (7680, 5120, 128, 1, 64, SK_STATIC, (False, 2400, "none"), (False, 2400, "none")),
        (128 * 300, 128, 64 * 300, 1, 96, SK_STATIC, (False, 86, "tree"),
         (False, 86, "tree")),
        # Version 2 launches one workgroup per tile exactly when the streaming
        # grid has more than two workgroups per usable CU: not at two, ...
        (128 * 1024, 128, 64, 2, 0, SK_STATIC, (False, 1024, "none"),
         (False, 512, "none")),
        # ... at three, past version 1's window of 2 to 32 tiles per CU ...
        (128 * 9216, 128, 64, 3, 0, SK_STATIC, (False, 768, "none"),
         (False, 9216, "none")),
        # ... and alike for a DataParallel kernel and a Hybrid one on a
        # static grid, whose grids select_grid_size picks too.
        (128 * 1100, 128, 64, 2, 0, dict(SK_STATIC, tile_processing_strategy=1),
         (False, 1100, "none"), (False, 440, "none")),
        (128 * 1100, 128, 64, 2, 0, HYBRID, (False, 1100, "none"), (False, 440, "none")),
    ],
)  # fmt: skip
def test_one_k_iteration_launches_by_rule_version(
    m, n, k, occupancy, budget, attrs, v1, v2
):
    ctx = fs.ExecutionContext(cu_budget=budget)
    assert _plan(m, n, k, occupancy, attrs, ctx, version=1) == v1
    assert _plan(m, n, k, occupancy, attrs, ctx, version=2) == v2
    assert _plan(m, n, k, occupancy, attrs, ctx) == v2
    assert _plan(m, n, k, occupancy, attrs, ctx, "gfx1250") == v2


def test_a_batch_of_one_k_iteration_keeps_its_grid_and_splits_no_tile():
    """A batch skips select_grid_size's one-workgroup-per-tile correction in
    both versions; version 2 reports no partial reduction for its one K
    iteration per tile either."""
    ctx = fs.EXCLUSIVE
    args = (128 * 1100, 128, 64, 2, SK_STATIC, ctx)
    assert _plan(*args, batch=2, version=1) == (False, 489, "tree")
    assert _plan(*args, batch=2) == _plan(*args, "gfx1250", 2) == (False, 489, "none")


# Thin GEMMs with one K iteration per tile, each with a static Stream-K kernel
# of 128 threads and depth 64 on a device of the CU96 library: (m, n, k,
# mt_m, mt_n, occupancy, plan), its plan under rule version 2, which the
# models of that library train on.
THIN_ONE_K_ITERATION = [
    (369, 198112, 45, 64, 96, 6, (False, 12384, "none")),
    (369, 198112, 45, 128, 128, 3, (False, 4644, "none")),
    (369, 198112, 45, 128, 128, 1, (False, 96, "none")),
    (64, 468021, 3, 64, 96, 6, (False, 4876, "none")),
    (64, 468021, 3, 64, 64, 6, (False, 7313, "none")),
    (64, 468021, 3, 64, 768, 1, (False, 94, "none")),
    (64, 468021, 3, 64, 384, 2, (False, 188, "none")),
    (64, 456960, 48, 64, 640, 1, (False, 95, "none")),
    (64, 456960, 48, 64, 64, 6, (False, 7140, "none")),
    (544, 73728, 30, 128, 128, 1, (False, 96, "none")),
    (544, 73728, 30, 32, 128, 6, (False, 9792, "none")),
    (544, 73728, 30, 80, 64, 6, (False, 8064, "none")),
    (83712, 3, 48, 256, 16, 1, (False, 93, "none")),
]


@pytest.mark.parametrize("row", THIN_ONE_K_ITERATION)
def test_thin_one_k_iteration_gemms_of_gfx1250(row):
    m, n, k, mt_m, mt_n, occupancy, plan = row
    prob = dict(m=m, n=n, k=k, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=mt_m, mt_n=mt_n, mt_k=64, occupancy=occupancy)
    attrs = dict(SK_STATIC, work_group_x=128, work_group_y=1, work_group_z=1)
    p = hl.launch_plan(prob, cfg, attrs, fs.EXCLUSIVE, 96, "gfx1250")
    assert (p.dynamic_queue, p.grid_wgs, p.partial_reduction) == plan


def test_rule_versions_differ_only_for_one_k_iteration_static_launches():
    """Rule versions 1 and 2 plan alike but for a static persistent launch
    (Stream-K, or DataParallel, whose grid select_grid_size picks too) with
    one K iteration per tile and more tiles than usable CUs, where they differ
    for some kernels and problems: in the grid for batch 1, and otherwise
    only in the partial reduction version 1 reported."""
    rng = random.Random(41)
    differ = batch_1 = 0
    for _ in range(3000):
        cfg, attrs = _random_kernel(rng)
        prob = _random_problem(rng)
        if rng.random() < 0.5:
            prob["k"] = rng.choice([1, 7, cfg["mt_k"]])
            prob["m"] = rng.choice([4096, 128 * 481, 65536])
        ctx = fs.ExecutionContext(
            cu_budget=rng.choice([0, 31, 64, 100, 224, 256]),
            schedule=rng.choice(fs.SCHEDULES),
        )
        n_cu = rng.choice([64, 256, 304])
        v1 = hl.launch_plan(prob, cfg, attrs, ctx, n_cu, "gfx950", version=1)
        v2 = hl.launch_plan(prob, cfg, attrs, ctx, n_cu, "gfx950", version=2)
        if v1 == v2:
            continue
        differ += 1
        budget = int(ctx.cu_budget)
        usable = budget if 0 < budget < n_cu else n_cu
        assert attrs.get("tile_processing_strategy") in (1, 2), (cfg, attrs, prob)
        assert not v1.dynamic_queue and not v2.dynamic_queue
        assert int(prob["k"]) <= int(cfg["mt_k"])
        assert fs.output_tiles(prob, cfg) > usable
        if int(prob["batch"]) > 1:
            assert v1.grid_wgs == v2.grid_wgs and v2.partial_reduction == "none"
        else:
            batch_1 += 1
    assert differ > 20 and batch_1 > 10


def test_non_persistent_kernels_launch_one_workgroup_per_tile():
    for attrs in ({}, {"tile_processing_strategy": 0, "work_assignment": 2}):
        assert _plan(4096, 1000, 512, 2, attrs, fs.EXCLUSIVE) == (False, 256, "none")
    prob = dict(m=4096, n=4096, k=512, batch=1)
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    assert hl.launch_plan(prob, cfg, None, fs.EXCLUSIVE, 0, "gfx950") == (
        fs.LaunchPlan(False, 1024, "none")
    )


def test_hipblaslt_launch_is_a_launch_source():
    source = hl.HipblasltLaunch("gfx942")
    assert repr(source) == "HipblasltLaunch('gfx942')"
    assert isinstance(hl.RULE_VERSION, int) and hl.RULE_VERSION >= 1
    assert source.id == f"hipblaslt:gfx942:{hl.RULE_VERSION}"
    assert hl.HipblasltLaunch("gfx950").id != source.id
    prob = dict(m=4096, n=4096, k=512, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    ctx = fs.ExecutionContext(cu_budget=224)
    assert source(prob, cfg, HYBRID, ctx, 256) == hl.launch_plan(
        prob, cfg, HYBRID, ctx, 256, "gfx942"
    )
    assert fs.Catalog("v3", (), source).launch is source
    assert not set(hl.LAUNCH_ATTRIBUTES) & {a.name for a in fs.DEFAULT_ATTRIBUTE_SPEC}


@pytest.mark.parametrize("name", hl.LAUNCH_ATTRIBUTES)
def test_launch_attributes_default_to_zero(name):
    prob = dict(m=4096, n=4096, k=512, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    attrs = dict(HYBRID, work_group_x=256, work_group_y=1, work_group_z=1)
    attrs[name] = 0
    without = {k: v for k, v in attrs.items() if k != name}
    assert hl.launch_plan(prob, cfg, attrs, fs.EXCLUSIVE, 256, "gfx950") == (
        hl.launch_plan(prob, cfg, without, fs.EXCLUSIVE, 256, "gfx950")
    )


SIGNATURE = ("mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k")


def _random_kernel(rng):
    cfg = dict(
        mt_m=rng.choice([16, 64, 128, 256]),
        mt_n=rng.choice([16, 64, 128, 256]),
        mt_k=rng.choice([16, 32, 64, 128]),
        mi_m=16, mi_n=16, mi_k=32, cache_hints_a=0, cache_hints_b=0,
        occupancy=rng.choice([-1, 0, 1, 2, 3, 4, 5]),
        grvw_a=rng.choice([1, 4, 8]), grvw_b=rng.choice([1, 4, 8]),
        gwvw_d=rng.choice([1, 4]),
    )  # fmt: skip
    attrs = {"unrelated": rng.choice([0, 1, 9])}
    for name, values in (
        ("tile_processing_strategy", [0, 1, 2, 2, 3]),
        ("work_assignment", [0, 1, 2]),
        ("stream_k_atomic", [0, 1]),
        ("work_group_x", [64, 128, 256]),
        ("work_group_y", [1, 2, 4]),
        ("work_group_z", [1, 2]),
        ("hand_written", [0, 1]),
    ):
        if rng.random() < 0.9:
            attrs[name] = rng.choice(values)
    return cfg, _loadable(attrs)


def _loadable(attrs):
    """`attrs` as TensileLite loads them: a DataParallel kernel takes the
    static grid only, and a Stream-K kernel carries the element size of its
    partial tiles."""
    if attrs.get("tile_processing_strategy") == 1:
        attrs["work_assignment"] = 0
    if attrs.get("tile_processing_strategy") == 2:
        attrs.setdefault("workspace_size_per_elem_c", 4)
    return attrs


def _variant(rng, cfg, attrs):
    """`cfg`/`attrs` with the signature kept and anything else perhaps
    changed, launch parameters included."""
    other_cfg, other_attrs = _random_kernel(rng)
    other_cfg.update({k: cfg[k] for k in SIGNATURE})
    other_cfg.update(
        cache_hints_a=cfg["cache_hints_a"], cache_hints_b=cfg["cache_hints_b"]
    )
    if rng.random() < 0.6:
        other_cfg["occupancy"] = cfg["occupancy"]
        for name in hl.LAUNCH_ATTRIBUTES:
            if rng.random() < 0.8:
                other_attrs.pop(name, None)
                if name in attrs:
                    other_attrs[name] = attrs[name]
    return other_cfg, _loadable(other_attrs)


def _random_problem(rng):
    return dict(
        m=rng.choice([1, 64, 100, 1024, 4096, 128 * 481, 65536]),
        n=rng.choice([1, 64, 128, 1000, 4096]),
        k=rng.choice([1, 64, 512, 4096, 64 * 300, 64 * 65536]),
        batch=rng.choice([1, 1, 2, 16]),
        mi_dtype=rng.choice(["bfloat16", "float", "double"]),
    )


def test_equal_launch_keys_launch_alike():
    rng = random.Random(17)
    equal = differing = 0
    for _ in range(4000):
        cfg, attrs = _random_kernel(rng)
        other_cfg, other_attrs = _variant(rng, cfg, attrs)
        same = hl.launch_key(cfg, attrs) == hl.launch_key(other_cfg, other_attrs)
        plans = []
        for _ in range(6):
            prob = _random_problem(rng)
            ctx = fs.ExecutionContext(
                cu_budget=rng.choice([0, 31, 64, 200, 224, 255, 256, 999]),
                schedule=rng.choice(fs.SCHEDULES),
            )
            n_cu = rng.choice([0, 64, 256, 304])
            arch = rng.choice(["gfx950", "gfx942", "gfx950:xnack-"])
            a = hl.launch_plan(prob, cfg, attrs, ctx, n_cu, arch)
            b = hl.launch_plan(prob, other_cfg, other_attrs, ctx, n_cu, arch)
            plans.append(a == b)
        if same:
            equal += 1
            assert all(plans), (cfg, attrs, other_cfg, other_attrs)
        elif not all(plans):
            differing += 1
    assert equal > 1000 and differing > 500


def test_launch_keys_merge_what_the_rule_cannot_tell_apart():
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    plain = {"tile_processing_strategy": 0, "work_assignment": 2}
    assert hl.launch_key(cfg, plain) == hl.launch_key(dict(cfg, occupancy=4), {})
    sk = dict(HYBRID, work_group_x=256, work_group_y=1, work_group_z=1)
    assert hl.launch_key(
        cfg, dict(sk, work_group_x=128, work_group_y=2)
    ) == hl.launch_key(cfg, sk)
    assert hl.launch_key(dict(cfg, occupancy=0), sk) == hl.launch_key(
        dict(cfg, occupancy=1), sk
    )
    dp = {"tile_processing_strategy": 1, "work_assignment": 0}
    assert hl.launch_key(cfg, dict(dp, stream_k_atomic=1)) == hl.launch_key(cfg, dp)
    assert hl.launch_key(cfg, dict(sk, stream_k_atomic=1)) != hl.launch_key(cfg, sk)
    assert hl.launch_key(dict(cfg, occupancy=3), sk) != hl.launch_key(cfg, sk)
    source = hl.HipblasltLaunch("gfx950")
    assert source.launch_key(cfg, sk) == hl.launch_key(cfg, sk)
    assert 0 <= hl.launch_key(cfg, sk) < 2**64


class _Strict(dict):
    """A mapping that fails on reading any key outside `allowed`."""

    def __init__(self, data, allowed):
        super().__init__(data)
        self.allowed = set(allowed)

    def _check(self, key):
        if key not in self.allowed:
            raise AssertionError(f"launch_plan read {key!r} outside its launch key")

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._check(key)
        return super().__contains__(key)

    def __iter__(self):
        raise AssertionError("launch_plan iterated a kernel's parameters")

    def items(self):
        raise AssertionError("launch_plan iterated a kernel's parameters")

    keys = values = items


def test_launch_plan_reads_the_kernel_only_through_launch_inputs(monkeypatch):
    """Everything launch_plan reads of a kernel beyond the signature's tile
    sizes must pass through `launch_inputs`, which the launch key covers."""
    real = hl.launch_inputs
    rng = random.Random(3)
    cases = [
        (
            _random_problem(rng),
            fs.ExecutionContext(
                cu_budget=rng.choice([0, 64, 224]), schedule=rng.choice(fs.SCHEDULES)
            ),
            rng.choice([0, 256]),
        )
        for _ in range(8)
    ]
    for _ in range(500):
        cfg, attrs = _random_kernel(rng)
        expect = [hl.launch_plan(p, cfg, attrs, c, n, "gfx950") for p, c, n in cases]
        monkeypatch.setattr(
            hl, "launch_inputs", lambda c, a, v=None: real(cfg, attrs, v)
        )
        strict_cfg = _Strict(cfg, hl.LAUNCH_SIGNATURE_FIELDS)
        strict_attrs = _Strict(attrs, ())
        got = [
            hl.launch_plan(p, strict_cfg, strict_attrs, c, n, "gfx950")
            for p, c, n in cases
        ]
        monkeypatch.setattr(hl, "launch_inputs", real)
        assert got == expect
    real(
        _Strict(dict(occupancy=2), hl.LAUNCH_CONFIG_FIELDS),
        _Strict(dict(HYBRID), hl.LAUNCH_ATTRIBUTES),
        2,
    )
    real(
        _Strict(dict(occupancy=2), hl.LAUNCH_CONFIG_FIELDS),
        _Strict(dict(HYBRID, workspace_size_per_elem_c=4), hl.LAUNCH_ATTRIBUTES_V3),
        3,
    )


# A hand-written Stream-K kernel on a static grid, like the one hipBLASLt's
# gfx950 TF32 libraries hold: macro tile 256x256x32, 256 threads.
HAND_WRITTEN_SK = {
    "tile_processing_strategy": 2,
    "work_assignment": 0,
    "work_group_x": 256,
    "work_group_y": 1,
    "work_group_z": 1,
    "workspace_size_per_elem_c": 4,
    "hand_written": 1,
}
GENERATED_SK = dict(HAND_WRITTEN_SK, hand_written=0)


def _sk_plan(m, n, k, context, attrs=HAND_WRITTEN_SK, batch=1):
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="xfloat32")
    cfg = dict(mt_m=256, mt_n=256, mt_k=32, occupancy=1)
    p = hl.launch_plan(prob, cfg, attrs, context, 256, "gfx950")
    return (p.dynamic_queue, p.grid_wgs, p.partial_reduction)


def test_hand_written_stream_k_kernels_reduce_in_a_tree():
    """TensileLite reduces the partial tiles of a hand-written Stream-K
    kernel in a tree (getSKReduction) where a generated one takes origami's
    parallel reduction, on the grid select_grid_size picks for the tree."""
    excl, sm64 = fs.EXCLUSIVE, fs.ExecutionContext(cu_budget=64)
    # Four tiles of 256 iterations each.
    assert _sk_plan(1024, 256, 8192, excl, GENERATED_SK) == (False, 128, "parallel")
    assert _sk_plan(1024, 256, 8192, excl) == (False, 64, "tree")
    assert _sk_plan(1024, 256, 8192, sm64, GENERATED_SK) == (False, 64, "parallel")
    assert _sk_plan(1024, 256, 8192, sm64) == (False, 64, "tree")
    assert _sk_plan(256, 256, 8192, excl, batch=4) == (False, 64, "tree")
    hybrid = dict(HAND_WRITTEN_SK, work_assignment=2)
    assert _sk_plan(1024, 256, 8192, excl, hybrid) == (False, 64, "tree")
    atomic = dict(HAND_WRITTEN_SK, stream_k_atomic=1)
    assert _sk_plan(1024, 256, 8192, excl, dict(atomic, hand_written=0)) == (
        False,
        128,
        "atomic",
    )
    assert _sk_plan(1024, 256, 8192, excl, atomic) == (False, 64, "atomic")
    # A tree for the generated kernel as well, no partial tile, a queue.
    dyn = fs.ExecutionContext(schedule="dynamic")
    queue = dict(HAND_WRITTEN_SK, work_assignment=1)
    for m, n, k, ctx, attrs, plan in (
        (2048, 512, 1024, excl, HAND_WRITTEN_SK, (False, 64, "tree")),
        (4096, 4096, 8192, excl, HAND_WRITTEN_SK, (False, 256, "none")),
        (1024, 256, 8192, dyn, hybrid, (True, 4, "none")),
        (1024, 256, 8192, excl, queue, (True, 4, "none")),
    ):
        assert _sk_plan(m, n, k, ctx, attrs) == plan
        assert _sk_plan(m, n, k, ctx, dict(attrs, hand_written=0)) == plan


# TensileLite's plans of the hand-written Stream-K kernel of a gfx950 TF32
# library (TilewrightLaunchTest.DumpsPoolPlans) next to the rule's plans of
# the kernel taken for a generated one: (m, n, k, batch, cu_budget,
# schedule, TensileLite's plan, the generated kernel's plan).
TENSILELITE_HAND_WRITTEN = [
    (3, 1201, 3008, 1, 0, "auto", (False, 40, "tree"), (False, 55, "parallel")),
    (105, 71, 10951, 1, 96, "default", (False, 16, "tree"), (False, 42, "parallel")),
    (1, 98, 36857, 4, 96, "auto", (False, 64, "tree"), (False, 96, "parallel")),
    (2, 1202, 44093, 1, 64, "auto", (False, 60, "tree"), (False, 60, "parallel")),
    (5238, 131, 32009, 1, 0, "auto", (False, 252, "tree"), (False, 252, "parallel")),
    (3659, 1, 22314, 1, 224, "dynamic", (False, 180, "tree"), (False, 210, "parallel")),
]


@pytest.mark.parametrize("row", TENSILELITE_HAND_WRITTEN)
def test_hand_written_stream_k_kernels_launch_as_tensilelite_launches_them(row):
    m, n, k, batch, budget, schedule, tensilelite, generated = row
    ctx = fs.ExecutionContext(cu_budget=budget, schedule=schedule)
    assert _sk_plan(m, n, k, ctx, batch=batch) == tensilelite
    assert _sk_plan(m, n, k, ctx, GENERATED_SK, batch) == generated


def test_hand_written_kernels_of_a_library_launch_as_tensilelite_launches_them(
    tmp_path,
):
    """The flag comes from the logic file's customKernel, reaches a library
    pool's kernels and the engine's launch model."""
    sk = dict(tileProcessingStrategy="StreamK", workAssignment="StaticGrid")
    sols = [solution(i, i, (256, 256, 32), **sk) for i in (5, 6, 7)]
    sols[0]["customKernel"] = {"name": sols[0]["name"], "generated": False}
    sols[1]["customKernel"] = {"name": sols[1]["name"], "generated": True}
    write_logic(tmp_path / f"{LIBRARY_STEM}.dat", sols, table=[5, 6, 7])
    pool = ev.pool_from_kernels(dat.load_library_kernels(tmp_path, LIBRARY_STEM))
    assert [k["attributes"].get("hand_written") for k in pool] == [1, None, None]
    prob = dict(m=1024, n=256, k=8192, batch=1, mi_dtype="bfloat16")
    launch = hl.HipblasltLaunch("gfx950")
    plans = [launch(prob, k, k["attributes"], fs.EXCLUSIVE, 256) for k in pool]
    assert [(p.grid_wgs, p.partial_reduction) for p in plans] == [
        (64, "tree"),
        (128, "parallel"),
        (128, "parallel"),
    ]
    keys = [launch.launch_key(k, k["attributes"]) for k in pool]
    assert keys[0] != keys[1] == keys[2]
    tw = pytest.importorskip("tilewright")
    engine = ev.engine_launch(tw, pool, launch, 256)
    problem = tw.Problem(
        size=tw.Dim3(1024, 256, 8192),
        a_dtype=tw.DataType.BFloat16,
        b_dtype=tw.DataType.BFloat16,
        c_dtype=tw.DataType.BFloat16,
        d_dtype=tw.DataType.BFloat16,
        mi_dtype=tw.DataType.BFloat16,
    )
    for i, want in enumerate(plans):
        config = ev.make_config(tw, pool[i], i)
        got = engine.plan(config, problem, tw.ExecutionContext())
        assert got.grid_wgs == want.grid_wgs
        assert got.partial_reduction == ev.make_plan(tw, want).partial_reduction
        assert engine.launch_key(config) == keys[i]


def test_the_engine_plans_under_the_calls_summation_order_and_workspace(tmp_path):
    """The engine's context reaches the launch model whole: under rule
    version 3 a launch without a workspace is the twin's, which differs from
    the launch in one that holds every partial tile, and the pipeline's
    context and the engine's carry the same order and workspace."""
    sk = dict(
        tileProcessingStrategy="StreamK",
        workAssignment="StaticGrid",
        workspaceSizePerElemC=4,
    )
    write_logic(
        tmp_path / f"{LIBRARY_STEM}.dat",
        [solution(5, 5, (256, 256, 32), **sk)],
        table=[5],
    )
    pool = ev.pool_from_kernels(dat.load_library_kernels(tmp_path, LIBRARY_STEM))
    prob = dict(m=1024, n=256, k=8192, batch=1, mi_dtype="bfloat16")
    launch = hl.HipblasltLaunch("gfx950", version=3)
    none = fs.ExecutionContext(uniform_summation_order=False, workspace_bytes=0)
    full = fs.ExecutionContext(uniform_summation_order=False)
    plans = [launch(prob, pool[0], pool[0]["attributes"], c, 256) for c in (none, full)]
    assert (plans[0].grid_wgs, plans[0].partial_reduction) != (
        plans[1].grid_wgs,
        plans[1].partial_reduction,
    )
    tw = pytest.importorskip("tilewright")
    for ctx in (none, full, fs.ExecutionContext(64, "auto", True, 1 << 20)):
        made = ev.make_context(tw, ctx)
        assert bool(made.uniform_summation_order) is ctx.uniform_summation_order
        assert made.workspace_bytes == (
            ev.ENGINE_UNBOUNDED_WORKSPACE
            if ctx.workspace_bytes is None
            else ctx.workspace_bytes
        )
    engine = ev.engine_launch(tw, pool, launch, 256)
    problem = tw.Problem(
        size=tw.Dim3(1024, 256, 8192),
        a_dtype=tw.DataType.BFloat16,
        b_dtype=tw.DataType.BFloat16,
        c_dtype=tw.DataType.BFloat16,
        d_dtype=tw.DataType.BFloat16,
        mi_dtype=tw.DataType.BFloat16,
    )
    config = ev.make_config(tw, pool[0], 0)
    for ctx, want in zip((none, full), plans):
        got = engine.plan(config, problem, ev.make_context(tw, ctx))
        assert got.grid_wgs == want.grid_wgs
        assert got.partial_reduction == ev.make_plan(tw, want).partial_reduction


def test_the_hand_written_flag_changes_only_parallel_reductions():
    """A hand-written kernel launches as the same kernel generated, except
    that it never reduces in parallel: where the generated kernel splits its
    tiles for a parallel reduction (or, atomic, adds partial tiles up on
    that reduction's grid), it launches the tree's grid, which the tree
    fix-up's bounds can turn back into one workgroup per tile."""
    rng = random.Random(29)
    seen = Counter()
    for _ in range(2000):
        cfg, attrs = _random_kernel(rng)
        attrs.pop("hand_written", None)
        for _ in range(5):
            prob = _random_problem(rng)
            ctx = fs.ExecutionContext(
                cu_budget=rng.choice([0, 64, 96, 200, 224, 256]),
                schedule=rng.choice(fs.SCHEDULES),
            )
            n_cu = rng.choice([0, 64, 256, 304])
            arch = rng.choice(["gfx950", "gfx942"])
            plain = hl.launch_plan(prob, cfg, attrs, ctx, n_cu, arch)
            hand = dict(attrs, hand_written=1)
            written = hl.launch_plan(prob, cfg, hand, ctx, n_cu, arch)
            assert written.dynamic_queue == plain.dynamic_queue
            if plain.partial_reduction == "parallel":
                assert written.partial_reduction in ("tree", "none")
            elif plain.partial_reduction == "atomic":
                assert written.partial_reduction in ("atomic", "none")
            else:
                assert written == plain
            seen[plain.partial_reduction, written.partial_reduction] += 1
    for pair in (("parallel", "tree"), ("atomic", "atomic"), ("tree", "tree")):
        assert seen[pair] > 50
    assert seen["parallel", "none"] + seen["atomic", "none"] > 0


def test_hand_written_stream_k_kernels_are_launch_classes_of_their_own():
    cfg = dict(mt_m=256, mt_n=256, mt_k=32, occupancy=1)
    for assign in (0, 2):
        hand = dict(HAND_WRITTEN_SK, work_assignment=assign)
        generated = dict(hand, hand_written=0)
        v2_inputs = hl.launch_inputs(cfg, generated, 2)
        assert hl.launch_inputs(cfg, hand, 2) == v2_inputs + (1,)
        assert hl.launch_key(cfg, hand) != hl.launch_key(cfg, generated)
    # A queue hands out whole tiles, and other kernels split none.
    for other in (
        {"work_assignment": 1},
        {"tile_processing_strategy": 1},
        {"tile_processing_strategy": 0},
    ):
        hand = dict(HAND_WRITTEN_SK, **other)
        generated = dict(hand, hand_written=0)
        assert hl.launch_inputs(cfg, hand) == hl.launch_inputs(cfg, generated)
        assert hl.launch_key(cfg, hand) == hl.launch_key(cfg, generated)


def test_launch_keys_of_generated_kernels_are_those_tensilelite_pins():
    """TensileLite's TilewrightLaunchTest.LaunchKeysMatchThePipeline pins
    these keys for its kernels (macro tile 128x128, DepthU 64, 256 threads)
    under rule version 2: a kernel that is not hand-written keeps the key it
    had."""

    def key(occupancy, **attrs):
        cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=occupancy)
        wg = {"work_group_x": 256, "work_group_y": 1, "work_group_z": 1}
        return hl.launch_key(cfg, dict(wg, **attrs), version=2)

    hybrid = {"tile_processing_strategy": 2, "work_assignment": 2}
    assert key(2, **hybrid) == 0x138281954B1A0008
    assert key(2, stream_k_atomic=1, **hybrid) == 0x30A54CB3ED1D0E1D
    assert key(2, tile_processing_strategy=1) == 0x862A415D97FBA99D
    assert key(2, tile_processing_strategy=1, stream_k_atomic=1) == 0x862A415D97FBA99D
    assert key(2) == key(5, work_assignment=2, stream_k_atomic=1) == 0x40D69E0CF0F65C45
    assert key(0, **hybrid) == key(1, **hybrid) == 0x7072D6B06BE7DE6B


def _ctx(budget=0, schedule="default"):
    return fs.ExecutionContext(cu_budget=budget, schedule=schedule)


WIDE = {"work_group_x": 1024, "work_group_y": 1024}
DP = {"tile_processing_strategy": 1, "work_assignment": 0}
# Every plan the tests above pin with _plan's kernel: (m, n, k, occupancy,
# attributes, context, arch, batch, plan).
PINNED = [
    (128 * 480, 128, 512, 2, HYBRID, _ctx(224), "gfx950", 1, (False, 320, "tree")),
    (128 * 481, 128, 512, 2, HYBRID, _ctx(224), "gfx950", 1, (True, 481, "none")),
    (4096, 4096, 512, 3, HYBRID, _ctx(224), "gfx950", 1, (True, 768, "none")),
    (4096, 4096, 512, 4, HYBRID, _ctx(224), "gfx950", 1, (False, 1024, "none")),
    (128 * 1883, 128, 512, 4, HYBRID, _ctx(224), "gfx950", 1, (False, 1883, "none")),
    (128 * 1884, 128, 512, 4, HYBRID, _ctx(224), "gfx950", 1, (True, 768, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(256), "gfx950", 1, (False, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(224), "gfx950", 1, (True, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(240), "gfx950", 1, (True, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(255), "gfx950", 1, (True, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(31), "gfx950", 1, (True, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(0), "gfx950", 1, (False, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(224, "auto"), "gfx950", 1, (True, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(0, "auto"), "gfx950", 1, (False, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(0, "dynamic"), "gfx950", 1, (True, 512, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(224), "gfx942", 1, (False, 1024, "none")),
    (4096, 4096, 512, 2, HYBRID, _ctx(224, "dynamic"), "gfx942", 1, (True, 512, "none")),
    (256, 256, 64 * 64, 1, HYBRID, _ctx(), "gfx950", 1, (False, 32, "parallel")),
    (256, 256, 64 * 32, 1, HYBRID, _ctx(), "gfx950", 1, (False, 16, "tree")),
    (256, 256, 64 * 32, 1, dict(HYBRID, stream_k_atomic=1), _ctx(), "gfx950", 1,
     (False, 16, "atomic")),
    (4096, 4096, 512, 2, DP, _ctx(), "gfx950", 1, (False, 512, "none")),
    (4096, 4096, 512, 5, QUEUE, _ctx(), "gfx950", 1, (True, 768, "none")),
    (4096, 4096, 512, 2, {}, _ctx(), "gfx950", 1, (False, 1024, "none")),
    (128 * 300, 128, 64 * 300, 1, HYBRID, _ctx(), "gfx950", 1, (False, 200, "tree")),
    (128 * 300, 128, 64 * 300, 1, dict(HYBRID, **WIDE), _ctx(), "gfx950", 1,
     (False, 300, "none")),
    (128 * 4500, 128, 512, 9, dict(HYBRID, **WIDE), _ctx(), "gfx950", 1,
     (False, 2048, "tree")),
    (1024, 1024, 512, 2, HYBRID, _ctx(), "gfx950", 16, (False, 512, "none")),
    (128 * 4500, 128, 512, 9, dict(HYBRID, **WIDE), _ctx(0, "dynamic"), "gfx950", 1,
     (True, 1792, "none")),
    (128 * 4500, 128, 512, 9, dict(QUEUE, **WIDE), _ctx(), "gfx950", 1,
     (True, 1792, "none")),
    (128 * 4500, 128, 512, 9, dict(QUEUE, **WIDE), _ctx(), "gfx942", 1,
     (True, 1792, "none")),
    (128 * 4500, 128, 512, 9, dict(QUEUE, **WIDE), _ctx(), "gfx90a", 1,
     (True, 2048 - 32, "none")),
    (128 * 4500, 128, 512, 9, dict(HYBRID, **WIDE), _ctx(), "gfx1250", 1,
     (False, 2048, "tree")),
    (4096, 4096, 512, 2, QUEUE, _ctx(), "gfx1250", 1, (True, 512, "none")),
    (4096, 1000, 512, 2, {}, _ctx(), "gfx950", 1, (False, 256, "none")),
    (4096, 1000, 512, 2, {"tile_processing_strategy": 0, "work_assignment": 2},
     _ctx(), "gfx950", 1, (False, 256, "none")),
]  # fmt: skip
# The plans of PINNED's parallel reductions with the kernel written by hand,
# by (m, n, k).
PINNED_IN_A_TREE = {(256, 256, 64 * 64): (False, 32, "tree")}


@pytest.mark.parametrize("row", PINNED)
def test_hand_written_kernels_keep_the_pinned_plans_but_parallel_reductions(row):
    """The launch fixtures above hold for generated kernels, and for the
    same kernels written by hand but where they reduce in parallel."""
    m, n, k, occupancy, attrs, ctx, arch, batch, plan = row
    for generated in (attrs, dict(attrs, hand_written=0)):
        assert _plan(m, n, k, occupancy, generated, ctx, arch, batch) == plan
    hand = dict(attrs, hand_written=1)
    want = PINNED_IN_A_TREE[(m, n, k)] if plan[2] == "parallel" else plan
    assert _plan(m, n, k, occupancy, hand, ctx, arch, batch) == want


@pytest.mark.parametrize("row", QUEUE_CLAMPED)
def test_hand_written_dynamic_queues_take_the_head_of_the_flag_region(row):
    m, n, k, batch, mt_m, mt_n, mt_k, wg, budget, schedule = row
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=mt_m, mt_n=mt_n, mt_k=mt_k, occupancy=8)
    attrs = dict(HYBRID, work_group_x=wg[0], work_group_y=wg[1], work_group_z=wg[2])
    ctx = fs.ExecutionContext(cu_budget=budget, schedule=schedule)
    hand = hl.launch_plan(prob, cfg, dict(attrs, hand_written=1), ctx, 256, "gfx950")
    assert hand == hl.launch_plan(prob, cfg, attrs, ctx, 256, "gfx950")


# ── rule version 3 ───────────────────────────────────────────────────────────

MIB = 1 << 20
# The worked examples' kernel: 256x256 tiles of depth 64, occupancy 1, 256
# threads, four bytes per partial element, static Stream-K with per-tile extras.
V3_SK = {
    "tile_processing_strategy": 2,
    "work_assignment": 0,
    "work_group_x": 256,
    "work_group_y": 1,
    "work_group_z": 1,
    "workspace_size_per_elem_c": 4,
    "per_tile_extra_iters": 1,
}
V3_DP = {
    "tile_processing_strategy": 1,
    "work_assignment": 0,
    "work_group_x": 256,
    "work_group_y": 1,
    "work_group_z": 1,
}


def _call(uso, workspace, budget=0, schedule="default"):
    return fs.ExecutionContext(
        cu_budget=budget,
        schedule=schedule,
        uniform_summation_order=uso,
        workspace_bytes=workspace,
    )


def _v3(m, n, k, attrs, context, occupancy=1, version=3, batch=1):
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=256, mt_n=256, mt_k=64, occupancy=occupancy)
    p = hl.launch_plan(prob, cfg, attrs, context, 256, "gfx950", version=version)
    return (p.dynamic_queue, p.grid_wgs, p.partial_reduction)


def test_the_execution_context_carries_the_calls_order_and_workspace():
    ctx = fs.ExecutionContext()
    assert (ctx.uniform_summation_order, ctx.workspace_bytes) == (False, None)
    assert _call(True, 0).is_exclusive() and not _call(True, 0, 64).is_exclusive()
    assert _call(False, (1 << 64) - 1).workspace_bytes == (1 << 64) - 1
    assert _call(True, MIB) != _call(False, MIB) != _call(False, None)
    for bad in (
        dict(uniform_summation_order=1),
        dict(workspace_bytes=-1),
        dict(workspace_bytes=1 << 64),
        dict(workspace_bytes=True),
        dict(workspace_bytes=1.0),
    ):
        with pytest.raises(ValueError):
            fs.ExecutionContext(**bad)


def test_rule_version_three_plans_the_worked_examples():
    """The worked examples of rule version 3, which
    TilewrightLaunchTest.RuleVersionThreePlansTheCallsSummationOrderAndWorkspace
    pins too."""
    sk = V3_SK
    no_per_tile = {k: v for k, v in sk.items() if k != "per_tile_extra_iters"}
    obstacle = dict(sk, uso_static_obstacle=2)
    queue = dict(sk, work_assignment=1)
    hybrid = dict(sk, work_assignment=2)
    excl = fs.EXCLUSIVE
    # A: 300 tiles of 64 iterations.
    a = (3840, 5120, 4096)
    assert _v3(*a, sk, excl, version=2) == (False, 200, "tree")
    assert _v3(*a, sk, _call(False, None)) == (False, 200, "tree")
    assert _v3(*a, sk, _call(False, 32 * MIB)) == (False, 300, "none")
    assert _v3(*a, sk, _call(True, 128 * MIB)) == (False, 300, "none")
    # B: 16 tiles of 35 iterations.
    b = (1024, 1024, 2240)
    assert _v3(*b, sk, excl, version=2) == (False, 64, "tree")
    assert _v3(*b, sk, _call(False, 8 * MIB)) == (False, 16, "none")
    assert _v3(*b, sk, _call(True, 128 * MIB)) == (False, 64, "tree")
    assert _v3(*b, no_per_tile, _call(True, 128 * MIB)) == (False, 16, "none")
    # D: a dynamic queue of 1000 tiles whose partials outgrow 128 MiB.
    d = (6400, 10240, 4096)
    assert _v3(*d, queue, excl, occupancy=3, version=2) == (True, 768, "none")
    assert _v3(*d, queue, _call(False, None), occupancy=3) == (True, 768, "none")
    for uso in (False, True):
        assert _v3(*d, queue, _call(uso, 128 * MIB), occupancy=3) == (
            True,
            1000,
            "none",
        )
    # E: 16 tiles of 128 iterations, parallel.
    e = (1024, 1024, 8192)
    assert _v3(*e, sk, excl, version=2) == (False, 256, "parallel")
    assert _v3(*e, sk, _call(False, 32 * MIB)) == (False, 16, "none")
    assert _v3(*e, sk, _call(True, 128 * MIB)) == (False, 256, "parallel")
    assert _v3(*e, obstacle, _call(True, 128 * MIB)) == (False, 256, "tree")
    # F: one iteration per tile.
    f = (3840, 5120, 64)
    assert _v3(*f, sk, excl, version=2) == (False, 200, "none")
    assert _v3(*f, sk, _call(False, 0)) == (False, 300, "none")
    assert _v3(*f, sk, _call(True, 128 * MIB)) == (False, 300, "none")
    # G: Hybrid at half the CUs takes the dynamic sub-mode, which takes no snap.
    for uso in (False, True):
        assert _v3(*d, hybrid, _call(uso, 128 * MIB, 128)) == (True, 256, "none")
    # H: DataParallel reads the workspace through Origami, never the order.
    assert _v3(*a, V3_DP, excl, version=2) == (False, 200, "none")
    for uso in (False, True):
        assert _v3(*a, V3_DP, _call(uso, 32 * MIB)) == (False, 300, "none")
        assert _v3(*a, V3_DP, _call(uso, None)) == (False, 200, "none")


@pytest.mark.parametrize(
    "g0, tiles, iters, per_tile, reduction, partial_bytes, workspace, flags, grid",
    [
        # Never upward: below the tiles one workgroup per tile, at the tiles unchanged.
        (200, 300, 64, True, "tree", 4, None, True, 300),
        (300, 300, 64, True, "tree", 4, None, True, 300),
        # No tiles or no grid: unchanged.
        (64, 0, 64, True, "tree", 4, None, True, 64),
        (0, 16, 64, True, "tree", 4, None, True, 0),
        # The largest factor up to 64 / 16 that passes every test.
        (64, 16, 64, True, "tree", 1, None, False, 64),
        # F | I, without per-tile extras and for tree only.
        (64, 16, 42, False, "tree", 1, None, False, 48),
        (64, 16, 35, False, "tree", 1, None, False, 16),
        (64, 16, 35, True, "tree", 1, None, False, 64),
        (64, 16, 35, False, "parallel", 1, None, False, 64),
        # At least MIN_ITERS_PER_CU iterations per workgroup.
        (64, 16, 24, True, "tree", 1, None, False, 48),
        (64, 16, 15, True, "tree", 1, None, False, 16),
        (256, 1, 64, True, "tree", 1, None, False, 8),
        # The partial tiles fit the workspace.
        (64, 16, 64, True, "tree", 1000, 48000, False, 48),
        (64, 16, 64, True, "tree", 1000, 0, False, 16),
        # Inside the flag region, when it binds.
        (4096, 1024, 64, True, "tree", 1, None, True, 2048),
        (4096, 1024, 64, True, "tree", 1, None, False, 4096),
    ],
)
def test_the_uniform_snap_takes_the_largest_admissible_split(
    g0, tiles, iters, per_tile, reduction, partial_bytes, workspace, flags, grid
):
    """getPersistentGridImpl's snap: the rows of
    TilewrightLaunchTest.UniformGridSnapsOntoAdmissibleGrids but the cluster
    one, since every kernel of the rule launches one workgroup per cluster."""
    got = hl._uniform_grid(
        g0, tiles, iters, per_tile, reduction, partial_bytes, workspace, flags
    )
    assert got == grid


def test_launch_keys_of_rule_version_three_are_those_tensilelite_pins():
    """TilewrightLaunchTest.LaunchKeysOfRuleVersionThree's kernels and keys:
    from version 3 on a Stream-K kernel hashes nine values; other kernels, and
    every kernel under versions 1 and 2, keep their keys."""
    cfg = dict(mt_m=128, mt_n=128, mt_k=64, occupancy=2)
    base = {
        "work_group_x": 256,
        "work_group_y": 1,
        "work_group_z": 1,
        "workspace_size_per_elem_c": 4,
    }
    hybrid = dict(base, tile_processing_strategy=2, work_assignment=2)

    def key(attrs, version=3):
        return hl.launch_key(cfg, attrs, version)

    assert key(hybrid) == 0x683F2C52BC20650C
    assert key(dict(hybrid, per_tile_extra_iters=1)) == 0x8EE6D58E798A4F6D
    assert key(dict(hybrid, stream_k_atomic=1)) == 0xB35A98B051C5B178
    assert key(dict(hybrid, work_assignment=0)) == 0x87F665DA871B8F8A
    queue = dict(hybrid, work_assignment=1, per_tile_extra_iters=1)
    assert key(queue) == 0xEDDA592695024407
    assert key(dict(hybrid, hand_written=1)) == 0x6D97DF395F16A58D
    assert key(dict(hybrid, workspace_size_per_elem_c=2)) == 0xCA4300C295F6348A
    assert key(dict(hybrid, uso_static_obstacle=2)) == 0x8739F35BC70FAF2D
    dp = dict(base, tile_processing_strategy=1, work_assignment=0)
    assert key(dp) == 0x862A415D97FBA99D
    assert key(base) == 0x40D69E0CF0F65C45
    for version in (1, 2):
        for attrs in (
            dict(hybrid, per_tile_extra_iters=1),
            dict(hybrid, workspace_size_per_elem_c=2),
            dict(hybrid, uso_static_obstacle=2),
        ):
            assert key(attrs, version) == 0x138281954B1A0008
    launch = hl.HipblasltLaunch("gfx950", version=3)
    assert launch.launch_key(cfg, hybrid) == key(hybrid)
    default = hl.HipblasltLaunch("gfx950").launch_key(cfg, hybrid)
    assert default == key(hybrid, hl.rule_version("gfx950"))
    without = {k: v for k, v in hybrid.items() if k != "workspace_size_per_elem_c"}
    with pytest.raises(ValueError, match="workspace_size_per_elem_c"):
        key(without)
    assert key(without, 2) == key(hybrid, 2)


@pytest.mark.parametrize("row", PINNED)
def test_rule_version_three_keeps_the_pinned_plans_without_the_order_and_workspace(
    row,
):
    """Version 3 with uniform summation order off and a workspace that holds
    every partial tile plans as version 2, for a kernel whose partial
    elements are the compute type's."""
    m, n, k, occupancy, attrs, ctx, arch, batch, plan = row
    attrs = dict(attrs, workspace_size_per_elem_c=4)
    assert _plan(m, n, k, occupancy, attrs, ctx, arch, batch, version=3) == plan


def _random_v3_kernel(rng):
    cfg, attrs = _random_kernel(rng)
    attrs["workspace_size_per_elem_c"] = rng.choice([2, 4, 8])
    for name, values in (
        ("per_tile_extra_iters", [0, 1]),
        ("uso_static_obstacle", [0, 0, 0, 2, 5]),
    ):
        if rng.random() < 0.7:
            attrs[name] = rng.choice(values)
    return cfg, attrs


def _random_call(rng):
    return _call(
        rng.random() < 0.5,
        rng.choice([None, 0, MIB, 32 * MIB, 128 * MIB]),
        rng.choice([0, 31, 64, 200, 224, 255, 256]),
        rng.choice(fs.SCHEDULES),
    )


def test_older_rule_versions_ignore_the_summation_order_and_workspace():
    """Versions 1 and 2 plan alike whatever the call's order and workspace and
    whatever the version 3 attributes; version 3 at order off and an
    unbounded workspace plans as version 2 for a kernel whose partial elements
    are the compute type's."""
    rng = random.Random(3)
    for _ in range(3000):
        cfg, attrs = _random_v3_kernel(rng)
        prob = _random_problem(rng)
        n_cu = rng.choice([64, 256, 304])
        budget = rng.choice([0, 64, 128, 224])
        call = _random_call(rng)
        plain = fs.ExecutionContext(cu_budget=call.cu_budget, schedule=call.schedule)
        for version in (1, 2):
            want = hl.launch_plan(
                prob, cfg, attrs, plain, n_cu, "gfx950", None, version
            )
            got = hl.launch_plan(prob, cfg, attrs, call, n_cu, "gfx950", None, version)
            assert got == want, (version, prob, cfg, attrs, call)
        compute = hl._WIDE_COMPUTE_BYTES.get(
            fs.normalize_dtype_name(prob["mi_dtype"]), 4
        )
        same = dict(attrs, workspace_size_per_elem_c=compute)
        unbounded = _call(False, None, budget)
        assert hl.launch_plan(
            prob, cfg, same, unbounded, n_cu, "gfx950", None, 3
        ) == hl.launch_plan(prob, cfg, same, unbounded, n_cu, "gfx950", None, 2)


def test_equal_rule_version_three_keys_launch_alike():
    """Within a version 3 launch-key class every call plans alike, the call's
    order and workspace included."""
    rng = random.Random(29)
    equal = differing = 0
    for _ in range(3000):
        cfg, attrs = _random_v3_kernel(rng)
        other_cfg, other_attrs = _random_v3_kernel(rng)
        other_cfg.update({k: cfg[k] for k in SIGNATURE})
        if rng.random() < 0.6:
            other_cfg["occupancy"] = cfg["occupancy"]
            for name in hl.LAUNCH_ATTRIBUTES_V3:
                if rng.random() < 0.8:
                    other_attrs.pop(name, None)
                    if name in attrs:
                        other_attrs[name] = attrs[name]
        other_attrs = _loadable(other_attrs)
        same = hl.launch_key(cfg, attrs, 3) == hl.launch_key(other_cfg, other_attrs, 3)
        plans = []
        for _ in range(6):
            prob, call = _random_problem(rng), _random_call(rng)
            n_cu = rng.choice([64, 256, 304])
            a = hl.launch_plan(prob, cfg, attrs, call, n_cu, "gfx950", None, 3)
            b = hl.launch_plan(
                prob, other_cfg, other_attrs, call, n_cu, "gfx950", None, 3
            )
            plans.append(a == b)
        if same:
            equal += 1
            assert all(plans), (cfg, attrs, other_cfg, other_attrs)
        elif not all(plans):
            differing += 1
    assert equal > 500 and differing > 300


def test_rule_version_three_reads_the_kernel_only_through_launch_inputs(monkeypatch):
    real = hl.launch_inputs
    rng = random.Random(7)
    cases = [(_random_problem(rng), _random_call(rng), 256) for _ in range(8)]
    for _ in range(500):
        cfg, attrs = _random_v3_kernel(rng)
        expect = [
            hl.launch_plan(p, cfg, attrs, c, n, "gfx950", None, 3) for p, c, n in cases
        ]
        monkeypatch.setattr(
            hl, "launch_inputs", lambda c, a, v=None: real(cfg, attrs, v)
        )
        strict_cfg = _Strict(cfg, hl.LAUNCH_SIGNATURE_FIELDS)
        strict_attrs = _Strict(attrs, ())
        got = [
            hl.launch_plan(p, strict_cfg, strict_attrs, c, n, "gfx950", None, 3)
            for p, c, n in cases
        ]
        monkeypatch.setattr(hl, "launch_inputs", real)
        assert got == expect


def test_uniform_summation_order_never_changes_a_data_parallel_plan():
    rng = random.Random(5)
    for _ in range(2000):
        cfg, _ = _random_kernel(rng)
        prob, call = _random_problem(rng), _random_call(rng)
        on = dict(uniform_summation_order=True)
        off = dict(uniform_summation_order=False)
        a = hl.launch_plan(
            prob, cfg, V3_DP, fs.ExecutionContext(**{**vars(call), **on}), 256, "gfx950",
            None, 3,
        )  # fmt: skip
        b = hl.launch_plan(
            prob, cfg, V3_DP, fs.ExecutionContext(**{**vars(call), **off}), 256, "gfx950",
            None, 3,
        )  # fmt: skip
        assert a == b


def test_the_snaps_workspace_and_flag_tests_never_bind_under_the_default_settings(
    monkeypatch,
):
    """Origami's grid already fits the call's workspace and holds at most
    the usable CUs, so the snap's workspace and flag-region tests never
    reject a split the others admit."""
    rng = random.Random(11)
    cases = []
    while len(cases) < 3000:
        cfg, attrs = _random_v3_kernel(rng)
        attrs.update(tile_processing_strategy=2, work_assignment=rng.choice([0, 2]))
        attrs.pop("uso_static_obstacle", None)
        call = _random_call(rng)
        call = fs.ExecutionContext(
            cu_budget=call.cu_budget,
            schedule="default",
            uniform_summation_order=True,
            workspace_bytes=call.workspace_bytes,
        )
        cases.append((_random_problem(rng), cfg, attrs, call))
    want = [hl.launch_plan(p, c, a, x, 256, "gfx950", None, 3) for p, c, a, x in cases]
    snapped = sum(
        p.grid_wgs > fs.output_tiles(q, c) for p, (q, c, _, _) in zip(want, cases)
    )
    real = hl._uniform_grid

    def loose(g0, tiles, iters, per_tile, reduction, partial_bytes, workspace, flags):
        return real(g0, tiles, iters, per_tile, reduction, partial_bytes, None, False)

    monkeypatch.setattr(hl, "_uniform_grid", loose)
    got = [hl.launch_plan(p, c, a, x, 256, "gfx950", None, 3) for p, c, a, x in cases]
    assert got == want
    assert snapped > 100


def test_the_uniform_summation_order_filter_mirror():
    """uso_obstacle names TensileLite's refusal of a kernel for a call with
    uniform summation order, from the kernel's attributes and its version 3
    plan."""
    prob = dict(m=1024, n=1024, k=2240, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=256, mt_n=256, mt_k=64, occupancy=1)
    on, off = _call(True, 128 * MIB), _call(False, 128 * MIB)

    def obstacle(attrs, plan=None, ctx=on):
        if plan is None:
            plan = hl.launch_plan(prob, cfg, attrs, ctx, 256, "gfx950", None, 3)
        return hl.uso_obstacle(prob, cfg, attrs, ctx, plan)

    no_per_tile = {k: v for k, v in V3_SK.items() if k != "per_tile_extra_iters"}
    # 16 tiles of 35 iterations: per-tile extras split every tile in four
    # workgroups, the kernel without them runs one workgroup per tile.
    assert obstacle(V3_SK) is None and obstacle(no_per_tile) is None
    assert obstacle(dict(V3_SK, hand_written=1), ctx=off) is None
    assert obstacle(dict(V3_SK, hand_written=1)) == "CustomKernel"
    assert obstacle(dict(V3_DP, hand_written=1)) == "CustomKernel"
    # One that declares the order passes unless a StaggerU is compiled into
    # it, which TensileLite asks after the Stream-K clauses.
    declared = dict(hand_written=1, uso_declared=1)
    assert obstacle(dict(V3_SK, **declared), fs.LaunchPlan(False, 64, "tree")) is None
    assert obstacle(dict(V3_DP, **declared)) is None
    assert obstacle(dict(V3_DP, **declared, stagger_u=4)) == "CompiledInStaggerU"
    assert obstacle(dict(V3_DP, stagger_u=4)) is None
    atomic_declared = dict(V3_SK, **declared, stagger_u=4, uso_static_obstacle=1)
    assert obstacle(atomic_declared) == "StreamKAtomic"
    assert obstacle(dict(V3_SK, uso_declared=1)) is None
    for position, token in enumerate(hl.USO_STATIC_OBSTACLES, 1):
        assert obstacle(dict(V3_SK, uso_static_obstacle=position)) == token
    assert obstacle(dict(V3_SK, stream_k_atomic=1)) == "StreamKAtomic"
    with pytest.raises(ValueError, match="names no obstacle"):
        obstacle(dict(V3_SK, uso_static_obstacle=len(hl.USO_STATIC_OBSTACLES) + 1))
    assert obstacle(dict(V3_DP, uso_static_obstacle=2)) is None
    assert obstacle(V3_SK, fs.LaunchPlan(False, 0, "none")) == "StreamKGridZero"
    # Tree splits: whole tiles, every tile alike, the per-tile extras.
    split = "StaticSplitNotRowUniform"
    assert obstacle(no_per_tile, fs.LaunchPlan(False, 16, "none")) is None
    assert obstacle(no_per_tile, fs.LaunchPlan(False, 48, "tree")) == split
    assert obstacle(V3_SK, fs.LaunchPlan(False, 48, "tree")) is None
    assert obstacle(no_per_tile, fs.LaunchPlan(False, 8, "none")) == split
    assert obstacle(no_per_tile, fs.LaunchPlan(False, 80, "tree")) is None
    # A dynamic queue takes whole tiles; parallel splits every tile alike.
    queue = dict(V3_SK, work_assignment=1)
    assert obstacle(queue, fs.LaunchPlan(True, 16, "none")) is None
    assert obstacle(V3_SK, fs.LaunchPlan(False, 256, "parallel")) is None


# Rule version 4: the Stream-K kernels of TensileLite's spatial-cluster layout.


def _abi2(x=2, y=1, **attrs):
    """A kernel of the spatial-cluster layout as TensileLite's build gives it:
    static non-atomic Stream-K of 256 threads, four bytes per partial element,
    ClusterDim [x, y, 1]."""
    out = {
        "tile_processing_strategy": 2,
        "work_assignment": 0,
        "work_group_x": 256,
        "work_group_y": 1,
        "work_group_z": 1,
        "workspace_size_per_elem_c": 4,
        "persistent_loop_args_version": 2,
        "cluster_dim_x": x,
        "cluster_dim_y": y,
    }
    out.update(attrs)
    return out


def test_spatial_cluster_kernels_are_the_stream_k_kernels_of_their_layout():
    assert hl.spatial_cluster_kernel(_abi2())
    assert hl.persistent_cluster_kernel(_abi2())
    assert not hl.spatial_cluster_kernel(_abi2(persistent_loop_args_version=0))
    assert not hl.spatial_cluster_kernel(_abi2(tile_processing_strategy=1))
    assert not hl.spatial_cluster_kernel(None)
    for x, y in ((2, 1), (4, 1), (8, 1), (1, 2), (1, 8), (2, 2), (4, 4), (8, 2)):
        assert hl.spatial_cluster_errors(_abi2(x, y)) == [], (x, y)
    # TensileLite launches no other layout kernel (validatePersistentLoopArgs).
    for x, y in ((1, 1), (16, 1), (3, 1), (1, 16), (4, 8), (1, 3)):
        assert hl.spatial_cluster_errors(_abi2(x, y)), (x, y)
    for bad in (
        dict(cluster_dim_z=2),
        dict(work_assignment=1),
        dict(work_assignment=2),
        dict(stream_k_atomic=1),
        dict(prefetch_across_persistent=1),
        dict(hand_written=1),
        dict(cluster_unsupported=1),
    ):
        assert len(hl.spatial_cluster_errors(_abi2(**bad))) == 1, bad


def _v4(m, n, k, attrs, context, tile=(256, 256, 64), occupancy=1, batch=1, n_cu=256):
    prob = dict(m=m, n=n, k=k, batch=batch, mi_dtype="bfloat16")
    cfg = dict(mt_m=tile[0], mt_n=tile[1], mt_k=tile[2], occupancy=occupancy)
    p = hl.launch_plan(prob, cfg, attrs, context, n_cu, "gfx1250", version=4)
    return (p.dynamic_queue, p.grid_wgs, p.partial_reduction)


# Calls of kernels of the spatial-cluster layout, one or more for every step
# of resolveClusteredStreamKSettings, and TensileLite's plans of them, which
# TilewrightLaunchTest.PlansSpatialClustersAsTheirSchedule pins against its
# own launch code: m, n, k, batch, the macro tile, occupancy, ClusterDim,
# per-tile extras, CUs, CU budget, uniform summation order, workspace (None:
# any) -> grid in workgroups, reduction.
SPATIAL_CLUSTER_PLANS = [
    # A parallel reduction over whole blocks.
    (
        1024,
        256,
        64 * 300,
        1,
        (256, 256, 64),
        1,
        (2, 2),
        1,
        256,
        0,
        True,
        None,
        256,
        "parallel",
    ),
    (
        1000,
        256,
        4096,
        4,
        (256, 256, 64),
        1,
        (4, 1),
        1,
        256,
        0,
        False,
        None,
        128,
        "parallel",
    ),
    (
        1024,
        1024,
        64 * 300,
        1,
        (256, 256, 64),
        2,
        (2, 4),
        0,
        96,
        192,
        False,
        None,
        96,
        "parallel",
    ),
    # A tree over the split of the blocks, held to the resident clusters.
    (
        8192,
        8192,
        16384,
        4,
        (256, 256, 64),
        2,
        (2, 1),
        1,
        96,
        0,
        False,
        128 * MIB,
        190,
        "tree",
    ),
    (
        8192,
        1024,
        64 * 300,
        4,
        (256, 256, 64),
        1,
        (8, 1),
        1,
        256,
        192,
        False,
        None,
        192,
        "tree",
    ),
    (
        1000,
        4096,
        16384,
        4,
        (256, 128, 256),
        2,
        (4, 4),
        1,
        256,
        192,
        False,
        None,
        336,
        "tree",
    ),
    # Under uniform summation order, equal K chunks of every block.
    (2048, 256, 4096, 1, (128, 128, 128), 1, (2, 1), 1, 96, 0, True, None, 96, "tree"),
    (
        1000,
        1000,
        16384,
        1,
        (128, 128, 128),
        2,
        (4, 4),
        1,
        256,
        192,
        True,
        None,
        192,
        "tree",
    ),
    # Whole blocks: past the tree's arithmetic bounds; partials the workspace
    # cannot hold; a whole-tile grid; K of 0; a split without partials; no
    # equal chunks under uniform summation order.
    (
        8192,
        256,
        64 * 65536,
        1,
        (256, 256, 64),
        1,
        (8, 1),
        0,
        256,
        64,
        True,
        None,
        32,
        "none",
    ),
    (
        16384,
        256,
        64 * 300,
        1,
        (256, 128, 256),
        1,
        (2, 1),
        1,
        96,
        64,
        False,
        4 * MIB,
        64,
        "none",
    ),
    (
        2048,
        4096,
        16384,
        1,
        (256, 256, 64),
        1,
        (2, 2),
        0,
        256,
        192,
        True,
        None,
        128,
        "none",
    ),
    (
        128 * 481,
        1000,
        0,
        1,
        (128, 128, 128),
        2,
        (2, 2),
        0,
        96,
        0,
        True,
        None,
        192,
        "none",
    ),
    (256, 256, 4096, 1, (256, 128, 256), 1, (8, 1), 1, 96, 0, False, None, 16, "none"),
    (
        256,
        1000,
        64 * 65536,
        4,
        (256, 256, 64),
        2,
        (8, 1),
        1,
        96,
        0,
        True,
        128 * MIB,
        128,
        "none",
    ),
]


@pytest.mark.parametrize("row", SPATIAL_CLUSTER_PLANS)
def test_rule_version_four_plans_spatial_clusters_as_their_schedule(row):
    m, n, k, batch, tile, occupancy, (x, y), per_tile, n_cu = row[:9]
    budget, uso, workspace, grid, reduction = row[9:]
    attrs = _abi2(x, y, per_tile_extra_iters=per_tile)
    ctx = _call(uso, workspace, budget)
    got = _v4(m, n, k, attrs, ctx, tile, occupancy, batch, n_cu)
    assert got == (False, grid, reduction)
    # The grid is physical workgroups, whole clusters of them.
    assert grid % (x * y) == 0
    # Versions 1 to 3 do not plan the layout: their plan is another one.
    assert hl.launch_key(dict(mt_m=1), attrs, 4) != hl.launch_key(
        dict(mt_m=1), attrs, 3
    )


def test_launch_keys_of_rule_version_four_are_those_tensilelite_pins():
    """TilewrightLaunchTest.LaunchKeysOfRuleVersionFour pins these: the
    version 3 values, then the layout, the cluster and the unit of K;
    version 3's key ignores the cluster."""
    cfg = dict(mt_m=256, mt_n=256, mt_k=64, occupancy=1)
    plain, per_tile = _abi2(), _abi2(per_tile_extra_iters=1)
    square = _abi2(4, 4, per_tile_extra_iters=1)
    tree_only, scaled = _abi2(cluster_tree_only=1), _abi2(8, 1, sk_k_granularity=4)
    assert hl.launch_key(cfg, plain, 4) == 0x0663633002D68789
    assert hl.launch_key(cfg, per_tile, 4) == 0x4BE1BAFA8AF4C8E8
    assert hl.launch_key(cfg, square, 4) == 0xE6CED38AB1E5622B
    assert hl.launch_key(cfg, tree_only, 4) == 0x57CE1E5EC6F562C8
    assert hl.launch_key(cfg, scaled, 4) == 0x7B65C0E4B328E7A6
    assert hl.launch_key(cfg, plain, 3) == 0xE95A5D101448F4E9
    assert hl.launch_key(cfg, square, 3) == 0xC2B2B3D456DF0A88
    assert hl.launch_key(cfg, tree_only, 3) == hl.launch_key(cfg, plain, 3)


def test_a_spatial_cluster_the_device_cannot_launch_has_no_plan():
    """TensileLite throws for a cluster of more workgroups than the device's
    resident budget holds; so does the rule."""
    with pytest.raises(ValueError, match="does not fit"):
        _v4(8192, 8192, 4096, _abi2(8, 1), _call(False, None), n_cu=4)
    assert _v4(8192, 8192, 4096, _abi2(8, 1), _call(False, None), n_cu=8)[1] % 8 == 0


def _random_abi2_kernel(rng):
    cfg = dict(
        mt_m=rng.choice([64, 128, 256]),
        mt_n=rng.choice([64, 128, 256]),
        mt_k=rng.choice([32, 64, 128, 256]),
        mi_m=16, mi_n=16, mi_k=32, cache_hints_a=0, cache_hints_b=0,
        occupancy=rng.choice([1, 1, 2]),
        grvw_a=4, grvw_b=4, gwvw_d=4,
    )  # fmt: skip
    attrs = _abi2(*rng.choice([(2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (2, 4), (4, 4)]))
    attrs["work_group_x"] = rng.choice([128, 256, 512])
    attrs["workspace_size_per_elem_c"] = rng.choice([2, 4, 8])
    for name, values in (
        ("per_tile_extra_iters", [0, 1]),
        ("uso_static_obstacle", [0, 0, 0, 2, 6]),
        ("sk_k_granularity", [1, 1, 2, 4]),
        ("cluster_tree_only", [0, 0, 1]),
    ):
        if rng.random() < 0.6:
            attrs[name] = rng.choice(values)
    return cfg, attrs


def _random_cluster_problem(rng):
    return dict(
        m=rng.choice([1, 100, 256, 1000, 1024, 4096, 8192, 128 * 481]),
        n=rng.choice([1, 64, 256, 1000, 4096, 8192]),
        k=rng.choice([0, 1, 64, 512, 4096, 64 * 300, 64 * 1027, 64 * 65536]),
        batch=rng.choice([1, 1, 2, 16]),
        mi_dtype="bfloat16",
    )


def test_rule_version_four_plans_every_other_kernel_as_version_three():
    """Versions 4 and 3 give every kernel but those of the spatial-cluster
    layout one key and one plan, kernels with clusters of another layout
    too."""
    rng = random.Random(41)
    for _ in range(3000):
        cfg, attrs = _random_v3_kernel(rng)
        if rng.random() < 0.3:
            attrs["cluster_dim_x"] = rng.choice([1, 2, 4])
            attrs["cluster_dim_y"] = rng.choice([1, 2])
        if rng.random() < 0.3:
            attrs["persistent_loop_args_version"] = rng.choice([0, 1])
        assert hl.launch_key(cfg, attrs, 4) == hl.launch_key(cfg, attrs, 3)
        prob, call = _random_problem(rng), _random_call(rng)
        n_cu = rng.choice([64, 256, 304])
        assert hl.launch_plan(
            prob, cfg, attrs, call, n_cu, "gfx950", None, 4
        ) == hl.launch_plan(prob, cfg, attrs, call, n_cu, "gfx950", None, 3)


def test_equal_rule_version_four_keys_launch_alike():
    """Kernels of the spatial-cluster layout of one signature and one version
    4 key plan alike for every call; of different keys they mostly do not."""
    rng = random.Random(43)
    equal = differing = 0
    for _ in range(2000):
        cfg, attrs = _random_abi2_kernel(rng)
        other_cfg, other_attrs = _random_abi2_kernel(rng)
        other_cfg.update({k: cfg[k] for k in SIGNATURE})
        if rng.random() < 0.6:
            other_cfg["occupancy"] = cfg["occupancy"]
            for name in hl.LAUNCH_ATTRIBUTES_V4:
                if rng.random() < 0.85:
                    other_attrs.pop(name, None)
                    if name in attrs:
                        other_attrs[name] = attrs[name]
        same = hl.launch_key(cfg, attrs, 4) == hl.launch_key(other_cfg, other_attrs, 4)
        plans = []
        for _ in range(6):
            prob, call = _random_cluster_problem(rng), _random_call(rng)
            n_cu = rng.choice([64, 96, 256])
            a = hl.launch_plan(prob, cfg, attrs, call, n_cu, "gfx1250", None, 4)
            b = hl.launch_plan(
                prob, other_cfg, other_attrs, call, n_cu, "gfx1250", None, 4
            )
            plans.append(a == b)
        if same:
            equal += 1
            assert all(plans), (cfg, attrs, other_cfg, other_attrs)
        elif not all(plans):
            differing += 1
    assert equal > 500 and differing > 300


def test_rule_version_four_reads_the_kernel_only_through_launch_inputs(monkeypatch):
    real = hl.launch_inputs
    rng = random.Random(47)
    cases = [
        (_random_cluster_problem(rng), _random_call(rng), rng.choice([96, 256]))
        for _ in range(8)
    ]
    for _ in range(300):
        cfg, attrs = _random_abi2_kernel(rng)
        expect = [
            hl.launch_plan(p, cfg, attrs, c, n, "gfx1250", None, 4) for p, c, n in cases
        ]
        monkeypatch.setattr(
            hl, "launch_inputs", lambda c, a, v=None: real(cfg, attrs, v)
        )
        strict_cfg = _Strict(cfg, hl.LAUNCH_SIGNATURE_FIELDS)
        strict_attrs = _Strict(attrs, ())
        got = [
            hl.launch_plan(p, strict_cfg, strict_attrs, c, n, "gfx1250", None, 4)
            for p, c, n in cases
        ]
        monkeypatch.setattr(hl, "launch_inputs", real)
        assert got == expect
    real(
        _Strict(dict(occupancy=2), hl.LAUNCH_CONFIG_FIELDS),
        _Strict(_abi2(per_tile_extra_iters=1), hl.LAUNCH_ATTRIBUTES_V4),
        4,
    )


def test_uniform_summation_order_refuses_a_spatial_cluster_only_for_its_kernel():
    """Under uniform summation order TensileLite snaps a cluster schedule onto
    equal K chunks of every block, so with whole iterations as its unit of K
    a kernel of the layout is refused only for a static obstacle; off, it is
    refused for none."""
    rng = random.Random(53)
    planned = 0
    for _ in range(3000):
        cfg, attrs = _random_abi2_kernel(rng)
        attrs["sk_k_granularity"] = 1
        attrs.pop("uso_static_obstacle", None)
        prob, call = _random_cluster_problem(rng), _random_call(rng)
        on = _call(True, call.workspace_bytes, call.cu_budget, call.schedule)
        plan = hl.launch_plan(prob, cfg, attrs, on, 256, "gfx1250", None, 4)
        assert hl.uso_obstacle(prob, cfg, attrs, on, plan) is None, (prob, cfg, attrs)
        planned += plan.partial_reduction == "tree"
        off = _call(False, call.workspace_bytes, call.cu_budget, call.schedule)
        plan = hl.launch_plan(prob, cfg, attrs, off, 256, "gfx1250", None, 4)
        assert hl.uso_obstacle(prob, cfg, attrs, off, plan) is None
    assert planned > 50
    prob = dict(m=1024, n=1024, k=4096, batch=1, mi_dtype="bfloat16")
    cfg = dict(mt_m=256, mt_n=256, mt_k=64, occupancy=1)
    obstacle = hl.USO_STATIC_OBSTACLES[5]
    attrs = _abi2(uso_static_obstacle=6)
    tree = fs.LaunchPlan(False, 32, "tree")
    assert hl.uso_obstacle(prob, cfg, attrs, _call(True, None), tree) == obstacle
    # A split of eight blocks of 64 iterations over 6 clusters is not
    # row-uniform; over 16 it is.
    for grid, token in ((12, "StaticSplitNotRowUniform"), (32, None)):
        plan = fs.LaunchPlan(False, grid, "tree")
        assert hl.uso_obstacle(prob, cfg, _abi2(), _call(True, None), plan) == token
    assert (
        hl.uso_obstacle(
            prob, cfg, _abi2(), _call(True, None), fs.LaunchPlan(False, 0, "none")
        )
        == "StreamKGridZero"
    )
