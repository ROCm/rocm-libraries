# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipBLASLt's launch of its kernels, as the framework-neutral
`features.LaunchPlan` that v3 launch features read.

The engine knows no framework's launch rule: an integration passes it a
`tilewright::LaunchModel` (PoolOptions::launch). This pipeline trains on
hipBLASLt measurements, so stage05 computes the plans of its training pairs
here, and `lib.evaluate` hands the engine the same plans. In deployment the
TensileLite adapter implements the LaunchModel with TensileLite's own launch
code; for every (config, problem, context) its plan must equal
`launch_plan`'s, or the model sees other launch features than it was trained
on:

    class LaunchModel {
      virtual LaunchPlan plan(const Config&, const Problem&,
                              const ExecutionContext&) const = 0;
      virtual std::uint64_t launch_key(const Config&) const;
    };
    LaunchPlan{dynamic_queue, grid_wgs, partial_reduction}

Its launch_key must split kernels as `launch_key` does: kernels with equal
signature and equal key launch alike (`launch_inputs`, the only kernel
parameters besides the signature's tile sizes that `launch_plan` reads), and
kernels in different launch classes stay separate candidates.

The rule is TensileLite's ContractionSolution.cpp (streamK5EffectiveDynamic,
getSKReduction, getPersistentGridImpl, resolveStreamKSettings) and the
Stream-K arithmetic it calls (select_hybrid_mode, select_reduction,
select_grid_size with k_split_aware), with the default TENSILE_PERSISTENT_*
settings. Rule versions 1 and 2 plan without uniform summation order (a call
that leaves it to a handle whose default is on launches otherwise;
lib/launch_settings.py records the mode the benches ran with) and with a
workspace that holds every partial tile; version 3 plans under the call's,
`context.uniform_summation_order` and `context.workspace_bytes`. The SM
target is `context.cu_budget` as ContractionProblem.hpp setSmCountTarget
keeps it, every CU it names; 0 means none.

TensileLite launches a hand-written kernel (attribute `hand_written`: its
logic file's customKernel names a kernel that was not generated) through
resolvePersistentSettings, whose getSKReduction gives a hand-written Stream-K
kernel the tree reduction whatever select_reduction would choose, so that
its static grid is select_grid_size's for the tree; its other choices are a
generated kernel's. Under uniform summation order it refuses one unless its
customKernel declares the order (attribute `uso_declared`), and one with a
StaggerU compiled in even then (`uso_obstacle`).

A persistent kernel, DataParallel or Stream-K, launched in clusters
(`persistent_cluster_kernel`: its ClusterDim spans more than one workgroup)
is outside rule versions 1 to 3: TensileLite leaves it out of the default
persistent launch (ContractionSolution::followsDefaultPersistentLaunch) and
plans its grid with cluster rules, so a run under them refuses a pool that
holds one (run_orchestrator.py preflight). Version 4 plans the Stream-K
kernels of the spatial-cluster argument layout (`spatial_cluster_kernel`)
as resolveClusteredStreamKSettings does; its preflight still refuses every
other clustered kernel, and those of that layout TensileLite does not
launch (`spatial_cluster_errors`).
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from . import features as fs
from .arch import canonical_arch

# Version of TensileLite's launch rule, part of `HipblasltLaunch.id`, which
# models record and the engine checks. Bump it when that rule changes, or when
# a change to `launch_plan` would make models trained before it see other
# plans than TensileLite gives them, so that those models are ranked with the
# plans they saw or refused. A fix that brings launch_plan closer to
# TensileLite does not count: TensileLite already gave those plans.
#
# Version 2 takes the grid Origami's select_grid_size chooses for a static
# Stream-K launch of batch 1 with one K iteration per tile and more tiles than
# usable CUs: one workgroup per tile once the streaming grid holds more than
# two workgroups per usable CU, where version 1 took one whenever the tiles
# numbered 2 to 32 times the usable CUs. It reports no partial reduction for a
# launch of one K iteration per tile, whose workgroups split no tile, where
# version 1 reported the reduction of every grid that does not divide the
# tiles.
#
# Version 3 plans a launch under the call's uniform summation order and
# workspace, where versions 1 and 2 take the order as off and a workspace
# that holds every partial tile: Origami's workspace guard in
# select_grid_size, and the workspace fit of resolveStreamKSettings, for any
# workspace; the reduction override of getSKReduction and the static two-tile
# grid snap of getPersistentGridImpl under a uniform summation order. Its
# launch key also covers the kernel inputs those steps read
# (LAUNCH_ATTRIBUTES_V3).
#
# Version 4 also plans the kernels of TensileLite's spatial-cluster argument
# layout (`spatial_cluster_kernel`), which versions 1 to 3 leave out, as
# resolveClusteredStreamKSettings does (`_spatial_cluster_plan`), under a
# launch key that covers the cluster inputs (LAUNCH_ATTRIBUTES_V4). It plans
# every other kernel as version 3 does, under the same key.
#
# The rule version models train on, but on the architectures of RULE_1_ARCHS
# and RULE_4_ARCHS.
RULE_VERSION = 3
# The rule versions launch_plan implements.
RULE_VERSIONS = (1, 2, 3, 4)
# Architectures whose models still train on rule version 1, and those whose
# models train on rule version 4. TensileLite's
# TilewrightLaunch::ruleVersion() and the C++ port's rule_version()
# (scripts/experiments/hipblaslt_launch_rule.hpp) hold the same tables, and
# TensileLite's TilewrightLaunchTest pins the ones test_hipblaslt_launch.py
# pins: change them together.
RULE_1_ARCHS: Tuple[str, ...] = ()
RULE_4_ARCHS: Tuple[str, ...] = ("gfx1250",)

# Kernel attributes (`lib.dat.kernel_dat_info`) the rule reads, 0 when a
# kernel lacks one (TensileLite's default).
LAUNCH_ATTRIBUTES = (
    "tile_processing_strategy",
    "work_assignment",
    "stream_k_atomic",
    "work_group_x",
    "work_group_y",
    "work_group_z",
    "hand_written",
)
# Kernel attributes rule version 3 reads besides LAUNCH_ATTRIBUTES, launch-only
# attributes lib.dat reads from the solution: whether the kernel redistributes
# the extra iterations of a Stream-K split within each tile
# (internalArgsSupport.perTileExtraIters), the bytes of one element of its
# partial tiles (sizeMapping.workspaceSizePerElemC, which a Stream-K kernel
# must have) and its static obstacle to a uniform summation order, the 1-based
# position of its token in USO_STATIC_OBSTACLES (0 for none).
LAUNCH_ATTRIBUTES_V3 = LAUNCH_ATTRIBUTES + (
    "per_tile_extra_iters",
    "workspace_size_per_elem_c",
    "uso_static_obstacle",
)
# Kernel attributes rule version 4 reads besides LAUNCH_ATTRIBUTES_V3, of a
# kernel of the spatial-cluster layout: the layout (the persistent loop args
# version of its InternalSupportParams as TensileLite's build sets it), its
# cluster shape, the unroll iterations of the unit of K Stream-K hands out
# whole (ContractionSolution::skKGranularity(), absent for 1) and whether
# TensileLite reduces its partial tiles in a tree whatever the problem (the
# terms of getSKReduction its kernel and problem type decide).
LAUNCH_ATTRIBUTES_V4 = LAUNCH_ATTRIBUTES_V3 + (
    "persistent_loop_args_version",
    "cluster_dim_x",
    "cluster_dim_y",
    "sk_k_granularity",
    "cluster_tree_only",
)
# The obstacles to a uniform summation order a kernel decides alone, by the
# tokens of TensileLite's streamKUniformSummationOrderObstacle, in the order
# it tests them (ContractionSolution::uniformSummationOrderKernelObstacle()
# names the first that holds). The packed index ones also hold for a problem
# with more than one free index on an operand, which no GEMM has.
USO_STATIC_OBSTACLES = (
    "StreamKAtomic",
    "UseInitialStridesCD",
    "PackedC0Index",
    "PackedC1Index",
    "WaveSplitK",
    "MXScaleFormat",
    "MXBlockSize",
    "MXMatrixInstK",
    "MXScaleSwizzleGranule",
)
# Config fields the rule reads besides the signature's tile sizes.
LAUNCH_CONFIG_FIELDS = ("occupancy",)
# Signature fields the rule reads.
LAUNCH_SIGNATURE_FIELDS = ("mt_m", "mt_n", "mt_k")

# The CUs an SM target is rounded down to a multiple of: one, since hipBLASLt
# takes every CU budget as its SM target as it is.
SM_TARGET_GRANULE = 1
# select_hybrid_mode: a Hybrid kernel turns dynamic only with more tiles than
# MIN_TILES_FOR_DYNAMIC, fewer usable CUs than the device has, and either an
# occupancy of at most MAX_OCCUPANCY_ALWAYS_DYNAMIC or more than
# TILES_PER_CU_DYNAMIC tiles per usable CU.
MIN_TILES_FOR_DYNAMIC = 480
MAX_OCCUPANCY_ALWAYS_DYNAMIC = 3
TILES_PER_CU_DYNAMIC = 8.41
# A dynamic queue launches at most this many workgroups per CU.
MAX_DYNAMIC_OCCUPANCY = 3
# k_split_aware keeps a grid whose partial tiles need more workspace than this
# only when the grid divides the tiles.
MAX_PARTIAL_WORKSPACE = 128 * 1024 * 1024
MIN_ITERS_PER_CU = 8
# Tree fix-up divides with 24-bit dividends and 16-bit divisors.
MAX_TREE_ITERS = 65536
MAX_TREE_DIVIDEND = 1 << 24
# Full Stream-K tiles per workgroup of a static split (TENSILE_STREAMK_FULL_TILES).
SK_FULL_TILES = 1
# Ready flags of the Stream-K fix-up, one per workgroup that uses them.
STREAM_K_FLAG_ELEMENTS = 2048
# A dynamic-queue launch puts one queue counter per XCD, each on its own cache
# line, at the base of the flag region and starts its flags after them
# (origami get_default_num_xcds and get_default_cache_line_bytes). The count
# is the one hipBLASLt compiles into those kernels: this table's for the
# architectures listed here, for any other the device's XCD count, which a
# run reads when it starts (`queue_count`).
QUEUE_COUNTS = {
    "gfx90a": 1,
    "gfx942": 8,
    "gfx950": 8,
    "gfx1100": 1,
    "gfx1101": 1,
    "gfx1150": 1,
    "gfx1151": 1,
    "gfx1152": 1,
    "gfx1153": 1,
    "gfx1200": 1,
    "gfx1201": 1,
}
CACHE_LINE_BYTES = 128
_FLAG_BYTES = 4
_U32_ITEMS = 1 << 32
# Bytes of one partial-tile element: the compute type's.
_WIDE_COMPUTE_BYTES = {"double": 8, "complexfloat": 8, "int64": 8, "complexdouble": 16}


def rule_version(arch: str) -> int:
    """The rule version models of `arch` train on: 1 for RULE_1_ARCHS, 4 for
    RULE_4_ARCHS, else RULE_VERSION."""
    arch = canonical_arch(str(arch))
    if arch in RULE_1_ARCHS:
        return 1
    return 4 if arch in RULE_4_ARCHS else RULE_VERSION


def served_rule_version(arch: str, launch_model: Optional[str]) -> int:
    """The rule version that ranks a model of `arch` recording the launch
    model `launch_model`, as TensileLite serves models: the version a model
    records when it is one of RULE_VERSIONS, whatever version `arch` trains
    on now, else `rule_version(arch)`, so that the engine refuses a model
    that records another launch model."""
    # Serving an older version keeps the models trained before their
    # architecture moved on servable, here, in
    # TilewrightLaunch::servedRuleVersion() and in the C++ port's
    # served_launch() alike.
    recorded = str(launch_model or "")
    for version in RULE_VERSIONS:
        if recorded == f"hipblaslt:{canonical_arch(str(arch))}:{version}":
            return version
    return rule_version(arch)


def hybrid_runtime_choice(arch: str) -> bool:
    """TensileLite lets Hybrid kernels switch to dynamic scheduling at run
    time only on gfx950."""
    return str(arch).split(":")[0] == "gfx950"


def queue_count(arch: str, num_xcd: Optional[int] = None) -> Optional[int]:
    """Queues of a dynamic-queue launch on `arch`: QUEUE_COUNTS', else the
    device's XCD count `num_xcd`; None when neither gives one."""
    known = QUEUE_COUNTS.get(canonical_arch(arch))
    if known is not None:
        return known
    return None if num_xcd is None else int(num_xcd)


def takes_dynamic_queue(
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    arch: str,
    schedules: Iterable[str] = ("default",),
    reduced_budget: bool = False,
) -> bool:
    """Whether `launch_plan` can give the kernel a dynamic-queue launch
    whose flag region it bounds, so needs `queue_count`, for some problem
    under the `schedules`, with budgets below the device's CUs when
    `reduced_budget`: a non-atomic StreamK kernel with DynamicWorkQueue, or
    with Hybrid where its dynamic sub-mode can run."""
    strategy, assign, atomic = launch_inputs(config_kwargs, attributes)[:3]
    if strategy != 2 or atomic:
        return False
    if assign == 1:
        return True
    return assign == 2 and (
        "dynamic" in set(schedules) or (hybrid_runtime_choice(arch) and reduced_budget)
    )


def _ceil_div(n: int, d: int) -> int:
    return 0 if d == 0 else -(-n // d)


def _hybrid_dynamic(
    schedule: str,
    runtime_choice: bool,
    tiles: int,
    sm_target: int,
    n_cu: int,
    occupancy: int,
) -> bool:
    """Whether a Hybrid kernel runs its dynamic sub-mode
    (streamK5EffectiveDynamic)."""
    if schedule == "dynamic":
        return True
    if schedule == "default" and sm_target == 0:
        return False
    if not runtime_choice or tiles <= MIN_TILES_FOR_DYNAMIC:
        return False
    available = min(sm_target, n_cu) if sm_target > 0 else n_cu
    if available == 0 or available >= n_cu:
        return False
    if occupancy <= MAX_OCCUPANCY_ALWAYS_DYNAMIC:
        return True
    return float(tiles) / float(available) > TILES_PER_CU_DYNAMIC


def _flag_entries(dynamic: bool, arch: str, num_xcd: Optional[int]) -> int:
    """Flags of the Stream-K fix-up a launch can index."""
    if not dynamic:
        return STREAM_K_FLAG_ELEMENTS
    queues = queue_count(arch, num_xcd)
    if queues is None:
        raise ValueError(
            f"the XCD count of the {arch} device is unknown, so the flags its "
            "dynamic-queue launches can index are unknown: set hardware.num_xcd, "
            "or read it from a device that runs as one partition"
        )
    prefix = queues * CACHE_LINE_BYTES // _FLAG_BYTES
    if prefix >= STREAM_K_FLAG_ELEMENTS:
        return STREAM_K_FLAG_ELEMENTS
    return STREAM_K_FLAG_ELEMENTS - prefix


def _static_grid(
    tiles: int,
    cu_count: int,
    n_cu: int,
    occupancy: int,
    iters: int,
    reduction: str,
    tile_bytes: int,
    batch: int,
    version: int,
    workspace: Optional[int] = None,
) -> int:
    """select_grid_size(k_split_aware) for a static persistent launch under
    rule `version`, with partial tiles of `tile_bytes` and a workspace of
    `workspace` bytes (None: one that holds every partial tile)."""
    grid = tiles
    if cu_count < n_cu:
        grid = min(grid, cu_count)
    if tiles > cu_count:
        virt = cu_count * occupancy if occupancy > 1 else cu_count
        min_even = max(1, tiles // virt)
        for frac in (0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0):
            g = int(float(tiles) / (float(min_even) + frac) + 0.5)
            if tiles % g != 0 and tile_bytes * g > MAX_PARTIAL_WORKSPACE:
                continue
            if g <= virt:
                grid = g
                break
    elif tiles < cu_count:
        if reduction == "parallel":
            grid = tiles * min(cu_count // tiles, iters // MIN_ITERS_PER_CU)
        else:
            for frac in (16, 12, 8, 6, 4, 3, 2, 1):
                if tiles * frac <= cu_count and iters // frac >= MIN_ITERS_PER_CU:
                    grid = tiles * frac
                    break
    # A grid that splits tiles needs their partial tiles in the workspace.
    if workspace is not None and grid > 0 and tiles % grid != 0:
        if tile_bytes * grid > workspace:
            grid = tiles
    if tiles <= cu_count or grid == 0 or grid >= tiles or batch > 1:
        return grid
    # A streaming grid becomes one workgroup per tile where that launch fills
    # its waves and the streaming one would split tiles. With one K iteration
    # per tile it splits none: there a grid of more than two workgroups per
    # CU becomes one workgroup per tile. Rule version 1 took one workgroup
    # per tile there for 2 to 32 tiles per CU instead, which differs from
    # TensileLite's grid; scripts/check_launch_plans.py counts such plans of
    # version 1 as a known class.
    if iters <= 1:
        if version >= 2:
            return tiles if grid > cu_count * 2 else grid
        return tiles if cu_count * 2 <= tiles < cu_count * 32 else grid
    dp_steps = _ceil_div(tiles, cu_count) * cu_count
    if float(tiles) < 0.8 * float(dp_steps):
        return grid
    return tiles if (tiles * iters // grid) % iters != 0 else grid


def _uniform_grid(
    g0: int,
    tiles: int,
    iters: int,
    per_tile_extra_iters: bool,
    reduction: str,
    partial_bytes: int,
    workspace: Optional[int],
    flag_region_binds: bool,
) -> int:
    """getPersistentGridImpl's snap of a static two-tile Stream-K grid `g0`
    under uniform summation order (PersistentLaunch::streamKUniformGrid):
    never upward. Above the tiles, the largest split factor F <= g0 / tiles
    that splits every tile evenly (unless the kernel redistributes per-tile
    extra iterations or the reduction is parallel), gives each workgroup at
    least MIN_ITERS_PER_CU of the `iters` (at least 1) per tile, fits its
    partial tiles in `workspace` and, when the flag region binds, stays in
    it; one workgroup per tile always does. Below the tiles, the tiles."""
    if tiles == 0 or g0 == 0 or g0 == tiles:
        return g0
    if g0 < tiles:
        return tiles
    for f in range(min(g0 // tiles, iters // MIN_ITERS_PER_CU), 1, -1):
        if reduction != "parallel" and not per_tile_extra_iters and iters % f != 0:
            continue
        if iters // f < MIN_ITERS_PER_CU:
            continue
        if workspace is not None and partial_bytes * tiles * f > workspace:
            continue
        if flag_region_binds and tiles * f > STREAM_K_FLAG_ELEMENTS:
            continue
        return tiles * f
    return tiles


def launch_inputs(
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    version: Optional[int] = None,
) -> Tuple[int, ...]:
    """(strategy, assignment, atomic, threads, occupancy) of a kernel as
    `launch_plan` uses them under rule `version` (default RULE_VERSION),
    followed by 1 for a hand-written Stream-K kernel that can run on a
    static grid (any assignment but DynamicWorkQueue), whose partial tiles
    TensileLite reduces in a tree only: what it reads of
    LAUNCH_CONFIG_FIELDS and LAUNCH_ATTRIBUTES, with the values it does not
    distinguish made equal (a non-persistent kernel launches one workgroup
    per tile whatever the rest, DataParallel ignores the atomic and
    hand-written flags, a dynamic queue the hand-written one).

    Under version 3 a Stream-K kernel has nine, the five, then whether it is
    tree only, whether it redistributes per-tile extra iterations (0 for a
    dynamic queue, which takes whole tiles), its workspace_size_per_elem_c
    and whether it has a static uniform-summation-order obstacle: what it
    reads of LAUNCH_ATTRIBUTES_V3. ValueError for such a kernel without
    workspace_size_per_elem_c.

    Under version 4 a kernel of the spatial-cluster layout has thirteen: the
    nine, tree only also where its cluster_tree_only says so, then the
    layout, its cluster's x and y and its sk_k_granularity, what it reads of
    LAUNCH_ATTRIBUTES_V4; every other kernel has its version 3 inputs."""
    version = RULE_VERSION if version is None else int(version)
    attrs = attributes or {}
    strategy = int(attrs.get("tile_processing_strategy", 0))
    if strategy not in (1, 2):
        return (0, 0, 0, 0, 0)
    threads = 1
    for name in ("work_group_x", "work_group_y", "work_group_z"):
        threads *= max(int(attrs.get(name, 0)), 0)
    assign = int(attrs.get("work_assignment", 0))
    atomic = int(attrs.get("stream_k_atomic", 0)) != 0 and strategy == 2
    inputs = (
        strategy,
        assign,
        int(atomic),
        threads,
        max(int(config_kwargs.get("occupancy", 1)), 1),
    )
    tree_only = strategy == 2 and assign != 1 and int(attrs.get("hand_written", 0)) != 0
    if version >= 3 and strategy == 2:
        if attrs.get("workspace_size_per_elem_c") is None:
            raise ValueError(
                "a Stream-K kernel without workspace_size_per_elem_c (its "
                "sizeMapping's workspaceSizePerElemC) has no launch under rule "
                "version 3: read its attributes from the library's .dat"
            )
        per_tile = assign != 1 and int(attrs.get("per_tile_extra_iters", 0)) != 0
        # An atomic kernel's first static obstacle is its atomic fix-up.
        obstacle = atomic or int(attrs.get("uso_static_obstacle", 0)) != 0
        cluster = version >= 4 and spatial_cluster_kernel(attrs)
        if cluster and int(attrs.get("cluster_tree_only", 0)) != 0:
            tree_only = True
        inputs += (
            int(tree_only),
            int(per_tile),
            int(attrs["workspace_size_per_elem_c"]),
            int(obstacle),
        )
        if not cluster:
            return inputs
        return inputs + (
            SPATIAL_CLUSTER_LAYOUT,
            int(attrs.get("cluster_dim_x", 1)),
            int(attrs.get("cluster_dim_y", 1)),
            max(int(attrs.get("sk_k_granularity", 1)), 1),
        )
    if tree_only:
        return inputs + (1,)
    return inputs


# Kernel attributes (`lib.dat.kernel_dat_info`, from the sizeMapping's
# clusterDim) whose product is the number of workgroups of one cluster.
CLUSTER_ATTRIBUTES = ("cluster_dim_x", "cluster_dim_y")


def persistent_cluster_kernel(attributes: Optional[Mapping[str, int]]) -> bool:
    """Whether a kernel is a persistent kernel (DataParallel or Stream-K)
    launched in clusters, whose ClusterDim spans more than one workgroup
    (the test of ContractionSolution::followsDefaultPersistentLaunch):
    TensileLite plans its launch with cluster rules `launch_plan` does not
    model. A non-persistent kernel launches one workgroup per tile with or
    without a cluster. An absent cluster dimension is 1, TensileLite's
    default."""
    attrs = attributes or {}
    if int(attrs.get("tile_processing_strategy", 0)) not in (1, 2):
        return False
    workgroups = 1
    for name in CLUSTER_ATTRIBUTES:
        workgroups *= int(attrs.get(name, 1))
    return workgroups > 1


# The persistent loop args version of TensileLite's spatial-cluster Stream-K
# argument layout (ContractionSolution::usesStreamKSpatialCluster()).
SPATIAL_CLUSTER_LAYOUT = 2
# The cluster shapes that layout launches (validatePersistentLoopArgs): x and
# y powers of two, x below MAX_SPATIAL_CLUSTER_X, y at most
# MAX_SPATIAL_CLUSTER_Y, z 1, and 2 to MAX_SPATIAL_CLUSTER workgroups.
MAX_SPATIAL_CLUSTER_X = 16
MAX_SPATIAL_CLUSTER_Y = 8
MAX_SPATIAL_CLUSTER = 16


def spatial_cluster_kernel(attributes: Optional[Mapping[str, int]]) -> bool:
    """Whether a kernel is a Stream-K kernel of TensileLite's spatial-cluster
    argument layout, which rule version 4 plans: persistent loop args version
    SPATIAL_CLUSTER_LAYOUT, which TensileLite's build gives every generated
    static Stream-K kernel launched in clusters whatever its logic file
    says."""
    attrs = attributes or {}
    return (
        int(attrs.get("tile_processing_strategy", 0)) == 2
        and int(attrs.get("persistent_loop_args_version", 0)) == SPATIAL_CLUSTER_LAYOUT
    )


def spatial_cluster_errors(attributes: Optional[Mapping[str, int]]) -> List[str]:
    """Why TensileLite does not launch a kernel of the spatial-cluster layout
    (ContractionSolution::validatePersistentLoopArgs), empty when it does:
    a static non-atomic Stream-K kernel without PrefetchAcrossPersistent, of
    a supported cluster shape, not written by hand and without the argument
    version or problem type terms attribute `cluster_unsupported` stands
    for."""
    attrs = attributes or {}
    errs = []
    if (
        int(attrs.get("work_assignment", 0)) != 0
        or int(attrs.get("stream_k_atomic", 0)) != 0
        or int(attrs.get("prefetch_across_persistent", 0)) != 0
    ):
        errs.append("not static non-atomic Stream-K without PrefetchAcrossPersistent")
    x, y, z = (int(attrs.get(f"cluster_dim_{a}", 1)) for a in "xyz")
    if not (
        z == 1
        and 0 < x < MAX_SPATIAL_CLUSTER_X
        and 0 < y <= MAX_SPATIAL_CLUSTER_Y
        and x & (x - 1) == 0
        and y & (y - 1) == 0
        and 2 <= x * y <= MAX_SPATIAL_CLUSTER
    ):
        errs.append(f"ClusterDim [{x}, {y}, {z}] is no spatial cluster it launches")
    if int(attrs.get("hand_written", 0)) != 0:
        errs.append("written by hand")
    if int(attrs.get("cluster_unsupported", 0)) != 0:
        errs.append("an argument version or problem type the layout does not take")
    return errs


def launch_key(
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    version: Optional[int] = None,
) -> int:
    """The kernel's launch class under rule `version` (default RULE_VERSION)
    as a tilewright launch key: kernels with equal signature and equal key
    get equal `launch_plan`s for every problem, context, device and arch.
    Versions 1 and 2 share one key."""
    return fs.view_key_hash(launch_inputs(config_kwargs, attributes, version))


def launch_plan(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    context: fs.ExecutionContext,
    n_cu: int,
    arch: str,
    num_xcd: Optional[int] = None,
    version: Optional[int] = None,
) -> fs.LaunchPlan:
    """hipBLASLt's launch of a kernel (`config_kwargs` with mt_m/n/k and
    occupancy, `attributes` with LAUNCH_ATTRIBUTES) for a problem under
    `context` on a device with `n_cu` CUs and `num_xcd` XCDs of `arch`, under
    rule `version` (default `rule_version(arch)`; from version 3 on, under
    the context's uniform summation order and workspace, with the
    attributes of LAUNCH_ATTRIBUTES_V3; from version 4 on, a kernel of the
    spatial-cluster layout with those of LAUNCH_ATTRIBUTES_V4 as
    `_spatial_cluster_plan`). A tile whose partial results atomic
    Stream-K kernels add up reports "atomic"; dynamic-queue and DataParallel
    launches report "none". ValueError for a kernel TensileLite does not
    load (DataParallel with a work queue), for a dynamic-queue launch whose
    flag region depends on a `queue_count` that is unknown, for a version
    not in RULE_VERSIONS, for a version 3 Stream-K kernel without
    workspace_size_per_elem_c and where `_spatial_cluster_plan` raises."""
    version = rule_version(arch) if version is None else int(version)
    if version not in RULE_VERSIONS:
        raise ValueError(
            f"launch rule version {version} is not one of {list(RULE_VERSIONS)}"
        )
    inputs = launch_inputs(config_kwargs, attributes, version)
    strategy, assign, atomic_flag, threads, occupancy = inputs[:5]
    tree_only = inputs[5:6] == (1,)
    if strategy == 1 and assign != 0:
        raise ValueError(
            "a DataParallel kernel takes WorkAssignment StaticGrid only; "
            "TensileLite does not load one with work_assignment "
            f"{assign}"
        )
    atomic = atomic_flag != 0
    mt_k = max(int(config_kwargs.get("mt_k", 1)), 1)
    mt_m = max(int(config_kwargs.get("mt_m", 1)), 1)
    mt_n = max(int(config_kwargs.get("mt_n", 1)), 1)
    k = int(problem_kwargs["k"])
    batch = int(problem_kwargs.get("batch", 1))
    n_cu = int(n_cu)

    tiles = fs.output_tiles(problem_kwargs, config_kwargs)
    # None (0) launches one workgroup per tile; DataParallel (1) and StreamK
    # (2) are persistent.
    if strategy not in (1, 2) or n_cu == 0:
        return fs.LaunchPlan(False, tiles, "none")
    sm_target = int(context.cu_budget) // SM_TARGET_GRANULE * SM_TARGET_GRANULE
    cu_count = sm_target if 0 < sm_target < n_cu else n_cu
    iters = _ceil_div(k, mt_k)
    # DynamicWorkQueue (1) always takes tiles from a queue; Hybrid (2) does in
    # its dynamic sub-mode.
    dynamic = assign == 1 or (
        assign == 2
        and _hybrid_dynamic(
            context.schedule,
            hybrid_runtime_choice(arch),
            tiles,
            sm_target,
            n_cu,
            occupancy,
        )
    )
    selected = "tree"
    if (
        not tree_only
        and tiles < cu_count
        and max(iters, 1) >= 64
        and tiles <= cu_count // 4
    ):
        selected = "parallel"
    # From version 3 on, the call's uniform summation order and workspace
    # (None: one that holds every partial tile), and a Stream-K kernel's
    # partial tiles of its own element size.
    uso = version >= 3 and bool(context.uniform_summation_order)
    workspace = context.workspace_bytes if version >= 3 else None
    v3_stream_k = version >= 3 and strategy == 2
    if v3_stream_k:
        per_tile_extra_iters, obstacle = inputs[6] != 0, inputs[8] != 0
        partial_bytes = mt_m * mt_n * inputs[7]
        if dynamic:
            selected = "tree"
        elif uso and not (selected == "parallel" and not atomic and not obstacle):
            # Static two-tile packing keeps parallel only for a non-atomic
            # kernel without a static obstacle (getSKReduction).
            selected = "tree"

    grid = tiles
    if dynamic:
        grid = min(tiles, n_cu * min(occupancy, MAX_DYNAMIC_OCCUPANCY))
    elif tiles > 0:
        mi_dtype = fs.normalize_dtype_name(problem_kwargs.get("mi_dtype", "bf16"))
        grid = _static_grid(
            tiles,
            cu_count,
            n_cu,
            occupancy,
            iters,
            "tree" if strategy == 1 else selected,
            (
                partial_bytes
                if v3_stream_k
                else mt_m * mt_n * _WIDE_COMPUTE_BYTES.get(mi_dtype, 4)
            ),
            batch,
            version,
            workspace,
        )
        if len(inputs) > 9:
            # The selected grid of a spatial cluster is the selection itself
            # (getPersistentGridImpl), which its schedule partitions.
            return _spatial_cluster_plan(
                problem_kwargs,
                config_kwargs,
                inputs,
                selected,
                grid,
                n_cu,
                workspace,
                uso,
            )
        if v3_stream_k and uso:
            grid = _uniform_grid(
                grid,
                tiles,
                max(iters, 1),
                per_tile_extra_iters,
                selected,
                partial_bytes,
                workspace,
                not atomic and selected != "parallel",
            )
    if strategy == 1:
        # Work-item counts are 32-bit.
        max_grid = (_U32_ITEMS - 1) // max(threads, 1)
        if grid > max_grid:
            grid = min(n_cu * occupancy, max_grid)
        return fs.LaunchPlan(dynamic, grid, "none")
    if selected == "tree" and grid > 0:
        max_tiles = _U32_ITEMS // max(threads, 1)
        if (
            iters >= MAX_TREE_ITERS
            or tiles * iters // grid >= MAX_TREE_ITERS
            or tiles * iters >= max_tiles
        ):
            grid = n_cu * occupancy if tiles >= max_tiles else tiles
    partial = grid > 0 and tiles % grid != 0
    if not atomic and selected != "parallel" and partial:
        grid = min(grid, _flag_entries(dynamic, arch, num_xcd))
    if v3_stream_k:
        # A parallel reduction needs two workgroups per tile
        # (streamKReconcileReduction).
        if selected == "parallel" and (tiles == 0 or grid // tiles < 2):
            selected = "tree"
        # A launch that writes partial tiles the workspace cannot hold runs
        # one workgroup per tile with a tree reduction (resolveStreamKSettings).
        straddles_tiles = grid > 0 and max(iters, 1) > 1 and tiles % grid != 0
        if (
            workspace is not None
            and grid > 0
            and (selected == "parallel" or straddles_tiles)
            and partial_bytes * grid > workspace
        ):
            selected, grid = "tree", tiles
    reduction = "none"
    # With one K iteration per tile every workgroup ends on a tile's edge, so
    # no tile has partial results (ContractionSolution.cpp
    # streamKStraddlesTiles); rule version 1 reported a reduction there too.
    straddles = iters > 1 or version < 2
    if not dynamic and grid > 0 and tiles % grid != 0 and straddles:
        reduction = "atomic" if atomic else selected
    return fs.LaunchPlan(dynamic, grid, reduction)


def _static_split_row_uniform(
    tiles: int, iters: int, grid: int, per_tile_extra_iters: bool
) -> bool:
    """Whether every tile of a tree static two-tile Stream-K split of `grid`
    workgroups over `tiles` tiles of `iters` (at least 1) iterations folds
    its partials alike under uniform summation order, on a device with the
    default one full Stream-K tile per workgroup (streamKStaticSplit and
    streamKStaticSplitRowUniform)."""
    sk_tiles = grid
    if tiles % grid != 0:
        sk_tiles = min(grid + tiles % grid if tiles > grid else tiles, tiles)
    per_wg = sk_tiles * iters // grid
    extra = sk_tiles * iters - per_wg * grid
    if sk_tiles == 0:
        return True
    if per_wg != 0 and extra == 0 and sk_tiles == tiles and iters % per_wg == 0:
        return True
    return (
        per_tile_extra_iters
        and sk_tiles == tiles
        and tiles != 0
        and grid % tiles == 0
        and per_wg != 0
    )


def _padded_iters(iters: int, k_unit: int) -> int:
    """skPaddedItersPerTile: `iters` rounded up to whole units of K."""
    if k_unit <= 1 or iters == 0:
        return iters
    return _ceil_div(iters, k_unit) * k_unit


def _static_split(
    tiles: int, iters: int, grid: int, full_tiles: int, k_unit: int
) -> Tuple[int, int, int]:
    """streamKStaticSplit: (Stream-K tiles, iterations per workgroup, extra
    iterations) of a static two-tile Stream-K split of `grid` workgroups over
    `tiles` tiles of `iters` iterations, with `full_tiles` full Stream-K
    tiles per workgroup and K handed out in units of `k_unit` iterations."""
    if grid == 0:
        return (0, 0, 0)
    sk_tiles = grid
    if tiles % grid != 0:
        sk_tiles = grid * full_tiles + tiles % grid if tiles > grid else tiles
        sk_tiles = min(sk_tiles, tiles)
    unit = max(k_unit, 1)
    sk_iters = sk_tiles * _padded_iters(iters, unit)
    per_wg = unit * (sk_iters // unit // grid)
    return (sk_tiles, per_wg, sk_iters - per_wg * grid)


def _static_split_uniform(
    split: Tuple[int, int, int],
    tiles: int,
    iters: int,
    grid: int,
    per_tile_extra_iters: bool,
    k_unit: int,
) -> bool:
    """streamKStaticSplitRowUniform under uniform summation order: whether
    every tile of `split` folds its partials alike."""
    sk_tiles, per_wg, extra = split
    if sk_tiles == 0:
        return True
    if (
        per_wg != 0
        and extra == 0
        and sk_tiles == tiles
        and _padded_iters(iters, k_unit) % per_wg == 0
    ):
        return True
    return (
        per_tile_extra_iters
        and sk_tiles == tiles
        and tiles != 0
        and grid % tiles == 0
        and per_wg != 0
    )


def _spatial_blocks(
    problem_kwargs: Mapping[str, Any], config_kwargs: Mapping[str, Any], inputs
) -> int:
    """The spatial blocks of a launch of a kernel of the spatial-cluster
    layout with launch `inputs` (spatialClusterGeometry): its clusters'
    shares of the output tiles, a partial one at each edge."""
    mt_m = max(int(config_kwargs.get("mt_m", 1)), 1)
    mt_n = max(int(config_kwargs.get("mt_n", 1)), 1)
    blocks_m = _ceil_div(_ceil_div(int(problem_kwargs["m"]), mt_m), inputs[10])
    blocks_n = _ceil_div(_ceil_div(int(problem_kwargs["n"]), mt_n), inputs[11])
    return blocks_m * blocks_n * int(problem_kwargs.get("batch", 1))


def _spatial_cluster_plan(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    inputs: Tuple[int, ...],
    selected: str,
    selected_grid: int,
    n_cu: int,
    workspace: Optional[int],
    uso: bool,
) -> fs.LaunchPlan:
    """resolveClusteredStreamKSettings: the launch of a kernel of the
    spatial-cluster layout, with launch `inputs` (`launch_inputs`, version
    4), whose reduction getSKReduction selects as `selected` and whose grid
    getPersistentGridImpl selects as `selected_grid` workgroups over the
    output tiles, on a device of `n_cu` CUs, under the default device
    settings and a `workspace` of that many bytes (None: one that holds
    every partial). Its schedule launches a grid of logical clusters over
    spatial blocks, which it reports in physical workgroups, the cluster's
    workgroups each; its reduction is the schedule's, "none" where every
    cluster takes whole blocks. A parallel reduction's partials are of D's
    compute type, whose element a generated Stream-K kernel's
    workspace_size_per_elem_c is. ValueError where TensileLite throws: a
    workgroup of no threads, a cluster the device's physical grid limits do
    not hold, and partial tiles of no bytes."""
    threads, occupancy = inputs[3], inputs[4]
    per_tile_extra_iters = inputs[6] != 0
    elem_bytes = inputs[7]
    cluster = inputs[10] * inputs[11]
    k_unit = inputs[12]
    mt_m = max(int(config_kwargs.get("mt_m", 1)), 1)
    mt_n = max(int(config_kwargs.get("mt_n", 1)), 1)
    partial_bytes = mt_m * mt_n * elem_bytes
    tiles = fs.output_tiles(problem_kwargs, config_kwargs)
    blocks = _spatial_blocks(problem_kwargs, config_kwargs, inputs)
    k = int(problem_kwargs["k"])
    iters = max(_ceil_div(k, max(int(config_kwargs.get("mt_k", 1)), 1)), 1)
    if blocks == 0:
        return fs.LaunchPlan(False, 0, "none")
    # The selection counts workgroups over tiles; a split of it pads to whole
    # blocks, and a launch of more workgroups than tiles keeps one cluster
    # per block.
    whole_tile = selected_grid == tiles
    budget = blocks * cluster if whole_tile else selected_grid
    if selected_grid > tiles:
        if selected == "parallel":
            budget = blocks * cluster * min(selected_grid // tiles, iters)
        else:
            budget = max(budget, blocks * cluster)
    if threads == 0:
        raise ValueError("a spatial cluster of workgroups of no threads")
    launch_limit = (_U32_ITEMS - 1) // threads
    # Whole blocks publish no partials and use no flags.
    max_physical = launch_limit
    if not whole_tile:
        max_physical = min(launch_limit, STREAM_K_FLAG_ELEMENTS, n_cu * occupancy)
    if max_physical < cluster:
        raise ValueError(
            f"a spatial cluster of {cluster} workgroups does not fit the "
            f"{max_physical} workgroups the device's grid limits leave it"
        )
    grid = min(max(budget, cluster) // cluster, max_physical // cluster, blocks * iters)

    def whole_blocks(grid: int) -> fs.LaunchPlan:
        return fs.LaunchPlan(False, min(grid, blocks) * cluster, "none")

    if selected == "parallel" and k != 0:
        splits = min(grid // blocks, iters)
        if splits >= 2:
            grid = blocks * splits
            output_bytes = (
                int(problem_kwargs["m"])
                * int(problem_kwargs["n"])
                * int(problem_kwargs.get("batch", 1))
                * elem_bytes
            )
            if workspace is not None and output_bytes * splits > workspace:
                return whole_blocks(grid)
            return fs.LaunchPlan(False, grid * cluster, "parallel")
    if k == 0 or whole_tile:
        return whole_blocks(grid)
    split = _static_split(
        blocks, iters, grid, min(SK_FULL_TILES, blocks // grid), k_unit
    )
    if uso and not _static_split_uniform(
        split, blocks, iters, grid, per_tile_extra_iters, k_unit
    ):
        # Equal K chunks of every block, never more clusters.
        splits = grid // blocks
        while splits >= 2 and (
            iters % splits != 0 or iters // splits < MIN_ITERS_PER_CU
        ):
            splits -= 1
        if splits < 2:
            return whole_blocks(grid)
        grid = blocks * splits
        split = _static_split(blocks, iters, grid, 0, k_unit)
    _, per_wg, extra = split
    if extra == 0 and per_wg % iters == 0:
        return whole_blocks(grid)
    if (
        iters >= MAX_TREE_ITERS
        or per_wg >= MAX_TREE_ITERS
        or blocks * iters >= MAX_TREE_DIVIDEND
    ):
        return whole_blocks(grid)
    if partial_bytes == 0:
        raise ValueError(
            "the partial tiles of a spatial cluster need its "
            "workspace_size_per_elem_c"
        )
    if workspace is not None and grid * cluster * partial_bytes > workspace:
        return whole_blocks(grid)
    return fs.LaunchPlan(False, grid * cluster, "tree")


def uso_obstacle(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    context: fs.ExecutionContext,
    plan: fs.LaunchPlan,
) -> Optional[str]:
    """The token of TensileLite's refusal of a kernel for a call with
    uniform summation order (ContractionSolution::
    uniformSummationOrderLaunchObstacle), None when the call has none or the
    kernel passes: of the clauses the kernel's attributes and its rule
    version 3 `plan` (version 4 for a kernel of the spatial-cluster layout)
    decide, CustomKernel (a kernel written by hand that
    does not declare the order, attribute `uso_declared`), the static
    obstacles (USO_STATIC_OBSTACLES), StreamKGridZero,
    ParallelReductionNotRowUniform, StaticSplitNotRowUniform and
    CompiledInStaggerU (one written by hand with a StaggerU), under the
    default device settings. The accumulation clause (auto-GSU and
    GlobalAccumulation), ResolvedStaggerU and the task predicate's
    workspace check read what a kernel's attributes do not hold; the bench,
    whose isAlgoSupported applies the same predicate, is the ground truth
    for them. ValueError for an unknown obstacle position."""
    if not context.uniform_summation_order:
        return None
    attrs = attributes or {}
    hand_written = int(attrs.get("hand_written", 0)) != 0
    if hand_written and int(attrs.get("uso_declared", 0)) == 0:
        return "CustomKernel"
    if int(attrs.get("tile_processing_strategy", 0)) == 2:
        token = _stream_k_uso_obstacle(problem_kwargs, config_kwargs, attrs, plan)
        if token is not None:
            return token
    if hand_written and int(attrs.get("stagger_u", 0)) != 0:
        return "CompiledInStaggerU"
    return None


def _stream_k_uso_obstacle(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    attrs: Mapping[str, int],
    plan: fs.LaunchPlan,
) -> Optional[str]:
    """The clauses of `uso_obstacle` for a Stream-K kernel."""
    position = int(attrs.get("uso_static_obstacle", 0))
    if position:
        if not 0 < position <= len(USO_STATIC_OBSTACLES):
            raise ValueError(f"uso_static_obstacle {position} names no obstacle")
        return USO_STATIC_OBSTACLES[position - 1]
    if int(attrs.get("stream_k_atomic", 0)) != 0:
        return USO_STATIC_OBSTACLES[0]
    grid = int(plan.grid_wgs)
    if grid == 0:
        return "StreamKGridZero"
    if spatial_cluster_kernel(attrs):
        return _spatial_cluster_uso_obstacle(problem_kwargs, config_kwargs, attrs, plan)
    tiles = fs.output_tiles(problem_kwargs, config_kwargs)
    if plan.partial_reduction == "parallel":
        # Parallel splits every tile over grid / tiles >= 2 workgroups of a
        # static two-tile launch, which is row-uniform.
        return None
    if plan.dynamic_queue:
        return None
    iters = max(
        _ceil_div(int(problem_kwargs["k"]), max(int(config_kwargs["mt_k"]), 1)), 1
    )
    per_tile = int(attrs.get("per_tile_extra_iters", 0)) != 0
    if not _static_split_row_uniform(tiles, iters, grid, per_tile):
        return "StaticSplitNotRowUniform"
    return None


def _spatial_cluster_uso_obstacle(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    attrs: Mapping[str, int],
    plan: fs.LaunchPlan,
) -> Optional[str]:
    """The row-uniformity clauses of `uso_obstacle` for a kernel of the
    spatial-cluster layout and its version 4 `plan`, which TensileLite
    tests on the split of its cluster schedule over spatial blocks: a
    parallel reduction splits every block alike and whole blocks split
    none. The split of a tree reduction is that of the plan's clusters,
    whose full Stream-K tiles only a grid below the blocks has."""
    if plan.partial_reduction != "tree":
        return None
    inputs = launch_inputs(config_kwargs, attrs, 4)
    grid = int(plan.grid_wgs) // (inputs[10] * inputs[11])
    blocks = _spatial_blocks(problem_kwargs, config_kwargs, inputs)
    mt_k = max(int(config_kwargs.get("mt_k", 1)), 1)
    iters = max(_ceil_div(int(problem_kwargs["k"]), mt_k), 1)
    split = _static_split(
        blocks, iters, grid, min(SK_FULL_TILES, blocks // grid), inputs[12]
    )
    if not _static_split_uniform(
        split, blocks, iters, grid, inputs[6] != 0, inputs[12]
    ):
        return "StaticSplitNotRowUniform"
    return None


def launch_arch(library_stem: str) -> str:
    """Arch of the launch model that ranks the TensileLite library
    `library_stem` (a stem or a path to its file): the last '_' part of the
    stem, which TensileLite reads, as the compiler target whose ISA it runs.
    Every library one model deploys to (gfx1250-strict's stems end in
    _gfx1250-strict, gfx1250's in _gfx1250) then names the same model."""
    stem = os.path.basename(str(library_stem)).split(".", 1)[0]
    _, sep, suffix = stem.rpartition("_")
    if not sep or not suffix:
        raise ValueError(f"library stem {library_stem!r} has no _<arch> suffix")
    return canonical_arch(suffix)


class HipblasltLaunch:
    """`features.Catalog.launch` of hipBLASLt kernels on `arch`, recorded as
    its compiler target, with their `launch_key`, on a device of `num_xcd`
    XCDs (None: unknown, which only a dynamic-queue launch needs), under rule
    `version` (default `rule_version(arch)`; ValueError for one not in
    RULE_VERSIONS)."""

    def __init__(
        self, arch: str, num_xcd: Optional[int] = None, version: Optional[int] = None
    ) -> None:
        self.arch = canonical_arch(str(arch))
        self.num_xcd = None if num_xcd is None else int(num_xcd)
        self.version = rule_version(self.arch) if version is None else int(version)
        if self.version not in RULE_VERSIONS:
            raise ValueError(
                f"launch rule version {self.version} is not one of "
                f"{list(RULE_VERSIONS)}"
            )

    @classmethod
    def for_library(
        cls, library_stem: str, num_xcd: Optional[int] = None
    ) -> "HipblasltLaunch":
        """The launch of the kernels of the library `library_stem`, under the
        rule version its models train on."""
        return cls(launch_arch(library_stem), num_xcd)

    @classmethod
    def for_model(
        cls, arch: str, launch_model: Optional[str], num_xcd: Optional[int] = None
    ) -> "HipblasltLaunch":
        """The launch that ranks a model of `arch` recording the launch model
        `launch_model` (`served_rule_version`)."""
        return cls(arch, num_xcd, served_rule_version(arch, launch_model))

    @property
    def id(self) -> str:
        """The launch model's id (tilewright LaunchModel::id) that MLREC_v3
        models trained with these plans record."""
        return f"hipblaslt:{self.arch}:{self.version}"

    def __call__(
        self,
        problem_kwargs: Mapping[str, Any],
        config_kwargs: Mapping[str, Any],
        attributes: Optional[Mapping[str, int]],
        context: fs.ExecutionContext,
        n_cu: int,
    ) -> fs.LaunchPlan:
        return launch_plan(
            problem_kwargs,
            config_kwargs,
            attributes,
            context,
            n_cu,
            self.arch,
            self.num_xcd,
            self.version,
        )

    def launch_key(
        self,
        config_kwargs: Mapping[str, Any],
        attributes: Optional[Mapping[str, int]],
    ) -> int:
        return launch_key(config_kwargs, attributes, self.version)

    def __repr__(self) -> str:
        args = [repr(self.arch)]
        if self.num_xcd is not None:
            args.append(f"num_xcd={self.num_xcd}")
        if self.version != rule_version(self.arch):
            args.append(f"version={self.version}")
        return f"HipblasltLaunch({', '.join(args)})"
