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
getSKReduction, getPersistentGridImpl) and the Stream-K arithmetic it calls
(select_hybrid_mode, select_reduction, select_grid_size with
k_split_aware), with the default TENSILE_PERSISTENT_* settings, no uniform
summation order and a workspace that holds every partial tile. The SM target
is `context.cu_budget` rounded down to a multiple of 32 (ContractionProblem.hpp
setSmCountTarget without TENSILE_PRECISE_SM_TARGET); 0 means none.

TensileLite launches a hand-written kernel (attribute `hand_written`: its
logic file's customKernel names a kernel that was not generated) through
resolvePersistentSettings, whose getSKReduction gives a hand-written Stream-K
kernel the tree reduction whatever select_reduction would choose, so that
its static grid is select_grid_size's for the tree; its other choices are a
generated kernel's.

A persistent kernel, DataParallel or Stream-K, launched in clusters
(`persistent_cluster_kernel`: its ClusterDim spans more than one workgroup)
is outside the rule: TensileLite leaves it out of the default persistent
launch (ContractionSolution::followsDefaultPersistentLaunch) and plans its
grid with cluster rules this module does not model, so a v3 run refuses a
pool that holds one (run_orchestrator.py preflight).
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Tuple

from . import features as fs
from .arch import canonical_arch

# Version of TensileLite's launch rule, part of `HipblasltLaunch.id`, which
# models record and the engine checks. Bump it when that rule changes, or when
# a change to `launch_plan` would make models trained before it see other
# plans than TensileLite gives them, so that those models are refused instead
# of ranked with plans they did not see. A fix that brings launch_plan closer
# to TensileLite does not count: TensileLite already gave those plans.
RULE_VERSION = 1

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
# Config fields the rule reads besides the signature's tile sizes.
LAUNCH_CONFIG_FIELDS = ("occupancy",)
# Signature fields the rule reads.
LAUNCH_SIGNATURE_FIELDS = ("mt_m", "mt_n", "mt_k")

SM_TARGET_GRANULE = 32
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
) -> int:
    """select_grid_size(k_split_aware) for a static persistent launch."""
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
    if tiles <= cu_count or grid == 0 or grid >= tiles or batch > 1:
        return grid
    # A streaming grid becomes one workgroup per tile where that launch fills
    # its waves and the streaming one would split tiles. With one K iteration
    # per tile this is rule version 1's window: TensileLite takes the grid
    # Origami's select_grid_size chooses there, which can differ, and
    # scripts/check_launch_plans.py counts such plans as a known class.
    if iters <= 1:
        return tiles if cu_count * 2 <= tiles < cu_count * 32 else grid
    dp_steps = _ceil_div(tiles, cu_count) * cu_count
    if float(tiles) < 0.8 * float(dp_steps):
        return grid
    return tiles if (tiles * iters // grid) % iters != 0 else grid


def launch_inputs(
    config_kwargs: Mapping[str, Any], attributes: Optional[Mapping[str, int]]
) -> Tuple[int, ...]:
    """(strategy, assignment, atomic, threads, occupancy) of a kernel as
    `launch_plan` uses them, followed by 1 for a hand-written Stream-K
    kernel that can run on a static grid (any assignment but
    DynamicWorkQueue), whose partial tiles TensileLite reduces in a tree
    only: what it reads of LAUNCH_CONFIG_FIELDS and LAUNCH_ATTRIBUTES, with
    the values it does not distinguish made equal (a non-persistent kernel
    launches one workgroup per tile whatever the rest, DataParallel ignores
    the atomic and hand-written flags, a dynamic queue the hand-written
    one)."""
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
    if strategy == 2 and assign != 1 and int(attrs.get("hand_written", 0)) != 0:
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


def launch_key(
    config_kwargs: Mapping[str, Any], attributes: Optional[Mapping[str, int]]
) -> int:
    """The kernel's launch class as a tilewright launch key: kernels with
    equal signature and equal key get equal `launch_plan`s for every
    problem, context, device and arch."""
    return fs.view_key_hash(launch_inputs(config_kwargs, attributes))


def launch_plan(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    context: fs.ExecutionContext,
    n_cu: int,
    arch: str,
    num_xcd: Optional[int] = None,
) -> fs.LaunchPlan:
    """hipBLASLt's launch of a kernel (`config_kwargs` with mt_m/n/k and
    occupancy, `attributes` with LAUNCH_ATTRIBUTES) for a problem under
    `context` on a device with `n_cu` CUs and `num_xcd` XCDs of `arch`. A
    tile whose partial results atomic Stream-K kernels add up reports
    "atomic"; dynamic-queue and DataParallel launches report "none".
    ValueError for a kernel TensileLite does not load (DataParallel with a
    work queue) and for a dynamic-queue launch whose flag region depends on
    a `queue_count` that is unknown."""
    inputs = launch_inputs(config_kwargs, attributes)
    strategy, assign, atomic_flag, threads, occupancy = inputs[:5]
    tree_only = inputs[5:] == (1,)
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
            mt_m * mt_n * _WIDE_COMPUTE_BYTES.get(mi_dtype, 4),
            batch,
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
    reduction = "none"
    if not dynamic and grid > 0 and tiles % grid != 0:
        reduction = "atomic" if atomic else selected
    return fs.LaunchPlan(dynamic, grid, reduction)


def launch_arch(library_stem: str) -> str:
    """Arch of the launch model that ranks the TensileLite library
    `library_stem` (a stem or a path to its file): the last '_' part of the
    stem, which TensileLite reads, as the compiler target whose ISA it runs.
    Every library one model deploys to (gfx1250-strict's stems end in
    _gfx1250-strict, those of hipBLASLt's gfx1250v0 subtree in _gfx1250)
    then names the same model."""
    stem = os.path.basename(str(library_stem)).split(".", 1)[0]
    _, sep, suffix = stem.rpartition("_")
    if not sep or not suffix:
        raise ValueError(f"library stem {library_stem!r} has no _<arch> suffix")
    return canonical_arch(suffix)


class HipblasltLaunch:
    """`features.Catalog.launch` of hipBLASLt kernels on `arch`, recorded as
    its compiler target, with their `launch_key`, on a device of `num_xcd`
    XCDs (None: unknown, which only a dynamic-queue launch needs)."""

    def __init__(self, arch: str, num_xcd: Optional[int] = None) -> None:
        self.arch = canonical_arch(str(arch))
        self.num_xcd = None if num_xcd is None else int(num_xcd)

    @classmethod
    def for_library(
        cls, library_stem: str, num_xcd: Optional[int] = None
    ) -> "HipblasltLaunch":
        """The launch of the kernels of the library `library_stem`."""
        return cls(launch_arch(library_stem), num_xcd)

    @property
    def id(self) -> str:
        """The launch model's id (tilewright LaunchModel::id) that MLREC_v3
        models trained with these plans record."""
        return f"hipblaslt:{self.arch}:{RULE_VERSION}"

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
        )

    def launch_key(
        self,
        config_kwargs: Mapping[str, Any],
        attributes: Optional[Mapping[str, int]],
    ) -> int:
        return launch_key(config_kwargs, attributes)

    def __repr__(self) -> str:
        if self.num_xcd is None:
            return f"HipblasltLaunch({self.arch!r})"
        return f"HipblasltLaunch({self.arch!r}, num_xcd={self.num_xcd})"
