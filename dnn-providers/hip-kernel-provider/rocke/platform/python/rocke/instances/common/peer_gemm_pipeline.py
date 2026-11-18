# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only fused/unfused peer GEMM orchestration using existing kernels.

No device SSA is emitted here. Unfused gather copies, local GEMM, and scatter
copies share the execution rank's stream; FIFO supplies the within-rank data
ordering. The caller observes outputs only after every execution rank drains.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...runtime.hip_module import Runtime, _HipFunctionHandle, device_scope
from ...runtime.packing import pack_args
from .peer_gemm import (
    PeerGemmAllocation,
    PeerGemmScheduleSpec,
    PeerGemmSpec,
    PeerGemmTables,
    _overlap,
    _require_region,
    make_peer_gemm_tables,
    peer_gemm_grid,
    peer_gemm_signature,
    scheduled_peer_gemm_grid,
)
from .peer_tile_copy import (
    PeerTileCopyProblem,
    PeerTileCopySpec,
    peer_tile_copy_grid,
    peer_tile_copy_signature,
)

__all__ = [
    "PeerCopyTask",
    "PeerGemmEnqueueSummary",
    "PeerGemmPipelinePlan",
    "enqueue_peer_gemm_pipeline",
    "make_peer_gemm_pipeline",
]


@dataclass(frozen=True)
class PeerCopyTask:
    """A bounded panel copy; references keep the caller's region owners alive."""

    source: PeerGemmAllocation
    destination: PeerGemmAllocation
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not int
            or type(self.cols) is not int
            or min(self.rows, self.cols) <= 0
        ):
            raise ValueError("copy panel dimensions must be positive integers")
        required = self.rows * self.cols * 2
        if min(self.source.nbytes, self.destination.nbytes) < required:
            raise ValueError("copy panel exceeds source or destination region")
        if _overlap(self.source, self.destination):
            raise ValueError("copy source and destination must not overlap")

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols * 2

    def values(self) -> dict[str, int]:
        """Existing copy ABI; both pointers already reference their panel origin."""
        problem = PeerTileCopyProblem(
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.cols,
            self.cols,
        )
        return problem.values(self.source.pointer, self.destination.pointer)


@dataclass(frozen=True)
class PeerGemmPipelinePlan:
    """Equivalent final layout for fused and copy/local-GEMM/copy paths.

    All local source/destination panels are copied when a phase is present,
    including the execution rank's own panel. Byte counts distinguish those
    local copies from peer payload. Logical payload is not a hardware traffic
    counter: the fused kernel may reread A for multiple output-column tiles.
    """

    spec: PeerGemmSpec
    copy_spec: PeerTileCopySpec
    execution_device: int
    fused_tables: PeerGemmTables
    local_tables: PeerGemmTables
    gather: tuple[PeerCopyTask, ...]
    scatter: tuple[PeerCopyTask, ...]
    required_peer_paths: tuple[tuple[int, int], ...]
    keepalive: tuple[PeerGemmAllocation, ...]

    def logical_bytes(self) -> dict[str, int]:
        """Workload payload/accounting only; no achieved bandwidth or traffic claim."""
        execution = self.execution_device
        return {
            "gather_copy_bytes": sum(task.nbytes for task in self.gather),
            "gather_peer_payload_bytes": sum(
                task.nbytes for task in self.gather if task.source.device != execution
            ),
            "scatter_copy_bytes": sum(task.nbytes for task in self.scatter),
            "scatter_peer_payload_bytes": sum(
                task.nbytes
                for task in self.scatter
                if task.destination.device != execution
            ),
            "gather_staging_bytes": sum(task.nbytes for task in self.gather),
            "scatter_staging_bytes": sum(task.nbytes for task in self.scatter),
        }


def _panel(parent: PeerGemmAllocation, index: int, nbytes: int) -> PeerGemmAllocation:
    offset = index * nbytes
    if offset + nbytes > parent.nbytes:
        raise ValueError("staging/output panel exceeds its allocation")
    return PeerGemmAllocation(
        parent.device, parent.pointer + offset, nbytes, owner=parent
    )


def make_peer_gemm_pipeline(
    spec: PeerGemmSpec,
    *,
    rank: int,
    rank_devices: tuple[int, ...],
    a_buffers: tuple[PeerGemmAllocation, ...],
    b_buffer: PeerGemmAllocation,
    c_buffers: tuple[PeerGemmAllocation, ...],
    a_staging: PeerGemmAllocation | None = None,
    c_staging: PeerGemmAllocation | None = None,
    input_placement: str = "gather",
    output_placement: str = "scatter",
) -> PeerGemmPipelinePlan:
    """Plan the unfused baseline without changing precision, GEMM, or final C layout.

    Gather materializes all A panels on the execution device. Scatter first
    materializes that rank's C column shard locally. Missing phases use the same
    original allocation as the fused path. Staging must not overlap any original
    input/output region or another staging region.
    """
    fused = make_peer_gemm_tables(
        spec,
        rank=rank,
        rank_devices=rank_devices,
        a_buffers=a_buffers,
        b_buffer=b_buffer,
        c_buffers=c_buffers,
        input_placement=input_placement,
        output_placement=output_placement,
    )
    execution = fused.execution_device
    a_bytes = spec.rows_per_source * spec.k * 2
    c_bytes = spec.rows_per_source * spec.cols_per_rank * 2
    staging = tuple(x for x in (a_staging, c_staging) if x is not None)
    originals = (*a_buffers, b_buffer, *c_buffers)
    if any(_overlap(stage, original) for stage in staging for original in originals):
        raise ValueError("staging must not overlap original input/output regions")
    if len(staging) == 2 and _overlap(staging[0], staging[1]):
        raise ValueError("A and C staging regions must not overlap")
    if input_placement == "gather":
        if a_staging is None:
            raise ValueError("gather baseline requires local A staging")
        _require_region(a_staging, execution, spec.sources * a_bytes)
        gather = tuple(
            PeerCopyTask(
                source, _panel(a_staging, index, a_bytes), spec.rows_per_source, spec.k
            )
            for index, source in enumerate(a_buffers)
        )
        local_a = (a_staging,)
    else:
        if a_staging is not None:
            raise ValueError("replicated input has no gather staging phase")
        gather = ()
        local_a = a_buffers
    if output_placement == "scatter":
        if c_staging is None:
            raise ValueError("scatter baseline requires local C staging")
        _require_region(c_staging, execution, spec.sources * c_bytes)
        scatter = tuple(
            PeerCopyTask(
                _panel(c_staging, source, c_bytes),
                _panel(target, rank, c_bytes),
                spec.rows_per_source,
                spec.cols_per_rank,
            )
            for source, target in enumerate(c_buffers)
        )
        local_c = (c_staging,)
    else:
        if c_staging is not None:
            raise ValueError("local output has no scatter staging phase")
        scatter = ()
        local_c = c_buffers
    local = make_peer_gemm_tables(
        spec,
        rank=rank,
        rank_devices=rank_devices,
        a_buffers=local_a,
        b_buffer=b_buffer,
        c_buffers=local_c,
        input_placement="replicated",
        output_placement="local",
    )
    assert not local.required_peer_paths
    return PeerGemmPipelinePlan(
        spec,
        PeerTileCopySpec(dtype=spec.dtype),
        execution,
        fused,
        local,
        gather,
        scatter,
        fused.required_peer_paths,
        (*originals, *staging),
    )


@dataclass(frozen=True)
class PeerGemmEnqueueSummary:
    """Kernel launch counts, not timing or a completion guarantee."""

    gather_launches: int
    gemm_launches: int
    scatter_launches: int

    @property
    def launches(self) -> int:
        return self.gather_launches + self.gemm_launches + self.scatter_launches


def enqueue_peer_gemm_pipeline(
    runtime: Runtime,
    plan: PeerGemmPipelinePlan,
    *,
    gemm_function: _HipFunctionHandle,
    copy_function: _HipFunctionHandle,
    a_table: PeerGemmAllocation,
    c_table: PeerGemmAllocation,
    fused: bool,
    stream: int = 0,
    schedule: PeerGemmScheduleSpec | None = None,
) -> PeerGemmEnqueueSummary:
    """Enqueue one rank's path with no host fence between its phases.

    Function modules and tables must belong to execution_device. Upload the
    selected plan's table bytes beforehand. The caller retains modules and drains
    all execution streams before readback/free; failures after partial enqueue
    still require a successful drain before releasing any referenced memory.
    """
    if type(fused) is not bool:
        raise ValueError("fused must be a boolean")
    if schedule is not None and schedule.gemm != plan.spec:
        raise ValueError("schedule GEMM spec must match the pipeline plan")
    grid = (
        peer_gemm_grid(plan.spec)
        if schedule is None
        else scheduled_peer_gemm_grid(schedule)
    )
    tables = plan.fused_tables if fused else plan.local_tables
    values = tables.kernel_values(a_table, c_table)
    if any(
        _overlap(table, region)
        for table in (a_table, c_table)
        for region in plan.keepalive
    ):
        raise ValueError("pointer tables must not overlap pipeline data or staging")
    copy_signature = peer_tile_copy_signature(plan.copy_spec)

    def copy(task: PeerCopyTask) -> None:
        runtime.launch(
            copy_function,
            peer_tile_copy_grid(task.rows, task.cols, plan.copy_spec),
            (plan.copy_spec.block_size, 1, 1),
            pack_args(copy_signature, task.values()),
            stream=stream,
            record_event=False,
        )
        runtime.retain_for_stream(stream, plan, task)

    with device_scope(plan.execution_device):
        if not fused:
            for task in plan.gather:
                copy(task)
        runtime.launch(
            gemm_function,
            grid,
            (plan.spec.block_size, 1, 1),
            pack_args(peer_gemm_signature(plan.spec), values),
            stream=stream,
            record_event=False,
        )
        runtime.retain_for_stream(stream, plan, tables, a_table, c_table)
        if not fused:
            for task in plan.scatter:
                copy(task)
    return PeerGemmEnqueueSummary(
        0 if fused else len(plan.gather), 1, 0 if fused else len(plan.scatter)
    )
