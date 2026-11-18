# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pointer-selected GEMM panels: direct peer inputs/outputs in one rank launch.

The compute loop reuses the existing one-wave MFMA GEMM helpers. Each grid-z
entry selects one A row panel and one C panel from device pointer tables; B is
local and shared across all source panels. HIP, not the builder, establishes
addressability and input publication. There is no collective or transfer API.

Scatter destinations are panel-packed: C_owner[compute_rank, rows, columns].
They are not an ordinary row-major full-width C shard. This explicit layout
lets the existing row-major store helper write directly to each destination.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ...core.ir import IRBuilder, KernelDef, PtrType, Value
from ...helpers.io import io_ir_type
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
    store_acc_to_global,
)
from ...helpers.spec import SignatureBuilder
from .mfma_gemm import MfmaGemmSpec
from .mfma_gemm import is_valid_spec as is_valid_mfma_spec

__all__ = [
    "PeerGemmAllocation",
    "PeerGemmScheduleSpec",
    "PeerGemmSpec",
    "PeerGemmTables",
    "build_peer_gemm",
    "build_scheduled_peer_gemm",
    "is_valid_schedule_spec",
    "is_valid_spec",
    "make_peer_gemm_tables",
    "peer_gemm_grid",
    "peer_gemm_signature",
    "scheduled_peer_gemm_cta_work",
    "scheduled_peer_gemm_grid",
    "scheduled_peer_gemm_tile_coordinate",
]


@dataclass(frozen=True)
class PeerGemmSpec:
    """Fixed equal row/column panels with full local K accumulation per output."""

    rows_per_source: int
    cols_per_rank: int
    k: int
    sources: int = 2
    dtype: str = "bf16"
    name: str = "rocke_peer_gemm"

    def to_mfma_spec(self) -> MfmaGemmSpec:
        """Select the portable existing 16x16x16 MFMA body, not a new pipeline."""
        return MfmaGemmSpec(
            M=self.rows_per_source,
            N=self.cols_per_rank,
            K=self.k,
            dtype=self.dtype,
            tile_m=16,
            tile_n=16,
            kpack=False,
            name=f"{self.name}_s{self.sources}",
        )

    @property
    def block_size(self) -> int:
        return 64

    def kernel_name(self) -> str:
        """Runtime rank, addresses, and placement do not change the code object."""
        return self.to_mfma_spec().kernel_name()


def is_valid_spec(spec: PeerGemmSpec, arch: str = "gfx950") -> tuple[bool, str]:
    """Validate initial shapes and the existing atom's exact target availability."""
    if arch not in ("gfx942", "gfx950"):
        return False, f"unsupported peer GEMM arch {arch!r}"
    if type(spec.sources) is not int or not 1 <= spec.sources <= 8:
        return False, "sources must be an integer in [1, 8]"
    if any(
        type(v) is not int or not 0 < v <= 0x7FFFFFFF
        for v in (spec.rows_per_source, spec.cols_per_rank, spec.k)
    ):
        return False, "panel dimensions must be positive i32 integers"
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if (
        not isinstance(spec.name, str)
        or len(spec.name) > 128
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.name)
    ):
        return False, "name must be an ASCII identifier of at most 128 characters"
    if (
        max(
            spec.rows_per_source * spec.k,
            spec.k * spec.cols_per_rank,
            spec.rows_per_source * spec.cols_per_rank,
        )
        > 0x7FFFFFFF
    ):
        return False, "each GEMM panel element span must fit i32 indexing"
    return is_valid_mfma_spec(spec.to_mfma_spec(), arch)


@dataclass(frozen=True)
class PeerGemmScheduleSpec:
    """Opt-in work assignment; the legacy GEMM spec, ABI, and layout stay intact.

    A zero CTA cap submits one CTA per output tile. A positive cap limits the
    grid size; each CTA visits tiles by a fixed grid stride. This bounds submitted
    workgroups, not residency, physical peer traffic, or outstanding memory requests.
    """

    gemm: PeerGemmSpec
    work_order: str = "source_grouped"
    max_ctas: int = 0

    @property
    def tiles_per_source(self) -> int:
        return (self.gemm.rows_per_source // 16) * (self.gemm.cols_per_rank // 16)

    @property
    def tile_count(self) -> int:
        return self.gemm.sources * self.tiles_per_source

    @property
    def cta_count(self) -> int:
        return min(self.max_ctas or self.tile_count, self.tile_count)

    def kernel_name(self) -> str:
        """Effective cap, not addresses/rank/placement, identifies the code object."""
        ok, why = is_valid_schedule_spec(self)
        if not ok:
            raise ValueError(f"invalid peer GEMM schedule spec: {why}")
        return f"{self.gemm.kernel_name()}_{self.work_order}_q{self.cta_count}"


def is_valid_schedule_spec(
    spec: PeerGemmScheduleSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    """Keep the existing aligned atom's constraints and bound i32 loop arithmetic."""
    if not isinstance(spec.gemm, PeerGemmSpec):
        return False, "gemm must be a PeerGemmSpec"
    ok, why = is_valid_spec(spec.gemm, arch)
    if not ok:
        return False, why
    if spec.work_order not in ("source_grouped", "tile_interleaved"):
        return False, "work_order must be 'source_grouped' or 'tile_interleaved'"
    if type(spec.max_ctas) is not int or not 0 <= spec.max_ctas <= 0x7FFFFFFF:
        return False, "max_ctas must be a non-negative i32 integer"
    # Legacy panel indexing bounds and sources <= 8 imply 2*tile_count fits i32,
    # including the grid-stride loop's final increment. Check it explicitly.
    if spec.tile_count > 0x7FFFFFFF // 2:
        return False, "tile count and final grid-stride increment must fit i32"
    return True, "ok"


def scheduled_peer_gemm_grid(spec: PeerGemmScheduleSpec) -> tuple[int, int, int]:
    """One-dimensional fixed CTA grid for a single consuming rank launch."""
    ok, why = is_valid_schedule_spec(spec)
    if not ok:
        raise ValueError(f"invalid peer GEMM schedule spec: {why}")
    return spec.cta_count, 1, 1


def scheduled_peer_gemm_tile_coordinate(
    spec: PeerGemmScheduleSpec, index: int
) -> tuple[int, int, int]:
    """Host oracle for (source, row tile, column tile), not hardware dispatch order."""
    scheduled_peer_gemm_grid(spec)
    if type(index) is not int or not 0 <= index < spec.tile_count:
        raise ValueError("tile index must be inside the scheduled tile range")
    if spec.work_order == "source_grouped":
        source, panel = divmod(index, spec.tiles_per_source)
    else:
        panel, source = divmod(index, spec.gemm.sources)
    row, column = divmod(panel, spec.gemm.cols_per_rank // 16)
    return source, row, column


def scheduled_peer_gemm_cta_work(
    spec: PeerGemmScheduleSpec, cta: int
) -> Iterator[tuple[int, int, int]]:
    """Finite unique work for one CTA, with no waits or cross-CTA dependencies."""
    scheduled_peer_gemm_grid(spec)
    if type(cta) is not int or not 0 <= cta < spec.cta_count:
        raise ValueError("CTA index must be inside the scheduled grid")
    return (
        scheduled_peer_gemm_tile_coordinate(spec, index)
        for index in range(cta, spec.tile_count, spec.cta_count)
    )


@dataclass(frozen=True)
class PeerGemmAllocation:
    """A caller-owned device-memory region, not an allocation or mapping API.

    Device is the visible HIP ordinal. The optional owner object is retained
    through table-plan lifetime; without it the caller must retain the actual
    allocation separately. Report the accessible region's true byte size.
    """

    device: int
    pointer: int
    nbytes: int
    owner: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.device) is not int or self.device < 0:
            raise ValueError("allocation device must be a non-negative integer")
        if type(self.pointer) is not int or self.pointer <= 0 or self.pointer % 16:
            raise ValueError("allocation pointer must be positive and 16-byte aligned")
        if (
            type(self.nbytes) is not int
            or self.nbytes <= 0
            or self.pointer + self.nbytes >= 1 << 64
        ):
            raise ValueError(
                "allocation byte range must fit a positive u64 address range"
            )


def _require_region(region: PeerGemmAllocation, device: int, size: int) -> None:
    if region.device != device:
        raise ValueError(
            "allocation owner does not match the panel's execution/row owner"
        )
    if region.nbytes < size:
        raise ValueError(
            f"allocation has {region.nbytes} bytes, requires at least {size}"
        )


def _overlap(a: PeerGemmAllocation, b: PeerGemmAllocation) -> bool:
    return a.device == b.device and max(a.pointer, b.pointer) < min(
        a.pointer + a.nbytes, b.pointer + b.nbytes
    )


@dataclass(frozen=True)
class PeerGemmTables:
    """Host placement plan; table bytes must be uploaded on execution_device.

    Keep this plan and both table allocations alive until the rank's stream is
    drained. Enable required_peer_paths and publish immutable inputs before any
    consuming launch. Peer addressability is a runtime check, not inferred here.
    """

    execution_device: int
    a_addresses: tuple[int, ...]
    b: PeerGemmAllocation
    c_addresses: tuple[int, ...]
    required_peer_paths: tuple[tuple[int, int], ...]
    keepalive: tuple[PeerGemmAllocation, ...]

    def __post_init__(self) -> None:
        if type(self.execution_device) is not int or self.execution_device < 0:
            raise ValueError("table execution device must be a non-negative integer")
        if not 1 <= len(self.a_addresses) <= 8 or len(self.c_addresses) != len(
            self.a_addresses
        ):
            raise ValueError("input/output tables need the same source count in [1, 8]")
        if any(
            type(pointer) is not int or not 0 < pointer < 1 << 64 or pointer % 16
            for pointer in (*self.a_addresses, *self.c_addresses)
        ):
            raise ValueError("panel addresses must be aligned positive u64 pointers")
        if self.b.device != self.execution_device:
            raise ValueError("B must be local to the table execution device")

    def a_table_bytes(self) -> bytes:
        """Packed 64-bit AMDGPU pointer slots, not data movement of A panels."""
        return struct.pack("<" + "Q" * len(self.a_addresses), *self.a_addresses)

    def c_table_bytes(self) -> bytes:
        """Packed output pointer slots; each slot is already inside its allocation."""
        return struct.pack("<" + "Q" * len(self.c_addresses), *self.c_addresses)

    def kernel_values(
        self, a_table: PeerGemmAllocation, c_table: PeerGemmAllocation
    ) -> dict[str, int]:
        """Bind locally uploaded tables; caller retains their owners through completion."""
        for table in (a_table, c_table):
            _require_region(table, self.execution_device, len(self.a_addresses) * 8)
            if any(_overlap(table, buffer) for buffer in self.keepalive):
                raise ValueError("pointer tables must not overlap input/output regions")
        if _overlap(a_table, c_table):
            raise ValueError("input/output pointer tables must not overlap")
        return {
            "A_ptrs": a_table.pointer,
            "B": self.b.pointer,
            "C_ptrs": c_table.pointer,
        }


def make_peer_gemm_tables(
    spec: PeerGemmSpec,
    *,
    rank: int,
    rank_devices: Sequence[int],
    a_buffers: Sequence[PeerGemmAllocation],
    b_buffer: PeerGemmAllocation,
    c_buffers: Sequence[PeerGemmAllocation],
    input_placement: str = "gather",
    output_placement: str = "local",
) -> PeerGemmTables:
    """Plan gathered/replicated inputs and local/panel-packed scattered outputs.

    Gather uses one A allocation per source rank. Replicated uses one full local
    A allocation and selects its row panels. Local C concatenates source panels;
    scattered C uses one allocation per row owner and selects this compute rank's
    disjoint column panel. No output reduction or in-place aliasing is supported.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid peer GEMM spec: {why}")
    devices = tuple(rank_devices)
    if (
        len(devices) != spec.sources
        or len(set(devices)) != len(devices)
        or any(type(d) is not int or d < 0 for d in devices)
    ):
        raise ValueError(
            "rank_devices must contain one unique non-negative device per source"
        )
    if type(rank) is not int or not 0 <= rank < spec.sources:
        raise ValueError("rank must be inside the source/rank range")
    execution = devices[rank]
    a_panel_bytes = spec.rows_per_source * spec.k * 2
    c_panel_bytes = spec.rows_per_source * spec.cols_per_rank * 2
    _require_region(b_buffer, execution, spec.k * spec.cols_per_rank * 2)
    inputs = tuple(a_buffers)
    outputs = tuple(c_buffers)
    if input_placement == "gather":
        if len(inputs) != spec.sources:
            raise ValueError("gather needs one A panel allocation per source")
        for source, buffer in enumerate(inputs):
            _require_region(buffer, devices[source], a_panel_bytes)
        a_addresses = tuple(buffer.pointer for buffer in inputs)
    elif input_placement == "replicated":
        if len(inputs) != 1:
            raise ValueError("replicated input needs one full local A allocation")
        _require_region(inputs[0], execution, spec.sources * a_panel_bytes)
        a_addresses = tuple(
            inputs[0].pointer + source * a_panel_bytes for source in range(spec.sources)
        )
    else:
        raise ValueError("input_placement must be 'gather' or 'replicated'")
    if output_placement == "local":
        if len(outputs) != 1:
            raise ValueError("local output needs one full local C allocation")
        _require_region(outputs[0], execution, spec.sources * c_panel_bytes)
        c_addresses = tuple(
            outputs[0].pointer + source * c_panel_bytes
            for source in range(spec.sources)
        )
    elif output_placement == "scatter":
        if len(outputs) != spec.sources:
            raise ValueError(
                "scatter needs one panel-packed C allocation per row owner"
            )
        for source, buffer in enumerate(outputs):
            _require_region(buffer, devices[source], spec.sources * c_panel_bytes)
        c_addresses = tuple(buffer.pointer + rank * c_panel_bytes for buffer in outputs)
    else:
        raise ValueError("output_placement must be 'local' or 'scatter'")
    if any(
        _overlap(source, destination)
        for source in (*inputs, b_buffer)
        for destination in outputs
    ):
        raise ValueError("GEMM input and output regions must not overlap")
    owners = {buffer.device for buffer in (*inputs, *outputs)}
    paths = tuple(sorted((execution, owner) for owner in owners if owner != execution))
    return PeerGemmTables(
        execution,
        a_addresses,
        b_buffer,
        c_addresses,
        paths,
        (*inputs, b_buffer, *outputs),
    )


def peer_gemm_grid(spec: PeerGemmSpec) -> tuple[int, int, int]:
    """Exact tile-aligned x/y geometry, one z entry per source row panel."""
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid peer GEMM spec: {why}")
    atom = spec.to_mfma_spec().atom
    return spec.cols_per_rank // atom.n, spec.rows_per_source // atom.m, spec.sources


def peer_gemm_signature(spec: PeerGemmSpec) -> list[dict[str, str]]:
    """Two device pointer tables and one local row-major B panel."""
    return (
        SignatureBuilder()
        .ptr("A_ptrs", f"ptr<{spec.dtype},global>")
        .ptr("B", spec.dtype)
        .ptr("C_ptrs", f"ptr<{spec.dtype},global>")
        .build()
    )


def build_peer_gemm(spec: PeerGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Fuse pointer-selected loads, the existing MFMA K-loop, and direct stores."""
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid peer GEMM spec: {why}")
    base = spec.to_mfma_spec()
    atom = base.atom
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    element = io_ir_type(spec.dtype)
    panel_pointer = PtrType(element, "global")
    table_pointer = PtrType(panel_pointer, "global")
    a_table = b.param("A_ptrs", table_pointer, noalias=True, readonly=True, align=8)
    local_b = b.param("B", panel_pointer, noalias=True, readonly=True, align=16)
    c_table = b.param("C_ptrs", table_pointer, noalias=True, readonly=True, align=8)
    source = b.block_id_z()
    valid = b.land(
        b.cmp_ge(source, b.const_i32(0)), b.cmp_lt(source, b.const_i32(spec.sources))
    )
    with b.scf_if(valid):
        selected_a = b.global_load(a_table, source, dtype=panel_pointer, align=8)
        selected_c = b.global_load(c_table, source, dtype=panel_pointer, align=8)
        lane = b.thread_id_x()
        bid_n = b.block_id_x()
        bid_m = b.block_id_y()
        _emit_peer_gemm_compute(
            b, spec, atom, selected_a, local_b, selected_c, lane, bid_n, bid_m
        )
    b.ret()
    return b.kernel


def _emit_peer_gemm_compute(
    b: IRBuilder,
    spec: PeerGemmSpec,
    atom,
    selected_a: Value,
    local_b: Value,
    selected_c: Value,
    lane: Value,
    bid_n: Value,
    bid_m: Value,
) -> None:
    """Canonical existing panel body shared by legacy and scheduled placement."""
    m_base = b.mul(bid_m, b.const_i32(atom.m))
    n_base = b.mul(bid_n, b.const_i32(atom.n))
    decode = decode_mfma_lanes(b, atom, lane)
    atom_k = b.const_i32(atom.k)

    def load_a(builder, kt):
        return load_a_row_major_contiguous(
            builder,
            A=selected_a,
            atom=atom,
            lane_decode=decode,
            m_tile_base=m_base,
            k_tile_base=builder.mul(kt, atom_k),
            K=spec.k,
        )

    def load_b(builder, kt):
        return load_b_col_strided_scalars(
            builder,
            B=local_b,
            atom=atom,
            lane_decode=decode,
            n_tile_base=n_base,
            k_tile_base=builder.mul(kt, atom_k),
            N=spec.cols_per_rank,
        )

    acc = mfma_k_loop(b, K=spec.k, atom=atom, load_a=load_a, load_b=load_b)
    store_acc_to_global(
        b,
        C=selected_c,
        atom=atom,
        lane_decode=decode,
        m_tile_base=m_base,
        n_tile_base=n_base,
        acc=acc,
        N=spec.cols_per_rank,
        out_dtype=spec.dtype,
    )


def build_scheduled_peer_gemm(
    spec: PeerGemmScheduleSpec, arch: str = "gfx950"
) -> KernelDef:
    """Reassign full output tiles; do not stage/publish tiles or split K.

    Extra x CTAs and nonzero y/z CTAs do no work. Valid CTAs independently
    execute a finite grid-stride loop around the same panel MFMA body. Logical
    source grouping/interleaving does not promise a physical dispatch ordering.
    """
    ok, why = is_valid_schedule_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid peer GEMM schedule spec: {why}")
    gemm = spec.gemm
    atom = gemm.to_mfma_spec().atom
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = gemm.block_size
    element = io_ir_type(gemm.dtype)
    panel_pointer = PtrType(element, "global")
    table_pointer = PtrType(panel_pointer, "global")
    a_table = b.param("A_ptrs", table_pointer, noalias=True, readonly=True, align=8)
    local_b = b.param("B", panel_pointer, noalias=True, readonly=True, align=16)
    c_table = b.param("C_ptrs", table_pointer, noalias=True, readonly=True, align=8)
    cta = b.block_id_x()
    valid = b.land(
        b.cmp_ge(cta, b.const_i32(0)), b.cmp_lt(cta, b.const_i32(spec.cta_count))
    )
    valid = b.land(valid, b.cmp_eq(b.block_id_y(), b.const_i32(0)))
    valid = b.land(valid, b.cmp_eq(b.block_id_z(), b.const_i32(0)))
    with b.scf_if(valid):  # noqa: SIM117 - explicit guarded SSA emission scope
        with b.scf_for(
            cta,
            b.const_i32(spec.tile_count),
            b.const_i32(spec.cta_count),
            iv_name="work",
        ) as index:
            if spec.work_order == "source_grouped":
                divisor = b.const_i32(spec.tiles_per_source)
                source = b.div(index, divisor)
                panel = b.mod(index, divisor)
            else:
                divisor = b.const_i32(gemm.sources)
                source = b.mod(index, divisor)
                panel = b.div(index, divisor)
            columns = b.const_i32(gemm.cols_per_rank // atom.n)
            bid_m = b.div(panel, columns)
            bid_n = b.mod(panel, columns)
            selected_a = b.global_load(a_table, source, dtype=panel_pointer, align=8)
            selected_c = b.global_load(c_table, source, dtype=panel_pointer, align=8)
            lane = b.thread_id_x()
            _emit_peer_gemm_compute(
                b, gemm, atom, selected_a, local_b, selected_c, lane, bid_n, bid_m
            )
    b.ret()
    return b.kernel
