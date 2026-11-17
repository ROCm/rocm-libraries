# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Masked strided tile copies for validating directly addressable peer memory.

The builder is unaware of ranks: HIP establishes mappings and supplies global
pointers. A peer source exercises remote reads; a peer destination exercises
remote writes. It is not a collective, a DMA submission, or a GEMM pipeline.

One CTA owns one logical output tile. Invalid source or destination coordinates
perform neither a load nor a store, leaving destination guards untouched. All
address arithmetic is signed i64 and values are copied without conversion.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ...core.ir import I64, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type
from ...helpers.spec import SignatureBuilder, kernel_name_join

__all__ = [
    "PeerTileCopyProblem",
    "PeerTileCopySpec",
    "build_peer_tile_copy",
    "is_valid_spec",
    "peer_tile_copy_grid",
    "peer_tile_copy_signature",
]

_SCALARS = (
    "rows",
    "cols",
    "src_rows",
    "src_cols",
    "dst_rows",
    "dst_cols",
    "src_row",
    "src_col",
    "dst_row",
    "dst_col",
    "src_stride_row",
    "src_stride_col",
    "dst_stride_row",
    "dst_stride_col",
)
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_GRID_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class PeerTileCopySpec:
    """Compile-time geometry; addresses and copy/view extents are launch arguments."""

    tile_rows: int = 4
    tile_cols: int = 64
    block_size: int = 64
    dtype: str = "bf16"
    name: str = "rocke_peer_tile_copy"

    def kernel_name(self) -> str:
        """Stable name independent of device ownership or runtime pointers."""
        return kernel_name_join(
            self.name,
            self.dtype,
            f"{self.tile_rows}x{self.tile_cols}",
            f"b{self.block_size}",
        )


def is_valid_spec(spec: PeerTileCopySpec, arch: str = "gfx950") -> tuple[bool, str]:
    """Validate the bounded initial geometry and explicitly supported targets."""
    if arch not in ("gfx942", "gfx950"):
        return False, f"unsupported peer tile copy arch {arch!r}"
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if any(type(v) is not int or v <= 0 for v in (spec.tile_rows, spec.tile_cols)):
        return False, "tile_rows and tile_cols must be positive integers"
    if type(spec.block_size) is not int or spec.block_size not in (64, 128, 256):
        return False, "block_size must be in {64, 128, 256}"
    if spec.tile_rows * spec.tile_cols > 4096:
        return False, "tile area must not exceed 4096 elements"
    if (
        not isinstance(spec.name, str)
        or len(spec.name) > 255
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.name)
    ):
        return False, "name must be an ASCII identifier of at most 255 characters"
    return True, "ok"


@dataclass(frozen=True)
class PeerTileCopyProblem:
    """Host-only argument validation; strides are in elements, not bytes.

    Origins may be negative to test clipping. Shapes and strides describe the
    accessible views, not just the copy rectangle. Destination rows must not
    overlap; source and destination allocations must also be disjoint (the
    caller's responsibility). Both allocations must cover the declared spans.
    This descriptor neither allocates memory nor establishes peer access.
    """

    rows: int
    cols: int
    src_rows: int
    src_cols: int
    dst_rows: int
    dst_cols: int
    src_stride_row: int
    dst_stride_row: int
    src_stride_col: int = 1
    dst_stride_col: int = 1
    src_row: int = 0
    src_col: int = 0
    dst_row: int = 0
    dst_col: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or not _I64_MIN <= value <= _I64_MAX:
                raise ValueError(f"{name} must be a signed i64 integer")
            if "stride" in name and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("rows", "cols", "src_rows", "src_cols", "dst_rows", "dst_cols"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for prefix in ("src", "dst"):
            for axis, extent in (("row", self.rows), ("col", self.cols)):
                origin = getattr(self, f"{prefix}_{axis}")
                if extent and origin + extent - 1 > _I64_MAX:
                    raise ValueError(f"{prefix}_{axis} + copy extent overflows i64")
            height, width = getattr(self, f"{prefix}_rows"), getattr(
                self, f"{prefix}_cols"
            )
            row_stride = getattr(self, f"{prefix}_stride_row")
            col_stride = getattr(self, f"{prefix}_stride_col")
            if height and width:
                span = (height - 1) * row_stride + (width - 1) * col_stride
                if span > (_I64_MAX - 1) // 2:
                    raise ValueError(f"{prefix} view byte span overflows i64")
        if (
            self.dst_rows > 1
            and self.dst_cols
            and self.dst_stride_row <= (self.dst_cols - 1) * self.dst_stride_col
        ):
            raise ValueError("destination rows must not overlap")

    def values(self, source: int, destination: int) -> dict[str, int]:
        """Return ABI values; retain allocation owners separately through all users."""
        return {"X": source, "Y": destination, **asdict(self)}


def peer_tile_copy_grid(
    rows: int, cols: int, spec: PeerTileCopySpec
) -> tuple[int, int, int]:
    """One CTA per tile; an empty rectangle returns a grid the caller must skip."""
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid peer tile copy spec: {why}")
    if any(type(v) is not int or not 0 <= v <= _I64_MAX for v in (rows, cols)):
        raise ValueError("rows and cols must be non-negative i64 integers")
    if not rows or not cols:
        return 0, 0, 1
    grid_x = (cols + spec.tile_cols - 1) // spec.tile_cols
    grid_y = (rows + spec.tile_rows - 1) // spec.tile_rows
    if max(grid_x, grid_y) > _GRID_MAX:
        raise ValueError("tile grid axis exceeds the supported signed i32 range")
    return grid_x, grid_y, 1


def peer_tile_copy_signature(spec: PeerTileCopySpec) -> list[dict[str, str]]:
    """Global pointers followed by signed i64 extents, origins, and element strides."""
    signature = SignatureBuilder().ptr("X", spec.dtype).ptr("Y", spec.dtype)
    for name in _SCALARS:
        signature.scalar(name, "i64")
    return signature.build()


def build_peer_tile_copy(spec: PeerTileCopySpec, arch: str = "gfx950") -> KernelDef:
    """Build ordinary masked global accesses; peer ownership is a runtime concern."""
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid peer tile copy spec: {why}")
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    dtype = io_ir_type(spec.dtype)
    source = b.param(
        "X", PtrType(dtype, "global"), noalias=True, readonly=True, align=2
    )
    destination = b.param(
        "Y", PtrType(dtype, "global"), noalias=True, writeonly=True, align=2
    )
    args = {name: b.param(name, I64) for name in _SCALARS}
    tid = b.zext(b.thread_id_x(), I64)
    block_col = b.zext(b.block_id_x(), I64)
    block_row = b.zext(b.block_id_y(), I64)
    tile_rows = b.const_i64(spec.tile_rows)
    tile_cols = b.const_i64(spec.tile_cols)
    row_base = b.mul(block_row, tile_rows)
    col_base = b.mul(block_col, tile_cols)
    zero = b.const_i64(0)
    area = b.const_i64(spec.tile_rows * spec.tile_cols)
    for chunk in range(
        (spec.tile_rows * spec.tile_cols + spec.block_size - 1) // spec.block_size
    ):
        lane_element = b.add(tid, b.const_i64(chunk * spec.block_size))
        row = b.add(row_base, b.div(lane_element, tile_cols))
        col = b.add(col_base, b.mod(lane_element, tile_cols))
        src_row = b.add(args["src_row"], row)
        src_col = b.add(args["src_col"], col)
        dst_row = b.add(args["dst_row"], row)
        dst_col = b.add(args["dst_col"], col)
        valid = b.cmp_lt(lane_element, area)
        valid = b.land(valid, b.cmp_lt(row, args["rows"]))
        valid = b.land(valid, b.cmp_lt(col, args["cols"]))
        for coordinate, limit in (
            (src_row, args["src_rows"]),
            (src_col, args["src_cols"]),
            (dst_row, args["dst_rows"]),
            (dst_col, args["dst_cols"]),
        ):
            valid = b.land(valid, b.cmp_ge(coordinate, zero))
            valid = b.land(valid, b.cmp_lt(coordinate, limit))
        with b.scf_if(valid):
            src_offset = b.add(
                b.mul(src_row, args["src_stride_row"]),
                b.mul(src_col, args["src_stride_col"]),
            )
            dst_offset = b.add(
                b.mul(dst_row, args["dst_stride_row"]),
                b.mul(dst_col, args["dst_stride_col"]),
            )
            value = b.global_load(source, src_offset, dtype=dtype, align=2)
            b.global_store(destination, dst_offset, value, align=2)
    return b.kernel
