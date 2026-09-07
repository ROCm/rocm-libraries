# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Local rank-staged reduction epilogues for MoE pipelines.

These builders intentionally stop at the single-device boundary.  rocKE has no
collective transport or peer-memory runtime, so callers first make every rank's
partial locally addressable in a contiguous rank-major workspace.  The kernels
then fuse the arithmetic that normally follows that transport:

* :func:`build_moe_rank_reduce_rmsnorm` reduces ``[R, M, N]`` partials and
  immediately applies RMSNorm.
* :func:`build_moe_rank_reduce_scatter` reduces only this rank's shard from
  ``[R, M, N]`` and writes ``[M, N / R]``.

Neither kernel is an all-reduce or reduce-scatter transport by itself.  Keeping
that boundary explicit prevents a local-memory kernel from being advertised as
a cross-device collective while still providing the fused compute epilogues a
communication runtime can invoke after filling the staging workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...core.arch import ArchTarget
from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.io import io_ir_type
from ...helpers.reduction import (
    REGISTER_TILE_MAX_ELEMS_PER_THREAD,
    block_lds_reduce,
    block_lds_reduce_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)

DType = Literal["f16", "bf16"]
_MAX_WORLD_SIZE = 16


@dataclass(frozen=True)
class MoeRankReduceRMSNormSpec:
    """Configuration for rank-major reduction followed by RMSNorm.

    ``width`` and ``world_size`` are compile-time constants. ``rows`` remains a
    runtime launch argument so one compiled kernel covers the decode batch
    range. When ``fp32_internal`` is false, the rank sum is rounded to
    ``dtype`` before the square and normalization, matching a narrow collective
    output consumed by a separate RMSNorm. The true path retains the sum in f32.
    """

    width: int
    world_size: int
    dtype: DType = "bf16"
    block_size: int = 256
    vec: int = 2
    wave_size: int = 64
    fp32_internal: bool = False
    name: str = "rocke_moe_rank_reduce_rmsnorm"

    @property
    def elems_per_thread(self) -> int:
        return self.width // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"N{self.width}",
            f"R{self.world_size}",
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"f32": self.fp32_internal},
        )


@dataclass(frozen=True)
class MoeRankReduceScatterSpec:
    """Configuration for reducing one rank's contiguous output shard."""

    width: int
    world_size: int
    dtype: DType = "bf16"
    block_size: int = 64
    vec: int = 2
    name: str = "rocke_moe_rank_reduce_scatter"

    @property
    def shard_width(self) -> int:
        return self.width // self.world_size

    @property
    def elems_per_thread(self) -> int:
        return self.shard_width // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"N{self.width}",
            f"R{self.world_size}",
            f"b{self.block_size}",
            f"v{self.vec}",
        )


def _validate_common(
    *,
    width: int,
    world_size: int,
    dtype: str,
    block_size: int,
    vec: int,
    n_per_block: int,
    arch: str,
    require_even_partition: bool,
) -> tuple[bool, str]:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if width <= 0:
        return False, f"width must be > 0 (got {width})"
    if not 1 <= world_size <= _MAX_WORLD_SIZE:
        return False, (
            f"world_size must be in [1, {_MAX_WORLD_SIZE}] (got {world_size})"
        )
    if require_even_partition and width % world_size:
        return False, f"width ({width}) must be divisible by world_size ({world_size})"
    ok, why = validate_io(
        IOSpecRule(
            dtype=dtype,
            block_size=block_size,
            vec=vec,
            n_per_block=n_per_block,
            max_elems_per_thread=REGISTER_TILE_MAX_ELEMS_PER_THREAD,
        )
    )
    if not ok:
        return False, why
    if block_size > target.max_threads_per_block:
        return False, (
            f"block_size {block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    return True, "ok"


def is_valid_rmsnorm_spec(
    spec: MoeRankReduceRMSNormSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    """Return whether a rank-reduce/RMSNorm configuration is buildable."""

    ok, why = _validate_common(
        width=spec.width,
        world_size=spec.world_size,
        dtype=spec.dtype,
        block_size=spec.block_size,
        vec=spec.vec,
        n_per_block=spec.width,
        arch=arch,
        require_even_partition=False,
    )
    if not ok:
        return False, why
    target = ArchTarget.from_gfx(arch)
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def is_valid_scatter_spec(
    spec: MoeRankReduceScatterSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    """Return whether a rank-reduce/scatter configuration is buildable."""

    return _validate_common(
        width=spec.width,
        world_size=spec.world_size,
        dtype=spec.dtype,
        block_size=spec.block_size,
        vec=spec.vec,
        n_per_block=spec.shard_width if spec.world_size > 0 else 0,
        arch=arch,
        require_even_partition=True,
    )


def _load_vec_as_f32(
    b: IRBuilder, ptr: Value, offset: Value, *, dtype, vec: int
) -> list[Value]:
    packed = b.global_load_vN(ptr, offset, dtype, vec)
    return [b.cast_to_f32(b.vec_extract(packed, i)) for i in range(vec)]


def _store_vec_from_f32(
    b: IRBuilder,
    ptr: Value,
    offset: Value,
    values: list[Value],
    *,
    dtype,
) -> None:
    packed = b.vec_pack([b.cast_f32_to(value, dtype) for value in values], dtype)
    b.global_store_vN(ptr, offset, packed, len(values))


def _rank_reduce_vec(
    b: IRBuilder,
    partials: Value,
    *,
    row: Value,
    rows: Value,
    width: int,
    world_size: int,
    column: Value,
    dtype,
    vec: int,
) -> list[Value]:
    accum = [b.const_f32(0.0) for _ in range(vec)]
    c_width = b.const_i32(width)
    for source_rank in range(world_size):
        rank_row = b.add(b.mul(b.const_i32(source_rank), rows), row)
        base = b.add(b.mul(rank_row, c_width), column)
        values = _load_vec_as_f32(b, partials, base, dtype=dtype, vec=vec)
        accum = [b.fadd(accum[i], values[i]) for i in range(vec)]
    return accum


def build_moe_rank_reduce_rmsnorm(
    spec: MoeRankReduceRMSNormSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build local rank reduction fused with RMSNorm.

    ABI::

        Partials : ptr<dtype>  # contiguous [world_size, rows, width]
        Gamma    : ptr<dtype>  # [width]
        Y        : ptr<dtype>  # [rows, width]
        rows     : i32
        width    : i32         # ABI check; equals spec.width
        eps      : f32

    Grid is ``(rows, 1, 1)`` with one workgroup per row.
    """

    ok, why = is_valid_rmsnorm_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid moe rank-reduce RMSNorm spec for {arch}: {why}")

    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    partials = b.param(
        "Partials",
        PtrType(io_ty, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    output = b.param(
        "Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
    )
    rows = b.param("rows", I32)
    _width = b.param("width", I32)
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()
    c_vec = b.const_i32(spec.vec)
    lds = b.smem_alloc(F32, [spec.block_size], name_hint="rank_reduce_rms")

    chunks_per_thread = spec.elems_per_thread // spec.vec
    cached: list[Value] = []
    sum_sq = b.const_f32(0.0)
    for chunk in range(chunks_per_thread):
        column = b.add(
            b.mul(b.const_i32(chunk * spec.block_size), c_vec),
            b.mul(tid, c_vec),
        )
        reduced = _rank_reduce_vec(
            b,
            partials,
            row=row,
            rows=rows,
            width=spec.width,
            world_size=spec.world_size,
            column=column,
            dtype=io_ty,
            vec=spec.vec,
        )
        if not spec.fp32_internal:
            reduced = [b.cast_to_f32(b.cast_f32_to(value, io_ty)) for value in reduced]
        squares = [b.fmul(value, value) for value in reduced]
        sum_sq = b.fadd(sum_sq, tree_reduce(b, b.fadd, squares))
        cached.extend(reduced)

    target = ArchTarget.from_gfx(arch)
    if target.wave_size == spec.wave_size and spec.block_size % spec.wave_size == 0:
        total_sq = block_lds_reduce_with_wave_prologue(
            b,
            sum_sq,
            lds,
            tid,
            block_size=spec.block_size,
            combine="sum",
            wave_size=spec.wave_size,
        )
    else:
        total_sq = block_lds_reduce(
            b,
            sum_sq,
            lds,
            tid,
            block_size=spec.block_size,
            combine="sum",
        )
    inv_rms = b.rsqrt(
        b.fadd(
            b.fmul(total_sq, b.rcp(b.const_f32(float(spec.width)))),
            eps,
        )
    )
    row_base = b.mul(row, b.const_i32(spec.width))
    for chunk in range(chunks_per_thread):
        column = b.add(
            b.mul(b.const_i32(chunk * spec.block_size), c_vec),
            b.mul(tid, c_vec),
        )
        gamma_values = _load_vec_as_f32(b, gamma, column, dtype=io_ty, vec=spec.vec)
        normalized = [
            b.fmul(
                cached[chunk * spec.vec + i],
                b.fmul(inv_rms, gamma_values[i]),
            )
            for i in range(spec.vec)
        ]
        _store_vec_from_f32(
            b,
            output,
            b.add(row_base, column),
            normalized,
            dtype=io_ty,
        )

    b.ret()
    return b.kernel


def build_moe_rank_reduce_scatter(
    spec: MoeRankReduceScatterSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build local rank reduction that materializes only one output shard.

    ABI::

        Partials : ptr<dtype>  # contiguous [world_size, rows, width]
        Y        : ptr<dtype>  # [rows, width / world_size]
        rows     : i32
        width    : i32         # ABI check; equals spec.width
        rank     : i32         # destination shard in [0, world_size)

    The caller validates ``rank`` before launch. Grid is ``(rows, 1, 1)``.
    """

    ok, why = is_valid_scatter_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid moe rank-reduce scatter spec for {arch}: {why}")

    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    partials = b.param(
        "Partials",
        PtrType(io_ty, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    output = b.param(
        "Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
    )
    rows = b.param("rows", I32)
    _width = b.param("width", I32)
    rank = b.param("rank", I32)

    tid = b.thread_id_x()
    row = b.block_id_x()
    c_vec = b.const_i32(spec.vec)
    shard_base = b.mul(rank, b.const_i32(spec.shard_width))
    output_row_base = b.mul(row, b.const_i32(spec.shard_width))
    chunks_per_thread = spec.elems_per_thread // spec.vec

    for chunk in range(chunks_per_thread):
        local_column = b.add(
            b.mul(b.const_i32(chunk * spec.block_size), c_vec),
            b.mul(tid, c_vec),
        )
        reduced = _rank_reduce_vec(
            b,
            partials,
            row=row,
            rows=rows,
            width=spec.width,
            world_size=spec.world_size,
            column=b.add(shard_base, local_column),
            dtype=io_ty,
            vec=spec.vec,
        )
        _store_vec_from_f32(
            b,
            output,
            b.add(output_row_base, local_column),
            reduced,
            dtype=io_ty,
        )

    b.ret()
    return b.kernel


def moe_rank_reduce_rmsnorm_grid(
    rows: int, spec: MoeRankReduceRMSNormSpec
) -> tuple[int, int, int]:
    """One workgroup per input row."""

    _ = spec
    return ceil_div_grid((rows, 1))


def moe_rank_reduce_scatter_grid(
    rows: int, spec: MoeRankReduceScatterSpec
) -> tuple[int, int, int]:
    """One workgroup per input row."""

    _ = spec
    return ceil_div_grid((rows, 1))


def moe_rank_reduce_rmsnorm_signature(
    spec: MoeRankReduceRMSNormSpec,
) -> list[dict[str, str]]:
    return (
        SignatureBuilder()
        .ptr("Partials", spec.dtype)
        .ptr("Gamma", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("rows", "i32")
        .scalar("width", "i32")
        .scalar("eps", "f32")
        .build()
    )


def moe_rank_reduce_scatter_signature(
    spec: MoeRankReduceScatterSpec,
) -> list[dict[str, str]]:
    return (
        SignatureBuilder()
        .ptr("Partials", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("rows", "i32")
        .scalar("width", "i32")
        .scalar("rank", "i32")
        .build()
    )


__all__ = [
    "MoeRankReduceRMSNormSpec",
    "MoeRankReduceScatterSpec",
    "build_moe_rank_reduce_rmsnorm",
    "build_moe_rank_reduce_scatter",
    "is_valid_rmsnorm_spec",
    "is_valid_scatter_spec",
    "moe_rank_reduce_rmsnorm_grid",
    "moe_rank_reduce_rmsnorm_signature",
    "moe_rank_reduce_scatter_grid",
    "moe_rank_reduce_scatter_signature",
]
