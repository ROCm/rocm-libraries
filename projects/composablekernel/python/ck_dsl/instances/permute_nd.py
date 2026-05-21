# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""N-D tensor permutation kernel instance.

DSL counterpart of CK Tile's ``example/ck_tile/06_permute``. Computes::

    Y[i_0, i_1, ..., i_{n-1}] = X[i_{pi(0)}, i_{pi(1)}, ..., i_{pi(n-1)}]

where ``pi`` is the permutation (so e.g. ``pi=(2,1,0)`` on a rank-3
tensor is equivalent to ``y = x.permute(2, 1, 0).contiguous()``).
``Y`` is row-major contiguous with shape
``(x_shape[pi[0]], x_shape[pi[1]], ..., x_shape[pi[n-1]])``.

What we cover today:

* Dtypes ``f16`` and ``bf16``. The CK Tile example also handles ``fp8``
  and ``fp32``; f8/f32 are a v2 follow-up that mostly needs a different
  ``global_load`` dispatch.
* Rank up to 8 (the same cap as CK Tile's ``GenericPermuteHostArgs::kMaxRanks``).
* Arbitrary permutations; the kernel is shape-specialised at build time
  (compile-time shape + permutation -> compile-time strides + the
  index-decomposition arithmetic folds at IR construction).
* No padding / alignment requirements on the input shape.

Layout:

* ``X`` is assumed to be row-major over ``x_shape``.
* ``Y`` is row-major over ``y_shape`` (the permuted shape).

Implementation notes:

* One thread per output element. Block size defaults to 256; the
  launch grid is ``(ceil(total_elements / block_size), 1, 1)``.
* The kernel decomposes the flat output index into an n-D coordinate
  using compile-time output strides (one div + one mod per axis), then
  recomposes the source index using compile-time input strides and the
  permutation. With rank <= 8 this is ~16 cheap integer ops per thread.
* Loads / stores use plain typed global ops with an ``scf.if`` guard
  on ``out_idx < total``; the kernel is bandwidth-bound, so the small
  per-thread bookkeeping is in the noise.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import List, Literal, Tuple

from ..core.ir import BF16, F16, IRBuilder, KernelDef, PtrType, Value
from ..helpers.io import io_ir_type
from ..helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join


DType = Literal["f16", "bf16"]
MAX_RANK = 8


@dataclass(frozen=True)
class PermuteSpec:
    """One concrete permutation kernel configuration.

    ``x_shape`` is the input tensor shape (row-major contiguous);
    ``perm`` is a tuple of length ``len(x_shape)`` containing a
    permutation of ``range(rank)``. The output shape is then
    ``y_shape[d] = x_shape[perm[d]]``.
    """

    x_shape: Tuple[int, ...]
    perm: Tuple[int, ...]
    dtype: DType = "f16"
    block_size: int = 256
    name: str = "ck_dsl_permute"

    @property
    def rank(self) -> int:
        return len(self.x_shape)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return tuple(self.x_shape[self.perm[i]] for i in range(self.rank))

    @property
    def total_elements(self) -> int:
        return functools.reduce(operator.mul, self.x_shape, 1)

    def kernel_name(self) -> str:
        shape_str = "x".join(str(d) for d in self.x_shape)
        perm_str = "".join(str(d) for d in self.perm)
        return kernel_name_join(
            self.name,
            f"s{shape_str}",
            f"p{perm_str}",
            self.dtype,
            f"b{self.block_size}",
        )


def is_valid_spec(spec: PermuteSpec) -> Tuple[bool, str]:
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"block_size {spec.block_size} not in {{64,128,256,512,1024}}"
    if spec.rank == 0:
        return False, "rank must be >= 1"
    if spec.rank > MAX_RANK:
        return False, f"rank {spec.rank} > {MAX_RANK} (CK Tile cap)"
    if len(spec.perm) != spec.rank:
        return (
            False,
            f"perm length {len(spec.perm)} != x_shape rank {spec.rank}",
        )
    if sorted(spec.perm) != list(range(spec.rank)):
        return (
            False,
            f"perm {list(spec.perm)} is not a permutation of range({spec.rank})",
        )
    if any(d <= 0 for d in spec.x_shape):
        return False, f"x_shape must be all positive, got {list(spec.x_shape)}"
    if spec.total_elements <= 0:
        return False, f"total_elements must be positive, got {spec.total_elements}"
    return True, "ok"


def _row_major_strides(shape: Tuple[int, ...]) -> List[int]:
    """Return row-major (C-contiguous) strides for ``shape``."""
    n = len(shape)
    strides = [1] * n
    for d in range(n - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def build_permute(spec: PermuteSpec) -> KernelDef:
    """Build the IR for one permutation instance.

    Kernel signature::

        (X: ptr<dtype, global>,   # input  (row-major over x_shape)
         Y: ptr<dtype, global>)   # output (row-major over y_shape)
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid permute spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    n = spec.rank
    x_shape = spec.x_shape
    y_shape = spec.y_shape
    perm = spec.perm
    x_strides = _row_major_strides(x_shape)
    y_strides = _row_major_strides(y_shape)
    total = spec.total_elements

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    out_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)
    c_total = b.const_i32(total)
    in_bounds = b.cmp_lt(out_idx, c_total)

    with b.scf_if(in_bounds):
        # Decompose out_idx into an n-D output coordinate ``a``:
        # a[d] = (out_idx // y_strides[d]) % y_shape[d]
        coords: List[Value] = []
        for d in range(n):
            if y_strides[d] == 1:
                axis = out_idx
            else:
                axis = b.div(out_idx, b.const_i32(y_strides[d]))
            if y_shape[d] != total:  # outermost axis: mod is redundant
                axis = b.mod(axis, b.const_i32(y_shape[d]))
            coords.append(axis)

        # Recompose source index using the input strides and the
        # permutation: x_idx = sum_d coords[perm[d]] * x_strides[d]
        x_idx: Value = b.const_i32(0)
        for d in range(n):
            src_axis = coords[perm[d]]
            stride = x_strides[d]
            term = src_axis if stride == 1 else b.mul(src_axis, b.const_i32(stride))
            x_idx = term if d == 0 else b.add(x_idx, term)

        if spec.dtype == "f16":
            loaded = b.global_load_f16(X, x_idx)
        else:  # bf16
            loaded = b.global_load_bf16(X, x_idx)
        b.global_store(Y, out_idx, loaded)

    return b.kernel


def permute_grid(spec: PermuteSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one thread per output element."""
    return ceil_div_grid((spec.total_elements, spec.block_size))


def permute_signature(spec: PermuteSpec):
    return SignatureBuilder().ptr("X", spec.dtype).ptr("Y", spec.dtype).build()


# ``BF16`` is intentionally re-exported so callers (e.g. tests) don't
# need to import it via the deeper ``ck_dsl.core.ir`` path just to
# build a PermuteSpec.
__all__ = [
    "BF16",
    "F16",
    "MAX_RANK",
    "PermuteSpec",
    "build_permute",
    "is_valid_spec",
    "permute_grid",
    "permute_signature",
]
