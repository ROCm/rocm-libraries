# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multiple-D GEMM kernel instance (CK Tile ``19_gemm_multi_d`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/19_gemm_multi_d``. The
kernel computes::

    E = f(A * B, D_0, D_1, ..., D_{n-1})

where ``f`` is a user-supplied elementwise fusion of the accumulator
output and N ``(M, N)`` side inputs. Typical uses:

* ``E = A*B + D0``                       (bias / residual add)
* ``E = (A*B + D0) * D1``                (bias + gating)
* ``E = relu(A*B + D0)``                 (bias + activation; composes
                                           with the existing ``ReLU`` op)
* ``E = (A*B + D0 + D1 + ...)``          (multi-residual sum)

Implementation re-uses the existing :class:`UniversalGemmSpec` body
(MFMA + cshuffle epilogue) and attaches a :class:`FusedEpilogue` whose
op chain is one :class:`ResidualAdd` / :class:`ResidualMul` per D
operand. Each D becomes an extra ``(M, N)`` row-major pointer
parameter on the kernel (named ``D0``, ``D1``, ..., in the order the
spec lists them).

What we cover today:

* Same tile / pipeline / scheduler space as :func:`build_universal_gemm`
  (fp16 RCR, MFMA 16x16x{16,32} and 32x32x{8,16}).
* Per-D op chosen from ``{"add", "mul"}``. The CK Tile example's
  ``CDEElementWise`` template is more general; ``add`` / ``mul`` cover
  the common bias / residual / gate cases.
* Up to 8 D operands (CK Tile defaults to 1-2 in the example; the
  cap here is the same ``MAX_D`` we expose on the spec).
* Requires ``epilogue="cshuffle"`` because the ``default`` epilogue
  doesn't have the fused-epilogue hook wired in. Validation rejects
  ``default`` so the failure mode is loud, not silent.

The shape contract: every D is laid out row-major as ``(M, N)``, with
the same M and N as the GEMM output, and stride-M defaulting to N
(contiguous). Non-contiguous D's can be supported by extending the
underlying :class:`ResidualAdd` op to thread through a per-D stride
argument — left as a v2 follow-on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import KernelDef
from ..helpers.fuse import (
    FusedEpilogue,
    ResidualAdd,
    ResidualMul,
    dtype_to_ir,
)
from ..helpers.spec import SignatureBuilder, kernel_name_join
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


MAX_D = 8
DOp = Tuple[str, str]  # (param_name, "add" | "mul")


@dataclass(frozen=True)
class GemmMultiDSpec:
    """One concrete multi-D GEMM kernel configuration.

    ``base`` is a :class:`UniversalGemmSpec` providing the GEMM tile /
    pipeline / data choices. ``d_operands`` is a tuple of
    ``(param_name, op)`` pairs; ``op`` is one of ``{"add", "mul"}``.
    The kernel param order matches the tuple order: the first entry's
    ``param_name`` becomes the third pointer arg (after ``A`` and
    ``B``), and so on, with the GEMM output ``C`` last in the pointer
    block.

    ``d_dtype`` is the element type used for every D operand (CK Tile
    allows heterogeneous D dtypes via its template tuple; we ship the
    homogeneous case in v1 since it matches every shipped example).
    """

    base: UniversalGemmSpec
    d_operands: Tuple[DOp, ...]
    d_dtype: str = "fp16"
    name: str = "ck_dsl_gemm_multi_d"

    @property
    def num_d(self) -> int:
        return len(self.d_operands)

    def kernel_name(self) -> str:
        d_suffix = "_".join(f"{name}{op}" for name, op in self.d_operands)
        return kernel_name_join(
            self.name,
            self.base.kernel_name(),
            f"md{self.num_d}",
            d_suffix,
            self.d_dtype,
        )


def is_valid_spec(spec: GemmMultiDSpec) -> Tuple[bool, str]:
    if spec.num_d == 0:
        return False, "d_operands must contain at least one entry"
    if spec.num_d > MAX_D:
        return False, f"num_d {spec.num_d} > MAX_D {MAX_D}"
    if spec.base.trait.epilogue != "cshuffle":
        return False, (
            "GemmMultiD requires base.trait.epilogue='cshuffle' "
            "(the default epilogue doesn't have the fused-op hook); "
            f"got {spec.base.trait.epilogue!r}"
        )
    names = set()
    for name, op in spec.d_operands:
        if op not in ("add", "mul"):
            return False, f"D op {op!r} not in {{'add','mul'}}"
        if not name:
            return False, "D param_name must be a non-empty string"
        if name in names:
            return False, f"duplicate D param_name {name!r}"
        if name in ("A", "B", "C", "M", "N", "K"):
            return False, (
                f"D param_name {name!r} collides with a reserved GEMM kernel parameter"
            )
        names.add(name)
    # Same arch validation as universal GEMM.
    from .gemm_universal import is_valid_spec as _is_valid_gemm

    ok_base, why_base = _is_valid_gemm(spec.base)
    if not ok_base:
        return False, f"base GEMM spec invalid: {why_base}"
    return True, "ok"


def _build_fused_epilogue(spec: GemmMultiDSpec) -> FusedEpilogue:
    """Compose the per-D ``ResidualAdd`` / ``ResidualMul`` chain."""
    d_ir_dtype = dtype_to_ir(spec.d_dtype)
    ops = []
    for name, op in spec.d_operands:
        if op == "add":
            ops.append(ResidualAdd(param_name=name, dtype=d_ir_dtype))
        elif op == "mul":
            ops.append(ResidualMul(param_name=name, dtype=d_ir_dtype))
        else:  # validated above, but defensive
            raise ValueError(f"unsupported D op {op!r}")
    return FusedEpilogue(ops=tuple(ops), dtype=d_ir_dtype)


def build_gemm_multi_d(spec: GemmMultiDSpec) -> KernelDef:
    """Build the IR for one multi-D GEMM instance.

    Kernel signature::

        (A: ptr<dtype_a, global>,
         B: ptr<dtype_b, global>,
         D0: ptr<d_dtype, global>,
         ...,
         D{num_d-1}: ptr<d_dtype, global>,
         C: ptr<dtype_c, global>,
         M: i32, N: i32, K: i32)

    The D pointers come after A/B and before C — same order the CK
    Tile example uses, so a host launcher can pack the args directly.

    Implementation: builds a :class:`FusedEpilogue` from the per-D op
    chain, attaches it to the base :class:`UniversalGemmSpec` via the
    `_fused_epilogue` side-channel, and delegates to
    :func:`build_universal_gemm`. The cshuffle epilogue picks up the
    fused chain and applies it inside the LDS-staged store loop.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid GemmMultiD spec: {why}")

    # Attach the fused epilogue to the base spec. The base spec is a
    # frozen dataclass; we use ``object.__setattr__`` (the same idiom
    # ``helpers.fuse.compile_fn`` uses) to drop the FusedEpilogue into
    # the ``_fused_epilogue`` side-channel slot the cshuffle epilogue
    # reads via ``getattr(spec, "_fused_epilogue", None)``.
    fused = _build_fused_epilogue(spec)

    # Build a fresh copy of the base spec so we don't mutate a spec
    # the caller may reuse elsewhere. The frozen dataclass forces us
    # through ``dataclasses.replace`` for any field change; here we
    # just rename so the lowered kernel symbol carries the multi-D
    # suffix, then attach the fused epilogue.
    import dataclasses

    base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name())
    object.__setattr__(base_renamed, "_fused_epilogue", fused)

    return build_universal_gemm(base_renamed)


def gemm_multi_d_signature(spec: GemmMultiDSpec):
    """Manifest-style signature for the multi-D GEMM kernel.

    The pointer order is: ``A, B, D0, ..., D{n-1}, C, then M / N / K``.
    """
    sb = (
        SignatureBuilder()
        .ptr("A", spec.base.data.dtype_a)
        .ptr("B", spec.base.data.dtype_b)
    )
    for name, _op in spec.d_operands:
        sb.ptr(name, spec.d_dtype)
    sb.ptr("C", spec.base.data.dtype_c).scalar("M", "i32").scalar("N", "i32").scalar(
        "K", "i32"
    )
    if spec.base.batched:
        sb.scalar("stride_a", "i32").scalar("stride_b", "i32").scalar("stride_c", "i32")
    return sb.build()


def gemm_multi_d_grid(spec: GemmMultiDSpec, m: int, n: int, batch: int = 1):
    """Same grid as :func:`build_universal_gemm`: ``(N_tiles, M_tiles, batch)``."""
    t = spec.base.tile
    nx = (n + t.tile_n - 1) // t.tile_n
    ny = (m + t.tile_m - 1) // t.tile_m
    return (nx, ny, batch if spec.base.batched else 1)


__all__ = [
    "DOp",
    "MAX_D",
    "GemmMultiDSpec",
    "build_gemm_multi_d",
    "gemm_multi_d_grid",
    "gemm_multi_d_signature",
    "is_valid_spec",
    # Re-exports for convenience so callers can build the base spec
    # without importing a second module.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
