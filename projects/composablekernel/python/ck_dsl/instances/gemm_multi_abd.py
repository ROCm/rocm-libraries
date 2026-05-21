# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multi A/B/D GEMM kernel instance (CK Tile ``22_gemm_multi_abd`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/22_gemm_multi_abd``.
Generalises :mod:`gemm_multi_d` to also accept multiple ``A`` and
multiple ``B`` operands that get combined via an ``AElementWise`` /
``BElementWise`` lambda before entering the MFMA path::

    A = AElementWise(A0, A1, ..., A_{na-1})
    B = BElementWise(B0, B1, ..., B_{nb-1})
    E = CDEElementWise(A * B, D0, D1, ..., D_{nd-1})

Typical use cases:

* Quantised GEMM with separate weight + per-column-scale matrices,
  where ``B = dequant(B0_int8, B1_scale)`` happens inside the load phase.
* Two A matrices that get summed (e.g. ``A = A0 + A1``) — useful for
  residual fan-in before a projection.
* Multi-D residuals / gates (same as :mod:`gemm_multi_d`).

What v1 covers:

* ``num_d`` operands with per-D ``add`` / ``mul`` ops (full parity
  with :func:`build_gemm_multi_d`).
* ``num_a == 1`` and ``num_b == 1`` only. The spec surface accepts
  larger tuples so callers can express their final intent today, but
  ``build_gemm_multi_abd`` raises a precise ``NotImplementedError`` if
  ``num_a > 1`` or ``num_b > 1`` until the v2 load-combine path lands
  (tracked as ``w1-gemm-multi-abd-load-combine`` in the wave plan).

The kernel signature in v1 (``num_a==num_b==1``) is exactly the same
as :func:`build_gemm_multi_d`'s::

    (A: ptr, B: ptr, D0: ptr, ..., C: ptr, M: i32, N: i32, K: i32)

When the v2 multi-A / multi-B path lands, the signature will grow
``A0..A{na-1}`` and ``B0..B{nb-1}`` between ``A`` and ``D0``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Tuple

from ..core.ir import KernelDef
from ..helpers.spec import SignatureBuilder, kernel_name_join
from .gemm_multi_d import (
    GemmMultiDSpec,
    MAX_D,
    build_gemm_multi_d,
    gemm_multi_d_grid,
)
from .gemm_multi_d import DOp  # noqa: F401
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)


MAX_A = 4
MAX_B = 4
AOperand = Tuple[str, str]  # (param_name, dtype_str)
BOperand = Tuple[str, str]
DOperand = Tuple[str, str]  # (param_name, "add" | "mul")


@dataclass(frozen=True)
class GemmMultiAbdSpec:
    """One concrete multi-A/B/D GEMM kernel configuration.

    See module docstring for the v1 vs v2 split. v1 requires
    ``num_a == num_b == 1`` and uses the standard A/B as the single
    inputs to MFMA; ``d_operands`` go through the existing fused
    cshuffle epilogue.

    ``a_operands`` / ``b_operands`` default to a single ``(name, dtype)``
    matching ``base.data.dtype_a`` / ``base.data.dtype_b``, so the
    common case (one A, one B, multiple D's) reads naturally.
    """

    base: UniversalGemmSpec
    a_operands: Tuple[AOperand, ...] = field(default_factory=lambda: (("A", "fp16"),))
    b_operands: Tuple[BOperand, ...] = field(default_factory=lambda: (("B", "fp16"),))
    d_operands: Tuple[DOperand, ...] = ()
    d_dtype: str = "fp16"
    name: str = "ck_dsl_gemm_multi_abd"

    @property
    def num_a(self) -> int:
        return len(self.a_operands)

    @property
    def num_b(self) -> int:
        return len(self.b_operands)

    @property
    def num_d(self) -> int:
        return len(self.d_operands)

    def kernel_name(self) -> str:
        d_suffix = "_".join(f"{n}{op}" for n, op in self.d_operands) or "noD"
        return kernel_name_join(
            self.name,
            self.base.kernel_name(),
            f"ma{self.num_a}",
            f"mb{self.num_b}",
            f"md{self.num_d}",
            d_suffix,
            self.d_dtype,
        )


def is_valid_spec(spec: GemmMultiAbdSpec) -> Tuple[bool, str]:
    if spec.num_a == 0:
        return False, "a_operands must contain at least one entry"
    if spec.num_b == 0:
        return False, "b_operands must contain at least one entry"
    if spec.num_a > MAX_A:
        return False, f"num_a {spec.num_a} > MAX_A {MAX_A}"
    if spec.num_b > MAX_B:
        return False, f"num_b {spec.num_b} > MAX_B {MAX_B}"
    if spec.num_d > MAX_D:
        return False, f"num_d {spec.num_d} > MAX_D {MAX_D}"
    if spec.base.trait.epilogue != "cshuffle":
        return False, (
            "GemmMultiAbd requires base.trait.epilogue='cshuffle'; "
            f"got {spec.base.trait.epilogue!r}"
        )
    # Validate D ops + names (delegated style; share rules with multi-d).
    seen = set()
    for name, op in spec.d_operands:
        if op not in ("add", "mul"):
            return False, f"D op {op!r} not in {{'add','mul'}}"
        if name in seen:
            return False, f"duplicate D param_name {name!r}"
        if name in ("M", "N", "K", "C"):
            return False, f"D param_name {name!r} reserved"
        seen.add(name)
    # Validate A / B name uniqueness across the three pools.
    a_names = {n for n, _ in spec.a_operands}
    b_names = {n for n, _ in spec.b_operands}
    d_names = {n for n, _ in spec.d_operands}
    overlap_ab = a_names & b_names
    if overlap_ab:
        return False, f"A and B share param_names {sorted(overlap_ab)}"
    overlap_ad = a_names & d_names
    if overlap_ad:
        return False, f"A and D share param_names {sorted(overlap_ad)}"
    overlap_bd = b_names & d_names
    if overlap_bd:
        return False, f"B and D share param_names {sorted(overlap_bd)}"
    return True, "ok"


def build_gemm_multi_abd(spec: GemmMultiAbdSpec) -> KernelDef:
    """Build the IR for one multi-A/B/D GEMM instance.

    v1 restriction: ``num_a == num_b == 1``. The single A and B drive
    the standard MFMA loop and the D list feeds the fused cshuffle
    epilogue (delegates to :func:`build_gemm_multi_d`).

    Raises :class:`NotImplementedError` if ``num_a > 1`` or
    ``num_b > 1`` so the v2 gap is loud rather than silently wrong.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid GemmMultiAbd spec: {why}")
    if spec.num_a > 1 or spec.num_b > 1:
        raise NotImplementedError(
            f"multi-A ({spec.num_a}) / multi-B ({spec.num_b}) load-combine "
            "is a v2 gap; today only num_a == num_b == 1 is implemented. "
            "Track this on the wave plan as the v2 follow-on to "
            "'w1-gemm-multi-abd'."
        )

    # Reuse the multi-D path with a renamed base spec so the kernel
    # symbol carries the multi-ABD suffix.
    base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name())
    md_spec = GemmMultiDSpec(
        base=base_renamed,
        d_operands=spec.d_operands,
        d_dtype=spec.d_dtype,
        name=spec.kernel_name(),
    )
    if spec.num_d == 0:
        # No D operands -> plain GEMM. Delegate directly so we don't
        # hit multi-d's "empty D list" guard.
        from .gemm_universal import build_universal_gemm

        return build_universal_gemm(base_renamed)
    return build_gemm_multi_d(md_spec)


def gemm_multi_abd_signature(spec: GemmMultiAbdSpec):
    """Manifest-style signature: ``A0 .. A{na-1}, B0 .. B{nb-1}, D0 .. D{nd-1}, C, M, N, K``.

    v1 (num_a==num_b==1) collapses to ``A, B, D0 .. , C, M, N, K`` —
    same surface the multi-D launcher already speaks.
    """
    sb = SignatureBuilder()
    for name, dtype in spec.a_operands:
        sb.ptr(name, dtype)
    for name, dtype in spec.b_operands:
        sb.ptr(name, dtype)
    for name, _op in spec.d_operands:
        sb.ptr(name, spec.d_dtype)
    sb.ptr("C", spec.base.data.dtype_c).scalar("M", "i32").scalar("N", "i32").scalar(
        "K", "i32"
    )
    if spec.base.batched:
        sb.scalar("stride_a", "i32").scalar("stride_b", "i32").scalar("stride_c", "i32")
    return sb.build()


def gemm_multi_abd_grid(spec: GemmMultiAbdSpec, m: int, n: int, batch: int = 1):
    """Same launch grid as :func:`build_universal_gemm`."""
    return gemm_multi_d_grid(
        GemmMultiDSpec(
            base=spec.base,
            d_operands=spec.d_operands or (("D0", "add"),),  # type: ignore[arg-type]
            d_dtype=spec.d_dtype,
        ),
        m,
        n,
        batch,
    )


__all__ = [
    "AOperand",
    "BOperand",
    "DOperand",
    "GemmMultiAbdSpec",
    "MAX_A",
    "MAX_B",
    "MAX_D",
    "build_gemm_multi_abd",
    "gemm_multi_abd_grid",
    "gemm_multi_abd_signature",
    "is_valid_spec",
    # Re-exports for convenience.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
