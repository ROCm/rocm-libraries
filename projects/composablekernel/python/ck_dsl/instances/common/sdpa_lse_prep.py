# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FMHA-backward softmax-statistics prep kernel.

The hipDNN forward emits a single natural-log LSE ``stats`` tensor,
head-major contiguous ``[B, Hq, Sq]`` (linear index
``(b*Hq + h)*Sq + q``). The CK FMHA-backward kernel
(:func:`ck_dsl.instances.common.fmha_bwd.build_fmha_bwd`) instead reads
two SEPARATE per-batch q-major inputs:

  * ``M_saved`` -- the saved softmax max, indexed ``q*Hq + h`` per batch,
    in the **log2** domain.
  * ``L_saved`` -- the saved softmax denominator, all ones.

Feeding the bwd kernel ``M = LSE2`` (LSE in the log2 domain) and
``L = 1`` reproduces the softmax probability exactly:
``p = exp2(s_log2 - M) / L = exp2(s_log2 - LSE2)`` (validated separately
via the ``--stats-mode bridge`` path of ``fmha_bwd_verify_hip``).

This kernel performs the two transforms the provider needs, fused, for
every ``(b, h, q)``::

    M_out[(b*Sq + q)*Hq + h] = stats[(b*Hq + h)*Sq + q] * LOG2E
    L_out[(b*Sq + q)*Hq + h] = 1.0

i.e. a head-major -> per-batch q-major transpose plus a natural-log ->
log2 rescale on ``M_out``, and a constant-one fill on ``L_out``.
``LOG2E = log2(e) = 1.4426950408889634``.

One thread handles one ``(b, h, q)`` element. The body is pure scalar
f32 load / multiply / store with a single ``q < Sq`` guard, so the
emitted IR is architecture-portable (identical on gfx942 and gfx950);
``arch`` is threaded through :func:`is_valid_spec` only to fail closed on
an unknown target via :class:`ck_dsl.core.arch.ArchTarget`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.spec import SignatureBuilder, kernel_name_join


__all__ = [
    "SdpaLsePrepSpec",
    "build_sdpa_lse_prep",
    "is_valid_spec",
    "sdpa_lse_prep_grid",
    "sdpa_lse_prep_signature",
]


# log2(e); the natural-log LSE -> log2-domain rescale factor.
LOG2E = 1.4426950408889634

# One thread per q-position within a (b, h) row; 64-wide CTA (one wave64
# warp on CDNA). The grid x-axis tiles Sq by this block.
_BLOCK_SIZE = 64


@dataclass(frozen=True)
class SdpaLsePrepSpec:
    """One FMHA-bwd stats-prep kernel instance.

    ``B`` / ``Hq`` / ``Sq`` are the batch, query-head, and query-seqlen
    extents of the forward ``stats`` tensor (head-major ``[B, Hq, Sq]``).
    """

    B: int
    Hq: int
    Sq: int
    name: str = "ck_dsl_sdpa_lse_prep"

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"B{self.B}",
            f"HQ{self.Hq}",
            f"Q{self.Sq}",
        )


def is_valid_spec(spec: SdpaLsePrepSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one stats-prep config on ``arch``.

    The body issues no MMA atoms and no atomics -- it is a plain
    transpose + rescale -- so the only architecture fact consulted is
    that ``arch`` resolves to a known :class:`ck_dsl.core.arch.ArchTarget`
    (fail closed on an unknown target). Shape validation is the obvious
    positivity check on the three extents.
    """
    from ...core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.B <= 0:
        return False, f"B must be > 0 (got {spec.B})"
    if spec.Hq <= 0:
        return False, f"Hq must be > 0 (got {spec.Hq})"
    if spec.Sq <= 0:
        return False, f"Sq must be > 0 (got {spec.Sq})"
    return True, "ok"


def _declare_params(b: IRBuilder) -> Tuple[Value, Value, Value, Value, Value, Value]:
    """Declare the kernel ABI in fixed order (shared by build + sig).

    ABI ORDER -- this is the fixed contract the provider depends on:

      0. ``stats`` : f32 ptr, read-only  (head-major source [B, Hq, Sq])
      1. ``M_out`` : f32 ptr, write       (per-batch q-major dest)
      2. ``L_out`` : f32 ptr, write       (per-batch q-major dest)
      3. ``B``     : i32 scalar
      4. ``Hq``    : i32 scalar
      5. ``Sq``    : i32 scalar

    Returns the declared param Values in that order.
    """
    stats = b.param(
        "stats", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    m_out = b.param(
        "M_out", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    l_out = b.param(
        "L_out", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    p_B = b.param("B", I32)
    p_Hq = b.param("Hq", I32)
    p_Sq = b.param("Sq", I32)
    return stats, m_out, l_out, p_B, p_Hq, p_Sq


def build_sdpa_lse_prep(spec: SdpaLsePrepSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one FMHA-bwd stats-prep instance.

    Grid (set by the caller via :func:`sdpa_lse_prep_grid`):
    ``(ceil(Sq/64), Hq, B)`` -- one CTA per ``(Sq-tile, head, batch)``,
    one thread per q-position. Inside the kernel::

        q = block_id_x * 64 + thread_id_x
        h = block_id_y
        b = block_id_z
        if q < Sq:
            in_off  = (b*Hq + h)*Sq + q     # head-major source
            out_off = (b*Sq + q)*Hq + h     # per-batch q-major dest
            M_out[out_off] = stats[in_off] * LOG2E
            L_out[out_off] = 1.0

    The default ``arch="gfx950"`` is byte-for-byte backward compatible;
    the emitted IR does not depend on the arch (only validation does).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid sdpa_lse_prep spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _BLOCK_SIZE

    stats, m_out, l_out, _p_B, p_Hq, p_Sq = _declare_params(b)

    # Grid coords: block_id_x tiles Sq, block_id_y = head, block_id_z =
    # batch. q = bx*BLOCK + tid; one thread owns one q-position.
    q = b.add(b.mul(b.block_id_x(), b.const_i32(_BLOCK_SIZE)), b.thread_id_x())
    h = b.block_id_y()
    bb = b.block_id_z()

    in_bounds = b.cmp_lt(q, p_Sq)
    with b.scf_if(in_bounds):
        # Source: head-major [B, Hq, Sq] -> (b*Hq + h)*Sq + q.
        in_off = b.add(b.mul(b.add(b.mul(bb, p_Hq), h), p_Sq), q)
        # Dest: per-batch q-major -> (b*Sq + q)*Hq + h.
        out_off = b.add(b.mul(b.add(b.mul(bb, p_Sq), q), p_Hq), h)
        # M_out: natural-log LSE -> log2 domain.
        val = b.fmul(b.global_load_f32(stats, in_off), b.const_f32(LOG2E))
        b.global_store(m_out, out_off, val, align=4)
        # L_out: constant ones (the bwd kernel's saved denominator).
        b.global_store(l_out, out_off, b.const_f32(1.0), align=4)

    b.ret()
    return b.kernel


def sdpa_lse_prep_grid(spec: SdpaLsePrepSpec) -> Tuple[int, int, int]:
    """Launch grid ``(ceil(Sq/64), Hq, B)`` for one stats-prep instance."""
    grid_x = (spec.Sq + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    return (grid_x, spec.Hq, spec.B)


def sdpa_lse_prep_signature(spec: SdpaLsePrepSpec):
    """ABI signature (stats, M_out, L_out, B, Hq, Sq) for one instance."""
    return (
        SignatureBuilder()
        .ptr("stats", "f32")
        .ptr("M_out", "f32")
        .ptr("L_out", "f32")
        .scalar("B", "i32")
        .scalar("Hq", "i32")
        .scalar("Sq", "i32")
        .build()
    )
