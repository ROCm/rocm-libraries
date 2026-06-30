# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture gating for the tiled MFMA unified-attention kernels.

There are two arch-specific tiled-attention builders, and this module decides
which targets each may run on:

  * **gfx950 (CDNA4)** — ``instances/gfx950/attention_tiled_2d.py`` is built
    around two gfx950 ISA features: the **wide-K MFMA atoms**
    ``mfma_f32_16x16x32_{f16,bf16}`` / ``mfma_f32_32x32x16_{f16,bf16}`` (the QK
    and PV matmuls use the wide-K form; K-step 32 / 16 head-dim elements per
    atom) and the **LDS transpose reads** ``ds_read_b64_tr_b16`` /
    ``ds_read_tr_b8`` (the PV ``B`` operand). Requesting either on a target that
    lacks it makes the AMDGPU backend abort with ``LLVM ERROR: Cannot select
    intrinsic`` -- a hard process crash, not a recoverable Python error.
  * **gfx942 (CDNA3)** — ``instances/gfx942/attention_tiled_2d.py`` is the
    narrow-atom variant: QK and PV both use the ``16x16x16`` f16/bf16 atom
    (K-step 16), and the V ``B`` operand is built from ordinary strided LDS
    loads, so it needs neither the wide-K atoms nor ``ds_read_*_tr_*``.

Rather than let comgr crash the whole process when a caller asks for a target
that has no matching path, the builders validate up front and raise a clean
structured error. Routing an admitted arch to the correct builder is the
caller's job (``instances/common/attention_unified.py:_tiled_2d_impl``);
admitting gfx942 here without that routing would drive the gfx950 builder onto
gfx942 ISA and crash comgr -- so the routing seam and this relaxation land
together.

The helpers here are deliberately catalog-driven (they query
:class:`rocke.core.arch.ArchTarget`) so a future architecture lights up the
matching path automatically. The one wrinkle is that the MFMA catalog lists the
``32x32x16`` bf16 atom as absent even on gfx950 (it compiles fine in practice),
so the wide-K check folds in a ``has_ds_read_tr`` cross-check: a target that
advertises the transpose-read family is taken to have the wide-K atoms too.
"""

from __future__ import annotations

from typing import Tuple

from ...core.arch import ArchTarget


def _wide_k_mfma_available(target: ArchTarget) -> bool:
    """True iff this target has the wide-K (K=32 / 32x32x16) f16/bf16 MFMA.

    Sourced from the MMA catalog (``16x16x32`` f16 is the canonical wide-K
    marker and is correctly reported per arch), with a transpose-read
    cross-check so a target that advertises the gfx950 ``ds_read_*_tr_*``
    family is always treated as wide-K capable even where the catalog is
    incomplete (the ``32x32x16`` bf16 atom).
    """
    if target.mma.has_shape(
        a_dtype="f16", b_dtype="f16", c_dtype="fp32", m=16, n=16, k=32
    ):
        return True
    return bool(target.memory.has_ds_read_tr)


# Arches that have a dedicated narrow-atom / strided-V tiled-2D variant routed
# by ``instances/common/attention_unified._tiled_2d_impl``. Only these may be
# admitted on the narrow path by ``validate_tiled_attention_arch`` -- an arch
# that is narrow-atom-capable but NOT listed here has no routed builder and
# would fall through to the gfx950 builder (wide-K / ds_read_tr ISA) and crash
# comgr. Keep this in lockstep with ``_tiled_2d_impl``.
_NARROW_TILED_2D_ARCHES = frozenset({"gfx942"})


def _narrow_k_mfma_available(target: ArchTarget) -> bool:
    """True iff this target has the narrow ``16x16x16`` f16 AND bf16 MFMA.

    This is the atom the gfx942 (CDNA3) tiled-2D variant
    (``instances/gfx942/attention_tiled_2d.py``) is built on: QK and PV both
    use ``16x16x16`` with a K-step of 16, and the V ``B`` operand is built from
    ordinary strided LDS loads instead of ``ds_read_*_tr_*``. Both f16 and bf16
    must be present (bf16 lowers through the ``_1k`` intrinsic).
    """
    return target.mma.has_shape(
        a_dtype="f16", b_dtype="f16", c_dtype="fp32", m=16, n=16, k=16
    ) and target.mma.has_shape(
        a_dtype="bf16", b_dtype="bf16", c_dtype="fp32", m=16, n=16, k=16
    )


def validate_tiled_attention_arch(arch: str) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for running a tiled attention kernel on ``arch``.

    Two arch families are admitted:

    * **gfx950 (CDNA4)** — the wide-K MFMA atoms (``mfma_f32_16x16x32`` /
      ``mfma_f32_32x32x16``) AND LDS transpose reads (``ds_read_b64_tr_b16``).
      This is the original gfx950 kernel's requirement, preserved unchanged.
    * **gfx942 (CDNA3)** — the narrow ``16x16x16`` f16/bf16 atom, with the V
      ``B`` operand built from strided LDS loads (no transpose read required).
      The dedicated ``instances/gfx942`` variant runs this path, so gfx942 does
      NOT need wide-K or ``has_ds_read_tr``.

    This predicate lets the selector / dispatcher drop a target that has neither
    path with a structured reason instead of letting comgr abort the process at
    lower time. Routing the admitted arch to the matching builder is the
    caller's responsibility (``_tiled_2d_impl(arch)``); admitting gfx942 here
    without that routing would drive the gfx950 builder onto gfx942 ISA.
    """
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    # gfx950 wide-K + transpose-read path (unchanged).
    if _wide_k_mfma_available(target) and target.memory.has_ds_read_tr:
        return True, "ok"
    # Narrow-atom + strided-V path. Admission is restricted to arches that
    # ``_tiled_2d_impl`` actually routes to a dedicated narrow variant -- only
    # gfx942 today. Admitting any narrow-atom-capable arch here would be unsafe:
    # an arch without a routed narrow builder (e.g. gfx90a) would fall through
    # ``_tiled_2d_impl`` to the gfx950 builder and emit wide-K / ds_read_tr ISA
    # it cannot run, crashing comgr. Adding another narrow arch means extending
    # BOTH this set and ``_tiled_2d_impl``.
    if arch in _NARROW_TILED_2D_ARCHES and _narrow_k_mfma_available(target):
        return True, "ok"
    if not _wide_k_mfma_available(target):
        return (
            False,
            f"tiled attention requires either the wide-K MFMA atoms "
            f"(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950) or, on a narrow "
            f"variant arch ({', '.join(sorted(_NARROW_TILED_2D_ARCHES))}), the "
            f"16x16x16 f16/bf16 atom; neither path is available on {arch}",
        )
    return (
        False,
        f"tiled attention requires LDS transpose reads "
        f"(ds_read_b64_tr_b16) for the wide-K path, absent on {arch}",
    )


def require_tiled_attention_arch(arch: str) -> None:
    """Raise :class:`NotImplementedError` if ``arch`` cannot run the kernel.

    Called at the very top of the tiled attention builders so a gfx942
    request fails with a clean Python error *before* any IR is emitted and
    long before comgr would hit ``LLVM ERROR: Cannot select intrinsic``.
    """
    ok, reason = validate_tiled_attention_arch(arch)
    if not ok:
        raise NotImplementedError(reason)
