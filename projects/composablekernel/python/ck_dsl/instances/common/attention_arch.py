# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture gating for the tiled MFMA unified-attention kernels.

The tiled 2D / 3D attention kernels are built around two CDNA4 (gfx950)
ISA features that do **not** exist on CDNA3 (gfx942):

  * the **wide-K MFMA atoms** ``mfma_f32_16x16x32_{f16,bf16}`` and
    ``mfma_f32_32x32x16_{f16,bf16}`` (the QK and PV matmuls use the wide-K
    form as their core math; the K-step is 32 / 16 head-dim elements per
    atom). Requesting one of these intrinsics on gfx942 makes the AMDGPU
    backend abort with ``LLVM ERROR: Cannot select intrinsic`` -- a hard
    process crash, not a recoverable Python error.
  * the **LDS transpose reads** ``ds_read_b64_tr_b16`` / ``ds_read_tr_b8``
    used to fetch the PV ``B`` operand in the MFMA distribution. These map
    to gfx950's ``ds_read_*_tr_*`` family (``ArchTarget.memory.has_ds_read_tr``)
    and are not available on gfx942.

Because both features are load-bearing for the kernel's hot loop -- there is
no narrow-atom / LDS-roundtrip fallback in this module -- the tiled attention
kernels are **gfx950-only**. Rather than let comgr crash the whole process
when a caller asks for gfx942, the builders validate the target up front and
raise a clean structured error.

The helpers here are deliberately catalog-driven (they query
:class:`ck_dsl.core.arch.ArchTarget`) so that if a future architecture adds
the wide-K atoms + transpose reads, the kernels light up automatically with
no edits here. The one wrinkle is that the MFMA catalog currently lists the
``32x32x16`` bf16 atom as absent even on gfx950 (it compiles fine in
practice), so the wide-K check folds in a ``has_ds_read_tr`` cross-check: on
a target that already advertises the transpose-read family the wide-K atoms
are taken to exist.
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


def validate_tiled_attention_arch(arch: str) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for running a tiled attention kernel on ``arch``.

    The tiled MFMA attention kernels require gfx950's wide-K MFMA atoms and
    LDS transpose reads (see the module docstring). This predicate lets the
    selector / dispatcher drop a gfx942 target with a structured reason
    instead of letting comgr abort the process at lower time.
    """
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if not _wide_k_mfma_available(target):
        return (
            False,
            f"tiled attention requires the wide-K MFMA atoms "
            f"(mfma_f32_16x16x32 / mfma_f32_32x32x16), absent on {arch}",
        )
    if not target.memory.has_ds_read_tr:
        return (
            False,
            f"tiled attention requires LDS transpose reads "
            f"(ds_read_b64_tr_b16), absent on {arch}",
        )
    return True, "ok"


def require_tiled_attention_arch(arch: str) -> None:
    """Raise :class:`NotImplementedError` if ``arch`` cannot run the kernel.

    Called at the very top of the tiled attention builders so a gfx942
    request fails with a clean Python error *before* any IR is emitted and
    long before comgr would hit ``LLVM ERROR: Cannot select intrinsic``.
    """
    ok, reason = validate_tiled_attention_arch(arch)
    if not ok:
        raise NotImplementedError(reason)
