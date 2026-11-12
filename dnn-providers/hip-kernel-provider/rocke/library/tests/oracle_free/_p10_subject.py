# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subject adapter of the coordinate-codeword placement probe (P10).

A case is resolved through the uniform-probability adapter, which already
asserts the architecture's dense candidate, the requested shape, and a
single-pass kernel with no sinks, LSE, ragged tail, bottom-right band, paging or
varlen. This module adds what the placement derivation needs on top.

Tile geometry, grid, persistence and scheduling knobs stay free, as in the
uniform-probability lane: a retune inside the scope keeps running. Dispatch and
the kernel modules are imported only inside the functions that use them, so CPU
collection needs neither a GPU nor ROCm.
"""

from __future__ import annotations

from ._analytical import UniformCase
from ._dense_subject import ResolvedDenseSubject, resolve_dense_subject
from ._p10_analytical import P10_QUICK_CASES, TileSchedule, require_p10_scope

# The derivation is for the gfx942 body (32x32x8 MFMA atom, its tile loop and its
# output epilogue). Another architecture needs its own.
P10_ARCHES = ("gfx942",)

P10_SUBJECT_CASES: tuple[tuple[str, UniformCase], ...] = tuple(
    (arch, case) for arch in P10_ARCHES for case in P10_QUICK_CASES
)


def _verify_p10_resolved_spec(tuning_spec) -> None:
    """Refuse a resolved schedule the placement derivation does not model.

    With ``window_wave_skip`` a wave skips the score, softmax and PV step of a
    tile that is fully masked for all of its rows. The bounds on what such a tile
    leaves in the accumulators are derived for the kernel that visits it, so the
    skipping schedule is refused before any launch, as a failure."""
    spec = tuning_spec.kernel_spec
    if spec.window_wave_skip:
        raise AssertionError(
            "the coordinate-codeword check requires window_wave_skip off, "
            f"got window_wave_skip={spec.window_wave_skip!r}"
        )


def resolve_p10_subject(arch: str, case: UniformCase) -> ResolvedDenseSubject:
    """Resolve one case on one architecture, inside the probe's scope."""
    if arch not in P10_ARCHES:
        raise ValueError(
            f"no coordinate-codeword subject - arch={arch!r}, "
            f"expected one of {P10_ARCHES}"
        )
    require_p10_scope(case)
    subject = resolve_dense_subject(arch, case)
    _verify_p10_resolved_spec(subject.tuning_spec)
    return subject


def p10_kernel_key(arch: str, kernel_spec) -> tuple:
    """Launcher-cache identity of the kernel a dense spec compiles to.

    The dense runner keeps one compiled kernel per key, so two specs with equal
    keys run the same binary. On the runtime-shape path the batch, both sequence
    lengths and both head counts are kernel arguments and are not part of it."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    return attention_dense_cache_key(kernel_spec, arch=arch)


def p10_schedule(subject: ResolvedDenseSubject) -> TileSchedule:
    """Tile geometry of the kernel that dispatch resolved for a subject.

    It is read from the resolved spec and never pinned: a retune changes what the
    emulation and the masked-tile analysis follow instead of failing the gate."""
    spec = subject.tuning_spec.kernel_spec
    return TileSchedule(block_m=int(spec.block_m), block_n=int(spec.block_n))
