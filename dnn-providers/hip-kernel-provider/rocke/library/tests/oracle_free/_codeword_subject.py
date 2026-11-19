# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subject adapter of the coordinate-codeword placement probe.

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

import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace

import numpy as np

from ._analytical import UniformCase
from ._codeword_analytical import (
    CODEWORD_QUICK_CASES,
    GuardedOutput,
    GuardLayout,
    TileSchedule,
    guard_layout,
    require_codeword_scope,
)
from ._dense_subject import (
    ResolvedDenseSubject,
    dense_runner_launch,
    launch_dense_buffers,
    resolve_dense_subject,
    validated_dense_subject,
)

_ELEMENT_BYTES = 2

# The derivation is for the gfx942 body (32x32x8 MFMA atom, its tile loop and its
# output epilogue). Another architecture needs its own.
CODEWORD_ARCHES = ("gfx942",)

CODEWORD_SUBJECT_CASES: tuple[tuple[str, UniformCase], ...] = tuple(
    (arch, case) for arch in CODEWORD_ARCHES for case in CODEWORD_QUICK_CASES
)


def _verify_codeword_resolved_spec(tuning_spec) -> None:
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


def resolve_codeword_subject(arch: str, case: UniformCase) -> ResolvedDenseSubject:
    """Resolve one case on one architecture, inside the probe's scope."""
    if arch not in CODEWORD_ARCHES:
        raise ValueError(
            f"no coordinate-codeword subject - arch={arch!r}, "
            f"expected one of {CODEWORD_ARCHES}"
        )
    require_codeword_scope(case)
    subject = resolve_dense_subject(arch, case)
    _verify_codeword_resolved_spec(subject.tuning_spec)
    return subject


def codeword_kernel_key(arch: str, kernel_spec) -> tuple:
    """Launcher-cache identity of the kernel a dense spec compiles to.

    The dense runner keeps one compiled kernel per key, so two specs with equal
    keys run the same binary. On the runtime-shape path the batch, both sequence
    lengths and both head counts are kernel arguments and are not part of it."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    return attention_dense_cache_key(kernel_spec, arch=arch)


def codeword_schedule(subject: ResolvedDenseSubject) -> TileSchedule:
    """Tile geometry of the kernel that dispatch resolved for a subject.

    It is read from the resolved spec and never pinned: a retune changes what the
    emulation and the masked-tile analysis follow instead of failing the gate."""
    spec = subject.tuning_spec.kernel_spec
    return TileSchedule(block_m=int(spec.block_m), block_n=int(spec.block_n))


def launch_guarded(
    launch: Callable[..., None],
    host: Mapping[str, np.ndarray],
    layout: GuardLayout,
    *,
    backend: SimpleNamespace | None = None,
) -> GuardedOutput:
    """Launch into the guarded ``out`` allocation of ``host`` and read it back.

    The kernel gets the address of the logical tensor inside the allocation, so
    the guards on both sides are device memory it does not own. The device path
    is the subject adapter's, with the guard as the output offset."""
    raw = launch_dense_buffers(
        launch, host, out_offset=layout.guard * _ELEMENT_BYTES, backend=backend
    )
    return GuardedOutput(layout, raw)


def run_codeword_case(
    arch: str,
    case: UniformCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    subject: ResolvedDenseSubject | None = None,
) -> GuardedOutput:
    """Launch one case on raw input bits into a guarded output allocation.

    ``q``/``k``/``v`` are ``uint16`` FP16/BF16 bit arrays in the runner layout.
    The output is allocated with a guard of one query tile on each side, prefilled
    with the canary and sentinel patterns, and read back whole and without any
    float conversion. The launch goes through the architecture's public dense
    runner, as in the uniform-probability lane; judging the bits is left to the
    analyzer."""
    require_codeword_scope(case)
    if subject is None:
        subject = resolve_codeword_subject(arch, case)
    resolved = validated_dense_subject(arch, case, q, k, v, subject=subject)
    _verify_codeword_resolved_spec(resolved.tuning_spec)
    layout = guard_layout(case, codeword_schedule(resolved))
    launch = dense_runner_launch(
        arch, resolved.tuning_spec.kernel_spec, scale=1.0 / math.sqrt(case.head_dim)
    )
    host = {"q": q.copy(), "k": k.copy(), "v": v.copy(), "out": layout.new_buffer()}
    return launch_guarded(launch, host, layout)
