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

import importlib
import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace

import numpy as np

from ._analytical import UniformCase
from ._dense_subject import (
    DENSE_SUBJECTS,
    ResolvedDenseSubject,
    _validate_device_buffers,
    _validate_host_bits,
    _verify_resolved_spec,
    resolve_dense_subject,
)
from ._p10_analytical import (
    P10_QUICK_CASES,
    GuardedOutput,
    GuardLayout,
    TileSchedule,
    guard_layout,
    require_p10_scope,
)

_ELEMENT_BYTES = 2

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


def _hip_backend() -> SimpleNamespace:
    """The HIP runtime pieces a launch needs, imported on use."""
    from rocke.runtime import DeviceMem, Runtime
    from rocke.runtime.host_buffers import as_u8_buffer

    return SimpleNamespace(
        DeviceMem=DeviceMem, Runtime=Runtime, as_u8_buffer=as_u8_buffer
    )


def _launch_guarded(
    launch: Callable[..., None],
    backend: SimpleNamespace,
    host: Mapping[str, np.ndarray],
    layout: GuardLayout,
) -> np.ndarray:
    """Upload ``q``/``k``/``v`` and the guarded ``out``, launch, read ``out`` back.

    The kernel gets the address of the logical tensor inside the allocation, so
    the guards on both sides are device memory it does not own. ``launch`` is
    called with the three input buffers and that interior address."""
    runtime = backend.Runtime()
    device = {name: backend.DeviceMem(array.nbytes) for name, array in host.items()}
    _validate_device_buffers(
        device, {name: int(array.nbytes) for name, array in host.items()}
    )
    for name, array in host.items():
        runtime.memcpy_h2d(
            device[name].ptr(), backend.as_u8_buffer(array), array.nbytes
        )
    interior = int(device["out"].ptr()) + layout.guard * _ELEMENT_BYTES
    launch(q=device["q"], k=device["k"], v=device["v"], out=interior)
    runtime.sync()
    raw = host["out"]
    runtime.memcpy_d2h(backend.as_u8_buffer(raw), device["out"].ptr(), raw.nbytes)
    return raw


def run_p10_case(
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
    resolved = resolve_p10_subject(arch, case) if subject is None else subject
    if (resolved.arch, resolved.case_id) != (arch, case.case_id):
        raise ValueError(
            f"run_p10_case failed - subject is {resolved.arch}/{resolved.case_id}, "
            f"expected {arch}/{case.case_id}"
        )
    require_p10_scope(case)
    _validate_host_bits(case, q, k, v)
    _verify_resolved_spec(arch, case, resolved.tuning_spec)
    _verify_p10_resolved_spec(resolved.tuning_spec)
    layout = guard_layout(case, p10_schedule(resolved))
    runner = importlib.import_module(DENSE_SUBJECTS[arch].runner_module)
    spec = resolved.tuning_spec.kernel_spec

    def launch(*, q, k, v, out) -> None:
        # Without sinks, LSE or a KV split the runner only packs device pointers,
        # so it takes the interior address of the output as a plain integer.
        runner.run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=1.0 / math.sqrt(case.head_dim)
        )

    host = {"q": q.copy(), "k": k.copy(), "v": v.copy(), "out": layout.new_buffer()}
    return GuardedOutput(layout, _launch_guarded(launch, _hip_backend(), host, layout))
