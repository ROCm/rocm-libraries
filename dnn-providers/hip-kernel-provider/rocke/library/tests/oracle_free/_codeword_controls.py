# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subject-side negative controls of the coordinate-codeword placement probe.

A control launches a kernel with one deliberate defect at one seam, on the
probe's own inputs and into the same guarded allocation. The analyzer must
reject it through the channel the kill matrix names for that seam. There is one
control per seam, named by the owned mutation it realizes.

An emitted control is a private build. Inside ``private_build`` the
architecture's dense runner constructs its IR through a builder that alters one
address or one arithmetic operation, and keeps what it compiles in a launcher
cache of its own. The shipped kernel is neither replaced nor reused, the shipped
compile and launch path is, and the defect is in the lowered IR, which must
differ from the shipped kernel's.

The row-bound control is host-side instead. On the runtime-shape path the query
length is a kernel argument, so one query row past a whole tile runs the shipped
binary itself; what must be bypassed is the host's refusal of that shape.
"""

from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from ._analytical import UniformCase, encode_bits
from ._codeword_analytical import (
    CODEWORD_QUICK_CASES,
    GuardedOutput,
    codeword_inputs,
    guard_layout,
    require_codeword_scope,
)
from ._codeword_policy import CODEWORD_KILL_MATRIX, CodewordMutation, in_scope
from ._codeword_subject import (
    CODEWORD_ARCHES,
    _hip_backend,
    _launch_guarded,
    codeword_schedule,
    resolve_codeword_subject,
    run_codeword_case,
)
from ._dense_subject import DENSE_SUBJECTS, ResolvedDenseSubject

# One control per seam of the kill matrix. Every control but the last is emitted.
CODEWORD_SUBJECT_CONTROLS = (
    CodewordMutation.KV_HEAD_ROTATE,
    CodewordMutation.OUT_HEAD_SWAP,
    CodewordMutation.BATCH_SWAP,
    CodewordMutation.D_HALF_SWAP,
    CodewordMutation.ADJACENT_ROW_SWAP,
    CodewordMutation.V_TOKEN_PLUS_ONE,
    CodewordMutation.DENOMINATOR_PLUS_ONE,
    CodewordMutation.DROPPED_STORE,
    CodewordMutation.BASE_UNDERRUN,
    CodewordMutation.ROW_BOUND_BYPASS,
)
CODEWORD_EMITTED_CONTROLS = CODEWORD_SUBJECT_CONTROLS[:-1]

# Each lane half of the epilogue owns four of every eight output dimensions.
_LANE_HALF = 4
# Elements the underrun control moves the output base back by.
UNDERRUN_ELEMENTS = 4

# An emitted control runs on every batch-2 case that can separate its mutation:
# batch 2 is where batch placement shows, and it covers both head configurations.
CODEWORD_EMITTED_CONTROL_CASES: tuple[
    tuple[str, UniformCase, CodewordMutation], ...
] = tuple(
    (arch, case, mutation)
    for arch in CODEWORD_ARCHES
    for case in CODEWORD_QUICK_CASES
    if case.batch == 2
    for mutation in CODEWORD_EMITTED_CONTROLS
    if in_scope(case, CODEWORD_KILL_MATRIX[mutation].scope)
)
# Swapping two query heads of one KV group stores the same values, so the same
# emitted defect must pass on a grouped case. It is the probe's stated blind spot.
CODEWORD_BLIND_CONTROL_CASES: tuple[tuple[str, UniformCase, CodewordMutation], ...] = (
    tuple(
        (arch, case, CodewordMutation.OUT_HEAD_SWAP)
        for arch in CODEWORD_ARCHES
        for case in CODEWORD_QUICK_CASES
        if case.batch == 2 and case.nhead_q > case.nhead_k
    )
)
# The row-bound control needs one batch: a second batch would start inside the
# rows the extra query tile stores.
CODEWORD_ROW_BOUND_CASES: tuple[tuple[str, UniformCase], ...] = tuple(
    (arch, case)
    for arch in CODEWORD_ARCHES
    for case in CODEWORD_QUICK_CASES
    if case.batch == 1 and case.nhead_q > case.nhead_k
)


def _single_bit(name: str, value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(
            f"emitted control failed - {name}={value}, expected a power of two so "
            "that flipping its bit stays inside the tensor"
        )
    return value


def _mutant_builder(base: type, mutation: CodewordMutation | None, case: UniformCase):
    """A builder that emits the shipped kernel with one defect; none for ``None``.

    The output index is ``((batch * Sq + row) * Hq + head) * D + dim`` in elements
    and the V load offset ``((batch * Sk + key) * Hkv + head) * D + dim`` in
    2-byte elements. With power-of-two extents each coordinate is a bit field, so
    flipping one bit swaps neighbours of one coordinate and every access stays
    inside its tensor. The two controls that leave it are bounded: a V load past
    the last key is clamped to zero by the buffer resource, and the underrun
    lands in the guard."""
    if mutation is not None and mutation not in CODEWORD_EMITTED_CONTROLS:
        raise ValueError(
            f"emitted control failed - mutation={mutation.value!r}, expected one "
            f"of {[control.value for control in CODEWORD_EMITTED_CONTROLS]}"
        )
    head = _single_bit("head_dim", case.head_dim)
    row = head * _single_bit("nhead_q", case.nhead_q)
    batch = row * _single_bit("seqlen_q", case.seqlen_q)
    output_bit = {
        CodewordMutation.OUT_HEAD_SWAP: head,
        CodewordMutation.ADJACENT_ROW_SWAP: row,
        CodewordMutation.BATCH_SWAP: batch,
        CodewordMutation.D_HALF_SWAP: _LANE_HALF,
    }.get(mutation)
    key_bytes = 2 * head * _single_bit("nhead_k", case.nhead_k)

    class MutantBuilder(base):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._control_params: dict = {}
            self._control_v_rsrc = None
            self._control_dropped = False

        def param(self, name, *args, **kwargs):
            value = super().param(name, *args, **kwargs)
            self._control_params[name] = value
            return value

        def buffer_rsrc(self, ptr, *args, **kwargs):
            rsrc = super().buffer_rsrc(ptr, *args, **kwargs)
            if ptr is self._control_params.get("v_ptr"):
                self._control_v_rsrc = rsrc
            return rsrc

        def async_buffer_load_lds_addr(self, rsrc, lds_addr, voffset, *args, **kwargs):
            if rsrc is self._control_v_rsrc:
                if mutation is CodewordMutation.KV_HEAD_ROTATE:
                    voffset = self.xor(voffset, self.const_i32(2 * head))
                elif mutation is CodewordMutation.V_TOKEN_PLUS_ONE:
                    voffset = self.add(voffset, self.const_i32(key_bytes))
            return super().async_buffer_load_lds_addr(
                rsrc, lds_addr, voffset, *args, **kwargs
            )

        def global_store_vN(self, ptr, idx, *args, **kwargs):
            if ptr is self._control_params.get("o_ptr"):
                if (
                    mutation is CodewordMutation.DROPPED_STORE
                    and not self._control_dropped
                ):
                    self._control_dropped = True
                    return None
                if output_bit is not None:
                    idx = self.xor(idx, self.const_i32(output_bit))
                elif mutation is CodewordMutation.BASE_UNDERRUN:
                    idx = self.sub(idx, self.const_i32(UNDERRUN_ELEMENTS))
            return super().global_store_vN(ptr, idx, *args, **kwargs)

        def rcp(self, a):
            if mutation is CodewordMutation.DENOMINATOR_PLUS_ONE:
                a = self.fadd(a, self.const_f32(1.0))
            return super().rcp(a)

    return MutantBuilder


def _runner(arch: str):
    return importlib.import_module(DENSE_SUBJECTS[arch].runner_module)


@contextmanager
def private_build(
    arch: str, case: UniformCase, mutation: CodewordMutation | None
) -> Iterator[None]:
    """Make the architecture's dense runner build and launch one emitted control.

    Inside the block the runner emits through the control's builder and caches
    its launchers privately; both are restored on exit, so the shipped kernel's
    launcher is neither served to the control nor replaced by it. ``None`` builds
    the shipped kernel through the same mechanism."""
    runner = _runner(arch)
    shipped = runner.IRBuilder, runner._DENSE_LAUNCHER_CACHE
    runner.IRBuilder = _mutant_builder(runner.IRBuilder, mutation, case)
    runner._DENSE_LAUNCHER_CACHE = {}
    try:
        yield
    finally:
        runner.IRBuilder, runner._DENSE_LAUNCHER_CACHE = shipped


def lowered_ir(
    arch: str,
    case: UniformCase,
    mutation: CodewordMutation | None = None,
    *,
    private: bool = True,
) -> str:
    """The LLVM IR a launch of ``case`` hands to the compiler.

    ``mutation`` selects an emitted control. Without one this is the shipped
    kernel, built directly or, with ``private``, through the control mechanism;
    the two must agree for a control's IR difference to mean its defect. The
    lowering is the dense launcher's own, without running the compiler."""
    from rocke.helpers.compile import lower_kernel_for_comgr

    spec = resolve_codeword_subject(arch, case).tuning_spec.kernel_spec

    def lower() -> str:
        kernel = _runner(arch).build_attention_dense(spec, arch=arch)
        return lower_kernel_for_comgr(kernel, arch=arch, backend="python").llvm_text

    if mutation is None and not private:
        return lower()
    with private_build(arch, case, mutation):
        return lower()


def run_codeword_control(
    arch: str,
    case: UniformCase,
    mutation: CodewordMutation,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    subject: ResolvedDenseSubject | None = None,
) -> GuardedOutput:
    """Launch one emitted control on raw input bits into a guarded allocation."""
    with private_build(arch, case, mutation):
        return run_codeword_case(arch, case, q, k, v, subject=subject)


def row_bound_bypass_spec(arch: str, case: UniformCase):
    """The shipped kernel spec of ``case`` with one query row past a whole tile.

    The spec type refuses that shape on construction unless it is ragged, and a
    ragged spec leaves the runtime-shape path and builds another kernel. So the
    query length is overwritten on a copy of the constructed shipped spec: every
    other field, and with them the compiled kernel, stays the shipped one."""
    spec = resolve_codeword_subject(arch, case).tuning_spec.kernel_spec
    if spec.persistent or not spec.runtime_shape:
        raise AssertionError(
            "the row-bound control requires the non-persistent runtime-shape "
            f"grid, got persistent={spec.persistent!r}, "
            f"runtime_shape={spec.runtime_shape!r}"
        )
    mutant = copy.copy(spec)
    object.__setattr__(mutant, "seqlen_q", int(spec.block_m) + 1)
    return mutant


def run_row_bound_bypass(
    arch: str, case: UniformCase
) -> tuple[UniformCase, GuardedOutput]:
    """Launch the shipped kernel on one query row past a whole query tile.

    Returns the case the logical output belongs to and the guarded allocation.
    The kernel stores without a row bound, so the second query tile stores its
    one logical row and then ``block_m - 1`` rows after the tensor, into the
    guard. Q is padded to whole tiles, so the unbounded Q loads stay in bounds.
    Only the host's row-bound refusal is bypassed: the patched predicate answers
    what the shipped one answers for the shipped shape."""
    if case.batch != 1:
        raise ValueError(
            f"run_row_bound_bypass failed - batch={case.batch}, expected 1 batch"
        )
    runner = _runner(arch)
    subject = resolve_codeword_subject(arch, case)
    shipped_spec = subject.tuning_spec.kernel_spec
    spec = row_bound_bypass_spec(arch, case)
    ragged = replace(case, seqlen_q=int(spec.seqlen_q))
    require_codeword_scope(ragged)
    layout = guard_layout(ragged, codeword_schedule(subject))

    _, k_values, v_values = codeword_inputs(case)
    tiles = -(-ragged.seqlen_q // int(spec.block_m))
    padded_q = (case.batch, tiles * int(spec.block_m), case.nhead_q, case.head_dim)
    host = {
        "q": np.zeros(padded_q, dtype=np.uint16),
        "k": encode_bits(k_values, case.dtype),
        "v": encode_bits(v_values, case.dtype),
        "out": layout.new_buffer(),
    }
    shipped_predicate = runner.supports_attention_dense

    def launch(*, q, k, v, out) -> None:
        runner.supports_attention_dense = lambda _spec, *, arch: shipped_predicate(
            shipped_spec, arch=arch
        )
        try:
            runner.run_attention_dense_torch(
                spec=spec, q=q, k=k, v=v, out=out, scale=1.0 / math.sqrt(case.head_dim)
            )
        finally:
            runner.supports_attention_dense = shipped_predicate

    raw = _launch_guarded(launch, _hip_backend(), host, layout)
    return ragged, GuardedOutput(layout, raw)
