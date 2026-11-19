# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU value-shift relation for the shipped dense attention kernels.

Attention weights sum to one over the keys of a row, so adding one vector ``c``
to every value token must add ``c`` to every output row. Each case launches the
subject on its base inputs, on the same inputs again, on the shifted values and
on two wrongly shifted ones, all on raw FP16/BF16 bits through the
architecture's public dense runner. The residual ``O(V + c) - O(V) - c`` is
judged against a tolerance derived from the kernel's arithmetic path; no
expected output is computed.

The only skip is a processor that is not the case's architecture, through the
shared ``rocke.tests.gpu.current_device_arch_in`` gate, so
``ROCKE_REQUIRE_<ARCH>=1`` turns that skip into a failure. On the architecture,
a dispatch outcome outside the tolerance's scope fails the test; tuning changes
inside the scope keep running.

Nothing is launched before the scope holds for the kernel under test: its
artifact is compiled, its lowered path is checked against the one the tolerance
is derived for, including that nothing computed from V reaches a weight or the
normalizer, the key-tile count is taken from that path, and the launched values
are checked to be the exact shift of the base ones.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from rocke.tests.gpu import current_device_arch_in

from ._analytical import encode_bits
from ._dense_subject import (
    VALUE_SHIFT_SUBJECT_CASES,
    ResolvedDenseSubject,
    compile_dense_subject,
    resolve_dense_subject,
    run_dense_case,
)
from ._permutation_path import fp_mode_violations
from ._value_shift import (
    OUTPUT_SCALE_SWEEP,
    ShiftMutation,
    ValueShiftCase,
    ValueShiftResult,
    analyze_value_shift,
    applied_shift,
    format_head_diagnostics,
    format_value_shift_result,
    output_scale_threshold,
    require_finite_scores,
    scale_output_bits,
    shift_mutation_failures,
    shift_vector,
    shifted_value_bits,
    value_shift_contract,
    value_shift_inputs,
)
from ._value_shift_path import REDERIVE, verified_key_tiles

pytestmark = pytest.mark.gpu

# Artifact checks already passed in this process, by architecture, kernel and key
# count: the attention profiles of one dtype launch the same code object.
_CHECKED_ARTIFACTS: dict[tuple[str, str, int], tuple[int, str]] = {}


def _case_label(arch: str, case: ValueShiftCase) -> str:
    return (
        f"{arch} value shift {case.dtype} {case.profile} unmasked B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _not_rejected(
    label: str,
    what: str,
    mutated: ValueShiftResult,
    positive: ValueShiftResult,
    failures: tuple[str, ...] = (),
) -> str:
    """Explain a mutation the relation did not reject, blaming the right side.

    The mutations are certain to be rejected only for a kernel whose correct
    shift passes. When that one fails, its violations are the diagnosis.
    ``failures`` are the reasons a wrong-shift launch does not count as
    rejected."""
    if positive.violations:
        return (
            f"{label}: {what} was not rejected and the correct shift itself "
            f"fails; {format_value_shift_result(positive)}"
        )
    if failures:
        return (
            f"{label}: {what} was not rejected as a wrong shift: "
            f"{'; '.join(failures)}; {format_value_shift_result(mutated)}"
        )
    return (
        f"{label}: {what} stayed within the tolerance; "
        f"{format_value_shift_result(mutated)}"
    )


def _artifact_key_tiles(
    label: str, arch: str, case: ValueShiftCase, subject: ResolvedDenseSubject
) -> int:
    """Check the artifact this case launches and return the number of key tiles
    one query row visits in it.

    The tolerance holds for one arithmetic path, one tile count and one
    floating-point mode, and its two-launch argument needs V to stay out of
    every weight. All of that is read from the kernel under test, not from the
    requested shape or from an earlier build: its LLVM IR must be the modeled
    path and its kernel descriptor must select round-to-nearest-even with
    subnormals kept. Anything else fails here, before any launch. The digest
    names the code object the outputs below come from."""
    from rocke.runtime.comgr import loaded_compiler_info

    spec = subject.tuning_spec.kernel_spec
    checked = (arch, str(spec.kernel_name()), case.seqlen_k)
    if checked not in _CHECKED_ARTIFACTS:
        artifact = compile_dense_subject(arch, subject)
        key_tiles = verified_key_tiles(
            artifact.llvm_text,
            case.dtype,
            seqlen_k=case.seqlen_k,
            block_n=int(spec.block_n),
            head_size=int(spec.head_size),
        )
        mode = fp_mode_violations(artifact.hsaco, artifact.kernel_name)
        assert not mode, f"{label}: {REDERIVE}: " + "; ".join(mode)
        compiler = loaded_compiler_info()
        _CHECKED_ARTIFACTS[checked] = (
            key_tiles,
            f"{artifact.kernel_name} "
            f"sha256={hashlib.sha256(artifact.hsaco).hexdigest()}; "
            f"compiler {'unknown' if compiler is None else compiler.describe()}",
        )
    key_tiles, description = _CHECKED_ARTIFACTS[checked]
    print(f"{label} artifact: {description}; {key_tiles} key tiles per query row")
    return key_tiles


def _subject_case_id(item: tuple[str, ValueShiftCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


@pytest.mark.parametrize(
    "subject_case", VALUE_SHIFT_SUBJECT_CASES, ids=_subject_case_id
)
def test_constant_value_shift(subject_case: tuple[str, ValueShiftCase]) -> None:
    """Gate one unmasked dense case on the shift of its outputs, and require a
    repeated launch to be identical and every mutation to be rejected."""
    arch, case = subject_case
    current_device_arch_in((arch,), subject=f"the {arch} dense value-shift gate")
    label = _case_label(arch, case)
    subject = resolve_dense_subject(arch, case)
    key_tiles = _artifact_key_tiles(label, arch, case, subject)
    q_values, k_values, v_values = value_shift_inputs(case)
    require_finite_scores(q_values, k_values)
    q = encode_bits(q_values, case.dtype)
    k = encode_bits(k_values, case.dtype)
    v = encode_bits(v_values, case.dtype)
    v_shifted = shifted_value_bits(v, shift_vector(case.head_dim), case.dtype)
    contract = value_shift_contract(case, v, v_shifted, key_tiles=key_tiles)

    def launch(values: np.ndarray) -> np.ndarray:
        return run_dense_case(arch, case, q, k, values, subject=subject)

    base = launch(v)
    replay = launch(v)
    follow_up = launch(v_shifted)
    positive = analyze_value_shift(contract, base, follow_up)
    print(f"{label}: {format_value_shift_result(positive)}")
    print(format_head_diagnostics(positive))

    # The two launches are compared as if they shared every weight, which needs
    # the kernel to be deterministic on identical bytes.
    differing = int(np.count_nonzero(base != replay))
    assert differing == 0, (
        f"{label}: a repeated launch on identical inputs changed {differing} "
        "stored values"
    )

    for mutation in ShiftMutation:
        applied = applied_shift(case.head_dim, mutation)
        v_mutated = shifted_value_bits(v, applied, case.dtype)
        mutated_contract = value_shift_contract(
            case, v, v_mutated, key_tiles=key_tiles, applied=applied
        )
        mutated = analyze_value_shift(mutated_contract, base, launch(v_mutated))
        print(
            f"{label} {mutation.value}: rho_max={mutated.max_ratio:.3g}; "
            f"{mutated.exceed_count} beyond the tolerance"
        )
        # The mutated tensor is a legitimate input: the launch must be sound and
        # its residuals must lie where the wrong shift puts them.
        failures = shift_mutation_failures(mutated_contract, applied, mutated)
        assert not failures, _not_rejected(
            label, mutation.value, mutated, positive, failures
        )

    # Normalization mutation on the real outputs: both scaled by 1 + eta after
    # the kernel's conversion. Above the derived threshold a passing pair is
    # certain to be rejected; below it the outcome is reported.
    threshold = output_scale_threshold(contract)
    floor = None
    unbroken = True
    for eta in OUTPUT_SCALE_SWEEP:
        scaled = analyze_value_shift(
            contract,
            scale_output_bits(base, eta, case.dtype),
            scale_output_bits(follow_up, eta, case.dtype),
        )
        rejected = bool(scaled.violations)
        unbroken = unbroken and rejected
        if unbroken:
            floor = eta
        print(
            f"{label} output scale 1+{eta:.6g}: "
            f"{'rejected' if rejected else 'not rejected'}; "
            f"rho_max={scaled.max_ratio:.3g}; {scaled.exceed_count} residuals "
            f"and {scaled.magnitude_count} magnitudes beyond their limits"
        )
        assert rejected or eta <= threshold, _not_rejected(
            label, f"an output scale of 1+{eta:.6g}", scaled, positive
        )
    print(
        f"{label} output scale: certain rejection above {threshold:.3g}; "
        f"rejected down to {'none' if floor is None else f'{floor:.6g}'}"
    )

    assert not positive.violations, f"{label}: {format_value_shift_result(positive)}"
