# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU structured-logit checks for the shipped gfx942 dense kernel.

Each quick case launches its fixture under the four basis exposures and the
folded exposure, twice, through a raw-bit adapter. Every stored coordinate must
lie in the Tier B set derived on the CPU from the pinned arithmetic path; no
measured value enters that set.

The only skip is a processor that is not gfx942, through the shared
``rocke.tests.gpu.current_device_arch_in`` gate, so ``ROCKE_REQUIRE_GFX942=1``
turns it into a failure. On gfx942 a missing or mismatched ``STRUCTURED_CALIBRATION``
entry fails after the records are written, so an unratified case fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from rocke.tests.gpu import current_device_arch_in

from ._structured_analytical import (
    STRUCTURED_EXPOSURES,
    STRUCTURED_QUICK_CASES,
    STRUCTURED_SEPARATION_ROWS,
    STRUCTURED_UNSEPARATED,
    StructuredCase,
    StructuredFixture,
    StructuredMutation,
    StructuredResult,
    analyze_structured_policy,
    applicable_mutations,
    epilogue_artifact_for,
    format_structured_result,
    mutation_separation,
    structured_diagnostics,
    structured_inputs,
    structured_records,
    require_structured_calibration,
)
from ._structured_shapes import Observation, encode_bits
from ._structured_subject import (
    ResolvedDenseSubject,
    resolve_dense_subject,
    run_dense_case,
)

pytestmark = pytest.mark.gpu

# The subject-side mutation: the kernel launched with scale 1 instead of
# 1/sqrt(D), the twin of the analyzer's ``omit_sqrt_d``.
_WRONG_SCALE = 1.0
_WRONG_SCALE_EXPOSURE = "basis@0"


def _run_exposures(
    case: StructuredCase,
    subject: ResolvedDenseSubject,
    exposures: tuple[str, ...] = STRUCTURED_EXPOSURES,
    *,
    scale_override: float | None = None,
) -> Observation:
    outputs = []
    for exposure in exposures:
        q, k, v = structured_inputs(case, exposure)
        outputs.append(
            (
                exposure,
                run_dense_case(
                    case,
                    encode_bits(q, case.dtype),
                    encode_bits(k, case.dtype),
                    encode_bits(v, case.dtype),
                    subject=subject,
                    scale_override=scale_override,
                ),
            )
        )
    return Observation(tuple(outputs))


def _same_bits(first: Observation, second: Observation) -> bool:
    return all(
        name == other and np.array_equal(left, right)
        for (name, left), (other, right) in zip(first.exposures, second.exposures)
    )


def _not_rejected(
    case: StructuredCase,
    what: str,
    mutated: StructuredResult,
    control: StructuredResult,
) -> str:
    """Explain a mutation the checker did not reject, blaming the right side: a
    kernel whose output already fails Tier B can match a mutated prediction."""
    if control.violations:
        return (
            f"{case.case_id}: {what} was not rejected and the kernel output itself "
            f"fails Tier B; control: {format_structured_result(control)}"
        )
    return (
        f"{case.case_id}: checker failed to reject {what}; "
        f"{format_structured_result(mutated)}"
    )


_CODE_OBJECTS: dict[str, dict[str, str]] = {}


def _code_object(subject: ResolvedDenseSubject) -> dict[str, str]:
    """The loaded compiler and the SHA-256 of the HSACO the launch ran.

    ``launch_attention_dense`` caches its compiled kernel without exposing it,
    so the same compile is repeated here; the digest names the epilogue
    artifact (A-F) the records compare against. Evidence only."""
    from kernels.gfx942.attention_dense import build_attention_dense
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.comgr import loaded_compiler_info

    spec = subject.tuning_spec.kernel_spec
    key = spec.kernel_name()
    if key not in _CODE_OBJECTS:
        artifact = compile_kernel(
            build_attention_dense(spec, arch="gfx942"),
            arch="gfx942",
            backend="python",
            capture_ir_text=False,
        )
        compiler = loaded_compiler_info()
        _CODE_OBJECTS[key] = {
            "kernel_name": artifact.kernel_name,
            "compiler": "unknown" if compiler is None else compiler.describe(),
            "hsaco_sha256": hashlib.sha256(artifact.hsaco).hexdigest(),
        }
    return _CODE_OBJECTS[key]


def _capture_records(
    case: StructuredCase, records: list[dict], observation: Observation
) -> None:
    """Write the records and the raw output bits, so ratification can audit
    every stored coordinate offline rather than trust the aggregates."""
    output_text = os.environ.get("ROCKE_ORACLE_FREE_CALIBRATION_OUT")
    if not output_text:
        return
    output_dir = Path(output_text).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{case.case_id}.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / f"{case.case_id}.npz", **dict(observation.exposures)
    )


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=lambda case: case.case_id)
def test_structured_logits(case: StructuredCase) -> None:
    """Gate a ratified case: deterministic, and every coordinate in Tier B.

    Mutation self-checks and record capture run before the gate, so an
    unratified case still records what ratification needs."""
    arch = current_device_arch_in(
        ("gfx942",), subject="the gfx942 dense structured-logit gate"
    )
    subject = resolve_dense_subject(case)
    observation = _run_exposures(case, subject)
    deterministic = _same_bits(observation, _run_exposures(case, subject))
    control = analyze_structured_policy(case, observation)
    results: dict[str, StructuredResult] = {"none": control}
    separations = {}

    # Analyzer-side mutations on the same real output (sampled rows suffice to
    # reject); cells proven inseparable on the CPU are owned elsewhere.
    for mutation in applicable_mutations(case):
        if (case.case_id, mutation) in STRUCTURED_UNSEPARATED:
            continue
        mutated = analyze_structured_policy(
            case, observation, mutation=mutation, rows=STRUCTURED_SEPARATION_ROWS
        )
        assert mutated.violations, _not_rejected(case, mutation.value, mutated, control)
        results[mutation.value] = mutated
        separations[mutation.value] = mutation_separation(case, mutation)

    # Subject-side mutation: the kernel itself launched with the wrong scale.
    if case.fixture is StructuredFixture.IDENTITY:
        wrong = _run_exposures(
            case, subject, (_WRONG_SCALE_EXPOSURE,), scale_override=_WRONG_SCALE
        )
        mutated = analyze_structured_policy(case, wrong, require_complete=False)
        what = f"the kernel launched with scale={_WRONG_SCALE}"
        assert mutated.violations, _not_rejected(case, what, mutated, control)
        results["subject_scale_override"] = mutated
        separations["subject_scale_override"] = mutation_separation(
            case, StructuredMutation.OMIT_SQRT_D
        )

    # The IR-exact and the launched artifact's epilogue agreement are recorded
    # separately (A-F); neither enters the verdict.
    code_object = _code_object(subject)
    artifact = epilogue_artifact_for(case.dtype, code_object["hsaco_sha256"])
    _capture_records(
        case,
        structured_records(
            case,
            subject.candidate_name,
            results,
            deterministic=deterministic,
            diagnostics=structured_diagnostics(case, observation, artifact=artifact),
            separations=separations,
            code_object=code_object,
        ),
        observation,
    )

    summary = f"deterministic={deterministic}; {format_structured_result(control)}"
    require_structured_calibration(
        arch, subject.candidate_name, case, subject.fingerprint, detail=summary
    )
    assert deterministic, f"{case.case_id}: repeated launches differ; {summary}"
    assert control.compared > 0, f"{case.case_id}: nothing compared; {summary}"
    assert not control.violations, f"{case.case_id}: {summary}"
