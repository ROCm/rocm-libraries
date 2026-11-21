# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launches and verdicts of one masked-token poisoning case, shared by the
quick-tier and the standard-tier GPU modules.

Nothing is launched before the scope holds for the kernel under test: its
artifact is compiled, its lowered path is checked against the one the bit
equality is derived for, the keys it leaves unmasked are compared with the
case's legal keys, and the encoded inputs of every launch pair are checked on
the bits the kernel reads.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ._analytical import encode_bits
from ._dense_subject import (
    ResolvedDenseSubject,
    compile_dense_subject,
    launch_geometry,
    resolve_dense_subject,
    run_dense_case,
)
from ._masked_poison import (
    POISON_ANCHORS,
    BoundaryControl,
    MaskedPoisonCase,
    PoisonFixture,
    PoisonResult,
    analyze_masked_poison,
    control_positions,
    control_witness,
    format_masked_poison_result,
    masked_poison_inputs,
    masked_poison_limits,
    poison_fixture,
    poison_mechanisms,
    poisoned_bit_violations,
    poisoned_tensors,
    sees_token,
)
from ._masked_poison_path import path_scope, verified_masked_path, visited_tiles
from ._permutation_path import REDERIVE, fp_mode_violations

_VERDICTS = ("validity", "sanity", "relation", "activity")


def _case_label(arch: str, case: MaskedPoisonCase) -> str:
    mask = f"window {case.window}" if case.mask == "window" else case.mask
    return (
        f"{arch} masked-token poisoning {case.dtype} {mask} B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _checked_artifact(
    label: str, arch: str, case: MaskedPoisonCase, subject: ResolvedDenseSubject
) -> tuple[np.ndarray, np.ndarray, int]:
    """Check the artifact this case launches and return, per query row, the
    first and one past the last key tile it visits, and its key-tile size.

    The bit equality holds for one masked path and one floating-point mode. They
    are read from the kernel under test, not from the requested shape or from an
    earlier build: its LLVM IR must be the modeled path, the keys its position
    arithmetic leaves unmasked must be the case's legal keys for every thread of
    the launch, and its kernel descriptor must select round-to-nearest-even with
    subnormals kept. Anything else fails here, before any launch. The digest
    names the code object the outputs come from."""
    from rocke.runtime.comgr import loaded_compiler_info

    artifact = compile_dense_subject(arch, subject)
    spec = subject.tuning_spec.kernel_spec
    grid, threads = launch_geometry(arch, subject)
    scope = path_scope(
        case,
        block_n=int(spec.block_n),
        head_size=int(spec.head_size),
        grid=grid,
        threads=threads,
    )
    path = verified_masked_path(artifact.llvm_text, case.dtype, **scope)
    mode = fp_mode_violations(artifact.hsaco, artifact.kernel_name)
    assert not mode, f"{label}: {REDERIVE}: " + "; ".join(mode)
    tile_low, tile_high = visited_tiles(
        artifact.llvm_text,
        path,
        **{name: scope[name] for name in scope if name not in ("block_n", "head_size")},
    )
    compiler = loaded_compiler_info()
    loops = ", ".join(f"{loop.counter} mask={loop.mask}" for loop in path.loops)
    print(
        f"{label} artifact: {artifact.kernel_name} "
        f"sha256={hashlib.sha256(artifact.hsaco).hexdigest()}; "
        f"compiler {'unknown' if compiler is None else compiler.describe()}; "
        f"grid {grid} x {threads} threads; key loops: {loops}"
    )
    return tile_low, tile_high, int(spec.block_n)


def _control_failure(
    label: str,
    fixture: PoisonFixture,
    control: BoundaryControl,
    result: PoisonResult,
) -> str | None:
    """Explain a boundary control the relation did not respond to as it must.

    The control moves one poisoned token a single position into the legal set
    of one target row. That row, in the head that copies the anchor key, must
    change and be taken over by the poison, for every batch element and KV
    head; no unaffected row that still cannot see a poisoned token may
    change."""
    case = fixture.case
    positions = control_positions(case, fixture.anchor, control)
    row, offset = control_witness(case, fixture.anchor, control)
    heads = [head for head in range(case.nhead_q) if head % case.group == offset]
    silent = [
        f"b{batch}:h{head}"
        for batch in range(case.batch)
        for head in heads
        if not (
            result.mismatched[batch, row, head] and result.dominated[batch, row, head]
        )
    ]
    exposed = fixture.unaffected & sees_token(case, positions)
    stray = [query for query in result.mismatched_rows if not exposed[query]]
    if not silent and not stray:
        return None
    return (
        f"{label} anchor {fixture.anchor} control {control.value} tokens "
        f"{positions}: row {row} did not respond in {silent or 'no head'}; rows "
        f"that cannot see the tokens changed: {stray or 'none'}; "
        f"{format_masked_poison_result(result)}"
    )


def check_masked_poison_case(arch: str, case: MaskedPoisonCase) -> None:
    """Launch one case and apply its four verdicts and its controls.

    Per anchor: one poisoned launch against the shared baseline, judged on raw
    bits; the same analysis with the baseline in the poisoned output's place,
    which the activity verdict must reject; and one launch per boundary
    control. The baseline is launched twice and must repeat."""
    label = _case_label(arch, case)
    subject = resolve_dense_subject(arch, case)
    tile_low, tile_high, block_n = _checked_artifact(label, arch, case, subject)
    limits = masked_poison_limits(case)
    baseline = masked_poison_inputs(case)
    q = encode_bits(baseline.q, case.dtype)
    k = encode_bits(baseline.k, case.dtype)
    v = encode_bits(baseline.v, case.dtype)
    print(
        f"{label}: key seed {baseline.key_seed:#x}; baseline ceiling "
        f"{limits.baseline_ceiling:.9g}, activity threshold {limits.threshold:.9g}, "
        f"takeover floor {limits.activity_floor:.9g}, target gap "
        f"{limits.target_gap:.6g} score units"
    )

    def launch(k_values: np.ndarray, v_values: np.ndarray) -> np.ndarray:
        return run_dense_case(
            arch,
            case,
            q,
            encode_bits(k_values, case.dtype),
            encode_bits(v_values, case.dtype),
            subject=subject,
        )

    base = run_dense_case(arch, case, q, k, v, subject=subject)
    replay = run_dense_case(arch, case, q, k, v, subject=subject)

    results: dict[int, PoisonResult] = {}
    control_failures: list[str] = []
    for anchor in POISON_ANCHORS:
        fixture = poison_fixture(case, baseline, anchor)
        invalid = poisoned_bit_violations(
            fixture,
            k,
            v,
            encode_bits(fixture.k, case.dtype),
            encode_bits(fixture.v, case.dtype),
        )
        assert not invalid, f"{label} anchor {anchor}: invalid inputs: {invalid}"
        mechanisms = poison_mechanisms(
            fixture, block_n=block_n, tile_low=tile_low, tile_high=tile_high
        )
        print(
            f"{label} anchor {anchor}: poisoned tokens {fixture.positions}; "
            + ", ".join(
                f"row {row} token {position}: {mechanism.value}"
                for (row, position), mechanism in mechanisms.items()
            )
        )

        result = analyze_masked_poison(
            fixture, limits, base, replay, launch(fixture.k, fixture.v)
        )
        results[anchor] = result
        print(f"{label} anchor {anchor}: {format_masked_poison_result(result)}")

        undelivered = analyze_masked_poison(fixture, limits, base, replay, base)
        assert undelivered.activity, (
            f"{label} anchor {anchor}: the activity verdict accepted an output "
            "that holds no poison"
        )

        for control in BoundaryControl:
            if control is BoundaryControl.WINDOW_KEY and case.mask != "window":
                continue
            positions = control_positions(case, anchor, control)
            moved = analyze_masked_poison(
                fixture,
                limits,
                base,
                replay,
                launch(*poisoned_tensors(case, baseline, anchor, positions)),
            )
            print(
                f"{label} anchor {anchor} control {control.value}: "
                f"{moved.mismatch_count} unaffected coordinate(s) changed in rows "
                f"{list(moved.mismatched_rows)}"
            )
            failure = _control_failure(label, fixture, control, moved)
            if failure:
                control_failures.append(failure)

    for verdict in _VERDICTS:
        failed = {
            anchor: format_masked_poison_result(result)
            for anchor, result in results.items()
            if getattr(result, verdict)
        }
        assert not failed, f"{label}: {verdict} failure: {failed}"
    assert not control_failures, "; ".join(control_failures)
