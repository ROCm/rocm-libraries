# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU key-permutation relation for the shipped dense attention kernels.

Unmasked attention must not change when the key tokens of K and V are permuted
together. Each case launches the subject on its base inputs, on every joint
permutation, and on three pairing mutations, all on raw FP16/BF16 bits through
the architecture's public dense runner. Two outputs are compared against a bound
derived from the kernel's arithmetic path; no expected output is computed.

The only skip is a processor that is not the case's architecture, through the
shared ``rocke.tests.gpu.current_device_arch_in`` gate, so
``ROCKE_REQUIRE_<ARCH>=1`` turns that skip into a failure. On the architecture,
a dispatch outcome outside the bound's scope fails the test; tuning changes
inside the scope keep running.

Nothing is launched before the scope holds for the kernel under test: its
artifact is compiled, its lowered path is checked against the one the bound is
derived for, the key-tile count of the bound is taken from that path, and the
input statistics of the bound are certified from above.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pytest
from rocke.tests.gpu import current_device_arch_in

from ._analytical import encode_bits
from ._dense_subject import (
    PERMUTATION_SUBJECT_CASES,
    ResolvedDenseSubject,
    compile_dense_subject,
    resolve_dense_subject,
    run_dense_case,
    softmax_scale,
)
from ._permutation import (
    PERMUTATION_MUTATION_MAP,
    PERMUTATION_QUICK_MAPS,
    PairingMutation,
    PermutationCase,
    PermutationResult,
    analyze_permutation,
    format_permutation_result,
    format_slice_diagnostics,
    mutated_token_maps,
    permutation_case_bound,
    permutation_inputs,
    permute_tokens,
    token_map,
)
from ._permutation_path import REDERIVE, fp_mode_violations, verified_key_tiles

pytestmark = pytest.mark.gpu


def _case_label(arch: str, case: PermutationCase) -> str:
    return (
        f"{arch} key permutation {case.dtype} unmasked B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _format_results(results: Mapping[str, PermutationResult]) -> str:
    return "; ".join(
        f"{name}: {format_permutation_result(result)}"
        for name, result in results.items()
    )


def _unresponsive(
    label: str,
    mutation: str,
    mutated: PermutationResult,
    positives: Mapping[str, PermutationResult],
) -> str:
    """Explain a pairing mutation some head's output did not respond to.

    A kernel that pairs scores with the wrong values already fails the joint
    permutations, so their violations are the diagnosis. With the joint
    permutations clean, the heads listed ignore which value belongs to which
    key."""
    heads = ", ".join(f"b{batch}:h{head}" for batch, head in mutated.unresponsive_heads)
    failed = {name: result for name, result in positives.items() if result.violations}
    if failed:
        return (
            f"{label}: {mutation} stayed within the bound on {heads} and the "
            f"joint permutations themselves fail it; {_format_results(failed)}"
        )
    return (
        f"{label}: the output of {heads} did not respond to {mutation}, so the "
        f"kernel ignores the key-value pairing there; "
        f"{format_permutation_result(mutated)}"
    )


def _artifact_key_tiles(
    label: str, arch: str, case: PermutationCase, subject: ResolvedDenseSubject
) -> int:
    """Check the artifact this case launches and return the number of key tiles
    one query row visits in it.

    The bound holds for one arithmetic path, one tile count and one
    floating-point mode. They are read from the kernel under test, not from the
    requested shape or from an earlier build: its LLVM IR must be the modeled
    path and its kernel descriptor must select round-to-nearest-even with
    subnormals kept. Anything else fails here, before any launch. The digest
    names the code object the outputs below come from."""
    from rocke.runtime.comgr import loaded_compiler_info

    artifact = compile_dense_subject(arch, subject)
    spec = subject.tuning_spec.kernel_spec
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
    print(
        f"{label} artifact: {artifact.kernel_name} "
        f"sha256={hashlib.sha256(artifact.hsaco).hexdigest()}; "
        f"compiler {'unknown' if compiler is None else compiler.describe()}; "
        f"{key_tiles} key tiles per query row"
    )
    return key_tiles


def _subject_case_id(item: tuple[str, PermutationCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


@pytest.mark.parametrize(
    "subject_case", PERMUTATION_SUBJECT_CASES, ids=_subject_case_id
)
def test_joint_key_value_permutation(
    subject_case: tuple[str, PermutationCase],
) -> None:
    """Gate one unmasked dense case on every joint permutation and require every
    pairing mutation to be visible in every head."""
    arch, case = subject_case
    current_device_arch_in((arch,), subject=f"the {arch} dense key-permutation gate")
    label = _case_label(arch, case)
    subject = resolve_dense_subject(arch, case)
    key_tiles = _artifact_key_tiles(label, arch, case, subject)
    q_values, k_values, v_values = permutation_inputs(case)
    bound = permutation_case_bound(
        case,
        q_values,
        k_values,
        v_values,
        key_tiles=key_tiles,
        scale=softmax_scale(case),
    )
    q = encode_bits(q_values, case.dtype)
    k = encode_bits(k_values, case.dtype)
    v = encode_bits(v_values, case.dtype)

    def launch(key_map: np.ndarray, value_map: np.ndarray) -> np.ndarray:
        return run_dense_case(
            arch,
            case,
            q,
            permute_tokens(k, key_map),
            permute_tokens(v, value_map),
            subject=subject,
        )

    base = run_dense_case(arch, case, q, k, v, subject=subject)
    positives: dict[str, PermutationResult] = {}
    for kind in PERMUTATION_QUICK_MAPS:
        source = token_map(kind, case.seqlen_k)
        result = analyze_permutation(case, bound, base, launch(source, source))
        positives[kind.value] = result
        print(f"{label} {kind.value}: {format_permutation_result(result)}")
        print(format_slice_diagnostics(result))

    for mutation in PairingMutation:
        key_map, value_map = mutated_token_maps(
            mutation, PERMUTATION_MUTATION_MAP, case.seqlen_k
        )
        mutated = analyze_permutation(case, bound, base, launch(key_map, value_map))
        fewest = min(min(counts) for counts in mutated.head_exceed_counts)
        print(
            f"{label} {mutation.value}: max_abs={mutated.max_abs:.9g}; "
            f"{mutated.exceed_count} beyond the bound, fewest in one head {fewest}"
        )
        assert not mutated.unresponsive_heads, _unresponsive(
            label, mutation.value, mutated, positives
        )

    failed = {name: result for name, result in positives.items() if result.violations}
    assert not failed, f"{label}: {_format_results(failed)}"
