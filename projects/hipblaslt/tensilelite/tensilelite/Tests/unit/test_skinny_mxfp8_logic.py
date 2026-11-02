# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural checks of the hipBLASLt skinny MXFP8 logic shard (ExactLogic routing)."""

from pathlib import Path

import pytest
import yaml

from skinny_mxfp8_predicates import emitted_predicates, failing, make_problem

pytestmark = pytest.mark.unit

SHARD = (
    Path(__file__).resolve().parents[4]
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Equality"
    / "gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml"
)

NUM_SOLUTIONS = 4
MAX_SKINNY = 4
NUM_SHAPES = 71
# Keys (both orientations) won by each kernel in the single TensileLite benchmark run.
EXPECTED_KEYS_PER_KERNEL = {0: 30, 1: 63, 2: 14, 3: 35}
EXPECTED_SIZE_CAP = {0: 1048577, 1: 1048577}
# K upper bounds of the kernels with signed 32-bit K arithmetic, keyed by solution index. The persistent MFMA
# kernels (1, 3) need K <= 2^30; the VALU kernel (2) needs K <= 2^31 - kStep with kStep 1024.
EXPECTED_K_BOUNDS = {0: {3: 2147483393}, 1: {3: (1 << 30) + 1}, 2: {3: 2147482625}, 3: {3: (1 << 30) + 1}}
# The MFMA kernels (0, 1, 3) use 0x80000000 as the out-of-range sentinel, so their buffers stay below 2 GiB.
EXPECTED_TENSOR_BYTES = {0: 1 << 31, 1: 1 << 31, 2: 1 << 32, 3: 1 << 31}

# Shapes dropped in both orientations because the generic kernel is faster: (wide, k, skinny extents).
PRUNED_SHAPES = ((128256, 8192, (1, 2, 4)),)
# Shapes the shard keeps although an earlier kernel set excluded them: (wide, skinny, k).
KEPT_SHAPES = ((4096, 4, 1280), (5120, 1, 2048), (5120, 2, 1536), (128256, 4, 4096))


@pytest.fixture(scope="module")
def shard():
    with open(SHARD) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def entries(shard):
    """Map (m, n, batch, k) to the full key and the (solution index, time) pair."""
    result = {}
    for key, value in shard["ExactLogic"]:
        assert tuple(key[:4]) not in result, f"duplicate ExactLogic key {key}"
        result[tuple(key[:4])] = (key, value)
    return result


def _is_skinny_n(m, n):
    return n <= MAX_SKINNY < m


def _is_skinny_m(m, n):
    return m <= MAX_SKINNY < n


def test_logic_type_is_equality(shard):
    assert shard["LibraryType"] == "Equality"
    assert shard["IndexOrder"] == [2, 3, 0, 1]


def test_keys_have_batch_one_and_consistent_leading_dims(entries):
    assert len(entries) == 2 * NUM_SHAPES
    for (m, n, batch, k), (key, _) in entries.items():
        assert batch == 1, key
        assert key[4:] == [m, m, k, k], key


def test_solution_indices_exist(shard, entries):
    defined = {solution["SolutionIndex"] for solution in shard["Solutions"]}
    assert defined == set(range(NUM_SOLUTIONS))
    for key, (_, (solution_index, _)) in entries.items():
        assert solution_index in defined, key


def test_every_entry_has_one_skinny_extent(entries):
    for m, n, _, _ in entries:
        assert _is_skinny_n(m, n) or _is_skinny_m(m, n), (m, n)


def test_entries_cover_both_orientations_equally(entries):
    skinny_n = [key for key in entries if _is_skinny_n(key[0], key[1])]
    skinny_m = [key for key in entries if _is_skinny_m(key[0], key[1])]
    assert len(skinny_n) == len(skinny_m) == NUM_SHAPES
    assert len({(max(m, n), min(m, n), k) for m, n, _, k in entries}) == NUM_SHAPES


def test_excluded_shapes_are_absent_in_both_orientations(entries):
    for wide, k, skinnies in PRUNED_SHAPES:
        for skinny in skinnies:
            assert (wide, skinny, 1, k) not in entries, (wide, skinny, k)
            assert (skinny, wide, 1, k) not in entries, (skinny, wide, k)
    for m, n, _, k in entries:
        wide, skinny = max(m, n), min(m, n)
        assert not any(wide == w and k == kk and skinny in sk for w, kk, sk in PRUNED_SHAPES), (m, n, k)


def test_kept_shapes_are_present_in_both_orientations(entries):
    # The exclusions are explicit: shapes a previous kernel set dropped are served again.
    for wide, skinny, k in KEPT_SHAPES:
        assert (wide, skinny, 1, k) in entries, (wide, skinny, k)
        assert (skinny, wide, 1, k) in entries, (skinny, wide, k)


def test_solutions_cap_the_wide_extent_symmetrically(shard):
    assert len(shard["Solutions"]) == NUM_SOLUTIONS
    for solution in shard["Solutions"]:
        expected = {**EXPECTED_SIZE_CAP, **EXPECTED_K_BOUNDS.get(solution["SolutionIndex"], {})}
        assert solution["AssertSizeLessThan"] == expected, solution["SolutionIndex"]


def test_solutions_cap_the_tensor_bytes_per_kernel(shard):
    assert len(shard["Solutions"]) == NUM_SOLUTIONS
    for solution in shard["Solutions"]:
        index = solution["SolutionIndex"]
        assert solution["AssertTensorBytesLessThan"] == EXPECTED_TENSOR_BYTES[index], index


def _solutions_by_index(shard):
    return {solution["SolutionIndex"]: solution for solution in shard["Solutions"]}


def _problem_of(key):
    m, n, batch, k, ldd, ldc, lda, ldb = key
    return make_problem(m, n, batch, k, ldd, ldc, lda, ldb)


def _mirror(key):
    m, n, batch, k, ldd, ldc, lda, ldb = key
    return [n, m, batch, k, n, n, ldb, lda]


def test_every_key_satisfies_the_predicates_of_its_solution(shard, entries):
    solutions = _solutions_by_index(shard)
    predicates = {index: emitted_predicates(solution) for index, solution in solutions.items()}

    for key, (index, _) in entries.values():
        violated = failing(predicates[index], _problem_of(key))
        assert not violated, (key, index, [(p.tag, p.index, p.value) for p in violated])


def test_every_key_has_an_exact_mirrored_twin_its_solution_accepts(shard, entries):
    solutions = _solutions_by_index(shard)
    predicates = {index: emitted_predicates(solution) for index, solution in solutions.items()}
    by_key = {tuple(full_key): index for full_key, (index, _) in entries.values()}

    for full_key in by_key:
        twin = tuple(_mirror(full_key))
        assert twin in by_key, f"missing mirror of {full_key}"
        # The twin may pick another kernel: the D store pattern differs per orientation.
        index = by_key[twin]
        assert index in solutions, twin
        violated = failing(predicates[index], _problem_of(twin))
        assert not violated, (twin, index, [(p.tag, p.index, p.value) for p in violated])


def test_keys_per_kernel_match_the_benchmark_run(entries):
    counts = {index: 0 for index in range(NUM_SOLUTIONS)}
    for _, (index, _) in entries.values():
        counts[index] += 1

    assert counts == EXPECTED_KEYS_PER_KERNEL


def test_every_kernel_is_used_in_both_orientations(shard, entries):
    used = {"skinny_n": set(), "skinny_m": set()}
    for (m, n, _, _), (_, (index, _)) in entries.items():
        used["skinny_n" if _is_skinny_n(m, n) else "skinny_m"].add(index)

    assert used["skinny_n"] == used["skinny_m"] == set(_solutions_by_index(shard)), used


def test_solutions_derive_the_macro_tile_from_the_kernel_orientation(shard):
    # MacroTile0 is the wide extent and MacroTile1 the skinny one.
    for solution in shard["Solutions"]:
        wide, skinny, _ = solution["CustomKernel"]["macrotile"]
        assert (solution["MacroTile0"], solution["MacroTile1"]) == (wide, skinny)
