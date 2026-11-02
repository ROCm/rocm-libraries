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

NUM_SOLUTIONS = 6
MAX_SKINNY = 4
EXPECTED_SIZE_CAP = {0: 1048577, 1: 1048577}
# K upper bounds of the kernels with signed 32-bit K arithmetic, keyed by solution index.
EXPECTED_K_BOUNDS = {4: {3: (1 << 31) - 1023}, 5: {3: (1 << 30) + 1}}

EXCLUDED_K = (1280, 1536)


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
    assert len(entries) == 122
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


def test_entries_cover_both_orientations(entries):
    assert any(_is_skinny_n(m, n) for m, n, _, _ in entries)
    assert any(_is_skinny_m(m, n) for m, n, _, _ in entries)


def test_excluded_shapes_are_absent(entries):
    for m, n, _, k in entries:
        wide, skinny = max(m, n), min(m, n)
        assert k not in EXCLUDED_K, (m, n, k)
        assert not (wide == 128256 and k == 8192 and skinny in (1, 2, 4)), (m, n, k)
        assert not (wide == 5120 and skinny == 1 and k == 2048), (m, n, k)


def test_solutions_cap_the_wide_extent_symmetrically(shard):
    assert len(shard["Solutions"]) == NUM_SOLUTIONS
    for solution in shard["Solutions"]:
        expected = {**EXPECTED_SIZE_CAP, **EXPECTED_K_BOUNDS.get(solution["SolutionIndex"], {})}
        assert solution["AssertSizeLessThan"] == expected, solution["SolutionIndex"]


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
