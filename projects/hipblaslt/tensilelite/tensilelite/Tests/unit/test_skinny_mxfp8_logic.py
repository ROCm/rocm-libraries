# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks the shipped skinny MXFP8 shard against the kernels and predicates it relies on.

Every ExactLogic key must satisfy the predicates of its solution, the shard must not lose a key to
another Equality file of its library, its solutions must be exactly the shipped skinny kernels, and
they must load with a ProblemType that rejects bias, activation and scale-alpha-vector problems.
These are checked here and not in TensileLogic.ValidCorpusConsistency, which only reads file
headers: they need a per-solution parse, which would cost a full parse of every Equality file in
each build and would flag duplicate keys outside this shard.
"""

import copy
import itertools
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

import tensilelite
import tensilelite.LibraryIO as L

from tensilelite.Contractions import ProblemPredicate
from tensilelite.Contractions import Solution as ContractionSolution
from tensilelite.Contractions import ProblemType as ContractionProblemType
from tensilelite.CustomKernels import readCustomKernelConfig
from tensilelite.LibraryIO import normalizeLogicProblemType
from tensilelite.SolutionStructs.Problem import ProblemType

from logic_corpus import needs_logic_corpus, require_asm_full_root

pytestmark = pytest.mark.unit

A_B_BYTES = 1
C_D_BYTES = 2


@dataclass(frozen=True)
class Tensor:
    sizes: tuple
    strides: tuple
    element_bytes: int

    def total_allocated_bytes(self):
        last = sum((size - 1) * stride for size, stride in zip(self.sizes, self.strides))
        return (last + 1) * self.element_bytes


@dataclass(frozen=True)
class Problem:
    sizes: tuple  # [m, n, batch, k]
    a: Tensor
    b: Tensor
    c: Tensor
    d: Tensor
    mx_scales_packed: bool = False  # the custom-kernel packed flag of either operand


def make_problem(m, n, batch, k, ldd, ldc, lda, ldb):
    """TN problem from an ExactLogic key: A is [k, m, batch], B [k, n, batch], C/D [m, n, batch]."""
    return Problem(
        sizes=(m, n, batch, k),
        a=Tensor((k, m, batch), (1, lda, lda * m), A_B_BYTES),
        b=Tensor((k, n, batch), (1, ldb, ldb * n), A_B_BYTES),
        c=Tensor((m, n, batch), (1, ldc, ldc * n), C_D_BYTES),
        d=Tensor((m, n, batch), (1, ldd, ldd * n), C_D_BYTES),
    )


def flatten(predicate):
    if predicate.tag == "And":
        for child in predicate.value:
            yield from flatten(child)
    else:
        yield predicate


def emitted_predicates(state):
    """The problem predicates the library builder derives from a solution's Assert keys."""
    result = []
    for pair in state.items():
        predicate = ProblemPredicate.FromOriginalKeyPair(pair)
        if predicate is not None:
            result.extend(flatten(predicate))
    return result


def _stride_multiple(stride, index, value):
    # A zero multiple or a broadcast (zero) stride is rejected, as in the C++ predicate.
    if value == 0 or index >= len(stride.strides) or stride.strides[index] == 0:
        return False
    return stride.strides[index] % value == 0


def evaluate(predicate, problem):
    # Mirrors ContractionProblemPredicates.hpp; an unknown tag raises so no predicate is skipped silently.
    tag, index, value = predicate.tag, predicate.index, predicate.value
    if tag == "Or":
        return any(evaluate(child, problem) for child in value)
    if tag == "SizeEqual":
        return problem.sizes[index] == value
    if tag == "SizeGreaterThan":
        return problem.sizes[index] > value
    if tag == "SizeLessThan":
        return problem.sizes[index] < value
    if tag == "MXScalesPreSwizzled":
        return not problem.mx_scales_packed
    if tag == "BoundSizeMultiple":
        # Index -1 is the last summation index, which is K.
        return problem.sizes[3] % value == 0
    tensors = {"A": problem.a, "B": problem.b, "C": problem.c, "D": problem.d}
    if tag.startswith("Stride") and tag.endswith("Equal"):
        return tensors[tag[len("Stride")]].strides[index] == value
    if tag in ("StrideAMultiple", "StrideBMultiple"):
        return _stride_multiple(tensors[tag[len("Stride")]], index, value)
    if tag == "TensorBytesLessThan":
        return all(t.total_allocated_bytes() < value for t in tensors.values())
    raise AssertionError(f"predicate tag {tag!r} has no evaluator")


def failing(predicates, problem):
    """The emitted predicates the problem violates."""
    return [p for p in predicates if not evaluate(p, problem)]


SKINNY_KERNEL_DIR = Path(tensilelite.__file__).parent / "CustomKernels" / "skinny_mxfp8"
SKINNY_KERNELS = sorted(path.stem for path in SKINNY_KERNEL_DIR.glob("*.s"))


def assert_items(mapping):
    return {key: value for key, value in mapping.items() if key.startswith("Assert")}


@pytest.fixture(scope="module")
def shard_path():
    matches = sorted((require_asm_full_root() / "gfx950/gfx950/Equality").glob("*_SkinnyGEMM.yaml"))
    if len(matches) != 1:
        pytest.fail(f"expected exactly one *_SkinnyGEMM.yaml shard, found {[m.name for m in matches]}", pytrace=False)
    return matches[0]


@pytest.fixture(scope="module")
def shard(shard_path):
    with open(shard_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def entries(shard):
    """Map (m, n, batch, k) to the full key and the solution index."""
    result = {}
    for key, (solution_index, _) in shard["ExactLogic"]:
        assert tuple(key[:4]) not in result, f"duplicate ExactLogic key {key}"
        result[tuple(key[:4])] = (key, solution_index)
    return result


def _config(solution):
    return readCustomKernelConfig(solution["CustomKernel"]["name"])


def _effective_predicates(solution):
    # custom.config wins at load and a logic file can only add predicates, so both apply.
    return emitted_predicates(solution) + emitted_predicates(_config(solution))


@needs_logic_corpus
def test_every_key_satisfies_the_predicates_of_its_solution(shard, entries):
    predicates = {
        solution["SolutionIndex"]: _effective_predicates(solution) for solution in shard["Solutions"]
    }
    for key, solution_index in entries.values():
        violated = failing(predicates[solution_index], make_problem(*key))
        assert not violated, (key, solution_index, [(p.tag, p.index, p.value) for p in violated])


def _best_efficiency_by_key(logic_data):
    """Map (m, n, batch, k) to (full key, best efficiency) over a logic file's ExactLogic table."""
    best = {}
    for key, (_, efficiency) in logic_data["ExactLogic"]:
        short_key = tuple(key[:4])
        if short_key not in best or efficiency > best[short_key][1]:
            best[short_key] = (tuple(key), efficiency)
    return best


def keys_lost_to(shard_data, other_data):
    """(full shard key, shard efficiency, other efficiency) for each (m, n, batch, k) in both tables where the other wins.

    Keys compare on their first four columns because MatchingLibrary.FromOriginalState keeps only
    those, so files that differ in the trailing columns still collide. The C++ loader sorts equal
    keys by descending speed (Serialization/MatchingLibrary.hpp) and the Equality lookup
    (PropertyMatching.hpp) takes the first, so a tie or a higher value elsewhere wins; each side
    therefore counts only its best efficiency per key.
    """
    other = _best_efficiency_by_key(other_data)
    lost = []
    for short_key, (full_key, efficiency) in _best_efficiency_by_key(shard_data).items():
        if short_key not in other:
            continue
        other_efficiency = other[short_key][1]
        if other_efficiency >= efficiency:
            lost.append((full_key, efficiency, other_efficiency))
    return lost


def _library_slot(logic_data, path):
    """Operation identifier and placeholder name, which together select the library a logic file lands in."""
    state = copy.deepcopy(logic_data["ProblemType"])
    normalizeLogicProblemType(state)
    problem_type = ContractionProblemType.FromOriginalState(
        ProblemType(state, False, srcFile=str(path), raiseOnTypeMismatch=False).state
    )
    return (
        problem_type.operationIdentifier,
        problem_type.placeholderStr(includeBatch=True, includeType=True),
    )


@needs_logic_corpus
def test_shard_wins_every_key_it_shares_with_an_equality_file_of_its_library(shard, shard_path):
    slot = _library_slot(shard, shard_path)
    for path in sorted(shard_path.parent.glob("*.yaml")):
        if path == shard_path:
            continue
        with open(path) as f:
            other = yaml.safe_load(f)
        if other.get("LibraryType") != "Equality" or _library_slot(other, path) != slot:
            continue
        lost = keys_lost_to(shard, other)
        assert not lost, (
            f"{path.name} wins {len(lost)} key(s) of the skinny shard, e.g. {lost[0]}: "
            "regenerate with UseEffLike False or drop the key from one file"
        )


# The ProblemType fields that make a logic file the skinny shard's MXFP8 TN problem type.
_MXFP8_TN_KEYS = (
    "DataTypeA",
    "DataTypeB",
    "DestDataType",
    "TransposeA",
    "TransposeB",
    "DataTypeMXSA",
    "MXBlockA",
    "MXBlockB",
)


@needs_logic_corpus
def test_no_chip_specific_logic_shadows_the_skinny_shard(shard, shard_path):
    # The shard is a fallback row (Device 75a0); an exact-chip row that returns a solution wins over it.
    wanted = [shard["ProblemType"].get(key) for key in _MXFP8_TN_KEYS]
    shadowing = []
    for path in sorted(shard_path.parents[2].glob("gfx950_id*/*/*.yaml")):
        with open(path) as f:  # Read only the header: the Solutions list is megabytes.
            header = yaml.safe_load("".join(itertools.takewhile(lambda line: not line.startswith("Solutions:"), f)))
        if [header["ProblemType"].get(key) for key in _MXFP8_TN_KEYS] == wanted:
            shadowing.append(path.name)
    assert not shadowing, (
        f"{shadowing} hold logic for the skinny shard's MXFP8 TN problem type: an exact-chip row wins over the "
        "Device 75a0 fallback row, so the skinny kernels would stop applying on that chip id."
    )


@needs_logic_corpus
def test_skinny_solutions_reject_problems_with_both_extents_above_4(shard):
    # make_problem arguments: m, n, batch, k, ldd, ldc, lda, ldb.
    both_wide = [
        make_problem(5, 5, 1, 4096, 5, 5, 4096, 4096),
        make_problem(5120, 5120, 1, 8192, 5120, 5120, 8192, 8192),
    ]
    for solution in shard["Solutions"]:
        predicates = _effective_predicates(solution)
        for problem in both_wide:
            assert failing(predicates, problem), (solution["CustomKernel"]["name"], problem.sizes)


@pytest.mark.parametrize("name", SKINNY_KERNELS)
def test_custom_config_rejects_packed_mx_scales(name):
    # The kernels read the pre-swizzled scale layout only; hipBLASLt flags every other format as packed.
    # Checked on the config alone so the predicate stays guarded if the shard copy is dropped.
    predicates = emitted_predicates(readCustomKernelConfig(name))
    plain = make_problem(4096, 1, 1, 4096, 4096, 4096, 4096, 4096)
    assert not failing(predicates, plain)
    assert {p.tag for p in failing(predicates, replace(plain, mx_scales_packed=True))} == {"MXScalesPreSwizzled"}


@needs_logic_corpus
@pytest.mark.parametrize("name", SKINNY_KERNELS)
def test_shard_matches_custom_config(shard, name):
    # The host launches from custom.config, but the shard copies still feed the predicates and sizeMapping.
    solution = next(s for s in shard["Solutions"] if s["CustomKernel"]["name"] == name)
    config = readCustomKernelConfig(name)
    descriptor = config["CustomKernel"]
    threads = math.prod(descriptor["threads"])
    mt0, mt1, depth = descriptor["macrotile"]
    assert assert_items(solution) == assert_items(config)
    assert solution["CustomKernel"]["threads"] == descriptor["threads"]
    assert solution["NumThreads"] == threads
    assert math.prod(solution["WorkGroup"]) == threads
    assert solution["MIWaveTile"] == config["MIWaveTile"]
    assert solution["MatrixInstruction"] == config["MatrixInstruction"]
    assert solution["CustomKernel"]["macrotile"] == descriptor["macrotile"]
    assert (solution["MacroTile0"], solution["MacroTile1"], solution["DepthU"]) == (mt0, mt1, depth)


def test_stride_multiples_are_not_swappable():
    # Different values per operand, so a swapped A/B mapping changes which problems fail.
    predicates = emitted_predicates({"AssertStrideAMultiple": {1: 4}, "AssertStrideBMultiple": {1: 8}})
    assert sorted(p.tag for p in predicates) == ["StrideAMultiple", "StrideBMultiple"]
    # make_problem arguments: m, n, batch, k, ldd, ldc, lda, ldb.
    assert not failing(predicates, make_problem(4, 64, 1, 4096, 4, 4, 4096, 4096))
    padded_b = make_problem(4, 64, 1, 4096, 4, 4, 4096, 4100)
    assert [p.tag for p in failing(predicates, padded_b)] == ["StrideBMultiple"]
    padded_a = make_problem(4, 64, 1, 4096, 4, 4, 4098, 4096)
    assert [p.tag for p in failing(predicates, padded_a)] == ["StrideAMultiple"]


@needs_logic_corpus
def test_shard_solutions_are_exactly_the_shipped_skinny_kernels(shard):
    assert SKINNY_KERNELS
    assert {solution["CustomKernel"]["name"] for solution in shard["Solutions"]} == set(SKINNY_KERNELS)


def _problem_type_predicates(problem_type):
    predicates = ContractionProblemType.FromOriginalState(problem_type).predicates(True, True, True)
    return [(p.tag, getattr(p, "index", None), getattr(p, "value", None)) for p in predicates]


@needs_logic_corpus
def test_skinny_solutions_reject_bias_activation_and_scale_alpha_vector(
        shard, shard_path, assembler, isa_info_map):
    state = copy.deepcopy(shard["ProblemType"])
    normalizeLogicProblemType(state)
    shared = _problem_type_predicates(
        ProblemType(state, False, srcFile=str(shard_path), raiseOnTypeMismatch=False)
    )
    assert ("UseBias", None, 1) in shared
    shared_activation = [p for p in shared if p[0] == "Activation"]

    data = L.read(str(shard_path), True)
    logic = L.parseLibraryLogicData(
        data, str(shard_path), assembler, False, False, False, isa_info_map, False
    )
    assert len(logic.solutions) == len(shard["Solutions"])
    for solution in logic.solutions:
        name = solution["CustomKernel"]["name"]
        predicates = _problem_type_predicates(solution["ProblemType"])
        assert ("UseBias", None, 0) in predicates, name
        assert ("UseScaleAlphaVec", None, 0) in predicates, name
        assert ("UseBias", None, 1) not in predicates, name
        assert ("UseScaleAlphaVec", None, 1) not in predicates, name
        activation = [p for p in predicates if p[0] == "Activation"]
        assert activation and activation != shared_activation, name
        assert not any(p[0] == "ActivationEnumWhiteList" for p in predicates), name


_SIZE_TAGS = ("SizeEqual", "SizeGreaterThan", "SizeLessThan")


def _size_predicates(predicate):
    """The size predicates of a problem predicate tree; the ProblemType ones need a full problem to evaluate."""
    if predicate.tag == "And":
        for child in predicate.value:
            yield from _size_predicates(child)
    elif predicate.tag in _SIZE_TAGS:
        yield predicate
    elif predicate.tag == "Or" and all(child.tag in _SIZE_TAGS for child in predicate.value):
        yield predicate


@needs_logic_corpus
def test_skinny_solutions_reject_batched_problems(shard_path, entries, assembler, isa_info_map):
    data = L.read(str(shard_path), True)
    logic = L.parseLibraryLogicData(
        data, str(shard_path), assembler, False, False, False, isa_info_map, False
    )
    key_of_solution = {}
    for key, solution_index in entries.values():
        key_of_solution.setdefault(solution_index, key)

    for solution in logic.solutions:
        name = solution["CustomKernel"]["name"]
        problem_predicate = ContractionSolution.FromSolutionStruct(
            solution, False, False, False, assembler, isa_info_map
        ).problemPredicate
        sizes = list(_size_predicates(problem_predicate))
        assert sizes, name

        m, n, _, k, *strides = key_of_solution[solution["SolutionIndex"]]
        assert not failing(sizes, make_problem(m, n, 1, k, *strides)), name
        violated = failing(sizes, make_problem(m, n, 2, k, *strides))
        assert any(p.tag == "SizeEqual" and p.index == 2 and p.value == 1 for p in violated), name
