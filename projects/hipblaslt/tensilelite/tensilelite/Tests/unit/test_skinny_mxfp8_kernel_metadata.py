# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks the six skinny MXFP8 kernels against their own custom.config.

The custom.config block embedded in each .s file is the single source of truth for the kernel
metadata. The Tests/custom YAML forks and the logic-shard solutions repeat it, so they are
checked against it here instead of against a second retyped table. The host sizes the launch
from this metadata alone (the kernel symbol is not parsed).
"""

import math
import os
from pathlib import Path

import pytest
import yaml

import tensilelite
from skinny_mxfp8_predicates import accepts, emitted_predicates, failing, make_problem
from tensilelite.CustomKernels import readCustomKernelConfig, validateCustomKernelStepAndSplitK

pytestmark = pytest.mark.unit

KERNEL_DIR = os.path.join(os.path.dirname(tensilelite.__file__), "CustomKernels", "skinny_mxfp8")
TESTS_CUSTOM_DIR = Path(tensilelite.__file__).parent / "Tests" / "custom"
SHARD = (
    Path(tensilelite.__file__).resolve().parents[2]
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Equality"
    / "gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml"
)

GRIDS = (["TilesWide", "TilesSkinny", "Batch"], ["TilesWideSkinnyBatch", "One", "One"])
K_TAGS = {"BoundSizeMultiple", "SizeMultiple", "SizeGreaterThan"}


# Signed 32-bit K arithmetic in these kernels: LANES_K16 shifts K arithmetically by 10 and GRID_3D0 computes
# (K + spk - 1) / spk, so each needs an exclusive K upper bound.
K_UPPER_BOUNDS = {"LANES_K16": {3: (1 << 31) - 1023}, "GRID_3D0": {3: (1 << 30) + 1}}


def kernel_kind(name):
    for kind in K_UPPER_BOUNDS:
        if kind in name:
            return kind
    return None


def kernel_names():
    return sorted(f[:-2] for f in os.listdir(KERNEL_DIR) if f.endswith(".s"))


KERNELS = kernel_names()


def assert_items(mapping):
    return {key: value for key, value in mapping.items() if key.startswith("Assert")}


def k_problem(k):
    """A TN problem whose only possible violation is K; leading dims are padded to 4."""
    ld = max(-(-k // 4) * 4, 4)
    return make_problem(m=64, n=2, batch=1, k=k, ldd=64, ldc=64, lda=ld, ldb=ld)


def k_multiple(predicates):
    """The K granule the predicates enforce: the lcm of every K multiple they emit."""
    multiple = 1
    for p in predicates:
        if p.tag == "BoundSizeMultiple" or (p.tag == "SizeMultiple" and p.index == 3):
            multiple = math.lcm(multiple, p.value)
    return multiple


def test_there_are_exactly_six_kernels():
    assert len(KERNELS) == 6


# ---------------------------------------------------------------------------
# Invariants of every kernel's custom.config.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", KERNELS)
def test_custom_kernel_invariants(name):
    config = readCustomKernelConfig(name)
    descriptor = config["CustomKernel"]

    assert descriptor["grid"] in GRIDS
    assert descriptor["noSplitK"] is True
    assert descriptor["threads"][1:] == [1, 1]
    assert config["MXScaleFormat"] == "HostPreSwizzle"
    # The Spk kernarg carries K rounded to kStep, next to the one SizeSum.
    semantics = [entry["semantic"] for entry in descriptor["args"]]
    assert semantics.count("SizeSum") == 1
    assert "Spk" in semantics
    validateCustomKernelStepAndSplitK(config)


@pytest.mark.parametrize("name", KERNELS)
def test_kernel_serves_both_orientations(name):
    # Only the wide-extent cap on both M and N; no guard that singles out one skinny side.
    config = readCustomKernelConfig(name)
    assert config["AssertSizeLessThan"] == {0: 1048577, 1: 1048577, **K_UPPER_BOUNDS.get(kernel_kind(name), {})}
    assert config["AssertSizeGreaterThan"] == {0: 0, 1: 0, 3: 0}
    predicates = emitted_predicates(config)
    for m, n in ((5120, 3), (3, 5120), (4, 8191), (8191, 4)):
        problem = make_problem(m, n, 1, 5120, m, m, 5120, 5120)
        assert accepts(predicates, problem), (name, m, n)


@pytest.mark.parametrize("name", KERNELS)
def test_k_step_divides_the_admitted_k_multiple(name):
    config = readCustomKernelConfig(name)
    kStep = config["CustomKernel"]["kStep"]

    assert k_multiple(emitted_predicates(config)) % kStep == 0


@pytest.mark.parametrize("name", KERNELS)
def test_k_zero_is_rejected(name):
    predicates = emitted_predicates(readCustomKernelConfig(name))

    assert [p.tag for p in failing(predicates, k_problem(0))].count("SizeGreaterThan") == 1


@pytest.mark.parametrize("name", KERNELS)
def test_k_must_be_a_multiple_of_the_k_step(name):
    config = readCustomKernelConfig(name)
    predicates = emitted_predicates(config)
    kStep = config["CustomKernel"]["kStep"]

    for k in (kStep, 2 * kStep, 5 * kStep, 25600, 131072):
        assert accepts(predicates, k_problem(k)) == (k % kStep == 0), k
    for k in (1, 100, 255, 257, kStep + 1, 2 * kStep - 1):
        violated = failing(predicates, k_problem(k))
        assert violated, k
        assert {p.tag for p in violated} <= K_TAGS, k


@pytest.mark.parametrize("name", KERNELS)
def test_k_upper_bound_follows_the_kernel_arithmetic(name):
    predicates = emitted_predicates(readCustomKernelConfig(name))
    kind = kernel_kind(name)
    # K predicates only: a K this large cannot be paired with a small tensor.
    k_only = [p for p in predicates if p.tag != "TensorBytesLessThan"]

    def k_accepted(k):
        return accepts(k_only, k_problem(k))

    if kind == "LANES_K16":
        assert not k_accepted(1 << 31)
        assert not k_accepted((1 << 31) - 256)
        assert k_accepted((1 << 31) - 1024)
    elif kind == "GRID_3D0":
        assert not k_accepted((1 << 30) + 256)
        assert k_accepted(1 << 30)
    else:
        assert k_accepted((1 << 32) - 256)


@pytest.mark.parametrize("name", KERNELS)
def test_name_fields_agree_with_the_metadata(name):
    # The host never parses the name, so this cross-checks the two independently.
    descriptor = readCustomKernelConfig(name)["CustomKernel"]
    assert descriptor["grid"] == (
        ["TilesWideSkinnyBatch", "One", "One"] if "GRID_3D0" in name else ["TilesWide", "TilesSkinny", "Batch"]
    )
    assert ("GRID_3D1" in name) != ("GRID_3D0" in name)
    waves = int(name.split("_WAVES")[1].split("_")[0])
    assert descriptor["threads"][0] == 64 * waves
    assert descriptor["macrotile"][1] == int(name.split("gemm_BM")[1].split("_")[0])
    assert descriptor["kStep"] == (1024 if "LANES_K16" in name else 256)


def test_only_the_k16_kernel_rejects_an_odd_multiple_of_256():
    # 1280 = 5 * 256 is a whole number of 256-element steps but not of 1024.
    rejecting = [
        name
        for name in KERNELS
        if not accepts(emitted_predicates(readCustomKernelConfig(name)), k_problem(1280))
    ]
    assert [name for name in KERNELS if "LANES_K16" in name] == rejecting
    for name in KERNELS:
        predicates = emitted_predicates(readCustomKernelConfig(name))
        assert accepts(predicates, k_problem(1024 * 5)), name
        assert accepts(predicates, k_problem(1024)), name


# ---------------------------------------------------------------------------
# The copies of the metadata agree with custom.config.
# ---------------------------------------------------------------------------


def load_custom_yaml_for(name):
    """The one Tests/custom YAML whose fork names this kernel, and its parsed pieces."""
    found = []
    for path in sorted(TESTS_CUSTOM_DIR.glob("custom_skinny_mxfp8_tn_sel*.yaml")):
        with open(path) as f:
            doc = yaml.safe_load(f)
        problem_type, benchmark = doc["BenchmarkProblems"][0]
        forks = {k: v for fork in benchmark["ForkParameters"] for k, v in fork.items()}
        if forks["CustomKernel"][0]["name"] == name:
            found.append((path.name, doc, problem_type, forks))
    assert len(found) == 1, f"expected one Tests/custom YAML for {name}, found {len(found)}"
    return found[0]


@pytest.mark.parametrize("name", KERNELS)
def test_load_custom_yaml_for_matches_custom_config(name):
    config = readCustomKernelConfig(name)
    _, doc, problem_type, forks = load_custom_yaml_for(name)

    assert forks["CustomKernel"] == [{**config["CustomKernel"], "name": name}]
    asserts = assert_items(forks)
    assert asserts == {key: [value] for key, value in assert_items(config).items()}
    assert forks["MXScaleFormat"] == [config["MXScaleFormat"]]
    assert forks["StaggerU"] == [config["StaggerU"]]
    assert forks["MatrixInstruction"][0][: len(config["MatrixInstruction"])] == config["MatrixInstruction"]
    for key, value in config["ProblemType"].items():
        assert problem_type[key] == value, key
    assert doc["LibraryLogic"]["ArchitectureName"] == "gfx950"


@pytest.mark.parametrize("name", KERNELS)
def test_load_custom_yaml_for_runs_only_on_gfx950_device_0(name):
    _, doc, _, _ = load_custom_yaml_for(name)
    marks = doc["TestParameters"]["marks"]

    assert "skip-gfx950" not in marks
    assert "skip-gfx942" in marks
    assert doc["GlobalParameters"]["Device"] == 0


@pytest.fixture(scope="module")
def shard_solutions():
    with open(SHARD) as f:
        shard = yaml.safe_load(f)
    return shard["DefaultSolution"], shard["Solutions"]


@pytest.mark.parametrize("name", KERNELS)
def test_shard_solution_matches_custom_config(name, shard_solutions):
    defaults, solutions = shard_solutions
    config = readCustomKernelConfig(name)
    named = [s for s in solutions if s["CustomKernel"]["name"] == name]
    assert len(named) == 1
    solution = named[0]

    assert solution["CustomKernel"] == {**config["CustomKernel"], "name": name}
    assert solution["MXScaleFormat"] == config["MXScaleFormat"]
    # The loader lets custom.config win, so a shard value that differed would be dead or,
    # for a key the config lacks, would narrow the kernel.
    keys = set(assert_items(solution)) | set(assert_items(config))
    assert "AssertTensorBytesLessThan" in keys
    for key in keys:
        assert solution.get(key, defaults.get(key)) == config.get(key, defaults.get(key)), key


TWO_TO_32 = 2**32


# (tensor, element bytes, contiguous extent, leading stride, extent hits 2^32 exactly).
# A is [k, m] and B is [k, n], so the wide side of either orientation is their free size;
# C and D are [m, n] bf16. Strides are in elements.
TENSOR_CASES = [
    ("A", 1, 4096, 4096, True),
    ("A", 1, 4096, 4100, True),
    ("A", 1, 256, 260, True),
    ("A", 1, 7168, 7180, False),
    ("B", 1, 4096, 4096, True),
    ("B", 1, 4096, 4100, True),
    ("B", 1, 12800, 12804, False),
    ("C", 2, 4, 4, True),
    ("C", 2, 3, 3, False),
    ("D", 2, 4096, 4096, True),
    ("D", 2, 2, 2, True),
    # Distinct leading dimensions: the other of C and D keeps the minimum, so dropping the named
    # tensor from the evaluator is caught.
    ("C", 2, 4, 8, False),
    ("D", 2, 4, 8, False),
    ("C", 2, 4096, 4100, False),
    ("D", 2, 4096, 4100, False),
]


def problem_with_tensor(tensor, contiguous, free, ld):
    """A problem in which only the named tensor is large; the others stay small."""
    if tensor == "A":
        return make_problem(free, 1, 1, contiguous, free, free, ld, contiguous)
    if tensor == "B":
        return make_problem(1, free, 1, contiguous, 1, 1, contiguous, ld)
    # C and D share the [m, n] shape; only the named one gets the large leading dimension.
    if tensor == "C":
        return make_problem(contiguous, free, 1, 1, contiguous, ld, 1, 1)
    return make_problem(contiguous, free, 1, 1, ld, contiguous, 1, 1)


def first_free_at_limit(element_bytes, contiguous, ld):
    """Smallest free size whose extent reaches 2^32 bytes, from the closed form."""
    elements = -(-TWO_TO_32 // element_bytes)
    return 1 + -(-(elements - contiguous) // ld)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("tensor, element_bytes, contiguous, ld, exact", TENSOR_CASES)
def test_tensor_bytes_limit_separates_the_last_fitting_size_from_two_to_the_32(
    name, tensor, element_bytes, contiguous, ld, exact
):
    config = readCustomKernelConfig(name)
    assert config["AssertTensorBytesLessThan"] == TWO_TO_32
    predicates = [p for p in emitted_predicates(config) if p.tag == "TensorBytesLessThan"]
    assert [p.value for p in predicates] == [TWO_TO_32]
    first = first_free_at_limit(element_bytes, contiguous, ld)

    accept = problem_with_tensor(tensor, contiguous, first - 1, ld)
    reject = problem_with_tensor(tensor, contiguous, first, ld)
    tensors = {"A": "a", "B": "b", "C": "c", "D": "d"}
    accept_bytes = getattr(accept, tensors[tensor]).total_allocated_bytes()
    reject_bytes = getattr(reject, tensors[tensor]).total_allocated_bytes()
    assert accept_bytes < TWO_TO_32 <= reject_bytes
    assert (reject_bytes == TWO_TO_32) == exact
    assert accepts(predicates, accept)
    assert not accepts(predicates, reject)
