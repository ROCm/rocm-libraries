# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import yaml

from Tensile.CustomYamlLoader import load_yaml_stream
from Tensile.LibraryIO import StrictTypeLoader


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[6]
CLASSIC_ROOT = (
    REPO_ROOT / "projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/gfx90c"
)
LITE_ROOT = (
    REPO_ROOT
    / "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic"
    / "asm_full/gfx90c/Equality"
)

LAYOUTS = {
    "Ailk_Bljk": {
        "transpose_a": False,
        "transpose_b": False,
        "assign_a": [0, 3, 2],
        "assign_b": [3, 1, 2],
        "tlua": True,
        "tlub": False,
    },
    "Ailk_Bjlk": {
        "transpose_a": False,
        "transpose_b": True,
        "assign_a": [0, 3, 2],
        "assign_b": [1, 3, 2],
        "tlua": True,
        "tlub": True,
    },
    "Alik_Bljk": {
        "transpose_a": True,
        "transpose_b": False,
        "assign_a": [3, 0, 2],
        "assign_b": [3, 1, 2],
        "tlua": False,
        "tlub": False,
    },
    "Alik_Bjlk": {
        "transpose_a": True,
        "transpose_b": True,
        "assign_a": [3, 0, 2],
        "assign_b": [1, 3, 2],
        "tlua": False,
        "tlub": True,
    },
}

VARIANTS = {
    "HHS_BH": {
        "data_type": 4,
        "dest_type": 4,
        "compute_type": 0,
        "hpa": True,
        "initial_strides_ab": False,
    },
    "SBIc": {
        "data_type": 0,
        "dest_type": 0,
        "compute_type": 0,
        "hpa": False,
        "initial_strides_ab": False,
    },
    "SBIIc": {
        "data_type": 0,
        "dest_type": 0,
        "compute_type": 0,
        "hpa": False,
        "initial_strides_ab": True,
    },
}

PROBLEM_LAYOUT_KEYS = (
    "TransposeA",
    "TransposeB",
    "IndexAssignmentsA",
    "IndexAssignmentsB",
    "TLUA",
    "TLUB",
)
PROBLEM_TYPE_KEYS = (
    "DataType",
    "DestDataType",
    "ComputeDataType",
    "HighPrecisionAccumulate",
    "UseInitialStridesAB",
)


def _load_classic(path):
    with path.open() as stream:
        return yaml.safe_load(stream)


def _load_lite(path):
    return load_yaml_stream(path, StrictTypeLoader)


def _package_name(layout, variant):
    return f"vega10_Cijk_{layout}_{variant}.yaml"


def _problem_value(problem, key):
    if key in ("ComputeDataType", "DestDataType"):
        return problem.get(key, problem["DataType"])
    if key == "UseInitialStridesAB":
        return problem.get(key, False)
    return problem[key]


def _expected_package_names():
    packages = {
        _package_name(layout, variant)
        for layout in LAYOUTS
        for variant in VARIANTS
    }
    packages.add("vega10_Cijk_Alik_Bljk_H_B_UserArgs.yaml")
    return packages


def test_complete_layout_variant_matrix_exists():
    assert {path.name for path in LITE_ROOT.glob("*.yaml")} == _expected_package_names()


@pytest.mark.parametrize("layout,layout_expected", LAYOUTS.items())
@pytest.mark.parametrize("variant,variant_expected", VARIANTS.items())
def test_package_problem_type_matches_filename_and_suffix(
    layout, layout_expected, variant, variant_expected
):
    logic = _load_lite(LITE_ROOT / _package_name(layout, variant))
    problem = logic[4]

    assert logic[1:3] == ["gfx90c", "gfx90c"]
    assert problem["TransposeA"] is layout_expected["transpose_a"]
    assert problem["TransposeB"] is layout_expected["transpose_b"]
    assert problem["IndexAssignmentsA"] == layout_expected["assign_a"]
    assert problem["IndexAssignmentsB"] == layout_expected["assign_b"]
    assert problem["TLUA"] is layout_expected["tlua"]
    assert problem["TLUB"] is layout_expected["tlub"]
    assert problem["DataType"] == variant_expected["data_type"]
    assert problem["DestDataType"] == variant_expected["dest_type"]
    assert _problem_value(problem, "ComputeDataType") == variant_expected["compute_type"]
    assert problem["HighPrecisionAccumulate"] is variant_expected["hpa"]
    assert (
        _problem_value(problem, "UseInitialStridesAB")
        is variant_expected["initial_strides_ab"]
    )


@pytest.mark.parametrize("name", sorted(_expected_package_names()))
def test_nested_solution_problem_types_match_package(name):
    logic = _load_lite(LITE_ROOT / name)
    package_problem = logic[4]

    assert logic[5]
    for solution in logic[5]:
        nested = solution["ProblemType"]
        for key in PROBLEM_LAYOUT_KEYS + PROBLEM_TYPE_KEYS:
            assert _problem_value(nested, key) == _problem_value(package_problem, key)


@pytest.mark.parametrize("name", sorted(_expected_package_names()))
def test_solution_invariants_and_exact_logic_reachability(name):
    logic = _load_lite(LITE_ROOT / name)
    solutions = logic[5]

    assert all(solution["ISA"] == [9, 0, 12] for solution in solutions)
    assert all(not solution["EnableMatrixInstruction"] for solution in solutions)
    assert all(solution["MatrixInstruction"] == [] for solution in solutions)
    assert all(solution["ScheduleIterAlg"] == 1 for solution in solutions)
    assert [solution["SolutionIndex"] for solution in solutions] == list(
        range(len(solutions))
    )

    selected = {entry[1][0] for entry in logic[7]}
    assert selected == set(range(len(solutions)))


def test_no_duplicate_catalog_problem_key_exists():
    problem_keys = {}
    for path in sorted(LITE_ROOT.glob("*.yaml")):
        problem = _load_lite(path)[4]
        key = yaml.safe_dump(problem, sort_keys=True)
        assert key not in problem_keys, (
            f"{path.name} duplicates the catalog key from {problem_keys.get(key)}"
        )
        problem_keys[key] = path.name


def test_lite_mac_vector_widths_are_current_and_codegen_compatible():
    for path in LITE_ROOT.glob("*.yaml"):
        logic = _load_lite(path)
        for solution in logic[5]:
            assert solution["VectorWidthA"] == solution["VectorWidthB"]
            assert solution["GlobalReadVectorWidthA"] in (
                1,
                solution["VectorWidthA"],
            )
            assert solution["GlobalReadVectorWidthB"] in (
                1,
                solution["VectorWidthB"],
            )


def test_prompt_gemm_mappings_use_tuned_solutions():
    hhh = _load_lite(LITE_ROOT / "vega10_Cijk_Alik_Bljk_H_B_UserArgs.yaml")
    qk = _load_classic(CLASSIC_ROOT / "vega10_Cijk_Alik_Bljk_SB_GB.yaml")

    hhh_solutions = {solution["SolutionIndex"]: solution for solution in hhh[5]}
    assert (hhh_solutions[0]["MacroTile0"], hhh_solutions[0]["MacroTile1"]) == (64, 128)
    assert (hhh_solutions[1]["MacroTile0"], hhh_solutions[1]["MacroTile1"]) == (128, 64)
    assert all(solution["ProblemType"]["SupportUserArgs"] for solution in hhh_solutions.values())

    hhh_mappings = {tuple(entry[0]): entry[1][0] for entry in hhh[7]}
    assert hhh_mappings[(8960, 16384, 1, 1536, 8960, 8960, 1536, 1536)] == 0
    assert hhh_mappings[(128, 16384, 12, 16384, 128, 128, 128, 16384)] == 1

    assert [(solution["MacroTile0"], solution["MacroTile1"], solution["DepthU"]) for solution in qk[5]] == [
        (256, 64, 12),
        (128, 64, 8),
    ]
    qk_mappings = {tuple(entry[0]): entry[1][0] for entry in qk[7]}
    assert qk_mappings[(256, 128, 12, 128, 256, 256, 256, 1536)] == 0
    assert qk_mappings[(16384, 16384, 12, 128, 16384, 16384, 256, 1536)] == 1


@pytest.mark.parametrize(
    "name,counts",
    {
        "vega10_Cijk_Ailk_Bljk_HHS_BH.yaml": ((109, 1540), (73, 844)),
        "vega10_Cijk_Ailk_Bljk_SBIIc.yaml": ((5, 12), (5, 12)),
        "vega10_Cijk_Ailk_Bljk_SBIc.yaml": ((5, 12), (5, 12)),
    }.items(),
)
def test_existing_nn_packages_remain_pipeline_specific(name, counts):
    classic = _load_classic(CLASSIC_ROOT / name)
    lite = _load_lite(LITE_ROOT / name)

    assert classic[1:3] == ["vega10", "gfx90c"]
    assert (len(classic[5]), len(classic[7])) == counts[0]
    assert (len(lite[5]), len(lite[7])) == counts[1]


def test_source_packages_do_not_contain_generated_databases_or_code_objects():
    allowed_suffixes = {".yaml", ".md"}

    for root in (CLASSIC_ROOT, LITE_ROOT.parent):
        unexpected = [
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and path.suffix not in allowed_suffixes
        ]
        assert unexpected == []
