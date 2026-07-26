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
    REPO_ROOT
    / "projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/gfx90c"
)
LITE_ROOT = (
    REPO_ROOT
    / "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx90c/Equality"
)

EXPECTED = {
    "vega10_Cijk_Ailk_Bljk_HB.yaml": ((85, 853), (87, 860)),
    "vega10_Cijk_Ailk_Bljk_HHS_BH.yaml": ((109, 1540), (73, 844)),
    "vega10_Cijk_Ailk_Bljk_SBIIc.yaml": ((5, 12), (5, 12)),
    "vega10_Cijk_Ailk_Bljk_SBIc.yaml": ((5, 12), (5, 12)),
}


def _load_classic(path):
    with path.open() as stream:
        return yaml.safe_load(stream)


def _load_lite(path):
    return load_yaml_stream(path, StrictTypeLoader)


@pytest.mark.parametrize("name,counts", EXPECTED.items())
def test_gfx90c_logic_packages_are_pipeline_specific(name, counts):
    classic = _load_classic(CLASSIC_ROOT / name)
    lite = _load_lite(LITE_ROOT / name)

    assert classic[1:3] == ["vega10", "gfx90c"]
    assert lite[1:3] == ["gfx90c", "gfx90c"]
    assert (len(classic[5]), len(classic[7])) == counts[0]
    assert (len(lite[5]), len(lite[7])) == counts[1]

    assert all(
        solution.get("ISA", [9, 0, 12]) == [9, 0, 12]
        for solution in classic[5]
    )
    assert all(solution["ISA"] == [9, 0, 12] for solution in lite[5])
    assert all(not solution["EnableMatrixInstruction"] for solution in lite[5])
    assert all(solution["MatrixInstruction"] == [] for solution in lite[5])
    assert all(solution["ScheduleIterAlg"] == 1 for solution in lite[5])

    assert [solution["SolutionIndex"] for solution in lite[5]] == list(
        range(len(lite[5]))
    )
    selected = {entry[1][0] for entry in lite[7]}
    assert selected == set(range(len(lite[5])))


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


def test_source_packages_do_not_contain_generated_databases_or_code_objects():
    allowed_suffixes = {".yaml", ".md"}

    for root in (CLASSIC_ROOT, LITE_ROOT.parent):
        unexpected = [
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and path.suffix not in allowed_suffixes
        ]
        assert unexpected == []
