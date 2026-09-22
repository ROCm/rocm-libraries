# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from multi_arch_test_projects import select_test_projects


@pytest.mark.parametrize(
    ("changed_project", "expected"),
    [
        (
            "shared/tensile",
            "hipblas,hipblaslt,rocblas,tensilelite",
        ),
        (
            "shared/origami",
            "hipblas,hipblaslt,origami,rocblas,tensilelite",
        ),
        (
            "shared/mxdatagenerator",
            "hipblas,hipblaslt,rocblas,rocroller,tensilelite",
        ),
        ("shared/rocroller", "rocroller"),
        (
            "shared/stinkytofu",
            "hipblas,hipblaslt,rocblas,tensilelite",
        ),
    ],
)
def test_all_shared_projects_select_multi_arch_tests(
    changed_project: str, expected: str
) -> None:
    assert select_test_projects(changed_project) == expected


def test_project_path_passes_through_unchanged() -> None:
    assert select_test_projects("projects/hipblaslt") == "projects/hipblaslt"


def test_mixed_projects_are_deduplicated_in_input_order() -> None:
    result = select_test_projects("shared/stinkytofu,shared/origami")

    assert result == "hipblas,hipblaslt,rocblas,tensilelite,origami"


def test_empty_project_list_selects_all_tests() -> None:
    assert select_test_projects("") == ""


def test_unmapped_shared_project_fails_loudly() -> None:
    with pytest.raises(
        ValueError,
        match="Shared project has no test mapping: shared/new-shared-library",
    ):
        select_test_projects("shared/new-shared-library")
