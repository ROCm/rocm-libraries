# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ solution-state tables have to stay the Python tables.

solutionFromState copies the data-type names and the build chip ids by hand.
The installed tensilelite test tree does not contain SolutionState.cpp, so
this test compares Python to the checked-in lists. SolutionRecord reads those
same lists back out of SolutionState.cpp.
"""

from pathlib import Path

import pytest

from rocisa.enum import DataTypeEnum

from tensilelite.Common.Architectures import GFX_CHIP_IDS

pytestmark = pytest.mark.unit


def _lines(name):
    path = Path(__file__).resolve().parent / "test_data" / name
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_build_chip_ids_match_python():
    python = [chip for chips in GFX_CHIP_IDS.values() for chip in chips]
    assert _lines("catalog_chip_ids.txt") == python


def test_data_type_names_match_python():
    python = [item.name for item in DataTypeEnum if item.name != "Count"]
    assert _lines("catalog_data_types.txt") == python
