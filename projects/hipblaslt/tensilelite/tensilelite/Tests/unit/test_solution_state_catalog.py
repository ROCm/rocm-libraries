# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ solution-state tables have to stay the Python tables.

solutionFromState copies the data-type names and the build chip ids by hand.
A name added on the Python side and not in SolutionState.cpp would build a
record the Python path would not.
"""

import re
from pathlib import Path

import pytest

from rocisa.enum import DataTypeEnum

from tensilelite.Common.Architectures import GFX_CHIP_IDS

pytestmark = pytest.mark.unit


def _solution_state_source():
    """Return SolutionState.cpp from the source tree or the checkout.

    TheRock copies this test to ``build/share/hipblaslt/tensilelite`` and runs
    it there. That copy has no ``src/msgpack``. The checkout that produced it
    is a parent of the copy.
    """
    here = Path(__file__).resolve()
    beside_package = here.parents[3] / "src" / "msgpack" / "SolutionState.cpp"
    if beside_package.is_file():
        return beside_package
    relative = Path("projects/hipblaslt/tensilelite/src/msgpack/SolutionState.cpp")
    for parent in here.parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(beside_package)


def _array(name):
    text = _solution_state_source().read_text()
    match = re.search(rf"{name}(?:\[\])?\s*=\s*\{{(.*?)\}};", text, re.S)
    assert match, name
    return match.group(1)


def test_build_chip_ids_match_python():
    quoted = re.findall(r'"([0-9a-fA-F]+)"', _array("SupportedBuildChipIds"))
    python = [chip for chips in GFX_CHIP_IDS.values() for chip in chips]
    assert quoted == python


def test_data_type_names_match_python():
    names = re.findall(r'\{"([^"]+)"', _array("DataTypes"))
    python = [item.name for item in DataTypeEnum if item.name != "Count"]
    assert names == python
