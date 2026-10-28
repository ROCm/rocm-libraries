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

_SOURCE = Path(__file__).resolve().parents[3] / "src" / "msgpack" / "SolutionState.cpp"


def _array(name):
    text = _SOURCE.read_text()
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
