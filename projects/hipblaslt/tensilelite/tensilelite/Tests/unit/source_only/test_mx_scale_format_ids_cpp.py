# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The Python MX scale format IDs match the C++ enum and predicate.

Reads the TensileLite headers from the source tree, which the installed test
artifact does not stage.
"""

import re
from pathlib import Path

import pytest

from tensilelite import Contractions as C

pytestmark = pytest.mark.unit

_INCLUDE = Path(__file__).resolve().parents[4] / "include" / "Tensile"


def test_scale_format_ids_match_cpp():
    enum = re.search(
        r"enum class MXScaleFormat : int\s*\{(.*?)\};",
        (_INCLUDE / "MXScaleFormat.hpp").read_text(encoding="utf-8"),
        re.S,
    )
    assert enum
    assert {
        name: int(value) for name, value in re.findall(r"(\w+)\s*=\s*(\d+)", enum.group(1))
    } == C.MX_SCALE_FORMAT_IDS
    first_exact = re.search(
        r"kFirstExactFormat\s*=\s*(\d+);",
        (_INCLUDE / "ContractionProblemPredicates.hpp").read_text(encoding="utf-8"),
    )
    assert first_exact
    assert int(first_exact.group(1)) == C.MX_SCALE_FORMAT_IDS["HostPreSwizzle128x4"]
