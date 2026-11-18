# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every hipblasStatus_t value becomes a HipblasLtError that names it.

The values come from the hipblas-common.h the bindings were built against, so a status added to
the header later fails here until status.hpp maps it.
"""

import os
import re

import pytest

import hipblaslt
from hipblaslt import _build_info

_HEADER = os.path.join(_build_info.INCLUDE_DIR, "hipblas-common", "hipblas-common.h")
_ENUM = re.compile(r"typedef\s+enum\s*\{(?P<body>[^}]*)\}\s*hipblasStatus_t\s*;", re.S)
_VALUE = re.compile(r"\b(HIPBLAS_STATUS_\w+)\s*=\s*(\d+)")


def _statuses() -> list[tuple[str, int]]:
    with open(_HEADER, encoding="utf-8", errors="replace") as f:
        m = _ENUM.search(f.read())
    assert m, f"no hipblasStatus_t enum in {_HEADER}"
    body = re.sub(r"/\*.*?\*/|//[^\n]*", "", m.group("body"), flags=re.S)
    return [(name, int(value)) for name, value in _VALUE.findall(body)]


STATUSES = _statuses()


def test_header_parse_found_the_statuses():
    # Guards against a parser that finds nothing and makes the parametrized test vacuous.
    names = {n for n, _ in STATUSES}
    assert {"HIPBLAS_STATUS_SUCCESS", "HIPBLAS_STATUS_INVALID_VALUE"} <= names
    assert len(STATUSES) >= 12


@pytest.mark.parametrize(
    "name,value", [s for s in STATUSES if s[1] != 0], ids=lambda s: str(s)
)
def test_status_raises_with_its_name(name, value):
    with pytest.raises(hipblaslt.HipblasLtError) as excinfo:
        hipblaslt._core._raise_test_status(value)
    message = str(excinfo.value)
    assert name in message
    assert f"({value})" in message
    assert "_raise_test_status" in message


def test_success_does_not_raise():
    hipblaslt._core._raise_test_status(0)


def test_unknown_status_still_raises():
    with pytest.raises(hipblaslt.HipblasLtError, match="unmapped"):
        hipblaslt._core._raise_test_status(9999)


def test_error_is_an_exception():
    assert issubclass(hipblaslt.HipblasLtError, Exception)
