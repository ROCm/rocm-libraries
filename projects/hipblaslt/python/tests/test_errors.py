# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every hipblasStatus_t value becomes a HipblasLtError that names it.

In a source checkout (as in CI) the values come from the checkout's own hipblas-common.h, so a PR
that adds a status fails here until status.hpp maps it, even when the installed ROCm predates it.
Otherwise they come from the hipblas-common.h the bindings were built against.
"""

import re
from pathlib import Path

import pytest

import hipblaslt
from hipblaslt import _build_info

_RELATIVE = Path("hipblas-common") / "hipblas-common.h"
_IN_TREE = (
    Path(__file__).resolve().parents[4]
    / "projects/hipblas-common/library/include"
    / _RELATIVE
)
_HEADER = str(
    _IN_TREE
    if _IN_TREE.is_file()
    else Path(_build_info.HIPBLAS_COMMON_INCLUDE_DIR) / _RELATIVE
)
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
