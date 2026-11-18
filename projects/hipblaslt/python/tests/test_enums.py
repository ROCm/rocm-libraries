# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every public enum value is bound, with the header's number (Review A, Decision 6).

The headers are read after the real preprocessor has run, so the build configuration decides
which opt-in values exist: a value inside `#if HIPBLASLT_HAS_...` counts only when the installed
hipblaslt-opt-in-features.h turns it on. In a source checkout (CI) these are develop's own
headers, paired with the three generated headers only an install has. Otherwise they're the
installed headers.

The pre-commit hook (scripts/check_enum_sync.py) is the first line of defense; this is the
backstop that sees the headers exactly as the library was configured.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import hipblaslt
from hipblaslt import _build_info
from hipblaslt._coverage import ALLOWED_MISSING, parse_enums

_REPO = Path(__file__).resolve().parents[4]
_IN_TREE = [
    _REPO / "projects" / "hipblaslt" / "library" / "include",
    _REPO / "projects" / "hipblas-common" / "library" / "include",
]
_HEADERS = ["hipblaslt/hipblaslt.h", "hipblaslt/hipblaslt-ext.hpp"]


def _uses_in_tree_headers() -> bool:
    return all(
        (d / "hipblaslt" / "hipblaslt.h").is_file() for d in _IN_TREE[:1]
    ) and all(d.is_dir() for d in _IN_TREE)


def _preprocess() -> str:
    includes = [str(d) for d in _IN_TREE] if _uses_in_tree_headers() else []
    # hipblas-common can live in a different prefix from hipBLASLt (invoke build --python).
    includes += [_build_info.INCLUDE_DIR, _build_info.HIPBLAS_COMMON_INCLUDE_DIR]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "headers.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.writelines(f"#include <{h}>\n" for h in _HEADERS)
        cmd = [_build_info.CXX_COMPILER, "-E", "-P", "-x", "c++", "-std=c++17"]
        cmd += ["-D__HIP_PLATFORM_AMD__=1", *(f"-I{d}" for d in includes), src]
        out = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    assert (
        out.returncode == 0
    ), f"preprocessing failed: {' '.join(cmd)}\n{out.stderr[-3000:]}"
    return out.stdout


def _owned(name: str) -> bool:
    """The enums the bindings own: every hipblasLt* enum (so a new one is caught), hipBLAS's
    common enums, the extension API's GemmType, and HIP's hipDataType."""
    return name.startswith("hipblasLt") or name in {
        "hipblasStatus_t",
        "hipblasOperation_t",
        "hipblasComputeType_t",
        "GemmType",
        "hipDataType",
    }


HEADER_ENUMS = parse_enums(_preprocess(), select=_owned)
REGISTRY = {
    c_enum: {c_name.split("::")[-1]: v for c_name, v in values.items()}
    for c_enum, values in hipblaslt._core._enum_registry().items()
}
_OURS = sorted(HEADER_ENUMS)
_SOURCE = (
    "develop's in-tree headers" if _uses_in_tree_headers() else "the installed headers"
)


def test_headers_were_found_and_parsed():
    # Guards against a preprocessing or parsing problem that makes the coverage test vacuous.
    for name in (
        "hipblasLtEpilogue_t",
        "hipblasLtMatmulDescAttributes_t",
        "hipDataType",
        "GemmType",
    ):
        assert len(HEADER_ENUMS[name].values) >= 2, f"{name} in {_SOURCE}"
    assert len(_OURS) >= 17


@pytest.mark.parametrize("c_enum", _OURS)
def test_every_value_is_bound_or_allowed(c_enum):
    bound = REGISTRY.get(c_enum, {})
    allowed = ALLOWED_MISSING.get(c_enum, {})
    missing = [
        v.name
        for v in HEADER_ENUMS[c_enum].values
        if v.name not in bound and v.name not in allowed
    ]
    assert not missing, (
        f"{c_enum} in {_SOURCE} has values the bindings don't bind: {missing}. Add a reg(...) line for each to "
        "projects/hipblaslt/python/src/enums.cpp, or add it to ALLOWED_MISSING in "
        "hipblaslt/_coverage.py with a reason."
    )


@pytest.mark.parametrize("c_enum", sorted(REGISTRY))
def test_bound_values_match_the_header(c_enum):
    header = HEADER_ENUMS[c_enum].by_name()
    wrong = {
        name: (
            py_value,
            header[name].value if name in header else "missing from the header",
        )
        for name, (_, _, py_value) in REGISTRY[c_enum].items()
        if name not in header or header[name].value != py_value
    }
    assert not wrong, f"{c_enum}: bound value vs header value: {wrong}"


@pytest.mark.parametrize("c_enum", sorted(REGISTRY))
def test_python_enum_matches_the_registry(c_enum):
    for _, (py_enum, py_name, value) in REGISTRY[c_enum].items():
        assert int(getattr(getattr(hipblaslt._core, py_enum), py_name)) == value


def test_allow_list_entries_have_reasons_and_exist():
    for c_enum, values in ALLOWED_MISSING.items():
        for name, reason in values.items():
            assert reason.strip(), f"{c_enum}.{name} has no reason"
            if c_enum in HEADER_ENUMS:
                assert (
                    name in HEADER_ENUMS[c_enum].by_name()
                ), f"{c_enum}.{name} isn't in the header"
            assert name not in REGISTRY.get(
                c_enum, {}
            ), f"{c_enum}.{name} is allow-listed and bound"
