# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Self-tests for check_enum_sync.py, run by the hipblaslt-python-enum-sync-tests pre-commit hook.

String fixtures only: nothing is built, and the hipblaslt package is never imported.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "check_enum_sync", _HERE / "check_enum_sync.py"
)
sync = importlib.util.module_from_spec(_spec)
sys.modules["check_enum_sync"] = sync
_spec.loader.exec_module(sync)
coverage = sync.load_coverage(_HERE.parents[3])

HEADER = """
typedef enum {
    HIPBLASLT_EPILOGUE_DEFAULT = 1,
    HIPBLASLT_EPILOGUE_RELU = 2, /**< {braces in a comment} */
} hipblasLtEpilogue_t;

typedef enum {
    HIPBLASLT_MATMUL_DESC_TRANSA = 0,
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE = 106,
#endif
    HIPBLASLT_MATMUL_DESC_MAX = 108,
} hipblasLtMatmulDescAttributes_t;

typedef enum { NOT_OURS_A = 0 } someOtherLibrary_t;
"""

ENUMS_CPP = """
    reg(e, "hipblasLtEpilogue_t", "DEFAULT", HIPBLASLT_EPILOGUE_DEFAULT);
    reg(e, "hipblasLtEpilogue_t", "RELU", HIPBLASLT_EPILOGUE_RELU);
    reg(e, "hipblasLtMatmulDescAttributes_t", "TRANSA", HIPBLASLT_MATMUL_DESC_TRANSA);
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    reg(e, "hipblasLtMatmulDescAttributes_t", "FUSED_EPILOGUE", HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE);
#endif
"""


def _check(header=HEADER, enums_cpp=ENUMS_CPP):
    return sync.check(
        {"projects/hipblaslt/library/include/hipblaslt/hipblaslt.h": header},
        enums_cpp,
        coverage,
    )


def test_in_sync_is_clean():
    # HIPBLASLT_MATMUL_DESC_MAX is a sentinel in the real ALLOWED_MISSING.
    assert _check() == []


def test_a_new_value_fails_with_the_fix_in_the_message():
    header = HEADER.replace("RELU = 2,", "RELU = 2,\n    HIPBLASLT_EPILOGUE_FOO = 200,")
    (problem,) = _check(header)
    assert "hipblaslt.h has `HIPBLASLT_EPILOGUE_FOO = 200`" in problem
    assert 'reg(e, "hipblasLtEpilogue_t", "FOO", HIPBLASLT_EPILOGUE_FOO);' in problem
    assert "ALLOWED_MISSING" in problem


def test_a_guarded_value_must_be_bound_under_its_guard():
    unguarded = ENUMS_CPP.replace("#if HIPBLASLT_HAS_GEMM_A2A_FUSION\n", "").replace(
        "#endif\n", ""
    )
    (problem,) = _check(enums_cpp=unguarded)
    assert (
        "under `#if HIPBLASLT_HAS_GEMM_A2A_FUSION`" in problem and "no guard" in problem
    )


def test_a_missing_guarded_value_says_to_use_the_guard():
    no_fused = "\n".join(l for l in ENUMS_CPP.split("\n") if "FUSED" not in l)
    (problem,) = _check(enums_cpp=no_fused)
    assert (
        "inside `#if HIPBLASLT_HAS_GEMM_A2A_FUSION`" in problem
        and "under the same guard" in problem
    )


def test_a_removed_header_value_fails():
    header = HEADER.replace(
        "    HIPBLASLT_EPILOGUE_RELU = 2, /**< {braces in a comment} */\n", ""
    )
    (problem,) = _check(header)
    assert "binds hipblasLtEpilogue_t.HIPBLASLT_EPILOGUE_RELU" in problem


def test_bound_and_allow_listed_at_once_fails():
    both = (
        ENUMS_CPP
        + '    reg(e, "hipblasLtMatmulDescAttributes_t", "MAX", HIPBLASLT_MATMUL_DESC_MAX);\n'
    )
    (problem,) = _check(enums_cpp=both)
    assert "both bound" in problem


def test_other_libraries_enums_are_ignored():
    assert all("someOtherLibrary_t" not in p for p in _check())


def test_unreadable_file_is_an_error_not_a_pass(tmp_path):
    with pytest.raises(SystemExit, match="can't read"):
        sync.read(tmp_path, "no/such/file.h")
