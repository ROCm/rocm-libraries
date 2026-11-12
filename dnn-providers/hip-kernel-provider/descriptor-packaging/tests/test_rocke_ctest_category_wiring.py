# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for deferred rocKE CTest category enrollment."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

_PROVIDER = Path(__file__).resolve().parents[2]
_ROCKE_CTEST_LISTS = _PROVIDER / "rocke" / "library" / "tests" / "ctest_lists.cmake"
_ROCKE_CATEGORIES = _PROVIDER / "ROCKE_ENGINE_test_categories_external.yaml"
_CATEGORY_PARSER = _PROVIDER.parents[1] / "shared/ctest/parse_ctest_categories.py"

_DEFERRED_TESTS = {
    "rocke_library_runner_contract_pytest",
    "rocke_library_worker_integration_pytest",
    "rocke_library_counted_structural_gate",
    "rocke_library_counted_compiler_gate",
}


def test_category_parser_leaves_deferred_tests_unenrolled(tmp_path):
    if not _CATEGORY_PARSER.is_file():
        pytest.skip("shared CTest category parser is unavailable")
    pytest.importorskip("yaml", reason="the shared category parser needs PyYAML")
    ctests = tmp_path / "CTestTestfile.cmake"
    ctests.write_text(
        "".join(f"add_test({name} python)\n" for name in sorted(_DEFERRED_TESTS)),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_CATEGORY_PARSER), str(_ROCKE_CATEGORIES), str(ctests)],
        check=True,
        capture_output=True,
        text=True,
    )
    labeled = {
        name
        for name, _value in re.findall(
            r'set_tests_properties\("([^"]+)" PROPERTIES LABELS "([^"]*)"\)',
            result.stdout,
        )
    }
    assert not _DEFERRED_TESTS & labeled


def test_deferred_tests_are_registered_without_ci_tier_owners():
    cmake = _ROCKE_CTEST_LISTS.read_text(encoding="utf-8")
    categories = _ROCKE_CATEGORIES.read_text(encoding="utf-8")

    # Fast counted/resource contracts remain source-runner coverage until a
    # provider CI tier has capacity for them.
    assert "test_attention_bwd_counted_sync.py" not in cmake
    assert "test_attention_bwd_resources.py" not in cmake
    assert 'pytest -m \\"not compiler_integration\\"' not in cmake

    for name in _DEFERRED_TESTS:
        assert f"add_test({name}" in cmake
        assert f'"{name}"' not in categories

    # Keep exact-name/manual CTest execution useful, but deliberately omit the
    # quick/standard/comprehensive/full labels selected by provider CI.
    assert 'LABELS \\"ckc\\\\;python\\\\;host\\\\;runner\\"' in cmake
    assert 'LABELS \\"ckc\\\\;python\\\\;host\\\\;runner\\\\;integration\\"' in cmake
    assert (
        'LABELS \\"ckc\\\\;python\\\\;attention\\\\;host\\\\;structural'
        '\\\\;hip-kernel-provider\\"' in cmake
    )
    assert (
        'LABELS \\"ckc\\\\;python\\\\;attention\\\\;host\\\\;compiler'
        '\\\\;hip-kernel-provider\\"' in cmake
    )

    assert "ROCKE_ISA_INSPECT_PROBE=" in cmake
    assert "ROCKE_BWD_NO_COMGR_PIN=" in cmake
    assert "ROCKE_BWD_NO_COMGR_PIN=1" not in cmake
