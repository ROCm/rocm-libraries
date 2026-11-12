# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for comprehensive-only rocKE CTest enrollment."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

_PROVIDER = Path(__file__).resolve().parents[2]
_PROVIDER_CMAKE = _PROVIDER / "src" / "CMakeLists.txt"
_ROCKE_CTEST_LISTS = _PROVIDER / "rocke" / "library" / "tests" / "ctest_lists.cmake"
_ROCKE_CATEGORIES = _PROVIDER / "ROCKE_ENGINE_test_categories_external.yaml"
_CATEGORY_PARSER = _PROVIDER.parents[1] / "shared/ctest/parse_ctest_categories.py"

_NEW_TESTS = {
    "rocke_library_runner_contract_pytest",
    "rocke_library_worker_integration_pytest",
    "rocke_library_counted_structural_gate",
    "rocke_library_counted_compiler_gate",
}


def _late_staging_block() -> str:
    text = _PROVIDER_CMAKE.read_text(encoding="utf-8")
    start = text.index("# Label what was just folded in")
    end = text.index("# Install the ctest files", start)
    return text[start:end]


def test_late_staging_drain_applies_provider_and_rocke_categories():
    block = _late_staging_block()
    assert '"${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"' in block
    assert '"${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"' in block
    assert block.count("apply_ctest_category_labels(") == 2


def test_category_parser_assigns_comprehensive_only(tmp_path):
    if not _CATEGORY_PARSER.is_file():
        pytest.skip("shared CTest category parser is unavailable")
    pytest.importorskip("yaml", reason="the shared category parser needs PyYAML")
    ctests = tmp_path / "CTestTestfile.cmake"
    ctests.write_text(
        "".join(f"add_test({name} python)\n" for name in sorted(_NEW_TESTS)),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_CATEGORY_PARSER), str(_ROCKE_CATEGORIES), str(ctests)],
        check=True,
        capture_output=True,
        text=True,
    )
    labels = {
        name: set(value.split(";"))
        for name, value in re.findall(
            r'set_tests_properties\("([^"]+)" PROPERTIES LABELS "([^"]*)"\)',
            result.stdout,
        )
    }
    assert set(labels) == _NEW_TESTS
    for name in _NEW_TESTS:
        assert {"comprehensive", "full"} <= labels[name]
        assert not {"quick", "standard"} & labels[name]


def test_new_tests_are_registered_outside_fast_ci_tiers():
    cmake = _ROCKE_CTEST_LISTS.read_text(encoding="utf-8")
    categories = _ROCKE_CATEGORIES.read_text(encoding="utf-8")

    # Fast counted/resource contracts remain source-runner coverage rather than
    # increasing the existing quick attention CTest.
    assert "test_attention_bwd_counted_sync.py" not in cmake
    assert "test_attention_bwd_resources.py" not in cmake
    assert 'pytest -m \\"not compiler_integration\\"' not in cmake

    for name in _NEW_TESTS:
        assert f"add_test({name}" in cmake
        assert categories.count(f'"{name}"') == 1

    # Fallback labels and YAML categories both keep the new entries away from
    # quick/standard pre-merge and post-merge selection.
    assert (
        'LABELS \\"ckc\\\\;python\\\\;host\\\\;runner\\\\;comprehensive'
        '\\\\;full\\"' in cmake
    )
    assert (
        'LABELS \\"ckc\\\\;python\\\\;host\\\\;runner\\\\;integration'
        '\\\\;comprehensive\\\\;full\\"' in cmake
    )
    assert "structural\\\\;comprehensive\\\\;full" in cmake
    assert "compiler\\\\;comprehensive\\\\;full" in cmake

    assert "ROCKE_ISA_INSPECT_PROBE=" in cmake
    assert "ROCKE_BWD_NO_COMGR_PIN=" in cmake
    assert "ROCKE_BWD_NO_COMGR_PIN=1" not in cmake
