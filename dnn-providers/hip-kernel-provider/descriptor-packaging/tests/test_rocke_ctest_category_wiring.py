# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for rocKE's pre-registered CTest category wiring."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

_PROVIDER = Path(__file__).resolve().parents[2]
_PROVIDER_CMAKE = _PROVIDER / "src" / "CMakeLists.txt"
_ROCKE_CMAKE = _PROVIDER / "rocke" / "platform" / "CMakeLists.txt"
_ROCKE_CATEGORIES = _PROVIDER / "ROCKE_ENGINE_test_categories_external.yaml"
_CATEGORY_PARSER = _PROVIDER.parents[1] / "shared/ctest/parse_ctest_categories.py"


def _late_staging_block() -> str:
    text = _PROVIDER_CMAKE.read_text(encoding="utf-8")
    start = text.index("# Second drain of the common external staging")
    end = text.index("# Install the ctest files", start)
    return text[start:end]


def test_late_staging_drain_applies_provider_and_rocke_categories():
    """A build without the early cross-provider drain still labels rocKE CTests."""
    block = _late_staging_block()
    assert '"${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"' in block
    assert '"${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"' in block
    assert block.count("apply_ctest_category_labels(") == 2


def test_category_parser_produces_counted_gate_tiers(tmp_path):
    if not _CATEGORY_PARSER.is_file():
        pytest.skip("shared CTest category parser is unavailable")
    pytest.importorskip("yaml", reason="the shared category parser needs PyYAML")
    ctests = tmp_path / "CTestTestfile.cmake"
    ctests.write_text(
        "add_test(rocke_library_runner_contract_pytest python)\n"
        "add_test(rocke_library_worker_integration_pytest python)\n"
        "add_test(rocke_library_counted_structural_gate python)\n"
        "add_test(rocke_library_counted_compiler_gate python)\n",
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
    assert {"quick", "standard", "comprehensive", "full"} <= labels[
        "rocke_library_runner_contract_pytest"
    ]
    worker = labels["rocke_library_worker_integration_pytest"]
    assert {"standard", "comprehensive", "full", "runner", "integration"} <= worker
    assert "quick" not in worker
    structural = labels["rocke_library_counted_structural_gate"]
    compiler = labels["rocke_library_counted_compiler_gate"]
    assert {
        "standard",
        "comprehensive",
        "full",
        "structural",
        "attention",
    } <= structural
    assert "quick" not in structural
    assert {"comprehensive", "full", "compiler", "attention"} <= compiler
    assert not {"quick", "standard"} & compiler


def test_counted_gates_have_installed_and_tier_owners():
    cmake = _ROCKE_CMAKE.read_text(encoding="utf-8")
    categories = _ROCKE_CATEGORIES.read_text(encoding="utf-8")
    assert "add_test(rocke_library_runner_contract_pytest" in cmake
    assert categories.count('"rocke_library_runner_contract_pytest"') == 2
    assert "add_test(rocke_library_worker_integration_pytest" in cmake
    assert categories.count('"rocke_library_worker_integration_pytest"') == 1
    assert "add_test(rocke_library_counted_structural_gate" in cmake
    assert "add_test(rocke_library_counted_compiler_gate" in cmake
    assert "ROCKE_ISA_INSPECT_PROBE=" in cmake
    assert "ROCKE_BWD_NO_COMGR_PIN=1" in cmake
    assert "probe_isa_inspect.py" in cmake
    assert "structural\\\\;standard\\\\;comprehensive\\\\;full" in cmake
    assert "compiler\\\\;comprehensive\\\\;full" in cmake
    structural_at = categories.index("  counted-structural-standard:")
    comprehensive_at = categories.index("  comprehensive:")
    before_structural = categories[:structural_at]
    structural = categories[structural_at:comprehensive_at]
    comprehensive = categories[comprehensive_at:]
    assert "rocke_library_counted_structural_gate" not in before_structural
    assert '"rocke_library_counted_structural_gate"' in structural
    for label in ("standard", "comprehensive", "full", "structural"):
        assert f'- "{label}"' in structural
    assert "rocke_library_counted_compiler_gate" not in structural
    assert '"rocke_library_counted_compiler_gate"' in comprehensive
    assert '- "compiler"' in comprehensive
