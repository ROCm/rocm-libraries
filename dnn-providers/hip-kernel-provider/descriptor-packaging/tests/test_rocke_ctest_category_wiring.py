# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static contracts for rocKE's pre-registered CTest category wiring."""

from pathlib import Path

_PROVIDER = Path(__file__).resolve().parents[2]
_PROVIDER_CMAKE = _PROVIDER / "src" / "CMakeLists.txt"
_ROCKE_CMAKE = _PROVIDER / "rocke" / "platform" / "CMakeLists.txt"
_ROCKE_CATEGORIES = _PROVIDER / "ROCKE_ENGINE_test_categories_external.yaml"


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


def test_counted_gates_have_installed_and_tier_owners():
    cmake = _ROCKE_CMAKE.read_text(encoding="utf-8")
    categories = _ROCKE_CATEGORIES.read_text(encoding="utf-8")
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
