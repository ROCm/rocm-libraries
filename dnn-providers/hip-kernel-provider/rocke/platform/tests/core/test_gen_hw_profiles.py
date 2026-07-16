"""Tests for the hardware profile supplement generator (gen_hw_profiles).

Verifies checksum validation to detect drift between Python HW_PROFILES
and generated C++ supplements.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rocke.heuristics import gen_hw_profiles as ghp


def test_checksum_computation_is_deterministic():
    """Verify the checksum is deterministic for the same data."""
    archs = list(ghp.HW_PROFILES.keys())
    checksum1 = ghp._compute_checksum(archs)
    checksum2 = ghp._compute_checksum(archs)
    assert checksum1 == checksum2
    assert len(checksum1) == 16  # First 16 hex digits


def test_checksum_changes_when_data_changes():
    """Verify checksum changes if supplement values change."""
    archs = list(ghp.HW_PROFILES.keys())
    original_checksum = ghp._compute_checksum(archs)

    # Temporarily modify one value
    original_value = ghp.HW_PROFILES["gfx942"]["hw_shader_engines"]
    ghp.HW_PROFILES["gfx942"]["hw_shader_engines"] = 999

    try:
        modified_checksum = ghp._compute_checksum(archs)
        assert modified_checksum != original_checksum
    finally:
        # Restore original value
        ghp.HW_PROFILES["gfx942"]["hw_shader_engines"] = original_value


def test_generated_checksum_matches_python_source(tmp_path):
    """Verify the generated C++ file's checksum matches Python HW_PROFILES."""
    # Generate to temp directory
    ghp.generate(tmp_path)

    # Read generated file
    generated = (tmp_path / "HardwareProfileSupplements.hpp").read_text()

    # Extract checksum from generated file
    for line in generated.split("\n"):
        if line.startswith("constexpr const char* kSupplementChecksum"):
            # Line format: constexpr const char* kSupplementChecksum = "bfd26a7ee849840f";
            embedded_checksum = line.split('"')[1]
            break
    else:
        pytest.fail("kSupplementChecksum not found in generated file")

    # Compute checksum from current Python data
    archs = list(ghp.HW_PROFILES.keys())
    expected_checksum = ghp._compute_checksum(archs)

    assert embedded_checksum == expected_checksum, (
        f"Generated checksum {embedded_checksum} doesn't match Python source {expected_checksum}. "
        "This indicates the generated file is stale or HW_PROFILES was modified without regenerating."
    )


def test_all_archs_have_required_supplement_fields():
    """Verify every arch in HW_PROFILES has all required supplement fields."""
    for arch, profile in ghp.HW_PROFILES.items():
        missing = [f for f in ghp.SUPPLEMENT_FIELDS if f not in profile]
        assert not missing, f"HW_PROFILES[{arch!r}] missing fields: {missing}"


def test_supplement_values_are_positive():
    """Verify supplement values are positive (negative would indicate error)."""
    for arch, profile in ghp.HW_PROFILES.items():
        for field in ghp.SUPPLEMENT_FIELDS:
            value = profile[field]
            # L3 cache can be 0 for some architectures (gfx1100)
            if field == "hw_l3_cache_kb":
                assert value >= 0, f"{arch}.{field} is negative: {value}"
            else:
                assert value > 0, f"{arch}.{field} must be positive, got {value}"
