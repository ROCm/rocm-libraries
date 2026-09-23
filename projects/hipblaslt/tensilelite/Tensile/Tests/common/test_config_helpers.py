# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for config_helpers.configMarks FFM-conditional xfail.

A config marked ``ffm_fail`` passes on real hardware but fails only under
FFM emulation. configMarks turns that mark into an ``xfail`` exclusively
when running under FFM — keyed on the emulator's ``HSA_MODEL_MEMFILE``
backing plus the gfx1250 arch — so it stays inert on hardware and on
other arches, where the test must still run.
"""

import os

import pytest

from config_helpers import configMarks, findAvailableArchs

# These validate the gfx1250 xfail/ffm marking logic in-process, so tag them into the gfx1250 arch suite (-m gfx1250) alongside the
# auto-added `common` mark, keeping them collected wherever gfx1250 marking is
# exercised.
pytestmark = pytest.mark.gfx1250

# configMarks takes rootDir only to compute the config's relpath (for the
# directory-name marks); the four gfx1250 configs live under Tensile/Tests.
_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_COMMON_DIR)

# A config tagged ``ffm_fail`` and a gfx1250 config that is not.
_FFM_FAIL_CONFIG = os.path.join(_COMMON_DIR, "gemm", "gfx12", "tdm_multicast_gfx1250.yaml")
_PLAIN_GFX1250_CONFIG = os.path.join(
    _COMMON_DIR, "streamk", "gfx1250", "core", "sk_mxf4_force_dp_only.yaml"
)
# A config tagged ``skip-gfx1250v0``.
_SKIP_GFX1250V0_CONFIG = os.path.join(
    _COMMON_DIR, "streamk", "gfx1250", "sk_mxf4gemm_tdm_ext.yaml"
)
# A config tagged ``skip-gfx1250`` (base arch).
_SKIP_GFX1250_CONFIG = os.path.join(_COMMON_DIR, "comm", "gfx950", "fused_a2a.yaml")

_FFM_MEMFILE = "/dev/shm/hsakmt_model_root_test"


def test_ffm_fail_xfails_under_ffm(monkeypatch):
    """memfile set + gfx1250 available + ffm_fail marked -> xfail added."""
    monkeypatch.setenv("HSA_MODEL_MEMFILE", _FFM_MEMFILE)
    marks = configMarks(_FFM_FAIL_CONFIG, _TESTS_ROOT, ["gfx1250"])
    assert pytest.mark.xfail in marks


def test_ffm_fail_inert_on_hardware(monkeypatch):
    """No memfile (real hardware) -> the ffm_fail config still runs."""
    monkeypatch.delenv("HSA_MODEL_MEMFILE", raising=False)
    marks = configMarks(_FFM_FAIL_CONFIG, _TESTS_ROOT, ["gfx1250"])
    assert pytest.mark.xfail not in marks


def test_ffm_fail_inert_on_other_arch(monkeypatch):
    """Under emulation but not gfx1250 -> the ffm_fail config still runs."""
    monkeypatch.setenv("HSA_MODEL_MEMFILE", _FFM_MEMFILE)
    marks = configMarks(_FFM_FAIL_CONFIG, _TESTS_ROOT, ["gfx942"])
    assert pytest.mark.xfail not in marks


# The chosen config lives under a ``core/`` dir; configMarks derives a mark
# from every path component, and ``core`` is intentionally unregistered — the
# resulting PytestUnknownMarkWarning is pre-existing repo behavior, not a
# defect in this test, so scope it out here.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")
def test_unmarked_config_never_xfails_under_ffm(monkeypatch):
    """A gfx1250 config without ffm_fail is untouched even under FFM."""
    monkeypatch.setenv("HSA_MODEL_MEMFILE", _FFM_MEMFILE)
    marks = configMarks(_PLAIN_GFX1250_CONFIG, _TESTS_ROOT, ["gfx1250"])
    assert pytest.mark.xfail not in marks


def test_find_available_archs_expands_versioned_target():
    """A versioned target yields BOTH the versioned string and its base arch, so
    configMarks matches gfx1250v0 marks AND base gfx1250 marks. A plain arch is
    returned unchanged."""
    assert findAvailableArchs("gfx1250v0") == ["gfx1250v0", "gfx1250"]
    assert findAvailableArchs("gfx942") == ["gfx942"]
    assert findAvailableArchs("gfx1250v0;gfx942") == ["gfx1250v0", "gfx1250", "gfx942"]


def test_skip_gfx1250v0_fires_on_v0_target():
    """A skip-gfx1250v0 config is skipped on a gfx1250v0 target."""
    archs = findAvailableArchs("gfx1250v0")  # -> ["gfx1250v0", "gfx1250"]
    marks = configMarks(_SKIP_GFX1250V0_CONFIG, _TESTS_ROOT, archs)
    assert pytest.mark.skip in marks


def test_skip_gfx1250v0_inert_on_base_target():
    """A skip-gfx1250v0 config runs on a base gfx1250 target — the gfx1250v0 skip stays inert."""
    archs = findAvailableArchs("gfx1250")  # -> ["gfx1250"]
    marks = configMarks(_SKIP_GFX1250V0_CONFIG, _TESTS_ROOT, archs)
    assert pytest.mark.skip not in marks


def test_base_skip_gfx1250_fires_on_v0_target():
    """A base skip-gfx1250 config is still skipped on a gfx1250v0 target — base marks
    must fire on a versioned target too."""
    archs = findAvailableArchs("gfx1250v0")  # -> ["gfx1250v0", "gfx1250"]
    marks = configMarks(_SKIP_GFX1250_CONFIG, _TESTS_ROOT, archs)
    assert pytest.mark.skip in marks
