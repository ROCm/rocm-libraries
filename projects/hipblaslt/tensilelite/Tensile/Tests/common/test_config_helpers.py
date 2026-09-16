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

from config_helpers import configMarks

# configMarks takes rootDir only to compute the config's relpath (for the
# directory-name marks); the four gfx1250 configs live under Tensile/Tests.
_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_COMMON_DIR)

# A config tagged ``ffm_fail`` and a gfx1250 config that is not.
_FFM_FAIL_CONFIG = os.path.join(_COMMON_DIR, "gemm", "gfx12", "tdm_multicast_gfx1250.yaml")
_PLAIN_GFX1250_CONFIG = os.path.join(
    _COMMON_DIR, "streamk", "gfx1250", "core", "sk_mxf4_force_dp_only.yaml"
)

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
