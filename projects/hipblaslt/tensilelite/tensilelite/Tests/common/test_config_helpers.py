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

from config_helpers import _usingStinkytofuBackend, configMarks, findAvailableArchs

# These validate the gfx1250 xfail/ffm marking logic in-process, so tag them into the gfx1250 arch suite (-m gfx1250) alongside the
# auto-added `common` mark, keeping them collected wherever gfx1250 marking is
# exercised.
pytestmark = pytest.mark.gfx1250

# configMarks takes rootDir only to compute the config's relpath (for the
# directory-name marks); the real gfx1250 configs below live under tensilelite/Tests.
_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_COMMON_DIR)

# A gfx1250 config that is not tagged ``ffm_fail``.
_PLAIN_GFX1250_CONFIG = os.path.join(
    _COMMON_DIR, "streamk", "gfx1250", "core", "data_parallel_static_mxf4.yaml"
)
# A base gfx1250 config tagged ``skip-gfx1250-strict``.
_SKIP_GFX1250_STRICT_CONFIG = os.path.join(
    _COMMON_DIR, "streamk", "gfx1250", "sk_mxf4gemm_tdm_ext.yaml"
)
# A gfx1250-strict config, tagged ``skip-gfx1250`` but not ``skip-gfx1250-strict``.
_STRICT_ONLY_CONFIG = os.path.join(_COMMON_DIR, "gemm", "gfx12", "bf16_gfx1250-strict.yaml")

# A gfx1250 config tagged ``skip-stinkytofu`` (subtile, not yet supported on the
# stinkytofu backend). On a base gfx1250 target none of its arch-skip marks fire,
# so skip-stinkytofu is the only thing that can add a skip -- which isolates the
# backend gating below.
_SKIP_STINKYTOFU_CONFIG = os.path.join(
    _COMMON_DIR, "gemm", "gfx12", "subtile_bf16_gfx1250_cluster.yaml"
)

# The reason configMarks attaches to the skip it adds for skip-stinkytofu.
_STINKYTOFU_SKIP_REASON = "Not yet supported in stinkytofu backend"

_FFM_MEMFILE = "/dev/shm/hsakmt_model_root_test"


@pytest.fixture
def ffm_fail_config(tmp_path):
    """A minimal gfx1250 config tagged ``ffm_fail``.

    Synthetic, so the tests don't depend on which real configs currently carry
    the mark: it comes off as soon as the emulator is fixed.
    """
    config = tmp_path / "ffm_fail_gfx1250.yaml"
    config.write_text(
        "TestParameters:\n"
        "  marks: [ffm_fail]\n"
        "GlobalParameters:\n"
        "  Architecture: gfx1250\n"
        "BenchmarkProblems:\n"
        "  - - OperationType: GEMM\n"
        "      DataType: s\n"
    )
    return str(config), str(tmp_path)


def test_ffm_fail_xfails_under_ffm(monkeypatch, ffm_fail_config):
    """memfile set + gfx1250 available + ffm_fail marked -> xfail added."""
    monkeypatch.setenv("HSA_MODEL_MEMFILE", _FFM_MEMFILE)
    marks = configMarks(*ffm_fail_config, ["gfx1250"])
    assert pytest.mark.xfail in marks


def test_ffm_fail_inert_on_hardware(monkeypatch, ffm_fail_config):
    """No memfile (real hardware) -> the ffm_fail config still runs."""
    monkeypatch.delenv("HSA_MODEL_MEMFILE", raising=False)
    marks = configMarks(*ffm_fail_config, ["gfx1250"])
    assert pytest.mark.xfail not in marks


def test_ffm_fail_inert_on_other_arch(monkeypatch, ffm_fail_config):
    """Under emulation but not gfx1250 -> the ffm_fail config still runs."""
    monkeypatch.setenv("HSA_MODEL_MEMFILE", _FFM_MEMFILE)
    marks = configMarks(*ffm_fail_config, ["gfx942"])
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


def test_find_available_archs_keeps_stepping_name():
    """A stepping is its own architecture: its name passes through whole, and
    does not also bring in its base arch."""
    assert findAvailableArchs("gfx1250-strict") == ["gfx1250-strict"]
    assert findAvailableArchs("gfx942") == ["gfx942"]
    assert findAvailableArchs("gfx1250-strict;gfx942") == ["gfx1250-strict", "gfx942"]


def test_find_available_archs_does_not_map_the_retired_v0_name_to_the_base():
    """gfx1250v0 was A0 silicon, now gfx1250-strict. Normalizing it to gfx1250
    would select the base stepping's configs for A0; kept whole, it names no
    architecture and fails loudly instead."""
    assert findAvailableArchs("gfx1250v0") == ["gfx1250v0"]


def test_skip_gfx1250_strict_fires_on_strict_target():
    """A skip-gfx1250-strict config is skipped on a gfx1250-strict target."""
    archs = findAvailableArchs("gfx1250-strict")
    marks = configMarks(_SKIP_GFX1250_STRICT_CONFIG, _TESTS_ROOT, archs)
    assert pytest.mark.skip in marks


def test_skip_gfx1250_strict_inert_on_base_target():
    """A skip-gfx1250-strict config runs on a base gfx1250 target."""
    archs = findAvailableArchs("gfx1250")
    marks = configMarks(_SKIP_GFX1250_STRICT_CONFIG, _TESTS_ROOT, archs)
    assert pytest.mark.skip not in marks


def test_base_skip_gfx1250_inert_on_strict_target():
    """A base skip-gfx1250 mark does not fire on a gfx1250-strict target. The two
    spellings carry mirrored marks, so inheriting the base's would skip every
    config written for the stepping."""
    strict = configMarks(_STRICT_ONLY_CONFIG, _TESTS_ROOT, findAvailableArchs("gfx1250-strict"))
    base = configMarks(_STRICT_ONLY_CONFIG, _TESTS_ROOT, findAvailableArchs("gfx1250"))
    assert pytest.mark.skip not in strict
    assert pytest.mark.skip in base


# --- skip-stinkytofu backend gating ---------------------------------------
#
# configMarks skips a skip-stinkytofu config whenever its kernels would be built
# on the stinkytofu backend. _usingStinkytofuBackend resolves that backend from
# (in order) ROCISA_BACKEND, the gfx1250 target, and whether rocisa has swapped
# in the stinkytofu adaptor. That last signal is process-global, so the fixture
# below pins rocisa.__file__ with monkeypatch to keep these in-process tests
# independent of the backend rocisa actually selected for the test process.


@pytest.fixture
def stinkytofu_backend(monkeypatch):
    """Pin _usingStinkytofuBackend's view of the loaded rocisa module.

    Returns a setter ``set_backend(adaptor=)`` that overrides ``rocisa.__file__``
    to look like the adaptor shim or the native module, so the unset-env +
    gfx1250 branch is deterministic regardless of the real backend.
    """
    import rocisa

    def set_backend(*, adaptor):
        monkeypatch.setattr(
            rocisa,
            "__file__",
            "/pkg/rocisa_stinkytofu_adaptor/__init__.py" if adaptor else "/pkg/rocisa/__init__.py",
            raising=False,
        )

    return set_backend


def _has_stinkytofu_skip(marks):
    """Whether configMarks added its skip-stinkytofu skip (matched by reason).

    The arch-skip path adds a bare ``pytest.mark.skip``; the skip-stinkytofu path
    adds ``pytest.mark.skip(reason=...)``, so match on the reason rather than on
    mark identity to tell the two apart.
    """
    return any(
        getattr(m, "name", None) == "skip"
        and getattr(m, "kwargs", {}).get("reason") == _STINKYTOFU_SKIP_REASON
        for m in marks
    )


def test_using_stinkytofu_env_rocisa_forces_native(monkeypatch):
    """ROCISA_BACKEND=rocisa -> native even on a gfx1250 target."""
    monkeypatch.setenv("ROCISA_BACKEND", "rocisa")
    assert _usingStinkytofuBackend(["gfx1250"]) is False


def test_using_stinkytofu_env_stinkytofu_ignores_arch(monkeypatch):
    """ROCISA_BACKEND=stinkytofu -> stinkytofu even on a non-gfx1250 target."""
    monkeypatch.setenv("ROCISA_BACKEND", "stinkytofu")
    assert _usingStinkytofuBackend(["gfx950"]) is True


def test_using_stinkytofu_unset_non_gfx1250_is_native(monkeypatch):
    """Unset env + no gfx1250 target -> native; only gfx1250 routes to stinkytofu."""
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    assert _usingStinkytofuBackend(["gfx950"]) is False


def test_using_stinkytofu_unset_gfx1250_adaptor_loaded(monkeypatch, stinkytofu_backend):
    """Unset env + gfx1250 target + adaptor already loaded -> stinkytofu."""
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    stinkytofu_backend(adaptor=True)
    assert _usingStinkytofuBackend(["gfx1250"]) is True


def test_using_stinkytofu_unset_gfx1250_native_fallback(monkeypatch, stinkytofu_backend):
    """Unset env + gfx1250 target + adaptor not loaded (stinkytofu unavailable, so
    rocisa stayed on native) -> NOT stinkytofu. Resolved from the loaded module,
    so no second nanobind binding is imported to find this out."""
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    stinkytofu_backend(adaptor=False)
    assert _usingStinkytofuBackend(["gfx1250"]) is False


def test_skip_stinkytofu_fires_when_autoselected(monkeypatch, stinkytofu_backend):
    """The regression this change fixes: a skip-stinkytofu config is skipped when
    stinkytofu is auto-selected (ROCISA_BACKEND unset) on a gfx1250 target, not
    only when the backend is requested explicitly."""
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    stinkytofu_backend(adaptor=True)
    marks = configMarks(_SKIP_STINKYTOFU_CONFIG, _TESTS_ROOT, ["gfx1250"])
    assert _has_stinkytofu_skip(marks)


def test_skip_stinkytofu_inert_on_native_fallback(monkeypatch, stinkytofu_backend):
    """A skip-stinkytofu config still runs when stinkytofu is unavailable and
    rocisa has stayed on native."""
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    stinkytofu_backend(adaptor=False)
    marks = configMarks(_SKIP_STINKYTOFU_CONFIG, _TESTS_ROOT, ["gfx1250"])
    assert not _has_stinkytofu_skip(marks)
