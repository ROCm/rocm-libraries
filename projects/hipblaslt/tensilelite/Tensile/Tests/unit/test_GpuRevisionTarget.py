# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx1250 v0/v1 ASIC-revision -> --gpu-targets mapping.

gfx1250 ships as two silicon revisions (v0 and v1) that share the same ISA/arch
name. Only hipDeviceProp_t::asicRevision distinguishes them (empirically v0 -> 0,
v1 -> 1). These tests pin the pure mapping and the detection wrapper's fallback
behavior; they never touch a real GPU (the probe is mocked).
"""

import pathlib
import subprocess
import sys

from unittest import mock

import pytest

# tasks.py lives at the tensilelite root (unit -> Tests -> Tensile -> tensilelite).
#
# tensilelite/tasks.py is dev-only tooling that is NOT shipped in ROCm test
# artifacts (the packaged tree under build/share/.../tensilelite has no tasks.py).
# Skip the whole module when it cannot be imported so the packaged test run is
# not aborted at collection; a source checkout still exercises every test.
_TENSILELITE_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_TENSILELITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TENSILELITE_ROOT))

try:
    import tasks  # noqa: E402  (tensilelite/tasks.py)
except ImportError:
    pytest.skip(
        "tensilelite/tasks.py is dev-only tooling and is absent from packaged "
        "test artifacts; GpuRevisionTarget tests require a source checkout.",
        allow_module_level=True,
    )


class TestRevisionToGpuTarget:
    """The pure mapping: base arch + asicRevision -> Tensile --gpu-targets value."""

    def test_gfx1250_rev0_is_v0(self):
        assert tasks._revision_to_gpu_target("gfx1250", 0) == "gfx1250v0"

    def test_gfx1250_rev1_is_plain_v1(self):
        assert tasks._revision_to_gpu_target("gfx1250", 1) == "gfx1250"

    def test_gfx1250_unknown_revision_defaults_to_v1(self):
        # -1 == HIP too old to expose the field; must not be mistaken for v0.
        assert tasks._revision_to_gpu_target("gfx1250", -1) == "gfx1250"

    def test_gfx1250_future_revision_defaults_to_v1(self):
        assert tasks._revision_to_gpu_target("gfx1250", 2) == "gfx1250"

    def test_non_gfx1250_arch_is_unchanged_even_at_rev0(self):
        # revision 0 only means v0 for gfx1250; other arches are returned as-is.
        assert tasks._revision_to_gpu_target("gfx942", 0) == "gfx942"

    def test_none_base_arch_is_passed_through(self):
        assert tasks._revision_to_gpu_target(None, 0) is None


class TestDetectGpuRevisionTarget:
    """The wrapper: detect base arch, probe only for gfx1250, fall back to v1."""

    def test_non_gfx1250_skips_probe(self):
        with mock.patch.object(tasks, "detect_gpu_arch", return_value="gfx942"), \
             mock.patch.object(tasks, "_probe_asic_revision") as probe:
            assert tasks.detect_gpu_revision_target() == "gfx942"
            probe.assert_not_called()

    def test_gfx1250_rev0_selects_v0(self):
        with mock.patch.object(tasks, "detect_gpu_arch", return_value="gfx1250"), \
             mock.patch.object(tasks, "_probe_asic_revision", return_value=("gfx1250", 0)) as probe:
            assert tasks.detect_gpu_revision_target() == "gfx1250v0"
            probe.assert_called_once()

    def test_gfx1250_rev0_with_feature_suffix_selects_v0(self):
        # Real hardware reports gcnArchName with feature suffixes; the base token
        # must still be recognized as gfx1250 so v0 detection is not dead.
        with mock.patch.object(tasks, "detect_gpu_arch", return_value="gfx1250"), \
             mock.patch.object(tasks, "_probe_asic_revision",
                               return_value=("gfx1250:sramecc+:xnack-", 0)):
            assert tasks.detect_gpu_revision_target() == "gfx1250v0"

    def test_gfx1250_rev1_selects_v1(self):
        with mock.patch.object(tasks, "detect_gpu_arch", return_value="gfx1250"), \
             mock.patch.object(tasks, "_probe_asic_revision", return_value=("gfx1250", 1)) as probe:
            assert tasks.detect_gpu_revision_target() == "gfx1250"
            probe.assert_called_once()

    def test_probe_failure_defaults_to_v1(self):
        with mock.patch.object(tasks, "detect_gpu_arch", return_value="gfx1250"), \
             mock.patch.object(tasks, "_probe_asic_revision", return_value=None):
            assert tasks.detect_gpu_revision_target() == "gfx1250"

    def test_probe_arch_mismatch_defaults_to_v1(self):
        # If the probe's own arch view disagrees, don't trust its revision.
        with mock.patch.object(tasks, "detect_gpu_arch", return_value="gfx1250"), \
             mock.patch.object(tasks, "_probe_asic_revision", return_value=("gfx1250x", 0)):
            assert tasks.detect_gpu_revision_target() == "gfx1250"

    def test_none_base_arch_is_passed_through(self):
        with mock.patch.object(tasks, "detect_gpu_arch", return_value=None), \
             mock.patch.object(tasks, "_probe_asic_revision") as probe:
            assert tasks.detect_gpu_revision_target() is None
            probe.assert_not_called()


def _completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class TestProbeAsicRevision:
    """The HIP probe wrapper: compile-on-demand + parse, never raises."""

    def test_hipcc_missing_returns_none(self):
        with mock.patch.object(tasks.shutil, "which", return_value=None):
            assert tasks._probe_asic_revision() is None

    def _fresh_probe(self, tmp_path):
        # Pre-create an up-to-date binary so the staleness check skips the
        # compile branch, leaving only the probe-run subprocess call to mock.
        binp = tmp_path / "gpu_revision_probe"
        binp.write_text("")
        return binp

    def test_success_parses_arch_and_revision(self, tmp_path):
        self._fresh_probe(tmp_path)
        with mock.patch.object(tasks.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(tasks.subprocess, "run",
                               return_value=_completed("gfx1250:xnack-\n0\n")) as run:
            assert tasks._probe_asic_revision(build_dir=str(tmp_path)) == ("gfx1250:xnack-", 0)
            run.assert_called_once()  # no recompile, just the probe run

    def test_nonzero_exit_returns_none(self, tmp_path):
        self._fresh_probe(tmp_path)
        with mock.patch.object(tasks.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(tasks.subprocess, "run",
                               return_value=_completed("", returncode=1, stderr="no device")):
            assert tasks._probe_asic_revision(build_dir=str(tmp_path)) is None

    def test_short_output_returns_none(self, tmp_path):
        self._fresh_probe(tmp_path)
        with mock.patch.object(tasks.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(tasks.subprocess, "run",
                               return_value=_completed("gfx1250\n")):
            assert tasks._probe_asic_revision(build_dir=str(tmp_path)) is None

    def test_unparsable_revision_returns_none(self, tmp_path):
        self._fresh_probe(tmp_path)
        with mock.patch.object(tasks.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(tasks.subprocess, "run",
                               return_value=_completed("gfx1250\nNaN\n")):
            assert tasks._probe_asic_revision(build_dir=str(tmp_path)) is None

    def test_compile_failure_returns_none(self, tmp_path):
        # No pre-existing binary -> stale -> compile branch runs and fails.
        with mock.patch.object(tasks.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(tasks.subprocess, "run",
                               side_effect=subprocess.CalledProcessError(1, "hipcc")):
            assert tasks._probe_asic_revision(build_dir=str(tmp_path)) is None

    def test_run_oserror_returns_none(self, tmp_path):
        self._fresh_probe(tmp_path)
        with mock.patch.object(tasks.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(tasks.subprocess, "run", side_effect=OSError("exec fail")):
            assert tasks._probe_asic_revision(build_dir=str(tmp_path)) is None
