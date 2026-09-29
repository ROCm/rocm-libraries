# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""gfx1250 v0/v1 ASIC-revision tests.

gfx1250 ships as two silicon revisions (v0, v1) that share one ISA, arch name
and compiler target; only hipDeviceProp_t::asicRevision tells them apart
(v0 -> 0, everything else -> v1). This file covers:

  * the pure revision -> --gpu-targets mapping, and
  * its probe wrapper.

No test touches a GPU: the probe is mocked.
"""

import os
import subprocess

from unittest import mock

import pytest

from tensilelite import GpuRevisionTarget as gpu_rev

pytestmark = pytest.mark.unit

# The revision mapping and probe wrapper live in the packaged TensileLite tree
# (tensilelite/GpuRevisionTarget.py, invoke-free), so they are present in ROCm test
# artifacts and CI exercises them directly -- not skipped.


# --------------------------------------------------------------------------- #
# The pure mapping and its probe wrapper (tensilelite/GpuRevisionTarget.py).
# --------------------------------------------------------------------------- #
class TestRevisionToGpuTarget:
    """base arch + asicRevision -> TensileLite --gpu-targets value."""

    @pytest.mark.parametrize("arch,revision,expected", [
        ("gfx1250", 0, "gfx1250v0"),   # the only v0 case
        ("gfx1250", 1, "gfx1250"),
        ("gfx1250", -1, "gfx1250"),    # HIP too old to expose the field
        ("gfx1250", 2, "gfx1250"),     # a revision this mapping has not seen
        ("gfx942", 0, "gfx942"),       # revision 0 means v0 only for gfx1250
        (None, 0, None),
    ])
    def test_mapping(self, arch, revision, expected):
        assert gpu_rev._revision_to_gpu_target(arch, revision) == expected


class TestDetectGpuRevisionTarget:
    """The wrapper: detect the arch, probe only for gfx1250, fall back to v1."""

    def _detect(self, arch, probe_result):
        with mock.patch.object(gpu_rev, "detect_gpu_arch", return_value=arch), \
             mock.patch.object(gpu_rev, "_probe_asic_revision", return_value=probe_result) as probe:
            return gpu_rev.detect_gpu_revision_target(), probe

    def test_non_gfx1250_skips_probe(self):
        target, probe = self._detect("gfx942", None)
        assert target == "gfx942"
        probe.assert_not_called()

    def test_none_arch_skips_probe(self):
        target, probe = self._detect(None, None)
        assert target is None
        probe.assert_not_called()

    def test_rev0_selects_v0(self):
        target, probe = self._detect("gfx1250", ("gfx1250", 0))
        assert target == "gfx1250v0"
        probe.assert_called_once()

    def test_rev0_with_feature_suffix_selects_v0(self):
        # Real hardware reports gcnArchName with suffixes; the base token must
        # still be recognized or v0 detection is dead.
        target, _ = self._detect("gfx1250", ("gfx1250:sramecc+:xnack-", 0))
        assert target == "gfx1250v0"

    @pytest.mark.parametrize("probe_result", [
        ("gfx1250", 1),        # a confirmed non-v0 part
        None,                  # probe could not run
        ("gfx1250x", 0),       # probe's own arch view disagrees; distrust it
    ])
    def test_anything_but_rev0_is_v1(self, probe_result):
        assert self._detect("gfx1250", probe_result)[0] == "gfx1250"

    @pytest.mark.parametrize("revision,target", [(2, "gfx1250"), (0, "gfx1250v0")])
    def test_the_probed_revision_number_is_reported(self, capsys, revision, target):
        # Everything but 0 maps to v1, so the raw number is the only thing that
        # separates a confirmed part from one reporting an unseen value (a
        # gfx1250 in the functional model reports 2).
        with mock.patch.object(gpu_rev, "detect_gpu_arch", return_value="gfx1250"), \
             mock.patch.object(gpu_rev, "_probe_asic_revision", return_value=("gfx1250", revision)):
            assert gpu_rev.detect_gpu_revision_target() == target
        # Reported on stderr so stdout stays a clean, capturable target value
        # (callers do TENSILE_TARGET=$(invoke get-gpu-revision-target)).
        captured = capsys.readouterr()
        assert str(revision) in captured.err
        assert captured.out == ""


def _completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class TestProbeAsicRevision:
    """The HIP probe wrapper: compile-on-demand + parse, never raises."""

    def _fresh_probe(self, tmp_path):
        # A fake probe source plus an up-to-date binary make the staleness check
        # skip the compile, leaving only the probe-run subprocess to mock. The
        # source is faked (and patched into gpu_rev via _REVISION_PROBE_SRC by the
        # callers) so these tests stay hermetic in packaged / sparse-checkout
        # artifacts, where tensilelite/tools/gpu_revision_probe.cpp is not shipped
        # and the real _REVISION_PROBE_SRC.stat() would otherwise raise.
        src = tmp_path / "gpu_revision_probe.cpp"
        src.write_text("// fake probe source\n")
        binary = tmp_path / "gpu_revision_probe"
        binary.write_text("")
        os.utime(src, (0, 0))  # force the source older than the fresh binary
        return src

    def test_hipcc_missing_returns_none(self):
        with mock.patch.object(gpu_rev.shutil, "which", return_value=None):
            assert gpu_rev._probe_asic_revision() is None

    def test_success_parses_arch_and_revision(self, tmp_path):
        src = self._fresh_probe(tmp_path)
        with mock.patch.object(gpu_rev, "_REVISION_PROBE_SRC", src), \
             mock.patch.object(gpu_rev.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(gpu_rev.subprocess, "run",
                               return_value=_completed("gfx1250:xnack-\n0\n")) as run:
            assert gpu_rev._probe_asic_revision(build_dir=str(tmp_path)) == ("gfx1250:xnack-", 0)
            run.assert_called_once()  # no recompile, just the probe run

    @pytest.mark.parametrize("run_kwargs", [
        {"return_value": _completed("", returncode=1, stderr="no device")},
        {"return_value": _completed("gfx1250\n")},          # too few lines
        {"return_value": _completed("gfx1250\nNaN\n")},     # unparsable revision
        {"side_effect": OSError("exec fail")},
    ])
    def test_probe_run_failures_return_none(self, tmp_path, run_kwargs):
        src = self._fresh_probe(tmp_path)
        with mock.patch.object(gpu_rev, "_REVISION_PROBE_SRC", src), \
             mock.patch.object(gpu_rev.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(gpu_rev.subprocess, "run", **run_kwargs):
            assert gpu_rev._probe_asic_revision(build_dir=str(tmp_path)) is None

    def test_compile_failure_returns_none(self, tmp_path):
        # No pre-existing binary -> stale -> the compile branch runs and fails.
        with mock.patch.object(gpu_rev.shutil, "which", return_value="/usr/bin/hipcc"), \
             mock.patch.object(gpu_rev.subprocess, "run",
                               side_effect=subprocess.CalledProcessError(1, "hipcc")):
            assert gpu_rev._probe_asic_revision(build_dir=str(tmp_path)) is None
