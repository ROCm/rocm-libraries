# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""gfx1250 v0/v1 ASIC-revision tests.

gfx1250 ships as two silicon revisions (v0, v1) that share one ISA, arch name
and compiler target; only hipDeviceProp_t::asicRevision tells them apart
(v0 -> 0, everything else -> v1). This file covers, in order:

  * the pure revision -> --gpu-targets mapping and its probe wrapper,
  * the invoke -> CMake build wiring that carries the revision, and
  * the shipped v0 logic tree, which is separated by ScheduleName alone.

No test touches a GPU: the probe is mocked and the logic tree is read from disk.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

from unittest import mock

import pytest

from Tensile.CustomYamlLoader import load_logic_gfx_arch, load_logic_schedule_name
from Tensile import GpuRevisionTarget as gpu_rev

pytestmark = pytest.mark.unit

# The revision mapping and probe wrapper live in the packaged Tensile tree
# (Tensile/GpuRevisionTarget.py, invoke-free), so they are present in ROCm test
# artifacts and CI exercises them directly -- not skipped.
#
# The invoke -> CMake build wiring lives in hipBLASLt's tasks.py at the project
# root (unit -> Tests -> Tensile -> tensilelite -> hipblaslt). That file is
# dev-only tooling absent from packaged artifacts and needs invoke, so the wiring
# tests load it by path and skip cleanly when it (or invoke) is unavailable,
# without aborting the module.
_TENSILELITE_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_TENSILELITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TENSILELITE_ROOT))


def _load_hipblaslt_tasks():
    # Returns None instead of skipping the whole module: only the wiring tests
    # need hipBLASLt's entry point (and invoke); the rest must still run.
    try:
        spec = importlib.util.spec_from_file_location(
            "hipblaslt_tasks", _TENSILELITE_ROOT.parent / "tasks.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001
        return None


hipblaslt_tasks = _load_hipblaslt_tasks()
_needs_hipblaslt_tasks = pytest.mark.skipif(
    hipblaslt_tasks is None, reason="hipBLASLt tasks.py (or invoke) not importable"
)

GFX1250 = "gfx1250"
GFX1250V0 = "gfx1250v0"
REVISION_OPT = "-DHIPBLASLT_ASIC_REVISION"


# --------------------------------------------------------------------------- #
# The pure mapping and its probe wrapper (Tensile/GpuRevisionTarget.py).
# --------------------------------------------------------------------------- #
class TestRevisionToGpuTarget:
    """base arch + asicRevision -> Tensile --gpu-targets value."""

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


# --------------------------------------------------------------------------- #
# The invoke -> CMake build wiring (hipBLASLt's tasks.py).
# --------------------------------------------------------------------------- #
@_needs_hipblaslt_tasks
class TestTargetsIncludeGfx1250:
    """Which --architecture values can produce gfx1250, and so are worth the
    probe's hipcc compile and device open."""

    @pytest.mark.parametrize("architecture", [
        "gfx1250", "gfx942;gfx1250", "gfx1250:xnack-", "gfx1250[cu=64]", " gfx1250 ",
        # 'all' (the default) and an empty list both expand to
        # BASE_ARCHITECTURES, which contains gfx1250; disagreeing would skip the
        # probe for a build that does produce it.
        "all", "gfx942;all", "", None,
    ])
    def test_targets_that_can_produce_gfx1250(self, architecture):
        assert hipblaslt_tasks._targets_include_gfx1250(architecture)

    @pytest.mark.parametrize("architecture", [
        "gfx942", "gfx942;gfx950", "gfx1200",
        "gfx12501", "gfx1250v0",  # substring matches would make these look benign
    ])
    def test_targets_that_cannot(self, architecture):
        assert not hipblaslt_tasks._targets_include_gfx1250(architecture)


@_needs_hipblaslt_tasks
class TestAsicRevisionOption:
    """What reaches CMake. A gfx1250 build ships both revisions' trees by
    default and lets the runtime pick by asicRevision, so the build machine no
    longer decides anything and is never probed. The option is emitted even when
    its value is empty: the CMake cache variable is sticky across incremental
    builds, so an unset one would keep a directory once pinned to v0 building
    only v0."""

    def test_the_default_builds_both_trees(self):
        assert hipblaslt_tasks._asic_revision_option("gfx1250", None) == f"{REVISION_OPT}="

    def test_the_default_does_not_look_at_the_local_gpu(self):
        # The build is machine-independent by construction: nothing here may
        # reach for a device, or two CI runners produce different packages.
        assert not hasattr(hipblaslt_tasks, "_detect_asic_revision")

    def test_a_build_without_gfx1250_emits_nothing(self):
        assert hipblaslt_tasks._asic_revision_option("gfx942", None) is None

    @pytest.mark.parametrize("pinned", ["v0", "v1"])
    def test_an_explicit_revision_prunes_to_one_tree(self, pinned):
        assert hipblaslt_tasks._asic_revision_option("gfx1250", pinned) == f"{REVISION_OPT}={pinned}"

    def test_an_explicit_revision_applies_even_without_gfx1250_targets(self):
        # Cross-building: the caller decides, so a target list naming no gfx1250
        # still emits the option (to keep the cache from going stale).
        assert hipblaslt_tasks._asic_revision_option("gfx942", "v0") == f"{REVISION_OPT}=v0"

    @pytest.mark.parametrize("bogus", ["0", "v2", "V0", "gfx1250v0"])
    def test_an_unrecognized_revision_is_rejected(self, bogus):
        # Anything else reaches CMake as a comparison matching neither branch,
        # which would quietly build the wrong set of trees.
        with pytest.raises(SystemExit) as exit_info:
            hipblaslt_tasks._asic_revision_option("gfx1250", bogus)
        assert exit_info.value.code == 2


@_needs_hipblaslt_tasks
class TestTheBuildSaysWhichRevisionItChose:
    """The log line is the only record of which trees an install actually got;
    the failure it guards against is a v0 part finding no library of its own."""

    def _option(self, capsys, architecture, pinned):
        hipblaslt_tasks._asic_revision_option(architecture, pinned)
        return capsys.readouterr().out

    def test_the_default_says_both_and_who_decides(self, capsys):
        out = self._option(capsys, "gfx1250", None)
        assert "gfx1250 ASIC revision: both" in out
        assert "runtime selects by asicRevision" in out

    def test_a_pinned_revision_says_it_was_pinned(self, capsys):
        out = self._option(capsys, "gfx1250", "v1")
        assert "gfx1250 ASIC revision: v1" in out
        assert "pinned by --asic-revision" in out
        assert "no gfx1250" not in out  # that caveat is for gfx1250-free builds

    def test_pinning_v1_does_not_read_as_the_default(self, capsys):
        # Pruning to the v1 tree leaves v0 silicon with no library of its
        # own, so it must not be reported the same way the both-trees default is.
        out = self._option(capsys, "gfx1250", "v1")
        assert "both" not in out
        assert "asicRevision" not in out

    def test_a_pinned_revision_without_gfx1250_targets_says_so(self, capsys):
        out = self._option(capsys, "gfx942", "v0")
        assert "gfx1250 ASIC revision: v0" in out
        assert "these targets contain no gfx1250" in out

    def test_a_build_that_cannot_produce_gfx1250_says_nothing(self, capsys):
        assert "ASIC revision" not in self._option(capsys, "gfx942", None)


@_needs_hipblaslt_tasks
class TestBuildTaskCommandLine:
    """invoke assigns short flags in signature order, so a new parameter's
    position is part of the interface: placed too early it steals a letter."""

    def _short_flags(self, flag):
        from invoke.parser import Context as ParserContext

        context = ParserContext(name="build", args=hipblaslt_tasks.build.get_arguments())
        return context.flags[flag].nicknames

    @pytest.mark.parametrize("flag,short", [
        ("--logic-filter", "f"),  # the first casualty if -g is stolen
        ("--gprof", "g"),
        ("--architecture", "a"),
    ])
    def test_existing_short_flags_are_unchanged(self, flag, short):
        assert self._short_flags(flag) == (short,)

    def test_the_revision_option_takes_no_letter(self):
        assert self._short_flags("--asic-revision") == ()


# --------------------------------------------------------------------------- #
# The shipped v0 logic tree. TensileCreateLibrary globs one tree and separates
# the two revisions by ScheduleName alone: both declare ArchitectureName:
# gfx1250 and the runtime resolves both to library/gfx1250/. A mis-tagged file
# fails silently -- dropped from v0, or leaked into every v1 build -- so the
# invariant is checked against the tree that actually ships.
# --------------------------------------------------------------------------- #
_LOGIC_ROOT = (
    _TENSILELITE_ROOT.parent
    / "library" / "src" / "amd_detail" / "rocblaslt" / "src"
    / "Tensile" / "Logic" / "asm_full"
)
_OVERLAY_ROOT = _LOGIC_ROOT / GFX1250V0

_needs_logic_dir = pytest.mark.xfail(
    not _LOGIC_ROOT.is_dir(),
    reason="Logic files not found: https://github.com/ROCm/rocm-libraries/issues/7481",
)


def _logic_root():
    # Asserted, not skipped: the comprehensions below pass vacuously over a
    # missing tree, and xfail_strict would turn that pass into a failure.
    assert _LOGIC_ROOT.is_dir(), f"Logic root not found: {_LOGIC_ROOT}"
    return _LOGIC_ROOT


def _overlay_files():
    return sorted((_logic_root() / GFX1250V0).rglob("*.yaml"))


@_needs_logic_dir
def test_the_overlay_ships_logic():
    # An empty overlay is a broken state: a v0 build reports success having
    # written a library with no solutions in it.
    assert _overlay_files()


@_needs_logic_dir
def test_every_overlay_file_declares_the_asic_revision_schedule_name():
    offenders = {
        str(p.relative_to(_LOGIC_ROOT)): load_logic_schedule_name(p)
        for p in _overlay_files()
        if load_logic_schedule_name(p) != GFX1250V0
    }
    assert offenders == {}


@_needs_logic_dir
def test_every_overlay_file_keeps_the_base_architecture_name():
    # ArchitectureName keys the master library and must stay the arch:
    # TensileCreateLibrary rejects a stepping there, and library/gfx1250v0/ is a
    # directory the runtime never reads.
    offenders = {
        str(p.relative_to(_LOGIC_ROOT)): load_logic_gfx_arch(p)
        for p in _overlay_files()
        if load_logic_gfx_arch(p) != GFX1250
    }
    assert offenders == {}


@_needs_logic_dir
def test_no_logic_outside_the_overlay_claims_the_asic_revision():
    offenders = [
        str(p.relative_to(_LOGIC_ROOT))
        for p in sorted(_logic_root().rglob("*.yaml"))
        if not p.is_relative_to(_OVERLAY_ROOT)
        and load_logic_schedule_name(p) == GFX1250V0
    ]
    assert offenders == []
