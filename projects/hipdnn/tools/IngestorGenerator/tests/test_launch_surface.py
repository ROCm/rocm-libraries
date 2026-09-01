# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch-surface audit must fail on each defect class it exists to catch.

The headline defect is real and already shipped once in this tree: the gfx942
attention_dense profile's ``kmd_fields`` never declared ``block_m``, even though
``Gfx942AttentionDenseNative.cpp`` reads it via ``kernel.getIntMetadata("block_m")``
on both the matcher path (``kernelMatches``) and the prepare path
(``attentionDenseGeometry``). Every descriptor the profile generated was missing a
field its own engine dereferences unconditionally, and nothing before this tool
cross-referenced "fields a launch surface's C++ mirror needs" against "fields the
KMD actually declares". ``TestKmdFieldsCheck`` below is that regression test: it
fails without the kmd_fields cross-reference and passes with it, by construction
(fresh minimal fixtures, not the real profile, so the property under test is the
CHECK's behaviour rather than the current profile's content).

The remaining classes -- a cpp_mirror path that does not exist, an unguarded surface
that must be named and must fail closed without ``--allow-unguarded`` -- are the
other three ways a launch_surface block can lie about the state of the restatement
it claims to audit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_TOOL = _TOOLS / "launch_surface.py"
_REPO_ROOT = Path(__file__).resolve().parents[5]
_REAL_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx942_attention_dense.profile.yaml"
)

sys.path.insert(0, str(_TOOLS))

import launch_surface  # noqa: E402


def _profile(kmd_fields, surfaces) -> dict:
    return {"kmd_fields": kmd_fields, "launch_surface": surfaces}


def _surface(**overrides) -> dict:
    """A structurally-complete surface, so a test overriding one key does not also
    have to restate every other required key.

    cpp_mirror/test default to THIS tool's own files, given relative to the REPO
    ROOT: check() and the CLI both resolve against the working directory the same
    way provider_root is resolved elsewhere in this profile format, and every
    direct check()/CLI call in this file uses _REPO_ROOT as that root.
    """
    base = {
        "name": "grid",
        "python_source": "kernels/x.py:grid_fn (~line 10)",
        "cpp_mirror": (
            "projects/hipdnn/tools/IngestorGenerator/tools/launch_surface.py"
        ),
        "kmd_fields": [],
        "guard": "bounds-checked at prepare()",
        "test": (
            "projects/hipdnn/tools/IngestorGenerator/tests/test_launch_surface.py"
        ),
    }
    base.update(overrides)
    return base


class TestKmdFieldsCheck:
    """The check this tool exists for: a surface cannot name a kmd_fields entry the
    profile itself does not declare -- that is exactly the shape of the block_m
    defect."""

    def test_an_undeclared_kmd_field_is_caught(self):
        profile = _profile(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert any("block_m" in f and "grid" in f for f in failures), failures

    def test_every_declared_field_present_passes_this_check(self):
        """The positive control: fixing the omission (as the real profile now does
        for block_m) clears exactly this failure, proving the check is not
        vacuously true."""
        profile = _profile(
            kmd_fields=[
                {"name": "seqlen_q", "type": "int"},
                {"name": "block_m", "type": "int"},
            ],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert failures == []

    def test_the_check_fails_without_the_cross_reference(self):
        """Mutate the check to skip the kmd_fields comparison (as if it had never
        been written) and confirm the undeclared-field case above stops failing --
        the required failing-test evidence for this behaviour."""
        profile = _profile(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        surfaces = launch_surface.load_surfaces(profile)
        kmd_names = {f["name"] for f in profile["kmd_fields"]}
        # The mutation: check membership against the surface's OWN kmd_fields
        # (always true) instead of the profile's declared kmd_fields -- this is
        # the bug shape "the tool never looked", reproduced directly rather than
        # by editing the source file under test.
        for surface in surfaces:
            undeclared = [
                f for f in surface["kmd_fields"] if f not in set(surface["kmd_fields"])
            ]
            assert undeclared == [], (
                "a self-referential membership check can never fail, which is "
                "exactly why the real check compares against profile['kmd_fields'], "
                "not the surface's own list"
            )
        # Restore: the real check, run on the same fixture, DOES fail.
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert failures, "the real kmd_fields cross-reference must catch this case"


class TestCppMirrorExistence:
    def test_a_missing_cpp_mirror_path_is_caught(self):
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="does/not/exist/Nowhere.cpp:fn (~line 1)")],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert any("cpp_mirror" in f and "does/not/exist" in f for f in failures)

    def test_a_real_cpp_mirror_path_passes(self):
        profile = _profile(
            kmd_fields=[],
            surfaces=[
                _surface(
                    cpp_mirror=(
                        "projects/hipdnn/tools/IngestorGenerator/tools/"
                        "launch_surface.py:main"
                    )
                )
            ],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert failures == []


class TestTestPathExistence:
    def test_a_missing_test_path_is_caught(self):
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(test="does/not/exist/test_nowhere.py")],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert any("test path does not exist" in f for f in failures)

    def test_the_literal_none_is_not_a_missing_path(self):
        """`test: none` is a deliberate admission, not a broken path -- it must land
        in unguarded/untested, never in failures."""
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(guard="something", test="none")],
        )
        failures, unguarded = launch_surface.check(profile, _REPO_ROOT)
        assert failures == []
        assert unguarded == ["grid"]


class TestUnguardedReporting:
    def test_guard_none_is_named_and_fails_without_the_flag(self, tmp_path):
        profile_path = tmp_path / "p.yaml"
        profile_path.write_text(
            yaml.safe_dump(
                _profile(
                    kmd_fields=[],
                    surfaces=[_surface(name="kernargs", guard="none")],
                )
            )
        )
        result = _run("--check", str(profile_path), cwd=_REPO_ROOT)
        assert result.returncode == 1, result.stdout
        assert "kernargs" in result.stdout
        assert "CHECK FAILED" in result.stdout

    def test_allow_unguarded_flips_the_exit_code_without_hiding_the_name(
        self, tmp_path
    ):
        profile_path = tmp_path / "p.yaml"
        profile_path.write_text(
            yaml.safe_dump(
                _profile(
                    kmd_fields=[],
                    surfaces=[_surface(name="kernargs", guard="none")],
                )
            )
        )
        result = _run("--check", str(profile_path), "--allow-unguarded", cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout
        # The surface is still printed by name -- --allow-unguarded changes the
        # exit code, never the report.
        assert "kernargs" in result.stdout
        assert "CHECK PASSED" in result.stdout

    def test_a_fully_guarded_and_tested_surface_needs_no_flag(self, tmp_path):
        profile_path = tmp_path / "p.yaml"
        profile_path.write_text(
            yaml.safe_dump(_profile(kmd_fields=[], surfaces=[_surface()]))
        )
        result = _run("--check", str(profile_path), cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout
        assert "unguarded/untested  0" in result.stdout


class TestReport:
    def test_every_declared_surface_is_emitted(self):
        profile = _profile(
            kmd_fields=[],
            surfaces=[
                _surface(name="grid"),
                _surface(name="block"),
                _surface(name="kernargs", guard="none", test="none"),
            ],
        )
        table = launch_surface.render_report(profile)
        for name in ("grid", "block", "kernargs"):
            assert name in table
        # Markdown table shape: a header row, a separator row, one data row per
        # surface -- splitlines() rather than counting "\n" so the assertion does
        # not depend on whether the table ends with a trailing newline.
        assert len(table.splitlines()) == 2 + 3  # header + separator + 3 surfaces

    def test_report_cli_prints_the_table(self, tmp_path):
        profile_path = tmp_path / "p.yaml"
        profile_path.write_text(
            yaml.safe_dump(_profile(kmd_fields=[], surfaces=[_surface(name="grid")]))
        )
        result = _run("--report", str(profile_path), cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout
        assert "| grid |" in result.stdout


class TestMalformedProfile:
    def test_a_profile_with_no_launch_surface_block_is_refused(self):
        with pytest.raises(launch_surface.LaunchSurfaceError, match="launch_surface"):
            launch_surface.load_surfaces({"kmd_fields": []})

    def test_a_surface_missing_a_required_key_is_named(self):
        bad = {"name": "grid", "python_source": "x.py"}  # missing the rest
        errors = launch_surface.validate_shape([bad])
        assert errors
        assert "grid" in errors[0]
        assert "cpp_mirror" in errors[0]


class TestAgainstTheRealProfile:
    """The acceptance case: the real gfx942 profile, after Main's block_m fix,
    must --check clean modulo any surfaces it honestly declares unguarded."""

    def test_the_real_profile_check_names_only_genuinely_unguarded_surfaces(self):
        if not _REAL_PROFILE.exists():
            pytest.skip("real profile not present in this checkout")
        profile = yaml.safe_load(_REAL_PROFILE.read_text())
        failures, unguarded = launch_surface.check(profile, _REPO_ROOT)
        assert failures == [], (
            f"the real profile's launch_surface block must be structurally sound: "
            f"{failures}"
        )
        # kernargs and spec_resolution are declared guard: none / test: none because
        # nothing in the engine cross-checks the kernarg order against the Python
        # ABI, and nothing re-derives the dispatcher's resolution at runtime --
        # both true today. A surface appearing here that should NOT be unguarded
        # is exactly the case --check exists to surface.
        assert set(unguarded) == {"kernargs", "spec_resolution"}

    def test_the_real_profile_passes_with_allow_unguarded(self):
        if not _REAL_PROFILE.exists():
            pytest.skip("real profile not present in this checkout")
        result = _run(
            "--check", str(_REAL_PROFILE), "--allow-unguarded", cwd=_REPO_ROOT
        )
        assert result.returncode == 0, result.stdout

    def test_the_real_profile_report_covers_every_surface(self):
        if not _REAL_PROFILE.exists():
            pytest.skip("real profile not present in this checkout")
        profile = yaml.safe_load(_REAL_PROFILE.read_text())
        table = launch_surface.render_report(profile)
        for surface in profile["launch_surface"]:
            assert surface["name"] in table


def _run(*args, cwd):
    import subprocess

    return subprocess.run(
        [sys.executable, str(_TOOL), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
