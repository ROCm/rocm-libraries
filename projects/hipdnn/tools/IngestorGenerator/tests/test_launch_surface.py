# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch-surface audit must fail on each defect class it exists to catch.

The headline class: a profile whose ``kmd_fields`` omits a field its own engine
dereferences unconditionally -- ``Gfx942AttentionDenseNative.cpp`` reads ``block_m``
through ``kernel.getIntMetadata`` on both ``kernelMatches`` and
``attentionDenseGeometry``. Nothing else cross-references a mirror against the KMD.

``TestKmdFieldsCheck`` uses fresh minimal fixtures; the classes below drive the same
check over a WHOLE profile from ``tests/fixtures/profiles/``, so every class runs on a
bare checkout. Those fixtures are controlled INPUTS: a literal like
``set(unguarded) == {"kernargs", "spec_resolution"}`` describes the fixture, not the
real pack.

An author auditing their own profile overrides the fixture per arch, one variable each
because the two profiles are not interchangeable:

    HIPDNN_INGESTOR_PROFILE_GFX942=/abs/path/to/gfx942.profile.yaml \\
    HIPDNN_INGESTOR_PROFILE_GFX950=/abs/path/to/gfx950.profile.yaml \\
        .venv/bin/python -m pytest tests/test_launch_surface.py

See ``tests/fixtures/profiles/README.md`` for the rest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_TOOL = _TOOLS / "launch_surface.py"

sys.path.insert(0, str(_TOOLS))

import launch_surface  # noqa: E402

_REPO_ROOT = launch_surface.find_repo_root(_TOOLS)

_PROFILE_VAR_942 = "HIPDNN_INGESTOR_PROFILE_GFX942"
_PROFILE_VAR_950 = "HIPDNN_INGESTOR_PROFILE_GFX950"

#: The committed synthetic profiles these classes audit by default; see that
#: directory's README for what each demonstrates.
_FIXTURE_PROFILES = Path(__file__).resolve().parent / "fixtures" / "profiles"


def _profile_path(var: str, fixture: str) -> Path:
    """The override named by ``var`` if set, else the committed fixture. A variable
    naming a non-file raises rather than reporting the fixture's clean result as the
    author's."""
    raw = os.environ.get(var)
    if not raw:
        return _FIXTURE_PROFILES / fixture
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(
            f"{var} is set to {raw!r}, which is not an existing file. Unset it to "
            f"audit the committed fixture {fixture}, or point it at a readable "
            "authoring profile."
        )
    return path


_PROFILE_942 = _profile_path(_PROFILE_VAR_942, "gfx942.profile.yaml")
_PROFILE_950 = _profile_path(_PROFILE_VAR_950, "gfx950.profile.yaml")


def _profile(kmd_fields, surfaces) -> dict:
    return {"kmd_fields": kmd_fields, "launch_surface": surfaces}


def _surface(**overrides) -> dict:
    """A structurally-complete surface, so a test overriding one key need not restate
    the rest. cpp_mirror/test default to this tool's own files, relative to the REPO
    ROOT, which every check()/CLI call here passes."""
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
    """A surface cannot name a kmd_fields entry the profile does not declare: the shape
    of the block_m defect."""

    def test_an_undeclared_kmd_field_is_caught(self):
        profile = _profile(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert any("block_m" in f and "grid" in f for f in failures), failures

    def test_every_declared_field_present_passes_this_check(self):
        """The positive control: the check is not vacuously true."""
        profile = _profile(
            kmd_fields=[
                {"name": "seqlen_q", "type": "int"},
                {"name": "block_m", "type": "int"},
            ],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert failures == []


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
        """`test: none` is a deliberate admission, so it lands in unguarded/untested."""
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
        # splitlines() rather than counting "\n", so the assertion does not depend
        # on whether the table ends with a trailing newline.
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


class TestAgainstTheGfx942Profile:
    """A whole gfx942-shaped profile must --check clean, modulo the surfaces it honestly
    declares unguarded.

    By default `tests/fixtures/profiles/gfx942.profile.yaml`, whose `grid` and
    `applicability` surfaces split their shared mirror's two required metadata fields,
    so passing means the union-over-shared-mirror rule fired. The unguarded set is a
    literal keyed to that fixture.
    """

    def test_the_profile_check_names_only_genuinely_unguarded_surfaces(self):
        profile = yaml.safe_load(_PROFILE_942.read_text(encoding="utf-8"))
        failures, unguarded = launch_surface.check(profile, _REPO_ROOT)
        assert failures == [], (
            f"the profile's launch_surface block must be structurally sound: "
            f"{failures}"
        )
        # The fixture declares kernargs and spec_resolution guard: none / test: none,
        # modelling the two surfaces a real integration cannot defend: nothing
        # cross-checks kernarg order against the Python ABI, and nothing re-derives the
        # dispatcher's resolution at runtime.
        assert set(unguarded) == {"kernargs", "spec_resolution"}

    def test_the_profile_passes_with_allow_unguarded(self):
        result = _run("--check", str(_PROFILE_942), "--allow-unguarded", cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout

    def test_the_profile_report_covers_every_surface(self):
        profile = yaml.safe_load(_PROFILE_942.read_text(encoding="utf-8"))
        table = launch_surface.render_report(profile)
        for surface in profile["launch_surface"]:
            assert surface["name"] in table


class TestMetadataFieldCoverage:
    """Check 1b: a metadata field a cpp_mirror reads through a REQUIRED accessor must be
    declared by SOME surface naming that mirror. See TestUndeclaredSurfaceLimit for the
    shape this cannot catch."""

    _CPP_DIRECT = """
        constexpr std::string_view SEQLEN_Q_FIELD = "seqlen_q";
        constexpr std::string_view BATCH_FIELD = "batch";
        int64_t f(const KernelDefinition& kernel) {
            return kernel.getIntMetadata(std::string(SEQLEN_Q_FIELD))
                 + kernel.getIntMetadata(std::string(BATCH_FIELD));
        }
    """

    _CPP_WRAPPER_LAMBDA = """
        constexpr std::string_view SEQLEN_Q_FIELD = "seqlen_q";
        constexpr std::string_view BATCH_FIELD = "batch";
        bool kernelMatches(const KernelDefinition& kernel) {
            const auto intField
                = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };
            return intField(SEQLEN_Q_FIELD) > 0 && intField(BATCH_FIELD) > 0;
        }
    """

    def test_a_required_field_with_no_declaring_surface_is_caught(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        profile = _profile(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:f", kmd_fields=["seqlen_q"])],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert any("batch" in f and "Mirror.cpp" in f for f in failures), failures

    def test_every_required_field_declared_passes(self, tmp_path):
        """Positive control: the check is not vacuously true."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        profile = _profile(
            kmd_fields=[
                {"name": "seqlen_q", "type": "int"},
                {"name": "batch", "type": "int"},
            ],
            surfaces=[
                _surface(
                    cpp_mirror="Mirror.cpp:f",
                    kmd_fields=["seqlen_q", "batch"],
                    test="none",
                )
            ],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []

    def test_the_forwarding_lambda_idiom_is_recognised(self, tmp_path):
        """The pack files' own idiom -- one lambda forwarding several field constants
        through a single accessor -- must resolve like a direct call."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_WRAPPER_LAMBDA)
        profile = _profile(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[
                _surface(cpp_mirror="Mirror.cpp:kernelMatches", kmd_fields=["seqlen_q"])
            ],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert any("batch" in f for f in failures), failures

    def test_two_surfaces_sharing_one_mirror_combine_their_kmd_fields(self, tmp_path):
        """grid and block cite the same geometry header, so the union of their
        kmd_fields, not either alone, must cover what it reads."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        profile = _profile(
            kmd_fields=[
                {"name": "seqlen_q", "type": "int"},
                {"name": "batch", "type": "int"},
            ],
            surfaces=[
                _surface(
                    name="a",
                    cpp_mirror="Mirror.cpp:f",
                    kmd_fields=["seqlen_q"],
                    test="none",
                ),
                _surface(
                    name="b",
                    cpp_mirror="Mirror.cpp:f",
                    kmd_fields=["batch"],
                    test="none",
                ),
            ],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []

    def test_trygetmetadata_fields_are_not_required(self, tmp_path):
        """tryGetMetadata is how this codebase spells 'may legitimately be absent'."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(
            """
            constexpr std::string_view USE_SINKS_FIELD = "use_sinks";
            bool f(const KernelDefinition& kernel) {
                const auto v = kernel.tryGetMetadata(std::string(USE_SINKS_FIELD));
                return v.has_value();
            }
            """
        )
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:f", kmd_fields=[], test="none")],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []

    def test_the_positive_control_a_field_named_only_in_a_comment_is_ignored(
        self, tmp_path
    ):
        """A regex scan over raw text risks matching a call site spelled out in a
        comment or string, so the extraction's comment/string exclusion is exercised
        directly."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(
            """
            // Do not read like: kernel.getIntMetadata(std::string(FAKE_FIELD));
            constexpr std::string_view FAKE_FIELD = "fake";
            const char* doc = "kernel.getIntMetadata(std::string(FAKE_FIELD))";
            int64_t f() { return 0; }
            """
        )
        fields = launch_surface.extract_required_metadata_fields(cpp.read_text())
        assert fields == set(), fields

    def test_the_positive_control_a_real_call_site_is_found(self, tmp_path):
        """The exclusion must discriminate real code from text, not just match
        nothing."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        fields = launch_surface.extract_required_metadata_fields(cpp.read_text())
        assert fields == {"seqlen_q", "batch"}


class TestSymbolExistence:
    """Checks 1c: a cpp_mirror/python_source locator's leading symbol, when it parses as
    one, must be real."""

    def test_a_nonexistent_cpp_symbol_is_caught(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("void realFunction() {}\n")
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:notARealFunction")],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert any(
            "notARealFunction" in f and "Mirror.cpp" in f for f in failures
        ), failures

    def test_a_real_cpp_symbol_passes(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("void realFunction() {}\n")
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:realFunction", test="none")],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []

    def test_a_class_qualified_cpp_symbol_checks_the_bare_method_name(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("class Handler { void launch() {} };\n")
        profile = _profile(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:Handler::launch", test="none")],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []

    def test_a_nonexistent_python_function_is_caught(self, tmp_path):
        (tmp_path / "provider").mkdir()
        (tmp_path / "provider" / "rocke").mkdir()
        (tmp_path / "provider" / "rocke" / "library").mkdir()
        py = tmp_path / "provider" / "rocke" / "library" / "mod.py"
        py.write_text("def real_function():\n    pass\n")
        profile = _profile(
            kmd_fields=[],
            surfaces=[
                _surface(
                    python_source="mod.py:not_a_real_function",
                    cpp_mirror="Mirror.cpp",
                )
            ],
        )
        profile["provider_root"] = "provider"
        (tmp_path / "Mirror.cpp").write_text("void f() {}\n")
        failures, _ = launch_surface.check(profile, tmp_path)
        assert any(
            "not_a_real_function" in f and "mod.py" in f for f in failures
        ), failures

    def test_a_real_python_function_passes(self, tmp_path):
        (tmp_path / "provider" / "rocke" / "library").mkdir(parents=True)
        py = tmp_path / "provider" / "rocke" / "library" / "mod.py"
        py.write_text("def real_function():\n    pass\n")
        (tmp_path / "Mirror.cpp").write_text("void f() {}\n")
        profile = _profile(
            kmd_fields=[],
            surfaces=[
                _surface(
                    python_source="mod.py:real_function",
                    cpp_mirror="Mirror.cpp",
                    test="none",
                )
            ],
        )
        profile["provider_root"] = "provider"
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []

    def test_a_prose_locator_with_no_leading_symbol_is_not_checked(self, tmp_path):
        """spec_resolution's real cpp_mirror is prose from the first token on:
        'prepare()' with parens is not a bare identifier."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("void unrelated() {}\n")
        profile = _profile(
            kmd_fields=[],
            surfaces=[
                _surface(
                    cpp_mirror="Mirror.cpp:prepare() trusts the KMD's own numbers",
                    test="none",
                )
            ],
        )
        failures, _ = launch_surface.check(profile, tmp_path)
        assert failures == []


class TestUndeclaredSurfaceLimit:
    """The documented residual gap: check 1b catches a deleted surface only when it
    uniquely covered a required metadata field. Both branches come from the gfx950
    profile, whose `applicability` and `kernargs` cite the SAME cpp_mirror."""

    def test_deleting_the_kernargs_surface_is_not_caught(self):
        """kernargs declares no kmd_fields and shares its mirror with applicability,
        which covers every required field, so the scan has no observable -- the real
        shape of a kernarg surface."""
        profile = yaml.safe_load(_PROFILE_950.read_text(encoding="utf-8"))
        profile["launch_surface"] = [
            s for s in profile["launch_surface"] if s["name"] != "kernargs"
        ]
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert failures == [], (
            "documenting a known limit: deleting kernargs is NOT caught because no "
            f"required field loses its only declarer; unexpected failures: {failures}"
        )

    def test_deleting_a_surface_with_unique_required_fields_is_caught(self):
        """The contrast: applicability is the sole declarer of the fields its cpp_mirror
        reads through a required accessor, so deleting it leaves them undeclared."""
        profile = yaml.safe_load(_PROFILE_950.read_text(encoding="utf-8"))
        profile["launch_surface"] = [
            s for s in profile["launch_surface"] if s["name"] != "applicability"
        ]
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert any("dtype" in f for f in failures), failures


class TestAgainstTheGfx950Profile:
    """The second profile shape, through the checks `TestAgainstTheGfx942Profile` runs
    over gfx942: one shared mirror with an exclusive declarer and one unguarded surface,
    where gfx942 splits mirror coverage across two surfaces. Its unguarded set is a
    literal keyed to the committed fixture."""

    def test_the_gfx950_profile_check_names_only_genuinely_unguarded_surfaces(self):
        profile = yaml.safe_load(_PROFILE_950.read_text(encoding="utf-8"))
        failures, unguarded = launch_surface.check(profile, _REPO_ROOT)
        assert failures == [], (
            f"the gfx950 profile's launch_surface block must be structurally "
            f"sound: {failures}"
        )
        # spec_resolution alone, so a second honest admission creeping into the fixture
        # cannot pass unnoticed. It models the surface where prepare() trusts the KMD's
        # own numbers.
        assert set(unguarded) == {"spec_resolution"}

    def test_the_gfx950_profile_passes_with_allow_unguarded(self):
        result = _run("--check", str(_PROFILE_950), "--allow-unguarded", cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout
        assert "CHECK PASSED" in result.stdout


class TestRepoRootResolution:
    """The CLI resolves cpp_mirror/test paths against the REPO ROOT, not the process
    cwd, so the same profile reports the same result from anywhere."""

    def test_find_repo_root_locates_the_git_checkout(self):
        found = launch_surface.find_repo_root(_TOOLS)
        assert (found / ".git").exists()

    def test_check_from_a_nested_cwd_matches_check_from_the_repo_root(self):
        from_root = _run(
            "--check", str(_PROFILE_950), "--allow-unguarded", cwd=_REPO_ROOT
        )
        nested_cwd = _PROFILE_950.parent  # wherever the profile happens to live
        from_nested = _run(
            "--check", str(_PROFILE_950), "--allow-unguarded", cwd=nested_cwd
        )
        assert from_root.returncode == 0, from_root.stdout
        assert from_nested.returncode == 0, from_nested.stdout
        assert "CHECK PASSED" in from_root.stdout
        assert "CHECK PASSED" in from_nested.stdout


def _run(*args, cwd):
    import subprocess

    return subprocess.run(
        [sys.executable, str(_TOOL), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
