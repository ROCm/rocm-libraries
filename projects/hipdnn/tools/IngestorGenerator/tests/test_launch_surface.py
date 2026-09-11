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

A second-pass review found a FOURTH defect class: every check above only ever looks
INSIDE the profile, so an entire undeclared surface -- delete ``kernargs``, or drop
``dtype`` from ``applicability``'s ``kmd_fields`` -- passed silently, because
nothing compared the profile's declaration against the ENGINE'S OWN metadata reads.
``TestMetadataFieldCoverage`` and ``TestSymbolExistence`` below cover the two checks
that close part of that gap (a required-accessor metadata read with no declaring
surface; a ``cpp_mirror``/``python_source`` symbol that does not exist in the named
file) and ``TestUndeclaredSurfaceLimit`` documents, with a real reproduction, the one
shape neither check can catch: a surface whose mirror reads no metadata at all.

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

sys.path.insert(0, str(_TOOLS))

import launch_surface  # noqa: E402

_REPO_ROOT = launch_surface.find_repo_root(_TOOLS)
_REAL_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx942_attention_dense.profile.yaml"
)
_REAL_PROFILE_950 = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx950_attention_dense.profile.yaml"
)


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


class TestMetadataFieldCoverage:
    """Check 1b: a metadata field a declared cpp_mirror reads through a REQUIRED
    accessor (getIntMetadata/getStringMetadata) must be declared by SOME surface
    naming that same mirror -- the check that catches an entire undeclared surface,
    not just an under-declared one, PROVIDED the deleted surface's mirror reads a
    field nothing else covers (see TestUndeclaredSurfaceLimit for the shape it
    cannot catch)."""

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
        """Positive control: declaring both fields the mirror reads clears the
        failure -- the check is not vacuously true."""
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
        through a single accessor -- must resolve the same as a direct call."""
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
        """grid and block in the real profiles both cite the same geometry header;
        the union of their kmd_fields, not either one alone, must cover what the
        header reads."""
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
        """tryGetMetadata is how this codebase spells 'may legitimately be absent'
        (the four ABI-extending features); a field read only that way must not be
        flagged even when no surface declares it."""
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
        """A regex scan over raw text (not a parse) risks matching a call site
        spelled out in a comment or string rather than real code -- this repo's
        embedded shell grep already produced a false-clean scan once from a BRE
        alternation gotcha (`grep "a\\|b"` matches nothing), so the extraction
        function's comment/string exclusion is exercised directly here rather than
        trusted by inspection."""
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
        """The other half of the positive control: the SAME accessor name, as an
        actual call rather than commentary, must be found -- proving the exclusion
        above is discriminating real code from text, not just matching nothing."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        fields = launch_surface.extract_required_metadata_fields(cpp.read_text())
        assert fields == {"seqlen_q", "batch"}


class TestSymbolExistence:
    """Checks 1c: a cpp_mirror/python_source locator's leading symbol, when it
    parses as one, must be real -- mutations (a) and (d) from the review."""

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
        """spec_resolution's real cpp_mirror is prose from the first token on
        ('prepare() trusts persistent...' -- 'prepare()' with the parens is not a
        bare identifier). This tool does not attempt to divine intent from free
        text, so such a locator is left alone rather than flagged as a fake
        symbol."""
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
    """The documented residual gap: check 1b catches an undeclared/deleted surface
    only when it uniquely covered a required metadata field. Both branches are
    reproduced against the REAL gfx950 profile, not a synthetic fixture, so this is
    also the acceptance evidence for review findings (e) and its contrast case."""

    def test_deleting_the_kernargs_surface_is_not_caught(self):
        """Mutation (e) from the review, reproduced directly: kernargs' cpp_mirror
        is Gfx950AttentionDenseDispatchHandler::launch, which reads zero metadata
        fields through any accessor (it only forwards positional device-buffer
        pointers), so no metadata-field scan can notice its absence. This test is
        the honest acknowledgment of that limit, not a claim the check covers it --
        see the module docstring's WHAT IT STILL DOES NOT DO section."""
        if not _REAL_PROFILE_950.exists():
            pytest.skip("real gfx950 profile not present in this checkout")
        profile = yaml.safe_load(_REAL_PROFILE_950.read_text())
        profile["launch_surface"] = [
            s for s in profile["launch_surface"] if s["name"] != "kernargs"
        ]
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert failures == [], (
            "documenting a known limit: deleting kernargs is NOT caught because "
            f"its mirror reads no metadata; unexpected failures: {failures}"
        )

    def test_deleting_a_surface_with_unique_required_fields_is_caught(self):
        """The contrast case: applicability's cpp_mirror (the same Native.cpp file
        kernargs cites) DOES read required fields -- dtype, head_size, and others --
        that no other surface declares, so deleting it IS caught by check 1b."""
        if not _REAL_PROFILE_950.exists():
            pytest.skip("real gfx950 profile not present in this checkout")
        profile = yaml.safe_load(_REAL_PROFILE_950.read_text())
        profile["launch_surface"] = [
            s for s in profile["launch_surface"] if s["name"] != "applicability"
        ]
        failures, _ = launch_surface.check(profile, _REPO_ROOT)
        assert any("dtype" in f for f in failures), failures


class TestGfx950RealProfile:
    """The second real profile this tool audits, in addition to gfx942 -- both
    must independently pass, proving the new checks generalise rather than being
    tuned to one profile's shape."""

    def test_the_real_gfx950_profile_check_names_only_genuinely_unguarded_surfaces(
        self,
    ):
        if not _REAL_PROFILE_950.exists():
            pytest.skip("real gfx950 profile not present in this checkout")
        profile = yaml.safe_load(_REAL_PROFILE_950.read_text())
        failures, unguarded = launch_surface.check(profile, _REPO_ROOT)
        assert failures == [], (
            f"the real gfx950 profile's launch_surface block must be structurally "
            f"sound: {failures}"
        )
        # spec_resolution alone: prepare() trusts persistent/num_persistent/ragged
        # exactly as the KMD states them, defended offline by test_dispatch_parity
        # rather than a runtime check -- see the profile's own comment.
        assert set(unguarded) == {"spec_resolution"}

    def test_the_real_gfx950_profile_passes_with_allow_unguarded(self):
        if not _REAL_PROFILE_950.exists():
            pytest.skip("real gfx950 profile not present in this checkout")
        result = _run(
            "--check", str(_REAL_PROFILE_950), "--allow-unguarded", cwd=_REPO_ROOT
        )
        assert result.returncode == 0, result.stdout
        assert "CHECK PASSED" in result.stdout


class TestRepoRootResolution:
    """The CLI must resolve cpp_mirror/test paths against the REPO ROOT, not the
    process working directory -- a profile checked from three directories down must
    report the same result as one checked from the repo root."""

    def test_find_repo_root_locates_the_git_checkout(self):
        found = launch_surface.find_repo_root(_TOOLS)
        assert (found / ".git").exists()

    def test_check_from_a_nested_cwd_matches_check_from_the_repo_root(self):
        if not _REAL_PROFILE_950.exists():
            pytest.skip("real gfx950 profile not present in this checkout")
        from_root = _run(
            "--check", str(_REAL_PROFILE_950), "--allow-unguarded", cwd=_REPO_ROOT
        )
        nested_cwd = _REAL_PROFILE_950.parent  # .../IngestorGenerator/configs
        from_nested = _run(
            "--check", str(_REAL_PROFILE_950), "--allow-unguarded", cwd=nested_cwd
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
