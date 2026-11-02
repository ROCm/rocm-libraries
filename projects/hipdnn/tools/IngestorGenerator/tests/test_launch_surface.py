# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch-surface audit must fail on each defect class it exists to catch.

The headline class: a manifest whose ``kmd_fields`` omits a metadata field its own
engine reads. Nothing else cross-references a mirror against the KMD.

``TestKmdFieldsCheck`` uses fresh minimal fixtures; the classes below drive the same
check over a WHOLE manifest from ``tests/fixtures/launch_surface/``, against mirror
files the test writes into a temporary check root, so every class runs on a bare
checkout. Those manifests and mirrors are controlled INPUTS: a literal like
``set(unguarded) == {"kernargs", "spec_resolution"}`` describes the fixture, not the
real pack. ``TestAgainstTheShippedEngineManifest`` audits the real shipped manifest.

An author auditing their own manifest overrides the fixture per arch, one variable each
because the two manifests are not interchangeable (the check then resolves paths
against the repository root):

    HIPDNN_INGESTOR_MANIFEST_GFX942=/abs/path/to/gfx942.engine.yaml \\
    HIPDNN_INGESTOR_MANIFEST_GFX950=/abs/path/to/gfx950.engine.yaml \\
        .venv/bin/python -m pytest tests/test_launch_surface.py

See ``tests/fixtures/launch_surface/README.md`` for the rest.
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

_MANIFEST_VAR_942 = "HIPDNN_INGESTOR_MANIFEST_GFX942"
_MANIFEST_VAR_950 = "HIPDNN_INGESTOR_MANIFEST_GFX950"

#: The committed synthetic manifests these classes audit by default; see that
#: directory's README for what each demonstrates.
_FIXTURE_MANIFESTS = Path(__file__).resolve().parent / "fixtures" / "launch_surface"


def _manifest_path(var: str, fixture: str) -> Path:
    """The override named by ``var`` if set, else the committed fixture. A variable
    naming a non-file raises rather than reporting the fixture's clean result as the
    author's."""
    raw = os.environ.get(var)
    if not raw:
        return _FIXTURE_MANIFESTS / fixture
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(
            f"{var} is set to {raw!r}, which is not an existing file. Unset it to "
            f"audit the committed fixture {fixture}, or point it at a readable "
            "authoring manifest."
        )
    return path


_MANIFEST_942 = _manifest_path(_MANIFEST_VAR_942, "gfx942.engine.yaml")
_MANIFEST_950 = _manifest_path(_MANIFEST_VAR_950, "gfx950.engine.yaml")

#: What the fixture manifests' paths resolve to: a python mirror defining the four
#: functions they name, a C++ mirror whose ``matches`` reads ``dtype`` and
#: ``block_size`` through required accessors, and a test file.
_FIXTURE_FILES = {
    "provider/rocke/library/kernels/mirror.py": (
        "def grid_fn():\n    pass\n\n\n"
        "def is_valid_fn():\n    pass\n\n\n"
        "def build_fn():\n    pass\n\n\n"
        "def resolve_fn():\n    pass\n"
    ),
    "provider/Mirror.cpp": """
        constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
        constexpr std::string_view DTYPE_FIELD = "dtype";
        struct Handler { void prepare() {} };
        bool matches(const KernelDefinition& kernel) {
            return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == "f16"
                && kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)) > 0;
        }
        void signature() {}
    """,
    "tests/test_mirror.py": "def test_it():\n    pass\n",
}


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("launch_surface_mirrors")
    for relative, text in _FIXTURE_FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def _audit_root(variable: str, fixture_root: Path) -> Path:
    """Where the audited manifest's paths resolve: the fixture's temporary root, or the
    repository when the author overrides the manifest."""
    return _REPO_ROOT if os.environ.get(variable) else fixture_root


def _main_at(monkeypatch, root: Path, *args) -> int:
    """`launch_surface.main` with its repo-root lookup answered by ``root``."""
    monkeypatch.setattr(launch_surface, "find_repo_root", lambda start: root)
    return launch_surface.main(list(args))


def _manifest(kmd_fields, surfaces) -> dict:
    return {"kmd_fields": kmd_fields, "launch_surface": surfaces}


#: A real python_source, relative to the repo root (no provider_root).
_PYTHON_SOURCE_PATH = "projects/hipdnn/tools/IngestorGenerator/tools/launch_surface.py"
_PYTHON_SOURCE = f"{_PYTHON_SOURCE_PATH}:main"


def _seed_python_source(root: Path) -> Path:
    """Copy the default python_source under a temporary check root; returns ``root``."""
    target = root / _PYTHON_SOURCE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_TOOL.read_text(encoding="utf-8"), encoding="utf-8")
    return root


def _surface(**overrides) -> dict:
    """A structurally-complete surface, so a test overriding one key need not restate
    the rest. python_source/cpp_mirror/test default to this tool's own files, relative
    to the REPO ROOT."""
    base = {
        "name": "grid",
        "python_source": _PYTHON_SOURCE,
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
    """A surface cannot name a kmd_fields entry the manifest does not declare: the shape
    of the block_m defect."""

    def test_an_undeclared_kmd_field_is_caught(self):
        manifest = _manifest(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        failures, _ = launch_surface.check(manifest, _REPO_ROOT)
        assert any("block_m" in f and "grid" in f for f in failures), failures

    def test_every_declared_field_present_passes_this_check(self):
        """The positive control: the check is not vacuously true."""
        manifest = _manifest(
            kmd_fields=[
                {"name": "seqlen_q", "type": "int"},
                {"name": "block_m", "type": "int"},
            ],
            surfaces=[_surface(kmd_fields=["seqlen_q", "block_m"])],
        )
        failures, _ = launch_surface.check(manifest, _REPO_ROOT)
        assert failures == []


class TestCppMirrorExistence:
    def test_a_missing_cpp_mirror_path_is_caught(self):
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="does/not/exist/Nowhere.cpp:fn (~line 1)")],
        )
        failures, _ = launch_surface.check(manifest, _REPO_ROOT)
        assert any("cpp_mirror" in f and "does/not/exist" in f for f in failures)

    def test_a_real_cpp_mirror_path_passes(self):
        manifest = _manifest(
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
        failures, _ = launch_surface.check(manifest, _REPO_ROOT)
        assert failures == []


class TestTestPathExistence:
    def test_a_missing_test_path_is_caught(self):
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(test="does/not/exist/test_nowhere.py")],
        )
        failures, _ = launch_surface.check(manifest, _REPO_ROOT)
        assert any("test path does not exist" in f for f in failures)

    def test_the_literal_none_is_not_a_missing_path(self):
        """`test: none` is a deliberate admission, so it lands in unguarded/untested."""
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(guard="something", test="none")],
        )
        failures, unguarded = launch_surface.check(manifest, _REPO_ROOT)
        assert failures == []
        assert unguarded == ["grid"]


class TestUnguardedReporting:
    def test_guard_none_is_named_and_fails_without_the_flag(self, tmp_path):
        manifest_path = tmp_path / "p.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                _manifest(
                    kmd_fields=[],
                    surfaces=[_surface(name="kernargs", guard="none")],
                )
            )
        )
        result = _run("--check", str(manifest_path), cwd=_REPO_ROOT)
        assert result.returncode == 1, result.stdout
        assert "kernargs" in result.stdout
        assert "CHECK FAILED" in result.stdout

    def test_allow_unguarded_flips_the_exit_code_without_hiding_the_name(
        self, tmp_path
    ):
        manifest_path = tmp_path / "p.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                _manifest(
                    kmd_fields=[],
                    surfaces=[_surface(name="kernargs", guard="none")],
                )
            )
        )
        result = _run(
            "--check", str(manifest_path), "--allow-unguarded", cwd=_REPO_ROOT
        )
        assert result.returncode == 0, result.stdout
        # The surface is still printed by name -- --allow-unguarded changes the
        # exit code, never the report.
        assert "kernargs" in result.stdout
        assert "CHECK PASSED" in result.stdout

    def test_a_fully_guarded_and_tested_surface_needs_no_flag(self, tmp_path):
        manifest_path = tmp_path / "p.yaml"
        manifest_path.write_text(
            yaml.safe_dump(_manifest(kmd_fields=[], surfaces=[_surface()]))
        )
        result = _run("--check", str(manifest_path), cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout
        assert "unguarded/untested  0" in result.stdout


class TestReport:
    def test_every_declared_surface_is_emitted(self):
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[
                _surface(name="grid", guard="a folded\nguard || other\n"),
                _surface(name="block"),
                _surface(name="kernargs", guard="none", test="none"),
            ],
        )
        table = launch_surface.render_report(manifest)
        for name in ("grid", "block", "kernargs"):
            assert name in table
        # splitlines() rather than counting "\n", so the assertion does not depend
        # on whether the table ends with a trailing newline.
        assert len(table.splitlines()) == 2 + 3  # header + separator + 3 surfaces
        assert all(r.replace("\\|", "").count("|") == 7 for r in table.splitlines())

    def test_report_cli_prints_the_table(self, tmp_path):
        manifest_path = tmp_path / "p.yaml"
        manifest_path.write_text(
            yaml.safe_dump(_manifest(kmd_fields=[], surfaces=[_surface(name="grid")]))
        )
        result = _run("--report", str(manifest_path), cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout
        assert "| grid |" in result.stdout


class TestMalformedManifest:
    @pytest.mark.parametrize(
        "suffix, text",
        [
            (".json", '{"launch_surface": [], "launch_surface": [1]}'),
            (".yaml", "launch_surface: []\nlaunch_surface: [1]\n"),
        ],
    )
    def test_a_key_declared_twice_is_exit_2_naming_the_key(
        self, tmp_path, capsys, suffix, text
    ):
        path = tmp_path / f"m{suffix}"
        path.write_text(text)
        assert launch_surface.main([str(path), "--report"]) == 2
        assert "launch_surface" in capsys.readouterr().err

    def test_a_tab_indented_json_manifest_loads(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text('{\n\t"launch_surface": [\n\t\t{"name": "grid"}\n\t]\n}\n')
        assert launch_surface._load_manifest(str(path))["launch_surface"] == [
            {"name": "grid"}
        ]

    def test_a_manifest_with_no_launch_surface_block_is_refused(self):
        with pytest.raises(launch_surface.LaunchSurfaceError, match="launch_surface"):
            launch_surface.load_surfaces({"kmd_fields": []})

    def test_a_surface_missing_a_required_key_is_named(self):
        bad = {"name": "grid", "python_source": "x.py"}  # missing the rest
        errors = launch_surface.validate_shape([bad])
        assert errors
        assert "grid" in errors[0]
        assert "cpp_mirror" in errors[0]


class TestAgainstTheGfx942Manifest:
    """A whole gfx942-shaped manifest must --check clean, modulo the surfaces it honestly
    declares unguarded.

    By default `tests/fixtures/launch_surface/gfx942.engine.yaml`, whose `grid` and
    `applicability` surfaces split their shared mirror's two required metadata fields,
    so passing means the union-over-shared-mirror rule fired. The unguarded set is a
    literal keyed to that fixture.
    """

    def test_the_manifest_check_names_only_genuinely_unguarded_surfaces(
        self, fixture_root
    ):
        manifest = yaml.safe_load(_MANIFEST_942.read_text(encoding="utf-8"))
        root = _audit_root(_MANIFEST_VAR_942, fixture_root)
        failures, unguarded = launch_surface.check(manifest, root)
        assert failures == [], (
            f"the manifest's launch_surface block must be structurally sound: "
            f"{failures}"
        )
        # The fixture declares kernargs and spec_resolution guard: none / test: none,
        # modelling the two surfaces a real integration cannot defend: nothing
        # cross-checks kernarg order against the Python ABI, and nothing re-derives the
        # dispatcher's resolution at runtime.
        assert set(unguarded) == {"kernargs", "spec_resolution"}

    def test_the_manifest_passes_with_allow_unguarded(
        self, fixture_root, monkeypatch, capsys
    ):
        root = _audit_root(_MANIFEST_VAR_942, fixture_root)
        rc = _main_at(
            monkeypatch, root, "--check", str(_MANIFEST_942), "--allow-unguarded"
        )
        assert rc == 0, capsys.readouterr().out

    def test_the_manifest_without_the_flag_fails_on_the_unguarded_surfaces(
        self, fixture_root, monkeypatch, capsys
    ):
        root = _audit_root(_MANIFEST_VAR_942, fixture_root)
        assert _main_at(monkeypatch, root, "--check", str(_MANIFEST_942)) == 1
        assert "kernargs" in capsys.readouterr().out

    def test_the_manifest_report_covers_every_surface(self):
        manifest = yaml.safe_load(_MANIFEST_942.read_text(encoding="utf-8"))
        table = launch_surface.render_report(manifest)
        for surface in manifest["launch_surface"]:
            assert surface["name"] in table


class TestMetadataFieldCoverage:
    """Check 1b: a metadata field a cpp_mirror reads through a required or declining
    accessor must be declared by SOME surface naming that mirror. See
    TestUndeclaredSurfaceLimit for the shape this cannot catch."""

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

    _CPP_DECLINING = """
        constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
        constexpr std::string_view BLOCK_M_FIELD = "block_m";
        std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
        {
            return std::nullopt;
        }
        bool kernelMatches(const KernelDefinition& kernel) {
            const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
            const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
            return headSize.has_value() && blockM.has_value();
        }
    """

    def test_a_required_field_with_no_declaring_surface_is_caught(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        manifest = _manifest(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:f", kmd_fields=["seqlen_q"])],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert any("batch" in f and "Mirror.cpp" in f for f in failures), failures

    def test_every_required_field_declared_passes(self, tmp_path):
        """Positive control: the check is not vacuously true."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        manifest = _manifest(
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
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert failures == []

    def test_the_forwarding_lambda_idiom_is_recognised(self, tmp_path):
        """The pack files' own idiom -- one lambda forwarding several field constants
        through a single accessor -- must resolve like a direct call."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_WRAPPER_LAMBDA)
        manifest = _manifest(
            kmd_fields=[{"name": "seqlen_q", "type": "int"}],
            surfaces=[
                _surface(cpp_mirror="Mirror.cpp:kernelMatches", kmd_fields=["seqlen_q"])
            ],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert any("batch" in f for f in failures), failures

    def test_two_surfaces_sharing_one_mirror_combine_their_kmd_fields(self, tmp_path):
        """grid and block cite the same geometry header, so the union of their
        kmd_fields, not either alone, must cover what it reads."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DIRECT)
        manifest = _manifest(
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
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
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
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:f", kmd_fields=[], test="none")],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert failures == []

    @pytest.mark.parametrize("declared", [["head_size"], ["head_size", "block_m"]])
    def test_a_declining_accessor_field_must_be_declared(self, tmp_path, declared):
        """Also proves the helper's own definition is not a read of a field."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text(self._CPP_DECLINING)
        manifest = _manifest(
            kmd_fields=[{"name": n, "type": "int"} for n in declared],
            surfaces=[
                _surface(
                    cpp_mirror="Mirror.cpp:kernelMatches",
                    kmd_fields=declared,
                    test="none",
                )
            ],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        if "block_m" in declared:
            assert failures == []
        else:
            assert any(
                "block_m" in f and "integerMetadata" in f for f in failures
            ), failures

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
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:notARealFunction")],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert any(
            "notARealFunction" in f and "Mirror.cpp" in f for f in failures
        ), failures

    def test_a_real_cpp_symbol_passes(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("void realFunction() {}\n")
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:realFunction", test="none")],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert failures == []

    def test_a_class_qualified_cpp_symbol_checks_the_bare_method_name(self, tmp_path):
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("class Handler { void launch() {} };\n")
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[_surface(cpp_mirror="Mirror.cpp:Handler::launch", test="none")],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert failures == []

    def test_a_nonexistent_python_function_is_caught(self, tmp_path):
        (tmp_path / "provider").mkdir()
        (tmp_path / "provider" / "rocke").mkdir()
        (tmp_path / "provider" / "rocke" / "library").mkdir()
        py = tmp_path / "provider" / "rocke" / "library" / "mod.py"
        py.write_text("def real_function():\n    pass\n")
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[
                _surface(
                    python_source="mod.py:not_a_real_function",
                    cpp_mirror="Mirror.cpp",
                )
            ],
        )
        manifest["provider_root"] = "provider"
        (tmp_path / "Mirror.cpp").write_text("void f() {}\n")
        failures, _ = launch_surface.check(manifest, tmp_path)
        assert any(
            "not_a_real_function" in f and "mod.py" in f for f in failures
        ), failures

    def test_a_real_python_function_passes(self, tmp_path):
        (tmp_path / "provider" / "rocke" / "library").mkdir(parents=True)
        py = tmp_path / "provider" / "rocke" / "library" / "mod.py"
        py.write_text("def real_function():\n    pass\n")
        (tmp_path / "Mirror.cpp").write_text("void f() {}\n")
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[
                _surface(
                    python_source="mod.py:real_function",
                    cpp_mirror="Mirror.cpp",
                    test="none",
                )
            ],
        )
        manifest["provider_root"] = "provider"
        failures, _ = launch_surface.check(manifest, tmp_path)
        assert failures == []

    def test_a_prose_locator_with_no_leading_symbol_is_not_checked(self, tmp_path):
        """spec_resolution's real cpp_mirror is prose from the first token on:
        'prepare()' with parens is not a bare identifier."""
        cpp = tmp_path / "Mirror.cpp"
        cpp.write_text("void unrelated() {}\n")
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[
                _surface(
                    cpp_mirror="Mirror.cpp:prepare() trusts the KMD's own numbers",
                    test="none",
                )
            ],
        )
        failures, _ = launch_surface.check(manifest, _seed_python_source(tmp_path))
        assert failures == []


class TestPythonSourceExistence:
    """Check 2b: python_source must name a real file under
    ``<provider_root>/rocke/library``, or the check root without provider_root."""

    @pytest.mark.parametrize(
        "provider_root,written,error",
        [
            ("provider", None, "provider/rocke/library"),
            ("provider", "mod.py", "provider/rocke/library"),  # outside provider_root
            (None, None, "repo root"),
            (None, "mod.py", None),
        ],
    )
    def test_python_source_must_exist(self, tmp_path, provider_root, written, error):
        (tmp_path / "provider" / "rocke" / "library").mkdir(parents=True)
        if written:
            (tmp_path / written).write_text("def fn():\n    pass\n")
        (tmp_path / "Mirror.cpp").write_text("void f() {}\n")
        manifest = _manifest(
            kmd_fields=[],
            surfaces=[
                _surface(
                    python_source="mod.py:fn", cpp_mirror="Mirror.cpp", test="none"
                )
            ],
        )
        if provider_root:
            manifest["provider_root"] = provider_root
        failures, _ = launch_surface.check(manifest, tmp_path)
        if error is None:
            assert failures == []
        else:
            assert any(
                "python_source" in f and "mod.py" in f and error in f for f in failures
            ), failures


class TestUndeclaredSurfaceLimit:
    """The documented residual gap: check 1b catches a deleted surface only when it
    uniquely covered a required metadata field. Both branches come from the gfx950
    fixture, whose `applicability` and `kernargs` cite the SAME cpp_mirror."""

    def test_deleting_the_kernargs_surface_is_not_caught(self, fixture_root):
        """kernargs declares no kmd_fields and shares its mirror with applicability,
        which covers every required field, so the scan has no observable -- the real
        shape of a kernarg surface."""
        manifest = yaml.safe_load(_MANIFEST_950.read_text(encoding="utf-8"))
        manifest["launch_surface"] = [
            s for s in manifest["launch_surface"] if s["name"] != "kernargs"
        ]
        failures, _ = launch_surface.check(
            manifest, _audit_root(_MANIFEST_VAR_950, fixture_root)
        )
        assert failures == [], (
            "documenting a known limit: deleting kernargs is NOT caught because no "
            f"required field loses its only declarer; unexpected failures: {failures}"
        )

    def test_deleting_a_surface_with_unique_required_fields_is_caught(
        self, fixture_root
    ):
        """The contrast: applicability is the sole declarer of the fields its cpp_mirror
        reads through a required accessor, so deleting it leaves them undeclared."""
        manifest = yaml.safe_load(_MANIFEST_950.read_text(encoding="utf-8"))
        manifest["launch_surface"] = [
            s for s in manifest["launch_surface"] if s["name"] != "applicability"
        ]
        failures, _ = launch_surface.check(
            manifest, _audit_root(_MANIFEST_VAR_950, fixture_root)
        )
        assert any("dtype" in f for f in failures), failures


class TestAgainstTheGfx950Manifest:
    """The second manifest shape, through the checks `TestAgainstTheGfx942Manifest` runs
    over gfx942: one shared mirror with an exclusive declarer and one unguarded surface,
    where gfx942 splits mirror coverage across two surfaces. Its unguarded set is a
    literal keyed to the committed fixture."""

    def test_the_gfx950_manifest_check_names_only_genuinely_unguarded_surfaces(
        self, fixture_root
    ):
        manifest = yaml.safe_load(_MANIFEST_950.read_text(encoding="utf-8"))
        root = _audit_root(_MANIFEST_VAR_950, fixture_root)
        failures, unguarded = launch_surface.check(manifest, root)
        assert failures == [], (
            f"the gfx950 manifest's launch_surface block must be structurally "
            f"sound: {failures}"
        )
        # spec_resolution alone, so a second honest admission creeping into the fixture
        # cannot pass unnoticed. It models the surface where prepare() trusts the KMD's
        # own numbers.
        assert set(unguarded) == {"spec_resolution"}

    def test_the_gfx950_manifest_passes_with_allow_unguarded(
        self, fixture_root, monkeypatch, capsys
    ):
        root = _audit_root(_MANIFEST_VAR_950, fixture_root)
        rc = _main_at(
            monkeypatch, root, "--check", str(_MANIFEST_950), "--allow-unguarded"
        )
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "CHECK PASSED" in out


class TestAgainstTheShippedEngineManifest:
    def test_every_declared_surface_is_real_guarded_and_tested(self):
        manifest = _TOOLS.parent / "configs" / "gfx950_attention_dense.engine.yaml"
        result = _run("--check", str(manifest), cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "unguarded/untested  0" in result.stdout


class TestRepoRootResolution:
    """The CLI resolves cpp_mirror/test paths against the REPO ROOT, not the process
    cwd, so the same manifest reports the same result from anywhere."""

    def test_find_repo_root_locates_the_tree_holding_this_tool(self):
        """A checkout or a `git archive` extract: either way the root is the
        directory holding projects/hipdnn, never the tools directory."""
        found = launch_surface.find_repo_root(_TOOLS)
        assert (found / "projects" / "hipdnn" / "tools" / "IngestorGenerator").is_dir()

    def test_a_tree_without_git_metadata_resolves_to_its_root(self, tmp_path):
        """`git archive` drops .git. Falling back to the start directory made every
        repo-relative profile path resolve under tools/."""
        if any((p / ".git").exists() for p in (tmp_path, *tmp_path.parents)):
            pytest.skip("the temporary directory sits inside a git checkout")
        tools = (
            tmp_path / "projects" / "hipdnn" / "tools" / "IngestorGenerator" / "tools"
        )
        tools.mkdir(parents=True)
        assert launch_surface.find_repo_root(tools) == tmp_path.resolve()

    def test_check_from_a_nested_cwd_matches_check_from_the_repo_root(self, tmp_path):
        manifest_path = tmp_path / "p.yaml"
        manifest_path.write_text(
            yaml.safe_dump(_manifest(kmd_fields=[], surfaces=[_surface()]))
        )
        from_root = _run("--check", str(manifest_path), cwd=_REPO_ROOT)
        from_nested = _run("--check", str(manifest_path), cwd=tmp_path)
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
