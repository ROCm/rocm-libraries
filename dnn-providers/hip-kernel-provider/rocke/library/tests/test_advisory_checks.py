# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for ``rocke/tools/advisory_checks.py``.

That script is the non-blocking pre-push wrapper around pyright, radon and vulture.
These tests pin down its contract: it covers ALL rocke Python, it never fails a
push, its radon report lists nested functions (radon scores them separately from
their parent), and vulture counts uses across all of rocke but reports only the
requested non-test files. Tests that run radon or vulture skip when that tool is
not installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

_ROCKE = Path(__file__).resolve().parents[2]  # tests -> library -> rocke
_SCRIPT = _ROCKE / "tools" / "advisory_checks.py"


@pytest.fixture(scope="module")
def adv():
    if not _SCRIPT.is_file():
        pytest.skip("rocke/tools is not part of this layout")
    spec = importlib.util.spec_from_file_location("advisory_checks", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("parts", "skipped"),
    [
        (("library", "kernels", "gfx942", "attention_tiled_2d.py"), False),
        (("platform", "python", "rocke", "core", "ir.py"), False),
        (("tools", "run_checks.py"), False),
        # A file whose NAME starts with "venv" is code, not a virtualenv.
        (("library", "venv_probe.py"), False),
        (("library", "kernels", "__pycache__", "a.py"), True),
        (("library", ".pytest_cache", "a.py"), True),
        (("platform", "python", "rocke.egg-info", "a.py"), True),
        (("library", "benchmarks", "kernel_cache", "a.py"), True),
        (("platform", "_deps", "x", "a.py"), True),
        (("library", ".venv-gfx942", "lib", "a.py"), True),
        (("library", "venv", "lib", "a.py"), True),
        (("library", "build", "lib", "a.py"), True),
        (("platform", "_build", "a.py"), True),
        (("tools", "scratch", "a.py"), True),
        (("library", "logs", "a.py"), True),
    ],
)
def test_artifact_dirs_are_skipped(adv, parts, skipped):
    assert adv._skipped(adv.ROCKE.joinpath(*parts)) is skipped


def test_whole_run_spans_every_tree(adv):
    top = {p.relative_to(adv.ROCKE).parts[0] for p in adv._all_files()}
    assert top == set(adv.TREES)


def test_whole_run_prunes_artifacts_before_descending(adv, monkeypatch, tmp_path):
    monkeypatch.setattr(adv, "ROCKE", tmp_path)
    library = tmp_path / "library"
    library.mkdir()
    source = library / "venv_probe.py"
    source.write_text("x = 1\n", encoding="utf-8")
    for name in (*adv.SKIP_DIRS, ".venv-gfx942", "venv-test", "pkg.egg-info"):
        nested = library / name / "nested"
        nested.mkdir(parents=True)
        (nested / "stale.py").write_text("x = 2\n", encoding="utf-8")
    scandir = os.scandir
    visited = []

    def record_scandir(path):
        visited.append(Path(path))
        return scandir(path)

    monkeypatch.setattr(os, "scandir", record_scandir)
    assert adv._all_files() == [source]
    assert not any(p.is_relative_to(library) and p != library for p in visited)


def test_requested_files_are_limited_to_python_inside_rocke(adv, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    requested = [
        str(outside),  # python, but outside rocke
        str(adv.ROCKE / "AGENTS.md"),  # inside rocke, but not python
        str(adv.ROCKE / "no_such_file.py"),  # does not exist
        # python inside rocke, spelled via "..": kept, as its resolved path
        str(adv.ROCKE / "library" / ".." / "tools" / _SCRIPT.name),
    ]
    assert adv._rocke_files(requested) == [_SCRIPT]


def test_no_tracked_python_lives_outside_the_covered_trees(adv):
    # A new top-level Python tree would silently escape every advisory check.
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    proc = subprocess.run(
        [git, "ls-files", "-z", "--", "*.py"],
        cwd=adv.ROCKE,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip("rocke is not in a git checkout")
    top = {Path(f).parts[0] for f in proc.stdout.split("\0") if f}
    assert top <= set(adv.TREES), (
        f"tracked Python outside {adv.TREES}: add the new tree to TREES in "
        "tools/advisory_checks.py and to `include` in pyrightconfig.json"
    )


@pytest.fixture
def pyright_config(adv):
    config = adv.ROCKE / "pyrightconfig.json"
    if not config.is_file():
        pytest.skip("pyrightconfig.json is not part of this layout")
    # pyright accepts // comments in its config; json does not.
    text = "\n".join(
        line
        for line in config.read_text().splitlines()
        if not line.lstrip().startswith("//")
    )
    return json.loads(text)


def test_pyright_includes_the_same_trees(adv, pyright_config):
    assert pyright_config["include"] == list(adv.TREES)


def test_pyright_whole_run_excludes_artifacts_not_source_names(
    adv, pyright_config, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("pyright")
    monkeypatch.setattr(adv, "ROCKE", tmp_path)
    (tmp_path / "pyrightconfig.json").write_text(
        json.dumps(pyright_config), encoding="utf-8"
    )
    library = tmp_path / "library"
    library.mkdir()
    for name in (
        "node_modules",
        "__pycache__",
        ".hidden",
        ".venv-gfx942",
        "venv",
        "venv-test",
        "build",
        "_build",
        "scratch",
        "logs",
        "_deps",
        "pkg.egg-info",
    ):
        artifact = library / name
        artifact.mkdir()
        (artifact / "stale.py").write_text("stale: int = 'bad'\n", encoding="utf-8")
    for name in ("source.py", "venv_probe.py"):
        (library / name).write_text("source: int = 'bad'\n", encoding="utf-8")
    adv.run_pyright([], whole=True)
    out = capsys.readouterr().out
    assert "source.py:1: error:" in out
    assert "venv_probe.py:1: error:" in out
    assert "stale.py" not in out
    assert "pyright: 2 error(s), 0 warning(s)" in out


def test_pyright_gets_names_no_longer_than_pre_commit_passed(adv, monkeypatch):
    # pre-commit sizes each batch for repo-relative names; absolute names are
    # longer, so a full Windows batch would overflow its 32,767-character limit.
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, cwd=kwargs.get("cwd"))
        report = {
            "generalDiagnostics": [],
            "summary": {"errorCount": 0, "warningCount": 0},
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(report), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adv.run_pyright([_SCRIPT], whole=False)
    assert seen["cmd"][-1] == str(Path("tools", "advisory_checks.py"))
    assert seen["cwd"] == adv.ROCKE


def test_pyright_reports_ranged_and_rangeless_diagnostics(adv, monkeypatch, capsys):
    report = {
        "generalDiagnostics": [
            {"file": str(_SCRIPT), "severity": "warning", "message": "No range"},
            {
                "file": str(_SCRIPT),
                "severity": "error",
                "message": "First line\nMore detail",
                "rule": "reportArgumentType",
                "range": {"start": {"line": 6, "character": 0}},
            },
        ],
        "summary": {"errorCount": 1, "warningCount": 1},
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, json.dumps(report), ""
        ),
    )
    adv.run_pyright([_SCRIPT], whole=False)
    out = capsys.readouterr().out
    assert f"{adv._show(_SCRIPT)}: warning: No range (warning)" in out
    assert f"{adv._show(_SCRIPT)}:7: error: First line (reportArgumentType)" in out
    assert "More detail" not in out
    assert "pyright: 1 error(s), 1 warning(s)" in out
    assert "      1  warning" in out
    assert "      1  reportArgumentType" in out


def test_pyright_non_json_failure_is_advisory(adv, monkeypatch, capsys):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, "startup failed\n", "No module named pyright\n"
        ),
    )
    monkeypatch.setattr(sys, "argv", ["advisory_checks.py", "pyright", str(_SCRIPT)])
    assert adv.main() == 0
    out = capsys.readouterr().out
    assert "startup failed\nNo module named pyright\n" in out
    assert "error(s)" not in out


@pytest.mark.parametrize("tool", ["pyright", "radon", "vulture"])
def test_system_hook_with_cold_cache_and_missing_tools(adv, tmp_path, tool):
    pytest.importorskip("pre_commit")
    import yaml

    config_path = _ROCKE.parents[2] / ".pre-commit-config.yaml"
    if not config_path.is_file():
        pytest.skip("repository hook config is not part of this layout")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hook = next(
        h
        for repo in config["repos"]
        if repo["repo"] == "local"
        for h in repo["hooks"]
        if h["id"] == f"rocke-{tool}"
    )
    assert hook["language"] == "system"
    assert not hook.get("additional_dependencies")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = Path("dnn-providers/hip-kernel-provider/rocke/tools/advisory_checks.py")
    (repo / script).parent.mkdir(parents=True)
    shutil.copyfile(_SCRIPT, repo / script)
    (repo / ".pre-commit-config.yaml").write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [hook]}]}),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    empty_env = tmp_path / "empty-env"
    venv.EnvBuilder(with_pip=False).create(empty_env)
    bin_dir = empty_env / ("Scripts" if os.name == "nt" else "bin")
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "VIRTUAL_ENV": str(empty_env),
        "PRE_COMMIT_HOME": str(tmp_path / "cold-cache"),
        "PIP_NO_INDEX": "1",
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pre_commit",
            "run",
            hook["id"],
            "--hook-stage",
            "pre-push",
            "--all-files",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "[advisory, non-blocking]" in out
    assert "not installed" in out or "No module named pyright" in out
    assert "Installing environment" not in out


@pytest.mark.parametrize(
    ("failure", "reported"),
    [
        (RuntimeError("tool exploded"), "crashed"),
        # vulture's file collection sys.exit()s when a listed file has vanished.
        (SystemExit("Error: gone.py could not be found."), "crashed"),
        (ImportError("No module named 'vulture'"), "not installed"),
    ],
    ids=["exception", "sys-exit", "not-installed"],
)
def test_a_broken_tool_never_fails_a_push(adv, monkeypatch, capsys, failure, reported):
    def runner(files, whole):
        raise failure

    monkeypatch.setitem(adv.RUNNERS, "vulture", runner)
    monkeypatch.setattr(sys, "argv", ["advisory_checks.py", "vulture", str(_SCRIPT)])
    assert adv.main() == 0
    assert reported in capsys.readouterr().out


def test_nothing_runs_when_no_file_is_in_rocke(adv, monkeypatch, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    calls = []
    monkeypatch.setitem(adv.RUNNERS, "radon", lambda files, whole: calls.append(files))
    monkeypatch.setattr(sys, "argv", ["advisory_checks.py", "radon", str(outside)])
    assert adv.main() == 0
    assert calls == []


@pytest.fixture
def mini_rocke(adv, monkeypatch, tmp_path):
    """Point ``adv.ROCKE`` at a rocke-shaped tree small enough to scan quickly."""
    pytest.importorskip("vulture")
    tree = {
        "platform/helpers.py": (
            "def cross_tree_helper():\n    return 1\n\n\n"
            "def orphan_in_platform():\n    return 2\n"
        ),
        "library/kernel.py": (
            "from helpers import cross_tree_helper\n\n\n"
            "def build():\n    return cross_tree_helper()\n\n\n"
            "def orphan_in_library():\n    return 3\n"
        ),
        "library/tests/test_kernel.py": (
            "from kernel import build\n\n\n"
            "def test_build():\n    assert build() == 1\n\n\n"
            "def orphan_in_tests():\n    return 4\n"
        ),
    }
    for rel, source in tree.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(adv, "ROCKE", tmp_path)
    return tmp_path


def test_vulture_counts_uses_across_trees_and_tests(adv, mini_rocke, capsys):
    adv.run_vulture(adv._all_files(), whole=True)
    out = capsys.readouterr().out
    # cross_tree_helper is used from another tree and build() only from a test:
    # neither is dead. Nothing is reported from inside the tests.
    assert re.findall(r"unused \w+ '(\w+)'", out) == [
        "orphan_in_library",
        "orphan_in_platform",
    ]
    assert "vulture: 2 finding(s)" in out


@pytest.mark.parametrize("directory", ["build", "_build", "scratch", "logs"])
def test_vulture_artifacts_cannot_hide_unused_code(adv, mini_rocke, capsys, directory):
    artifact = mini_rocke / "library" / directory / "lib"
    artifact.mkdir(parents=True)
    stale = artifact / "stale.py"
    stale.write_text("orphan_in_library()\norphan_in_platform()\n", encoding="utf-8")
    assert adv._rocke_files([str(stale)]) == []
    adv.run_vulture(adv._all_files(), whole=True)
    out = capsys.readouterr().out
    assert "orphan_in_library" in out
    assert "orphan_in_platform" in out
    assert "vulture: 2 finding(s)" in out


def test_vulture_reports_only_the_requested_files(adv, mini_rocke, monkeypatch, capsys):
    # Paths as typed (relative, through ".."), while vulture reports resolved
    # ones: if the two spellings did not match, the report would silently be
    # empty. A requested test file is still not reported on.
    monkeypatch.chdir(mini_rocke / "library")
    helpers = str(Path("..", "platform", "helpers.py"))
    test_file = str(Path("tests", "test_kernel.py"))
    adv.run_vulture(adv._rocke_files([helpers, test_file]), whole=False)
    out = capsys.readouterr().out
    assert re.findall(r"unused \w+ '(\w+)'", out) == ["orphan_in_platform"]
    # Located relative to the cwd, with its line, so an editor can jump to it.
    assert f"{helpers}:5: unused function 'orphan_in_platform' (60% confidence)" in out
    assert "vulture: 1 finding(s)" in out


def test_vulture_skips_a_dangling_py_symlink(adv, mini_rocke, capsys):
    # Emacs marks a file with unsaved edits by a dangling `.#<name>` symlink. It
    # matches *.py, and vulture aborts the whole scan on a path that does not exist.
    try:
        (mini_rocke / "library" / ".#kernel.py").symlink_to("no-such-file")
    except OSError:
        pytest.skip("cannot create symlinks here")
    adv.run_vulture(adv._all_files(), whole=True)
    assert "vulture: 2 finding(s)" in capsys.readouterr().out


def test_vulture_scan_launches_no_child_process(adv, mini_rocke, monkeypatch, capsys):
    # The real tree is ~1,150 files; as arguments to a child process their names
    # make a ~62,000-character command line, past Windows' 32,767-character limit.
    def no_process(*args, **kwargs):
        raise AssertionError("vulture must not be launched as a child process")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    adv.run_vulture(adv._all_files(), whole=True)
    # The scan really ran: a launch whose error was swallowed reports nothing.
    assert "vulture: 2 finding(s)" in capsys.readouterr().out


def test_a_missing_vulture_is_reported_not_silently_empty(adv, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "vulture", None)  # `import vulture` fails
    monkeypatch.setattr(sys, "argv", ["advisory_checks.py", "vulture", str(_SCRIPT)])
    assert adv.main() == 0
    out = capsys.readouterr().out
    assert "not installed" in out
    assert "finding(s)" not in out


def test_radon_scores_nested_functions_separately(adv):
    pytest.importorskip("radon")
    from radon.complexity import cc_visit
    from radon.visitors import Function

    source = (
        "def outer(a):\n"
        "    def inner(x):\n"
        "        if x:\n"
        "            return 1\n"
        "        return 0\n"
        "    return inner(a)\n"
    )
    [outer] = [b for b in cc_visit(source) if isinstance(b, Function)]
    scored = {name: fn.complexity for name, fn in adv._with_closures(outer)}
    # The nested branch does not count toward its parent.
    assert scored == {"outer": 1, "outer.inner": 2}


def test_radon_report_lists_a_complex_function(adv, tmp_path, capsys):
    pytest.importorskip("radon")
    branches = "".join(f"    if a == {i}:\n        a += 1\n" for i in range(12))
    sample = tmp_path / "sample.py"
    sample.write_text(f"def busy(a):\n{branches}    return a\n", encoding="utf-8")
    adv.run_radon([sample], whole=False)
    out = capsys.readouterr().out
    assert "Most complex functions" in out
    # 12 branches + 1 = complexity 13, rank C.
    assert re.search(r"\bC\s+13\s+\S*sample\.py:1\s+busy\b", out)


def test_radon_report_is_quiet_for_simple_code(adv, tmp_path, capsys):
    pytest.importorskip("radon")
    sample = tmp_path / "simple.py"
    sample.write_text("def f(a):\n    return a\n", encoding="utf-8")
    adv.run_radon([sample], whole=False)
    out = capsys.readouterr().out
    assert "avg CC 1.0 (A)" in out
    assert "Most complex functions" not in out
