# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for ``rocke/tools/static_analysis/static_analysis_checks.py``.

That script is the non-blocking pre-push wrapper around pyright, radon and vulture.
These tests pin down its contract: it covers ALL rocke Python (the tracked files
that git lists, with no directory list of its own), it never fails a push, its
radon report lists nested functions (radon scores them separately from their
parent), and vulture counts uses across all of rocke but reports only the
requested non-test files. Tests that run pyright, radon or vulture skip when that
tool is not installed; tests that need git skip without it.
"""

from __future__ import annotations

import importlib.util
import io
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
_SCRIPT = _ROCKE / "tools" / "static_analysis" / "static_analysis_checks.py"
_PYRIGHT_CONFIG = _SCRIPT.with_name("pyrightconfig.json")


@pytest.fixture(scope="module")
def adv():
    if not _SCRIPT.is_file():
        pytest.skip("rocke/tools is not part of this layout")
    spec = importlib.util.spec_from_file_location("static_analysis_checks", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clean_git_env(monkeypatch):
    """Keep the caller's git setup out of every test.

    A hook, ``git rebase --exec`` or a linked worktree exports GIT_DIR and friends;
    with them set, even the scratch ``git init`` below would rewrite the caller's real
    repository. The user's and the system's git config (commit signing, global
    ignores, hooks) must not matter either.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def require_git():
    if shutil.which("git") is None:
        pytest.skip("git is not available")


def _git(cwd, *args):
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr}"


@pytest.fixture
def git_rocke(adv, require_git, monkeypatch, tmp_path):
    """An empty git repository standing in for rocke: ``adv.ROCKE`` points at it."""
    _git(tmp_path, "init", "-q")
    monkeypatch.setattr(adv, "ROCKE", tmp_path)
    return tmp_path


def _track(repo, files):
    """Write ``{relative path: source}`` under ``repo`` and ``git add`` it all."""
    for rel, source in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    _git(repo, "add", "--force", "-A")  # --force: a global ignore file must not matter


def test_whole_run_lists_exactly_the_tracked_python(adv, git_rocke):
    _track(
        git_rocke,
        {
            "library/kernel.py": "x = 1\n",
            "platform/python/rocke/ir.py": "x = 1\n",
            # A new top-level tree needs no registration, and a tracked file under
            # an artifact-looking name (build/, venv_*) is source: git says so.
            "newtree/tool.py": "x = 1\n",
            "library/build/real.py": "x = 1\n",
            "library/venv_probe.py": "x = 1\n",
            "library/k\u00e9rnel.py": "x = 1\n",  # git quotes this name without -z
            "docs/notes.md": "not python\n",
        },
    )
    for artifact in (".venv-gfx942/lib", "__pycache__", "pkg.egg-info"):
        stale = git_rocke / "library" / artifact / "stale.py"
        stale.parent.mkdir(parents=True)
        stale.write_text("x = 2\n", encoding="utf-8")  # on disk, never `git add`ed
    names = [p.relative_to(git_rocke).as_posix() for p in adv._all_files()]
    assert names == [
        "library/build/real.py",
        "library/kernel.py",
        "library/k\u00e9rnel.py",
        "library/venv_probe.py",
        "newtree/tool.py",
        "platform/python/rocke/ir.py",
    ]


def test_whole_run_lists_this_checkouts_own_python(adv, require_git):
    # The real tree, not a scratch repository: the listing finds the wrapper itself.
    try:
        files = adv._all_files()
    except RuntimeError:
        pytest.skip("rocke is not in a git checkout")
    assert _SCRIPT in files


def test_whole_run_lists_a_conflicted_file_once(adv, tmp_path, monkeypatch):
    # `git ls-files` prints a path once per index stage: three times for a file with
    # an unresolved merge conflict. radon would report it three times.
    (tmp_path / "library").mkdir()
    (tmp_path / "library" / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(adv, "ROCKE", tmp_path)
    listing = b"library/a.py\0" * 3
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, listing, b""),
    )
    assert adv._all_files() == [tmp_path / "library" / "a.py"]


def test_whole_run_drops_a_tracked_file_deleted_from_disk(adv, git_rocke):
    # vulture exits on a path that does not exist, and radon cannot read one.
    _track(git_rocke, {"library/kept.py": "x = 1\n", "library/gone.py": "x = 2\n"})
    (git_rocke / "library" / "gone.py").unlink()  # deleted, still in the index
    assert adv._all_files() == [git_rocke / "library" / "kept.py"]


def test_whole_run_drops_a_tracked_dangling_symlink(adv, git_rocke):
    _track(git_rocke, {"library/kept.py": "x = 1\n"})
    try:
        (git_rocke / "library" / "link.py").symlink_to("no-such-file")
    except OSError:
        pytest.skip("cannot create symlinks here")
    _git(git_rocke, "add", "--force", "-A")
    assert adv._all_files() == [git_rocke / "library" / "kept.py"]


def test_file_list_ignores_the_repository_a_linked_worktree_hook_exports(
    adv, require_git, monkeypatch, tmp_path
):
    # A hook run from a linked worktree gets an absolute GIT_DIR (see
    # _GIT_REPO_ENV in the wrapper): without the wrapper dropping it, the file
    # list would be the whole repository, not rocke.
    main = tmp_path / "main"
    (main / "rocke").mkdir(parents=True)
    (main / "rocke" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (main / "elsewhere").mkdir()
    (main / "elsewhere" / "b.py").write_text("x = 2\n", encoding="utf-8")
    _git(main, "init", "-q")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "--no-verify", "-m", "init")
    worktree = tmp_path / "worktree"
    _git(main, "worktree", "add", "-q", "-b", "linked", str(worktree))
    git_dir = (worktree / ".git").read_text(encoding="utf-8").split("gitdir:")[1]
    monkeypatch.setenv("GIT_DIR", git_dir.strip())
    monkeypatch.setattr(adv, "ROCKE", worktree / "rocke")
    assert adv._all_files() == [worktree / "rocke" / "a.py"]


def test_file_list_ignores_an_index_file_a_hook_exports(adv, git_rocke, monkeypatch):
    # `git commit` exports GIT_INDEX_FILE (a temporary index) to its hooks; an index
    # other than the repository's would give another file list.
    _track(git_rocke, {"library/kernel.py": "x = 1\n"})
    monkeypatch.setenv("GIT_INDEX_FILE", str(git_rocke / "no-such-index"))
    assert adv._all_files() == [git_rocke / "library" / "kernel.py"]


@pytest.mark.parametrize("name", ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"])
def test_git_starts_without_the_repository_variables_a_hook_exports(
    adv, git_rocke, monkeypatch, name
):
    # Whether one of these changes what `git ls-files` prints depends on the git
    # version and on which others are set with it (GIT_WORK_TREE alone changes
    # nothing visible), so the contract is checked at its boundary: none of them
    # reaches the git process.
    monkeypatch.setenv(name, str(git_rocke / "elsewhere"))
    seen = {}
    run = subprocess.run

    def spy(cmd, **kwargs):
        seen["env"] = kwargs["env"]
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    adv._all_files()
    assert name not in seen["env"]


def test_a_failing_git_is_advisory(adv, require_git, monkeypatch, tmp_path, capsys):
    # Not a repository: the whole-tree file list cannot be built, and that must be
    # reported, never raised. The ceiling stops git finding a repository above tmp.
    monkeypatch.setattr(adv, "ROCKE", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(sys, "argv", ["static_analysis_checks.py", "radon"])
    assert adv.main() == 0
    out = capsys.readouterr().out
    assert "radon: crashed (RuntimeError: git ls-files failed" in out


def test_an_unencodable_message_cannot_fail_a_push(adv, monkeypatch):
    # pre-commit pipes the output, so on Windows it is encoded strictly with the
    # ANSI code page. A message naming a path with a character outside it (here a
    # Polish letter, absent from cp1252) must still end in exit 0.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "argv", ["static_analysis_checks.py", "radon"])

    def broken_listing():
        raise RuntimeError("git ls-files failed: C:/Users/\u0141ukasz")

    monkeypatch.setattr(adv, "_all_files", broken_listing)
    assert adv.main() == 0
    stream.flush()
    assert b"radon: crashed" in raw.getvalue()


def test_requested_files_are_limited_to_python_inside_rocke(adv, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    requested = [
        str(outside),  # python, but outside rocke
        str(adv.ROCKE / "AGENTS.md"),  # inside rocke, but not python
        str(adv.ROCKE / "no_such_file.py"),  # does not exist
        # python inside rocke, spelled via "..": kept, as its resolved path
        str(adv.ROCKE / "library" / ".." / "tools" / "static_analysis" / _SCRIPT.name),
    ]
    assert adv._rocke_files(requested) == [_SCRIPT]


@pytest.fixture
def pyright_config():
    # The real rocke's config, even in a test that points `adv.ROCKE` elsewhere.
    if not _PYRIGHT_CONFIG.is_file():
        pytest.skip("pyrightconfig.json is not part of this layout")
    # pyright accepts // comments in its config; json does not.
    text = "\n".join(
        line
        for line in _PYRIGHT_CONFIG.read_text().splitlines()
        if not line.lstrip().startswith("//")
    )
    return json.loads(text)


def test_pyright_config_keeps_no_file_list_of_its_own(pyright_config):
    # The wrapper gives pyright the files git tracks, and that list replaces
    # `include`. An `exclude` would still filter it, and could drop tracked source.
    assert "include" not in pyright_config
    assert "exclude" not in pyright_config


def test_pyright_config_paths_exist_relative_to_the_config(pyright_config):
    # pyright reads every `root` and `extraPaths` entry relative to the config file's
    # own directory, so moving the file silently breaks the import roots unless the
    # paths move with it.
    base = _PYRIGHT_CONFIG.parent
    for env in pyright_config["executionEnvironments"]:
        for rel in [env["root"], *env.get("extraPaths", [])]:
            assert (
                base / rel
            ).is_dir(), f"{rel} (relative to {base}) is not a directory"


def test_pyright_whole_run_checks_what_git_tracks(
    adv, git_rocke, pyright_config, capsys
):
    pytest.importorskip("pyright")
    bad = "source: int = 'bad'\n"
    _track(
        git_rocke,
        {
            "library/source.py": bad,
            "library/venv_probe.py": bad,
            "library/build/real.py": bad,  # tracked, so source despite its directory
            # An import that only resolves through the config's import roots.
            "platform/python/rocke/__init__.py": "VALUE = 1\n",
            "library/kernels/uses_rocke.py": "import rocke\n",
            # ...and one that only resolves with rocke/ itself on the import path.
            "library/uses_root.py": "import library.source\n",
        },
    )
    config = git_rocke / adv.PYRIGHT_CONFIG
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(pyright_config), encoding="utf-8")
    for artifact in (".venv-gfx942", "venv-test", "node_modules"):
        stale = git_rocke / "library" / artifact / "stale.py"
        stale.parent.mkdir()
        stale.write_text(bad, encoding="utf-8")  # on disk, never `git add`ed
    adv.run_pyright(adv._all_files())
    out = capsys.readouterr().out
    for name in ("source.py", "venv_probe.py", "real.py"):
        assert f"{name}:1: error:" in out, out
    assert "stale.py" not in out, out
    assert "pyright: 3 error(s), 0 warning(s)" in out, out
    assert "reportMissingImports" not in out, out  # `import rocke` resolved


def test_pyright_gets_its_file_names_on_stdin(adv, monkeypatch):
    # All of rocke is ~1,900 names (~98,000 characters) as arguments, past
    # Windows' 32,767-character command-line limit; on stdin the command line
    # stays the same length however many files there are.
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(
            cmd=cmd,
            cwd=kwargs.get("cwd"),
            stdin=kwargs.get("input"),
            encoding=kwargs.get("encoding"),
            env=kwargs.get("env"),
        )
        report = {
            "generalDiagnostics": [],
            "summary": {"errorCount": 0, "warningCount": 0},
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(report), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    files = [adv.ROCKE / "library" / f"kernel_{i:04d}.py" for i in range(3000)]
    adv.run_pyright(files[:1])
    command_for_one_file = seen["cmd"]
    adv.run_pyright(files)
    assert seen["cmd"] == command_for_one_file  # same command line for any file count
    assert seen["cmd"][-1] == "-"  # pyright reads its file list from stdin
    # The config sits in tools/static_analysis/, not at the project's cwd; losing it
    # would silently drop every import root.
    project = seen["cmd"][seen["cmd"].index("--project") + 1]
    assert project == str(
        adv.ROCKE / "tools" / "static_analysis" / "pyrightconfig.json"
    )
    # rocke/ stays an import root now that the config is not in it.
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(adv.ROCKE)
    assert seen["cwd"] == adv.ROCKE
    # Node reads the names and writes the JSON as UTF-8; text=True would use the
    # locale's code page (cp1252 on Windows) and mangle a non-ASCII name.
    assert seen["encoding"] == "utf-8"
    # rocke-relative, one per line, in order
    lines = seen["stdin"].splitlines()
    assert lines[0] == str(Path("library", "kernel_0000.py"))
    assert len(lines) == 3000


def test_pyright_keeps_the_callers_pythonpath_after_rocke(adv, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["env"] = kwargs["env"]
        report = {
            "generalDiagnostics": [],
            "summary": {"errorCount": 0, "warningCount": 0},
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(report), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/caller/path")
    adv.run_pyright([adv.ROCKE / "library" / "kernel.py"])
    assert seen["env"]["PYTHONPATH"] == os.pathsep.join(
        [str(adv.ROCKE), "/caller/path"]
    )


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
    adv.run_pyright([_SCRIPT])
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
    monkeypatch.setattr(
        sys, "argv", ["static_analysis_checks.py", "pyright", str(_SCRIPT)]
    )
    assert adv.main() == 0
    out = capsys.readouterr().out
    assert "startup failed\nNo module named pyright\n" in out
    assert "error(s)" not in out


def _hook_repo(tmp_path, tool):
    """A scratch repository with the wrapper and the repo's own ``rocke-<tool>`` hook.

    The hook definition comes from the repository's ``.pre-commit-config.yaml``, so
    these tests run what a contributor would run. Returns ``(repo, hook)``.
    """
    pytest.importorskip("pre_commit")
    if shutil.which("bash") is None:
        pytest.skip("the hook entry is a bash snippet")
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
    repo = tmp_path / "repo"
    repo.mkdir()
    script = Path(
        "dnn-providers/hip-kernel-provider/rocke/tools/static_analysis"
        "/static_analysis_checks.py"
    )
    (repo / script).parent.mkdir(parents=True)
    shutil.copyfile(_SCRIPT, repo / script)
    (repo / ".pre-commit-config.yaml").write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [hook]}]}),
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    return repo, hook


def _run_hook(repo, hook, env):
    """Run the hook the way ``git push`` would; returns ``(exit code, output)``."""
    env = {k: v for k, v in env.items() if k != "SKIP"}  # pre-commit's skip list
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
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("tool", ["pyright", "radon", "vulture"])
def test_system_hook_with_cold_cache_and_missing_tools(
    adv, require_git, tmp_path, tool
):
    repo, hook = _hook_repo(tmp_path, tool)
    assert hook["language"] == "system"
    assert not hook.get("additional_dependencies")
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
    returncode, out = _run_hook(repo, hook, env)
    assert returncode == 0, out
    assert "[advisory, non-blocking]" in out
    assert "not installed" in out or "No module named pyright" in out
    assert "Installing environment" not in out


@pytest.mark.parametrize("tool", ["pyright", "radon", "vulture"])
@pytest.mark.parametrize(
    ("programs", "ran"),
    [
        pytest.param({}, False, id="no-python"),
        pytest.param({"python3": "real"}, True, id="python3-only"),
        # What the Windows Store stub does: `python` exists, but only fails.
        pytest.param({"python": "#!/bin/sh\nexit 9\n"}, False, id="unusable-python"),
        # The probe checks the version, not just that the program starts.
        pytest.param({"python": "old"}, False, id="old-python"),
        # "The first that works": a bad `python` must not hide a good `python3`.
        pytest.param(
            {"python": "#!/bin/sh\nexit 9\n", "python3": "real"},
            True,
            id="unusable-python-then-python3",
        ),
    ],
)
def test_hook_runs_a_usable_python_or_skips_but_never_fails(
    adv, require_git, tmp_path, tool, programs, ran
):
    # pre-commit fails a system hook whose program is absent, before the wrapper
    # could skip anything; the hook entry is a bash snippet that does the skipping.
    if os.name == "nt":
        pytest.skip("builds a PATH of symlinks")
    repo, hook = _hook_repo(tmp_path, tool)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for needed in ("bash", "git"):
        (bin_dir / needed).symlink_to(shutil.which(needed))
    for name, body in programs.items():
        program = bin_dir / name
        if body == "real":
            # A working Python with no analyzers: an empty virtual environment, as in
            # the cold-cache test. A symlink to the base interpreter would see whatever
            # is installed system-wide.
            empty_env = tmp_path / "empty-env"
            venv.EnvBuilder(with_pip=False).create(empty_env)
            body = f'#!/bin/sh\nexec "{empty_env / "bin" / "python"}" "$@"\n'
        elif body == "old":
            # Reports itself as Python 3.9; only the version probe ever runs it.
            fake = "import sys; sys.version_info = (3, 9, 0, 'final', 0)"
            body = f'#!/bin/sh\nexec "{sys.executable}" -c "{fake}\n$2"\n'
        program.write_text(body, encoding="utf-8")
        program.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir),
        "PRE_COMMIT_HOME": str(tmp_path / "cold-cache"),
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
    }
    env.pop("VIRTUAL_ENV", None)
    returncode, out = _run_hook(repo, hook, env)
    assert returncode == 0, out
    skip_note = f"rocKE advisory ({tool}): no Python 3.10+ on PATH; skipped."
    if ran:
        assert f"[advisory, non-blocking] {tool} on" in out
        # `exec` hands the hook over to the wrapper. Without it the snippet runs on
        # after the wrapper: it would try the next python, then claim it skipped.
        assert out.count("[advisory, non-blocking]") == 1, out
        assert skip_note not in out, out
    else:
        assert skip_note in out
        assert "[advisory, non-blocking]" not in out
        assert "not found" not in out  # the probe's own errors stay hidden


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
    def runner(files):
        raise failure

    monkeypatch.setitem(adv.RUNNERS, "vulture", runner)
    monkeypatch.setattr(
        sys, "argv", ["static_analysis_checks.py", "vulture", str(_SCRIPT)]
    )
    assert adv.main() == 0
    assert reported in capsys.readouterr().out


def test_nothing_runs_when_no_file_is_in_rocke(adv, monkeypatch, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    calls = []
    monkeypatch.setitem(adv.RUNNERS, "radon", lambda files: calls.append(files))
    monkeypatch.setattr(
        sys, "argv", ["static_analysis_checks.py", "radon", str(outside)]
    )
    assert adv.main() == 0
    assert calls == []


@pytest.fixture
def mini_rocke(git_rocke):
    """A rocke-shaped git repository small enough to scan quickly."""
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
    _track(git_rocke, tree)
    return git_rocke


def test_vulture_counts_uses_across_trees_and_tests(adv, mini_rocke, capsys):
    adv.run_vulture(adv._all_files())
    out = capsys.readouterr().out
    # cross_tree_helper is used from another tree and build() only from a test:
    # neither is dead. Nothing is reported from inside the tests.
    assert re.findall(r"unused \w+ '(\w+)'", out) == [
        "orphan_in_library",
        "orphan_in_platform",
    ]
    assert "vulture: 2 finding(s)" in out


def test_vulture_untracked_files_cannot_hide_unused_code(adv, mini_rocke, capsys):
    # A stale build copy that calls the orphans would hide them if it were scanned.
    stale = mini_rocke / "library" / "build" / "lib" / "stale.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("orphan_in_library()\norphan_in_platform()\n", encoding="utf-8")
    adv.run_vulture(adv._all_files())
    out = capsys.readouterr().out
    assert "orphan_in_library" in out
    assert "orphan_in_platform" in out
    assert "vulture: 2 finding(s)" in out


def test_vulture_scans_a_requested_file_git_does_not_track_yet(adv, mini_rocke, capsys):
    # A new file named on the command line is scanned and reported too; left out of
    # the scan it would print "0 finding(s)" for code that has some.
    new = mini_rocke / "library" / "new_kernel.py"
    new.write_text("def orphan_in_new_file():\n    return 5\n", encoding="utf-8")
    adv.run_vulture(adv._rocke_files([str(new)]))
    out = capsys.readouterr().out
    assert re.findall(r"unused \w+ '(\w+)'", out) == ["orphan_in_new_file"]


def test_vulture_reports_only_the_requested_files(adv, mini_rocke, monkeypatch, capsys):
    # Paths as typed (relative, through ".."), while vulture reports resolved
    # ones: if the two spellings did not match, the report would silently be
    # empty. A requested test file is still not reported on.
    monkeypatch.chdir(mini_rocke / "library")
    helpers = str(Path("..", "platform", "helpers.py"))
    test_file = str(Path("tests", "test_kernel.py"))
    adv.run_vulture(adv._rocke_files([helpers, test_file]))
    out = capsys.readouterr().out
    assert re.findall(r"unused \w+ '(\w+)'", out) == ["orphan_in_platform"]
    # Located relative to the cwd, with its line, so an editor can jump to it.
    assert f"{helpers}:5: unused function 'orphan_in_platform' (60% confidence)" in out
    assert "vulture: 1 finding(s)" in out


def test_vulture_scan_launches_only_git(adv, mini_rocke, monkeypatch, capsys):
    # The real tree is ~1,900 files; as arguments to a child process their names
    # make a command line of over 98,000 characters, past Windows' 32,767 limit.
    # Only `git ls-files` (a fixed, short command line) may run.
    popen = subprocess.Popen

    def only_git(cmd, *args, **kwargs):
        if cmd[0] != "git":
            raise AssertionError(f"vulture must not be launched as {cmd[0]!r}")
        return popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", only_git)
    adv.run_vulture(adv._all_files())
    # The scan really ran: a launch whose error was swallowed reports nothing.
    assert "vulture: 2 finding(s)" in capsys.readouterr().out


def test_a_missing_vulture_is_reported_not_silently_empty(adv, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "vulture", None)  # `import vulture` fails
    monkeypatch.setattr(
        sys, "argv", ["static_analysis_checks.py", "vulture", str(_SCRIPT)]
    )
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
    adv.run_radon([sample])
    out = capsys.readouterr().out
    assert "Most complex functions" in out
    # 12 branches + 1 = complexity 13, rank C.
    assert re.search(r"\bC\s+13\s+\S*sample\.py:1\s+busy\b", out)


def test_radon_report_is_quiet_for_simple_code(adv, tmp_path, capsys):
    pytest.importorskip("radon")
    sample = tmp_path / "simple.py"
    sample.write_text("def f(a):\n    return a\n", encoding="utf-8")
    adv.run_radon([sample])
    out = capsys.readouterr().out
    assert "avg CC 1.0 (A)" in out
    assert "Most complex functions" not in out
