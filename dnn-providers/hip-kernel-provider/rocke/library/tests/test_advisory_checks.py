# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for ``rocke/tools/advisory_checks.py``.

That script is the non-blocking pre-push wrapper around pyright, radon and vulture.
These tests pin down its contract: it covers ALL rocke Python, it never fails a
push, and its radon report lists nested functions (radon scores them separately
from their parent). The radon tests skip when radon is not installed.
"""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
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
    ],
)
def test_artifact_dirs_are_skipped(adv, parts, skipped):
    assert adv._skipped(adv.ROCKE.joinpath(*parts)) is skipped


def test_whole_run_spans_every_tree(adv):
    top = {p.relative_to(adv.ROCKE).parts[0] for p in adv._all_files()}
    assert top == set(adv.TREES)


def test_requested_files_are_limited_to_python_inside_rocke(adv, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    requested = [
        str(outside),  # python, but outside rocke
        str(adv.ROCKE / "AGENTS.md"),  # inside rocke, but not python
        str(adv.ROCKE / "no_such_file.py"),  # does not exist
        str(_SCRIPT),
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


def test_pyright_includes_the_same_trees(adv):
    config = adv.ROCKE / "pyrightconfig.json"
    if not config.is_file():
        pytest.skip("pyrightconfig.json is not part of this layout")
    # pyright accepts // comments in its config; json does not.
    text = "\n".join(
        line
        for line in config.read_text().splitlines()
        if not line.lstrip().startswith("//")
    )
    assert json.loads(text)["include"] == list(adv.TREES)


@pytest.mark.parametrize(
    ("failure", "reported"),
    [
        (RuntimeError("tool exploded"), "crashed"),
        (ImportError("No module named 'vulture'"), "not installed"),
    ],
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
