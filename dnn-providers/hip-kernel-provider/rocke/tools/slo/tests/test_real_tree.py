# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Against the real rocke tree.

1. The committed baseline loads and `check` runs on today's tree.
2. Breaking real kernels on purpose produces the right check id.
"""
import ast
import json
import shutil
import subprocess

import pytest
from conftest import ROCKE

import slo
import slo_measure as MS

# Folders a copy needs: the measured code, the helpers, and the IR harness.
COPY = ("library", "platform/python", "platform/tests")


def test_committed_baseline_runs_on_the_tree(tmp_path):
    out = tmp_path / "f.json"
    assert slo.main(["check", "--json", str(out)]) == 0
    assert all(f["kind"] != "error" for f in json.loads(out.read_text())["findings"])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A copy of the real tree in its own git repo, with a fresh baseline."""
    top = tmp_path_factory.mktemp("mut")
    root = top / "rocke"
    for d in COPY:
        shutil.copytree(
            ROCKE / d, root / d, ignore=shutil.ignore_patterns("__pycache__")
        )
    for a in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(top), *a], check=True, capture_output=True)
    bl = top / "baseline.json"
    assert (
        slo.main(["baseline", "--update", "--root", str(root), "--baseline", str(bl)])
        == 0
    )
    files = MS.measure(root)["files"]
    # the file with the longest function, and a mid-size kernel file to grow
    longest = max(files.values(), key=lambda r: r["longest_fn"])
    mid = min(
        (
            r
            for r in files.values()
            if 300 <= r["sloc"] <= 600 and r["path"] != longest["path"]
        ),
        key=lambda r: r["path"],
    )
    return root, bl, longest["path"], longest["longest_fn_name"], mid["path"]


def run_check(root, bl, tmp_path, *extra):
    out = tmp_path / "f.json"
    rc = slo.main(
        [
            "check",
            "--root",
            str(root),
            "--baseline",
            str(bl),
            "--json",
            str(out),
            *extra,
        ]
    )
    return rc, json.loads(out.read_text())["findings"]


def kinds(findings):
    return {(f["check"], f["kind"], f["path"]) for f in findings}


def test_mutations_are_caught(tree, tmp_path):
    root, bl, long_rel, long_fn, mid_rel = tree
    pasted = str(mid_rel).rsplit("/", 1)[0] + "/pasted.py"
    mid, long_ = root / mid_rel, root / long_rel
    mid_src, long_src = mid.read_text(), long_.read_text()
    try:
        # 1. a new function full of raw builder calls
        dead = (
            "\n\ndef _never_built(b, x):\n"
            + "".join(f"    x = b.add(x, b.const_i32({i}))\n" for i in range(200))
            + "    return x\n"
        )
        mid.write_text(mid_src + dead)
        # 2. the longest function grows by one line
        long_.write_text(_grow_function(long_src, long_fn))
        # 3. a copy-pasted new file
        (root / pasted).write_text(long_src)

        rc, f = run_check(root, bl, tmp_path, "--base", "HEAD")
        k = kinds(f)
        assert rc == 0  # warn only
        assert ("SLO101", "worse", mid_rel) in k
        assert ("SLO103", "worse", mid_rel) in k
        assert ("SLO107", "worse", mid_rel) in k  # const pattern count rose
        assert ("SLO102", "worse", long_rel) in k
        assert ("SLO106", "over_target", pasted) in k
        assert all(x["path"] in (mid_rel, long_rel, pasted) for x in f)

        rc, _ = run_check(root, bl, tmp_path, "--fail-on", "increase")
        assert rc == 1
    finally:
        mid.write_text(mid_src)
        long_.write_text(long_src)
        (root / pasted).unlink(missing_ok=True)

    rc, f = run_check(root, bl, tmp_path, "--fail-on", "increase")
    assert (rc, f) == (0, [])


def _grow_function(src: str, name: str) -> str:
    """Add one statement as the first line of `name`'s body."""
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    lines = src.splitlines(keepends=True)
    first = fn.body[0]
    line = lines[first.lineno - 1]
    indent = line[: len(line) - len(line.lstrip())]
    is_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    at = first.end_lineno if is_doc else first.lineno - 1
    lines.insert(at, f"{indent}_slo_grown = 0\n")
    return "".join(lines)
