# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SLO201 / SLO202: how much kernel code the IR golden harness actually runs.

Runs the harness once under coverage.py, in a subprocess, with the same
interpreter. coverage.py is the only non-stdlib dependency of the SLO tool, and
only these checks need it.

Two numbers per file:
  cov_file  every statement coverage.py sees. Too kind:
            `def` lines, class bodies and constants run on import even when no
            kernel is built.
  cov_body  only statements inside function bodies. This is the one we track.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import slo_config as C


class CoverageUnavailable(RuntimeError):
    pass


def _body_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            lines.update(range(n.body[0].lineno, n.end_lineno + 1))
    return lines


def run(root: Path, files: list[str], python: str | None = None, scope=None) -> dict:
    """Return {"harness_ok": bool, "harness_tail": str, "files": {rel: {...}}}."""
    try:
        import coverage
    except ImportError as e:  # the static checks still run without it
        raise CoverageUnavailable(
            "coverage.py is not installed (pip install coverage)"
        ) from e

    python = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="slo_cov_") as tmp:
        data = Path(tmp) / ".coverage"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / p) for p in C.PYTHONPATH]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        cmd = [
            python,
            "-m",
            "coverage",
            "run",
            f"--data-file={data}",
            "--source=" + ",".join(str(root / s) for s in (scope or C.SCOPE)),
            str(root / C.HARNESS),
            "--check",
            str(root / C.HARNESS_GOLDEN),
            "--flavor",
            C.HARNESS_FLAVOR,
        ]
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        if not data.exists():
            raise CoverageUnavailable(
                "harness produced no coverage data: " + " | ".join(tail)
            )
        cov = coverage.Coverage(data_file=str(data))
        cov.load()
        out = {}
        for rel in files:
            path = root / rel
            try:
                _, stmts, _, missing, _ = cov.analysis2(str(path))
            except coverage.CoverageException:
                continue
            stmts, missing = set(stmts), set(missing)
            ran = stmts - missing
            body = stmts & _body_lines(
                ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            )
            out[rel] = {
                "cov_file": round(100 * len(ran) / len(stmts), 1) if stmts else None,
                "cov_body": (
                    round(100 * len(body & ran) / len(body), 1) if body else None
                ),
                "missing_body": sorted(body - ran),
                "executed": sorted(ran),
                "statements": sorted(stmts),
            }
    return {
        "harness_ok": proc.returncode == 0,
        "harness_tail": "\n".join(tail),
        "files": out,
    }
