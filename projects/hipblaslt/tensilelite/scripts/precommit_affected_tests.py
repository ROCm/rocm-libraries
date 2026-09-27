#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pre-commit runner for the TensileLite tests affected by staged changes.

Computes the staged file set via git, selects the affected unit +
characterization tests, and runs them with ``uv run pytest``; falls back to the
full suite when the set cannot be narrowed. Bypass with ``git commit --no-verify``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TL_REL = Path("projects/hipblaslt/tensilelite")
TESTS_REL = Path("Tensile/Tests/unit")
SRC_REL = Path("Tensile")

BROAD_TRIGGER_PARTS = (
    "conftest.py",
    "__snapshots__",
    "pyproject.toml",
    "tox.ini",
    "setup.py",
    "requirements",
)

MATCH_TOO_MANY_FRACTION = 0.40

IMPORT_MODULE_RE = re.compile(r"""import_module\(\s*["']([\w.]+)["']""")
DOTTED_STRING_RE = re.compile(r"""["'](Tensile\.[\w.]+)["']""")


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr)


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def staged_files(root: Path) -> list[Path]:
    """Return staged (added/copied/modified/renamed/deleted) paths, repo-relative.

    Deletions are included: removing a source module can break tests that import
    it, so those tests must still be selected. Deleted test files are dropped
    later (they cannot be run) by an existence check on the final target set.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD", "-z"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return [Path(p) for p in out.stdout.split("\0") if p]


def module_dotted(rel_to_tl: Path) -> str | None:
    """``Tensile/Common/Utilities.py`` -> ``Tensile.Common.Utilities``."""
    if rel_to_tl.suffix != ".py":
        return None
    parts = list(rel_to_tl.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def referenced_modules(test_file: Path) -> set[str]:
    """Dotted module names a test file references (imports, import_module, patch strings)."""
    refs: set[str] = set()
    try:
        text = test_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return refs
    try:
        tree = ast.parse(text, filename=str(test_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refs.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                if node.module:
                    refs.add(node.module)
                    for alias in node.names:
                        refs.add(f"{node.module}.{alias.name}")
    except SyntaxError:
        pass
    refs.update(IMPORT_MODULE_RE.findall(text))
    refs.update(DOTTED_STRING_RE.findall(text))
    return refs


def build_test_index(tests_root: Path) -> dict[Path, set[str]]:
    return {tf: referenced_modules(tf) for tf in tests_root.rglob("test_*.py")}


def tests_for_module(module: str, index: dict[Path, set[str]]) -> set[Path]:
    prefix = module + "."
    hits = set()
    for tf, refs in index.items():
        for r in refs:
            if r == module or r.startswith(prefix):
                hits.add(tf)
                break
    return hits


def failed_test_files(tl_root: Path) -> list[str]:
    """Test files that just failed, from pytest's last-failed cache (``[]`` if absent)."""
    cache = tl_root / ".pytest_cache" / "v" / "cache" / "lastfailed"
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted({str(nodeid).split("::", 1)[0] for nodeid in data})


def preflight_env(tl_root: Path) -> tuple[bool, str]:
    """Check the uv virtualenv can import pytest and pytest-xdist before running.

    Returns ``(ok, output)``. When ``ok`` is False, ``output`` is uv's combined
    stdout+stderr, used to diagnose *why* the environment is unusable. This runs
    the same ``uv run --no-sync`` path the real test command uses, so it surfaces
    a broken/unwritable .venv as an environment problem instead of letting the
    failure masquerade as a test failure. ``xdist`` is probed because the suite
    runs with ``-n 8``; a missing pytest-xdist would otherwise fail as a usage
    error rather than an environment problem.
    """
    probe = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", "import pytest, xdist"],
        cwd=tl_root,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return True, ""
    return False, (probe.stdout or "") + (probe.stderr or "")


def diagnose_env_failure(output: str, tl_root: Path) -> list[str]:
    """Turn an environment-probe failure into targeted remediation lines."""
    venv_disp = TL_REL / ".venv"
    tl_disp = TL_REL
    low = output.lower()

    owned_by_other = False
    if hasattr(os, "getuid"):
        try:
            venv = tl_root / ".venv"
            owned_by_other = venv.exists() and venv.stat().st_uid != os.getuid()
        except OSError:
            owned_by_other = False

    permission = "permission denied" in low or "os error 13" in low
    stale_interp = (
        "non-existent python interpreter" in low
        or "ignoring existing virtual environment" in low
    )
    missing_dep = "modulenotfounderror" in low or "no module named" in low
    recreate = [
        f"    rm -rf {venv_disp}",
        f"    (cd {tl_disp} && uv sync)",
    ]

    if owned_by_other or permission:
        return [
            f"The test virtualenv ({venv_disp}) is owned by another user, so uv",
            "cannot rebuild it -- it was almost certainly created inside a",
            "container. Recreate it from a shell that can delete it:",
            *recreate,
            "",
            "If 'rm' also reports 'Permission denied', the files are root-owned;",
            "delete them from the container that created them, or use sudo.",
        ]
    if stale_interp:
        return [
            f"The test virtualenv ({venv_disp}) points at a Python interpreter",
            "that no longer exists. Recreate it:",
            *recreate,
        ]
    if missing_dep:
        return [
            "The test virtualenv is missing required packages. Sync it:",
            f"    (cd {tl_disp} && uv sync)",
        ]
    return [
        "uv could not prepare the test environment (see its output above).",
        "Most environment issues are fixed by recreating the virtualenv:",
        *recreate,
    ]


def classify_staged(tl_staged: list[Path]):
    """Bucket tensilelite-relative staged paths into broad/test/source/ignored."""
    broad_reasons: list[str] = []
    changed_tests: set[Path] = set()
    changed_sources: list[Path] = []
    ignored: list[Path] = []
    for p in tl_staged:
        rel = p.relative_to(TL_REL)
        rel_str = str(rel)
        if any(part in rel_str for part in BROAD_TRIGGER_PARTS) or rel_str.startswith("scripts/"):
            broad_reasons.append(rel_str)
        elif rel.parts[:1] == ("rocisa",):
            broad_reasons.append(rel_str + " (native ext)")
        elif rel.parts[:3] == ("Tensile", "Tests", "unit") and rel.name.startswith("test_"):
            changed_tests.add(rel)
        elif rel.parts[:2] == ("Tensile", "Tests"):
            broad_reasons.append(rel_str + " (test support)")
        elif rel.suffix != ".py":
            ignored.append(rel)
        elif rel.parts[:1] == ("Tensile",):
            changed_sources.append(rel)
        else:
            ignored.append(rel)
    return broad_reasons, changed_tests, changed_sources, ignored


def select_tests(changed_sources: list[Path], index: dict[Path, set[str]]):
    """Map changed sources to tests; escalate when unmappable or matching too many."""
    selected: set[Path] = set()
    escalations: list[str] = []
    total = len(index) or 1
    for src in changed_sources:
        module = module_dotted(src)
        if not module:
            escalations.append(f"{src} (unmappable path)")
            continue
        hits = tests_for_module(module, index)
        if not hits:
            escalations.append(f"{src} -> {module} (no referencing tests)")
        elif len(hits) > MATCH_TOO_MANY_FRACTION * total:
            escalations.append(f"{src} -> {module} ({len(hits)}/{total} tests, too broad)")
        else:
            selected.update(hits)
    return selected, escalations


def main() -> int:
    root = repo_root()
    tl_root = root / TL_REL
    tests_root = tl_root / TESTS_REL

    staged = staged_files(root)
    tl_staged = [p for p in staged if str(p).startswith(str(TL_REL) + os.sep)]
    if not tl_staged:
        return 0

    broad_reasons, changed_tests, changed_sources, ignored = classify_staged(tl_staged)

    selected: set[Path] = set(changed_tests)
    escalations: list[str] = []
    if not broad_reasons:
        index = build_test_index(tests_root)
        sel, escalations = select_tests(changed_sources, index)
        selected.update(sel)

    selected = {p for p in selected if (tl_root / p).is_file()}

    run_full = bool(broad_reasons) or bool(escalations)

    log("[tensilelite-tests] " + "-" * 50)
    if broad_reasons:
        log("[tensilelite-tests] broad change -> full suite:")
        for r in broad_reasons:
            log(f"    {r}")
    if escalations:
        log("[tensilelite-tests] could not narrow -> full suite:")
        for r in escalations:
            log(f"    {r}")

    if run_full:
        nodes = [str(TESTS_REL)]
        log("[tensilelite-tests] running FULL unit + characterization suite")
    else:
        if not selected:
            log("[tensilelite-tests] no affected unit/char tests for staged changes")
            return 0
        nodes = sorted(
            os.path.relpath(p, tl_root) if Path(p).is_absolute() else str(p)
            for p in selected
        )
        log(f"[tensilelite-tests] running {len(nodes)} affected test file(s):")
        for n in nodes:
            log(f"    {n}")
    log("[tensilelite-tests] " + "-" * 50)

    if not shutil.which("uv"):
        log("[tensilelite-tests] ERROR: `uv` not found on PATH.")
        log("    Tests run via `uv run`; install uv or commit from an env that has it.")
        return 1

    env_ok, env_output = preflight_env(tl_root)
    if not env_ok:
        bar = "=" * 64
        log("")
        log(bar)
        log("  !  TENSILELITE TEST ENVIRONMENT NOT READY -- COMMIT BLOCKED")
        log(bar)
        log("  The tests never ran: uv could not prepare the virtualenv.")
        log("  THIS IS NOT A TEST FAILURE -- your changes are fine.")
        if env_output.strip():
            log("")
            log("  uv reported:")
            for ln in env_output.strip().splitlines():
                log("      " + ln)
        log("")
        for ln in diagnose_env_failure(env_output, tl_root):
            log("  " + ln)
        log("")
        log("  After fixing, re-run your commit. To bypass once: git commit --no-verify")
        log(bar)
        return 1

    # --no-sync: use the provisioned .venv without rewriting uv.lock mid-commit.
    # -n 8: fixed; -n auto = os.cpu_count() over-subscribes large CI/dev hosts.
    argv = ["uv", "run", "--no-sync", "pytest", "-q", "-ra", "-n", "8", *nodes]
    result = subprocess.run(argv, cwd=tl_root)
    rc = result.returncode
    if rc == 5:  # pytest: no tests collected
        log("[tensilelite-tests] no tests collected (treated as pass)")
        return 0
    if rc == 0:
        log("[tensilelite-tests] OK -- affected tests passed")
        return 0

    bar = "=" * 64
    update_targets = failed_test_files(tl_root) or nodes
    update_cmd = "uv run --no-sync pytest --snapshot-update " + " ".join(update_targets)
    log("")
    log(bar)
    log("  X  TENSILELITE TESTS FAILED (rc=%d) -- COMMIT BLOCKED" % rc)
    log(bar)
    log("  Scroll up for pytest's 'short test summary info' (the FAILED lines).")
    log("")
    log("  If a failure is an .ambr snapshot mismatch AND the new output is")
    log("  intentional, refresh the snapshot(s) with:")
    log("")
    log("      " + update_cmd)
    log("")
    log("  then review the diff ('git diff' the .ambr files) before committing.")
    log("  To bypass this hook once: git commit --no-verify")
    log(bar)
    return rc


if __name__ == "__main__":
    sys.exit(main())
