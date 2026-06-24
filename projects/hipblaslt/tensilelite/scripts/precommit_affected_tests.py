#!/usr/bin/env python3
"""Pre-commit runner for TensileLite unit + characterization tests.

Selects the tests affected by the staged changes and runs them. When the
affected set cannot be narrowed confidently, it runs the full unit +
characterization suite (the safe fallback).

Wired in from ``projects/hipblaslt/.pre-commit-config.yaml`` as an
``always_run`` / ``pass_filenames: false`` local hook, so this script computes
the staged file set itself (via git) rather than relying on pre-commit's file
filtering. That also means the monorepo root config's ``projects/hipblaslt/.*``
opt-out does not suppress it.

Tests run via ``uv run pytest`` from the tensilelite directory: uv provisions
the workspace env (deps + the rocisa native extension) from ``uv.lock`` and runs
pytest in it. Nothing else -- no backend selection, no tunables. To bypass the
hook for a single commit, use ``git commit --no-verify``.
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
    """Return staged (added/copied/modified/renamed) paths, repo-relative."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
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
                if node.level:  # relative import; not a Tensile.* module path
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
    """Test files that just failed, from pytest's last-failed cache.

    Keys are ``path::nodeid`` relative to the pytest rootdir (the tensilelite
    directory), so the bare paths are exactly what a follow-up ``uv run pytest``
    from that directory expects. Returns ``[]`` if the cache is absent or
    unreadable, so callers can fall back to the full run set.
    """
    cache = tl_root / ".pytest_cache" / "v" / "cache" / "lastfailed"
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted({str(nodeid).split("::", 1)[0] for nodeid in data})


def main() -> int:
    root = repo_root()
    tl_root = root / TL_REL
    tests_root = tl_root / TESTS_REL

    staged = staged_files(root)
    tl_staged = [p for p in staged if str(p).startswith(str(TL_REL) + os.sep)]
    if not tl_staged:
        return 0

    broad_reasons: list[str] = []
    changed_tests: set[Path] = set()
    changed_sources: list[Path] = []
    ignored: list[Path] = []

    for p in tl_staged:
        rel = p.relative_to(TL_REL)
        rel_str = str(rel)
        if any(part in rel_str for part in BROAD_TRIGGER_PARTS) or rel_str.startswith("scripts/"):
            broad_reasons.append(rel_str)
            continue
        if rel.parts[:1] == ("rocisa",):
            broad_reasons.append(rel_str + " (native ext)")
            continue
        if rel.suffix != ".py":
            ignored.append(rel)
            continue
        # Python file under tensilelite.
        if rel.parts[:3] == ("Tensile", "Tests", "unit") and rel.name.startswith("test_"):
            changed_tests.add(rel)
        elif rel.parts[:2] == ("Tensile", "Tests"):
            ignored.append(rel)  # non-unit tests (need GPU client); out of scope
        elif rel.parts[:1] == ("Tensile",):
            changed_sources.append(rel)
        else:
            ignored.append(rel)

    selected: set[Path] = set(changed_tests)
    escalations: list[str] = []

    if not broad_reasons:
        index = build_test_index(tests_root)
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

    # --no-sync: run in the already-provisioned .venv without syncing or
    # rewriting uv.lock. The hook must not mutate tracked files mid-commit:
    # pre-commit stashes unstaged changes first, so a sync here would re-resolve
    # against the reverted lockfile and rewrite uv.lock, which pre-commit then
    # flags as "files were modified by this hook". Provision once via `uv sync`.
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
    update_cmd = "uv run pytest --snapshot-update " + " ".join(update_targets)
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
