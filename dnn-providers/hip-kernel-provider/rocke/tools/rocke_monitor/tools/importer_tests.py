# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The test files that import the Python modules a PR changes, for the checks' "Tests that import the changed modules"
section: the PR's own tests ran already, and a test the PR does not touch can still break on its change (and the
failure would otherwise show only in CI's logs).

A module is a changed `.py` file under `platform/python/` (imported as `rocke...`) or `library/` (on the tests' path as
the library root), not a test file or a conftest. A test file (`test_*.py` anywhere in the rocKE tree) imports it when a
line reads `import <module>`, `from <module> import ...`, or `from <package> import ... <name> ...` (parenthesised over
lines too); a changed package's `__init__.py` counts for `import <package>` and `from <package> import`. The PR's own
test files (`--exclude`) are left out. The files that import the most changed modules come first, then the smallest;
at most `--cap` run, and the rest are named as not run.

Usage: python3 -I importer_tests.py --tree ROCKE_ROOT --list FILE [--map JSON] [--cap 12] [--exclude FILE ...] -- MODULE ...
Prints the selection; writes the chosen files (relative to ROCKE_ROOT, space-separated) to FILE, and with --map each
chosen file's changed modules to JSON.
       python3 -I importer_tests.py --fails RUN_DIR/tests_pr_failed.txt --map JSON
After the run (tools/pytest_vs_base.py): one `IMPORTERS_FAIL <file> (<module>, ...)` line per importer that fails on
the merge only (develop passes it), naming the changed modules it imports.
"""
import argparse
import json
import pathlib
import re
import sys

CAP = 12
ROOTS = ("platform/python/", "library/")  # the import roots on the tests' path
TEST_FILE = re.compile(r"(^|/)test_[^/]*\.py$")
SHOWN = 10


def dotted(path: str) -> tuple[str, bool] | None:
    """The module name a changed file is imported by, and whether it is a package's __init__; None for a file that is
    no importable module (a test, a conftest, a file outside the import roots)."""
    if (
        not path.endswith(".py")
        or TEST_FILE.search(path)
        or path.endswith("conftest.py")
    ):
        return None
    for root in ROOTS:
        if path.startswith(root):
            rel = path[len(root) : -len(".py")]
            init = rel.endswith("/__init__") or rel == "__init__"
            rel = rel[: -len("/__init__")] if rel.endswith("/__init__") else rel
            name = rel.replace("/", ".")
            return (name, init) if name and name != "__init__" else None
    return None


def patterns(name: str, init: bool) -> list[re.Pattern]:
    """The import statements that reach module `name` (a package's __init__ when `init`)."""
    esc = re.escape(name)
    out = [
        re.compile(rf"^[ \t]*import[ \t]+[^\n#]*\b{esc}\b(?![.\w])", re.M),
        re.compile(rf"^[ \t]*from[ \t]+{esc}[ \t]+import\b", re.M),
    ]
    if "." in name:
        parent, stem = name.rsplit(".", 1)
        out.append(
            re.compile(
                rf"^[ \t]*from[ \t]+{re.escape(parent)}[ \t]+import[ \t]+(?:\([^)]*|[^\n]*)"
                rf"\b{re.escape(stem)}\b",
                re.M,
            )
        )
    if not init:
        out.append(re.compile(rf"^[ \t]*import[ \t]+[^\n#]*\b{esc}\b", re.M))
    return out


def select(
    tree: pathlib.Path, modules: list[str], exclude: set[str], cap: int = CAP
) -> dict:
    """The importing test files: {"modules": [...], "chosen": [...], "rest": [...], "by_file": {file: [module, ...]}}."""
    mods = {m: d for m in modules if (d := dotted(m))}
    pats = {m: patterns(*d) for m, d in mods.items()}
    by_file: dict[str, list[str]] = {}
    sizes: dict[str, int] = {}
    for p in sorted(tree.rglob("test_*.py")):
        rel = str(p.relative_to(tree))
        if rel in exclude or "/." in "/" + rel:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hit = [m for m, ps in pats.items() if any(x.search(text) for x in ps)]
        if hit:
            by_file[rel], sizes[rel] = hit, len(text)
    order = sorted(by_file, key=lambda f: (-len(by_file[f]), sizes[f], f))
    return {
        "modules": sorted(mods),
        "chosen": order[:cap],
        "rest": order[cap:],
        "by_file": by_file,
    }


def report(sel: dict, cap: int) -> list[str]:
    mods = sel["modules"]
    out = [
        f"changed modules: {len(mods)}"
        + (f" ({', '.join(mods[:6])}{' …' if len(mods) > 6 else ''})" if mods else "")
    ]
    total = len(sel["chosen"]) + len(sel["rest"])
    out.append(
        f"test files the PR does not change that import them: {total}; run {len(sel['chosen'])} (cap {cap}), "
        "the most changed modules first, then the smallest"
    )
    for f in sel["chosen"]:
        out.append(
            f"  {f} (imports {', '.join(m.rsplit('/', 1)[-1] for m in sel['by_file'][f][:3])}"
            f"{' …' if len(sel['by_file'][f]) > 3 else ''})"
        )
    if sel["rest"]:
        out.append(
            f"NOT RUN (over the cap, {len(sel['rest'])}): "
            + ", ".join(sel["rest"][:SHOWN])
            + (
                f" (+{len(sel['rest']) - SHOWN} more)"
                if len(sel["rest"]) > SHOWN
                else ""
            )
        )
    return out


RUN_ERROR = re.compile(r"^RUN-ERROR\(rc=\d+\):")


def failed_files(failed: pathlib.Path) -> list[str]:
    """The test files of a pytest_vs_base.py run's `tests_pr_failed.txt` (the ids failing on the merge only, and a
    batch pytest itself stopped on as `RUN-ERROR(rc=N):<file>`), in their first order; [] when there is no such file.
    """
    try:
        lines = failed.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[str] = []
    for ln in lines:
        f = RUN_ERROR.sub("", ln.strip()).split("::", 1)[0].strip()
        if f and f not in out:
            out.append(f)
    return out


def fail_lines(failed: pathlib.Path, by_file: dict[str, list[str]]) -> list[str]:
    """`IMPORTERS_FAIL <file> (<module>, ...)` for each importer failing on the merge only, its changed modules dotted."""
    out = []
    for f in failed_files(failed):
        mods = [d[0] for m in by_file.get(f, []) if (d := dotted(m))]
        out.append(f"IMPORTERS_FAIL {f} ({', '.join(mods) or 'a changed module'})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree")
    ap.add_argument("--list")
    ap.add_argument(
        "--map", help="write {file: [changed module, ...]} of the chosen files as JSON"
    )
    ap.add_argument(
        "--fails",
        help="a run's tests_pr_failed.txt: print IMPORTERS_FAIL lines (with --map) and nothing else",
    )
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("modules", nargs="*")
    a = ap.parse_args(argv)
    if a.fails:
        try:
            by_file = (
                json.loads(pathlib.Path(a.map).read_text(encoding="utf-8"))
                if a.map
                else {}
            )
        except (OSError, ValueError):
            by_file = {}
        lines = fail_lines(pathlib.Path(a.fails), by_file)
        if lines:
            print("\n".join(lines))
        return 0
    if not (a.tree and a.list):
        ap.error("--tree and --list are needed (or --fails)")
    sel = select(pathlib.Path(a.tree), a.modules, set(a.exclude), a.cap)
    pathlib.Path(a.list).write_text(" ".join(sel["chosen"]) + "\n", encoding="utf-8")
    if a.map:
        pathlib.Path(a.map).write_text(
            json.dumps({f: sel["by_file"][f] for f in sel["chosen"]}, indent=1) + "\n",
            encoding="utf-8",
        )
    print("\n".join(report(sel, a.cap)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
