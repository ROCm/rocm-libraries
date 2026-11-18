#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that the Python bindings bind every enum value in hipBLASLt's public headers.

This is the `hipblaslt-python-enum-sync` pre-commit hook. It compares the enums in
- projects/hipblaslt/library/include/hipblaslt/hipblaslt.h,
- projects/hipblaslt/library/include/hipblaslt/hipblaslt-ext.hpp (GemmType),
- projects/hipblas-common/library/include/hipblas-common/hipblas-common.h
with the reg(...) calls in projects/hipblaslt/python/src/enums.cpp and ALLOWED_MISSING in
projects/hipblaslt/python/hipblaslt/_coverage.py. It needs only the standard library and builds
nothing, so it flags a new enum value on the PR that adds it.

CI's pre-commit checkout is sparse: it has only the projects a PR changed. Files that aren't on
disk are read with `git show HEAD:<path>`, which the blobless clone fetches on demand. A file that
can't be read either way is an error, never a silent pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

HEADERS = [
    "projects/hipblaslt/library/include/hipblaslt/hipblaslt.h",
    "projects/hipblaslt/library/include/hipblaslt/hipblaslt-ext.hpp",
    "projects/hipblas-common/library/include/hipblas-common/hipblas-common.h",
]
ENUMS_CPP = "projects/hipblaslt/python/src/enums.cpp"
COVERAGE = "projects/hipblaslt/python/hipblaslt/_coverage.py"

# Enums the bindings own in these headers.
_COMMON = {"hipblasStatus_t", "hipblasOperation_t", "hipblasComputeType_t", "GemmType"}


def owned(enum_name: str) -> bool:
    return enum_name.startswith("hipblasLt") or enum_name in _COMMON


def read(repo: Path, rel: str) -> str:
    path = repo / rel
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        return subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            f"check_enum_sync: can't read {rel} from disk or from git ({e})"
        ) from e


def load_coverage(repo: Path) -> ModuleType:
    """Load _coverage.py by path; importing the hipblaslt package would load _core."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_coverage.py"
        path.write_text(read(repo, COVERAGE), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("hipblaslt_coverage", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def _suggested_python_name(c_name: str, bound: dict) -> str:
    """Guess the Python name from a sibling: HIPBLASLT_EPILOGUE_RELU is RELU, so strip that prefix."""
    for other_c, b in bound.items():
        if other_c.endswith(b.py_name) and other_c != b.py_name:
            prefix = other_c[: -len(b.py_name)]
            if c_name.startswith(prefix) and c_name[len(prefix) :][:1].isalpha():
                return c_name[len(prefix) :]
    return c_name


def check(headers: dict[str, str], enums_cpp: str, coverage: ModuleType) -> list[str]:
    """Every problem found, as messages that name the value and the fix."""
    problems = []
    bound_all = coverage.bound_values(enums_cpp)
    allowed_all = coverage.ALLOWED_MISSING
    seen = set()
    for header_path, text in headers.items():
        header = header_path.rsplit("/", 1)[-1]
        for enum_name, enum in coverage.parse_enums(text).items():
            if not owned(enum_name):
                continue
            seen.add(enum_name)
            bound = bound_all.get(enum_name, {})
            allowed = allowed_all.get(enum_name, {})
            header_names = set()
            for v in enum.values:
                header_names.add(v.name)
                if v.name in allowed:
                    if v.name in bound:
                        problems.append(
                            f"{v.name} is both bound in {ENUMS_CPP} and in ALLOWED_MISSING in "
                            f"{COVERAGE}. Remove one."
                        )
                    continue
                if v.name not in bound:
                    py = _suggested_python_name(v.name, bound)
                    guard = f" inside `#{v.guard}`" if v.guard else ""
                    problems.append(
                        f"{header} has `{v.name} = {v.value}`{guard} in {enum_name}, and the Python "
                        f'bindings don\'t bind it. Add `reg(e, "{enum_name}", "{py}", {v.name});` to '
                        f"{ENUMS_CPP}{' under the same guard' if v.guard else ''}, or add it to "
                        f"ALLOWED_MISSING in {COVERAGE} with a reason."
                    )
                elif bound[v.name].guard != v.guard:
                    want = f"`#{v.guard}`" if v.guard else "no guard"
                    have = (
                        f"`#{bound[v.name].guard}`"
                        if bound[v.name].guard
                        else "no guard"
                    )
                    problems.append(
                        f"{v.name} is under {want} in {header}, but under {have} in {ENUMS_CPP}. "
                        "Bind it under the same condition, or the default build won't compile."
                    )
            for name in sorted(set(bound) - header_names):
                problems.append(
                    f"{ENUMS_CPP} binds {enum_name}.{name}, which {header} doesn't have any more. "
                    "Remove its reg(...) line."
                )
            for name in sorted(set(allowed) - header_names):
                problems.append(
                    f"ALLOWED_MISSING in {COVERAGE} lists {enum_name}.{name}, which {header} doesn't have."
                )
    for enum_name in sorted(n for n in bound_all if owned(n) and n not in seen):
        problems.append(
            f"{ENUMS_CPP} binds {enum_name}, which none of the checked headers defines."
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="rocm-libraries checkout to check (default: the one this script is in)",
    )
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    coverage = load_coverage(repo)
    problems = check(
        {h: read(repo, h) for h in HEADERS}, read(repo, ENUMS_CPP), coverage
    )
    for p in problems:
        print(p)
    if problems:
        print(
            f"\n{len(problems)} problem(s). See 'Python bindings enum sync' in projects/hipblaslt/TESTING.md."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
