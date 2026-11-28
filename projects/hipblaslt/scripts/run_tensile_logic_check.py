#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run TensileLogic --check-all on library logic. Cross-platform (Windows and Unix).
#
# How to run (from hipblaslt project root):
#   python scripts/run_tensile_logic_check.py [--changed-files=FILE] [LIBLOGIC_PATH]
# --changed-files=FILE takes the changed paths, one per line, and checks only
# the architectures of the changed logic YAMLs; a changed validator, known-bugs
# list or this script, or an empty value, checks everything. Each changed logic
# YAML is also compared with its base revision (git show HEAD^): a new file
# must be dict format, and a solution that had a SolutionUID must keep one.
# If the current Python is missing deps (e.g. joblib), the script will re-run itself
# with .venv/bin/python (Unix) or .venv\Scripts\python.exe (Windows) if that venv exists.
# Requires: a full build first so rocisa exists under build/tensilelite/rocisa
# (nanobind package: rocisa/_rocisa*.so) or legacy build/tensilelite/rocisa/lib,
# or a pip-installed rocisa.

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _find_hipblaslt_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "tensilelite").is_dir() or not (root / "library").is_dir():
        raise SystemExit(
            "Error: Cannot find hipblaslt root (expected tensilelite/ and library/). "
            "Run from hipblaslt root or keep scripts/ in the project tree."
        )
    return root


def _ensure_paths(root: Path, build_dir: Path, lib_logic_path: Path) -> None:
    """Prepend sys.path entries so `import rocisa` and tensilelite resolve.

    CMake places the nanobind module under build/tensilelite/rocisa/rocisa/
    (_rocisa*.so). The importable package root is build/tensilelite/rocisa.
    Older trees may only have a single rocisa*.so under rocisa/lib/; prefer
    the modern layout when both exist so a stale lib/ artifact cannot shadow.
    """
    tensilelite = root / "tensilelite"
    rocisa_build = build_dir / "tensilelite" / "rocisa"
    rocisa_pkg = rocisa_build / "rocisa"
    rocisa_lib = rocisa_build / "lib"

    def _has_modern_rocisa() -> bool:
        if not rocisa_pkg.is_dir():
            return False
        return bool(
            list(rocisa_pkg.glob("_rocisa*.so")) + list(rocisa_pkg.glob("_rocisa*.pyd"))
        )

    def _has_legacy_rocisa() -> bool:
        if not rocisa_lib.is_dir():
            return False
        return bool(
            list(rocisa_lib.glob("rocisa*.so")) + list(rocisa_lib.glob("rocisa*.pyd"))
        )

    def _has_installed_rocisa() -> bool:
        # A pip-installed rocisa (e.g. `tox -e logic-corpus`). A spec without an
        # origin is the rocisa/ source directory seen as a namespace package.
        spec = importlib.util.find_spec("rocisa")
        return spec is not None and spec.origin is not None

    if _has_modern_rocisa():
        rocisa_path_entries = [rocisa_build]
    elif _has_legacy_rocisa():
        rocisa_path_entries = [rocisa_lib]
    elif _has_installed_rocisa():
        rocisa_path_entries = []
    else:
        raise SystemExit(
            "Error: rocisa not built. Run a full build first, or pip install "
            "tensilelite/rocisa. Expected either:\n"
            f"  {rocisa_pkg} with _rocisa*.so or .pyd, or legacy\n"
            f"  {rocisa_lib} with rocisa*.so or .pyd"
        )

    if not lib_logic_path.exists():
        raise SystemExit(f"Error: Library logic path not found: {lib_logic_path}")

    for path in (*rocisa_path_entries, tensilelite):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _parse_argv(root: Path):
    """Split script argv into library path (or default), the --changed-files value
    (None if not given), and options to pass to TensileLogic (-j, -v, etc.)."""
    default_lib = root / "library"
    args = sys.argv[1:]
    lib_path = None
    changed_files = None
    passthrough = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--changed-files="):
            changed_files = a.split("=", 1)[1]
            i += 1
        elif a == "--changed-files":
            raise SystemExit("Error: use --changed-files=FILE (an empty value checks everything)")
        elif a in ("-j", "--jobs", "-v", "--verbose") or a.startswith("--jobs="):
            passthrough.append(a)
            i += 1
            if i < len(args) and a in ("-j", "--jobs", "-v", "--verbose") and not args[i].startswith("-"):
                passthrough.append(args[i])
                i += 1
        elif a == "--check-all":
            i += 1  # we add it below
        elif a == "--known-bugs":
            passthrough.append(a)
            i += 1
            if i < len(args) and not args[i].startswith("-"):
                passthrough.append(args[i])
                i += 1
            else:
                raise SystemExit("Error: --known-bugs requires a file path")
        elif not a.startswith("-"):
            lib_path = Path(a)
            i += 1
            passthrough.extend(args[i:])
            break
        else:
            passthrough.append(a)
            i += 1
    return (lib_path if lib_path is not None else default_lib, changed_files, passthrough)


def _changed_architectures(root: Path, list_file: Path):
    """Return the architectures to check for the changed paths in list_file, or
    None to check everything.

    Paths are one per line, absolute or relative to the repository root (the
    form `git diff --name-only` prints). A change to the validators, the YAML
    loader, the known-bugs list or this script can affect every architecture,
    so it means everything. Otherwise each changed logic YAML contributes the
    architecture its header declares, and other paths contribute nothing. A
    deleted logic file is resolved through a YAML still in its directory, since a
    directory holds one architecture's logic; if the directory is gone too, the
    answer is everything.
    """
    from tensilelite.CustomYamlLoader import load_logic_gfx_arch

    repo_root = root.parents[1]
    lib_root = (root / "library").resolve()
    validators = [
        (root / "tensilelite" / "tensilelite" / "logic").resolve(),
        (root / "tensilelite" / "tensilelite" / "CustomYamlLoader.py").resolve(),
        Path(__file__).resolve(),
    ]
    archs = set()
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        path = (Path(line) if Path(line).is_absolute() else repo_root / line).resolve()
        if any(path == v or v in path.parents for v in validators):
            return None
        if path.suffix != ".yaml" or lib_root not in path.parents:
            continue
        if not path.is_file():
            siblings = sorted(path.parent.glob("*.yaml")) if path.parent.is_dir() else []
            if not siblings:
                return None
            path = siblings[0]
        arch = load_logic_gfx_arch(path)
        if not isinstance(arch, str):
            return None
        archs.add(arch)
    return archs


def _base_revision_text(repo_root: Path, rel: str) -> str | None:
    """Return ``rel`` at ``HEAD^``, or None when that blob does not exist.

    Args:
        repo_root: Git repository root.
        rel: Repository-relative posix path.

    Returns:
        Decoded file text, or None when git cannot show the blob.

    Raises:
        OSError: If git cannot be started.
    """
    proc = subprocess.run(
        ["git", "show", f"HEAD^:{rel}"], cwd=repo_root, capture_output=True, check=False
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def _check_logic_changes(root: Path, list_file: Path) -> None:
    """Compare the changed logic files in ``list_file`` with ``HEAD^``.

    Args:
        root: hipBLASLt project directory.
        list_file: Changed paths, one per line.

    Returns:
        None.

    Raises:
        SystemExit: If ``HEAD^`` is missing or a changed file breaks a rule.
    """
    from tensilelite.TensileLogic.ValidLogicChange import changed_logic_paths, check_changed_logic

    repo_root = root.parents[1]
    if not changed_logic_paths(repo_root, list_file):
        return
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "HEAD^"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "Error: HEAD^ is not available, so changed logic files cannot be "
            "compared with their base revision. Fetch the parent commit."
        )
    errors = check_changed_logic(
        repo_root, list_file, lambda rel: _base_revision_text(repo_root, rel)
    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    root = _find_hipblaslt_root()
    build_dir = root / "build"
    lib_logic_path, changed_files, passthrough = _parse_argv(root)
    _ensure_paths(root, build_dir, lib_logic_path)

    if changed_files:
        _check_logic_changes(root, Path(changed_files))
        archs = _changed_architectures(root, Path(changed_files))
        if archs is None:
            print("The changes can affect every architecture; checking everything.")
        elif not archs:
            print("No library logic or validator files changed; nothing to check.")
            return
        else:
            print(f"Checking the architectures of the changed logic files: {', '.join(sorted(archs))}")
            passthrough = ["--architecture", ";".join(sorted(archs))] + passthrough

    has_known_bugs_option = any(
        arg in ("--known-bugs", "--use-bundled-known-bugs")
        or arg.startswith("--known-bugs=")
        for arg in passthrough
    )
    if not has_known_bugs_option:
        passthrough = ["--use-bundled-known-bugs"] + passthrough

    # TensileLogic: LOGIC_PATH [options] --check-all
    sys.argv = ["TensileLogic", str(lib_logic_path.resolve())] + passthrough + ["--check-all"]

    from tensilelite.logic import main as tensile_logic_main
    tensile_logic_main()


def _try_venv_reexec() -> None:
    """If key deps are missing but project .venv exists, re-exec with venv Python (never returns)."""
    if os.environ.get("HIPBLASLT_LOGIC_CHECK_VENV"):
        return  # Already running under venv
    try:
        import joblib  # noqa: F401
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        venv_py = root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = root / ".venv" / "bin" / "python"
    if not venv_py.is_file():
        return
    os.environ["HIPBLASLT_LOGIC_CHECK_VENV"] = "1"
    script = Path(__file__).resolve()
    os.execv(venv_py, [str(venv_py), str(script)] + sys.argv[1:])
    # execv does not return on success


if __name__ == "__main__":
    _try_venv_reexec()
    main()
