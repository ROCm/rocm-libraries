#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run TensileLogic --check-all on library logic. Cross-platform (Windows and Unix).
#
# How to run (from hipblaslt project root):
#   python scripts/run_tensile_logic_check.py [LIBLOGIC_PATH]
# If the current Python is missing deps (e.g. joblib), the script will re-run itself
# with .venv/bin/python (Unix) or .venv\Scripts\python.exe (Windows) if that venv exists.
# Requires: a full build first so rocisa exists under build/tensilelite/rocisa
# (nanobind package: rocisa/_rocisa*.so) or legacy build/tensilelite/rocisa/lib.

import os
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
    """Prepend sys.path entries so `import rocisa` and Tensile resolve.

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

    if _has_modern_rocisa():
        rocisa_path_entries = [rocisa_build]
    elif _has_legacy_rocisa():
        rocisa_path_entries = [rocisa_lib]
    else:
        raise SystemExit(
            "Error: rocisa not built. Run a full build first. Expected either:\n"
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
    """Split script argv into library path (or default) and options to pass to TensileLogic (-j, -v, etc.)."""
    default_lib = root / "library"
    args = sys.argv[1:]
    lib_path = None
    passthrough = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-j", "--jobs", "-v", "--verbose") or a.startswith("--jobs="):
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
    return (lib_path if lib_path is not None else default_lib, passthrough)


def main() -> None:
    root = _find_hipblaslt_root()
    build_dir = root / "build"
    lib_logic_path, passthrough = _parse_argv(root)
    _ensure_paths(root, build_dir, lib_logic_path)

    default_known_bugs = (
        root / "tensilelite" / "Tensile" / "TensileLogic" / "known_bugs.yaml"
    )
    if default_known_bugs.is_file() and "--known-bugs" not in passthrough:
        passthrough = ["--known-bugs", str(default_known_bugs)] + passthrough

    # TensileLogic: LOGIC_PATH [options] --check-all
    sys.argv = ["TensileLogic", str(lib_logic_path.resolve())] + passthrough + ["--check-all"]

    from Tensile.TensileLogic import main as tensile_logic_main
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
