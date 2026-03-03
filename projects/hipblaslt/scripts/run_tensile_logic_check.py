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
# Requires: a full build first so build/tensilelite/rocisa/lib exists.

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
    rocisa_lib = build_dir / "tensilelite" / "rocisa" / "lib"
    if not rocisa_lib.is_dir():
        raise SystemExit(
            f"Error: rocisa not built. Run a full build first so this exists:\n  {rocisa_lib}"
        )
    so_files = list(rocisa_lib.glob("rocisa*.so")) + list(rocisa_lib.glob("rocisa*.pyd"))
    if not so_files:
        raise SystemExit(
            f"Error: rocisa module not found in {rocisa_lib} (no .so/.pyd). Run a full build."
        )
    if not lib_logic_path.exists():
        raise SystemExit(f"Error: Library logic path not found: {lib_logic_path}")

    # Prepend so Tensile can import rocisa and find the Tensile package
    tensilelite = root / "tensilelite"
    for path in (rocisa_lib, tensilelite):
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
