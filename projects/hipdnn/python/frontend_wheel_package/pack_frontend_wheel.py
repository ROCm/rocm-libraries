#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pack pre-built hipDNN Python bindings into a wheel.

CMake builds only the nanobind extension. This script owns wheel/package staging:
it creates an importable hipdnn_frontend package from the source template,
overlays the built extension, and delegates to `python -m build --wheel` so
METADATA, RECORD, and wheel tags are created by standard packaging tooling.

Usage:
    python pack_frontend_wheel.py \
        --build-dir /path/to/cmake-build \
        --wheel-dir /path/to/wheel-output

A pre-staged package can also be packed directly:
    python pack_frontend_wheel.py \
        --pkg-dir /path/to/hipdnn_frontend \
        --wheel-dir /path/to/wheel-output
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYPROJECT_FILE = SCRIPT_DIR / "pyproject.toml"
PACKAGE_TEMPLATE_DIR = SCRIPT_DIR / "src" / "hipdnn_frontend"
EXPECTED_PKG_NAME = "hipdnn_frontend"
NATIVE_EXT_SUFFIXES = (".so", ".pyd")
NATIVE_EXT_PREFIXES = ("hipdnn_frontend_python", "libhipdnn_frontend_python")
IGNORED_SEARCH_DIRS = {"CMakeFiles", "_deps", "Testing"}
PACKAGE_TREE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

SETUP_PY_TEMPLATE = """# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

\"\"\"Generated setup file for the pre-built hipdnn_frontend wheel.\"\"\"

import os
import sysconfig

from setuptools import Distribution, find_packages, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


_pkg = "hipdnn_frontend"
_packages = find_packages(where="src", include=[_pkg, f"{_pkg}.*"])
if not _packages:
    raise RuntimeError(f"find_packages found no {_pkg!r} package; wheel staging is broken")

setup(
    distclass=BinaryDistribution,
    package_dir={"": "src"},
    packages=_packages,
    package_data={_pkg: ["*.so", "*.pyd"]},
    exclude_package_data={_pkg: ["__pycache__/*", "*.pyc", "*.pyo"]},
    include_package_data=False,
    zip_safe=False,
    options={
        "bdist_wheel": {
            "plat_name": os.getenv("ROCM_SDK_WHEEL_PLATFORM_TAG", sysconfig.get_platform()),
            "py_limited_api": "cp312",
        },
    },
)
"""

logger = logging.getLogger(__name__)


def is_native_extension(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in NATIVE_EXT_SUFFIXES
        and any(path.name.startswith(prefix) for prefix in NATIVE_EXT_PREFIXES)
    )


def is_searchable_candidate(path: Path) -> bool:
    return is_native_extension(path) and not (set(path.parts) & IGNORED_SEARCH_DIRS)


def preferred_extension_dirs(build_dir: Path) -> tuple[Path, ...]:
    return (
        build_dir,
        build_dir / "Release",
        build_dir / "RelWithDebInfo",
        build_dir / "Debug",
        build_dir / "release",
        build_dir / "debug",
        build_dir / "lib",
        build_dir / "bin",
        build_dir / "Release" / "lib",
        build_dir / "Release" / "bin",
        build_dir / "RelWithDebInfo" / "lib",
        build_dir / "RelWithDebInfo" / "bin",
        build_dir / "Debug" / "lib",
        build_dir / "Debug" / "bin",
        build_dir / "release" / "lib",
        build_dir / "release" / "bin",
        build_dir / "debug" / "lib",
        build_dir / "debug" / "bin",
        build_dir / "wheel_package" / EXPECTED_PKG_NAME,
    )


def raise_multiple_native_extensions(matches: list[Path]) -> None:
    formatted = "\n".join(f"  - {path}" for path in matches)
    raise SystemExit(
        "Multiple native extensions found; pass --extension to select one:\n"
        f"{formatted}"
    )


def find_native_extension(build_dir: Path) -> Path:
    for directory in preferred_extension_dirs(build_dir):
        if not directory.is_dir():
            continue
        matches = sorted(
            path for path in directory.iterdir() if is_native_extension(path)
        )
        if len(matches) > 1:
            raise_multiple_native_extensions(matches)
        if matches:
            return matches[0]

    matches = sorted(
        path for path in build_dir.rglob("*") if is_searchable_candidate(path)
    )
    if not matches:
        raise SystemExit(
            f"No native extension named {NATIVE_EXT_PREFIXES[0]}* with suffix "
            f"{'/'.join(NATIVE_EXT_SUFFIXES)} found under {build_dir}. Build the "
            "CMake project first."
        )
    if len(matches) > 1:
        raise_multiple_native_extensions(matches)
    return matches[0]


def validate_pkg_dir(pkg_dir: Path) -> None:
    if not pkg_dir.is_dir():
        raise SystemExit(f"Package directory does not exist: {pkg_dir}")
    if pkg_dir.name != EXPECTED_PKG_NAME:
        raise SystemExit(
            f"Package directory basename must be {EXPECTED_PKG_NAME!r}; got {pkg_dir.name!r}"
        )
    if not (pkg_dir / "__init__.py").is_file():
        raise SystemExit(f"Package directory is missing __init__.py: {pkg_dir}")
    if not any(is_native_extension(path) for path in pkg_dir.rglob("*")):
        raise SystemExit(
            f"No native extension ({'/'.join(NATIVE_EXT_SUFFIXES)}) found under "
            f"{pkg_dir}; refusing to build a platform wheel from pure-Python files"
        )


def stage_package_from_build(
    *, build_dir: Path, package_dir: Path, extension: Path | None
) -> Path:
    if (
        PACKAGE_TEMPLATE_DIR.name != EXPECTED_PKG_NAME
        or not PACKAGE_TEMPLATE_DIR.is_dir()
    ):
        raise SystemExit(
            f"Package template directory is missing: {PACKAGE_TEMPLATE_DIR}"
        )

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PACKAGE_TEMPLATE_DIR, package_dir, ignore=PACKAGE_TREE_IGNORE)

    native_extension = (
        extension.resolve()
        if extension is not None
        else find_native_extension(build_dir)
    )
    if not is_native_extension(native_extension):
        raise SystemExit(
            f"--extension is not a hipDNN Python native extension: {native_extension}"
        )
    shutil.copy2(native_extension, package_dir / native_extension.name)
    return package_dir


def write_project_files(build_root: Path) -> None:
    readme_file = SCRIPT_DIR / "README.md"
    if not readme_file.is_file():
        readme_file = SCRIPT_DIR.parent / "README.md"
    shutil.copy2(PYPROJECT_FILE, build_root / "pyproject.toml")
    (build_root / "setup.py").write_text(SETUP_PY_TEMPLATE, encoding="utf-8")
    shutil.copy2(readme_file, build_root / "README.md")


def build_wheel(*, staged_pkg_dir: Path, wheel_dir: Path) -> list[Path]:
    validate_pkg_dir(staged_pkg_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        build_root = temp_dir / "wheel-src"
        build_root.mkdir()
        shutil.copytree(
            staged_pkg_dir,
            build_root / "src" / EXPECTED_PKG_NAME,
            ignore=PACKAGE_TREE_IGNORE,
        )
        write_project_files(build_root)

        temp_wheel_dir = temp_dir / "wheelhouse"
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(temp_wheel_dir),
                str(build_root),
            ]
        )

        built_wheels = sorted(temp_wheel_dir.glob("hipdnn_frontend-*.whl"))
        if not built_wheels:
            raise SystemExit(
                f"python -m build produced no hipdnn_frontend wheel in {temp_wheel_dir}"
            )

        for stale_wheel in wheel_dir.glob("hipdnn_frontend-*.whl"):
            stale_wheel.unlink()

        copied_wheels = []
        for wheel in built_wheels:
            destination = wheel_dir / wheel.name
            shutil.copy2(wheel, destination)
            copied_wheels.append(destination)
        return copied_wheels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build directory containing the built hipDNN frontend extension",
    )
    source.add_argument(
        "--pkg-dir",
        type=Path,
        help=f"Pre-staged {EXPECTED_PKG_NAME} package directory containing the native extension",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        help="Explicit native extension path to copy when --build-dir has multiple candidates",
    )
    parser.add_argument(
        "--wheel-dir",
        required=True,
        type=Path,
        help="Output directory for the staged package and .whl file",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    wheel_dir = args.wheel_dir.resolve()

    if args.pkg_dir is not None:
        staged_pkg_dir = args.pkg_dir.resolve()
        validate_pkg_dir(staged_pkg_dir)
    else:
        build_dir = args.build_dir.resolve()
        if not build_dir.is_dir():
            raise SystemExit(f"--build-dir is not a directory: {build_dir}")
        staged_pkg_dir = stage_package_from_build(
            build_dir=build_dir,
            package_dir=wheel_dir / EXPECTED_PKG_NAME,
            extension=args.extension,
        )

    wheels = build_wheel(staged_pkg_dir=staged_pkg_dir, wheel_dir=wheel_dir)
    logger.info(
        "Wheel(s) written to %s: %s", wheel_dir, [wheel.name for wheel in wheels]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
