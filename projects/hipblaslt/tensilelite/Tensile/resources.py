# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Package resource helpers for TensileLite generator data files."""

from contextlib import ExitStack
from importlib import resources
from pathlib import Path, PureWindowsPath
import shutil
import sys
from typing import List, Tuple

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:
    from importlib.abc import Traversable


_PACKAGE = __package__

_STATIC_HEADER_NAMES: Tuple[str, ...] = (
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "KernelHeader.h",
    "ReductionTemplate.h",
    "memory_gfx.h",
)


def _root() -> Traversable:
    return resources.files(_PACKAGE)


def _resource(*parts: str) -> Traversable:
    """Return a package resource under the Tensile package root."""
    item = _root()
    for part in parts:
        item = item / part
    return item


def _resource_text(*parts: str, encoding: str = "utf-8") -> str:
    """Read a package resource under the Tensile package root as text."""
    return _resource(*parts).read_text(encoding=encoding)


def static_header_paths() -> Tuple[Traversable, ...]:
    """Return the packaged static-header resources in copy order."""
    source = _resource("Source")
    return tuple(source / name for name in _STATIC_HEADER_NAMES)


def copy_static_headers(output_dir) -> List[str]:
    """Copy packaged static headers into output_dir and return their names."""
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"Static-header output path is not a directory: {output_path}"
        )

    # A resource may need to be unpacked to a temporary file. Keep every
    # as_file() context alive until all resources have been preflighted and
    # copied, then let ExitStack clean up the extracted files.
    with ExitStack() as stack:
        sources = []
        for source in static_header_paths():
            source_path = stack.enter_context(resources.as_file(source))
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Packaged static header not found: {source.name}"
                )
            sources.append((source.name, source_path))

        output_path.mkdir(parents=True, exist_ok=True)
        for name, source_path in sources:
            shutil.copy(source_path, output_path / name)

    return [name for name, _ in sources]


def _custom_kernels() -> Traversable:
    return _resource("CustomKernels")


def _validate_custom_kernel_resource_name(name: str) -> None:
    """Reject path-bearing names at the bundled-resource boundary."""
    if "/" in name or "\\" in name or PureWindowsPath(name).drive:
        raise ValueError(
            f"Custom kernel resource name must not be a path: {name!r}"
        )


def custom_kernel_names() -> List[str]:
    """Return bundled custom kernel names in deterministic order."""
    return sorted(
        resource.name[:-2]
        for resource in _custom_kernels().iterdir()
        if resource.is_file() and resource.name.endswith(".s")
    )


def custom_kernel_text(name: str) -> str:
    """Read a bundled custom kernel assembly resource."""
    _validate_custom_kernel_resource_name(name)
    return _resource_text("CustomKernels", f"{name}.s")


def known_bugs_text() -> str:
    """Read the bundled TensileLogic known-bugs YAML resource."""
    return _resource_text("TensileLogic", "known_bugs.yaml")


def ductile_defaults_text() -> str:
    """Read the bundled ductile defaults YAML resource."""
    return _resource_text("ductile", "config", "defaults.yaml")
