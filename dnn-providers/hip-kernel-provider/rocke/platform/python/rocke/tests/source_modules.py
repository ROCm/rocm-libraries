# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Load a Python source file that no package holds, by its path.

The tools and scripts outside the ``rocke`` package (``platform/tools``,
``rocke/tools``) are not importable by name; a test loads the one it checks
through :func:`load_module_from_path`. :func:`load_entry_time_check` loads the
entry-time check, which several test files share, once per session under
:data:`ENTRY_TIME_CHECK_MODULE`, from the source tree or an installed tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from ..assets import platform_root

__all__ = ["ENTRY_TIME_CHECK_MODULE", "load_entry_time_check", "load_module_from_path"]

# The module name the entry-time check (rocke/tools/check_entry_times.py) loads
# under, so every test of a session shares one copy of its classes.
ENTRY_TIME_CHECK_MODULE = "check_entry_times"


def load_module_from_path(name: str, path: Path) -> ModuleType:
    """Execute the Python file ``path`` as the module ``name`` and return it.

    The module is registered in ``sys.modules`` under ``name`` first, as an
    import would register it: a dataclass looks its module up there. Raises
    ``ImportError`` when ``path`` is not a Python source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"loading a module failed - path={path}, expected a Python source file"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_entry_time_check(installed_tools_dir: Path) -> ModuleType | None:
    """The entry-time check, ``rocke/tools/check_entry_times.py``, loaded once
    per session as :data:`ENTRY_TIME_CHECK_MODULE`: from the source checkout,
    else from ``installed_tools_dir``, where an installed test tree stages it
    (its ``tests/tools``). None when neither holds it (a platform-only tree)."""
    if ENTRY_TIME_CHECK_MODULE in sys.modules:
        return sys.modules[ENTRY_TIME_CHECK_MODULE]
    for directory in (platform_root().parent / "tools", installed_tools_dir):
        path = directory / f"{ENTRY_TIME_CHECK_MODULE}.py"
        if path.is_file():
            return load_module_from_path(ENTRY_TIME_CHECK_MODULE, path)
    return None
