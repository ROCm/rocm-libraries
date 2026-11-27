# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Load a Python source file that no package holds, by its path.

The tools and scripts outside the ``rocke`` package (``platform/tools``,
``rocke/tools``) are not importable by name; a test loads the one it checks
through :func:`load_module_from_path`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["load_module_from_path"]


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
