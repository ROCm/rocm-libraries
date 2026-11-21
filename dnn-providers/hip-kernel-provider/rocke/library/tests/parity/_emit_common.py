# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# library/tests/parity/_emit_common.py -- the library emitters' handle on the
# platform parity driver (platform/tests/instances/parity/_emit_common.py),
# the one home of run_emit, hip_case and MODES.
#
# The library emitters import it the way the platform emitters do,
# `from _emit_common import ...`, and this directory comes first on their path,
# so this module stands in under that name and re-exports the platform driver.
# Both files are named _emit_common, so the platform one is loaded from its
# file under another module name. It is found where the emitter runners put it,
# on sys.path (run_diff.py, in a source checkout and in the installed test
# tree, lists the platform parity directory after this one); outside those
# runners it is taken from the source checkout through rocke.assets.
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import PathFinder
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PLATFORM_MODULE_NAME = "_rocke_platform_emit_common"


def _platform_driver_path() -> Path:
    """The platform ``_emit_common.py``: the first one on ``sys.path`` outside
    this directory, else the source checkout's."""
    other_entries = [
        entry for entry in sys.path if entry and Path(entry).resolve() != _HERE
    ]
    spec = PathFinder.find_spec("_emit_common", other_entries)
    if spec is not None and spec.origin is not None:
        return Path(spec.origin)
    from rocke.assets import platform_root

    return platform_root() / "tests" / "instances" / "parity" / "_emit_common.py"


def _load_platform_driver():
    """The platform driver module, loaded once under its own name."""
    if _PLATFORM_MODULE_NAME in sys.modules:
        return sys.modules[_PLATFORM_MODULE_NAME]
    path = _platform_driver_path()
    if not path.is_file():
        raise ImportError(
            f"_emit_common failed - path={str(path)!r}, expected the platform "
            "parity driver; put platform/tests/instances/parity on PYTHONPATH"
        )
    spec = importlib.util.spec_from_file_location(_PLATFORM_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PLATFORM_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_platform_driver = _load_platform_driver()
MODES = _platform_driver.MODES
hip_case = _platform_driver.hip_case
run_emit = _platform_driver.run_emit

__all__ = ["MODES", "hip_case", "run_emit"]
