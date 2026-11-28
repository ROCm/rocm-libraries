# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# library/tests/parity/_emit_common.py -- the library emitters' handle on the
# platform parity driver (_emit_common.py in rocke.assets.platform_parity_dir()),
# the one home of run_emit, hip_case and MODES.
#
# The library emitters import it the way the platform emitters do,
# `from _emit_common import ...`, and this directory comes first on their path,
# so this module stands in under that name and re-exports the platform driver.
# Both files are named _emit_common, so the platform one is loaded from its
# file under another module name.
from __future__ import annotations

import importlib.util
import sys

from rocke.assets import platform_parity_dir

_PLATFORM_MODULE_NAME = "_rocke_platform_emit_common"


def _load_platform_driver():
    """The platform driver module, loaded once under its own name."""
    if _PLATFORM_MODULE_NAME in sys.modules:
        return sys.modules[_PLATFORM_MODULE_NAME]
    path = platform_parity_dir() / "_emit_common.py"
    if not path.is_file():
        raise ImportError(
            f"_emit_common failed - path={str(path)!r}, expected the platform "
            "parity driver; point ROCKE_PLATFORM_PARITY_DIR at the platform "
            "tests/instances/parity directory"
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
