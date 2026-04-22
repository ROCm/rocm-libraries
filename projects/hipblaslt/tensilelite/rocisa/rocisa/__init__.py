# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import types
from ._rocisa import *
from . import _rocisa

# Register nanobind submodules under the rocisa.* namespace so that
# `from rocisa.enum import X` and `import rocisa.instruction as ri` work.
for _name, _obj in vars(_rocisa).items():
    if isinstance(_obj, types.ModuleType) and not _name.startswith("_"):
        sys.modules.setdefault(f"rocisa.{_name}", _obj)
