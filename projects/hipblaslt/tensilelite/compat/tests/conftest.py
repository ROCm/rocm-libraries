# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys


_COMPAT_ROOT = str(Path(__file__).resolve().parents[1])
if _COMPAT_ROOT not in sys.path:
    sys.path.insert(0, _COMPAT_ROOT)
