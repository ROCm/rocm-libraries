# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compatibility exports for the production subprocess launcher."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .._python_subprocess import (
        _run,
        rocke_python_command,
        rocke_subprocess_env,
    )
else:
    _PYTHON = Path(__file__).resolve().parents[2]
    if str(_PYTHON) not in sys.path:
        sys.path.insert(0, str(_PYTHON))
    from rocke._python_subprocess import (
        _run,
        rocke_python_command,
        rocke_subprocess_env,
    )

__all__ = ["rocke_python_command", "rocke_subprocess_env"]


if __name__ == "__main__":
    _run(sys.argv[1:])
