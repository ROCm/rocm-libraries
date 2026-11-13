# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Windows-only launch barrier for assigning file workers to a Job Object."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    gate = Path(sys.argv[1])
    command = sys.argv[2:]
    deadline = time.monotonic() + 30
    while not gate.is_file():
        if time.monotonic() >= deadline:
            return 125
        time.sleep(0.01)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
