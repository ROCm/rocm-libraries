#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On a host: undo tools/gh_split_apply.py, so gh's own folder has the write login again, in place of the read
login's copy (tools/gh_split_apply.py says how, and what it checks).

  tools/gh_split_rollback.py [--no-probe]
"""

import pathlib
import runpy
import sys

if __name__ == "__main__":
    split = runpy.run_path(
        str(pathlib.Path(__file__).resolve().with_name("gh_split_apply.py")),
        run_name="gh_split",
    )
    sys.exit(split["main"](sys.argv[1:], action="rollback"))
