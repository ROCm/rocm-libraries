# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The suite runs its git commands in temporary repositories of its own. A GIT_DIR, GIT_WORK_TREE or GIT_INDEX_FILE that
the caller exported (a git hook, `git rebase --exec`) would point those commands at this repository instead, so pytest
(RKMON_PYTEST=1) runs the suite without them: rkmon_lib/selftest.py GIT_NAMES, which `rk selftest` drops too. A gh that
refuses every call goes first on the PATH (tests/test_00_no_github.py), as under unittest: no test reaches GitHub.
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import test_00_no_github  # noqa: E402,F401
from rkmon_lib import selftest  # noqa: E402

for _name in selftest.GIT_NAMES:
    os.environ.pop(_name, None)
