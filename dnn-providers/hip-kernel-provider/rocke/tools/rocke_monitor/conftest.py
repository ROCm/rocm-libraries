# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Keeps pytest, when it runs from rocKE or the repository's root, from collecting this package's tests. They are
unittest tests, offline and with no secrets, that run with `./rk selftest` (or `python3 -B rkmon.py selftest` in this
folder); no rocKE test lane runs them. RKMON_PYTEST=1 lets pytest collect them anyway.
"""

import os

collect_ignore_glob = [] if os.environ.get("RKMON_PYTEST") else ["*"]
