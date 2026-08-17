# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
os_type = platform.system().lower()

logging.basicConfig(level=logging.INFO)

TEST_TO_IGNORE = {
    # TODO(#3709): Re-enable gfx110X tests once issues are resolved
    "gfx110X-all": {
        "windows": [
            "miopen_plugin_integration_tests",
        ]
    }
}

# If you increase the timeout here you need to also increase the timeout for the job
# See file build_tools/github_actions/fetch_test_configurations.py and search for miopenprovider
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/miopen_plugin",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "1200",
]

if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
    ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
    cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    environ_vars["GTEST_FILTER"] = "-Full*"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)
