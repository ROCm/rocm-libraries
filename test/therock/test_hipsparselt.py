# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()


def is_asan():
    """Return True for ASAN builds, detected via the ARTIFACT_GROUP env var."""
    return "asan" in os.getenv("ARTIFACT_GROUP", "")


logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

if is_asan():
    # Disable OpenMP parallelism: it provides limited speedups and introduces
    # high overhead in ASAN builds.
    environ_vars["OMP_NUM_THREADS"] = "1"

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")

test_filter = []
if test_type == "quick":
    quick_filter = "--gtest_filter=*smoke*-*known_bug*"
    test_filter.append(quick_filter)
elif test_type == "standard":
    test_filter.append("--gtest_filter=*smoke*:*quick*:*pre_checkin*-*known_bug*")

cmd = [f"{THEROCK_BIN_DIR}/hipsparselt-test"] + test_filter

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
