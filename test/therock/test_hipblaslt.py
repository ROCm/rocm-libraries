# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()
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
    # Only device ASAN requires XNACK; host ASAN preserves the caller's setting.
    if "host-asan" not in os.getenv("ARTIFACT_GROUP", ""):
        environ_vars["HSA_XNACK"] = "1"
    environ_vars["OMP_NUM_THREADS"] = "1"

test_process_options = {}
diagnostic_ptracer = os.getenv("HIPBLASLT_DIAGNOSTIC_PTRACER_PID")
if is_asan() and platform == "linux" and diagnostic_ptracer is not None:
    import ctypes

    ptracer_pid = int(diagnostic_ptracer)
    if not 0 < ptracer_pid <= 2147483647:
        raise ValueError("HIPBLASLT_DIAGNOSTIC_PTRACER_PID must be a positive PID")
    # Prepare ctypes before forking. This driver has no background threads.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int

    def allow_diagnostic_ptrace():
        # Linux preserves this across exec, but not fork: grant it in each
        # native test child, only to the collector and its debugger children.
        if libc.prctl(0x59616D61, ptracer_pid, 0, 0, 0) != 0:  # PR_SET_PTRACER
            error = ctypes.get_errno()
            os.write(
                2, f"PR_SET_PTRACER({ptracer_pid}): {os.strerror(error)}\n".encode()
            )
            raise OSError(error, os.strerror(error))

    test_process_options["preexec_fn"] = allow_diagnostic_ptrace

# ---------------------------------------------------------------------------
# Pre-flight: validate the installed library tree's physical layout BEFORE
# launching hipblaslt-test. The runtime probe in
# rocblaslt_auxiliary.cpp has no fallback for misplaced files; a layout
# regression silently turns into hipModuleLoad errors at first dispatch.
# Running the validator here surfaces the real cause and a path to the
# offending file BEFORE the gtest tries to load anything.
#
# This is hipblaslt-specific by construction: the validator lives under
# projects/hipblaslt/tools/scripts/ and this driver runs only for the hipblaslt
# test component. The check fires only on the first shard (it's a
# filesystem walk; running it once per shard is wasted work).
# ---------------------------------------------------------------------------
if int(SHARD_INDEX) == 1:
    install_root = Path(THEROCK_BIN_DIR).resolve().parent  # <output>/bin -> <output>
    validator_dir = (
        THEROCK_DIR / "rocm-libraries" / "projects" / "hipblaslt" / "tools" / "scripts"
    )
    # In case rocm-libraries is not checked out as a sibling (e.g. when
    # running this driver against a standalone hipblaslt build tree),
    # fall back to walking up from this file's location.
    if not (validator_dir / "validate_library_layout.py").is_file():
        validator_dir = (
            SCRIPT_DIR.parent.parent / "projects" / "hipblaslt" / "tools" / "scripts"
        )
    if (validator_dir / "validate_library_layout.py").is_file():
        sys.path.insert(0, str(validator_dir))
        import validate_library_layout

        logging.info(f"++ Layout check on {install_root}")
        violations = validate_library_layout.validate(install_root)
        if violations:
            logging.error(
                f"hipblaslt library layout has {len(violations)} violation(s):"
            )
            for v in violations:
                logging.error(f"  - {v}")
            sys.exit(1)
        logging.info("++ Layout check OK")
    else:
        logging.warning(
            f"validate_library_layout.py not found near {SCRIPT_DIR}; "
            "skipping pre-flight layout check"
        )

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")

# Only run quick tests (less memory intensive) for Windows strix-halo, issue: https://github.com/ROCm/TheRock/issues/1750
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"

test_filter = []
if test_type == "quick":
    test_filter.append("--gtest_filter=*smoke*")
elif test_type == "quick":
    test_filter.append("--gtest_filter=*quick*")

# The tuning store's tests are a separate binary. They need no GPU and take well
# under a second, so the first shard runs all of them, without gtest sharding.
if int(SHARD_INDEX) == 1:
    store_cmd = [f"{THEROCK_BIN_DIR}/hipblaslt-test-tuning-store"]
    store_env = {
        name: value
        for name, value in environ_vars.items()
        if name not in ("GTEST_SHARD_INDEX", "GTEST_TOTAL_SHARDS")
    }
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(store_cmd)}")
    subprocess.run(
        store_cmd, cwd=THEROCK_DIR, check=True, env=store_env, **test_process_options
    )

cmd = [f"{THEROCK_BIN_DIR}/hipblaslt-test"] + test_filter

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd, cwd=THEROCK_DIR, check=True, env=environ_vars, **test_process_options
)
