# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

logging.basicConfig(level=logging.INFO)

# The per-test step-summary renderer lives in the rocKE tree (so it is triggered
# and tested by the component's own CI). This runner sits outside that tree, so
# it imports the module from the source checkout — two levels up from test/therock.
_ROCKE_TESTS_DIR = (
    SCRIPT_DIR.parent.parent
    / "dnn-providers"
    / "hip-kernel-provider"
    / "rocke"
    / "platform"
    / "tests"
)


def _write_step_summary(junit_path: Path) -> None:
    """Best-effort: render the per-test table; never fail the job over it."""
    try:
        sys.path.insert(0, str(_ROCKE_TESTS_DIR))
        from ctest_summary import write_step_summary

        write_step_summary(
            junit_path,
            arch=os.getenv("AMDGPU_FAMILIES", ""),
            summary_path=os.getenv("GITHUB_STEP_SUMMARY"),
        )
    except Exception as exc:  # importing/rendering must never fail the test job
        logging.warning("rocKE test-summary step skipped: %s", exc)


def main() -> int:
    junit_path = THEROCK_DIR / "test_logs" / "ctest-junit-hipkernelprovider.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop any prior run's file so a ctest that dies before rewriting it surfaces
    # as the loud "missing report" marker rather than a stale (possibly all-green) table.
    junit_path.unlink(missing_ok=True)

    environ_vars = os.environ.copy()
    # Some of our runtime kernel compilations have been relying on either ROCM_PATH being set, or ROCm being installed at
    # /opt/rocm. Neither of these is true in TheRock so we need to supply ROCM_PATH to our tests.
    environ_vars["ROCM_PATH"] = str(Path(THEROCK_BIN_DIR).resolve().parent)

    cmd = [
        "ctest",
        "--test-dir",
        f"{THEROCK_BIN_DIR}/hip_kernel_provider",
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "600",
        # JUnit XML feeds the step-summary table below; ctest still streams to
        # stdout, which the workflow tees to the job log.
        "--output-junit",
        str(junit_path),
    ]

    if os.getenv("TEST_TYPE", "full") == "smoke":
        environ_vars["GTEST_FILTER"] = "-Full*"

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

    # check=False so the summary still runs on failure and the ctest exit code is
    # propagated exactly. ctest's exit code is the sole pass/fail authority.
    result = subprocess.run(cmd, cwd=THEROCK_DIR, env=environ_vars, check=False)

    _write_step_summary(junit_path)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
