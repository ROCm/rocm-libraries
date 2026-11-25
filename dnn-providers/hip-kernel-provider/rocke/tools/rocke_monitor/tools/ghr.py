#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gh under the read login, for the tools and the review checks: `python3 -I ghr.py <gh arguments>`.

It runs gh (the configured binary: $RKMON_GH in a job) with GH_CONFIG_DIR set to the read login's folder, as rk and
its jobs give it, and no token variable; a call that is not a read (a REST method or body flag, a GraphQL mutation, a
subcommand that writes) is refused with exit 2 (rkmon_lib/gh.py check_read). gh's output and exit code are its own.
"""

import pathlib
import sys

sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time (and -I ignores PYTHONDONTWRITEBYTECODE)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import gh  # noqa: E402


def main(argv: list[str]) -> int:
    try:
        gh.exec_read(argv)
    except gh.GhError as exc:
        print(f"ghr: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ghr: gh did not run: {exc}", file=sys.stderr)
        return 127
    return 0  # exec_read does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
