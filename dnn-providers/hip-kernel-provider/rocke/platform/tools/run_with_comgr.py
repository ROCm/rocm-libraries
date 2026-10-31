# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Apply a build-tree compiler default without replacing a caller's override."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-comgr", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a test command is required")
    env = dict(os.environ)
    if not env.get("ROCKE_COMGR_LIB"):
        env["ROCKE_COMGR_LIB"] = args.default_comgr
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
