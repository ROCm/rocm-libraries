#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Refresh the build binding, then run one TensileLite generator command."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from tensilelite_configure_client import configure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Python command is required after --")
    configure(args.client, ensure=True)
    return subprocess.run([str(args.python), *command], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
