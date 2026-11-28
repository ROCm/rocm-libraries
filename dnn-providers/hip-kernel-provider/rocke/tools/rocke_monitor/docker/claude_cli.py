#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where the Claude Agent SDK keeps the Claude Code CLI it brings (claude_agent_sdk/_bundled/claude), for the image's
build: `--link PATH` links it there (/usr/local/bin/claude), so `claude --version`, rk doctor and the container's
`claude` command find the same CLI the orchestrator's session runs. Run it with the orchestrator venv's python.
"""

import argparse
import os
import pathlib
import sys


def bundled() -> pathlib.Path | None:
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    p = pathlib.Path(claude_agent_sdk.__file__).resolve().parent / "_bundled" / "claude"
    return p if p.is_file() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--link", help="make this path a symlink to the bundled CLI")
    a = ap.parse_args(argv)
    cli = bundled()
    if cli is None:
        print(
            "the Claude Agent SDK brings no CLI here (an sdist install?): install Claude Code and set its path",
            file=sys.stderr,
        )
        return 3
    if a.link:
        dest = pathlib.Path(a.link)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(cli, dest)
    print(cli)
    return 0


if __name__ == "__main__":
    sys.exit(main())
