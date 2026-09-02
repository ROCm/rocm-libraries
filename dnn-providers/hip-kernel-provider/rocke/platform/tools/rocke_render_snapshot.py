#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reconstruct and render a rocKE value snapshot outside rocGDB."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rocke.debug import load_snapshot, logical_snapshot, render_readable


def render_human(record: dict[str, Any], *, show_sources: bool = False) -> str:
    """Render capture identity and logical values without hiding partial data."""
    capture = record["capture"]
    target = record["target"]
    lines = [
        f"capture scope={capture['scope']} complete={str(capture['complete']).lower()}",
        f"target architecture={target.get('architecture') or '?'} "
        f"kernel={target.get('kernel') or '?'}",
    ]
    for issue in capture.get("issues", []):
        lines.append(
            f"issue value={issue.get('value', '?')} "
            f"status={issue.get('status', '?')}: {issue.get('detail', '')}"
        )
    for wave in record["waves"]:
        lines.append(
            f"wave thread={wave['thread_id']} status={wave['status']} "
            f"pc={wave.get('pc') or '?'} exec={wave.get('exec') or '?'}"
        )
        lines.append(render_readable(wave["values"], show_sources=show_sources))
    return "\n".join(line for line in lines if line)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument("--show-sources", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        physical = load_snapshot(args.snapshot)
        logical = logical_snapshot(physical.to_dict())
    except (KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(logical, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(render_human(logical, show_sources=args.show_sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
