#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Predict LDS conflicts from a JSON request using the production rocKE API."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rocke.analysis.lds import LdsAccess, dumps, predict_lds_conflicts


def _load_request(path: str) -> Mapping[str, Any]:
    document = (
        sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    )
    value = json.loads(document)
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("request must be a JSON object with string keys")

    required = {"target", "opcode", "wave_size", "accesses"}
    optional = {"coordinate_axes"}
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(
            f"request is missing required fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ValueError(f"request has unknown fields: {', '.join(sorted(unknown))}")
    if not isinstance(value["accesses"], list):
        raise ValueError("request.accesses must be an array")
    return value


def _predict(request: Mapping[str, Any]) -> str:
    accesses = tuple(LdsAccess.from_dict(value) for value in request["accesses"])
    coordinate_axes = request.get("coordinate_axes", ())
    result = predict_lds_conflicts(
        target=request["target"],
        opcode=request["opcode"],
        wave_size=request["wave_size"],
        accesses=accesses,
        coordinate_axes=coordinate_axes,
    )
    return dumps(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict LDS conflicts and write canonical semantic JSON."
    )
    parser.add_argument(
        "request", help="request JSON path, or '-' to read standard input"
    )
    args = parser.parse_args(argv)

    try:
        output = _predict(_load_request(args.request))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(f"{output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
