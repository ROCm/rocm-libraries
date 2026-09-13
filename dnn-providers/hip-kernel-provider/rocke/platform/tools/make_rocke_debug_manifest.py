#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate a layout-aware manifest for ``rocke value``.

Example::

    python tools/make_rocke_debug_manifest.py \
      --arch gfx942 --op-id mfma_f32_16x16x16_f16 --role acc \
      --name acc --dtype f32 --storage-dtype f32 \
      --location '$v40' --location '$v41' \
      --location '$v42' --location '$v43' \
      --output acc.json

Physical locations remain an explicit prototype input. Logical shape and
lane/slot coordinates come from rocKE's verified ``LayoutMap``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rocke.core import (
    bind_logical_value,
    debug_manifest,
    logical_value_description,
    register_value_binding,
)
from rocke.core.arch import ArchTarget, LayoutMap, MmaOp


def _layout_and_shape(op: MmaOp, role: str) -> tuple[LayoutMap, tuple[int, int]]:
    if role == "a":
        return op.a_layout(), (op.m, op.k)
    if role == "b":
        return op.b_layout(), (op.k, op.n)
    return op.acc_layout(), (op.m, op.n)


def build_manifest(
    *,
    arch: str,
    op_id: str,
    role: str,
    name: str,
    dtype: str,
    storage_dtype: str,
    locations: list[str],
) -> dict[str, Any]:
    """Resolve an MMA layout and build one portable logical-value manifest."""
    target = ArchTarget.from_gfx(arch)
    op = target.mma.by_op_id(op_id)
    if op is None:
        raise ValueError(f"architecture {arch!r} has no MMA operation {op_id!r}")
    layout, shape = _layout_and_shape(op, role)
    logical = logical_value_description(
        name=name,
        dtype=dtype,
        shape=shape,
        layout=layout,
        layout_name=f"{op_id}.{role}",
    )
    binding = register_value_binding(
        storage_dtype=storage_dtype,
        locations=locations,
        fragment_length=layout.frag_len,
    )
    return debug_manifest(bind_logical_value(logical, binding))


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--op-id", required=True)
    parser.add_argument("--role", choices=("a", "b", "acc"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--dtype", choices=("f16", "bf16", "f32", "fp8e4m3", "bf8e5m2"), required=True
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("f32", "f16x2", "bf16x2", "fp8e4m3x4", "bf8e5m2x4"),
        required=True,
    )
    parser.add_argument("--location", action="append", required=True)
    parser.add_argument("--output", default="-")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        manifest = build_manifest(
            arch=args.arch,
            op_id=args.op_id,
            role=args.role,
            name=args.name,
            dtype=args.dtype,
            storage_dtype=args.storage_dtype,
            locations=args.location,
        )
    except (KeyError, NotImplementedError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(rendered)
    else:
        Path(args.output).write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
