#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Python oracle for ``ck_tile::dispatcher::KernelKey::encode_identifier()``.

This is the Python half of parity deliverable (b): "ensure the generated kernel
identifier matches across codegen and runtime". It reproduces the C++
``encode_identifier()`` in ``kernel_key.hpp`` *byte-for-byte* from a dispatcher
config dict (the output of ``te_to_dispatcher.translate``).

The contract is deliberately dumb: every field in the config dict is already in
*canonical dispatcher to_string() form* (the translator did the TE -> dispatcher
mapping exactly once). So the identifier is pure concatenation -- there is no
mapping logic here, which is what makes the Python/C++ agreement provable.

The C++ source of truth (kernel_key.hpp, encode_identifier):

    dtype_a "_" layout_a layout_b layout_c "_"
    pipeline "_" epilogue "_" scheduler "_"
    padM "_" padN "_" padK "_" persistent "_"      (each "True"/"False")
    {tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}_{wt_m}x{wt_n}x{wt_k}
    [ "_splitk{n}"  if split_k > 1 ]
    [ "_{op}"       if elementwise_op not in ("", "PassThrough") ]
    [ "_d{n}"       if num_d_tensors > 0 ]
    [ "_sparse"     if structured_sparsity ]
    [ "_preshuffle" if preshuffle ]

Note block_size is intentionally NOT part of the identifier (it is part of the
KernelKey tie() for equality, but encode_identifier() omits it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _cpp_bool(value: Any) -> str:
    """Mirror C++ ``(flag ? "True" : "False")`` used in encode_identifier()."""
    return "True" if value else "False"


def encode_identifier(cfg: Dict[str, Any]) -> str:
    """Reproduce ``KernelKey::encode_identifier()`` from a dispatcher config dict.

    ``cfg`` is one element of ``te_to_dispatcher.translate(...)`` -- it has
    ``signature`` and ``algorithm`` sub-dicts whose string fields are already in
    canonical to_string() form.
    """
    sig = cfg["signature"]
    alg = cfg["algorithm"]

    parts = []
    parts.append(f"{sig['dtype_a']}_")
    parts.append(f"{sig['layout_a']}{sig['layout_b']}{sig['layout_c']}_")
    parts.append(f"{alg['pipeline']}_")
    parts.append(f"{alg['epilogue']}_")
    parts.append(f"{alg['scheduler']}_")
    parts.append(f"{_cpp_bool(alg['pad_m'])}_")
    parts.append(f"{_cpp_bool(alg['pad_n'])}_")
    parts.append(f"{_cpp_bool(alg['pad_k'])}_")
    parts.append(f"{_cpp_bool(alg['persistent'])}_")
    parts.append(
        f"{alg['tile_m']}x{alg['tile_n']}x{alg['tile_k']}"
        f"_{alg['warp_m']}x{alg['warp_n']}x{alg['warp_k']}"
        f"_{alg['warp_tile_m']}x{alg['warp_tile_n']}x{alg['warp_tile_k']}"
    )

    identifier = "".join(parts)

    # Optional suffixes -- emitted in the exact order of the C++ implementation.
    split_k = sig.get("split_k", 1)
    if split_k > 1:
        identifier += f"_splitk{split_k}"

    op = sig.get("elementwise_op", "")
    if op and op != "PassThrough":
        identifier += f"_{op}"

    num_d = sig.get("num_d_tensors", 0)
    if num_d > 0:
        identifier += f"_d{num_d}"

    if sig.get("structured_sparsity", False):
        identifier += "_sparse"

    if alg.get("preshuffle", False):
        identifier += "_preshuffle"

    return identifier


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config",
        type=Path,
        help="JSON file: a single dispatcher config dict or a list of them "
        "(e.g. the output of `te_to_dispatcher.py --json`).",
    )
    args = ap.parse_args()

    try:
        with open(args.config) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    configs = data if isinstance(data, list) else [data]
    for cfg in configs:
        print(encode_identifier(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
