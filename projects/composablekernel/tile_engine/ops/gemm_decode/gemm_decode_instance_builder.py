# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instance builder for the gemm_decode tile_engine.

P0 wires the universal family with a single hand-written instance header
(`universal/gemm_decode_universal_single_default.hpp`). This script is the
P0 stub for the full Python codegen path that lands in P1+: it reads
`configs/default_config.json`, validates each tile/trait combination via
`gemm_decode_validation_utils`, and emits the expected `--list_kernels`
output and per-config `.hpp` instantiations.

The P0 implementation keeps the CLI surface compatible with the
`gemm_universal_instance_builder.py` invocation pattern used by CMake so
that wiring the codegen later is a drop-in change rather than a contract
change.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from gemm_decode_validation_utils import (
        is_tile_config_valid,
        is_trait_combination_valid,
    )
except ImportError:  # pragma: no cover - exercised only when run by hand.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gemm_decode_validation_utils import (
        is_tile_config_valid,
        is_trait_combination_valid,
    )


def _expand(values_block: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    keys = list(values_block.keys())
    grids = [values_block[k]["values"] for k in keys]
    for combo in itertools.product(*grids):
        yield dict(zip(keys, combo))


def _enumerate_kernels(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tiles = list(_expand(config["tile_config"]))
    traits = list(_expand(config["trait_config"]))
    out: List[Dict[str, Any]] = []
    for tile in tiles:
        if not is_tile_config_valid(tile):
            continue
        for trait in traits:
            if not is_trait_combination_valid(trait):
                continue
            out.append({"tile": tile, "trait": trait})
    return out


def _kernel_name(datatype: str, layout: str, tile: Dict[str, Any], trait: Dict[str, Any]) -> str:
    return (
        f"gemm_decode_universal_{datatype}_{layout}_"
        f"{trait['pipeline']}_{trait['epilogue']}_{trait['scheduler']}_"
        f"split{trait['split_k']}_v{tile['vector_size']}_"
        f"m{tile['m_per_warp']}n{tile['n_per_warp']}"
    )


def _write_kernel_count(working_path: Path, count: int) -> None:
    (working_path / "gemm_decode_universal_kernel_count.txt").write_text(str(count))


def _write_kernel_list(
    working_path: Path,
    datatype: str,
    layout: str,
    kernels: List[Dict[str, Any]],
) -> None:
    lines = []
    for k in kernels:
        tile, trait = k["tile"], k["trait"]
        name = _kernel_name(datatype, layout, tile, trait)
        # Match gemm_universal's `<name>|<tile_config>|<trait_combo>` format.
        tile_token = (
            f"{tile['tile_m']}x{tile['tile_n']}x{tile['tile_k']}_"
            f"{tile['m_per_warp']}x{tile['n_per_warp']}x{tile['warps_per_block']}_"
            f"v{tile['vector_size']}"
        )
        trait_token = f"{trait['pipeline']}_{trait['epilogue']}_{trait['scheduler']}"
        lines.append(f"{name}|{tile_token}|{trait_token}")
    (working_path / "gemm_decode_universal_kernel_list.txt").write_text("\n".join(lines))


def _list_kernels(args: argparse.Namespace) -> int:
    working_path = Path(args.working_path)
    working_path.mkdir(parents=True, exist_ok=True)

    with open(args.config_json, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    kernels = _enumerate_kernels(config)
    _write_kernel_count(working_path, len(kernels))
    _write_kernel_list(working_path, args.datatype, args.layout, kernels)
    print(f"gemm_decode: {len(kernels)} kernel configurations enumerated")
    return 0


def _gen_single(args: argparse.Namespace) -> int:
    # P0 ships exactly one hand-written instance header. Once the full
    # codegen lands, this branch will materialize one .hpp per
    # (datatype, layout, tile, trait) tuple.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="gemm_decode instance builder (P0 stub)")
    parser.add_argument("--working_path", required=True)
    parser.add_argument("--datatype", default="bf16")
    parser.add_argument("--layout", default="rrr")
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--list_kernels", action="store_true")
    parser.add_argument("--gen_single", action="store_true")
    parser.add_argument("--gen_all_individual", action="store_true")
    parser.add_argument("--gpu_target", default="")
    parser.add_argument("--kernel_name", default="")
    parser.add_argument("--tile_config", default="")
    parser.add_argument("--trait_combo", default="")
    parser.add_argument("--max-instances", default=None)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()

    if args.list_kernels:
        return _list_kernels(args)
    if args.gen_single or args.gen_all_individual:
        return _gen_single(args)

    parser.error("one of --list_kernels / --gen_single / --gen_all_individual is required")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
