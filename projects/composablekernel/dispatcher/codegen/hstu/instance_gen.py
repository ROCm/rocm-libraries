#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSTU instance expansion — kernel config grid from sweep JSON."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from hstu_utils import HstuKernelConfig  # noqa: E402


def _expand_values(spec: Any) -> List[Any]:
    if isinstance(spec, dict) and "values" in spec:
        return list(spec["values"])
    if isinstance(spec, list):
        return spec
    return [spec]


def expand_sweep(
    config_path: Optional[str | Path],
    arch: str = "gfx950",
    receipt: int = 0,
) -> List[HstuKernelConfig]:
    """Expand sweep JSON trait_config into HstuKernelConfig list."""
    del receipt  # reserved for future receipt filtering
    if config_path is None:
        raise ValueError("config_path required for HSTU sweep expansion")

    with open(config_path) as f:
        cfg = json.load(f)

    trait = cfg.get("trait_config", {})
    axes = {
        "data_type": _expand_values(trait.get("data_type", {"values": ["bf16"]})),
        "use_causal": _expand_values(trait.get("use_causal", {"values": [True]})),
        "max_k": _expand_values(trait.get("max_k", {"values": [128]})),
        "mtile": _expand_values(trait.get("mtile", {"values": [64, 128]})),
        "use_splitkv": _expand_values(trait.get("use_splitkv", {"values": [False]})),
    }

    configs: List[HstuKernelConfig] = []
    for dt, causal, max_k, mtile, splitkv in itertools.product(
        axes["data_type"],
        axes["use_causal"],
        axes["max_k"],
        axes["mtile"],
        axes["use_splitkv"],
    ):
        if splitkv and mtile != 64:
            continue
        name = (
            f"jagged_{dt}_causal{int(causal)}_maxk{max_k}_mtile{mtile}"
            f"_splitkv{int(splitkv)}"
        )
        configs.append(
            HstuKernelConfig(
                name=name,
                data_type=dt,
                use_causal=bool(causal),
                max_k=int(max_k),
                mtile=int(mtile),
                use_splitkv=bool(splitkv),
                disable_splitkv=not bool(splitkv),
                gfx_arch=arch,
            )
        )
    return configs


def apply_filter(
    configs: List[HstuKernelConfig],
    filter_expr: str = "",
    filter_file: str = "",
) -> List[HstuKernelConfig]:
    """Filter configs with a Python expression or filter_file defining filter_config(c)."""
    if filter_file:
        ns: Dict[str, Any] = {}
        exec(Path(filter_file).read_text(), ns)  # noqa: S102
        fn = ns.get("filter_config")
        if fn is None:
            raise ValueError(f"{filter_file} must define filter_config(c)")
        return [c for c in configs if fn(c)]

    if not filter_expr:
        return configs

    return [c for c in configs if eval(filter_expr, {"__builtins__": {}}, {"c": c})]  # noqa: S307


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand HSTU sweep JSON")
    parser.add_argument("config", type=Path, help="Sweep config JSON")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    configs = expand_sweep(args.config, args.arch)
    if args.list:
        for c in configs:
            print(c.name)
        print(f"\nTotal: {len(configs)}")
    else:
        print(json.dumps([c.__dict__ for c in configs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
