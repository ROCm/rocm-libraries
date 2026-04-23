#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate depthwise conv fwd instance .inc file from config."""

import argparse
from pathlib import Path


def parse_config(conf_path: Path) -> list[list[int]]:
    instances = []
    for line in conf_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        params = [int(x.strip()) for x in line.split(",")]
        assert len(params) == 12, f"Expected 12 params, got {len(params)}: {line}"
        instances.append(params)
    return instances


def generate_inc(instances: list[list[int]], output_path: Path):
    lines = [
        "// Auto-generated from ngchw_depthwise.conf — do not edit manually",
        "// Parameters: TileH, TileW, Filter, StrH, StrW, PadH, PadW,",
        "//             NBatch, SubTileH, SubTileW, InVecSize, OutVecSize",
        "",
    ]
    for params in instances:
        args = ", ".join(str(p) for p in params)
        lines.append(f"CK_TILE_DEPTHWISE_TRY_INSTANCE({args});")
    lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"Generated {len(instances)} instances -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/forward/profiler/ngchw_depthwise.conf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "instances/forward/depthwise_fwd_instances.inc",
    )
    args = parser.parse_args()

    instances = parse_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_inc(instances, args.output)


if __name__ == "__main__":
    main()
