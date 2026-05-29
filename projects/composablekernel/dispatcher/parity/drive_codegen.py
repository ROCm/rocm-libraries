#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Drive dispatcher codegen for a single config -- parity deliverable (c).

Takes a Tile Engine config JSON, translates it (te_to_dispatcher), selects ONE
config (by index), and invokes the dispatcher's ``unified_gemm_codegen.py`` to
emit exactly that one kernel header.

The bridge between the two systems is the raw TE trait strings the translator
stashes under ``_te`` (codegen wants "compv3"/"intrawave"/"default", not the
canonical dispatcher "auto" etc.). We rebuild a *minimal* TE config -- one value
per parameter -- so codegen's well-tested ``--config`` path produces a single
kernel rather than a cartesian product.

The expected registry identifier (from identifier.encode_identifier) is printed
so the caller can locate the generated ``Kernel_<id>`` struct / SelectedKernel
alias that the C++ harness (deliverable d) includes.

Usage:
    python drive_codegen.py configs/single_fp16_rcr.json
    python drive_codegen.py configs/single_fp16_rcr.json --index 0 --output-dir ./generated
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from identifier import encode_identifier
from te_to_dispatcher import TranslationError, translate_file

_HERE = Path(__file__).resolve().parent
_CODEGEN = _HERE.parent / "codegen" / "unified_gemm_codegen.py"


def _minimal_te_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a one-value-per-parameter config for the chosen dispatcher config.

    ``unified_gemm_codegen._load_config`` reads ``--config`` verbatim and expects
    every tile/trait parameter to already be a *flat list* (it iterates
    ``tc["tile_m"]`` directly), NOT the TE ``{"values": [...]}`` form. We emit one
    single-element list per parameter so codegen produces exactly one kernel, and
    we use the raw TE trait strings (cfg["_te"]) so it sees the names it expects.
    """
    te = cfg["_te"]
    alg = cfg["algorithm"]

    return {
        "tile_config": {
            "tile_m": [alg["tile_m"]],
            "tile_n": [alg["tile_n"]],
            "tile_k": [alg["tile_k"]],
            "warp_m": [alg["warp_m"]],
            "warp_n": [alg["warp_n"]],
            "warp_k": [alg["warp_k"]],
            "warp_tile_m": [alg["warp_tile_m"]],
            "warp_tile_n": [alg["warp_tile_n"]],
            "warp_tile_k": [alg["warp_tile_k"]],
        },
        "trait_config": {
            "pipeline": [te["pipeline"]],
            "epilogue": [te["epilogue"]],
            "scheduler": [te["scheduler"]],
            "pad_m": [alg["pad_m"]],
            "pad_n": [alg["pad_n"]],
            "pad_k": [alg["pad_k"]],
            "persistent": [alg["persistent"]],
        },
    }


def drive(
    te_config_path: str | Path,
    index: int,
    output_dir: Path,
    kernel_set: str,
    dry_run: bool,
) -> int:
    configs = translate_file(te_config_path)
    if not configs:
        print(f"error: no valid dispatcher configs from {te_config_path}", file=sys.stderr)
        return 1
    if not (0 <= index < len(configs)):
        print(f"error: index {index} out of range (0..{len(configs)-1})", file=sys.stderr)
        return 1

    cfg = configs[index]
    identifier = encode_identifier(cfg)
    te = cfg["_te"]
    minimal = _minimal_te_config(cfg)

    print(f"Selected config #{index} of {len(configs)}")
    print(f"  identifier: {identifier}")
    print(f"  datatype={te['datatype']} layout={te['layout']} gpu={cfg['gfx_arch']}")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(minimal, tmp)
    tmp.close()

    cmd = [
        sys.executable,
        str(_CODEGEN),
        "--output-dir",
        str(output_dir),
        "--datatype",
        te["datatype"],
        "--layout",
        te["layout"],
        "--gpu-target",
        cfg["gfx_arch"],
        "--config",
        tmp.name,
        "--variants",
        "standard",
        "--kernel-set",
        kernel_set,
    ]

    print("\ncodegen command:")
    print("  " + " ".join(cmd))

    if dry_run:
        print("\n--dry-run: not invoking codegen. Minimal TE config written to:")
        print(f"  {tmp.name}")
        return 0

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"\nerror: codegen failed (rc={proc.returncode})", file=sys.stderr)
        return proc.returncode

    # Surface generated headers for the chosen kernel set.
    set_dir = output_dir / kernel_set
    headers = sorted(set_dir.rglob("*.hpp")) if set_dir.exists() else []
    print(f"\nGenerated {len(headers)} header(s) under {set_dir}:")
    for h in headers:
        print(f"  {h}")
    print(f"\nExpected registry identifier: {identifier}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="Tile Engine config JSON")
    ap.add_argument("--index", type=int, default=0, help="Which translated config to generate")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "generated",
        help="Codegen output directory",
    )
    ap.add_argument("--kernel-set", default="parity_single", help="Kernel set subdirectory name")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the codegen command and minimal config without invoking it",
    )
    args = ap.parse_args()

    try:
        return drive(args.config, args.index, args.output_dir, args.kernel_set, args.dry_run)
    except (TranslationError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
