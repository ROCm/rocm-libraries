#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tile Engine config JSON -> dispatcher config objects.

This is deliverable (a) of the parity bring-up: a translator that turns a
tile_engine GEMM config file (the ``{tile_config, trait_config}`` schema, with
either ``{"values": [...]}`` or ``{"min","max","step"}`` parameter specs) into
concrete *dispatcher* configuration objects.

A "dispatcher config object" here is a plain dict shaped 1:1 like the C++
``ck_tile::dispatcher::KernelKey`` (Signature + Algorithm). Every field uses the
*canonical dispatcher string* form -- i.e. the exact output of the C++
``to_string()`` overloads in ``kernel_key.hpp`` -- so that the identifier oracle
(``identifier.py``) can produce the registry key by pure concatenation and the
C++ runtime will agree byte-for-byte (see ``check_identifier_parity.py``).

The TE -> dispatcher mapping (scheduler "default" -> "auto", fp8/bf8 output ->
fp16, etc.) is applied exactly ONCE, here, using tables that mirror
``codegen/codegen_common.py`` and ``kernel_key.hpp``.

CLI:
    python te_to_dispatcher.py configs/single_fp16_rcr.json
    python te_to_dispatcher.py configs/single_fp16_rcr.json --json   # dump configs
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Canonical mappings (mirror codegen_common.CommonTypeMappings + kernel_key.hpp)
# --------------------------------------------------------------------------- #

# fp8/bf8 accumulate in fp32 but store output as fp16 (8-bit too narrow for C).
_OUTPUT_DTYPE = {"fp8": "fp16", "bf8": "fp16"}

# Acc type per input label (matches operation_support_matrix.md).
_ACC_DTYPE = {
    "fp16": "fp32",
    "bf16": "fp32",
    "fp8": "fp32",
    "bf8": "fp32",
    "int8": "int32",
    "fp32": "fp32",
}

# TE scheduler string -> dispatcher Scheduler::to_string() form.
# codegen_common maps "default" -> Scheduler::Auto, whose to_string() is "auto".
_SCHEDULER_CANON = {
    "intrawave": "intrawave",
    "interwave": "interwave",
    "default": "auto",
    "auto": "auto",
}

# TE pipeline names already match Pipeline::to_string() for the set the tile
# engine emits. Listed explicitly so an unknown pipeline is a hard error rather
# than a silent passthrough that would desync the identifier.
#
# Pipelines that have NO codegen path in codegen_common.PIPELINE_TO_DISPATCHER
# or unified_gemm_codegen are excluded. If a TE config specifies one, we raise
# TranslationError immediately (at translation time) rather than letting codegen
# fail with an opaque error message.
_PIPELINE_CANON = {
    "mem": "mem",
    "compv3": "compv3",
    "compv4": "compv4",
    "compv5": "compv5",
    "preshufflev2": "preshufflev2",
}

# Pipelines recognized in TE configs but NOT supported by the dispatcher codegen.
# Raising TranslationError for these at translation time gives a clear diagnostic
# rather than an opaque codegen failure.
_UNSUPPORTED_PIPELINES = frozenset({"compv1", "compv2", "preshufflev1"})

_EPILOGUE_CANON = {
    "default": "default",
    "cshuffle": "cshuffle",
    "none": "none",
}

_LAYOUT_CHAR = {"r": "r", "c": "c", "p": "p"}  # p = PackedExternal

# Pipelines that imply double SMEM buffering (matches unified_gemm_codegen).
_DOUBLE_BUFFER_PIPELINES = {"compv4", "preshufflev2"}


class TranslationError(ValueError):
    """Raised when a TE config cannot be mapped to a dispatcher config."""


# --------------------------------------------------------------------------- #
# TE JSON parameter expansion
# --------------------------------------------------------------------------- #


def _values(spec: Dict[str, Any], key: str, default: List) -> List:
    """Extract a parameter list from a TE config entry.

    Supports ``{"values": [...]}`` and ``{"min","max","step"}`` forms, matching
    kernel_config_loader._get_values so behavior is identical to existing tools.
    """
    if key not in spec:
        return list(default)
    item = spec[key]
    if isinstance(item, dict) and "values" in item:
        return list(item["values"])
    if isinstance(item, dict) and "min" in item and "max" in item:
        step = item.get("step", 1)
        return list(range(item["min"], item["max"] + 1, step))
    if isinstance(item, list):
        return list(item)
    raise TranslationError(f"Cannot interpret parameter spec for '{key}': {item!r}")


@dataclass(frozen=True)
class _Tile:
    # NOTE: Naming trap — TE uses "warp_m/n/k" to mean wave counts per block
    # (how many waves/warps tile the block). The dispatcher calls these same
    # values "wave_shape.m/n/k". What the dispatcher calls "warp_tile" is the
    # per-warp MFMA shape (tile_m/n/k per wave). They map one-to-one but the
    # vocabularies are swapped; mixing them produces valid-looking but wrong kernels.
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    def is_valid(self) -> bool:
        # Mirror codegen_common.TileConfig.is_valid: block tile must divide
        # evenly into warp_count * warp_tile along every axis.
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            return False
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        )


# (pipeline, epilogue, scheduler) combos CK Tile does not support. Mirrors
# codegen_common.TraitConfigBase._UNSUPPORTED (compute pipelines + interwave).
_UNSUPPORTED_TRAITS = frozenset(
    (p, e, "interwave")
    for p in ("compv3", "compv4", "compv5", "compv6", "comp_async", "basic_async_v1")
    for e in ("cshuffle", "default")
)


# --------------------------------------------------------------------------- #
# Translator
# --------------------------------------------------------------------------- #


def translate_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load a TE config JSON and return a list of dispatcher config dicts."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return translate(data)


def translate_with_rejections(
    data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Like translate(), but also returns a list of rejected combination reasons.

    Returns:
        (valid_configs, rejections) where each rejection is a dict with keys
        {combo, reason} suitable for writing to a CSV rejection manifest.
    """
    return _translate_impl(data, collect_rejections=True)


def translate(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate a parsed TE config dict into dispatcher config dicts.

    Returns one dict per valid (tile x trait) combination. Invalid combinations
    (tile divisibility / unsupported traits) are dropped, matching codegen.
    """
    configs, _ = _translate_impl(data, collect_rejections=False)
    return configs


def _translate_impl(
    data: Dict[str, Any],
    collect_rejections: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Shared implementation for translate() and translate_with_rejections()."""
    datatype = data.get("datatype", "fp16")
    if isinstance(datatype, dict):
        # tile_engine sometimes nests dtype as {"a","b","c","acc"}; take A.
        datatype = datatype.get("a", "fp16")
    layout = data.get("layout", "rcr")
    gfx_arch = data.get("gpu_target") or (data.get("gpu_targets") or ["gfx942"])[0]

    if len(layout) < 3:
        raise TranslationError(f"Layout must be >=3 chars (got {layout!r})")
    layout_a, layout_b, layout_c = layout[0], layout[1], layout[2]
    for ch in (layout_a, layout_b, layout_c):
        if ch not in _LAYOUT_CHAR:
            raise TranslationError(f"Unknown layout char {ch!r} in {layout!r}")

    block_size = data.get("block_size", 256)
    k_block_per_cu = data.get("k_block_per_cu", 1)
    num_wave_groups = data.get("num_wave_groups", 1)
    split_k = data.get("split_k", 1)

    tc = data.get("tile_config", {})
    tr = data.get("trait_config", {})

    tiles = [
        _Tile(*combo)
        for combo in itertools.product(
            _values(tc, "tile_m", [128]),
            _values(tc, "tile_n", [128]),
            _values(tc, "tile_k", [32]),
            _values(tc, "warp_m", [2]),
            _values(tc, "warp_n", [2]),
            _values(tc, "warp_k", [1]),
            _values(tc, "warp_tile_m", [32]),
            _values(tc, "warp_tile_n", [32]),
            _values(tc, "warp_tile_k", [16]),
        )
    ]

    trait_combos = list(
        itertools.product(
            _values(tr, "pipeline", ["compv4"]),
            _values(tr, "epilogue", ["cshuffle"]),
            _values(tr, "scheduler", ["intrawave"]),
            _values(tr, "pad_m", [False]),
            _values(tr, "pad_n", [False]),
            _values(tr, "pad_k", [False]),
            _values(tr, "persistent", [False]),
        )
    )

    configs: List[Dict[str, Any]] = []
    rejections: List[Dict[str, str]] = []

    for tile in tiles:
        if not tile.is_valid():
            if collect_rejections:
                rejections.append({
                    "combo": str(tile),
                    "reason": "invalid_tile_divisibility",
                })
            continue
        for (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent) in trait_combos:
            if (pipeline, epilogue, scheduler) in _UNSUPPORTED_TRAITS:
                if collect_rejections:
                    rejections.append({
                        "combo": f"{tile}+{pipeline}_{epilogue}_{scheduler}",
                        "reason": f"unsupported_trait_combo:{pipeline}_{epilogue}_{scheduler}",
                    })
                continue
            configs.append(
                _build_config(
                    datatype=datatype,
                    layout=(layout_a, layout_b, layout_c),
                    gfx_arch=gfx_arch,
                    tile=tile,
                    pipeline=pipeline,
                    epilogue=epilogue,
                    scheduler=scheduler,
                    pad_m=bool(pad_m),
                    pad_n=bool(pad_n),
                    pad_k=bool(pad_k),
                    persistent=bool(persistent),
                    block_size=block_size,
                    k_block_per_cu=k_block_per_cu,
                    num_wave_groups=num_wave_groups,
                    split_k=split_k,
                )
            )
    return configs, rejections


def _build_config(
    *,
    datatype: str,
    layout,
    gfx_arch: str,
    tile: _Tile,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    pad_m: bool,
    pad_n: bool,
    pad_k: bool,
    persistent: bool,
    block_size: int,
    k_block_per_cu: int,
    num_wave_groups: int,
    split_k: int,
) -> Dict[str, Any]:
    if pipeline in _UNSUPPORTED_PIPELINES:
        raise TranslationError(
            f"Pipeline {pipeline!r} has no dispatcher codegen path "
            f"(compv1/compv2/preshufflev1 are not supported by unified_gemm_codegen)"
        )
    if pipeline not in _PIPELINE_CANON:
        raise TranslationError(f"Unknown pipeline {pipeline!r}")
    if epilogue not in _EPILOGUE_CANON:
        raise TranslationError(f"Unknown epilogue {epilogue!r}")
    if scheduler not in _SCHEDULER_CANON:
        raise TranslationError(f"Unknown scheduler {scheduler!r}")
    if datatype not in _ACC_DTYPE:
        raise TranslationError(f"Unknown datatype {datatype!r}")

    out_dtype = _OUTPUT_DTYPE.get(datatype, datatype)
    acc_dtype = _ACC_DTYPE[datatype]
    layout_a, layout_b, layout_c = layout

    return {
        # Raw TE trait strings, retained for driving unified_gemm_codegen.py
        # (which expects "compv3"/"intrawave"/"default" etc.).
        "_te": {
            "datatype": datatype,
            "layout": layout_a + layout_b + layout_c,
            "pipeline": pipeline,
            "epilogue": epilogue,
            "scheduler": scheduler,
        },
        # Canonical dispatcher KernelKey shape. Strings here are already in
        # to_string() form so identifier = pure concatenation.
        "signature": {
            "dtype_a": datatype,
            "dtype_b": datatype,
            "dtype_c": out_dtype,
            "dtype_acc": acc_dtype,
            "layout_a": _LAYOUT_CHAR[layout_a],
            "layout_b": _LAYOUT_CHAR[layout_b],
            "layout_c": _LAYOUT_CHAR[layout_c],
            "transpose_a": False,
            "transpose_b": False,
            "grouped": False,
            "split_k": split_k,
            "elementwise_op": "PassThrough",
            "num_d_tensors": 0,
            "structured_sparsity": False,
        },
        "algorithm": {
            "tile_m": tile.tile_m,
            "tile_n": tile.tile_n,
            "tile_k": tile.tile_k,
            "warp_m": tile.warp_m,
            "warp_n": tile.warp_n,
            "warp_k": tile.warp_k,
            "warp_tile_m": tile.warp_tile_m,
            "warp_tile_n": tile.warp_tile_n,
            "warp_tile_k": tile.warp_tile_k,
            "pipeline": _PIPELINE_CANON[pipeline],
            "scheduler": _SCHEDULER_CANON[scheduler],
            "epilogue": _EPILOGUE_CANON[epilogue],
            "block_size": block_size,
            "double_buffer": pipeline in _DOUBLE_BUFFER_PIPELINES,
            "persistent": persistent,
            "preshuffle": pipeline in ("preshufflev1", "preshufflev2"),
            "transpose_c": False,
            "num_wave_groups": num_wave_groups,
            "k_block_per_cu": k_block_per_cu,
            "pad_m": pad_m,
            "pad_n": pad_n,
            "pad_k": pad_k,
        },
        "gfx_arch": gfx_arch,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="Tile Engine config JSON")
    ap.add_argument("--json", action="store_true", help="Dump full config objects as JSON")
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Write one JSON per valid config into this directory (Phase 2 foundation). "
             "Each file is named <identifier>.json.",
    )
    ap.add_argument(
        "--rejection-csv", type=Path, default=None,
        help="Write a CSV manifest of rejected (invalid/unsupported) combinations to this path. "
             "Columns: combo, reason. Enables auditing what was dropped and why.",
    )
    args = ap.parse_args()

    try:
        with open(args.config) as f:
            data = json.load(f)
        configs, rejections = translate_with_rejections(data)
    except (TranslationError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(configs, indent=2))
        return 0

    # Lazy import: identifier.py depends on te_to_dispatcher indirectly, so we
    # defer it to main() to keep translate() dependency-free.
    from identifier import encode_identifier  # noqa: PLC0415

    print(f"{len(configs)} dispatcher config(s) from {args.config} "
          f"({len(rejections)} rejected):")
    for cfg in configs:
        print(f"  {encode_identifier(cfg)}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            ident = encode_identifier(cfg)
            out = args.output_dir / f"{ident}.json"
            out.write_text(json.dumps(cfg, indent=2))
        print(f"\nWrote {len(configs)} config JSON files to {args.output_dir}")

    if args.rejection_csv:
        with open(args.rejection_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["combo", "reason"])
            writer.writeheader()
            writer.writerows(rejections)
        print(f"Wrote {len(rejections)} rejection(s) to {args.rejection_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
