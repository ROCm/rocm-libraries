# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instance builder for the gemm_decode tile_engine.

Reads `configs/default_config.json`, validates each tile/trait combination via
`gemm_decode_validation_utils`, and:

  * `--list_kernels`        enumerates the sweep and writes the kernel
                            count/list files CMake consumes.
  * `--gen_single`          materializes the one instance header named by
                            `--kernel_name`.
  * `--gen_all_individual`  materializes every codegen-able instance header.

Codegen reuses the macro-parameterized blueprints in `universal/`
(`gemm_decode_universal_single_default.hpp`, `..._fp8_smallm_pertensor.hpp`):
the emitted header `#define`s the register-tile / swizzle knobs
(`GEMM_DECODE_M_PER_WARP` / `_N_PER_WARP` / `_VECTOR` / `_CHIPLET_*`) and
`#include`s the matching blueprint, which fixes the data types, dot2 path,
bias, and scale layout. Combinations without a blueprint (fp16, bias, block
scale) are skipped. The CLI mirrors `gemm_universal_instance_builder.py` so the
CMake driver wiring is a drop-in.
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


def _enumerate_kernels(config: Dict[str, Any], family: str = "universal") -> List[Dict[str, Any]]:
    tiles = list(_expand(config["tile_config"]))
    traits = list(_expand(config["trait_config"]))
    out: List[Dict[str, Any]] = []
    for tile in tiles:
        if not is_tile_config_valid(tile):
            continue
        for trait in traits:
            if not is_trait_combination_valid(trait, family):
                continue
            out.append({"tile": tile, "trait": trait})
    return out


def _chiplet_token(tile: Dict[str, Any]) -> str:
    if not bool(tile.get("chiplet_swizzle", False)):
        return "noswz"
    return f"swz{int(tile.get('chiplet_num_xcds', 8))}x{int(tile.get('chiplet_chunk_size', 8))}"


def _recipe_token(tile: Dict[str, Any]) -> str:
    """Compact tag for the P0 wvSplitKQ-recipe knobs.

    Empty for the default (single-warp, no recipe) so the existing instance
    names/headers are unchanged; otherwise encodes the active levers so the
    multi-warp / fat-WG variants get distinct compiled identities. All four
    levers (warps_per_block / stage_a_in_lds / stream_b / persistent) are
    compile-time Problem flags, so they live together as tile knobs.
    """
    wpb = int(tile.get("warps_per_block", 1))
    parts = []
    if wpb > 1:
        parts.append(f"wpb{wpb}")
    if bool(tile.get("stage_a_in_lds", False)):
        parts.append("alds")
    if bool(tile.get("stream_b", False)):
        parts.append("ntb")
    if bool(tile.get("persistent", False)):
        parts.append("pers")
    return ("_" + "_".join(parts)) if parts else ""


def _scale_token(trait: Dict[str, Any]) -> str:
    x_scale = str(trait.get("x_scale_layout", "void"))
    w_scale = str(trait.get("w_scale_layout", "void"))
    if x_scale == "PerTensor" and w_scale == "PerTensor":
        tag = "pertensor"
    elif x_scale == "PerToken" and w_scale == "PerTensor":
        tag = "pertoken"
    elif x_scale.startswith("Block2D") and w_scale.startswith("Block2D"):
        tag = "blockscale"
    else:
        tag = "unscaled"
    return f"{tag}{'_bias' if bool(trait.get('has_bias', False)) else ''}"


def _kernel_name(
    datatype: str,
    layout: str,
    tile: Dict[str, Any],
    trait: Dict[str, Any],
    family: str = "universal",
) -> str:
    # The scale/bias token disambiguates configs that share the same register
    # tile but differ in epilogue/scale layout (so the generated headers and
    # benchmark targets get unique names).
    return (
        f"gemm_decode_{family}_{datatype}_{layout}_"
        f"{trait['pipeline']}_{trait['epilogue']}_{trait['scheduler']}_"
        f"split{trait['split_k']}_{_scale_token(trait)}_v{tile['vector_size']}_"
        f"m{tile['m_per_warp']}n{tile['n_per_warp']}_{_chiplet_token(tile)}"
        f"{_recipe_token(tile)}"
    )


# The blueprint headers in universal/ (and blockscale/) fix the data types, dot2
# path, bias, and scale layout; the generated per-config header only overrides
# the register-tile and swizzle knobs via #define before including one of them.
_UNIVERSAL_DIR  = Path(__file__).resolve().parent / "universal"
_BLOCKSCALE_DIR = Path(__file__).resolve().parent / "blockscale"


def _blueprint_dir_for(family: str) -> Path:
    return _BLOCKSCALE_DIR if family == "blockscale" else _UNIVERSAL_DIR


def _blueprint_for(datatype: str, trait: Dict[str, Any], family: str = "universal"):
    """Return (header_filename, honors_vector) for the (datatype, trait), or None.

    Only the combinations that have a hand-written blueprint are codegen-able;
    everything else (fp16, bias epilogue) is skipped until its blueprint exists.
    `family` selects the blueprint set: "universal" (unscaled / per-tensor /
    per-token) or "blockscale" (Block2D<.,.> scales).
    """
    x_scale = str(trait.get("x_scale_layout", "void"))
    w_scale = str(trait.get("w_scale_layout", "void"))
    if bool(trait.get("has_bias", False)):
        return None  # bias epilogue is a separate (P0b) blueprint.

    if family == "blockscale":
        # The blockscale kernel pins mp=np=wpb=1 and kVector=16; only the XCD
        # swizzle is a codegen axis. DSV3 convention: X=Block2D<1,128>,
        # W=Block2D<128,128>.
        if x_scale.startswith("Block2D") and w_scale.startswith("Block2D") and datatype == "fp8":
            return ("gemm_decode_blockscale_single_dsv3.hpp", False)
        return None

    unscaled = x_scale in {"void", "", "unscaled"} and w_scale in {"void", "", "unscaled"}
    per_tensor = x_scale == "PerTensor" and w_scale == "PerTensor"
    per_token = x_scale == "PerToken" and w_scale == "PerTensor"
    if unscaled and datatype == "bf16":
        return ("gemm_decode_universal_single_default.hpp", True)
    if per_tensor and datatype == "fp8":
        # kVector is pinned by the dot2 contract on this path, not a knob.
        return ("gemm_decode_universal_fp8_smallm_pertensor.hpp", False)
    if per_token and datatype == "fp8":
        # Per-token activation quant (X = [M] FP32 scale, W = per-tensor scalar);
        # same pinned-kVector dot2 K-loop as per-tensor, per-row X-scale epilogue.
        return ("gemm_decode_universal_fp8_smallm_pertoken.hpp", False)
    return None


def _instance_stem(
    datatype: str,
    layout: str,
    tile: Dict[str, Any],
    trait: Dict[str, Any],
    honors_vector: bool,
    family: str = "universal",
) -> str:
    """Compile-time instance identity.

    split_k (k_batch) and the atomic-add epilogue are *runtime* knobs in
    gemm_decode (k_batch is a kernel argument; the atomic-add store is taken
    automatically when k_batch > 1), so they do not multiply the compiled
    instances. kVector is part of the identity only on paths that honor it
    (the FP8 dot2 path pins it).
    """
    v_tag = f"v{int(tile['vector_size'])}_" if honors_vector else ""
    return (
        f"gemm_decode_{family}_{datatype}_{layout}_{_scale_token(trait)}_"
        f"{v_tag}m{int(tile['m_per_warp'])}n{int(tile['n_per_warp'])}_"
        f"{_chiplet_token(tile)}{_recipe_token(tile)}"
    )


def _instance_header_text(
    stem: str,
    blueprint: str,
    honors_vector: bool,
    tile: Dict[str, Any],
    trait: Dict[str, Any],
    family: str = "universal",
) -> str:
    swizzle = "true" if bool(tile.get("chiplet_swizzle", False)) else "false"
    # P0 wvSplitKQ-recipe knobs (warps_per_block / stage_a_in_lds / stream_b /
    # persistent) are all compile-time Problem flags, so they are tile knobs.
    # Default values reproduce the original single-warp, no-recipe instance, so a
    # non-recipe row emits the same kernel as before.
    a_lds = "true" if bool(tile.get("stage_a_in_lds", False)) else "false"
    stream_b = "true" if bool(tile.get("stream_b", False)) else "false"
    persistent = "true" if bool(tile.get("persistent", False)) else "false"
    lines = [
        f"// Generated gemm_decode instance: {stem}",
        "// Auto-emitted by gemm_decode_instance_builder.py. Overrides the register-tile",
        f"// / swizzle / recipe knobs and includes the {blueprint} blueprint.",
        "#pragma once",
        f"#define GEMM_DECODE_M_PER_WARP {int(tile['m_per_warp'])}",
        f"#define GEMM_DECODE_N_PER_WARP {int(tile['n_per_warp'])}",
    ]
    if honors_vector:
        lines.append(f"#define GEMM_DECODE_VECTOR {int(tile['vector_size'])}")
    lines += [
        f"#define GEMM_DECODE_CHIPLET_SWIZZLE {swizzle}",
        f"#define GEMM_DECODE_CHIPLET_NUM_XCDS {int(tile.get('chiplet_num_xcds', 8))}",
        f"#define GEMM_DECODE_CHIPLET_CHUNK {int(tile.get('chiplet_chunk_size', 8))}",
        f"#define GEMM_DECODE_WARPS_PER_BLOCK {int(tile.get('warps_per_block', 1))}",
        f"#define GEMM_DECODE_STAGE_A_IN_LDS {a_lds}",
        f"#define GEMM_DECODE_STREAM_B {stream_b}",
        f"#define GEMM_DECODE_PERSISTENT {persistent}",
        f'#include "{(_blueprint_dir_for(family) / blueprint).as_posix()}"',
        "",
    ]
    return "\n".join(lines)


def _emit_instance_header(
    working_path: Path,
    datatype: str,
    layout: str,
    tile: Dict[str, Any],
    trait: Dict[str, Any],
    family: str = "universal",
):
    """Materialize one per-config instance header; return its Path or None."""
    blueprint = _blueprint_for(datatype, trait, family)
    if blueprint is None:
        return None
    header_name, honors_vector = blueprint
    stem = _instance_stem(datatype, layout, tile, trait, honors_vector, family)
    text = _instance_header_text(stem, header_name, honors_vector, tile, trait, family)
    out = working_path / f"{stem}.hpp"
    out.write_text(text)
    return out


def _write_kernel_count(working_path: Path, count: int) -> None:
    (working_path / "gemm_decode_universal_kernel_count.txt").write_text(str(count))


def _write_kernel_list(
    working_path: Path,
    datatype: str,
    layout: str,
    kernels: List[Dict[str, Any]],
    family: str = "universal",
) -> None:
    lines = []
    for k in kernels:
        tile, trait = k["tile"], k["trait"]
        name = _kernel_name(datatype, layout, tile, trait, family)
        # Match gemm_universal's `<name>|<tile_config>|<trait_combo>` format.
        tile_token = (
            f"{tile['tile_m']}x{tile['tile_n']}x{tile['tile_k']}_"
            f"{tile['m_per_warp']}x{tile['n_per_warp']}x{tile['warps_per_block']}_"
            f"v{tile['vector_size']}_{_chiplet_token(tile)}"
        )
        trait_token = f"{trait['pipeline']}_{trait['epilogue']}_{trait['scheduler']}"
        lines.append(f"{name}|{tile_token}|{trait_token}")
    (working_path / "gemm_decode_universal_kernel_list.txt").write_text("\n".join(lines))


def _list_kernels(args: argparse.Namespace) -> int:
    working_path = Path(args.working_path)
    working_path.mkdir(parents=True, exist_ok=True)

    with open(args.config_json, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    family = getattr(args, "family", "universal")
    kernels = _enumerate_kernels(config, family)
    _write_kernel_count(working_path, len(kernels))
    _write_kernel_list(working_path, args.datatype, args.layout, kernels, family)
    print(f"gemm_decode: {len(kernels)} kernel configurations enumerated")
    return 0


def _gen_single(args: argparse.Namespace) -> int:
    """Materialize the single instance header named by --kernel_name."""
    if not args.kernel_name:
        print("gemm_decode --gen_single requires --kernel_name", file=sys.stderr)
        return 2
    working_path = Path(args.working_path)
    working_path.mkdir(parents=True, exist_ok=True)

    with open(args.config_json, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    family = getattr(args, "family", "universal")
    for k in _enumerate_kernels(config, family):
        tile, trait = k["tile"], k["trait"]
        if _kernel_name(args.datatype, args.layout, tile, trait, family) != args.kernel_name:
            continue
        out = _emit_instance_header(working_path, args.datatype, args.layout, tile, trait, family)
        if out is None:
            print(
                f"gemm_decode: {args.kernel_name} has no blueprint (skipped)",
                file=sys.stderr,
            )
            return 1
        print(f"gemm_decode: emitted {out}")
        return 0

    print(f"gemm_decode: no config matches kernel_name {args.kernel_name!r}", file=sys.stderr)
    return 1


def _gen_all_individual(args: argparse.Namespace) -> int:
    """Materialize every codegen-able instance header + a manifest of them."""
    working_path = Path(args.working_path)
    working_path.mkdir(parents=True, exist_ok=True)

    with open(args.config_json, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    family = getattr(args, "family", "universal")
    emitted: List[str] = []
    skipped = 0
    for k in _enumerate_kernels(config, family):
        tile, trait = k["tile"], k["trait"]
        out = _emit_instance_header(working_path, args.datatype, args.layout, tile, trait, family)
        if out is None:
            skipped += 1
            continue
        emitted.append(out.name)

    (working_path / "gemm_decode_universal_generated_headers.txt").write_text(
        "\n".join(sorted(set(emitted)))
    )
    print(
        f"gemm_decode: emitted {len(set(emitted))} instance headers "
        f"({skipped} configs skipped: no blueprint)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="gemm_decode instance builder (P0 stub)")
    parser.add_argument("--working_path", required=True)
    parser.add_argument("--datatype", default="bf16")
    parser.add_argument("--layout", default="rrr")
    parser.add_argument(
        "--family",
        default="universal",
        choices=["universal", "blockscale"],
        help="Kernel family / blueprint set to codegen.",
    )
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
    if args.gen_all_individual:
        return _gen_all_individual(args)
    if args.gen_single:
        return _gen_single(args)

    parser.error("one of --list_kernels / --gen_single / --gen_all_individual is required")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
