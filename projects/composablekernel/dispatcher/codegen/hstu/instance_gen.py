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
        # Block-tile shape axes for sequence<kM0,kN0,kN0Sub,kN1,kK1,...>. Default
        # [0] == "use the base dim", so configs that omit these (e.g. the base-tile
        # sweep_fast.json) expand to exactly the same kernels/names as before.
        "km0": _expand_values(trait.get("km0", {"values": [0]})),
        "kn0": _expand_values(trait.get("kn0", {"values": [0]})),
        "kn0sub": _expand_values(trait.get("kn0sub", {"values": [0]})),
        "kn1": _expand_values(trait.get("kn1", {"values": [0]})),
        "kk1": _expand_values(trait.get("kk1", {"values": [0]})),
        # Warp-K (16x16x{K} bf16 MFMA). Default [0] == dispatch default (WarpK=16);
        # accepts the config key "warp_k". d=64 needs 32 (16x16x32) to satisfy the
        # pipeline WarpGemm assertion.
        "warp_k": _expand_values(trait.get("warp_k", {"values": [0]})),
    }

    configs: List[HstuKernelConfig] = []
    for (
        dt,
        causal,
        max_k,
        mtile,
        splitkv,
        km0,
        kn0,
        kn0sub,
        kn1,
        kk1,
        warp_k,
    ) in itertools.product(
        axes["data_type"],
        axes["use_causal"],
        axes["max_k"],
        axes["mtile"],
        axes["use_splitkv"],
        axes["km0"],
        axes["kn0"],
        axes["kn0sub"],
        axes["kn1"],
        axes["kk1"],
        axes["warp_k"],
    ):
        if splitkv and mtile != 64:
            continue
        # Tile-shape overrides are wired only through the no-softmax non-splitkv
        # dispatch (jagged_forward_causal_softmax_bias_dropout_dispatch). The
        # split-KV dispatch still uses the fixed ...TileSettingW form, so a
        # split-KV kernel cannot honor a tile override -- skip those combos
        # instead of silently ignoring the requested shape (this matches the
        # sweep_exhaustive config note "splitkv off to isolate tile shape").
        tile_active = any(v for v in (km0, kn0, kn0sub, kn1, kk1))
        n_tile_active = sum(1 for v in (km0, kn0, kn0sub, kn1, kk1) if v)
        if splitkv and tile_active:
            continue
        # Tile dims are all-or-nothing: a partial override (some of km0/kn0/
        # kn0sub/kn1/kk1 set, the rest left at base 0) is not a well-formed
        # sequence<kM0,kN0,kN0Sub,kN1,kK1,...>. Skipping mixed combos lets a sweep
        # express a tile family as {0, X} per dim and collapse to base-tile +
        # full-tile kernels only (no nonsensical partial tiles).
        if n_tile_active not in (0, 5):
            continue
        # warp_k <-> tile coupling (d=64). A tile override is VALID at the
        # dispatch-default WarpK (warp_k 0 -> 16, 16x16x16): the deployed example
        # binary instantiates exactly sequence<192,32,32,64,32,64> with
        # /*WarpK=*/16, /*KN0=*/32 (hstu_attention_jagged_forward_dispatch.hpp
        # try_run_tuned_jagged_forward, ~line 358-376), so the prior
        # "WarpK=16/16x16x16 fails to compile" claim was false. WarpK=32
        # (16x16x32) is also a valid tile variant, so a tile override may be
        # emitted at warp_k in {0/default(16), 16, 32}. Only the converse still
        # holds: pinning WarpK=32 on a base-tile kernel breaks it (maxk128+wk32
        # fails to compile, maxk96+wk32 is ~5x slower), so warp_k==32 REQUIRES a
        # tile override. Base-tile kernels keep the dispatch-default WarpK.
        if warp_k == 32 and not tile_active:
            continue
        # Tile-shape overrides are only wired/validated for the native d=64 path
        # (max_k == 64). Padded max_k (96/128) with a tile override is unsupported
        # for the light fast sweep and unneeded by the exhaustive grid (max_k=64
        # pinned). Base-tile kernels keep every max_k value.
        if tile_active and int(max_k) != 64:
            continue
        # Validity gates implied by the block-tile contract
        # (hstu_attention_fwd_setting.hpp:20 "MaxK % N1 == 0, N0 % K1 == 0" and
        # the sweep_exhaustive note "kN0Sub == kK1"). Gates apply only to the
        # active (nonzero) tile dims so base-tile configs are never filtered.
        if kk1 and kn0sub and kn0sub != kk1:
            continue
        if kk1 and kn0 and (kn0 % kk1) != 0:
            continue
        if kn1 and (int(max_k) % kn1) != 0:
            continue
        # WarpK must be 16 or 32 (HstuChooseWarpTile_16x16) and BlockTile::kK1 >=
        # WarpK so the warp tile fits one MFMA (override static_assert). Only
        # checked when kK1 is explicitly overridden; base kK1 (d=64 -> 32) is
        # assumed to satisfy it.
        if warp_k and warp_k not in (16, 32):
            continue
        if warp_k and kk1 and kk1 < warp_k:
            continue
        name = (
            f"jagged_{dt}_causal{int(causal)}_maxk{max_k}_mtile{mtile}"
            f"_splitkv{int(splitkv)}"
        )
        # Append distinguishing tile tokens ONLY for overridden (nonzero) dims so
        # existing 5-axis configs keep byte-identical kernel names (and JIT cache
        # identities). A base-tile sweep produces no tokens at all.
        if km0:
            name += f"_km0{km0}"
        if kn0:
            name += f"_n0{kn0}"
        if kn0sub:
            name += f"_n0s{kn0sub}"
        if kn1:
            name += f"_n1{kn1}"
        if kk1:
            name += f"_k1{kk1}"
        if warp_k:
            name += f"_wk{warp_k}"
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
                km0=int(km0),
                kn0=int(kn0),
                kn0sub=int(kn0sub),
                kn1=int(kn1),
                kk1=int(kk1),
                warp_k=int(warp_k),
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
