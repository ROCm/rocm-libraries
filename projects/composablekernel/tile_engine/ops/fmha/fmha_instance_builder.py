#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA kernel sweep builder for the tile engine.

Expands JSON sweep configs via the self-contained pipeline rules in
fmha_pipeline_rules.py, achieving exact parity with CK's get_fwd_blobs().

Usage:
    python fmha_instance_builder.py configs/receipt0_fwd.json --arch gfx950
    python fmha_instance_builder.py configs/fwd_ci.json --arch gfx950 --list
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from fmha_utils import FmhaKernelConfig, get_dispatcher_root  # noqa: E402
from fmha_pipeline_rules import (  # noqa: E402
    ARCH_DTYPES,
    SPLITKV_TILES_FP16,
    SPLITKV_TILES_FP8,
    SPLITKV_COMBINE_HDIMS_FP16,
    SPLITKV_COMBINE_HDIMS_FP8,
    PAGEDKV_TILES_FP16,
    PAGEDKV_TILES_FP8,
    APPENDKV_TILES_FP16,
    APPENDKV_TILES_FP8,
    BATCH_PREFILL_TILES_FP16,
    BATCH_PREFILL_TILES_FP8,
    BWD_DQ_DK_DV_TILES_FP16,
    BWD_DQ_DK_DV_EXTRA_TILES,
    BWD_DOT_DO_O_HDIMS,
    BWD_CONVERT_DQ_HDIMS,
    BWD_CONVERT_DQ_TILE_GROUPS,
    get_bwd_dq_dk_dv_pipelines,
    get_bwd_dq_dk_dv_extra_pipelines,
    get_bwd_dot_do_o_pipelines,
    get_bwd_convert_dq_pipelines,
    get_pipelines_for_config,
    get_splitkv_pipelines,
    get_splitkv_combine_pipelines,
    get_pagedkv_pipelines,
    get_appendkv_pipelines,
    get_batch_prefill_pipelines,
    tile_compatible,
    _check_mode,
)

VARIANT_TO_FAMILY = {
    "fwd": "fwd",
    "bwd": "bwd_dq_dk_dv",
    "splitkv": "fwd_splitkv",
    "appendkv": "fwd_appendkv",
    "pagedkv": "fwd_pagedkv",
    "batch_prefill": "batch_prefill",
}

MODES = ["batch", "group"]


def _pad_val(s: str) -> int:
    """Convert pad string to int: 'f'->0, 't'->1, '8'->8."""
    if s == "f":
        return 0
    if s == "t":
        return 1
    return int(s)


# Maps from PipelineSpec feature flags to FmhaKernelConfig field values
_MASK_MAP = {"no": "no", "causal": "top_left", "generic": "generic"}
_BIAS_MAP = {"no": "no", "bias": "bias", "alibi": "alibi"}


def _load_arch_specs() -> dict:
    specs_path = get_dispatcher_root() / "codegen" / "fmha_arch_specs.json"
    with open(specs_path) as f:
        return json.load(f)


def _get_tile_lookup(
    arch_specs: dict, arch: str
) -> Dict[str, Dict[Tuple[int, int], List[Tuple]]]:
    """Build {dtype -> {(hdim_q, hdim_v) -> [full_tile_tuple, ...]}} from arch_specs."""
    arch_info = None
    for a, info in arch_specs.get("architectures", {}).items():
        if a == arch or arch.startswith(a[:5]):
            arch_info = info
            break
    if arch_info is None:
        return {}

    combos = arch_info.get("hdim_tile_combos", {})
    lookup: Dict[str, Dict[Tuple[int, int], List[Tuple]]] = {}
    for dtype, hdim_map in combos.items():
        lookup[dtype] = {}
        for hdim_key, tiles in hdim_map.items():
            parts = hdim_key.split("_")
            hq, hv = int(parts[0]), int(parts[1])
            lookup[dtype][(hq, hv)] = [tuple(t) for t in tiles]
    return lookup


def expand_sweep(
    config_path: str, arch: str, receipt: int = 0
) -> List[FmhaKernelConfig]:
    """Expand JSON sweep config using self-contained pipeline rules.

    Pipeline rules (fmha_pipeline_rules.py) generate ALL valid kernels for the
    receipt. The JSON trait_config acts as an allow-list filter: if a trait key
    is present, only the listed values survive. If absent, all values pass.

    This means:
      - receipt0_fwd.json (no trait_config) -> full 11,980 kernels
      - fwd_ci.json (fp16, qr_async, no mask, no bias) -> small subset
    """
    with open(config_path) as f:
        config = json.load(f)

    variant = config["variant"]
    family = VARIANT_TO_FAMILY[variant]

    arch_specs = _load_arch_specs()
    tile_lookup = _get_tile_lookup(arch_specs, arch)

    # Build allow-list filters from JSON trait_config
    trait_cfg = config.get("trait_config", {})

    def _allow(key: str, default=None):
        entry = trait_cfg.get(key)
        if entry is None:
            return default
        return set(entry.get("values", []))

    allowed_dtypes = _allow("data_type")
    allowed_pipes = _allow("pipeline")
    allowed_masks = _allow("mask")
    allowed_biases = _allow("bias")
    allowed_modes = _allow("mode")
    allowed_lse = _allow("lse")
    allowed_dropout = _allow("dropout")
    allowed_logits = _allow("logits")
    allowed_sink = _allow("sink")
    allowed_paged_kv = _allow("paged_kv")

    # Intersect requested dtypes with arch support
    arch_dtypes = set(ARCH_DTYPES.get(arch, ARCH_DTYPES.get("gfx950", [])))
    if allowed_dtypes is not None:
        dtypes = sorted(allowed_dtypes & arch_dtypes)
    else:
        dtypes = sorted(arch_dtypes)

    configs: List[FmhaKernelConfig] = []

    def _resolve_tiles(dtype):
        dt = tile_lookup.get(dtype, {})
        if not dt:
            for alias in ("fp16", "bf16"):
                if alias in tile_lookup:
                    return tile_lookup[alias]
        return dt

    def _tile_params(tile, hq, dtype, var="fwd"):
        """Compute wave/warp parameters from tile shape, matching CK's codegen.

        Returns (m0, n0, k0, n1, k1, k0max, wave_m, warp_m, warp_k).
        warp_m/warp_k are the per-warp tile sizes; wave_m is the repeat count.
        """
        m0, n0, k0 = tile[0], tile[1], tile[2]
        n1 = tile[3] if len(tile) > 3 else hq
        k1 = tile[4] if len(tile) > 4 else k0
        k0max = tile[5] if len(tile) > 5 else hq
        is_fp8 = "fp8" in dtype

        # Determine warp_m from variant and tile, matching CK's factory tile objects.
        # FWD: warp_m depends on tile size (trload tiles use 16, standard uses 32)
        # SplitKV/PagedKV/BatchPrefill: always 16x16x16
        if is_fp8:
            warp_m = 32
            warp_k = 32
        elif var in ("splitkv", "pagedkv", "appendkv", "batch_prefill"):
            warp_m = 16
            warp_k = 16
        else:
            # FWD/BWD: warp_m derived from CK's tile objects
            # CK uses wm0 = min(32, bm0) for the warp M dimension,
            # with bm0 <= 64 tiles using wm0 = min(bm0, 16 or 32) depending on pipeline
            if m0 <= 16:
                warp_m = 16
            elif m0 <= 64:
                # bm0=32 standard -> wm0=32, bm0=64 trload -> wm0=16
                warp_m = 16 if m0 == 64 and n0 < 128 else 32 if m0 == 32 else 16
            else:
                warp_m = 32
            warp_k = 16

        # wave_m: repeat count in M dimension = bm0 / warp_m
        if len(tile) > 6:
            wave_m = tile[6]
        elif is_fp8:
            wave_m = min(m0 // warp_m, hq // warp_k) if warp_m > 0 and warp_k > 0 else 2
        else:
            wave_m = m0 // warp_m if warp_m > 0 else 4

        return m0, n0, k0, n1, k1, k0max, wave_m, warp_m, warp_k

    if variant == "fwd":
        for dtype in dtypes:
            dtype_tiles = _resolve_tiles(dtype)
            if not dtype_tiles:
                continue
            for (hq, hv), tiles in sorted(dtype_tiles.items()):
                pipeline_specs = get_pipelines_for_config(arch, dtype, hq, hv, receipt)
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in pipeline_specs:
                        if not _check_mode(mode, spec):
                            continue
                        if allowed_pipes is not None and spec.tag not in allowed_pipes:
                            continue
                        mm = _MASK_MAP.get(spec.mask, spec.mask)
                        mb = _BIAS_MAP.get(spec.bias, spec.bias)
                        lv, dv, lgv, sv, skv = (
                            spec.lse == "t",
                            spec.dropout == "t",
                            spec.logits == "t",
                            spec.sink == "t",
                            spec.skip == "t",
                        )
                        if allowed_masks is not None and mm not in allowed_masks:
                            continue
                        if allowed_biases is not None and mb not in allowed_biases:
                            continue
                        if allowed_lse is not None and lv not in allowed_lse:
                            continue
                        if allowed_dropout is not None and dv not in allowed_dropout:
                            continue
                        if allowed_logits is not None and lgv not in allowed_logits:
                            continue
                        if allowed_sink is not None and sv not in allowed_sink:
                            continue
                        for tile in tiles:
                            if not tile_compatible(arch, dtype, hq, hv, spec.tag, tile):
                                continue
                            m0, n0, k0, n1, k1, k0max, wave_m, warp_m, warp_k = (
                                _tile_params(tile, hv, dtype)
                            )
                            configs.append(
                                FmhaKernelConfig(
                                    family=family,
                                    data_type=dtype,
                                    mode=mode,
                                    hdim_q=hq,
                                    hdim_v=hv,
                                    pipeline=spec.tag,
                                    tile_m0=m0,
                                    tile_n0=n0,
                                    tile_k0=k0,
                                    tile_n1=n1,
                                    tile_k1=k1,
                                    tile_k0max=k0max,
                                    wave_m0=wave_m,
                                    wave_n0=1,
                                    wave_k0=1,
                                    wave_m1=wave_m,
                                    wave_n1=1,
                                    wave_k1=1,
                                    warp_m0=warp_m,
                                    warp_n0=warp_m,
                                    warp_k0=warp_k,
                                    warp_m1=warp_m,
                                    warp_n1=warp_m,
                                    warp_k1=warp_k,
                                    pad_s=_pad_val(spec.spad),
                                    pad_sk=_pad_val(spec.skpad),
                                    pad_d=_pad_val(spec.dpad),
                                    pad_dv=_pad_val(spec.dvpad),
                                    mask=mm,
                                    bias=mb,
                                    lse=lv,
                                    dropout=dv,
                                    logits=lgv,
                                    sink=sv,
                                    skip_min_seqlen_q=skv,
                                    qscale=spec.qscale,
                                    gfx_arch=arch,
                                )
                            )

    elif variant == "splitkv":
        for dtype in dtypes:
            sk_tiles = (
                SPLITKV_TILES_FP16
                if dtype in ("fp16", "bf16")
                else SPLITKV_TILES_FP8
                if dtype in ("fp8", "bf8")
                else {}
            )
            if not sk_tiles:
                continue
            for (hq, hv), tiles_or_tile in sorted(sk_tiles.items()):
                tile_list = (
                    tiles_or_tile
                    if isinstance(tiles_or_tile, list)
                    else [tiles_or_tile]
                )
                sk_specs = get_splitkv_pipelines(dtype, hq, receipt)
                for tile in tile_list:
                    for mode in MODES:
                        if allowed_modes is not None and mode not in allowed_modes:
                            continue
                        for spec in sk_specs:
                            if mode == "group" and not (
                                spec.spad == "t" and spec.skpad == "t"
                            ):
                                continue
                            mm = _MASK_MAP.get(spec.mask, spec.mask)
                            mb = _BIAS_MAP.get(spec.bias, spec.bias)
                            if allowed_masks is not None and mm not in allowed_masks:
                                continue
                            if allowed_biases is not None and mb not in allowed_biases:
                                continue
                            lgv = (spec.logits == "t")
                            sv = (spec.sink == "t")
                            pkv = (spec.pagedkv == "t")
                            if allowed_logits is not None and lgv not in allowed_logits:
                                continue
                            if allowed_sink is not None and sv not in allowed_sink:
                                continue
                            if allowed_paged_kv is not None and pkv not in allowed_paged_kv:
                                continue
                            m0, n0, k0, n1, k1, k0max, wave_m, warp_m, warp_k = (
                                _tile_params(tile, hv, dtype, var="splitkv")
                            )
                            configs.append(
                                FmhaKernelConfig(
                                    family="fwd_splitkv",
                                    data_type=dtype,
                                    mode=mode,
                                    hdim_q=hq,
                                    hdim_v=hv,
                                    pipeline=spec.tag,
                                    tile_m0=m0,
                                    tile_n0=n0,
                                    tile_k0=k0,
                                    tile_n1=n1,
                                    tile_k1=k1,
                                    tile_k0max=k0max,
                                    wave_m0=wave_m,
                                    wave_n0=1,
                                    wave_k0=1,
                                    wave_m1=wave_m,
                                    wave_n1=1,
                                    wave_k1=1,
                                    warp_m0=warp_m,
                                    warp_n0=warp_m,
                                    warp_k0=warp_k,
                                    warp_m1=warp_m,
                                    warp_n1=warp_m,
                                    warp_k1=warp_k,
                                    pad_s=_pad_val(spec.spad),
                                    pad_sk=_pad_val(spec.skpad),
                                    pad_d=_pad_val(spec.dpad),
                                    pad_dv=_pad_val(spec.dvpad),
                                    mask=mm,
                                    bias=mb,
                                    lse=True,
                                    logits=lgv,
                                    sink=sv,
                                    paged_kv=pkv,
                                    gfx_arch=arch,
                                )
                            )
        # Also generate combine kernels
        for dtype in dtypes:
            comb_specs = get_splitkv_combine_pipelines(dtype, receipt)
            if not comb_specs:
                continue
            hdims = (
                SPLITKV_COMBINE_HDIMS_FP16
                if dtype in ("fp16", "bf16")
                else SPLITKV_COMBINE_HDIMS_FP8
                if dtype in ("fp8", "bf8")
                else []
            )
            for hv in hdims:
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in comb_specs:
                        if mode == "group" and spec.spad != "t":
                            continue
                        configs.append(
                            FmhaKernelConfig(
                                family="fwd_splitkv_combine",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hv,
                                hdim_v=hv,
                                pipeline="splitkv_combine",
                                tile_m0=32,
                                tile_n0=hv,
                                tile_k0=32,
                                tile_n1=32,
                                pad_s=_pad_val(spec.spad),
                                pad_dv=_pad_val(spec.dvpad),
                                lse=(spec.lse == "t"),
                                gfx_arch=arch,
                            )
                        )

    elif variant == "pagedkv":
        for dtype in dtypes:
            pk_tiles = (
                PAGEDKV_TILES_FP16
                if dtype in ("fp16", "bf16")
                else PAGEDKV_TILES_FP8
                if dtype in ("fp8", "bf8")
                else {}
            )
            if not pk_tiles:
                continue
            for (hq, hv), tiles_or_tile in sorted(pk_tiles.items()):
                tile_list = (
                    tiles_or_tile
                    if isinstance(tiles_or_tile, list)
                    else [tiles_or_tile]
                )
                pk_specs = get_pagedkv_pipelines(dtype, hq, receipt)
                for tile in tile_list:
                    for mode in MODES:
                        if allowed_modes is not None and mode not in allowed_modes:
                            continue
                        for spec in pk_specs:
                            if mode == "group" and not (
                                spec.spad == "t" and spec.skpad == "t"
                            ):
                                continue
                            mm = _MASK_MAP.get(spec.mask, spec.mask)
                            mb = _BIAS_MAP.get(spec.bias, spec.bias)
                            if allowed_masks is not None and mm not in allowed_masks:
                                continue
                            if allowed_biases is not None and mb not in allowed_biases:
                                continue
                            m0, n0, k0, n1, k1, k0max, wave_m, warp_m, warp_k = (
                                _tile_params(tile, hv, dtype, var="pagedkv")
                            )
                            configs.append(
                                FmhaKernelConfig(
                                    family="fwd_pagedkv",
                                    data_type=dtype,
                                    mode=mode,
                                    hdim_q=hq,
                                    hdim_v=hv,
                                    pipeline=spec.tag,
                                    tile_m0=m0,
                                    tile_n0=n0,
                                    tile_k0=k0,
                                    tile_n1=n1,
                                    tile_k1=k1,
                                    tile_k0max=k0max,
                                    wave_m0=wave_m,
                                    wave_n0=1,
                                    wave_k0=1,
                                    wave_m1=wave_m,
                                    wave_n1=1,
                                    wave_k1=1,
                                    warp_m0=warp_m,
                                    warp_n0=warp_m,
                                    warp_k0=warp_k,
                                    warp_m1=warp_m,
                                    warp_n1=warp_m,
                                    warp_k1=warp_k,
                                    pad_s=_pad_val(spec.spad),
                                    pad_sk=_pad_val(spec.skpad),
                                    pad_d=_pad_val(spec.dpad),
                                    pad_dv=_pad_val(spec.dvpad),
                                    mask=mm,
                                    bias=mb,
                                    logits=(spec.logits == "t"),
                                    skip_min_seqlen_q=(spec.skip == "t"),
                                    sink=(spec.sink == "t"),
                                    paged_kv=True,
                                    gfx_arch=arch,
                                )
                            )

    elif variant == "appendkv":
        for dtype in dtypes:
            ak_tiles = (
                APPENDKV_TILES_FP16
                if dtype in ("fp16", "bf16")
                else APPENDKV_TILES_FP8
                if dtype in ("fp8", "bf8")
                else {}
            )
            if not ak_tiles:
                continue
            ak_specs = get_appendkv_pipelines(dtype, 0, receipt)
            for hdim, tile in sorted(ak_tiles.items()):
                for spec in ak_specs:
                    configs.append(
                        FmhaKernelConfig(
                            family="fwd_appendkv",
                            data_type=dtype,
                            mode="batch",
                            hdim_q=hdim,
                            hdim_v=hdim,
                            pipeline="appendkv",
                            tile_m0=tile[0],
                            tile_n0=tile[1],
                            tile_k0=tile[2],
                            tile_n1=tile[3] if len(tile) > 3 else hdim,
                            pad_s=_pad_val(spec.spad),
                            pad_sk=_pad_val(spec.skpad),
                            pad_d=_pad_val(spec.dpad),
                            pad_dv=_pad_val(spec.dvpad),
                            rope={
                                "none": "none",
                                "interleaved": "interleaved",
                                "half_rotated": "half_rotated",
                            }.get(spec.rope, spec.rope),
                            paged_kv=(spec.pagedkv == "t"),
                            gfx_arch=arch,
                        )
                    )

    elif variant == "batch_prefill":
        page_sizes = [1, 16, 1024]

        for dtype in dtypes:
            bp_tiles = (
                BATCH_PREFILL_TILES_FP16
                if dtype in ("fp16", "bf16")
                else BATCH_PREFILL_TILES_FP8
                if dtype in ("fp8bf16",)
                else {}
            )
            if not bp_tiles:
                continue
            bp_specs = get_batch_prefill_pipelines(dtype, 128, receipt)
            for (hq, hv), tiles in sorted(bp_tiles.items()):
                for tile in tiles:
                    for mode in ["group"]:  # batch_prefill is always group mode
                        for spec in bp_specs:
                            mm = _MASK_MAP.get(spec.mask, spec.mask)
                            mb = _BIAS_MAP.get(spec.bias, spec.bias)
                            if allowed_masks is not None and mm not in allowed_masks:
                                continue
                            if allowed_biases is not None and mb not in allowed_biases:
                                continue
                            m0, n0, k0, n1, k1, k0max, wave_m, warp_m, warp_k = (
                                _tile_params(tile, hv, dtype, var="batch_prefill")
                            )
                            for ps in page_sizes:
                                # page_size=1 only with kv_layout=linear
                                if ps == 1 and spec.kv_memory_layout != "linear":
                                    continue
                                # kv_blockscale requires page_size >= bn0
                                if spec.qscale == "kv_blockscale" and ps < n0:
                                    continue
                                configs.append(
                                    FmhaKernelConfig(
                                        family="batch_prefill",
                                        data_type=dtype,
                                        mode=mode,
                                        hdim_q=hq,
                                        hdim_v=hv,
                                        pipeline="qr_async",
                                        tile_m0=m0,
                                        tile_n0=n0,
                                        tile_k0=k0,
                                        tile_n1=n1,
                                        tile_k1=k1,
                                        tile_k0max=k0max,
                                        wave_m0=wave_m,
                                        wave_n0=1,
                                        wave_k0=1,
                                        wave_m1=wave_m,
                                        wave_n1=1,
                                        wave_k1=1,
                                        warp_m0=warp_m,
                                        warp_n0=warp_m,
                                        warp_k0=warp_k,
                                        warp_m1=warp_m,
                                        warp_n1=warp_m,
                                        warp_k1=warp_k,
                                        pad_s=1,
                                        pad_sk=1,
                                        pad_d=1,
                                        pad_dv=1,
                                        mask=mm,
                                        bias=mb,
                                        lse=(spec.lse == "t"),
                                        dropout=(spec.dropout == "t"),
                                        logits=(spec.logits == "t"),
                                        paged_kv=True,
                                        page_size=ps,
                                        kv_memory_layout=spec.kv_memory_layout,
                                        kv_lookup_table=spec.kv_lookup_table,
                                        qscale=spec.qscale,
                                        gfx_arch=arch,
                                    )
                                )

    elif variant == "bwd":
        for dtype in dtypes:
            if dtype not in ("fp16", "bf16"):
                continue

            # --- dot_do_o ---
            dot_specs = get_bwd_dot_do_o_pipelines(dtype)
            for hd in BWD_DOT_DO_O_HDIMS:
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in dot_specs:
                        if mode == "group" and spec.spad != "t":
                            continue
                        configs.append(
                            FmhaKernelConfig(
                                family="bwd_dot_do_o",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hd,
                                hdim_v=hd,
                                pipeline="qr",
                                tile_m0=64,
                                pad_s=_pad_val(spec.spad),
                                pad_dv=_pad_val(spec.dvpad),
                                gfx_arch=arch,
                            )
                        )

            # Exact wave/warp lookup for bwd_dq_dk_dv, extracted from CK's codegen.
            # Key is (bm0, bn0, bk0, trload). warp_k1 differs between trload/non-trload.
            BWD_DQ_WAVE_WARP = {
                (16, 16, 128, "t"): {
                    "wave": (1, 1, 1, 1, 1, 1, 1, 1, 1),
                    "warp_k1": 16,
                },
                (16, 64, 256, "f"): {
                    "wave": (1, 4, 1, 4, 1, 1, 1, 4, 1),
                    "warp_k1": 16,
                },
                (16, 128, 128, "f"): {
                    "wave": (1, 4, 1, 4, 1, 1, 1, 4, 1),
                    "warp_k1": 16,
                },
                (16, 192, 128, "t"): {
                    "wave": (1, 4, 1, 4, 1, 1, 1, 4, 1),
                    "warp_k1": 16,
                },
                (32, 16, 64, "t"): {"wave": (1, 1, 1, 1, 1, 1, 1, 1, 1), "warp_k1": 16},
                (32, 128, 32, "f"): {
                    "wave": (1, 4, 1, 4, 1, 1, 2, 2, 1),
                    "warp_k1": 16,
                },
                (32, 128, 64, "f"): {
                    "wave": (1, 4, 1, 4, 1, 1, 1, 4, 1),
                    "warp_k1": 16,
                },
                (32, 128, 64, "t"): {
                    "wave": (1, 4, 1, 4, 1, 1, 1, 4, 1),
                    "warp_k1": 32,
                },
                (32, 128, 96, "f"): {
                    "wave": (1, 4, 1, 4, 1, 1, 2, 2, 1),
                    "warp_k1": 16,
                },
                (32, 128, 128, "t"): {
                    "wave": (1, 4, 1, 4, 1, 1, 1, 4, 1),
                    "warp_k1": 32,
                },
            }

            def _bwd_dq_wave_warp(tile, hq, trload=False):
                trl = "t" if trload else "f"
                key = (tile[0], tile[1], tile[2], trl)
                entry = BWD_DQ_WAVE_WARP.get(key)
                if entry is None:
                    # Fallback: try without trload key, default warp_k1=16
                    for k, v in BWD_DQ_WAVE_WARP.items():
                        if k[:3] == (tile[0], tile[1], tile[2]):
                            entry = v
                            break
                if entry is None:
                    bn0 = tile[1]
                    wn = min(4, max(1, bn0 // 32))
                    return {
                        "wave_m0": 1,
                        "wave_n0": wn,
                        "wave_k0": 1,
                        "wave_m1": 4,
                        "wave_n1": 1,
                        "wave_k1": 1,
                        "wave_m2": 1,
                        "wave_n2": wn,
                        "wave_k2": 1,
                        "warp_m0": 16,
                        "warp_n0": 16,
                        "warp_k0": 32,
                        "warp_m1": 16,
                        "warp_n1": 16,
                        "warp_k1": 16,
                        "warp_m2": 16,
                        "warp_n2": 16,
                        "warp_k2": 16,
                    }
                w = entry["wave"]
                wk1 = entry["warp_k1"]
                return {
                    "wave_m0": w[0],
                    "wave_n0": w[1],
                    "wave_k0": w[2],
                    "wave_m1": w[3],
                    "wave_n1": w[4],
                    "wave_k1": w[5],
                    "wave_m2": w[6],
                    "wave_n2": w[7],
                    "wave_k2": w[8],
                    "warp_m0": 16,
                    "warp_n0": 16,
                    "warp_k0": 32,
                    "warp_m1": 16,
                    "warp_n1": 16,
                    "warp_k1": wk1,
                    "warp_m2": 16,
                    "warp_n2": 16,
                    "warp_k2": 16,
                }

            # --- dq_dk_dv: main tiles ---
            dq_specs = get_bwd_dq_dk_dv_pipelines(dtype, receipt)
            for (hq, hv), tile in sorted(BWD_DQ_DK_DV_TILES_FP16.items()):
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in dq_specs:
                        mm = _MASK_MAP.get(spec.mask, spec.mask)
                        mb = _BIAS_MAP.get(spec.bias, spec.bias)
                        if allowed_masks is not None and mm not in allowed_masks:
                            continue
                        if allowed_biases is not None and mb not in allowed_biases:
                            continue
                        ww = _bwd_dq_wave_warp(tile, hq)
                        configs.append(
                            FmhaKernelConfig(
                                family="bwd_dq_dk_dv",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline="qr",
                                tile_m0=tile[0],
                                tile_n0=tile[1],
                                tile_k0=tile[2],
                                tile_n1=tile[3] if len(tile) > 3 else hv,
                                tile_k1=tile[4] if len(tile) > 4 else tile[2],
                                tile_k0max=tile[5] if len(tile) > 5 else hq,
                                tile_bwd6=tile[6] if len(tile) > 6 else 0,
                                tile_bwd7=tile[7] if len(tile) > 7 else 0,
                                tile_bwd8=tile[8] if len(tile) > 8 else 0,
                                pad_s=_pad_val(spec.spad),
                                pad_sk=_pad_val(spec.skpad),
                                pad_d=_pad_val(spec.dpad),
                                pad_dv=_pad_val(spec.dvpad),
                                mask=mm,
                                bias=mb,
                                dbias=(spec.dbias == "t"),
                                dropout=(spec.dropout != "no"),
                                dropout_variant=spec.dropout,
                                deterministic=(spec.deterministic == "t"),
                                gfx_arch=arch,
                                **ww,
                            )
                        )

            # --- dq_dk_dv: extra tiles use reduced pad product ---
            for (hq, hv), extra_entries in BWD_DQ_DK_DV_EXTRA_TILES.items():
                for tile, tag, is_batch_only in extra_entries:
                    dq_extra_specs = get_bwd_dq_dk_dv_extra_pipelines(
                        dtype, is_small=is_batch_only, receipt=receipt
                    )
                    for mode in ["batch"] if is_batch_only else MODES:
                        if allowed_modes is not None and mode not in allowed_modes:
                            continue
                        for spec in dq_extra_specs:
                            mm = _MASK_MAP.get(spec.mask, spec.mask)
                            mb = _BIAS_MAP.get(spec.bias, spec.bias)
                            ww = _bwd_dq_wave_warp(tile, hq, trload=(tag == "trload"))
                            configs.append(
                                FmhaKernelConfig(
                                    family="bwd_dq_dk_dv",
                                    data_type=dtype,
                                    mode=mode,
                                    hdim_q=hq,
                                    hdim_v=hv,
                                    pipeline="qr",
                                    tile_m0=tile[0],
                                    tile_n0=tile[1],
                                    tile_k0=tile[2],
                                    tile_n1=tile[3] if len(tile) > 3 else hv,
                                    tile_k1=tile[4] if len(tile) > 4 else tile[2],
                                    tile_k0max=tile[5] if len(tile) > 5 else hq,
                                    tile_bwd6=tile[6] if len(tile) > 6 else 0,
                                    tile_bwd7=tile[7] if len(tile) > 7 else 0,
                                    tile_bwd8=tile[8] if len(tile) > 8 else 0,
                                    tile_tag=tag,
                                    use_trload=(tag == "trload"),
                                    pad_s=_pad_val(spec.spad),
                                    pad_sk=_pad_val(spec.skpad),
                                    pad_d=_pad_val(spec.dpad),
                                    pad_dv=_pad_val(spec.dvpad),
                                    mask=mm,
                                    bias=mb,
                                    dbias=(spec.dbias == "t"),
                                    dropout=(spec.dropout != "no"),
                                    dropout_variant=spec.dropout,
                                    deterministic=(spec.deterministic == "t"),
                                    gfx_arch=arch,
                                    **ww,
                                )
                            )

            # --- convert_dq: one per (hdim, tile_group, spad, deterministic, mode) ---
            for hd in BWD_CONVERT_DQ_HDIMS:
                cvt_specs = get_bwd_convert_dq_pipelines(dtype, hd)
                n_tile_groups = BWD_CONVERT_DQ_TILE_GROUPS.get(hd, 1)
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in cvt_specs:
                        if mode == "group" and spec.spad != "t":
                            continue
                        for tile_grp in range(n_tile_groups):
                            configs.append(
                                FmhaKernelConfig(
                                    family="bwd_convert_dq",
                                    data_type=dtype,
                                    mode=mode,
                                    hdim_q=hd,
                                    hdim_v=hd,
                                    pipeline="qr",
                                    tile_m0=64,
                                    tile_tag=f"g{tile_grp}" if tile_grp > 0 else "",
                                    pad_s=_pad_val(spec.spad),
                                    pad_d=_pad_val(spec.dpad),
                                    deterministic=(spec.deterministic == "t"),
                                    gfx_arch=arch,
                                )
                            )

    # Dedup truly identical configs (same name = same compiled kernel)
    seen: set = set()
    unique: List[FmhaKernelConfig] = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    return unique


def apply_filter(
    configs: List[FmhaKernelConfig], expr: str = "", filter_file: str = ""
) -> List[FmhaKernelConfig]:
    """Apply user-defined filters to a config list.

    Args:
        expr: Python expression evaluated per config with 'c' as the config.
              Example: "c.hdim_q == 128 and c.pipeline == 'qr_async'"
        filter_file: Path to a .py file defining filter_config(c) -> bool.

    Both can be combined (AND logic).
    """
    result = configs

    if filter_file:
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_filter", filter_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "filter_config")
        result = [c for c in result if fn(c)]

    if expr:
        # Developer-only CLI flag -- not user-facing, not exposed via web APIs.
        result = [c for c in result if eval(expr, {"c": c})]  # noqa: S307

    return result


# --- Sample filter file (save as e.g. my_filter.py) ---
#
# def filter_config(c):
#     """Keep only h128 kernels with 128x128 tiles, no dropout."""
#     if c.hdim_q != 128:
#         return False
#     if c.tile_m0 != 128 or c.tile_n0 != 128:
#         return False
#     if c.dropout:
#         return False
#     return True


def main():
    parser = argparse.ArgumentParser(description="FMHA tile engine sweep builder")
    parser.add_argument("config", help="Sweep config JSON")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--receipt", type=int, default=0)
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expression per config, e.g. "c.hdim_q == 128"',
    )
    parser.add_argument(
        "--filter-file",
        default="",
        help="Path to .py file with filter_config(c) -> bool",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    configs = expand_sweep(args.config, args.arch, args.receipt)
    before = len(configs)
    configs = apply_filter(configs, args.filter_expr, args.filter_file)
    filtered = before - len(configs)

    print(
        f"Expanded {args.config} -> {before} configs"
        f"{f' (filtered {filtered}, kept {len(configs)})' if filtered else ''}"
    )

    if args.count_only:
        return

    if args.list:
        for i, c in enumerate(configs):
            print(f"  [{i}] {c.name}")


if __name__ == "__main__":
    main()
