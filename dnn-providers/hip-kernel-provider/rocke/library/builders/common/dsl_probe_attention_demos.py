# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention-specific CLI demos for the platform DSL probe framework.

The platform probe files (``probe_occupancy``, ``probe_config_sweep``,
``probe_intrinsic_counts``, ``probe_isa_inspect``, ``probe_lowering_compare``)
are general-purpose tools that live in ``platform/dsl_docs``.  Their built-in
attention demos were removed to keep platform/ free of ``kernels`` imports.
This driver re-exposes exactly those demos from the library side where
``from kernels import ...`` is legal.

Usage::

    # occupancy table for the tiled-2D attention sweep
    python dsl_probe_attention_demos.py --probe occupancy

    # config sweep (build-only, no GPU required)
    python dsl_probe_attention_demos.py --probe config_sweep

    # LLVM-IR intrinsic histogram
    python dsl_probe_attention_demos.py --probe intrinsic_counts

    # ISA category summary
    python dsl_probe_attention_demos.py --probe isa_inspect [--arch gfx950]

    # LLVM vs HIP backend HSACO size comparison
    python dsl_probe_attention_demos.py --probe lowering_compare [--arch gfx950]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The DSL probe framework is a loose (non-package) tool tree under
# platform/dsl_docs; locate it via the sanctioned rocke.assets accessor (proper
# library→platform import) rather than raw path math.
from rocke.assets import dsl_docs_dir  # noqa: E402

_PROBES = dsl_docs_dir() / "optimization" / "utilities" / "tools" / "dsl_probes"
sys.path.insert(0, str(_PROBES))

# ---- platform probe framework imports ----------------------------------
from probe_occupancy import (  # noqa: E402
    ARCH_BY_NAME,
    probe_occupancy,
)
from probe_config_sweep import probe_config_sweep  # noqa: E402
from probe_intrinsic_counts import probe_intrinsic_counts  # noqa: E402
from probe_isa_inspect import probe_isa_inspect  # noqa: E402
from probe_lowering_compare import probe_lowering_compare  # noqa: E402

# ---- library kernel import ---------------------------------------------
from kernels.gfx950.attention_tiled_2d import (  # noqa: E402
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)

# ---- shared attention base spec ----------------------------------------
_BASE_SINKS = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=True,
    sliding_window=0,
    has_softcap=False,
)
_BASE_NO_SINKS = dict(_BASE_SINKS, use_sinks=False)


# ---- demo functions (spec bodies copied verbatim from removed probes) --


def _demo_occupancy(arch_name: str) -> None:
    arch = ARCH_BY_NAME[arch_name]
    specs = [
        (
            "w4t64_mw16",
            UnifiedAttention2DTiledSpec(
                **_BASE_SINKS, num_warps=4, tile_size=64, block_m_per_warp=16
            ),
        ),
        (
            "w4t64_mw32",
            UnifiedAttention2DTiledSpec(
                **_BASE_SINKS, num_warps=4, tile_size=64, block_m_per_warp=32
            ),
        ),
        (
            "w2t64_mw32",
            UnifiedAttention2DTiledSpec(
                **_BASE_SINKS, num_warps=2, tile_size=64, block_m_per_warp=32
            ),
        ),
        (
            "w1t32",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=1, tile_size=32),
        ),
        (
            "w4t32",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=4, tile_size=32),
        ),
        (
            "w4t64",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=4, tile_size=64),
        ),
        (
            "w8t64",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=8, tile_size=64),
        ),
    ]
    entries = [
        (label, build_unified_attention_2d_tiled(spec), spec.num_warps)
        for label, spec in specs
    ]
    probe_occupancy(entries, arch=arch)


def _demo_config_sweep() -> None:
    base = UnifiedAttention2DTiledSpec(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
    )
    overrides = [
        {},  # baseline
        dict(num_warps=2, tile_size=64),
        dict(num_warps=4, tile_size=64),
        dict(num_warps=4, tile_size=64, waves_per_eu=3),
        dict(num_warps=4, tile_size=64, block_m_per_warp=32),
        # mfma_32x32 requires block_m_per_warp=32 — set both together.
        dict(num_warps=4, tile_size=64, block_m_per_warp=32, use_mfma_32x32=True),
        dict(
            num_warps=4,
            tile_size=64,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
        ),
        # Intentionally-invalid override to exercise the SPEC-FAIL path.
        dict(num_warps=4, tile_size=64, use_mfma_32x32=True),
    ]
    probe_config_sweep(
        build_fn=build_unified_attention_2d_tiled,
        base_spec=base,
        overrides=overrides,
        only_build=True,
    )


def _demo_intrinsic_counts() -> None:
    specs = [
        (
            "baseline_mw16",
            UnifiedAttention2DTiledSpec(
                **_BASE_NO_SINKS, num_warps=1, tile_size=64, block_m_per_warp=16
            ),
        ),
        (
            "mfma32",
            UnifiedAttention2DTiledSpec(
                **_BASE_NO_SINKS,
                num_warps=1,
                tile_size=64,
                block_m_per_warp=32,
                use_mfma_32x32=True,
            ),
        ),
        (
            "mfma32_transposed",
            UnifiedAttention2DTiledSpec(
                **_BASE_NO_SINKS,
                num_warps=1,
                tile_size=64,
                block_m_per_warp=32,
                use_mfma_32x32=True,
                use_transposed_qk_32x32=True,
            ),
        ),
    ]
    entries = [(label, build_unified_attention_2d_tiled(spec)) for label, spec in specs]
    probe_intrinsic_counts(entries)


def _demo_isa_inspect(arch: str) -> None:
    specs = [
        (
            "w4t64_prefill",
            UnifiedAttention2DTiledSpec(
                **_BASE_SINKS, num_warps=4, tile_size=64, waves_per_eu=3
            ),
        ),
        (
            "w1t32_decode",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=1, tile_size=32),
        ),
    ]
    entries = [(label, build_unified_attention_2d_tiled(spec)) for label, spec in specs]
    probe_isa_inspect(entries, mcpu=arch)


def _demo_lowering_compare(arch: str) -> None:
    specs = [
        (
            "decode",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=1, tile_size=32),
        ),
        (
            "prefill",
            UnifiedAttention2DTiledSpec(**_BASE_SINKS, num_warps=4, tile_size=64),
        ),
    ]
    entries = [(label, build_unified_attention_2d_tiled(spec)) for label, spec in specs]
    probe_lowering_compare(entries, arch=arch)


# ---- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--probe",
        choices=[
            "occupancy",
            "config_sweep",
            "intrinsic_counts",
            "isa_inspect",
            "lowering_compare",
        ],
        required=True,
        help="which probe demo to run",
    )
    p.add_argument(
        "--arch",
        choices=list(ARCH_BY_NAME.keys()),
        default="gfx950",
        help="GPU arch (used by occupancy, isa_inspect, lowering_compare)",
    )
    args = p.parse_args(argv)

    if args.probe == "occupancy":
        _demo_occupancy(args.arch)
    elif args.probe == "config_sweep":
        _demo_config_sweep()
    elif args.probe == "intrinsic_counts":
        _demo_intrinsic_counts()
    elif args.probe == "isa_inspect":
        _demo_isa_inspect(args.arch)
    elif args.probe == "lowering_compare":
        _demo_lowering_compare(args.arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
