#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm TensorQuant (single per-tensor scale) Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_tensor_quant_ctypes_lib.cpp

Scope (behavioral parity with Old-TE gemm_quant_tensor.cpp):
    QuantType::TensorQuant -- ONE scalar scale for A and ONE scalar scale for B.
    dtypes:   fp8, bf8   (A==B; QDataType=float; CDataType=half; Acc=float)
    layout:   rcr only   (RowMajor A, ColumnMajor B, RowMajor C)
    pipeline: compv3      (GemmPipelineAgBgCrCompV3 -- the "regular" pipeline;
                           TensorQuant reuses the non-quant compute pipeline and
                           applies the scalar scales in the epilogue)
    scheduler: intrawave

TensorQuant vs BQuantGrouped differences (mirrors run_gemm_quant_example.inc):
    - PipelineProblem   = GemmRowColTensorQuantPipelineProblem (NOT GemmBQuantPipelineProblem)
    - GemmPipeline      = GemmPipelineAgBgCrCompV3            (NOT BQuant pipeline)
    - Base pipeline     = BaseWeightPreshufflePipelineAGmemBGmemCRegV2 (else-branch, PreshuffleB=false)
    - QuantGroupSize    = QuantGroupShape<1,1,1> (placeholder; tensor path ignores it)
    - Both aq_ptr AND bq_ptr are single scalar floats (read as *aq_ptr / *bq_ptr in kernel)
    - Epilogue is invoked with (aq_scale, bq_scale) extra args
    - TiledPermuteN = GemmConfig::TiledMMAPermuteN (false for GemmConfigQuantDecode -> cshuffle)

Naming convention (byte-exact with TensorQuantKernelConfig.name in gemm_tensor_quant_utils.py):
    gemm_tensor_quant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_quant_tensor.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp  (GemmConfigQuantDecode)
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry: A==B element dtype; C=half; Q=float (scalar scale); Acc=float.
# Matches gemm_quant_tensor.cpp: GemmQuantTypeConfig<fp8_t, fp8_t, half_t, float>
#                                GemmQuantTypeConfig<bf8_t, bf8_t, half_t, float>
# =============================================================================

TENSOR_QUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "dtype_a": "fp8",
        "dtype_b": "fp8",
        "dtype_c": "half",
        "dtype_q": "float",
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    "bf8": {
        "dtype_a": "bf8",
        "dtype_b": "bf8",
        "dtype_c": "half",
        "dtype_q": "float",
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
}

# Layout strings supported: only rcr (RowMajor A, ColumnMajor B, RowMajor C).
# run_gemm_quant_example.inc only dispatches a_layout=="R" && b_layout=="C" for
# these fp8/bf8 types, and CLayout is static_asserted to RowMajor.
TENSOR_QUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# TensorQuant uses the regular (non-quant) compute pipeline; only compv3 is
# emitted by GemmConfigQuantDecode (Scheduler=Intrawave).
TENSOR_QUANT_PIPELINE_MAP = {
    "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
}

# For TensorQuant (PreshuffleB=false, not AQuant/ABQuant, IS_FP8BLOCKSCALE=false)
# run_gemm_quant_example.inc's base_gemm_pipeline lambda falls through to the
# final else branch -> BaseWeightPreshufflePipelineAGmemBGmemCRegV2.
TENSOR_QUANT_BASE_PIPELINE_MAP = {
    "compv3": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
}

TENSOR_QUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# =============================================================================
# Kernel name construction (shared by codegen + utils so they stay byte-exact)
# =============================================================================


def make_tensor_quant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
) -> str:
    """Return the canonical TensorQuant kernel name used as KERNEL_NAME.

    The epilogue segment reflects the epilogue the codegen actually emits
    (computed from tile params via tensor_quant_effective_epilogue) so the name
    always matches the compiled kernel.
    """
    effective_epilogue = tensor_quant_effective_epilogue(tile_n, warp_n, warp_tile_n)
    parts = [
        "gemm_tensor_quant",
        variant_key,
        layout,
        pipeline,
        effective_epilogue,
        scheduler,
        f"{tile_m}x{tile_n}x{tile_k}",
        f"{warp_m}x{warp_n}x{warp_k}",
        f"{warp_tile_m}x{warp_tile_n}x{warp_tile_k}",
    ]
    return "_".join(parts)


def tensor_quant_effective_epilogue(tile_n: int, warp_n: int, warp_tile_n: int) -> str:
    """Return the epilogue tag the codegen will emit for the given tile params.

    Mirrors run_gemm_quant_example.inc TensorQuant path:
        TiledPermuteN = GemmConfig::TiledMMAPermuteN   (BQuantGroupSize::kN==1 always here)
    For GemmConfigQuantDecode, TiledMMAPermuteN is false, so this returns
    "cshuffle".  We still compute N_Repeat parity to stay future-proof if a
    caller supplies a preshuffle-style config with TiledMMAPermuteN semantics.

    NOTE: TensorQuant configs (GemmConfigQuantDecode) hardcode TiledMMAPermuteN=false,
    so this always returns "cshuffle" for the supported set.
    """
    # GemmConfigQuantDecode.TiledMMAPermuteN == false (inherited GemmConfigBase),
    # so the tensor path always selects CShuffleEpilogue.
    return "cshuffle"


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class TensorQuantTileConfig:
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
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            return False
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        )


@dataclass
class TensorQuantKernelSpec:
    """Complete specification for one TensorQuant kernel."""

    variant_key: str          # "fp8" or "bf8"
    layout: str               # "rcr"
    pipeline: str             # "compv3"
    epilogue: str             # "cshuffle"
    scheduler: str            # "intrawave"
    tile: TensorQuantTileConfig
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_tensor_quant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
        )


# =============================================================================
# Header generator
# =============================================================================


class TensorQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one TensorQuantKernelSpec."""

    def generate(self, spec: TensorQuantKernelSpec) -> str:
        variant = TENSOR_QUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = TENSOR_QUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = TENSOR_QUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = TENSOR_QUANT_LAYOUT_TO_CK[spec.layout[2]]
        # AQ/BQ layouts are placeholders for TensorQuant (single scalar scale);
        # mirror the example which passes Col{} for both AQ and BQ in the rcr path.
        layout_aq_ck = TENSOR_QUANT_LAYOUT_TO_CK["c"]
        layout_bq_ck = TENSOR_QUANT_LAYOUT_TO_CK["c"]

        pipeline_ck = TENSOR_QUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = TENSOR_QUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = TENSOR_QUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # TensorQuant (GemmConfigQuantDecode) always uses the CShuffle epilogue
        # (TiledMMAPermuteN=false). The CShuffleEpilogue is invoked with the two
        # scalar scales (aq_scale, bq_scale) inside the kernel for TensorQuant.
        epilogue_block = f"""\
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<
                    typename PipelineProblem::AComputeDataType,
                    typename PipelineProblem::BComputeDataType,
                    ck_tile::tuple<>,
                    AccDataType,
                    CDataType,
                    ck_tile::tuple<>,
                    {ns}::CLayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpM, WarpN,
                    WarpTileM, WarpTileN, WarpTileK,
                    TransposeC>>;"""

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated Gemm TensorQuant kernel header.
// DO NOT EDIT -- regenerate via unified_gemm_tensor_quant_codegen.py
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue.hpp"

namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
using BQLayout = {layout_bq_ck};

// Placeholder QuantGroupSize -- TensorQuant applies one scalar scale for A and
// one scalar scale for B, so the group size is unused (matches the example's
// QuantGroupShape<1,1,1> place holder).
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 1>>;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using QDataType   = {ns}::QDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t TileM      = {t.tile_m};
    static constexpr ck_tile::index_t TileN      = {t.tile_n};
    static constexpr ck_tile::index_t TileK      = {t.tile_k};
    static constexpr ck_tile::index_t WarpM      = {t.warp_m};
    static constexpr ck_tile::index_t WarpN      = {t.warp_n};
    static constexpr ck_tile::index_t WarpK      = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM  = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN  = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK  = {t.warp_tile_k};
    static constexpr ck_tile::index_t BlockSize  = {spec.block_size};
    static constexpr int               kBlockPerCu = {spec.k_block_per_cu};

    static constexpr bool kPadM            = {pad_m};
    static constexpr bool kPadN            = {pad_n};
    static constexpr bool kPadK            = {pad_k};
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB      = false;
    static constexpr bool TransposeC       = false;
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<TileShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::TensorQuant,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;

    // ComputeDataType for the base pipeline problem: the example uses
    // AComputeDataType=void for the non-fp8-blockscale TensorQuant path.
    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
        // hot-loop / tail dispatch -- mirrors run_gemm_quant_example.inc.
        // K1 = WarpTileK; K_split uses K_Tile for k_batch==1.
        constexpr ck_tile::index_t K1 = WarpTileK;
        const ck_tile::index_t K_split =
            (args.k_batch == 1)
                ? ck_tile::integer_least_multiple(args.K, TileK)
                : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);

        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto has_hot_loop_, auto tail_number_) {{
            // TensorQuant reuses the regular GEMM compute pipeline via
            // GemmRowColTensorQuantPipelineProblem; the scalar scales are applied
            // in the epilogue (see gemm_quant_kernel.hpp TensorQuant branch).
            // Mirrors run_gemm_quant_example.inc: the compute pipeline's C slot is
            // AccDataType (float), NOT the final CDataType (half). The half
            // down-conversion happens in the CShuffle epilogue. Passing CDataType
            // here selects a non-existent fp8x(16x16x128)->half warp-gemm.
            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,      // CDataType slot = AccDataType (float)
                AccDataType,
                TileShape,
                GemmTraits,
                TransposeC,
                void,             // AComputeDataType (void for non-fp8-blockscale TensorQuant; mirrors run_gemm_quant_example.inc)
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::TensorQuant>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                return -1.0f;

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};
using AccDataType = {ck_acc};
using QuantGroupSize = {ns}::QuantGroupSize;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


_DEFAULT_GFX_ARCH = "gfx950"


def _fp8_warp_tile_k_for_arch(gfx_arch: str) -> int:
    """Arch-derived WarpTileK for fp8/bf8 with M_Warp_Tile=16.

    Mirrors ck_tile::get_k_warp_tile<fp8_t/bf8_t, M_Warp_Tile=16>()
    (include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp): 128 on gfx950,
    32 on gfx942. Using 128 on gfx942 compiles but produces all-zeros output
    (no valid 16x16x128 fp8/bf8 warp-gemm on gfx942).
    """
    return 128 if "gfx950" in gfx_arch else 32


def _default_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> dict:
    """Default sweep config matching GemmConfigQuantDecode tile defaults.

    GemmConfigQuantDecode<fp8_t/bf8_t>: M=16, N=64, K=256/sizeof(8bit)=256,
    warp 1x4x1, warp_tile 16x16x K_warp. WarpTileK is arch-derived
    (get_k_warp_tile<fp8_t/bf8_t, M_Warp_Tile=16>() = 128 on gfx950, 32 on gfx942).
    """
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16,
             "warp_tile_k": _fp8_warp_tile_k_for_arch(gfx_arch)},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
    }


def _build_specs(config: dict) -> List[TensorQuantKernelSpec]:
    specs = []
    pipeline  = config.get("pipeline", "compv3")
    epilogue  = config.get("epilogue", "cshuffle")
    scheduler = config.get("scheduler", "intrawave")
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", True)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)

    for variant_key, layout, tile_dict in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
    ):
        if variant_key not in TENSOR_QUANT_VARIANTS:
            log.warning("Unknown variant_key %s -- skipping", variant_key)
            continue
        if pipeline not in TENSOR_QUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s -- skipping", pipeline)
            continue
        if layout != "rcr":
            log.warning("Unsupported layout %s (only rcr) -- skipping", layout)
            continue

        tile = TensorQuantTileConfig(
            tile_m=tile_dict["tile_m"],
            tile_n=tile_dict["tile_n"],
            tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"],
            warp_n=tile_dict["warp_n"],
            warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"],
            warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
        )
        if not tile.is_valid():
            log.debug("Invalid tile config %s -- skipping", tile)
            continue

        specs.append(TensorQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile=tile,
            double_smem_buffer=double_smem_buffer,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs


# =============================================================================
# Generation entry point
# =============================================================================


def generate_kernels(
    output_dir: Path,
    config: Optional[dict] = None,
    parallel: bool = True,
) -> List[Path]:
    """Generate all TensorQuant kernel headers into output_dir.

    Returns list of generated .hpp paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No kernel specs produced from config -- check variant_keys and tile_configs")
        return []

    log.info("Generating %d TensorQuant kernel headers into %s", len(specs), output_dir)

    gen = TensorQuantKernelHeaderGenerator()
    generated: List[Path] = []

    def _generate_one(spec: TensorQuantKernelSpec) -> Path:
        header = gen.generate(spec)
        out_path = output_dir / f"{spec.name}.hpp"
        out_path.write_text(header)
        log.info("  wrote %s", out_path.name)
        return out_path

    if parallel and len(specs) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(_generate_one, s): s for s in specs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    generated.append(fut.result())
                except Exception as e:
                    log.error("Failed generating %s: %s", futures[fut].name, e)
    else:
        for spec in specs:
            try:
                generated.append(_generate_one(spec))
            except Exception as e:
                log.error("Failed generating %s: %s", spec.name, e)

    log.info("Generated %d / %d headers", len(generated), len(specs))
    return generated


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemm TensorQuant kernel header generator"
    )
    parser.add_argument("--output-dir", type=Path,
                        help="Directory to write generated .hpp files (required unless --list-names)")
    parser.add_argument("--config", type=Path,
                        help="JSON config file (defaults to built-in sweep)")
    parser.add_argument("--config-json", type=str,
                        help="Inline JSON config string")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel generation")
    parser.add_argument("--list-names", action="store_true",
                        help="Print kernel names that would be generated and exit")
    parser.add_argument("--gfx-arch", type=str, default=_DEFAULT_GFX_ARCH,
                        help="Target GPU arch for the built-in default config's "
                             "arch-derived WarpTileK (gfx942 -> 32, gfx950 -> 128)")
    args = parser.parse_args()

    cfg: Optional[dict] = None
    if args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            log.error("Invalid --config-json: %s", e)
            return 1
    elif args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    if args.list_names:
        specs = _build_specs(cfg or _default_config(args.gfx_arch))
        for s in specs:
            print(s.name)
        return 0

    if args.output_dir is None:
        log.error("--output-dir is required unless --list-names is given")
        return 1

    paths = generate_kernels(
        output_dir=args.output_dir,
        config=cfg,
        parallel=not args.no_parallel,
    )
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
