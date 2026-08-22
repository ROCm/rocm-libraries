#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm ABQuant (A-and-B block-scale) Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_abquant_ctypes_lib.cpp

ABQuant quantizes BOTH A and B with independent block-scale group shapes:
    AQuantGroupSize is always 1x1x128 (K-wise scale on A)
    BQuantGroupSize is 1x{gN}x{gK}  (gN in {1, 128}, gK = 128)

Scope (100% parity with the Old-TE example matrix in
example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped*.cpp):
    fp8 / bf8 / fp4 dtypes, rcr layout, compv3 pipeline,
    non-preshuffle + preshuffleb + preshufflequant + preshuffleb_preshufflequant,
    optional eight_waves (gfx950 fp8/bf8 blockscale) fast path.

Naming convention (byte-exact with ABQuantKernelConfig.name in gemm_abquant_utils.py):
    gemm_abquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    aqg1x1x{aqK}_bqg1x{bqN}x{bqK}[_preshuffleb][_preshufflebq][_eightwaves]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped_{fp8,bf8,fp4}.cpp
    example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped_preshuffleb_*.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc  (gemm_calc_quant)
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp              (GemmConfig* structs)
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from codegen_common import (
    make_gemm_abquant_kernel_name,
    bquant_effective_epilogue,
    emit_generated_header_preamble,
    emit_single_kernel_include_footer,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry: A/B/C/Q ck_tile qualified type names.
# Matches example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped_*.cpp:
#   GemmQuantTypeConfig<A, B, C=half_t, Q=float>
# =============================================================================

ABQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    "bf8": {
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    # fp4: TypeConfig uses pk_fp4_t for A/B; GemmConfig template param is pk_fp4_raw_t.
    "fp4": {
        "ck_a": "ck_tile::pk_fp4_t",
        "ck_b": "ck_tile::pk_fp4_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
}

# The subset of variants that are fp8/bf8 blockscale, i.e. eligible for the
# gfx950 eight_waves fast path (IS_FP8BLOCKSCALE in run_gemm_quant_example.inc).
_FP8_BLOCKSCALE_VARIANTS = {"fp8", "bf8"}

# Layout strings supported: only rcr (RowMajor A, ColMajor B, RowMajor C).
# CLayout must be RowMajor (static_assert in gemm_calc_quant).
ABQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# ABQuant pipeline map (mirrors the ABQuantPipeline selection in
# run_gemm_quant_example.inc lines 191-196):
#   eight_waves                     -> ABQuantGemmPipelineAgBgCrEightWaves
#   DoubleSmemBuffer && PreshuffleB -> WPABQuantBPipelineAgBgCrV2
#   otherwise (compv3)              -> ABQuantGemmPipelineAgBgCrCompV3
ABQUANT_PIPELINE_MAP = {
    "compv3":      "ck_tile::ABQuantGemmPipelineAgBgCrCompV3",
    "preshuffleb": "ck_tile::WPABQuantBPipelineAgBgCrV2",
    "eightwaves":  "ck_tile::ABQuantGemmPipelineAgBgCrEightWaves",
}

# Base pipeline map (mirrors base_gemm_pipeline selection, lines 89-103):
#   eight_waves -> BaseGemmPipelineAgBgCrCompV3
#   PreshuffleB -> BaseWeightPreshufflePipelineAGmemBGmemCRegV2
#   else ABQuant-> BaseGemmPipelineAgBgCrMem
ABQUANT_BASE_PIPELINE_MAP = {
    "compv3":      "ck_tile::BaseGemmPipelineAgBgCrMem",
    "preshuffleb": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
    "eightwaves":  "ck_tile::BaseGemmPipelineAgBgCrCompV3",
}

ABQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class ABQuantTileConfig:
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
class ABQuantKernelSpec:
    """Complete specification for one ABQuant kernel."""

    variant_key: str          # "fp8" | "bf8" | "fp4"
    layout: str               # "rcr"
    pipeline: str             # "compv3" | "preshuffleb" | "eightwaves"
    epilogue: str             # "cshuffle" (permute_n derived from tile params)
    scheduler: str            # "intrawave"
    tile: ABQuantTileConfig
    # AQuantGroupSize is always 1x1x{aquant_group_k}; only K is configurable.
    aquant_group_k: int = 128
    # BQuantGroupSize is 1x{bquant_group_n}x{bquant_group_k}
    bquant_group_n: int = 1
    bquant_group_k: int = 128
    preshuffle_b: bool = False
    preshuffle_bquant: bool = False
    double_smem_buffer: bool = False
    eight_waves: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False      # ABQuant configs set kPadK=false (GemmConfigABQuantPrefill)
    transpose_c: bool = False
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_gemm_abquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            aquant_group_k=self.aquant_group_k,
            bquant_group_n=self.bquant_group_n,
            bquant_group_k=self.bquant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_bquant=self.preshuffle_bquant,
            eight_waves=self.eight_waves,
        )


# =============================================================================
# Header generator
# =============================================================================


class ABQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one ABQuantKernelSpec."""

    def generate(self, spec: ABQuantKernelSpec) -> str:
        variant = ABQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = ABQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = ABQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = ABQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # ABQuant kernel constraint (static_assert in gemm_quant_kernel.hpp):
        #   BQ layout MUST be ColumnMajor.
        # AQ layout is RowMajor for all configs EXCEPT the n=128 EightWaves fast
        # path, which Old-TE compiles with AQLayout=ColumnMajor (StrideAQ=M); see
        # run_gemm_quant_example.inc:1013-1021:
        #   ABQuantGrouped && !APreshuffleQuant && BQuantGroupSize::kN==128 &&
        #   (M_Warp*N_Warp*K_Warp==8)  ->  Row,Col,Col,Col,Row  (else all RowMajor AQ).
        # APreshuffleQuant is always false in this codegen, so the predicate reduces
        # to kN==128 && warps==8 (true only for the 4x2x1 EightWaves configs).
        # A RowMajor-AQ kernel here builds a different, slower kernel (+9..25%), so
        # emit ColumnMajor to match Old-TE exactly.
        aq_column_major = (
            spec.bquant_group_n == 128 and (t.warp_m * t.warp_n * t.warp_k == 8)
        )
        layout_aq_ck = ABQUANT_LAYOUT_TO_CK["c" if aq_column_major else "r"]
        layout_bq_ck = ABQUANT_LAYOUT_TO_CK["c"]

        pipeline_ck = ABQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = ABQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = ABQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_b = str(spec.preshuffle_b).lower()
        preshuffle_bquant = str(spec.preshuffle_bquant).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()
        transpose_c = str(spec.transpose_c).lower()

        # AComputeDataType: for fp8-blockscale ABQuant kernels the compute type is
        # the A dtype (IS_FP8BLOCKSCALE branch); otherwise void (default compute).
        # We follow the same rule the example uses.
        is_fp8_blockscale = spec.variant_key in _FP8_BLOCKSCALE_VARIANTS
        a_compute = ck_a if is_fp8_blockscale else "void"

        # Determine effective epilogue. GemmConfig::TiledMMAPermuteN is a per-config
        # property: ONLY the preshuffleB configs (GemmConfigPreshuffleB_*_Prefill)
        # override it to (N_Repeat % 2 == 0). The compv3 (GemmConfigABQuantPrefill /
        # GemmConfigPreshuffleBQuantPrefill) and eight_waves (GemmConfig*EightWaves)
        # configs inherit TiledMMAPermuteN=false from GemmConfigBase, so they always
        # use CShuffle. PermuteN is further disabled when BQuantGroupSize::kN > 1
        # (mirrors run_gemm_quant_example.inc:208-209). Keep this in lockstep with
        # make_gemm_abquant_kernel_name so the emitted name matches the emitted epilogue.
        use_permute_n_epilogue = (spec.preshuffle_b and not spec.eight_waves) and (
            bquant_effective_epilogue(
                t.tile_n, t.warp_n, t.warp_tile_n, spec.bquant_group_n
            )
            == "permute_n"
        )

        # GemmConfig::TiledMMAPermuteN drives whether the B weight matrix is
        # pre-shuffled via shuffle_b_permuteN (permute_n) or plain shuffle_b.
        # Only the non-eight_waves preshuffleB configs override it to (N_Repeat % 2
        # == 0); every other config inherits false from GemmConfigBase. Mirror the
        # same rule the example uses (run_gemm_quant_example.inc:773 selects
        # shuffle_b_permuteN when TiledMMAPermuteN && BQuantGroupSize::kN == 1).
        n_repeat = t.tile_n // (t.warp_n * t.warp_tile_n) if (t.warp_n * t.warp_tile_n) else 0
        tiled_mma_permute_n = (
            spec.preshuffle_b and not spec.eight_waves and (n_repeat % 2 == 0)
        )
        tiled_mma_permute_n_str = str(tiled_mma_permute_n).lower()
        aq_column_major_str = str(aq_column_major).lower()

        if use_permute_n_epilogue:
            epilogue_block = f"""\
            using GemmEpilogue = ck_tile::PermuteNEpilogue<
                ck_tile::PermuteNEpilogueProblem<
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
                    TransposeC,
                    false,
                    1>>;"""
        else:
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

        return emit_generated_header_preamble(
            "ABQuant (A+B block-scale) GEMM", "unified_gemm_abquant_codegen.py"
        ) + f"""\
namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};
using AccDataType = {ck_acc};
// AComputeDataType: A dtype for fp8/bf8 blockscale, else void (default compute).
using AComputeDataType = {a_compute};
using BComputeDataType = AComputeDataType;

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
using BQLayout = {layout_bq_ck};

// AQuantGroupSize is always 1x1x{spec.aquant_group_k} (K-wise scale on A).
using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    1, 1, {spec.aquant_group_k}>>;
// BQuantGroupSize is 1x{spec.bquant_group_n}x{spec.bquant_group_k}.
using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    1, {spec.bquant_group_n}, {spec.bquant_group_k}>>;

struct {struct} {{
    using ADataType        = {ns}::ADataType;
    using BDataType        = {ns}::BDataType;
    using CDataType        = {ns}::CDataType;
    using QDataType        = {ns}::QDataType;
    using AccDataType      = {ns}::AccDataType;
    using AComputeDataType = {ns}::AComputeDataType;
    using BComputeDataType = {ns}::BComputeDataType;

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
    static constexpr ck_tile::index_t AGroupSizeK = {spec.aquant_group_k};
    static constexpr ck_tile::index_t BGroupSizeK = {spec.bquant_group_k};
    static constexpr ck_tile::index_t BGroupSizeN = {spec.bquant_group_n};

    static constexpr bool kPadM            = {pad_m};
    static constexpr bool kPadN            = {pad_n};
    static constexpr bool kPadK            = {pad_k};
    // ABQuant "preshufflequant" preshuffles only the B-quant scales
    // (GemmConfigPreshuffleBQuantPrefill / *_PreshuffleBQuant_Prefill set only
    // BPreshuffleQuant=true; APreshuffleQuant stays false).
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = {preshuffle_bquant};
    static constexpr bool PreshuffleB      = {preshuffle_b};
    static constexpr bool TransposeC       = {transpose_c};
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};
    // TiledMMAPermuteN: selects shuffle_b_permuteN vs plain shuffle_b for the B
    // weight matrix (see gemm_abquant_ctypes_lib.cpp). Mirrors GemmConfig struct.
    static constexpr bool TiledMMAPermuteN = {tiled_mma_permute_n_str};
    // AQIsColumnMajor: true only for the n=128 EightWaves fast path (StrideAQ=M).
    static constexpr bool AQIsColumnMajor  = {aq_column_major_str};

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    // Config exposing the member names ck_tile::shuffle_b / shuffle_b_permuteN
    // expect (N_Warp, N_Warp_Tile, K_Warp_Tile, N_Tile). Used by the ctypes lib
    // to pre-shuffle the B weight matrix for PreshuffleB kernels, matching
    // Old-TE's host-side shuffle in run_gemm_quant_example.inc:770-789.
    struct BShuffleConfig {{
        static constexpr ck_tile::index_t N_Tile      = TileN;
        static constexpr ck_tile::index_t N_Warp      = WarpN;
        static constexpr ck_tile::index_t N_Warp_Tile = WarpTileN;
        static constexpr ck_tile::index_t K_Warp_Tile = WarpTileK;
    }};

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::ABQuantGrouped,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits,
        AComputeDataType, BComputeDataType>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
        // hot-loop / tail dispatch -- mirrors run_gemm_quant_example.inc
        const ck_tile::index_t K_split =
            (args.k_batch == 1)
                ? ck_tile::integer_least_multiple(args.K, TileK)
                : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, WarpTileK);

        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto has_hot_loop_, auto tail_number_) {{
            using PipelineProblem = ck_tile::GemmABQuantPipelineProblem<
                ADataType,
                QDataType,       // AQ dtype
                BDataType,
                QDataType,       // BQ dtype
                AccDataType,
                TileShape,
                GemmTraits,
                AQuantGroupSize,
                BQuantGroupSize,
                TransposeC,
                AComputeDataType,
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::ABQuantGrouped>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                return -1.0f;

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            using k_attr_t = ck_tile::kernel_attr<{str(spec.eight_waves).lower()}>;
            return ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<kBlockPerCu, k_attr_t>(
                    Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

""" + emit_single_kernel_include_footer(
            ns=ns,
            struct=struct,
            ck_a=ck_a,
            ck_b=ck_b,
            ck_c=ck_c,
            ck_q=ck_q,
            ck_acc=ck_acc,
            extra_lines=(
                f"using AQuantGroupSize = {ns}::AQuantGroupSize;\n"
                f"using BQuantGroupSize = {ns}::BQuantGroupSize;\n"
                f"constexpr ck_tile::index_t AGroupSizeK = {ns}::{struct}::AGroupSizeK;\n"
                f"constexpr ck_tile::index_t BGroupSizeK = {ns}::{struct}::BGroupSizeK;\n"
                f"constexpr ck_tile::index_t BGroupSizeN = {ns}::{struct}::BGroupSizeN;"
            ),
        )


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    """Default sweep config matching GemmConfigABQuantPrefill tile defaults.

    fp8/bf8 non-preshuffle, 1x1x128 A-quant / 1x1x128 B-quant, prefill tile
    128x128x128 (GemmConfigQuantPrefill), warp 1x4x1, warp_tile 16x16x32.
    """
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            {"tile_m": 128, "tile_n": 128, "tile_k": 128,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 32},
        ],
        "aquant_group_k": 128,
        "bquant_groups": [
            {"bquant_group_n": 1, "bquant_group_k": 128},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "transpose_c": False,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
        "preshuffle_b": False,
        "preshuffle_bquant": False,
        "eight_waves": False,
    }


def _build_specs(config: dict) -> List[ABQuantKernelSpec]:
    specs = []
    pipeline  = config.get("pipeline", "compv3")
    epilogue  = config.get("epilogue", "cshuffle")
    scheduler = config.get("scheduler", "intrawave")
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", False)
    transpose_c        = config.get("transpose_c", False)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)
    preshuffle_b       = config.get("preshuffle_b", False)
    preshuffle_bquant  = config.get("preshuffle_bquant", False)
    eight_waves        = config.get("eight_waves", False)
    aquant_group_k     = config.get("aquant_group_k", 128)

    for variant_key, layout, tile_dict, bqg in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        config.get("bquant_groups", [{"bquant_group_n": 1, "bquant_group_k": 128}]),
    ):
        if variant_key not in ABQUANT_VARIANTS:
            log.warning("Unknown variant_key %s -- skipping", variant_key)
            continue
        if pipeline not in ABQUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s -- skipping", pipeline)
            continue

        tile = ABQuantTileConfig(
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

        specs.append(ABQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile=tile,
            aquant_group_k=aquant_group_k,
            bquant_group_n=bqg.get("bquant_group_n", 1),
            bquant_group_k=bqg.get("bquant_group_k", 128),
            preshuffle_b=preshuffle_b,
            preshuffle_bquant=preshuffle_bquant,
            double_smem_buffer=double_smem_buffer,
            eight_waves=eight_waves,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            transpose_c=transpose_c,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs


# =============================================================================
# Generation entry point
# =============================================================================


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    return run_codegen_cli(
        description="ABQuant (A+B block-scale) GEMM kernel header generator",
        op_label="ABQuant",
        make_generator=ABQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
