#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
AQuant (A-only quantized) GEMM Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_aquant_ctypes_lib.cpp

Scope (matches Old-TE gemm_aquant_quantgrouped*.cpp):
  dtypes : fp8, bf8, fp8i4 (A=pk_int4), bf8i4 (A=pk_int4)
  layouts: rcr, rrr, crr, ccr  (non-preshufflequant)
           rcr, rrr, crr       (preshufflequant -- ccr rejected by Old-TE)
  pipeline: compv3  ->  AQuantGemmPipelineAgBgCrMem       (non-preshufflequant)
                        AQuantGemmPipelineAgBgCrCompV3    (preshufflequant)
  host args = ck_tile::QuantGemmHostArgs (aq_ptr set, bq_ptr = nullptr)

Naming convention (byte-exact with AQuantKernelConfig.name in gemm_aquant_utils.py):
    gemm_aquant_{variant}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    qg{gM}x{gN}x{gK}[_preshufflequant]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp
        (GemmConfigQuantDecodeInterwave, GemmConfigPreshuffleQuantDecode)
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from codegen_common import (
    make_gemm_aquant_kernel_name,
    gemm_aquant_effective_epilogue,
    emit_generated_header_preamble,
    emit_single_kernel_include_footer,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry maps a variant key to the (A, B, C, Q) CK types.
# For AQuant the *A* matrix is the quantized operand:
#   fp8/bf8  : A and B are the same 8-bit float, Q (A-scale) is float
#   fp8i4    : A = pk_int4 (quantized weight), B = fp8, Q (A-scale) = fp8
#   bf8i4    : A = pk_int4 (quantized weight), B = bf8, Q (A-scale) = bf8
# Matches GemmQuantTypeConfig<A, B, C, Q> in the Old-TE aquant .cpp files.
# =============================================================================

AQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
        # PrecType passed to GemmConfig<PrecType> drives K_Tile = 256/sizeof(PrecType).
        "prec_bytes": 1,
    },
    "bf8": {
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
        "prec_bytes": 1,
    },
    "fp8i4": {
        "ck_a": "ck_tile::pk_int4_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::fp8_t",
        "ck_acc": "float",
        # GemmConfig<ck_tile::fp8_t> -> K_Tile = 256/sizeof(fp8_t) = 256.
        "prec_bytes": 1,
    },
    "bf8i4": {
        "ck_a": "ck_tile::pk_int4_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::bf8_t",
        "ck_acc": "float",
        "prec_bytes": 1,
    },
}

# Layout characters -> CK layout type.
AQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# The 3-char layout tag encodes (A, B, C).  C is always RowMajor for quant kernels
# (static_assert in gemm_calc_quant).  The AQ layout is derived from the Old-TE
# run_gemm_example_prec_type dispatch table:
#   rcr : A=R B=C C=R -> AQ=R BQ=C
#   rrr : A=R B=R C=R -> AQ=R BQ=C
#   crr : A=C B=R C=R -> AQ=R BQ=C
#   ccr : A=C B=C C=R -> AQ=C BQ=C   (non-preshufflequant only)
AQUANT_AQ_LAYOUT = {
    "rcr": "r",
    "rrr": "r",
    "crr": "r",
    "ccr": "c",
}

# Pipeline map for AQuant kernels.
#   non-preshufflequant -> AQuantGemmPipelineAgBgCrMem   (base: BaseGemmPipelineAgBgCrMem)
#   preshufflequant     -> AQuantGemmPipelineAgBgCrCompV3(base: BaseGemmPipelineAgBgCrCompV3)
AQUANT_PIPELINE_MAP = {
    "mem": "ck_tile::AQuantGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::AQuantGemmPipelineAgBgCrCompV3",
}

AQUANT_BASE_PIPELINE_MAP = {
    "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
}

AQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class AQuantTileConfig:
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
class AQuantKernelSpec:
    """Complete specification for one AQuant kernel."""

    variant_key: str          # "fp8", "bf8", "fp8i4", "bf8i4"
    layout: str               # "rcr", "rrr", "crr", "ccr"
    scheduler: str            # "interwave" (decode) or "intrawave" (preshufflequant)
    tile: AQuantTileConfig
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    preshuffle_aquant: bool = False
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def pipeline_key(self) -> str:
        """Pipeline map key: preshufflequant -> compv3, else mem."""
        return "compv3" if self.preshuffle_aquant else "mem"

    @property
    def name(self) -> str:
        t = self.tile
        return make_gemm_aquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline_key,
            epilogue="cshuffle",
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_aquant=self.preshuffle_aquant,
        )


# =============================================================================
# Header generator
# =============================================================================


class AQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one AQuantKernelSpec."""

    def generate(self, spec: AQuantKernelSpec) -> str:
        variant = AQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = AQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = AQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = AQUANT_LAYOUT_TO_CK[spec.layout[2]]
        layout_aq_ck = AQUANT_LAYOUT_TO_CK[AQUANT_AQ_LAYOUT[spec.layout]]
        # BQ layout is unused for AQuant-only (bq_ptr=nullptr); Old-TE passes ColumnMajor.
        layout_bq_ck = AQUANT_LAYOUT_TO_CK["c"]

        pipeline_key = spec.pipeline_key
        pipeline_ck = AQUANT_PIPELINE_MAP[pipeline_key]
        base_pipeline_ck = AQUANT_BASE_PIPELINE_MAP[pipeline_key]
        scheduler_ck = AQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_aquant = str(spec.preshuffle_aquant).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # AQuant configs never enable TiledMMAPermuteN (see gemm_aquant_effective_epilogue),
        # so the epilogue is always CShuffle. Kept as a computed value for parity with
        # the bquant codegen and to fail loudly if the assumption ever changes.
        use_permute_n_epilogue = (
            gemm_aquant_effective_epilogue(t.tile_n, t.warp_n, t.warp_tile_n, spec.quant_group_n)
            == "permute_n"
        )
        assert not use_permute_n_epilogue, "AQuant does not support PermuteN epilogue"

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
            "AQuant (A-only quantized) GEMM", "unified_gemm_aquant_codegen.py"
        ) + f"""\
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

// QuantGroupShape<sequence<gM, gN, gK>> -- same type used for the AQ slot in the
// pipeline template; BQ is disabled via bq_ptr=nullptr at runtime for AQuant-only.
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    {spec.quant_group_m}, {spec.quant_group_n}, {spec.quant_group_k}>>;

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
    static constexpr ck_tile::index_t GroupSizeK = {spec.quant_group_k};

    static constexpr bool kPadM           = {pad_m};
    static constexpr bool kPadN           = {pad_n};
    static constexpr bool kPadK           = {pad_k};
    static constexpr bool APreshuffleQuant = {preshuffle_aquant};
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB     = false;
    static constexpr bool TransposeC      = false;
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
        ck_tile::QuantType::AQuantGrouped,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

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
            // GemmAQuantPipelineProblem<A, AQ, B, C(=Acc), Shape, Traits,
            //   AQuantGroupSize, TransposeC, ComputeDataType, Scheduler, hot, tail>
            using PipelineProblem = ck_tile::GemmAQuantPipelineProblem<
                ADataType,
                QDataType,
                BDataType,
                AccDataType,
                TileShape,
                GemmTraits,
                QuantGroupSize,
                TransposeC,
                ADataType,        // ComputeDataType
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::AQuantGrouped>;

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

""" + emit_single_kernel_include_footer(
            ns=ns,
            struct=struct,
            ck_a=ck_a,
            ck_b=ck_b,
            ck_c=ck_c,
            ck_q=ck_q,
            ck_acc=ck_acc,
            extra_lines=(
                f"using QuantGroupSize = {ns}::QuantGroupSize;\n"
                f"using ALayout = {ns}::ALayout;\n"
                f"using BLayout = {ns}::BLayout;\n"
                f"using CLayout = {ns}::CLayout;\n"
                "// AQ scale-tensor layout: RowMajor for rcr/rrr/crr, ColumnMajor for ccr.\n"
                "// The ctypes lib derives stride_AQ from this (RowMajor -> QK_A, ColumnMajor -> M).\n"
                f"using AQLayout = {ns}::AQLayout;\n"
                f"constexpr ck_tile::index_t GroupSizeK = {ns}::{struct}::GroupSizeK;"
            ),
        )


# =============================================================================
# Config sweep
# =============================================================================


def _k_tile_for(variant_key: str) -> int:
    """K_Tile = 256 / sizeof(PrecType) for the decode config.

    PrecType is the 8-bit float weight type (fp8_t / bf8_t), so sizeof==1 and
    K_Tile==256 for every supported variant (fp8, bf8, fp8i4, bf8i4).
    """
    return 256 // AQUANT_VARIANTS[variant_key]["prec_bytes"]


def _warp_tile_k_for_arch(gfx_arch: str, preshuffle_aquant: bool = False) -> int:
    """Arch-derived WarpTileK for AQuant with M_Warp_Tile=16.

    Every AQuant variant (fp8, bf8, fp8i4, bf8i4) instantiates the GEMM config with
    an 8-bit float PrecType (fp8_t/bf8_t; the pk_int4 A operand does not drive the K
    warp tile -- see gemm_aquant_quantgrouped{,_preshufflequant}.cpp GemmConfig<fp8/bf8_t>).
    Mirrors ck_tile::get_k_warp_tile<fp8_t/bf8_t, M_Warp_Tile=16, IsFlatMM>()
    (include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp):
      gfx950                         -> 128 (decode and preshufflequant)
      gfx942/other, decode           ->  32
      gfx942/other, preshufflequant  ->  64
    Using 128 on gfx942 compiles but produces all-zeros output (no valid 16x16x128
    fp8/bf8 warp-gemm on gfx942).
    """
    if "gfx950" in gfx_arch:
        return 128
    return 64 if preshuffle_aquant else 32


def _default_config(gfx_arch: str = "gfx950") -> dict:
    """Default sweep config matching GemmConfigQuantDecodeInterwave tile defaults.

    Non-preshufflequant decode kernels for every dtype x layout Old-TE supports.
    WarpTileK is arch-derived (get_k_warp_tile<fp8/bf8_t, 16>() = 128 on gfx950,
    32 on gfx942 for the decode path).
    """
    return {
        "variant_keys": ["fp8", "bf8", "fp8i4", "bf8i4"],
        "layouts": ["rcr", "rrr", "crr", "ccr"],
        "scheduler": "interwave",
        "tile_configs": [
            # GemmConfigQuantDecodeInterwave: M=16, N=64, K=256/sizeof(PrecType)=256
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16,
             "warp_tile_k": _warp_tile_k_for_arch(gfx_arch, preshuffle_aquant=False)},
        ],
        "quant_groups": [
            {"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
        "preshuffle_aquant": False,
    }


def _build_specs(config: dict) -> List[AQuantKernelSpec]:
    specs = []
    preshuffle_aquant = config.get("preshuffle_aquant", False)
    # Decode config uses Interwave; preshufflequant uses Intrawave (GemmConfigBase default).
    default_scheduler = "intrawave" if preshuffle_aquant else "interwave"
    scheduler = config.get("scheduler", default_scheduler)
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", True)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)

    for variant_key, layout, tile_dict, qg in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        config.get("quant_groups", [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        if variant_key not in AQUANT_VARIANTS:
            log.warning("Unknown variant_key %s -- skipping", variant_key)
            continue
        if layout not in AQUANT_AQ_LAYOUT:
            log.warning("Unsupported layout %s -- skipping", layout)
            continue
        # Old-TE rejects the ccr layout for the preshufflequant path.
        if preshuffle_aquant and layout == "ccr":
            log.warning("ccr layout is unsupported for preshufflequant -- skipping")
            continue

        tile = AQuantTileConfig(
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

        specs.append(AQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            scheduler=scheduler,
            tile=tile,
            quant_group_m=qg.get("quant_group_m", 1),
            quant_group_n=qg.get("quant_group_n", 1),
            quant_group_k=qg.get("quant_group_k", 128),
            preshuffle_aquant=preshuffle_aquant,
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


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    return run_codegen_cli(
        description="AQuant (A-only quantized) GEMM kernel header generator",
        op_label="AQuant",
        make_generator=AQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
        arch_aware=True,
        default_gfx_arch="gfx950",
    )


if __name__ == "__main__":
    raise SystemExit(main())
