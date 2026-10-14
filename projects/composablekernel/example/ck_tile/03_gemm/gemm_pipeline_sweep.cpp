// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"
#include "ck_tile/core/utility/gemm_validation.hpp"

#ifndef CK_GEMM_SWEEP_PIPELINE_TDM
#define CK_GEMM_SWEEP_PIPELINE_TDM 0
#endif

#ifndef CK_GEMM_SWEEP_M_TILE
#define CK_GEMM_SWEEP_M_TILE 128
#endif

#ifndef CK_GEMM_SWEEP_N_TILE
#define CK_GEMM_SWEEP_N_TILE 128
#endif

#ifndef CK_GEMM_SWEEP_K_TILE
#define CK_GEMM_SWEEP_K_TILE 64
#endif

#ifndef CK_GEMM_SWEEP_M_WARP
#define CK_GEMM_SWEEP_M_WARP 2
#endif

#ifndef CK_GEMM_SWEEP_N_WARP
#define CK_GEMM_SWEEP_N_WARP 4
#endif

#ifndef CK_GEMM_SWEEP_K_WARP
#define CK_GEMM_SWEEP_K_WARP 1
#endif

template <typename PrecType>
struct GemmPipelineSweepConfig : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = CK_GEMM_SWEEP_M_TILE;
    static constexpr ck_tile::index_t N_Tile = CK_GEMM_SWEEP_N_TILE;
    static constexpr ck_tile::index_t K_Tile = CK_GEMM_SWEEP_K_TILE;

    static constexpr ck_tile::index_t M_Warp = CK_GEMM_SWEEP_M_WARP;
    static constexpr ck_tile::index_t N_Warp = CK_GEMM_SWEEP_N_WARP;
    static constexpr ck_tile::index_t K_Warp = CK_GEMM_SWEEP_K_WARP;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool DoubleSmemBuffer = false;
    static constexpr ck_tile::GemmPipeline Pipeline =
        CK_GEMM_SWEEP_PIPELINE_TDM ? ck_tile::GemmPipeline::COMPUTE_TDM_V1
                                   : ck_tile::GemmPipeline::BASIC_V1;

    static constexpr bool EnableSmallerVectorLoadFallback = true;
    static constexpr bool LargeTensors                    = true;
};

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    const std::string data_type = arg_parser.get_str("prec");
    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string c_layout  = arg_parser.get_str("c_layout");

    const auto [m, n, k] = parse_gemm_size(arg_parser);
    ck_tile::validate_gemm_stride(a_layout,
                                  b_layout,
                                  c_layout,
                                  m,
                                  n,
                                  k,
                                  arg_parser.get_int("stride_a"),
                                  arg_parser.get_int("stride_b"),
                                  arg_parser.get_int("stride_c"));

    if(c_layout != "R")
        throw std::runtime_error("The sweep executable supports row-major C only.");

    using Invoker = UniversalInvoker;
    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmPipelineSweepConfig<ck_tile::half_t>,
                                          Invoker,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmPipelineSweepConfig<ck_tile::bf16_t>,
                                          Invoker,
                                          ck_tile::bf16_t>(a_layout, b_layout, arg_parser);
    }
#if (!CK_TILE_USE_WMMA || defined(CK_USE_WMMA_FP8)) && CK_GEMM_SWEEP_K_TILE >= 64
    if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmPipelineSweepConfig<ck_tile::fp8_t>,
                                          Invoker,
                                          ck_tile::fp8_t,
                                          ck_tile::fp8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
#endif

    throw std::runtime_error(
        "The sweep executable supports fp16 and bf16, plus fp8 when KTile is at least 64.");
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    if(!arg_parser.parse(argc, argv))
        return EXIT_FAILURE;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(const std::exception& error)
    {
        std::cerr << "Runtime error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
