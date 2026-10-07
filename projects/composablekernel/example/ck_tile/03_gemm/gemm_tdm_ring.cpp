// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"

// GEMM on the TDM ring pipeline (comp_tdm_ring, default policy: depth 3, every wave issues,
// wait-then-fill) with the TDM epilogue. gfx1250 only.
template <typename PrecType>
struct GemmConfigTDMRing : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_TDM_RING;
};

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    const std::string data_type = arg_parser.get_str("prec");
    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfigTDMRing<ck_tile::half_t>,
                                          UniversalInvoker,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfigTDMRing<ck_tile::bf16_t>,
                                          UniversalInvoker,
                                          ck_tile::bf16_t>(a_layout, b_layout, arg_parser);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for the TDM ring GEMM example!");
    }
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    auto result     = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
