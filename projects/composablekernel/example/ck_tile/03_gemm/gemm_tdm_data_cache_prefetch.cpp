// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"
#include "ck_tile/core/utility/gemm_validation.hpp"

// Template function to run GEMM with optional prefetch comparison.
// GemmConfig takes (PrecType, UseDataCachePrefetch, DataCachePrefetchToL1, ClusterM, ClusterN).
template <template <typename, bool, bool, ck_tile::index_t, ck_tile::index_t> class GemmConfig,
          ck_tile::index_t ClusterM,
          ck_tile::index_t ClusterN,
          typename ADataType,
          typename... BCAccDataTypes>
bool run_gemm_with_prefetch_comparison(const std::string& a_layout,
                                       const std::string& b_layout,
                                       ck_tile::ArgParser& arg_parser,
                                       bool compare_with_non_prefetch,
                                       bool prefetch_to_l1)
{
    using Invoker = UniversalInvoker;

    std::cout << "\n=== Running with DataCache Prefetch ENABLED (";
    std::cout << (prefetch_to_l1 ? "L1" : "L2") << ") ===\n" << std::endl;

    bool pass_prefetch;
    if(prefetch_to_l1)
    {
        pass_prefetch =
            run_gemm_example_prec_type<GemmConfig<ADataType, true, true, ClusterM, ClusterN>,
                                       Invoker,
                                       ADataType,
                                       BCAccDataTypes...>(a_layout, b_layout, arg_parser);
    }
    else
    {
        pass_prefetch =
            run_gemm_example_prec_type<GemmConfig<ADataType, true, false, ClusterM, ClusterN>,
                                       Invoker,
                                       ADataType,
                                       BCAccDataTypes...>(a_layout, b_layout, arg_parser);
    }

    if(compare_with_non_prefetch)
    {
        std::cout << "\n=== Running with DataCache Prefetch DISABLED ===\n" << std::endl;
        bool pass_no_prefetch =
            run_gemm_example_prec_type<GemmConfig<ADataType, false, false, ClusterM, ClusterN>,
                                       Invoker,
                                       ADataType,
                                       BCAccDataTypes...>(a_layout, b_layout, arg_parser);

        std::cout << "\n=== Comparison Summary ===" << std::endl;
        std::cout << "Note: Check the timing results above to compare performance." << std::endl;
        std::cout << "With prefetch vs without prefetch - speedup can be observed in the "
                     "timing outputs."
                  << std::endl;

        return pass_prefetch && pass_no_prefetch;
    }

    return pass_prefetch;
}

// Common GEMM example runner
template <template <typename, bool, bool, ck_tile::index_t, ck_tile::index_t> class GemmConfig,
          ck_tile::index_t ClusterM,
          ck_tile::index_t ClusterN>
int run_gemm_example_with_prefetch(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");
    std::string c_layout  = arg_parser.get_str("c_layout");

    std::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t> gemm_sizes =
        parse_gemm_size(arg_parser);

    int m = std::get<0>(gemm_sizes);
    int n = std::get<1>(gemm_sizes);
    int k = std::get<2>(gemm_sizes);

    int stride_a = arg_parser.get_int("stride_a");
    int stride_b = arg_parser.get_int("stride_b");
    int stride_c = arg_parser.get_int("stride_c");

    bool compare_with_non_prefetch = arg_parser.get_int("compare") == 1;
    bool prefetch_to_l1            = arg_parser.get_int("prefetch_l1") == 1;

    ck_tile::validate_gemm_stride(
        a_layout, b_layout, c_layout, m, n, k, stride_a, stride_b, stride_c);

    if(data_type == "fp16")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ClusterM,
                                                 ClusterN,
                                                 ck_tile::half_t,
                                                 ck_tile::half_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ClusterM,
                                                 ClusterN,
                                                 ck_tile::bf16_t,
                                                 ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ClusterM,
                                                 ClusterN,
                                                 ck_tile::fp8_t,
                                                 ck_tile::fp8_t,
                                                 ck_tile::half_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ClusterM,
                                                 ClusterN,
                                                 ck_tile::bf8_t,
                                                 ck_tile::bf8_t,
                                                 ck_tile::half_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "i8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ClusterM,
                                                 ClusterN,
                                                 ck_tile::int8_t,
                                                 ck_tile::int8_t,
                                                 int32_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for GEMM with prefetch!");
    }
}

// TDM V1 GEMM Configuration with Data Cache Prefetch control
template <typename PrecType,
          bool UseDataCachePrefetch_,
          bool DataCachePrefetchToL1_     = false,
          ck_tile::index_t kClusterSizeM_ = 1,
          ck_tile::index_t kClusterSizeN_ = 1>
struct GemmConfigTDMV1Prefetch : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_TDM_V1;
    static constexpr bool UseDataCachePrefetch      = UseDataCachePrefetch_;
    static constexpr bool DataCachePrefetchToL1     = DataCachePrefetchToL1_;

    static constexpr ck_tile::index_t kClusterSizeM = kClusterSizeM_;
    static constexpr ck_tile::index_t kClusterSizeN = kClusterSizeN_;
};

// TDM V2 GEMM Configuration with Data Cache Prefetch control
template <typename PrecType,
          bool UseDataCachePrefetch_,
          bool DataCachePrefetchToL1_     = false,
          ck_tile::index_t kClusterSizeM_ = 1,
          ck_tile::index_t kClusterSizeN_ = 1>
struct GemmConfigTDMV2Prefetch : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    // TDM V2 (requires 4 waves):  M_Warp * N_Warp * K_Warp == 4
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_TDM_V2;
    static constexpr bool UseDataCachePrefetch      = UseDataCachePrefetch_;
    static constexpr bool DataCachePrefetchToL1     = DataCachePrefetchToL1_;

    static constexpr ck_tile::index_t kClusterSizeM = kClusterSizeM_;
    static constexpr ck_tile::index_t kClusterSizeN = kClusterSizeN_;
};

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    const std::string pipeline = arg_parser.get_str("pipeline");
    const bool use_cluster_2x2 = arg_parser.get_int("use_cluster_2x2") == 1;
    const bool is_v2           = (pipeline == "v2");

    if(!is_v2 && pipeline != "v1")
        std::cerr << "Unknown pipeline '" << pipeline << "', defaulting to v1." << std::endl;

    if(is_v2)
    {
        if(use_cluster_2x2)
            return run_gemm_example_with_prefetch<GemmConfigTDMV2Prefetch, 2, 2>(arg_parser);
        else
            return run_gemm_example_with_prefetch<GemmConfigTDMV2Prefetch, 1, 1>(arg_parser);
    }
    else
    {
        if(use_cluster_2x2)
            return run_gemm_example_with_prefetch<GemmConfigTDMV1Prefetch, 2, 2>(arg_parser);
        else
            return run_gemm_example_with_prefetch<GemmConfigTDMV1Prefetch, 1, 1>(arg_parser);
    }
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    arg_parser.insert(
        "pipeline",
        "v1",
        "TDM pipeline version to use: v1 (8 waves) or v2 (4 waves, wave-specialized)");
    arg_parser.insert("use_cluster_2x2",
                      "0",
                      "0: single workgroup, 1: enable 2x2 cluster launch for TDM multicast");
    arg_parser.insert(
        "compare",
        "0",
        "0: Run with data cache prefetch only, 1: Compare with/without data cache prefetch");
    arg_parser.insert("prefetch_l1", "0", "0: Prefetch to L2 cache, 1: Prefetch to L1 cache");
    auto result = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
