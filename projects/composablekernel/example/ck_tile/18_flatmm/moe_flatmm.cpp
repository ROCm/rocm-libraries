// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

#include "moe_flatmm.hpp"
#include "moe_flatmm_impl.hpp"

#include "ck_tile/host/reference/reference_moe_gemm.hpp"

#include "run_moe_flatmm_example.inc"

template <template <typename PreType> typename FlatmmConfig>
int run_moe_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    const std::string prec_type = arg_parser.get_str("prec");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        const std::string gemm_kind = arg_parser.get_str("gemm_kind");
        if(gemm_kind == "gemm1_gate_up")
        {
            if(prec_type == "fp8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::fp8_t,
                    FlatmmConfig<ck_tile::fp8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bf8_t,
                    FlatmmConfig<ck_tile::bf8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    FlatmmConfig<ck_tile::bfloat16_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::half_t,
                    FlatmmConfig<ck_tile::half_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_up!");
            }
        }
        else if(gemm_kind == "gemm1_gate_only")
        {
            if(prec_type == "fp8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::fp8_t,
                    FlatmmConfig<ck_tile::fp8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bf8_t,
                    FlatmmConfig<ck_tile::bf8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    FlatmmConfig<ck_tile::bfloat16_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::half_t,
                    FlatmmConfig<ck_tile::half_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp4xfp4")
            {
                return run_mx_moe_gemm_example_with_layouts<
                    MXGemmTypeConfig_fp4xfp4,
                    MXMoeFlatmmConfig16,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp8xfp4")
            {
                return run_mx_moe_gemm_example_with_layouts<
                    MXGemmTypeConfig_fp8xfp4,
                    MXMoeFlatmmConfig16,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_only!");
            }
        }
        else if(gemm_kind == "gemm2")
        {
            if(prec_type == "fp8")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::fp8_t,
                                                         FlatmmConfig<ck_tile::fp8_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf8")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::bf8_t,
                                                         FlatmmConfig<ck_tile::bf8_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf16")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::bfloat16_t,
                                                         FlatmmConfig<ck_tile::bfloat16_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp16")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::half_t,
                                                         FlatmmConfig<ck_tile::half_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_up!");
            }
        }
        else
        {
            throw std::runtime_error("Unrecoginized gemm_kind parameter, only accept value "
                                     "[gemm1_gate_only | gemm1_gate_up | gemm2]");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;

    try
    {
#if CK_TILE_USE_WMMA
        return !run_moe_flatmm_example<FlatmmConfig16_Wmma>(argc, argv);
#else
        int warp_tile = arg_parser.get_int("warp_tile");
        if(warp_tile == 0)
        {
            return !run_moe_flatmm_example<FlatmmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            return !run_moe_flatmm_example<FlatmmConfig32>(argc, argv);
        }
        else if(warp_tile == 2)
        {
            return !run_moe_flatmm_example<FlatmmConfig16_950>(argc, argv);
        }
        else
        {
            return !run_moe_flatmm_example<FlatmmConfig32_950>(argc, argv);
        }
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
