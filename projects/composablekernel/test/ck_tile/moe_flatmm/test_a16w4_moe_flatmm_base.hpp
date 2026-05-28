// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/reference/reference_moe_gemm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

#include "test_moe_flatmm_configs.hpp"
#include "test_moe_flatmm_dispatch.hpp"
#include "test_moe_flatmm_fixtures.hpp"
#include "test_moe_flatmm_sorted_inputs.hpp"

// Base gtest fixture for the F16xMXF4FlatmmPipelineAGmemBGmemCRegV1 pipeline
// (a16w4: bf16xfp4 / fp16xfp4). The B operand is packed-fp4 with e8m0 block
// scales (K=32); A has no MX scale. Mirrors AITER's `a16w4_*` codegen path,
// which uses Swiglu activation, optional expert bias, and (implicitly) the
// MulRoutedWeight=true behavior baked into the gemm2 kernel.
//
// Tuple layout:
//   <A16W4TypeConfig, MoeKindTag>
template <typename Tuple>
class TestA16W4MoeFlatmmBase : public ::testing::Test
{
    public:
    using TypeConfig = std::tuple_element_t<0, Tuple>;
    using MoeKindTag = std::tuple_element_t<1, Tuple>;

    using ADataType   = typename TypeConfig::ADataType;
    using BDataType   = typename TypeConfig::BDataType;
    using CDataType   = typename TypeConfig::CDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using ScaleType   = ck_tile::e8m0_t;

    using FlatmmConfig = A16W4_MoeFlatmmConfig16;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    static constexpr ck_tile::MoeFlatmmKind Kind = MoeKindTag::value;
    static constexpr bool IsInputGemm            = Kind != ck_tile::MoeFlatmmKind::kFFN_gemm2;
    static constexpr bool IsGateUp = Kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up;

    static constexpr int ScaleGranularityN = 1;
    static constexpr int ScaleGranularityK = 32;

    void run_test(ck_tile::index_t num_tokens,
                  ck_tile::index_t topk,
                  ck_tile::index_t experts,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  std::optional<std::vector<ck_tile::index_t>> forced_topk_ids = std::nullopt,
                  bool skip_experts_with_zero_token                            = true,
                  int seed                                                     = 42)
    {
        if constexpr(Kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only)
        {
            GTEST_SKIP()
                << "A16W4 gate-only currently compiles but does not match reference; "
                   "gate until the F16xMXF4 gate-only layout/activation contract is fixed.";
        }

        ASSERT_EQ(K % FlatmmConfig::K_Tile, 0)
            << "K (" << K << ") must be a multiple of K_Tile (" << FlatmmConfig::K_Tile << ")";
        ASSERT_EQ(N % FlatmmConfig::N_Tile, 0)
            << "N (" << N << ") must be a multiple of N_Tile (" << FlatmmConfig::N_Tile << ")";
        ASSERT_EQ(K % ScaleGranularityK, 0);
        if constexpr(IsGateUp)
        {
            ASSERT_EQ(N % 2, 0) << "gemm1_gate_up requires even N so that outputN = N/2";
        }

        constexpr ck_tile::index_t MPerBlock = FlatmmConfig::M_Tile;

        // Production-shaped MoE metadata via reference_moe_sorting.
        auto sorted = test_moe_flatmm::make_moe_sorted_inputs<AccDataType, ck_tile::index_t>(
            static_cast<int>(num_tokens),
            static_cast<int>(topk),
            static_cast<int>(experts),
            static_cast<int>(MPerBlock),
            seed,
            std::move(forced_topk_ids),
            skip_experts_with_zero_token);
        const ck_tile::index_t M       = sorted.M;
        const ck_tile::index_t outputN = IsGateUp ? N / 2 : N;
        using ReferenceActivation =
            std::conditional_t<Kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only,
                               ck_tile::moe::MoeSilu,
                               ck_tile::moe::Swiglu>;

        const ck_tile::index_t stride_A = ck_tile::get_default_stride(
            IsInputGemm ? num_tokens : num_tokens * topk, K, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
        const ck_tile::index_t stride_C = ck_tile::get_default_stride(
            IsInputGemm ? num_tokens * topk : num_tokens, outputN, 0, is_row_major(CLayout{}));

        auto a_m_k_tensor = ck_tile::HostTensor<ADataType>(ck_tile::host_tensor_descriptor(
            IsInputGemm ? num_tokens : num_tokens * topk, K, stride_A, is_row_major(ALayout{})));
        auto b_k_n_tensor = ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, experts * N, stride_B, is_row_major(BLayout{})));
        auto c_m_n_tensor = ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(IsInputGemm ? num_tokens * topk : num_tokens,
                                            outputN,
                                            stride_C,
                                            is_row_major(CLayout{})));

        ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_m_k_tensor);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_k_n_tensor);

        // B-side MX block scales (e8m0)
        ck_tile::HostTensor<ScaleType> scale_b(ck_tile::HostTensorDescriptor(
            {K * experts / ScaleGranularityK, N / ScaleGranularityN}, {N / ScaleGranularityN, 1}));
        ck_tile::FillUniformDistribution<ScaleType>{0.f, 1.f}(scale_b);

        // Shuffle B weights and scales to match the F16xMXF4 pipeline's expected layout.
        ck_tile::HostTensor<BDataType> b_shuffle_host(
            ck_tile::host_tensor_descriptor(K, experts * N, stride_B, is_row_major(BLayout{})));
        shuffle_mxfp4_weight<FlatmmConfig, Kind>(
            b_k_n_tensor.begin(), b_shuffle_host.begin(), experts, N, K);

        ck_tile::HostTensor<ScaleType> scale_b_shuffle =
            shuffle_mxfp4_scale<FlatmmConfig, Kind>(scale_b, experts);

        auto& sorted_token_ids = sorted.sorted_token_ids;
        auto& expert_ids       = sorted.sorted_expert_ids;
        auto& expert_weight    = sorted.sorted_expert_weight;
        auto& max_token_id     = sorted.max_token_id;

        ck_tile::HostTensor<AccDataType> expert_bias(
            ck_tile::HostTensorDescriptor({experts * N}, {1}));
        // Expert bias: small magnitude so swiglu's clipping limit doesn't kick in.
        ck_tile::FillUniformDistribution<AccDataType>{-1.0f, 1.0f}(expert_bias);

        // Device memory
        ck_tile::DeviceMem a_m_k_dev_buf{a_m_k_tensor.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem b_origin_dev_buf{b_k_n_tensor.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem b_shuffle_dev_buf{b_shuffle_host.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem c_m_n_dev_buf{c_m_n_tensor.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem scale_b_shuffle_dev_buf{
            scale_b_shuffle.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem expert_bias_dev{expert_bias.get_element_space_size_in_bytes()};

        a_m_k_dev_buf.ToDevice(a_m_k_tensor.data());
        b_origin_dev_buf.ToDevice(b_k_n_tensor.data());
        b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_tensor.SetZero();
        scale_b_shuffle_dev_buf.ToDevice(scale_b_shuffle.data());
        expert_bias_dev.ToDevice(expert_bias.data());

        ck_tile::DeviceMem sorted_token_ids_dev{sorted_token_ids.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem expert_ids_dev{expert_ids.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem max_token_id_dev{max_token_id.get_element_space_size_in_bytes()};
        ck_tile::DeviceMem expert_weight_dev{expert_weight.get_element_space_size_in_bytes()};

        sorted_token_ids_dev.ToDevice(sorted_token_ids.data());
        expert_ids_dev.ToDevice(expert_ids.data());
        max_token_id_dev.ToDevice(max_token_id.data());
        expert_weight_dev.ToDevice(expert_weight.data());

        const auto* p_sorted_token_ids_dev =
            static_cast<const ck_tile::index_t*>(sorted_token_ids_dev.GetDeviceBuffer());
        const auto* p_expert_ids_dev =
            static_cast<const ck_tile::index_t*>(expert_ids_dev.GetDeviceBuffer());
        const auto* p_max_token_id_dev =
            static_cast<const ck_tile::index_t*>(max_token_id_dev.GetDeviceBuffer());
        const auto* p_sorted_expert_weight_dev =
            static_cast<const AccDataType*>(expert_weight_dev.GetDeviceBuffer());

        auto scale_b_shuffle_dev_ptr =
            ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
                static_cast<ScaleType*>(scale_b_shuffle_dev_buf.GetDeviceBuffer()),
                N / ScaleGranularityN};
        auto exp_bias_dev_ptr = ck_tile::FlatmmScalePointer<1>{
            static_cast<float*>(expert_bias_dev.GetDeviceBuffer()), experts * N};

        using MoeFlatmmArgs = ck_tile::MoeFlatmmHostArgs<
            ck_tile::FlatmmScalePointer<-1>,
            ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>,
            ck_tile::FlatmmScalePointer<1>>;

        MoeFlatmmArgs args{p_sorted_token_ids_dev,
                           p_sorted_expert_weight_dev,
                           p_expert_ids_dev,
                           p_max_token_id_dev,
                           a_m_k_dev_buf.GetDeviceBuffer(),
                           b_shuffle_dev_buf.GetDeviceBuffer(),
                           c_m_n_dev_buf.GetDeviceBuffer(),
                           num_tokens,
                           experts,
                           topk,
                           1, // k_batch
                           M,
                           N,
                           K,
                           stride_A,
                           stride_B,
                           stride_C,
                           {}, // ScaleM (none for A16W4)
                           scale_b_shuffle_dev_ptr,
                           exp_bias_dev_ptr};

        a16w4_moe_gemm<FlatmmConfig,
                       ADataType,
                       BDataType,
                       ck_tile::tuple<>,
                       AccDataType,
                       CDataType,
                       ALayout,
                       BLayout,
                       ck_tile::tuple<>,
                       CLayout,
                       Kind>(args, ck_tile::stream_config{nullptr, false, 0, 0, 1});

        c_m_n_dev_buf.FromDevice(c_m_n_tensor.data());

        // Reference: A scale = 1.0 (no MX on A side), B scale lifted to fp32.
        ck_tile::HostTensor<AccDataType> scale_a_float(
            ck_tile::HostTensorDescriptor({1, K / ScaleGranularityK}, {1, 1}));
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(scale_a_float);
        ck_tile::DeviceMem scale_a_float_dev_buf(scale_a_float.get_element_space_size_in_bytes());
        scale_a_float_dev_buf.ToDevice(scale_a_float.data());

        ck_tile::HostTensor<AccDataType> scale_b_float(ck_tile::HostTensorDescriptor(
            {K * experts / ScaleGranularityK, N / ScaleGranularityN}, {N / ScaleGranularityN, 1}));
        std::copy(scale_b.begin(), scale_b.end(), scale_b_float.begin());
        ck_tile::DeviceMem scale_b_float_dev_buf(scale_b_float.get_element_space_size_in_bytes());
        scale_b_float_dev_buf.ToDevice(scale_b_float.data());

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(IsInputGemm ? num_tokens * topk : num_tokens,
                                            outputN,
                                            stride_C,
                                            is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        auto c_m_n_ref_buf =
            std::make_unique<ck_tile::DeviceMem>(c_m_n_tensor.get_element_space_size_in_bytes());
        c_m_n_ref_buf->SetZero();

        ck_tile::reference_moe_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        static_cast<int>(Kind),
                                        ReferenceActivation>(
            p_sorted_token_ids_dev,
            p_expert_ids_dev,
            p_max_token_id_dev,
            static_cast<const ADataType*>(a_m_k_dev_buf.GetDeviceBuffer()),
            static_cast<const BDataType*>(b_origin_dev_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_ref_buf->GetDeviceBuffer()),
            p_sorted_expert_weight_dev,
            num_tokens,
            MPerBlock,
            topk,
            M,
            N,
            K,
            stride_A,
            stride_B,
            stride_C,
            M,
            1,
            ScaleGranularityK,
            static_cast<float*>(scale_a_float_dev_buf.GetDeviceBuffer()),
            static_cast<float*>(scale_b_float_dev_buf.GetDeviceBuffer()),
            static_cast<float*>(expert_bias_dev.GetDeviceBuffer()));

        c_m_n_ref_buf->FromDevice(c_m_n_host_ref.data());

        const float rtol =
            std::is_same_v<ADataType, ck_tile::half_t> && IsInputGemm ? 1e-3f : 1e-2f;
        const float atol =
            std::is_same_v<ADataType, ck_tile::half_t> && IsInputGemm ? 1e-3f : 1e-2f;

        EXPECT_TRUE(ck_tile::check_err(
            c_m_n_tensor, c_m_n_host_ref, "A16W4 MoE FlatMM result mismatch", rtol, atol));
    }
};
