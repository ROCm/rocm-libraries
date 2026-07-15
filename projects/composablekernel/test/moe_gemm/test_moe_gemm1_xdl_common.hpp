// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstring>
#include <tuple>

#include "gtest/gtest.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_moe_gemm.hpp"

namespace moe_gemm_test {

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using D0DataType = float;
using D1DataType = float;
using D2DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType, D2DataType>;
using DsLayout   = ck::Tuple<Row, Col, Row>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AElementOp  = PassThrough;
using BElementOp  = PassThrough;

template <typename EDataType>
struct MulABScaleExpertWeight
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const
    {
        (void)d0;
        (void)d1;
        (void)d2;
        e = ck::type_convert<EDataType>(c);
    }
};

// Tile configuration (independent of the compute type).
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr ck::index_t MPerBlock   = 128;
static constexpr ck::index_t NPerBlock   = 128;
static constexpr ck::index_t MNPerXDL    = 16;
static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * 1);
static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * 4);
static constexpr ck::index_t BLOCKSIZE   = 256;
static constexpr ck::index_t Nswizzle    = false;
static constexpr bool MulRoutedWeight    = false;

// Compute-type-dependent packing parameters.
template <typename T>
static constexpr ck::index_t KPerBlockOf = 128 / sizeof(T);
template <typename T>
static constexpr ck::index_t K1Of = 16 / sizeof(T);
template <typename T>
static constexpr ck::index_t EVecOf = 8 / sizeof(T);

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          ck::index_t ActOP>
using DeviceMoeGemm1Instance = ck::tensor_operation::device::DeviceMoeGemm
    // clang-format off
        <      Row,      Col, DsLayout, Row, ADataType, BDataType, DsDataType, EDataType, AccDataType, EDataType,
               AElementOp,  BElementOp, MulABScaleExpertWeight<EDataType>,       GemmSpec,
               BLOCKSIZE,   MPerBlock,   NPerBlock,    KPerBlockOf<ADataType>,
               K1Of<ADataType>,   K1Of<BDataType>,
               MNPerXDL,   MNPerXDL,
               MXDLPerWave,  NXDLPerWave,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, K1Of<ADataType>, K1Of<ADataType>, 0,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, K1Of<BDataType>, K1Of<BDataType>, 0,
               2,    2,   S<1, 32, 1, 8>, S<EVecOf<EDataType>, 1, 1, 1>,
               // IsInputGemm=true -> GEMM1 (gate/up projection, gu-fusion activation)
               ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, ActOP, Nswizzle, true, MulRoutedWeight, true, int32_t, ADataType>;
// clang-format on

inline void preShuffleBuffer(const void* src_v, void* dst_v, int N, int K, int NXdl, int elem_size)
{
    const char* src = static_cast<const char*>(src_v);
    char* dst       = static_cast<char*>(dst_v);

    int KPack = 16 / elem_size;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            std::memcpy(dst + static_cast<size_t>(outputIndex) * elem_size,
                        src + static_cast<size_t>(n * K + k) * elem_size,
                        elem_size);
        }
    }
}

inline ck::index_t compute_tile_num(ck::index_t tokens, ck::index_t topk, ck::index_t experts)
{
    const ck::index_t tiles_needed = (tokens * topk + MPerBlock - 1) / MPerBlock;
    ck::index_t tile_num           = ((tiles_needed + experts - 1) / experts) * experts;
    if(tile_num < experts)
    {
        tile_num = experts;
    }
    return tile_num;
}

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          ck::index_t ActOP>
bool run_moe_gemm1(ck::index_t N,
                   ck::index_t K,
                   ck::index_t tokens,
                   ck::index_t experts,
                   ck::index_t topk,
                   double rtol = 1e-2,
                   double atol = 2e-1)
{
    using RefComputeType = AccDataType;

    const ck::index_t sorted_tile_num = compute_tile_num(tokens, topk, experts);
    const ck::index_t valid_tile_num  = sorted_tile_num;
    const ck::index_t sorted_size     = sorted_tile_num * MPerBlock;
    const ck::index_t valid_size      = valid_tile_num * MPerBlock;

    const ck::index_t StrideA        = K;
    const ck::index_t StrideB        = K;
    const ck::index_t StrideE        = N;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    const auto StrideDs              = std::array<ck::index_t, NumDTensor>{1, 1, 1};
    const ck::index_t KBatch         = 1;

    Tensor<ck::index_t> expert_ids(HostTensorDescriptor({sorted_tile_num}, {1}));
    Tensor<ck::index_t> sorted_token_ids(HostTensorDescriptor({sorted_size}, {1}));
    Tensor<ck::index_t> max_token_id(HostTensorDescriptor({1 + sorted_tile_num}));
    max_token_id.mData = {valid_size};

    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = i / (valid_tile_num / experts);
    }

    const int token_per_tile = (tokens * topk + valid_tile_num - 1) / valid_tile_num;
    int tokenid              = 0;
    for(int i = 0; i < sorted_size; i++)
    {
        const int tile_off = i % MPerBlock;
        if(tile_off < token_per_tile && tokenid < tokens * topk)
        {
            sorted_token_ids.mData[i] = (tokenid % tokens) | ((tokenid / tokens) << 24);
            tokenid++;
        }
        else
        {
            sorted_token_ids.mData[i] = tokens;
        }
    }

    Tensor<ADataType> a0_t_k(HostTensorDescriptor({tokens, K}, {K, 1}));
    Tensor<BDataType> b0_e_n_k(HostTensorDescriptor({experts, K, N * 2}, {N * 2 * K, 1, K}, Col{}));
    Tensor<BDataType> b0_preshuffled(
        HostTensorDescriptor({experts, K, N * 2}, {N * 2 * K, 1, K}, Col{}));
    Tensor<D0DataType> d0_t_n(HostTensorDescriptor({tokens, N}, {StrideDs[0], 0}));
    Tensor<D1DataType> d1_e_n(
        HostTensorDescriptor({experts, N * 2}, {StrideDs[1] * N * 2, StrideDs[1]}));
    Tensor<D2DataType> d2_e_n(HostTensorDescriptor({sorted_size, N}, {1, 0}, Bypass{}));
    Tensor<EDataType> e_t_n_host_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}, Row{}));
    Tensor<EDataType> e_t_n_device_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}, Row{}));

    a0_t_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.1, 0.1});
    d0_t_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
    d1_e_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0.0, 1.0});

    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) *
                                   sorted_token_ids.mDesc.GetElementSpaceSize());
    DeviceMem expert_ids_dev(sizeof(ck::index_t) * expert_ids.mDesc.GetElementSpaceSize());
    DeviceMem max_token_id_dev(sizeof(ck::index_t) * max_token_id.mDesc.GetElementSpaceSize());
    DeviceMem a0_device_buf(sizeof(ADataType) * a0_t_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(BDataType) * b0_e_n_k.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_t_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_e_n.mDesc.GetElementSpaceSize());
    DeviceMem d2_device_buf(sizeof(D2DataType) * d2_e_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_t_n_device_result.mDesc.GetElementSpaceSize());

    sorted_token_ids_dev.ToDevice(sorted_token_ids.mData.data());
    expert_ids_dev.ToDevice(expert_ids.mData.data());
    max_token_id_dev.ToDevice(max_token_id.mData.data());
    a0_device_buf.ToDevice(a0_t_k.mData.data());
    d0_device_buf.ToDevice(d0_t_n.mData.data());
    d1_device_buf.ToDevice(d1_e_n.mData.data());
    d2_device_buf.ToDevice(d2_e_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = MulABScaleExpertWeight<EDataType>{};

    auto device_op = DeviceMoeGemm1Instance<ADataType, BDataType, EDataType, AccDataType, ActOP>{};

    const int NPerXdl = device_op.GetPreShuffleParameters();
    preShuffleBuffer(b0_e_n_k.mData.data(),
                     b0_preshuffled.mData.data(),
                     N * 2 * experts,
                     K,
                     NPerXdl,
                     sizeof(BDataType));
    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker  = device_op.MakeInvoker();
    auto argument = device_op.MakeArgument(
        sorted_token_ids_dev.GetDeviceBuffer(),
        expert_ids_dev.GetDeviceBuffer(),
        max_token_id_dev.GetDeviceBuffer(),
        a0_device_buf.GetDeviceBuffer(),
        b0_device_buf.GetDeviceBuffer(),
        std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                            d1_device_buf.GetDeviceBuffer(),
                                            d2_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        tokens,
        topk,
        sorted_size,
        N,
        K,
        StrideA,
        StrideB,
        StrideDs,
        StrideE,
        KBatch,
        a_element_op,
        b_element_op,
        cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        ADD_FAILURE() << "device op does not support this MoE GEMM1 problem: N=" << N << " K=" << K
                      << " tokens=" << tokens << " experts=" << experts << " topk=" << topk;
        return false;
    }

    invoker.Run(argument, StreamConfig{nullptr, false, 0, 0, 1});
    e_device_buf.FromDevice(e_t_n_device_result.mData.data());

    Tensor<RefComputeType> c_t_k_n({tokens, topk, N}, {topk * N, N, 1}, Row{});

    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMoeGemm<ADataType,
                                                                               BDataType,
                                                                               RefComputeType,
                                                                               D2DataType,
                                                                               AccDataType,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ActOP,
                                                                               MulRoutedWeight>;
    auto ref_moe_gemm = ReferenceGemmInstance{};
    auto ref_invoker  = ref_moe_gemm.MakeInvoker();
    auto ref_argument = ref_moe_gemm.MakeArgument(sorted_token_ids,
                                                  expert_ids,
                                                  max_token_id,
                                                  MPerBlock,
                                                  a0_t_k,
                                                  d0_t_n,
                                                  b0_e_n_k,
                                                  d1_e_n,
                                                  c_t_k_n,
                                                  d2_e_n,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});
    ref_invoker.Run(ref_argument);

    for(int m = 0; m < valid_size; ++m)
    {
        const int fuse_t  = sorted_token_ids.mData[m];
        const int t       = fuse_t & 0xffffff;
        const int topk_id = (fuse_t & 0xff000000) >> 24;
        if(t >= tokens)
        {
            continue;
        }
        const int e = expert_ids(m / MPerBlock);
        for(int n = 0; n < N; ++n)
        {
            cde_element_op(e_t_n_host_result(t, topk_id, n),
                           c_t_k_n(t, topk_id, n),
                           d0_t_n(t, n),
                           d1_e_n(e, n),
                           d2_e_n(e, n));
        }
    }

    return ck::utils::check_err(
        e_t_n_device_result, e_t_n_host_result, "Error: Incorrect results!", rtol, atol);
}

template <typename ADataType, typename BDataType, typename EDataType, typename AccDataType>
void run_moe_gemm1(
    ck::index_t N, ck::index_t K, ck::index_t tokens, ck::index_t experts, ck::index_t topk)
{
    EXPECT_TRUE(
        (run_moe_gemm1<ADataType, BDataType, EDataType, AccDataType, 0>(N, K, tokens, experts, topk)))
        << "gelu_and_mul failed for N=" << N << " K=" << K << " tokens=" << tokens
        << " experts=" << experts << " topk=" << topk;
    EXPECT_TRUE(
        (run_moe_gemm1<ADataType, BDataType, EDataType, AccDataType, 1>(N, K, tokens, experts, topk)))
        << "silu_and_mul failed for N=" << N << " K=" << K << " tokens=" << tokens
        << " experts=" << experts << " topk=" << topk;
    EXPECT_TRUE(
        (run_moe_gemm1<ADataType, BDataType, EDataType, AccDataType, 4>(N, K, tokens, experts, topk)))
        << "gelu_tanh_and_mul failed for N=" << N << " K=" << K << " tokens=" << tokens
        << " experts=" << experts << " topk=" << topk;
}

} // namespace moe_gemm_test
