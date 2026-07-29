// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_blockscale_bpreshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

namespace {

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using A1DataType       = F32;
using B0DataType       = FP8;
using B1DataType       = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using A0Layout = Row;
using A1Layout = Col;
using B0Layout = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr ck::index_t Scale_Block_M = 1;
static constexpr ck::index_t Scale_Block_N = 128;
static constexpr ck::index_t Scale_Block_K = 128;

using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultiD_BlockScale_Xdl_CShuffle_V3_BPreshuffle
    // clang-format off
        <Row, Col, DsLayout, ELayout,
         A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
         AElementOp,  BElementOp, CDEElementOp, GemmSpec,
         256, Scale_Block_M, Scale_Block_N, Scale_Block_K,
         128,  128,
         128, 16, 16,
         16,   16,
         8,    2,
         S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
         S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
         2,    1,  S<1, 32, 1, 8>,  S<8>,
         ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, FP8>;
// clang-format on

void PreShuffleBuffer(const FP8* src, FP8* dst, int N, int K, int NXdl)
{
    const int KPack = 16;
    const int NLane = NXdl;
    const int KLane = 64 / NLane;
    const int K0    = K / (KLane * KPack);

    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            const int n0 = n / NLane;
            const int n1 = n % NLane;

            const int k0    = k / (KLane * KPack);
            const int tempk = k % (KLane * KPack);
            const int k1    = tempk / KPack;
            const int k2    = tempk % KPack;

            dst[n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                n1 * KPack + k2] = src[n * K + k];
        }
    }
}

HostTensorDescriptor MakeDescriptor(std::size_t row, std::size_t col, std::size_t stride, bool is_row_major)
{
    using namespace ck::literals;

    return is_row_major ? HostTensorDescriptor({row, col}, {stride, 1_uz})
                        : HostTensorDescriptor({row, col}, {1_uz, stride});
}

// Fixed inputs shared by every KBatch of one shape, so any difference in the
// output is attributable to the split alone.
class SplitKFixture
{
    public:
    SplitKFixture(int M, int N, int K)
        : M_{M},
          N_{N},
          K_{K},
          a0_m_k_(MakeDescriptor(M, K, K, true)),
          b0_k_n_(MakeDescriptor(K, N, K, false)),
          a1_m_k_(MakeDescriptor((M + Scale_Block_M - 1) / Scale_Block_M,
                                 (K + Scale_Block_K - 1) / Scale_Block_K,
                                 (M + Scale_Block_M - 1) / Scale_Block_M,
                                 false)),
          b1_k_n_(MakeDescriptor((K + Scale_Block_K - 1) / Scale_Block_K,
                                 (N + Scale_Block_N - 1) / Scale_Block_N,
                                 (K + Scale_Block_K - 1) / Scale_Block_K,
                                 false))
    {
        a0_m_k_.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n_.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_m_k_.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b1_k_n_.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
    }

    // Returns false when the device op does not support this KBatch.
    bool Run(int KBatch, Tensor<EDataType>& e_m_n)
    {
        auto device_op = DeviceOpInstance{};

        Tensor<B0DataType> b0_preshuffled(MakeDescriptor(K_, N_, K_, false));
        PreShuffleBuffer(b0_k_n_.mData.data(),
                         b0_preshuffled.mData.data(),
                         N_,
                         K_,
                         device_op.GetPreShuffleParameters());

        DeviceMem a0_buf(sizeof(A0DataType) * a0_m_k_.mDesc.GetElementSpaceSize());
        DeviceMem b0_buf(sizeof(B0DataType) * b0_k_n_.mDesc.GetElementSpaceSize());
        DeviceMem a1_buf(sizeof(A1DataType) * a1_m_k_.mDesc.GetElementSpaceSize());
        DeviceMem b1_buf(sizeof(B1DataType) * b1_k_n_.mDesc.GetElementSpaceSize());
        DeviceMem e_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpaceSize());

        a0_buf.ToDevice(a0_m_k_.mData.data());
        b0_buf.ToDevice(b0_preshuffled.mData.data());
        a1_buf.ToDevice(a1_m_k_.mData.data());
        b1_buf.ToDevice(b1_k_n_.mData.data());
        e_buf.SetZero();

        constexpr ck::index_t NumDTensor = DsDataType::Size();

        auto argument = device_op.MakeArgument(a0_buf.GetDeviceBuffer(),
                                               b0_buf.GetDeviceBuffer(),
                                               std::array<const void*, NumDTensor>{},
                                               e_buf.GetDeviceBuffer(),
                                               M_,
                                               N_,
                                               K_,
                                               K_,
                                               K_,
                                               std::array<ck::index_t, NumDTensor>{},
                                               N_,
                                               a1_buf.GetDeviceBuffer(),
                                               b1_buf.GetDeviceBuffer(),
                                               AElementOp{},
                                               BElementOp{},
                                               CDEElementOp{},
                                               KBatch);

        if(!device_op.IsSupportedArgument(argument))
        {
            return false;
        }

        device_op.MakeInvoker().Run(argument, StreamConfig{nullptr, false});
        e_buf.FromDevice(e_m_n.mData.data());

        return true;
    }

    bool IsSupported(int KBatch)
    {
        auto device_op = DeviceOpInstance{};

        constexpr ck::index_t NumDTensor = DsDataType::Size();

        auto argument = device_op.MakeArgument(nullptr,
                                               nullptr,
                                               std::array<const void*, NumDTensor>{},
                                               nullptr,
                                               M_,
                                               N_,
                                               K_,
                                               K_,
                                               K_,
                                               std::array<ck::index_t, NumDTensor>{},
                                               N_,
                                               nullptr,
                                               nullptr,
                                               AElementOp{},
                                               BElementOp{},
                                               CDEElementOp{},
                                               KBatch);

        return device_op.IsSupportedArgument(argument);
    }

    Tensor<EDataType> MakeOutputTensor() const
    {
        return Tensor<EDataType>(MakeDescriptor(M_, N_, N_, true));
    }

    private:
    int M_;
    int N_;
    int K_;
    Tensor<A0DataType> a0_m_k_;
    Tensor<B0DataType> b0_k_n_;
    Tensor<A1DataType> a1_m_k_;
    Tensor<B1DataType> b1_k_n_;
};

} // namespace

// Splitting K must not change the result: the slices accumulate into C with
// atomics, so the only permitted difference is the rounding that comes from
// summing the partials in a different order.
TEST(TestGemmBlockScaleWPSplitK, MatchesSinglePass)
{
    constexpr int N        = 512;
    constexpr int K        = 1024;
    // M starts at 2 because a 1-row column-major scale tensor is rejected by
    // HostTensorDescriptor's stride validation, which is a host-helper limit.
    const std::vector<int> Ms{2, 16, 128};
    const std::vector<int> KBatches{2, 4, 8};

    for(int M : Ms)
    {
        SplitKFixture fixture{M, N, K};

        auto e_single = fixture.MakeOutputTensor();
        if(!fixture.Run(1, e_single))
        {
            GTEST_SKIP() << "device op does not support this problem on the current device";
        }

        for(int KBatch : KBatches)
        {
            SCOPED_TRACE(::testing::Message() << "M=" << M << " N=" << N << " K=" << K
                                              << " KBatch=" << KBatch);

            auto e_split = fixture.MakeOutputTensor();
            ASSERT_TRUE(fixture.Run(KBatch, e_split));

            // Each atomic accumulation rounds to bf16, which keeps 8 mantissa
            // bits, so one ULP is 2e-2 relative and is 1-4 absolute at the
            // magnitudes this problem produces. The absolute floor matters for
            // the handful of outputs where the K sum cancels to near zero and a
            // relative bound would be meaningless. A split that read the wrong
            // scales or failed to accumulate would be off by whole multiples of
            // the value, not by an ULP.
            EXPECT_TRUE(ck::utils::check_err(
                e_split, e_single, "split-K result differs from single-pass", 2e-2, 2.0));
        }
    }
}

// KRead = ceil(K / (KBatch * lcm(AK1, BK1))) * lcm(AK1, BK1). With K = 1024 and
// KBatch = 3 that is 352, which is not a multiple of Scale_Block_K, so a slice
// would own a fraction of a scale and the split must be rejected rather than
// silently computed with the wrong scales.
TEST(TestGemmBlockScaleWPSplitK, RejectsSplitOffScaleBlockBoundary)
{
    SplitKFixture fixture{128, 512, 1024};

    if(!fixture.IsSupported(1))
    {
        GTEST_SKIP() << "device op does not support this problem on the current device";
    }

    EXPECT_TRUE(fixture.IsSupported(2));
    EXPECT_FALSE(fixture.IsSupported(3));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
