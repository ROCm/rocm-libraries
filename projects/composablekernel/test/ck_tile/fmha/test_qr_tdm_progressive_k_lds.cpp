// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/fmha/block/variants.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/device_memory.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>

namespace qr_tdm_progressive_k_lds_test {

template <ck_tile::index_t M>
using TestFmhaShape = ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<16, 16, 32>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<16, 16, 32>,
                                             true>;

using TestFmhaTraits = ck_tile::TileFmhaTraits<false,
                                               false,
                                               false,
                                               false,
                                               false,
                                               ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                               false,
                                               false,
                                               false,
                                               ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE>;

template <ck_tile::index_t M,
          bool UseDoubleKVLdsBuffer = false,
          bool ProgressiveDsLoadK = false,
          typename DataType = ck_tile::half_t>
using TestFmhaProblem =
    ck_tile::BlockFmhaPipelineProblem<DataType,
                                      DataType,
                                      DataType,
                                      float,
                                      float,
                                      DataType,
                                      uint8_t,
                                      float,
                                      DataType,
                                      float,
                                      DataType,
                                      TestFmhaShape<M>,
                                      false,
                                      ck_tile::ComposedAttention<0>,
                                      ck_tile::SimplifiedGenericAttentionMask<false>,
                                      false,
                                      TestFmhaTraits,
                                      UseDoubleKVLdsBuffer,
                                      ProgressiveDsLoadK>;

using SingleBufferM64Problem  = TestFmhaProblem<64>;
using DoubleBufferM64Problem  = TestFmhaProblem<64, true>;
using ProgressiveM64Problem   = TestFmhaProblem<64, true, true>;
using DoubleBufferM128Problem = TestFmhaProblem<128, true>;
using ProgressiveM128Problem  = TestFmhaProblem<128, true, true>;

static_assert(!SingleBufferM64Problem::kUseDoubleKVLdsBuffer);
static_assert(!SingleBufferM64Problem::kProgressiveDsLoadK);
static_assert(DoubleBufferM64Problem::kUseDoubleKVLdsBuffer);
static_assert(!DoubleBufferM64Problem::kProgressiveDsLoadK);
static_assert(ProgressiveM64Problem::kUseDoubleKVLdsBuffer);
static_assert(ProgressiveM64Problem::kProgressiveDsLoadK);
static_assert(DoubleBufferM128Problem::kUseDoubleKVLdsBuffer);
static_assert(!DoubleBufferM128Problem::kProgressiveDsLoadK);
static_assert(ProgressiveM128Problem::kUseDoubleKVLdsBuffer);
static_assert(ProgressiveM128Problem::kProgressiveDsLoadK);

template <typename Problem>
using TestPipeline = ck_tile::BlockFmhaPipelineQRKSVSTdm<Problem>;

static_assert(!TestPipeline<SingleBufferM64Problem>::kKLoadOnce);
static_assert(TestPipeline<DoubleBufferM64Problem>::kKLoadOnce);
static_assert(TestPipeline<DoubleBufferM128Problem>::kKLoadOnce);
static_assert(TestPipeline<ProgressiveM64Problem>::kKLoadOnce);
static_assert(TestPipeline<ProgressiveM128Problem>::kKLoadOnce);

template <typename Problem>
using TestGemm0 = ck_tile::remove_cvref_t<
    decltype(ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetQKBlockGemm<Problem>())>;

static_assert(TestGemm0<ProgressiveM64Problem>::MIterPerWarp == 1);
static_assert(TestGemm0<ProgressiveM128Problem>::MIterPerWarp == 2);
static_assert(ProgressiveM64Problem::BlockFmhaShape::kSubQKHeaddim /
                  ProgressiveM64Problem::BlockFmhaShape::kK0 ==
              4);
static_assert(ProgressiveM128Problem::BlockFmhaShape::kSubQKHeaddim /
                  ProgressiveM128Problem::BlockFmhaShape::kK0 ==
              4);

template <typename Problem>
using TestEpilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename Problem::OaccDataType,
                                                                 typename Problem::ODataType,
                                                                 false,
                                                                 false>>;

template <typename Problem>
using TestKernel = ck_tile::FmhaFwdKernel<TestPipeline<Problem>, TestEpilogue<Problem>>;

constexpr ck_tile::index_t kBatch   = 1;
constexpr ck_tile::index_t kHeads   = 1;
constexpr ck_tile::index_t kSeqlenK = 64;
constexpr ck_tile::index_t kHeadDim = 128;

template <typename DataType = ck_tile::half_t>
std::vector<DataType>
make_input(std::size_t element_count, int multiplier, int modulus, int center, float scale)
{
    std::vector<DataType> values(element_count);
    for(std::size_t i = 0; i < element_count; ++i)
    {
        const int value = (static_cast<int>(i % modulus) * multiplier) % modulus - center;
        values[i]       = ck_tile::type_convert<DataType>(value * scale);
    }
    return values;
}

template <typename Problem>
std::vector<typename Problem::ODataType> run_kernel(const ck_tile::DeviceMem& q_device,
                                                   const ck_tile::DeviceMem& k_device,
                                                   const ck_tile::DeviceMem& v_device,
                                                   ck_tile::index_t seqlen_q,
                                                   ck_tile::index_t seqlen_k = kSeqlenK)
{
    using Kernel   = TestKernel<Problem>;
    using DataType = typename Problem::ODataType;

    std::vector<DataType> output(kBatch * kHeads * seqlen_q * kHeadDim);
    ck_tile::DeviceMem output_device(output.size() * sizeof(DataType));
    output_device.SetBytePattern(0x7f);

    typename Kernel::Kargs args{};
    args.q_ptr              = q_device.GetDeviceBuffer();
    args.k_ptr              = k_device.GetDeviceBuffer();
    args.v_ptr              = v_device.GetDeviceBuffer();
    args.o_ptr              = output_device.GetDeviceBuffer();
    args.seqlen_q           = seqlen_q;
    args.seqlen_k           = seqlen_k;
    args.hdim_q             = kHeadDim;
    args.hdim_v             = kHeadDim;
    args.num_head_q         = kHeads;
    args.nhead_ratio_qk     = 1;
    constexpr float scale_s = 0.08838834764831843f; // 1 / sqrt(128)
#if CK_TILE_FMHA_FWD_FAST_EXP2
    args.scale_s = scale_s * ck_tile::log2e_v<>;
#else
    args.scale_s = scale_s;
#endif
    args.stride_q         = kHeadDim;
    args.stride_k         = kHeadDim;
    args.stride_v         = kHeadDim;
    args.stride_o         = kHeadDim;
    args.nhead_stride_q   = seqlen_q * kHeadDim;
    args.nhead_stride_k   = seqlen_k * kHeadDim;
    args.nhead_stride_v   = seqlen_k * kHeadDim;
    args.nhead_stride_o   = seqlen_q * kHeadDim;
    args.num_head_q_total = kHeads;
    args.batch_stride_q   = kHeads * args.nhead_stride_q;
    args.batch_stride_k   = kHeads * args.nhead_stride_k;
    args.batch_stride_v   = kHeads * args.nhead_stride_v;
    args.batch_stride_o   = kHeads * args.nhead_stride_o;

    const ck_tile::stream_config stream{};
    ck_tile::launch_kernel(stream,
                           ck_tile::make_kernel<Kernel::kBlockPerCu, ck_tile::gfx125_t>(
                               Kernel{},
                               Kernel::GridSize(kBatch, kHeads, seqlen_q, kHeadDim),
                               Kernel::BlockSize(),
                               0,
                               args));
    output_device.FromDevice(output.data());
    return output;
}

template <typename DataType>
void expect_finite_nonzero(const std::vector<DataType>& output)
{
    bool has_nonzero = false;
    for(std::size_t i = 0; i < output.size(); ++i)
    {
        const float value = ck_tile::type_convert<float>(output[i]);
        ASSERT_TRUE(std::isfinite(value)) << "non-finite output at index " << i;
        has_nonzero = has_nonzero || std::abs(value) > 1.0e-3f;
    }
    EXPECT_TRUE(has_nonzero);
}

template <typename DataType>
void expect_close(const std::vector<DataType>& actual,
                  const std::vector<DataType>& expected,
                  const std::string& message)
{
    constexpr double rtol = 1.0e-3;
    constexpr double atol = 1.0e-3;
    EXPECT_TRUE(ck_tile::check_err(actual, expected, message, rtol, atol));
}

TEST(QrTdmProgressiveKLds, M64BufferingModesProduceEquivalentOutput)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 64;
    const auto q = make_input(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input(kBatch * kHeads * kSeqlenK * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input(kBatch * kHeads * kSeqlenK * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(ck_tile::half_t));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto single = run_kernel<SingleBufferM64Problem>(q_device, k_device, v_device, seqlen_q);
    const auto baseline =
        run_kernel<DoubleBufferM64Problem>(q_device, k_device, v_device, seqlen_q);
    const auto progressive =
        run_kernel<ProgressiveM64Problem>(q_device, k_device, v_device, seqlen_q);

    expect_finite_nonzero(single);
    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(baseline, single, "M64 double-buffer baseline differs from single-buffer output");
    expect_close(progressive, single, "M64 progressive output differs from single-buffer output");
}

template <typename DataType>
class QrTdmProgressiveKLdsM128 : public ::testing::Test
{
};

using M128DataTypes = ::testing::Types<ck_tile::half_t, ck_tile::bf16_t>;
TYPED_TEST_SUITE(QrTdmProgressiveKLdsM128, M128DataTypes);

TYPED_TEST(QrTdmProgressiveKLdsM128, SingleKBlockProducesEquivalentOutput)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 128;
    const auto q = make_input<TypeParam>(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input<TypeParam>(kBatch * kHeads * kSeqlenK * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input<TypeParam>(kBatch * kHeads * kSeqlenK * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(TypeParam));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto baseline =
        run_kernel<TestFmhaProblem<128, true, false, TypeParam>>(q_device, k_device, v_device, seqlen_q);
    const auto progressive =
        run_kernel<TestFmhaProblem<128, true, true, TypeParam>>(q_device, k_device, v_device, seqlen_q);

    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(progressive, baseline, "M128 progressive output differs from baseline output");
}

TYPED_TEST(QrTdmProgressiveKLdsM128, MultipleKBlocksProduceEquivalentOutput)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 128;
    constexpr ck_tile::index_t seqlen_k = 128;
    const auto q = make_input<TypeParam>(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input<TypeParam>(kBatch * kHeads * seqlen_k * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input<TypeParam>(kBatch * kHeads * seqlen_k * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(TypeParam));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto baseline = run_kernel<TestFmhaProblem<128, true, false, TypeParam>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k);
    const auto progressive = run_kernel<TestFmhaProblem<128, true, true, TypeParam>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k);

    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(
        progressive, baseline, "M128 multi-block progressive output differs from baseline");
}

} // namespace qr_tdm_progressive_k_lds_test
