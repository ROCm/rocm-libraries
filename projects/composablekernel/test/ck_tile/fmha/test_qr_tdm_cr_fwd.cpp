// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_cr_kernel.hpp"

// Keep native device code under the production build settings, but exercise
// the host adapter's tuning-contract assertions even in Release test builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename Data, int D, bool Group, bool Mask, bool Pad = true, int M = 128>
struct Model
{
    static constexpr int V = D < 128 ? D : 128;
    static constexpr int N = D == 192 ? 64 : M;
    using Shape            = ck_tile::TileFmhaShape<ck_tile::sequence<M, N, 32, V, 32, D>,
                                                    ck_tile::sequence<4, 1, 1>,
                                                    ck_tile::sequence<16, 16, 32>,
                                                    ck_tile::sequence<4, 1, 1>,
                                                    ck_tile::sequence<16, 16, 32>,
                                                    true>;
    using Traits           = ck_tile::TileFmhaTraits<Pad,
                                                     Pad,
                                                     false,
                                                     false,
                                                     false,
                                                     ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                     false,
                                                     true,
                                                     false,
                                                     ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                                     1,
                                                     false,
                                                     false>;
    using Problem          = ck_tile::BlockFmhaPipelineProblem<Data,
                                                               Data,
                                                               Data,
                                                               float,
                                                               float,
                                                               Data,
                                                               uint8_t,
                                                               float,
                                                               Data,
                                                               float,
                                                               Data,
                                                               Shape,
                                                               Group,
                                                               ck_tile::ComposedAttention<0, true>,
                                                               ck_tile::SimplifiedGenericAttentionMask<Mask>,
                                                               false,
                                                               Traits>;
    using Pipeline         = ck_tile::BlockFmhaPipelineQRKSVSTdmCr<Problem>;
    using Epilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, Data, true, true>>;
    using Kernel = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
};

class CrForward : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        int count         = 0;
        const auto status = hipGetDeviceCount(&count);
        if(status == hipErrorNoDevice || (status == hipSuccess && count == 0))
            GTEST_SKIP() << "No HIP device";
        ASSERT_EQ(status, hipSuccess);
        if(ck_tile::get_device_name().compare(0, 7, "gfx1250") != 0)
            GTEST_SKIP() << "CR requires gfx1250";
    }
};

// Inputs are exactly represented multiples of 1/16. A later K tile has larger
// logits, exercising conditional rescaling as well as the first-score path.
template <typename Data, int D, bool Group, bool Mask, int M = 128, bool Pad = true, int B = 2>
void CheckForward(int sq, int sk, int kv_heads, int mask_type = 1, int left = -1)
{
    static_assert(!Group || B == 2, "The packed test metadata describes two groups");
    using Test      = Model<Data, D, Group, Mask, Pad, M>;
    using Kernel    = typename Test::Kernel;
    constexpr int H = 8, V = Test::V;
    std::vector<Data> q(B * sq * H * D), k(B * sk * kv_heads * D), v(B * sk * kv_heads * V);
    std::vector<Data> o(B * sq * H * V, ck_tile::type_convert<Data>(NAN));
    std::vector<float> lse(B * H * sq, NAN);
    for(std::size_t i = 0; i < q.size(); ++i)
        q[i] = ck_tile::type_convert<Data>((int(i % 13) - 6) / 16.f);
    for(std::size_t i = 0; i < q.size(); i += D)
        q[i] = ck_tile::type_convert<Data>(0.5f);
    for(std::size_t i = 0; i < k.size(); ++i)
        k[i] = ck_tile::type_convert<Data>((int(i % 17) - 8) / 16.f);
    for(int b = 0; b < B; ++b)
        for(int row = 0; row < sk; ++row)
            for(int head = 0; head < kv_heads; ++head)
                k[((b * sk + row) * kv_heads + head) * D] =
                    ck_tile::type_convert<Data>(row < sk / 2 ? -64.f : 64.f);
    for(std::size_t i = 0; i < v.size(); ++i)
        v[i] = ck_tile::type_convert<Data>((int(i % 23) - 11) / 16.f);
    if constexpr(!Mask)
        for(int b = 0; b < B; ++b)
            for(int row = 0; row < sk / 2; ++row)
                for(int head = 0; head < kv_heads; ++head)
                    for(int d = 0; d < V; ++d)
                        v[((b * sk + row) * kv_heads + head) * V + d] =
                            ck_tile::type_convert<Data>(32.f);
    ck_tile::DeviceMem dq(q.size() * sizeof(Data)),
        dk(std::max(k.size(), std::size_t{1}) * sizeof(Data));
    ck_tile::DeviceMem dv(std::max(v.size(), std::size_t{1}) * sizeof(Data)),
        dout(o.size() * sizeof(Data));
    ck_tile::DeviceMem dlse(lse.size() * sizeof(float));
    dq.ToDevice(q.data());
    if(sk)
    {
        dk.ToDevice(k.data());
        dv.ToDevice(v.data());
    }
    dout.ToDevice(o.data());
    dlse.ToDevice(lse.data());
    // The second group has fewer Q rows and no keys. Its empty launch tiles must
    // not read K/V, and its live Q rows must produce zero O and -infinity LSE.
    const std::vector<int32_t> starts_q{0, sq, 2 * sq}, starts_k{0, sk, 2 * sk};
    const std::vector<int32_t> lengths_q{sq, std::min(sq, 17)}, lengths_k{sk, 0};
    ck_tile::DeviceMem dsq(starts_q.size() * sizeof(int32_t)),
        dsk(starts_k.size() * sizeof(int32_t));
    ck_tile::DeviceMem dlq(lengths_q.size() * sizeof(int32_t)),
        dlk(lengths_k.size() * sizeof(int32_t));
    dsq.ToDevice(starts_q.data());
    dsk.ToDevice(starts_k.data());
    dlq.ToDevice(lengths_q.data());
    dlk.ToDevice(lengths_k.data());
    fmha_fwd_args a{};
    a.q_ptr          = dq.GetDeviceBuffer();
    a.k_ptr          = dk.GetDeviceBuffer();
    a.v_ptr          = dv.GetDeviceBuffer();
    a.o_ptr          = dout.GetDeviceBuffer();
    a.lse_ptr        = dlse.GetDeviceBuffer();
    a.batch          = B;
    a.nhead_q        = H;
    a.nhead_k        = kv_heads;
    a.hdim_q         = D;
    a.hdim_v         = V;
    a.seqlen_q       = Group ? 2 * sq : sq;
    a.max_seqlen_q   = sq;
    a.seqlen_k       = Group ? 2 * sk : sk;
    a.scale_s        = 0.5f;
    a.stride_q       = H * D;
    a.stride_k       = kv_heads * D;
    a.stride_v       = kv_heads * V;
    a.stride_o       = H * V;
    a.nhead_stride_q = a.nhead_stride_k = D;
    a.nhead_stride_v = a.nhead_stride_o = V;
    a.batch_stride_q                    = sq * H * D;
    a.batch_stride_k                    = sk * kv_heads * D;
    a.batch_stride_v                    = sk * kv_heads * V;
    a.batch_stride_o                    = sq * H * V;
    // Group LSE is [head, total physical Q], batch LSE is [batch, head, Q].
    a.nhead_stride_lse  = Group ? B * sq : sq;
    a.batch_stride_lse  = H * sq;
    a.mask_type         = Mask ? mask_type : 0;
    a.window_size_left  = left;
    a.window_size_right = Mask ? 0 : -1;
    if constexpr(Group)
    {
        a.seqstart_q_ptr = dsq.GetDeviceBuffer();
        a.seqstart_k_ptr = dsk.GetDeviceBuffer();
        a.seqlen_q_ptr   = dlq.GetDeviceBuffer();
        a.seqlen_k_ptr   = dlk.GetDeviceBuffer();
    }
    ASSERT_TRUE(Kernel::IsSupportedInput(a));
    using Strided = typename Kernel::StridedKernel;
    auto native   = Strided::MakeKargs(a);
    ASSERT_TRUE(Strided::IsSupportedArgument(native));
    native.stride_o += 1;
    EXPECT_FALSE(Strided::IsSupportedArgument(native));
    // A direct typed adapter forces CR even in a build containing other pipelines.
    ASSERT_GE(
        (FmhaFwdLaunch<Kernel, ck_tile::gfx125_t>::Run(ck_tile::stream_config{nullptr, false}, a)),
        0);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    dout.FromDevice(o.data());
    dlse.FromDevice(lse.data());
    for(int b = 0; b < B; ++b)
    {
        // Long regressions execute the full B32/H8 grid; sample its first and
        // last batch/head to bound the independent host reference cost.
        if(sq >= 16384 && b != 0 && b != B - 1)
            continue;
        const int qlen = Group ? lengths_q[b] : sq, klen = Group ? lengths_k[b] : sk;
        // Sample both sides of tile boundaries plus first/last rows. Every value
        // dimension is checked; the GPU still computes all rows.
        for(int row : {0,
                       std::min(16, qlen - 1),
                       std::min(63, qlen - 1),
                       std::min(128, qlen - 1),
                       std::max(0, qlen / 2 - 1),
                       qlen / 2,
                       qlen - 1})
            for(int head = 0; head < H; ++head)
            {
                if(sq >= 16384 && head != 0 && head != H - 1)
                    continue;
                const int kh = head / (H / kv_heads);
                std::vector<double> scores(klen, -std::numeric_limits<double>::infinity());
                double maximum = -std::numeric_limits<double>::infinity();
                for(int col = 0; col < klen; ++col)
                {
                    const int center = row + (mask_type == 2 ? klen - qlen : 0);
                    if(Mask && (col > center || (left >= 0 && col < center - left)))
                        continue;
                    double dot = 0;
                    for(int d = 0; d < D; ++d)
                        dot +=
                            ck_tile::type_convert<float>(q[((b * sq + row) * H + head) * D + d]) *
                            ck_tile::type_convert<float>(
                                k[((b * sk + col) * kv_heads + kh) * D + d]);
                    scores[col] = dot * a.scale_s;
                    maximum     = std::max(maximum, scores[col]);
                }
                if constexpr(!Mask)
                    if(klen > 2 * Test::N)
                    {
                        // Exceed both the first-score +8 initial base and the
                        // subsequent +8 update threshold (scores here use exp).
                        const double first_max =
                            *std::max_element(scores.begin(), scores.begin() + Test::N);
                        EXPECT_GT(maximum - first_max, 16 * std::log(2.0));
                    }
                double sum = 0;
                std::vector<double> expected(V, 0);
                if(std::isfinite(maximum))
                    for(int col = 0; col < klen; ++col)
                    {
                        const double weight = std::exp(scores[col] - maximum);
                        sum += weight;
                        for(int d = 0; d < V; ++d)
                            expected[d] +=
                                weight * ck_tile::type_convert<float>(
                                             v[((b * sk + col) * kv_heads + kh) * V + d]);
                    }
                for(int d = 0; d < V; ++d)
                {
                    const double actual =
                        ck_tile::type_convert<float>(o[((b * sq + row) * H + head) * V + d]);
                    // P and O are rounded to the input dtype (7 BF16 fraction bits).
                    // The dominant late V values have magnitude <1, allowing 1/64
                    // error. Dense early V=32 amplifies a missing output rescale;
                    // its correct contribution after the logit jump is negligible.
                    EXPECT_NEAR(actual, sum ? expected[d] / sum : 0, 1.0 / 64)
                        << "b=" << b << " h=" << head << " q=" << row << " d=" << d;
                }
                const float actual_lse =
                    lse[Group ? head * B * sq + b * sq + row : (b * H + head) * sq + row];
                if(sum)
                    EXPECT_NEAR(actual_lse, maximum + std::log(sum), 0.002);
                else
                    EXPECT_EQ(actual_lse, -std::numeric_limits<float>::infinity());
            }
    }
}

#if defined(CK_TILE_TEST_CR_LONG)
TEST_F(CrForward, Static16384Dense)
{
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false, 32>(16384, 16384, 8);
}
TEST_F(CrForward, Static16384Causal)
{
    CheckForward<ck_tile::bf16_t, 128, false, true, 128, false, 32>(16384, 16384, 8);
}
TEST_F(CrForward, Static32768Dense)
{
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false, 32>(32768, 32768, 8);
}
TEST_F(CrForward, Static32768Causal)
{
    CheckForward<ck_tile::bf16_t, 128, false, true, 128, false, 32>(32768, 32768, 8);
}
#elif defined(CK_TILE_TEST_CR_D64)
TEST_F(CrForward, GqaAndSlidingWindow)
{
    CheckForward<ck_tile::half_t, 64, false, true, 64>(129, 257, 2, 2, 31);
    CheckForward<ck_tile::bf16_t, 64, false, false, 128>(257, 515, 2);
}
TEST_F(CrForward, GroupAndEmptyKeys)
{
    CheckForward<ck_tile::half_t, 64, true, false, 64>(129, 257, 2);
}
#else
TEST_F(CrForward, CompactTailAndRescaling)
{
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false, 32>(512, 512, 8);
    CheckForward<ck_tile::bf16_t, 128, false, true, 128, false, 32>(512, 512, 8);
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false>(512, 512, 8);
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false>(1024, 2048, 8);
    CheckForward<ck_tile::bf16_t, 128, false, false>(639, 639, 8);
    CheckForward<ck_tile::bf16_t, 128, false, true>(257, 515, 8);
}
TEST_F(CrForward, GqaAndSlidingWindow)
{
    CheckForward<ck_tile::bf16_t, 128, false, true, 64>(257, 129, 1, 2, 15);
    CheckForward<ck_tile::half_t, 128, false, false>(257, 515, 2);
    CheckForward<ck_tile::half_t, 192, false, true>(129, 257, 2, 2, 31);
}
TEST_F(CrForward, GroupAndEmptyKeys)
{
    CheckForward<ck_tile::bf16_t, 128, true, true, 64>(257, 129, 1, 2, 31);
    CheckForward<ck_tile::bf16_t, 192, true, true, 64>(129, 257, 2, 2, 31);
    CheckForward<ck_tile::bf16_t, 128, false, false>(129, 0, 8);
}

TEST(CrArgumentContract, FixedTuningAndCompactLayout)
{
    using Kernel = Model<ck_tile::bf16_t, 128, false, false, false>::Kernel;
    using Static = Kernel::WithTuning<ck_tile::FmhaTdmCrTuning<512>>;
    using Tiles  = Kernel::WithTuning<ck_tile::FmhaTdmCrTileTuning<4>>;
    alignas(16) float storage[4]{};
    Kernel::Kargs a{storage, storage, storage, storage, storage, 32, 8, 512, 512, 0.5f};
    EXPECT_TRUE(Static::IsSupportedArgument(a));
    EXPECT_TRUE(Tiles::IsSupportedArgument(a));
    a.batch = 1;
    EXPECT_FALSE(Static::IsSupportedArgument(a));
    EXPECT_TRUE(Tiles::IsSupportedArgument(a));
    a.heads = 4;
    EXPECT_FALSE(Tiles::IsSupportedArgument(a));
    a.heads = std::numeric_limits<int>::max();
    EXPECT_FALSE(Kernel::IsSupportedArgument(a));
    a.heads = 8;
    a.sk    = 513;
    EXPECT_FALSE(Kernel::IsSupportedArgument(a));
    using TailKernel = Model<ck_tile::bf16_t, 128, false, false>::Kernel;
    using Tail       = TailKernel::WithTuning<ck_tile::FmhaTdmCrTileTuning<5, 0, true>>;
    a.sq = a.sk = 639;
    EXPECT_TRUE(Tail::IsSupportedArgument(a));
    a.sq = 638;
    EXPECT_FALSE(Tail::IsSupportedArgument(a));
}

#endif

} // namespace
