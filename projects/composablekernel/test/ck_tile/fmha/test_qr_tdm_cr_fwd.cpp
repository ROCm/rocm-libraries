// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp"

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

template <typename Data>
__global__ void CompareRepeatedOutput(
    const Data* reference, const Data* output, std::size_t size, int iteration, int* first_mismatch)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size &&
       ck_tile::bit_cast<uint16_t>(reference[i]) != ck_tile::bit_cast<uint16_t>(output[i]))
        atomicMin(first_mismatch, iteration);
}

template <typename Data,
          int D,
          bool Group,
          bool Mask,
          bool Pad      = true,
          int M         = 128,
          bool StoreLSE = true>
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
                                                     StoreLSE,
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

enum class InputPattern
{
    Rescaling,
    Random
};

struct ForwardTestOptions
{
    InputPattern input_pattern = InputPattern::Rescaling;
    bool output_head_major     = false;
    int output_padding         = 0;
    int output_offset          = 0;
    int input_padding          = 0;
    int input_offset           = 0;
    int repeats                = 1;
    float scale_s              = std::numeric_limits<float>::quiet_NaN();
    bool fixed_tolerance       = false;
};

// Rescaling inputs use exact multiples of 1/16 and a later K tile with larger
// logits. Random inputs expose sparse stale LDS values. Repetition preserves
// the selected input pattern and scale.
template <typename Data,
          int D,
          bool Group,
          bool Mask,
          int M         = 128,
          bool Pad      = true,
          int B         = 2,
          bool StoreLSE = true>
void CheckForward(
    int sq, int sk, int kv_heads, int mask_type = 1, int left = -1, ForwardTestOptions options = {})
{
    static_assert(!Group || B == 2, "The packed test metadata describes two groups");
    using Test      = Model<Data, D, Group, Mask, Pad, M, StoreLSE>;
    using Kernel    = typename Test::Kernel;
    constexpr int H = 8, V = Test::V;
    std::vector<Data> q(B * sq * H * D), k(B * sk * kv_heads * D), v(B * sk * kv_heads * V);
    const bool output_head_major  = options.output_head_major;
    const int output_padding      = options.output_padding;
    const int output_stride       = (output_head_major ? V : H * V) + output_padding;
    const int output_head_stride  = output_head_major ? (Group ? B * sq : sq) * output_stride : V;
    const int output_batch_stride = sq * output_stride * (output_head_major ? H : 1);
    const int output_offset       = options.output_offset;
    std::vector<Data> o(B * output_batch_stride + 2 * output_offset,
                        ck_tile::type_convert<Data>(NAN));
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
    if(options.input_pattern == InputPattern::Random)
    {
        // Repeated patterns can hide sparse stale LDS values in the output.
        std::mt19937 engine{42};
        std::normal_distribution<float> normal{0.f, 1.f};
        for(auto* input : {&q, &k, &v})
            for(auto& value : *input)
                value = ck_tile::type_convert<Data>(normal(engine));
    }
    const int input_padding = options.input_padding, input_offset = options.input_offset;
    const int q_stride = H * D + input_padding, k_stride = kv_heads * D + input_padding,
              v_stride = kv_heads * V + input_padding;
    ck_tile::DeviceMem dq((B * sq * q_stride + 2 * input_offset) * sizeof(Data)),
        dk(std::max(B * sk * k_stride + 2 * input_offset, 1) * sizeof(Data));
    ck_tile::DeviceMem dv(std::max(B * sk * v_stride + 2 * input_offset, 1) * sizeof(Data)),
        dout(o.size() * sizeof(Data));
    ck_tile::DeviceMem dlse(lse.size() * sizeof(float));
    auto upload = [&](ck_tile::DeviceMem& device, const auto& data, int rows, int width) {
        if(input_padding == 0 && input_offset == 0)
            device.ToDevice(data.data());
        else
        {
            std::vector<Data> padded(rows * (width + input_padding) + 2 * input_offset,
                                     ck_tile::type_convert<Data>(NAN));
            for(int row = 0; row < rows; ++row)
                std::copy_n(data.begin() + row * width,
                            width,
                            padded.begin() + input_offset + row * (width + input_padding));
            device.ToDevice(padded.data());
        }
    };
    upload(dq, q, B * sq, H * D);
    if(sk)
    {
        upload(dk, k, B * sk, kv_heads * D);
        upload(dv, v, B * sk, kv_heads * V);
    }
    ASSERT_EQ(hipMemset(dout.GetDeviceBuffer(), 0xff, o.size() * sizeof(Data)), hipSuccess);
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
    a.q_ptr        = static_cast<Data*>(dq.GetDeviceBuffer()) + input_offset;
    a.k_ptr        = static_cast<Data*>(dk.GetDeviceBuffer()) + input_offset;
    a.v_ptr        = static_cast<Data*>(dv.GetDeviceBuffer()) + input_offset;
    a.o_ptr        = static_cast<Data*>(dout.GetDeviceBuffer()) + output_offset;
    a.lse_ptr      = StoreLSE ? dlse.GetDeviceBuffer() : nullptr;
    a.batch        = B;
    a.nhead_q      = H;
    a.nhead_k      = kv_heads;
    a.hdim_q       = D;
    a.hdim_v       = V;
    a.seqlen_q     = Group ? 2 * sq : sq;
    a.max_seqlen_q = sq;
    a.seqlen_k     = Group ? 2 * sk : sk;
    a.scale_s = options.input_pattern == InputPattern::Random ? 1.f / std::sqrt(float(D)) : 0.5f;
    if(std::isfinite(options.scale_s))
        a.scale_s = options.scale_s;
    a.stride_q       = q_stride;
    a.stride_k       = k_stride;
    a.stride_v       = v_stride;
    a.stride_o       = output_stride;
    a.nhead_stride_q = a.nhead_stride_k = D;
    a.nhead_stride_v                    = V;
    a.nhead_stride_o                    = output_head_stride;
    a.batch_stride_q                    = sq * q_stride;
    a.batch_stride_k                    = sk * k_stride;
    a.batch_stride_v                    = sk * v_stride;
    a.batch_stride_o                    = output_batch_stride;
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
    using Strided          = typename Kernel::StridedKernel;
    auto native            = Strided::MakeKargs(a);
    const bool vector_path = a.scale_s > 0 && ck_tile::detail::fmha_fwd_cr_vector_output(a);
    EXPECT_EQ(Strided::IsSupportedArgument(native), vector_path);
    native.stride_o += 1;
    EXPECT_FALSE(Strided::IsSupportedArgument(native));
    // A direct typed adapter forces CR even in a build containing other pipelines.
    ASSERT_GE(
        (FmhaFwdLaunch<Kernel, ck_tile::gfx125_t>::Run(ck_tile::stream_config{nullptr, false}, a)),
        0);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    dout.FromDevice(o.data());
    dlse.FromDevice(lse.data());
    // Sparse output races can miss the sampled reference rows below. Compare
    // every output bit across repeated launches of the same attention problem.
    if(options.repeats > 1)
    {
        // Keep the comparisons on device so host copies do not serialize each launch.
        ck_tile::DeviceMem reference(o.size() * sizeof(Data)), mismatch(sizeof(int));
        reference.ToDevice(o.data());
        int first_mismatch = std::numeric_limits<int>::max();
        mismatch.ToDevice(&first_mismatch);
        for(int iteration = 1; iteration < options.repeats; ++iteration)
        {
            ASSERT_EQ(
                hipMemsetAsync(dout.GetDeviceBuffer(), 0xff, o.size() * sizeof(Data), nullptr),
                hipSuccess);
            ASSERT_GE((FmhaFwdLaunch<Kernel, ck_tile::gfx125_t>::Run(
                          ck_tile::stream_config{nullptr, false}, a)),
                      0);
            hipLaunchKernelGGL((CompareRepeatedOutput<Data>),
                               dim3((o.size() + 255) / 256),
                               dim3(256),
                               0,
                               nullptr,
                               static_cast<const Data*>(reference.GetDeviceBuffer()),
                               static_cast<const Data*>(dout.GetDeviceBuffer()),
                               o.size(),
                               iteration,
                               static_cast<int*>(mismatch.GetDeviceBuffer()));
            ASSERT_EQ(hipGetLastError(), hipSuccess);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        mismatch.FromDevice(&first_mismatch);
        ASSERT_EQ(first_mismatch, std::numeric_limits<int>::max())
            << "Output changed on iteration " << first_mismatch;
    }
    // Padding between output rows must not be touched by TDM iteration.
    if(output_padding)
        for(int row = 0; row < B * output_batch_stride / output_stride; ++row)
            for(int col = output_stride - output_padding; col < output_stride; ++col)
                ASSERT_TRUE(std::isnan(
                    ck_tile::type_convert<float>(o[output_offset + row * output_stride + col])));
    for(int i = 0; i < output_offset; ++i)
    {
        EXPECT_TRUE(std::isnan(ck_tile::type_convert<float>(o[i])));
        EXPECT_TRUE(std::isnan(ck_tile::type_convert<float>(o[o.size() - 1 - i])));
    }
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
                    if(options.input_pattern == InputPattern::Rescaling && klen > 2 * Test::N)
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
                    const double actual = ck_tile::type_convert<float>(
                        o[output_offset +
                          (Group ? b * sq * output_stride : b * output_batch_stride) +
                          row * output_stride + head * output_head_stride + d]);
                    // P and O are rounded to the input dtype (7 BF16 fraction bits).
                    // The dominant late V values have magnitude <1, allowing 1/64
                    // error. Dense early V=32 amplifies a missing output rescale;
                    // its correct contribution after the logit jump is negligible.
                    const double expected_value = sum ? expected[d] / sum : 0;
                    const double tolerance =
                        options.fixed_tolerance
                            ? (std::is_same_v<Data, ck_tile::half_t> ? 0.001 : 0.01) *
                                  (1 + std::abs(expected_value))
                            : 1.0 / 64;
                    EXPECT_NEAR(actual, expected_value, tolerance)
                        << "b=" << b << " h=" << head << " q=" << row << " d=" << d;
                }
                if constexpr(StoreLSE)
                {
                    const float actual_lse =
                        lse[Group ? head * B * sq + b * sq + row : (b * H + head) * sq + row];
                    if(sum)
                        EXPECT_NEAR(actual_lse, maximum + std::log(sum), 0.002);
                    else
                        EXPECT_EQ(actual_lse, -std::numeric_limits<float>::infinity());
                }
            }
    }
}

template <typename Data, int D>
void CheckGeneralViews()
{
    ForwardTestOptions options;
    options.input_pattern   = InputPattern::Random;
    options.repeats         = 8;
    options.fixed_tolerance = true;
    // Zero and negative scales must preserve mask sentinels and handle tails.
    for(float scale : {0.f, -0.25f})
    {
        options.scale_s = scale;
        CheckForward<Data, D, false, true, 64>(257, 130, 2, 2, 31, options);
        // The empty/valid boundary cuts through a tile. M128 also exercises
        // both query fragments and the D192 rectangular geometry.
        CheckForward<Data, D, false, true, 128>(257, 130, 2, 2, 31, options);
        CheckForward<Data, D, true, false, 64>(129, 257, 2, 0, -1, options);
    }
    options.scale_s           = 0.125f;
    options.output_offset     = 1;
    options.output_padding    = 1;
    options.output_head_major = true;
    CheckForward<Data, D, false, true, 64>(257, 129, 2, 2, 31, options);
    CheckForward<Data, D, false, true, 128>(257, 130, 2, 2, 31, options);
    options.output_offset = options.output_padding = 0;
    options.input_offset = options.input_padding = 1;
    CheckForward<Data, D, false, true, 64>(257, 129, 2, 2, 31, options);
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
TEST_F(CrForward, GeneralScaleAndOutputViews)
{
    CheckGeneralViews<ck_tile::half_t, 64>();
    CheckGeneralViews<ck_tile::bf16_t, 64>();
}

TEST_F(CrForward, RepeatedCausalOutputWithoutLse)
{
    ForwardTestOptions options;
    options.input_pattern = InputPattern::Random;
    options.repeats       = 16384;
    CheckForward<ck_tile::bf16_t, 64, false, true, 64, true, 2, false>(
        8192, 8192, 2, 1, -1, options);
}

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
TEST_F(CrForward, GeneralScaleAndOutputViews)
{
    CheckGeneralViews<ck_tile::half_t, 128>();
    CheckGeneralViews<ck_tile::bf16_t, 128>();
    CheckGeneralViews<ck_tile::half_t, 192>();
    CheckGeneralViews<ck_tile::bf16_t, 192>();
}

// Exercise the output descriptor independently of attention: incorrect iteration
// units, row counts or LDS pitches must corrupt either the values or the canaries.
template <int Width, int Pitch>
__global__ void StoreRows(uint16_t* output, int rows, int stride)
{
    __shared__ __align__(16) uint16_t lds[64 * Pitch];
    const int wave = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    for(int i = lane; i < 16 * Pitch; i += 32)
    {
        const int row          = wave * 16 + i / Pitch;
        const int col          = i % Pitch;
        lds[row * Pitch + col] = col < Width ? uint16_t(row * Width + col + 1) : 0xbeef;
    }
    if(wave * 16 < rows)
        ck_tile::detail::fmha_cr::store_strided<16, Width, Pitch>(lds + wave * 16 * Pitch,
                                                                  output +
                                                                      int64_t(wave * 16) * stride,
                                                                  rows - wave * 16,
                                                                  uint32_t(stride));
    ck_tile::detail::fmha_cr::wait();
}

template <int Width, int Pitch>
void CheckStoreRows()
{
    constexpr int Guard = 32;
    for(int stride : {Width, Width + 32, 65544})
    {
        std::vector<uint16_t> output(64 * stride + 2 * Guard);
        ck_tile::DeviceMem device(output.size() * sizeof(uint16_t));
        // Include empty fragments and tails at every wave boundary. The wide
        // stride also detects accidental truncation to a 16-bit TDM increment.
        for(int rows : {0, 1, 15, 16, 17, 31, 32, 47, 48, 63, 64})
        {
            SCOPED_TRACE(::testing::Message() << "width=" << Width << " pitch=" << Pitch
                                              << " stride=" << stride << " rows=" << rows);
            std::fill(output.begin(), output.end(), uint16_t{0xcdcd});
            device.ToDevice(output.data());
            hipLaunchKernelGGL((StoreRows<Width, Pitch>),
                               dim3(1),
                               dim3(128),
                               0,
                               nullptr,
                               static_cast<uint16_t*>(device.GetDeviceBuffer()) + Guard,
                               rows,
                               stride);
            ASSERT_EQ(hipGetLastError(), hipSuccess);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            device.FromDevice(output.data());
            for(std::size_t i = 0; i < output.size(); ++i)
            {
                uint16_t expected = 0xcdcd;
                if(i >= Guard && i < output.size() - Guard)
                {
                    const auto row = (i - Guard) / stride;
                    const auto col = (i - Guard) % stride;
                    if(row < std::size_t(rows) && col < Width)
                        expected = uint16_t(row * Width + col + 1);
                }
                ASSERT_EQ(output[i], expected) << "element=" << i;
            }
        }
    }
}

TEST_F(CrForward, TdmOutputPitchAndTail)
{
    CheckStoreRows<64, 72>();
    CheckStoreRows<128, 136>();
    CheckStoreRows<128, 216>(); // D192 Q/K storage reused for D128 output.
}

TEST_F(CrForward, CompactTailAndRescaling)
{
    ForwardTestOptions repeated;
    repeated.repeats = 2;
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false, 32>(512, 512, 8, 1, -1, repeated);
    CheckForward<ck_tile::bf16_t, 128, false, true, 128, false, 32>(512, 512, 8);
    // Exercise causal LSE peeling in the compact 128-row and 64-row paths.
    CheckForward<ck_tile::bf16_t, 128, false, true, 128, false, 64>(512, 512, 8);
    CheckForward<ck_tile::bf16_t, 128, false, true, 128, false, 4>(1024, 2048, 8);
    CheckForward<ck_tile::bf16_t, 128, false, true, 64, false>(512, 512, 8);
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false>(512, 512, 8);
    CheckForward<ck_tile::bf16_t, 128, false, false, 128, false>(1024, 2048, 8);
    CheckForward<ck_tile::bf16_t, 128, false, false>(639, 639, 8);
    CheckForward<ck_tile::bf16_t, 128, false, true>(257, 515, 8);
    // Samples 199/200 straddle the causal/padding boundary within the same Q tile.
    CheckForward<ck_tile::bf16_t, 128, false, true>(400, 200, 8);
}
TEST_F(CrForward, GqaAndSlidingWindow)
{
    CheckForward<ck_tile::bf16_t, 128, false, true, 64>(257, 129, 1, 2, 15);
    CheckForward<ck_tile::half_t, 128, false, false>(257, 515, 2);
    ForwardTestOptions head_major;
    head_major.output_head_major = true;
    head_major.output_padding    = 8;
    head_major.repeats           = 2;
    CheckForward<ck_tile::half_t, 128, false, true>(257, 515, 2, 1, -1, head_major);
    ForwardTestOptions padded;
    padded.output_padding = 16;
    CheckForward<ck_tile::half_t, 192, false, true>(129, 257, 2, 2, 31, padded);
}
TEST_F(CrForward, GroupAndEmptyKeys)
{
    CheckForward<ck_tile::bf16_t, 128, true, true, 64>(257, 129, 1, 2, 31);
    CheckForward<ck_tile::bf16_t, 128, true, true>(257, 129, 1, 2, 31);
    CheckForward<ck_tile::bf16_t, 192, true, true, 64>(129, 257, 2, 2, 31);
    CheckForward<ck_tile::bf16_t, 128, false, false>(129, 0, 8);
}

struct UnsupportedTuningSelection
{
    template <typename Args, typename Visitor>
    static float VisitKernels(const Args&, Visitor&& visitor)
    {
        using Kernel = Model<ck_tile::bf16_t, 128, false, false, false>::Kernel;
        return visitor(typename Kernel::WithTuning<ck_tile::FmhaTdmCrTuning<512>>{});
    }
};

TEST(CrArgumentContract, FixedTuningAndCompactLayout)
{
    // Metadata may contain both physical padding and logical lengths. An
    // explicit zero length must override a nonempty physical or cumulative span.
    constexpr int32_t starts[]     = {0, 128, 384};
    constexpr int32_t cumulative[] = {0, 96, 160};
    constexpr int32_t lengths[]    = {80, 0};
    using ck_tile::detail::fmha_fwd_sequence_length;
    static_assert(fmha_fwd_sequence_length(true, 1, 512, lengths, cumulative, starts) == 0);
    static_assert(fmha_fwd_sequence_length(true, 1, 512, nullptr, cumulative, starts) == 64);
    static_assert(fmha_fwd_sequence_length(true, 1, 512, nullptr, nullptr, starts) == 256);
    static_assert(fmha_fwd_sequence_length(false, 1, 512, nullptr, cumulative, nullptr) == 64);
    static_assert(fmha_fwd_sequence_length(false, 1, 512, nullptr, nullptr, nullptr) == 512);

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

    // Even if a selector chooses an incompatible tuning, the common launcher
    // must reject it in Release builds before touching these small host buffers.
    fmha_fwd_args input{};
    input.q_ptr = input.k_ptr = input.v_ptr = storage;
    input.o_ptr = input.lse_ptr = storage;
    input.batch                 = 1; // The fixed tuning requires B32.
    input.nhead_q               = 8;
    input.seqlen_q = input.seqlen_k = 512;
    input.scale_s                   = 0.5f;
    EXPECT_EQ((FmhaFwdLaunch<UnsupportedTuningSelection, ck_tile::gfx125_t>::Run(
                  ck_tile::stream_config{nullptr, false}, input)),
              -1.0f);
}

#endif

} // namespace
