// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp"

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
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
          bool Pad                             = true,
          int M                                = 128,
          bool StoreLSE                        = true,
          bool HeadPad                         = false,
          ck_tile::BlockAttentionBiasEnum Bias = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool SoftCap                         = false,
          bool Sink                            = false,
          bool Skip                            = false,
          int KeyTile =
              D == 64 || D == 128 ||
                      (HeadPad &&
                       (D == 32 || (D == 96 && Bias != ck_tile::BlockAttentionBiasEnum::ALIBI)))
                  ? M
                  : 64>
struct Model
{
    static constexpr int V = D == 192 ? 128 : D;
    static constexpr int N = KeyTile;
    using Shape            = ck_tile::TileFmhaShape<ck_tile::sequence<M, N, 32, V, 32, D>,
                                                    ck_tile::sequence<4, 1, 1>,
                                                    ck_tile::sequence<16, 16, 32>,
                                                    ck_tile::sequence<4, 1, 1>,
                                                    ck_tile::sequence<16, 16, 32>,
                                                    true>;
    using Traits           = ck_tile::TileFmhaTraits<Pad,
                                                     Pad,
                                                     HeadPad,
                                                     HeadPad,
                                                     SoftCap,
                                                     Bias,
                                                     false,
                                                     StoreLSE,
                                                     false,
                                                     ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                                     1,
                                                     Skip,
                                                     Sink>;
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
    using Kernel        = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
    using CompactKernel = ck_tile::FmhaFwdKernel<
        ck_tile::FmhaTdmCrCompactPipelineVariant<Pipeline, ck_tile::FmhaTdmCrTuning<>>,
        Epilogue>;
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
    MaskedRescaling,
    Random
};

struct ForwardTestOptions
{
    InputPattern input_pattern    = InputPattern::Rescaling;
    bool output_head_major        = false;
    int output_padding            = 0;
    int output_offset             = 0;
    int input_padding             = 0;
    int input_offset              = 0;
    int repeats                   = 1;
    float scale_s                 = std::numeric_limits<float>::quiet_NaN();
    bool fixed_tolerance          = false;
    bool check_all_rows           = false;
    bool check_output_writes      = false;
    int hdim_q                    = 0;
    int hdim_v                    = 0;
    bool second_group_has_keys    = false;
    bool shared_bias              = false;
    bool masked_bias              = false;
    bool first_column_masked_bias = false;
    bool fragment_masked_bias     = false;
    bool wide_bias_stride         = false;
    float alibi_slope_scale       = 1.f;
    float logits_soft_cap         = 8.f;
    int sink_size                 = 0;
    int min_seqlen_q              = 0;
    std::vector<float> virtual_sink;
    std::optional<bool> causal_sink_tuning;
};

// Rescaling inputs use exact multiples of 1/16 and a later K tile with larger
// logits. Random inputs expose sparse stale LDS values. Repetition preserves
// the selected input pattern and scale.
template <typename Data,
          int D,
          bool Group,
          bool Mask,
          int M                                = 128,
          bool Pad                             = true,
          int B                                = 2,
          bool StoreLSE                        = true,
          bool HeadPad                         = false,
          ck_tile::BlockAttentionBiasEnum Bias = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool SoftCap                         = false,
          bool Sink                            = false,
          bool Skip                            = false,
          int KeyTile                          = D == 64 || D == 128 ||
                                (HeadPad &&
                                 (D == 32 ||
                                  (D == 96 && Bias != ck_tile::BlockAttentionBiasEnum::ALIBI)))
                                                     ? M
                                                     : 64>
void CheckForward(
    int sq, int sk, int kv_heads, int mask_type = 1, int left = -1, ForwardTestOptions options = {})
{
    static_assert(!Group || B == 2, "The packed test metadata describes two groups");
    using Test =
        Model<Data, D, Group, Mask, Pad, M, StoreLSE, HeadPad, Bias, SoftCap, Sink, Skip, KeyTile>;
    using Kernel    = typename Test::Kernel;
    constexpr int H = 8;
    const int QDim  = options.hdim_q ? options.hdim_q : D;
    const int V     = options.hdim_v ? options.hdim_v : Test::V;
    std::vector<Data> q(B * sq * H * QDim), k(B * sk * kv_heads * QDim), v(B * sk * kv_heads * V);
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
    for(std::size_t i = 0; i < q.size(); i += QDim)
        q[i] = ck_tile::type_convert<Data>(0.5f);
    for(std::size_t i = 0; i < k.size(); ++i)
        k[i] = ck_tile::type_convert<Data>((int(i % 17) - 8) / 16.f);
    for(int b = 0; b < B; ++b)
        for(int row = 0; row < sk; ++row)
            for(int head = 0; head < kv_heads; ++head)
                k[((b * sk + row) * kv_heads + head) * QDim] =
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
    else if(options.input_pattern == InputPattern::MaskedRescaling)
    {
        std::fill(q.begin(), q.end(), ck_tile::type_convert<Data>(0.f));
        std::fill(k.begin(), k.end(), ck_tile::type_convert<Data>(0.f));
        for(std::size_t i = 0; i < q.size(); i += QDim)
            q[i] = ck_tile::type_convert<Data>(1.f);
        for(int b = 0; b < B; ++b)
            for(int row = 128; row < sk; ++row)
                for(int head = 0; head < kv_heads; ++head)
                    k[((b * sk + row) * kv_heads + head) * QDim] =
                        ck_tile::type_convert<Data>(32.f);
    }
    const int input_padding = options.input_padding, input_offset = options.input_offset;
    const int q_stride = H * QDim + input_padding, k_stride = kv_heads * QDim + input_padding,
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
    upload(dq, q, B * sq, H * QDim);
    if(sk)
    {
        upload(dk, k, B * sk, kv_heads * QDim);
        upload(dv, v, B * sk, kv_heads * V);
    }
    // A finite fill detects missing writes when the expected output is NaN.
    // Compare guards by bits so a NaN store cannot hide an out-of-bounds write.
    const int output_fill    = options.check_output_writes ? 0x3f : 0xff;
    const uint16_t o_guard   = output_fill * 0x0101u;
    const uint32_t lse_guard = output_fill * 0x01010101u;
    ASSERT_EQ(hipMemset(dout.GetDeviceBuffer(), output_fill, o.size() * sizeof(Data)), hipSuccess);
    ASSERT_EQ(hipMemset(dlse.GetDeviceBuffer(), output_fill, lse.size() * sizeof(float)),
              hipSuccess);
    // Keep physical padding after the second group's live rows. With keys it
    // exercises nonzero Q/K/V offsets and both M128 query fragments; without
    // keys its live rows must produce zero O and -infinity LSE.
    const std::vector<int32_t> starts_q{0, sq, 2 * sq}, starts_k{0, sk, 2 * sk};
    const std::vector<int32_t> lengths_q{sq,
                                         std::min(sq, options.second_group_has_keys ? 145 : 17)};
    const std::vector<int32_t> lengths_k{sk, options.second_group_has_keys ? std::min(sk, 193) : 0};
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
    a.hdim_q       = QDim;
    a.hdim_v       = V;
    a.seqlen_q     = Group ? 2 * sq : sq;
    a.max_seqlen_q = sq;
    a.seqlen_k     = Group ? 2 * sk : sk;
    a.scale_s = options.input_pattern == InputPattern::Random ? 1.f / std::sqrt(float(QDim)) : 0.5f;
    if(std::isfinite(options.scale_s))
        a.scale_s = options.scale_s;
    a.stride_q       = q_stride;
    a.stride_k       = k_stride;
    a.stride_v       = v_stride;
    a.stride_o       = output_stride;
    a.nhead_stride_q = a.nhead_stride_k = QDim;
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
    std::vector<Data> bias;
    std::vector<float> slopes;
    ck_tile::DeviceMem dbias([&]() -> std::size_t {
        if constexpr(Bias == ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            a.stride_bias       = sk + 3;
            a.nhead_stride_bias = options.shared_bias ? 0 : (Group ? B : 1) * sq * a.stride_bias;
            a.batch_stride_bias = options.shared_bias ? 0 : H * sq * a.stride_bias;
            bias.resize(B * H * sq * a.stride_bias);
            for(std::size_t i = 0; i < bias.size(); ++i)
            {
                const int row = (i / a.stride_bias) % sq, col = i % a.stride_bias;
                const float value =
                    col >= sk                                                           ? NAN
                    : options.first_column_masked_bias && row % 2 == 0 && col == 0      ? -INFINITY
                    : options.fragment_masked_bias && (col % 16 == 5 || col % 16 == 13) ? -INFINITY
                    : options.masked_bias && (row % 19 == 0 || col < 64)                ? -INFINITY
                    : options.masked_bias && row % 2 == 0 && col >= 128                 ? 32.f
                                                                        : (int(i % 37) - 18) / 8.f;
                bias[i] = ck_tile::type_convert<Data>(value);
            }
            if(options.wide_bias_stride)
            {
                // One shared row can exercise a wide native stride without
                // allocating the unused gap between rows.
                EXPECT_EQ(sq, 1);
                EXPECT_FALSE(Group);
                EXPECT_TRUE(options.shared_bias);
            }
            return bias.size() * sizeof(Data);
        }
        else if constexpr(Bias == ck_tile::BlockAttentionBiasEnum::ALIBI)
        {
            a.stride_bias = options.shared_bias ? 0 : H;
            slopes.resize(B * H);
            for(int i = 0; i < B * H; ++i)
                slopes[i] = options.alibi_slope_scale * (i + 1) / 128.f;
            return slopes.size() * sizeof(float);
        }
        else
            return 1;
    }());
    if constexpr(Bias != ck_tile::BlockAttentionBiasEnum::NO_BIAS)
    {
        a.bias_ptr = dbias.GetDeviceBuffer();
        if constexpr(Bias == ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            dbias.ToDevice(bias.data());
        else
            dbias.ToDevice(slopes.data());
    }
    if constexpr(SoftCap)
        a.logits_soft_cap = options.logits_soft_cap;
    a.sink_size    = options.sink_size;
    a.min_seqlen_q = options.min_seqlen_q;
    ck_tile::DeviceMem dsink(H * sizeof(float));
    if(!options.virtual_sink.empty())
    {
        ASSERT_EQ(options.virtual_sink.size(), std::size_t{H});
        dsink.ToDevice(options.virtual_sink.data());
        a.sink_ptr = dsink.GetDeviceBuffer();
    }
    EXPECT_EQ(Kernel::IsSupportedInput(a), options.virtual_sink.empty());
    auto validate_native = [&](auto selected) {
        using Selected         = decltype(selected);
        auto native            = Selected::MakeKargs(a);
        const bool vector_path = a.scale_s > 0 && ck_tile::detail::fmha_fwd_cr_vector_output(a);
        EXPECT_EQ(Selected::IsSupportedArgument(native),
                  HeadPad || Selected::kGeneral || vector_path);
        if constexpr(!HeadPad && !Selected::kGeneral &&
                     !Selected::LaunchPolicy::kUsesCompactArguments)
        {
            native.stride_o += 1;
            EXPECT_FALSE(Selected::IsSupportedArgument(native));
        }
    };
    using Strided = typename Kernel::StridedKernel;
    if(options.virtual_sink.empty())
        validate_native(Strided{});
    else
        validate_native(typename Strided::template WithTuning<ck_tile::FmhaTdmCrSinkTuning>{});
    if(options.causal_sink_tuning.has_value())
    {
        const auto selection_status = Kernel::VisitKernels(a, [&](auto selected) {
            using Selected = decltype(selected);
            using Tuning   = typename Selected::FmhaPipeline::Tuning;
            EXPECT_EQ((std::is_same_v<Tuning, ck_tile::FmhaTdmCrCausalSinkTuning>),
                      *options.causal_sink_tuning);
            EXPECT_TRUE(Selected::IsSupportedInput(a));
            validate_native(selected);
            return 0.f;
        });
        EXPECT_GE(selection_status, 0);
    }
    auto launch = [&]() {
        if constexpr(Bias == ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            if(options.wide_bias_stride)
            {
                // The example API uses int32 strides. Check the native wide
                // stride contract on row zero without allocating its unused gap.
                EXPECT_TRUE(options.virtual_sink.empty());
                EXPECT_GE(Kernel::VisitKernels(
                              a,
                              [](auto selected) {
                                  EXPECT_TRUE((std::is_same_v<decltype(selected), Strided>));
                                  return 0.f;
                              }),
                          0);
                auto [native, grid] = fmha_fwd_create_kargs_and_grids<Strided>(a);
                native.stride_bias  = (int64_t{1} << 31) + 3;
                EXPECT_GT(native.stride_bias, std::numeric_limits<int32_t>::max());
                if(!Strided::IsSupportedArgument(native))
                    return -1.f;
                return ck_tile::launch_kernel(
                    ck_tile::stream_config{nullptr, false},
                    ck_tile::make_kernel<Strided::kBlockPerCu, ck_tile::gfx125_t>(
                        Strided{}, grid, Strided::BlockSize(), 0, native));
            }
        }
        // A direct typed adapter forces CR even with other pipelines in the build.
        return FmhaFwdLaunch<Kernel, ck_tile::gfx125_t>::Run(ck_tile::stream_config{nullptr, false},
                                                             a);
    };
    ASSERT_GE(launch(), 0);
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
            ASSERT_EQ(hipMemsetAsync(
                          dout.GetDeviceBuffer(), output_fill, o.size() * sizeof(Data), nullptr),
                      hipSuccess);
            ASSERT_GE(launch(), 0);
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
                ASSERT_EQ(ck_tile::bit_cast<uint16_t>(o[output_offset + row * output_stride + col]),
                          o_guard);
    for(int i = 0; i < output_offset; ++i)
    {
        EXPECT_EQ(ck_tile::bit_cast<uint16_t>(o[i]), o_guard);
        EXPECT_EQ(ck_tile::bit_cast<uint16_t>(o[o.size() - 1 - i]), o_guard);
    }
    for(int b = 0; b < B; ++b)
    {
        if constexpr(Group)
            for(int row = lengths_q[b]; row < sq; ++row)
                for(int head = 0; head < H; ++head)
                {
                    for(int d = 0; d < V; ++d)
                        ASSERT_EQ(ck_tile::bit_cast<uint16_t>(
                                      o[output_offset + (b * sq + row) * output_stride +
                                        head * output_head_stride + d]),
                                  o_guard)
                            << "group padding b=" << b << " h=" << head << " q=" << row;
                    if constexpr(StoreLSE)
                        ASSERT_EQ(ck_tile::bit_cast<uint32_t>(lse[head * B * sq + b * sq + row]),
                                  lse_guard);
                }
        // Long regressions execute the full B32/H8 grid; sample its first and
        // last batch/head to bound the independent host reference cost.
        if(sq >= 16384 && b != 0 && b != B - 1)
            continue;
        const int qlen = Group ? lengths_q[b] : sq, klen = Group ? lengths_k[b] : sk;
        if constexpr(Skip)
            if(qlen <= options.min_seqlen_q)
            {
                for(int row = 0; row < qlen; ++row)
                    for(int head = 0; head < H; ++head)
                    {
                        for(int d = 0; d < V; ++d)
                            ASSERT_EQ(ck_tile::bit_cast<uint16_t>(
                                          o[output_offset + (b * sq + row) * output_stride +
                                            head * output_head_stride + d]),
                                      o_guard);
                        if constexpr(StoreLSE)
                            ASSERT_EQ(
                                ck_tile::bit_cast<uint32_t>(lse[head * B * sq + b * sq + row]),
                                lse_guard);
                    }
                continue;
            }
        // Sample both sides of tile boundaries plus first/last rows. Every value
        // dimension is checked; the GPU still computes all rows.
        std::vector<int> checked_rows = {0,
                                         std::min(16, qlen - 1),
                                         std::min(63, qlen - 1),
                                         std::min(128, qlen - 1),
                                         std::max(0, qlen / 2 - 1),
                                         qlen / 2,
                                         qlen - 1};
        if(options.check_all_rows)
        {
            checked_rows.resize(qlen);
            for(int row = 0; row < qlen; ++row)
                checked_rows[row] = row;
        }
        for(int row : checked_rows)
            for(int head = 0; head < H; ++head)
            {
                if(sq >= 16384 && head != 0 && head != H - 1)
                    continue;
                const int kh = head / (H / kv_heads);
                std::vector<double> scores(klen, -std::numeric_limits<double>::infinity());
                double maximum = -std::numeric_limits<double>::infinity();
                for(int col = 0; col < klen; ++col)
                {
                    const int center = row + (mask_type != 1 ? klen - qlen : 0);
                    const int mask_y = 1 + (left < 0 ? (mask_type == 1 ? qlen : klen) - 1 : left) +
                                       (mask_type == 1 ? 0 : qlen - klen);
                    const bool in_sink = Sink && col < options.sink_size && col <= center &&
                                         mask_y < qlen && row < klen;
                    if(Mask && !in_sink && (col > center || (left >= 0 && col < center - left)))
                        continue;
                    double dot = 0;
                    for(int d = 0; d < QDim; ++d)
                        dot += ck_tile::type_convert<float>(
                                   q[((b * sq + row) * H + head) * QDim + d]) *
                               ck_tile::type_convert<float>(
                                   k[((b * sk + col) * kv_heads + kh) * QDim + d]);
                    scores[col] = dot * a.scale_s;
                    if constexpr(SoftCap)
                    {
#if CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH
                        scores[col] =
                            a.logits_soft_cap * std::tanh(scores[col] / a.logits_soft_cap);
#else
                        scores[col] /= 1 + std::abs(scores[col] / a.logits_soft_cap);
#endif
                    }
                    if constexpr(Bias == ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    {
                        const auto offset =
                            (Group ? b * sq * a.stride_bias : b * a.batch_stride_bias) +
                            head * a.nhead_stride_bias + row * a.stride_bias + col;
                        scores[col] += ck_tile::type_convert<float>(bias[offset]);
                    }
                    else if constexpr(Bias == ck_tile::BlockAttentionBiasEnum::ALIBI)
                    {
                        // Pure causal uses a row-constant-equivalent vertical bias.
                        const double position =
                            Mask && left < 0
                                ? col
                                : -std::abs(row + ((!Mask || mask_type == 2) ? klen - qlen : 0) -
                                            col);
                        scores[col] += slopes[b * a.stride_bias + head] * position;
                    }
                    maximum = std::max(maximum, scores[col]);
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
                const double sink = options.virtual_sink.empty()
                                        ? -std::numeric_limits<double>::infinity()
                                        : options.virtual_sink[head];
                maximum           = std::max(maximum, sink);
                double sum        = std::isfinite(maximum) ? std::exp(sink - maximum) : 0;
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
                    if(std::isnan(sink))
                        EXPECT_TRUE(std::isnan(actual));
                    else
                        EXPECT_NEAR(actual, expected_value, tolerance)
                            << "b=" << b << " h=" << head << " q=" << row << " d=" << d;
                }
                if constexpr(StoreLSE)
                {
                    const float actual_lse =
                        lse[Group ? head * B * sq + b * sq + row : (b * H + head) * sq + row];
                    if(std::isnan(sink))
                        EXPECT_TRUE(std::isnan(actual_lse));
                    else if(sink == std::numeric_limits<double>::infinity())
                        EXPECT_EQ(actual_lse, std::numeric_limits<float>::infinity());
                    else if(sum)
                        EXPECT_NEAR(actual_lse, maximum + std::log(sum), 0.002);
                    else
                        EXPECT_EQ(actual_lse, -std::numeric_limits<float>::infinity());
                }
            }
    }
}

template <typename Data, int D, bool PackedD32 = false>
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
        if constexpr(D != 32 || !PackedD32)
        {
            CheckForward<Data, D, false, true, 64>(257, 130, 2, 2, 31, options);
            CheckForward<Data, D, true, false, 64>(129, 257, 2, 0, -1, options);
        }
        // The empty/valid boundary cuts through a tile. M128 also exercises
        // both query fragments and the D192 rectangular geometry.
        if constexpr(D != 32)
            CheckForward<Data, D, false, true, 128>(257, 130, 2, 2, 31, options);
        else if constexpr(PackedD32)
            CheckForward<Data,
                         D,
                         false,
                         true,
                         64,
                         true,
                         2,
                         true,
                         false,
                         ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                         false,
                         false,
                         false,
                         128>(257, 130, 2, 2, 31, options);
        if constexpr(D == 192)
        {
            // The typed API can select dense M128 General even though the
            // generated dispatcher reserves this geometry for virtual sinks.
            auto dense_options           = options;
            dense_options.check_all_rows = true;
            CheckForward<Data, D, false, false, 128>(129, 65, 2, 0, -1, dense_options);
            CheckForward<Data, D, false, false, 128, true, 2, false>(
                129, 65, 2, 0, -1, dense_options);
        }
    }
    options.scale_s           = 0.125f;
    options.output_offset     = 1;
    options.output_padding    = 1;
    options.output_head_major = true;
    if constexpr(D != 32 || !PackedD32)
        CheckForward<Data, D, false, true, 64>(257, 129, 2, 2, 31, options);
    if constexpr(D != 32)
        CheckForward<Data, D, false, true, 128>(257, 130, 2, 2, 31, options);
    else if constexpr(PackedD32)
        CheckForward<Data,
                     D,
                     false,
                     true,
                     64,
                     true,
                     2,
                     true,
                     false,
                     ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                     false,
                     false,
                     false,
                     128>(257, 130, 2, 2, 31, options);
    options.output_offset = options.output_padding = 0;
    options.input_offset = options.input_padding = 1;
    if constexpr(D != 32 || !PackedD32)
        CheckForward<Data, D, false, true, 64>(257, 129, 2, 2, 31, options);
}

template <typename Data, int D, bool PackedD32 = false>
void CheckHeadDimensions()
{
    ForwardTestOptions options;
    options.input_pattern   = InputPattern::Random;
    options.fixed_tolerance = true;
    options.repeats         = 8;
    if constexpr(D == 32 && PackedD32)
    {
        // Long exact D32 heads use M64/N128. Cover its key tail and
        // fully masked query tiles with the packed-register build.
        using Bias = ck_tile::BlockAttentionBiasEnum;
        CheckForward<Data,
                     D,
                     false,
                     false,
                     64,
                     true,
                     2,
                     true,
                     false,
                     Bias::NO_BIAS,
                     false,
                     false,
                     false,
                     128>(129, 257, 2, 0, -1, options);
        CheckForward<Data,
                     D,
                     false,
                     true,
                     64,
                     true,
                     2,
                     false,
                     false,
                     Bias::NO_BIAS,
                     false,
                     false,
                     false,
                     128>(257, 130, 2, 2, -1, options);
        options.second_group_has_keys = true;
        CheckForward<Data,
                     D,
                     true,
                     true,
                     64,
                     true,
                     2,
                     true,
                     false,
                     Bias::NO_BIAS,
                     false,
                     false,
                     false,
                     128>(257, 130, 1, 2, 31, options);
    }
    if constexpr(D != 32 || !PackedD32)
    {
        CheckForward<Data, D, false, false, 64>(129, 257, 2, 0, -1, options);
        CheckForward<Data, D, false, true, 64>(257, 130, 2, 2, 31, options);
    }
    if constexpr(D != 32)
    {
        CheckForward<Data, D, false, false, 128>(257, 515, 2, 0, -1, options);
        CheckForward<Data, D, false, true, 128>(257, 515, 2, 1, -1, options);
        CheckForward<Data, D, true, true, 128>(257, 130, 1, 2, 31, options);
        options.second_group_has_keys = true;
        CheckForward<Data, D, true, false, 128>(257, 515, 2, 0, -1, options);
        CheckForward<Data, D, true, true, 128>(257, 515, 2, 2, 31, options);
    }
    if constexpr(D == 64 || D == 192)
    {
        // Native heads also use the virtual-sink entry when no prefix tokens
        // are retained. Cover both tile sizes and fully masked query rows.
        auto sink_options           = options;
        sink_options.check_all_rows = true;
        sink_options.virtual_sink   = {-INFINITY, -80.f, -2.f, 0.f, 2.f, 8.f, 80.f, INFINITY};
        CheckForward<Data, D, false, false, 64>(129, 257, 2, 0, -1, sink_options);
        CheckForward<Data, D, false, false, 128>(257, 515, 2, 0, -1, sink_options);
        CheckForward<Data, D, false, true, 64>(257, 129, 2, 2, 31, sink_options);
        CheckForward<Data, D, false, true, 128>(257, 33, 2, 2, -1, sink_options);
        CheckForward<Data, D, true, true, 128>(257, 130, 1, 2, 31, sink_options);
        if constexpr(D == 64)
        {
            // Cooperative M64/N128 has a distinct BF16 fully visible prefix.
            // Cover that prefix, partial K tiles, and fully masked Q tiles.
            using Bias = ck_tile::BlockAttentionBiasEnum;
            CheckForward<Data,
                         D,
                         false,
                         true,
                         64,
                         true,
                         1,
                         true,
                         false,
                         Bias::NO_BIAS,
                         false,
                         false,
                         false,
                         128>(257, 515, 2, 2, -1, sink_options);
            CheckForward<Data,
                         D,
                         false,
                         true,
                         64,
                         true,
                         1,
                         false,
                         false,
                         Bias::NO_BIAS,
                         false,
                         false,
                         false,
                         128>(257, 129, 2, 2, -1, sink_options);
        }
        if constexpr(D == 192)
        {
            // Empty tiles use vector stores only when both the pointer and
            // row stride are aligned. Exercise each fallback independently.
            sink_options.causal_sink_tuning  = false;
            sink_options.check_output_writes = true;
            sink_options.virtual_sink[0]     = NAN;
            sink_options.output_padding      = 8;
            CheckForward<Data, D, false, true, 64>(257, 33, 2, 2, -1, sink_options);
            sink_options.output_offset = 4;
            CheckForward<Data, D, false, true, 64>(257, 33, 2, 2, -1, sink_options);
            CheckForward<Data, D, false, true, 128>(257, 33, 2, 2, -1, sink_options);
            sink_options.output_offset  = 0;
            sink_options.output_padding = 4;
            CheckForward<Data, D, false, true, 64>(257, 33, 2, 2, -1, sink_options);
            sink_options.output_padding    = 8;
            sink_options.output_head_major = true;
            CheckForward<Data, D, false, true, 128>(257, 33, 2, 2, -1, sink_options);
            CheckForward<Data, D, true, false, 128>(129, 0, 2, 0, -1, sink_options);
            // The causal NLSE entry has a different empty-tile layout. Check
            // its selection and argument contract alongside actual outputs.
            sink_options.causal_sink_tuning = true;
            sink_options.virtual_sink[1]    = -INFINITY;
            sink_options.output_head_major  = false;
            CheckForward<Data, D, false, true, 64, true, 2, false>(257, 33, 2, 2, -1, sink_options);
            sink_options.output_offset = sink_options.output_padding = 1;
            sink_options.second_group_has_keys                       = false;
            for(float scale : {0.f, -0.25f, 0.125f})
            {
                sink_options.scale_s = scale;
                CheckForward<Data, D, false, true, 64, true, 2, false>(
                    33, 257, 2, 1, -1, sink_options);
                CheckForward<Data, D, true, true, 64, true, 2, false>(
                    257, 33, 2, 2, -1, sink_options);
            }
            sink_options.causal_sink_tuning = false;
            CheckForward<Data, D, false, true, 64, true, 2, false>(
                257, 129, 2, 2, 31, sink_options);
        }
    }
    options.second_group_has_keys = false;
    constexpr int V               = Model<Data, D, false, false>::V;
    for(int width : {1, D - 1, D})
    {
        options.hdim_q       = width;
        options.hdim_v       = width == D ? V : std::min(width, V - 1);
        options.input_offset = options.output_offset = 1;
        options.input_padding = options.output_padding = 1;
        if constexpr(D != 32 || !PackedD32)
        {
            CheckForward<Data, D, false, false, 64, true, 2, true, true>(
                129, 257, 2, 0, -1, options);
            CheckForward<Data, D, true, true, 64, true, 2, true, true>(257, 130, 1, 2, 31, options);
        }
        if constexpr(D != 32 || PackedD32)
        {
            CheckForward<Data, D, false, true, 128, true, 2, true, true>(
                257, 515, 2, 2, 31, options);
            CheckForward<Data, D, true, true, 128, true, 2, true, true>(
                257, 130, 1, 2, 31, options);
        }
    }
    if constexpr(D != 32 || PackedD32)
    {
        options.hdim_q = D / 2;
        options.hdim_v = V;
        CheckForward<Data, D, false, false, 128, true, 2, true, true>(257, 515, 2, 0, -1, options);
        options.second_group_has_keys = true;
        options.hdim_v                = V - 1;
        CheckForward<Data, D, true, false, 128, true, 2, true, true>(257, 515, 2, 0, -1, options);
        CheckForward<Data, D, true, true, 128, true, 2, true, true>(257, 515, 2, 2, 31, options);
        if constexpr(D == 96)
        {
            // Padded dense virtual sinks retain the M128/N64 specialization.
            options.hdim_q       = 65;
            options.hdim_v       = 95;
            options.virtual_sink = {-INFINITY, -80.f, -2.f, 0.f, 2.f, 8.f, 80.f, INFINITY};
            CheckForward<Data,
                         D,
                         false,
                         false,
                         128,
                         true,
                         2,
                         true,
                         true,
                         ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                         false,
                         false,
                         false,
                         64>(257, 515, 2, 0, -1, options);
        }
    }
}

template <typename Data>
void CheckScoreTransforms()
{
    using Bias = ck_tile::BlockAttentionBiasEnum;
    ForwardTestOptions options;
    options.input_pattern   = InputPattern::Random;
    options.fixed_tolerance = true;
    options.check_all_rows  = true;
    options.repeats         = 4;
    options.output_padding  = 1;
    options.output_offset   = 1;
    for(bool shared : {false, true})
    {
        options.shared_bias = shared;
        options.masked_bias = shared;
        options.scale_s     = shared ? 0.f : -0.25f;
        CheckForward<Data, 32, false, false, 64, true, 2, true, false, Bias::ELEMENTWISE_BIAS>(
            131, 257, 2, 0, -1, options);
        options.hdim_q                = 65;
        options.hdim_v                = 95;
        options.second_group_has_keys = true;
        // M64 uses exact odd-width K descriptors for elementwise bias;
        // the padded LDS columns must stay zero across the key tail.
        CheckForward<Data, 96, false, true, 64, true, 2, true, true, Bias::ELEMENTWISE_BIAS>(
            257, 129, 2, 2, 31, options);
        CheckForward<Data, 96, true, true, 128, true, 2, true, true, Bias::ELEMENTWISE_BIAS>(
            131, 257, 2, 2, 33, options);
        options.scale_s = std::numeric_limits<float>::quiet_NaN();
        options.hdim_q = options.hdim_v = 0;
        CheckForward<Data, 96, true, true, 128, true, 2, true, false, Bias::ALIBI>(
            257, 131, 2, 2, 33, options);
        CheckForward<Data, 96, true, true, 128, true, 2, true, false, Bias::ALIBI>(
            257, 131, 2, 3, 33, options);
        for(int mask_type : {1, 2})
            for(int left : {-1, 33})
                CheckForward<Data, 96, false, true, 128, true, 2, true, false, Bias::ALIBI>(
                    131, 257, 2, mask_type, left, options);
    }
    // Bottom-right causal ALiBi includes fully masked and valid rows in the
    // same wave. Rescaling a valid row must preserve the empty rows' zero O
    // and -infinity LSE, including when the slope changes quickly across a tile.
    options.shared_bias = false;
    for(float slope_scale : {1.f, 8.f})
    {
        options.alibi_slope_scale = slope_scale;
        CheckForward<Data, 128, false, true, 64, true, 2, true, false, Bias::ALIBI>(
            515, 257, 2, 2, -1, options);
        CheckForward<Data, 96, false, true, 128, true, 2, true, false, Bias::ALIBI>(
            513, 513, 2, 2, -1, options);
    }
    // The wide ALiBi path uses a finite maximum even for FP16. Exercise a
    // fully masked prefix followed by valid rows, and a window with a virtual sink.
    for(float scale : {-0.25f, 0.f, 0.25f})
    {
        options.scale_s = scale;
        CheckForward<Data, 160, false, true, 128, true, 2, true, false, Bias::ALIBI>(
            515, 259, 2, 2, -1, options);
        options.virtual_sink = {-INFINITY, -80.f, -2.f, 0.f, 2.f, 8.f, 80.f, INFINITY};
        CheckForward<Data, 160, false, true, 128, true, 2, true, false, Bias::ALIBI>(
            515, 259, 2, 2, 63, options);
        options.virtual_sink.clear();
    }
    auto group_options = options;
    for(bool has_keys : {false, true})
    {
        group_options.second_group_has_keys = has_keys;
        // Group M64 score entries without LSE prepare offsets before lengths.
        CheckForward<Data, 128, true, true, 64, true, 2, false, false, Bias::ALIBI>(
            257, 515, 2, 2, -1, group_options);
        CheckForward<Data, 160, true, true, 128, true, 2, true, false, Bias::ALIBI>(
            515, 259, 2, 2, 63, group_options);
    }
    options.alibi_slope_scale = 1.f;
    // Bias can mask the first key while leaving the rest of its tile valid.
    // Cover both the initial LSE and rescaling across a subsequent K tile.
    options.masked_bias              = false;
    options.first_column_masked_bias = true;
    options.scale_s                  = 0.f;
    for(int keys : {64, 129})
    {
        CheckForward<Data, 32, false, false, 64, true, 2, true, false, Bias::ELEMENTWISE_BIAS>(
            131, keys, 2, 0, -1, options);
        CheckForward<Data, 96, false, false, 128, true, 2, true, false, Bias::ELEMENTWISE_BIAS>(
            131, keys, 2, 0, -1, options);
    }
    options.first_column_masked_bias = false;
    options.fragment_masked_bias     = true;
    for(int keys : {13, 67, 129})
    {
        CheckForward<Data, 128, false, true, 64, true, 2, false, false, Bias::ELEMENTWISE_BIAS>(
            131, keys, 2, 2, -1, options);
        CheckForward<Data, 96, false, false, 128, true, 2, true, false, Bias::ELEMENTWISE_BIAS>(
            131, keys, 2, 0, -1, options);
    }
    options.shared_bias      = true;
    options.wide_bias_stride = true;
    CheckForward<Data, 128, false, true, 64, true, 2, false, false, Bias::ELEMENTWISE_BIAS>(
        1, 67, 2, 2, -1, options);
    CheckForward<Data, 96, false, false, 128, true, 2, true, false, Bias::ELEMENTWISE_BIAS>(
        1, 67, 2, 0, -1, options);
    options.wide_bias_stride = options.fragment_masked_bias = false;
    for(float cap : {0.5f, 8.f})
        for(float scale : {-0.25f, 0.f, 0.25f})
        {
            options.logits_soft_cap = cap;
            options.scale_s         = scale;
            CheckForward<Data, 32, false, false, 64, true, 2, true, false, Bias::NO_BIAS, true>(
                131, 257, 2, 0, -1, options);
            CheckForward<Data, 160, false, true, 64, true, 2, true, false, Bias::NO_BIAS, true>(
                131, 257, 2, 2, 33, options);
        }
    options.scale_s = std::numeric_limits<float>::quiet_NaN();
    CheckForward<Data, 32, false, false, 64, true, 2, true, false, Bias::ALIBI>(
        3, 65569, 2, 0, -1, options);
}

template <typename Data>
void CheckSinkAndSkip()
{
    using Bias = ck_tile::BlockAttentionBiasEnum;
    ForwardTestOptions options;
    options.input_pattern   = InputPattern::Random;
    options.fixed_tolerance = true;
    options.check_all_rows  = true;
    options.repeats         = 4;
    options.output_padding  = 1;
    options.output_offset   = 1;
    {
        // Odd K widths may prefetch an extra column. Poison that column to
        // verify its LDS repair across logical group tails and sink jumps.
        auto padded_options                  = options;
        padded_options.hdim_q                = 65;
        padded_options.hdim_v                = 95;
        padded_options.input_padding         = 1;
        padded_options.input_offset          = 1;
        padded_options.sink_size             = 65;
        padded_options.second_group_has_keys = true;
        CheckForward<Data, 96, true, true, 128, true, 2, true, true, Bias::NO_BIAS, false, true>(
            257, 257, 1, 2, 31, padded_options);
    }
    for(int sink_size : {1, 63, 64, 65, 129, 600})
    {
        options.sink_size = sink_size;
        CheckForward<Data, 32, false, true, 64, true, 2, true, false, Bias::NO_BIAS, false, true>(
            257, 515, 2, 2, 31, options);
        options.hdim_q                = 65;
        options.hdim_v                = 95;
        options.second_group_has_keys = true;
        CheckForward<Data, 96, true, true, 128, true, 2, true, true, Bias::ALIBI, false, true>(
            257, 193, 1, 1, 63, options);
        options.hdim_q = options.hdim_v = 0;
    }
    // Bias tiles follow the same disjoint sink/window traversal. Cover both
    // a rectangular tile in a later group and the virtual softmax denominator.
    options.masked_bias = true;
    for(int sink_size : {65, 129})
    {
        options.sink_size = sink_size;
        options.hdim_q    = 65;
        options.hdim_v    = 95;
        CheckForward<Data,
                     96,
                     true,
                     true,
                     128,
                     true,
                     2,
                     true,
                     true,
                     Bias::ELEMENTWISE_BIAS,
                     false,
                     true>(257, 515, 2, 2, 31, options);
        options.virtual_sink = {-INFINITY, -80.f, -2.f, 0.f, 2.f, 8.f, 80.f, INFINITY};
        CheckForward<Data,
                     96,
                     true,
                     true,
                     128,
                     true,
                     2,
                     true,
                     true,
                     Bias::ELEMENTWISE_BIAS,
                     false,
                     true>(257, 515, 2, 2, 31, options);
        options.hdim_q = options.hdim_v = 0;
        CheckForward<Data,
                     128,
                     false,
                     true,
                     64,
                     true,
                     2,
                     true,
                     false,
                     Bias::ELEMENTWISE_BIAS,
                     false,
                     true>(257, 515, 2, 2, 31, options);
        options.virtual_sink.clear();
    }
    options.masked_bias  = false;
    options.sink_size    = 0;
    options.virtual_sink = {-INFINITY, -80.f, -2.f, 0.f, 2.f, 8.f, 80.f, INFINITY};
    for(float scale : {0.f, -0.25f, 0.125f})
    {
        options.scale_s = scale;
        CheckForward<Data, 128, false, false, 64>(131, 257, 2, 0, -1, options);
        CheckForward<Data, 160, false, true, 128>(257, 130, 1, 2, 31, options);
        CheckForward<Data, 32, false, false, 64>(131, 0, 2, 0, -1, options);
        CheckForward<Data, 96, false, false, 128>(131, 257, 2, 0, -1, options);
        CheckForward<Data, 96, false, false, 128>(131, 0, 2, 0, -1, options);
        CheckForward<Data, 96, true, false, 128>(257, 515, 2, 0, -1, options);
    }
    options.scale_s         = 0.125f;
    options.sink_size       = 0;
    options.hdim_q          = 31;
    options.hdim_v          = 17;
    options.virtual_sink[0] = NAN;
    CheckForward<Data, 32, false, false, 64, true, 2, true, true>(131, 0, 2, 0, -1, options);
    options.virtual_sink.clear();
    options.hdim_q = options.hdim_v = 0;
    options.sink_size               = 129;
    CheckForward<Data, 128, false, true, 128, true, 2, true, false, Bias::NO_BIAS, false, true>(
        257, 1025, 2, 2, 31, options);
    CheckForward<Data, 128, false, true, 128, true, 2, true, false, Bias::NO_BIAS, false, true>(
        257, 515, 2, 1, -1, options);
    options.sink_size = 0;
    // The first group's length is 257, the second 145: the threshold is inclusive.
    options.second_group_has_keys = true;
    for(int minimum : {144, 145, 256, 257})
    {
        options.min_seqlen_q = minimum;
        CheckForward<Data,
                     32,
                     true,
                     true,
                     64,
                     true,
                     2,
                     true,
                     false,
                     Bias::ELEMENTWISE_BIAS,
                     false,
                     false,
                     true>(257, 515, 2, 2, 31, options);
        CheckForward<Data,
                     128,
                     true,
                     true,
                     128,
                     true,
                     2,
                     true,
                     false,
                     Bias::NO_BIAS,
                     false,
                     false,
                     true>(257, 515, 2, 2, 31, options);
        // Exercise the skip-only TDM epilogue as well as scalar General output.
        options.output_padding = options.output_offset = 0;
        CheckForward<Data,
                     128,
                     true,
                     true,
                     128,
                     true,
                     2,
                     true,
                     false,
                     Bias::NO_BIAS,
                     false,
                     false,
                     true>(257, 515, 2, 2, 31, options);
        options.output_padding = options.output_offset = 1;
    }
}

// D32's M64/N128 and M128 tiles use the same packed-register compiler
// setting as D64. Keep their feature cases in that existing build partition.
template <typename Data>
void CheckPackedD32Features()
{
    using Bias = ck_tile::BlockAttentionBiasEnum;
    ForwardTestOptions options;
    options.input_pattern         = InputPattern::Random;
    options.fixed_tolerance       = true;
    options.check_all_rows        = true;
    options.repeats               = 4;
    options.output_padding        = 1;
    options.output_offset         = 1;
    options.hdim_q                = 31;
    options.hdim_v                = 17;
    options.second_group_has_keys = true;
    for(bool shared : {false, true})
    {
        options.shared_bias = shared;
        options.masked_bias = shared;
        CheckForward<Data, 32, false, true, 128, true, 2, true, true, Bias::ELEMENTWISE_BIAS>(
            257, 129, 2, 2, -1, options);
        CheckForward<Data, 32, true, true, 128, true, 2, false, true, Bias::ELEMENTWISE_BIAS>(
            257, 129, 2, 2, -1, options);
        CheckForward<Data, 32, false, false, 128, true, 2, false, true, Bias::ALIBI>(
            257, 131, 2, 0, -1, options);
    }
    options.masked_bias           = false;
    options.second_group_has_keys = false;
    for(float cap : {0.5f, 8.f})
        for(float scale : {-0.25f, 0.f, 0.25f})
        {
            options.logits_soft_cap = cap;
            options.scale_s         = scale;
            CheckForward<Data, 32, true, true, 128, true, 2, true, true, Bias::NO_BIAS, true>(
                131, 257, 2, 2, -1, options);
        }
    options.hdim_q = options.hdim_v = 0;
    options.virtual_sink            = {-INFINITY, -80.f, -2.f, 0.f, 2.f, 8.f, 80.f, INFINITY};
    // A subnormal scale also exercises the folding guard at zero.
    for(float scale : {0.f, -0.25f, 1.e-40f, 0.125f})
    {
        options.scale_s = scale;
        // The long dense virtual-sink specialization retains M128/N64.
        CheckForward<Data, 32, false, false, 128>(257, 130, 2, 0, -1, options);
    }
    options.sink_size = 65;
    CheckForward<Data,
                 32,
                 true,
                 true,
                 64,
                 true,
                 2,
                 true,
                 false,
                 Bias::NO_BIAS,
                 true,
                 true,
                 false,
                 128>(257, 515, 2, 2, 31, options);
    options.sink_size = 0;
    options.hdim_q    = 31;
    options.hdim_v    = 17;
    CheckForward<Data, 32, false, true, 128, true, 2, true, true>(257, 130, 1, 2, 31, options);
    options.virtual_sink[0] = NAN;
    CheckForward<Data, 32, false, true, 128, true, 2, true, true>(257, 130, 1, 2, 31, options);
}

#if defined(CK_TILE_TEST_CR_HEAD_DIM)
TEST_F(CrForward, SinkAndSkipFp16) { CheckSinkAndSkip<ck_tile::half_t>(); }
TEST_F(CrForward, SinkAndSkipBf16) { CheckSinkAndSkip<ck_tile::bf16_t>(); }
TEST_F(CrForward, ScoreTransformsFp16) { CheckScoreTransforms<ck_tile::half_t>(); }
TEST_F(CrForward, ScoreTransformsBf16) { CheckScoreTransforms<ck_tile::bf16_t>(); }

TEST_F(CrForward, HeadDimensionsAndPaddingFp16)
{
    CheckHeadDimensions<ck_tile::half_t, 32>();
    CheckHeadDimensions<ck_tile::half_t, 96>();
    CheckHeadDimensions<ck_tile::half_t, 128>();
    CheckHeadDimensions<ck_tile::half_t, 160>();
    CheckHeadDimensions<ck_tile::half_t, 192>();
}

TEST_F(CrForward, HeadDimensionsAndPaddingBf16)
{
    CheckHeadDimensions<ck_tile::bf16_t, 32>();
    CheckHeadDimensions<ck_tile::bf16_t, 96>();
    CheckHeadDimensions<ck_tile::bf16_t, 128>();
    CheckHeadDimensions<ck_tile::bf16_t, 160>();
    CheckHeadDimensions<ck_tile::bf16_t, 192>();
}

TEST_F(CrForward, HeadDimensionsGeneralViews)
{
    CheckGeneralViews<ck_tile::half_t, 32>();
    CheckGeneralViews<ck_tile::bf16_t, 32>();
    CheckGeneralViews<ck_tile::half_t, 96>();
    CheckGeneralViews<ck_tile::bf16_t, 96>();
    CheckGeneralViews<ck_tile::half_t, 160>();
    CheckGeneralViews<ck_tile::bf16_t, 160>();
}

TEST_F(CrForward, RepeatedD160Output)
{
    ForwardTestOptions options;
    options.input_pattern   = InputPattern::Random;
    options.fixed_tolerance = true;
    options.repeats         = 16384;
    // Five 32-element TDM chunks, with tails in both query and key sequences.
    CheckForward<ck_tile::half_t, 160, false, true, 128>(257, 515, 2, 1, -1, options);
    CheckForward<ck_tile::bf16_t, 160, false, true, 128>(257, 515, 2, 1, -1, options);
}
#elif defined(CK_TILE_TEST_CR_LONG)
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
TEST_F(CrForward, HeadDimensionsAndPadding)
{
    // D64 and D32's large tiles use packed register allocation in codegen.
    CheckHeadDimensions<ck_tile::half_t, 32, true>();
    CheckHeadDimensions<ck_tile::bf16_t, 32, true>();
    CheckHeadDimensions<ck_tile::half_t, 64>();
    CheckHeadDimensions<ck_tile::bf16_t, 64>();
}

TEST_F(CrForward, GeneralScaleAndOutputViews)
{
    CheckGeneralViews<ck_tile::half_t, 32, true>();
    CheckGeneralViews<ck_tile::bf16_t, 32, true>();
    CheckGeneralViews<ck_tile::half_t, 64>();
    CheckGeneralViews<ck_tile::bf16_t, 64>();
}

TEST_F(CrForward, PackedD32ScoreAndSequenceFeatures)
{
    CheckPackedD32Features<ck_tile::half_t>();
    CheckPackedD32Features<ck_tile::bf16_t>();
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
TEST_F(CrForward, RescalingPreservesFullyMaskedRows)
{
    ForwardTestOptions options;
    options.input_pattern   = InputPattern::MaskedRescaling;
    options.scale_s         = 1.f;
    options.fixed_tolerance = true;
    // At K=128, row 160 rescales while rows 163..175 in the same wave are
    // fully masked. Their zero accumulators and -infinity LSE must survive.
    options.check_all_rows = true;
    CheckForward<ck_tile::half_t, 128, false, true, 64>(176, 130, 2, 1, 33, options);
    CheckForward<ck_tile::bf16_t, 128, false, true, 64>(176, 130, 2, 1, 33, options);
    CheckForward<ck_tile::half_t, 128, false, true, 128>(176, 130, 2, 1, 33, options);
    CheckForward<ck_tile::bf16_t, 128, false, true, 128>(176, 130, 2, 1, 33, options);
    // The scalar-aligned output adapter shares the same rescaling guard.
    options.output_offset  = 1;
    options.output_padding = 1;
    CheckForward<ck_tile::half_t, 128, false, true, 64>(176, 130, 2, 1, 33, options);
    CheckForward<ck_tile::bf16_t, 128, false, true, 64>(176, 130, 2, 1, 33, options);
    CheckForward<ck_tile::half_t, 128, false, true, 64, true, 2, false>(
        176, 130, 2, 1, 33, options);
    CheckForward<ck_tile::bf16_t, 128, false, true, 64, true, 2, false>(
        176, 130, 2, 1, 33, options);
}

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
    CheckStoreRows<32, 40>();
    CheckStoreRows<64, 72>();
    CheckStoreRows<96, 120>();
    CheckStoreRows<128, 136>();
    CheckStoreRows<160, 200>();
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
        using Kernel = Model<ck_tile::bf16_t, 128, false, false, false>::CompactKernel;
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

    using Kernel = Model<ck_tile::bf16_t, 128, false, false, false>::CompactKernel;
    static_assert(std::is_same_v<Kernel::Kargs, ck_tile::detail::fmha_cr::Arguments>);
    static_assert(sizeof(Kernel::Kargs) == 64);
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
    using TailKernel = Model<ck_tile::bf16_t, 128, false, false>::CompactKernel;
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

TEST(CrArgumentContract, SharedPreparationPreservesWideStrides)
{
    using Definition = Model<ck_tile::half_t,
                             128,
                             false,
                             true,
                             true,
                             128,
                             true,
                             false,
                             ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS>;
    using Kernel     = Definition::Kernel;
    static_assert(std::is_same_v<Kernel, Kernel::WithTuning<ck_tile::FmhaTdmCrTuning<>>>);
    using Reference =
        ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSTdm<typename Definition::Problem>,
                               typename Definition::Epilogue>;
    static_assert(std::is_same_v<decltype(std::declval<typename Reference::Kargs>().nhead_stride_q),
                                 ck_tile::index_t>);

    alignas(16) float storage[4]{};
    // The typed kernel API permits large tensor offsets independently of the
    // example API's index_t fields. No device allocation or launch is needed.
    struct WideInput : ck_tile::detail::FmhaFwdCrInput
    {
        int head_start       = 0;
        int num_head_q_total = 0;
    };
    WideInput input{};
    input.q_ptr = input.k_ptr = input.v_ptr = input.bias_ptr = storage;
    input.o_ptr = input.lse_ptr = storage;
    input.batch                 = 2;
    input.nhead_q               = 4;
    input.nhead_k               = 2;
    input.hdim_q = input.hdim_v = 128;
    input.seqlen_q = input.max_seqlen_q = 17;
    input.seqlen_k                      = 33;
    input.mask_type                     = 1;
    input.window_size_left              = -1;
    input.window_size_right             = 0;
    input.scale_s                       = 0.25f;
    constexpr int64_t stride            = (int64_t{1} << 33) + 128;
    input.nhead_stride_q = input.nhead_stride_k = input.nhead_stride_v = input.nhead_stride_o =
        stride;
    input.batch_stride_q = input.batch_stride_k = input.batch_stride_v = input.batch_stride_o =
        4 * stride;
    input.nhead_stride_lse  = stride;
    input.batch_stride_lse  = 4 * stride;
    input.stride_bias       = stride;
    input.nhead_stride_bias = 2 * stride;
    input.batch_stride_bias = 8 * stride;
    auto args               = Kernel::MakeKargs(input);
    EXPECT_EQ(args.nhead_stride_q, stride);
    EXPECT_EQ(args.batch_stride_o, 4 * stride);
    EXPECT_EQ(args.nhead_stride_lse, stride);
    EXPECT_EQ(args.stride_bias, stride);
    EXPECT_EQ(args.batch_stride_bias, 8 * stride);
    EXPECT_FLOAT_EQ(args.scale_s, input.scale_s);
    EXPECT_TRUE(Kernel::IsSupportedArgument(args));
    EXPECT_TRUE(Kernel::IsSupportedInput(input));
    input.max_seqlen_q = input.seqlen_q + 128;
    EXPECT_FALSE(Kernel::IsSupportedInput(input));
    input.max_seqlen_q = input.seqlen_q - 1;
    EXPECT_FALSE(Kernel::IsSupportedInput(input));
    input.max_seqlen_q = input.seqlen_q;
    input.head_start   = 1;
    EXPECT_FALSE(Kernel::IsSupportedInput(input));
    EXPECT_FALSE(Kernel::IsSupportedArgument(Kernel::MakeKargs(input)));
}

template <typename Definition, bool VirtualSink = false, bool General = true>
void CheckSharedFeatureArguments()
{
    using Problem = typename Definition::Problem;
    using Tuning  = std::
        conditional_t<VirtualSink, ck_tile::FmhaTdmCrSinkTuning, ck_tile::FmhaTdmCrGeneralTuning>;
    using Kernel = std::conditional_t<General || VirtualSink,
                                      typename Definition::Kernel::template WithTuning<Tuning>,
                                      typename Definition::Kernel>;
    alignas(16) float storage[4]{};
    int32_t starts[] = {0, 17, 34};
    ck_tile::detail::FmhaFwdCrInput input{};
    input.q_ptr = input.k_ptr = input.v_ptr = input.bias_ptr = storage;
    input.o_ptr = input.lse_ptr = storage;
    input.batch                 = 2;
    input.nhead_q               = 4;
    input.nhead_k               = 2;
    input.hdim_q = input.hdim_v = 32;
    if constexpr(Problem::kPadHeadDimQ || Problem::kPadHeadDimV)
    {
        input.hdim_q = 31;
        input.hdim_v = 17;
    }
    input.seqlen_q = input.max_seqlen_q = 17;
    input.seqlen_k                      = 33;
    input.scale_s                       = 0.25f;
    input.mask_type                     = 1;
    input.window_size_left              = 8;
    input.window_size_right             = 0;
    input.sink_size                     = 3;
    input.min_seqlen_q                  = 7;
    input.logits_soft_cap               = 4.0f;
    input.stride_q = input.stride_k = input.stride_v = input.stride_o = 128;
    input.nhead_stride_q = input.nhead_stride_k = input.nhead_stride_v = input.nhead_stride_o = 32;
    input.batch_stride_q = input.batch_stride_o = 17 * 128;
    input.batch_stride_k = input.batch_stride_v = 33 * 128;
    input.nhead_stride_lse                      = 17;
    input.batch_stride_lse                      = 4 * 17;
    input.stride_bias                           = 33;
    input.nhead_stride_bias                     = 17 * 33;
    input.batch_stride_bias                     = 4 * 17 * 33;
    if constexpr(Problem::kIsGroupMode)
    {
        input.seqstart_q_ptr = input.seqstart_k_ptr = starts;
        input.seqlen_q_ptr = input.seqlen_k_ptr = starts;
    }
    if constexpr(VirtualSink)
        input.sink_ptr = storage;
    const auto check = [&] {
        EXPECT_EQ(Kernel::IsSupportedArgument(Kernel::MakeKargs(input)),
                  Kernel::IsSupportedInput(input));
    };
    EXPECT_TRUE(Kernel::IsSupportedInput(input));
    check();
    auto args = Kernel::MakeKargs(input);
    EXPECT_EQ(args.nhead_ratio_qk, 2);
    EXPECT_EQ(args.sink_ptr, input.sink_ptr);
    EXPECT_EQ(args.nhead_stride_lse, input.nhead_stride_lse);
    EXPECT_EQ(args.window_size_left, input.window_size_left);
    if constexpr(Problem::kIsGroupMode)
    {
        EXPECT_EQ(args.seqstart_q_ptr, starts);
        EXPECT_EQ(args.seqlen_q_ptr, starts);
    }
    if constexpr(Problem::kHasSink)
    {
        EXPECT_EQ(args.sink_size, input.sink_size);
        input.sink_size = -1;
        EXPECT_FALSE(Kernel::IsSupportedInput(input));
        check();
        input.sink_size = 3;
    }
    if constexpr(Problem::kSkipMinSeqlenQ)
        EXPECT_EQ(args.min_seqlen_q, input.min_seqlen_q);
    if constexpr(Problem::BiasEnum != ck_tile::BlockAttentionBiasEnum::NO_BIAS)
    {
        if constexpr(Problem::BiasEnum == ck_tile::BlockAttentionBiasEnum::ALIBI)
        {
            EXPECT_EQ(args.alibi_slope_ptr, input.bias_ptr);
            EXPECT_EQ(args.alibi_slope_stride, input.stride_bias);
        }
        else
        {
            EXPECT_EQ(args.bias_ptr, input.bias_ptr);
            EXPECT_EQ(args.stride_bias, input.stride_bias);
            EXPECT_EQ(args.nhead_stride_bias, input.nhead_stride_bias);
        }
        input.bias_ptr = nullptr;
        EXPECT_FALSE(Kernel::IsSupportedInput(input));
        check();
        input.bias_ptr = storage;
    }
    if constexpr(Problem::kHasLogitsSoftCap)
    {
        EXPECT_EQ(args.logits_soft_cap, input.logits_soft_cap);
        input.logits_soft_cap = 0;
        EXPECT_FALSE(Kernel::IsSupportedInput(input));
        check();
    }
    input.stride_q = -1;
    EXPECT_FALSE(Kernel::IsSupportedInput(input));
    check();
    if constexpr(!General && !VirtualSink)
    {
        input.stride_q = 128;
        input.o_ptr    = reinterpret_cast<char*>(storage) + 2;
        // The selector may accept this input via General, but the vector entry
        // must reject a direct launch with unaligned output.
        EXPECT_TRUE(Kernel::IsSupportedInput(input));
        EXPECT_FALSE(Kernel::IsSupportedArgument(Kernel::MakeKargs(input)));
    }
}

TEST(CrArgumentContract, SharedFeatureArgumentsRoundTrip)
{
    using B = ck_tile::BlockAttentionBiasEnum;
    CheckSharedFeatureArguments<
        Model<ck_tile::bf16_t, 32, false, true, true, 64, true, false, B::ELEMENTWISE_BIAS>>();
    CheckSharedFeatureArguments<Model<ck_tile::bf16_t,
                                      32,
                                      true,
                                      true,
                                      true,
                                      64,
                                      true,
                                      false,
                                      B::ALIBI,
                                      false,
                                      true,
                                      true>>();
    CheckSharedFeatureArguments<
        Model<ck_tile::half_t, 32, false, true, true, 64, true, false, B::NO_BIAS, true, true>,
        true>();
    CheckSharedFeatureArguments<Model<ck_tile::half_t,
                                      32,
                                      true,
                                      true,
                                      true,
                                      64,
                                      true,
                                      false,
                                      B::NO_BIAS,
                                      false,
                                      false,
                                      true>,
                                false,
                                false>();
    CheckSharedFeatureArguments<
        Model<ck_tile::bf16_t, 64, true, true, true, 64, true, true, B::NO_BIAS>>();
}

#endif

} // namespace
