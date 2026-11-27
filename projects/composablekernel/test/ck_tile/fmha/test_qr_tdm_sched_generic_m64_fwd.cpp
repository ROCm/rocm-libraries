// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/device_memory.hpp"
#include "fmha_reference.hpp"
#include "fmha_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#ifndef DataTypeConfig
#define DataTypeConfig FmhaFwdFp16
#endif

namespace {

using Data = typename FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
static_assert(std::is_same_v<DataTypeConfig, FmhaFwdFp16> ||
              std::is_same_v<DataTypeConfig, FmhaFwdBf16>);
constexpr const char* kDataType = std::is_same_v<DataTypeConfig, FmhaFwdFp16> ? "fp16" : "bf16";
constexpr int kTileN = 128, kSpareRows = 64;
namespace reference = ck_tile::test::reference;

// Group-mode problems whose M128 grid underfills the CUs, so the generated dispatcher selects
// the M64 tile, the only tile in this instance set. Q, K, V and O are packed BSHD.
struct M64Case
{
    const char* name;
    int dimension, query_heads, kv_heads;
    std::vector<int32_t> q_lengths, k_lengths;
    mask_enum mask;
    int window_size_left, window_size_right;
};

const std::vector<M64Case> kCases = {
    // D64 M64 also inherits the D64 N128 causal contract (max_seqlen_q >= 4096).
    {"D64Causal", 64, 1, 1, {4096, 2500}, {4096, 2500}, mask_enum::mask_top_left, -1, 0},
    // K longer than Q moves the causal diagonal; GQA 4:2; 1000 rows end mid M64 tile.
    {"D128CausalBottomRight",
     128,
     4,
     2,
     {1000, 777},
     {1500, 777},
     mask_enum::mask_bottom_right,
     -1,
     0},
    // window_generic is the mask type that selects the GenericMask (mg) kernels. It aligns the
    // window to the bottom right, the same as the top left here because Q and K lengths match.
    {"D128Window", 128, 4, 2, {1000, 777}, {1000, 777}, mask_enum::window_generic, 200, 7},
};

std::string ExpectedKernel(const M64Case& c, bool lse)
{
    const bool d64 = c.dimension == 64;
    return std::string("fmha_fwd_d") + std::to_string(c.dimension) + "_" + kDataType +
           (d64 ? "_group_b64x128x32x64x32x64" : "_group_b64x128x32x128x32x128") +
           "_r4x1x1_r4x1x1_w16x16x32_w16x16x32" + (d64 ? "_o3" : "_o1") + "_qr_tdm_sched_vr" +
           (d64 ? "_pssk" : "_psskddv") + "_nlogits_nbias" +
           (c.mask == mask_enum::window_generic ? "_mg_" : "_mc_") + (lse ? "lse" : "nlse") +
           "_ndropout_nskip_nqscale_ntrload_nsink";
}

std::vector<Data> MakeInput(int rows, int heads, int dimension, int phase)
{
    std::vector<Data> data(std::size_t(rows) * heads * dimension);
    for(int row = 0; row < rows; ++row)
        for(int head = 0; head < heads; ++head)
            for(int dim = 0; dim < dimension; ++dim)
            {
                const float value = 0.35f * std::sin((row + 1) * (0.061f + phase * 0.017f) +
                                                     (dim + 1) * 0.137f + head * 0.71f) +
                                    0.11f * std::cos((row + 1) * (dim % 7 + 1) * 0.031f +
                                                     phase * 0.4f + head * 0.29f);
                data[(std::size_t(row) * heads + head) * dimension + dim] =
                    ck_tile::type_convert<Data>(value);
            }
    return data;
}

std::vector<float> Decode(const std::vector<Data>& data)
{
    std::vector<float> decoded(data.size());
    std::transform(data.begin(), data.end(), decoded.begin(), [](Data value) {
        return ck_tile::type_convert<float>(value);
    });
    return decoded;
}

class QrTdmSchedGenericM64Forward
    : public ck_tile::test::Gfx125FmhaTest,
      public ::testing::WithParamInterface<std::tuple<std::size_t, bool>>
{
};

TEST_P(QrTdmSchedGenericM64Forward, GeneratedM64FullReference)
{
    const auto& c  = kCases[std::get<0>(GetParam())];
    const bool lse = std::get<1>(GetParam());
    int device     = 0;
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);

    const int batch        = static_cast<int>(c.q_lengths.size());
    const int max_seqlen_q = *std::max_element(c.q_lengths.begin(), c.q_lengths.end());
    // Exercise the generated small-grid M64 rule (M128 blocks < CUs), not a forced launch.
    const long long m128_blocks =
        static_cast<long long>(batch) * c.query_heads * ((max_seqlen_q + 127) / 128);
    ASSERT_LT(m128_blocks, properties.multiProcessorCount);

    std::vector<int32_t> q_starts(batch + 1, 0), k_starts(batch + 1, 0);
    std::partial_sum(c.q_lengths.begin(), c.q_lengths.end(), q_starts.begin() + 1);
    std::partial_sum(c.k_lengths.begin(), c.k_lengths.end(), k_starts.begin() + 1);
    const int total_q       = q_starts.back();
    const int total_k       = k_starts.back();
    const std::size_t q_row = std::size_t(c.query_heads) * c.dimension;
    const std::size_t k_row = std::size_t(c.kv_heads) * c.dimension;

    const auto q    = MakeInput(total_q, c.query_heads, c.dimension, 0);
    const auto k    = MakeInput(total_k, c.kv_heads, c.dimension, 1);
    const auto v    = MakeInput(total_k, c.kv_heads, c.dimension, 2);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // Spare O rows past the last sequence catch stores beyond its end.
    std::vector<Data> output((std::size_t(total_q) + kSpareRows) * q_row,
                             ck_tile::type_convert<Data>(nan));
    std::vector<float> lse_data(std::size_t(c.query_heads) * total_q, nan);
    ck_tile::DeviceMem q_device(q.size() * sizeof(Data)), k_device(k.size() * sizeof(Data)),
        v_device(v.size() * sizeof(Data)), o_device(output.size() * sizeof(Data)),
        lse_device(lse_data.size() * sizeof(float)),
        q_starts_device(q_starts.size() * sizeof(int32_t)),
        k_starts_device(k_starts.size() * sizeof(int32_t));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    o_device.ToDevice(output.data());
    lse_device.ToDevice(lse_data.data());
    q_starts_device.ToDevice(q_starts.data());
    k_starts_device.ToDevice(k_starts.data());

    std::string selected_kernel;
    fmha_fwd_args args{};
    args.selected_kernel_name = &selected_kernel;
    args.q_ptr                = q_device.GetDeviceBuffer();
    args.k_ptr                = k_device.GetDeviceBuffer();
    args.v_ptr                = v_device.GetDeviceBuffer();
    args.o_ptr                = o_device.GetDeviceBuffer();
    args.lse_ptr              = lse ? lse_device.GetDeviceBuffer() : nullptr;
    args.seqstart_q_ptr       = q_starts_device.GetDeviceBuffer();
    args.seqstart_k_ptr       = k_starts_device.GetDeviceBuffer();
    args.batch                = batch;
    args.nhead_q              = c.query_heads;
    args.nhead_k              = c.kv_heads;
    args.hdim_q = args.hdim_v = c.dimension;
    args.seqlen_q             = total_q;
    args.seqlen_k             = total_k;
    args.max_seqlen_q         = max_seqlen_q;
    args.stride_q = args.stride_o = static_cast<ck_tile::index_t>(q_row);
    args.stride_k = args.stride_v = static_cast<ck_tile::index_t>(k_row);
    args.nhead_stride_q = args.nhead_stride_k = args.nhead_stride_v = args.nhead_stride_o =
        c.dimension;
    args.batch_stride_q = args.batch_stride_o = static_cast<ck_tile::index_t>(q.size());
    args.batch_stride_k = args.batch_stride_v = static_cast<ck_tile::index_t>(k.size());
    args.nhead_stride_lse                     = total_q;
    args.batch_stride_lse                     = static_cast<ck_tile::index_t>(lse_data.size());
    args.scale_s                              = 1.0f / std::sqrt(float(c.dimension));
    args.mask_type                            = static_cast<int>(c.mask);
    args.window_size_left                     = c.window_size_left;
    args.window_size_right                    = c.window_size_right;
    fmha_fwd_traits traits{};
    traits.hdim_q = traits.hdim_v = c.dimension;
    traits.data_type              = kDataType;
    traits.is_group_mode          = true;
    traits.is_v_rowmajor          = true;
    traits.mask_type              = c.mask;
    traits.bias_type              = bias_enum::no_bias;
    traits.has_lse                = lse;
    traits.qscale_type            = quant_scale_enum::no_scale;

    const float result = fmha_fwd(traits, args, {nullptr, false, 0, 0, 1});
    ASSERT_GE(result, 0) << "Generated generic M64 dispatcher has no supported instance";
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(selected_kernel, ExpectedKernel(c, lse));
    o_device.FromDevice(output.data());
    if(lse)
        lse_device.FromDevice(lse_data.data());

    const auto qf = Decode(q), kf = Decode(k), vf = Decode(v), of = Decode(output);
    std::vector<reference::Sequence> sequences;
    std::vector<reference::Output> outputs;
    std::size_t logical_rows = 0;
    for(int s = 0; s < batch; ++s)
    {
        const std::size_t q_offset  = std::size_t(q_starts[s]) * q_row;
        const std::size_t k_offset  = std::size_t(k_starts[s]) * k_row;
        const std::size_t dimension = c.dimension;
        reference::Sequence input;
        input.q               = {qf.data(), qf.size(), q_offset, dimension, q_row, 1};
        input.k               = {kf.data(), kf.size(), k_offset, dimension, k_row, 1};
        input.v               = {vf.data(), vf.size(), k_offset, dimension, k_row, 1};
        input.query_length    = c.q_lengths[s];
        input.key_length      = c.k_lengths[s];
        input.query_heads     = c.query_heads;
        input.kv_heads        = c.kv_heads;
        input.query_dimension = input.value_dimension = dimension;
        input.score_scale                             = args.scale_s;
        // Signed intervals, independent of CK's mask implementation; every mask type here
        // except mask_top_left aligns the diagonal to the bottom right.
        const int64_t alignment =
            c.mask == mask_enum::mask_top_left ? 0 : int64_t(c.k_lengths[s]) - c.q_lengths[s];
        input.is_masked = [alignment, left = c.window_size_left, right = c.window_size_right](
                              std::size_t query, std::size_t key) {
            const auto center = static_cast<int64_t>(query) + alignment;
            const auto column = static_cast<int64_t>(key);
            return (left >= 0 && column < center - left) || (right >= 0 && column > center + right);
        };
        logical_rows += std::size_t(c.q_lengths[s]) * c.query_heads;
        sequences.push_back(std::move(input));
        reference::Output actual{{of.data(), of.size(), q_offset, dimension, q_row, 1},
                                 std::nullopt};
        if(lse)
            actual.lse = reference::FloatView{lse_data.data(),
                                              lse_data.size(),
                                              std::size_t(q_starts[s]),
                                              std::size_t(total_q),
                                              1,
                                              1};
        outputs.push_back(actual);
    }
    const auto expected = reference::Compute(sequences);
    const reference::AccuracyContract contract{std::is_same_v<Data, ck_tile::half_t> ? 10 : 7,
                                               kTileN};
    const auto metrics        = reference::Compare(sequences, expected, outputs, {}, contract);
    std::size_t written_spare = 0;
    for(std::size_t i = std::size_t(total_q) * q_row; i < of.size(); ++i)
        written_spare += !std::isnan(of[i]);
    EXPECT_TRUE(metrics.passed) << "maxabs=" << metrics.max_abs_error
                                << " max_component_ratio=" << metrics.max_component_ratio
                                << " lse_maxabs=" << metrics.max_lse_abs_error;
    EXPECT_EQ(metrics.checked_rows, logical_rows);
    EXPECT_EQ(metrics.checked_elements, logical_rows * c.dimension);
    EXPECT_EQ(metrics.checked_lse_rows, lse ? logical_rows : 0);
    EXPECT_EQ(written_spare, 0u) << "O stores past the last sequence";
    std::cout << "generic_m64_dispatch case=" << c.name << " dtype=" << kDataType << " lse=" << lse
              << " m128_blocks=" << m128_blocks << " cus=" << properties.multiProcessorCount
              << " actual_kernel=" << selected_kernel << " checked_rows=" << metrics.checked_rows
              << " checked_elements=" << metrics.checked_elements
              << " checked_lse_rows=" << metrics.checked_lse_rows
              << " maxabs=" << metrics.max_abs_error << " lse_maxabs=" << metrics.max_lse_abs_error
              << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    GroupSmallGrid,
    QrTdmSchedGenericM64Forward,
    ::testing::Combine(::testing::Range<std::size_t>(0, kCases.size()), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<std::size_t, bool>>& parameter) {
        return std::string(kCases[std::get<0>(parameter.param)].name) +
               (std::get<1>(parameter.param) ? "Lse" : "NoLse");
    });

} // namespace
