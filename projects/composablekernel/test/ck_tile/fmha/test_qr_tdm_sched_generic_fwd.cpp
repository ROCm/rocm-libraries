// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/device_memory.hpp"
#include "fmha_reference.hpp"
#include "fmha_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
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
constexpr int kLength = 4096, kDimension = 64, kTileM = 128, kTileN = 128;
constexpr std::size_t kElements = std::size_t(kLength) * kDimension;
namespace reference             = ck_tile::test::reference;

std::vector<Data> MakeInput(int phase)
{
    std::vector<Data> data(kElements);
    for(int row = 0; row < kLength; ++row)
        for(int dim = 0; dim < kDimension; ++dim)
        {
            const float value =
                0.35f * std::sin((row + 1) * (0.061f + phase * 0.017f) + (dim + 1) * 0.137f) +
                0.11f * std::cos((row + 1) * (dim % 7 + 1) * 0.031f + phase * 0.4f);
            data[std::size_t(row) * kDimension + dim] = ck_tile::type_convert<Data>(value);
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

class QrTdmSchedGenericForward : public ck_tile::test::Gfx125FmhaTest,
                                 public ::testing::WithParamInterface<bool>
{
};

TEST_P(QrTdmSchedGenericForward, GeneratedCausalD64N128FullReference)
{
    const bool lse = GetParam();
    int device     = 0;
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
    // Exercise the actual generic mc N128 crossover, not a forced direct launch.
    ASSERT_LE(kLength / kTileM, 8LL * properties.multiProcessorCount);

    const auto q = MakeInput(0), k = MakeInput(1), v = MakeInput(2);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<Data> output(kElements, ck_tile::type_convert<Data>(nan));
    std::vector<float> lse_data(kLength, nan);
    ck_tile::DeviceMem q_device(q.size() * sizeof(Data)), k_device(k.size() * sizeof(Data)),
        v_device(v.size() * sizeof(Data)), o_device(output.size() * sizeof(Data)),
        lse_device(lse_data.size() * sizeof(float));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    o_device.ToDevice(output.data());
    lse_device.ToDevice(lse_data.data());

    std::string selected_kernel;
    fmha_fwd_args args{};
    args.selected_kernel_name = &selected_kernel;
    args.q_ptr                = q_device.GetDeviceBuffer();
    args.k_ptr                = k_device.GetDeviceBuffer();
    args.v_ptr                = v_device.GetDeviceBuffer();
    args.o_ptr                = o_device.GetDeviceBuffer();
    args.lse_ptr              = lse ? lse_device.GetDeviceBuffer() : nullptr;
    args.batch = args.nhead_q = args.nhead_k = 1;
    args.hdim_q = args.hdim_v = kDimension;
    args.seqlen_q = args.seqlen_k = args.max_seqlen_q = kLength;
    args.stride_q = args.stride_k = args.stride_v = args.stride_o = kDimension;
    args.nhead_stride_q = args.nhead_stride_k = args.nhead_stride_v = args.nhead_stride_o =
        kElements;
    args.batch_stride_q = args.batch_stride_k = args.batch_stride_v = args.batch_stride_o =
        kElements;
    args.nhead_stride_lse = args.batch_stride_lse = kLength;
    args.scale_s                                  = 1.0f / std::sqrt(float(kDimension));
    args.mask_type                                = static_cast<int>(mask_enum::mask_top_left);
    args.window_size_left                         = -1;
    args.window_size_right                        = 0;
    fmha_fwd_traits traits{};
    traits.hdim_q = traits.hdim_v = kDimension;
    traits.data_type              = kDataType;
    traits.is_group_mode          = false;
    traits.is_v_rowmajor          = true;
    traits.mask_type              = mask_enum::mask_top_left;
    traits.bias_type              = bias_enum::no_bias;
    traits.has_lse                = lse;
    traits.qscale_type            = quant_scale_enum::no_scale;

    const float result = fmha_fwd(traits, args, {nullptr, false, 0, 0, 1});
    ASSERT_GE(result, 0) << "Generated generic mc dispatcher has no supported instance";
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    const std::string expected_kernel =
        std::string("fmha_fwd_d64_") + kDataType +
        "_batch_b128x128x32x64x32x64_r4x1x1_r4x1x1_w16x16x32_w16x16x32_o3_qr_tdm_sched_vr"
        "_npad_nlogits_nbias_mc_" +
        (lse ? "lse" : "nlse") + "_ndropout_nskip_nqscale_ntrload_nsink";
    ASSERT_EQ(selected_kernel, expected_kernel);
    o_device.FromDevice(output.data());
    if(lse)
        lse_device.FromDevice(lse_data.data());

    const auto qf = Decode(q), kf = Decode(k), vf = Decode(v), of = Decode(output);
    reference::Sequence sequence;
    sequence.q            = {qf.data(), qf.size(), 0, kElements, kDimension, 1};
    sequence.k            = {kf.data(), kf.size(), 0, kElements, kDimension, 1};
    sequence.v            = {vf.data(), vf.size(), 0, kElements, kDimension, 1};
    sequence.query_length = sequence.key_length = kLength;
    sequence.query_heads = sequence.kv_heads = 1;
    sequence.query_dimension = sequence.value_dimension = kDimension;
    sequence.score_scale                                = args.scale_s;
    sequence.is_masked = [](std::size_t row, std::size_t key) { return key > row; };
    reference::Output actual{{of.data(), of.size(), 0, kElements, kDimension, 1}, std::nullopt};
    if(lse)
        actual.lse = reference::FloatView{lse_data.data(), lse_data.size(), 0, kLength, 1, 1};
    const std::vector<reference::Sequence> sequences{sequence};
    const auto expected = reference::Compute(sequences);
    const reference::AccuracyContract contract{std::is_same_v<Data, ck_tile::half_t> ? 10 : 7,
                                               kTileN};
    const auto metrics = reference::Compare(sequences, expected, {actual}, {}, contract);
    EXPECT_TRUE(metrics.passed) << "maxabs=" << metrics.max_abs_error
                                << " max_component_ratio=" << metrics.max_component_ratio
                                << " lse_maxabs=" << metrics.max_lse_abs_error;
    EXPECT_EQ(metrics.checked_rows, std::size_t(kLength));
    EXPECT_EQ(metrics.checked_elements, kElements);
    EXPECT_EQ(metrics.checked_lse_rows, lse ? std::size_t(kLength) : 0);
    std::cout << "generic_generated_dispatch dtype=" << kDataType << " lse=" << lse
              << " query_wgs=" << kLength / kTileM << " cus=" << properties.multiProcessorCount
              << " actual_kernel=" << selected_kernel << " checked_rows=" << metrics.checked_rows
              << " checked_elements=" << metrics.checked_elements
              << " checked_lse_rows=" << metrics.checked_lse_rows
              << " maxabs=" << metrics.max_abs_error << " lse_maxabs=" << metrics.max_lse_abs_error
              << '\n';
}

INSTANTIATE_TEST_SUITE_P(FullCausal,
                         QrTdmSchedGenericForward,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& parameter) {
                             return parameter.param ? "Lse" : "NoLse";
                         });

} // namespace
