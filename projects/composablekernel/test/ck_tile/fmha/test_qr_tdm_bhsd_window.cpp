// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_test_common.hpp"
#include "fmha_reference.hpp"
#include "ck_tile/host/device_memory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <limits>
#include <tuple>
#include <vector>

namespace {

#ifdef CK_TILE_FMHA_BHSD_WINDOW_FP16
using Data = ck_tile::half_t;
#else
using Data = ck_tile::bf16_t;
#endif

namespace reference = ck_tile::test::reference;

template <bool Scheduled, typename Problem>
struct SelectPipeline;

template <typename Problem>
struct SelectPipeline<false, Problem>
{
    using type = ck_tile::BlockFmhaPipelineQRKSVSTdm<Problem>;
};

template <typename Problem>
struct SelectPipeline<true, Problem>
{
    using type = ck_tile::BlockFmhaPipelineQRKSVSTdmSched<Problem>;
};

template <int D, int V, int N, bool Scheduled, bool Lse>
struct WindowKernel
{
    using Shape                    = ck_tile::TileFmhaShape<ck_tile::sequence<128, N, 32, V, 32, D>,
                                                            ck_tile::sequence<4, 1, 1>,
                                                            ck_tile::sequence<16, 16, 32>,
                                                            ck_tile::sequence<4, 1, 1>,
                                                            ck_tile::sequence<16, 16, 32>,
                                                            true>;
    static constexpr bool kPadHead = D == 192 || (Scheduled && D == 128);
    using Traits                   = ck_tile::TileFmhaTraits<true,
                                                             true,
                                                             kPadHead,
                                                             kPadHead,
                                                             false,
                                                             ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                             false,
                                                             Lse,
                                                             false,
                                                             ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                           Scheduled && D == 64 ? 3 : -1,
                                                             false,
                                                             false>;
    using Problem =
        ck_tile::BlockFmhaPipelineProblem<Data,
                                          Data,
                                          Data,
                                          float,
                                          float,
                                          Data,
                                          std::uint8_t,
                                          float,
                                          Data,
                                          float,
                                          Data,
                                          Shape,
                                          false,
                                          ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                          ck_tile::SimplifiedGenericAttentionMask<true>,
                                          false,
                                          Traits>;
    using Pipeline = typename SelectPipeline<Scheduled, Problem>::type;
    using Epilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, Data, true, true>>;
    using type = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
};

constexpr std::array<const char*, 9> kVariants{"QrD32",
                                               "QrD64",
                                               "QrD96",
                                               "QrD128",
                                               "QrD160",
                                               "QrD192",
                                               "SchedD64N64",
                                               "SchedD64N128",
                                               "SchedD128"};
constexpr std::array<int, 3> kLengths{384, 512, 768};
constexpr std::array<int, 3> kLeftWindows{31, 127, 255};

template <int D, int V, int N, bool Scheduled, bool Lse>
void CheckWindow(int length, int left, int seed)
{
    using Kernel         = typename WindowKernel<D, V, N, Scheduled, Lse>::type;
    constexpr int kBatch = 2, kHeads = 4, kKvHeads = 2;
    const auto make_input = [&](int heads, int dim, int phase) {
        std::vector<Data> data(std::size_t(kBatch) * heads * length * dim);
        for(int b = 0; b < kBatch; ++b)
            for(int h = 0; h < heads; ++h)
                for(int s = 0; s < length; ++s)
                    for(int d = 0; d < dim; ++d)
                    {
                        const float x = 0.31f * std::sin((s + 1) * (0.061f + phase * 0.017f) +
                                                         (d + 1) * 0.137f + h * 0.53f + b * 0.29f +
                                                         seed * 0.11f) +
                                        0.13f * std::cos((s + 1) * (d % 7 + 1) * 0.031f +
                                                         phase * 0.4f + seed * 0.23f);
                        data[((std::size_t(b) * heads + h) * length + s) * dim + d] =
                            ck_tile::type_convert<Data>(x);
                    }
        return data;
    };
    const auto q    = make_input(kHeads, D, 0);
    const auto k    = make_input(kKvHeads, D, 1);
    const auto v    = make_input(kKvHeads, V, 2);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<Data> output(std::size_t(kBatch) * kHeads * length * V,
                             ck_tile::type_convert<Data>(nan));
    std::vector<float> lse(std::size_t(kBatch) * kHeads * length, nan);
    ck_tile::DeviceMem q_device(q.size() * sizeof(Data)), k_device(k.size() * sizeof(Data)),
        v_device(v.size() * sizeof(Data)), o_device(output.size() * sizeof(Data)),
        lse_device(lse.size() * sizeof(float));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    o_device.ToDevice(output.data());
    lse_device.ToDevice(lse.data());
    fmha_fwd_args args{};
    args.q_ptr    = q_device.GetDeviceBuffer();
    args.k_ptr    = k_device.GetDeviceBuffer();
    args.v_ptr    = v_device.GetDeviceBuffer();
    args.o_ptr    = o_device.GetDeviceBuffer();
    args.lse_ptr  = Lse ? lse_device.GetDeviceBuffer() : nullptr;
    args.batch    = kBatch;
    args.nhead_q  = kHeads;
    args.nhead_k  = kKvHeads;
    args.hdim_q   = D;
    args.hdim_v   = V;
    args.seqlen_q = args.seqlen_k = args.max_seqlen_q = length;
    args.stride_q = args.stride_k = D;
    args.stride_v = args.stride_o = V;
    args.nhead_stride_q = args.nhead_stride_k = length * D;
    args.nhead_stride_v = args.nhead_stride_o = length * V;
    args.batch_stride_q                       = kHeads * length * D;
    args.batch_stride_k                       = kKvHeads * length * D;
    args.batch_stride_v                       = kKvHeads * length * V;
    args.batch_stride_o                       = kHeads * length * V;
    args.nhead_stride_lse                     = length;
    args.batch_stride_lse                     = kHeads * length;
    args.scale_s                              = 1.0f / std::sqrt(float(D));
    args.window_size_left                     = left;
    args.window_size_right                    = 0;
    args.mask_type                            = static_cast<int>(mask_enum::mask_top_left);
    const auto [kargs, grid]                  = fmha_fwd_create_kargs_and_grids<Kernel>(args);
    ASSERT_TRUE(Kernel::IsSupportedArgument(kargs));
    ASSERT_EQ(kargs.seqlen_q, kargs.seqlen_k);
    ASSERT_EQ(kargs.stride_q, kargs.hdim_q);
    ASSERT_GT(kargs.nhead_stride_q, kargs.hdim_q);
    ASSERT_EQ(kargs.window_size_right, 0);
    ASSERT_GE(kargs.window_size_left, 0);
    ASSERT_NE(kargs.mask_type, ck_tile::GenericAttentionMaskEnum::MASK_GENERIC);
    const auto launch = [&](const ck_tile::stream_config& stream) {
        return ck_tile::launch_kernel(stream,
                                      ck_tile::make_kernel<Kernel::kBlockPerCu, ck_tile::gfx125_t>(
                                          Kernel{}, grid, Kernel::BlockSize(), 0, kargs));
    };
    launch({nullptr, false, 0, 0, 1});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    o_device.FromDevice(output.data());
    if constexpr(Lse)
        lse_device.FromDevice(lse.data());
    const auto decode = [](const std::vector<Data>& data) {
        std::vector<float> values(data.size());
        std::transform(data.begin(), data.end(), values.begin(), [](Data x) {
            return ck_tile::type_convert<float>(x);
        });
        return values;
    };
    const auto qf = decode(q), kf = decode(k), vf = decode(v), of = decode(output);
    std::vector<reference::Sequence> sequences;
    std::vector<reference::Output> outputs;
    for(int b = 0; b < kBatch; ++b)
    {
        reference::Sequence sequence;
        sequence.q            = {qf.data(),
                                 qf.size(),
                                 std::size_t(b) * kHeads * length * D,
                                 std::size_t(length) * D,
                                 D,
                                 1};
        sequence.k            = {kf.data(),
                                 kf.size(),
                                 std::size_t(b) * kKvHeads * length * D,
                                 std::size_t(length) * D,
                                 D,
                                 1};
        sequence.v            = {vf.data(),
                                 vf.size(),
                                 std::size_t(b) * kKvHeads * length * V,
                                 std::size_t(length) * V,
                                 V,
                                 1};
        sequence.query_length = sequence.key_length = length;
        sequence.query_heads                        = kHeads;
        sequence.kv_heads                           = kKvHeads;
        sequence.query_dimension                    = D;
        sequence.value_dimension                    = V;
        sequence.score_scale                        = args.scale_s;
        sequence.is_masked                          = [left](std::size_t row, std::size_t key) {
            return key > row || std::int64_t(key) < std::int64_t(row) - left;
        };
        reference::Output actual{{of.data(),
                                  of.size(),
                                  std::size_t(b) * kHeads * length * V,
                                  std::size_t(length) * V,
                                  V,
                                  1},
                                 std::nullopt};
        if constexpr(Lse)
            actual.lse = reference::FloatView{lse.data(),
                                              lse.size(),
                                              std::size_t(b) * kHeads * length,
                                              std::size_t(length),
                                              1,
                                              1};
        sequences.push_back(sequence);
        outputs.push_back(actual);
    }
    const auto expected = reference::Compute(sequences);
    const reference::AccuracyContract contract{std::is_same_v<Data, ck_tile::half_t> ? 10 : 7, N};
    const auto metrics = reference::Compare(sequences, expected, outputs, {}, contract);
    EXPECT_TRUE(metrics.passed) << "maxabs=" << metrics.max_abs_error
                                << " max_component_ratio=" << metrics.max_component_ratio
                                << " lse_maxabs=" << metrics.max_lse_abs_error;
    EXPECT_EQ(metrics.checked_rows, std::size_t(kBatch) * kHeads * length);
    EXPECT_EQ(metrics.checked_elements, output.size());
    EXPECT_EQ(metrics.checked_lse_rows, Lse ? lse.size() : 0);
    if(std::getenv("CK_TILE_FMHA_BHSD_WINDOW_TIMING") != nullptr)
    {
        const float ms = launch({nullptr, true, 0, 3, 100});
        std::cout << std::setprecision(10)
                  << "BHSD_WINDOW_TIMING pipeline=" << (Scheduled ? "qr_tdm_sched" : "qr_tdm")
                  << " d=" << D << " n=" << N << " s=" << length << " left=" << left
                  << " lse=" << Lse << " seed=" << seed << " ms=" << ms << '\n';
    }
}

template <bool Lse>
void SelectWindow(int variant, int length, int left, int seed)
{
    switch(variant)
    {
    case 0: CheckWindow<32, 32, 64, false, Lse>(length, left, seed); break;
    case 1: CheckWindow<64, 64, 64, false, Lse>(length, left, seed); break;
    case 2: CheckWindow<96, 96, 64, false, Lse>(length, left, seed); break;
    case 3: CheckWindow<128, 128, 64, false, Lse>(length, left, seed); break;
    case 4: CheckWindow<160, 160, 64, false, Lse>(length, left, seed); break;
    case 5: CheckWindow<192, 128, 64, false, Lse>(length, left, seed); break;
    case 6: CheckWindow<64, 64, 64, true, Lse>(length, left, seed); break;
    case 7: CheckWindow<64, 64, 128, true, Lse>(length, left, seed); break;
    case 8: CheckWindow<128, 128, 128, true, Lse>(length, left, seed); break;
    }
}

class QrTdmBhsdWindow : public ck_tile::test::Gfx125FmhaTest,
                        public ::testing::WithParamInterface<std::tuple<int, int, bool, int>>
{
};

TEST_P(QrTdmBhsdWindow, SquareFiniteLeftFullReference)
{
    const auto [variant, shape, lse, seed] = GetParam();
    if(lse)
        SelectWindow<true>(variant, kLengths[shape], kLeftWindows[shape], seed);
    else
        SelectWindow<false>(variant, kLengths[shape], kLeftWindows[shape], seed);
}

std::string WindowCaseName(const ::testing::TestParamInfo<QrTdmBhsdWindow::ParamType>& case_info)
{
    const auto [variant, shape, lse, seed] = case_info.param;
    return std::string{kVariants[variant]} + "S" + std::to_string(kLengths[shape]) + "L" +
           std::to_string(kLeftWindows[shape]) + (lse ? "Lse" : "NoLse") + "Seed" +
           std::to_string(seed);
}

INSTANTIATE_TEST_SUITE_P(AllAffectedPipelines,
                         QrTdmBhsdWindow,
                         ::testing::Combine(::testing::Range(0, 9),
                                            ::testing::Range(0, 3),
                                            ::testing::Bool(),
                                            ::testing::Range(0, 2)),
                         WindowCaseName);
} // namespace
