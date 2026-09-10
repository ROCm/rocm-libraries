// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include "ck_tile/host.hpp"
#include "tdm_fmha_v128_test_common.hpp"

namespace {

using namespace ck_tile;
#if defined(CK_TILE_FMHA_TDM_V128_TEST_FP16)
using Data = half_t;
#else
using Data = bf16_t;
#endif

CK_TILE_HOST_DEVICE constexpr std::uint16_t Tag(int row, int col, int width)
{
    return static_cast<std::uint16_t>(0x1000 + row * width + col);
}

template <typename Tensor, typename Visitor>
CK_TILE_DEVICE void VisitElements(Tensor& tensor, const Visitor& visitor)
{
    constexpr auto distribution = typename Tensor::StaticTileDistribution{};
    constexpr auto lengths      = to_sequence(distribution.get_ys_to_d_descriptor().get_lengths());
    const auto partitions       = get_partition_index(distribution);
    static_ford<remove_cvref_t<decltype(lengths)>>{}([&](auto ys) {
        constexpr index_t offset = distribution.get_ys_to_d_descriptor().calculate_offset(ys);
        const auto coord         = make_tensor_adaptor_coordinate(
            distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partitions, to_array<index_t, ys.size()>(ys)));
        const auto xy = coord.get_bottom_index();
        visitor(tensor.get_thread_buffer().template at<offset>(), xy[number<0>{}], xy[number<1>{}]);
    });
}

struct Args
{
    const Data* k;
    const Data* v;
    std::uint8_t* arena_dump;
    std::uint32_t* errors;
    int rows;
};

template <typename A, typename B>
constexpr bool SameBlockEncoding()
{
    using AWarp = typename A::WarpGemm;
    using BWarp = typename B::WarpGemm;
    return std::is_same_v<typename AWarp::AWarpDstrEncoding, typename BWarp::AWarpDstrEncoding> &&
           std::is_same_v<typename AWarp::BWarpDstrEncoding, typename BWarp::BWarpDstrEncoding> &&
           std::is_same_v<typename AWarp::CWarpDstrEncoding, typename BWarp::CWarpDstrEncoding> &&
           std::is_same_v<remove_cvref_t<decltype(A::MakeABlockDistributionEncode())>,
                          remove_cvref_t<decltype(B::MakeABlockDistributionEncode())>> &&
           std::is_same_v<remove_cvref_t<decltype(A::MakeBBlockDistributionEncode())>,
                          remove_cvref_t<decltype(B::MakeBBlockDistributionEncode())>> &&
           std::is_same_v<remove_cvref_t<decltype(A::MakeCBlockDistributionEncode())>,
                          remove_cvref_t<decltype(B::MakeCBlockDistributionEncode())>>;
}

template <typename Problem, typename Q, typename K, typename V, typename P>
struct OperandProblem : Problem
{
    using QDataType = Q;
    using KDataType = K;
    using VDataType = V;
    using PDataType = P;
};

template <int Dim>
constexpr bool CheckDtypeEncoding()
{
    using B      = typename tdm_v128_test::Model<Dim>::Problem;
    using F      = typename tdm_v128_test::Model<Dim, false, false, false, half_t>::Problem;
    using BP     = tdm_v128_test::Policy<B, 0>;
    using FP     = tdm_v128_test::Policy<F, 0>;
    using BQK    = remove_cvref_t<decltype(BP::template GetQKBlockGemmSu<B>())>;
    using FQK    = remove_cvref_t<decltype(FP::template GetQKBlockGemmSu<F>())>;
    using BPV    = remove_cvref_t<decltype(BP::template GetPVBlockGemm<B>())>;
    using FPV    = remove_cvref_t<decltype(FP::template GetPVBlockGemm<F>())>;
    using BRead  = remove_cvref_t<decltype(BP::template MakeVRegTileDistribution<B>())>;
    using FRead  = remove_cvref_t<decltype(FP::template MakeVRegTileDistribution<F>())>;
    using BQRead = remove_cvref_t<decltype(BP::template MakeQRegTileDistribution<B>())>;
    using FQRead = remove_cvref_t<decltype(FP::template MakeQRegTileDistribution<F>())>;
    using BKRead = remove_cvref_t<decltype(BP::template MakeKSuRegTileDistribution<B>())>;
    using FKRead = remove_cvref_t<decltype(FP::template MakeKSuRegTileDistribution<F>())>;
    return BP::template IsSupportedProblem<B>() && FP::template IsSupportedProblem<F>() &&
           !BP::template IsSupportedProblem<F>() && !FP::template IsSupportedProblem<B>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, bf16_t, half_t, half_t, half_t>>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, half_t, bf16_t, half_t, half_t>>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, half_t, half_t, bf16_t, half_t>>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, half_t, half_t, half_t, bf16_t>>() &&
           SameBlockEncoding<BQK, FQK>() && SameBlockEncoding<BPV, FPV>() &&
           std::is_same_v<typename BRead::DstrEncode, typename FRead::DstrEncode> &&
           std::is_same_v<typename BQRead::DstrEncode, typename FQRead::DstrEncode> &&
           std::is_same_v<typename BKRead::DstrEncode, typename FKRead::DstrEncode> &&
           std::is_same_v<typename BP::OutputFragments, typename FP::OutputFragments> &&
           BP::kKFootprintBytes == FP::kKFootprintBytes &&
           BP::kVFootprintBytes == FP::kVFootprintBytes && BP::kLdsArenaSize == FP::kLdsArenaSize;
}
static_assert(CheckDtypeEncoding<128>() && CheckDtypeEncoding<192>());

template <int Dim, bool Pack>
struct LayoutProbe
{
    using Problem  = typename tdm_v128_test::Model<Dim, false, false, false, Data>::Problem;
    using Policy   = tdm_v128_test::Policy<Problem, 0, Pack>;
    using Pipeline = BlockFmhaPipelineQRKSVSTdmV128<Problem, Policy>;
    static constexpr index_t kBlockSize = 128;

    CK_TILE_DEVICE void operator()(Args args) const
    {
        __shared__ char arena[Policy::kLdsArenaSize];
        for(index_t i = threadIdx.x; i < Policy::kLdsArenaSize; i += kBlockSize)
            arena[i] = static_cast<char>(0xa5);
        block_sync_lds<0>();
        std::uint32_t errors = 0;
        auto input_k         = make_naive_tensor_view<address_space_enum::global>(
            args.k, make_tuple(args.rows, Dim), make_tuple(Dim + 16, 1));
        auto input_v = make_naive_tensor_view<address_space_enum::global>(
            args.v, make_tuple(args.rows, 128), make_tuple(144, 1));

        // Q occupies K0 temporarily; every Q reader completes before K0 is overwritten.
        auto q_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<Data*>(arena), Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_write  = make_tile_window(q_view, make_tuple(number<128>{}, number<Dim>{}), {0, 0});
        auto q_global = make_tile_window(input_k,
                                         make_tuple(number<128>{}, number<Dim>{}),
                                         {0, 0},
                                         Policy::template MakeQDramTileDistribution<Problem>());
        load_tile_tdm(TDMConfig{}, q_write, q_global);
        s_wait_tensorcnt_barrier<0>();
        auto q   = load_tile(make_tile_window(q_view,
                                            make_tuple(number<128>{}, number<Dim>{}),
                                              {0, 0},
                                            Policy::template MakeQRegTileDistribution<Problem>()));
        using Qk = remove_cvref_t<decltype(Policy::template GetQKBlockGemmSu<Problem>())>;
        static_assert(std::is_same_v<typename decltype(q)::StaticTileDistribution::DstrEncode,
                                     remove_cvref_t<decltype(Qk::MakeABlockDistributionEncode())>>);
        VisitElements(q, [&](auto value, int row, int col) {
            errors += bit_cast<std::uint16_t>(value) != (row < args.rows ? Tag(row, col, Dim) : 0);
        });
        block_sync_lds<0>();

        static_for<0, 2, 1>{}([&](auto buffer) {
            constexpr int base = buffer == 0 ? Policy::kLdsOffsetK0 : Policy::kLdsOffsetK1;
            auto view          = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<Data*>(arena + base),
                Policy::template MakeKLdsWriteBlockDescriptor<Problem>());
            auto write = make_tile_window(
                view, make_tuple(number<128>{}, number<Policy::kKPhysicalStride>{}), {0, 0});
            auto global =
                make_tile_window(input_k,
                                 make_tuple(number<128>{}, number<Policy::kKPhysicalStride>{}),
                                 {0, 0},
                                 Policy::template MakeKDramTileDistribution<Problem>());
            load_tile_tdm(TDMConfig{}, write, global);
            s_wait_tensorcnt_barrier<0>();
            static_for<0, 4, 1>{}([&](auto su) {
                auto window =
                    make_tile_window(view,
                                     make_tuple(number<32>{}, number<Dim>{}),
                                     {su * 32, 0},
                                     Policy::template MakeKSuRegTileDistribution<Problem>());
                using Window = decltype(window);
                static_assert(Window::NumAccessPerCoord == Dim / 8);
                decltype(load_tile(window)) values;
                static_for<0, Window::NumAccessPerCoord, 1>{}([&](auto access) {
                    Policy::KLoad::template LoadInstruction<access>(values, window);
                });
                s_wait_dscnt<0>();
                static_assert(
                    std::is_same_v<typename decltype(values)::StaticTileDistribution::DstrEncode,
                                   remove_cvref_t<decltype(Qk::MakeBBlockDistributionEncode())>>);
                VisitElements(values, [&](auto value, int row, int col) {
                    row += su * 32;
                    errors += bit_cast<std::uint16_t>(value) !=
                              (row < args.rows ? Tag(row, col, Dim) : 0);
                });
            });
            block_sync_lds<0>();
        });

        using Pv = remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
        static_for<0, 2, 1>{}([&](auto buffer) {
            constexpr int base = buffer == 0 ? Policy::kLdsOffsetV0 : Policy::kLdsOffsetV1;
            auto view          = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<Data*>(arena + base),
                Policy::template MakeVLdsWriteBlockDescriptor<Problem>());
            auto write  = make_tile_window(view, make_tuple(number<128>{}, number<128>{}), {0, 0});
            auto global = make_tile_window(input_v,
                                           make_tuple(number<128>{}, number<128>{}),
                                           {0, 0},
                                           Policy::template MakeVDramTileDistribution<Problem>());
            TDMConfig config{};
            config.pad_enable              = true;
            config.pad_config.pad_interval = Policy::kVPadInterval;
            config.pad_config.pad_amount   = Policy::kVPadAmount;
            load_tile_tdm(config, write, global);
            s_wait_tensorcnt_barrier<0>();
            static_for<0, 4, 1>{}([&](auto stage) {
                auto window =
                    make_tile_window(view,
                                     make_tuple(number<32>{}, number<128>{}),
                                     {stage * 32, 0},
                                     Policy::template MakeVRegTileDistribution<Problem>());
                using Window = decltype(window);
                static_assert(Window::NumAccessPerCoord == 16);
                decltype(load_tile_transpose(window)) values;
                static_for<0, 16, 1>{}([&](auto access) {
                    Policy::VLoad::template LoadAccess<access>(values, window);
                });
                s_wait_dscnt<0>();
                static_assert(
                    std::is_same_v<typename decltype(values)::StaticTileDistribution::DstrEncode,
                                   remove_cvref_t<decltype(Pv::MakeBBlockDistributionEncode())>>);
                VisitElements(values, [&](auto value, int col, int row) {
                    row += stage * 32;
                    errors += bit_cast<std::uint16_t>(value) !=
                              (row < args.rows ? Tag(row, col, 128) : 0);
                });
            });
            block_sync_lds<0>();
        });

        using ScoreGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
        auto score      = ScoreGemm::MakeCBlockTile();
        VisitElements(score, [](auto& value, int row, int col) {
            value = type_convert<float>(bit_cast<Data>(Tag(row, col, 128)));
        });
        auto p = Pipeline::template MakePForGemm1<Pv>(score);
        static_for<0, 4, 1>{}([&](auto stage) {
            auto operand =
                get_slice_tile(p, sequence<0, stage * 32>{}, sequence<128, (stage + 1) * 32>{});
            static_assert(
                std::is_same_v<typename decltype(operand)::StaticTileDistribution::DstrEncode,
                               remove_cvref_t<decltype(Pv::MakeABlockDistributionEncode())>>);
            VisitElements(operand, [&](auto value, int row, int col) {
                errors += bit_cast<std::uint16_t>(value) != Tag(row, col + stage * 32, 128);
            });
        });
        args.errors[threadIdx.x] = errors;
        for(index_t i = threadIdx.x; i < Policy::kLdsArenaSize; i += kBlockSize)
            args.arena_dump[i] = static_cast<std::uint8_t>(arena[i]);
    }
};

template <int Dim, bool Pack>
bool Run(int rows)
{
    using Probe  = LayoutProbe<Dim, Pack>;
    using Policy = typename Probe::Policy;
    std::vector<std::uint16_t> k(128 * (Dim + 16), 0x7fff), v(128 * 144, 0x7fff);
    for(int row = 0; row < rows; ++row)
    {
        for(int col = 0; col < Dim; ++col)
            k[row * (Dim + 16) + col] = Tag(row, col, Dim);
        for(int col = 0; col < 128; ++col)
            v[row * 144 + col] = Tag(row, col, 128);
    }
    DeviceMem dk(k.size() * 2), dv(v.size() * 2), dump(Policy::kLdsArenaSize), counts(128 * 4);
    dk.ToDevice(k.data());
    dv.ToDevice(v.data());
    const Args args{static_cast<const Data*>(dk.GetDeviceBuffer()),
                    static_cast<const Data*>(dv.GetDeviceBuffer()),
                    static_cast<std::uint8_t*>(dump.GetDeviceBuffer()),
                    static_cast<std::uint32_t*>(counts.GetDeviceBuffer()),
                    rows};
    launch_kernel(stream_config{nullptr, false, 0},
                  make_kernel<1, gfx125_t>(Probe{}, dim3(1), dim3(128), 0, args));
    hip_check_error(hipDeviceSynchronize());
    std::array<std::uint32_t, 128> errors{};
    counts.FromDevice(errors.data());
    std::vector<std::uint8_t> actual(Policy::kLdsArenaSize), expected(actual.size(), 0xa5);
    dump.FromDevice(actual.data());
    for(int region = 0; region < 4; ++region)
    {
        const bool is_k  = region < 2;
        const int base   = std::array<int, 4>{Policy::kLdsOffsetK0,
                                              Policy::kLdsOffsetK1,
                                              Policy::kLdsOffsetV0,
                                              Policy::kLdsOffsetV1}[region];
        const int stride = is_k ? Dim + 8 : 144;
        const int width  = is_k ? Dim : 128;
        for(int row = 0; row < 128; ++row)
            for(int col = 0; col < (is_k ? stride : width); ++col)
            {
                const std::uint16_t bits = row < rows && col < width ? Tag(row, col, width) : 0;
                const int offset         = base + 2 * (row * stride + col);
                expected[offset]         = static_cast<std::uint8_t>(bits);
                expected[offset + 1]     = static_cast<std::uint8_t>(bits >> 8);
            }
    }
    std::uint64_t failures = 0;
    for(auto count : errors)
        failures += count;
    for(std::size_t i = 0; i < actual.size(); ++i)
        failures += actual[i] != expected[i];
    std::cout << (std::is_same_v<Data, half_t> ? "fp16 " : "bf16 ") << "D" << Dim
              << " pack=" << Pack << " rows=" << rows << " arena=" << actual.size()
              << " Q/K/V/P tags + arena mismatches=" << failures << '\n';
    return failures == 0;
}

} // namespace

int main()
{
    int device = 0;
    hipDeviceProp_t properties{};
    hip_check_error(hipGetDevice(&device));
    hip_check_error(hipGetDeviceProperties(&properties, device));
    if(!std::string_view{properties.gcnArchName}.starts_with("gfx1250"))
        return 77;
    for(int rows : {128, 127})
        if(!Run<128, false>(rows) || !Run<128, true>(rows) || !Run<192, false>(rows) ||
           !Run<192, true>(rows))
            return 1;
    return 0;
}
