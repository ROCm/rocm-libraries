// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_deferred_p.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"

namespace {

using DataType = ck_tile::bf16_t;
using Layout   = ck_tile::FmhaD192DeferredPLdsLayout;

using D192BlockShape = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, 192>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true>;

struct D192Problem
{
    struct AttentionVariant
    {
    };
    struct FmhaMask
    {
        static constexpr bool IsMasking = false;
    };

    using QDataType           = DataType;
    using KDataType           = DataType;
    using VDataType           = DataType;
    using SaccDataType        = float;
    using SMPLComputeDataType = float;
    using PDataType           = DataType;
    using OaccDataType        = float;
    using BlockFmhaShape      = D192BlockShape;

    static constexpr ck_tile::index_t kBlockSize = 128;
    static constexpr bool kHasLogitsSoftCap      = false;
    static constexpr auto BiasEnum               = ck_tile::BlockAttentionBiasEnum::NO_BIAS;
};

using Policy = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy;

static_assert(Policy::IsSupportedProblem<D192Problem>());
static_assert(Layout::kArenaBytes == Policy::kLdsArenaSize);
static_assert(Layout::kValuesPerThread == 32);
static_assert(Layout::kValuesPerMHalf == 16);
static_assert(Layout::GetSlotBase<0>() == Policy::kLdsOffsetK0 + Policy::kKFootprintBytes);
static_assert(Layout::GetSlotBase<0>() + Layout::kSlotBytes == Policy::kLdsOffsetK1);
static_assert(Layout::GetSlotBase<1>() == Policy::kLdsOffsetK1 + Policy::kKFootprintBytes);
static_assert(Layout::GetSlotBase<1>() + Layout::kSlotBytes == Policy::kLdsOffsetV0);
static_assert(Layout::GetSlotBase<2>() == Policy::kLdsOffsetV0 + Policy::kVFootprintBytes);
static_assert(Layout::GetSlotBase<2>() + Layout::kSlotBytes == Layout::GetSlotBase<3>());
static_assert(Layout::GetSlotBase<3>() + Layout::kSlotBytes == Policy::kLdsOffsetV1);

constexpr std::uint8_t kArenaCanary                         = 0xa5;
constexpr std::array<ck_tile::index_t, 4> kCandidateStrides = {32, 40, 48, 56};

struct ProbeArgs
{
    void* arena_dump;
    std::uint32_t* mismatches;
};

struct BenchmarkArgs
{
    std::uint64_t* cycles;
    std::uint32_t* checksums;
    ck_tile::index_t repetitions;
};

CK_TILE_HOST_DEVICE constexpr std::uint16_t make_pattern(ck_tile::index_t lane,
                                                         ck_tile::index_t element)
{
    return static_cast<std::uint16_t>(1 + lane * 257 + element);
}

template <ck_tile::index_t RowStride>
struct DeferredPProbeKernel
{
    static constexpr ck_tile::index_t kBlockSize = 128;

    CK_TILE_DEVICE void operator()(ProbeArgs args) const
    {
        __shared__ char arena[Layout::kArenaBytes];
        auto* arena_bytes = reinterpret_cast<std::uint8_t*>(arena);

        for(ck_tile::index_t i = threadIdx.x; i < Layout::kArenaBytes; i += blockDim.x)
        {
            arena_bytes[i] = kArenaCanary;
        }
        __syncthreads();

        auto p_tile = ck_tile::make_static_distributed_tensor<DataType>(
            Policy::MakePRegTileDistribution<D192Problem>());
        constexpr ck_tile::index_t kThreadValues = decltype(p_tile)::get_thread_buffer_size();
        static_assert(kThreadValues == Layout::kNumGroups * Layout::kValuesPerThread);

        ck_tile::static_for<0, kThreadValues, 1>{}([&](auto i) {
            const auto bits               = make_pattern(threadIdx.x, decltype(i)::value);
            p_tile.get_thread_buffer()[i] = ck_tile::bit_cast<DataType>(bits);
        });

        ck_tile::static_for<0, Layout::kNumGroups, 1>{}([&](auto group) {
            constexpr ck_tile::index_t g = decltype(group)::value;
            auto p_group                 = ck_tile::get_slice_tile(
                p_tile,
                ck_tile::sequence<0, g * Layout::kColumnsPerGroup>{},
                ck_tile::sequence<Layout::kRows, (g + 1) * Layout::kColumnsPerGroup>{});
            static_assert(decltype(p_group)::get_thread_buffer_size() == Layout::kValuesPerThread);

            auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                reinterpret_cast<DataType*>(arena + Layout::GetSlotBase<g>()),
                Layout::MakeGroupLdsDescriptor<RowStride>());
            auto lds_window = ck_tile::make_tile_window(
                lds_view,
                ck_tile::make_tuple(ck_tile::number<Layout::kRows>{},
                                    ck_tile::number<Layout::kColumnsPerGroup>{}),
                {0, 0},
                p_group.get_tile_distribution());

            ck_tile::static_for<0, Layout::kMHalves, 1>{}([&](auto m_half) {
                Layout::MHalf values{};
                constexpr ck_tile::index_t begin =
                    decltype(m_half)::value * Layout::kValuesPerMHalf;
                ck_tile::static_for<0, Layout::kValuesPerMHalf, 1>{}([&](auto i) {
                    values[decltype(i)::value] =
                        p_group.get_thread_buffer()[begin + decltype(i)::value];
                });
                Layout::StoreMHalf<decltype(m_half)::value>(lds_window, values);
            });
        });

        ck_tile::block_sync_lds();

        std::uint32_t lane_mismatches = 0;
        ck_tile::static_for<0, Layout::kNumGroups, 1>{}([&](auto group) {
            constexpr ck_tile::index_t g = decltype(group)::value;
            auto expected                = ck_tile::get_slice_tile(
                p_tile,
                ck_tile::sequence<0, g * Layout::kColumnsPerGroup>{},
                ck_tile::sequence<Layout::kRows, (g + 1) * Layout::kColumnsPerGroup>{});

            auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                reinterpret_cast<DataType*>(arena + Layout::GetSlotBase<g>()),
                Layout::MakeGroupLdsDescriptor<RowStride>());
            auto lds_window = ck_tile::make_tile_window(
                lds_view,
                ck_tile::make_tuple(ck_tile::number<Layout::kRows>{},
                                    ck_tile::number<Layout::kColumnsPerGroup>{}),
                {0, 0},
                expected.get_tile_distribution());
            ck_tile::static_for<0, Layout::kMHalves, 1>{}([&](auto m_half) {
                constexpr ck_tile::index_t begin =
                    decltype(m_half)::value * Layout::kValuesPerMHalf;
                const auto loaded = Layout::LoadMHalf<decltype(m_half)::value>(lds_window);
                ck_tile::static_for<0, Layout::kValuesPerMHalf, 1>{}([&](auto i) {
                    constexpr ck_tile::index_t element = begin + decltype(i)::value;
                    const auto actual_bits =
                        ck_tile::bit_cast<std::uint16_t>(loaded[decltype(i)::value]);
                    const auto expected_bits =
                        ck_tile::bit_cast<std::uint16_t>(expected.get_thread_buffer()[element]);
                    lane_mismatches += actual_bits != expected_bits;
                });
            });
        });

        if(lane_mismatches != 0)
        {
            atomicAdd(args.mismatches, lane_mismatches);
        }

        __syncthreads();
        auto* dump = static_cast<std::uint8_t*>(args.arena_dump);
        for(ck_tile::index_t i = threadIdx.x; i < Layout::kArenaBytes; i += blockDim.x)
        {
            dump[i] = arena_bytes[i];
        }
    }
};

template <ck_tile::index_t RowStride>
struct DeferredPBenchmarkKernel
{
    static constexpr ck_tile::index_t kBlockSize = 128;

    CK_TILE_DEVICE void operator()(BenchmarkArgs args) const
    {
        __shared__ char arena[Layout::kArenaBytes];
        auto p_tile = ck_tile::make_static_distributed_tensor<DataType>(
            Policy::MakePRegTileDistribution<D192Problem>());
        constexpr ck_tile::index_t kThreadValues = decltype(p_tile)::get_thread_buffer_size();

        ck_tile::static_for<0, kThreadValues, 1>{}([&](auto i) {
            const auto bits               = make_pattern(threadIdx.x, decltype(i)::value);
            p_tile.get_thread_buffer()[i] = ck_tile::bit_cast<DataType>(bits);
        });

        ck_tile::block_sync_lds();
        const std::uint64_t begin = clock64();
        std::uint32_t checksum    = 0;

        for(ck_tile::index_t repeat = 0; repeat < args.repetitions; ++repeat)
        {
            ck_tile::static_for<0, Layout::kNumGroups, 1>{}([&](auto group) {
                constexpr ck_tile::index_t g = decltype(group)::value;
                auto p_group                 = ck_tile::get_slice_tile(
                    p_tile,
                    ck_tile::sequence<0, g * Layout::kColumnsPerGroup>{},
                    ck_tile::sequence<Layout::kRows, (g + 1) * Layout::kColumnsPerGroup>{});
                auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    reinterpret_cast<DataType*>(arena + Layout::GetSlotBase<g>()),
                    Layout::MakeGroupLdsDescriptor<RowStride>());
                auto lds_window = ck_tile::make_tile_window(
                    lds_view,
                    ck_tile::make_tuple(ck_tile::number<Layout::kRows>{},
                                        ck_tile::number<Layout::kColumnsPerGroup>{}),
                    {0, 0},
                    p_group.get_tile_distribution());
                ck_tile::static_for<0, Layout::kMHalves, 1>{}([&](auto m_half) {
                    Layout::MHalf values{};
                    constexpr ck_tile::index_t begin =
                        decltype(m_half)::value * Layout::kValuesPerMHalf;
                    ck_tile::static_for<0, Layout::kValuesPerMHalf, 1>{}([&](auto i) {
                        values[decltype(i)::value] =
                            p_group.get_thread_buffer()[begin + decltype(i)::value];
                    });
                    Layout::StoreMHalf<decltype(m_half)::value>(lds_window, values);
                });
            });

            ck_tile::block_sync_lds();

            ck_tile::static_for<0, Layout::kNumGroups, 1>{}([&](auto group) {
                constexpr ck_tile::index_t g = decltype(group)::value;
                auto p_group                 = ck_tile::get_slice_tile(
                    p_tile,
                    ck_tile::sequence<0, g * Layout::kColumnsPerGroup>{},
                    ck_tile::sequence<Layout::kRows, (g + 1) * Layout::kColumnsPerGroup>{});
                auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    reinterpret_cast<DataType*>(arena + Layout::GetSlotBase<g>()),
                    Layout::MakeGroupLdsDescriptor<RowStride>());
                auto lds_window = ck_tile::make_tile_window(
                    lds_view,
                    ck_tile::make_tuple(ck_tile::number<Layout::kRows>{},
                                        ck_tile::number<Layout::kColumnsPerGroup>{}),
                    {0, 0},
                    p_group.get_tile_distribution());
                ck_tile::static_for<0, Layout::kMHalves, 1>{}([&](auto m_half) {
                    const auto loaded = Layout::LoadMHalf<decltype(m_half)::value>(lds_window);
                    ck_tile::static_for<0, Layout::kValuesPerMHalf, 1>{}([&](auto i) {
                        checksum += ck_tile::bit_cast<std::uint16_t>(loaded[decltype(i)::value]);
                    });
                });
            });

            ck_tile::block_sync_lds();
        }

        const std::uint64_t end     = clock64();
        args.checksums[threadIdx.x] = checksum;
        if(threadIdx.x == 0)
        {
            *args.cycles = end - begin;
        }
    }
};

void check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

template <ck_tile::index_t RowStride>
bool run_probe()
{
    ck_tile::DeviceMem arena_dump(Layout::kArenaBytes);
    ck_tile::DeviceMem mismatch_buffer(sizeof(std::uint32_t));
    const std::uint32_t zero = 0;
    mismatch_buffer.ToDevice(&zero);

    const ProbeArgs args{arena_dump.GetDeviceBuffer(),
                         static_cast<std::uint32_t*>(mismatch_buffer.GetDeviceBuffer())};
    ck_tile::stream_config stream_config{nullptr, false, 0, 0, 1};
    ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<1>(DeferredPProbeKernel<RowStride>{}, dim3(1), dim3(128), 0, args));
    check_hip(hipGetLastError(), "deferred-P probe launch");
    check_hip(hipDeviceSynchronize(), "deferred-P probe synchronize");

    std::uint32_t roundtrip_mismatches = 0;
    mismatch_buffer.FromDevice(&roundtrip_mismatches);

    std::vector<std::uint8_t> arena(Layout::kArenaBytes);
    arena_dump.FromDevice(arena.data());

    std::size_t canary_mismatches = 0;
    for(ck_tile::index_t offset = 0; offset < Layout::kArenaBytes; ++offset)
    {
        bool may_be_written = false;
        for(ck_tile::index_t group = 0; group < Layout::kNumGroups; ++group)
        {
            const ck_tile::index_t relative = offset - Layout::GetSlotBase(group);
            if(relative >= 0 && relative < Layout::kSlotBytes)
            {
                const ck_tile::index_t row_bytes = RowStride * sizeof(DataType);
                const ck_tile::index_t row       = relative / row_bytes;
                const ck_tile::index_t row_byte  = relative % row_bytes;
                may_be_written =
                    row < Layout::kRows && row_byte < Layout::kColumnsPerGroup * sizeof(DataType);
            }
        }
        if(!may_be_written)
        {
            canary_mismatches += arena[offset] != kArenaCanary;
        }
    }

    const auto footprint = Layout::GetGroupFootprintBytes<RowStride>();
    std::cout << "{\"stride\":" << RowStride << ",\"slot_bytes\":" << Layout::kSlotBytes
              << ",\"footprint_bytes\":" << footprint
              << ",\"roundtrip_mismatches\":" << roundtrip_mismatches
              << ",\"canary_mismatches\":" << canary_mismatches << ",\"result\":\""
              << (roundtrip_mismatches == 0 && canary_mismatches == 0 ? "pass" : "fail") << "\"}\n";
    return roundtrip_mismatches == 0 && canary_mismatches == 0;
}

template <ck_tile::index_t RowStride>
bool run_benchmark(ck_tile::index_t repetitions)
{
    ck_tile::DeviceMem cycles_buffer(sizeof(std::uint64_t));
    ck_tile::DeviceMem checksum_buffer(128 * sizeof(std::uint32_t));
    const BenchmarkArgs args{static_cast<std::uint64_t*>(cycles_buffer.GetDeviceBuffer()),
                             static_cast<std::uint32_t*>(checksum_buffer.GetDeviceBuffer()),
                             repetitions};

    ck_tile::stream_config stream_config{nullptr, false, 0, 0, 1};
    ck_tile::launch_kernel(stream_config,
                           ck_tile::make_kernel<1>(
                               DeferredPBenchmarkKernel<RowStride>{}, dim3(1), dim3(128), 0, args));
    check_hip(hipGetLastError(), "deferred-P benchmark launch");
    check_hip(hipDeviceSynchronize(), "deferred-P benchmark synchronize");

    std::uint64_t cycles = 0;
    cycles_buffer.FromDevice(&cycles);
    std::array<std::uint32_t, 128> checksums{};
    checksum_buffer.FromDevice(checksums.data());
    const bool nonzero = checksums[0] != 0;

    std::cout << "{\"benchmark_stride\":" << RowStride << ",\"repetitions\":" << repetitions
              << ",\"cycles\":" << cycles << ",\"lane0_checksum\":" << checksums[0]
              << ",\"result\":\"" << (nonzero ? "pass" : "fail") << "\"}\n";
    return nonzero;
}

} // namespace

int main()
{
    try
    {
        int device = 0;
        check_hip(hipGetDevice(&device), "hipGetDevice");
        hipDeviceProp_t props{};
        check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
        if(std::string(props.gcnArchName).find("gfx125") == std::string::npos)
        {
            std::cerr << "test_tdm_fmha_d192_deferred_p requires gfx125, got " << props.gcnArchName
                      << '\n';
            return 77;
        }

        bool passed = true;
        passed &= run_probe<kCandidateStrides[0]>();
        passed &= run_probe<kCandidateStrides[1]>();
        passed &= run_probe<kCandidateStrides[2]>();
        passed &= run_probe<kCandidateStrides[3]>();
        constexpr ck_tile::index_t kBenchmarkRepetitions = 256;
        passed &= run_benchmark<kCandidateStrides[0]>(kBenchmarkRepetitions);
        passed &= run_benchmark<kCandidateStrides[1]>(kBenchmarkRepetitions);
        passed &= run_benchmark<kCandidateStrides[2]>(kBenchmarkRepetitions);
        passed &= run_benchmark<kCandidateStrides[3]>(kBenchmarkRepetitions);
        return passed ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "{\"result\":\"error\",\"message\":\"" << error.what() << "\"}\n";
        return 2;
    }
}
