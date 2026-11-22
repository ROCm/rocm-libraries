// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Round trip global -> LDS -> global through the gfx1250 b128 async LDS copies:
// amd_async_global_load_to_lds (asm path) and amd_async_global_store_from_lds_b128, each with the
// default and the nontemporal cache policy, ordered by s_wait_asynccnt_pinned.

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/amd_buffer_addressing_builtins.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace {

constexpr int kWaveSize = 32;            // gfx1250 is wave32
constexpr int kHalf     = kWaveSize * 4; // int32 elements per half buffer, 16 bytes per lane

// One wave copies two halves, 16 bytes per lane each. The second half goes through the immediate
// offset of both copies, which applies to the global and to the LDS address. The coherence
// enumerators differ per target, so the policy is chosen in device code.
template <bool Nontemporal>
struct AsyncLdsRoundTripKernel
{
    static constexpr int kBlockSize = kWaveSize;

    CK_TILE_DEVICE void operator()(const int32_t* __restrict__ src, int32_t* __restrict__ dst) const
    {
#if defined(__gfx125__)
        constexpr auto Coherence = Nontemporal
                                       ? ck_tile::amd_buffer_coherence_enum::NT
                                       : ck_tile::amd_buffer_coherence_enum::coherence_default;
        // lds is the only LDS allocation, so lane 0 copies from LDS address 0 (the case a cast
        // through a generic pointer gets wrong).
        __shared__ int32_t lds[2 * kHalf];
        const int lane = threadIdx.x;

        ck_tile::amd_async_global_load_to_lds<int32_t, 4, 0, true, Coherence>(
            lds + lane * 4, src, lane * 4, true);
        ck_tile::amd_async_global_load_to_lds<int32_t, 4, kHalf, true, Coherence>(
            lds + lane * 4, src, kHalf + lane * 4, true);
        ck_tile::s_wait_asynccnt_pinned<0>();

        const uint32_t lds_addr =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds + lane * 4));
        ck_tile::amd_async_global_store_from_lds_b128<0, Coherence>(dst + lane * 4, lds_addr);
        ck_tile::amd_async_global_store_from_lds_b128<kHalf * sizeof(int32_t), Coherence>(
            dst + lane * 4, lds_addr);
        ck_tile::s_wait_asynccnt_pinned<0>();
#else
        ck_tile::ignore = src;
        ck_tile::ignore = dst;
#endif
    }
};

template <bool Nontemporal>
void run_round_trip()
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "async LDS copies require gfx1250; device reports '"
                     << ck_tile::get_device_name() << "'";

    std::vector<int32_t> h_src(2 * kHalf), h_dst(2 * kHalf);
    for(int i = 0; i < 2 * kHalf; ++i)
        h_src[i] = static_cast<int32_t>(0x9E3779B9u * static_cast<uint32_t>(i + 1));

    ck_tile::DeviceMem d_src(h_src.size() * sizeof(int32_t));
    ck_tile::DeviceMem d_dst(h_dst.size() * sizeof(int32_t));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    using Kernel = AsyncLdsRoundTripKernel<Nontemporal>;
    ck_tile::launch_and_check(
        ck_tile::stream_config{},
        ck_tile::make_kernel(Kernel{},
                             dim3(1),
                             dim3(Kernel::kBlockSize),
                             0,
                             static_cast<const int32_t*>(d_src.GetDeviceBuffer()),
                             static_cast<int32_t*>(d_dst.GetDeviceBuffer())));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    d_dst.FromDevice(h_dst.data());

    EXPECT_EQ(std::memcmp(h_dst.data(), h_src.data(), h_src.size() * sizeof(int32_t)), 0);
}

} // namespace

TEST(AsyncLdsRoundTrip, B128_Default) { run_round_trip<false>(); }

TEST(AsyncLdsRoundTrip, B128_Nontemporal) { run_round_trip<true>(); }
