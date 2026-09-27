// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"

// On gfx942, bf16 atomic adds are issued as global atomics instead of buffer atomics.
// They must still drop accesses past the buffer size, as buffer atomics do in hardware.

namespace {

using ck_tile::bf16_t;
using ck_tile::index_t;

constexpr index_t kBufferElements = 32;
constexpr index_t kAllocElements  = 64;
constexpr index_t kInRange        = 10;
constexpr index_t kPastTheEnd     = 40;

template <bool Raw>
__global__ void atomic_add_bf16_pair([[maybe_unused]] bf16_t* p, [[maybe_unused]] index_t offset)
{
#if defined(__gfx942__)
    if(threadIdx.x != 0)
        return;
    ck_tile::thread_buffer<bf16_t, 2> v;
    v[0] = ck_tile::type_convert<bf16_t>(1.0f);
    v[1] = ck_tile::type_convert<bf16_t>(1.0f);
    if constexpr(Raw)
        ck_tile::amd_buffer_atomic_add_raw<bf16_t, 2>(v, p, offset, 0, true, kBufferElements);
    else
        ck_tile::amd_buffer_atomic_add<bf16_t, 2>(v, p, offset, true, kBufferElements);
#endif
}

bool IsGfx942()
{
    hipDeviceProp_t props;
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return false;
    return std::string(props.gcnArchName).rfind("gfx942", 0) == 0;
}

template <bool Raw>
std::vector<float> RunPairAdds()
{
    bf16_t* d = nullptr;
    EXPECT_EQ(hipMalloc(&d, kAllocElements * sizeof(bf16_t)), hipSuccess);
    EXPECT_EQ(hipMemset(d, 0, kAllocElements * sizeof(bf16_t)), hipSuccess);
    atomic_add_bf16_pair<Raw><<<1, 64>>>(d, kInRange);
    atomic_add_bf16_pair<Raw><<<1, 64>>>(d, kPastTheEnd);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<bf16_t> h(kAllocElements);
    EXPECT_EQ(hipMemcpy(h.data(), d, kAllocElements * sizeof(bf16_t), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(hipFree(d), hipSuccess);
    std::vector<float> out(kAllocElements);
    for(index_t i = 0; i < kAllocElements; ++i)
        out[i] = ck_tile::type_convert<float>(h[i]);
    return out;
}

template <bool Raw>
void CheckPairAdds()
{
    if(!IsGfx942())
        GTEST_SKIP() << "bf16 global-atomic fallback is gfx942-only";
    const auto out = RunPairAdds<Raw>();
    EXPECT_EQ(out[kInRange], 1.0f);
    EXPECT_EQ(out[kInRange + 1], 1.0f);
    for(index_t i = kBufferElements; i < kAllocElements; ++i)
        EXPECT_EQ(out[i], 0.0f) << "element " << i << " is past the buffer end";
}

} // namespace

TEST(AtomicAddOutOfBounds, Bf16BufferAtomicAdd) { CheckPairAdds<false>(); }

TEST(AtomicAddOutOfBounds, Bf16BufferAtomicAddRaw) { CheckPairAdds<true>(); }
