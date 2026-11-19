// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/convolution.hpp>

#include <cstdint>
#include <set>
#include <thread>
#include <vector>

namespace {

uint32_t NewDescriptorSeed()
{
    return miopen::ConvolutionDescriptor{}.attribute.fp8rounding_mode.GetSeed();
}

} // namespace

TEST(CPU_ConvFP8RoundingSeed_NONE, DistinctPerDescriptor)
{
    if(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED)
        GTEST_SKIP() << "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED is set";

    EXPECT_NE(NewDescriptorSeed(), NewDescriptorSeed());
}

// The seed engine is per thread, so threads must not start from the same state.
TEST(CPU_ConvFP8RoundingSeed_NONE, DistinctAcrossThreads)
{
    if(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED)
        GTEST_SKIP() << "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED is set";

    constexpr std::size_t num_threads = 8;
    std::vector<uint32_t> seeds(num_threads);
    std::vector<std::thread> threads;
    for(std::size_t i = 0; i < num_threads; ++i)
        threads.emplace_back([&seeds, i] { seeds[i] = NewDescriptorSeed(); });
    for(auto& t : threads)
        t.join();

    EXPECT_EQ(std::set<uint32_t>(seeds.begin(), seeds.end()).size(), num_threads);
}
