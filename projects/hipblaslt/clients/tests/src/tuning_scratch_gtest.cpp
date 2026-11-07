// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../../library/src/amd_detail/rocblaslt/src/include/tuning_scratch.hpp"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

namespace
{
    TEST(TuningScratchSeed_pre_checkin, GradientAuxiliaryCopiesInputIntoEveryBlock)
    {
        int devices = 0;
        if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
            GTEST_SKIP() << "No GPU available";

        struct Buffers
        {
            void*       source      = nullptr;
            void*       destination = nullptr;
            hipStream_t stream      = nullptr;
            ~Buffers()
            {
                if(stream)
                    static_cast<void>(hipStreamSynchronize(stream));
                if(destination)
                    static_cast<void>(hipFree(destination));
                if(source)
                    static_cast<void>(hipFree(source));
                if(stream)
                    static_cast<void>(hipStreamDestroy(stream));
            }
        } buffers;
        constexpr size_t                      elements = 5, stride = 8, blocks = 3;
        const std::array<uint32_t, elements>  input{1, 7, 19, 31, 47};
        std::array<uint32_t, stride * blocks> output;
        output.fill(0xabababab);
        ASSERT_EQ(hipStreamCreate(&buffers.stream), hipSuccess);
        ASSERT_EQ(hipMalloc(&buffers.source, sizeof(input)), hipSuccess);
        ASSERT_EQ(hipMalloc(&buffers.destination, sizeof(output)), hipSuccess);
        ASSERT_EQ(hipMemcpy(buffers.source, input.data(), sizeof(input), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpy(buffers.destination, output.data(), sizeof(output), hipMemcpyHostToDevice),
            hipSuccess);

        ASSERT_TRUE(rocblaslt::seedGradientAuxiliary(buffers.destination,
                                                     buffers.source,
                                                     sizeof(input),
                                                     stride * sizeof(uint32_t),
                                                     blocks,
                                                     buffers.stream));
        ASSERT_EQ(hipStreamSynchronize(buffers.stream), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(output.data(), buffers.destination, sizeof(output), hipMemcpyDeviceToHost),
            hipSuccess);
        for(size_t block = 0; block < blocks; ++block)
            for(size_t element = 0; element < stride; ++element)
                EXPECT_EQ(output[block * stride + element],
                          element < elements ? input[element] : 0xabababab)
                    << "block=" << block << " element=" << element;
        std::array<uint32_t, elements> unchanged;
        ASSERT_EQ(hipMemcpy(unchanged.data(), buffers.source, sizeof(input), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(unchanged, input);
    }
}
