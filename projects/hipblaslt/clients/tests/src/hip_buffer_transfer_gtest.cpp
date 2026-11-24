// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipBuffer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace
{
    template <std::size_t N>
    void copyToHost(HipHostBuffer& buffer, const std::array<int32_t, N>& values)
    {
        std::copy(values.begin(), values.end(), buffer.as<int32_t>());
    }

    template <std::size_t N>
    void expectHostEquals(const HipHostBuffer& buffer, const std::array<int32_t, N>& expected)
    {
        const auto* actual = buffer.as<int32_t>();
        for(std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_EQ(actual[i], expected[i]) << "mismatch at element " << i;
    }
}

TEST(HipBufferTransfer, smoke_RejectsZeroTransferCounts)
{
    HipDeviceBuffer device(HIP_R_32I, 4);
    HipHostBuffer   host(HIP_R_32I, 4);
    ASSERT_EQ(device.memcheck(), hipSuccess);

    EXPECT_EQ(synchronize(device, host, 0), hipErrorInvalidValue);
    EXPECT_EQ(broadcast(device, 0), hipErrorInvalidValue);
}

TEST(HipBufferTransfer, smoke_RejectsInvalidSwizzleGeometry)
{
    constexpr std::array<int32_t, 3> values{11, 22, 33};
    constexpr std::array<int32_t, 3> sentinel{-1, -1, -1};

    HipHostBuffer input(HIP_R_32I, values.size());
    copyToHost(input, values);
    HipDeviceBuffer device(HIP_R_32I, values.size());
    ASSERT_EQ(device.memcheck(), hipSuccess);
    ASSERT_EQ(synchronize(device, input), hipSuccess);

    HipHostBuffer observed(HIP_R_32I, sentinel.size());
    copyToHost(observed, sentinel);
    EXPECT_EQ(synchronize(observed, device, 1, 3, 1, 2, sizeof(int32_t), true),
              hipErrorInvalidValue);
    expectHostEquals(observed, sentinel);
}

class HipBufferTransferMemoryMode : public ::testing::TestWithParam<bool>
{
};

TEST_P(HipBufferTransferMemoryMode, smoke_TransfersAndBroadcastsNormally)
{
    const bool hmm = GetParam();

    constexpr std::size_t             repeats = 4;
    constexpr std::array<int32_t, 3>  firstValues{11, -7, 42};
    constexpr std::array<int32_t, 12> firstExpected{11, -7, 42, 11, -7, 42, 11, -7, 42, 11, -7, 42};
    constexpr std::array<int32_t, 12> broadcastInput{9, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<int32_t, 12> broadcastExpected{9, 8, 7, 9, 8, 7, 9, 8, 7, 9, 8, 7};

    // A nondefault stream checks that the device-side block copies follow the upload.
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    const std::unique_ptr<std::remove_pointer_t<hipStream_t>, decltype(&hipStreamDestroy)>
        streamOwner(stream, &hipStreamDestroy);

    HipHostBuffer first(HIP_R_32I, firstValues.size());
    copyToHost(first, firstValues);
    HipDeviceBuffer device(HIP_R_32I, firstExpected.size(), hmm);
    ASSERT_EQ(device.memcheck(), hipSuccess);
    ASSERT_EQ(synchronize(device, first, repeats, stream), hipSuccess);

    HipHostBuffer observed(HIP_R_32I, firstExpected.size());
    ASSERT_EQ(synchronize(observed, device), hipSuccess);
    expectHostEquals(observed, firstExpected);

    HipHostBuffer fullInput(HIP_R_32I, broadcastInput.size());
    copyToHost(fullInput, broadcastInput);
    ASSERT_EQ(synchronize(device, fullInput, 1, stream), hipSuccess);
    ASSERT_EQ(broadcast(device, repeats), hipSuccess);
    ASSERT_EQ(synchronize(observed, device), hipSuccess);
    expectHostEquals(observed, broadcastExpected);
}

// 11 bytes in 3 blocks: each block is 3 bytes at offset block * 11 / 3, leaving the
// remainder bytes 6 and 10 untouched.
TEST_P(HipBufferTransferMemoryMode, smoke_TransfersPreserveTruncatedBlockOffsets)
{
    const bool hmm = GetParam();

    constexpr std::array<uint8_t, 11> expected{11, 22, 33, 11, 22, 33, 0xa5, 11, 22, 33, 0xa5};
    HipHostBuffer input(HIP_R_8U, expected.size());
    std::fill_n(input.as<uint8_t>(), expected.size(), 0xa5);
    std::copy_n(expected.begin(), 3, input.as<uint8_t>());
    HipDeviceBuffer device(HIP_R_8U, expected.size(), hmm);
    ASSERT_EQ(device.memcheck(), hipSuccess);
    ASSERT_EQ(synchronize(device, input), hipSuccess);
    ASSERT_EQ(synchronize(device, input, 3), hipSuccess);
    HipHostBuffer observed(HIP_R_8U, expected.size());
    ASSERT_EQ(synchronize(observed, device), hipSuccess);
    for(std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(observed.as<uint8_t>()[i], expected[i]) << "mismatch at byte " << i;
}

INSTANTIATE_TEST_SUITE_P(MemoryModes,
                        HipBufferTransferMemoryMode,
                        ::testing::Values(false, true),
                        [](const ::testing::TestParamInfo<bool>& info) {
                            return info.param ? "Managed" : "Device";
                        });

TEST(HipBufferTransfer, smoke_TransfersValidSwizzleGeometry)
{
    constexpr std::size_t             row = 2;
    constexpr std::size_t             col = 3;
    constexpr std::size_t             lda = 4;
    constexpr std::array<int32_t, 6>  compact{1, 2, 3, 4, 5, 6};
    constexpr std::array<int32_t, 12> paddedExpected{1, 2, -1, -1, 3, 4, -1, -1, 5, 6, -1, -1};

    HipHostBuffer input(HIP_R_32I, compact.size());
    copyToHost(input, compact);
    HipDeviceBuffer device(HIP_R_32I, compact.size());
    ASSERT_EQ(device.memcheck(), hipSuccess);
    ASSERT_EQ(synchronize(device, input), hipSuccess);

    HipHostBuffer plain(HIP_R_32I, compact.size());
    ASSERT_EQ(synchronize(plain, device, 1, 3, 2, 0, sizeof(int32_t), false), hipSuccess);
    expectHostEquals(plain, compact);

    HipHostBuffer padded(HIP_R_32I, paddedExpected.size());
    std::fill_n(padded.as<int32_t>(), paddedExpected.size(), -1);
    ASSERT_EQ(synchronize(padded, device, 1, row, col, lda, sizeof(int32_t), true), hipSuccess);
    expectHostEquals(padded, paddedExpected);
}
