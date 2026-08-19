// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipBuffer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>

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

TEST(HipBuffer, smoke_MoveConstructionTransfersOwnership)
{
    constexpr std::array<int32_t, 5> values{3, 1, 4, 1, 5};
    HipHostBuffer                    input(HIP_R_32I, values.size());
    copyToHost(input, values);

    std::optional<HipDeviceBuffer> moved;
    void*                          allocation = nullptr;
    {
        HipDeviceBuffer source(HIP_R_32I, values.size());
        ASSERT_EQ(source.memcheck(), hipSuccess);
        ASSERT_EQ(synchronize(source, input), hipSuccess);
        allocation = source.buf();

        moved.emplace(std::move(source));

        EXPECT_EQ(moved->buf(), allocation);
        EXPECT_EQ(moved->getNumBytes(), values.size() * sizeof(int32_t));
        EXPECT_EQ(source.buf(), nullptr);
        EXPECT_EQ(source.getNumBytes(), 0);
        EXPECT_EQ(source.memcheck(), hipSuccess);
    }

    HipHostBuffer observed(HIP_R_32I, values.size());
    ASSERT_EQ(synchronize(observed, *moved), hipSuccess);
    expectHostEquals(observed, values);
}

TEST(HipBuffer, smoke_MoveAssignmentReleasesPriorOwnership)
{
    constexpr std::array<int32_t, 5> values{2, 7, 1, 8, 2};
    HipHostBuffer                    input(HIP_R_32I, values.size());
    copyToHost(input, values);

    std::optional<HipDeviceBuffer> destination;
    destination.emplace(HIP_R_8U, 7);
    ASSERT_EQ(destination->memcheck(), hipSuccess);
    void* priorAllocation = destination->buf();

    void* transferredAllocation = nullptr;
    {
        HipDeviceBuffer source(HIP_R_32I, values.size());
        ASSERT_EQ(source.memcheck(), hipSuccess);
        ASSERT_EQ(synchronize(source, input), hipSuccess);
        transferredAllocation = source.buf();
        ASSERT_NE(transferredAllocation, priorAllocation);

        *destination = std::move(source);

        EXPECT_EQ(destination->buf(), transferredAllocation);
        EXPECT_EQ(destination->getNumBytes(), values.size() * sizeof(int32_t));
        EXPECT_EQ(source.buf(), nullptr);
        EXPECT_EQ(source.getNumBytes(), 0);
        EXPECT_EQ(source.memcheck(), hipSuccess);
    }

    HipHostBuffer observed(HIP_R_32I, values.size());
    ASSERT_EQ(synchronize(observed, *destination), hipSuccess);
    expectHostEquals(observed, values);

    HipDeviceBuffer replacement(HIP_R_8U, 7);
    EXPECT_EQ(replacement.memcheck(), hipSuccess);
}

TEST(HipBuffer, smoke_RejectsZeroTransferCounts)
{
    HipDeviceBuffer device(HIP_R_32I, 4);
    HipHostBuffer   host(HIP_R_32I, 4);
    ASSERT_EQ(device.memcheck(), hipSuccess);

    EXPECT_EQ(synchronize(device, host, 0), hipErrorInvalidValue);
    EXPECT_EQ(broadcast(device, 0), hipErrorInvalidValue);
}

TEST(HipBuffer, smoke_RejectsInvalidSwizzleGeometry)
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
    EXPECT_EQ(synchronize(observed,
                          device,
                          /*batch=*/1,
                          /*row=*/3,
                          /*col=*/1,
                          /*lda=*/2,
                          sizeof(int32_t),
                          /*needSwizzle=*/true),
              hipErrorInvalidValue);
    expectHostEquals(observed, sentinel);
}

TEST(HipBuffer, smoke_TransfersAndBroadcastsNormally)
{
    constexpr std::size_t             repeats = 4;
    constexpr std::array<int32_t, 3>  firstValues{11, -7, 42};
    constexpr std::array<int32_t, 12> firstExpected{11, -7, 42, 11, -7, 42, 11, -7, 42, 11, -7, 42};
    constexpr std::array<int32_t, 12> broadcastInput{9, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<int32_t, 12> broadcastExpected{9, 8, 7, 9, 8, 7, 9, 8, 7, 9, 8, 7};

    HipHostBuffer first(HIP_R_32I, firstValues.size());
    copyToHost(first, firstValues);
    HipDeviceBuffer device(HIP_R_32I, firstExpected.size());
    ASSERT_EQ(device.memcheck(), hipSuccess);
    ASSERT_EQ(synchronize(device, first, repeats), hipSuccess);

    HipHostBuffer observed(HIP_R_32I, firstExpected.size());
    ASSERT_EQ(synchronize(observed, device), hipSuccess);
    expectHostEquals(observed, firstExpected);

    HipHostBuffer fullInput(HIP_R_32I, broadcastInput.size());
    copyToHost(fullInput, broadcastInput);
    ASSERT_EQ(synchronize(device, fullInput), hipSuccess);
    ASSERT_EQ(broadcast(device, repeats), hipSuccess);
    ASSERT_EQ(synchronize(observed, device), hipSuccess);
    expectHostEquals(observed, broadcastExpected);
}

TEST(HipBuffer, smoke_TransfersValidSwizzleGeometry)
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
    ASSERT_EQ(synchronize(plain,
                          device,
                          /*batch=*/1,
                          /*row=*/3,
                          /*col=*/2,
                          /*lda=*/0,
                          sizeof(int32_t),
                          /*needSwizzle=*/false),
              hipSuccess);
    expectHostEquals(plain, compact);

    HipHostBuffer padded(HIP_R_32I, paddedExpected.size());
    std::fill_n(padded.as<int32_t>(), paddedExpected.size(), -1);
    ASSERT_EQ(synchronize(padded,
                          device,
                          /*batch=*/1,
                          row,
                          col,
                          lda,
                          sizeof(int32_t),
                          /*needSwizzle=*/true),
              hipSuccess);
    expectHostEquals(padded, paddedExpected);
}
