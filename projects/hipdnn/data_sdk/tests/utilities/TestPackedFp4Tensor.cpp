// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp4Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

using namespace hipdnn_data_sdk::utilities;
using hipdnn_data_sdk::types::fp4_e2m1;

namespace
{
// Low nibble = even logical index, high nibble = odd.
uint8_t nibbleAt(const uint8_t* packed, size_t index)
{
    const uint8_t byte = packed[index / 2];
    return static_cast<uint8_t>((index % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F));
}
} // namespace

TEST(TestPackedFp4Tensor, ConstructionAndShape)
{
    const std::vector<int64_t> dims = {2, 16};
    const std::vector<int64_t> strides = {16, 1};
    const PackedFp4Tensor tensor(dims, strides);

    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
    EXPECT_EQ(tensor.elementCount(), 32u);
    EXPECT_EQ(tensor.elementSpace(), 32u);
    EXPECT_TRUE(tensor.isPacked());
}

TEST(TestPackedFp4Tensor, ElementSizeThrows)
{
    // elementCount() * elementSize() would overrun the half-size packed buffer,
    // so elementSize() must fail loudly rather than return a misleading value.
    const PackedFp4Tensor tensor({4}, {1});
    EXPECT_THROW(tensor.elementSize(), std::logic_error);
}

TEST(TestPackedFp4Tensor, DimsStridesSizeMismatchThrows)
{
    EXPECT_THROW(PackedFp4Tensor({2, 16}, {1}), std::invalid_argument);
}

TEST(TestPackedFp4Tensor, NonDenseStridesThrow)
{
    // Padded layout: span (6) != element count (4), so nibble packing is undefined.
    EXPECT_THROW(PackedFp4Tensor({2, 2}, {4, 1}), std::invalid_argument);
}

TEST(TestPackedFp4Tensor, NonPositiveDimsAndStridesThrow)
{
    EXPECT_THROW(PackedFp4Tensor({-1}, {1}), std::invalid_argument);
    EXPECT_THROW(PackedFp4Tensor({0, 16}, {16, 1}), std::invalid_argument);
    EXPECT_THROW(PackedFp4Tensor({2, 16}, {16, -1}), std::invalid_argument);
}

TEST(TestPackedFp4Tensor, FillWithValueSetsBothNibbles)
{
    PackedFp4Tensor tensor({4}, {1});
    tensor.fillTensorWithValue(2.0f);

    const auto nibble = static_cast<uint8_t>(fp4_e2m1(2.0f).data & 0x0F);
    const auto expected = static_cast<uint8_t>(nibble | (nibble << 4));

    const auto* host = static_cast<const uint8_t*>(tensor.rawHostData());
    for(size_t byte = 0; byte < 2; ++byte)
    {
        EXPECT_EQ(host[byte], expected);
    }
}

TEST(TestPackedFp4Tensor, OddElementCountLeavesFinalHighNibbleUnused)
{
    // 5 elements -> 3 bytes; the last element occupies byte 2's low nibble, and
    // the random fill (which zeroes bytes first) leaves its high nibble unused.
    PackedFp4Tensor packed({5}, {1});
    Tensor<fp4_e2m1> unpacked({5}, {1});
    packed.fillTensorWithRandomValues(-6.0f, 6.0f, 7u);
    unpacked.fillTensorWithRandomValues(-6.0f, 6.0f, 7u);

    const auto* host = static_cast<const uint8_t*>(packed.rawHostData());
    const auto* ref = static_cast<const fp4_e2m1*>(unpacked.rawHostData());
    EXPECT_EQ(nibbleAt(host, 4), static_cast<uint8_t>(ref[4].data & 0x0F));
    EXPECT_EQ(static_cast<uint8_t>((host[2] >> 4) & 0x0F), 0u);
}

// The reason this class exists: its randomization must agree value-for-value with
// the unpacked Tensor<fp4_e2m1> given the same (seed, min, max). Bounds deliberately
// not FP4-representable (3.5 rounds to 4.0): Tensor<T> rounds them through T before
// building the distribution, so a raw-float writer would diverge here.
TEST(TestPackedFp4Tensor, FillWithRandomValuesMirrorsUnpackedTensor)
{
    const std::vector<int64_t> dims = {3, 8};
    const std::vector<int64_t> strides = {8, 1};
    const float minValue = -3.5f;
    const float maxValue = 3.5f;
    const unsigned int seedValue = 1337u;

    PackedFp4Tensor packed(dims, strides);
    Tensor<fp4_e2m1> unpacked(dims, strides);
    packed.fillTensorWithRandomValues(minValue, maxValue, seedValue);
    unpacked.fillTensorWithRandomValues(minValue, maxValue, seedValue);

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    const auto* unpackedHost = static_cast<const fp4_e2m1*>(unpacked.rawHostData());
    for(size_t i = 0; i < packed.elementCount(); ++i)
    {
        EXPECT_EQ(nibbleAt(packedHost, i), static_cast<uint8_t>(unpackedHost[i].data & 0x0F))
            << "mismatch at logical index " << i;
    }
}

// The mirror must hold at every COORDINATE, not merely slot for slot, or the FP4 MX
// suites compare a GPU bundle against a reference bundle holding different values.
// Column-major strides separate the two: logical element i no longer lives in nibble
// slot i, so a fill that ignores strides lands every draw on the wrong coordinate.
TEST(TestPackedFp4Tensor, FillWithRandomValuesMirrorsUnpackedTensorWhenColumnMajor)
{
    const std::vector<int64_t> dims = {3, 8};
    const std::vector<int64_t> strides = {1, 3}; // column-major, dense
    const float minValue = -6.0f;
    const float maxValue = 6.0f;
    const unsigned int seedValue = 1337u;

    PackedFp4Tensor packed(dims, strides);
    Tensor<fp4_e2m1> unpacked(dims, strides);
    packed.fillTensorWithRandomValues(minValue, maxValue, seedValue);
    unpacked.fillTensorWithRandomValues(minValue, maxValue, seedValue);

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    for(int64_t i0 = 0; i0 < dims[0]; ++i0)
    {
        for(int64_t i1 = 0; i1 < dims[1]; ++i1)
        {
            // Read the packed buffer the way a kernel does: nibble at the stride offset.
            const auto slot = static_cast<size_t>((i0 * strides[0]) + (i1 * strides[1]));
            EXPECT_EQ(nibbleAt(packedHost, slot),
                      static_cast<uint8_t>(unpacked.getHostValue(i0, i1).data & 0x0F))
                << "mismatch at coordinate (" << i0 << "," << i1 << ")";
        }
    }
}

TEST(TestPackedFp4Tensor, FillWithDataCopiesAtMostPackedBytes)
{
    PackedFp4Tensor tensor({4}, {1}); // 2 packed bytes
    const std::vector<uint8_t> source = {0x12, 0x34, 0x56};

    const size_t copied = tensor.fillWithData(source.data(), source.size());
    EXPECT_EQ(copied, 2u);

    const auto* host = static_cast<const uint8_t*>(tensor.rawHostData());
    EXPECT_EQ(host[0], 0x12);
    EXPECT_EQ(host[1], 0x34);
}

TEST(TestPackedFp4Tensor, PerElementHostAccessThrows)
{
    PackedFp4Tensor tensor({4}, {1});
    const PackedFp4Tensor& constTensor = tensor;
    EXPECT_THROW(tensor.hostDataOffsetFromIndex(0), std::logic_error);
    EXPECT_THROW(constTensor.hostDataOffsetFromIndex(0), std::logic_error);
}
