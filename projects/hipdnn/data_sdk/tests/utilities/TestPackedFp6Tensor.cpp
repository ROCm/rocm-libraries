// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp6Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

using namespace hipdnn_data_sdk::utilities;
using hipdnn_data_sdk::types::fp6_e2m3;
using hipdnn_data_sdk::types::fp6_e3m2;
using hipdnn_data_sdk::types::fp6x2_e2m3;
using hipdnn_data_sdk::types::fp6x4_e2m3;

namespace
{
constexpr size_t BITS_PER_ELEMENT = 6;
constexpr size_t BITS_PER_BYTE = 8;
constexpr unsigned CODE_MASK = 0x3F;

// Mirrors the writer's addressing, so it is an oracle for the mirror tests rather
// than an independent check of it; the raw-byte tests below pin the bit pattern.
uint8_t unpackElementAt(const uint8_t* packed, size_t index)
{
    const size_t bitOffset = index * BITS_PER_ELEMENT;
    const size_t byteIndex = bitOffset / BITS_PER_BYTE;
    const auto bitIndex = static_cast<unsigned>(bitOffset % BITS_PER_BYTE);

    auto window = static_cast<unsigned>(packed[byteIndex]);
    if(bitIndex > BITS_PER_BYTE - BITS_PER_ELEMENT)
    {
        window |= static_cast<unsigned>(packed[byteIndex + 1]) << BITS_PER_BYTE;
    }
    return static_cast<uint8_t>((window >> bitIndex) & CODE_MASK);
}

template <typename T>
uint8_t encodedValue(const T& value)
{
    return static_cast<uint8_t>(value.data & CODE_MASK);
}

// For indexing and loop bounds. The sizing rule itself is pinned against
// literals in PackedBufferSizeCoversEveryBitResidue, not against this.
size_t packedBytes(size_t elements)
{
    return (elements * BITS_PER_ELEMENT + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
}
} // namespace

// ---------------------------------------------------------------------------
// Shape / validation
// ---------------------------------------------------------------------------

TEST(TestPackedFp6Tensor, ConstructionAndShape)
{
    const std::vector<int64_t> dims = {2, 16};
    const std::vector<int64_t> strides = {16, 1};
    const PackedFp6Tensor<fp6_e2m3> tensor(dims, strides);

    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
    EXPECT_EQ(tensor.elementCount(), 32u);
    EXPECT_EQ(tensor.elementSpace(), 32u);
    EXPECT_TRUE(tensor.isPacked());
}

TEST(TestPackedFp6Tensor, ElementSizeThrows)
{
    // elementCount() * elementSize() would over-report the packed buffer by 4/3.
    const PackedFp6Tensor<fp6_e2m3> tensor({4}, {1});
    EXPECT_THROW(tensor.elementSize(), std::logic_error);
}

TEST(TestPackedFp6Tensor, DimsStridesSizeMismatchThrows)
{
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({2, 16}, {1}), std::invalid_argument);
}

TEST(TestPackedFp6Tensor, NonDenseStridesThrow)
{
    // Padded layout: span (6) != element count (4).
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({2, 2}, {4, 1}), std::invalid_argument);
}

TEST(TestPackedFp6Tensor, NonPositiveDimsAndStridesThrow)
{
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({-1}, {1}), std::invalid_argument);
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({0, 16}, {16, 1}), std::invalid_argument);
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({2, 16}, {16, -1}), std::invalid_argument);
}

TEST(TestPackedFp6Tensor, EmptyDimsThrow)
{
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({}, {}), std::invalid_argument);
}

// A dims product of exactly 2^64 wraps to 0, and the span wraps to match, so the
// dense check would accept it. The zero-length buffer that follows makes the
// constructor's tail write dereference null at SIZE_MAX.
TEST(TestPackedFp6Tensor, WrappedElementCountThrows)
{
    const int64_t root = 1LL << 32;
    EXPECT_THROW(PackedFp6Tensor<fp6_e2m3>({root, root}, {root, 1}), std::invalid_argument);
}

TEST(TestPackedFp6Tensor, PerElementHostAccessThrows)
{
    PackedFp6Tensor<fp6_e3m2> tensor({4}, {1});
    const PackedFp6Tensor<fp6_e3m2>& constTensor = tensor;
    EXPECT_THROW(tensor.hostDataOffsetFromIndex(0), std::logic_error);
    EXPECT_THROW(constTensor.hostDataOffsetFromIndex(0), std::logic_error);
}

// ---------------------------------------------------------------------------
// Byte sizing
// ---------------------------------------------------------------------------

TEST(TestPackedFp6Tensor, PackedBufferSizeCoversEveryBitResidue)
{
    // ceil(N * 6 / 8) over every value of 6N mod 8, observed through
    // fillWithData, which clamps its copy to the buffer size.
    struct SizeCase
    {
        int64_t elements;
        size_t exactBytes;
    };
    const std::array<SizeCase, 7> cases = {{
        {4, 3}, // 6N mod 8 == 0
        {5, 4}, // 6N mod 8 == 6
        {6, 5}, // 6N mod 8 == 4
        {7, 6}, // 6N mod 8 == 2
        {32, 24},
        {100, 75},
        {128, 96},
    }};

    const std::vector<uint8_t> oversized(128, 0);
    for(const auto& testCase : cases)
    {
        PackedFp6Tensor<fp6_e2m3> tensor({testCase.elements}, {1});
        EXPECT_EQ(tensor.fillWithData(oversized.data(), oversized.size()), testCase.exactBytes)
            << "elements = " << testCase.elements;
    }
}

// Guards a write one byte past the bitstream. It happens when the final element
// starts at bit 0 or 2 of the last byte, i.e. N % 4 is 0 or 1; the other counts
// here are along for the ride. The bad write preserves the byte's value, so no
// assertion can see it -- this test only means something under ASAN.
TEST(TestPackedFp6Tensor, FillDoesNotWritePastPackedBuffer)
{
    for(const int64_t elements : {1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 100, 128, 4096})
    {
        PackedFp6Tensor<fp6_e2m3> tensor({elements}, {1});
        tensor.fillTensorWithRandomValues(-1.0f, 1.0f, 42u);
        tensor.fillTensorWithValue(1.0f);
        tensor.fillWithSentinelValue();
    }
}

// 5 elements -> 30 bits -> 4 bytes, so the top 2 bits of byte 3 must stay clear.
TEST(TestPackedFp6Tensor, PartialFinalByteLeavesUnusedBitsZero)
{
    PackedFp6Tensor<fp6_e2m3> packed({5}, {1});
    Tensor<fp6_e2m3> unpacked({5}, {1});
    packed.fillTensorWithRandomValues(-1.0f, 1.0f, 7u);
    unpacked.fillTensorWithRandomValues(-1.0f, 1.0f, 7u);

    ASSERT_EQ(packedBytes(5), 4u);
    const auto* host = static_cast<const uint8_t*>(packed.rawHostData());
    const auto* ref = static_cast<const fp6_e2m3*>(unpacked.rawHostData());
    EXPECT_EQ(unpackElementAt(host, 4), encodedValue(ref[4]));
    EXPECT_EQ(static_cast<uint8_t>(host[3] >> 6), 0u);
}

// ---------------------------------------------------------------------------
// Packing correctness
// ---------------------------------------------------------------------------

// Pins the reader against the canonical bit pattern: codes 1, 2, 3, 4 pack to
// 0x81, 0x30, 0x10 LSB-first, matching this SDK's fp6x4_e2m3 storage.
TEST(TestPackedFp6Tensor, CodeReaderMatchesCanonicalBitPattern)
{
    PackedFp6Tensor<fp6_e2m3> tensor({4}, {1});
    const std::vector<uint8_t> canonical = {0x81, 0x30, 0x10};

    ASSERT_EQ(tensor.fillWithData(canonical.data(), canonical.size()), 3u);

    const auto* host = static_cast<const uint8_t*>(tensor.rawHostData());
    for(size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(unpackElementAt(host, i), static_cast<uint8_t>(i + 1)) << "at index " << i;
    }
}

// Cross-checks the writer against the SDK's own canonical fp6x4_e2m3 packing.
TEST(TestPackedFp6Tensor, PackingMatchesFp6x4StorageType)
{
    PackedFp6Tensor<fp6_e2m3> packed({4}, {1});
    Tensor<fp6_e2m3> unpacked({4}, {1});
    packed.fillTensorWithRandomValues(-1.0f, 1.0f, 99u);
    unpacked.fillTensorWithRandomValues(-1.0f, 1.0f, 99u);

    const auto* unpackedHost = static_cast<const fp6_e2m3*>(unpacked.rawHostData());
    const fp6x4_e2m3 expected(fp6x2_e2m3(unpackedHost[0], unpackedHost[1]),
                              fp6x2_e2m3(unpackedHost[2], unpackedHost[3]));

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    EXPECT_TRUE(std::equal(expected.data.begin(), expected.data.end(), packedHost));
}

TEST(TestPackedFp6Tensor, FillWithValueWritesThreeBytePeriod)
{
    // Hand-derived bytes, so the writer is checked independently of the reader.
    PackedFp6Tensor<fp6_e2m3> tensor({8}, {1});
    tensor.fillTensorWithValue(2.0f);

    const auto code = static_cast<unsigned>(encodedValue(fp6_e2m3(2.0f)));
    const std::array<uint8_t, 3> period = {
        static_cast<uint8_t>((code | (code << 6)) & 0xFF),
        static_cast<uint8_t>(((code >> 2) | (code << 4)) & 0xFF),
        static_cast<uint8_t>(((code >> 4) | (code << 2)) & 0xFF),
    };

    const auto* host = static_cast<const uint8_t*>(tensor.rawHostData());
    for(size_t byte = 0; byte < packedBytes(8); ++byte)
    {
        EXPECT_EQ(host[byte], period[byte % 3]) << "at byte " << byte;
    }
    for(size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(unpackElementAt(host, i), static_cast<uint8_t>(code)) << "at index " << i;
    }
}

TEST(TestPackedFp6Tensor, SentinelUsesMaxBecauseFp6HasNoNaN)
{
    PackedFp6Tensor<fp6_e3m2> packed({4}, {1});
    Tensor<fp6_e3m2> unpacked({4}, {1});
    packed.fillWithSentinelValue();
    unpacked.fillWithSentinelValue();

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    const auto* unpackedHost = static_cast<const fp6_e3m2*>(unpacked.rawHostData());
    for(size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(unpackElementAt(packedHost, i), encodedValue(unpackedHost[i]))
            << "at index " << i;
    }
}

TEST(TestPackedFp6Tensor, FillWithDataCopiesAtMostPackedBytes)
{
    PackedFp6Tensor<fp6_e2m3> tensor({4}, {1}); // 3 packed bytes
    const std::vector<uint8_t> source = {0x12, 0x34, 0x56, 0x78};

    const size_t copied = tensor.fillWithData(source.data(), source.size());
    EXPECT_EQ(copied, 3u);

    const auto* host = static_cast<const uint8_t*>(tensor.rawHostData());
    EXPECT_EQ(host[0], 0x12);
    EXPECT_EQ(host[1], 0x34);
    EXPECT_EQ(host[2], 0x56);
}

TEST(TestPackedFp6Tensor, FillWithDataLeavesUncoveredBytesAlone)
{
    // A short copy touches only the bytes it covers, matching Tensor<T>.
    PackedFp6Tensor<fp6_e2m3> tensor({8}, {1}); // 6 packed bytes
    tensor.fillTensorWithValue(2.0f);
    const auto* host = static_cast<const uint8_t*>(tensor.rawHostData());
    const std::vector<uint8_t> untouched(host + 1, host + packedBytes(8));

    const std::vector<uint8_t> source = {0xAB};
    EXPECT_EQ(tensor.fillWithData(source.data(), source.size()), 1u);

    EXPECT_EQ(host[0], 0xAB);
    EXPECT_TRUE(std::equal(untouched.begin(), untouched.end(), host + 1));
}

// ---------------------------------------------------------------------------
// Randomization must match the unpacked Tensor<fp6_*> value-for-value for the
// same (seed, min, max), or the packed GPU bundle and the CPU reference bundle
// diverge silently.
// ---------------------------------------------------------------------------

template <typename T>
class PackedFp6TensorMirror : public ::testing::Test
{
};

using Fp6Types = ::testing::Types<fp6_e2m3, fp6_e3m2>;
TYPED_TEST_SUITE(PackedFp6TensorMirror, Fp6Types, );

TYPED_TEST(PackedFp6TensorMirror, FillWithRandomValuesMirrorsUnpackedTensor)
{
    // 24 elements covers every bitIndex residue (0, 6, 4, 2) several times over.
    const std::vector<int64_t> dims = {3, 8};
    const std::vector<int64_t> strides = {8, 1};
    // Bounds deliberately not FP6-representable: Tensor<T> rounds them through T
    // before building the distribution, so a raw-float writer would diverge here.
    const float minValue = -0.9f;
    const float maxValue = 1.1f;
    const unsigned int seedValue = 1337u;

    PackedFp6Tensor<TypeParam> packed(dims, strides);
    Tensor<TypeParam> unpacked(dims, strides);
    packed.fillTensorWithRandomValues(minValue, maxValue, seedValue);
    unpacked.fillTensorWithRandomValues(minValue, maxValue, seedValue);

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    const auto* unpackedHost = static_cast<const TypeParam*>(unpacked.rawHostData());
    for(size_t i = 0; i < packed.elementCount(); ++i)
    {
        ASSERT_EQ(unpackElementAt(packedHost, i), encodedValue(unpackedHost[i]))
            << "at index " << i;
    }
}

TYPED_TEST(PackedFp6TensorMirror, FillWithRandomValuesMirrorsUnpackedTensorWhenColumnMajor)
{
    const std::vector<int64_t> dims = {3, 8};
    const std::vector<int64_t> strides = {1, 3}; // column-major, dense
    const float minValue = -0.9f;
    const float maxValue = 1.1f;
    const unsigned int seedValue = 1337u;

    PackedFp6Tensor<TypeParam> packed(dims, strides);
    Tensor<TypeParam> unpacked(dims, strides);
    packed.fillTensorWithRandomValues(minValue, maxValue, seedValue);
    unpacked.fillTensorWithRandomValues(minValue, maxValue, seedValue);

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    for(int64_t i0 = 0; i0 < dims[0]; ++i0)
    {
        for(int64_t i1 = 0; i1 < dims[1]; ++i1)
        {
            // Read the packed buffer the way a kernel does: code at the stride offset.
            const auto slot = static_cast<size_t>((i0 * strides[0]) + (i1 * strides[1]));
            EXPECT_EQ(unpackElementAt(packedHost, slot),
                      encodedValue(unpacked.getHostValue(i0, i1)))
                << "mismatch at coordinate (" << i0 << "," << i1 << ")";
        }
    }
}

TYPED_TEST(PackedFp6TensorMirror, FillWithRandomValuesMirrorsAtMxOperandShape)
{
    // A real MX operand shape.
    const std::vector<int64_t> dims = {32, 128};
    const std::vector<int64_t> strides = {128, 1};

    PackedFp6Tensor<TypeParam> packed(dims, strides);
    Tensor<TypeParam> unpacked(dims, strides);
    packed.fillTensorWithRandomValues(-1.0f, 1.0f, 2024u);
    unpacked.fillTensorWithRandomValues(-1.0f, 1.0f, 2024u);

    const auto* packedHost = static_cast<const uint8_t*>(packed.rawHostData());
    const auto* unpackedHost = static_cast<const TypeParam*>(unpacked.rawHostData());
    for(size_t i = 0; i < packed.elementCount(); ++i)
    {
        ASSERT_EQ(unpackElementAt(packedHost, i), encodedValue(unpackedHost[i]))
            << "at index " << i;
    }
}
