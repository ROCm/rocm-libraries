// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <string>
#include <vector>

using hipdnn_flatbuffers_sdk::utilities::formatUuid;
using hipdnn_flatbuffers_sdk::utilities::isUuidV4;
using hipdnn_flatbuffers_sdk::utilities::parseUuid;
using hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid;
using hipdnn_flatbuffers_sdk::utilities::toUuidBytes;
using hipdnn_flatbuffers_sdk::utilities::UuidBytes;

namespace
{
constexpr UuidBytes ALL_ZERO_BYTES{};
constexpr UuidBytes ALL_ONES_BYTES{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
} // namespace

TEST(TestUuid, FormatsAndParsesCanonicalText)
{
    const auto bytes = parseUuid("01234567-89AB-4DEF-8123-456789ABCDEF");
    EXPECT_EQ(formatUuid(bytes), "01234567-89ab-4def-8123-456789abcdef");
    EXPECT_TRUE(isUuidV4(bytes));

    const auto flatbufferUuid = toFlatbufferUuid(bytes);
    EXPECT_EQ(toUuidBytes(flatbufferUuid), bytes);
}

TEST(TestUuid, FormatUuidLowercasesHexDigits)
{
    const UuidBytes bytes{0xAB,
                          0xCD,
                          0xEF,
                          0x01,
                          0x23,
                          0x45,
                          0x67,
                          0x89,
                          0xAB,
                          0xCD,
                          0xEF,
                          0x01,
                          0x23,
                          0x45,
                          0x67,
                          0x89};
    EXPECT_EQ(formatUuid(bytes), "abcdef01-2345-6789-abcd-ef0123456789");
}

TEST(TestUuid, FormatUuidPlacesDashesAtCanonicalPositions)
{
    const std::string text = formatUuid(ALL_ZERO_BYTES);
    ASSERT_EQ(text.size(), 36U);
    EXPECT_EQ(text[8], '-');
    EXPECT_EQ(text[13], '-');
    EXPECT_EQ(text[18], '-');
    EXPECT_EQ(text[23], '-');
    EXPECT_EQ(text, "00000000-0000-0000-0000-000000000000");
}

TEST(TestUuid, FormatUuidHandlesAllOnesBoundary)
{
    EXPECT_EQ(formatUuid(ALL_ONES_BYTES), "ffffffff-ffff-ffff-ffff-ffffffffffff");
}

TEST(TestUuid, ParseUuidIsCaseInsensitiveAndRoundTrips)
{
    EXPECT_EQ(parseUuid("ABCDEF01-2345-6789-ABCD-EF0123456789"),
              parseUuid("abcdef01-2345-6789-abcd-ef0123456789"));
}

TEST(TestUuid, ParseUuidRejectsWrongLength)
{
    EXPECT_THROW((void)parseUuid("01234567-89ab-4def-8123-456789abcde"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid("01234567-89ab-4def-8123-456789abcdef0"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid(""), std::invalid_argument);
}

TEST(TestUuid, ParseUuidRejectsMisplacedDashes)
{
    // Length 36, but the dash sits one column right of its canonical slot each time,
    // shifting a hex digit into it.
    EXPECT_THROW((void)parseUuid("01234567_89ab-4def-8123-456789abcdef"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid("01234567-89ab_4def-8123-456789abcdef"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid("01234567-89ab-4def_8123-456789abcdef"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid("01234567-89ab-4def-8123_456789abcdef"), std::invalid_argument);
}

TEST(TestUuid, ParseUuidRejectsNonHexCharacters)
{
    EXPECT_THROW((void)parseUuid("g1234567-89ab-4def-8123-456789abcdef"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid("01234567-89ab-4def-8123-456789abcdeZ"), std::invalid_argument);
    EXPECT_THROW((void)parseUuid("0123456 -89ab-4def-8123-456789abcdef"), std::invalid_argument);
}

TEST(TestUuid, ParseUuidAcceptsAllHexDigitRanges)
{
    // Exercises every branch of parseHexDigit: '0'-'9', 'a'-'f', 'A'-'F'.
    const auto bytes = parseUuid("01234567-89ab-cdEF-ABcd-ef0123456789");
    EXPECT_EQ(formatUuid(bytes), "01234567-89ab-cdef-abcd-ef0123456789");
}

TEST(TestUuid, ParsesOpaque128BitValuesRegardlessOfVersionOrVariant)
{
    for(const auto& id : {"01234567-89ab-3def-8123-456789abcdef",
                          "01234567-89ab-4def-4123-456789abcdef",
                          "00000000-0000-0000-0000-000000000000",
                          "ffffffff-ffff-ffff-ffff-ffffffffffff"})
    {
        EXPECT_EQ(formatUuid(parseUuid(id)), id);
    }
}

TEST(TestUuid, IsUuidV4ChecksVersionNibbleIndependentlyOfVariant)
{
    // Version nibble (byte 6 high bits) must be 0100; variant (byte 8 high bits) checked
    // separately, so wrong version with a valid-looking variant still fails.
    EXPECT_TRUE(isUuidV4(parseUuid("01234567-89ab-4000-8123-456789abcdef")));
    EXPECT_FALSE(isUuidV4(parseUuid("01234567-89ab-3000-8123-456789abcdef")));
    EXPECT_FALSE(isUuidV4(parseUuid("01234567-89ab-5000-8123-456789abcdef")));
}

TEST(TestUuid, IsUuidV4ChecksVariantNibbleIndependentlyOfVersion)
{
    // Variant nibble (byte 8 high bits) must be 10xx (0x80-0xbf); version held at v4.
    EXPECT_TRUE(isUuidV4(parseUuid("01234567-89ab-4000-8123-456789abcdef")));
    EXPECT_TRUE(isUuidV4(parseUuid("01234567-89ab-4000-bfff-456789abcdef")));
    EXPECT_FALSE(isUuidV4(parseUuid("01234567-89ab-4000-0123-456789abcdef")));
    EXPECT_FALSE(isUuidV4(parseUuid("01234567-89ab-4000-c123-456789abcdef")));
}

TEST(TestUuid, IsUuidV4RejectsNilAndMaxUuids)
{
    EXPECT_FALSE(isUuidV4(ALL_ZERO_BYTES));
    EXPECT_FALSE(isUuidV4(ALL_ONES_BYTES));
}

TEST(TestUuid, ToFlatbufferUuidRoundTripsThroughEqualityOperator)
{
    const auto bytes = parseUuid("01234567-89ab-4def-8123-456789abcdef");
    const auto lhs = toFlatbufferUuid(bytes);
    const auto rhs = toFlatbufferUuid(bytes);
    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(toUuidBytes(lhs), bytes);
}

TEST(TestUuid, ToFlatbufferUuidPreservesByteOrder)
{
    UuidBytes bytes{};
    for(size_t i = 0; i < bytes.size(); ++i)
    {
        bytes[i] = static_cast<uint8_t>(i);
    }
    EXPECT_EQ(toUuidBytes(toFlatbufferUuid(bytes)), bytes);
}

TEST(TestUuid, DistinctUuidsCompareUnequal)
{
    const auto a = toFlatbufferUuid(parseUuid("01234567-89ab-4def-8123-456789abcdef"));
    const auto b = toFlatbufferUuid(parseUuid("00000000-0000-0000-0000-000000000000"));
    EXPECT_NE(a, b);
}
