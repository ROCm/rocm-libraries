// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

// The header in front of a saved ingestor plan: its exact bytes, and the order of the
// reader's checks. No GPU is used.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{
namespace
{

constexpr size_t MAJOR_OFFSET = 4;
constexpr size_t MINOR_OFFSET = 6;
constexpr size_t KIND_OFFSET = 8;
constexpr size_t HEADER_SIZE_OFFSET = 10;
constexpr size_t DIGEST_OFFSET = 16;
constexpr size_t HEADER_SIZE = 48;

IngestorPlanDigest knownDigest()
{
    IngestorPlanDigest digest{};
    for(size_t index = 0; index < digest.size(); ++index)
    {
        digest[index] = static_cast<uint8_t>(0xA0U + index);
    }
    return digest;
}

std::vector<uint8_t> validHeader()
{
    return encodeIngestorPlanHeader(IngestorPlanKind::SINGLE_KERNEL_PLAN, knownDigest());
}

void setU16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value)
{
    bytes[offset] = static_cast<uint8_t>(value & 0xFFU);
    bytes[offset + 1] = static_cast<uint8_t>(value >> 8U);
}

void expectHeaderRefusal(const std::vector<uint8_t>& bytes,
                         size_t size,
                         hipdnnPluginStatus_t status,
                         const std::string& phrase)
{
    expectIngestorPlanRefusal(
        [&]() { decodeIngestorPlanHeader(bytes.data(), size); }, status, phrase);
}

TEST(TestIngestorPlanHeader, WritesTheFortyEightByteLayoutLittleEndian)
{
    // Integers are little-endian.
    std::vector<uint8_t> expected;
    const auto append = [&expected](std::initializer_list<uint8_t> bytes) {
        expected.insert(expected.end(), bytes.begin(), bytes.end());
    };
    append({0x48, 0x4B, 0x49, 0x50}); // marker HKIP
    append({0x01, 0x00}); // format major 1
    append({0x00, 0x00}); // format minor 0
    append({0x01, 0x00}); // kind SINGLE_KERNEL_PLAN
    append({0x30, 0x00}); // header size 48
    append({0x00, 0x00, 0x00, 0x00}); // reserved
    const IngestorPlanDigest digest = knownDigest();
    expected.insert(expected.end(), digest.begin(), digest.end());

    EXPECT_EQ(validHeader(), expected);
    EXPECT_EQ(INGESTOR_PLAN_HEADER_SIZE, HEADER_SIZE);
    EXPECT_EQ(INGESTOR_PLAN_DIGEST_OFFSET, DIGEST_OFFSET);
}

TEST(TestIngestorPlanHeader, RejectsAPayloadShorterThanTheFixedFields)
{
    const std::vector<uint8_t> bytes = validHeader();
    for(size_t size = 0; size < DIGEST_OFFSET; ++size)
    {
        SCOPED_TRACE("size " + std::to_string(size));
        expectHeaderRefusal(bytes, size, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "truncated");
    }
    expectIngestorPlanRefusal([]() { decodeIngestorPlanHeader(nullptr, HEADER_SIZE); },
                              HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                              "truncated");
}

TEST(TestIngestorPlanHeader, RejectsAWrongMarker)
{
    std::vector<uint8_t> bytes = validHeader();
    bytes[1] = 'X';
    expectHeaderRefusal(
        bytes, bytes.size(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "not an ingestor plan");
}

TEST(TestIngestorPlanHeader, RefusesAnotherMajorVersion)
{
    for(const uint16_t major : {uint16_t{0}, uint16_t{2}})
    {
        SCOPED_TRACE("major " + std::to_string(major));
        std::vector<uint8_t> bytes = validHeader();
        setU16(bytes, MAJOR_OFFSET, major);
        const std::string payloadVersion = "version " + std::to_string(major) + ".0 ";
        expectHeaderRefusal(
            bytes, bytes.size(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, payloadVersion);
        expectHeaderRefusal(bytes, bytes.size(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "1.0 to 1.0");
    }
}

TEST(TestIngestorPlanHeader, RefusesANewerMinorVersion)
{
    std::vector<uint8_t> bytes = validHeader();
    setU16(bytes, MINOR_OFFSET, 1);
    expectHeaderRefusal(bytes, bytes.size(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "version 1.1 ");
}

TEST(TestIngestorPlanHeader, ReadableVersionRuleAcceptsAnyMinorUpToTheReaders)
{
    for(uint16_t minor = 0; minor <= 3; ++minor)
    {
        EXPECT_TRUE(detail::isReadableIngestorPlanVersion(1, minor, 1, 3)) << "1." << minor;
    }
    EXPECT_FALSE(detail::isReadableIngestorPlanVersion(1, 4, 1, 3));
    for(uint16_t minor = 0; minor <= 4; ++minor)
    {
        EXPECT_FALSE(detail::isReadableIngestorPlanVersion(0, minor, 1, 3)) << "0." << minor;
        EXPECT_FALSE(detail::isReadableIngestorPlanVersion(2, minor, 1, 3)) << "2." << minor;
    }
}

TEST(TestIngestorPlanHeader, RefusesAnUnknownKind)
{
    for(const uint16_t kind : {uint16_t{0}, uint16_t{2}})
    {
        SCOPED_TRACE("kind " + std::to_string(kind));
        std::vector<uint8_t> bytes = validHeader();
        setU16(bytes, KIND_OFFSET, kind);
        expectHeaderRefusal(bytes,
                            bytes.size(),
                            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                            "unknown ingestor plan kind " + std::to_string(kind));
    }
}

TEST(TestIngestorPlanHeader, RejectsABadHeaderSize)
{
    // A valid header followed by padding, so a larger header size still fits the payload.
    std::vector<uint8_t> bytes = validHeader();
    bytes.resize(HEADER_SIZE + 32, 0);

    for(const uint16_t headerSize : {uint16_t{16}, uint16_t{40}, uint16_t{56}, uint16_t{96}})
    {
        SCOPED_TRACE("header size " + std::to_string(headerSize));
        setU16(bytes, HEADER_SIZE_OFFSET, headerSize);
        expectHeaderRefusal(bytes, bytes.size(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "header size");
    }

    setU16(bytes, HEADER_SIZE_OFFSET, 64);
    const IngestorPlanHeader header = decodeIngestorPlanHeader(bytes.data(), bytes.size());
    EXPECT_EQ(header.headerSize, 64);
    EXPECT_EQ(header.formatMajor, INGESTOR_PLAN_FORMAT_MAJOR);
    EXPECT_EQ(header.formatMinor, INGESTOR_PLAN_FORMAT_MINOR);
    EXPECT_EQ(header.kind, IngestorPlanKind::SINGLE_KERNEL_PLAN);
    EXPECT_EQ(header.bodyDigest, knownDigest());
}

TEST(TestIngestorPlanHeader, ChecksTheVersionBeforeAnythingElse)
{
    std::vector<uint8_t> bytes = validHeader();
    bytes.resize(HEADER_SIZE + 64, 0xEE);
    setU16(bytes, MAJOR_OFFSET, 2);
    setU16(bytes, HEADER_SIZE_OFFSET, 3);
    for(size_t index = DIGEST_OFFSET; index < HEADER_SIZE; ++index)
    {
        bytes[index] = 0;
    }
    expectHeaderRefusal(bytes, bytes.size(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "version 2.0 ");
}

TEST(TestIngestorPlanHeader, ConvertsADigestBetweenRawAndHex)
{
    const std::string hex = "a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf";
    EXPECT_EQ(detail::ingestorPlanDigestToHex(knownDigest()), hex);
    EXPECT_EQ(detail::ingestorPlanDigestFromHex(hex), knownDigest());
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
