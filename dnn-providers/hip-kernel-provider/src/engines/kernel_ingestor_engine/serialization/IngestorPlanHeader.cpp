// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstring>
#include <string_view>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{
namespace
{

constexpr size_t MARKER_OFFSET = 0;
constexpr size_t FORMAT_MAJOR_OFFSET = MARKER_OFFSET + INGESTOR_PLAN_MARKER.size();
constexpr size_t FORMAT_MINOR_OFFSET = FORMAT_MAJOR_OFFSET + sizeof(uint16_t);
constexpr size_t KIND_OFFSET = FORMAT_MINOR_OFFSET + sizeof(uint16_t);
constexpr size_t HEADER_SIZE_OFFSET = KIND_OFFSET + sizeof(uint16_t);
constexpr size_t RESERVED_OFFSET = HEADER_SIZE_OFFSET + sizeof(uint16_t);
constexpr size_t FIXED_FIELDS_SIZE = RESERVED_OFFSET + sizeof(uint32_t);

static_assert(FIXED_FIELDS_SIZE == INGESTOR_PLAN_DIGEST_OFFSET,
              "The body digest must follow the fixed header fields.");
static_assert(INGESTOR_PLAN_HEADER_SIZE <= UINT16_MAX,
              "The header size must fit its 16-bit header field.");

constexpr std::string_view HEX_DIGITS = "0123456789abcdef";

void writeU16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value)
{
    bytes[offset] = static_cast<uint8_t>(value & 0xFFU);
    bytes[offset + 1] = static_cast<uint8_t>(value >> 8U);
}

uint16_t readU16(const uint8_t* data, size_t offset)
{
    std::array<uint8_t, sizeof(uint16_t)> bytes{};
    std::memcpy(bytes.data(), data + offset, bytes.size());
    return static_cast<uint16_t>(static_cast<unsigned>(bytes[0])
                                 | (static_cast<unsigned>(bytes[1]) << 8U));
}

std::string formatVersion(uint16_t major, uint16_t minor)
{
    return std::to_string(major) + "." + std::to_string(minor);
}

int hexDigitValue(char digit)
{
    const auto position = HEX_DIGITS.find(digit);
    return position == std::string_view::npos ? -1 : static_cast<int>(position);
}

} // namespace

std::vector<uint8_t> encodeIngestorPlanHeader(IngestorPlanKind kind,
                                              const IngestorPlanDigest& bodyDigest)
{
    std::vector<uint8_t> bytes(INGESTOR_PLAN_HEADER_SIZE, 0);
    std::memcpy(
        bytes.data() + MARKER_OFFSET, INGESTOR_PLAN_MARKER.data(), INGESTOR_PLAN_MARKER.size());
    writeU16(bytes, FORMAT_MAJOR_OFFSET, INGESTOR_PLAN_FORMAT_MAJOR);
    writeU16(bytes, FORMAT_MINOR_OFFSET, INGESTOR_PLAN_FORMAT_MINOR);
    writeU16(bytes, KIND_OFFSET, static_cast<uint16_t>(kind));
    writeU16(bytes, HEADER_SIZE_OFFSET, static_cast<uint16_t>(INGESTOR_PLAN_HEADER_SIZE));
    std::memcpy(bytes.data() + INGESTOR_PLAN_DIGEST_OFFSET, bodyDigest.data(), bodyDigest.size());
    return bytes;
}

IngestorPlanHeader decodeIngestorPlanHeader(const uint8_t* data, size_t size)
{
    if(data == nullptr || size < FIXED_FIELDS_SIZE)
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "truncated: the payload holds " + std::to_string(size)
                               + " bytes, and the fixed header fields need "
                               + std::to_string(FIXED_FIELDS_SIZE));
    }

    if(std::memcmp(data + MARKER_OFFSET, INGESTOR_PLAN_MARKER.data(), INGESTOR_PLAN_MARKER.size())
       != 0)
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "not an ingestor plan: the payload does not start with the HKIP "
                           "marker");
    }

    IngestorPlanHeader header;
    header.formatMajor = readU16(data, FORMAT_MAJOR_OFFSET);
    header.formatMinor = readU16(data, FORMAT_MINOR_OFFSET);
    if(!detail::isReadableIngestorPlanVersion(header.formatMajor,
                                              header.formatMinor,
                                              INGESTOR_PLAN_FORMAT_MAJOR,
                                              INGESTOR_PLAN_FORMAT_MINOR))
    {
        refuseIngestorPlan(
            IngestorPlanRefusal::INCOMPATIBLE,
            "payload format version " + formatVersion(header.formatMajor, header.formatMinor)
                + " is not readable; this provider reads format versions "
                + formatVersion(INGESTOR_PLAN_FORMAT_MAJOR, 0) + " to "
                + formatVersion(INGESTOR_PLAN_FORMAT_MAJOR, INGESTOR_PLAN_FORMAT_MINOR));
    }

    const uint16_t kind = readU16(data, KIND_OFFSET);
    if(kind != static_cast<uint16_t>(IngestorPlanKind::SINGLE_KERNEL_PLAN))
    {
        refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                           "unknown ingestor plan kind " + std::to_string(kind));
    }
    header.kind = IngestorPlanKind::SINGLE_KERNEL_PLAN;

    header.headerSize = readU16(data, HEADER_SIZE_OFFSET);
    if(header.headerSize < INGESTOR_PLAN_HEADER_SIZE
       || header.headerSize % INGESTOR_PLAN_BODY_ALIGNMENT != 0 || header.headerSize > size)
    {
        refuseIngestorPlan(
            IngestorPlanRefusal::DAMAGED,
            "header size " + std::to_string(header.headerSize) + " is invalid for a payload of "
                + std::to_string(size) + " bytes; it must be at least "
                + std::to_string(INGESTOR_PLAN_HEADER_SIZE) + ", a multiple of "
                + std::to_string(INGESTOR_PLAN_BODY_ALIGNMENT) + " and no larger than the payload");
    }

    std::memcpy(
        header.bodyDigest.data(), data + INGESTOR_PLAN_DIGEST_OFFSET, header.bodyDigest.size());
    return header;
}

namespace detail
{

bool isReadableIngestorPlanVersion(uint16_t major,
                                   uint16_t minor,
                                   uint16_t readerMajor,
                                   uint16_t readerMinor)
{
    return major == readerMajor && minor <= readerMinor;
}

IngestorPlanDigest ingestorPlanDigestFromHex(const std::string& hex)
{
    IngestorPlanDigest digest{};
    if(hex.size() != digest.size() * 2)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "a SHA-256 digest needs " + std::to_string(digest.size() * 2)
                + " hex characters, but the text has " + std::to_string(hex.size()));
    }
    for(size_t index = 0; index < digest.size(); ++index)
    {
        const int high = hexDigitValue(hex[2 * index]);
        const int low = hexDigitValue(hex[(2 * index) + 1]);
        if(high < 0 || low < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "a SHA-256 digest contains a character that is not hex: '" + hex + "'");
        }
        digest[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return digest;
}

std::string ingestorPlanDigestToHex(const IngestorPlanDigest& digest)
{
    std::string hex;
    hex.reserve(digest.size() * 2);
    for(const uint8_t byte : digest)
    {
        hex += HEX_DIGITS[static_cast<size_t>(byte >> 4U)];
        hex += HEX_DIGITS[static_cast<size_t>(byte & 0x0FU)];
    }
    return hex;
}

} // namespace detail

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
