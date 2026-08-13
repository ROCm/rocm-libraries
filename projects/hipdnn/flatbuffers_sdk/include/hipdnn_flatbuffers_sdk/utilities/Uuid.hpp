// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <flatbuffers/array.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hipdnn_flatbuffers_sdk::utilities
{

using UuidBytes = std::array<uint8_t, 16>;

inline UuidBytes toUuidBytes(const data_objects::Uuid& uuid)
{
    UuidBytes bytes{};
    for(size_t i = 0; i < bytes.size(); ++i)
    {
        bytes[i] = uuid.bytes()->Get(static_cast<flatbuffers::uoffset_t>(i));
    }
    return bytes;
}

inline data_objects::Uuid toFlatbufferUuid(const UuidBytes& bytes)
{
    return {flatbuffers::span<const uint8_t, 16>(bytes)};
}

inline bool isUuidV4(const UuidBytes& bytes)
{
    return (bytes[6] & 0xf0U) == 0x40U && (bytes[8] & 0xc0U) == 0x80U;
}

inline std::string formatUuid(const UuidBytes& bytes)
{
    static constexpr std::string_view HEX = "0123456789abcdef";
    std::string result(36, '-');
    size_t output = 0;
    for(const auto byte : bytes)
    {
        if(output == 8 || output == 13 || output == 18 || output == 23)
        {
            ++output;
        }
        result[output++] = HEX[byte >> 4U];
        result[output++] = HEX[byte & 0x0fU];
    }
    return result;
}

inline UuidBytes parseUuid(std::string_view text)
{
    if(text.size() != 36 || text[8] != '-' || text[13] != '-' || text[18] != '-' || text[23] != '-')
    {
        throw std::invalid_argument("UUID must use xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx format");
    }
    const auto parseHexDigit = [](char value) -> uint8_t {
        if(value >= '0' && value <= '9')
        {
            return static_cast<uint8_t>(value - '0');
        }
        if(value >= 'a' && value <= 'f')
        {
            return static_cast<uint8_t>(value - 'a' + 10);
        }
        if(value >= 'A' && value <= 'F')
        {
            return static_cast<uint8_t>(value - 'A' + 10);
        }
        throw std::invalid_argument("UUID contains a non-hexadecimal character");
    };

    UuidBytes bytes{};
    size_t input = 0;
    for(auto& byte : bytes)
    {
        if(input == 8 || input == 13 || input == 18 || input == 23)
        {
            ++input;
        }
        const auto high = parseHexDigit(text[input++]);
        const auto low = parseHexDigit(text[input++]);
        byte = static_cast<uint8_t>((high << 4U) | low);
    }
    return bytes;
}

} // namespace hipdnn_flatbuffers_sdk::utilities
