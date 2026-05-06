// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILLI_PLUGIN_INCLUDE_FUSILLI_SERIALIZED_PLAN_PAYLOAD_H
#define FUSILLI_PLUGIN_INCLUDE_FUSILLI_SERIALIZED_PLAN_PAYLOAD_H

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace fusilli_plugin::serialized_plan_payload {

constexpr std::array<uint8_t, 8> MAGIC = {
    static_cast<uint8_t>('F'), static_cast<uint8_t>('U'),
    static_cast<uint8_t>('S'), static_cast<uint8_t>('P'),
    static_cast<uint8_t>('L'), static_cast<uint8_t>('A'),
    static_cast<uint8_t>('N'), 0};
constexpr uint16_t CURRENT_FORMAT_MAJOR = 1;
constexpr uint16_t CURRENT_FORMAT_MINOR = 0;
constexpr uint32_t PAYLOAD_KIND_HIPDNN_GRAPH = 1;
constexpr size_t HEADER_SIZE = 20;

enum class DecodeStatus {
  Success,
  InvalidHeader,
  UnsupportedVersion,
  UnsupportedPayloadKind,
  MalformedPayload,
};

struct DecodedPayload {
  uint32_t kind = 0;
  hipdnnPluginConstData_t bytes{nullptr, 0};
};

namespace detail {

inline void appendUint16LE(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
  bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
}

inline void appendUint32LE(std::vector<uint8_t> &bytes, uint32_t value) {
  bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
  bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
  bytes.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
  bytes.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
}

inline uint16_t readUint16LE(const uint8_t *bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8);
}

inline uint32_t readUint32LE(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

} // namespace detail

inline std::unique_ptr<std::vector<uint8_t>>
wrapHipdnnGraphPayload(const std::vector<uint8_t> &graphBytes) {
  auto payload = std::make_unique<std::vector<uint8_t>>();
  payload->reserve(HEADER_SIZE + graphBytes.size());
  payload->insert(payload->end(), MAGIC.begin(), MAGIC.end());
  detail::appendUint16LE(*payload, CURRENT_FORMAT_MAJOR);
  detail::appendUint16LE(*payload, CURRENT_FORMAT_MINOR);
  detail::appendUint32LE(*payload, PAYLOAD_KIND_HIPDNN_GRAPH);
  detail::appendUint32LE(*payload, static_cast<uint32_t>(HEADER_SIZE));
  payload->insert(payload->end(), graphBytes.begin(), graphBytes.end());
  return payload;
}

inline DecodeStatus unwrapPayload(const hipdnnPluginConstData_t &serialized,
                                  DecodedPayload &decoded) {
  const auto *bytes = static_cast<const uint8_t *>(serialized.ptr);
  if (bytes == nullptr || serialized.size < HEADER_SIZE ||
      std::memcmp(bytes, MAGIC.data(), MAGIC.size()) != 0) {
    return DecodeStatus::InvalidHeader;
  }

  const auto major = detail::readUint16LE(bytes + 8);
  if (major != CURRENT_FORMAT_MAJOR) {
    return DecodeStatus::UnsupportedVersion;
  }

  const auto payloadKind = detail::readUint32LE(bytes + 12);
  if (payloadKind != PAYLOAD_KIND_HIPDNN_GRAPH) {
    return DecodeStatus::UnsupportedPayloadKind;
  }

  const auto headerSize = static_cast<size_t>(detail::readUint32LE(bytes + 16));
  if (headerSize < HEADER_SIZE || headerSize >= serialized.size) {
    return DecodeStatus::MalformedPayload;
  }

  decoded.kind = payloadKind;
  decoded.bytes.ptr = bytes + headerSize;
  decoded.bytes.size = serialized.size - headerSize;
  return DecodeStatus::Success;
}

} // namespace fusilli_plugin::serialized_plan_payload

#endif // FUSILLI_PLUGIN_INCLUDE_FUSILLI_SERIALIZED_PLAN_PAYLOAD_H
