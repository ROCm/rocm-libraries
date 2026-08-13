// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace hipdnn_backend::heuristics::uhd
{

/// Compute SHA-256 hash of a byte buffer.
/// Returns lowercase hex string (64 characters).
std::string sha256(const uint8_t* data, size_t size);

/// Compute SHA-256 hash of a string.
/// Returns lowercase hex string (64 characters).
std::string sha256(const std::string& input);

} // namespace hipdnn_backend::heuristics::uhd
