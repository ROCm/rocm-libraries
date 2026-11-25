// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace hip_kernel_provider::utilities
{

/// XXH64 of a buffer, seed 0: a fast non-cryptographic checksum (several GB/s) for detecting
/// accidental damage, such as a truncated or bit-flipped cache entry. Not a defence against a
/// deliberate change; content identity uses SHA-256 (Digest.hpp).
std::uint64_t xxh64(const void* data, std::size_t size);

} // namespace hip_kernel_provider::utilities
