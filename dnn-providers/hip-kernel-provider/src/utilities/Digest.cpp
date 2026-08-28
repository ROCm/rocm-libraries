// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utilities/Digest.hpp"

#include "utilities/sha256.h"

namespace hip_kernel_provider::utilities
{

std::string sha256Hex(const void* data, std::size_t size)
{
    // sha256_read_hex writes exactly 64 characters and no terminator, so the string is
    // sized first and written into rather than read back from a bare char[64].
    std::string hex(64, '\0');
    sha256_easy_hash_hex(data, size, hex.data());
    return hex;
}

} // namespace hip_kernel_provider::utilities
