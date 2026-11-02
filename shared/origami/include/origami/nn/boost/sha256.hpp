// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

#include "origami/origami_export.h"

namespace origami::nn::boost {

/// Lowercase hex sha256 of a byte buffer (the .orr checksum and the kernel name digests).
ORIGAMI_EXPORT std::string sha256_hex(const void* data, std::size_t size);

}  // namespace origami::nn::boost
