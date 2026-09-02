// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <string>

namespace hip_kernel_provider::utilities
{

/// SHA-256 of a buffer as 64 lowercase hex characters -- the spelling
/// `hashlib.sha256(...).hexdigest()` produces, so comparing against a descriptor's
/// `kernel.source.sha256` is plain string equality.
///
/// Not inlined: the vendored header defines an unqualified global `SHA256` class, and
/// declaring it here keeps that in one translation unit rather than every consumer.
std::string sha256Hex(const void* data, std::size_t size);

} // namespace hip_kernel_provider::utilities
