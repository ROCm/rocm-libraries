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
/// Declared rather than defined here so consumers need only this header, not the
/// SHA-256 implementation behind it.
std::string sha256Hex(const void* data, std::size_t size);

} // namespace hip_kernel_provider::utilities
