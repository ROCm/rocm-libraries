// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Derived from LekKit's SHA-256 (https://github.com/LekKit/sha256) at commit
// c6cf8e509a713ef28c0b1b2f12578735b6402323, adapted to this project's naming, container
// and constness conventions. The algorithm is unchanged; TestDigest.cpp pins it to the
// FIPS 180-2 vectors and to what hashlib.sha256 produces, which is what makes the
// adaptation checkable. Upstream's licence follows and applies to the derived work:
//
//     MIT License
//
//     Copyright (c) 2020 LekKit https://github.com/LekKit
//
//     Permission is hereby granted, free of charge, to any person obtaining a copy
//     of this software and associated documentation files (the "Software"), to deal
//     in the Software without restriction, including without limitation the rights
//     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//     copies of the Software, and to permit persons to whom the Software is
//     furnished to do so, subject to the following conditions:
//
//     The above copyright notice and this permission notice shall be included in all
//     copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//     SOFTWARE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace hip_kernel_provider::utilities
{

/// Streaming SHA-256 state. Pass it to sha256Init before any other call.
struct Sha256Buff
{
    std::uint64_t dataSize;
    std::array<std::uint32_t, 8> h;
    std::array<std::uint8_t, 64> lastChunk;
    std::uint8_t chunkSize;
};

/// Initialisation, must be called before any further use.
void sha256Init(Sha256Buff& buff);

/// Process a block of data of arbitrary length; usable across a stream of calls.
void sha256Update(Sha256Buff& buff, const void* data, std::size_t size);

/// Produce the final digest values, ready to be read. Reusing the buffer afterwards
/// means calling sha256Init again.
void sha256Finalize(Sha256Buff& buff);

/// Read the digest into a 32-byte binary array.
void sha256Read(const Sha256Buff& buff, std::uint8_t* hash);

/// Read the digest into 64 hex characters, without a terminator.
void sha256ReadHex(const Sha256Buff& buff, char* hex);

/// Hash one contiguous block and read the digest into a 32-byte binary array.
void sha256EasyHash(const void* data, std::size_t size, std::uint8_t* hash);

/// Hash one contiguous block and read the digest into 64 hex characters, without a
/// terminator.
void sha256EasyHashHex(const void* data, std::size_t size, char* hex);

} // namespace hip_kernel_provider::utilities
