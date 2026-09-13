// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Derived from LekKit's SHA-256 (https://github.com/LekKit/sha256); see sha256.hpp for
// the provenance note and upstream's licence. Algorithm details:
// https://en.wikipedia.org/wiki/SHA-2

#include "utilities/sha256.hpp"

#include <cstring>

namespace hip_kernel_provider::utilities
{

namespace
{

constexpr std::size_t CHUNK_BYTES = 64;
constexpr std::size_t DIGEST_BYTES = 32;

/// The first 32 bits of the fractional parts of the cube roots of the first 64 primes.
constexpr std::array<std::uint32_t, 64> ROUND_CONSTANTS = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

/// std::rotr would say this directly, but this provider builds as C++17 and that is C++20.
constexpr std::uint32_t rotateRight(std::uint32_t value, std::uint32_t bits)
{
    return (value >> bits) | (value << (32U - bits));
}

void sha256CalcChunk(Sha256Buff& buff, const std::uint8_t* chunk)
{
    std::array<std::uint32_t, 64> w{};
    std::array<std::uint32_t, 8> tv{};

    for(std::size_t i = 0; i < 16; ++i)
    {
        w[i] = static_cast<std::uint32_t>(chunk[0]) << 24
               | static_cast<std::uint32_t>(chunk[1]) << 16
               | static_cast<std::uint32_t>(chunk[2]) << 8 | static_cast<std::uint32_t>(chunk[3]);
        chunk += 4;
    }

    for(std::size_t i = 16; i < 64; ++i)
    {
        const std::uint32_t sigma0
            = rotateRight(w[i - 15], 7) ^ rotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3U);
        const std::uint32_t sigma1
            = rotateRight(w[i - 2], 17) ^ rotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10U);
        w[i] = w[i - 16] + sigma0 + w[i - 7] + sigma1;
    }

    for(std::size_t i = 0; i < 8; ++i)
    {
        tv[i] = buff.h[i];
    }

    for(std::size_t i = 0; i < 64; ++i)
    {
        const std::uint32_t sum1
            = rotateRight(tv[4], 6) ^ rotateRight(tv[4], 11) ^ rotateRight(tv[4], 25);
        const std::uint32_t ch = (tv[4] & tv[5]) ^ (~tv[4] & tv[6]);
        const std::uint32_t temp1 = tv[7] + sum1 + ch + ROUND_CONSTANTS[i] + w[i];
        const std::uint32_t sum0
            = rotateRight(tv[0], 2) ^ rotateRight(tv[0], 13) ^ rotateRight(tv[0], 22);
        const std::uint32_t maj = (tv[0] & tv[1]) ^ (tv[0] & tv[2]) ^ (tv[1] & tv[2]);
        const std::uint32_t temp2 = sum0 + maj;

        tv[7] = tv[6];
        tv[6] = tv[5];
        tv[5] = tv[4];
        tv[4] = tv[3] + temp1;
        tv[3] = tv[2];
        tv[2] = tv[1];
        tv[1] = tv[0];
        tv[0] = temp1 + temp2;
    }

    for(std::size_t i = 0; i < 8; ++i)
    {
        buff.h[i] += tv[i];
    }
}

void binToHex(const void* data, std::size_t len, char* out)
{
    constexpr std::array<char, 16> HEX_DIGITS
        = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for(std::size_t i = 0; i < len; ++i)
    {
        const std::uint8_t c = bytes[i];
        out[i * 2] = HEX_DIGITS[c >> 4U];
        out[(i * 2) + 1] = HEX_DIGITS[c & 15U];
    }
}

} // namespace

void sha256Init(Sha256Buff& buff)
{
    buff.h[0] = 0x6a09e667;
    buff.h[1] = 0xbb67ae85;
    buff.h[2] = 0x3c6ef372;
    buff.h[3] = 0xa54ff53a;
    buff.h[4] = 0x510e527f;
    buff.h[5] = 0x9b05688c;
    buff.h[6] = 0x1f83d9ab;
    buff.h[7] = 0x5be0cd19;
    buff.dataSize = 0;
    buff.chunkSize = 0;
}

void sha256Update(Sha256Buff& buff, const void* data, std::size_t size)
{
    const auto* ptr = static_cast<const std::uint8_t*>(data);
    buff.dataSize += size;

    // Whatever is held over from the previous call is concatenated with the front of this
    // one so it can be processed as a whole chunk.
    if(size + buff.chunkSize >= CHUNK_BYTES)
    {
        const std::size_t held = buff.chunkSize;
        std::array<std::uint8_t, CHUNK_BYTES> tmpChunk{};
        std::memcpy(tmpChunk.data(), buff.lastChunk.data(), held);
        std::memcpy(tmpChunk.data() + held, ptr, CHUNK_BYTES - held);
        ptr += CHUNK_BYTES - held;
        size -= CHUNK_BYTES - held;
        buff.chunkSize = 0;
        sha256CalcChunk(buff, tmpChunk.data());
    }

    while(size >= CHUNK_BYTES)
    {
        sha256CalcChunk(buff, ptr);
        ptr += CHUNK_BYTES;
        size -= CHUNK_BYTES;
    }

    // The remainder stays in the buffer for the next call, or for finalize.
    std::memcpy(buff.lastChunk.data() + buff.chunkSize, ptr, size);
    buff.chunkSize = static_cast<std::uint8_t>(buff.chunkSize + size);
}

void sha256Finalize(Sha256Buff& buff)
{
    buff.lastChunk[buff.chunkSize] = 0x80;
    buff.chunkSize++;
    std::memset(buff.lastChunk.data() + buff.chunkSize, 0, CHUNK_BYTES - buff.chunkSize);

    // Without room left for the 64-bit length, this chunk is padded out and flushed so the
    // length lands in the next one.
    if(buff.chunkSize > 56)
    {
        sha256CalcChunk(buff, buff.lastChunk.data());
        buff.lastChunk.fill(0);
    }

    // Total size in bits, as a big-endian 64-bit value in the last eight bytes.
    std::uint64_t size = buff.dataSize * 8;
    for(std::size_t i = 8; i > 0; --i)
    {
        buff.lastChunk[55 + i] = static_cast<std::uint8_t>(size & 255U);
        size >>= 8U;
    }

    sha256CalcChunk(buff, buff.lastChunk.data());
}

void sha256Read(const Sha256Buff& buff, std::uint8_t* hash)
{
    for(std::size_t i = 0; i < 8; ++i)
    {
        hash[i * 4] = static_cast<std::uint8_t>((buff.h[i] >> 24U) & 255U);
        hash[(i * 4) + 1] = static_cast<std::uint8_t>((buff.h[i] >> 16U) & 255U);
        hash[(i * 4) + 2] = static_cast<std::uint8_t>((buff.h[i] >> 8U) & 255U);
        hash[(i * 4) + 3] = static_cast<std::uint8_t>(buff.h[i] & 255U);
    }
}

void sha256ReadHex(const Sha256Buff& buff, char* hex)
{
    std::array<std::uint8_t, DIGEST_BYTES> hash{};
    sha256Read(buff, hash.data());
    binToHex(hash.data(), DIGEST_BYTES, hex);
}

void sha256EasyHash(const void* data, std::size_t size, std::uint8_t* hash)
{
    Sha256Buff buff{};
    sha256Init(buff);
    sha256Update(buff, data, size);
    sha256Finalize(buff);
    sha256Read(buff, hash);
}

void sha256EasyHashHex(const void* data, std::size_t size, char* hex)
{
    std::array<std::uint8_t, DIGEST_BYTES> hash{};
    sha256EasyHash(data, size, hash.data());
    binToHex(hash.data(), DIGEST_BYTES, hex);
}

} // namespace hip_kernel_provider::utilities
