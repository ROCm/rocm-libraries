// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

/// @file Sha256.hpp
/// @brief Dependency-free SHA-256 (FIPS 180-4) for UHD fingerprints (RFC 0019 §6.3), so
/// the hash cannot drift with a third-party library version.
namespace hipdnn_plugin_sdk::uhd
{

namespace detail
{

// `inline` gives the header-scope table one definition instead of one per TU.
inline constexpr std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr uint32_t rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

constexpr uint32_t ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

constexpr uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

constexpr uint32_t sigma0(uint32_t x)
{
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

constexpr uint32_t sigma1(uint32_t x)
{
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

constexpr uint32_t gamma0(uint32_t x)
{
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

constexpr uint32_t gamma1(uint32_t x)
{
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

inline constexpr size_t BLOCK_BYTES = 64;

/// Compress one 64-byte block into @p h.
inline void sha256Block(std::array<uint32_t, 8>& h, const uint8_t* block)
{
    std::array<uint32_t, 64> w{};
    for(size_t i = 0; i < 16; ++i)
    {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24)
               | (static_cast<uint32_t>(block[i * 4 + 1]) << 16)
               | (static_cast<uint32_t>(block[i * 4 + 2]) << 8)
               | (static_cast<uint32_t>(block[i * 4 + 3]));
    }
    for(size_t i = 16; i < 64; ++i)
    {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    auto a = h[0];
    auto b = h[1];
    auto c = h[2];
    auto d = h[3];
    auto e = h[4];
    auto f = h[5];
    auto g = h[6];
    auto hh = h[7];

    for(size_t i = 0; i < 64; ++i)
    {
        const uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        const uint32_t t2 = sigma0(a) + maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

inline std::string sha256Impl(const uint8_t* data, size_t length)
{
    std::array<uint32_t, 8> h = {0x6a09e667,
                                 0xbb67ae85,
                                 0x3c6ef372,
                                 0xa54ff53a,
                                 0x510e527f,
                                 0x9b05688c,
                                 0x1f83d9ab,
                                 0x5be0cd19};

    // Whole blocks are read in place; only the tail is copied, to append the padding.
    const size_t whole = length - length % BLOCK_BYTES;
    for(size_t offset = 0; offset < whole; offset += BLOCK_BYTES)
    {
        sha256Block(h, data + offset);
    }

    // The 0x80 marker and the 8-byte big-endian bit length follow the tail: one block if
    // they fit after it, else two.
    std::array<uint8_t, 2 * BLOCK_BYTES> tail{};
    const size_t rest = length - whole;
    std::copy(data + whole, data + length, tail.begin());
    tail[rest] = 0x80;
    const size_t tailBytes = rest < BLOCK_BYTES - 8 ? BLOCK_BYTES : 2 * BLOCK_BYTES;
    const uint64_t bitLen = static_cast<uint64_t>(length) * 8;
    for(size_t i = 0; i < 8; ++i)
    {
        tail[tailBytes - 1 - i] = static_cast<uint8_t>(bitLen >> (i * 8));
    }
    for(size_t offset = 0; offset < tailBytes; offset += BLOCK_BYTES)
    {
        sha256Block(h, tail.data() + offset);
    }

    // Formatted by hand rather than through a stream, whose global locale may group digits:
    // the digest is compared as text against the one Python wrote.
    constexpr std::string_view HEX_DIGITS = "0123456789abcdef";
    constexpr size_t NIBBLES_PER_WORD = 8;
    std::string digest;
    digest.reserve(h.size() * NIBBLES_PER_WORD);
    for(const auto word : h)
    {
        for(size_t nibble = NIBBLES_PER_WORD; nibble-- > 0;)
        {
            digest.push_back(HEX_DIGITS[(word >> (nibble * 4)) & 0xFU]);
        }
    }
    return digest;
}

} // namespace detail

/// SHA-256 of a byte buffer as 64 lowercase hex characters.
inline std::string sha256(const uint8_t* data, size_t size)
{
    return detail::sha256Impl(data, size);
}

/// SHA-256 of a string as 64 lowercase hex characters.
inline std::string sha256(const std::string& input)
{
    return detail::sha256Impl(reinterpret_cast<const uint8_t*>(input.data()), input.size());
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
