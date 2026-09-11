// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/// @file Sha256.hpp
/// @brief Minimal SHA-256, used to fingerprint a UHD's feature signature
/// (RFC 0019 §6.3). No external dependencies, so the hash a provider computes
/// cannot drift with a third-party library version.
///
/// Based on RFC 6234 / FIPS 180-4.
namespace hipdnn_plugin_sdk::ingestor::uhd
{

namespace detail
{

// `inline` is load-bearing on the table below: a namespace-scope `constexpr` *variable*
// in a header otherwise has internal linkage, so each including translation unit gets its
// own copy. The `constexpr` functions need no such marking -- they are implicitly inline.
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

    // Pre-processing: padding
    std::vector<uint8_t> msg(data, data + length);
    const uint64_t bitLen = msg.size() * 8;
    msg.push_back(0x80);
    while((msg.size() % 64) != 56)
    {
        msg.push_back(0x00);
    }
    for(int i = 7; i >= 0; --i)
    {
        msg.push_back(static_cast<uint8_t>((bitLen >> (i * 8)) & 0xFF));
    }

    // Process each 512-bit chunk
    for(size_t chunk = 0; chunk < msg.size(); chunk += 64)
    {
        std::array<uint32_t, 64> w{};
        for(size_t i = 0; i < 16; ++i)
        {
            w[i] = (static_cast<uint32_t>(msg[chunk + i * 4]) << 24)
                   | (static_cast<uint32_t>(msg[chunk + i * 4 + 1]) << 16)
                   | (static_cast<uint32_t>(msg[chunk + i * 4 + 2]) << 8)
                   | (static_cast<uint32_t>(msg[chunk + i * 4 + 3]));
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

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for(auto hv : h)
    {
        oss << std::setw(8) << hv;
    }
    return oss.str();
}

} // namespace detail

/// Compute SHA-256 hash of a byte buffer.
/// Returns lowercase hex string (64 characters).
inline std::string sha256(const uint8_t* data, size_t size)
{
    return detail::sha256Impl(data, size);
}

/// Compute SHA-256 hash of a string.
/// Returns lowercase hex string (64 characters).
inline std::string sha256(const std::string& input)
{
    return detail::sha256Impl(reinterpret_cast<const uint8_t*>(input.data()), input.size());
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
