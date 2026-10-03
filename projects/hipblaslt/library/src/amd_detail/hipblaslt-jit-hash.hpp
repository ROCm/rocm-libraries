// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace hipblaslt_jit
{
    // 64-bit FNV-1a for names and change detection; it is not collision resistant.
    class Fnv1a
    {
    public:
        // Ends each field with a separator so that field boundaries change the hash.
        Fnv1a& add(std::string_view field) noexcept
        {
            for(unsigned char c : field)
                m_hash = (m_hash ^ c) * 1099511628211ull;
            m_hash = (m_hash ^ 0xff) * 1099511628211ull;
            return *this;
        }
        std::string hex() const
        {
            char text[17];
            std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(m_hash));
            return text;
        }

    private:
        uint64_t m_hash = 14695981039346656037ull;
    };

    // SHA-256 of bytes as 64 lowercase hex digits (FIPS 180-4).
    inline std::string sha256Hex(std::string_view bytes)
    {
        static constexpr uint32_t k[64]
            = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
               0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
               0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
               0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
               0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
               0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
               0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
               0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
               0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
               0xc67178f2};
        uint32_t   h[8] = {0x6a09e667,
                         0xbb67ae85,
                         0x3c6ef372,
                         0xa54ff53a,
                         0x510e527f,
                         0x9b05688c,
                         0x1f83d9ab,
                         0x5be0cd19};
        const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
        std::string message(bytes);
        const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8;
        message.push_back(static_cast<char>(0x80));
        while(message.size() % 64 != 56)
            message.push_back('\0');
        for(int shift = 56; shift >= 0; shift -= 8)
            message.push_back(static_cast<char>((bits >> shift) & 0xff));
        for(size_t block = 0; block < message.size(); block += 64)
        {
            uint32_t w[64];
            for(int i = 0; i < 16; ++i)
            {
                w[i] = 0;
                for(int j = 0; j < 4; ++j)
                    w[i] = (w[i] << 8) | static_cast<unsigned char>(message[block + 4 * i + j]);
            }
            for(int i = 16; i < 64; ++i)
            {
                const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i]          = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t v[8];
            for(int i = 0; i < 8; ++i)
                v[i] = h[i];
            for(int i = 0; i < 64; ++i)
            {
                const auto s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
                const auto t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
                const auto s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
                const auto t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
                for(int j = 7; j > 0; --j)
                    v[j] = v[j - 1];
                v[4] += t1;
                v[0] = t1 + t2;
            }
            for(int i = 0; i < 8; ++i)
                h[i] += v[i];
        }
        char text[65];
        for(int i = 0; i < 8; ++i)
            std::snprintf(text + 8 * i, 9, "%08x", h[i]);
        return text;
    }
}
