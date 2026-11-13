#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace hipconv
{

constexpr uint32_t rotl32(uint32_t x, int8_t r)
{
    return (x << r) | (x >> (32 - r));
}

// MurmurHash3 by Austin Appleby, public domain
// https://github.com/aappleby/smhasher
constexpr auto murmur3_32_c(const char* key, size_t len, std::uint32_t seed = 0) -> std::uint32_t
{
    const int nblocks = len / 4;

    std::uint32_t h1 = seed;

    const std::uint32_t c1 = 0xcc9e2d51;
    const std::uint32_t c2 = 0x1b873593;

    // Body — process 4-byte blocks
    for(int i = 0; i < nblocks; i++)
    {
        const size_t j   = i * 4;
        std::uint32_t k1 = std::bit_cast<std::uint32_t>(
            std::array<char, 4>{key[j], key[j + 1], key[j + 2], key[j + 3]});

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    // Tail — remaining bytes
    const char* tail = key + nblocks * 4;
    std::uint32_t k1 = 0;

    switch(len & 3)
    {
    case 3:
        k1 ^= static_cast<std::uint8_t>(tail[2]) << 16;
        [[fallthrough]];
    case 2:
        k1 ^= static_cast<std::uint8_t>(tail[1]) << 8;
        [[fallthrough]];
    case 1:
        k1 ^= static_cast<std::uint8_t>(tail[0]);
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    }

    // Finalization
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}

constexpr auto murmur3_32(std::string_view str, std::uint32_t seed = 0) -> std::uint32_t
{
    return murmur3_32_c(str.data(), str.size(), seed);
}

constexpr auto operator""_murmur3_32(char const* str, std::size_t size) -> std::uint32_t
{
    return murmur3_32_c(str, size, 0);
}

} // namespace hipconv
