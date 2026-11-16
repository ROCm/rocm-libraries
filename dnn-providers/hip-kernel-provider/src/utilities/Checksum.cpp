// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utilities/Checksum.hpp"

namespace hip_kernel_provider::utilities
{

namespace
{

constexpr std::uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t PRIME3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

std::uint64_t rotateLeft(std::uint64_t value, int bits)
{
    return (value << bits) | (value >> (64 - bits));
}

/// Little-endian loads, so the checksum is the same on every host.
std::uint64_t load64(const unsigned char* bytes)
{
    std::uint64_t value = 0;
    for(int i = 7; i >= 0; --i)
    {
        value = (value << 8) | bytes[i];
    }
    return value;
}

std::uint64_t load32(const unsigned char* bytes)
{
    std::uint64_t value = 0;
    for(int i = 3; i >= 0; --i)
    {
        value = (value << 8) | bytes[i];
    }
    return value;
}

std::uint64_t round(std::uint64_t accumulator, std::uint64_t input)
{
    accumulator += input * PRIME2;
    accumulator = rotateLeft(accumulator, 31);
    return accumulator * PRIME1;
}

std::uint64_t mergeRound(std::uint64_t accumulator, std::uint64_t value)
{
    accumulator ^= round(0, value);
    return accumulator * PRIME1 + PRIME4;
}

} // namespace

std::uint64_t xxh64(const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    const unsigned char* const end = bytes + size;
    std::uint64_t hash = 0;

    if(size >= 32)
    {
        std::uint64_t v1 = PRIME1 + PRIME2;
        std::uint64_t v2 = PRIME2;
        std::uint64_t v3 = 0;
        std::uint64_t v4 = 0 - PRIME1;
        const unsigned char* const limit = end - 32;
        do
        {
            v1 = round(v1, load64(bytes));
            v2 = round(v2, load64(bytes + 8));
            v3 = round(v3, load64(bytes + 16));
            v4 = round(v4, load64(bytes + 24));
            bytes += 32;
        } while(bytes <= limit);
        hash = rotateLeft(v1, 1) + rotateLeft(v2, 7) + rotateLeft(v3, 12) + rotateLeft(v4, 18);
        hash = mergeRound(hash, v1);
        hash = mergeRound(hash, v2);
        hash = mergeRound(hash, v3);
        hash = mergeRound(hash, v4);
    }
    else
    {
        hash = PRIME5;
    }
    hash += static_cast<std::uint64_t>(size);

    while(bytes + 8 <= end)
    {
        hash ^= round(0, load64(bytes));
        hash = rotateLeft(hash, 27) * PRIME1 + PRIME4;
        bytes += 8;
    }
    if(bytes + 4 <= end)
    {
        hash ^= load32(bytes) * PRIME1;
        hash = rotateLeft(hash, 23) * PRIME2 + PRIME3;
        bytes += 4;
    }
    while(bytes < end)
    {
        hash ^= static_cast<std::uint64_t>(*bytes) * PRIME5;
        hash = rotateLeft(hash, 11) * PRIME1;
        ++bytes;
    }

    hash ^= hash >> 33;
    hash *= PRIME2;
    hash ^= hash >> 29;
    hash *= PRIME3;
    hash ^= hash >> 32;
    return hash;
}

} // namespace hip_kernel_provider::utilities
