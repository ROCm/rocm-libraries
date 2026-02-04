// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <cstring>
#include <bit>
#include <random>
#include <type_traits>

#include "unique_path.hpp"

#if defined(_WIN32) || defined(__CYGWIN__) // Windows default, including MinGW and Cygwin
#define MIOPEN_WINDOWS_API
#else // defined(_WIN32) || defined(__CYGWIN__)
#define MIOPEN_POSIX_API
#endif // defined(_WIN32) || defined(__CYGWIN__)

namespace {

template <typename T = uint64_t, typename G = std::mt19937_64>
requires std::is_integral_v<T>
static void generate_random_data_block(void* buf, size_t len)
{
    std::random_device rd;
    G gen(rd());
    std::uniform_int_distribution<T> distrib(0);
    size_t pos{0};
    T* pBuffer{std::bit_cast<T*>(buf)};

    while(pos < len)
    {
        const T random_value{distrib(gen)};
        const size_t remaining_size{len - pos};

        if(sizeof(T) <= remaining_size)
        {
            *pBuffer++ = random_value;
            pos += sizeof(T);
        }
        else
        {
            // Copy remaining bytes manually, in case 'len' is not an exact multiple of 'sizeof(T)'.
            memcpy(pBuffer, &random_value, remaining_size);
            pos += remaining_size;
        }
    }
}

#ifdef MIOPEN_WINDOWS_API
const constexpr wchar_t hex[]   = L"0123456789abcdef";
const constexpr wchar_t percent = L'%';
#else
const constexpr char hex[]   = "0123456789abcdef";
const constexpr char percent = '%';
#endif

} // namespace

namespace miopen {

fs::path unique_path(fs::path const& model)
{
    fs::path::string_type s(model.native());

    char ran[16] = {}; // init to avoid clang static analyzer message

    const constexpr unsigned int max_nibbles = 2u * sizeof(ran); // 4-bits per nibble
    unsigned int nibbles_used                = max_nibbles;

    for(auto& sch : s)
    {
        if(sch == percent) // digit request
        {
            if(nibbles_used == max_nibbles)
            {
                generate_random_data_block(ran, sizeof(ran));
                nibbles_used = 0;
            }

            unsigned int c = ran[nibbles_used / 2u];
            c >>= 4u * (nibbles_used++ & 1u); // if odd, shift right 1 nibble
            sch = hex[c & 0xf];               // convert to hex digit and replace
        }
    }

    return s;
}

} // namespace miopen
