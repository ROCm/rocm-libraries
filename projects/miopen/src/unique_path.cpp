// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <cstring>
#include <array>
#include <random>

#include <miopen/unique_path.hpp>

namespace {

using DataBlock = std::array<uint64_t, 2>;

void generate_random_data_block(DataBlock& buf)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<std::remove_reference_t<decltype(buf[0])>> distrib(0);

    for(auto& x : buf)
    {
        x = distrib(gen);
    }
}

#ifdef _WIN32
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
    DataBlock ran;

    const constexpr unsigned int max_nibbles =
        2u * ran.size() * sizeof(ran[0]); // 4-bits per nibble
    unsigned int nibbles_used = max_nibbles;

    for(auto& sch : s)
    {
        if(sch == percent) // digit request
        {
            if(nibbles_used == max_nibbles)
            {
                generate_random_data_block(ran);
                nibbles_used = 0;
            }

            unsigned int c = reinterpret_cast<const uint8_t*>(ran.data())[nibbles_used / 2u];
            c >>= 4u * (nibbles_used++ & 1u); // if odd, shift right 1 nibble
            sch = hex[c & 0xf];               // convert to hex digit and replace
        }
    }

    return s;
}

} // namespace miopen
