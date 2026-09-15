// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the fused-A2A fields on Arguments.

#include "hipblaslt_arguments.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    // Reproduces hipblaslt_gentest.py's write_signature(): for each field in
    // FOR_EACH_ARGUMENT order, zero-fill the gap since the previous field's end,
    // then emit sizeof(field) canary bytes (sig ^ byte_index), advancing sig by 89
    // (mod 256) per field.
    std::vector<unsigned char> build_expected_canary()
    {
        std::vector<unsigned char> bytes;
        unsigned                   sig      = 0;
        std::ptrdiff_t             last_ofs = 0;

        auto append_field = [&](std::size_t offset, std::size_t size) {
            std::ptrdiff_t gap = static_cast<std::ptrdiff_t>(offset) - last_ofs;
            for(std::ptrdiff_t i = 0; i < gap; ++i)
                bytes.push_back(0);
            for(std::size_t i = 0; i < size; ++i)
                bytes.push_back(static_cast<unsigned char>(sig ^ i));
            sig      = (sig + 89) % 256;
            last_ofs = static_cast<std::ptrdiff_t>(offset + size);
        };

#define A2A_CANARY_FIELD(NAME) append_field(offsetof(Arguments, NAME), sizeof(Arguments::NAME))
        FOR_EACH_ARGUMENT(A2A_CANARY_FIELD, ;);
#undef A2A_CANARY_FIELD

        bytes.resize(sizeof(Arguments), 0);
        return bytes;
    }

    std::string build_validate_stream()
    {
        std::string stream("hipBLASLt");
        stream.push_back('\0');
        auto canary = build_expected_canary();
        stream.append(reinterpret_cast<const char*>(canary.data()), canary.size());
        stream += "HIPblaslT";
        stream.push_back('\0');
        return stream;
    }
} // namespace

TEST(arguments_a2a, validate_accepts_the_current_layout)
{
    std::istringstream iss(build_validate_stream());
    Arguments::validate(iss);
}

TEST(arguments_a2a, defaults_are_the_single_rank_case)
{
    Arguments arg;
    arg.init();

    EXPECT_EQ(arg.a2a_world, 1);
    EXPECT_EQ(arg.a2a_extent, 0);
}

TEST(arguments_a2a, fields_survive_a_round_trip)
{
    Arguments arg;
    arg.init();
    arg.a2a_world  = 4;
    arg.a2a_extent = 10240;

    Arguments copy;
    std::memcpy(&copy, &arg, sizeof(Arguments));

    EXPECT_EQ(copy.a2a_world, 4);
    EXPECT_EQ(copy.a2a_extent, 10240);
}
