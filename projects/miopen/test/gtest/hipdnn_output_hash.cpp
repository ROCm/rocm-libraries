// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Unit tests for OutputHash in hipdnn_shim_helpers.hpp, which the parity harness compares
// across its two runs. Kept out of the hipdnn_shim_*.cpp binaries and off the HipdnnShim
// filter, so the parity replay runs only GPU shim tests. No GPU is needed.

#include "hipdnn_shim_helpers.hpp"

#include <initializer_list>
#include <string>
#include <vector>

// The published FNV-1a 64 test vectors. They pin the constants and the byte order of
// OutputHash, and the split case shows that hashing in several Add calls is the same as
// hashing the joined bytes, which the multi-output parity hashes rely on.
TEST(CPU_HipdnnOutputHash_NONE, MatchesPublishedFnv1aVectors)
{
    const auto hash_of = [](std::initializer_list<std::string> parts) {
        hipdnn_shim_test::OutputHash hash;
        for(const auto& part : parts)
            hash.Add(std::vector<char>(part.begin(), part.end()));
        return hash.Hex();
    };
    EXPECT_EQ(hash_of({}), "cbf29ce484222325");
    EXPECT_EQ(hash_of({"a"}), "af63dc4c8601ec8c");
    EXPECT_EQ(hash_of({"foobar"}), "85944171f73967e8");
    EXPECT_EQ(hash_of({"foo", "bar"}), "85944171f73967e8");
}
