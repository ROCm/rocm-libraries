// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for benchmark_collective.hpp: no GPU/HIP calls, pure logic.

#include "benchmark_collective.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

using hipblaslt_bench::CollectiveAgreement;
using hipblaslt_bench::MaxOp;
using hipblaslt_bench::MinOp;

namespace
{
    // Stands in for a rank group: the caller's contribution lands in slot 0 and
    // the other ranks' fixed contributions follow it.
    template <typename T>
    CollectiveAgreement group(std::vector<T> others)
    {
        CollectiveAgreement a;
        a.world     = uint32_t(others.size() + 1);
        a.allgather = [others](const void* send, void* recv, size_t bytes) {
            T* const out = static_cast<T*>(recv);
            std::memcpy(out, send, bytes);
            for(size_t j = 0; j < others.size(); ++j)
                out[j + 1] = others[j];
            return true;
        };
        return a;
    }

    CollectiveAgreement unreachable_group()
    {
        CollectiveAgreement a;
        a.world     = 2;
        a.allgather = [](const void*, void*, size_t) { return false; };
        return a;
    }
} // namespace

TEST(benchmark_collective_smoke, empty_agreement_is_identity)
{
    const CollectiveAgreement none;

    EXPECT_EQ(none.agree(12.5, MaxOp{}), 12.5);
    EXPECT_EQ(none.agree(12.5, MinOp{}), 12.5);
    EXPECT_TRUE(none.agree(true, std::logical_and<>{}));
    EXPECT_FALSE(none.agree(false, std::logical_and<>{}));
    EXPECT_TRUE(none.agree(true, std::logical_or<>{}));
    EXPECT_FALSE(none.agree(false, std::logical_or<>{}));
}

TEST(benchmark_collective_smoke, value_reduces_both_directions)
{
    const CollectiveAgreement a = group<double>({3.0, 7.0});

    EXPECT_EQ(a.agree(5.0, MaxOp{}), 7.0);
    EXPECT_EQ(a.agree(5.0, MinOp{}), 3.0);
}

TEST(benchmark_collective_smoke, flag_all_requires_every_rank)
{
    EXPECT_TRUE(group<bool>({true, true}).agree(true, std::logical_and<>{}));
    EXPECT_FALSE(group<bool>({true, false}).agree(true, std::logical_and<>{}));
}

TEST(benchmark_collective_smoke, flag_any_takes_one_rank)
{
    EXPECT_TRUE(group<bool>({false, true}).agree(false, std::logical_or<>{}));
    EXPECT_FALSE(group<bool>({false, false}).agree(false, std::logical_or<>{}));
}

TEST(benchmark_collective_smoke, failed_allgather_throws)
{
    const CollectiveAgreement a = unreachable_group();

    EXPECT_THROW(a.agree(1.0, MaxOp{}), std::runtime_error);
    EXPECT_THROW(a.agree(true, std::logical_and<>{}), std::runtime_error);
}
