// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "norm.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace
{
    constexpr int64_t kM = 4, kN = 3, kLd = 5, kBatches = 2;

    // Gold is all ones; result matches it except in the batches listed in bad, where every
    // element is doubled (relative Frobenius error 1).
    struct Batches
    {
        std::vector<std::vector<double>> gold, result;
        std::vector<void*>               goldPtrs, resultPtrs;

        explicit Batches(std::vector<int> const& bad)
            : gold(kBatches, std::vector<double>(kLd * kN, 1.0))
            , result(gold)
        {
            for(int b : bad)
                for(auto& v : result[b])
                    v = 2.0;
            for(int b = 0; b < kBatches; b++)
            {
                goldPtrs.push_back(gold[b].data());
                resultPtrs.push_back(result[b].data());
            }
        }

        // The same batches packed contiguously, as strided mode sees them.
        double stridedError() const
        {
            std::vector<double> g, r;
            for(int b = 0; b < kBatches; b++)
            {
                g.insert(g.end(), gold[b].begin(), gold[b].end());
                r.insert(r.end(), result[b].begin(), result[b].end());
            }
            return norm_check_general(
                'F', kM, kN, kLd, kLd * kN, g.data(), r.data(), kBatches, HIP_R_64F);
        }

        double pointerArrayError() const
        {
            return norm_check_pointer_array('F', kM, kN, kLd, goldPtrs, resultPtrs, HIP_R_64F);
        }
    };
} // namespace

// A wrong D in any batch, not just the last, must fail the pointer-array norm check.
TEST(HostUnitNormCheck, smoke_PointerArrayReportsErrorInFirstBatch)
{
    Batches batches({0});
    EXPECT_GT(batches.pointerArrayError(), 0.5);
}

TEST(HostUnitNormCheck, smoke_PointerArrayReportsErrorInLastBatch)
{
    Batches batches({kBatches - 1});
    EXPECT_GT(batches.pointerArrayError(), 0.5);
}

TEST(HostUnitNormCheck, smoke_PointerArrayMatchesStrided)
{
    for(auto bad : std::vector<std::vector<int>>{{}, {0}, {1}, {0, 1}})
    {
        Batches batches(bad);
        EXPECT_DOUBLE_EQ(batches.pointerArrayError(), batches.stridedError())
            << "bad batches: " << bad.size();
    }
}
