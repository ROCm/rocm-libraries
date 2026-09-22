// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression test for the serialized-solution work-dimensions boundary check.
//
// Work-dimension vectors (std::vector<size_t>) deserialized from a serialized
// Solution blob via miopenLoadSolution are copied into a fixed
// std::array<size_t, 3> in the kernel constructors; unbounded input there is
// an out-of-bounds write.
//
// The trust-boundary fix rejects out-of-range work-dimension vectors during
// deserialization (SerializedSolutionKernelInfo::from_json) before any kernel
// object is constructed. That validation is factored into the exported helper
// miopen::ValidateSerializedWorkDims, which this test exercises directly. This
// keeps the test CPU-only (no GPU, no find, no full Solution round-trip).

#include <miopen/errors.hpp>
#include <miopen/solution.hpp>

#include <gtest/gtest.h>

#include <vector>

TEST(CPU_SolutionLoadWorkDims_NONE, RejectsTooManyWorkDims)
{
    // A vector with more than 3 entries is the exact malicious payload that
    // would overrun the std::array<size_t, 3> destination; it must be rejected.
    const std::vector<size_t> too_many{1, 1, 1, 1};

    EXPECT_THROW(miopen::ValidateSerializedWorkDims(too_many, "local"), miopen::Exception);
    EXPECT_THROW(miopen::ValidateSerializedWorkDims(too_many, "global"), miopen::Exception);

    // The rejection must carry the load-time "invalid value" status.
    try
    {
        miopen::ValidateSerializedWorkDims(too_many, "local");
        FAIL() << "Expected miopen::Exception for oversized work dimensions.";
    }
    catch(const miopen::Exception& e)
    {
        EXPECT_EQ(e.status, miopenStatusInvalidValue);
    }
}

TEST(CPU_SolutionLoadWorkDims_NONE, RejectsEmptyWorkDims)
{
    // An empty vector violates the pre-existing non-empty invariant.
    const std::vector<size_t> empty{};

    EXPECT_THROW(miopen::ValidateSerializedWorkDims(empty, "local"), miopen::Exception);
    EXPECT_THROW(miopen::ValidateSerializedWorkDims(empty, "global"), miopen::Exception);
}

TEST(CPU_SolutionLoadWorkDims_NONE, AcceptsValidWorkDims)
{
    // 1..3 entries fit the destination array and must be accepted.
    EXPECT_NO_THROW(miopen::ValidateSerializedWorkDims(std::vector<size_t>{256}, "local"));
    EXPECT_NO_THROW(miopen::ValidateSerializedWorkDims(std::vector<size_t>{256, 1}, "local"));
    EXPECT_NO_THROW(miopen::ValidateSerializedWorkDims(std::vector<size_t>{256, 1, 1}, "global"));
}
