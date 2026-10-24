// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// In-tree tests always track the current experimental API version; external
// consumers must pin the literal version they have reviewed instead.
#define HIPBLASLT_ENABLE_EXPERIMENTAL_API HIPBLASLT_EXPERIMENTAL_API_VERSION
#include <hipblaslt/hipblaslt-experimental.h>

#include <gtest/gtest.h>

#include <limits>

namespace
{
    // Experimental enumerators are cast to stable (int-sized) enum types, so the
    // marker bit must stay a positive int.
    static_assert(HIPBLASLT_EXPERIMENTAL_ENUM_BIT > 0
                      && HIPBLASLT_EXPERIMENTAL_ENUM_BIT <= std::numeric_limits<int>::max(),
                  "HIPBLASLT_EXPERIMENTAL_ENUM_BIT must fit in a positive int");

    TEST(ExperimentalApiSmoke, RuntimeVersionMatchesHeader)
    {
        EXPECT_EQ(hipblasLtExperimentalGetApiVersion(), HIPBLASLT_EXPERIMENTAL_API_VERSION);
    }

    TEST(ExperimentalApiSmoke, StableEpiloguesDoNotUseExperimentalBit)
    {
        // hipblasLtEpilogue_t is the stable enum with the largest values; guard
        // its highest current enumerators against colliding with the marker bit.
        for(int e : {static_cast<int>(HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT),
                     static_cast<int>(HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT)})
        {
            EXPECT_EQ(e & HIPBLASLT_EXPERIMENTAL_ENUM_BIT, 0);
        }
    }
}
