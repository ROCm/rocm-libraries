// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
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

    TEST(ExperimentalApiSmoke, RuntimeRevisionMatchesHeader)
    {
        // In-tree tests always track the current revision; consumers compare against the
        // literal revision they wrote their code for.
        EXPECT_EQ(hipblasLtExperimentalGetFeatureRevision("DEMO"), HIPBLASLT_EXPERIMENTAL_DEMO);
    }

    TEST(ExperimentalApiSmoke, UnknownFeatureHasRevisionZero)
    {
        EXPECT_EQ(hipblasLtExperimentalGetFeatureRevision("NO_SUCH_FEATURE"), 0);
        EXPECT_EQ(hipblasLtExperimentalGetFeatureRevision("demo"), 0);
        EXPECT_EQ(hipblasLtExperimentalGetFeatureRevision(""), 0);
        EXPECT_EQ(hipblasLtExperimentalGetFeatureRevision(nullptr), 0);
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
