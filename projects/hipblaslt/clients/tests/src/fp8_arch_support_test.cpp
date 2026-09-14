// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only smoke test for the FP8-capable architecture list, a pure function
// with no HIP dependency (see rocblaslt_fp8_arch.hpp). Included by relative
// path so the white-box test stays self-contained without adding the internal
// rocblaslt include dir to the whole test target.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_fp8_arch.hpp"

namespace
{
    TEST(Fp8ArchSupportSmoke, CdnaArchesSupportFp8)
    {
        // CDNA3/CDNA4 reach FP8 through MFMA.
        EXPECT_TRUE(rocblaslt_arch_supports_fp8("gfx942"));
        EXPECT_TRUE(rocblaslt_arch_supports_fp8("gfx950"));
    }

    TEST(Fp8ArchSupportSmoke, Rdna4ArchesSupportFp8)
    {
        // RDNA4 exposes the same OCP E4M3FN/E5M2 formats through WMMA.
        EXPECT_TRUE(rocblaslt_arch_supports_fp8("gfx1200"));
        EXPECT_TRUE(rocblaslt_arch_supports_fp8("gfx1201"));
    }

    TEST(Fp8ArchSupportSmoke, ArchesWithoutFp8HardwareAreRejected)
    {
        // No FP8 matrix hardware: CDNA1/CDNA2, and RDNA2/RDNA3 which have
        // WMMA but no FP8 variant of it.
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx908"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx90a"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx1030"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx1100"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx1101"));
    }

    TEST(Fp8ArchSupportSmoke, MatchIsExactNotPrefix)
    {
        // The list is compared whole-string: a longer name that merely starts
        // with a supported one must not be accepted, or a future part could
        // silently inherit FP8 dispatch it cannot execute.
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx94"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx9420"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8("gfx12010"));
        EXPECT_FALSE(rocblaslt_arch_supports_fp8(""));
    }
}
