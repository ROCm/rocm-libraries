// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for rocRollerKernelNameMatchesTile, the name-vs-tile half
// of matchesRocRollerSelectedKernel (tensile_host.cpp). That function builds the
// type/transpose prefix from a RocblasltContractionProblem and then calls
// straight through to the predicate under test here, so these cases exercise the
// shipping code rather than a copy of it. See README.md in this directory for
// the dual-target build and the HostUnit/smoke_ naming.
//
// The predicate decides which checked-in rocRoller kernel an off-build may
// substitute for a ranked tile, so a false positive silently runs the wrong
// kernel. Three distinctions are easy to lose and each has its own cases below:
//
//   StreamK   "_WGT_16x16x128_UR_" is a substring of both the plain and the
//             StreamK name of the same tile, so the tile token alone cannot tell
//             them apart. "_SK" immediately precedes "_WGT_" on a StreamK name.
//   Swizzle   The prefix match stops at "_Float_" and the tile match starts at
//             "_WGT_", so the scale tokens between them are never inspected
//             unless something looks for them. A pre-swizzled kernel carries
//             PSTA_/PTA_/PSTB_/PTB_ there.
//   WGM       A StreamK name never carries "_WGM_" even when the ranked tile has
//             workgroupMapping set, because the generator forces
//             workgroupMappingDim = -1 for every StreamK kernel.
//
// The header is included by relative path, following secure_env_gtest.cpp and
// arch_revision_gtest.cpp: putting the library's internal src/include on this
// target's include path shadows the clients' own utility.hpp (see the note in
// ../src/CMakeLists.txt). The header is dependency-free for the same reason.

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocroller_kernel_match.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace
{
    // Up to and including the trailing "_Float_", as matchesRocRollerSelectedKernel
    // builds it for a TN problem.
    const std::string fp8Prefix = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_";
    const std::string fp4Prefix = "RR_GEMM_TN_FP4_FP4_Float_Float_Float_";

    // Names as the kernels are actually checked in (MX scales on both operands).
    const char* const plainFp8_16x16
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2";
    const char* const skFp8_16x16
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2";
    const char* const skFp8_64x32
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2";
    const char* const wgmFp8_256x256
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_";
    const char* const skFp4_16x32
        = "RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2";
    const char* const skFp4_256x192
        = "RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_256x192x128_UR_4";
    // A pre-swizzled-scale kernel: identical tile token, extra PSTA_/PTA_/PSTB_/PTB_
    // tokens ahead of it. None are checked in today.
    const char* const preSwizzledFp8_16x16
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_PSTA_32x8x4_PTA_32x8_SB_BE8M0_32_"
          "PSTB_32x8x4_PTB_8x32_WGT_16x16x128_UR_2";

    // {m, n, k, workgroupMapping, streamK, nonTemporalA, nonTemporalB}. The
    // non-temporal bits are rejected by the caller, ahead of this predicate, so
    // they stay false throughout.
    constexpr RocRollerSelectedTile plainTile16x16{16, 16, 128, false, false, false, false};
    constexpr RocRollerSelectedTile skTile16x16{16, 16, 128, true, true, false, false};
    constexpr RocRollerSelectedTile skTile64x32{64, 32, 128, true, true, false, false};
    constexpr RocRollerSelectedTile wgmTile256x256{256, 256, 128, true, false, false, false};
    constexpr RocRollerSelectedTile skTile16x32{16, 32, 128, true, true, false, false};
    constexpr RocRollerSelectedTile skTile256x192{256, 192, 128, true, true, false, false};

    struct MatchCase
    {
        const char*           kernel;
        const std::string*    prefix;
        RocRollerSelectedTile tile;
        bool                  expected;
        const char*           why;
    };

    const std::vector<MatchCase>& matchCases()
    {
        static const std::vector<MatchCase> cases = {
            {skFp8_16x16, &fp8Prefix, skTile16x16, true, "SK kernel matches its SK tile"},
            {skFp8_64x32, &fp8Prefix, skTile64x32, true, "SK 64x32x128"},
            {skFp4_16x32, &fp4Prefix, skTile16x32, true, "FP4 SK 16x32x128"},
            {skFp4_256x192, &fp4Prefix, skTile256x192, true, "FP4 SK 256x192x128, UR_4"},
            {plainFp8_16x16, &fp8Prefix, plainTile16x16, true, "plain kernel matches its plain tile"},
            {wgmFp8_256x256, &fp8Prefix, wgmTile256x256, true, "WGM plain kernel matches its WGM tile"},
            {plainFp8_16x16,
             &fp8Prefix,
             skTile16x16,
             false,
             "an SK tile must not take the plain kernel of the same tile"},
            {skFp8_16x16,
             &fp8Prefix,
             plainTile16x16,
             false,
             "a plain tile must not take the SK kernel of the same tile"},
            {preSwizzledFp8_16x16,
             &fp8Prefix,
             plainTile16x16,
             false,
             "a pre-swizzled kernel must not match a problem whose scales are not pre-swizzled"},
            {skFp8_16x16, &fp4Prefix, skTile16x16, false, "the type prefix must still match"},
            {skFp8_64x32, &fp8Prefix, skTile16x16, false, "a different tile shape must not match"},
            {wgmFp8_256x256,
             &fp8Prefix,
             {256, 256, 128, false, false, false, false},
             false,
             "a non-WGM tile must not take a WGM kernel"},
            {plainFp8_16x16,
             &fp8Prefix,
             {16, 16, 128, true, false, false, false},
             false,
             "a WGM tile must not take a kernel without _WGM_"},
        };
        return cases;
    }
}

TEST(HostUnitRocRollerKernelMatch, smoke_MatchesOnlyTheKernelDescribingThatTile)
{
    for(const auto& c : matchCases())
    {
        SCOPED_TRACE(::testing::Message() << c.why << "\n  kernel: " << c.kernel << "\n  tile:   "
                                          << c.tile.m << "x" << c.tile.n << "x" << c.tile.k
                                          << " wgm=" << c.tile.workgroupMapping
                                          << " sk=" << c.tile.streamK);
        EXPECT_EQ(rocRollerKernelNameMatchesTile(c.kernel, *c.prefix, c.tile), c.expected);
    }
}

// The tile token alone cannot separate these two names, so pin the pairing
// directly: each name matches exactly one of the two tiles.
TEST(HostUnitRocRollerKernelMatch, smoke_StreamKIsNotInterchangeableWithPlain)
{
    EXPECT_TRUE(rocRollerKernelNameMatchesTile(skFp8_16x16, fp8Prefix, skTile16x16));
    EXPECT_FALSE(rocRollerKernelNameMatchesTile(skFp8_16x16, fp8Prefix, plainTile16x16));

    EXPECT_TRUE(rocRollerKernelNameMatchesTile(plainFp8_16x16, fp8Prefix, plainTile16x16));
    EXPECT_FALSE(rocRollerKernelNameMatchesTile(plainFp8_16x16, fp8Prefix, skTile16x16));
}

// Each pre-swizzle token on its own is enough to disqualify a kernel: a name
// carrying only the A-side pair, or only the B-side pair, must still be refused.
TEST(HostUnitRocRollerKernelMatch, smoke_RejectsEveryPreSwizzledScaleToken)
{
    const char* const oneSided[] = {
        "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_PSTA_32x8x4_SB_BE8M0_32_WGT_16x16x128_UR_2",
        "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_PTA_32x8_SB_BE8M0_32_WGT_16x16x128_UR_2",
        "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_PSTB_32x8x4_WGT_16x16x128_UR_2",
        "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_PTB_8x32_WGT_16x16x128_UR_2",
    };
    for(const char* kernel : oneSided)
    {
        SCOPED_TRACE(kernel);
        EXPECT_FALSE(rocRollerKernelNameMatchesTile(kernel, fp8Prefix, plainTile16x16));
    }
}

// A StreamK kernel never carries _WGM_, so a StreamK tile with workgroupMapping
// set must still match it. The companion false case covers a StreamK name that
// did carry _WGM_, which the generator does not emit.
TEST(HostUnitRocRollerKernelMatch, smoke_StreamKIgnoresWorkgroupMappingOnTheTile)
{
    constexpr RocRollerSelectedTile skWithWgm{16, 16, 128, true, true, false, false};
    constexpr RocRollerSelectedTile skNoWgm{16, 16, 128, false, true, false, false};
    EXPECT_TRUE(rocRollerKernelNameMatchesTile(skFp8_16x16, fp8Prefix, skWithWgm));
    EXPECT_TRUE(rocRollerKernelNameMatchesTile(skFp8_16x16, fp8Prefix, skNoWgm));

    const std::string skPlusWgm = std::string(skFp8_16x16) + "_WGM_";
    EXPECT_FALSE(rocRollerKernelNameMatchesTile(skPlusWgm, fp8Prefix, skWithWgm));
}

// The tile token is matched whole, delimited by "_WGT_" ahead of it and "_UR_"
// after the K extent, so a name whose extents merely contain the requested
// digits is not a match.
TEST(HostUnitRocRollerKernelMatch, smoke_NearMissExtentsDoNotMatch)
{
    const char* const widerM
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_216x16x128_UR_2";
    EXPECT_FALSE(rocRollerKernelNameMatchesTile(widerM, fp8Prefix, plainTile16x16));

    const char* const widerK
        = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x1280_UR_2";
    EXPECT_FALSE(rocRollerKernelNameMatchesTile(widerK, fp8Prefix, plainTile16x16));
}
