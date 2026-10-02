// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"

using namespace stinkytofu;

TEST(ComgrProbeTest, HasComgrSupport) {
    EXPECT_TRUE(hasComgrSupport());
}

TEST(ComgrProbeTest, ValidInstructionAssembles) {
    constexpr const char* kIsa = "amdgcn-amd-amdhsa--gfx1250";
    if (!comgrSupportsIsa(kIsa)) {
        GTEST_SKIP() << "Installed comgr does not list " << kIsa
                     << "; skipping assembler round-trip test.";
    }
    EXPECT_TRUE(tryAssembleWithComgr("s_nop 0", kIsa, 32));
}

// Joblib workers close stdin. comgr then opens its output on FD 0, and LLVM's
// raw_fd_ostream::close() asserts ShouldClose. Restore stdin before the
// assertion so a failure does not leak into later tests.
TEST(ComgrProbeTest, AssemblesWhenStdinIsClosed) {
#ifndef _WIN32
    constexpr const char* kIsa = "amdgcn-amd-amdhsa--gfx1250";
    if (!comgrSupportsIsa(kIsa)) {
        GTEST_SKIP() << "Installed comgr does not list " << kIsa
                     << "; skipping assembler round-trip test.";
    }
    const int savedStdin = ::dup(STDIN_FILENO);
    ASSERT_NE(savedStdin, -1);
    ASSERT_EQ(::close(STDIN_FILENO), 0);
    const bool assembled = tryAssembleWithComgr("s_nop 0", kIsa, 32);
    ASSERT_EQ(::dup2(savedStdin, STDIN_FILENO), STDIN_FILENO);
    ::close(savedStdin);
    EXPECT_TRUE(assembled);
#else
    GTEST_SKIP() << "stdin-closed assemble regression is covered on POSIX";
#endif
}

TEST(ComgrProbeTest, InvalidInstructionFails) {
    EXPECT_FALSE(tryAssembleWithComgr("s_bogus_not_real 0", "amdgcn-amd-amdhsa--gfx1250", 32));
}

TEST(ComgrProbeTest, InvalidIsaFails) {
    EXPECT_FALSE(tryAssembleWithComgr("s_nop 0", "amdgcn-amd-amdhsa--gfx0000", 32));
}

TEST(ToolchainCapsTest, ProbeGfx1250ReturnsNonNone) {
    constexpr const char* kIsa = "amdgcn-amd-amdhsa--gfx1250";
    if (!comgrSupportsIsa(kIsa)) {
        GTEST_SKIP() << "Installed comgr does not list " << kIsa
                     << "; ToolchainCaps cannot probe vgprMsbMode.";
    }
    auto caps = ToolchainCaps::probe(GfxArchID::Gfx1250);
    EXPECT_NE(caps.vgprMsbMode, VgprMsbMode::None);
}

TEST(ToolchainCapsTest, ProbeIsCached) {
    auto caps1 = ToolchainCaps::probe(GfxArchID::Gfx1250);
    auto caps2 = ToolchainCaps::probe(GfxArchID::Gfx1250);
    EXPECT_EQ(caps1.vgprMsbMode, caps2.vgprMsbMode);
}
