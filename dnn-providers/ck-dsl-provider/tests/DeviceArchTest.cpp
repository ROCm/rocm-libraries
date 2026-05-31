// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <optional>
#include <string>

#include "runtime/DeviceArch.hpp"

namespace {

using ck_dsl_provider::detectDeviceArch;
using ck_dsl_provider::stripArchFeatureSuffix;

// --- stripArchFeatureSuffix: pure string logic, no HIP device needed ---

TEST(DeviceArchStripSuffix, StripsRocmFeatureSuffix) {
    EXPECT_EQ(stripArchFeatureSuffix("gfx950:sramecc+:xnack-"), "gfx950");
}

TEST(DeviceArchStripSuffix, StripsSingleFeatureSuffix) {
    EXPECT_EQ(stripArchFeatureSuffix("gfx1151:xnack-"), "gfx1151");
}

TEST(DeviceArchStripSuffix, LeavesBareTokenUnchanged) {
    EXPECT_EQ(stripArchFeatureSuffix("gfx942"), "gfx942");
}

TEST(DeviceArchStripSuffix, StripsFromFirstColon) {
    // Everything from the first ':' onward goes, even if more colons follow.
    EXPECT_EQ(stripArchFeatureSuffix("gfx900:a:b:c"), "gfx900");
}

TEST(DeviceArchStripSuffix, EmptyStaysEmpty) {
    EXPECT_EQ(stripArchFeatureSuffix(""), "");
}

TEST(DeviceArchStripSuffix, LeadingColonYieldsEmpty) {
    EXPECT_EQ(stripArchFeatureSuffix(":gfx950"), "");
}

// --- detectDeviceArch: branch on actual device visibility so the test
//     is device-independent (passes on host-only CI and on a GPU box) ---

TEST(DeviceArchDetect, MatchesDeviceVisibility) {
    int deviceCount = 0;
    const hipError_t countErr = hipGetDeviceCount(&deviceCount);

    if (countErr != hipSuccess || deviceCount == 0) {
        // Host-only: the one benign "no arch" outcome is nullopt.
        EXPECT_EQ(detectDeviceArch(/*stream=*/nullptr), std::nullopt);
        return;
    }

    // A device is present: detection must yield a bare gfx token --
    // non-empty, no feature suffix, and the gfx prefix the DSL catalog
    // keys on.
    std::optional<std::string> arch = detectDeviceArch(/*stream=*/nullptr);
    ASSERT_TRUE(arch.has_value());
    EXPECT_FALSE(arch->empty());
    EXPECT_EQ(arch->find(':'), std::string::npos) << "feature suffix not stripped: " << *arch;
    EXPECT_EQ(arch->rfind("gfx", 0), 0u) << "not a gfx token: " << *arch;
}

TEST(DeviceArchDetect, IsIdempotent) {
    // Whatever the verdict (nullopt on host-only, a token on a GPU box),
    // a second call returns the same result -- the per-ordinal memo must
    // not change the answer.
    EXPECT_EQ(detectDeviceArch(/*stream=*/nullptr), detectDeviceArch(/*stream=*/nullptr));
}

}  // namespace
