// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/CodeObjectTarget.hpp"

/**
 * @file TestCodeObjectTarget.cpp
 * @brief isCodeObjectTargetCompatible: which devices can run a stored code object.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

TEST(TestCodeObjectTarget, KpackAcceptsExactlyTheDevicesThatCouldSelectTheEntry)
{
    // A bare entry serves the bare device and every decorated form of it.
    EXPECT_TRUE(isCodeObjectTargetCompatible(KernelSourceKind::KPACK, "gfx90a", "gfx90a"));
    EXPECT_TRUE(
        isCodeObjectTargetCompatible(KernelSourceKind::KPACK, "gfx90a", "gfx90a:sramecc+:xnack-"));
    EXPECT_TRUE(
        isCodeObjectTargetCompatible(KernelSourceKind::KPACK, "gfx90a:xnack-", "gfx90a:xnack-"));

    // A decorated entry serves only a device whose name starts with the same features.
    EXPECT_FALSE(isCodeObjectTargetCompatible(
        KernelSourceKind::KPACK, "gfx90a:xnack-", "gfx90a:sramecc+:xnack-"));
    EXPECT_FALSE(
        isCodeObjectTargetCompatible(KernelSourceKind::KPACK, "gfx90a:xnack-", "gfx90a:xnack+"));

    EXPECT_FALSE(isCodeObjectTargetCompatible(KernelSourceKind::KPACK, "gfx942", "gfx90a"));
}

TEST(TestCodeObjectTarget, OtherKindsNeedTheExactTarget)
{
    EXPECT_FALSE(isCodeObjectTargetCompatible(
        KernelSourceKind::EMBEDDED_SOURCE, "gfx90a", "gfx90a:sramecc+:xnack-"));
    EXPECT_TRUE(isCodeObjectTargetCompatible(
        KernelSourceKind::EMBEDDED_SOURCE, "gfx90a:sramecc+:xnack-", "gfx90a:sramecc+:xnack-"));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
