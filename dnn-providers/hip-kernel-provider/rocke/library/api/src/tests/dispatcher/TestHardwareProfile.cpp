// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "dispatcher/HardwareProfile.hpp"

namespace rocke_client::dispatcher
{
namespace
{

// Test that fromDeviceWithSupplement handles unknown arch gracefully
TEST(TestHardwareProfile, UnknownArchUsesHipValuesOnly)
{
    SKIP_IF_NO_DEVICES();

    const HardwareProfile hw = HardwareProfile::fromDeviceWithSupplement(0, "gfx_unknown");

    // HIP-queried fields should be populated (non-zero on real hardware)
    EXPECT_GT(hw.num_cus, 0);
    EXPECT_GT(hw.max_clock_mhz, 0);
    EXPECT_GT(hw.wavefront_size, 0);
    EXPECT_GT(hw.lds_capacity, 0);

    // Supplement fields should remain at default (0) for unknown arch
    EXPECT_EQ(hw.shader_engines, 0);
    EXPECT_EQ(hw.num_xcd, 0);
    EXPECT_EQ(hw.max_waves_per_cu, 0);
    EXPECT_EQ(hw.l1_cache_kb, 0);
    EXPECT_EQ(hw.l2_cache_kb, 0);
    EXPECT_EQ(hw.l3_cache_kb, 0);
}

// Test that fromDeviceWithSupplement handles empty arch string
TEST(TestHardwareProfile, EmptyArchUsesHipValuesOnly)
{
    SKIP_IF_NO_DEVICES();

    const HardwareProfile hw = HardwareProfile::fromDeviceWithSupplement(0, "");

    // HIP-queried fields should still be populated
    EXPECT_GT(hw.num_cus, 0);
    EXPECT_GT(hw.wavefront_size, 0);

    // Supplement fields should be 0
    EXPECT_EQ(hw.shader_engines, 0);
    EXPECT_EQ(hw.num_xcd, 0);
}

// Test that fromDeviceWithSupplement populates all fields for known arch
TEST(TestHardwareProfile, KnownArchPopulatesAllFields)
{
    SKIP_IF_NO_DEVICES();

    // Use gfx942 which should exist in the supplement table
    const HardwareProfile hw = HardwareProfile::fromDeviceWithSupplement(0, "gfx942");

    // HIP-queried fields
    EXPECT_GT(hw.num_cus, 0);
    EXPECT_GT(hw.max_clock_mhz, 0);
    EXPECT_GT(hw.wavefront_size, 0);
    EXPECT_GT(hw.lds_capacity, 0);

    // Supplement fields should be non-zero for known arch
    EXPECT_GT(hw.shader_engines, 0);
    EXPECT_GT(hw.num_xcd, 0);
    EXPECT_GT(hw.simds_per_cu, 0);
    EXPECT_GT(hw.max_waves_per_cu, 0);
    // Note: Not all archs have L3, so only check L1/L2
    EXPECT_GT(hw.l1_cache_kb, 0);
    EXPECT_GT(hw.l2_cache_kb, 0);
}

// Test that total_simds derived field works correctly
TEST(TestHardwareProfile, TotalSimdsDerivedCorrectly)
{
    SKIP_IF_NO_DEVICES();

    const HardwareProfile hw = HardwareProfile::fromDeviceWithSupplement(0, "gfx942");

    // total_simds() = num_cus * simds_per_cu
    const int expected_simds = hw.num_cus * hw.simds_per_cu;
    EXPECT_EQ(hw.total_simds(), expected_simds);
}

// Test fromDevice still works (legacy path without supplement)
TEST(TestHardwareProfile, FromDevicePopulatesHipFields)
{
    SKIP_IF_NO_DEVICES();

    const HardwareProfile hw = HardwareProfile::fromDevice(0);

    // HIP-queried fields should be populated
    EXPECT_GT(hw.num_cus, 0);
    EXPECT_GT(hw.max_clock_mhz, 0);
    EXPECT_GT(hw.wavefront_size, 0);
    EXPECT_GT(hw.lds_capacity, 0);

    // Supplement fields should be at defaults (fromDevice doesn't populate them)
    EXPECT_EQ(hw.shader_engines, 0);
    EXPECT_EQ(hw.num_xcd, 0);
}

} // namespace
} // namespace rocke_client::dispatcher
