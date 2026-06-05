// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_gpu_ref/detail/ShallowDeviceOnlyMigratableMemory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstddef>
#include <stdexcept>

using hipdnn_data_sdk::utilities::MemoryLocation;
using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_gpu_ref::detail::ShallowDeviceOnlyMigratableMemory;

TEST(TestShallowDeviceOnlyMigratableMemory, DefaultConstruction)
{
    ShallowDeviceOnlyMigratableMemory<float> mem;

    EXPECT_EQ(mem.count(), 0u);
    EXPECT_TRUE(mem.empty());
    EXPECT_EQ(mem.location(), MemoryLocation::DEVICE);
    EXPECT_EQ(mem.deviceData(), nullptr);
    EXPECT_EQ(mem.deviceDataAsync(), nullptr);
}

TEST(TestShallowDeviceOnlyMigratableMemory, DeviceDataReturnsPointer)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t COUNT = 16;
    const Workspace workspace(COUNT * sizeof(float));

    ShallowDeviceOnlyMigratableMemory<float> mem(workspace.get(), COUNT);

    EXPECT_EQ(mem.deviceData(), workspace.get());
    EXPECT_EQ(mem.deviceDataAsync(), workspace.get());
    EXPECT_EQ(mem.count(), COUNT);
    EXPECT_FALSE(mem.empty());
}

TEST(TestShallowDeviceOnlyMigratableMemory, HostAccessThrows)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowDeviceOnlyMigratableMemory<float> mem(workspace.get(), 4);

    EXPECT_THROW(mem.hostData(), std::runtime_error);
    EXPECT_THROW(mem.hostDataAsync(), std::runtime_error);

    const auto& constMem = mem;
    EXPECT_THROW(constMem.hostData(), std::runtime_error);
    EXPECT_THROW(constMem.hostDataAsync(), std::runtime_error);

    EXPECT_THROW(mem.markHostModified(), std::runtime_error);
}

TEST(TestShallowDeviceOnlyMigratableMemory, MarkDeviceModifiedIsNoOp)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowDeviceOnlyMigratableMemory<float> mem(workspace.get(), 4);

    EXPECT_NO_THROW(mem.markDeviceModified());
}

TEST(TestShallowDeviceOnlyMigratableMemory, ResizeAndClearThrow)
{
    ShallowDeviceOnlyMigratableMemory<float> mem;

    EXPECT_THROW(mem.resize(10), std::runtime_error);
    EXPECT_THROW(mem.clear(), std::runtime_error);
}

TEST(TestShallowDeviceOnlyMigratableMemory, MoveConstruction)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t COUNT = 8;
    const Workspace workspace(COUNT * sizeof(float));

    ShallowDeviceOnlyMigratableMemory<float> source(workspace.get(), COUNT);
    ShallowDeviceOnlyMigratableMemory<float> dest(std::move(source));

    EXPECT_EQ(dest.deviceData(), workspace.get());
    EXPECT_EQ(dest.count(), COUNT);
    EXPECT_FALSE(dest.empty());

    // Source is nulled out — intentionally inspecting moved-from state
    EXPECT_EQ(source.deviceData(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.count(), 0u);
    EXPECT_TRUE(source.empty());
}

TEST(TestShallowDeviceOnlyMigratableMemory, MoveAssignment)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t COUNT = 8;
    const Workspace workspace(COUNT * sizeof(float));

    ShallowDeviceOnlyMigratableMemory<float> source(workspace.get(), COUNT);
    ShallowDeviceOnlyMigratableMemory<float> dest;

    dest = std::move(source);

    EXPECT_EQ(dest.deviceData(), workspace.get());
    EXPECT_EQ(dest.count(), COUNT);
    EXPECT_FALSE(dest.empty());

    // Source is nulled out — intentionally inspecting moved-from state
    EXPECT_EQ(source.deviceData(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.count(), 0u);
    EXPECT_TRUE(source.empty());
}
