// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/ShallowHostOnlyMigratableMemory.hpp>
#include <stdexcept>

using namespace hipdnn_sdk::utilities;

TEST(TestShallowHostOnlyMigratableMemory, HostAccess)
{
    std::array<int, 8> backing = {0, 1, 2, 3, 4, 5, 6, 7};
    ShallowHostOnlyMigratableMemory<int> mem(backing.data());

    auto* p1 = mem.hostData();
    auto* p2 = mem.hostDataAsync();
    EXPECT_EQ(p1, backing.data());
    EXPECT_EQ(p2, backing.data());

    const ShallowHostOnlyMigratableMemory<int>& cref = mem;
    auto* cp1 = cref.hostData();
    auto* cp2 = cref.hostDataAsync();
    EXPECT_EQ(cp1, backing.data());
    EXPECT_EQ(cp2, backing.data());
}

TEST(TestShallowHostOnlyMigratableMemory, ThrowsOnDeviceAccess)
{
    std::array<int, 2> backing = {0, 1};
    ShallowHostOnlyMigratableMemory<int> mem(backing.data());
    EXPECT_THROW(mem.deviceData(), std::runtime_error);
    EXPECT_THROW(mem.deviceDataAsync(), std::runtime_error);
}

TEST(TestShallowHostOnlyMigratableMemory, MarkHostModifiedNoThrow)
{
    std::array<int, 1> backing = {42};
    ShallowHostOnlyMigratableMemory<int> mem(backing.data());
    EXPECT_NO_THROW(mem.markHostModified());
}

TEST(TestShallowHostOnlyMigratableMemory, MarkDeviceModifiedThrows)
{
    std::array<int, 1> backing = {0};
    ShallowHostOnlyMigratableMemory<int> mem(backing.data());
    EXPECT_THROW(mem.markDeviceModified(), std::runtime_error);
}

TEST(TestShallowHostOnlyMigratableMemory, LocationIsHost)
{
    std::array<int, 1> backing = {0};
    ShallowHostOnlyMigratableMemory<int> mem(backing.data());
    EXPECT_EQ(mem.location(), MemoryLocation::HOST);
}

TEST(TestShallowHostOnlyMigratableMemory, SizeAndMutationOperationsThrow)
{
    std::array<int, 4> backing = {0, 1, 2, 3};
    ShallowHostOnlyMigratableMemory<int> mem(backing.data());
    EXPECT_THROW(mem.count(), std::runtime_error);
    EXPECT_THROW(mem.empty(), std::runtime_error);
    EXPECT_THROW(mem.resize(10), std::runtime_error);
    EXPECT_THROW(mem.clear(), std::runtime_error);
}
