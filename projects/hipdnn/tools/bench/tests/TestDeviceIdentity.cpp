// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestDeviceIdentity.cpp
 * @brief A row's device is the kind of GPU that measured it, not the GPU.
 */

#include <hipdnn_bench/DeviceIdentity.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

namespace
{

hipDeviceProp_t mi300x(unsigned char board)
{
    hipDeviceProp_t properties{};
    std::snprintf(
        properties.gcnArchName, sizeof(properties.gcnArchName), "%s", "gfx942:sramecc+:xnack-");
    std::snprintf(properties.name, sizeof(properties.name), "%s", "AMD Instinct MI300X");
    properties.warpSize = 64;
    properties.multiProcessorCount = 304;
    properties.totalGlobalMem = 206158430208;
    properties.memoryBusWidth = 8192;
    properties.memoryClockRate = 2600000;
    properties.sharedMemPerBlock = 65536;
    // What differs between two boards of one SKU.
    properties.pciBusID = board;
    std::memset(properties.uuid.bytes, board, sizeof(properties.uuid.bytes));
    return properties;
}

} // namespace

namespace hipdnn_bench
{

TEST(TestDeviceIdentity, TwoCardsOfOneKindAreOneDevice)
{
    EXPECT_EQ(deviceKindId(mi300x(1)), deviceKindId(mi300x(2)));
}

TEST(TestDeviceIdentity, AnotherKindOfCardIsAnotherDevice)
{
    auto mi325x = mi300x(1);
    mi325x.totalGlobalMem = 274877906944;
    mi325x.memoryClockRate = 2933000;
    EXPECT_NE(deviceKindId(mi300x(1)), deviceKindId(mi325x));
}

TEST(TestDeviceIdentity, TheIdIsTheKeyTheCatalogPageReports)
{
    const auto properties = mi300x(1);
    EXPECT_EQ(deviceKindId(properties),
              std::to_string(
                  hipdnn_plugin_sdk::ingestor::DeviceKey{devicePropertiesFrom(properties)}.hash()));
}

// A structured binding names every member, so this stops compiling when DeviceProperties
// gains a field: copy it in devicePropertiesFrom(), then extend this binding.
TEST(TestDeviceIdentity, EveryKeyedPropertyIsCopiedFromHip)
{
    const auto copied = devicePropertiesFrom(mi300x(1));
    const auto& [gcnArchName,
                 warpSize,
                 multiProcessorCount,
                 totalGlobalMem,
                 memoryBusWidth,
                 memoryClockRate,
                 sharedMemPerBlock]
        = copied;
    EXPECT_EQ(gcnArchName, "gfx942:sramecc+:xnack-");
    EXPECT_EQ(warpSize, 64);
    EXPECT_EQ(multiProcessorCount, 304);
    EXPECT_EQ(totalGlobalMem, 206158430208U);
    EXPECT_EQ(memoryBusWidth, 8192);
    EXPECT_EQ(memoryClockRate, 2600000);
    EXPECT_EQ(sharedMemPerBlock, 65536U);
}

} // namespace hipdnn_bench
