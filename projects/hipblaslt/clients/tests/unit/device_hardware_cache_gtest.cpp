// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the per-device hardware cache that TensileHost fills when it
// starts. The cache is reached by relative #include, because the library's
// include directory would shadow the clients' utility.hpp. The tests answer its
// device queries with stand-ins for hipGetDeviceProperties and hip::GetDevice,
// so they need no GPU, let alone two of one architecture with different CU
// counts.

#include "../../../library/src/amd_detail/rocblaslt/src/include/device_hardware_cache.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace
{
    // What a launch on a device reads from its hardware description.
    struct StubHardware
    {
        int device;
        int computeUnits;
    };

    using Cache = rocblaslt::DeviceHardwareCache<StubHardware>;

    struct DeviceSpec
    {
        const char* gcnArchName;
        int         multiProcessorCount;
        int         pciBusID;
    };

    // A Radeon PRO V620 and an RX 6900 XT are both gfx1030, with 36 and 40 CUs as
    // HIP counts them. The third device is a second V620.
    constexpr int        kDevices = 3;
    constexpr DeviceSpec kMachine[kDevices]
        = {{"gfx1030", 36, 0x03}, {"gfx1030", 40, 0x43}, {"gfx1030", 36, 0x83}};

    // Provide properties for three same-architecture devices.
    struct Machine
    {

        hipDeviceProp_t properties(int device)
        {
            hipDeviceProp_t prop{};
            std::snprintf(
                prop.gcnArchName, sizeof(prop.gcnArchName), "%s", kMachine[device].gcnArchName);
            prop.multiProcessorCount = kMachine[device].multiProcessorCount;
            prop.pciBusID            = kMachine[device].pciBusID;
            return prop;
        }

        std::shared_ptr<StubHardware> hardware(int device)
        {
            return std::make_shared<StubHardware>(
                StubHardware{device, kMachine[device].multiProcessorCount});
        }
    };

    Cache filledCache(Machine& machine)
    {
        Cache cache;
        cache.populate(
            kDevices,
            [&](int device) { return machine.properties(device); },
            [&](int device) { return machine.hardware(device); });
        return cache;
    }
}

// Devices of one architecture that differ in CU count each keep an entry of their
// own. A cache keyed by architecture asked only about the first of them and
// served its properties and hardware for all.
TEST(HostUnitDeviceHardwareCache, smoke_SameArchitectureDevicesKeepTheirOwnEntries)
{
    Machine     machine;
    const Cache cache = filledCache(machine);

    for(int device = 0; device < kDevices; device++)
    {
        SCOPED_TRACE("device " + std::to_string(device));

        const auto& properties = cache.properties(device);
        ASSERT_NE(properties.get(), nullptr);
        EXPECT_STREQ(properties->gcnArchName, "gfx1030");
        EXPECT_EQ(properties->multiProcessorCount, kMachine[device].multiProcessorCount);
        EXPECT_EQ(properties->pciBusID, kMachine[device].pciBusID);

        const auto hardware = cache.hardware(device);
        ASSERT_NE(hardware.get(), nullptr);
        EXPECT_EQ(hardware->device, device);
        EXPECT_EQ(hardware->computeUnits, kMachine[device].multiProcessorCount);
    }
}

