// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the library's per-device arch-name cache, driven
// through hipblaslt_arch_name_lookups_for_test over fake HIP calls.

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" void
    hipblaslt_arch_name_lookups_for_test(hipError_t (*getDeviceCount)(int*),
                                         hipError_t (*getDeviceProperties)(hipDeviceProp_t*, int),
                                         const int* deviceIds,
                                         size_t     count,
                                         char*      names,
                                         size_t     nameSize,
                                         bool*      ok);

namespace
{
    const char* const kArchNames[] = {"gfx950:sramecc+:xnack-", "gfx942"};

    // Fake HIP state, reset by the fixture.
    int countCalls;
    int countFailures; // the next count queries that fail
    int propCalls[2];
    int propFailures[2]; // per device, the next property queries that fail

    hipError_t fakeGetDeviceCount(int* count)
    {
        ++countCalls;
        if(countFailures > 0)
        {
            --countFailures;
            return hipErrorNoDevice;
        }
        *count = 2;
        return hipSuccess;
    }

    hipError_t fakeGetDeviceProperties(hipDeviceProp_t* prop, int deviceId)
    {
        if(deviceId < 0 || deviceId >= 2)
            return hipErrorInvalidDevice;
        ++propCalls[deviceId];
        if(propFailures[deviceId] > 0)
        {
            --propFailures[deviceId];
            return hipErrorInvalidValue;
        }
        std::memset(prop, 0, sizeof(*prop));
        std::strcpy(prop->gcnArchName, kArchNames[deviceId]);
        return hipSuccess;
    }

    struct Lookup
    {
        bool        ok;
        std::string name;
    };

    class HostUnitArchNameCache : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            countCalls    = 0;
            countFailures = 0;
            for(int d = 0; d < 2; ++d)
            {
                propCalls[d]    = 0;
                propFailures[d] = 0;
            }
        }

        // Looks up the IDs in order through one fresh cache.
        static std::vector<Lookup> lookups(const std::vector<int>& deviceIds)
        {
            constexpr size_t        nameSize = 64;
            std::vector<char>       names(deviceIds.size() * nameSize, 'x');
            std::unique_ptr<bool[]> ok(new bool[deviceIds.size()]);
            hipblaslt_arch_name_lookups_for_test(fakeGetDeviceCount,
                                                 fakeGetDeviceProperties,
                                                 deviceIds.data(),
                                                 deviceIds.size(),
                                                 names.data(),
                                                 nameSize,
                                                 ok.get());
            std::vector<Lookup> result;
            for(size_t i = 0; i < deviceIds.size(); ++i)
                result.push_back({ok[i], ok[i] ? std::string(&names[i * nameSize]) : ""});
            return result;
        }
    };

    TEST_F(HostUnitArchNameCache, smoke_EachDeviceGetsItsOwnName)
    {
        auto r = lookups({1, 0, 1});
        ASSERT_EQ(r.size(), 3u);
        EXPECT_TRUE(r[0].ok);
        EXPECT_EQ(r[0].name, "gfx942");
        EXPECT_TRUE(r[1].ok);
        EXPECT_EQ(r[1].name, "gfx950");
        EXPECT_TRUE(r[2].ok);
        EXPECT_EQ(r[2].name, "gfx942");
    }

    TEST_F(HostUnitArchNameCache, smoke_QualifiersAreStripped)
    {
        auto r = lookups({0});
        ASSERT_TRUE(r[0].ok);
        EXPECT_EQ(r[0].name, "gfx950");
    }

    TEST_F(HostUnitArchNameCache, smoke_SuccessIsCachedPerDevice)
    {
        auto r = lookups({0, 1, 0, 1});
        for(const auto& l : r)
            EXPECT_TRUE(l.ok);
        EXPECT_EQ(countCalls, 1);
        EXPECT_EQ(propCalls[0], 1);
        EXPECT_EQ(propCalls[1], 1);
    }

    TEST_F(HostUnitArchNameCache, smoke_PropertiesFailureIsRetried)
    {
        propFailures[0] = 1;
        auto r          = lookups({0, 0, 0});
        EXPECT_FALSE(r[0].ok);
        EXPECT_TRUE(r[1].ok);
        EXPECT_EQ(r[1].name, "gfx950");
        EXPECT_TRUE(r[2].ok);
        EXPECT_EQ(r[2].name, "gfx950");
        EXPECT_EQ(propCalls[0], 2);
    }

    TEST_F(HostUnitArchNameCache, smoke_PropertiesFailureLeavesOtherDevices)
    {
        propFailures[0] = 1;
        auto r          = lookups({0, 1});
        EXPECT_FALSE(r[0].ok);
        EXPECT_TRUE(r[1].ok);
        EXPECT_EQ(r[1].name, "gfx942");
    }

    TEST_F(HostUnitArchNameCache, smoke_OutOfRangeIdFails)
    {
        auto r = lookups({2, -1, 0});
        EXPECT_FALSE(r[0].ok);
        EXPECT_FALSE(r[1].ok);
        EXPECT_TRUE(r[2].ok);
        EXPECT_EQ(r[2].name, "gfx950");
    }

    TEST_F(HostUnitArchNameCache, smoke_DeviceCountFailureIsRetried)
    {
        countFailures = 1;
        auto r        = lookups({1, 1, 1});
        EXPECT_FALSE(r[0].ok);
        EXPECT_TRUE(r[1].ok);
        EXPECT_EQ(r[1].name, "gfx942");
        EXPECT_TRUE(r[2].ok);
        EXPECT_EQ(countCalls, 2);
        EXPECT_EQ(propCalls[1], 1);
    }
}
