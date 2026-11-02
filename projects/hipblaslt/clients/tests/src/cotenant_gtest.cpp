// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the cotenant helper (common/include/cotenant.hpp).
//
// The option-validation and LDS-sizing tests are host-only. The Scoped tests
// need a GPU and skip without one. Linux only: the cotenant is not supported
// on Windows, so CMake does not build this file there.

#include "cotenant.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
    using namespace hipblaslt_cotenant;

    constexpr int kLds64K  = 64 * 1024;
    constexpr int kLds160K = 160 * 1024;

    TEST(CotenantOptions, DefaultsAreValid)
    {
        EXPECT_TRUE(validate_options(0, 1).empty());
    }

    TEST(CotenantOptions, RejectsNegativeCount)
    {
        EXPECT_FALSE(validate_options(-1, 1).empty());
    }

    TEST(CotenantOptions, RejectsOccupancyOutsideRange)
    {
        EXPECT_FALSE(validate_options(0, 0).empty());
        EXPECT_FALSE(validate_options(0, -1).empty());
        EXPECT_FALSE(validate_options(0, kMaxOccupancy + 1).empty());
    }

    TEST(CotenantOptions, AcceptsOccupancyBounds)
    {
        EXPECT_TRUE(validate_options(0, 1).empty());
        EXPECT_TRUE(validate_options(0, kMaxOccupancy).empty());
    }

    TEST(CotenantOptions, AcceptsPositiveCount)
    {
        EXPECT_TRUE(validate_options(4, 1).empty());
        EXPECT_TRUE(validate_options(4, kMaxOccupancy).empty());
    }

    TEST(CotenantLds, MatchesKnownReservations)
    {
        struct Case
        {
            int lds, occupancy, expected;
        };
        const Case cases[] = {
            {kLds64K, 1, 65536},
            {kLds64K, 3, 21760},
            {kLds64K, 5, 13056},
            {kLds64K, 7, 9216},
            {kLds64K, 64, 1024},
            {kLds160K, 1, 163840},
            {kLds160K, 3, 54528},
            {kLds160K, 6, 27136},
            {kLds160K, 16, 10240},
            {kLds160K, 64, 2560},
        };
        for(const auto& c : cases)
            EXPECT_EQ(lds_reservation(c.lds, c.lds, c.occupancy), c.expected)
                << "lds=" << c.lds << " occupancy=" << c.occupancy;
    }

    TEST(CotenantLds, GranuleAlignedAndNeverAllowsMoreThanRequested)
    {
        for(const int lds : {kLds64K, kLds160K})
        {
            for(int n = 1; n <= kMaxOccupancy; ++n)
            {
                const int reserve = lds_reservation(lds, lds, n);
                EXPECT_EQ(reserve % 256, 0) << "lds=" << lds << " n=" << n;
                EXPECT_LE(lds / reserve, n) << "lds=" << lds << " n=" << n;
                EXPECT_LT(std::abs(double(reserve) - double(lds) / n), 256.0)
                    << "lds=" << lds << " n=" << n;
            }
        }
    }

    TEST(CotenantLds, RespectsPerBlockLimit)
    {
        EXPECT_EQ(lds_reservation(kLds160K, 65536, 3), 54528);
        EXPECT_THROW(lds_reservation(kLds160K, 65536, 1), std::runtime_error);
        // A zero opt-in limit means no limit below the per-CU total.
        EXPECT_EQ(lds_reservation(kLds64K, 0, 1), 65536);
    }

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // CU count of the current device. Call only when gpuAvailable().
    int deviceCuCount()
    {
        int             dev = 0;
        hipDeviceProp_t prop{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess)
        {
            ADD_FAILURE() << "Could not query the device";
            return 0;
        }
        return prop.multiProcessorCount;
    }

    // Smallest max occupancy the current device can honour. Occupancy 1 needs all of
    // a CU's LDS in one workgroup, which exceeds the per-workgroup limit on some
    // targets (e.g. RDNA). Call only when gpuAvailable().
    int minDeviceOccupancy()
    {
        int dev = 0, lds_per_cu = 0, optin = 0;
        if(hipGetDevice(&dev) != hipSuccess
           || hipDeviceGetAttribute(
                  &lds_per_cu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev)
                  != hipSuccess
           || hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, dev)
                  != hipSuccess)
        {
            ADD_FAILURE() << "Could not query the device";
            return 1;
        }
        for(int n = 1; n <= kMaxOccupancy; ++n)
        {
            try
            {
                lds_reservation(lds_per_cu, optin, n);
                return n;
            }
            catch(const std::runtime_error&)
            {
            }
        }
        ADD_FAILURE() << "No max occupancy fits the device's per-workgroup LDS limit";
        return kMaxOccupancy;
    }

    TEST(CotenantScoped, DisabledScopeDoesNothing)
    {
        EXPECT_NO_THROW(Scoped(0, 1, nullptr));
    }

    TEST(CotenantScoped, StartsAndStopsRepeatedly)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";
        const int cus = deviceCuCount();
        ASSERT_GT(cus, 0);
        if(cus < 2)
            GTEST_SKIP() << "Needs a GPU with at least 2 CUs";

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        const int n         = std::min(8, cus / 2);
        const int occupancy = minDeviceOccupancy();
        for(int i = 0; i < 5; ++i)
        {
            // Construction returns only once all n workgroups are resident.
            EXPECT_NO_THROW(Scoped(n, occupancy, stream)) << "iteration " << i;
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        }
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    }

    TEST(CotenantScoped, CountMayEqualCuCount)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";
        const int cus = deviceCuCount();
        ASSERT_GT(cus, 0);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_NO_THROW(Scoped(cus, 8, stream));
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    }

    TEST(CotenantScoped, RejectsInvalidArgumentsAndStaysUsable)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";
        const int cus = deviceCuCount();
        ASSERT_GT(cus, 0);
        if(cus < 2)
            GTEST_SKIP() << "Needs a GPU with at least 2 CUs";

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_THROW(Scoped(cus + 1, 1, stream), std::runtime_error);
        EXPECT_THROW(Scoped(1, 0, stream), std::runtime_error);
        EXPECT_THROW(Scoped(1, kMaxOccupancy + 1, stream), std::runtime_error);
        // Failed construction releases everything it acquired.
        EXPECT_NO_THROW(Scoped(1, minDeviceOccupancy(), stream));
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
} // namespace
