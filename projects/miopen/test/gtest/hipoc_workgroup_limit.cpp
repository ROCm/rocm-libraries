// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace {
using miopen::CheckWorkgroupCount;
using miopen::GetMaxWorkgroupsYZ;
using miopen::ValidateLaunch;

constexpr std::array<size_t, 3> ldims{8, 4, 1};

// Y and Z workgroup counts; one workgroup in X.
std::array<size_t, 3> Grid(size_t y_workgroups, size_t z_workgroups)
{
    return {8, 4 * y_workgroups, z_workgroups};
}

// Runs \p f and returns the status of the miopen::Exception it throws, or miopenStatusSuccess.
template <class F>
miopenStatus_t ThrownStatus(F&& f)
{
    try
    {
        f();
    }
    catch(const miopen::Exception& e)
    {
        return e.status;
    }
    return miopenStatusSuccess;
}

} // namespace

TEST(CPU_hipoc_workgroup_limit_NONE, arch_limits)
{
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1201"), 65536u);
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1200"), 65536u);
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1250"), 65535u);
    // Feature suffixes reported by gcnArchName.
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1250-strict"), 65535u);
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1201:sramecc+:xnack-"), 65536u);
    for(const char* arch : {"gfx942", "gfx950", "gfx1100", "gfx1151", "gfx90a"})
        EXPECT_EQ(GetMaxWorkgroupsYZ(arch), std::numeric_limits<size_t>::max()) << arch;
}

TEST(CPU_hipoc_workgroup_limit_NONE, boundaries)
{
    struct Case
    {
        const char* arch;
        std::array<size_t, 3> gdims;
        bool throws;
    };
    // 65536 workgroups already run nothing on gfx1250.
    const Case cases[] = {
        {"gfx1200", Grid(65536, 1), false},
        {"gfx1200", Grid(1, 65536), false},
        {"gfx1200", Grid(65537, 1), true},
        {"gfx1200", Grid(1, 65537), true},
        {"gfx1201", Grid(65536, 1), false},
        {"gfx1201", Grid(1, 65536), false},
        {"gfx1201", Grid(65537, 1), true},
        {"gfx1201", Grid(1, 65537), true},
        {"gfx1250", Grid(65535, 1), false},
        {"gfx1250", Grid(1, 65535), false},
        {"gfx1250", Grid(65536, 1), true},
        {"gfx1250", Grid(1, 65536), true},
        // 65536 full workgroups plus one work-item is 65537 workgroups.
        {"gfx1201", {8, 4 * 65536 + 1, 1}, true},
        {"gfx1250", {8, 1, 65536}, true},
        {"gfx942", Grid(1000000, 1000000), false},
        // X is not limited.
        {"gfx1250", {8 * 1000000, 4, 1}, false},
    };

    for(const auto& c : cases)
    {
        const auto status = ThrownStatus(
            [&] { CheckWorkgroupCount(c.gdims, ldims, "k", GetMaxWorkgroupsYZ(c.arch)); });
        EXPECT_EQ(status, c.throws ? miopenStatusInternalError : miopenStatusSuccess)
            << c.arch << " y=" << c.gdims[1] << " z=" << c.gdims[2];

        const auto launch_status = ThrownStatus([&] {
            ValidateLaunch(c.gdims, ldims, "k", [&] { return GetMaxWorkgroupsYZ(c.arch); });
        });
        EXPECT_EQ(launch_status, c.throws ? miopenStatusInternalError : miopenStatusSuccess)
            << c.arch << " y=" << c.gdims[1] << " z=" << c.gdims[2];
    }
}

TEST(CPU_hipoc_workgroup_limit_NONE, validate_launch_skips_device_query_below_smallest_limit)
{
    int queries     = 0;
    const auto find = [&] {
        ++queries;
        return GetMaxWorkgroupsYZ("gfx1250");
    };

    EXPECT_NO_THROW(ValidateLaunch(Grid(65535, 65535), ldims, "k", find));
    EXPECT_EQ(queries, 0);

    EXPECT_EQ(ThrownStatus([&] { ValidateLaunch(Grid(65536, 1), ldims, "k", find); }),
              miopenStatusInternalError);
    EXPECT_EQ(queries, 1);
}

TEST(CPU_hipoc_workgroup_limit_NONE, validate_launch_checks_global_work_size)
{
    constexpr size_t k32BitLimit = size_t{1} << 32;
    EXPECT_EQ(ThrownStatus([&] {
                  ValidateLaunch({k32BitLimit, 4, 1}, ldims, "k", [] {
                      return std::numeric_limits<size_t>::max();
                  });
              }),
              miopenStatusInternalError);
}
