// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "engines/hip_flash2_engine/HipFlash2KernelResolver.hpp"

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hip_flash2_engine
{
namespace
{

/// Stands in for the hardware and the kernel directory. A fake module carries no
/// hipModule_t, so destroying it touches no HIP state.
class FakeFlash2Resolver : public Flash2KernelResolver
{
public:
    explicit FakeFlash2Resolver(std::map<int, std::string> architectures)
        : Flash2KernelResolver(
              "gfx942", "/kernels/hip_flash2_fwd_gfx942.co", "flash2_fn", 256, "w4")
        , _architectures(std::move(architectures))
    {
    }

    mutable std::vector<int> archQueries;
    mutable std::vector<int> loads;

protected:
    hipError_t queryDeviceArch(int deviceOrdinal, std::string& reportedArch) const override
    {
        archQueries.push_back(deviceOrdinal);
        const auto found = _architectures.find(deviceOrdinal);
        if(found == _architectures.end())
        {
            return hipErrorInvalidDevice;
        }
        reportedArch = found->second;
        return hipSuccess;
    }

    Flash2Module loadModule(int deviceOrdinal) const override
    {
        loads.push_back(deviceOrdinal);
        return std::make_shared<const HipModuleGuard>();
    }

private:
    std::map<int, std::string> _architectures;
};

TEST(TestHipFlash2KernelResolver, LoadsASecondOrdinalOfTheSameArchitecture)
{
    const FakeFlash2Resolver kernels(
        {{0, "gfx942:sramecc+:xnack-"}, {1, "gfx942:sramecc+:xnack-"}});

    const auto first = kernels.forOrdinal(0);
    const auto second = kernels.forOrdinal(1);

    // Each device gets its own module rather than device 0's.
    EXPECT_NE(first, second);
    EXPECT_EQ(kernels.loads, (std::vector<int>{0, 1}));
}

TEST(TestHipFlash2KernelResolver, AnswersASeenOrdinalWithoutQueryingOrLoading)
{
    const FakeFlash2Resolver kernels({{1, "gfx942"}});

    kernels.forOrdinal(1);
    kernels.forOrdinal(1);

    EXPECT_EQ(kernels.archQueries, std::vector<int>{1});
    EXPECT_EQ(kernels.loads, std::vector<int>{1});
}

TEST(TestHipFlash2KernelResolver, RefusesAnOrdinalOfAnotherArchitecture)
{
    const FakeFlash2Resolver kernels({{0, "gfx942"}, {1, "gfx950"}});

    try
    {
        kernels.forOrdinal(1);
        FAIL() << "expected a device of another architecture to be refused";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("flash2_fn"), std::string::npos) << error.what();
    }
    EXPECT_TRUE(kernels.loads.empty());
}

} // namespace
} // namespace hip_flash2_engine
