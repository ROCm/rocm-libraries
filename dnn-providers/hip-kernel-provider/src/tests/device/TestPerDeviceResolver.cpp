// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "device/PerDeviceResolver.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hip_kernel_provider::device
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

/// Stands in for the hardware: each ordinal reports an architecture from a table, and a
/// resolution yields the ordinal it was made for. Every query and resolution is recorded.
class FakeResolver : public PerDeviceResolver<int>
{
public:
    FakeResolver(const std::string& builtArch, std::map<int, std::string> architectures)
        : PerDeviceResolver<int>(builtArch, "fake kernel")
        , _architectures(std::move(architectures))
    {
    }

    mutable std::vector<int> archQueries;
    mutable std::vector<int> resolves;

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

    int resolveForDevice(int deviceOrdinal, const std::string& /*reportedArch*/) const override
    {
        resolves.push_back(deviceOrdinal);
        return deviceOrdinal;
    }

private:
    std::map<int, std::string> _architectures;
};

TEST(TestPerDeviceResolver, ASeededOrdinalIsAnsweredWithoutQueryingOrResolving)
{
    FakeResolver resolver("gfx942", {{0, "gfx942"}});
    resolver.seed(0, 100);

    EXPECT_EQ(resolver.forOrdinal(0), 100);
    EXPECT_TRUE(resolver.archQueries.empty());
    EXPECT_TRUE(resolver.resolves.empty());
}

TEST(TestPerDeviceResolver, ResolvesASecondOrdinalOfTheSameArchitecture)
{
    FakeResolver resolver("gfx942:sramecc+:xnack-",
                          {{0, "gfx942:sramecc+:xnack-"}, {1, "gfx942:sramecc+:xnack-"}});
    resolver.seed(0, 100);

    // Device 1 gets its own resolution rather than device 0's.
    EXPECT_EQ(resolver.forOrdinal(1), 1);
    EXPECT_EQ(resolver.resolves, std::vector<int>{1});
    EXPECT_EQ(resolver.forOrdinal(0), 100);
    EXPECT_EQ(resolver.resolvedCount(), 2U);
}

TEST(TestPerDeviceResolver, AnswersASeenOrdinalWithoutQueryingOrResolving)
{
    const FakeResolver resolver("gfx942", {{1, "gfx942"}});

    resolver.forOrdinal(1);
    resolver.forOrdinal(1);

    EXPECT_EQ(resolver.archQueries, std::vector<int>{1});
    EXPECT_EQ(resolver.resolves, std::vector<int>{1});
}

TEST(TestPerDeviceResolver, ComparesArchitecturesWithoutTheirFeatureSuffix)
{
    // Validity keys on the base architecture: the feature suffix describes the device,
    // not the kernel selection.
    const FakeResolver resolver("gfx942", {{1, "gfx942:sramecc-:xnack+"}});

    EXPECT_EQ(resolver.forOrdinal(1), 1);
    EXPECT_EQ(resolver.builtArch(), "gfx942");
}

TEST(TestPerDeviceResolver, RefusesAnOrdinalOfAnotherArchitecture)
{
    const FakeResolver resolver("gfx942", {{1, "gfx950"}});

    try
    {
        resolver.forOrdinal(1);
        FAIL() << "expected a device of another architecture to be refused";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        const std::string message = error.what();
        EXPECT_NE(message.find("fake kernel"), std::string::npos) << message;
        EXPECT_NE(message.find("gfx942"), std::string::npos) << message;
        EXPECT_NE(message.find("gfx950"), std::string::npos) << message;
    }

    // Refused before anything was loaded, and not remembered as resolved.
    EXPECT_TRUE(resolver.resolves.empty());
    EXPECT_EQ(resolver.resolvedCount(), 0U);
}

TEST(TestPerDeviceResolver, ReportsADeviceThatCannotBeQueried)
{
    const FakeResolver resolver("gfx942", {});

    try
    {
        resolver.forOrdinal(3);
        FAIL() << "expected an unqueryable device to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
    EXPECT_TRUE(resolver.resolves.empty());
}

TEST(TestPerDeviceResolver, RefusesANegativeOrdinal)
{
    FakeResolver resolver("gfx942", {{0, "gfx942"}});

    EXPECT_THROW(resolver.forOrdinal(-1), HipdnnPluginException);
    EXPECT_THROW(resolver.seed(-1, 0), HipdnnPluginException);
    EXPECT_TRUE(resolver.archQueries.empty());
}

} // namespace
} // namespace hip_kernel_provider::device
