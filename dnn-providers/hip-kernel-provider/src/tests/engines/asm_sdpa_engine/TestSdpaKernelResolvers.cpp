// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/plans/SdpaKernelResolvers.hpp"

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace asm_sdpa_engine
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

/// One recorded module load: which entry, on which device.
struct Load
{
    std::string tocKey;
    int deviceOrdinal;

    bool operator==(const Load& other) const
    {
        return tocKey == other.tocKey && deviceOrdinal == other.deviceOrdinal;
    }
};

/// Stands in for the hardware and the archive. A fake module carries no hipModule_t, so
/// destroying it touches no HIP state.
struct FakeDevices
{
    std::map<int, std::string> architectures;
    std::map<int, bool> mi308;

    hipError_t archOf(int deviceOrdinal, std::string& reportedArch) const
    {
        const auto found = architectures.find(deviceOrdinal);
        if(found == architectures.end())
        {
            return hipErrorInvalidDevice;
        }
        reportedArch = found->second;
        return hipSuccess;
    }
};

class FakeFwdResolver : public SdpaFwdKernelResolver
{
public:
    FakeFwdResolver(FakeDevices devices, const std::string& arch)
        : SdpaFwdKernelResolver(unusedCache(), arch, "fmha_v3_fwd/fwd.co", "fwd_fn")
        , _devices(std::move(devices))
    {
    }

    mutable std::vector<int> archQueries;
    mutable std::vector<int> mi308Queries;
    mutable std::vector<Load> loads;

protected:
    hipError_t queryDeviceArch(int deviceOrdinal, std::string& reportedArch) const override
    {
        archQueries.push_back(deviceOrdinal);
        return _devices.archOf(deviceOrdinal, reportedArch);
    }

    bool queryIsMi308(int deviceOrdinal) const override
    {
        mi308Queries.push_back(deviceOrdinal);
        return _devices.mi308.at(deviceOrdinal);
    }

    CachedModule loadModule(const std::string& tocKey, int deviceOrdinal) const override
    {
        loads.push_back({tocKey, deviceOrdinal});
        return std::make_shared<HipModuleGuard>();
    }

private:
    // Never consulted: loadModule is replaced.
    static SdpaModuleCache& unusedCache()
    {
        static SdpaModuleCache s_cache;
        return s_cache;
    }

    FakeDevices _devices;
};

class FakeBwdResolver : public SdpaBwdKernelResolver
{
public:
    FakeBwdResolver(FakeDevices devices, std::optional<SdpaKernelRef> post)
        : SdpaBwdKernelResolver(unusedCache(),
                                "gfx942",
                                SdpaKernelRef{"bwd/odo.co", "odo_fn"},
                                SdpaKernelRef{"bwd/dqdkdv.co", "dqdkdv_fn"},
                                std::move(post))
        , _devices(std::move(devices))
    {
    }

    mutable std::vector<Load> loads;

protected:
    hipError_t queryDeviceArch(int deviceOrdinal, std::string& reportedArch) const override
    {
        return _devices.archOf(deviceOrdinal, reportedArch);
    }

    CachedModule loadModule(const SdpaKernelRef& kernel, int deviceOrdinal) const override
    {
        loads.push_back({kernel.tocKey, deviceOrdinal});
        return std::make_shared<HipModuleGuard>();
    }

private:
    // Never consulted: loadModule is replaced.
    static SdpaModuleCache& unusedCache()
    {
        static SdpaModuleCache s_cache;
        return s_cache;
    }

    FakeDevices _devices;
};

FakeDevices twoGfx942(bool firstIsMi308, bool secondIsMi308)
{
    return FakeDevices{{{0, "gfx942:sramecc+:xnack-"}, {1, "gfx942:sramecc+:xnack-"}},
                       {{0, firstIsMi308}, {1, secondIsMi308}}};
}

TEST(TestSdpaKernelResolvers, Mi308ChipIdsAreRecognised)
{
    EXPECT_TRUE(isMi308ChipId(0x74a2));
    EXPECT_TRUE(isMi308ChipId(0x74bc));
    EXPECT_FALSE(isMi308ChipId(0x74a1));
}

TEST(TestSdpaKernelResolvers, FwdTocKeySelectsTheBoardDirectoryOnGfx942Only)
{
    EXPECT_EQ(fwdKernelTocKey("fmha_v3_fwd/fwd.co", "gfx942", false), "fmha_v3_fwd/MI300/fwd.co");
    EXPECT_EQ(fwdKernelTocKey("fmha_v3_fwd/fwd.co", "gfx942", true), "fmha_v3_fwd/MI308/fwd.co");
    EXPECT_EQ(fwdKernelTocKey("fmha_v3_fwd/fwd.co", "gfx950", true), "fmha_v3_fwd/fwd.co");
}

TEST(TestSdpaKernelResolvers, FwdLoadsASecondOrdinalOfTheSameArchitectureOnThatDevice)
{
    const FakeFwdResolver kernels(twoGfx942(false, false), "gfx942");

    const auto first = kernels.forOrdinal(0);
    const auto second = kernels.forOrdinal(1);

    // Each device gets its own module, loaded on that device.
    EXPECT_NE(first, second);
    EXPECT_EQ(
        kernels.loads,
        (std::vector<Load>{{"fmha_v3_fwd/MI300/fwd.co", 0}, {"fmha_v3_fwd/MI300/fwd.co", 1}}));
}

TEST(TestSdpaKernelResolvers, FwdRederivesTheBoardBinaryForEachDevice)
{
    // MI300- and MI308-class boards share gfx942 but not their forward binaries, so a
    // plan built on one and executed on the other must load the other's entry.
    const FakeFwdResolver kernels(twoGfx942(false, true), "gfx942");

    kernels.forOrdinal(0);
    kernels.forOrdinal(1);

    EXPECT_EQ(
        kernels.loads,
        (std::vector<Load>{{"fmha_v3_fwd/MI300/fwd.co", 0}, {"fmha_v3_fwd/MI308/fwd.co", 1}}));
}

TEST(TestSdpaKernelResolvers, FwdAnswersASeenOrdinalWithoutQueryingOrLoading)
{
    const FakeFwdResolver kernels(twoGfx942(false, false), "gfx942");

    const auto first = kernels.forOrdinal(1);
    const auto repeat = kernels.forOrdinal(1);

    EXPECT_EQ(first, repeat);
    EXPECT_EQ(kernels.archQueries, std::vector<int>{1});
    EXPECT_EQ(kernels.mi308Queries, std::vector<int>{1});
    EXPECT_EQ(kernels.loads.size(), 1U);
}

TEST(TestSdpaKernelResolvers, FwdRefusesAnOrdinalOfAnotherArchitecture)
{
    const FakeFwdResolver kernels(FakeDevices{{{0, "gfx942"}, {1, "gfx950"}}, {{0, false}}},
                                  "gfx942");

    try
    {
        kernels.forOrdinal(1);
        FAIL() << "expected a device of another architecture to be refused";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("fwd_fn"), std::string::npos) << error.what();
    }
    EXPECT_TRUE(kernels.loads.empty());
    EXPECT_TRUE(kernels.mi308Queries.empty());
}

TEST(TestSdpaKernelResolvers, FwdSkipsTheBoardQueryOffGfx942)
{
    const FakeFwdResolver kernels(FakeDevices{{{0, "gfx950"}}, {}}, "gfx950");

    kernels.forOrdinal(0);

    EXPECT_TRUE(kernels.mi308Queries.empty());
    EXPECT_EQ(kernels.loads, (std::vector<Load>{{"fmha_v3_fwd/fwd.co", 0}}));
}

TEST(TestSdpaKernelResolvers, BwdLoadsEveryStageOnASecondOrdinal)
{
    const FakeBwdResolver kernels(twoGfx942(false, false),
                                  SdpaKernelRef{"bwd/dq_convert.co", "dq_convert_fn"});

    kernels.forOrdinal(0);
    const auto second = kernels.forOrdinal(1);

    EXPECT_TRUE(second.post.has_value());
    EXPECT_EQ(kernels.loads,
              (std::vector<Load>{{"bwd/odo.co", 0},
                                 {"bwd/dqdkdv.co", 0},
                                 {"bwd/dq_convert.co", 0},
                                 {"bwd/odo.co", 1},
                                 {"bwd/dqdkdv.co", 1},
                                 {"bwd/dq_convert.co", 1}}));

    // A repeat on a seen ordinal loads nothing.
    kernels.forOrdinal(1);
    EXPECT_EQ(kernels.loads.size(), 6U);
}

TEST(TestSdpaKernelResolvers, BwdA16LoadsNoDqConvertStage)
{
    const FakeBwdResolver kernels(twoGfx942(false, false), std::nullopt);

    const auto modules = kernels.forOrdinal(1);

    EXPECT_FALSE(modules.post.has_value());
    EXPECT_EQ(kernels.loads, (std::vector<Load>{{"bwd/odo.co", 1}, {"bwd/dqdkdv.co", 1}}));
}

TEST(TestSdpaKernelResolvers, BwdRefusesAnOrdinalOfAnotherArchitecture)
{
    const FakeBwdResolver kernels(FakeDevices{{{0, "gfx942"}, {1, "gfx90a"}}, {}}, std::nullopt);

    try
    {
        kernels.forOrdinal(1);
        FAIL() << "expected a device of another architecture to be refused";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
    EXPECT_TRUE(kernels.loads.empty());
}

} // namespace
} // namespace asm_sdpa_engine
