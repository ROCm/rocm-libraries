// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

namespace hipdnn_plugin_sdk::ingestor::testing
{
namespace
{

DeviceProperties propertiesFor(std::string arch, int warpSize = 64, int computeUnits = 304)
{
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = warpSize;
    properties.multiProcessorCount = computeUnits;
    return properties;
}

// THE FIELD-SET PIN. A structured binding must name every member of the aggregate, so
// this stops compiling the moment DeviceProperties grows a field -- which is the point.
// A new field is not hashed until DeviceKey::fold() emits it, and nothing else in the
// build would notice. If this fails to compile: add the field to fold(), add a
// discriminates-on test below, then extend this binding.
TEST(TestIngestorDeviceKey, TheHashedFieldSetIsPinnedAtCompileTime)
{
    const auto properties = propertiesFor("gfx942");
    const auto& [gcnArchName, warpSize, multiProcessorCount] = properties;

    EXPECT_EQ(gcnArchName, "gfx942");
    EXPECT_EQ(warpSize, 64);
    EXPECT_EQ(multiProcessorCount, 304);
}

TEST(TestIngestorDeviceKey, IdenticalPropertiesCompareEqual)
{
    EXPECT_EQ(DeviceKey{propertiesFor("gfx942")}, DeviceKey{propertiesFor("gfx942")});
}

// The reason C6 keys on the whole struct rather than the arch string: two parts reporting
// the same arch can differ in compute units, and a kernel timed on one is not necessarily
// the winner on the other.
TEST(TestIngestorDeviceKey, DevicesDifferingOnlyInComputeUnitsCompareUnequal)
{
    const auto small = propertiesFor("gfx942", 64, 228);
    const auto large = propertiesFor("gfx942", 64, 304);

    EXPECT_NE(DeviceKey{small}, DeviceKey{large});
}

TEST(TestIngestorDeviceKey, DevicesDifferingOnlyInWarpSizeCompareUnequal)
{
    EXPECT_NE(DeviceKey{propertiesFor("gfx942", 32)}, DeviceKey{propertiesFor("gfx942", 64)});
}

TEST(TestIngestorDeviceKey, ADifferentArchComparesUnequal)
{
    EXPECT_NE(DeviceKey{propertiesFor("gfx942")}, DeviceKey{propertiesFor("gfx950")});
}

// gcnArchName is raw and suffixed. The cache key is exact, deliberately: unlike the
// pack-level arch gate, which PREFIX-matches, a measurement taken with one feature set is
// not claimed to hold for another.
TEST(TestIngestorDeviceKey, ASuffixedArchIsNotTheSameKeyAsItsBaseIdentifier)
{
    EXPECT_NE(DeviceKey{propertiesFor("gfx942")},
              DeviceKey{propertiesFor("gfx942:sramecc+:xnack-")});
}

// Length precedes the arch characters in the fold, so a shorter name with a larger
// warpSize cannot serialize to the same bytes as a longer name with a smaller one.
TEST(TestIngestorDeviceKey, FieldBoundariesDoNotBleedIntoOneAnother)
{
    EXPECT_NE(DeviceKey{propertiesFor("gfx9", 42)}, DeviceKey{propertiesFor("gfx942", 0)});
}

TEST(TestIngestorDeviceKey, AnUnresolvedDeviceStillKeysDistinctlyFromAResolvedOne)
{
    const DeviceProperties unresolved;

    EXPECT_NE(DeviceKey{unresolved}, DeviceKey{propertiesFor("gfx942")});
}

TEST(TestIngestorDeviceKey, StdHashAgreesWithTheKeysOwnHash)
{
    const DeviceKey key{propertiesFor("gfx942")};

    EXPECT_EQ(std::hash<DeviceKey>{}(key), static_cast<size_t>(key.hash));
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
