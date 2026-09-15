// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <set>
#include <string>

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
    const auto& [gcnArchName,
                 warpSize,
                 multiProcessorCount,
                 totalGlobalMem,
                 memoryBusWidth,
                 memoryClockRate,
                 sharedMemPerBlock]
        = properties;

    EXPECT_EQ(gcnArchName, "gfx942");
    EXPECT_EQ(warpSize, 64);
    EXPECT_EQ(multiProcessorCount, 304);
    EXPECT_EQ(totalGlobalMem, 0U);
    EXPECT_EQ(memoryBusWidth, 0);
    EXPECT_EQ(memoryClockRate, 0);
    EXPECT_EQ(sharedMemPerBlock, 0U);
}

TEST(TestIngestorDeviceKey, TwoBoardsOfOneArchSharingCuCountStillGetDifferentKeys)
{
    // The case that motivated widening the struct. One arch spans several boards, and
    // boards exist that carry the same compute-unit count and different memory. Keyed
    // on arch, warp size and CUs alone these two collide, and a ranking benchmarked on
    // the smaller-bandwidth card is served to the faster one as if it had been measured
    // there -- silently, because a cache hit looks exactly like a correct answer.
    auto slower = propertiesFor("gfx942");
    slower.totalGlobalMem = 192ULL * 1024 * 1024 * 1024;
    slower.memoryBusWidth = 8192;
    slower.memoryClockRate = 2600000;

    auto faster = propertiesFor("gfx942");
    faster.totalGlobalMem = 256ULL * 1024 * 1024 * 1024;
    faster.memoryBusWidth = 8192;
    faster.memoryClockRate = 3000000;

    ASSERT_EQ(slower.multiProcessorCount, faster.multiProcessorCount)
        << "the fixture must model the collision it is testing";
    EXPECT_NE(DeviceKey{slower}, DeviceKey{faster});
    EXPECT_NE(DeviceKey{slower}.hash(), DeviceKey{faster}.hash());
}

TEST(TestIngestorDeviceKey, LdsSizeDiscriminates)
{
    auto small = propertiesFor("gfx942");
    small.sharedMemPerBlock = 64 * 1024;
    auto large = propertiesFor("gfx942");
    large.sharedMemPerBlock = 160 * 1024;

    EXPECT_NE(DeviceKey{small}, DeviceKey{large});
}

TEST(TestIngestorDeviceKey, IdenticalPropertiesCompareEqual)
{
    EXPECT_EQ(DeviceKey{propertiesFor("gfx942")}, DeviceKey{propertiesFor("gfx942")});
}

// Why the key folds the whole struct rather than the arch string: two parts reporting
// the same arch can differ in compute units, and a kernel timed on one is not necessarily
// the winner on the other.
TEST(TestIngestorDeviceKey, DevicesDifferingOnlyInComputeUnitsCompareUnequal)
{
    // Not named "small": that is a macro once <windows.h> is in the translation unit
    // (rpcndr.h defines it as char), and this file is one include away from pulling it in.
    const auto fewerUnits = propertiesFor("gfx942", 64, 228);
    const auto moreUnits = propertiesFor("gfx942", 64, 304);

    EXPECT_NE(DeviceKey{fewerUnits}, DeviceKey{moreUnits});
}

TEST(TestIngestorDeviceKey, DevicesDifferingOnlyInWarpSizeCompareUnequal)
{
    EXPECT_NE(DeviceKey{propertiesFor("gfx942", 32)}, DeviceKey{propertiesFor("gfx942", 64)});
}

TEST(TestIngestorDeviceKey, ADifferentArchComparesUnequal)
{
    EXPECT_NE(DeviceKey{propertiesFor("gfx942")}, DeviceKey{propertiesFor("gfx950")});
}

// gcnArchName is raw and suffixed; the cache key compares it exactly, unlike the
// pack-level arch gate, which PREFIX-matches: a measurement taken with one feature set
// is not claimed to hold for another.
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

    EXPECT_EQ(std::hash<DeviceKey>{}(key), static_cast<size_t>(key.hash()));
}

/// The hash narrows; the fields decide. Each case forces the fold to agree and varies
/// exactly one field, so a comparison that dropped that field would serve a ranking
/// measured on one device for another. Varying the arch alone would not catch that:
/// `gcnArchName` is the field a regressed comparison is likeliest to keep.
class TestIngestorDeviceKeyCollision : public ::testing::TestWithParam<DeviceProperties>
{
protected:
    struct CollidingKey : DeviceKey
    {
        using DeviceKey::DeviceKey;

        // Force the narrowing half to agree, leaving only the field comparison.
        static CollidingKey with(DeviceProperties properties, uint64_t forcedHash)
        {
            CollidingKey key{std::move(properties)};
            key.forceHash(forcedHash);
            return key;
        }
    };
};

TEST_P(TestIngestorDeviceKeyCollision, EqualHashesWithADifferingFieldStillCompareUnequal)
{
    auto baseline = CollidingKey::with(propertiesFor("gfx942"), 0xABCD);
    auto variant = CollidingKey::with(GetParam(), 0xABCD);

    ASSERT_EQ(baseline.hash(), variant.hash()) << "the collision must actually be forced";
    EXPECT_NE(static_cast<const DeviceKey&>(baseline), static_cast<const DeviceKey&>(variant))
        << "a hash collision with differing properties must resolve to a miss";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestIngestorDeviceKeyCollision,
                         ::testing::Values(propertiesFor("gfx950"),
                                           propertiesFor("gfx942", /*warpSize=*/32),
                                           propertiesFor("gfx942", 64, /*computeUnits=*/228)),
                         [](const ::testing::TestParamInfo<DeviceProperties>& info) {
                             switch(info.index)
                             {
                             case 0:
                                 return "ByArch";
                             case 1:
                                 return "ByWarpSize";
                             default:
                                 return "ByComputeUnits";
                             }
                         });


// ---- the $device.* vocabulary ------------------------------------------------------
//
// One arch spans several boards and a UHD is arch-keyed, so a gfx942 model is trained on
// a corpus merged from MI300X, MI325X and MI308X. Two things read the device facts and
// they have to agree: FeatureExtractor binds them at scoring time, and the benchmark
// recorder writes them as `device.*` columns into that corpus. A name in one and not the
// other is a feature trained on a column the runtime cannot produce, or a binding no
// corpus ever held -- neither throws, and both yield a model that is quietly wrong.

TEST(TestIngestorDeviceVocabulary, EveryNameIsBoundExactlyOnce)
{
    auto properties = propertiesFor("gfx942");
    properties.totalGlobalMem = 192ULL * 1024 * 1024 * 1024;
    properties.memoryBusWidth = 8192;
    properties.memoryClockRate = 2600000;
    properties.sharedMemPerBlock = 64 * 1024;

    std::set<std::string> names;
    for(const auto& entry : deviceFeatureValues(properties))
    {
        EXPECT_TRUE(names.insert(entry.first).second)
            << entry.first << " is emitted twice; one of them silently wins";
    }

    // Both spellings of the CU count, because signatures exist against each.
    EXPECT_EQ(names.count("cu_count"), 1U);
    EXPECT_EQ(names.count("multi_processor_count"), 1U);
    // The fields that separate boards of one arch.
    EXPECT_EQ(names.count("total_global_mem"), 1U);
    EXPECT_EQ(names.count("memory_bus_width"), 1U);
    EXPECT_EQ(names.count("memory_clock_rate"), 1U);
    EXPECT_EQ(names.count("peak_memory_bandwidth"), 1U);
    EXPECT_EQ(names.count("lds_size"), 1U);
    // Never the arch: it selects which UHD runs, so a model splitting on it would be
    // splitting on the thing that chose it (RFC 0019 3.1).
    EXPECT_EQ(names.count("arch"), 0U);
    EXPECT_EQ(names.count("gcn_arch_name"), 0U);
}

TEST(TestIngestorDeviceVocabulary, PeakBandwidthIsDerivedFromClockAndWidth)
{
    auto properties = propertiesFor("gfx942");
    properties.memoryBusWidth = 8192;
    properties.memoryClockRate = 2600000;  // kHz

    // 2 (DDR) * 2.6e9 Hz * 1024 bytes.
    EXPECT_DOUBLE_EQ(peakMemoryBandwidth(properties), 2.0 * 2600000.0 * 1000.0 * 1024.0);
}

TEST(TestIngestorDeviceVocabulary, PeakBandwidthIsZeroRatherThanWrongWhenUnresolved)
{
    // hipGetDeviceProperties leaves 0 on fields it cannot answer. Multiplying those out
    // would report a device with zero bandwidth as though it were measured, so the
    // derived value stays 0 and a signature reading it sees the same "unknown" the
    // inputs carry.
    auto properties = propertiesFor("gfx942");
    EXPECT_DOUBLE_EQ(peakMemoryBandwidth(properties), 0.0);

    properties.memoryBusWidth = 8192;
    EXPECT_DOUBLE_EQ(peakMemoryBandwidth(properties), 0.0) << "clock still unresolved";
}
} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
