// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU-only serialization tests for ConvHipConv's performance config. No GPU and no
// hipconv build required: Serialize/Deserialize are header-inline.
//
// The config is stored as its kernel descriptor, family[field=value,...]. Multi-field
// descriptors contain commas, so they must be read back whole rather than split into
// fields. These tests cover the perf-config picker path (Deserialize of a ranked
// descriptor) and the perf-db path (a text record parsed by DbRecord).

#include <gtest/gtest.h>

#include <miopen/conv/solvers.hpp>
#include <miopen/db_record.hpp>

#include <sstream>
#include <string>
#include <vector>

namespace {

using miopen::solver::conv::PerformanceConfigConvHipConv;

// One descriptor per shape seen in the shipped pcfg catalog: comma-separated fields,
// space-separated fields, and a family with no fields.
const std::vector<std::string>& Descriptors()
{
    static const std::vector<std::string> d = {
        "direct[tile_size_k=128,tile_size_n=1,tile_size_h=16,tile_size_w=16,aligned=false]",
        "depthwise_1d_toeplitz[dw1d k11 s1 D w1 c8 nc b2]",
        "patch_embed",
    };
    return d;
}

std::string Serialized(const PerformanceConfigConvHipConv& c)
{
    std::ostringstream ss;
    c.Serialize(ss);
    return ss.str();
}

TEST(CPU_ConvHipConvPerfConfig_NONE, DeserializeKeepsWholeDescriptor)
{
    for(const auto& desc : Descriptors())
    {
        PerformanceConfigConvHipConv c;
        ASSERT_TRUE(c.Deserialize(desc)) << desc;
        EXPECT_EQ(c.descriptor, desc);
        EXPECT_EQ(Serialized(c), desc);
    }
}

TEST(CPU_ConvHipConvPerfConfig_NONE, DeserializeClearsResolvedIndex)
{
    // A stale index would make IsValid() skip resolving the new descriptor.
    PerformanceConfigConvHipConv c;
    c.index = 5;
    ASSERT_TRUE(c.Deserialize(Descriptors().front()));
    EXPECT_EQ(c.index, -1);
}

TEST(CPU_ConvHipConvPerfConfig_NONE, DeserializeRejectsEmpty)
{
    PerformanceConfigConvHipConv c;
    EXPECT_FALSE(c.Deserialize(""));
}

TEST(CPU_ConvHipConvPerfConfig_NONE, RoundTripsThroughPerfDbRecord)
{
    for(const auto& desc : Descriptors())
    {
        // A comma in the stored value is what the field reader split on. ':' and ';'
        // are DbRecord's own id/entry separators, which a value must not contain.
        ASSERT_EQ(desc.find_first_of(":;"), std::string::npos) << desc;

        PerformanceConfigConvHipConv written;
        ASSERT_TRUE(written.Deserialize(desc));

        // Store and reload through DbRecord the way a perf-db update/load does
        // (Serialize into the record, Deserialize back out).
        miopen::DbRecord record(miopen::DbKinds::PerfDb, std::string{"key"});
        ASSERT_TRUE(record.SetValues("ConvHipConv", written));

        PerformanceConfigConvHipConv loaded;
        ASSERT_TRUE(record.GetValues("ConvHipConv", loaded)) << desc;
        EXPECT_EQ(loaded.descriptor, desc);
    }
}

} // namespace
