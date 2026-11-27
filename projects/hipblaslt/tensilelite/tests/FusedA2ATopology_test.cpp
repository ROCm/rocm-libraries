// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "FusedA2ATopology.hpp"

using namespace TensileLite::Client;

namespace
{
    PciBusId parse(char const* text)
    {
        PciBusId id;
        EXPECT_TRUE(parsePciBusId(text, id)) << text;
        return id;
    }
}

TEST(FusedA2ATopology, ParsesPciBusId)
{
    const PciBusId id = parse("0001:c5:1f.1");
    EXPECT_EQ(id.domain, 0x1u);
    EXPECT_EQ(id.bus, 0xc5u);
    EXPECT_EQ(id.device, 0x1fu);
    EXPECT_EQ(id.function, 0x1u);

    PciBusId bad;
    EXPECT_FALSE(parsePciBusId("0000:c5:00", bad));
    EXPECT_FALSE(parsePciBusId("", bad));
}

TEST(FusedA2ATopology, BdfIdTellsTheFunctionsOfACardApart)
{
    EXPECT_EQ(pciBdfId(parse("0000:04:00.0")), 1024u);
    EXPECT_EQ(pciBdfId(parse("0000:04:00.1")), 1025u);
    EXPECT_EQ(pciBdfId(parse("0001:c5:1f.1")), 0xc5f9u);
}
