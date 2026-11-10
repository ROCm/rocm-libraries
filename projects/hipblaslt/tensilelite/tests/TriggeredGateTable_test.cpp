// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "TriggeredGateValidator.hpp"

#include <Tensile/GateKernArg.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

TEST(TriggeredGateTable, LayoutPutsGatesOnTheirOwnLines)
{
    auto layout = gateTableLayout(4, 56);
    EXPECT_EQ(layout.gatesOffset, 256u); // 48 + 4 * 48 = 240, rounded up to 64
    EXPECT_EQ(layout.bytes, 256u + 4 * 64);
}

TEST(TriggeredGateTable, FillWritesHeaderRecordsAndZeroGates)
{
    uint64_t const       base   = 0x7f0000001000ull;
    auto                 layout = gateTableLayout(4, 56);
    std::vector<uint8_t> host;
    fillGateTable(host, layout, base);
    ASSERT_EQ(host.size(), layout.bytes);

    GateTableHeader header;
    std::memcpy(&header, host.data(), sizeof(header));
    EXPECT_EQ(header.abiVersion, GATE_ABI_VERSION);
    EXPECT_EQ(header.tileRecordSize, sizeof(GateTileRecord));
    EXPECT_EQ(header.tileCount, 4u);
    EXPECT_EQ(header.reserved, 0u);
    EXPECT_EQ(header.heapBase, base);
    EXPECT_EQ(header.tiles, base + sizeof(GateTableHeader));

    for(uint32_t b = 0; b < 4; b++)
    {
        GateTileRecord record;
        std::memcpy(&record,
                    host.data() + sizeof(GateTableHeader) + b * sizeof(GateTileRecord),
                    sizeof(record));
        EXPECT_EQ(record.tileId, b);
        EXPECT_EQ(record.roleFlags & GATE_ROLE_PUBLISH, GATE_ROLE_PUBLISH);
        EXPECT_EQ(record.publisherCount, 56u);
        EXPECT_EQ(record.readyOffset, layout.gatesOffset + b * 64);
    }

    for(size_t i = layout.gatesOffset; i < layout.bytes; i++)
        ASSERT_EQ(host[i], 0) << "gate byte " << i;
}
