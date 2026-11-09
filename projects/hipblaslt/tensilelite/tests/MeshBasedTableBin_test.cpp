// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/msgpack/MessagePack.hpp>

using namespace TensileLite;
using Serialization::MessagePackInput;

namespace
{
    using Key      = std::array<int64_t, 4>;
    using Distance = Matching::GridBasedDistance<Key>;
    using Element  = std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>;
    using Table = Matching::DistanceMatchingTable<Key,
                                                  ContractionProblemGemm,
                                                  Element,
                                                  std::shared_ptr<ContractionSolution>,
                                                  Distance>;

    struct Record
    {
        int32_t k0, k1, k2, k3, idx;
    };

    // Builds a msgpack MAP with table_bin (BIN) and table_count (INT).
    // If countOverride >= 0, it overrides the actual record count.
    msgpack::sbuffer packTableBinMap(std::vector<Record> const& records,
                                     int                        countOverride = -1)
    {
        int32_t count = countOverride >= 0 ? countOverride
                                           : static_cast<int32_t>(records.size());

        std::vector<uint8_t> bin(records.size() * 20);
        for(size_t i = 0; i < records.size(); ++i)
        {
            uint8_t* rec = bin.data() + i * 20;
            std::memcpy(rec,      &records[i].k0,  4);
            std::memcpy(rec + 4,  &records[i].k1,  4);
            std::memcpy(rec + 8,  &records[i].k2,  4);
            std::memcpy(rec + 12, &records[i].k3,  4);
            std::memcpy(rec + 16, &records[i].idx, 4);
        }

        msgpack::sbuffer                  buf;
        msgpack::packer<msgpack::sbuffer> pk(buf);

        pk.pack_map(2);

        pk.pack(std::string("table_count"));
        pk.pack(count);

        pk.pack(std::string("table_bin"));
        pk.pack_bin(static_cast<uint32_t>(bin.size()));
        pk.pack_bin_body(reinterpret_cast<const char*>(bin.data()),
                         static_cast<uint32_t>(bin.size()));

        return buf;
    }

    // Deserializes a table_bin msgpack buffer into table, returns errors.
    std::vector<std::string> deserializeTable(msgpack::sbuffer const&                buf,
                                              LibraryIOContext<ContractionSolution>& ctx,
                                              Table&                                 table)
    {
        auto             handle = msgpack::unpack(buf.data(), buf.size());
        MessagePackInput reader(handle.get());
        reader.context = &ctx;
        Serialization::MappingTraits<Table, MessagePackInput>::mapping(reader, table);
        return reader.error;
    }

    bool hasError(std::vector<std::string> const& errors, std::string const& substr)
    {
        for(auto const& e : errors)
            if(e.find(substr) != std::string::npos)
                return true;
        return false;
    }
}

TEST(MeshBasedTableBinTest, DecodesKeysAndSolutionIndices)
{
    std::vector<Record> records = {
        {10, 20, 1, 128,  0},
        {30, 40, 1, 256,  1},
        {50, 60, 1, 512,  2},
    };

    SolutionMap<ContractionSolution> solutions;
    for(int i = 0; i < 3; ++i)
    {
        auto sol   = std::make_shared<ContractionSolution>();
        sol->index = i;
        solutions[i] = sol;
    }

    LibraryIOContext<ContractionSolution> ctx{};
    ctx.solutions = &solutions;

    auto  buf    = packTableBinMap(records);
    Table table;
    auto  errors = deserializeTable(buf, ctx, table);

    ASSERT_TRUE(errors.empty()) << "unexpected error: " << errors[0];
    ASSERT_EQ(table.table.size(), 3u);

    // table is sorted by key after deserialization, so find entries by key
    for(auto const& rec : records)
    {
        Key expected = {rec.k0, rec.k1, rec.k2, rec.k3};
        auto it      = std::find_if(table.table.begin(), table.table.end(),
                                    [&](auto const& e) { return e.key == expected; });
        ASSERT_NE(it, table.table.end())
            << "key {" << rec.k0 << "," << rec.k1 << "," << rec.k2 << "," << rec.k3
            << "} not found";

        auto* single = dynamic_cast<
            SingleSolutionLibrary<ContractionProblemGemm, ContractionSolution>*>(
            it->value.get());
        ASSERT_NE(single, nullptr);
        ASSERT_NE(single->solution, nullptr);
        EXPECT_EQ(single->solution->index, rec.idx);
    }
}

TEST(MeshBasedTableBinTest, SizeMismatchSetsError)
{
    // 3 records in the blob but table_count says 2
    std::vector<Record> records = {
        {1, 2, 3, 4, 0},
        {5, 6, 7, 8, 0},
        {9, 10, 11, 12, 0},
    };

    SolutionMap<ContractionSolution> solutions;
    auto sol   = std::make_shared<ContractionSolution>();
    sol->index = 0;
    solutions[0] = sol;

    LibraryIOContext<ContractionSolution> ctx{};
    ctx.solutions = &solutions;

    auto  buf    = packTableBinMap(records, /*countOverride=*/2);
    Table table;
    auto  errors = deserializeTable(buf, ctx, table);

    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(hasError(errors, "size mismatch")) << "no size-mismatch error found";
}

TEST(MeshBasedTableBinTest, InvalidSolutionIndexSetsError)
{
    // Record references index 99 which is not in the solutions map
    std::vector<Record> records = {
        {1, 2, 3, 4, 99},
    };

    SolutionMap<ContractionSolution> solutions;
    auto sol   = std::make_shared<ContractionSolution>();
    sol->index = 0;
    solutions[0] = sol;

    LibraryIOContext<ContractionSolution> ctx{};
    ctx.solutions = &solutions;

    auto  buf    = packTableBinMap(records);
    Table table;
    auto  errors = deserializeTable(buf, ctx, table);

    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(hasError(errors, "Invalid solution index"))
        << "no invalid-index error found";
}
