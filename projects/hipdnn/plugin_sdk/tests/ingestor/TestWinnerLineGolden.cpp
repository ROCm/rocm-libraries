// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>

#include "../TestResourcePaths.hpp"

/**
 * @file TestWinnerLineGolden.cpp
 * @brief The C++ half of the winner-line guard. `winner_lines_golden.jsonl` is the shard
 * `tools/IngestorGenerator/tools/variant_reduce.py` reads; `test_winner_line_golden.py`
 * reads the same file and asserts the same values. This test decodes it with the SDK
 * reader and compares what it decodes, and what the encoder writes back, with those
 * values field by field (semantic JSON equality, not bytes).
 */
namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

constexpr const char* UPDATE_REDUCER
    = "The winner-cache line format changed. Update "
      "projects/hipdnn/tools/IngestorGenerator/tools/variant_reduce.py (its accepted line "
      "versions and line decoding) and the golden shard "
      "projects/hipdnn/tools/IngestorGenerator/tests/fixtures/winner_lines_golden.jsonl, "
      "then run tests/test_winner_line_golden.py.";

constexpr const char* GOLDEN_FILE = "winner_lines_golden.jsonl";

std::vector<std::string> goldenLines()
{
    const auto path
        = hipdnn_plugin_sdk::test::testDataDir() / HIPDNN_WINNER_GOLDEN_SUBDIR / GOLDEN_FILE;
    std::ifstream stream(path);
    EXPECT_TRUE(stream.is_open()) << "golden shard not staged: " << path;
    std::vector<std::string> lines;
    for(std::string line; std::getline(stream, line);)
    {
        lines.push_back(line);
    }
    // Line 1 is the shard's version stamp; the records follow.
    EXPECT_FALSE(lines.empty());
    if(!lines.empty())
    {
        lines.erase(lines.begin());
    }
    return lines;
}

DescriptorId uuid(const char* text)
{
    return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
}

struct ExpectedEntry
{
    const char* kernel;
    const char* pack;
    const char* dispatch;
    double timeMs;
};

void expectRanking(const WinnerRecord& record,
                   const std::vector<ExpectedEntry>& expected,
                   const std::string& line)
{
    ASSERT_EQ(record.size(), expected.size()) << line << "\n" << UPDATE_REDUCER;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(record[i].kernelId, uuid(expected[i].kernel)) << line;
        EXPECT_EQ(record[i].packId, uuid(expected[i].pack)) << line;
        EXPECT_EQ(record[i].dispatchId, uuid(expected[i].dispatch)) << line;
        EXPECT_DOUBLE_EQ(record[i].timeMs, expected[i].timeMs) << line;
    }
}

void expectGoldenDevice(const DeviceProperties& device, const std::string& line)
{
    EXPECT_EQ(device.gcnArchName, "gfx950") << line;
    EXPECT_EQ(device.warpSize, 64) << line;
    EXPECT_EQ(device.multiProcessorCount, 256) << line;
    EXPECT_EQ(device.totalGlobalMem, 309220868096ULL) << line;
    EXPECT_EQ(device.memoryBusWidth, 8192) << line;
    EXPECT_EQ(device.memoryClockRate, 1900000) << line;
    EXPECT_EQ(device.sharedMemPerBlock, 163840U) << line;
}

constexpr const char* GRAPH_CONTENT
    = "HAAAABgAHAAIAAAAAAAHAAAAAAAAAAAAAAAMABgAAAAAAAABFAAAAAECAwQFBgcICQoLDA0ODxAGAAAAZ29sZGVu"
      "AAA=";

constexpr const char* K1 = "c1000000-0000-4000-8000-000000000001";
constexpr const char* K2 = "c2000000-0000-4000-8000-000000000002";
constexpr const char* K3 = "c3000000-0000-4000-8000-000000000003";
constexpr const char* K4 = "c4000000-0000-4000-8000-000000000004";
constexpr const char* K5 = "c5000000-0000-4000-8000-000000000005";
constexpr const char* P1 = "a1000000-0000-4000-8000-000000000001";
constexpr const char* P2 = "a2000000-0000-4000-8000-000000000002";
constexpr const char* P3 = "a3000000-0000-4000-8000-000000000003";
constexpr const char* D1 = "b1000000-0000-4000-8000-000000000001";
constexpr const char* D2 = "b2000000-0000-4000-8000-000000000002";
constexpr const char* D3 = "b3000000-0000-4000-8000-000000000003";

TEST(TestWinnerLineGolden, TheLineVersionsAreTheOnesTheReducerAccepts)
{
    EXPECT_EQ(detail::WINNER_LINE_FORMAT_VERSION, 2) << UPDATE_REDUCER;
    EXPECT_EQ(detail::WINNER_LINE_SLOTTED_FORMAT_VERSION, 3) << UPDATE_REDUCER;
}

TEST(TestWinnerLineGolden, TheSdkReaderDecodesEveryGoldenLineToTheExpectedFields)
{
    const auto lines = goldenLines();
    ASSERT_EQ(lines.size(), 4U);

    std::vector<std::optional<DecodedWinnerLine>> decoded;
    for(const auto& line : lines)
    {
        decoded.push_back(decodeWinnerLine(line));
        ASSERT_TRUE(decoded.back().has_value()) << line << "\n" << UPDATE_REDUCER;
        EXPECT_EQ(decoded.back()->key.graph.toJson().at("content_base64").get<std::string>(),
                  GRAPH_CONTENT);
        expectGoldenDevice(decoded.back()->key.device.properties(), line);
    }

    // v2: a whole-graph record.
    EXPECT_TRUE(decoded[0]->key.slot.empty());
    EXPECT_FALSE(decoded[0]->program.has_value());
    expectRanking(decoded[0]->record, {{K1, P1, D1, 1.5}, {K2, P1, D1, 2.25}}, lines[0]);

    // v3: one slot record per slot.
    EXPECT_EQ(decoded[1]->key.slot, "dkdv");
    EXPECT_FALSE(decoded[1]->program.has_value());
    expectRanking(decoded[1]->record, {{K3, P2, D2, 0.5}, {K4, P2, D2, 0.75}}, lines[1]);
    EXPECT_EQ(decoded[2]->key.slot, "dq");
    expectRanking(decoded[2]->record, {{K5, P3, D3, 1.25}}, lines[2]);

    // v3: the program line, which ranks nothing.
    ASSERT_TRUE(decoded[3]->program.has_value());
    const auto& program = *decoded[3]->program;
    ASSERT_EQ(program.picks.size(), 2U);
    EXPECT_EQ(program.picks[0].slot, "dkdv");
    EXPECT_EQ(program.picks[0].kernelId, uuid(K3));
    EXPECT_EQ(program.picks[1].slot, "dq");
    EXPECT_EQ(program.picks[1].kernelId, uuid(K5));
    EXPECT_DOUBLE_EQ(program.timeMs, 4.0);
    EXPECT_FALSE(decodeWinnerRecordLine(lines[3]).has_value());
}

TEST(TestWinnerLineGolden, TheEncoderWritesTheGoldenFieldsBack)
{
    const auto lines = goldenLines();
    ASSERT_EQ(lines.size(), 4U);
    for(const auto& line : lines)
    {
        const auto decoded = decodeWinnerLine(line);
        ASSERT_TRUE(decoded.has_value()) << line << "\n" << UPDATE_REDUCER;
        const auto encoded = decoded->program.has_value()
                                 ? encodeProgramLine(decoded->key, *decoded->program)
                                 : encodeWinnerRecordLine(decoded->key, decoded->record);
        EXPECT_EQ(nlohmann::json::parse(encoded), nlohmann::json::parse(line))
            << "encoded: " << encoded << "\ngolden:  " << line << "\n"
            << UPDATE_REDUCER;
    }
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
