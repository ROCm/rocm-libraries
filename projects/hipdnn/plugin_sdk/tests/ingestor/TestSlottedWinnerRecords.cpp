// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>

#include "KernelIngestorTestFixtures.hpp"
#include "flatbuffer_utilities/ContentCarryingTestGraph.hpp"

/**
 * @file TestSlottedWinnerRecords.cpp
 * @brief winners.jsonl v3 (slot records and program lines), slot-keyed record adoption, program
 *        record validity at read time, and the one-locked-append `recordSlotted` (RFC 0021).
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::ContentCarryingTestGraph;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::makeGraphId;

using Manager = KernelIngestorStateManager<StubHandle>;
using SlotRecords = std::vector<std::pair<std::string, WinnerRecord>>;

constexpr const char* GRAPH_MATCH = "hipdnn.kernel_ingestor.test.slotted_records.graph_match";
constexpr const char* TILE = "tile";

constexpr uint8_t PREP_PACK = 0x10;
constexpr uint8_t DKDV_PACK = 0x20;
constexpr uint8_t DQ_PACK = 0x30;

class ScopedGraphMatch
{
public:
    ScopedGraphMatch()
    {
        GraphMatchRegistry::registerSymbol(GRAPH_MATCH, &acceptAnyGraph);
    }

    ~ScopedGraphMatch()
    {
        GraphMatchRegistry::unregisterSymbol(GRAPH_MATCH);
    }

    ScopedGraphMatch(const ScopedGraphMatch&) = delete;
    ScopedGraphMatch& operator=(const ScopedGraphMatch&) = delete;
    ScopedGraphMatch(ScopedGraphMatch&&) = delete;
    ScopedGraphMatch& operator=(ScopedGraphMatch&&) = delete;
};

MetadataSchema slotSchema(uint8_t seed, const std::string& name)
{
    return {testId(seed), name, {{TILE, MetadataType::INT, std::nullopt}}};
}

KernelDescriptorPack
    slotPack(uint8_t packSeed, const std::string& slot, const std::vector<uint8_t>& kernelSeeds)
{
    std::vector<KernelDescriptor> kernels;
    kernels.reserve(kernelSeeds.size());
    for(const auto seed : kernelSeeds)
    {
        kernels.push_back(makeSlotKernel(seed,
                                         slot + "_" + std::to_string(seed),
                                         {{TILE, MetadataValue{static_cast<int64_t>(seed)}}}));
    }
    return makeSlotPack(packSeed, slot, std::move(kernels));
}

KernelSlotInput slotInput(const std::string& name,
                          uint8_t schemaSeed,
                          uint8_t packSeed,
                          const std::vector<uint8_t>& kernels)
{
    return {name,
            slotSchema(schemaSeed, name + " kmd"),
            {slotPack(packSeed, name, kernels)},
            std::make_shared<UnrankedKernelHeuristic>()};
}

/// prep {0x11, 0x12}, dkdv {0x21, 0x22, 0x23}, dq {0x31, 0x32}; @p extraDkdv adds kernel 0x24.
std::unique_ptr<Manager> makeManager(const EngineIdentity& engine,
                                     bool extraDkdv = false,
                                     size_t winnerCapacity = Manager::DEFAULT_WINNER_CACHE_CAPACITY)
{
    std::vector<KernelSlotInput> slots;
    slots.reserve(3);
    slots.push_back(slotInput("prep", 0xA1, PREP_PACK, {0x11, 0x12}));
    slots.push_back(slotInput("dkdv",
                              0xA2,
                              DKDV_PACK,
                              extraDkdv ? std::vector<uint8_t>{0x21, 0x22, 0x23, 0x24}
                                        : std::vector<uint8_t>{0x21, 0x22, 0x23}));
    slots.push_back(slotInput("dq", 0xA3, DQ_PACK, {0x31, 0x32}));
    return std::make_unique<Manager>(std::move(slots),
                                     std::vector<MatchDescriptor>{},
                                     makeStubDispatches(),
                                     GRAPH_MATCH,
                                     std::string{},
                                     "engine 'slotted records'",
                                     Manager::DEFAULT_CATALOG_CACHE_CAPACITY,
                                     engine,
                                     winnerCapacity);
}

const EngineIdentity& namedEngine()
{
    static const EngineIdentity s_engine{
        "test:SlottedRecords", {1, 0, 0}, toString(ENGINE_ID), "abcd1234abcd1234"};
    return s_engine;
}

ContentCarryingTestGraph makeGraph()
{
    ContentCarryingTestGraph::Spec spec;
    spec.graphId = makeGraphId(1);
    return ContentCarryingTestGraph{std::move(spec)};
}

WinnerKey baseKeyFor(const ContentCarryingTestGraph& graph, const DeviceProperties& properties)
{
    return WinnerKey{GraphContentKey{graph}, DeviceKey{properties}};
}

/// A ranking of @p packSeed's kernels in the given order, with ascending times.
WinnerRecord ranking(uint8_t packSeed, const std::vector<uint8_t>& kernels)
{
    return makeRanking(testId(packSeed), DISPATCH_ID, kernels);
}

/// Winners 0x12 / 0x23 / 0x31, each ranking covering its slot's catalog.
SlotRecords goodSlotRecords()
{
    return {{"prep", ranking(PREP_PACK, {0x12, 0x11})},
            {"dkdv", ranking(DKDV_PACK, {0x23, 0x21, 0x22})},
            {"dq", ranking(DQ_PACK, {0x31, 0x32})}};
}

ProgramRecord programOf(uint8_t prep, uint8_t dkdv, uint8_t dq)
{
    return ProgramRecord{{{"prep", testId(prep)}, {"dkdv", testId(dkdv)}, {"dq", testId(dq)}}, 3.5};
}

ProgramRecord goodProgram()
{
    return programOf(0x12, 0x23, 0x31);
}

/// Lets the manager see every slot's catalog for @p graph, as a plan build would.
void seeCatalogs(const Manager& manager,
                 const ContentCarryingTestGraph& graph,
                 const DeviceProperties& properties)
{
    const MatchContext context{graph, 0, properties};
    for(size_t slot = 0; slot < manager.slotCount(); ++slot)
    {
        (void)manager.sortedCatalog(slot, context);
    }
}

std::vector<std::string> shardLines(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::vector<std::string> lines;
    for(std::string line; std::getline(file, line);)
    {
        lines.push_back(line);
    }
    return lines;
}

std::filesystem::path shardPath(const DeviceProperties& properties)
{
    return winnerCacheShardPath(namedEngine(), properties.gcnArchName);
}

std::vector<std::string> kernelOrder(const Catalog& catalog)
{
    std::vector<std::string> names;
    names.reserve(catalog.entries.size());
    for(const auto& entry : catalog.entries)
    {
        names.push_back(entry->name);
    }
    return names;
}

void recordGood(Manager& manager, const WinnerKey& key)
{
    manager.recordSlotted(key,
                          goodSlotRecords(),
                          goodProgram(),
                          WinnerWriteCause::FRESH_MISS,
                          WinnerWriteCause::FRESH_MISS);
}

class TestSlottedWinnerRecordsInMemory : public ::testing::Test
{
public:
    const ScopedGraphMatch match;
    const ContentCarryingTestGraph graph = makeGraph();
    const DeviceProperties properties = testDeviceProperties();
    const WinnerKey base = baseKeyFor(graph, properties);
};

class TestSlottedWinnerRecords : public TestSlottedWinnerRecordsInMemory
{
protected:
    const ScopedCacheDir _cacheDir{::testing::UnitTest::GetInstance()->current_test_info()->name()};
};

/// Writes the slot records and a good program for the base key through one manager, and returns
/// a fresh manager over the same shard after @p append added lines to it.
std::unique_ptr<Manager> managerOverShardWith(const TestSlottedWinnerRecordsInMemory& scenario,
                                              const std::vector<std::string>& append)
{
    recordGood(*makeManager(namedEngine()), scenario.base);
    {
        EXPECT_TRUE(std::filesystem::exists(shardPath(scenario.properties)));
        std::ofstream out(shardPath(scenario.properties), std::ios::app);
        EXPECT_TRUE(out.is_open());
        for(const auto& line : append)
        {
            out << line << "\n";
        }
    }
    auto reader = makeManager(namedEngine());
    seeCatalogs(*reader, scenario.graph, scenario.properties);
    return reader;
}

// ---------------------------------------------------------------------------
// The codec: v2 stays byte-identical, v3 carries the slot and program lines
// ---------------------------------------------------------------------------

TEST(TestSlottedWinnerRecordCodec, ASlotRecordIsAV3LineWithASlotField)
{
    const auto graph = makeGraph();
    const auto properties = testDeviceProperties();
    WinnerKey key = baseKeyFor(graph, properties);
    key.slot = "dkdv";
    const auto record = ranking(DKDV_PACK, {0x23, 0x21});

    const auto line = encodeWinnerRecordLine(key, record);

    nlohmann::json expected;
    expected["v"] = 3;
    expected["slot"] = "dkdv";
    expected["graph"] = key.graph.toJson();
    expected["device"] = nlohmann::json::parse(
        encodeWinnerRecordLine(baseKeyFor(graph, properties), record))["device"];
    expected["entries"] = nlohmann::json::parse(
        encodeWinnerRecordLine(baseKeyFor(graph, properties), record))["entries"];
    EXPECT_EQ(line, expected.dump());

    const auto decoded = decodeWinnerRecordLine(line);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(decoded->first, key);
    EXPECT_EQ(decoded->first.slot, "dkdv");
    ASSERT_EQ(decoded->second.size(), 2U);
    EXPECT_EQ(decoded->second.front().kernelId, testId(0x23));
}

TEST(TestSlottedWinnerRecordCodec, AProgramLineIsAV3LineCarryingTheSlotToKernelMap)
{
    const auto graph = makeGraph();
    const auto properties = testDeviceProperties();
    const auto key = baseKeyFor(graph, properties);

    const auto line = encodeProgramLine(key, goodProgram());

    const auto json = nlohmann::json::parse(line);
    EXPECT_EQ(json.at("v"), 3);
    EXPECT_FALSE(json.contains("entries"));
    ASSERT_EQ(json.at("program").size(), 3U);
    EXPECT_EQ(json.at("program")[1].at("slot"), "dkdv");
    EXPECT_EQ(json.at("program")[1].at("kernel_id"), toString(testId(0x23)));
    EXPECT_DOUBLE_EQ(json.at("time_ms").get<double>(), 3.5);

    const auto decoded = decodeWinnerLine(line);
    ASSERT_TRUE(decoded.has_value());
    ASSERT_TRUE(decoded->program.has_value());
    EXPECT_EQ(decoded->key, key);
    ASSERT_EQ(decoded->program->picks.size(), 3U);
    EXPECT_EQ(decoded->program->picks[0].slot, "prep");
    EXPECT_EQ(decoded->program->picks[2].kernelId, testId(0x31));
    EXPECT_DOUBLE_EQ(decoded->program->timeMs, 3.5);
    EXPECT_EQ(encodeProgramLine(decoded->key, *decoded->program), line);

    EXPECT_FALSE(decodeWinnerRecordLine(line).has_value()) << "a program line is not a ranking";
}

TEST(TestSlottedWinnerRecordCodec, AV3LineThatIsNeitherASlotRecordNorAProgramIsSkipped)
{
    const auto graph = makeGraph();
    const auto properties = testDeviceProperties();
    auto slotless = nlohmann::json::parse(
        encodeWinnerRecordLine(baseKeyFor(graph, properties), ranking(DKDV_PACK, {0x21})));
    slotless["v"] = 3;
    EXPECT_FALSE(decodeWinnerLine(slotless.dump()).has_value())
        << "v3 must never read back as an unslotted record";

    WinnerKey dkdvKey = baseKeyFor(graph, properties);
    dkdvKey.slot = "dkdv";
    auto emptySlot
        = nlohmann::json::parse(encodeWinnerRecordLine(dkdvKey, ranking(DKDV_PACK, {0x21})));
    ASSERT_TRUE(decodeWinnerLine(emptySlot.dump()).has_value());
    emptySlot["slot"] = "";
    EXPECT_FALSE(decodeWinnerLine(emptySlot.dump()).has_value())
        << "a v3 line with an empty slot must never read back as a whole-graph record";

    auto emptyProgram
        = nlohmann::json::parse(encodeProgramLine(baseKeyFor(graph, properties), goodProgram()));
    emptyProgram["program"] = nlohmann::json::array();
    EXPECT_FALSE(decodeWinnerLine(emptyProgram.dump()).has_value());

    auto futureVersion
        = nlohmann::json::parse(encodeProgramLine(baseKeyFor(graph, properties), goodProgram()));
    futureVersion["v"] = 4;
    EXPECT_FALSE(decodeWinnerLine(futureVersion.dump()).has_value());
}

/// Each edit of a valid program line leaves one that cannot be read; none may decode, and a
/// torn tail (a crash mid-append) must not either.
TEST(TestSlottedWinnerRecordCodec, AMalformedProgramLineIsSkipped)
{
    const auto graph = makeGraph();
    const auto properties = testDeviceProperties();
    const auto good
        = nlohmann::json::parse(encodeProgramLine(baseKeyFor(graph, properties), goodProgram()));
    ASSERT_TRUE(decodeWinnerLine(good.dump()).has_value());

    const std::vector<std::pair<std::string, std::function<void(nlohmann::json&)>>> edits{
        {"missing time_ms", [](nlohmann::json& j) { j.erase("time_ms"); }},
        {"null time_ms", [](nlohmann::json& j) { j["time_ms"] = nullptr; }},
        {"kernel id not a uuid", [](nlohmann::json& j) { j["program"][1]["kernel_id"] = "zz"; }},
        {"pick without kernel id", [](nlohmann::json& j) { j["program"][1].erase("kernel_id"); }},
        {"pick without slot", [](nlohmann::json& j) { j["program"][1].erase("slot"); }},
        {"empty pick slot", [](nlohmann::json& j) { j["program"][1]["slot"] = ""; }},
        {"program not an array", [](nlohmann::json& j) { j["program"] = "x"; }},
        {"no graph", [](nlohmann::json& j) { j.erase("graph"); }},
        {"v2 carrying a program", [](nlohmann::json& j) { j["v"] = 2; }},
    };
    for(const auto& [name, edit] : edits)
    {
        auto line = good;
        edit(line);
        EXPECT_FALSE(decodeWinnerLine(line.dump()).has_value()) << name;
    }

    const auto text = good.dump();
    for(const auto length : {text.size() / 2, text.size() - 1})
    {
        EXPECT_FALSE(decodeWinnerLine(text.substr(0, length)).has_value()) << length;
    }
}

// ---------------------------------------------------------------------------
// recordSlotted: one locked append, program line only behind valid slot records
// ---------------------------------------------------------------------------

TEST_F(TestSlottedWinnerRecords, SlotRecordsThenTheProgramLineLandInOneAppend)
{
    recordGood(*makeManager(namedEngine()), base);

    const auto lines = shardLines(shardPath(properties));
    ASSERT_EQ(lines.size(), 5U) << "version line, three slot records, one program line";
    for(size_t i = 1; i <= 3; ++i)
    {
        const auto decoded = decodeWinnerLine(lines[i]);
        ASSERT_TRUE(decoded.has_value());
        EXPECT_FALSE(decoded->program.has_value());
        EXPECT_FALSE(decoded->key.slot.empty());
    }
    const auto program = decodeWinnerLine(lines[4]);
    ASSERT_TRUE(program.has_value());
    EXPECT_TRUE(program->program.has_value()) << "the program line comes last";
}

TEST_F(TestSlottedWinnerRecords, AValidProgramIsServedByAFreshManagerOnceItHasSeenTheCatalogs)
{
    recordGood(*makeManager(namedEngine()), base);

    const auto reader = makeManager(namedEngine());
    EXPECT_FALSE(reader->programFor(base).has_value())
        << "a program cannot be checked against catalogs the manager has not seen";

    seeCatalogs(*reader, graph, properties);
    const auto program = reader->programFor(base);
    ASSERT_TRUE(program.has_value());
    ASSERT_EQ(program->picks.size(), 3U);
    EXPECT_EQ(program->picks[1].kernelId, testId(0x23));
}

TEST_F(TestSlottedWinnerRecords, ARewrittenSlotWinnerInvalidatesTheProgramAtReadTime)
{
    const auto manager = makeManager(namedEngine());
    recordGood(*manager, base);
    seeCatalogs(*manager, graph, properties);
    ASSERT_TRUE(manager->programFor(base).has_value());

    // A re-benchmark crowns another dkdv kernel; the program line is untouched.
    manager->recordSlotted(base,
                           {{"dkdv", ranking(DKDV_PACK, {0x22, 0x23, 0x21})}},
                           std::nullopt,
                           WinnerWriteCause::COVERAGE_REBENCHMARK,
                           WinnerWriteCause::FRESH_MISS);

    EXPECT_FALSE(manager->programFor(base).has_value());
    seeCatalogs(*manager, graph, properties);
    const auto fresh = makeManager(namedEngine());
    seeCatalogs(*fresh, graph, properties);
    EXPECT_FALSE(fresh->programFor(base).has_value())
        << "the rewrite is what the shard holds, not just what this manager remembers";
}

TEST_F(TestSlottedWinnerRecords, ACatalogThatGainedAKernelInvalidatesTheProgram)
{
    recordGood(*makeManager(namedEngine()), base);

    const auto grown = makeManager(namedEngine(), true);
    seeCatalogs(*grown, graph, properties);

    EXPECT_FALSE(grown->programFor(base).has_value())
        << "kernel 0x24 was never measured against the recorded winners";
}

TEST_F(TestSlottedWinnerRecordsInMemory, AProgramNamingAKernelOutsideTheCatalogIsInvalid)
{
    const auto manager = makeManager({});
    // The winners agree with the program, but the winner is not a kernel of the dq catalog.
    manager->recordSlotted(base,
                           {{"prep", ranking(PREP_PACK, {0x12, 0x11})},
                            {"dkdv", ranking(DKDV_PACK, {0x23, 0x21, 0x22})},
                            {"dq", ranking(DQ_PACK, {0x39, 0x31, 0x32})}},
                           programOf(0x12, 0x23, 0x39),
                           WinnerWriteCause::FRESH_MISS,
                           WinnerWriteCause::FRESH_MISS);
    seeCatalogs(*manager, graph, properties);

    EXPECT_FALSE(manager->programFor(base).has_value());
}

TEST_F(TestSlottedWinnerRecordsInMemory, ALaterProgramLineThatFailsAtReadTimeIsInvalid)
{
    // The kernels are the recorded winners; only the picks differ.
    const std::vector<std::pair<std::string, ProgramRecord>> rows{
        {"reordered",
         ProgramRecord{{{"dkdv", testId(0x23)}, {"prep", testId(0x12)}, {"dq", testId(0x31)}},
                       1.0}},
        {"too_few", ProgramRecord{{{"prep", testId(0x12)}, {"dkdv", testId(0x23)}}, 1.0}},
        {"too_many",
         ProgramRecord{{{"prep", testId(0x12)},
                        {"dkdv", testId(0x23)},
                        {"dq", testId(0x31)},
                        {"dq", testId(0x31)}},
                       1.0}}};

    for(const auto& [name, program] : rows)
    {
        const ScopedCacheDir cacheDir(name);
        const auto reader = managerOverShardWith(*this, {encodeProgramLine(base, program)});
        EXPECT_FALSE(reader->programFor(base).has_value()) << name;
    }
}

TEST_F(TestSlottedWinnerRecords, UnreadableLinesAfterAProgramLeaveItStanding)
{
    auto emptySlot = nlohmann::json::parse(encodeProgramLine(base, goodProgram()));
    emptySlot["program"][1]["slot"] = "";
    auto noTime = nlohmann::json::parse(encodeProgramLine(base, goodProgram()));
    noTime.erase("time_ms");

    const auto reader = managerOverShardWith(
        *this,
        {emptySlot.dump(), noTime.dump(), R"({"v":3,"program":[{"slot":"prep","kernel_id":"1)"});

    const auto program = reader->programFor(base);
    ASSERT_TRUE(program.has_value());
    EXPECT_DOUBLE_EQ(program->timeMs, 3.5);
}

/// A shard is shared by engines of both kinds under one identity: a slotted engine reads the
/// slot and program lines, never a whole-graph (v2) line.
TEST_F(TestSlottedWinnerRecords, ASlottedEngineIgnoresAWholeGraphRecordInItsShard)
{
    const auto reader
        = managerOverShardWith(*this, {encodeWinnerRecordLine(base, ranking(DKDV_PACK, {0x21}))});

    EXPECT_FALSE(reader->winnerFor(base).has_value());
    EXPECT_TRUE(reader->programFor(base).has_value());
}

TEST_F(TestSlottedWinnerRecords, AProgramLineNeverLandsBehindSlotRecordsWithOtherWinners)
{
    const auto manager = makeManager(namedEngine());

    // dkdv's record crowns 0x23, but the program names 0x21.
    manager->recordSlotted(base,
                           goodSlotRecords(),
                           programOf(0x12, 0x21, 0x31),
                           WinnerWriteCause::FRESH_MISS,
                           WinnerWriteCause::FRESH_MISS);

    const auto lines = shardLines(shardPath(properties));
    ASSERT_EQ(lines.size(), 4U) << "the three slot records land, the program line does not";
    for(size_t i = 1; i < lines.size(); ++i)
    {
        EXPECT_FALSE(decodeWinnerLine(lines[i])->program.has_value());
    }
    seeCatalogs(*manager, graph, properties);
    EXPECT_FALSE(manager->programFor(base).has_value());
}

TEST_F(TestSlottedWinnerRecords, AProgramOnlyCallWithNoSlotRecordsOnDiskWritesNothing)
{
    makeManager(namedEngine())
        ->recordSlotted(
            base, {}, goodProgram(), WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS);

    const auto lines = shardLines(shardPath(properties));
    EXPECT_EQ(lines.size(), 1U) << "only the version line; no program line without slot records";
}

TEST_F(TestSlottedWinnerRecords, AProgramOnlyCallIsLegalOnceTheSlotRecordsAreOnDisk)
{
    makeManager(namedEngine())
        ->recordSlotted(base,
                        goodSlotRecords(),
                        std::nullopt,
                        WinnerWriteCause::FRESH_MISS,
                        WinnerWriteCause::FRESH_MISS);
    ASSERT_EQ(shardLines(shardPath(properties)).size(), 4U);

    const auto manager = makeManager(namedEngine());
    manager->recordSlotted(
        base, {}, goodProgram(), WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS);

    EXPECT_EQ(shardLines(shardPath(properties)).size(), 5U);
    seeCatalogs(*manager, graph, properties);
    EXPECT_TRUE(manager->programFor(base).has_value());
}

TEST(TestSlottedWinnerRecordsOwnSetup, AnUnkeyableGraphsRecordsAreNeitherCachedNorWritten)
{
    const ScopedGraphMatch match;
    const ScopedCacheDir cacheDir("unkeyable");
    const BytelessTestGraph graph;
    const auto properties = testDeviceProperties();
    const WinnerKey base{GraphContentKey{graph}, DeviceKey{properties}};
    ASSERT_FALSE(base.graph.isUsable());
    const auto manager = makeManager(namedEngine());

    recordGood(*manager, base);

    EXPECT_EQ(manager->winnerCacheSize(), 0U);
    EXPECT_TRUE(shardLines(shardPath(properties)).empty()) << "no shard is opened for it";
}

TEST_F(TestSlottedWinnerRecords, ARacingWritersProgramLineIsKeptWhenTheCallersProgramIsDropped)
{
    // The loser has already read the (empty) shard when the winner appends its records.
    const auto loser = makeManager(namedEngine());
    seeCatalogs(*loser, graph, properties);
    recordGood(*makeManager(namedEngine()), base);
    ASSERT_EQ(shardLines(shardPath(properties)).size(), 5U);

    // Different winners: the loser adopts the winner's slot records, so its own program names
    // kernels they do not crown.
    const SlotRecords other = {{"prep", ranking(PREP_PACK, {0x11, 0x12})},
                               {"dkdv", ranking(DKDV_PACK, {0x21, 0x22, 0x23})},
                               {"dq", ranking(DQ_PACK, {0x32, 0x31})}};
    loser->recordSlotted(base,
                         other,
                         programOf(0x11, 0x21, 0x32),
                         WinnerWriteCause::FRESH_MISS,
                         WinnerWriteCause::COVERAGE_REBENCHMARK);

    EXPECT_EQ(shardLines(shardPath(properties)).size(), 5U) << "nothing new lands";
    seeCatalogs(*loser, graph, properties);
    const auto program = loser->programFor(base);
    ASSERT_TRUE(program.has_value()) << "the shard holds a valid program; so must the loser";
    EXPECT_EQ(program->picks[0].kernelId, testId(0x12));
}

TEST_F(TestSlottedWinnerRecords, EachPartFollowsItsOwnWriteCause)
{
    const auto manager = makeManager(namedEngine());
    recordGood(*manager, base);
    ASSERT_EQ(shardLines(shardPath(properties)).size(), 5U);

    // A fresh miss adopts what is on disk, for both parts.
    recordGood(*manager, base);
    EXPECT_EQ(shardLines(shardPath(properties)).size(), 5U);

    // Slot records superseded, program line adopted: three new lines.
    manager->recordSlotted(base,
                           goodSlotRecords(),
                           goodProgram(),
                           WinnerWriteCause::COVERAGE_REBENCHMARK,
                           WinnerWriteCause::FRESH_MISS);
    EXPECT_EQ(shardLines(shardPath(properties)).size(), 8U);

    // Only the program line superseded.
    manager->recordSlotted(base,
                           goodSlotRecords(),
                           goodProgram(),
                           WinnerWriteCause::FRESH_MISS,
                           WinnerWriteCause::COVERAGE_REBENCHMARK);
    EXPECT_EQ(shardLines(shardPath(properties)).size(), 9U);
}

// ---------------------------------------------------------------------------
// Slot-keyed adoption and coverage
// ---------------------------------------------------------------------------

TEST_F(TestSlottedWinnerRecordsInMemory, ACoveringSlotRecordOrdersThatSlotsCatalog)
{
    const MatchContext context{graph, 0, properties};
    const auto manager = makeManager({});
    ASSERT_EQ(kernelOrder(manager->sortedCatalog(1, context)),
              (std::vector<std::string>{"dkdv_33", "dkdv_34", "dkdv_35"}));

    manager->recordSlotted(base,
                           {{"dkdv", ranking(DKDV_PACK, {0x23, 0x21, 0x22})}},
                           std::nullopt,
                           WinnerWriteCause::FRESH_MISS,
                           WinnerWriteCause::FRESH_MISS);

    EXPECT_EQ(kernelOrder(manager->sortedCatalog(1, context)),
              (std::vector<std::string>{"dkdv_35", "dkdv_33", "dkdv_34"}))
        << "the record postdates the cached ranker order and still wins";
    EXPECT_EQ(kernelOrder(manager->sortedCatalog(0, context)),
              (std::vector<std::string>{"prep_17", "prep_18"}))
        << "another slot's catalog is untouched";
    EXPECT_NE(manager->sortedCatalog(1, context).measuredRecord, nullptr);
}

TEST_F(TestSlottedWinnerRecordsInMemory, APartialSlotRecordLeavesTheRankerOrderIntact)
{
    const MatchContext context{graph, 0, properties};
    const auto manager = makeManager({});
    manager->recordSlotted(base,
                           {{"dkdv", ranking(DKDV_PACK, {0x23, 0x21})}},
                           std::nullopt,
                           WinnerWriteCause::FRESH_MISS,
                           WinnerWriteCause::FRESH_MISS);

    const auto catalog = manager->sortedCatalog(1, context);

    EXPECT_EQ(catalog.measuredRecord, nullptr);
    EXPECT_EQ(kernelOrder(catalog), (std::vector<std::string>{"dkdv_33", "dkdv_34", "dkdv_35"}));
}

// ---------------------------------------------------------------------------
// Guards, in-memory operation and LRU accounting
// ---------------------------------------------------------------------------

TEST_F(TestSlottedWinnerRecordsInMemory, SlottedAndUnslottedEntryPointsRefuseTheOtherKind)
{
    WinnerKey dkdvKey = base;
    dkdvKey.slot = "dkdv";
    const auto manager = makeManager({});

    EXPECT_THROW(
        manager->recordWinner(base, ranking(DKDV_PACK, {0x21}), WinnerWriteCause::FRESH_MISS),
        std::logic_error);
    WinnerKey unknownSlot = base;
    unknownSlot.slot = "nope";
    EXPECT_THROW(manager->recordWinner(
                     unknownSlot, ranking(DKDV_PACK, {0x21}), WinnerWriteCause::FRESH_MISS),
                 std::invalid_argument);
    EXPECT_THROW(
        manager->recordSlotted(
            dkdvKey, {}, std::nullopt, WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS),
        std::invalid_argument);
    EXPECT_THROW(manager->recordSlotted(base,
                                        {{"nope", ranking(DKDV_PACK, {0x21})}},
                                        std::nullopt,
                                        WinnerWriteCause::FRESH_MISS,
                                        WinnerWriteCause::FRESH_MISS),
                 std::invalid_argument);
    EXPECT_THROW(manager->recordSlotted(base,
                                        {},
                                        ProgramRecord{{{"dkdv", testId(0x21)}}, 1.0},
                                        WinnerWriteCause::FRESH_MISS,
                                        WinnerWriteCause::FRESH_MISS),
                 std::invalid_argument)
        << "a program must name every slot, in launch order";

    // A single slot record through the per-record entry point is legal and lands as v3.
    EXPECT_NO_THROW(manager->recordWinner(
        dkdvKey, ranking(DKDV_PACK, {0x23, 0x21, 0x22}), WinnerWriteCause::FRESH_MISS));
    EXPECT_TRUE(manager->winnerFor(dkdvKey).has_value());
}

TEST(TestSlottedWinnerRecordsOwnSetup, AnUnslottedEngineWritesNoSlotOrProgramLine)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto graph = makeGraph();
    const auto properties = testDeviceProperties();
    const auto base = baseKeyFor(graph, properties);
    const auto unslotted = makeIdentifiedStateManager(EngineIdentity{});
    WinnerKey slotKey = base;
    slotKey.slot = "dkdv";

    EXPECT_THROW(
        unslotted->recordSlotted(
            base, {}, std::nullopt, WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS),
        std::logic_error);
    EXPECT_THROW(
        unslotted->recordWinner(slotKey, ranking(DKDV_PACK, {0x21}), WinnerWriteCause::FRESH_MISS),
        std::logic_error);
    EXPECT_FALSE(unslotted->programFor(base).has_value());
}

TEST_F(TestSlottedWinnerRecordsInMemory, WithoutAShardTheInMemoryRecordsStandInAndAreValidated)
{
    const auto manager = makeManager({});

    manager->recordSlotted(
        base, {}, goodProgram(), WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS);
    seeCatalogs(*manager, graph, properties);
    EXPECT_FALSE(manager->programFor(base).has_value()) << "no slot records, no program";

    recordGood(*manager, base);
    EXPECT_TRUE(manager->programFor(base).has_value());
}

TEST_F(TestSlottedWinnerRecordsInMemory, ProgramAndSlotRecordsAreLruAccountedSeparately)
{
    // Room for the three slot records, and for the program record in its own cache.
    const auto manager = makeManager({}, false, 3);

    recordGood(*manager, base);

    seeCatalogs(*manager, graph, properties);
    EXPECT_TRUE(manager->programFor(base).has_value())
        << "a program record must not evict a slot record it depends on";

    // A fourth slot record pushes the oldest slot record out, as for any ranking, and the
    // program that names it goes with it.
    const auto otherGraph = ContentCarryingTestGraph{[] {
        ContentCarryingTestGraph::Spec spec;
        spec.tensors[0].dims = {16, 8};
        return spec;
    }()};
    manager->recordSlotted(baseKeyFor(otherGraph, properties),
                           {{"prep", ranking(PREP_PACK, {0x12, 0x11})}},
                           std::nullopt,
                           WinnerWriteCause::FRESH_MISS,
                           WinnerWriteCause::FRESH_MISS);
    auto prepKey = base;
    prepKey.slot = "prep";
    auto dkdvKey = base;
    dkdvKey.slot = "dkdv";
    EXPECT_FALSE(manager->winnerFor(prepKey).has_value());
    EXPECT_TRUE(manager->winnerFor(dkdvKey).has_value());
    EXPECT_FALSE(manager->programFor(base).has_value());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
