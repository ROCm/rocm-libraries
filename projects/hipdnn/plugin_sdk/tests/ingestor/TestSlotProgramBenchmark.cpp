// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkDetail.hpp>
#include <hipdnn_plugin_sdk/ingestor/SlotProgramBenchmark.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipStreamWaitFaults.hpp"
#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestSlotProgramBenchmark.cpp
 * @brief The benchmark of a slotted engine's program (RFC 0021 §7) over the synthetic three-slot
 *        engine with fake timings: run accounting, trigger, reference fallback, and records.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using namespace hipdnn_plugin_sdk::ingestor::testing::slotted;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;

using Settings = std::vector<std::pair<std::string, int64_t>>;
/// A program as the seeds of its kernels, in launch order.
using Program = std::vector<uint8_t>;

constexpr std::array<uint8_t, 7> ALL_SEEDS{0x11, 0x12, 0x21, 0x22, 0x23, 0x31, 0x32};
constexpr size_t WORKSPACE_BYTES = 1024 + 16384;
constexpr int SAMPLES_PER_RUN = BENCHMARK_ITERATIONS;

// Each kernel's own cost; a program's time is the sum of its kernels'. The reference program
// (prep_a, dkdv_a, dq_a) takes 19, the winners program (prep_b, dkdv_c, dq_a) 14.
const std::map<uint8_t, double> COSTS{
    {0x11, 5.0}, {0x12, 3.0}, {0x21, 10.0}, {0x22, 12.0}, {0x23, 7.0}, {0x31, 4.0}, {0x32, 6.0}};

const Program REFERENCE{0x11, 0x21, 0x31};
const Program WINNERS{0x12, 0x23, 0x31};

uint8_t seedOf(const DescriptorId& id)
{
    for(const auto seed : ALL_SEEDS)
    {
        if(testId(seed) == id)
        {
            return seed;
        }
    }
    ADD_FAILURE() << "launched an unknown kernel";
    return 0;
}

/// A timer that executes the program it is handed and answers with the sum of its kernels'
/// costs, recording every program it timed.
class FakeClock
{
public:
    void attach(const SlotLog& log)
    {
        _log = &log;
    }

    /// Answers nothing for @p program, as a timer that cannot time it.
    void failToTime(Program program)
    {
        _failing = std::move(program);
    }

    /// Makes @p seed's kernel cost @p cost instead of its COSTS entry.
    void setCost(uint8_t seed, double cost)
    {
        _costs[seed] = cost;
    }

    void clearFailure()
    {
        _failing.reset();
    }

    BenchmarkPlan<StubHandle>::Timer timer()
    {
        return [this](const hipdnn_plugin_sdk::IPlan<StubHandle>& plan,
                      const StubHandle& handle,
                      const hipdnnPluginDeviceBuffer_t* buffers,
                      uint32_t count,
                      void* workspace) -> std::optional<double> {
            const auto before = _log->launches.size();
            plan.execute(handle, buffers, count, workspace);
            Program program;
            for(auto index = before; index < _log->launches.size(); ++index)
            {
                program.push_back(seedOf(_log->launches[index].kernel));
            }
            _samples.push_back(program);
            if(_failing.has_value() && program == *_failing)
            {
                return std::nullopt;
            }
            double total = 0.0;
            for(const auto seed : program)
            {
                total += _costs.at(seed);
            }
            return total;
        };
    }

    /// The programs timed, in order, one entry per run: the samples of a run are identical.
    std::vector<Program> runs() const
    {
        std::vector<Program> runs;
        for(const auto& sample : _samples)
        {
            if(runs.empty() || runs.back() != sample)
            {
                runs.push_back(sample);
            }
        }
        return runs;
    }

    size_t sampleCount() const
    {
        return _samples.size();
    }

    void forget()
    {
        _samples.clear();
    }

private:
    const SlotLog* _log = nullptr;
    std::vector<Program> _samples;
    std::optional<Program> _failing;
    std::map<uint8_t, double> _costs = COSTS;
};

using PredictionResult = hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT;
using hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;

/// The engine, its fake clock, and a graph with a usable content key. With @p defaultTimer the
/// engine times with the real HIP timer, and the fake clock stays unused.
class Bench
{
public:
    explicit Bench(bool defaultTimer = false,
                   const std::vector<std::vector<SlotKernel>>& kernels = defaultSlotKernels(),
                   EngineIdentity identity = {})
        : _fixture(kernels,
                   defaultTimer ? BenchmarkPlan<StubHandle>::Timer{} : _clock.timer(),
                   {},
                   std::move(identity))
    {
        _clock.attach(_fixture.log());
    }

    SlottedEngineFixture& fixture()
    {
        return _fixture;
    }

    FakeClock& clock()
    {
        return _clock;
    }

    const TestGraph& graph() const
    {
        return _graph;
    }

    /// Builds the plan with benchmarking on (and @p extra settings), executes it once and
    /// returns the program a second execution then runs.
    Program benchmarkAndServe(const Settings& extra = {})
    {
        Settings settings{{hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1}};
        settings.insert(settings.end(), extra.begin(), extra.end());
        flatbuffers::FlatBufferBuilder builder;
        const auto config = makeIntKnobsEngineConfig(builder, settings);
        const auto context = _fixture.planFor(_graph, config);

        std::vector<char> workspace(WORKSPACE_BYTES);
        const StubHandle handle;
        context.plan().execute(handle, nullptr, 0, workspace.data());

        const auto before = _fixture.log().launches.size();
        context.plan().execute(handle, nullptr, 0, workspace.data());
        Program served;
        for(auto index = before; index < _fixture.log().launches.size(); ++index)
        {
            served.push_back(seedOf(_fixture.log().launches[index].kernel));
        }
        return served;
    }

    /// The workspace size the benchmarking plan reports for @p extra settings.
    size_t benchmarkingWorkspaceSize(const Settings& extra = {})
    {
        Settings settings{{hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1}};
        settings.insert(settings.end(), extra.begin(), extra.end());
        flatbuffers::FlatBufferBuilder builder;
        const auto config = makeIntKnobsEngineConfig(builder, settings);
        const auto context = _fixture.planFor(_graph, config);
        const StubHandle handle;
        return context.plan().getWorkspaceSize(handle);
    }

    /// The engine's workspace estimate with benchmarking on (RFC 0021 §6, #14105).
    size_t benchmarkingWorkspaceEstimate()
    {
        flatbuffers::FlatBufferBuilder builder;
        const auto config = makeIntKnobsEngineConfig(
            builder, Settings{{hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1}});
        const StubHandle handle;
        return _fixture.engine().getMaxWorkspaceSize(handle, _graph, config);
    }

    WinnerKey baseKey() const
    {
        return WinnerKey{GraphContentKey{_graph}, DeviceKey{testDeviceProperties()}};
    }

    std::optional<WinnerRecord> slotRecord(const std::string& slot) const
    {
        auto key = baseKey();
        key.slot = slot;
        return _fixture.manager().winnerFor(key);
    }

    std::optional<ProgramRecord> programRecord() const
    {
        return _fixture.manager().programFor(baseKey());
    }

    /// Writes a record covering @p slot's catalog, as an earlier benchmark would have.
    void seedSlotRecord(size_t slot, const std::vector<uint8_t>& ranking)
    {
        WinnerRecord record;
        for(const auto seed : ranking)
        {
            record.push_back(RankedEntry{testId(seed),
                                         testId(static_cast<uint8_t>(0x10 * (slot + 1))),
                                         testId(static_cast<uint8_t>(0xD0 + slot)),
                                         1.0});
        }
        // Reading the catalog notes it, which a program record's validity is judged against.
        const auto properties = testDeviceProperties();
        const MatchContext context{
            _graph, 0, properties, hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC};
        static_cast<void>(_fixture.manager().sortedCatalog(slot, context));
        _fixture.manager().recordSlotted(baseKey(),
                                         {{slotNames()[slot], std::move(record)}},
                                         std::nullopt,
                                         WinnerWriteCause::FRESH_MISS,
                                         WinnerWriteCause::FRESH_MISS);
    }

    /// The prediction of the engine's configuration for the graph under @p settings.
    PredictionResult predict(const Settings& settings = {})
    {
        // The time metric: the test graph carries no FLOP count for the default one.
        hipdnn_flatbuffers_sdk::data_objects::EngineConfigT config;
        config.engine_id = ENGINE_ID.front();
        config.ranking_metric = "time";
        for(const auto& [name, value] : settings)
        {
            auto setting = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>();
            setting->knob_id = name;
            hipdnn_flatbuffers_sdk::data_objects::IntValueT integer;
            integer.value = value;
            setting->value.Set(integer);
            config.knobs.push_back(std::move(setting));
        }
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(builder, &config));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper wrapper(
            builder.GetBufferPointer(), builder.GetSize());
        StubHandle handle;
        return _fixture.engine().getPrediction(
            handle, _graph, wrapper, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    }

    /// Writes a covering record for every slot and the winners program, without any plan build
    /// having shown the manager the catalogs, as a record left by an earlier process would be.
    void recordMeasuredProgramWithoutBuildingCatalogs()
    {
        const auto rankingOf = [](size_t slot, const std::vector<uint8_t>& seeds) {
            WinnerRecord record;
            for(const auto seed : seeds)
            {
                record.push_back(RankedEntry{testId(seed),
                                             testId(static_cast<uint8_t>(0x10 * (slot + 1))),
                                             testId(static_cast<uint8_t>(0xD0 + slot)),
                                             1.0});
            }
            return record;
        };
        ProgramRecord program;
        for(size_t slot = 0; slot < 3; ++slot)
        {
            program.picks.push_back({slotNames()[slot], testId(WINNERS[slot])});
        }
        program.timeMs = 14.0;
        _fixture.manager().recordSlotted(baseKey(),
                                         {{"prep", rankingOf(0, {0x12, 0x11})},
                                          {"dkdv", rankingOf(1, {0x23, 0x21, 0x22})},
                                          {"dq", rankingOf(2, {0x31, 0x32})}},
                                         program,
                                         WinnerWriteCause::FRESH_MISS,
                                         WinnerWriteCause::FRESH_MISS);
    }

private:
    FakeClock _clock;
    SlottedEngineFixture _fixture;
    TestGraph _graph{makeGraphId(7)};
};

std::vector<uint8_t> seedsOf(const WinnerRecord& record)
{
    std::vector<uint8_t> seeds;
    for(const auto& entry : record)
    {
        seeds.push_back(seedOf(entry.kernelId));
    }
    return seeds;
}

/// The int knob values @p config pins, by knob id.
std::map<std::string, int64_t>
    pinnedKnobs(const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& config)
{
    std::map<std::string, int64_t> pinned;
    for(const auto& setting : config.knobs)
    {
        pinned.emplace(setting->knob_id, setting->value.AsIntValue()->value);
    }
    return pinned;
}

Program picksOf(const ProgramRecord& program)
{
    Program picks;
    for(const auto& pick : program.picks)
    {
        picks.push_back(seedOf(pick.kernelId));
    }
    return picks;
}

TEST(TestSlotProgramBenchmark, TimesTheReferenceThenOneSlotVariedRunsThenTheWinnersProgram)
{
    Bench bench;

    const auto served = bench.benchmarkAndServe();

    // 1 + sum(K - 1) = 1 + 1 + 2 + 1 runs, then the winners program.
    const std::vector<Program> expected{REFERENCE,
                                        {0x12, 0x21, 0x31},
                                        {0x11, 0x22, 0x31},
                                        {0x11, 0x23, 0x31},
                                        {0x11, 0x21, 0x32},
                                        WINNERS};
    EXPECT_EQ(bench.clock().runs(), expected);
    EXPECT_EQ(bench.clock().sampleCount(), expected.size() * SAMPLES_PER_RUN);
    EXPECT_EQ(served, WINNERS);
}

TEST(TestSlotProgramBenchmark, ABenchmarkingPlanIsNotSaveable)
{
    Bench bench;
    flatbuffers::FlatBufferBuilder builder;
    const auto config = makeIntKnobsEngineConfig(
        builder, Settings{{hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1}});
    const auto context = bench.fixture().planFor(bench.graph(), config);

    // Before and after the benchmark has picked a program: RFC 0021 §11, a slot program is not
    // serialized (RFC 0009), though every slot's pick is a saveable GenericPlan.
    EXPECT_EQ(context.plan().saveablePlan(), nullptr);
    std::vector<char> workspace(WORKSPACE_BYTES);
    const StubHandle handle;
    context.plan().execute(handle, nullptr, 0, workspace.data());
    EXPECT_EQ(context.plan().saveablePlan(), nullptr);
}

TEST(TestSlotProgramBenchmark, TheWorkspaceIsTheAlignedSharedRegionPlusTheLargestScratchOfAnyKernel)
{
    Bench bench;

    // The shared region is 1000 bytes, 1024 aligned; the largest scratch is dkdv_b's 16384, and a
    // benchmark run may launch any kernel, so the caller's buffer must hold it.
    EXPECT_EQ(bench.benchmarkingWorkspaceSize(), WORKSPACE_BYTES);

    // A limit prunes dkdv_b and dq_b: the largest remaining scratch is dkdv_c's 8192.
    EXPECT_EQ(bench.benchmarkingWorkspaceSize(
                  {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 9300}}),
              1024U + 8192U);
}

TEST(TestSlotProgramBenchmark, WritesPerSlotRankingsFromWholeProgramTimesAndTheProgramRecord)
{
    Bench bench;
    bench.benchmarkAndServe();

    const auto prep = bench.slotRecord("prep");
    const auto dkdv = bench.slotRecord("dkdv");
    const auto dq = bench.slotRecord("dq");
    ASSERT_TRUE(prep.has_value() && dkdv.has_value() && dq.has_value());
    EXPECT_EQ(seedsOf(*prep), (std::vector<uint8_t>{0x12, 0x11}));
    EXPECT_EQ(seedsOf(*dkdv), (std::vector<uint8_t>{0x23, 0x21, 0x22}));
    EXPECT_EQ(seedsOf(*dq), (std::vector<uint8_t>{0x31, 0x32}));
    // Whole-program times: the slot's kernel, the others at their reference picks.
    EXPECT_DOUBLE_EQ((*prep)[0].timeMs, 3.0 + 10.0 + 4.0);
    EXPECT_DOUBLE_EQ((*prep)[1].timeMs, 19.0);
    EXPECT_DOUBLE_EQ((*dkdv)[0].timeMs, 5.0 + 7.0 + 4.0);
    EXPECT_DOUBLE_EQ((*dkdv)[2].timeMs, 5.0 + 12.0 + 4.0);
    EXPECT_DOUBLE_EQ((*dq)[1].timeMs, 5.0 + 10.0 + 6.0);
    // Pack and dispatch ids travel with each entry, as the coverage check requires.
    EXPECT_EQ((*dkdv)[0].packId, testId(0x20));
    EXPECT_EQ((*dkdv)[0].dispatchId, testId(0xD1));

    const auto program = bench.programRecord();
    ASSERT_TRUE(program.has_value()) << "the program record is valid against the slot records";
    EXPECT_EQ(picksOf(*program), WINNERS);
    EXPECT_DOUBLE_EQ(program->timeMs, 14.0);
}

/// Every log line that parses as a record of @p event. Parsed rather than substring-matched, so
/// a row that stopped being valid JSON fails the assertions.
template <typename TRecorder>
std::vector<nlohmann::json> loggedRecords(const TRecorder& recorder, const std::string& event)
{
    std::vector<nlohmann::json> records;
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        const auto start = recorded.message.find('{');
        if(start == std::string::npos)
        {
            continue;
        }
        auto parsed = nlohmann::json::parse(recorded.message.substr(start), nullptr, false);
        if(!parsed.is_discarded() && parsed.contains("event") && parsed["event"] == event)
        {
            records.push_back(std::move(parsed));
        }
    }
    return records;
}

/// The id of a program of @p seeds as the program row names it.
std::string programId(const Program& seeds)
{
    std::string id;
    for(size_t slot = 0; slot < seeds.size(); ++slot)
    {
        id += (slot == 0 ? "" : "|") + slotNames()[slot] + "=" + toString(testId(seeds[slot]));
    }
    return id;
}

TEST(TestSlotProgramBenchmark, ABenchmarkLogsOneRowPerTimedKernelAndOneForTheWinnersProgram)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Bench bench;
    bench.benchmarkAndServe();

    const auto candidates = loggedRecords(recorder, "ingestor.benchmark.candidate");
    ASSERT_EQ(candidates.size(), ALL_SEEDS.size());
    for(const auto seed : ALL_SEEDS)
    {
        const auto slot = static_cast<size_t>(seed >> 4) - 1;
        const auto row = std::find_if(
            candidates.begin(), candidates.end(), [seed](const nlohmann::json& record) {
                return record["kernel"] == toString(testId(seed));
            });
        ASSERT_NE(row, candidates.end()) << static_cast<int>(seed);
        EXPECT_EQ((*row)["slot"], slotNames()[slot]);
        EXPECT_EQ((*row)["pack"], toString(testId(static_cast<uint8_t>(seed & 0xF0))));
        EXPECT_EQ((*row)["dispatch"], toString(testId(static_cast<uint8_t>(0xD0 + slot))));
        EXPECT_EQ((*row)["status"], "ok");
        // A whole-program time: the kernel, the other slots at their reference picks.
        EXPECT_DOUBLE_EQ((*row)["robust_mean_ms"].get<double>(),
                         19.0 - COSTS.at(REFERENCE[slot]) + COSTS.at(seed));
        EXPECT_EQ((*row)["benchmark"], candidates.front()["benchmark"]);
    }

    const auto programs = loggedRecords(recorder, "ingestor.benchmark.program");
    ASSERT_EQ(programs.size(), 1U);
    EXPECT_EQ(programs.front()["program"], programId(WINNERS));
    EXPECT_EQ(programs.front()["status"], "ok");
    EXPECT_DOUBLE_EQ(programs.front()["robust_mean_ms"].get<double>(), 14.0);
    EXPECT_EQ(programs.front()["benchmark"], candidates.front()["benchmark"]);
}

TEST(TestSlotProgramBenchmark, AKernelThatFailsToLaunchIsLoggedAsAFailedRowWithItsReason)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Bench bench;
    bench.fixture().log().failingLaunch.insert(testId(0x22));
    bench.benchmarkAndServe();

    const auto candidates = loggedRecords(recorder, "ingestor.benchmark.candidate");
    ASSERT_EQ(candidates.size(), ALL_SEEDS.size());
    size_t failed = 0;
    for(const auto& row : candidates)
    {
        if(row["status"] == "ok")
        {
            EXPECT_TRUE(row.contains("robust_mean_ms"));
            continue;
        }
        ++failed;
        EXPECT_EQ(row["status"], "failed");
        EXPECT_EQ(row["kernel"], toString(testId(0x22)));
        EXPECT_EQ(row["slot"], "dkdv");
        EXPECT_FALSE(row["reason"].get<std::string>().empty());
        EXPECT_FALSE(row.contains("robust_mean_ms"));
    }
    EXPECT_EQ(failed, 1U);
}

TEST(TestSlotProgramBenchmark, AFailedWinnersProgramIsLoggedAsTheFailingRowNotTheReference)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Bench bench;
    bench.clock().failToTime(WINNERS);
    ASSERT_EQ(bench.benchmarkAndServe(), REFERENCE);

    const auto programs = loggedRecords(recorder, "ingestor.benchmark.program");
    ASSERT_EQ(programs.size(), 1U);
    EXPECT_EQ(programs.front()["status"], "failed");
    EXPECT_EQ(programs.front()["program"], programId(WINNERS))
        << "the row names the program that failed, not the reference that was served";
    EXPECT_FALSE(programs.front()["reason"].get<std::string>().empty());
}

TEST(TestSlotProgramBenchmark, EqualProgramTimesRankTheLowestKernelIndexFirst)
{
    Bench bench;
    // dkdv_b now costs what dkdv_a does: a tie between the reference and the first variation.
    bench.clock().setCost(0x22, COSTS.at(0x21));
    bench.benchmarkAndServe();

    const auto dkdv = bench.slotRecord("dkdv");
    ASSERT_TRUE(dkdv.has_value());
    EXPECT_EQ(seedsOf(*dkdv), (std::vector<uint8_t>{0x23, 0x21, 0x22}));
}

TEST(TestSlotProgramBenchmark, ABenchmarkedProblemIsServedFromItsRecordsWithoutTimingAgain)
{
    Bench bench;
    bench.benchmarkAndServe();
    bench.clock().forget();

    const auto served = bench.benchmarkAndServe();

    EXPECT_EQ(bench.clock().sampleCount(), 0U);
    EXPECT_EQ(served, WINNERS);
}

TEST(TestSlotProgramBenchmark, AFailedWinnersProgramServesTheReferenceAndWritesNoProgramLine)
{
    Bench bench;
    bench.clock().failToTime(WINNERS);

    const auto served = bench.benchmarkAndServe();

    EXPECT_EQ(served, REFERENCE);
    EXPECT_EQ(bench.clock().runs().back(), WINNERS) << "the winners program was attempted";
    EXPECT_FALSE(bench.programRecord().has_value());
    // The slot records stand: their measurements are sound.
    ASSERT_TRUE(bench.slotRecord("dkdv").has_value());
    EXPECT_EQ(seedsOf(*bench.slotRecord("dkdv")), (std::vector<uint8_t>{0x23, 0x21, 0x22}));
}

TEST(TestSlotProgramBenchmark, AStaleProgramLineIsSupersededWhenItsWinnersAreTimedAgain)
{
    // A named engine keeps its records in a shard on disk, where a stale program line can be
    // adopted instead of superseded.
    const ScopedCacheDir cacheDir("stale_program");
    Bench bench(
        false,
        defaultSlotKernels(),
        EngineIdentity{"test:StaleProgram", {1, 0, 0}, toString(ENGINE_ID), "abcd1234abcd1234"});
    bench.benchmarkAndServe();
    ASSERT_EQ(picksOf(*bench.programRecord()), WINNERS);
    // The slot records are rewritten with another winner for prep, so the program line on disk
    // names a winner prep no longer has.
    bench.fixture().manager().recordSlotted(
        bench.baseKey(),
        {{"prep",
          WinnerRecord{{testId(0x11), testId(0x10), testId(0xD0), 1.0},
                       {testId(0x12), testId(0x10), testId(0xD0), 2.0}}}},
        std::nullopt,
        WinnerWriteCause::COVERAGE_REBENCHMARK,
        WinnerWriteCause::FRESH_MISS);
    ASSERT_FALSE(bench.programRecord().has_value()) << "the old program line is now invalid";
    bench.clock().forget();

    const auto served = bench.benchmarkAndServe();

    const Program expected{0x11, 0x23, 0x31};
    EXPECT_EQ(bench.clock().runs(), (std::vector<Program>{expected}));
    EXPECT_EQ(served, expected);
    const auto program = bench.programRecord();
    ASSERT_TRUE(program.has_value()) << "the new program line must supersede the stale one";
    EXPECT_EQ(picksOf(*program), expected);
}

TEST(TestSlotProgramBenchmark, WithOnlyTheProgramRecordInvalidOnlyTheWinnersProgramIsTimed)
{
    Bench bench;
    bench.clock().failToTime(WINNERS);
    bench.benchmarkAndServe();
    ASSERT_FALSE(bench.programRecord().has_value());
    bench.clock().clearFailure();
    bench.clock().forget();

    const auto served = bench.benchmarkAndServe();

    // Every slot is covered, so there is no reference program and no varied run.
    EXPECT_EQ(bench.clock().runs(), (std::vector<Program>{WINNERS}));
    EXPECT_EQ(bench.clock().sampleCount(), static_cast<size_t>(SAMPLES_PER_RUN));
    EXPECT_EQ(served, WINNERS);
    const auto program = bench.programRecord();
    ASSERT_TRUE(program.has_value());
    EXPECT_EQ(picksOf(*program), WINNERS);
    EXPECT_DOUBLE_EQ(program->timeMs, 14.0);
}

TEST(TestSlotProgramBenchmark, ACoveredSlotIsNotMeasuredAndItsRecordIsNotRewritten)
{
    Bench bench;
    // dkdv's catalog is covered, with dkdv_c ahead.
    bench.seedSlotRecord(1, {0x23, 0x21, 0x22});

    const auto served = bench.benchmarkAndServe();

    // The reference holds dkdv at its winner; only prep and dq are varied.
    const std::vector<Program> expected{
        {0x11, 0x23, 0x31}, {0x12, 0x23, 0x31}, {0x11, 0x23, 0x32}, WINNERS};
    EXPECT_EQ(bench.clock().runs(), expected);
    EXPECT_EQ(served, WINNERS);
    ASSERT_TRUE(bench.slotRecord("dkdv").has_value());
    EXPECT_DOUBLE_EQ((*bench.slotRecord("dkdv"))[0].timeMs, 1.0) << "the seeded entry stands";
    EXPECT_EQ(seedsOf(*bench.slotRecord("prep")), (std::vector<uint8_t>{0x12, 0x11}));
    ASSERT_TRUE(bench.programRecord().has_value());
}

TEST(TestSlotProgramBenchmark, AReferencePickThatFailsToLaunchAdvancesItsSlotToTheNextKernel)
{
    Bench bench;
    bench.fixture().log().failingLaunch.insert(testId(0x11));

    const auto served = bench.benchmarkAndServe();

    // prep_a never launches: prep's reference is prep_b, and prep has nothing left to vary.
    const std::vector<Program> expected{
        {0x12, 0x21, 0x31}, {0x12, 0x22, 0x31}, {0x12, 0x23, 0x31}, {0x12, 0x21, 0x32}, WINNERS};
    EXPECT_EQ(bench.clock().runs(), expected);
    EXPECT_EQ(served, WINNERS);
    // The failed kernel is omitted, never ranked.
    ASSERT_TRUE(bench.slotRecord("prep").has_value());
    EXPECT_EQ(seedsOf(*bench.slotRecord("prep")), (std::vector<uint8_t>{0x12}));
    EXPECT_DOUBLE_EQ((*bench.slotRecord("prep"))[0].timeMs, 19.0 - 5.0 + 3.0);
}

TEST(TestSlotProgramBenchmark, ASlotWhoseReferenceHasNoNextKernelServesTheProgramWithoutRecords)
{
    Bench bench;
    // dq's two kernels both fail to launch.
    bench.fixture().log().failingLaunch.insert(testId(0x31));
    bench.fixture().log().failingLaunch.insert(testId(0x32));

    // The reference program is served as it stands, and fails when run.
    EXPECT_THROW(bench.benchmarkAndServe(), std::runtime_error);

    EXPECT_EQ(bench.clock().sampleCount(), 0U);
    for(const auto* slot : {"prep", "dkdv", "dq"})
    {
        EXPECT_FALSE(bench.slotRecord(slot).has_value()) << slot;
    }
    EXPECT_FALSE(bench.programRecord().has_value());
}

TEST(TestSlotProgramBenchmark, AKnobFilterTimesTheFilteredCatalogsAndWritesNoRecords)
{
    Bench bench;

    // dkdv is pinned to dkdv_a alone, so only prep and dq are varied.
    const auto served = bench.benchmarkAndServe({{"dkdv.block_m", 64}});

    const std::vector<Program> expected{
        REFERENCE, {0x12, 0x21, 0x31}, {0x11, 0x21, 0x32}, {0x12, 0x21, 0x31}};
    EXPECT_EQ(bench.clock().runs(), expected);
    EXPECT_EQ(served, (Program{0x12, 0x21, 0x31}));
    for(const auto* slot : {"prep", "dkdv", "dq"})
    {
        EXPECT_FALSE(bench.slotRecord(slot).has_value()) << slot;
    }
    EXPECT_FALSE(bench.programRecord().has_value());
}

TEST(TestSlotProgramBenchmark, AWorkspaceLimitTimesThePrunedCatalogsAndWritesNoRecords)
{
    Bench bench;

    const auto servedLimited
        = bench.benchmarkAndServe({{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 9300}});
    // dkdv_b (16384) and dq_b (12288) are pruned, dkdv_c (8192 + 1024) is not.
    const std::vector<Program> expected{
        REFERENCE, {0x12, 0x21, 0x31}, {0x11, 0x23, 0x31}, {0x12, 0x23, 0x31}};
    EXPECT_EQ(bench.clock().runs(), expected);
    EXPECT_EQ(servedLimited, (Program{0x12, 0x23, 0x31}));
    for(const auto* slot : {"prep", "dkdv", "dq"})
    {
        EXPECT_FALSE(bench.slotRecord(slot).has_value()) << slot;
    }
    EXPECT_FALSE(bench.programRecord().has_value());
}

TEST(TestSlotProgramBenchmark, ANarrowedPlanWhoseSlotsAreAllCoveredIsServedWithoutTiming)
{
    Bench bench;
    bench.seedSlotRecord(0, {0x12, 0x11});
    bench.seedSlotRecord(1, {0x23, 0x21, 0x22});
    bench.seedSlotRecord(2, {0x31, 0x32});
    ASSERT_FALSE(bench.programRecord().has_value());

    // No program record exists, so only the narrowing keeps the benchmark from running.
    const auto served = bench.benchmarkAndServe({{"dkdv.block_m", 64}});

    EXPECT_EQ(bench.clock().sampleCount(), 0U);
    EXPECT_EQ(served, (Program{0x12, 0x21, 0x31}));
    EXPECT_FALSE(bench.programRecord().has_value());
}

TEST(TestSlotProgramBenchmark, ACandidateWhoseBuildRejectsTheProblemEndsTheBenchmarkPlanBuild)
{
    Bench bench;
    // INVALID_VALUE is the caller's problem, not the kernel's: no later candidate is tried.
    bench.fixture().log().failingPrepareWithStatus[testId(0x22)]
        = HIPDNN_PLUGIN_STATUS_INVALID_VALUE;

    try
    {
        bench.benchmarkAndServe();
        FAIL() << "the plan build must not drop an INVALID_VALUE candidate";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
}

TEST(TestSlotProgramBenchmark, ACandidateWhoseBuildFailsOtherwiseIsDroppedFromTheBenchmark)
{
    Bench bench;
    bench.fixture().log().failingPrepareWithStatus[testId(0x22)]
        = HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;

    const auto served = bench.benchmarkAndServe();

    // dkdv_b never builds, so dkdv varies over dkdv_c alone.
    for(const auto& run : bench.clock().runs())
    {
        EXPECT_NE(run[1], 0x22);
    }
    EXPECT_EQ(served, WINNERS);
}

TEST(TestSlotProgramBenchmark, AMeasuredProgramIsPredictedWithItsTimeAndPinnedKnobs)
{
    Bench bench;
    bench.benchmarkAndServe();

    const auto prediction = bench.predict();

    ASSERT_EQ(prediction.status, PredictionStatus::AVAILABLE) << prediction.reason;
    EXPECT_DOUBLE_EQ(prediction.value, 14.0);
    ASSERT_NE(prediction.engine_config, nullptr);
    EXPECT_EQ(pinnedKnobs(*prediction.engine_config),
              (std::map<std::string, int64_t>{
                  {"prep.block_m", 128}, {"dkdv.block_m", 256}, {"dq.block_m", 64}}));
}

TEST(TestSlotProgramBenchmark, ASettingThatSelectsAnotherProgramMakesThePredictionUnavailable)
{
    Bench bench;
    bench.benchmarkAndServe();

    // dkdv_b is not the measured winner.
    const auto prediction = bench.predict({{"dkdv.block_m", 128}});

    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.engine_config, nullptr);
}

TEST(TestSlotProgramBenchmark, AProgramRecordIsPredictedByAnEngineThatHasNotBuiltTheCatalogs)
{
    Bench bench;
    bench.recordMeasuredProgramWithoutBuildingCatalogs();

    const auto prediction = bench.predict();

    EXPECT_EQ(prediction.status, PredictionStatus::AVAILABLE) << prediction.reason;
    EXPECT_DOUBLE_EQ(prediction.value, 14.0);
}

TEST(TestSlotProgramBenchmark, AServedKernelSharingItsKnobTupleMakesThePredictionUnavailable)
{
    // Both dkdv kernels carry block_m = 64, the slot's only knob: no setting identifies either.
    Bench bench(false,
                {{{0x11, "prep_a", 64, 0}},
                 {{0x21, "dkdv_a", 64, 4096}, {0x22, "dkdv_b", 64, 8192}},
                 {{0x31, "dq_a", 64, 0}}});
    bench.seedSlotRecord(0, {0x11});
    bench.seedSlotRecord(1, {0x22, 0x21});
    bench.seedSlotRecord(2, {0x31});
    ProgramRecord program;
    program.picks = {{"prep", testId(0x11)}, {"dkdv", testId(0x22)}, {"dq", testId(0x31)}};
    program.timeMs = 14.0;
    bench.fixture().manager().recordSlotted(
        bench.baseKey(), {}, program, WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS);
    ASSERT_TRUE(bench.programRecord().has_value());

    const auto prediction = bench.predict();

    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE) << prediction.reason;
    EXPECT_EQ(prediction.engine_config, nullptr);
}

TEST(TestSlotProgramBenchmark, ACoveredSlotWhoseWinnerFailsToBuildIsTimedAgain)
{
    Bench bench;
    bench.seedSlotRecord(0, {0x12, 0x11});
    bench.seedSlotRecord(1, {0x23, 0x21, 0x22});
    bench.seedSlotRecord(2, {0x31, 0x32});
    bench.fixture().log().failingPrepare.insert(testId(0x23));

    const auto served = bench.benchmarkAndServe();

    // dkdv is no longer covered: its buildable kernels are varied rather than the program
    // being timed once with an arbitrary stand-in for the winner.
    const std::vector<Program> expected{{0x12, 0x21, 0x31}, {0x12, 0x22, 0x31}, {0x12, 0x21, 0x31}};
    EXPECT_EQ(bench.clock().runs(), expected);
    EXPECT_EQ(served, (Program{0x12, 0x21, 0x31}));
    EXPECT_EQ(seedsOf(*bench.slotRecord("dkdv")), (std::vector<uint8_t>{0x21, 0x22}));
}

/// IEngine::getMaxWorkspaceSize with `global.benchmarking` SHOULD cover the priming plan, so a
/// caller that sizes its workspace from the estimate can run the benchmark. The slotted estimate
/// is the aligned shared region plus the largest scratch of every surviving kernel of every slot,
/// which the benchmark's own plan can only equal or undercut.
TEST(TestSlotProgramBenchmark, TheBenchmarkingWorkspaceEstimateCoversThePrimingPlan)
{
    Bench bench;

    const auto primingPlan = bench.benchmarkingWorkspaceSize();
    const auto estimate = bench.benchmarkingWorkspaceEstimate();

    EXPECT_GT(primingPlan, 0U);
    EXPECT_GE(estimate, primingPlan);
}

#ifdef HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES
/// Every slot's ranking and the program record are written, and the served program is the
/// recorded one: the comparison completed rather than serving the reference unrecorded.
void expectCompleteComparison(Bench& bench, const Program& served)
{
    for(const auto& [slot, kernels] :
        std::vector<std::pair<std::string, size_t>>{{"prep", 2}, {"dkdv", 3}, {"dq", 2}})
    {
        const auto record = bench.slotRecord(slot);
        ASSERT_TRUE(record.has_value()) << slot;
        EXPECT_EQ(record->size(), kernels) << slot;
    }
    const auto program = bench.programRecord();
    ASSERT_TRUE(program.has_value());
    EXPECT_EQ(picksOf(*program), served);
}

/// A device that cannot wait on a stream value has no stall gate to arm. The slot program
/// benchmark then times every program unstalled in one pass: nothing is discarded or re-timed.
class TestSlotProgramBenchmarkUnsupportedDevice : public StreamWaitFaultsFixture
{
};

TEST_F(TestSlotProgramBenchmarkUnsupportedDevice, StartsUnstalledAndKeepsEverySample)
{
    // The unstalled launches the benchmark makes on a device that never stalls.
    size_t unstalledLaunches = 0;
    {
        Bench reference;
        static_cast<void>(reference.benchmarkAndServe());
        unstalledLaunches = reference.fixture().log().launches.size();
    }

    gStreamWaitFaults.unsupported = true;
    // A stalled reading would carry a value no unstalled reading of these short kernels nears.
    constexpr float STALLED_READING_MS = 1.0e6F;
    gStreamWaitFaults.stalledReadingMs = STALLED_READING_MS;
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    ASSERT_GT(gStreamWaitFaults.capabilityQueries, 0);
    EXPECT_EQ(gStreamWaitFaults.armCalls, 0);
    EXPECT_EQ(gStreamWaitFaults.stalledReadings, 0);
    EXPECT_EQ(bench.fixture().log().launches.size(), unstalledLaunches)
        << "samples were discarded or re-timed";
    expectCompleteComparison(bench, served);
    for(const std::string slot : {"prep", "dkdv", "dq"})
    {
        const auto record = bench.slotRecord(slot);
        ASSERT_TRUE(record.has_value()) << slot;
        for(const auto& entry : *record)
        {
            EXPECT_LT(entry.timeMs, STALLED_READING_MS) << slot;
        }
    }
}

/// The real default timer arms a stall gate on every sample of a stalled pass. A fault injected
/// at the HIP boundary makes one sample unstallable, which must discard the whole stalled pass
/// and re-measure every program unstalled (RFC 0021 §7).
class TestSlotProgramBenchmarkStallRestart : public StreamWaitFaultsFixture
{
protected:
    void SetUp() override
    {
        StreamWaitFaultsFixture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        SKIP_UNLESS_STREAM_WAIT_VALUE();
    }
};

TEST_F(TestSlotProgramBenchmarkStallRestart, AStalledPassThatCompletesIsNotRestarted)
{
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    // The reference, the four one-slot variations and the winners program all armed the gate.
    EXPECT_EQ(gStreamWaitFaults.rejectedArms, 0);
    EXPECT_GE(gStreamWaitFaults.armCalls, 6 * SAMPLES_PER_RUN);
    expectCompleteComparison(bench, served);
}

TEST_F(TestSlotProgramBenchmarkStallRestart, AnUnstallableReferenceRestartsTheComparisonUnstalled)
{
    // The reference program's first sample cannot arm the gate.
    gStreamWaitFaults.failFromArm = 1;
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    // The stalled pass stops at that arm; the unstalled restart never arms again. The reference
    // is re-timed, not given up on: records are written.
    EXPECT_EQ(gStreamWaitFaults.rejectedArms, 1);
    EXPECT_EQ(gStreamWaitFaults.armCalls, 1);
    expectCompleteComparison(bench, served);
}

TEST_F(TestSlotProgramBenchmarkStallRestart,
       AnUnstallableVariedSlotRunStopsTheStalledPassAndRestartsTheComparisonUnstalled)
{
    // The reference program is timed stalled; the first varied program's first sample cannot arm.
    gStreamWaitFaults.failFromArm = SAMPLES_PER_RUN + 1;
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    // No later program of the stalled pass is attempted, and the restart never arms.
    EXPECT_EQ(gStreamWaitFaults.rejectedArms, 1);
    EXPECT_EQ(gStreamWaitFaults.armCalls, SAMPLES_PER_RUN + 1);
    expectCompleteComparison(bench, served);
}

TEST_F(TestSlotProgramBenchmarkStallRestart,
       AnUnstallableWinnersProgramRestartsTheComparisonUnstalled)
{
    // The reference and the four one-slot variations are timed stalled; the winners program's
    // first sample cannot arm.
    gStreamWaitFaults.failFromArm = 5 * SAMPLES_PER_RUN + 1;
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    // The stalled pass stops at that arm and the unstalled restart never arms again.
    EXPECT_EQ(gStreamWaitFaults.rejectedArms, 1);
    EXPECT_EQ(gStreamWaitFaults.armCalls, 5 * SAMPLES_PER_RUN + 1);
    expectCompleteComparison(bench, served);
}

/// The winners program is the last program timed, so no later stalled arm exposes a restart that
/// re-times only the winners program. Stalled readings carry a value no unstalled reading of
/// these short kernels approaches: a ranking or program record holding one mixed a stalled
/// sample into the comparison.
TEST_F(TestSlotProgramBenchmarkStallRestart,
       AnUnstallableWinnersProgramKeepsNoStalledSampleInTheRestartedComparison)
{
    constexpr float STALLED_READING_MS = 1.0e6F;
    gStreamWaitFaults.failFromArm = 5 * SAMPLES_PER_RUN + 1;
    gStreamWaitFaults.stalledReadingMs = STALLED_READING_MS;
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    // The reference and the four variations were timed stalled before the winners arm failed.
    EXPECT_EQ(gStreamWaitFaults.stalledReadings, 5 * SAMPLES_PER_RUN);
    expectCompleteComparison(bench, served);
    for(const std::string slot : {"prep", "dkdv", "dq"})
    {
        const auto record = bench.slotRecord(slot);
        ASSERT_TRUE(record.has_value()) << slot;
        for(const auto& entry : *record)
        {
            EXPECT_LT(entry.timeMs, STALLED_READING_MS) << slot;
        }
    }
    const auto program = bench.programRecord();
    ASSERT_TRUE(program.has_value());
    EXPECT_LT(program->timeMs, STALLED_READING_MS);
}

/// Every stalled reading is negative, the shape the Windows HIP runtime reports for short stalled
/// work. A candidate that spends its retry budget on negative stalled readings must restart the
/// comparison unstalled rather than be dropped, so every slot's ranking is still recorded and
/// the served program is the recorded one.
TEST_F(TestSlotProgramBenchmarkStallRestart,
       APersistentlyNegativeStalledComparisonRestartsTheComparisonUnstalled)
{
    gStreamWaitFaults.negativeStalledReadings = std::numeric_limits<int>::max();
    Bench bench(/*defaultTimer=*/true);

    const auto served = bench.benchmarkAndServe();

    // The first reference sample spends its retries, then the stalled pass stops: nothing later
    // is timed stalled, and the restart arms no more gates.
    EXPECT_EQ(gStreamWaitFaults.stalledReadings, 1 + MAX_NEGATIVE_SAMPLE_RETRIES);
    expectCompleteComparison(bench, served);
    for(const std::string slot : {"prep", "dkdv", "dq"})
    {
        const auto record = bench.slotRecord(slot);
        ASSERT_TRUE(record.has_value()) << slot;
        for(const auto& entry : *record)
        {
            EXPECT_GE(entry.timeMs, 0.0) << slot;
        }
    }
}
#endif

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
