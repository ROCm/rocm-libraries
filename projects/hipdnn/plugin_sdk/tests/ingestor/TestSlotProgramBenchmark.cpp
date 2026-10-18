// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkDetail.hpp>
#include <hipdnn_plugin_sdk/ingestor/SlotProgramBenchmark.hpp>
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
                total += COSTS.at(seed);
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
};

using PredictionResult = hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT;
using hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;

/// The engine, its fake clock, and a graph with a usable content key. With @p defaultTimer the
/// engine times with the real HIP timer, and the fake clock stays unused.
class Bench
{
public:
    explicit Bench(bool defaultTimer = false)
        : _fixture(defaultSlotKernels(),
                   defaultTimer ? BenchmarkPlan<StubHandle>::Timer{} : _clock.timer())
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

#ifdef HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES
/// The real default timer arms a stall gate on every sample of a stalled pass. A fault injected
/// at the HIP boundary makes one sample unstallable, which must discard the whole stalled pass
/// and re-measure every program unstalled (RFC 0021 §7).
class TestSlotProgramBenchmarkStallRestart : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gStreamWaitFaults = {};
        SKIP_IF_NO_DEVICES();
        int device = 0;
        int canWait = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipDeviceGetAttribute(&canWait, hipDeviceAttributeCanUseStreamWaitValue, device),
                  hipSuccess);
        if(canWait == 0)
        {
            GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
        }
        gStreamWaitFaults.enabled = true;
    }

    void TearDown() override
    {
        gStreamWaitFaults = {};
        static_cast<void>(hipGetLastError());
    }

    /// Every slot's ranking and the program record are written, and the served program is the
    /// recorded one: the comparison completed rather than serving the reference unrecorded.
    static void expectCompleteComparison(Bench& bench, const Program& served)
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
#endif

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
