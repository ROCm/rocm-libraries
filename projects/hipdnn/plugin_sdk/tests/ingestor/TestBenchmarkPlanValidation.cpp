// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkValidation.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "HostValidationHooks.hpp"

/**
 * @file TestBenchmarkPlanValidation.cpp
 * @brief Unit tests for BenchmarkPlan's output validation: which candidates may win, what is
 *        recorded, what is logged, and that validation adds no launches. Candidates are host
 *        doubles that write bytes into host "device" buffers; the fill and snapshot hooks are
 *        host memset and memcpy, so no device is needed.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::output_agreement::firstDisagreement;
using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_plugin_sdk::ingestor::detail::findDeviceBuffer;
using hipdnn_plugin_sdk::ingestor::host_hooks::entryIds;
using hipdnn_plugin_sdk::ingestor::host_hooks::floatBytes;
using hipdnn_plugin_sdk::ingestor::host_hooks::HookCounters;
using hipdnn_plugin_sdk::ingestor::host_hooks::hostValidationHooks;
using hipdnn_test_sdk::utilities::SharedLogRecorder;

/// A handle satisfying HasGetStream, which BenchmarkPlan's constructor static_asserts.
struct ValidationTestHandle
{
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    hipStream_t getStream() const
    {
        return nullptr;
    }
};

using ValidationPlan = BenchmarkPlan<ValidationTestHandle>;
using PlanInterface = hipdnn_plugin_sdk::IPlan<ValidationTestHandle>;

constexpr int64_t OUTPUT_UID = 1;
constexpr size_t FOUR_FLOATS = 4 * sizeof(float);

DescriptorId idOf(uint8_t seed)
{
    DescriptorId id{};
    id.fill(seed);
    return id;
}

DescriptorId packId()
{
    return idOf(0xE6);
}

DescriptorId dispatchId()
{
    return idOf(0xE5);
}

KernelTriple tripleOf(uint8_t seed)
{
    return {idOf(seed), packId(), dispatchId()};
}

CatalogDigest digestOf(std::initializer_list<uint8_t> seeds)
{
    std::vector<KernelTriple> triples;
    for(const uint8_t seed : seeds)
    {
        triples.push_back(tripleOf(seed));
    }
    return catalogDigest(std::move(triples));
}

/// What every correct kernel writes, up to the float tolerance.
std::vector<uint8_t> correctImage()
{
    return floatBytes({1.0F, 2.0F, 3.0F, 4.0F});
}

/// Within the float tolerance of correctImage(), yet not byte-identical: tells which correct
/// candidate's output a buffer holds.
std::vector<uint8_t> variantImage()
{
    return floatBytes({1.0000001F, 2.0F, 3.0F, 4.0F});
}

std::vector<uint8_t> wrongImage()
{
    return floatBytes({9.0F, 9.0F, 9.0F, 9.0F});
}

std::vector<OutputBufferSpec> floatOutput()
{
    return {{OUTPUT_UID, DataType::FLOAT, FOUR_FLOATS}};
}

/// A plan double that writes fixed bytes into the host buffer bound to each uid, counting
/// launches. The first @p throwForCalls calls throw before writing anything.
class WritingPlan : public PlanInterface
{
public:
    using Writes = std::map<int64_t, std::vector<uint8_t>>;

    explicit WritingPlan(Writes writes, int throwForCalls = 0)
        : _writes(std::move(writes))
        , _throwForCalls(throwForCalls)
    {
    }

    size_t getWorkspaceSize(const ValidationTestHandle& /*handle*/) const override
    {
        return 0;
    }

    void execute(const ValidationTestHandle& /*handle*/,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* /*workspace*/ = nullptr) const override
    {
        ++_calls;
        if(_calls <= _throwForCalls)
        {
            throw std::runtime_error("WritingPlan: simulated failure");
        }
        ++_launches;
        for(const auto& [uid, bytes] : _writes)
        {
            void* ptr = findDeviceBuffer(deviceBuffers, numDeviceBuffers, uid);
            if(ptr != nullptr)
            {
                std::memcpy(ptr, bytes.data(), bytes.size());
            }
        }
    }

    int launchCount() const
    {
        return _launches;
    }

private:
    Writes _writes;
    int _throwForCalls;
    mutable int _calls = 0;
    mutable int _launches = 0;
};

/// A timer returning a fixed duration per plan and forwarding the real execute(), so launches
/// still accrue. A plan absent from @p durations cannot be timed, which scores it unusable.
ValidationPlan::Timer makeTimer(std::map<const PlanInterface*, double> durations)
{
    return [durations = std::move(durations)](const PlanInterface& plan,
                                              const ValidationTestHandle& handle,
                                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                              uint32_t numDeviceBuffers,
                                              void* workspace) -> std::optional<double> {
        const auto found = durations.find(&plan);
        if(found == durations.end())
        {
            return std::nullopt;
        }
        plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        return found->second;
    };
}

/// One BenchmarkPlan over writing candidates, host buffers and host hooks, with both sinks
/// (the legacy ranking callback and the validated record) captured.
class Scenario
{
public:
    explicit Scenario(std::vector<OutputBufferSpec> outputs, std::vector<int64_t> inputUids = {})
        : _outputs(std::move(outputs))
        , _inputUids(std::move(inputUids))
    {
        for(const auto& spec : _outputs)
        {
            addBuffer(spec.uid, spec.byteSize);
        }
    }

    Scenario(const Scenario&) = delete;
    Scenario& operator=(const Scenario&) = delete;

    /// A zeroed host buffer bound to @p uid.
    void addBuffer(int64_t uid, size_t bytes)
    {
        auto& storage = _storage[uid];
        storage.assign(bytes, 0x00);
        _deviceBuffers.push_back({uid, storage.data()});
    }

    void aliasBuffer(int64_t uid, int64_t targetUid)
    {
        bufferEntry(uid).ptr = bufferEntry(targetUid).ptr;
    }

    void nullBuffer(int64_t uid)
    {
        bufferEntry(uid).ptr = nullptr;
    }

    /// @p milliseconds absent leaves the candidate untimeable.
    void addCandidate(uint8_t seed,
                      WritingPlan::Writes writes,
                      std::optional<double> milliseconds,
                      int throwForCalls = 0)
    {
        auto plan = std::make_unique<WritingPlan>(std::move(writes), throwForCalls);
        _plans.push_back(plan.get());
        if(milliseconds.has_value())
        {
            _durations[plan.get()] = *milliseconds;
        }
        _candidates.push_back({idOf(seed), std::move(plan), packId(), dispatchId()});
    }

    /// Builds the plan once; validation mode when @p validationOn.
    ValidationPlan& build(bool validationOn)
    {
        std::optional<BenchmarkValidation> validation;
        if(validationOn)
        {
            BenchmarkValidation configured;
            configured.outputs = _outputs;
            configured.inputUids = _inputUids;
            configured.hooks = hostValidationHooks(hooks);
            configured.record
                = [this](WinnerRecord record) { records.push_back(std::move(record)); };
            validation = std::move(configured);
        }
        _plan = std::make_unique<ValidationPlan>(
            std::move(_candidates),
            ValidationTestHandle{},
            makeTimer(_durations),
            [this](std::vector<RankedEntry> entries) { rankings.push_back(std::move(entries)); },
            "validation-test",
            "device-0",
            std::move(validation));
        return *_plan;
    }

    /// The first execute(), which resolves the winner.
    void execute()
    {
        _plan->execute(ValidationTestHandle{},
                       _deviceBuffers.data(),
                       static_cast<uint32_t>(_deviceBuffers.size()),
                       nullptr);
    }

    /// Index of the chosen candidate in the order added.
    size_t winner() const
    {
        const PlanInterface* chosen = _plan->chosenPlan();
        for(size_t i = 0; i < _plans.size(); ++i)
        {
            if(_plans[i] == chosen)
            {
                return i;
            }
        }
        return _plans.size();
    }

    int launches(size_t candidate) const
    {
        return _plans[candidate]->launchCount();
    }

    const std::vector<uint8_t>& bufferBytes(int64_t uid) const
    {
        return _storage.at(uid);
    }

    HookCounters hooks;
    std::vector<WinnerRecord> records;
    std::vector<std::vector<RankedEntry>> rankings;

private:
    hipdnnPluginDeviceBuffer_t& bufferEntry(int64_t uid)
    {
        for(auto& entry : _deviceBuffers)
        {
            if(entry.uid == uid)
            {
                return entry;
            }
        }
        throw std::logic_error("Scenario: no buffer with that uid");
    }

    std::vector<OutputBufferSpec> _outputs;
    std::vector<int64_t> _inputUids;
    std::map<int64_t, std::vector<uint8_t>> _storage;
    std::vector<hipdnnPluginDeviceBuffer_t> _deviceBuffers;
    std::vector<const WritingPlan*> _plans;
    std::vector<ValidationPlan::Candidate> _candidates;
    std::map<const PlanInterface*, double> _durations;
    std::unique_ptr<ValidationPlan> _plan;
};

std::vector<std::string> messagesAt(const SharedLogRecorder& recorder, hipdnnSeverity_t severity)
{
    std::vector<std::string> messages;
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        if(recorded.severity == severity)
        {
            messages.push_back(recorded.message);
        }
    }
    return messages;
}

/// True when one message contains every needle.
bool anyContainsAll(const std::vector<std::string>& messages,
                    std::initializer_list<std::string> needles)
{
    for(const auto& message : messages)
    {
        bool all = true;
        for(const auto& needle : needles)
        {
            all = all && message.find(needle) != std::string::npos;
        }
        if(all)
        {
            return true;
        }
    }
    return false;
}

/// The `reason` of the failed candidate record logged for @p kernelId, if one was logged.
std::optional<std::string> failedReasonOf(const SharedLogRecorder& recorder,
                                          const DescriptorId& kernelId)
{
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        const auto start = recorded.message.find('{');
        if(start == std::string::npos)
        {
            continue;
        }
        const auto parsed = nlohmann::json::parse(recorded.message.substr(start), nullptr, false);
        if(!parsed.is_object() || parsed.value("event", "") != "ingestor.benchmark.candidate"
           || parsed.value("status", "") != "failed"
           || parsed.value("kernel", "") != toString(kernelId))
        {
            continue;
        }
        return parsed.value("reason", "");
    }
    return std::nullopt;
}

/// A validated record exists and is the only one.
::testing::AssertionResult hasOneValidatedRecord(const Scenario& scenario)
{
    if(scenario.records.size() != 1U)
    {
        return ::testing::AssertionFailure()
               << "expected one validated record, got " << scenario.records.size();
    }
    if(!scenario.records.front().validated.has_value())
    {
        return ::testing::AssertionFailure() << "the record is not marked validated";
    }
    return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Which candidate may win
// ---------------------------------------------------------------------------

/// Index 0 is correct and slower; index 1 is wrong and faster.
void addFastWrongSlowCorrect(Scenario& scenario)
{
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 5.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, wrongImage()}}, 2.0);
}

TEST(TestIngestorBenchmarkPlanValidation,
     AFastWrongCandidateIsExcludedAndTheSlowerCorrectHeuristicFirstOneWins)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario validated(floatOutput());
    addFastWrongSlowCorrect(validated);
    validated.build(true);
    validated.execute();

    EXPECT_EQ(validated.winner(), 0U);
    ASSERT_TRUE(hasOneValidatedRecord(validated));
    EXPECT_EQ(entryIds(validated.records.front()), std::vector<DescriptorId>{idOf(0x10)});
    EXPECT_TRUE(validated.rankings.empty()) << "validation replaces the legacy ranking sink";

    // The same fixture without validation: the fast wrong candidate wins.
    Scenario unvalidated(floatOutput());
    addFastWrongSlowCorrect(unvalidated);
    unvalidated.build(false);
    unvalidated.execute();
    EXPECT_EQ(unvalidated.winner(), 1U);
    EXPECT_TRUE(unvalidated.records.empty());
    EXPECT_EQ(unvalidated.hooks.fills + unvalidated.hooks.snapshots, 0);
}

/// Two correct candidates (indexes 0 and 1) and a faster wrong one at index 2.
void addTwoCorrectAndOneFasterWrong(Scenario& scenario)
{
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 5.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.addCandidate(0x12, {{OUTPUT_UID, wrongImage()}}, 1.0);
}

TEST(TestIngestorBenchmarkPlanValidation,
     TwoCorrectAndOneFasterWrongCandidateElectTheCorrectCohortByMajority)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario validated(floatOutput());
    addTwoCorrectAndOneFasterWrong(validated);
    validated.build(true);
    validated.execute();
    EXPECT_EQ(validated.winner(), 1U) << "the fastest correct candidate";
    ASSERT_TRUE(hasOneValidatedRecord(validated));
    EXPECT_EQ(entryIds(validated.records.front()),
              (std::vector<DescriptorId>{idOf(0x11), idOf(0x10)}));

    Scenario unvalidated(floatOutput());
    addTwoCorrectAndOneFasterWrong(unvalidated);
    unvalidated.build(false);
    unvalidated.execute();
    EXPECT_EQ(unvalidated.winner(), 2U);
}

TEST(TestIngestorBenchmarkPlanValidation, AMajorityBeatsAWrongHeuristicFirstCandidate)
{
    const auto addFixture = [](Scenario& scenario) {
        scenario.addCandidate(0x10, {{OUTPUT_UID, wrongImage()}}, 1.0);
        scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 3.0);
        scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, 4.0);
    };

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario validated(floatOutput());
    addFixture(validated);
    validated.build(true);
    validated.execute();
    EXPECT_EQ(validated.winner(), 1U);
    EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {toString(idOf(0x10))}))
        << recorder.getRecordedLogsAsString();

    Scenario unvalidated(floatOutput());
    addFixture(unvalidated);
    unvalidated.build(false);
    unvalidated.execute();
    EXPECT_EQ(unvalidated.winner(), 0U);
}

TEST(TestIngestorBenchmarkPlanValidation, TheLogNamesTheExcludedKernelIdAndTheObservedError)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    addFastWrongSlowCorrect(scenario);
    scenario.build(true);
    scenario.execute();

    // The wrong candidate's image against the passing cohort founder's.
    const std::string observed
        = firstDisagreement("output 1", DataType::FLOAT, wrongImage(), correctImage());
    ASSERT_FALSE(observed.empty());
    EXPECT_TRUE(
        anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {toString(idOf(0x11)), observed}))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(
        anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {toString(idOf(0x10)), observed}))
        << "the passing candidate must not be named as excluded";
}

// ---------------------------------------------------------------------------
// Untouched outputs
// ---------------------------------------------------------------------------

/// Index 0 never writes and is the fastest, so only the untouched rule can exclude it: voting
/// alone would tie and the heuristic would pick it. Index 1 writes a pattern that is neither
/// the fill nor a NaN.
void expectANonWriterIsExcludedForDtype(DataType dataType, size_t elementBytes, uint8_t writerByte)
{
    SCOPED_TRACE(EnumNameDataType(dataType));
    const size_t bytes = 4 * elementBytes;
    const std::vector<OutputBufferSpec> outputs{{OUTPUT_UID, dataType, bytes}};
    const WritingPlan::Writes writer{{OUTPUT_UID, std::vector<uint8_t>(bytes, writerByte)}};

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario validated(outputs);
    validated.addCandidate(0x10, {}, 1.0);
    validated.addCandidate(0x11, writer, 2.0);
    validated.build(true);
    validated.execute();
    EXPECT_EQ(validated.winner(), 1U);
    EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN),
                               {toString(idOf(0x10)), "left untouched"}))
        << recorder.getRecordedLogsAsString();
    ASSERT_TRUE(hasOneValidatedRecord(validated));
    EXPECT_EQ(entryIds(validated.records.front()), std::vector<DescriptorId>{idOf(0x11)});

    // Without validation the non-writer wins: the case discriminates.
    Scenario unvalidated(outputs);
    unvalidated.addCandidate(0x10, {}, 1.0);
    unvalidated.addCandidate(0x11, writer, 2.0);
    unvalidated.build(false);
    unvalidated.execute();
    EXPECT_EQ(unvalidated.winner(), 0U);
}

TEST(TestIngestorBenchmarkPlanValidation, AnUntouchedOutputOfFloatDtypeIsExcluded)
{
    expectANonWriterIsExcludedForDtype(DataType::FLOAT, 4, 0x3F);
}

TEST(TestIngestorBenchmarkPlanValidation, AnUntouchedOutputOfHalfDtypeIsExcluded)
{
    expectANonWriterIsExcludedForDtype(DataType::HALF, 2, 0x3C);
}

TEST(TestIngestorBenchmarkPlanValidation, AnUntouchedOutputOfBfloat16DtypeIsExcluded)
{
    expectANonWriterIsExcludedForDtype(DataType::BFLOAT16, 2, 0x3F);
}

TEST(TestIngestorBenchmarkPlanValidation, AnUntouchedOutputOfDoubleDtypeIsExcluded)
{
    expectANonWriterIsExcludedForDtype(DataType::DOUBLE, 8, 0x3F);
}

/// FP8 is never value-compared, so the untouched rule is the only mechanism. E4M3, E5M2 and
/// E8M0 fill with 0xFF; the FNUZ types with 0x80.
TEST(TestIngestorBenchmarkPlanValidation, AnUntouchedOutputOfFp8DtypeIsExcluded)
{
    for(const auto dataType : {DataType::FP8_E4M3,
                               DataType::FP8_E5M2,
                               DataType::FP8_E8M0,
                               DataType::FP8_E4M3_FNUZ,
                               DataType::FP8_E5M2_FNUZ})
    {
        expectANonWriterIsExcludedForDtype(dataType, 1, 0x01);
    }
}

TEST(TestIngestorBenchmarkPlanValidation,
     ABufferLeftUntouchedAmongSeveralOutputsExcludesTheCandidate)
{
    constexpr int64_t FIRST_UID = 11;
    constexpr int64_t SECOND_UID = 22;
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const std::vector<OutputBufferSpec> outputs{{FIRST_UID, DataType::FLOAT, FOUR_FLOATS},
                                                {SECOND_UID, DataType::FLOAT, FOUR_FLOATS}};
    Scenario scenario(outputs);
    // Fastest and heuristic-first, but only the first output is written.
    scenario.addCandidate(0x10, {{FIRST_UID, correctImage()}}, 1.0);
    scenario.addCandidate(0x11, {{FIRST_UID, correctImage()}, {SECOND_UID, correctImage()}}, 2.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN),
                               {toString(idOf(0x10)), "output 22 left untouched"}))
        << recorder.getRecordedLogsAsString();
}

TEST(TestIngestorBenchmarkPlanValidation, AnUnwrittenIntegerOutputIsCaughtOnlyByDisagreement)
{
    const std::vector<OutputBufferSpec> outputs{{OUTPUT_UID, DataType::INT32, 4 * sizeof(int32_t)}};
    const std::vector<uint8_t> written(4 * sizeof(int32_t), 0x05);

    {
        // Two writers outvote the non-writer; no untouched rule is involved.
        auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        Scenario scenario(outputs);
        scenario.addCandidate(0x10, {}, 1.0);
        scenario.addCandidate(0x11, {{OUTPUT_UID, written}}, 2.0);
        scenario.addCandidate(0x12, {{OUTPUT_UID, written}}, 3.0);
        scenario.build(true);
        scenario.execute();
        EXPECT_EQ(scenario.winner(), 1U);
        EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {toString(idOf(0x10))}))
            << recorder.getRecordedLogsAsString();
        EXPECT_FALSE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {"left untouched"}));
    }
    {
        // One writer against one non-writer is a tie, so the heuristic-first non-writer wins:
        // the documented limit for types without a NaN encoding.
        Scenario scenario(outputs);
        scenario.addCandidate(0x10, {}, 1.0);
        scenario.addCandidate(0x11, {{OUTPUT_UID, written}}, 2.0);
        scenario.build(true);
        scenario.execute();
        EXPECT_EQ(scenario.winner(), 0U);
        ASSERT_TRUE(hasOneValidatedRecord(scenario));
        EXPECT_EQ(entryIds(scenario.records.front()), std::vector<DescriptorId>{idOf(0x10)});
    }
}

TEST(TestIngestorBenchmarkPlanValidation, OutputsThatAreNotValueComparedWarnAndSayWhichCheckApplied)
{
    struct Case
    {
        DataType dataType;
        const char* expected;
        const char* unexpected;
    };
    const std::array cases{
        Case{DataType::FP8_E4M3, "untouched check applied", "neither check applied"},
        Case{DataType::FP8_E4M3_FNUZ, "untouched check applied", "neither check applied"},
        Case{DataType::FP4_E2M1, "neither check applied", "untouched check applied"},
        Case{DataType::FP6_E2M3, "neither check applied", "untouched check applied"},
        Case{DataType::FP6_E3M2, "neither check applied", "untouched check applied"},
        Case{DataType::INT4, "neither check applied", "untouched check applied"},
    };
    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(EnumNameDataType(testCase.dataType));
        auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        const std::vector<OutputBufferSpec> outputs{{OUTPUT_UID, testCase.dataType, 4}};
        Scenario scenario(outputs);
        scenario.addCandidate(0x10, {{OUTPUT_UID, {0x01, 0x02, 0x03, 0x04}}}, 2.0);
        scenario.addCandidate(0x11, {{OUTPUT_UID, {0x05, 0x06, 0x07, 0x08}}}, 3.0);
        scenario.build(true);
        scenario.execute();

        const auto warnings = messagesAt(recorder, HIPDNN_SEV_WARN);
        EXPECT_TRUE(anyContainsAll(warnings, {"value comparison skipped", testCase.expected}))
            << recorder.getRecordedLogsAsString();
        EXPECT_FALSE(anyContainsAll(warnings, {testCase.unexpected}))
            << recorder.getRecordedLogsAsString();
    }
}

// ---------------------------------------------------------------------------
// What is recorded
// ---------------------------------------------------------------------------

TEST(TestIngestorBenchmarkPlanValidation,
     NoPassingCandidateFallsBackToTheFrontLogsErrorAndRecordsNothing)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    // Candidate 0 is the slowest, so a fallback to the ranking would not pick it.
    scenario.addCandidate(0x10, {}, 9.0);
    scenario.addCandidate(0x11, {}, 2.0);
    scenario.addCandidate(0x12, {}, 3.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 0U);
    EXPECT_TRUE(scenario.records.empty());
    EXPECT_TRUE(scenario.rankings.empty());
    EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_ERROR), {toString(idOf(0x10))}))
        << recorder.getRecordedLogsAsString();
    for(const int seed : {0x10, 0x11, 0x12})
    {
        EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN),
                                   {toString(idOf(static_cast<uint8_t>(seed))), "left untouched"}))
            << recorder.getRecordedLogsAsString();
    }
}

TEST(TestIngestorBenchmarkPlanValidation,
     TheRecordHoldsOnlyPassingEntriesAndTheDigestCoversEveryClassifiedCandidate)
{
    Scenario scenario(floatOutput());
    addTwoCorrectAndOneFasterWrong(scenario);
    scenario.build(true);
    scenario.execute();

    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), (std::vector<DescriptorId>{idOf(0x11), idOf(0x10)}));
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x11, 0x12}));
    EXPECT_NE(*record.validated, digestOf({0x10, 0x11}));
    EXPECT_TRUE(scenario.rankings.empty());
}

TEST(TestIngestorBenchmarkPlanValidation,
     ALoneComputingCandidatePassesUnverifiedAndIsRecordedValidated)
{
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {}, 1.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 5.0);
    scenario.addCandidate(0x12, {}, 2.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), std::vector<DescriptorId>{idOf(0x11)});
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x11, 0x12}));
}

TEST(TestIngestorBenchmarkPlanValidation, AnInconclusiveResolveIsRecordedValidated)
{
    const std::vector<OutputBufferSpec> outputs{{OUTPUT_UID, DataType::INT4, 4}};
    Scenario scenario(outputs);
    scenario.addCandidate(0x10, {{OUTPUT_UID, {0x01, 0x02, 0x03, 0x04}}}, 5.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, {0x09, 0x08, 0x07, 0x06}}}, 3.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), (std::vector<DescriptorId>{idOf(0x11), idOf(0x10)}));
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x11}));
}

TEST(TestIngestorBenchmarkPlanValidation,
     AnUntouchedCandidateIsInTheDigestSoTheRecordCoversTheCatalog)
{
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 3.0);
    scenario.addCandidate(0x12, {}, 1.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    const auto ids = entryIds(record);
    EXPECT_EQ(ids, (std::vector<DescriptorId>{idOf(0x11), idOf(0x10)}))
        << "the untouched kernel is absent from the entries";
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x11, 0x12}))
        << "and present in the digest, so the record covers the whole catalog";
}

// ---------------------------------------------------------------------------
// Skipped validation
// ---------------------------------------------------------------------------

/// Validation skipped: no hook ran, nothing was excluded (the fastest candidate wins even
/// though it is the wrong one), and the skip is warned about. Records nothing itself.
void expectSkipped(Scenario& scenario, const SharedLogRecorder& recorder, const char* reason)
{
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.hooks.fills, 0);
    EXPECT_EQ(scenario.hooks.snapshots, 0);
    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_TRUE(
        anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {"validation skipped", reason}))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(scenario.rankings.empty());
}

/// A skip caused by the caller's buffers: the resolve records nothing, so a call with usable
/// buffers can still benchmark and record.
void expectSkippedWithoutARecord(Scenario& scenario,
                                 const SharedLogRecorder& recorder,
                                 const char* reason)
{
    expectSkipped(scenario, recorder, reason);
    EXPECT_TRUE(scenario.records.empty());
}

TEST(TestIngestorBenchmarkPlanValidation, AnOutputAliasingAnInputSkipsValidationWithAWarning)
{
    constexpr int64_t INPUT_UID = 2;
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput(), {INPUT_UID});
    scenario.addBuffer(INPUT_UID, FOUR_FLOATS);
    scenario.aliasBuffer(INPUT_UID, OUTPUT_UID);
    addFastWrongSlowCorrect(scenario);
    expectSkippedWithoutARecord(scenario, recorder, "alias");
}

TEST(TestIngestorBenchmarkPlanValidation, ANullOutputPointerSkipsValidation)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    scenario.nullBuffer(OUTPUT_UID);
    addFastWrongSlowCorrect(scenario);
    expectSkippedWithoutARecord(scenario, recorder, "null");
}

/// Two float outputs, so a check that stops after the first one is caught.
constexpr int64_t SECOND_OUTPUT_UID = 3;

std::vector<OutputBufferSpec> twoFloatOutputs()
{
    return {{OUTPUT_UID, DataType::FLOAT, FOUR_FLOATS},
            {SECOND_OUTPUT_UID, DataType::FLOAT, FOUR_FLOATS}};
}

TEST(TestIngestorBenchmarkPlanValidation, ANullSecondOutputPointerSkipsValidation)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(twoFloatOutputs());
    scenario.nullBuffer(SECOND_OUTPUT_UID);
    addFastWrongSlowCorrect(scenario);
    expectSkippedWithoutARecord(scenario, recorder, "null");
}

TEST(TestIngestorBenchmarkPlanValidation,
     ASecondOutputAliasingAnInputSkipsValidationAndLeavesTheInputUnfilled)
{
    constexpr int64_t INPUT_UID = 2;
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(twoFloatOutputs(), {INPUT_UID});
    scenario.addBuffer(INPUT_UID, FOUR_FLOATS);
    scenario.aliasBuffer(INPUT_UID, SECOND_OUTPUT_UID);
    addFastWrongSlowCorrect(scenario);
    expectSkippedWithoutARecord(scenario, recorder, "alias");
    EXPECT_THAT(scenario.bufferBytes(SECOND_OUTPUT_UID), ::testing::Each(0x00));
}

TEST(TestIngestorBenchmarkPlanValidation, AnEmptyOutputListSkipsValidation)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(std::vector<OutputBufferSpec>{});
    scenario.addBuffer(OUTPUT_UID, FOUR_FLOATS);
    addFastWrongSlowCorrect(scenario);
    expectSkipped(scenario, recorder, "no output");
    // Nothing to check is a property of the graph, so the resolve is recorded validated.
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), (std::vector<DescriptorId>{idOf(0x11), idOf(0x10)}));
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x11}));
}

// ---------------------------------------------------------------------------
// The caller's buffers and the launch count
// ---------------------------------------------------------------------------

TEST(TestIngestorBenchmarkPlanValidation, AfterTheFirstExecuteTheBuffersHoldAPassingWinnersOutput)
{
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 5.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, variantImage()}}, 3.0);
    // Sampled last, wrong and fastest: it wins without validation, and it is what the buffer holds
    // after sampling unless the delegated execute rewrites it.
    scenario.addCandidate(0x12, {{OUTPUT_UID, wrongImage()}}, 1.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_EQ(scenario.bufferBytes(OUTPUT_UID), variantImage());
}

TEST(TestIngestorBenchmarkPlanValidation,
     AfterTheFirstExecuteTheBuffersHoldTheFallbackCandidatesOutput)
{
    constexpr int64_t NEVER_WRITTEN_UID = 5;
    constexpr int64_t WRITTEN_UID = 6;
    const std::vector<OutputBufferSpec> outputs{{NEVER_WRITTEN_UID, DataType::FLOAT, FOUR_FLOATS},
                                                {WRITTEN_UID, DataType::FLOAT, FOUR_FLOATS}};
    Scenario scenario(outputs);
    // Every candidate leaves the first output untouched, so none passes and candidate 0 is served.
    scenario.addCandidate(0x10, {{WRITTEN_UID, correctImage()}}, 9.0);
    scenario.addCandidate(0x11, {{WRITTEN_UID, variantImage()}}, 2.0);
    scenario.addCandidate(0x12, {{WRITTEN_UID, wrongImage()}}, 3.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 0U);
    EXPECT_EQ(scenario.bufferBytes(WRITTEN_UID), correctImage());
    EXPECT_TRUE(scenario.records.empty());
    EXPECT_TRUE(scenario.rankings.empty());
}

TEST(TestIngestorBenchmarkPlanValidation, ValidationAddsNoLaunches)
{
    const auto addFixture = [](Scenario& scenario) {
        scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 5.0);
        scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 3.0);
        scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, 4.0);
    };
    Scenario validated(floatOutput());
    addFixture(validated);
    validated.build(true);
    validated.execute();

    Scenario unvalidated(floatOutput());
    addFixture(unvalidated);
    unvalidated.build(false);
    unvalidated.execute();

    ASSERT_EQ(validated.winner(), unvalidated.winner());
    for(size_t candidate = 0; candidate < 3; ++candidate)
    {
        EXPECT_EQ(validated.launches(candidate), unvalidated.launches(candidate)) << candidate;
    }
    // One fill and one snapshot per candidate for the one pass.
    EXPECT_EQ(validated.hooks.fills, 3);
    EXPECT_EQ(validated.hooks.snapshots, 3);
}

// ---------------------------------------------------------------------------
// Voting and timing
// ---------------------------------------------------------------------------

TEST(TestIngestorBenchmarkPlanValidation,
     ASnapshotFailureScoresTheCandidateUnusableAndItDoesNotVote)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    // The failing copy belongs to the first candidate, which would otherwise win on time.
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 3.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, 5.0);
    scenario.hooks.failSnapshotAtCall = 1;
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_EQ(failedReasonOf(recorder, idOf(0x10)),
              std::optional<std::string>{"validation-copy-failed"})
        << recorder.getRecordedLogsAsString();
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), (std::vector<DescriptorId>{idOf(0x11), idOf(0x12)}));
    EXPECT_EQ(*record.validated, digestOf({0x11, 0x12}))
        << "a failed copy is not a failed validation: outside the digest";
}

TEST(TestIngestorBenchmarkPlanValidation, AShortSnapshotScoresTheCandidateUnusableAndItDoesNotVote)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, wrongImage()}}, 3.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, 5.0);
    // The hook reports success for the first candidate but returns no image for it.
    scenario.hooks.shortSnapshotAtCall = 1;
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_EQ(failedReasonOf(recorder, idOf(0x10)),
              std::optional<std::string>{"validation-copy-failed"})
        << recorder.getRecordedLogsAsString();
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), (std::vector<DescriptorId>{idOf(0x11), idOf(0x12)}));
    EXPECT_EQ(*record.validated, digestOf({0x11, 0x12}))
        << "an unvoted candidate is outside the digest, so it cannot have been classified";
}

TEST(TestIngestorBenchmarkPlanValidation, AThrowingWarmupDoesNotVote)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 1.0, /*throwForCalls=*/1);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, 5.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_EQ(scenario.hooks.snapshots, 2) << "the candidate that threw takes no snapshot";
    EXPECT_FALSE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {"left untouched"}))
        << "its buffer still holds the fill, but it never voted";
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    EXPECT_EQ(*scenario.records.front().validated, digestOf({0x11, 0x12}));
}

TEST(TestIngestorBenchmarkPlanValidation, AnUntimedWrongCandidateIsStillExcludedAndNamed)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, 3.0);
    scenario.addCandidate(0x12, {{OUTPUT_UID, wrongImage()}}, std::nullopt);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U);
    EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_WARN), {toString(idOf(0x12))}))
        << recorder.getRecordedLogsAsString();
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), (std::vector<DescriptorId>{idOf(0x11), idOf(0x10)}));
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x11, 0x12}));
    EXPECT_NE(*record.validated, digestOf({0x10, 0x11}));
}

TEST(TestIngestorBenchmarkPlanValidation, AnUntimedCorrectCandidateStillCountsTowardTheMajority)
{
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, wrongImage()}}, 3.0);
    scenario.addCandidate(0x11, {{OUTPUT_UID, correctImage()}}, std::nullopt);
    scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, 4.0);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 2U) << "without the untimed vote the tie-break picks index 0";
    ASSERT_TRUE(hasOneValidatedRecord(scenario));
    const auto& record = scenario.records.front();
    EXPECT_EQ(entryIds(record), std::vector<DescriptorId>{idOf(0x12)});
    EXPECT_EQ(*record.validated, digestOf({0x10, 0x12}))
        << "the failed index 0 is inside the digest, the untimed passing index 1 is not";
}

TEST(TestIngestorBenchmarkPlanValidation,
     AllTimingUnusableWithModeOnServesTheFirstPassingCandidateAndRecordsNothing)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Scenario scenario(floatOutput());
    scenario.addCandidate(0x10, {{OUTPUT_UID, wrongImage()}}, std::nullopt);
    scenario.addCandidate(0x11, {{OUTPUT_UID, variantImage()}}, std::nullopt);
    scenario.addCandidate(0x12, {{OUTPUT_UID, correctImage()}}, std::nullopt);
    scenario.build(true);
    scenario.execute();

    EXPECT_EQ(scenario.winner(), 1U) << "the lowest-index passing candidate, never index 0";
    EXPECT_TRUE(scenario.records.empty());
    EXPECT_TRUE(scenario.rankings.empty());
    EXPECT_TRUE(anyContainsAll(messagesAt(recorder, HIPDNN_SEV_ERROR), {toString(idOf(0x11))}))
        << recorder.getRecordedLogsAsString();
    // Index 2 was sampled last and also holds a correct image: only the delegated execute of
    // index 1 leaves the variant.
    EXPECT_EQ(scenario.bufferBytes(OUTPUT_UID), variantImage());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
