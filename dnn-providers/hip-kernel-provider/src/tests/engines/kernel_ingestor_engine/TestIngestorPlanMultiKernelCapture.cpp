// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/SlotProgramBenchmark.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/RestoredSlotProgramPlan.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
#include "utilities/Digest.hpp"
#include "version.h"

/**
 * @file TestIngestorPlanMultiKernelCapture.cpp
 * @brief captureIngestorPlan for the plans a kind 1 payload cannot hold: a plan that launches
 *        several kernels, and a slot program, saved as a kind 2 payload.
 *
 * The handlers are stubs of PackedPlanTestSupport.hpp. They prepare real code objects of the
 * packed multi_launch shard, which capture reads back, and need no device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::BenchmarkPlan;
using hipdnn_plugin_sdk::ingestor::DescriptorId;
using hipdnn_plugin_sdk::ingestor::GenericPlan;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SlotProgramBenchmark;
using hipdnn_plugin_sdk::ingestor::SlotProgramPlan;
using hipdnn_plugin_sdk::ingestor::SlotStage;
using serialization::IngestorMultiKernelPayload;
using serialization::IngestorSavedPlan;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using packs::MultiKernelStubHandler;
using packs::NO_WINNER_PHRASE;
using packs::PACKED_CONSUME_LAUNCH;
using packs::PACKED_PRODUCE_LAUNCH;
using packs::RegisteredStubHandler;
using Config = MultiKernelStubHandler::Config;
using Launch = MultiKernelStubHandler::Launch;

constexpr size_t ALIGNED_SHARED_BYTES = 1024;

// A handler that launches both packed launches under their names, and opts in.
Config twoLaunchConfig(const std::string& alias, const std::string& tag)
{
    Config config;
    config.alias = alias;
    config.tag = tag;
    config.launches = {Launch{PACKED_PRODUCE_LAUNCH, PACKED_PRODUCE_LAUNCH},
                       Launch{PACKED_CONSUME_LAUNCH, PACKED_CONSUME_LAUNCH}};
    config.optsIn = true;
    return config;
}

// A handler that launches the one `source` of a single-launch kernel.
Config oneLaunchConfig(const std::string& alias, const std::string& tag)
{
    Config config;
    config.alias = alias;
    config.tag = tag;
    return config;
}

class TestIngestorPlanMultiKernelCapture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_NO_FATAL_FAILURE(packs::loadPackedLaunchSource(_source));
    }

    // The packed kernel with its two launches in `sources`, under its own id.
    KernelDefinition multiLaunchKernel(uint8_t seed) const
    {
        auto kernel = _source.kernel;
        kernel.kernelId.fill(seed);
        return kernel;
    }

    // The packed kernel as one launch of `launch`, under its own id.
    KernelDefinition singleLaunchKernel(uint8_t seed,
                                        const std::string& launch = PACKED_PRODUCE_LAUNCH) const
    {
        auto kernel = _source.singleLaunchKernel(launch);
        kernel.kernelId.fill(seed);
        return kernel;
    }

    const hipdnn_plugin_sdk::ingestor::KernelSource& sourceOf(const std::string& launch) const
    {
        return _source.kernel.sources.at(launch);
    }

    std::string archBase() const
    {
        return std::string(hipdnn_plugin_sdk::stripArchFeatures(_source.arch));
    }

    // Captures `plan` and returns the kind 2 payload it holds.
    static IngestorMultiKernelPayload captureKindTwo(const hipdnn_plugin_sdk::IPlan<Handle>& plan)
    {
        const Handle handle;
        const IngestorSavedPlan saved = captureIngestorPlan(plan, handle);
        const auto* payload = std::get_if<IngestorMultiKernelPayload>(&saved);
        EXPECT_NE(payload, nullptr) << "expected a kind 2 payload";
        return payload != nullptr ? *payload : IngestorMultiKernelPayload{};
    }

    static std::string expectSaveRefusal(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                                         hipdnnPluginStatus_t status,
                                         const std::string& phrase)
    {
        return serialization::expectIngestorPlanSaveRefusal(
            [&plan]() {
                const Handle handle;
                static_cast<void>(captureIngestorPlan(plan, handle));
            },
            status,
            phrase);
    }

    static std::shared_ptr<GenericPlan<Handle>> stagePlan(const MultiKernelStubHandler& handler,
                                                          KernelDefinition kernel)
    {
        return {packs::makeStubPlan(handler, std::move(kernel))};
    }

    packs::PackedLaunchSource _source;
};

// ---------------------------------------------------------------------------
// Kind choice
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelCapture, SavesAMultiLaunchPlanOfAnOptedInHandlerAsKindTwo)
{
    auto config = twoLaunchConfig("hipkernel.capture_test.two_launch.dispatch.v1", "main");
    config.workspaceBytes = 4096;
    const RegisteredStubHandler stub(config, _source.arch);
    const auto kernel = multiLaunchKernel(0x31);
    const auto plan = packs::makeStubPlan(stub.handler, kernel);

    const auto payload = captureKindTwo(*plan);

    ASSERT_EQ(payload.stages.size(), 1U);
    const auto& stage = payload.stages.front();
    EXPECT_EQ(stage.slot, "");
    EXPECT_EQ(stage.kernelDescriptorId, kernel.kernelId);
    EXPECT_EQ(stage.dispatchSymbol, config.alias);
    EXPECT_EQ(stage.launchValues, (MetadataValues{{"tag", std::string("main")}}));

    ASSERT_EQ(stage.kernels.size(), 2U);
    EXPECT_EQ(stage.kernels[0].launchName, PACKED_PRODUCE_LAUNCH);
    EXPECT_EQ(stage.kernels[0].symbol, sourceOf(PACKED_PRODUCE_LAUNCH).symbol);
    EXPECT_EQ(stage.kernels[1].launchName, PACKED_CONSUME_LAUNCH);
    EXPECT_EQ(stage.kernels[1].symbol, sourceOf(PACKED_CONSUME_LAUNCH).symbol);
    EXPECT_NE(stage.kernels[0].symbol, stage.kernels[1].symbol);
    EXPECT_TRUE(serialization::detail::sameKernelSignature(
        stage.kernels[0].recordedSignature, sourceOf(PACKED_PRODUCE_LAUNCH).signature));
    EXPECT_TRUE(serialization::detail::sameKernelSignature(
        stage.kernels[1].recordedSignature, sourceOf(PACKED_CONSUME_LAUNCH).signature));

    ASSERT_EQ(payload.codeObjects.size(), 1U);
    const auto& codeObject = payload.codeObjects.front();
    EXPECT_EQ(codeObject.sourceKind, KernelSourceKind::KPACK);
    EXPECT_EQ(codeObject.target, archBase());
    EXPECT_EQ(codeObject.sha256, sourceOf(PACKED_PRODUCE_LAUNCH).sha256);
    EXPECT_EQ(utilities::sha256Hex(codeObject.codeObject.data(), codeObject.codeObject.size()),
              codeObject.sha256);

    EXPECT_EQ(payload.sharedWorkspaceBytes, 0U);
    EXPECT_EQ(payload.workspaceBytes, 4096U);
    EXPECT_TRUE(payload.runtimePassByValueUids.empty());
    EXPECT_EQ(payload.providerVersion, HIP_KERNEL_PROVIDER_VERSION_STRING);
}

TEST_F(TestIngestorPlanMultiKernelCapture, SavesASlotProgramAsKindTwoWithStagesInSlotOrder)
{
    // The first stage's handler launches one kernel and does not opt in. A stage that
    // launches one kernel needs no more than a handler that saves.
    const RegisteredStubHandler prep(
        oneLaunchConfig("hipkernel.capture_test.prep.dispatch.v1", "prep"), _source.arch);
    const RegisteredStubHandler dkdv(
        twoLaunchConfig("hipkernel.capture_test.dkdv.dispatch.v1", "dkdv"), _source.arch);
    const RegisteredStubHandler dq(oneLaunchConfig("hipkernel.capture_test.dq.dispatch.v1", "dq"),
                                   _source.arch);

    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"prep", stagePlan(prep.handler, singleLaunchKernel(0x41))});
    stages.push_back({"dkdv", stagePlan(dkdv.handler, multiLaunchKernel(0x42))});
    stages.push_back(
        {"dq", stagePlan(dq.handler, singleLaunchKernel(0x43, PACKED_CONSUME_LAUNCH))});
    const SlotProgramPlan<Handle> program(std::move(stages), ALIGNED_SHARED_BYTES);

    const auto payload = captureKindTwo(program);

    ASSERT_EQ(payload.stages.size(), 3U);
    const std::vector<std::string> slots{"prep", "dkdv", "dq"};
    const std::vector<std::string> dispatches{
        prep.handler.config().alias, dkdv.handler.config().alias, dq.handler.config().alias};
    const std::vector<size_t> kernelCounts{1U, 2U, 1U};
    for(size_t index = 0; index < slots.size(); ++index)
    {
        EXPECT_EQ(payload.stages[index].slot, slots[index]);
        EXPECT_EQ(payload.stages[index].dispatchSymbol, dispatches[index]);
        EXPECT_EQ(payload.stages[index].launchValues, (MetadataValues{{"tag", slots[index]}}));
        EXPECT_EQ(payload.stages[index].kernels.size(), kernelCounts[index]);
    }
    EXPECT_EQ(payload.stages[0].kernels.front().launchName, "");
    EXPECT_EQ(payload.stages[0].kernels.front().symbol, sourceOf(PACKED_PRODUCE_LAUNCH).symbol);
    EXPECT_EQ(payload.stages[2].kernels.front().symbol, sourceOf(PACKED_CONSUME_LAUNCH).symbol);
    EXPECT_EQ(payload.stages[0].kernelDescriptorId.front(), 0x41);
    EXPECT_EQ(payload.stages[1].kernelDescriptorId.front(), 0x42);
    EXPECT_EQ(payload.stages[2].kernelDescriptorId.front(), 0x43);
}

TEST_F(TestIngestorPlanMultiKernelCapture, SavesAnUnslottedMultiLaunchPlanAsOneStageWithNoSlot)
{
    const RegisteredStubHandler stub(
        twoLaunchConfig("hipkernel.capture_test.unslotted.dispatch.v1", "u"), _source.arch);
    const auto plan = packs::makeStubPlan(stub.handler, multiLaunchKernel(0x44));

    const auto payload = captureKindTwo(*plan);

    ASSERT_EQ(payload.stages.size(), 1U);
    EXPECT_EQ(payload.stages.front().slot, "");
    EXPECT_EQ(payload.sharedWorkspaceBytes, 0U);
}

TEST_F(TestIngestorPlanMultiKernelCapture, StoresASharedCodeObjectOnce)
{
    // One code object holds both launches, in this stage and across the stages of a slot
    // program, so the payload holds it once and every kernel points at it.
    const RegisteredStubHandler first(
        twoLaunchConfig("hipkernel.capture_test.shared_a.dispatch.v1", "a"), _source.arch);
    const RegisteredStubHandler second(
        oneLaunchConfig("hipkernel.capture_test.shared_b.dispatch.v1", "b"), _source.arch);
    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"first", stagePlan(first.handler, multiLaunchKernel(0x45))});
    stages.push_back({"second", stagePlan(second.handler, singleLaunchKernel(0x46))});
    const SlotProgramPlan<Handle> program(std::move(stages), 0);

    const auto payload = captureKindTwo(program);

    ASSERT_EQ(payload.codeObjects.size(), 1U);
    for(const auto& stage : payload.stages)
    {
        for(const auto& kernel : stage.kernels)
        {
            EXPECT_EQ(kernel.codeObjectIndex, 0U);
        }
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelCapture, RefusesAStageWhoseSourceKindSavingDoesNotSupport)
{
    const RegisteredStubHandler fine(
        oneLaunchConfig("hipkernel.capture_test.fine.dispatch.v1", "fine"), _source.arch);
    const RegisteredStubHandler embedded(
        oneLaunchConfig("hipkernel.capture_test.embedded.dispatch.v1", "e"), _source.arch);
    auto kernel = singleLaunchKernel(0x47);
    kernel.source.kind = KernelSourceKind::EMBEDDED_SOURCE;

    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"fine", stagePlan(fine.handler, singleLaunchKernel(0x48))});
    stages.push_back({"embedded", stagePlan(embedded.handler, std::move(kernel))});
    const SlotProgramPlan<Handle> program(std::move(stages), 0);

    const std::string message = expectSaveRefusal(
        program, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "which saving does not yet support");
    EXPECT_NE(message.find("embedded_source"), std::string::npos) << message;
}

TEST_F(TestIngestorPlanMultiKernelCapture,
       RefusesAMultiLaunchPlanWhoseSecondLaunchSavingDoesNotSupport)
{
    const RegisteredStubHandler stub(
        twoLaunchConfig("hipkernel.capture_test.second_embedded.dispatch.v1", "x"), _source.arch);
    auto kernel = multiLaunchKernel(0x49);
    kernel.sources.at(PACKED_CONSUME_LAUNCH).kind = KernelSourceKind::EMBEDDED_SOURCE;
    const auto plan = packs::makeStubPlan(stub.handler, std::move(kernel));

    const std::string message = expectSaveRefusal(
        *plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "which saving does not yet support");
    EXPECT_NE(message.find("embedded_source"), std::string::npos) << message;
}

TEST_F(TestIngestorPlanMultiKernelCapture, RejectsAHandlerThatReportsSeveralLaunchesWithoutOptingIn)
{
    // A handler that launches several kernels and does not say it saves them all is a
    // provider fault, not a plan to refuse.
    auto config = twoLaunchConfig("hipkernel.capture_test.no_opt_in.dispatch.v1", "n");
    config.optsIn = false;
    const RegisteredStubHandler stub(config, _source.arch);
    const auto plan = packs::makeStubPlan(stub.handler, multiLaunchKernel(0x4A));

    serialization::expectPluginInternalError(
        [&plan]() {
            const Handle handle;
            static_cast<void>(captureIngestorPlan(*plan, handle));
        },
        "supportsSavingAllLaunches");
}

TEST_F(TestIngestorPlanMultiKernelCapture, RejectsAHandlerThatNamesOneOfSeveralLaunchesWithNothing)
{
    // Launch names key the saved launches of a stage, so a stage of several launches cannot
    // leave one unnamed. That is a provider fault, not a plan to refuse.
    auto config = twoLaunchConfig("hipkernel.capture_test.empty_name.dispatch.v1", "e");
    config.launches[1].name = "";
    const RegisteredStubHandler stub(config, _source.arch);
    const auto plan = packs::makeStubPlan(stub.handler, multiLaunchKernel(0x56));

    serialization::expectPluginInternalError(
        [&plan]() {
            const Handle handle;
            static_cast<void>(captureIngestorPlan(*plan, handle));
        },
        "empty launch name");
}

TEST_F(TestIngestorPlanMultiKernelCapture, RefusesASlotProgramWhoseLaterStageCannotSave)
{
    auto laterConfig = oneLaunchConfig("hipkernel.capture_test.cannot_save.dispatch.v1", "later");
    laterConfig.saves = false;
    const RegisteredStubHandler first(
        oneLaunchConfig("hipkernel.capture_test.can_save.dispatch.v1", "first"), _source.arch);
    const RegisteredStubHandler later(laterConfig, _source.arch);

    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"first", stagePlan(first.handler, singleLaunchKernel(0x57))});
    stages.push_back({"later", stagePlan(later.handler, singleLaunchKernel(0x58))});
    const SlotProgramPlan<Handle> program(std::move(stages), 0);

    const std::string message = expectSaveRefusal(
        program, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not support saving");
    EXPECT_NE(message.find("slot 'later'"), std::string::npos) << message;
}

TEST_F(TestIngestorPlanMultiKernelCapture, RefusesARestoredSlotProgramPlan)
{
    const packs::StubDispatchHandler handler;
    std::vector<RestoredSlotProgramPlan<Handle>::Stage> stages;
    stages.push_back(
        {"only", &handler, std::make_unique<hipdnn_plugin_sdk::ingestor::PreparedDispatch>()});
    const RestoredSlotProgramPlan<Handle> plan(std::move(stages), 0, 0);

    expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "loaded from a saved payload");
    const std::string message = expectSaveRefusal(
        plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not call the provider");
    EXPECT_EQ(message.find(NO_WINNER_PHRASE), std::string::npos) << message;
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelCapture, RecordsTheSharedWorkspaceAndTheTotalWorkspace)
{
    auto smallConfig = oneLaunchConfig("hipkernel.capture_test.ws_small.dispatch.v1", "small");
    smallConfig.workspaceBytes = 100;
    auto largeConfig = oneLaunchConfig("hipkernel.capture_test.ws_large.dispatch.v1", "large");
    largeConfig.workspaceBytes = 300;
    const RegisteredStubHandler small(smallConfig, _source.arch);
    const RegisteredStubHandler large(largeConfig, _source.arch);

    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"small", stagePlan(small.handler, singleLaunchKernel(0x4B))});
    stages.push_back({"large", stagePlan(large.handler, singleLaunchKernel(0x4C))});
    const SlotProgramPlan<Handle> program(std::move(stages), ALIGNED_SHARED_BYTES);
    const Handle handle;
    ASSERT_EQ(program.getWorkspaceSize(handle), ALIGNED_SHARED_BYTES + 300U);

    const auto payload = captureKindTwo(program);

    EXPECT_EQ(payload.sharedWorkspaceBytes, program.alignedSharedBytes());
    EXPECT_EQ(payload.workspaceBytes, program.getWorkspaceSize(handle));
}

// ---------------------------------------------------------------------------
// Runtime pass-by-value scalars
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelCapture, RecordsTheUserSuppliedUidsOfAKindTwoPlan)
{
    // Two stages bind scalars, and one UID is bound by both: the payload lists the union,
    // sorted, once each. No launch value holds a value for a user-supplied role.
    auto firstConfig = oneLaunchConfig("hipkernel.capture_test.uids_a.dispatch.v1", "a");
    firstConfig.uids = {5};
    auto secondConfig = twoLaunchConfig("hipkernel.capture_test.uids_b.dispatch.v1", "b");
    secondConfig.uids = {5, 9};
    const RegisteredStubHandler first(firstConfig, _source.arch);
    const RegisteredStubHandler second(secondConfig, _source.arch);

    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"first", stagePlan(first.handler, singleLaunchKernel(0x4D))});
    stages.push_back({"second", stagePlan(second.handler, multiLaunchKernel(0x4E))});
    const SlotProgramPlan<Handle> program(std::move(stages), 0);

    const auto payload = captureKindTwo(program);
    ASSERT_EQ(payload.stages.size(), 2U);

    EXPECT_EQ(payload.runtimePassByValueUids, (std::vector<int64_t>{5, 9}));
    EXPECT_EQ(payload.stages[0].launchValues, (MetadataValues{{"tag", std::string("a")}}));
    EXPECT_EQ(payload.stages[1].launchValues, (MetadataValues{{"tag", std::string("b")}}));
}

TEST_F(TestIngestorPlanMultiKernelCapture, KeepsAnEarlierStagesUidsThatNoLaterStageBinds)
{
    auto firstConfig = oneLaunchConfig("hipkernel.capture_test.uids_first.dispatch.v1", "a");
    firstConfig.uids = {5, 9};
    auto secondConfig = oneLaunchConfig("hipkernel.capture_test.uids_second.dispatch.v1", "b");
    secondConfig.uids = {5};
    const RegisteredStubHandler first(firstConfig, _source.arch);
    const RegisteredStubHandler second(secondConfig, _source.arch);

    std::vector<SlotStage<Handle>> stages;
    stages.push_back({"first", stagePlan(first.handler, singleLaunchKernel(0x59))});
    stages.push_back({"second", stagePlan(second.handler, singleLaunchKernel(0x5A))});
    const SlotProgramPlan<Handle> program(std::move(stages), 0);

    EXPECT_EQ(captureKindTwo(program).runtimePassByValueUids, (std::vector<int64_t>{5, 9}));
}

TEST_F(TestIngestorPlanMultiKernelCapture, FillsNoUidForAKindTwoPlanWithoutUserSuppliedScalars)
{
    const RegisteredStubHandler stub(
        twoLaunchConfig("hipkernel.capture_test.no_uids_two.dispatch.v1", "n"), _source.arch);
    const auto plan = packs::makeStubPlan(stub.handler, multiLaunchKernel(0x4F));

    EXPECT_TRUE(captureKindTwo(*plan).runtimePassByValueUids.empty());
}

// ---------------------------------------------------------------------------
// A benchmarking slot program, and a benchmarking plan over a multi-launch kernel
// ---------------------------------------------------------------------------

// Each slot's name and the built kernels of it.
using SlotKernels
    = std::vector<std::pair<std::string, std::vector<std::shared_ptr<GenericPlan<Handle>>>>>;

// Times a program by its workspace: the program that needs less is the faster one.
std::optional<double> timeByWorkspace(const hipdnn_plugin_sdk::IPlan<Handle>& program,
                                      const Handle& handle,
                                      const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                                      uint32_t /*numDeviceBuffers*/,
                                      void* /*workspace*/)
{
    return static_cast<double>(program.getWorkspaceSize(handle));
}

SlotProgramBenchmark<Handle> makeBenchmark(SlotKernels slots)
{
    std::vector<SlotProgramBenchmark<Handle>::Slot> built;
    for(auto& [name, plans] : slots)
    {
        auto& slot = built.emplace_back();
        slot.name = name;
        for(auto& plan : plans)
        {
            auto& kernel = slot.kernels.emplace_back();
            kernel.kernelId = plan->kernel().kernelId;
            kernel.plan = std::move(plan);
        }
    }
    return {std::move(built), ALIGNED_SHARED_BYTES, &timeByWorkspace};
}

// The `dkdv` slot offers two kernels whose scratch is 16384 and 8192, so the benchmark's own
// workspace counts the larger and the program it serves needs the smaller. The `prep` slot
// has one kernel.
class TestIngestorPlanMultiKernelCaptureBenchmark : public TestIngestorPlanMultiKernelCapture
{
protected:
    void SetUp() override
    {
        TestIngestorPlanMultiKernelCapture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        _prep = std::make_unique<RegisteredStubHandler>(
            oneLaunchConfig("hipkernel.capture_test.bench_prep.dispatch.v1", "prep"), _source.arch);
        auto largeConfig
            = oneLaunchConfig("hipkernel.capture_test.bench_large.dispatch.v1", "dkdv_large");
        largeConfig.workspaceBytes = 16384;
        _large = std::make_unique<RegisteredStubHandler>(largeConfig, _source.arch);
        auto smallConfig
            = oneLaunchConfig("hipkernel.capture_test.bench_small.dispatch.v1", "dkdv_small");
        smallConfig.workspaceBytes = 8192;
        _small = std::make_unique<RegisteredStubHandler>(smallConfig, _source.arch);
    }

    SlotProgramBenchmark<Handle> benchmark() const
    {
        SlotKernels slots;
        slots.push_back({"prep", {stagePlan(_prep->handler, singleLaunchKernel(0x51))}});
        slots.push_back({"dkdv",
                         {stagePlan(_large->handler, singleLaunchKernel(0x52)),
                          stagePlan(_small->handler, singleLaunchKernel(0x53))}});
        return makeBenchmark(std::move(slots));
    }

    // Executes `benchmark` once, which settles the program it serves.
    static void executeOnce(const SlotProgramBenchmark<Handle>& benchmark)
    {
        const Handle handle;
        std::vector<char> workspace(benchmark.getWorkspaceSize(handle));
        benchmark.execute(handle, nullptr, 0, workspace.data());
    }

    std::unique_ptr<RegisteredStubHandler> _prep;
    std::unique_ptr<RegisteredStubHandler> _large;
    std::unique_ptr<RegisteredStubHandler> _small;
};

TEST_F(TestIngestorPlanMultiKernelCaptureBenchmark,
       RefusesABenchmarkingSlotProgramBeforeItsFirstExecute)
{
    const auto program = benchmark();
    ASSERT_EQ(program.servedProgram(), nullptr);

    expectSaveRefusal(program, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "execute it once");
    expectSaveRefusal(program, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "benchmarking off");
    const std::string message
        = expectSaveRefusal(program, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, NO_WINNER_PHRASE);
    EXPECT_EQ(message.find(packs::NOT_AN_INGESTOR_ENGINE_PHRASE), std::string::npos) << message;
}

TEST_F(TestIngestorPlanMultiKernelCaptureBenchmark, SavesTheProgramABenchmarkingSlotProgramServes)
{
    const auto program = benchmark();
    executeOnce(program);
    ASSERT_NE(program.servedProgram(), nullptr);

    const auto payload = captureKindTwo(program);

    ASSERT_EQ(payload.stages.size(), 2U);
    EXPECT_EQ(payload.stages[0].slot, "prep");
    EXPECT_EQ(payload.stages[0].launchValues, (MetadataValues{{"tag", std::string("prep")}}));
    // The smaller kernel of the slot is the faster one, so it is the one the program serves.
    EXPECT_EQ(payload.stages[1].slot, "dkdv");
    EXPECT_EQ(payload.stages[1].launchValues, (MetadataValues{{"tag", std::string("dkdv_small")}}));
    EXPECT_EQ(payload.stages[1].kernelDescriptorId.front(), 0x53);
}

TEST_F(TestIngestorPlanMultiKernelCaptureBenchmark, RecordsTheServedProgramsOwnWorkspace)
{
    const auto program = benchmark();
    executeOnce(program);
    const Handle handle;
    ASSERT_EQ(program.getWorkspaceSize(handle), ALIGNED_SHARED_BYTES + 16384U);

    const auto payload = captureKindTwo(program);

    // The benchmark counts the largest scratch of every candidate, including the one that did
    // not win. The saved plan runs the served program only.
    EXPECT_EQ(payload.workspaceBytes, ALIGNED_SHARED_BYTES + 8192U);
    EXPECT_EQ(payload.workspaceBytes, program.servedProgram()->getWorkspaceSize(handle));
    EXPECT_EQ(payload.sharedWorkspaceBytes, program.servedProgram()->alignedSharedBytes());
}

TEST_F(TestIngestorPlanMultiKernelCapture, SavesABenchmarkPlanThatWrapsAMultiLaunchGenericPlan)
{
    auto multiConfig = twoLaunchConfig("hipkernel.capture_test.bench_multi.dispatch.v1", "multi");
    multiConfig.workspaceBytes = 64;
    const RegisteredStubHandler multi(multiConfig, _source.arch);
    auto otherConfig = oneLaunchConfig("hipkernel.capture_test.bench_other.dispatch.v1", "other");
    otherConfig.workspaceBytes = 128;
    const RegisteredStubHandler other(otherConfig, _source.arch);

    auto multiPlan = packs::makeStubPlan(multi.handler, multiLaunchKernel(0x54));
    auto otherPlan = packs::makeStubPlan(other.handler, singleLaunchKernel(0x55));
    const hipdnn_plugin_sdk::IPlan<Handle>* const winner = multiPlan.get();
    const DescriptorId multiId = multiPlan->kernel().kernelId;
    const DescriptorId otherId = otherPlan->kernel().kernelId;

    std::vector<BenchmarkPlan<Handle>::Candidate> candidates;
    candidates.push_back({otherId, std::move(otherPlan)});
    candidates.push_back({multiId, std::move(multiPlan)});
    const Handle handle;
    const BenchmarkPlan<Handle> plan(std::move(candidates),
                                     handle,
                                     [winner](const hipdnn_plugin_sdk::IPlan<Handle>& candidate,
                                              const Handle& /*handle*/,
                                              const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                                              uint32_t /*numDeviceBuffers*/,
                                              void* /*workspace*/) -> std::optional<double> {
                                         return &candidate == winner ? 1.0 : 5.0;
                                     });

    const std::string message
        = expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, NO_WINNER_PHRASE);
    EXPECT_EQ(message.find(packs::NOT_AN_INGESTOR_ENGINE_PHRASE), std::string::npos) << message;

    std::vector<char> workspace(plan.getWorkspaceSize(handle));
    plan.execute(handle, nullptr, 0, workspace.data());

    const auto payload = captureKindTwo(plan);
    ASSERT_EQ(payload.stages.size(), 1U);
    EXPECT_EQ(payload.stages.front().kernelDescriptorId, multiId);
    EXPECT_EQ(payload.stages.front().kernels.size(), 2U);
    EXPECT_EQ(payload.workspaceBytes, 64U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
