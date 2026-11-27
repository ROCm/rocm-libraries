// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "TestDescriptorRoot.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/MultiLaunchTestSupport.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwisePackedCase.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestMultiLaunchSaveLoad.cpp
 * @brief A multi-launch plan saved and loaded through the plugin hooks, on device.
 *
 * The plan is the packed multi_launch/two_stage shard (one kernel, two launches that share one
 * code object) served by a handler of the cases' own. That handler opts in to saving every
 * launch and is registered under a versioned name, as a production handler is. Each case
 * saves the plan of a freshly built engine through the save hook, loads the bytes through the
 * load hook on a second handle, and compares the loaded plan's output with the fresh plan's,
 * byte for byte. A damaged variant of the saved bytes is a re-encoded payload, or a resealed
 * one, so that each refusal is the one check it targets.
 *
 * The kind 1 counterpart is the pointwise packed case: a plan of one launch is saved as a
 * format 1.0 payload.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::multi_launch;
using namespace hipdnn_plugin_sdk::ingestor;
namespace hooks = hip_kernel_provider::kernel_ingestor_engine::testing::plan_hooks;
using serialization::IngestorPlanRefusal;

/// The two-stage handler, opted in to saving every launch. It adds only what saving needs: the
/// launch inputs (none, the geometry is a constant) under the versioned name, and a restore
/// that rebuilds both launches from the saved code objects.
class SavingTwoStageHandler final : public TwoStageDispatchHandler
{
public:
    SavingTwoStageHandler()
        : TwoStageDispatchHandler({PRODUCE_LAUNCH, CONSUME_LAUNCH})
    {
    }

    bool supportsSaving() const override
    {
        return true;
    }

    bool supportsSavingAllLaunches() const override
    {
        return true;
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        if(dynamic_cast<const PreparedTwoStage*>(&prepared) == nullptr)
        {
            return std::nullopt;
        }
        return SavedLaunchInputs{DISPATCH_SYMBOL_V1, {}};
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        std::vector<SavedKernelCode> codes;
        codes.push_back(std::move(code));
        return restoreLaunches(inputs, std::move(codes), deviceOrdinal);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunches(const SavedLaunchInputs& inputs,
                                                      std::vector<SavedKernelCode> codes,
                                                      int deviceOrdinal) const override
    {
        std::set<std::string> held;
        for(const auto& code : codes)
        {
            held.insert(code.launchName);
        }
        if(inputs.dispatchSymbol != DISPATCH_SYMBOL_V1 || !inputs.values.empty()
           || held.size() != codes.size() || held != launchNames())
        {
            serialization::refuseIngestorPlan(
                IngestorPlanRefusal::INCOMPATIBLE,
                "dispatch '" + inputs.dispatchSymbol
                    + "' launches {'consume', 'produce'} and takes no "
                      "launch values, and the plan holds "
                    + std::to_string(codes.size()) + " launches, " + describe(held) + ", with "
                    + std::to_string(inputs.values.size()) + " launch values");
        }

        std::optional<IngestorKernelCode> produce;
        std::optional<IngestorKernelCode> consume;
        for(auto& code : codes)
        {
            auto& slot = code.launchName == PRODUCE_LAUNCH ? produce : consume;
            std::string label = "multi-launch probe launch '" + code.launchName + "'";
            slot.emplace(IngestorKernelCode::fromSavedKernelCode(
                std::move(code), std::move(label), deviceOrdinal));
            slot->setBlockSize(BLOCK_SIZE);
            slot->setGridSize(static_cast<unsigned int>(ELEMENTS / BLOCK_SIZE));
        }
        return std::make_unique<PreparedTwoStage>(std::move(*produce), std::move(*consume));
    }

private:
    static std::string describe(const std::set<std::string>& names)
    {
        std::string text = "{";
        for(const auto& name : names)
        {
            text += (text.size() > 1 ? ", '" : "'") + name + "'";
        }
        return text + "}";
    }
};

/// Executes @p plan on @p handle over the fixed input, with a poisoned output and workspace,
/// and returns the output's bytes.
std::vector<uint8_t> runOnFixedInput(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                                     const Handle& handle)
{
    EXPECT_EQ(plan.getWorkspaceSize(handle), WORKSPACE_BYTES);
    const DeviceFloats x(inputValues());
    const DeviceFloats y(std::vector<float>(ELEMENTS, POISON));
    const DeviceFloats workspace(std::vector<float>(ELEMENTS, POISON));
    const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers{
        hipdnnPluginDeviceBuffer_t{INPUT_A_UID, x.get()},
        hipdnnPluginDeviceBuffer_t{OUTPUT_UID, y.get()}};
    EXPECT_NO_THROW(plan.execute(
        handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
    return y.readBytes();
}

class TestGpuMultiLaunchSaveLoad : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }

    /// Builds the plan of the opted-in handler on the packed shard of this device. Skips when
    /// the build packed nothing for the device.
    void buildTwoStagePlan()
    {
        PackedShard shard;
        ASSERT_NO_FATAL_FAILURE(deviceShardOrSkip(shard));
        if(shard.arch.empty())
        {
            return;
        }

        _harness = std::make_unique<MultiLaunchHarness>(_handler, true);
        std::vector<ValidatedSet<Handle>> validated;
        const KernelDefinition* kept = nullptr;
        ASSERT_NO_FATAL_FAILURE(
            loadWithKeptKernel(*_harness, shard, currentDeviceProperties(), validated, kept));
        for(const auto& [launch, source] : kept->sources)
        {
            _distinctObjects.insert(source.sha256);
        }

        _handle = hooks::makeHandle();
        ASSERT_NO_FATAL_FAILURE(makeTwoStageContext(validated[0], *_handle, _engine, _context));

        // The plan the cases save computes what the two launches compute, and nothing else:
        // the comparisons below are of its output with the loaded plan's.
        _fresh = runOnFixedInput(_context.plan(), *_handle);
        const auto expected = expectedOutput();
        ASSERT_EQ(_fresh.size(), expected.size() * sizeof(float));
        ASSERT_EQ(std::memcmp(_fresh.data(), expected.data(), _fresh.size()), 0)
            << "the fresh plan does not compute what its two launches compute";
    }

    /// The bytes the save hook writes for the fresh plan.
    void saveTwoStagePlan(std::vector<uint8_t>& bytes)
    {
        ASSERT_NO_FATAL_FAILURE(hooks::save(*_handle, _context, bytes));
    }

    SavingTwoStageHandler _handler;
    std::unique_ptr<MultiLaunchHarness> _harness;
    std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> _engine;
    std::unique_ptr<Handle> _handle;
    Context _context;
    std::vector<uint8_t> _fresh;
    std::set<std::string> _distinctObjects;
};

TEST_F(TestGpuMultiLaunchSaveLoad, RoundTripsAMultiLaunchPlanWithExactOutput)
{
    ASSERT_NO_FATAL_FAILURE(buildTwoStagePlan());
    if(IsSkipped())
    {
        return;
    }
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(saveTwoStagePlan(bytes));

    const auto loadHandle = hooks::makeHandle();
    hooks::LoadedPlan loaded(*loadHandle);
    ASSERT_NO_FATAL_FAILURE(loaded.load(bytes));

    for(int run = 0; run < 2; ++run)
    {
        EXPECT_EQ(runOnFixedInput(loaded.plan(), *loadHandle), _fresh) << "run " << run;
    }
}

TEST_F(TestGpuMultiLaunchSaveLoad, SavesAMultiLaunchPlanAsKindTwoWithOneStage)
{
    ASSERT_NO_FATAL_FAILURE(buildTwoStagePlan());
    if(IsSkipped())
    {
        return;
    }
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(saveTwoStagePlan(bytes));

    const auto header = hooks::headerOf(bytes);
    EXPECT_EQ(header.kind, serialization::IngestorPlanKind::MULTI_KERNEL_PLAN);
    EXPECT_EQ(header.formatMajor, serialization::INGESTOR_PLAN_FORMAT_MAJOR);
    EXPECT_EQ(header.formatMinor, serialization::INGESTOR_PLAN_MULTI_KERNEL_MINOR);

    serialization::IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    ASSERT_EQ(payload.stages.size(), 1U);
    const auto& stage = payload.stages[0];
    EXPECT_EQ(stage.slot, "");
    EXPECT_EQ(stage.dispatchSymbol, DISPATCH_SYMBOL_V1);
    ASSERT_EQ(stage.kernels.size(), 2U);
    // Launch order, not the sorted order of the names.
    EXPECT_EQ(stage.kernels[0].launchName, PRODUCE_LAUNCH);
    EXPECT_EQ(stage.kernels[0].symbol, PRODUCE_SYMBOL);
    EXPECT_EQ(stage.kernels[1].launchName, CONSUME_LAUNCH);
    EXPECT_EQ(stage.kernels[1].symbol, CONSUME_SYMBOL);
    EXPECT_EQ(payload.workspaceBytes, WORKSPACE_BYTES);
    EXPECT_EQ(payload.sharedWorkspaceBytes, 0U);
    EXPECT_TRUE(payload.runtimePassByValueUids.empty());
}

TEST_F(TestGpuMultiLaunchSaveLoad, StoresEachCodeObjectOnceAndAsManyAsTheShardHas)
{
    ASSERT_NO_FATAL_FAILURE(buildTwoStagePlan());
    if(IsSkipped())
    {
        return;
    }
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(saveTwoStagePlan(bytes));

    serialization::IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));

    // The packer put both launches in one code object, so the distinct objects of the kernel's
    // two launches are the objects the payload holds, each once.
    ASSERT_EQ(_distinctObjects.size(), 1U);
    std::set<std::string> stored;
    for(const auto& object : payload.codeObjects)
    {
        stored.insert(object.sha256);
    }
    EXPECT_EQ(payload.codeObjects.size(), _distinctObjects.size());
    EXPECT_EQ(stored, _distinctObjects);
    for(const auto& kernel : payload.stages.at(0).kernels)
    {
        EXPECT_EQ(kernel.codeObjectIndex, 0U) << kernel.launchName;
    }
}

TEST_F(TestGpuMultiLaunchSaveLoad, RefusesALaunchSetThatDiffersFromTheHandlersAtTheHook)
{
    ASSERT_NO_FATAL_FAILURE(buildTwoStagePlan());
    if(IsSkipped())
    {
        return;
    }
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(saveTwoStagePlan(bytes));
    serialization::IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));

    const auto loadHandle = hooks::makeHandle();
    {
        // The control: the saved plan loads, so the refusals below are the changed launches'.
        hooks::LoadedPlan loaded(*loadHandle);
        ASSERT_NO_FATAL_FAILURE(loaded.load(bytes));
    }

    payload.stages.at(0).kernels.at(1).launchName = "renamed";
    const auto renamed = serialization::encodeIngestorPlan(payload);
    const auto message = hooks::expectLoadRefusal(
        *loadHandle, renamed, IngestorPlanRefusal::INCOMPATIBLE, "'renamed'");
    EXPECT_NE(message.find(DISPATCH_SYMBOL_V1), std::string::npos) << message;
}

TEST_F(TestGpuMultiLaunchSaveLoad, RefusesADamagedCodeObjectInALaterLaunch)
{
    ASSERT_NO_FATAL_FAILURE(buildTwoStagePlan());
    if(IsSkipped())
    {
        return;
    }
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(saveTwoStagePlan(bytes));
    serialization::IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    ASSERT_EQ(payload.codeObjects.size(), 1U);

    // The shard's launches share one object, so the second launch gets an object of its own:
    // a copy that differs in one byte, with its own correct SHA-256. That payload is valid.
    payload.codeObjects.push_back(hooks::damagedCopyOf(payload.codeObjects.front()));
    payload.stages.at(0).kernels.at(1).codeObjectIndex = 1;
    auto encoded = serialization::encodeIngestorPlan(payload);

    // The stored bytes of the second object change after its SHA-256 was written, and the
    // body digest is written again, so only the check of that object can tell.
    ASSERT_NO_FATAL_FAILURE(
        hooks::damageStoredCodeObject(encoded, payload.codeObjects[1].codeObject));

    const auto loadHandle = hooks::makeHandle();
    hooks::expectLoadRefusal(
        *loadHandle, encoded, IngestorPlanRefusal::DAMAGED, "of code object 1 does not match");
}

// ---------------------------------------------------------------------------
// The kind 1 counterpart
// ---------------------------------------------------------------------------

TEST_F(TestGpuMultiLaunchSaveLoad, ASingleLaunchPlanIsSavedWithAFormat1Point0Header)
{
    namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;
    const packs::PackedPlanCase& packedCase = packs::POINTWISE_PACKED_PLAN_CASE;

    const auto scratch = hipdnn_test_sdk::utilities::claimScratchDirectory("singlelaunchsaveload");
    packs::PackedPlan built;
    ASSERT_NO_FATAL_FAILURE(
        packs::buildPackedPlan(packedCase, packedCase.roundTripGraph(), scratch.path(), built));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }
    packedCase.fillInputs(built.tensors);

    const auto saveHandle = hooks::makeHandle();
    std::vector<uint8_t> fresh;
    ASSERT_NO_FATAL_FAILURE(packs::runPlan(*built.plan, *saveHandle, built.tensors, fresh));
    packedCase.expectExactOutput(fresh);

    Context context;
    context.setPlan(std::move(built.plan));
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(hooks::save(*saveHandle, context, bytes));

    // A plan served by one launch is a kind 1 payload stamped 1.0, which a 1.0 reader reads.
    const auto header = hooks::headerOf(bytes);
    EXPECT_EQ(header.kind, serialization::IngestorPlanKind::SINGLE_KERNEL_PLAN);
    EXPECT_EQ(header.formatMajor, 1U);
    EXPECT_EQ(header.formatMinor, 0U);

    const auto loadHandle = hooks::makeHandle();
    hooks::LoadedPlan loaded(*loadHandle);
    ASSERT_NO_FATAL_FAILURE(loaded.load(bytes));
    std::vector<uint8_t> output;
    ASSERT_NO_FATAL_FAILURE(packs::runPlan(loaded.plan(), *loadHandle, built.tensors, output));
    EXPECT_EQ(output, fresh);
    packedCase.expectExactOutput(output);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
