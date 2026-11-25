// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/RestoredSlotProgramPlan.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
#include "utilities/Digest.hpp"

/**
 * @file TestIngestorPlanMultiKernelRestore.cpp
 * @brief restoreIngestorSavedPlan for a kind 2 payload: what each stage hands its handler, the
 *        checks that refuse a payload this provider cannot restore, and the plan it returns.
 *
 * The payloads are structs, so no case needs the kind 2 codec. A stub handler restores them, so
 * no case needs a device or a code object that loads.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::DeviceId;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::IDeviceResolver;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using serialization::expectIngestorPlanRefusal;
using serialization::IngestorCodeObjectPayload;
using serialization::IngestorMultiKernelPayload;
using serialization::IngestorStageKernelPayload;
using serialization::IngestorStagePayload;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using packs::MultiKernelStubHandler;
using packs::RegisteredStubHandler;
using Config = MultiKernelStubHandler::Config;
using Launch = MultiKernelStubHandler::Launch;

constexpr const char* DEVICE_ARCH = "gfx90a:sramecc+:xnack-";
constexpr const char* DEVICE_TARGET = "gfx90a";
constexpr const char* FOREIGN_TARGET = "gfx942";
constexpr int DEVICE_ORDINAL = 3;

// Reports one device with a chosen arch.
class FakeDeviceResolver : public IDeviceResolver<Handle>
{
public:
    FakeDeviceResolver()
    {
        _properties.gcnArchName = DEVICE_ARCH;
        _properties.warpSize = 64;
    }

    DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return DEVICE_ORDINAL;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties;
};

// A buffer and `extents` by-value arguments.
std::vector<KernelArgument> signatureOf(size_t extents)
{
    std::vector<KernelArgument> signature{KernelArgument{"global_buffer", 8, 0, ""}};
    signature.insert(signature.end(), extents, KernelArgument{"by_value", 4, 0, ""});
    return signature;
}

IngestorCodeObjectPayload codeObject(uint8_t fill, const std::string& target = DEVICE_TARGET)
{
    IngestorCodeObjectPayload object;
    object.sourceKind = KernelSourceKind::KPACK;
    object.target = target;
    object.codeObject = std::vector<uint8_t>(64, fill);
    object.sha256 = utilities::sha256Hex(object.codeObject.data(), object.codeObject.size());
    return object;
}

IngestorStageKernelPayload stageKernel(const std::string& launchName,
                                       uint32_t codeObjectIndex = 0,
                                       std::vector<KernelArgument> signature = signatureOf(1))
{
    IngestorStageKernelPayload kernel;
    kernel.launchName = launchName;
    kernel.symbol = "symbol_" + (launchName.empty() ? std::string("main") : launchName);
    kernel.recordedSignature = std::move(signature);
    kernel.codeObjectIndex = codeObjectIndex;
    return kernel;
}

IngestorStagePayload stageOf(const std::string& slot,
                             const std::string& alias,
                             std::vector<IngestorStageKernelPayload> kernels,
                             uint8_t id = 0x61)
{
    IngestorStagePayload stage;
    stage.slot = slot;
    stage.kernelDescriptorId.fill(id);
    stage.dispatchSymbol = alias;
    stage.launchValues = {{"tag", slot}};
    stage.kernels = std::move(kernels);
    return stage;
}

// One stage that launches `produce` and `consume`, both from one code object.
IngestorMultiKernelPayload twoLaunchPayload(const std::string& alias)
{
    IngestorMultiKernelPayload payload;
    payload.stages.push_back(stageOf("", alias, {stageKernel("produce"), stageKernel("consume")}));
    payload.codeObjects.push_back(codeObject(0xAB));
    payload.workspaceBytes = 128;
    payload.providerVersion = "test";
    return payload;
}

Config twoLaunchConfig(const std::string& alias)
{
    Config config;
    config.alias = alias;
    config.optsIn = true;
    config.launchNames = std::set<std::string>{"produce", "consume"};
    return config;
}

class TestIngestorPlanMultiKernelRestore : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Every pack's handler is registered beside the stubs, as in a provider process.
        registerNativeIngestorSymbols();
    }

    std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
        restore(const IngestorMultiKernelPayload& payload) const
    {
        IngestorPlanRestoreEnvironment environment;
        environment.deviceResolver = &_resolver;
        return restoreIngestorSavedPlan(payload, _handle, environment);
    }

    std::string expectRefusal(const IngestorMultiKernelPayload& payload,
                              hipdnnPluginStatus_t status,
                              const std::string& phrase) const
    {
        return expectIngestorPlanRefusal(
            [&]() { static_cast<void>(restore(payload)); }, status, phrase);
    }

private:
    Handle _handle;
    FakeDeviceResolver _resolver;
};

// ---------------------------------------------------------------------------
// What a stage hands its handler
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelRestore,
       HandsEveryKernelOfAStageToRestoreLaunchesWithItsLaunchName)
{
    const RegisteredStubHandler stub(twoLaunchConfig("hipkernel.restore_test.two.dispatch.v1"));
    const auto payload = twoLaunchPayload(stub.handler.config().alias);

    ASSERT_NE(restore(payload), nullptr);

    ASSERT_EQ(stub.handler.received().size(), 1U);
    const auto& received = stub.handler.received().front();
    EXPECT_TRUE(received.several);
    EXPECT_EQ(received.deviceOrdinal, DEVICE_ORDINAL);
    EXPECT_EQ(received.inputs.dispatchSymbol, payload.stages.front().dispatchSymbol);
    EXPECT_EQ(received.inputs.values, payload.stages.front().launchValues);
    ASSERT_EQ(received.codes.size(), 2U);
    EXPECT_EQ(received.codes[0].launchName, "produce");
    EXPECT_EQ(received.codes[1].launchName, "consume");
    EXPECT_EQ(received.codes[0].symbol, "symbol_produce");
    EXPECT_EQ(received.codes[1].symbol, "symbol_consume");
    for(const auto& code : received.codes)
    {
        EXPECT_EQ(code.kernelId, payload.stages.front().kernelDescriptorId);
        EXPECT_EQ(code.sourceKind, KernelSourceKind::KPACK);
        EXPECT_EQ(code.target, DEVICE_TARGET);
        EXPECT_EQ(code.sha256, payload.codeObjects.front().sha256);
    }
}

TEST_F(TestIngestorPlanMultiKernelRestore, HandsASharedCodeObjectToEveryKernelThatUsesIt)
{
    // One code object serves both kernels of the first stage and the kernel of the second.
    // The last use takes the bytes and every earlier one gets a copy, so each kernel holds
    // the whole object.
    const Config first = twoLaunchConfig("hipkernel.restore_test.shared_a.dispatch.v1");
    Config second;
    second.alias = "hipkernel.restore_test.shared_b.dispatch.v1";
    const RegisteredStubHandler firstStub(first);
    const RegisteredStubHandler secondStub(second);

    auto payload = twoLaunchPayload(first.alias);
    payload.stages.push_back(stageOf("later", second.alias, {stageKernel("")}, 0x62));

    ASSERT_NE(restore(payload), nullptr);

    const auto& object = payload.codeObjects.front();
    ASSERT_EQ(firstStub.handler.received().size(), 1U);
    ASSERT_EQ(secondStub.handler.received().size(), 1U);
    for(const auto& code : firstStub.handler.received().front().codes)
    {
        EXPECT_EQ(code.codeObject, object.codeObject);
    }
    ASSERT_EQ(secondStub.handler.received().front().codes.size(), 1U);
    EXPECT_EQ(secondStub.handler.received().front().codes.front().codeObject, object.codeObject);
}

TEST_F(TestIngestorPlanMultiKernelRestore,
       UsesRestoreLaunchForAHandlerThatDoesNotOptInAndRestoreLaunchesForOneThatDoes)
{
    Config plain;
    plain.alias = "hipkernel.restore_test.plain.dispatch.v1";
    Config optedIn;
    optedIn.alias = "hipkernel.restore_test.opted_in.dispatch.v1";
    optedIn.optsIn = true;
    const RegisteredStubHandler plainStub(plain);
    const RegisteredStubHandler optedInStub(optedIn);

    IngestorMultiKernelPayload payload;
    payload.stages.push_back(stageOf("plain", plain.alias, {stageKernel("")}, 0x63));
    payload.stages.push_back(stageOf("opted_in", optedIn.alias, {stageKernel("")}, 0x64));
    payload.codeObjects.push_back(codeObject(0xAB));
    payload.providerVersion = "test";

    ASSERT_NE(restore(payload), nullptr);

    ASSERT_EQ(plainStub.handler.received().size(), 1U);
    EXPECT_FALSE(plainStub.handler.received().front().several);
    ASSERT_EQ(optedInStub.handler.received().size(), 1U);
    EXPECT_TRUE(optedInStub.handler.received().front().several);
    EXPECT_EQ(optedInStub.handler.received().front().codes.size(), 1U);
}

// ---------------------------------------------------------------------------
// The plan it returns
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelRestore, RestoresEveryKindTwoPayloadAsARestoredSlotProgramPlan)
{
    // An unslotted payload (one stage, no slot) restores as a slot program too.
    const RegisteredStubHandler stub(twoLaunchConfig("hipkernel.restore_test.plan.dispatch.v1"));
    auto payload = twoLaunchPayload(stub.handler.config().alias);
    payload.workspaceBytes = 777;

    const auto plan = restore(payload);

    ASSERT_NE(plan, nullptr);
    EXPECT_NE(dynamic_cast<const RestoredSlotProgramPlan<Handle>*>(plan.get()), nullptr);
    EXPECT_EQ(plan->getWorkspaceSize(Handle{}), 777U);
    EXPECT_EQ(plan->saveablePlan(), nullptr);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

// A refusal by the handler of stage N reaches the caller as an incompatible plan, and no later
// stage's handler is asked to restore.
TEST_F(TestIngestorPlanMultiKernelRestore, RefusesALaunchSetTheHandlerDoesNotLaunch)
{
    const RegisteredStubHandler stub(twoLaunchConfig("hipkernel.restore_test.set.dispatch.v1"));
    Config later;
    later.alias = "hipkernel.restore_test.set_later.dispatch.v1";
    const RegisteredStubHandler laterStub(later);
    const auto alias = stub.handler.config().alias;

    auto missing = twoLaunchPayload(alias);
    missing.stages.push_back(stageOf("later", later.alias, {stageKernel("")}, 0x6A));
    missing.stages.front().kernels.pop_back();
    expectRefusal(
        missing, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not launch this set of launches");
    EXPECT_EQ(stub.handler.received().size(), 1U);
    EXPECT_TRUE(laterStub.handler.received().empty());

    auto extra = twoLaunchPayload(alias);
    extra.stages.push_back(stageOf("later", later.alias, {stageKernel("")}, 0x6A));
    extra.stages.front().kernels.push_back(stageKernel("extra"));
    expectRefusal(
        extra, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not launch this set of launches");
    EXPECT_EQ(stub.handler.received().size(), 2U);
    EXPECT_TRUE(laterStub.handler.received().empty());
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesAStageWhoseHandlerRefusesItsLaunchValues)
{
    auto config = twoLaunchConfig("hipkernel.restore_test.values.dispatch.v1");
    config.rejectsLaunchValues = true;
    const RegisteredStubHandler stub(config);
    Config later;
    later.alias = "hipkernel.restore_test.values_later.dispatch.v1";
    const RegisteredStubHandler laterStub(later);

    auto payload = twoLaunchPayload(config.alias);
    payload.stages.push_back(stageOf("later", later.alias, {stageKernel("")}, 0x6A));
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "refuses its launch values");
    EXPECT_EQ(stub.handler.received().size(), 1U);
    EXPECT_TRUE(laterStub.handler.received().empty());
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesASignatureTheSecondKernelDoesNotLaunch)
{
    auto config = twoLaunchConfig("hipkernel.restore_test.signature.dispatch.v1");
    config.signatures = {{"produce", signatureOf(1)}, {"consume", signatureOf(1)}};
    const RegisteredStubHandler stub(config);

    auto payload = twoLaunchPayload(config.alias);
    // The first kernel agrees with the handler. The second records one argument fewer.
    payload.stages.front().kernels[1].recordedSignature = signatureOf(0);

    const std::string message
        = expectRefusal(payload,
                        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                        "the saved kernel's arguments do not match what this provider launches");
    EXPECT_NE(message.find("symbol 'symbol_consume'"), std::string::npos) << message;
    EXPECT_EQ(message.find("symbol 'symbol_produce'"), std::string::npos) << message;
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesALaterCodeObjectForAnotherDeviceArch)
{
    // The first code object runs on the device. The second, used by the second stage's
    // kernel, does not. The refusal comes before any handler restores anything.
    Config first;
    first.alias = "hipkernel.restore_test.arch_a.dispatch.v1";
    Config second;
    second.alias = "hipkernel.restore_test.arch_b.dispatch.v1";
    const RegisteredStubHandler firstStub(first);
    const RegisteredStubHandler secondStub(second);

    IngestorMultiKernelPayload payload;
    payload.stages.push_back(stageOf("a", first.alias, {stageKernel("", 0)}, 0x65));
    payload.stages.push_back(stageOf("b", second.alias, {stageKernel("", 1)}, 0x66));
    payload.codeObjects.push_back(codeObject(0xAB));
    payload.codeObjects.push_back(codeObject(0xCD, FOREIGN_TARGET));
    payload.providerVersion = "test";

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "saved for 'gfx942'");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, DEVICE_ARCH);
    EXPECT_TRUE(firstStub.handler.received().empty());
    EXPECT_TRUE(secondStub.handler.received().empty());
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesAStageWithAnUnknownDispatchAlias)
{
    const RegisteredStubHandler known(twoLaunchConfig("hipkernel.restore_test.known.dispatch.v1"));
    auto payload = twoLaunchPayload(known.handler.config().alias);
    const std::string unknown = "hipkernel.restore_test.unknown.dispatch.v1";
    payload.stages.push_back(stageOf("second", unknown, {stageKernel("")}, 0x67));

    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "no dispatch handler is registered under '" + unknown + "'");
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesAStageWhoseHandlerDoesNotRestore)
{
    auto config = twoLaunchConfig("hipkernel.restore_test.no_restore.dispatch.v1");
    config.restores = false;
    const RegisteredStubHandler stub(config);

    expectRefusal(twoLaunchPayload(config.alias),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "does not support restoring");
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesAMultiKernelStageForAHandlerThatDoesNotOptIn)
{
    auto config = twoLaunchConfig("hipkernel.restore_test.not_opted_in.dispatch.v1");
    config.optsIn = false;
    const RegisteredStubHandler stub(config);

    expectRefusal(twoLaunchPayload(config.alias),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "does not restore several launches");
    // The handler was not asked to restore a launch set it never said it handles.
    EXPECT_TRUE(stub.handler.received().empty());
}

// ---------------------------------------------------------------------------
// Runtime pass-by-value scalars
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelRestore,
       RefusesAPlanWhoseHandlersBindOtherPassByValueUidsThanTheListedOnes)
{
    auto config = twoLaunchConfig("hipkernel.restore_test.uids.dispatch.v1");
    config.uids = {5, 9};
    const RegisteredStubHandler stub(config);

    auto payload = twoLaunchPayload(config.alias);
    payload.runtimePassByValueUids = {5};
    const std::string extra = expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "{5, 9}");
    EXPECT_NE(extra.find("{5}"), std::string::npos) << extra;

    // The same payload restores when the list is the handler's.
    payload.runtimePassByValueUids = {5, 9};
    EXPECT_NE(restore(payload), nullptr);

    auto unbound = twoLaunchConfig("hipkernel.restore_test.uids_none.dispatch.v1");
    const RegisteredStubHandler unboundStub(unbound);
    auto listed = twoLaunchPayload(unbound.alias);
    listed.runtimePassByValueUids = {5};
    const std::string missing = expectRefusal(listed, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "{}");
    EXPECT_NE(missing.find("{5}"), std::string::npos) << missing;
}

TEST_F(TestIngestorPlanMultiKernelRestore, TakesTheUnionOfTheUidsOfEveryStage)
{
    Config first;
    first.alias = "hipkernel.restore_test.union_a.dispatch.v1";
    first.uids = {9};
    Config second;
    second.alias = "hipkernel.restore_test.union_b.dispatch.v1";
    second.uids = {5, 9};
    const RegisteredStubHandler firstStub(first);
    const RegisteredStubHandler secondStub(second);

    IngestorMultiKernelPayload payload;
    payload.stages.push_back(stageOf("a", first.alias, {stageKernel("")}, 0x68));
    payload.stages.push_back(stageOf("b", second.alias, {stageKernel("")}, 0x69));
    payload.codeObjects.push_back(codeObject(0xAB));
    payload.providerVersion = "test";

    payload.runtimePassByValueUids = {5, 9};
    EXPECT_NE(restore(payload), nullptr);

    // One stage's UIDs alone are not the plan's.
    payload.runtimePassByValueUids = {9};
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "{5, 9}");
}

TEST_F(TestIngestorPlanMultiKernelRestore, KeepsAnEarlierStagesUidsThatNoLaterStageBinds)
{
    Config first;
    first.alias = "hipkernel.restore_test.union_first.dispatch.v1";
    first.uids = {5, 9};
    Config second;
    second.alias = "hipkernel.restore_test.union_second.dispatch.v1";
    second.uids = {9};
    const RegisteredStubHandler firstStub(first);
    const RegisteredStubHandler secondStub(second);

    IngestorMultiKernelPayload payload;
    payload.stages.push_back(stageOf("a", first.alias, {stageKernel("")}, 0x6B));
    payload.stages.push_back(stageOf("b", second.alias, {stageKernel("")}, 0x6C));
    payload.codeObjects.push_back(codeObject(0xAB));
    payload.providerVersion = "test";

    payload.runtimePassByValueUids = {5, 9};
    EXPECT_NE(restore(payload), nullptr);

    // The last stage's UIDs alone are not the plan's.
    payload.runtimePassByValueUids = {9};
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "{5, 9}");
}

TEST_F(TestIngestorPlanMultiKernelRestore,
       ContributesNoUidForAPreparedDispatchThatIsNotAnIngestorPreparedDispatch)
{
    auto config = twoLaunchConfig("hipkernel.restore_test.foreign.dispatch.v1");
    config.restoresForeignPrepared = true;
    config.uids = {5};
    const RegisteredStubHandler stub(config);

    auto payload = twoLaunchPayload(config.alias);
    payload.runtimePassByValueUids = {5};
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "{}");

    payload.runtimePassByValueUids.clear();
    EXPECT_NE(restore(payload), nullptr);
}

// ---------------------------------------------------------------------------
// Damage the payload structs can carry past the codec
// ---------------------------------------------------------------------------

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesACodeObjectIndexOutOfRange)
{
    const RegisteredStubHandler stub(twoLaunchConfig("hipkernel.restore_test.range.dispatch.v1"));
    auto payload = twoLaunchPayload(stub.handler.config().alias);
    payload.stages.front().kernels[1].codeObjectIndex = 1;

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "code object");
    EXPECT_TRUE(stub.handler.received().empty());
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesAPayloadWithNoStages)
{
    IngestorMultiKernelPayload payload;
    payload.codeObjects.push_back(codeObject(0xAB));
    payload.providerVersion = "test";

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "has no stages");
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesASharedRegionLargerThanTheWorkspace)
{
    const RegisteredStubHandler stub(twoLaunchConfig("hipkernel.restore_test.shared.dispatch.v1"));
    auto payload = twoLaunchPayload(stub.handler.config().alias);
    payload.workspaceBytes = 128;
    payload.sharedWorkspaceBytes = 256;

    expectRefusal(
        payload, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "is larger than the plan's workspace");
    EXPECT_TRUE(stub.handler.received().empty());
}

TEST_F(TestIngestorPlanMultiKernelRestore, RefusesAStageThatLaunchesNoKernel)
{
    const RegisteredStubHandler stub(twoLaunchConfig("hipkernel.restore_test.empty.dispatch.v1"));
    auto payload = twoLaunchPayload(stub.handler.config().alias);
    payload.stages.front().kernels.clear();

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "launches no kernel");
    EXPECT_TRUE(stub.handler.received().empty());
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
