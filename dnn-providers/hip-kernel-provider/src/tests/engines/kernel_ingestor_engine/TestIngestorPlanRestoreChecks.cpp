// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/**
 * @file TestIngestorPlanRestoreChecks.cpp
 * @brief restoreIngestorPlan: each load check refuses on its own, and the checks run in
 *        their documented order.
 *
 * No case needs a device. A fake resolver reports the device, and every case stops at or
 * before the launch-value check, so no case reaches the module load. The code bytes are
 * not a code object.
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
using hipdnn_plugin_sdk::ingestor::SymbolScope;
using serialization::expectIngestorPlanRefusal;
using serialization::IngestorPlanPayload;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* LOADED_ENGINE_NAME = "hipkernel:restore_test_engine";
constexpr const char* UNLOADED_ENGINE_NAME = "hipkernel:restore_test_absent_engine";
// The fake engine lookup reports another name for the id of this name.
constexpr const char* RENAMED_ENGINE_NAME = "hipkernel:restore_test_renamed_engine";
constexpr const char* OTHER_LOADED_NAME = "hipkernel:restore_test_other_engine";
constexpr const char* DEVICE_ARCH = "gfx90a:sramecc+:xnack-";
constexpr const char* FOREIGN_TARGET = "gfx942";
constexpr const char* UNKNOWN_ALIAS = "hipkernel.conv_fwd.dispatch.v9";

// Reports one device with a chosen arch, or no device at all.
class FakeDeviceResolver : public IDeviceResolver<Handle>
{
public:
    explicit FakeDeviceResolver(DeviceId ordinal = 0, std::string arch = DEVICE_ARCH)
        : _deviceId(ordinal)
    {
        _properties.gcnArchName = std::move(arch);
        _properties.warpSize = 64;
    }

    DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return _deviceId;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceId _deviceId;
    DeviceProperties _properties;
};

// The id of an engine name, in the hex form refusal messages use.
std::string engineIdHexOf(const std::string& engineName)
{
    return hipdnn_data_sdk::utilities::formatEngineIdHex(
        hipdnn_data_sdk::utilities::engineNameToId(engineName));
}

// Knows two loaded engines. The engine with the id of RENAMED_ENGINE_NAME has another name.
std::optional<std::string> fakeLoadedEngineName(int64_t engineId)
{
    if(engineId == hipdnn_data_sdk::utilities::engineNameToId(LOADED_ENGINE_NAME))
    {
        return std::string(LOADED_ENGINE_NAME);
    }
    if(engineId == hipdnn_data_sdk::utilities::engineNameToId(RENAMED_ENGINE_NAME))
    {
        return std::string(OTHER_LOADED_NAME);
    }
    return std::nullopt;
}

// The ten arguments the conv-forward pack launches with: three pointers and seven ints.
std::vector<KernelArgument> convFwdSignature()
{
    const KernelArgument buffer{"global_buffer", 8, 0, ""};
    const KernelArgument extent{"by_value", 4, 0, ""};
    return {buffer, buffer, buffer, extent, extent, extent, extent, extent, extent, extent};
}

// A conv-forward payload that passes every check before the module load. Its code bytes
// are not a code object.
IngestorPlanPayload validConvFwdPayload()
{
    IngestorPlanPayload payload;
    payload.engineName = LOADED_ENGINE_NAME;
    payload.kernelDescriptorId.fill(0x42);
    payload.workspaceBytes = 0;
    payload.dispatchSymbol = std::string(CONV_FWD_DISPATCH_SYMBOL_V1);
    payload.launchValues = convFwdLaunchValues(
        ConvFwdBinding{1, 2, 3}, ConvFwdExtents{1, 1, 3, 5, 1, 2, 2}, int64_t{64});
    payload.sourceKind = KernelSourceKind::KPACK;
    payload.symbol = "ConvFwd";
    payload.target = "gfx90a";
    payload.sha256 = std::string(64, 'a');
    payload.recordedSignature = convFwdSignature();
    payload.codeObject = std::vector<uint8_t>(256, 0xAB);
    payload.providerVersion = "test";
    return payload;
}

class TestIngestorPlanRestoreChecks : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Every pack's handler, under its unversioned name and its versioned alias. Once
        // per process, so other suites that register the packs do not collide with this.
        registerNativeIngestorSymbols();
    }

    std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> restore(const std::vector<uint8_t>& encoded,
                                                              const FakeDeviceResolver& resolver
                                                              = FakeDeviceResolver()) const
    {
        IngestorPlanRestoreEnvironment environment;
        environment.loadedIngestorEngineName = fakeLoadedEngineName;
        environment.deviceResolver = &resolver;
        return restoreIngestorPlan(encoded.data(), encoded.size(), _handle, environment);
    }

    void expectRefusal(const IngestorPlanPayload& payload,
                       hipdnnPluginStatus_t status,
                       const std::string& phrase) const
    {
        const auto encoded = serialization::encodeIngestorPlan(payload);
        expectIngestorPlanRefusal([&]() { static_cast<void>(restore(encoded)); }, status, phrase);
    }

private:
    Handle _handle;
};

TEST_F(TestIngestorPlanRestoreChecks, SurfacesAnUnreadableVersion)
{
    auto encoded = serialization::encodeIngestorPlan(validConvFwdPayload());
    // The format major is the little-endian u16 at offset 4.
    encoded[4] = 2;
    encoded[5] = 0;

    expectIngestorPlanRefusal([&]() { static_cast<void>(restore(encoded)); },
                              HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                              "payload format version 2.0");
    expectIngestorPlanRefusal(
        [&]() { static_cast<void>(restore(encoded)); }, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "1.0");
}

TEST_F(TestIngestorPlanRestoreChecks, RunsTheDecoderBeforeAnyLoadCheck)
{
    auto payload = validConvFwdPayload();
    payload.engineName = UNLOADED_ENGINE_NAME;
    payload.target = FOREIGN_TARGET;
    payload.dispatchSymbol = UNKNOWN_ALIAS;
    auto encoded = serialization::encodeIngestorPlan(payload);
    encoded.back() = static_cast<uint8_t>(encoded.back() ^ 0xFFU);

    expectIngestorPlanRefusal([&]() { static_cast<void>(restore(encoded)); },
                              HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                              "body digest mismatch");
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesAnEngineThatIsNotLoaded)
{
    auto payload = validConvFwdPayload();
    payload.engineName = UNLOADED_ENGINE_NAME;

    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  std::string("no ingestor engine named '") + UNLOADED_ENGINE_NAME + "'");
    expectRefusal(
        payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, engineIdHexOf(UNLOADED_ENGINE_NAME));
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "descriptor set");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "424242");
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesALoadedEngineWithAnotherName)
{
    auto payload = validConvFwdPayload();
    payload.engineName = RENAMED_ENGINE_NAME;

    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  std::string("is named '") + OTHER_LOADED_NAME + "', not '" + RENAMED_ENGINE_NAME
                      + "'");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, engineIdHexOf(RENAMED_ENGINE_NAME));
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "424242");
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesWhenTheHandleNamesNoDevice)
{
    const auto encoded = serialization::encodeIngestorPlan(validConvFwdPayload());
    const FakeDeviceResolver noDevice(hipdnn_plugin_sdk::ingestor::NO_DEVICE);

    serialization::expectPluginInternalError(
        [&]() { static_cast<void>(restore(encoded, noDevice)); },
        "cannot resolve the handle's device");
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesAnotherDeviceArch)
{
    auto payload = validConvFwdPayload();
    payload.target = FOREIGN_TARGET;
    payload.dispatchSymbol = UNKNOWN_ALIAS;

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "target 'gfx942'");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, DEVICE_ARCH);
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "kpack");
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesAnUnknownAlias)
{
    auto payload = validConvFwdPayload();
    payload.dispatchSymbol = UNKNOWN_ALIAS;

    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  std::string("no dispatch handler is registered under '") + UNKNOWN_ALIAS + "'");
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesAHandlerWithoutRestoreSupport)
{
    const std::string alias = "hipkernel.restore_test.no_restore.dispatch.v1";
    const packs::StubDispatchHandler handler;
    // Not committed: the alias is unregistered again when the scope ends.
    SymbolScope<Handle> scope;
    scope.add(alias, &handler);

    auto payload = validConvFwdPayload();
    payload.dispatchSymbol = alias;

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not support restoring");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, alias);
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesASignatureThePackDoesNotLaunch)
{
    auto payload = validConvFwdPayload();
    payload.recordedSignature.pop_back();
    ASSERT_EQ(payload.recordedSignature.size(), 9U);

    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  hipdnn_plugin_sdk::ingestor::describeKernelSignature(payload.recordedSignature));
    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  hipdnn_plugin_sdk::ingestor::describeKernelSignature(convFwdSignature()));
    expectRefusal(
        payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, std::string(CONV_FWD_DISPATCH_SYMBOL_V1));
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesAnUnversionedAlias)
{
    // The unversioned name stays registered for descriptors, so it resolves to the conv
    // handler. Its reader accepts only the versioned contract.
    auto payload = validConvFwdPayload();
    payload.dispatchSymbol = "hipkernel.conv_fwd.dispatch";

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "names no launch-value contract");
}

struct OrderCase
{
    std::string name;
    // Breaks two adjacent checks of a valid payload.
    std::function<void(IngestorPlanPayload&)> breakBoth;
    // A phrase only the earlier of the two checks writes.
    std::string phrase;
};

class TestIngestorPlanRestoreCheckOrder : public TestIngestorPlanRestoreChecks,
                                          public ::testing::WithParamInterface<OrderCase>
{
};

TEST_P(TestIngestorPlanRestoreCheckOrder, ChecksRunInTheDocumentedOrder)
{
    auto payload = validConvFwdPayload();
    GetParam().breakBoth(payload);

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, GetParam().phrase);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestIngestorPlanRestoreCheckOrder,
    ::testing::Values(
        OrderCase{"EngineBeforeTarget",
                  [](IngestorPlanPayload& payload) {
                      payload.engineName = UNLOADED_ENGINE_NAME;
                      payload.target = FOREIGN_TARGET;
                  },
                  "no ingestor engine named"},
        OrderCase{"TargetBeforeAlias",
                  [](IngestorPlanPayload& payload) {
                      payload.target = FOREIGN_TARGET;
                      payload.dispatchSymbol = UNKNOWN_ALIAS;
                  },
                  "cannot run"},
        OrderCase{"SignatureBeforeLaunchValues",
                  [](IngestorPlanPayload& payload) {
                      payload.recordedSignature.pop_back();
                      payload.launchValues["block_size"] = int64_t{0};
                  },
                  "do not match what this provider launches"},
        // The code bytes are not a code object, so a module load attempted first would
        // refuse with the driver's message.
        OrderCase{
            "LaunchValuesBeforeModuleLoad",
            [](IngestorPlanPayload& payload) { payload.launchValues["block_size"] = int64_t{0}; },
            "launch value 'block_size'"}),
    [](const ::testing::TestParamInfo<OrderCase>& info) { return info.param.name; });

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
