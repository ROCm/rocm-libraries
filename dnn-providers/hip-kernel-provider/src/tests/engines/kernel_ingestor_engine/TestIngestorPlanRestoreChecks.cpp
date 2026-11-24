// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

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
#include "engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
#include "utilities/Digest.hpp"

/**
 * @file TestIngestorPlanRestoreChecks.cpp
 * @brief restoreIngestorPlan: each load check refuses on its own, and the checks run in
 *        their documented order.
 *
 * No case needs a device. A fake resolver reports the device, or the case stops before the
 * device query. Every case stops at or before the launch-value check, so no case reaches the
 * module load. The code bytes are not a code object.
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
using testing::launchValuesUnder;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

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
    payload.kernelDescriptorId.fill(0x42);
    payload.workspaceBytes = 0;
    payload.dispatchSymbol = std::string(CONV_FWD_DISPATCH_SYMBOL_V1);
    payload.launchValues = convFwdLaunchValues(
        ConvFwdBinding{1, 2, 3}, ConvFwdExtents{1, 1, 3, 5, 1, 2, 2}, int64_t{64});
    payload.sourceKind = KernelSourceKind::KPACK;
    payload.symbol = "ConvFwd";
    payload.target = "gfx90a";
    payload.recordedSignature = convFwdSignature();
    payload.codeObject = std::vector<uint8_t>(256, 0xAB);
    payload.sha256 = utilities::sha256Hex(payload.codeObject.data(), payload.codeObject.size());
    payload.providerVersion = "test";
    return payload;
}

// The gfx950 dense-attention arguments, named as the engine declares them: v1 stops at
// seqlen_kv, v2 adds the two head counts.
std::vector<KernelArgument> gfx950AttentionDenseSignature(bool withHeads)
{
    std::vector<KernelArgument> signature{KernelArgument{"global_buffer", 8, 0, "q_ptr"},
                                          KernelArgument{"global_buffer", 8, 8, "k_ptr"},
                                          KernelArgument{"global_buffer", 8, 16, "v_ptr"},
                                          KernelArgument{"global_buffer", 8, 24, "o_ptr"},
                                          KernelArgument{"by_value", 4, 32, "scale"},
                                          KernelArgument{"by_value", 4, 36, "batch"},
                                          KernelArgument{"by_value", 4, 40, "seqlen_q"},
                                          KernelArgument{"by_value", 4, 44, "seqlen_kv"}};
    if(withHeads)
    {
        signature.push_back(KernelArgument{"by_value", 4, 48, "num_query_heads"});
        signature.push_back(KernelArgument{"by_value", 4, 52, "num_kv_heads"});
    }
    return signature;
}

// A gfx950 dense-attention payload under `dispatchSymbol` that passes every check before the
// module load when its signature is the one that contract launches.
IngestorPlanPayload gfx950AttentionDensePayload(std::string_view dispatchSymbol, bool withHeads)
{
    const AttentionDenseBinding binding{1, 2, 3, 4, 1, 0, 0.125F};
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::BFLOAT16;

    IngestorPlanPayload payload = validConvFwdPayload();
    payload.dispatchSymbol = std::string(dispatchSymbol);
    payload.launchValues = launchValuesUnder(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2,
                                             gfx950AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256});
    payload.symbol = "attention_dense";
    payload.recordedSignature = gfx950AttentionDenseSignature(withHeads);
    return payload;
}

// A gfx950 dense-attention bias plan under v3, the contract of a bias kernel: the v2 values
// plus a [B, 1, Sq, Skv] bias. Its recorded signature is v2's ten arguments, then the bias
// pointer and its three strides when `withBiasTail`.
IngestorPlanPayload gfx950AttentionDenseBiasPayload(bool withBiasTail)
{
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 0, 0.125F};
    binding.bias = AttentionDenseBias{5, {512 * 512, 0, 512}};
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::BFLOAT16;

    IngestorPlanPayload payload = validConvFwdPayload();
    payload.dispatchSymbol = std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3);
    payload.launchValues = launchValuesUnder(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3,
                                             gfx950AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256});
    payload.symbol = "attention_dense";
    payload.recordedSignature = gfx950AttentionDenseSignature(/*withHeads=*/true);
    if(withBiasTail)
    {
        payload.recordedSignature.push_back(
            KernelArgument{"global_buffer", 8, 56, "additive_bias_ptr"});
        payload.recordedSignature.push_back(
            KernelArgument{"by_value", 4, 64, "additive_bias_batch_stride"});
        payload.recordedSignature.push_back(
            KernelArgument{"by_value", 4, 68, "additive_bias_head_stride"});
        payload.recordedSignature.push_back(
            KernelArgument{"by_value", 4, 72, "additive_bias_sq_stride"});
    }
    return payload;
}

// A gfx942 dense-attention payload under its v1 contract, saved with stats uid `stats`
// (ATTENTION_DENSE_NO_STATS for none), that passes every check before the module load when
// `withHeads` is true and `writesLse` says whether `stats` is bound: gfx942 launches the
// same ten arguments as gfx950's v2, and lse_ptr after them to a kernel that writes LSE.
IngestorPlanPayload gfx942AttentionDensePayload(bool withHeads,
                                                int64_t stats = ATTENTION_DENSE_NO_STATS,
                                                bool writesLse = false)
{
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 128, 0.125F};
    binding.stats = stats;
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::HALF;

    IngestorPlanPayload payload = validConvFwdPayload();
    payload.dispatchSymbol = std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1);
    payload.launchValues = launchValuesUnder(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
                                             gfx942AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256});
    payload.symbol = "attention_dense";
    payload.recordedSignature = gfx950AttentionDenseSignature(withHeads);
    if(writesLse)
    {
        payload.recordedSignature.push_back(KernelArgument{"global_buffer", 8, 56, "lse_ptr"});
    }
    return payload;
}

// A gfx942 dense-attention bias plan under v2, the contract of a bias kernel: v1's values plus
// a [B, 1, Sq, Skv] bias, saved with stats uid `stats` (ATTENTION_DENSE_NO_STATS for none).
// Its recorded signature is v1's ten arguments, lse_ptr when `writesLse`, then the bias
// pointer and its three strides when `withBiasTail`: last, after lse_ptr.
IngestorPlanPayload gfx942AttentionDenseBiasPayload(bool withBiasTail,
                                                    int64_t stats = ATTENTION_DENSE_NO_STATS,
                                                    bool writesLse = false)
{
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 128, 0.125F};
    binding.stats = stats;
    binding.bias = AttentionDenseBias{5, {512 * 512, 0, 512}};
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::HALF;

    IngestorPlanPayload payload = validConvFwdPayload();
    payload.dispatchSymbol = std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2);
    payload.launchValues = launchValuesUnder(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2,
                                             gfx942AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256});
    payload.symbol = "attention_dense";
    payload.recordedSignature = gfx950AttentionDenseSignature(/*withHeads=*/true);
    uint32_t offset = 56;
    if(writesLse)
    {
        payload.recordedSignature.push_back(KernelArgument{"global_buffer", 8, offset, "lse_ptr"});
        offset += 8;
    }
    if(withBiasTail)
    {
        payload.recordedSignature.push_back(
            KernelArgument{"global_buffer", 8, offset, "additive_bias_ptr"});
        payload.recordedSignature.push_back(
            KernelArgument{"by_value", 4, offset + 8, "additive_bias_batch_stride"});
        payload.recordedSignature.push_back(
            KernelArgument{"by_value", 4, offset + 12, "additive_bias_head_stride"});
        payload.recordedSignature.push_back(
            KernelArgument{"by_value", 4, offset + 16, "additive_bias_sq_stride"});
    }
    return payload;
}

/// A gfx942 payload saved on the persistent grid at @p savedDecode (304 workgroups) whose code
/// object records the pairs of @p recordedDecode, or none for an empty one, and lse_ptr
/// after them when @p lse.
IngestorPlanPayload
    gfx942PersistentPayload(attention_dense::AttentionDensePersistDecode savedDecode,
                            const std::vector<std::string>& recordedDivisors,
                            bool lse = false)
{
    attention_dense::AttentionDenseGrid grid;
    grid.order = attention_dense::AttentionDenseGridOrder::PERSISTENT;
    grid.decode = savedDecode;
    grid.numPersistent = 304;
    auto payload = gfx942AttentionDensePayload(/*withHeads=*/true, lse ? int64_t{7} : int64_t{-1});
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 128, 0.125F};
    binding.stats = lse ? int64_t{7} : ATTENTION_DENSE_NO_STATS;
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::HALF;
    payload.launchValues = launchValuesUnder(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
                                             gfx942AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256},
                                             grid);
    uint32_t offset = 56;
    for(const auto& divisor : recordedDivisors)
    {
        for(const char* part : {"_magic", "_shift"})
        {
            payload.recordedSignature.push_back(
                KernelArgument{"by_value", 4, offset, divisor + part});
            offset += 4;
        }
    }
    if(lse)
    {
        payload.recordedSignature.push_back(
            KernelArgument{"global_buffer", 8, (offset + 7) / 8 * 8, "lse_ptr"});
    }
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
    payload.target = FOREIGN_TARGET;
    payload.dispatchSymbol = UNKNOWN_ALIAS;
    auto encoded = serialization::encodeIngestorPlan(payload);
    encoded.back() = static_cast<uint8_t>(encoded.back() ^ 0xFFU);

    expectIngestorPlanRefusal([&]() { static_cast<void>(restore(encoded)); },
                              HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                              "body digest mismatch");
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

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "saved for 'gfx942'");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, DEVICE_ARCH);
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "kpack");
}

// The unknown alias stops the restore at the next check, before any module load.
TEST_F(TestIngestorPlanRestoreChecks, AdmitsAPlanSavedOnTheSameBaseArch)
{
    auto payload = validConvFwdPayload();
    payload.target = "gfx1151";
    payload.dispatchSymbol = UNKNOWN_ALIAS;
    const auto encoded = serialization::encodeIngestorPlan(payload);
    const FakeDeviceResolver sameBaseArch(0, "gfx1151:xnack-");

    const std::string what
        = expectIngestorPlanRefusal([&]() { static_cast<void>(restore(encoded, sameBaseArch)); },
                                    HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                    "no dispatch handler is registered under");
    EXPECT_EQ(what.find("saved for '"), std::string::npos) << what;
    EXPECT_EQ(what.find("cannot run"), std::string::npos) << what;
}

TEST_F(TestIngestorPlanRestoreChecks, RefusesAnUnknownAlias)
{
    auto payload = validConvFwdPayload();
    payload.dispatchSymbol = UNKNOWN_ALIAS;

    expectRefusal(payload,
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  std::string("no dispatch handler is registered under '") + UNKNOWN_ALIAS + "'");
    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "424242");
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

// Each gfx950 dense-attention contract restores only the code object it launches: v1 the
// 8-argument kernels that bake the head counts, v2 the 10-argument kernels that read them,
// and v3 the 14-argument bias kernels.
TEST_F(TestIngestorPlanRestoreChecks, Gfx950AttentionDenseContractsRefuseTheOtherSignature)
{
    expectRefusal(gfx950AttentionDensePayload(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1, true),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
    expectRefusal(gfx950AttentionDensePayload(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2, false),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
    expectRefusal(gfx950AttentionDenseBiasPayload(/*withBiasTail=*/false),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
}

// The matching signature passes: the plan reaches the launch-value check, which a broken
// block_m then refuses. No case reaches the module load, so no device is needed.
TEST_F(TestIngestorPlanRestoreChecks, Gfx950AttentionDenseContractsAcceptTheirOwnSignature)
{
    for(const auto& [dispatchSymbol, withHeads] :
        {std::pair{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1, false},
         std::pair{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2, true}})
    {
        SCOPED_TRACE(std::string(dispatchSymbol));
        auto payload = gfx950AttentionDensePayload(dispatchSymbol, withHeads);
        payload.launchValues["block_m"] = int64_t{64};
        expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");
    }

    auto bias = gfx950AttentionDenseBiasPayload(/*withBiasTail=*/true);
    bias.launchValues["block_m"] = int64_t{64};
    expectRefusal(bias, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");
}

// gfx942's v1 contract restores only the 10-argument code object it launches.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseContractRefusesTheEightArgumentSignature)
{
    expectRefusal(gfx942AttentionDensePayload(/*withHeads=*/false),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
}

// The matching signature passes: the plan reaches the launch-value check, which a block_m
// gfx942 does not build, or a window that leaves query rows no key in a plan with a stats
// output, then refuses. No case reaches the module load, so no device is needed.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseContractAcceptsItsOwnSignature)
{
    auto unbuiltTile = gfx942AttentionDensePayload(/*withHeads=*/true);
    unbuiltTile.launchValues["block_m"] = int64_t{48};
    expectRefusal(unbuiltTile, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");

    auto keylessRows
        = gfx942AttentionDensePayload(/*withHeads=*/true, /*stats=*/7, /*writesLse=*/true);
    keylessRows.launchValues["seqlen_kv"] = int64_t{256};
    keylessRows.launchValues["sliding_window"] = int64_t{256};
    expectRefusal(
        keylessRows, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'sliding_window'");
}

// gfx942's bias contract, v2, restores only the code object that takes the bias tail last,
// after lse_ptr; its plain contract, v1, refuses the tail.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseBiasContractRefusesTheOtherSignatures)
{
    constexpr const char* MISMATCH
        = "the saved kernel's arguments do not match what this provider launches";
    expectRefusal(gfx942AttentionDenseBiasPayload(/*withBiasTail=*/false),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  MISMATCH);
    expectRefusal(gfx942AttentionDenseBiasPayload(
                      /*withBiasTail=*/false, int64_t{7}, /*writesLse=*/true),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  MISMATCH);
    // A bias plan whose code object records lse_ptr after the tail, the order a kernel
    // never takes, or whose stats uid disagrees with the lse_ptr it records.
    auto tailBeforeLse = gfx942AttentionDenseBiasPayload(/*withBiasTail=*/true);
    tailBeforeLse.recordedSignature.push_back(KernelArgument{"global_buffer", 8, 80, "lse_ptr"});
    tailBeforeLse.launchValues["stats.uid"] = int64_t{7};
    expectRefusal(tailBeforeLse, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, MISMATCH);
    expectRefusal(gfx942AttentionDenseBiasPayload(
                      /*withBiasTail=*/true, ATTENTION_DENSE_NO_STATS, /*writesLse=*/true),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  MISMATCH);

    // v1 holds no bias: a plan relabelled v1 that records the tail is refused too, its
    // code object taking four arguments v1 never passes.
    auto biasUnderV1 = gfx942AttentionDenseBiasPayload(/*withBiasTail=*/true);
    biasUnderV1.dispatchSymbol = std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1);
    expectRefusal(biasUnderV1, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, MISMATCH);
}

// The matching signature passes: the plan reaches the launch-value check, which a block_m
// gfx942 does not build then refuses. No case loads the code object, so no device is
// needed.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseBiasContractAcceptsItsOwnSignature)
{
    for(const bool writesLse : {false, true})
    {
        SCOPED_TRACE(writesLse);
        auto payload = gfx942AttentionDenseBiasPayload(
            /*withBiasTail=*/true, writesLse ? int64_t{7} : ATTENTION_DENSE_NO_STATS, writesLse);
        payload.launchValues["block_m"] = int64_t{48};
        expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");
    }
}

// The saved stats uid selects the argument list before anything is checked: a plan with a
// stats output restores the 11-argument code object that writes the LSE, reaching the
// launch-value check, which a broken block_m then refuses.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseRestoresTheLseFormOfAPlanWithStats)
{
    for(const int64_t stats : {int64_t{0}, int64_t{7}})
    {
        SCOPED_TRACE(stats);
        auto payload = gfx942AttentionDensePayload(/*withHeads=*/true, stats, /*writesLse=*/true);
        payload.launchValues["block_m"] = int64_t{48};
        expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");
    }
}

// A kernel whose arguments disagree with the saved stats uid is refused by the signature
// check: an LSE kernel saved without a stats output would store through an argument the
// launch never passes, and a non-LSE kernel saved with one would never write it.
TEST_F(TestIngestorPlanRestoreChecks,
       Gfx942AttentionDenseRefusesAKernelThatDisagreesWithItsStatsUid)
{
    expectRefusal(gfx942AttentionDensePayload(/*withHeads=*/true, int64_t{7}, /*writesLse=*/false),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
    expectRefusal(gfx942AttentionDensePayload(
                      /*withHeads=*/true, ATTENTION_DENSE_NO_STATS, /*writesLse=*/true),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "the saved kernel's arguments do not match what this provider launches");
}

// The selector reads the uid leniently and the reader strictly: a uid below -1 selects the
// LSE form and is then refused as a value; a missing one selects the plain form and is then
// refused as missing.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseRefusesAStatsUidItNeverSaves)
{
    expectRefusal(gfx942AttentionDensePayload(/*withHeads=*/true, int64_t{-2}, /*writesLse=*/true),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  "launch value 'stats.uid'");

    auto missing = gfx942AttentionDensePayload(/*withHeads=*/true);
    missing.launchValues.erase("stats.uid");
    expectRefusal(missing, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "needs launch value 'stats.uid'");
}

// ---------------------------------------------------------------------------
// The strided contracts (gfx950 v4, gfx942 v3) against the legacy ones
//
// Each strided contract restores only a code object that takes the twelve strides last, and
// each legacy contract only one that does not, so a plan never launches a kernel with the
// wrong argument count.
// ---------------------------------------------------------------------------

/// Appends the twelve strides to @p signature as a strided kernel records them: last, packed
/// after whatever precedes them.
void appendRecordedStrides(std::vector<KernelArgument>& signature)
{
    for(const char* tensor : {"q", "k", "v", "o"})
    {
        for(const char* axis : {"batch", "head", "token"})
        {
            const uint32_t end = signature.back().offset + signature.back().size;
            signature.push_back(
                KernelArgument{"by_value", 4, end, std::string("stride_") + tensor + "_" + axis});
        }
    }
}

/// The packed strides of the restore fixtures' shape: batch 2, seqlen 512, 8 query heads
/// over 4 kv heads of 128 elements.
AttentionDenseStrides fixtureStrides()
{
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    return attention_dense::packedI32LaunchStrides(problem).value();
}

/// A gfx950 payload under the strided contract that passes every check before the module
/// load when @p recordsStrides: its code object records v2's ten arguments, then the strides
/// unless told otherwise. Saved as prepare() saves a plan for a grid kernel, whose body reads
/// the strides, unless @p readsStrides says otherwise (a gqa_pair or bias body).
IngestorPlanPayload gfx950AttentionDenseStridedPayload(bool recordsStrides,
                                                       bool readsStrides = true)
{
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 0, 0.125F};
    binding.strides = fixtureStrides();
    binding.blockN = 64;
    binding.readsStrides = readsStrides;
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::BFLOAT16;

    auto payload = gfx950AttentionDensePayload(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED,
                                               /*withHeads=*/true);
    payload.launchValues = launchValuesUnder(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED,
                                             gfx950AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256});
    if(recordsStrides)
    {
        appendRecordedStrides(payload.recordedSignature);
    }
    return payload;
}

/// The same for gfx942, saved with stats uid @p stats; lse_ptr precedes the strides when
/// @p writesLse.
IngestorPlanPayload gfx942AttentionDenseStridedPayload(bool recordsStrides,
                                                       int64_t stats = ATTENTION_DENSE_NO_STATS,
                                                       bool writesLse = false)
{
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 128, 0.125F};
    binding.stats = stats;
    binding.strides = fixtureStrides();
    binding.blockN = 64;
    binding.readsStrides = true;
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::HALF;

    auto payload = gfx942AttentionDensePayload(/*withHeads=*/true, stats, writesLse);
    payload.dispatchSymbol = std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED);
    payload.launchValues = launchValuesUnder(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED,
                                             gfx942AttentionDenseArch(),
                                             binding,
                                             problem,
                                             int64_t{256});
    if(recordsStrides)
    {
        appendRecordedStrides(payload.recordedSignature);
    }
    return payload;
}

// The matching signature passes: the plan reaches the launch-value check, which a block_m
// the arch does not build then refuses.
TEST_F(TestIngestorPlanRestoreChecks, AttentionDenseStridedContractsAcceptTheirOwnSignature)
{
    auto gfx950 = gfx950AttentionDenseStridedPayload(/*recordsStrides=*/true);
    gfx950.launchValues["block_m"] = int64_t{64};
    expectRefusal(gfx950, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");

    for(const auto& [stats, writesLse] :
        {std::pair{ATTENTION_DENSE_NO_STATS, false}, std::pair{int64_t{7}, true}})
    {
        SCOPED_TRACE(stats);
        auto gfx942 = gfx942AttentionDenseStridedPayload(/*recordsStrides=*/true, stats, writesLse);
        gfx942.launchValues["block_m"] = int64_t{48};
        expectRefusal(gfx942, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");
    }
}

// A code object without the strides, such as one packed before them, does not restore
// under a strided contract: the launch would write twelve arguments the kernel never reads.
TEST_F(TestIngestorPlanRestoreChecks, AttentionDenseStridedContractsRefuseAKernelWithoutStrides)
{
    for(const auto& payload :
        {gfx950AttentionDenseStridedPayload(/*recordsStrides=*/false),
         gfx942AttentionDenseStridedPayload(/*recordsStrides=*/false),
         gfx942AttentionDenseStridedPayload(/*recordsStrides=*/false, int64_t{7}, true)})
    {
        SCOPED_TRACE(payload.dispatchSymbol);
        expectRefusal(payload,
                      HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                      "the saved kernel's arguments do not match what this provider launches");
    }
}

// The reverse: a legacy contract restores only its own older code object, never a kernel
// that takes the strides, whose last twelve arguments the legacy launch would not write. The
// bias contracts too: their kernels take the bias tail last, after any lse_ptr.
TEST_F(TestIngestorPlanRestoreChecks, AttentionDenseLegacyContractsRefuseAKernelWithStrides)
{
    auto v1 = gfx950AttentionDensePayload(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1, false);
    auto v2 = gfx950AttentionDensePayload(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2, true);
    auto v3 = gfx950AttentionDenseBiasPayload(/*withBiasTail=*/true);
    auto gfx942V1 = gfx942AttentionDensePayload(/*withHeads=*/true);
    auto gfx942V2 = gfx942AttentionDenseBiasPayload(/*withBiasTail=*/true);
    auto gfx942V2Lse
        = gfx942AttentionDenseBiasPayload(/*withBiasTail=*/true, int64_t{7}, /*writesLse=*/true);
    for(auto* payload : {&v1, &v2, &v3, &gfx942V1, &gfx942V2, &gfx942V2Lse})
    {
        SCOPED_TRACE(payload->dispatchSymbol);
        appendRecordedStrides(payload->recordedSignature);
        expectRefusal(*payload,
                      HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                      "the saved kernel's arguments do not match what this provider launches");
    }
}

// The restore re-runs the graph match's layout gate on the saved strides, after the
// signature check. A plan whose kernel does not read the strides (reads_strides 0, here a
// gfx950 gqa_pair body) holds packed strides only, so one that is not the packed stride of the
// saved dims is refused by its own name; a plan whose kernel reads them (every grid kernel of
// either arch) holds any layout the stride predicate admits, so a stride the predicate declines
// is refused under the predicate's reason.
TEST_F(TestIngestorPlanRestoreChecks, AttentionDenseStridedContractsRefuseAStrideTheirGateDeclines)
{
    auto packedOnly
        = gfx950AttentionDenseStridedPayload(/*recordsStrides=*/true, /*readsStrides=*/false);
    packedOnly.launchValues["stride_k_token"] = int64_t{1024};
    expectRefusal(packedOnly, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'stride_k_token'");

    auto gfx950 = gfx950AttentionDenseStridedPayload(/*recordsStrides=*/true);
    gfx950.launchValues["stride_v_head"] = int64_t{64};
    expectRefusal(gfx950, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "(overlap)");

    auto gfx942 = gfx942AttentionDenseStridedPayload(/*recordsStrides=*/true);
    gfx942.launchValues["stride_o_batch"] = int64_t{0};
    expectRefusal(gfx942, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "(stride_alignment)");

    // A head stride below the head size folds heads onto each other.
    auto overlapping = gfx942AttentionDenseStridedPayload(/*recordsStrides=*/true);
    overlapping.launchValues["stride_q_head"] = int64_t{64};
    expectRefusal(overlapping, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "(overlap)");
}

/// What the handler registered under @p payload's dispatch name throws when it restores
/// @p payload on no device (ordinal -1): every check before the device load runs, and the load
/// then refuses the ordinal. Empty when nothing is thrown, which no case expects.
std::string restoreOnNoDevice(const IngestorPlanPayload& payload)
{
    const auto* handler
        = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(payload.dispatchSymbol);
    EXPECT_NE(handler, nullptr) << payload.dispatchSymbol;
    if(handler == nullptr)
    {
        return {};
    }
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs inputs{payload.dispatchSymbol,
                                                                payload.launchValues};
    hipdnn_plugin_sdk::ingestor::SavedKernelCode code;
    code.kernelId = payload.kernelDescriptorId;
    code.sourceKind = payload.sourceKind;
    code.symbol = payload.symbol;
    code.target = payload.target;
    code.sha256 = payload.sha256;
    code.recordedSignature = payload.recordedSignature;
    code.codeObject = payload.codeObject;
    try
    {
        static_cast<void>(handler->restoreLaunch(inputs, std::move(code), -1));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return error.getMessage();
    }
    return {};
}

// The gfx950 strided plan as prepare() saves it for a grid kernel, whose body reads the strides
// (reads_strides 1), with head-major operands: the handler restores it past the signature check
// and the stride checks, and stops only at the device load. The same plan saved for a body that
// does not read the strides is refused by the name of the first stride that is not packed.
TEST_F(TestIngestorPlanRestoreChecks, Gfx950AttentionDenseRestoresAHeadMajorPlanOfAGridKernel)
{
    // Head-major (B, H, S, D) storage of the fixtures' shape: batch H * S * D, head S * D (a
    // slab), token D for each of Q, K, V and O.
    constexpr int64_t HEAD = 128;
    constexpr int64_t SLAB = 512 * HEAD;
    const AttentionDenseStrides headMajor{
        8 * SLAB, SLAB, HEAD, 4 * SLAB, SLAB, HEAD, 4 * SLAB, SLAB, HEAD, 8 * SLAB, SLAB, HEAD};
    auto payload = gfx950AttentionDenseStridedPayload(/*recordsStrides=*/true);
    for(std::size_t index = 0; index < headMajor.size(); ++index)
    {
        payload.launchValues[std::string(attention_dense::DENSE_STRIDE_NAMES.at(index))]
            = headMajor.at(index);
    }
    const std::string restored = restoreOnNoDevice(payload);
    EXPECT_NE(restored.find("which names no device"), std::string::npos) << restored;

    payload.launchValues["reads_strides"] = int64_t{0};
    const std::string packedOnly = restoreOnNoDevice(payload);
    EXPECT_NE(packedOnly.find("launch value 'stride_q_head'"), std::string::npos) << packedOnly;
    EXPECT_NE(packedOnly.find("reads_strides is 0"), std::string::npos) << packedOnly;
}

/// A payload under @p contract, a legacy contract of @p arch, with the restore fixtures' shape,
/// a [B, 1, Sq, Skv] bias where the contract adds one, and the argument list the contract
/// launches (attention_dense::kernelSignature on the qb_minor grid, without lse_ptr). A contract
/// that takes the strides saves them as prepare() does for a body that reads packed operands
/// only (the bias bodies among them): the packed strides, a block_n both arches build, and
/// reads_strides 0.
IngestorPlanPayload
    legacyAttentionDensePayload(const attention_dense::AttentionDenseArch& arch,
                                const attention_dense::AttentionDenseContract& contract)
{
    AttentionDenseBinding binding{1, 2, 3, 4, 1, 0, 0.125F};
    if(attention_dense::takesStrides(contract.abi))
    {
        binding.strides = fixtureStrides();
        binding.blockN = 64;
        binding.readsStrides = false;
    }
    if(attention_dense::addsBias(contract.abi))
    {
        binding.bias = AttentionDenseBias{5, {512 * 512, 0, 512}};
    }
    AttentionDenseProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 512;
    problem.seqLenKv = 512;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 4;
    problem.headSize = 128;
    problem.dataType = data_objects::DataType::BFLOAT16;

    IngestorPlanPayload payload = validConvFwdPayload();
    payload.dispatchSymbol = std::string(contract.symbol);
    payload.launchValues = launchValuesUnder(contract.symbol, arch, binding, problem, int64_t{256});
    payload.symbol = "attention_dense";
    payload.recordedSignature = attention_dense::kernelSignature(
        contract.abi, attention_dense::AttentionDenseGrid{}, /*writesLse=*/false);
    return payload;
}

// Every legacy contract of each arch's contract table restores the code object that takes its
// own arguments, reaching the launch-value check, which a block_m neither arch builds then
// refuses; and no code object that takes twelve more arguments after them (the strides, for a
// contract that predates them). Driven by the tables, and a contract that takes the strides is
// saved as prepare() saves one, so the rows a later contract change turns legacy (the strided
// ones) are covered with no edit here; the hand-written payloads above pin each list's argument
// names.
TEST_F(TestIngestorPlanRestoreChecks, EveryLegacyAttentionDenseContractRestoresOnlyItsOwnArguments)
{
    // The legacy contracts that predate the strides, which no later change adds to: gfx950 v1,
    // v2 and v3, gfx942 v1 and v2.
    for(const auto& [arch, preStride] : {std::pair{&gfx950AttentionDenseArch(), std::size_t{3}},
                                         std::pair{&gfx942AttentionDenseArch(), std::size_t{2}}})
    {
        SCOPED_TRACE(std::string(arch->label));
        std::size_t preStrideRows = 0;
        for(const auto& contract : arch->legacyContracts)
        {
            if(contract.symbol.empty())
            {
                continue;
            }
            preStrideRows += attention_dense::takesStrides(contract.abi) ? 0 : 1;
            SCOPED_TRACE(std::string(contract.symbol));
            auto own = legacyAttentionDensePayload(*arch, contract);
            own.launchValues["block_m"] = int64_t{48};
            expectRefusal(own, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");

            auto strided = legacyAttentionDensePayload(*arch, contract);
            appendRecordedStrides(strided.recordedSignature);
            expectRefusal(strided,
                          HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                          "the saved kernel's arguments do not match what this provider launches");
        }
        EXPECT_EQ(preStrideRows, preStride);
    }
}

// The saved grid selects the argument list before anything is checked: a persistent plan
// restores the code object that takes its decode's fast-division pairs (and lse_ptr after
// them), reaching the launch-value check, which a broken block_m then refuses.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseRestoresThePersistentFormOfItsDecode)
{
    using attention_dense::AttentionDensePersistDecode;
    for(const bool lse : {false, true})
    {
        SCOPED_TRACE(lse);
        auto payload = gfx942PersistentPayload(
            AttentionDensePersistDecode::BT_HKV_MINOR, {"batch", "num_query_heads"}, lse);
        payload.launchValues["block_m"] = int64_t{48};
        expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");

        auto swz = gfx942PersistentPayload(
            AttentionDensePersistDecode::HQ_MINOR_SWZ, {"nqb", "num_query_heads", "gqa"}, lse);
        swz.launchValues["block_m"] = int64_t{48};
        expectRefusal(swz, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "launch value 'block_m'");
    }
}

// A code object whose arguments disagree with the saved grid is refused by the signature
// check: a grid kernel under a persistent plan, a persistent one under a grid plan, and
// another decode's pairs.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseRefusesAKernelThatDisagreesWithItsGrid)
{
    using attention_dense::AttentionDensePersistDecode;
    constexpr const char* MISMATCH
        = "the saved kernel's arguments do not match what this provider launches";
    expectRefusal(gfx942PersistentPayload(AttentionDensePersistDecode::QB_MAJOR, {}),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  MISMATCH);
    expectRefusal(gfx942PersistentPayload(AttentionDensePersistDecode::QB_MAJOR,
                                          {"batch", "num_query_heads"}),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  MISMATCH);
    expectRefusal(gfx942PersistentPayload(AttentionDensePersistDecode::HKV_MAJOR,
                                          {"batch", "num_query_heads", "gqa"}),
                  HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                  MISMATCH);

    auto gridPlan = gfx942PersistentPayload(AttentionDensePersistDecode::QB_MAJOR,
                                            {"batch", "num_query_heads", "gqa"});
    gridPlan.launchValues["grid_order"] = std::string("bt_hkv_minor");
    gridPlan.launchValues["persist_decode"] = std::string();
    gridPlan.launchValues["num_persistent"] = int64_t{0};
    expectRefusal(gridPlan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, MISMATCH);
}

// The unversioned gfx942 name resolves to the catalog's handler, the strided contract's,
// which reads only that contract: a plan with the strided kernel and values passes the
// signature check and is refused by name.
TEST_F(TestIngestorPlanRestoreChecks, Gfx942AttentionDenseRefusesItsUnversionedAlias)
{
    auto payload = gfx942AttentionDenseStridedPayload(/*recordsStrides=*/true);
    payload.dispatchSymbol = "hipkernel.gfx942_attention_dense.dispatch";

    expectRefusal(payload, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "names no launch-value contract");
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
