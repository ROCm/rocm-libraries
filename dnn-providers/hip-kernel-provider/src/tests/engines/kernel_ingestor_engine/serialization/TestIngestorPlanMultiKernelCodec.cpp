// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
#include "engines/kernel_ingestor_engine/serialization/ingestor_plan_generated.h"

// The kind 2 (multi-kernel) ingestor plan codec: the round trip of every field, the body
// layout, and the structure checks of the reader. No GPU is used. A test that builds a
// damaged or foreign body writes a new header digest over the changed body, unless it tests
// the digest itself. The check under test is then the one that refuses.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{
namespace
{

using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

constexpr size_t FORMAT_MAJOR_OFFSET = 4;
constexpr size_t KIND_OFFSET = 8;
constexpr const char* MULTI_KERNEL_IDENTIFIER = "HKMP";

IngestorCodeObjectPayload makeCodeObject(KernelSourceKind sourceKind,
                                         const std::string& target,
                                         size_t size,
                                         uint8_t seed)
{
    IngestorCodeObjectPayload codeObject;
    codeObject.sourceKind = sourceKind;
    codeObject.target = target;
    codeObject.codeObject = patternBytes(size, seed);
    codeObject.sha256 = sha256Of(codeObject.codeObject);
    return codeObject;
}

IngestorStageKernelPayload
    makeKernel(const std::string& launchName, const std::string& symbol, uint32_t codeObjectIndex)
{
    IngestorStageKernelPayload kernel;
    kernel.launchName = launchName;
    kernel.symbol = symbol;
    kernel.recordedSignature
        = {KernelArgument{"global_buffer", 8, 0, ""}, KernelArgument{"by_value", 4, 8, "count"}};
    kernel.codeObjectIndex = codeObjectIndex;
    return kernel;
}

IngestorStagePayload makeStage(const std::string& slot,
                               const std::string& dispatchSymbol,
                               uint8_t idSeed,
                               std::vector<IngestorStageKernelPayload> kernels)
{
    IngestorStagePayload stage;
    stage.slot = slot;
    for(size_t index = 0; index < stage.kernelDescriptorId.size(); ++index)
    {
        stage.kernelDescriptorId[index] = static_cast<uint8_t>(idSeed + index);
    }
    stage.dispatchSymbol = dispatchSymbol;
    stage.launchValues = sampleLaunchValues();
    stage.kernels = std::move(kernels);
    return stage;
}

// Three stages over two code objects. The second stage launches two kernels. The first and
// second stage share a code object.
IngestorMultiKernelPayload makeMaximalPayload()
{
    IngestorMultiKernelPayload payload;
    payload.codeObjects = {makeCodeObject(KernelSourceKind::KPACK, "gfx942", 37, 0x31),
                           makeCodeObject(KernelSourceKind::HSACO_FILE, "gfx942:xnack-", 64, 0x47)};
    payload.stages = {
        makeStage(
            "slot_prep", "hipkernel.test.prep.dispatch.v1", 0x10, {makeKernel("", "prep", 0)}),
        makeStage(
            "slot_dq",
            "hipkernel.test.dq.dispatch.v1",
            0x30,
            {makeKernel("dq", "dq_kernel", 0), makeKernel("dbias_reduce", "reduce_kernel", 1)}),
        makeStage(
            "slot_dkdv", "hipkernel.test.dkdv.dispatch.v1", 0x50, {makeKernel("", "dkdv", 1)}),
    };
    payload.sharedWorkspaceBytes = 1024;
    payload.workspaceBytes = 8192;
    payload.runtimePassByValueUids = {7, 9};
    payload.providerVersion = "9.8.7";
    return payload;
}

// One unslotted stage over the given code objects, one kernel per code object.
IngestorMultiKernelPayload makeOneStagePayload(std::vector<IngestorCodeObjectPayload> codeObjects)
{
    IngestorMultiKernelPayload payload;
    std::vector<IngestorStageKernelPayload> kernels;
    kernels.reserve(codeObjects.size());
    for(size_t index = 0; index < codeObjects.size(); ++index)
    {
        kernels.push_back(makeKernel("launch" + std::to_string(index),
                                     "kernel" + std::to_string(index),
                                     static_cast<uint32_t>(index)));
    }
    payload.codeObjects = std::move(codeObjects);
    payload.stages = {makeStage("", "hipkernel.test.dispatch.v1", 0x10, std::move(kernels))};
    payload.workspaceBytes = 512;
    payload.providerVersion = "1.2.3";
    return payload;
}

IngestorPlanPayload makeSingleKernelPayload()
{
    IngestorPlanPayload payload;
    payload.workspaceBytes = 64;
    payload.dispatchSymbol = "hipkernel.test.dispatch.v1";
    payload.sourceKind = KernelSourceKind::KPACK;
    payload.symbol = "single";
    payload.target = "gfx942";
    payload.codeObject = patternBytes(20, 0x11);
    payload.sha256 = sha256Of(payload.codeObject);
    payload.providerVersion = "1.2.3";
    return payload;
}

IngestorMultiKernelPayload decode(const std::vector<uint8_t>& bytes)
{
    return std::get<IngestorMultiKernelPayload>(
        decodeIngestorSavedPlan(bytes.data(), bytes.size()));
}

void expectDecodeRefusal(const std::vector<uint8_t>& bytes,
                         hipdnnPluginStatus_t status,
                         const std::string& phrase)
{
    expectIngestorPlanRefusal(
        [&]() { decodeIngestorSavedPlan(bytes.data(), bytes.size()); }, status, phrase);
}

void expectEncodeInternalError(const IngestorMultiKernelPayload& payload, const std::string& phrase)
{
    expectPluginInternalError([&]() { encodeIngestorPlan(payload); }, phrase);
}

struct RawKernel
{
    std::string launchName;
    uint32_t codeObjectIndex = 0;
};

struct RawStage
{
    std::string slot;
    std::vector<RawKernel> kernels;
    std::vector<RawLaunchValue> values = {RawLaunchValue{"count"}};
};

struct RawCodeObject
{
    std::vector<uint8_t> bytes;
    // The digest the body records. The digest of `bytes` when absent.
    std::optional<std::string> sha256 = std::nullopt;
    fb::SourceKind sourceKind = fb::SourceKind::KPACK;
};

// The body fields a raw build can set to values that encodeIngestorPlan never writes.
struct RawPlan
{
    std::vector<RawStage> stages;
    std::vector<RawCodeObject> codeObjects;
    std::optional<uint64_t> sharedWorkspaceBytes = 64;
    std::optional<uint64_t> workspaceBytes = 256;
    std::vector<int64_t> runtimePassByValueUids;
};

// Two stages over two code objects: [dq, dbias_reduce] and [""].
RawPlan makeValidRawPlan()
{
    RawPlan raw;
    raw.codeObjects
        = {RawCodeObject{patternBytes(16, 0x55)}, RawCodeObject{patternBytes(24, 0x66)}};
    raw.stages = {RawStage{"first", {RawKernel{"dq", 0}, RawKernel{"dbias_reduce", 1}}},
                  RawStage{"second", {RawKernel{"", 1}}}};
    return raw;
}

// Builds a sealed kind 2 payload with the generated builder, without the writer's checks.
std::vector<uint8_t> buildRawPayload(const RawPlan& raw)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<fb::CodeObject>> codeObjects;
    for(const RawCodeObject& object : raw.codeObjects)
    {
        builder.ForceVectorAlignment(object.bytes.size(), sizeof(uint8_t), ALIGNMENT);
        const auto bytes = builder.CreateVector(object.bytes);
        const auto target = builder.CreateString("gfx942");
        const auto sha256 = builder.CreateString(object.sha256.value_or(sha256Of(object.bytes)));
        codeObjects.push_back(
            fb::CreateCodeObject(builder, object.sourceKind, target, sha256, bytes));
    }
    const auto codeObjectsVector = builder.CreateVector(codeObjects);

    std::vector<flatbuffers::Offset<fb::Stage>> stages;
    for(const RawStage& stage : raw.stages)
    {
        std::vector<flatbuffers::Offset<fb::StageKernel>> kernels;
        for(const RawKernel& kernel : stage.kernels)
        {
            const auto signature
                = builder.CreateVector(std::vector<flatbuffers::Offset<fb::KernelArgument>>{});
            const auto launchName = builder.CreateString(kernel.launchName);
            const auto symbol = builder.CreateString("raw_kernel");
            kernels.push_back(fb::CreateStageKernel(
                builder, launchName, symbol, signature, kernel.codeObjectIndex));
        }
        const auto kernelsVector = builder.CreateVector(kernels);

        const auto valuesVector = writeRawLaunchValues(builder, stage.values);
        const auto dispatchSymbol = builder.CreateString("raw.dispatch.v1");
        const auto dispatchData = fb::CreateDispatchData(builder, dispatchSymbol, valuesVector);

        const auto slot = builder.CreateString(stage.slot);
        const fb::Uuid kernelDescriptorId;
        stages.push_back(
            fb::CreateStage(builder, slot, &kernelDescriptorId, dispatchData, kernelsVector));
    }
    const auto stagesVector = builder.CreateVector(stages);

    const auto uids = builder.CreateVector(raw.runtimePassByValueUids);
    const auto providerVersion = builder.CreateString("0.0.1");

    fb::MultiKernelPlanBuilder plan(builder);
    plan.add_stages(stagesVector);
    plan.add_code_objects(codeObjectsVector);
    if(raw.sharedWorkspaceBytes.has_value())
    {
        plan.add_shared_workspace_bytes(*raw.sharedWorkspaceBytes);
    }
    if(raw.workspaceBytes.has_value())
    {
        plan.add_workspace_bytes(*raw.workspaceBytes);
    }
    plan.add_runtime_pass_by_value_uids(uids);
    plan.add_provider_version(providerVersion);
    // End the table directly. The generated Finish() asserts that every required field is set.
    const flatbuffers::Offset<fb::MultiKernelPlan> root(builder.EndTable(plan.start_));
    builder.Finish(root, MULTI_KERNEL_IDENTIFIER);

    const uint8_t* body = builder.GetBufferPointer();
    return sealIngestorPlanBody(IngestorPlanKind::MULTI_KERNEL_PLAN,
                                std::vector<uint8_t>(body, body + builder.GetSize()));
}

TEST(TestIngestorPlanMultiKernelCodec, RoundTripsAMaximalPayload)
{
    const IngestorMultiKernelPayload payload = makeMaximalPayload();
    const std::vector<uint8_t> encoded = encodeIngestorPlan(payload);
    const IngestorSavedPlan saved = decodeIngestorSavedPlan(encoded.data(), encoded.size());

    ASSERT_TRUE(std::holds_alternative<IngestorMultiKernelPayload>(saved));
    const auto& decoded = std::get<IngestorMultiKernelPayload>(saved);
    EXPECT_EQ(decoded.stages.size(), 3U);
    EXPECT_EQ(decoded.stages[1].kernels.size(), 2U);
    EXPECT_EQ(decoded.codeObjects.size(), 2U);
    EXPECT_EQ(decoded.sharedWorkspaceBytes, 1024U);
    EXPECT_EQ(decoded.workspaceBytes, 8192U);
    EXPECT_EQ(decoded.runtimePassByValueUids, (std::vector<int64_t>{7, 9}));
    EXPECT_EQ(decoded.providerVersion, "9.8.7");
    for(size_t index = 0; index < decoded.stages.size(); ++index)
    {
        SCOPED_TRACE("stage " + std::to_string(index));
        EXPECT_EQ(decoded.stages[index].launchValues.size(), 5U);
        EXPECT_TRUE(decoded.stages[index].launchValues == payload.stages[index].launchValues);
    }
    EXPECT_TRUE(decoded == payload);
}

TEST(TestIngestorPlanMultiKernelCodec, BufferEncodingRoundTrips)
{
    const IngestorMultiKernelPayload payload = makeMaximalPayload();
    const EncodedIngestorPlan encoded = encodeIngestorPlanToBuffer(payload);

    ASSERT_NE(encoded.bytes.get(), nullptr);
    ASSERT_GT(encoded.size, HEADER_SIZE);
    const IngestorPlanHeader header = decodeIngestorPlanHeader(encoded.bytes.get(), encoded.size);
    EXPECT_EQ(header.kind, IngestorPlanKind::MULTI_KERNEL_PLAN);

    // The header digest covers every byte after the header, and the body carries HKMP.
    const uint8_t* body = encoded.bytes.get() + HEADER_SIZE;
    const size_t bodySize = encoded.size - HEADER_SIZE;
    EXPECT_EQ(header.bodyDigest, detail::computeIngestorPlanBodyDigest(body, bodySize));
    EXPECT_TRUE(flatbuffers::BufferHasIdentifier(body, MULTI_KERNEL_IDENTIFIER));

    const IngestorSavedPlan decoded = decodeIngestorSavedPlan(encoded.bytes.get(), encoded.size);
    EXPECT_TRUE(std::get<IngestorMultiKernelPayload>(decoded) == payload);
}

TEST(TestIngestorPlanMultiKernelCodec, ReadsAKindOnePayloadAsKindOne)
{
    const IngestorPlanPayload payload = makeSingleKernelPayload();
    const std::vector<uint8_t> encoded = encodeIngestorPlan(payload);

    const IngestorSavedPlan saved = decodeIngestorSavedPlan(encoded.data(), encoded.size());
    ASSERT_TRUE(std::holds_alternative<IngestorPlanPayload>(saved));
    EXPECT_TRUE(std::get<IngestorPlanPayload>(saved) == payload);
}

TEST(TestIngestorPlanMultiKernelCodec, StoresEachCodeObjectOnce)
{
    constexpr size_t CODE_SIZE = 4096;
    const IngestorCodeObjectPayload codeObject
        = makeCodeObject(KernelSourceKind::KPACK, "gfx942", CODE_SIZE, 0x31);

    // Two entries with the same bytes, one for each kernel of the stage.
    const IngestorMultiKernelPayload duplicated = makeOneStagePayload({codeObject, codeObject});
    const std::vector<uint8_t> encoded = encodeIngestorPlan(duplicated);
    const IngestorMultiKernelPayload decoded = decode(encoded);

    ASSERT_EQ(decoded.codeObjects.size(), 1U);
    EXPECT_TRUE(decoded.codeObjects[0] == codeObject);
    ASSERT_EQ(decoded.stages[0].kernels.size(), 2U);
    EXPECT_EQ(decoded.stages[0].kernels[0].codeObjectIndex, 0U);
    EXPECT_EQ(decoded.stages[0].kernels[1].codeObjectIndex, 0U);
    EXPECT_LT(encoded.size(), 2 * CODE_SIZE);

    // A payload that already holds the shared entry once encodes to the same bytes.
    IngestorMultiKernelPayload shared = makeOneStagePayload({codeObject});
    shared.stages[0].kernels.push_back(makeKernel("launch1", "kernel1", 0));
    EXPECT_EQ(encodeIngestorPlan(shared), encoded);
}

TEST(TestIngestorPlanMultiKernelCodec, KeepsCodeObjectsThatDifferInTargetOrSourceKind)
{
    const IngestorCodeObjectPayload base
        = makeCodeObject(KernelSourceKind::KPACK, "gfx942", 40, 0x31);
    IngestorCodeObjectPayload otherTarget = base;
    otherTarget.target = "gfx90a";
    IngestorCodeObjectPayload otherKind = base;
    otherKind.sourceKind = KernelSourceKind::HSACO_FILE;

    const IngestorMultiKernelPayload payload = makeOneStagePayload({base, otherTarget, otherKind});
    EXPECT_TRUE(decode(encodeIngestorPlan(payload)) == payload);
}

TEST(TestIngestorPlanMultiKernelCodec, KeepsStageOrderAndLaunchOrder)
{
    IngestorMultiKernelPayload payload;
    payload.codeObjects = {makeCodeObject(KernelSourceKind::KPACK, "gfx942", 20, 0x11)};
    payload.stages = {
        makeStage("z", "hipkernel.test.z.dispatch.v1", 0x10, {makeKernel("", "z_kernel", 0)}),
        makeStage("a",
                  "hipkernel.test.a.dispatch.v1",
                  0x20,
                  {makeKernel("c", "c_kernel", 0),
                   makeKernel("a", "a_kernel", 0),
                   makeKernel("b", "b_kernel", 0)}),
        makeStage("m", "hipkernel.test.m.dispatch.v1", 0x30, {makeKernel("", "m_kernel", 0)}),
    };
    payload.workspaceBytes = 16;
    payload.providerVersion = "1.0.0";

    const IngestorMultiKernelPayload decoded = decode(encodeIngestorPlan(payload));

    ASSERT_EQ(decoded.stages.size(), 3U);
    EXPECT_EQ(decoded.stages[0].slot, "z");
    EXPECT_EQ(decoded.stages[1].slot, "a");
    EXPECT_EQ(decoded.stages[2].slot, "m");
    ASSERT_EQ(decoded.stages[1].kernels.size(), 3U);
    EXPECT_EQ(decoded.stages[1].kernels[0].launchName, "c");
    EXPECT_EQ(decoded.stages[1].kernels[1].launchName, "a");
    EXPECT_EQ(decoded.stages[1].kernels[2].launchName, "b");
}

TEST(TestIngestorPlanMultiKernelCodec, RoundTripsAMinimalPayload)
{
    IngestorMultiKernelPayload payload
        = makeOneStagePayload({makeCodeObject(KernelSourceKind::EMBEDDED_SOURCE, "", 0, 0x00)});
    payload.stages[0].launchValues.clear();
    payload.stages[0].kernels[0].recordedSignature.clear();
    payload.stages[0].kernels[0].launchName.clear();
    EXPECT_TRUE(decode(encodeIngestorPlan(payload)) == payload);
}

TEST(TestIngestorPlanMultiKernelCodec, EncodesAKindTwoPayloadAsFormat1Point1)
{
    const std::vector<uint8_t> encoded = encodeIngestorPlan(makeMaximalPayload());

    ASSERT_GT(encoded.size(), HEADER_SIZE);
    EXPECT_EQ(encoded[FORMAT_MAJOR_OFFSET], 0x01); // major 1
    EXPECT_EQ(encoded[FORMAT_MAJOR_OFFSET + 1], 0x00);
    EXPECT_EQ(encoded[FORMAT_MAJOR_OFFSET + 2], 0x01); // minor 1
    EXPECT_EQ(encoded[FORMAT_MAJOR_OFFSET + 3], 0x00);
    EXPECT_EQ(encoded[KIND_OFFSET], 0x02);
    EXPECT_EQ(encoded[KIND_OFFSET + 1], 0x00);
}

TEST(TestIngestorPlanMultiKernelCodec, EachCodeObjectIsSixteenByteAligned)
{
    for(size_t codeSize = 1; codeSize <= 17; ++codeSize)
    {
        SCOPED_TRACE("code size " + std::to_string(codeSize));
        const std::vector<uint8_t> encoded = encodeIngestorPlan(makeOneStagePayload(
            {makeCodeObject(KernelSourceKind::KPACK, "gfx942", codeSize, 0x31),
             makeCodeObject(KernelSourceKind::KPACK, "gfx942", codeSize + 3, 0x32),
             makeCodeObject(KernelSourceKind::KPACK, "gfx942", codeSize + 5, 0x33)}));

        std::vector<detail::IngestorPlanAlignedBlock> storage;
        const uint8_t* body = detail::alignedIngestorPlanBody(
            encoded.data() + HEADER_SIZE, encoded.size() - HEADER_SIZE, storage);
        const auto* plan = flatbuffers::GetRoot<fb::MultiKernelPlan>(body);
        ASSERT_EQ(plan->code_objects()->size(), 3U);
        for(const fb::CodeObject* codeObject : *plan->code_objects())
        {
            EXPECT_EQ(static_cast<size_t>(codeObject->code_object()->data() - body) % ALIGNMENT,
                      0U);
        }
        EXPECT_EQ(decode(encoded).codeObjects[0].codeObject.size(), codeSize);
    }
}

TEST(TestIngestorPlanMultiKernelCodec, DecodeIngestorPlanRefusesAKindTwoPayload)
{
    const std::vector<uint8_t> encoded = encodeIngestorPlan(makeMaximalPayload());
    expectIngestorPlanRefusal([&]() { decodeIngestorPlan(encoded.data(), encoded.size()); },
                              HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                              "is not a single-kernel plan");
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesAPayloadWithNoStages)
{
    RawPlan raw = makeValidRawPlan();
    EXPECT_NO_THROW(decode(buildRawPayload(raw)));
    raw.stages.clear();
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "no stages");

    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.stages.clear();
    expectEncodeInternalError(payload, "no stages");
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesAStageWithNoKernels)
{
    RawPlan raw = makeValidRawPlan();
    raw.stages[1].kernels.clear();
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "stage 1 has no kernels");

    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.stages[2].kernels.clear();
    expectEncodeInternalError(payload, "stage 2 has no kernels");
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesACodeObjectIndexOutOfRange)
{
    RawPlan raw = makeValidRawPlan();
    raw.stages[1].kernels[0].codeObjectIndex = 2;
    expectDecodeRefusal(buildRawPayload(raw),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "names code object 2, and the plan holds 2");

    raw.stages[1].kernels[0].codeObjectIndex = 1;
    EXPECT_NO_THROW(decode(buildRawPayload(raw)));

    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.stages[1].kernels[1].codeObjectIndex = 2;
    expectEncodeInternalError(payload, "names code object 2, and the plan holds 2");
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesACodeObjectWhoseShaDoesNotMatchItsBytesInALaterStage)
{
    RawPlan raw = makeValidRawPlan();
    const std::string otherDigest = sha256Of(patternBytes(24, 0x01));
    ASSERT_NE(otherDigest, sha256Of(raw.codeObjects[1].bytes));
    raw.codeObjects[1].sha256 = otherDigest;

    // The second stage is the only one that launches code object 1 alone, and the first
    // code object is intact.
    const std::string refusal
        = expectIngestorPlanRefusal([&]() { decode(buildRawPayload(raw)); },
                                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "of code object 1 does not match the code object");
    EXPECT_NE(refusal.find(otherDigest), std::string::npos) << refusal;
    EXPECT_NE(refusal.find(sha256Of(raw.codeObjects[1].bytes)), std::string::npos) << refusal;
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesAMalformedCodeObjectDigestOrSourceKind)
{
    RawPlan shortDigest = makeValidRawPlan();
    shortDigest.codeObjects[0].sha256 = std::string(63, 'a');
    expectDecodeRefusal(buildRawPayload(shortDigest),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "of code object 0 is not 64 lowercase hex characters");

    RawPlan upperCaseDigest = makeValidRawPlan();
    upperCaseDigest.codeObjects[1].sha256 = std::string(64, 'A');
    expectDecodeRefusal(buildRawPayload(upperCaseDigest),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "of code object 1 is not 64 lowercase hex characters");

    RawPlan unset = makeValidRawPlan();
    unset.codeObjects[1].sourceKind = fb::SourceKind::UNSET;
    expectDecodeRefusal(buildRawPayload(unset),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "source kind 0 of code object 1");

    RawPlan unknown = makeValidRawPlan();
    unknown.codeObjects[0].sourceKind = static_cast<fb::SourceKind>(9);
    expectDecodeRefusal(buildRawPayload(unknown),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "source kind 9 of code object 0");
}

// A stage with no launch values reads. A stage with a malformed value, in any stage, is refused.
TEST(TestIngestorPlanMultiKernelCodec, RefusesAnEmptyLaunchValueSetWithAMalformedValue)
{
    RawPlan raw = makeValidRawPlan();
    raw.stages[1].values.clear();
    EXPECT_TRUE(decode(buildRawPayload(raw)).stages[1].launchValues.empty());

    raw.stages[1].values = {RawLaunchValue{"none", fb::LaunchValueData::NONE}};
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "unknown value type 0");

    raw.stages[1].values = {RawLaunchValue{"future", static_cast<fb::LaunchValueData>(9)}};
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "unknown value type 9");

    raw.stages[1].values = {RawLaunchValue{""}};
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "has no name");

    raw.stages[1].values = {RawLaunchValue{"twice"}, RawLaunchValue{"twice"}};
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "occurs more than once");

    raw.stages[1].values = {RawLaunchValue{"empty", fb::LaunchValueData::Int64Value, false}};
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "no data");

    raw.stages[1].values = {RawLaunchValue{"count"}};
    EXPECT_EQ(std::get<int64_t>(decode(buildRawPayload(raw)).stages[1].launchValues.at("count")),
              5);
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesAStageWithDuplicateLaunchNames)
{
    RawPlan raw = makeValidRawPlan();
    raw.stages[0].kernels = {RawKernel{"dq", 0}, RawKernel{"dq", 1}, RawKernel{"dbias_reduce", 1}};
    expectDecodeRefusal(buildRawPayload(raw),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "launch name 'dq' occurs more than once in stage 0");

    // The same name in two stages is not a repeat.
    raw.stages[0].kernels = {RawKernel{"dq", 0}, RawKernel{"dbias_reduce", 1}};
    raw.stages[1].kernels = {RawKernel{"dq", 1}};
    EXPECT_NO_THROW(decode(buildRawPayload(raw)));

    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.stages[1].kernels.insert(payload.stages[1].kernels.begin(),
                                     makeKernel("dq", "again", 0));
    expectEncodeInternalError(payload, "launch name 'dq' occurs more than once in stage 1");
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesAPayloadWithoutItsWorkspaceFields)
{
    RawPlan withoutShared = makeValidRawPlan();
    withoutShared.sharedWorkspaceBytes.reset();
    expectDecodeRefusal(buildRawPayload(withoutShared),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "the required field shared_workspace_bytes is missing");

    RawPlan withoutTotal = makeValidRawPlan();
    withoutTotal.workspaceBytes.reset();
    expectDecodeRefusal(buildRawPayload(withoutTotal),
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "the required field workspace_bytes is missing");

    // A field that holds 0 is present.
    RawPlan zeros = makeValidRawPlan();
    zeros.sharedWorkspaceBytes = 0;
    zeros.workspaceBytes = 0;
    const IngestorMultiKernelPayload decoded = decode(buildRawPayload(zeros));
    EXPECT_EQ(decoded.sharedWorkspaceBytes, 0U);
    EXPECT_EQ(decoded.workspaceBytes, 0U);
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesASharedRegionLargerThanTheTotalWorkspace)
{
    RawPlan raw = makeValidRawPlan();
    raw.sharedWorkspaceBytes = 257;
    raw.workspaceBytes = 256;
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "more than the total workspace");

    raw.sharedWorkspaceBytes = 256;
    EXPECT_EQ(decode(buildRawPayload(raw)).sharedWorkspaceBytes, 256U);

    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.sharedWorkspaceBytes = payload.workspaceBytes + 1;
    expectEncodeInternalError(payload, "more than the total workspace");
}

TEST(TestIngestorPlanMultiKernelCodec, RefusesAKindByteThatDisagreesWithTheBody)
{
    // A kind 2 body under a kind 1 header.
    std::vector<uint8_t> kindTwoAsOne = encodeIngestorPlan(makeMaximalPayload());
    kindTwoAsOne[KIND_OFFSET] = static_cast<uint8_t>(IngestorPlanKind::SINGLE_KERNEL_PLAN);
    expectDecodeRefusal(kindTwoAsOne, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "verifier");

    // A kind 1 body under a kind 2 header. The header digest covers the body only.
    std::vector<uint8_t> kindOneAsTwo = encodeIngestorPlan(makeSingleKernelPayload());
    kindOneAsTwo[KIND_OFFSET] = static_cast<uint8_t>(IngestorPlanKind::MULTI_KERNEL_PLAN);
    expectDecodeRefusal(kindOneAsTwo, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "verifier");
}

TEST(TestIngestorPlanMultiKernelCodec, ReadsAPayloadThatListsRuntimePassByValueUids)
{
    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.runtimePassByValueUids = {-3, 7, 9};
    EXPECT_EQ(decode(encodeIngestorPlan(payload)).runtimePassByValueUids,
              payload.runtimePassByValueUids);
}

TEST(TestIngestorPlanMultiKernelCodec, WriterRejectsInputsItOwnsAsAnInternalError)
{
    IngestorMultiKernelPayload noProviderVersion = makeMaximalPayload();
    noProviderVersion.providerVersion.clear();
    expectEncodeInternalError(noProviderVersion, "provider version is empty");

    IngestorMultiKernelPayload noDispatch = makeMaximalPayload();
    noDispatch.stages[2].dispatchSymbol.clear();
    expectEncodeInternalError(noDispatch, "dispatch symbol of stage 2 is empty");

    IngestorMultiKernelPayload noSymbol = makeMaximalPayload();
    noSymbol.stages[1].kernels[1].symbol.clear();
    expectEncodeInternalError(noSymbol, "kernel symbol of stage 1 is empty");

    IngestorMultiKernelPayload badDigest = makeMaximalPayload();
    badDigest.codeObjects[1].sha256 = std::string(64, 'A');
    expectEncodeInternalError(badDigest, "digest 'AAAA");

    IngestorMultiKernelPayload badKind = makeMaximalPayload();
    badKind.codeObjects[0].sourceKind = static_cast<KernelSourceKind>(99);
    expectEncodeInternalError(badKind, "source kind 99");
}

// The checks that hold for every payload the reader returns, whatever a damaged body byte
// changed. A length read out of a damaged body cannot exceed the body it came from.
void expectSelfConsistentPayload(const IngestorMultiKernelPayload& decoded, size_t bodySize)
{
    EXPECT_LE(decoded.sharedWorkspaceBytes, decoded.workspaceBytes);
    EXPECT_LE(decoded.runtimePassByValueUids.size(), bodySize);
    EXPECT_LE(decoded.providerVersion.size(), bodySize);
    EXPECT_LE(decoded.codeObjects.size(), bodySize);
    EXPECT_FALSE(decoded.stages.empty());
    EXPECT_LE(decoded.stages.size(), bodySize);

    for(const IngestorCodeObjectPayload& codeObject : decoded.codeObjects)
    {
        EXPECT_EQ(codeObject.sha256.size(), SHA256_HEX_LENGTH);
        EXPECT_EQ(codeObject.sha256, sha256Of(codeObject.codeObject));
        EXPECT_LE(codeObject.target.size(), bodySize);
    }
    for(const IngestorStagePayload& stage : decoded.stages)
    {
        EXPECT_LE(stage.slot.size(), bodySize);
        EXPECT_LE(stage.dispatchSymbol.size(), bodySize);
        EXPECT_LE(stage.launchValues.size(), bodySize);
        for(const auto& value : stage.launchValues)
        {
            EXPECT_FALSE(value.first.empty());
        }
        EXPECT_FALSE(stage.kernels.empty());
        EXPECT_LE(stage.kernels.size(), bodySize);
        std::set<std::string> launchNames;
        for(const IngestorStageKernelPayload& kernel : stage.kernels)
        {
            EXPECT_TRUE(launchNames.insert(kernel.launchName).second) << kernel.launchName;
            EXPECT_LT(kernel.codeObjectIndex, decoded.codeObjects.size());
            EXPECT_LE(kernel.symbol.size(), bodySize);
            EXPECT_LE(kernel.recordedSignature.size(), bodySize);
        }
    }
}

// Decodes a payload whose body carries damage under a matching digest. Two outcomes are
// allowed: a load refusal, or a payload that is self-consistent. The refusal must carry a
// load status and the message prefix of its category. Any other exception type leaves this
// function and fails the test. Returns true when the reader refused.
bool decodeRefusesOrReadsAConsistentPayload(const std::vector<uint8_t>& payload)
{
    try
    {
        expectSelfConsistentPayload(decode(payload), payload.size() - HEADER_SIZE);
        return false;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        const std::string message = error.getMessage();
        const bool incompatible = error.getStatus() == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE;
        EXPECT_TRUE(incompatible || error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
            << "status " << static_cast<int>(error.getStatus()) << ": " << message;
        const std::string_view prefix
            = incompatible ? INGESTOR_PLAN_INCOMPATIBLE_PREFIX : INGESTOR_PLAN_DAMAGED_PREFIX;
        EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
        return true;
    }
}

// Every body byte in turn, flipped and sealed again, so the damage passes the digest and
// reaches the verifier and the structure checks. A flip in most field data still decodes.
// A flip in an offset, a length, a vtable entry, a code object digest or the code bytes is
// refused.
TEST(TestIngestorPlanMultiKernelCodec, RefusesOrReadsEveryResealedBodyByteFlipOfAKindTwoPayload)
{
    // Covers every body byte. A larger stride samples the body instead.
    constexpr size_t STRIDE = 1;
    constexpr uint8_t FLIP_MASK = 0xFF;

    IngestorMultiKernelPayload payload = makeMaximalPayload();
    payload.codeObjects[0].codeObject.resize(16);
    payload.codeObjects[0].sha256 = sha256Of(payload.codeObjects[0].codeObject);
    payload.codeObjects[1].codeObject.resize(16);
    payload.codeObjects[1].sha256 = sha256Of(payload.codeObjects[1].codeObject);
    const std::vector<uint8_t> encoded = encodeIngestorPlan(payload);
    const size_t bodySize = encoded.size() - HEADER_SIZE;

    size_t refusedCount = 0;
    size_t readCount = 0;
    for(size_t offset = 0; offset < bodySize; offset += STRIDE)
    {
        SCOPED_TRACE("body byte " + std::to_string(offset) + " of " + std::to_string(bodySize));
        std::vector<uint8_t> damaged = encoded;
        const size_t index = HEADER_SIZE + offset;
        damaged[index] = static_cast<uint8_t>(damaged[index] ^ FLIP_MASK);
        resealIngestorPlan(damaged);

        // An exact-size heap copy lets ASAN see any read past the end of the payload.
        const std::vector<uint8_t> probe(damaged.begin(), damaged.end());
        if(decodeRefusesOrReadsAConsistentPayload(probe))
        {
            ++refusedCount;
        }
        else
        {
            ++readCount;
        }
    }

    // The sweep reaches both outcomes, so neither branch of the check above is dead.
    EXPECT_GT(refusedCount, 0U);
    EXPECT_GT(readCount, 0U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
