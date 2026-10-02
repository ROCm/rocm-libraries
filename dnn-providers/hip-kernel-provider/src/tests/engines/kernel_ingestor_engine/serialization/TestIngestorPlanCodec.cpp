// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
#include "engines/kernel_ingestor_engine/serialization/ingestor_plan_generated.h"
#include "utilities/Digest.hpp"

// The ingestor plan codec: the round trip of every field, the body layout, and the order of
// the reader's checks. No GPU is used. A test that builds a damaged or foreign body writes a
// new header digest over the changed body, unless it tests the digest itself. The check
// under test is then the one that refuses.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

constexpr size_t HEADER_SIZE = INGESTOR_PLAN_HEADER_SIZE;
constexpr size_t ALIGNMENT = INGESTOR_PLAN_BODY_ALIGNMENT;
constexpr size_t FILE_IDENTIFIER_OFFSET = sizeof(flatbuffers::uoffset_t);
constexpr std::array<uint8_t, 4> PLAN_IDENTIFIER = {0x48, 0x4B, 0x53, 0x50}; // HKSP
constexpr std::array<uint8_t, 4> FOREIGN_IDENTIFIER = {0x58, 0x58, 0x58, 0x58}; // XXXX

std::vector<uint8_t> patternBytes(size_t size, uint8_t seed)
{
    std::vector<uint8_t> bytes(size);
    for(size_t index = 0; index < size; ++index)
    {
        bytes[index] = static_cast<uint8_t>(seed + (index * 7U));
    }
    return bytes;
}

IngestorPlanPayload makePayload(KernelSourceKind sourceKind, size_t codeSize = 37)
{
    IngestorPlanPayload payload;
    payload.engineId = 0x123456789A;
    for(size_t index = 0; index < payload.kernelId.size(); ++index)
    {
        payload.kernelId[index] = static_cast<uint8_t>(0x10U + index);
    }
    payload.workspaceBytes = 4096;
    payload.dispatchSymbol = "hipkernel.test.dispatch.v1";
    payload.launchValues = {
        {"flag", true},
        {"count", int64_t{-42}},
        {"scale", 0.25},
        {"layout", std::string("bshd")},
        {"shape", std::vector<int64_t>{2, 3, 5}},
    };
    payload.sourceKind = sourceKind;
    payload.symbol = "test_kernel";
    payload.target = "gfx90a:xnack-";
    payload.codeObject = patternBytes(codeSize, 0x31);
    payload.sha256 = utilities::sha256Hex(payload.codeObject.data(), payload.codeObject.size());
    payload.recordedSignature
        = {KernelArgument{"global_buffer", 8, 0, ""}, KernelArgument{"by_value", 4, 8, "count"}};
    payload.providerVersion = "9.8.7";
    return payload;
}

std::vector<uint8_t> bodyOf(const std::vector<uint8_t>& payload)
{
    return {payload.begin() + static_cast<std::ptrdiff_t>(HEADER_SIZE), payload.end()};
}

// Puts a header with the matching digest in front of body.
std::vector<uint8_t> seal(const std::vector<uint8_t>& body)
{
    std::vector<uint8_t> bytes
        = encodeIngestorPlanHeader(IngestorPlanKind::SINGLE_KERNEL_PLAN,
                                   detail::computeIngestorPlanBodyDigest(body.data(), body.size()));
    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}

// Writes the header digest again after a test changes body bytes.
void reseal(std::vector<uint8_t>& payload)
{
    payload = seal(bodyOf(payload));
}

// The offset of the code bytes from the start of the body.
size_t codeObjectOffset(const std::vector<uint8_t>& payload)
{
    std::vector<detail::IngestorPlanAlignedBlock> storage;
    const uint8_t* body = detail::alignedIngestorPlanBody(
        payload.data() + HEADER_SIZE, payload.size() - HEADER_SIZE, storage);
    const auto* plan = fb::GetIngestorPlan(body);
    return static_cast<size_t>(plan->kernel()->code_object()->data() - body);
}

IngestorPlanPayload decode(const std::vector<uint8_t>& payload)
{
    return decodeIngestorPlan(payload.data(), payload.size());
}

void expectDecodeRefusal(const std::vector<uint8_t>& payload,
                         hipdnnPluginStatus_t status,
                         const std::string& phrase)
{
    expectIngestorPlanRefusal([&]() { decode(payload); }, status, phrase);
}

// One launch value as the raw builder writes it. A type other than NONE gets an Int64Value
// table as its data when withData is true.
struct RawLaunchValue
{
    std::string name;
    fb::LaunchValueData type = fb::LaunchValueData::Int64Value;
    bool withData = true;
};

// The body fields a raw build can set to values that encodeIngestorPlan never writes.
struct RawPlan
{
    std::optional<int64_t> engineId = 11;
    std::optional<uint64_t> workspaceBytes = 256;
    std::vector<int64_t> runtimePassByValueUids;
    std::vector<RawLaunchValue> values = {RawLaunchValue{"count"}};
    fb::SourceKind sourceKind = fb::SourceKind::EMBEDDED_SOURCE;
    std::string sha256 = std::string(64, 'a');
};

// Builds a sealed payload with the generated builder, without the writer's checks.
std::vector<uint8_t> buildRawPayload(const RawPlan& raw)
{
    flatbuffers::FlatBufferBuilder builder;

    const auto signature
        = builder.CreateVector(std::vector<flatbuffers::Offset<fb::KernelArgument>>{});
    const std::vector<uint8_t> code = patternBytes(16, 0x55);
    builder.ForceVectorAlignment(code.size(), sizeof(uint8_t), ALIGNMENT);
    const auto codeObject = builder.CreateVector(code);
    const auto symbol = builder.CreateString("raw_kernel");
    const auto target = builder.CreateString("gfx942");
    const auto sha256 = builder.CreateString(raw.sha256);
    const auto kernel = fb::CreateKernelImage(
        builder, raw.sourceKind, symbol, target, sha256, signature, codeObject);

    std::vector<flatbuffers::Offset<fb::LaunchValue>> values;
    for(const RawLaunchValue& value : raw.values)
    {
        flatbuffers::Offset<void> data;
        if(value.type != fb::LaunchValueData::NONE && value.withData)
        {
            data = fb::CreateInt64Value(builder, 5).Union();
        }
        const auto name = builder.CreateString(value.name);
        values.push_back(fb::CreateLaunchValue(builder, name, value.type, data));
    }
    const auto valuesVector = builder.CreateVector(values);
    const auto dispatchSymbol = builder.CreateString("raw.dispatch.v1");
    const auto launch = fb::CreateLaunchInputs(builder, dispatchSymbol, valuesVector);
    const auto uids = builder.CreateVector(raw.runtimePassByValueUids);
    const fb::Uuid kernelId;

    fb::IngestorPlanBuilder plan(builder);
    if(raw.engineId.has_value())
    {
        plan.add_engine_id(*raw.engineId);
    }
    plan.add_kernel_id(&kernelId);
    if(raw.workspaceBytes.has_value())
    {
        plan.add_workspace_bytes(*raw.workspaceBytes);
    }
    plan.add_runtime_pass_by_value_uids(uids);
    plan.add_launch(launch);
    plan.add_kernel(kernel);
    fb::FinishIngestorPlanBuffer(builder, plan.Finish());

    const uint8_t* body = builder.GetBufferPointer();
    return seal(std::vector<uint8_t>(body, body + builder.GetSize()));
}

class TestIngestorPlanCodecRoundTrip : public ::testing::TestWithParam<KernelSourceKind>
{
};

// The codec writes every source kind. Whether a provider saves or loads a kind is decided
// outside the codec.
TEST_P(TestIngestorPlanCodecRoundTrip, RoundTripsEveryField)
{
    const IngestorPlanPayload payload = makePayload(GetParam());
    const IngestorPlanPayload decoded = decode(encodeIngestorPlan(payload));

    EXPECT_EQ(decoded.engineId, payload.engineId);
    EXPECT_EQ(decoded.kernelId, payload.kernelId);
    EXPECT_EQ(decoded.workspaceBytes, payload.workspaceBytes);
    EXPECT_EQ(decoded.runtimePassByValueUids, payload.runtimePassByValueUids);
    EXPECT_EQ(decoded.dispatchSymbol, payload.dispatchSymbol);
    EXPECT_EQ(decoded.launchValues.size(), 5U);
    EXPECT_TRUE(decoded.launchValues == payload.launchValues);
    EXPECT_EQ(decoded.sourceKind, payload.sourceKind);
    EXPECT_EQ(decoded.symbol, payload.symbol);
    EXPECT_EQ(decoded.target, payload.target);
    EXPECT_EQ(decoded.sha256, payload.sha256);
    EXPECT_TRUE(detail::sameKernelSignature(decoded.recordedSignature, payload.recordedSignature));
    EXPECT_EQ(decoded.codeObject, payload.codeObject);
    EXPECT_EQ(decoded.providerVersion, payload.providerVersion);
    EXPECT_TRUE(decoded == payload);
}

INSTANTIATE_TEST_SUITE_P(SourceKinds,
                         TestIngestorPlanCodecRoundTrip,
                         ::testing::Values(KernelSourceKind::EMBEDDED_SOURCE,
                                           KernelSourceKind::KPACK,
                                           KernelSourceKind::HSACO_FILE,
                                           KernelSourceKind::ROCKE_BUILDER),
                         [](const ::testing::TestParamInfo<KernelSourceKind>& info) {
                             switch(info.param)
                             {
                             case KernelSourceKind::EMBEDDED_SOURCE:
                                 return std::string("EMBEDDED_SOURCE");
                             case KernelSourceKind::KPACK:
                                 return std::string("KPACK");
                             case KernelSourceKind::HSACO_FILE:
                                 return std::string("HSACO_FILE");
                             case KernelSourceKind::ROCKE_BUILDER:
                                 return std::string("ROCKE_BUILDER");
                             default:
                                 return std::to_string(info.index);
                             }
                         });

TEST(TestIngestorPlanCodec, RoundTripsEmptyOptionalContent)
{
    IngestorPlanPayload payload = makePayload(KernelSourceKind::EMBEDDED_SOURCE, 0);
    payload.launchValues.clear();
    payload.recordedSignature.clear();
    payload.providerVersion.clear();
    payload.target.clear();
    EXPECT_TRUE(decode(encodeIngestorPlan(payload)) == payload);
}

TEST(TestIngestorPlanCodec, RoundTripsADoubleLaunchValueExactly)
{
    IngestorPlanPayload payload = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    payload.launchValues = {{"x", 0.1}};
    const IngestorPlanPayload decoded = decode(encodeIngestorPlan(payload));

    const double expected = 0.1;
    const double actual = std::get<double>(decoded.launchValues.at("x"));
    uint64_t expectedBits = 0;
    uint64_t actualBits = 0;
    std::memcpy(&expectedBits, &expected, sizeof(expected));
    std::memcpy(&actualBits, &actual, sizeof(actual));
    EXPECT_EQ(actualBits, expectedBits);
}

TEST(TestIngestorPlanCodec, BodyStartsAtTheHeaderSizeAndCodeIsSixteenByteAligned)
{
    for(size_t codeSize = 1; codeSize <= 17; ++codeSize)
    {
        SCOPED_TRACE("code size " + std::to_string(codeSize));
        const std::vector<uint8_t> payload
            = encodeIngestorPlan(makePayload(KernelSourceKind::EMBEDDED_SOURCE, codeSize));

        const IngestorPlanHeader header = decodeIngestorPlanHeader(payload.data(), payload.size());
        EXPECT_EQ(static_cast<size_t>(header.headerSize), HEADER_SIZE);
        EXPECT_TRUE(fb::IngestorPlanBufferHasIdentifier(payload.data() + header.headerSize));
        EXPECT_EQ(codeObjectOffset(payload) % ALIGNMENT, 0U);
        EXPECT_EQ(decode(payload).codeObject.size(), codeSize);
    }
}

TEST(TestIngestorPlanCodec, ReadsABodyThatStartsMisaligned)
{
    const IngestorPlanPayload payload = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    const std::vector<uint8_t> encoded = encodeIngestorPlan(payload);

    for(size_t shift = 1; shift < ALIGNMENT; ++shift)
    {
        SCOPED_TRACE("shift " + std::to_string(shift));
        std::vector<detail::IngestorPlanAlignedBlock> blocks((encoded.size() / ALIGNMENT) + 2);
        uint8_t* start = reinterpret_cast<uint8_t*>(blocks.data()) + shift;
        std::memcpy(start, encoded.data(), encoded.size());

        EXPECT_TRUE(decodeIngestorPlan(start, encoded.size()) == payload);

        std::vector<detail::IngestorPlanAlignedBlock> storage;
        const uint8_t* aligned = detail::alignedIngestorPlanBody(
            start + HEADER_SIZE, encoded.size() - HEADER_SIZE, storage);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned) % ALIGNMENT, 0U);
        EXPECT_EQ(std::memcmp(aligned, start + HEADER_SIZE, encoded.size() - HEADER_SIZE), 0);
    }
}

TEST(TestIngestorPlanCodec, RejectsAChangedBodyByteAtTheDigestStage)
{
    const std::vector<uint8_t> encoded
        = encodeIngestorPlan(makePayload(KernelSourceKind::EMBEDDED_SOURCE));

    std::vector<uint8_t> changedCode = encoded;
    const size_t codeByte = HEADER_SIZE + codeObjectOffset(encoded);
    changedCode[codeByte] = static_cast<uint8_t>(changedCode[codeByte] ^ 0x01U);
    expectDecodeRefusal(changedCode, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "body digest mismatch");

    std::vector<uint8_t> changedRoot = encoded;
    changedRoot[HEADER_SIZE] = static_cast<uint8_t>(changedRoot[HEADER_SIZE] ^ 0x01U);
    expectDecodeRefusal(changedRoot, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "body digest mismatch");
}

TEST(TestIngestorPlanCodec, RejectsEveryTruncation)
{
    const std::vector<uint8_t> encoded
        = encodeIngestorPlan(makePayload(KernelSourceKind::EMBEDDED_SOURCE, 5));

    for(size_t size = 0; size < encoded.size(); ++size)
    {
        SCOPED_TRACE("prefix of " + std::to_string(size) + " bytes");
        // An exact-size heap copy lets ASAN see any read past the end of the prefix.
        const std::vector<uint8_t> prefix(encoded.begin(),
                                          encoded.begin() + static_cast<std::ptrdiff_t>(size));
        expectIngestorPlanRefusal([&]() { decodeIngestorPlan(prefix.data(), prefix.size()); },
                                  HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                  "");
    }
}

TEST(TestIngestorPlanCodec, VerifierRejectsACorruptRootOffsetWithAValidDigest)
{
    std::vector<uint8_t> payload
        = encodeIngestorPlan(makePayload(KernelSourceKind::EMBEDDED_SOURCE));
    const auto outside = static_cast<flatbuffers::uoffset_t>(payload.size());
    std::memcpy(payload.data() + HEADER_SIZE, &outside, sizeof(outside));
    reseal(payload);
    expectDecodeRefusal(payload, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "verifier");
}

TEST(TestIngestorPlanCodec, RejectsABodyWithAnotherFileIdentifier)
{
    std::vector<uint8_t> payload
        = encodeIngestorPlan(makePayload(KernelSourceKind::EMBEDDED_SOURCE));
    uint8_t* identifier = payload.data() + HEADER_SIZE + FILE_IDENTIFIER_OFFSET;
    ASSERT_EQ(std::memcmp(identifier, PLAN_IDENTIFIER.data(), PLAN_IDENTIFIER.size()), 0);
    std::memcpy(identifier, FOREIGN_IDENTIFIER.data(), FOREIGN_IDENTIFIER.size());
    reseal(payload);
    expectDecodeRefusal(payload, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "verifier");
}

TEST(TestIngestorPlanCodec, RejectsAnOversizeBody)
{
    const auto maxSize = static_cast<size_t>(FLATBUFFERS_MAX_BUFFER_SIZE);
    expectIngestorPlanRefusal([&]() { detail::checkIngestorPlanBodySize(maxSize); },
                              HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                              "body size");
    EXPECT_NO_THROW(detail::checkIngestorPlanBodySize(maxSize - 1));

    const size_t minSize = FLATBUFFERS_MIN_BUFFER_SIZE;
    expectIngestorPlanRefusal([&]() { detail::checkIngestorPlanBodySize(minSize - 1); },
                              HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                              "body size");
    EXPECT_NO_THROW(detail::checkIngestorPlanBodySize(minSize));
}

TEST(TestIngestorPlanCodec, RefusesANonEmptyRuntimePassByValueList)
{
    IngestorPlanPayload payload = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    payload.runtimePassByValueUids = {7};
    expectDecodeRefusal(encodeIngestorPlan(payload),
                        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                        "pass-by-value tensors (UIDs 7)");
}

TEST(TestIngestorPlanCodec, DamageIsReportedBeforeIncompatibility)
{
    RawPlan raw;
    raw.runtimePassByValueUids = {7};
    raw.values = {RawLaunchValue{""}};
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "has no name");

    raw.values = {RawLaunchValue{"count"}};
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "UIDs 7");
}

TEST(TestIngestorPlanCodec, RejectsMalformedLaunchValues)
{
    RawPlan raw;

    raw.values = {RawLaunchValue{"none", fb::LaunchValueData::NONE}};
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "unknown value type 0");

    raw.values = {RawLaunchValue{"future", static_cast<fb::LaunchValueData>(9)}};
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "unknown value type 9");

    raw.values = {RawLaunchValue{""}};
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "has no name");

    raw.values = {RawLaunchValue{"twice"}, RawLaunchValue{"twice"}};
    expectDecodeRefusal(
        buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "occurs more than once");

    raw.values = {RawLaunchValue{"empty", fb::LaunchValueData::Int64Value, false}};
    expectDecodeRefusal(buildRawPayload(raw), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "no data");

    raw.values = {RawLaunchValue{"count"}};
    EXPECT_EQ(std::get<int64_t>(decode(buildRawPayload(raw)).launchValues.at("count")), 5);
}

TEST(TestIngestorPlanCodec, RejectsAMissingRequiredScalar)
{
    RawPlan withoutEngine;
    withoutEngine.engineId.reset();
    expectDecodeRefusal(
        buildRawPayload(withoutEngine), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "engine_id");

    RawPlan withoutWorkspace;
    withoutWorkspace.workspaceBytes.reset();
    expectDecodeRefusal(
        buildRawPayload(withoutWorkspace), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "workspace_bytes");
}

TEST(TestIngestorPlanCodec, RejectsAMalformedKernelDigest)
{
    RawPlan shortDigest;
    shortDigest.sha256 = std::string(63, 'a');
    expectDecodeRefusal(
        buildRawPayload(shortDigest), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "kernel digest");

    RawPlan upperCaseDigest;
    upperCaseDigest.sha256 = std::string(64, 'A');
    expectDecodeRefusal(
        buildRawPayload(upperCaseDigest), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "kernel digest");
}

TEST(TestIngestorPlanCodec, RejectsAnUnsetOrUnknownSourceKind)
{
    RawPlan unset;
    unset.sourceKind = fb::SourceKind::UNSET;
    expectDecodeRefusal(
        buildRawPayload(unset), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "source kind 0");

    RawPlan unknown;
    unknown.sourceKind = static_cast<fb::SourceKind>(9);
    expectDecodeRefusal(
        buildRawPayload(unknown), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "source kind 9");
}

TEST(TestIngestorPlanCodec, WriterRejectsInputsItOwnsAsAnInternalError)
{
    const auto expectInternalError = [](const IngestorPlanPayload& payload,
                                        const std::string& phrase) {
        try
        {
            encodeIngestorPlan(payload);
            ADD_FAILURE() << "expected an internal error that contains '" << phrase << "'";
        }
        catch(const HipdnnPluginException& error)
        {
            EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << error.getMessage();
            EXPECT_NE(error.getMessage().find(phrase), std::string::npos) << error.getMessage();
        }
    };

    IngestorPlanPayload noSymbol = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    noSymbol.symbol.clear();
    expectInternalError(noSymbol, "kernel symbol is empty");

    IngestorPlanPayload noDispatch = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    noDispatch.dispatchSymbol.clear();
    expectInternalError(noDispatch, "dispatch symbol is empty");

    IngestorPlanPayload badDigest = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    badDigest.sha256 = std::string(64, 'A');
    expectInternalError(badDigest, "kernel digest");

    IngestorPlanPayload badKind = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    badKind.sourceKind = static_cast<KernelSourceKind>(99);
    expectInternalError(badKind, "source kind 99");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
