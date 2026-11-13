// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
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
constexpr size_t SHA256_HEX_LENGTH = INGESTOR_PLAN_DIGEST_SIZE * 2;
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
    for(size_t index = 0; index < payload.kernelDescriptorId.size(); ++index)
    {
        payload.kernelDescriptorId[index] = static_cast<uint8_t>(0x10U + index);
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
    const auto* plan = fb::GetExecutionPlan(body);
    return static_cast<size_t>(plan->kernel_image()->code_object()->data() - body);
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

void expectEncodeInternalError(const IngestorPlanPayload& payload, const std::string& phrase)
{
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
}

// One launch value as the raw builder writes it. A type other than NONE gets an Int64Value
// table as its data when withData is true.
struct RawLaunchValue
{
    std::string name;
    fb::LaunchValueData type = fb::LaunchValueData::Int64Value;
    bool withData = true;
};

// The code bytes of every raw build.
std::vector<uint8_t> rawCodeObject()
{
    return patternBytes(16, 0x55);
}

std::string rawCodeObjectSha256()
{
    const std::vector<uint8_t> code = rawCodeObject();
    return utilities::sha256Hex(code.data(), code.size());
}

// The body fields a raw build can set to values that encodeIngestorPlan never writes.
struct RawPlan
{
    std::optional<std::string> providerVersion = std::string("0.0.1");
    std::optional<uint64_t> workspaceBytes = 256;
    std::vector<int64_t> runtimePassByValueUids;
    std::vector<RawLaunchValue> values = {RawLaunchValue{"count"}};
    fb::SourceKind sourceKind = fb::SourceKind::EMBEDDED_SOURCE;
    std::string sha256 = rawCodeObjectSha256();
};

// Builds a sealed payload with the generated builder, without the writer's checks.
std::vector<uint8_t> buildRawPayload(const RawPlan& raw)
{
    flatbuffers::FlatBufferBuilder builder;

    const auto signature
        = builder.CreateVector(std::vector<flatbuffers::Offset<fb::KernelArgument>>{});
    const std::vector<uint8_t> code = rawCodeObject();
    builder.ForceVectorAlignment(code.size(), sizeof(uint8_t), ALIGNMENT);
    const auto codeObject = builder.CreateVector(code);
    const auto symbol = builder.CreateString("raw_kernel");
    const auto target = builder.CreateString("gfx942");
    const auto sha256 = builder.CreateString(raw.sha256);
    const auto kernelImage = fb::CreateKernelImage(
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
    const auto dispatchData = fb::CreateDispatchData(builder, dispatchSymbol, valuesVector);
    const auto uids = builder.CreateVector(raw.runtimePassByValueUids);
    flatbuffers::Offset<flatbuffers::String> providerVersion;
    if(raw.providerVersion.has_value())
    {
        providerVersion = builder.CreateString(*raw.providerVersion);
    }
    const fb::Uuid kernelDescriptorId;

    fb::ExecutionPlanBuilder plan(builder);
    plan.add_kernel_descriptor_id(&kernelDescriptorId);
    if(raw.workspaceBytes.has_value())
    {
        plan.add_workspace_bytes(*raw.workspaceBytes);
    }
    plan.add_runtime_pass_by_value_uids(uids);
    plan.add_dispatch_data(dispatchData);
    plan.add_kernel_image(kernelImage);
    if(raw.providerVersion.has_value())
    {
        plan.add_provider_version(providerVersion);
    }
    // End the table directly. The generated Finish() asserts that every required field is set.
    const flatbuffers::Offset<fb::ExecutionPlan> root(builder.EndTable(plan.start_));
    fb::FinishExecutionPlanBuffer(builder, root);

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

    EXPECT_EQ(decoded.kernelDescriptorId, payload.kernelDescriptorId);
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

TEST(TestIngestorPlanCodec, BufferEncodingRoundTrips)
{
    for(const KernelSourceKind sourceKind : {KernelSourceKind::EMBEDDED_SOURCE,
                                             KernelSourceKind::KPACK,
                                             KernelSourceKind::HSACO_FILE,
                                             KernelSourceKind::ROCKE_BUILDER})
    {
        SCOPED_TRACE("source kind " + std::to_string(static_cast<int>(sourceKind)));
        const IngestorPlanPayload payload = makePayload(sourceKind);
        const EncodedIngestorPlan encoded = encodeIngestorPlanToBuffer(payload);

        ASSERT_NE(encoded.bytes.get(), nullptr);
        ASSERT_GT(encoded.size, HEADER_SIZE);
        const IngestorPlanHeader header
            = decodeIngestorPlanHeader(encoded.bytes.get(), encoded.size);
        EXPECT_EQ(static_cast<size_t>(header.headerSize), HEADER_SIZE);

        // The body is every byte after the header, and the header digest covers all of it.
        const uint8_t* body = encoded.bytes.get() + HEADER_SIZE;
        const size_t bodySize = encoded.size - HEADER_SIZE;
        EXPECT_TRUE(fb::ExecutionPlanBufferHasIdentifier(body));
        EXPECT_EQ(header.bodyDigest, detail::computeIngestorPlanBodyDigest(body, bodySize));

        EXPECT_TRUE(decodeIngestorPlan(encoded.bytes.get(), encoded.size) == payload);
    }
}

TEST(TestIngestorPlanCodec, RoundTripsEmptyOptionalContent)
{
    IngestorPlanPayload payload = makePayload(KernelSourceKind::EMBEDDED_SOURCE, 0);
    payload.launchValues.clear();
    payload.recordedSignature.clear();
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
        EXPECT_TRUE(fb::ExecutionPlanBufferHasIdentifier(payload.data() + header.headerSize));
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

// The checks that hold for every payload the reader returns, whatever a damaged body byte
// changed. A length read out of a damaged body cannot exceed the body it came from.
void expectSelfConsistentPayload(const IngestorPlanPayload& decoded, size_t bodySize)
{
    EXPECT_EQ(decoded.sha256.size(), SHA256_HEX_LENGTH);
    EXPECT_TRUE(std::all_of(decoded.sha256.begin(), decoded.sha256.end(), [](char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    })) << decoded.sha256;
    EXPECT_TRUE(decoded.runtimePassByValueUids.empty());
    EXPECT_LE(decoded.codeObject.size(), bodySize);
    EXPECT_LE(decoded.symbol.size(), bodySize);
    EXPECT_LE(decoded.target.size(), bodySize);
    EXPECT_LE(decoded.dispatchSymbol.size(), bodySize);
    EXPECT_LE(decoded.providerVersion.size(), bodySize);
    EXPECT_LE(decoded.recordedSignature.size(), bodySize);
    EXPECT_LE(decoded.launchValues.size(), bodySize);
    for(const auto& value : decoded.launchValues)
    {
        EXPECT_FALSE(value.first.empty());
        EXPECT_LE(value.first.size(), bodySize);
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
    catch(const HipdnnPluginException& error)
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
// A flip in an offset, a length, a vtable entry, the kernel digest or the code bytes is refused.
TEST(TestIngestorPlanCodec, RefusesOrReadsEveryResealedBodyByteFlip)
{
    // Covers every body byte. A larger stride samples the body instead.
    constexpr size_t STRIDE = 1;
    constexpr uint8_t FLIP_MASK = 0xFF;

    const std::vector<uint8_t> encoded
        = encodeIngestorPlan(makePayload(KernelSourceKind::EMBEDDED_SOURCE));
    const size_t bodySize = encoded.size() - HEADER_SIZE;

    size_t refusedCount = 0;
    size_t readCount = 0;
    for(size_t offset = 0; offset < bodySize; offset += STRIDE)
    {
        SCOPED_TRACE("body byte " + std::to_string(offset) + " of " + std::to_string(bodySize));
        std::vector<uint8_t> damaged = encoded;
        const size_t index = HEADER_SIZE + offset;
        damaged[index] = static_cast<uint8_t>(damaged[index] ^ FLIP_MASK);
        reseal(damaged);

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
    RawPlan withoutWorkspace;
    withoutWorkspace.workspaceBytes.reset();
    expectDecodeRefusal(
        buildRawPayload(withoutWorkspace), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "workspace_bytes");
}

TEST(TestIngestorPlanCodec, VerifierRejectsABodyWithoutAProviderVersion)
{
    RawPlan withoutProviderVersion;
    withoutProviderVersion.providerVersion.reset();
    expectDecodeRefusal(
        buildRawPayload(withoutProviderVersion), HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "verifier");

    const IngestorPlanPayload decoded = decode(buildRawPayload(RawPlan{}));
    EXPECT_EQ(decoded.providerVersion, "0.0.1");
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

TEST(TestIngestorPlanCodec, RefusesAKernelDigestThatDoesNotMatchTheCodeObject)
{
    IngestorPlanPayload payload = makePayload(KernelSourceKind::KPACK);
    const std::string otherDigest(64, 'a');
    ASSERT_NE(payload.sha256, otherDigest);
    payload.sha256 = otherDigest;

    const std::string refusal
        = expectIngestorPlanRefusal([&]() { decode(encodeIngestorPlan(payload)); },
                                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "does not match the code object");
    EXPECT_NE(refusal.find(otherDigest), std::string::npos) << refusal;
    const std::string actual
        = utilities::sha256Hex(payload.codeObject.data(), payload.codeObject.size());
    EXPECT_NE(refusal.find(actual), std::string::npos) << refusal;
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
    IngestorPlanPayload noSymbol = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    noSymbol.symbol.clear();
    expectEncodeInternalError(noSymbol, "kernel symbol is empty");

    IngestorPlanPayload noDispatch = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    noDispatch.dispatchSymbol.clear();
    expectEncodeInternalError(noDispatch, "dispatch symbol is empty");

    IngestorPlanPayload badDigest = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    badDigest.sha256 = std::string(64, 'A');
    expectEncodeInternalError(badDigest, "kernel digest");

    IngestorPlanPayload badKind = makePayload(KernelSourceKind::EMBEDDED_SOURCE);
    badKind.sourceKind = static_cast<KernelSourceKind>(99);
    expectEncodeInternalError(badKind, "source kind 99");
}

TEST(TestIngestorPlanCodec, WriterRefusesAnEmptyProviderVersion)
{
    IngestorPlanPayload noProviderVersion = makePayload(KernelSourceKind::KPACK);
    noProviderVersion.providerVersion.clear();
    expectEncodeInternalError(noProviderVersion, "provider version is empty");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
