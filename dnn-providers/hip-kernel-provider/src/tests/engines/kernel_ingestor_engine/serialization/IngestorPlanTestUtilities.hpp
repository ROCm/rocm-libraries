// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/ingestor_plan_generated.h"
#include "utilities/Digest.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

// Runs action and expects a plugin exception. Checks the status, the message prefix and a
// phrase that names the failing check. An empty prefix matches every message. Any other
// exception fails the test. Returns the message, or an empty string when no plugin
// exception occurs.
template <typename Action>
std::string expectPluginException(Action&& action,
                                  hipdnnPluginStatus_t status,
                                  std::string_view prefix,
                                  const std::string& phrase)
{
    try
    {
        action();
        ADD_FAILURE() << "expected a refusal that contains '" << phrase << "'";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        const std::string message = error.getMessage();
        EXPECT_EQ(error.getStatus(), status) << message;
        EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
        EXPECT_NE(message.find(phrase), std::string::npos) << message;
        return message;
    }
    catch(const std::exception& error)
    {
        ADD_FAILURE() << "expected a refusal that contains '" << phrase << "', got "
                      << typeid(error).name() << ": " << error.what();
    }
    catch(...)
    {
        ADD_FAILURE() << "expected a refusal that contains '" << phrase
                      << "', got an exception of an unknown type";
    }
    return {};
}

// Runs action and expects an ingestor plan load refusal with the category prefix of status.
template <typename Action>
std::string expectIngestorPlanRefusal(Action&& action,
                                      hipdnnPluginStatus_t status,
                                      const std::string& phrase)
{
    return expectPluginException(std::forward<Action>(action),
                                 status,
                                 status == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE
                                     ? INGESTOR_PLAN_INCOMPATIBLE_PREFIX
                                     : INGESTOR_PLAN_DAMAGED_PREFIX,
                                 phrase);
}

// Runs action and expects an ingestor plan save refusal with the category prefix of status.
template <typename Action>
std::string expectIngestorPlanSaveRefusal(Action&& action,
                                          hipdnnPluginStatus_t status,
                                          const std::string& phrase)
{
    return expectPluginException(std::forward<Action>(action),
                                 status,
                                 status == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE
                                     ? INGESTOR_PLAN_SAVE_INCOMPATIBLE_PREFIX
                                     : INGESTOR_PLAN_SAVE_DAMAGED_PREFIX,
                                 phrase);
}

// Runs action and expects an internal error, which has no category prefix.
template <typename Action>
std::string expectPluginInternalError(Action&& action, const std::string& phrase)
{
    return expectPluginException(
        std::forward<Action>(action), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "", phrase);
}

// The FlatBuffer body of an encoded payload: everything after the 48-byte header.
inline std::vector<uint8_t> bodyOfIngestorPlan(const std::vector<uint8_t>& payload)
{
    return {payload.begin() + static_cast<std::ptrdiff_t>(INGESTOR_PLAN_HEADER_SIZE),
            payload.end()};
}

// Puts a header of the given kind, with the matching digest, in front of body.
inline std::vector<uint8_t> sealIngestorPlanBody(IngestorPlanKind kind,
                                                 const std::vector<uint8_t>& body)
{
    std::vector<uint8_t> bytes = encodeIngestorPlanHeader(
        kind, detail::computeIngestorPlanBodyDigest(body.data(), body.size()));
    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}

// Writes the header digest again after a test changes body bytes. Keeps the kind of the
// payload's own header.
inline void resealIngestorPlan(std::vector<uint8_t>& payload)
{
    const IngestorPlanKind kind = decodeIngestorPlanHeader(payload.data(), payload.size()).kind;
    payload = sealIngestorPlanBody(kind, bodyOfIngestorPlan(payload));
}

constexpr size_t HEADER_SIZE = INGESTOR_PLAN_HEADER_SIZE;
constexpr size_t ALIGNMENT = INGESTOR_PLAN_BODY_ALIGNMENT;
constexpr size_t SHA256_HEX_LENGTH = INGESTOR_PLAN_DIGEST_SIZE * 2;

// Deterministic bytes: seed plus a stride of 7 per index.
inline std::vector<uint8_t> patternBytes(size_t size, uint8_t seed)
{
    std::vector<uint8_t> bytes(size);
    for(size_t index = 0; index < size; ++index)
    {
        bytes[index] = static_cast<uint8_t>(seed + (index * 7U));
    }
    return bytes;
}

inline std::string sha256Of(const std::vector<uint8_t>& bytes)
{
    return utilities::sha256Hex(bytes.data(), bytes.size());
}

// Launch values of every MetadataValue alternative.
inline hipdnn_plugin_sdk::ingestor::MetadataValues sampleLaunchValues()
{
    return {
        {"flag", true},
        {"count", int64_t{-42}},
        {"scale", 0.25},
        {"layout", std::string("bshd")},
        {"shape", std::vector<int64_t>{2, 3, 5}},
    };
}

// One launch value as the raw builder writes it. A type other than NONE gets an Int64Value
// table as its data when withData is true.
struct RawLaunchValue
{
    std::string name;
    fb::LaunchValueData type = fb::LaunchValueData::Int64Value;
    bool withData = true;
};

// Writes the launch values with the generated builder, without the writer's checks.
inline flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fb::LaunchValue>>>
    writeRawLaunchValues(flatbuffers::FlatBufferBuilder& builder,
                         const std::vector<RawLaunchValue>& values)
{
    std::vector<flatbuffers::Offset<fb::LaunchValue>> offsets;
    offsets.reserve(values.size());
    for(const RawLaunchValue& value : values)
    {
        flatbuffers::Offset<void> data;
        if(value.type != fb::LaunchValueData::NONE && value.withData)
        {
            data = fb::CreateInt64Value(builder, 5).Union();
        }
        const auto name = builder.CreateString(value.name);
        offsets.push_back(fb::CreateLaunchValue(builder, name, value.type, data));
    }
    return builder.CreateVector(offsets);
}

// ---------------------------------------------------------------------------
// Checked-in golden samples
// ---------------------------------------------------------------------------

// The code object every golden sample carries: a few bytes standing in for an HSACO, so a
// checked-in sample stays small. Nothing on this path loads it.
inline constexpr std::array<uint8_t, 8> SAMPLE_CODE_OBJECT
    = {0x7F, 0x45, 0x4C, 0x46, 0x02, 0x01, 0x01, 0x40};

// A golden sample's payload: the fields outside the dispatch contract are fixed across the
// samples of a file, so they differ only in the contract under test.
inline IngestorPlanPayload
    goldenSamplePayload(std::string dispatchSymbol,
                        hipdnn_plugin_sdk::ingestor::MetadataValues launchValues,
                        std::string target,
                        std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> recordedSignature)
{
    IngestorPlanPayload payload;
    for(size_t index = 0; index < payload.kernelDescriptorId.size(); ++index)
    {
        payload.kernelDescriptorId[index] = static_cast<uint8_t>(0x20U + index);
    }
    payload.workspaceBytes = 4096;
    payload.dispatchSymbol = std::move(dispatchSymbol);
    payload.launchValues = std::move(launchValues);
    payload.sourceKind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
    payload.symbol = "golden_kernel";
    payload.target = std::move(target);
    payload.recordedSignature = std::move(recordedSignature);
    payload.codeObject.assign(SAMPLE_CODE_OBJECT.begin(), SAMPLE_CODE_OBJECT.end());
    payload.sha256 = utilities::sha256Hex(payload.codeObject.data(), payload.codeObject.size());
    payload.providerVersion = "1.0.0";
    return payload;
}

// Prints the payload fields that identify a sample, for a failed comparison.
inline std::string describeGoldenPayload(const IngestorPlanPayload& payload)
{
    return "target=" + payload.target + " symbol=" + payload.symbol
           + " dispatchSymbol=" + payload.dispatchSymbol + " sha256=" + payload.sha256
           + " workspaceBytes=" + std::to_string(payload.workspaceBytes) + " signature=["
           + hipdnn_plugin_sdk::ingestor::describeKernelSignature(payload.recordedSignature)
           + "] values=" + std::to_string(payload.launchValues.size());
}

// Decodes a sample, compares the whole payload with the expected payload, and hands back the
// dispatch contract it carries.
template <size_t SIZE>
hipdnn_plugin_sdk::ingestor::SavedLaunchInputs
    readGoldenSample(const std::array<uint8_t, SIZE>& golden, const IngestorPlanPayload& expected)
{
    const IngestorPlanPayload decoded = decodeIngestorPlan(golden.data(), golden.size());
    EXPECT_TRUE(decoded == expected) << "decoded:  " << describeGoldenPayload(decoded) << "\n"
                                     << "expected: " << describeGoldenPayload(expected);
    return hipdnn_plugin_sdk::ingestor::SavedLaunchInputs{decoded.dispatchSymbol,
                                                          decoded.launchValues};
}

// Writes @p bytes as a C++ array named @p name, ready to paste beside the samples already
// checked in.
inline void printGoldenSample(std::string_view name, const std::vector<uint8_t>& bytes)
{
    constexpr std::string_view DIGITS = "0123456789ABCDEF";
    constexpr size_t PER_LINE = 12;
    std::cout << "constexpr std::array<uint8_t, " << bytes.size() << "> " << name << " = {\n";
    for(size_t index = 0; index < bytes.size(); ++index)
    {
        const bool lineStart = index % PER_LINE == 0;
        const bool last = index + 1 == bytes.size();
        std::cout << (lineStart ? "    0x" : " 0x")
                  << DIGITS[static_cast<size_t>(bytes[index] >> 4U)]
                  << DIGITS[static_cast<size_t>(bytes[index] & 0x0FU)] << (last ? "" : ",")
                  << ((last || index % PER_LINE == PER_LINE - 1) ? "\n" : "");
    }
    std::cout << "};\n\n";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
