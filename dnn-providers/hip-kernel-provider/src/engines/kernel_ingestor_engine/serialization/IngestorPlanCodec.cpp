// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/ingestor_plan_generated.h"
#include "utilities/Digest.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MetadataType;
using hipdnn_plugin_sdk::ingestor::MetadataValue;
using hipdnn_plugin_sdk::ingestor::MetadataValues;

constexpr size_t MAX_BODY_SIZE = static_cast<size_t>(FLATBUFFERS_MAX_BUFFER_SIZE);
constexpr size_t MIN_BODY_SIZE = FLATBUFFERS_MIN_BUFFER_SIZE;

constexpr size_t KERNEL_ID_SIZE = std::tuple_size_v<hipdnn_plugin_sdk::ingestor::DescriptorId>;

constexpr size_t SHA256_HEX_LENGTH = INGESTOR_PLAN_DIGEST_SIZE * 2;

// Builder space for everything other than the code bytes.
constexpr size_t BUILDER_HEADROOM = 4096;

constexpr size_t MESSAGE_TEXT_LIMIT = 80;

// Shortens a string from the payload for use in an error message.
std::string messageText(const std::string& text)
{
    if(text.size() <= MESSAGE_TEXT_LIMIT)
    {
        return text;
    }
    return text.substr(0, MESSAGE_TEXT_LIMIT) + "... (" + std::to_string(text.size())
           + " characters)";
}

[[noreturn]] void throwInternal(const std::string& message)
{
    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                "cannot encode the ingestor plan: " + message);
}

bool isLowercaseSha256Hex(const std::string& text)
{
    if(text.size() != SHA256_HEX_LENGTH)
    {
        return false;
    }
    for(const char character : text)
    {
        const bool digit = character >= '0' && character <= '9';
        const bool letter = character >= 'a' && character <= 'f';
        if(!digit && !letter)
        {
            return false;
        }
    }
    return true;
}

bool toFlatBufferSourceKind(KernelSourceKind kind, fb::SourceKind& result)
{
    switch(kind)
    {
    case KernelSourceKind::EMBEDDED_SOURCE:
        result = fb::SourceKind::EMBEDDED_SOURCE;
        return true;
    case KernelSourceKind::KPACK:
        result = fb::SourceKind::KPACK;
        return true;
    case KernelSourceKind::HSACO_FILE:
        result = fb::SourceKind::HSACO_FILE;
        return true;
    case KernelSourceKind::ROCKE_BUILDER:
        result = fb::SourceKind::ROCKE_BUILDER;
        return true;
    default:
        return false;
    }
}

bool fromFlatBufferSourceKind(fb::SourceKind kind, KernelSourceKind& result)
{
    switch(kind)
    {
    case fb::SourceKind::EMBEDDED_SOURCE:
        result = KernelSourceKind::EMBEDDED_SOURCE;
        return true;
    case fb::SourceKind::KPACK:
        result = KernelSourceKind::KPACK;
        return true;
    case fb::SourceKind::HSACO_FILE:
        result = KernelSourceKind::HSACO_FILE;
        return true;
    case fb::SourceKind::ROCKE_BUILDER:
        result = KernelSourceKind::ROCKE_BUILDER;
        return true;
    case fb::SourceKind::UNSET:
    default:
        return false;
    }
}

fb::SourceKind checkWriterInputs(const IngestorPlanPayload& payload)
{
    fb::SourceKind sourceKind = fb::SourceKind::UNSET;
    if(!toFlatBufferSourceKind(payload.sourceKind, sourceKind))
    {
        throwInternal("source kind " + std::to_string(static_cast<int>(payload.sourceKind))
                      + " is not a known kernel source kind");
    }
    if(payload.symbol.empty())
    {
        throwInternal("the kernel symbol is empty");
    }
    if(payload.dispatchSymbol.empty())
    {
        throwInternal("the dispatch symbol is empty");
    }
    if(!isLowercaseSha256Hex(payload.sha256))
    {
        throwInternal("the kernel digest '" + messageText(payload.sha256)
                      + "' is not 64 lowercase hex characters");
    }
    return sourceKind;
}

flatbuffers::Offset<fb::LaunchValue> writeLaunchValue(flatbuffers::FlatBufferBuilder& builder,
                                                      const std::string& name,
                                                      const MetadataValue& value)
{
    fb::LaunchValueData type = fb::LaunchValueData::NONE;
    flatbuffers::Offset<void> data;
    switch(hipdnn_plugin_sdk::ingestor::metadataTypeOf(value))
    {
    case MetadataType::BOOL:
        type = fb::LaunchValueData::BoolValue;
        data = fb::CreateBoolValue(builder, std::get<bool>(value)).Union();
        break;
    case MetadataType::INT:
        type = fb::LaunchValueData::Int64Value;
        data = fb::CreateInt64Value(builder, std::get<int64_t>(value)).Union();
        break;
    case MetadataType::FLOAT:
        type = fb::LaunchValueData::DoubleValue;
        data = fb::CreateDoubleValue(builder, std::get<double>(value)).Union();
        break;
    case MetadataType::STRING:
    {
        const auto text = builder.CreateString(std::get<std::string>(value));
        type = fb::LaunchValueData::StringValue;
        data = fb::CreateStringValue(builder, text).Union();
        break;
    }
    case MetadataType::INT_LIST:
    {
        const auto values = builder.CreateVector(std::get<std::vector<int64_t>>(value));
        type = fb::LaunchValueData::Int64ListValue;
        data = fb::CreateInt64ListValue(builder, values).Union();
        break;
    }
    default:
        throwInternal("launch value '" + messageText(name)
                      + "' has a type the payload format cannot hold");
    }
    const auto nameOffset = builder.CreateString(name);
    return fb::CreateLaunchValue(builder, nameOffset, type, data);
}

flatbuffers::Offset<fb::KernelImage> writeKernelImage(flatbuffers::FlatBufferBuilder& builder,
                                                      const IngestorPlanPayload& payload,
                                                      fb::SourceKind sourceKind)
{
    std::vector<flatbuffers::Offset<fb::KernelArgument>> arguments;
    arguments.reserve(payload.recordedSignature.size());
    for(const KernelArgument& argument : payload.recordedSignature)
    {
        const auto kind = builder.CreateString(argument.kind);
        flatbuffers::Offset<flatbuffers::String> name;
        if(!argument.name.empty())
        {
            name = builder.CreateString(argument.name);
        }
        arguments.push_back(
            fb::CreateKernelArgument(builder, kind, argument.size, argument.offset, name));
    }
    const auto signature = builder.CreateVector(arguments);

    // The low-level vector calls ignore the schema's force_align. Align the code bytes
    // directly before the vector is created.
    builder.ForceVectorAlignment(
        payload.codeObject.size(), sizeof(uint8_t), INGESTOR_PLAN_BODY_ALIGNMENT);
    const auto codeObject = builder.CreateVector(payload.codeObject);

    const auto symbol = builder.CreateString(payload.symbol);
    const auto target = builder.CreateString(payload.target);
    const auto sha256 = builder.CreateString(payload.sha256);
    return fb::CreateKernelImage(
        builder, sourceKind, symbol, target, sha256, signature, codeObject);
}

const fb::IngestorPlan* verifyIngestorPlanBody(const uint8_t* alignedBody, size_t bodySize)
{
    const flatbuffers::Verifier::Options options{};
    flatbuffers::Verifier verifier(alignedBody, bodySize, options);
    if(!fb::VerifyIngestorPlanBuffer(verifier))
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "the FlatBuffers verifier rejects the payload body, or the body "
                           "does not carry the HKSP file identifier");
    }
    return fb::GetIngestorPlan(alignedBody);
}

[[noreturn]] void refuseMissingLaunchData(const std::string& name)
{
    refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                       "launch value '" + messageText(name) + "' has a type but no data");
}

MetadataValue readLaunchValueData(const fb::LaunchValue& value, const std::string& name)
{
    switch(value.value_type())
    {
    case fb::LaunchValueData::BoolValue:
    {
        const auto* data = value.value_as_BoolValue();
        if(data == nullptr)
        {
            refuseMissingLaunchData(name);
        }
        return data->value();
    }
    case fb::LaunchValueData::Int64Value:
    {
        const auto* data = value.value_as_Int64Value();
        if(data == nullptr)
        {
            refuseMissingLaunchData(name);
        }
        return data->value();
    }
    case fb::LaunchValueData::DoubleValue:
    {
        const auto* data = value.value_as_DoubleValue();
        if(data == nullptr)
        {
            refuseMissingLaunchData(name);
        }
        return data->value();
    }
    case fb::LaunchValueData::StringValue:
    {
        const auto* data = value.value_as_StringValue();
        if(data == nullptr)
        {
            refuseMissingLaunchData(name);
        }
        return data->value()->str();
    }
    case fb::LaunchValueData::Int64ListValue:
    {
        const auto* data = value.value_as_Int64ListValue();
        if(data == nullptr)
        {
            refuseMissingLaunchData(name);
        }
        return std::vector<int64_t>(data->values()->begin(), data->values()->end());
    }
    case fb::LaunchValueData::NONE:
    default:
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "launch value '" + messageText(name) + "' has unknown value type "
                               + std::to_string(static_cast<int>(value.value_type())));
    }
}

// Reads the launch values. Refuses an empty or repeated name, an unknown type and missing data.
MetadataValues readLaunchValues(const fb::LaunchInputs& launch)
{
    MetadataValues values;
    flatbuffers::uoffset_t index = 0;
    for(const fb::LaunchValue* value : *launch.values())
    {
        const std::string name = value->name()->str();
        if(name.empty())
        {
            refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                               "launch value at index " + std::to_string(index) + " has no name");
        }
        if(values.find(name) != values.end())
        {
            refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                               "launch value name '" + messageText(name)
                                   + "' occurs more than once");
        }
        values.emplace(name, readLaunchValueData(*value, name));
        ++index;
    }
    return values;
}

// The fields that the structure check reads.
struct CheckedFields
{
    int64_t engineId = 0;
    uint64_t workspaceBytes = 0;
    KernelSourceKind sourceKind{};
    MetadataValues launchValues;
};

// Checks the structure rules that the FlatBuffers verifier does not check.
CheckedFields checkIngestorPlanStructure(const fb::IngestorPlan& plan)
{
    CheckedFields fields;

    const auto engineId = plan.engine_id();
    if(!engineId.has_value())
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED, "the required field engine_id is missing");
    }
    fields.engineId = engineId.value();

    const auto workspaceBytes = plan.workspace_bytes();
    if(!workspaceBytes.has_value())
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "the required field workspace_bytes is missing");
    }
    fields.workspaceBytes = workspaceBytes.value();

    const fb::KernelImage& kernel = *plan.kernel();
    if(!fromFlatBufferSourceKind(kernel.source_kind(), fields.sourceKind))
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "kernel source kind "
                               + std::to_string(static_cast<int>(kernel.source_kind()))
                               + " is unset or unknown");
    }

    const std::string sha256 = kernel.sha256()->str();
    if(!isLowercaseSha256Hex(sha256))
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "the kernel digest '" + messageText(sha256)
                               + "' is not 64 lowercase hex characters");
    }

    fields.launchValues = readLaunchValues(*plan.launch());
    return fields;
}

// Refuses a plan that takes runtime pass-by-value tensors.
void checkIngestorPlanCompatibility(const fb::IngestorPlan& plan)
{
    const auto& uids = *plan.runtime_pass_by_value_uids();
    if(uids.empty())
    {
        return;
    }
    std::string listed;
    for(const int64_t uid : uids)
    {
        if(!listed.empty())
        {
            listed += ", ";
        }
        listed += std::to_string(uid);
    }
    refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                       "the plan takes runtime pass-by-value tensors (UIDs " + listed
                           + "), and this provider cannot restore such a plan");
}

IngestorPlanPayload toIngestorPlanPayload(const fb::IngestorPlan& plan, CheckedFields fields)
{
    IngestorPlanPayload payload;
    payload.engineId = fields.engineId;
    for(size_t index = 0; index < payload.kernelId.size(); ++index)
    {
        payload.kernelId[index]
            = plan.kernel_id()->bytes()->Get(static_cast<flatbuffers::uoffset_t>(index));
    }
    payload.workspaceBytes = fields.workspaceBytes;
    const auto& uids = *plan.runtime_pass_by_value_uids();
    payload.runtimePassByValueUids.assign(uids.begin(), uids.end());

    payload.dispatchSymbol = plan.launch()->dispatch_symbol()->str();
    payload.launchValues = std::move(fields.launchValues);

    const fb::KernelImage& kernel = *plan.kernel();
    payload.sourceKind = fields.sourceKind;
    payload.symbol = kernel.symbol()->str();
    payload.target = kernel.target()->str();
    payload.sha256 = kernel.sha256()->str();
    payload.recordedSignature.reserve(kernel.signature()->size());
    for(const fb::KernelArgument* argument : *kernel.signature())
    {
        KernelArgument copy;
        copy.kind = argument->kind()->str();
        copy.size = argument->size();
        copy.offset = argument->offset();
        if(argument->name() != nullptr)
        {
            copy.name = argument->name()->str();
        }
        payload.recordedSignature.push_back(std::move(copy));
    }
    const auto& codeObject = *kernel.code_object();
    payload.codeObject.assign(codeObject.data(), codeObject.data() + codeObject.size());

    if(plan.provider_version() != nullptr)
    {
        payload.providerVersion = plan.provider_version()->str();
    }
    return payload;
}

} // namespace

std::vector<uint8_t> encodeIngestorPlan(const IngestorPlanPayload& payload)
{
    const fb::SourceKind sourceKind = checkWriterInputs(payload);

    if(payload.codeObject.size() >= MAX_BODY_SIZE)
    {
        refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                           "the code object holds " + std::to_string(payload.codeObject.size())
                               + " bytes, and the payload format holds less than "
                               + std::to_string(MAX_BODY_SIZE));
    }

    flatbuffers::FlatBufferBuilder builder(payload.codeObject.size() + BUILDER_HEADROOM);

    const auto kernel = writeKernelImage(builder, payload, sourceKind);

    std::vector<flatbuffers::Offset<fb::LaunchValue>> values;
    values.reserve(payload.launchValues.size());
    for(const auto& [name, value] : payload.launchValues)
    {
        values.push_back(writeLaunchValue(builder, name, value));
    }
    const auto valuesVector = builder.CreateVector(values);
    const auto dispatchSymbol = builder.CreateString(payload.dispatchSymbol);
    const auto launch = fb::CreateLaunchInputs(builder, dispatchSymbol, valuesVector);

    const auto uids = builder.CreateVector(payload.runtimePassByValueUids);

    flatbuffers::Offset<flatbuffers::String> providerVersion;
    if(!payload.providerVersion.empty())
    {
        providerVersion = builder.CreateString(payload.providerVersion);
    }

    const fb::Uuid kernelId{flatbuffers::span<const uint8_t, KERNEL_ID_SIZE>(payload.kernelId)};

    fb::IngestorPlanBuilder plan(builder);
    plan.add_engine_id(payload.engineId);
    plan.add_kernel_id(&kernelId);
    plan.add_workspace_bytes(payload.workspaceBytes);
    plan.add_runtime_pass_by_value_uids(uids);
    plan.add_launch(launch);
    plan.add_kernel(kernel);
    if(!payload.providerVersion.empty())
    {
        plan.add_provider_version(providerVersion);
    }
    fb::FinishIngestorPlanBuffer(builder, plan.Finish());

    if(builder.GetBufferMinAlignment() > INGESTOR_PLAN_BODY_ALIGNMENT)
    {
        throwInternal("the body needs " + std::to_string(builder.GetBufferMinAlignment())
                      + "-byte alignment, and the header keeps only "
                      + std::to_string(INGESTOR_PLAN_BODY_ALIGNMENT));
    }
    const size_t bodySize = builder.GetSize();
    if(bodySize >= MAX_BODY_SIZE)
    {
        throwInternal("the body holds " + std::to_string(bodySize)
                      + " bytes, and the payload format holds less than "
                      + std::to_string(MAX_BODY_SIZE));
    }

    const uint8_t* body = builder.GetBufferPointer();
    std::vector<uint8_t> bytes
        = encodeIngestorPlanHeader(IngestorPlanKind::SINGLE_KERNEL_PLAN,
                                   detail::computeIngestorPlanBodyDigest(body, bodySize));
    bytes.insert(bytes.end(), body, body + bodySize);
    return bytes;
}

IngestorPlanPayload decodeIngestorPlan(const uint8_t* data, size_t size)
{
    const IngestorPlanHeader header = decodeIngestorPlanHeader(data, size);
    const uint8_t* body = data + header.headerSize;
    const size_t bodySize = size - header.headerSize;

    detail::checkIngestorPlanBodySize(bodySize);
    detail::checkIngestorPlanBodyDigest(body, bodySize, header.bodyDigest);

    std::vector<detail::IngestorPlanAlignedBlock> storage;
    const uint8_t* alignedBody = detail::alignedIngestorPlanBody(body, bodySize, storage);
    const fb::IngestorPlan& plan = *verifyIngestorPlanBody(alignedBody, bodySize);

    CheckedFields fields = checkIngestorPlanStructure(plan);
    checkIngestorPlanCompatibility(plan);
    return toIngestorPlanPayload(plan, std::move(fields));
}

namespace detail
{

IngestorPlanDigest computeIngestorPlanBodyDigest(const uint8_t* body, size_t bodySize)
{
    return ingestorPlanDigestFromHex(utilities::sha256Hex(body, bodySize));
}

void checkIngestorPlanBodySize(size_t bodySize)
{
    if(bodySize < MIN_BODY_SIZE || bodySize >= MAX_BODY_SIZE)
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "body size " + std::to_string(bodySize)
                               + " is outside the FlatBuffers limits: at least "
                               + std::to_string(MIN_BODY_SIZE) + " and less than "
                               + std::to_string(MAX_BODY_SIZE));
    }
}

void checkIngestorPlanBodyDigest(const uint8_t* body,
                                 size_t bodySize,
                                 const IngestorPlanDigest& headerDigest)
{
    // The header stores the raw digest, and sha256Hex returns hex text. Compare them as hex.
    const std::string expected = ingestorPlanDigestToHex(headerDigest);
    const std::string actual = utilities::sha256Hex(body, bodySize);
    if(actual != expected)
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "body digest mismatch: the header records " + expected
                               + ", and the body hashes to " + actual);
    }
}

const uint8_t* alignedIngestorPlanBody(const uint8_t* body,
                                       size_t bodySize,
                                       std::vector<IngestorPlanAlignedBlock>& storage)
{
    if(reinterpret_cast<std::uintptr_t>(body) % INGESTOR_PLAN_BODY_ALIGNMENT == 0)
    {
        return body;
    }
    storage.assign((bodySize + INGESTOR_PLAN_BODY_ALIGNMENT - 1) / INGESTOR_PLAN_BODY_ALIGNMENT,
                   IngestorPlanAlignedBlock{});
    std::memcpy(storage.data(), body, bodySize);
    return reinterpret_cast<const uint8_t*>(storage.data());
}

} // namespace detail

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
