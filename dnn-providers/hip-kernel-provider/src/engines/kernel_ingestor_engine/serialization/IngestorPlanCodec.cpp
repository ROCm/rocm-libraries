// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
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

constexpr size_t KERNEL_DESCRIPTOR_ID_SIZE
    = std::tuple_size_v<hipdnn_plugin_sdk::ingestor::DescriptorId>;

constexpr size_t SHA256_HEX_LENGTH = INGESTOR_PLAN_DIGEST_SIZE * 2;

// Builder space for everything other than the code bytes.
constexpr size_t BUILDER_HEADROOM = 4096;

constexpr size_t MESSAGE_TEXT_LIMIT = 80;

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
    if(payload.providerVersion.empty())
    {
        throwInternal("the provider version is empty");
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
        throwInternal("the kernel digest '" + ingestorPlanMessageText(payload.sha256)
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
        throwInternal("launch value '" + ingestorPlanMessageText(name)
                      + "' has a type the payload format cannot hold");
    }
    const auto nameOffset = builder.CreateString(name);
    return fb::CreateLaunchValue(builder, nameOffset, type, data);
}

flatbuffers::Offset<fb::DispatchData> writeDispatchData(flatbuffers::FlatBufferBuilder& builder,
                                                        const std::string& dispatchSymbol,
                                                        const MetadataValues& launchValues)
{
    std::vector<flatbuffers::Offset<fb::LaunchValue>> values;
    values.reserve(launchValues.size());
    for(const auto& [name, value] : launchValues)
    {
        values.push_back(writeLaunchValue(builder, name, value));
    }
    const auto valuesVector = builder.CreateVector(values);
    const auto symbol = builder.CreateString(dispatchSymbol);
    return fb::CreateDispatchData(builder, symbol, valuesVector);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fb::KernelArgument>>>
    writeSignature(flatbuffers::FlatBufferBuilder& builder,
                   const std::vector<KernelArgument>& signature)
{
    std::vector<flatbuffers::Offset<fb::KernelArgument>> arguments;
    arguments.reserve(signature.size());
    for(const KernelArgument& argument : signature)
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
    return builder.CreateVector(arguments);
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
    writeCodeBytes(flatbuffers::FlatBufferBuilder& builder, const std::vector<uint8_t>& codeObject)
{
    // The low-level vector calls ignore the schema's force_align. Align the code bytes
    // directly before the vector is created.
    builder.ForceVectorAlignment(codeObject.size(), sizeof(uint8_t), INGESTOR_PLAN_BODY_ALIGNMENT);
    return builder.CreateVector(codeObject);
}

flatbuffers::Offset<fb::KernelImage> writeKernelImage(flatbuffers::FlatBufferBuilder& builder,
                                                      const IngestorPlanPayload& payload,
                                                      fb::SourceKind sourceKind)
{
    const auto signature = writeSignature(builder, payload.recordedSignature);
    const auto codeObject = writeCodeBytes(builder, payload.codeObject);
    const auto symbol = builder.CreateString(payload.symbol);
    const auto target = builder.CreateString(payload.target);
    const auto sha256 = builder.CreateString(payload.sha256);
    return fb::CreateKernelImage(
        builder, sourceKind, symbol, target, sha256, signature, codeObject);
}

// Checks the rules of a kind 2 payload that the caller owns. Returns the FlatBuffers source
// kind of each code object.
std::vector<fb::SourceKind> checkMultiKernelWriterInputs(const IngestorMultiKernelPayload& payload)
{
    if(payload.providerVersion.empty())
    {
        throwInternal("the provider version is empty");
    }
    if(payload.stages.empty())
    {
        throwInternal("the plan has no stages");
    }
    if(payload.sharedWorkspaceBytes > payload.workspaceBytes)
    {
        throwInternal(
            "the shared workspace region holds " + std::to_string(payload.sharedWorkspaceBytes)
            + " bytes, more than the total workspace of " + std::to_string(payload.workspaceBytes));
    }

    std::vector<fb::SourceKind> sourceKinds;
    sourceKinds.reserve(payload.codeObjects.size());
    for(size_t index = 0; index < payload.codeObjects.size(); ++index)
    {
        const IngestorCodeObjectPayload& codeObject = payload.codeObjects[index];
        fb::SourceKind sourceKind = fb::SourceKind::UNSET;
        if(!toFlatBufferSourceKind(codeObject.sourceKind, sourceKind))
        {
            throwInternal("source kind " + std::to_string(static_cast<int>(codeObject.sourceKind))
                          + " of code object " + std::to_string(index)
                          + " is not a known kernel source kind");
        }
        if(!isLowercaseSha256Hex(codeObject.sha256))
        {
            throwInternal("the digest '" + ingestorPlanMessageText(codeObject.sha256)
                          + "' of code object " + std::to_string(index)
                          + " is not 64 lowercase hex characters");
        }
        sourceKinds.push_back(sourceKind);
    }

    for(size_t stageIndex = 0; stageIndex < payload.stages.size(); ++stageIndex)
    {
        const IngestorStagePayload& stage = payload.stages[stageIndex];
        const std::string stageName = "stage " + std::to_string(stageIndex);
        if(stage.dispatchSymbol.empty())
        {
            throwInternal("the dispatch symbol of " + stageName + " is empty");
        }
        if(stage.kernels.empty())
        {
            throwInternal(stageName + " has no kernels");
        }
        std::unordered_set<std::string_view> launchNames;
        for(const IngestorStageKernelPayload& kernel : stage.kernels)
        {
            if(kernel.symbol.empty())
            {
                throwInternal("a kernel symbol of " + stageName + " is empty");
            }
            if(kernel.codeObjectIndex >= payload.codeObjects.size())
            {
                throwInternal("a kernel of " + stageName + " names code object "
                              + std::to_string(kernel.codeObjectIndex) + ", and the plan holds "
                              + std::to_string(payload.codeObjects.size()));
            }
            if(!launchNames.insert(kernel.launchName).second)
            {
                throwInternal("launch name '" + ingestorPlanMessageText(kernel.launchName)
                              + "' occurs more than once in " + stageName);
            }
        }
    }
    return sourceKinds;
}

constexpr const char* MULTI_KERNEL_FILE_IDENTIFIER = "HKMP";

// The verified root of a body of table type @p Root, which carries @p fileIdentifier.
template <typename Root>
const Root* verifyBody(const uint8_t* alignedBody, size_t bodySize, const char* fileIdentifier)
{
    const flatbuffers::Verifier::Options options{};
    flatbuffers::Verifier verifier(alignedBody, bodySize, options);
    if(!verifier.VerifyBuffer<Root>(fileIdentifier))
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "the FlatBuffers verifier rejects the payload body, or the body "
                           "does not carry the "
                               + std::string(fileIdentifier) + " file identifier");
    }
    return flatbuffers::GetRoot<Root>(alignedBody);
}

// Checks that a code object's recorded digest is well formed and matches its bytes. The
// messages read "<before> '<digest>'<after>", so @p before and @p after name the digest.
void requireCodeObjectDigest(const flatbuffers::Vector<uint8_t>& bytes,
                             const std::string& sha256,
                             const std::string& before,
                             const std::string& after)
{
    if(!isLowercaseSha256Hex(sha256))
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           before + " '" + ingestorPlanMessageText(sha256) + "'" + after
                               + " is not 64 lowercase hex characters");
    }
    const std::string actualSha256 = utilities::sha256Hex(bytes.data(), bytes.size());
    if(actualSha256 != sha256)
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           before + " " + sha256 + after
                               + " does not match the code object, which hashes to "
                               + actualSha256);
    }
}

// The value of a required scalar field. The verifier does not check that it is present.
template <typename T>
T requireField(const flatbuffers::Optional<T>& field, const char* name)
{
    if(!field.has_value())
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           std::string("the required field ") + name + " is missing");
    }
    return field.value();
}

hipdnn_plugin_sdk::ingestor::DescriptorId readDescriptorId(const fb::Uuid& uuid)
{
    hipdnn_plugin_sdk::ingestor::DescriptorId id{};
    for(size_t index = 0; index < id.size(); ++index)
    {
        id[index] = uuid.bytes()->Get(static_cast<flatbuffers::uoffset_t>(index));
    }
    return id;
}

fb::Uuid toFlatBufferUuid(const hipdnn_plugin_sdk::ingestor::DescriptorId& id)
{
    return fb::Uuid{flatbuffers::span<const uint8_t, KERNEL_DESCRIPTOR_ID_SIZE>(id)};
}

[[noreturn]] void refuseMissingLaunchData(const std::string& name)
{
    refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                       "launch value '" + ingestorPlanMessageText(name)
                           + "' has a type but no data");
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
                           "launch value '" + ingestorPlanMessageText(name)
                               + "' has unknown value type "
                               + std::to_string(static_cast<int>(value.value_type())));
    }
}

// Reads the launch values. Refuses an empty or repeated name, an unknown type and missing data.
MetadataValues readLaunchValues(const fb::DispatchData& dispatchData)
{
    MetadataValues values;
    flatbuffers::uoffset_t index = 0;
    for(const fb::LaunchValue* value : *dispatchData.values())
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
                               "launch value name '" + ingestorPlanMessageText(name)
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
    uint64_t workspaceBytes = 0;
    KernelSourceKind sourceKind{};
    MetadataValues launchValues;
};

// Checks the structure rules that the FlatBuffers verifier does not check.
CheckedFields checkIngestorPlanStructure(const fb::ExecutionPlan& plan)
{
    CheckedFields fields;

    fields.workspaceBytes = requireField(plan.workspace_bytes(), "workspace_bytes");

    const fb::KernelImage& kernelImage = *plan.kernel_image();
    if(!fromFlatBufferSourceKind(kernelImage.source_kind(), fields.sourceKind))
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "kernel source kind "
                               + std::to_string(static_cast<int>(kernelImage.source_kind()))
                               + " is unset or unknown");
    }

    requireCodeObjectDigest(
        *kernelImage.code_object(), kernelImage.sha256()->str(), "the kernel digest", "");

    fields.launchValues = readLaunchValues(*plan.dispatch_data());
    return fields;
}

std::vector<KernelArgument>
    readSignature(const flatbuffers::Vector<flatbuffers::Offset<fb::KernelArgument>>& signature)
{
    std::vector<KernelArgument> result;
    result.reserve(signature.size());
    for(const fb::KernelArgument* argument : signature)
    {
        KernelArgument copy;
        copy.kind = argument->kind()->str();
        copy.size = argument->size();
        copy.offset = argument->offset();
        if(argument->name() != nullptr)
        {
            copy.name = argument->name()->str();
        }
        result.push_back(std::move(copy));
    }
    return result;
}

IngestorPlanPayload toIngestorPlanPayload(const fb::ExecutionPlan& plan, CheckedFields fields)
{
    IngestorPlanPayload payload;
    payload.kernelDescriptorId = readDescriptorId(*plan.kernel_descriptor_id());
    payload.workspaceBytes = fields.workspaceBytes;
    const auto& uids = *plan.runtime_pass_by_value_uids();
    payload.runtimePassByValueUids.assign(uids.begin(), uids.end());

    payload.dispatchSymbol = plan.dispatch_data()->dispatch_symbol()->str();
    payload.launchValues = std::move(fields.launchValues);

    const fb::KernelImage& kernelImage = *plan.kernel_image();
    payload.sourceKind = fields.sourceKind;
    payload.symbol = kernelImage.symbol()->str();
    payload.target = kernelImage.target()->str();
    payload.sha256 = kernelImage.sha256()->str();
    payload.recordedSignature = readSignature(*kernelImage.signature());
    const auto& codeObject = *kernelImage.code_object();
    payload.codeObject.assign(codeObject.data(), codeObject.data() + codeObject.size());

    payload.providerVersion = plan.provider_version()->str();
    return payload;
}

// Reads a verified kind 2 body into a payload. Refuses the damage that the verifier does not
// see: a missing workspace field, a shared region above the total, a plan or stage without
// kernels, a code object index out of range, a repeated launch name, and a code object whose
// digest is malformed or does not match its bytes.
IngestorMultiKernelPayload readMultiKernelPlan(const fb::MultiKernelPlan& plan)
{
    IngestorMultiKernelPayload payload;

    payload.sharedWorkspaceBytes
        = requireField(plan.shared_workspace_bytes(), "shared_workspace_bytes");
    payload.workspaceBytes = requireField(plan.workspace_bytes(), "workspace_bytes");
    if(payload.sharedWorkspaceBytes > payload.workspaceBytes)
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                           "the shared workspace region holds "
                               + std::to_string(payload.sharedWorkspaceBytes)
                               + " bytes, more than the total workspace of "
                               + std::to_string(payload.workspaceBytes));
    }

    const auto& stages = *plan.stages();
    if(stages.empty())
    {
        refuseIngestorPlan(IngestorPlanRefusal::DAMAGED, "the plan has no stages");
    }

    const auto& codeObjects = *plan.code_objects();
    payload.codeObjects.reserve(codeObjects.size());
    flatbuffers::uoffset_t objectIndex = 0;
    for(const fb::CodeObject* object : codeObjects)
    {
        IngestorCodeObjectPayload codeObject;
        if(!fromFlatBufferSourceKind(object->source_kind(), codeObject.sourceKind))
        {
            refuseIngestorPlan(
                IngestorPlanRefusal::DAMAGED,
                "source kind " + std::to_string(static_cast<int>(object->source_kind()))
                    + " of code object " + std::to_string(objectIndex) + " is unset or unknown");
        }
        codeObject.target = object->target()->str();
        codeObject.sha256 = object->sha256()->str();
        const auto& bytes = *object->code_object();
        requireCodeObjectDigest(bytes,
                                codeObject.sha256,
                                "the digest",
                                " of code object " + std::to_string(objectIndex));
        codeObject.codeObject.assign(bytes.data(), bytes.data() + bytes.size());
        payload.codeObjects.push_back(std::move(codeObject));
        ++objectIndex;
    }

    payload.stages.reserve(stages.size());
    flatbuffers::uoffset_t stageIndex = 0;
    for(const fb::Stage* stage : stages)
    {
        const std::string stageName = "stage " + std::to_string(stageIndex);
        IngestorStagePayload stagePayload;
        stagePayload.slot = stage->slot()->str();
        stagePayload.kernelDescriptorId = readDescriptorId(*stage->kernel_descriptor_id());
        stagePayload.dispatchSymbol = stage->dispatch_data()->dispatch_symbol()->str();
        stagePayload.launchValues = readLaunchValues(*stage->dispatch_data());

        const auto& kernels = *stage->kernels();
        if(kernels.empty())
        {
            refuseIngestorPlan(IngestorPlanRefusal::DAMAGED, stageName + " has no kernels");
        }
        std::unordered_set<std::string_view> launchNames;
        stagePayload.kernels.reserve(kernels.size());
        for(const fb::StageKernel* kernel : kernels)
        {
            IngestorStageKernelPayload kernelPayload;
            kernelPayload.launchName = kernel->launch_name()->str();
            if(!launchNames
                    .insert(std::string_view(kernel->launch_name()->c_str(),
                                             kernel->launch_name()->size()))
                    .second)
            {
                refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                                   "launch name '"
                                       + ingestorPlanMessageText(kernelPayload.launchName)
                                       + "' occurs more than once in " + stageName);
            }
            kernelPayload.symbol = kernel->symbol()->str();
            kernelPayload.recordedSignature = readSignature(*kernel->signature());
            kernelPayload.codeObjectIndex = kernel->code_object_index();
            if(kernelPayload.codeObjectIndex >= payload.codeObjects.size())
            {
                refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                                   "a kernel of " + stageName + " names code object "
                                       + std::to_string(kernelPayload.codeObjectIndex)
                                       + ", and the plan holds "
                                       + std::to_string(payload.codeObjects.size()));
            }
            stagePayload.kernels.push_back(std::move(kernelPayload));
        }
        payload.stages.push_back(std::move(stagePayload));
        ++stageIndex;
    }

    const auto& uids = *plan.runtime_pass_by_value_uids();
    payload.runtimePassByValueUids.assign(uids.begin(), uids.end());
    payload.providerVersion = plan.provider_version()->str();
    return payload;
}

// Puts the header of `kind` in front of the finished body in `builder`.
EncodedIngestorPlan finishEncodedPlan(IngestorPlanKind kind,
                                      const flatbuffers::FlatBufferBuilder& builder)
{
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
    const std::vector<uint8_t> header
        = encodeIngestorPlanHeader(kind, detail::computeIngestorPlanBodyDigest(body, bodySize));

    EncodedIngestorPlan encoded;
    encoded.size = header.size() + bodySize;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    encoded.bytes = std::make_unique<uint8_t[]>(encoded.size);
    std::memcpy(encoded.bytes.get(), header.data(), header.size());
    std::memcpy(encoded.bytes.get() + header.size(), body, bodySize);
    return encoded;
}

EncodedIngestorPlan encodeSingleKernelPlanToBuffer(IngestorPlanPayload payload)
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

    const auto kernelImage = writeKernelImage(builder, payload, sourceKind);
    // The builder holds the code bytes now. Free the payload copy to lower the peak memory.
    std::vector<uint8_t>().swap(payload.codeObject);

    const auto dispatchData
        = writeDispatchData(builder, payload.dispatchSymbol, payload.launchValues);

    const auto uids = builder.CreateVector(payload.runtimePassByValueUids);

    const auto providerVersion = builder.CreateString(payload.providerVersion);

    const fb::Uuid kernelDescriptorId = toFlatBufferUuid(payload.kernelDescriptorId);

    fb::ExecutionPlanBuilder plan(builder);
    plan.add_kernel_descriptor_id(&kernelDescriptorId);
    plan.add_workspace_bytes(payload.workspaceBytes);
    plan.add_runtime_pass_by_value_uids(uids);
    plan.add_dispatch_data(dispatchData);
    plan.add_kernel_image(kernelImage);
    plan.add_provider_version(providerVersion);
    fb::FinishExecutionPlanBuffer(builder, plan.Finish());

    return finishEncodedPlan(IngestorPlanKind::SINGLE_KERNEL_PLAN, builder);
}

EncodedIngestorPlan encodeMultiKernelPlanToBuffer(IngestorMultiKernelPayload payload)
{
    const std::vector<fb::SourceKind> sourceKinds = checkMultiKernelWriterInputs(payload);

    // Equal code objects share one entry: the code object list holds each (digest, source
    // kind, target) once, and the kernels name the surviving entry.
    std::vector<size_t> written;
    std::vector<uint32_t> writtenIndexOf(payload.codeObjects.size());
    size_t codeBytes = 0;
    for(size_t index = 0; index < payload.codeObjects.size(); ++index)
    {
        const IngestorCodeObjectPayload& codeObject = payload.codeObjects[index];
        const auto same = std::find_if(written.begin(), written.end(), [&](size_t earlier) {
            const IngestorCodeObjectPayload& other = payload.codeObjects[earlier];
            return other.sha256 == codeObject.sha256 && other.sourceKind == codeObject.sourceKind
                   && other.target == codeObject.target;
        });
        if(same != written.end())
        {
            writtenIndexOf[index] = static_cast<uint32_t>(same - written.begin());
            continue;
        }
        writtenIndexOf[index] = static_cast<uint32_t>(written.size());
        written.push_back(index);
        codeBytes += codeObject.codeObject.size();
    }
    if(codeBytes >= MAX_BODY_SIZE)
    {
        refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                           "the code objects hold " + std::to_string(codeBytes)
                               + " bytes, and the payload format holds less than "
                               + std::to_string(MAX_BODY_SIZE));
    }

    flatbuffers::FlatBufferBuilder builder(codeBytes + BUILDER_HEADROOM);

    std::vector<flatbuffers::Offset<fb::CodeObject>> codeObjects;
    codeObjects.reserve(written.size());
    for(const size_t index : written)
    {
        IngestorCodeObjectPayload& codeObject = payload.codeObjects[index];
        const auto bytes = writeCodeBytes(builder, codeObject.codeObject);
        // The builder holds the code bytes now. Free the payload copy to lower the peak memory.
        std::vector<uint8_t>().swap(codeObject.codeObject);
        const auto target = builder.CreateString(codeObject.target);
        const auto sha256 = builder.CreateString(codeObject.sha256);
        codeObjects.push_back(
            fb::CreateCodeObject(builder, sourceKinds[index], target, sha256, bytes));
    }
    const auto codeObjectsVector = builder.CreateVector(codeObjects);

    std::vector<flatbuffers::Offset<fb::Stage>> stages;
    stages.reserve(payload.stages.size());
    for(const IngestorStagePayload& stage : payload.stages)
    {
        std::vector<flatbuffers::Offset<fb::StageKernel>> kernels;
        kernels.reserve(stage.kernels.size());
        for(const IngestorStageKernelPayload& kernel : stage.kernels)
        {
            const auto signature = writeSignature(builder, kernel.recordedSignature);
            const auto launchName = builder.CreateString(kernel.launchName);
            const auto symbol = builder.CreateString(kernel.symbol);
            kernels.push_back(fb::CreateStageKernel(
                builder, launchName, symbol, signature, writtenIndexOf[kernel.codeObjectIndex]));
        }
        const auto kernelsVector = builder.CreateVector(kernels);

        const auto dispatchData
            = writeDispatchData(builder, stage.dispatchSymbol, stage.launchValues);

        const auto slot = builder.CreateString(stage.slot);
        const fb::Uuid kernelDescriptorId = toFlatBufferUuid(stage.kernelDescriptorId);
        stages.push_back(
            fb::CreateStage(builder, slot, &kernelDescriptorId, dispatchData, kernelsVector));
    }
    const auto stagesVector = builder.CreateVector(stages);

    const auto uids = builder.CreateVector(payload.runtimePassByValueUids);
    const auto providerVersion = builder.CreateString(payload.providerVersion);

    const auto plan = fb::CreateMultiKernelPlan(builder,
                                                stagesVector,
                                                codeObjectsVector,
                                                payload.sharedWorkspaceBytes,
                                                payload.workspaceBytes,
                                                uids,
                                                providerVersion);
    builder.Finish(plan, MULTI_KERNEL_FILE_IDENTIFIER);

    return finishEncodedPlan(IngestorPlanKind::MULTI_KERNEL_PLAN, builder);
}

} // namespace

EncodedIngestorPlan encodeIngestorPlanToBuffer(IngestorSavedPlan payload)
{
    if(auto* singleKernel = std::get_if<IngestorPlanPayload>(&payload))
    {
        return encodeSingleKernelPlanToBuffer(std::move(*singleKernel));
    }
    return encodeMultiKernelPlanToBuffer(std::move(std::get<IngestorMultiKernelPayload>(payload)));
}

std::vector<uint8_t> encodeIngestorPlan(const IngestorSavedPlan& payload)
{
    const EncodedIngestorPlan encoded = encodeIngestorPlanToBuffer(payload);
    return {encoded.bytes.get(), encoded.bytes.get() + encoded.size};
}

std::string ingestorPlanMessageText(const std::string& text)
{
    if(text.size() <= MESSAGE_TEXT_LIMIT)
    {
        return text;
    }
    return text.substr(0, MESSAGE_TEXT_LIMIT) + "... (" + std::to_string(text.size())
           + " characters)";
}

IngestorSavedPlan decodeIngestorSavedPlan(const uint8_t* data, size_t size)
{
    const IngestorPlanHeader header = decodeIngestorPlanHeader(data, size);
    const uint8_t* body = data + header.headerSize;
    const size_t bodySize = size - header.headerSize;

    detail::checkIngestorPlanBodySize(bodySize);
    detail::checkIngestorPlanBodyDigest(body, bodySize, header.bodyDigest);

    std::vector<detail::IngestorPlanAlignedBlock> storage;
    const uint8_t* alignedBody = detail::alignedIngestorPlanBody(body, bodySize, storage);
    if(header.kind == IngestorPlanKind::MULTI_KERNEL_PLAN)
    {
        return readMultiKernelPlan(
            *verifyBody<fb::MultiKernelPlan>(alignedBody, bodySize, MULTI_KERNEL_FILE_IDENTIFIER));
    }
    const fb::ExecutionPlan& plan
        = *verifyBody<fb::ExecutionPlan>(alignedBody, bodySize, fb::ExecutionPlanIdentifier());

    CheckedFields fields = checkIngestorPlanStructure(plan);
    return toIngestorPlanPayload(plan, std::move(fields));
}

IngestorPlanPayload decodeIngestorPlan(const uint8_t* data, size_t size)
{
    const IngestorPlanHeader header = decodeIngestorPlanHeader(data, size);
    if(header.kind != IngestorPlanKind::SINGLE_KERNEL_PLAN)
    {
        refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                           "ingestor plan kind "
                               + std::to_string(static_cast<uint16_t>(header.kind))
                               + " is not a single-kernel plan");
    }
    return std::get<IngestorPlanPayload>(decodeIngestorSavedPlan(data, size));
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
