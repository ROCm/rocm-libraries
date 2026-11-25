// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/PhiloxBackward.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <flatbuffers/flatbuffers.h>

namespace hipdnn_integration_tests
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

const fb::TensorAttributes* findTensor(const fb::Graph& graph, int64_t uid)
{
    if(graph.tensors() == nullptr)
    {
        return nullptr;
    }
    for(const auto* tensor : *graph.tensors())
    {
        if(tensor->uid() == uid)
        {
            return tensor;
        }
    }
    return nullptr;
}

const fb::SdpaBackwardAttributes* findSdpaBackward(const fb::Graph& graph)
{
    if(graph.nodes() == nullptr)
    {
        return nullptr;
    }
    for(const auto* node : *graph.nodes())
    {
        if(const auto* attributes = node->attributes_as_SdpaBackwardAttributes())
        {
            return attributes;
        }
    }
    return nullptr;
}

} // namespace

std::optional<PhiloxBackwardDropout> findPhiloxBackwardDropout(const fb::Graph& graph,
                                                               std::string& unsupported)
{
    unsupported.clear();
    const auto* attributes = findSdpaBackward(graph);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    const bool probability = attributes->dropout_probability().has_value()
                             && attributes->dropout_probability().value() > 0.0F;
    if(!probability || attributes->dropout_mask_tensor_uid().has_value()
       || !attributes->seed_tensor_uid().has_value()
       || !attributes->offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const auto* q = findTensor(graph, attributes->q_tensor_uid());
    const auto* k = findTensor(graph, attributes->k_tensor_uid());
    if(q == nullptr || k == nullptr || q->dims() == nullptr || k->dims() == nullptr
       || q->dims()->size() != 4 || k->dims()->size() != 4)
    {
        unsupported = "Philox dropout needs rank-4 Q and K";
        return std::nullopt;
    }
    if(q->ragged_offset_tensor_uid().has_value() || k->ragged_offset_tensor_uid().has_value())
    {
        unsupported = "Philox dropout on a ragged graph is not modelled";
        return std::nullopt;
    }

    int64_t maxUid = 0;
    for(const auto* tensor : *graph.tensors())
    {
        maxUid = std::max(maxUid, tensor->uid());
    }

    PhiloxBackwardDropout dropout;
    dropout.probability = attributes->dropout_probability().value();
    dropout.seedUid = attributes->seed_tensor_uid().value();
    dropout.offsetUid = attributes->offset_tensor_uid().value();
    dropout.maskUid = maxUid + 1;
    dropout.scaleUid = maxUid + 2;
    dropout.scaleInvUid = maxUid + 3;
    dropout.extent
        = philox::MaskExtent{(*q->dims())[0], (*q->dims())[1], (*q->dims())[2], (*k->dims())[2]};
    return dropout;
}

namespace
{

// The seed or offset a scalar tensor holds: INT32 sign-extended to 64 bits, or INT64.
std::optional<std::string> readScalar(const fb::Graph& graph,
                                      int64_t uid,
                                      const char* what,
                                      const InputTensorMap& inputs,
                                      uint64_t& out)
{
    const auto* tensor = findTensor(graph, uid);
    if(tensor == nullptr)
    {
        return std::string("the ") + what + " tensor (uid " + std::to_string(uid)
               + ") is not in the graph";
    }
    const auto type = tensor->data_type();
    if(type != fb::DataType::INT32 && type != fb::DataType::INT64)
    {
        return std::string("the ") + what + " tensor is not INT32 or INT64";
    }

    const auto input = inputs.find(uid);
    if(input != inputs.end() && input->second != nullptr)
    {
        auto& value = *input->second;
        if(value.elementSpace() == 0)
        {
            return std::string("the ") + what + " tensor is empty";
        }
        if(type == fb::DataType::INT32)
        {
            int32_t raw = 0;
            std::memcpy(&raw, value.rawHostData(), sizeof(raw));
            out = static_cast<uint64_t>(static_cast<int64_t>(raw));
        }
        else
        {
            int64_t raw = 0;
            std::memcpy(&raw, value.rawHostData(), sizeof(raw));
            out = static_cast<uint64_t>(raw);
        }
        return std::nullopt;
    }
    if(type == fb::DataType::INT32 && tensor->value_as_Int32Value() != nullptr)
    {
        out = static_cast<uint64_t>(static_cast<int64_t>(tensor->value_as_Int32Value()->value()));
        return std::nullopt;
    }
    if(type == fb::DataType::INT64 && tensor->value_as_Int64Value() != nullptr)
    {
        out = static_cast<uint64_t>(tensor->value_as_Int64Value()->value());
        return std::nullopt;
    }
    return std::string("the ") + what
           + " is neither a graph input nor a value baked into the graph";
}

} // namespace

std::optional<std::string> resolvePhiloxSeedAndOffset(const fb::Graph& graph,
                                                      const PhiloxBackwardDropout& dropout,
                                                      const InputTensorMap& inputs,
                                                      uint64_t& seed,
                                                      uint64_t& offset)
{
    if(auto why = readScalar(graph, dropout.seedUid, "seed", inputs, seed))
    {
        return why;
    }
    return readScalar(graph, dropout.offsetUid, "offset", inputs, offset);
}

std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    makePhiloxKeepMask(const PhiloxBackwardDropout& dropout, uint64_t seed, uint64_t offset)
{
    const auto& e = dropout.extent;
    const std::vector<int64_t> dims = {e.batch, e.heads, e.seqQ, e.seqKv};
    const std::vector<int64_t> strides = {e.heads * e.seqQ * e.seqKv, e.seqQ * e.seqKv, e.seqKv, 1};
    auto mask = std::make_shared<hipdnn_data_sdk::utilities::Tensor<float>>(dims, strides);
    philox::fillKeepMask(
        static_cast<float*>(mask->rawHostData()), seed, offset, e, dropout.probability);
    mask->markHostModified();
    return mask;
}

std::vector<uint8_t> rewritePhiloxBackwardWithExplicitMask(const fb::Graph& graph,
                                                           const PhiloxBackwardDropout& dropout)
{
    std::unique_ptr<fb::GraphT> rewritten(graph.UnPack());

    fb::SdpaBackwardAttributesT* attributes = nullptr;
    for(auto& node : rewritten->nodes)
    {
        if(auto* a = node->attributes.AsSdpaBackwardAttributes())
        {
            attributes = a;
            break;
        }
    }
    if(attributes == nullptr)
    {
        throw std::invalid_argument("the graph has no SDPA backward node");
    }
    attributes->dropout_probability = flatbuffers::nullopt;
    attributes->seed_tensor_uid = flatbuffers::nullopt;
    attributes->offset_tensor_uid = flatbuffers::nullopt;
    attributes->dropout_mask_tensor_uid = dropout.maskUid;
    attributes->dropout_scale_tensor_uid = dropout.scaleUid;
    attributes->dropout_scale_inv_tensor_uid = dropout.scaleInvUid;

    // The seed and offset are not read by the explicit-mask form.
    rewritten->tensors.erase(
        std::remove_if(rewritten->tensors.begin(),
                       rewritten->tensors.end(),
                       [&dropout](const std::unique_ptr<fb::TensorAttributesT>& tensor) {
                           return tensor->uid == dropout.seedUid
                                  || tensor->uid == dropout.offsetUid;
                       }),
        rewritten->tensors.end());

    const auto& e = dropout.extent;
    auto mask = std::make_unique<fb::TensorAttributesT>();
    mask->uid = dropout.maskUid;
    mask->name = "philox_keep_mask";
    mask->data_type = fb::DataType::FLOAT;
    mask->dims = {e.batch, e.heads, e.seqQ, e.seqKv};
    mask->strides = {e.heads * e.seqQ * e.seqKv, e.seqQ * e.seqKv, e.seqKv, 1};
    mask->virtual_ = false;
    rewritten->tensors.push_back(std::move(mask));

    const auto addScalar = [&rewritten](int64_t uid, const char* name, float value) {
        auto scalar = std::make_unique<fb::TensorAttributesT>();
        scalar->uid = uid;
        scalar->name = name;
        scalar->data_type = fb::DataType::FLOAT;
        scalar->dims = {1, 1, 1, 1};
        scalar->strides = {1, 1, 1, 1};
        scalar->virtual_ = false;
        scalar->is_runtime_pass_by_value = false;
        scalar->value.Set(fb::Float32Value(value));
        rewritten->tensors.push_back(std::move(scalar));
    };
    addScalar(dropout.scaleUid, "philox_dropout_scale", philox::dropoutScale(dropout.probability));
    addScalar(dropout.scaleInvUid, "philox_dropout_scale_inv", 1.0F - dropout.probability);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::Graph::Pack(builder, rewritten.get()));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace hipdnn_integration_tests
