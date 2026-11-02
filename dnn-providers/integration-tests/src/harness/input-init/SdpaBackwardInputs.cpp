// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/SdpaBackwardInputs.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_integration_tests
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::ShallowTensor;

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

// Names the first term the forward applies besides the scale, the diagonal band and
// the additive mask, none of which the derivation reproduces.
std::optional<std::string> unmodeledForwardTerm(const fb::SdpaBackwardAttributes& a)
{
    if(a.alibi_mask())
    {
        return "ALiBi";
    }
    if(a.padding_mask() || a.seq_len_q_tensor_uid().has_value()
       || a.seq_len_kv_tensor_uid().has_value())
    {
        return "padding or variable sequence lengths";
    }
    // A zero probability is no dropout, as the engines read it.
    if((a.dropout_probability().has_value() && a.dropout_probability().value() != 0.f)
       || a.seed_tensor_uid().has_value() || a.offset_tensor_uid().has_value()
       || a.dropout_mask_tensor_uid().has_value() || a.dropout_scale_tensor_uid().has_value()
       || a.dropout_scale_inv_tensor_uid().has_value())
    {
        return "dropout";
    }
    return std::nullopt;
}

// The Q, K, V, O and mask types the CPU forward is instantiated for; the CPU backward
// registers the same ones.
bool isReferenceType(fb::DataType type)
{
    return type == fb::DataType::FLOAT || type == fb::DataType::HALF
           || type == fb::DataType::BFLOAT16;
}

// The CPU forward reads the mask right-aligned to the [B, H, Sq, Skv] scores, with
// size-1 dims broadcast, and does not check its shape, so it is checked here.
bool maskBroadcastsToScores(const ITensor& mask, const ITensor& q, const ITensor& k)
{
    const auto& maskDims = mask.dims();
    if(maskDims.empty() || maskDims.size() > 4)
    {
        return false;
    }
    if(q.dims().size() != 4 || k.dims().size() != 4)
    {
        // The forward rejects these shapes before it reads the mask.
        return true;
    }
    const std::array<int64_t, 4> scoreDims = {q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]};
    const std::size_t offset = scoreDims.size() - maskDims.size();
    for(std::size_t i = 0; i < maskDims.size(); ++i)
    {
        if(maskDims[i] != 1 && maskDims[i] != scoreDims[offset + i])
        {
            return false;
        }
    }
    return true;
}

template <typename T, typename Mask>
void forwardInto(ITensor& q,
                 ITensor& k,
                 ITensor& v,
                 ITensor& o,
                 ITensor& stats,
                 ITensor* mask,
                 std::optional<float> scale,
                 const DiagonalBandParams& band)
{
    const ShallowTensor<T> qView(q.rawHostData(), q.dims(), q.strides());
    const ShallowTensor<T> kView(k.rawHostData(), k.dims(), k.strides());
    const ShallowTensor<T> vView(v.rawHostData(), v.dims(), v.strides());
    ShallowTensor<T> oView(o.rawHostData(), o.dims(), o.strides());
    ShallowTensor<float> lseView(stats.rawHostData(), stats.dims(), stats.strides());
    std::optional<ShallowTensor<Mask>> maskView;
    if(mask != nullptr)
    {
        maskView.emplace(mask->rawHostData(), mask->dims(), mask->strides());
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward<T, T, T, T, float, Mask>(
        qView,
        kView,
        vView,
        oView,
        scale,
        maskView.has_value() ? &*maskView : nullptr,
        band.leftBound,
        band.rightBound,
        band.topLeftAlignment,
        &lseView);

    // The views write the host copies; a device-resident tensor uploads them only
    // once told the host copy is the current one.
    o.markHostModified();
    stats.markHostModified();
}

// The caller has checked maskType with isReferenceType(); without a mask it is FLOAT.
template <typename T>
void forwardWithMaskType(ITensor& q,
                         ITensor& k,
                         ITensor& v,
                         ITensor& o,
                         ITensor& stats,
                         ITensor* mask,
                         fb::DataType maskType,
                         std::optional<float> scale,
                         const DiagonalBandParams& band)
{
    switch(maskType)
    {
    case fb::DataType::HALF:
        forwardInto<T, hipdnn_data_sdk::types::half>(q, k, v, o, stats, mask, scale, band);
        break;
    case fb::DataType::BFLOAT16:
        forwardInto<T, hipdnn_data_sdk::types::bfloat16>(q, k, v, o, stats, mask, scale, band);
        break;
    default:
        forwardInto<T, float>(q, k, v, o, stats, mask, scale, band);
        break;
    }
}

} // anonymous namespace

std::optional<std::string>
    deriveSdpaBackwardForwardOutputs(const fb::Graph& graph,
                                     const fb::SdpaBackwardAttributes& attributes,
                                     InputTensorMap& inputs,
                                     const std::vector<int64_t>& ownedUids)
{
    const auto isOwned = [&ownedUids](int64_t uid) {
        return std::find(ownedUids.begin(), ownedUids.end(), uid) != ownedUids.end();
    };

    // In a fused forward-backward graph the forward node computes O and stats.
    const bool ownsO = isOwned(attributes.o_tensor_uid());
    const bool ownsStats = isOwned(attributes.stats_tensor_uid());
    if(!ownsO && !ownsStats)
    {
        return std::nullopt;
    }
    if(!ownsO || !ownsStats)
    {
        return "only one of O and stats is a graph input";
    }

    if(const auto term = unmodeledForwardTerm(attributes))
    {
        return "the derivation does not model " + *term;
    }

    std::vector<std::pair<std::string, int64_t>> operands
        = {{"Q", attributes.q_tensor_uid()},
           {"K", attributes.k_tensor_uid()},
           {"V", attributes.v_tensor_uid()},
           {"O", attributes.o_tensor_uid()},
           {"stats", attributes.stats_tensor_uid()}};
    const auto maskUid = attributes.attn_mask_tensor_uid();
    if(maskUid.has_value())
    {
        operands.emplace_back("the attention mask", *maskUid);
    }
    for(const auto& [name, uid] : operands)
    {
        const auto* tensor = findTensor(graph, uid);
        if(tensor == nullptr)
        {
            return name + " (uid " + std::to_string(uid) + ") is not in the graph";
        }
        if(!isOwned(uid))
        {
            return name
                   + " is computed by the graph, so its values are not known when the "
                     "inputs are filled";
        }
        if(tensor->ragged_offset_tensor_uid().has_value())
        {
            return name + " is ragged";
        }
    }

    const auto ioType = findTensor(graph, attributes.q_tensor_uid())->data_type();
    for(const int64_t uid :
        {attributes.k_tensor_uid(), attributes.v_tensor_uid(), attributes.o_tensor_uid()})
    {
        if(findTensor(graph, uid)->data_type() != ioType)
        {
            return "Q, K, V and O do not share one data type";
        }
    }
    if(!isReferenceType(ioType))
    {
        return std::string("the CPU forward does not take ") + fb::EnumNameDataType(ioType);
    }
    if(findTensor(graph, attributes.stats_tensor_uid())->data_type() != fb::DataType::FLOAT)
    {
        return "stats is not FLOAT";
    }
    auto maskType = fb::DataType::FLOAT;
    if(maskUid.has_value())
    {
        maskType = findTensor(graph, *maskUid)->data_type();
        if(!isReferenceType(maskType))
        {
            return std::string("the CPU forward does not take a ") + fb::EnumNameDataType(maskType)
                   + " attention mask";
        }
    }

    // A scale tensor takes precedence over attn_scale_value, as in the CPU reference's
    // plan; neither means no scaling. The reference reads a scale tensor only as a
    // pass-by-value scalar.
    std::optional<float> scale = attributes.attn_scale_value();
    if(attributes.scale_tensor_uid().has_value())
    {
        const int64_t scaleUid = *attributes.scale_tensor_uid();
        const auto* scaleTensor = findTensor(graph, scaleUid);
        if(!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scaleTensor))
        {
            return "the scale tensor is not a pass-by-value scalar";
        }
        if(scaleTensor->is_runtime_pass_by_value() && !isOwned(scaleUid))
        {
            return "the runtime scale is computed by the graph";
        }
        std::unordered_map<int64_t, void*> variantPack;
        if(isOwned(scaleUid))
        {
            variantPack.emplace(scaleUid, inputs.at(scaleUid)->rawHostData());
        }
        scale = hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
            hipdnn_test_sdk::detail::unpackTensorAttributes(*scaleTensor),
            variantPack,
            "SDPA backward scale");
    }

    auto& q = *inputs.at(attributes.q_tensor_uid());
    auto& k = *inputs.at(attributes.k_tensor_uid());
    auto& v = *inputs.at(attributes.v_tensor_uid());
    auto& o = *inputs.at(attributes.o_tensor_uid());
    auto& stats = *inputs.at(attributes.stats_tensor_uid());
    ITensor* mask = maskUid.has_value() ? inputs.at(*maskUid).get() : nullptr;
    if(mask != nullptr && !maskBroadcastsToScores(*mask, q, k))
    {
        return "the attention mask does not broadcast to [B, H, Sq, Skv]";
    }

    // The reference's own checks decide which shapes and band parameters can be derived,
    // including the [B, H, Sq, 1] stats layout it requires. It validates before writing
    // anything, so a rejected graph keeps its filled O and stats for the backward
    // reference to fail.
    try
    {
        const auto band = extractDiagonalBandParams(attributes, "SDPA backward input fill");
        switch(ioType)
        {
        case fb::DataType::HALF:
            forwardWithMaskType<hipdnn_data_sdk::types::half>(
                q, k, v, o, stats, mask, maskType, scale, band);
            break;
        case fb::DataType::BFLOAT16:
            forwardWithMaskType<hipdnn_data_sdk::types::bfloat16>(
                q, k, v, o, stats, mask, maskType, scale, band);
            break;
        default: // FLOAT, the only other type isReferenceType() lets through
            forwardWithMaskType<float>(q, k, v, o, stats, mask, maskType, scale, band);
            break;
        }
    }
    catch(const std::invalid_argument&)
    {
        return std::nullopt;
    }
    return std::nullopt;
}

} // namespace hipdnn_integration_tests
