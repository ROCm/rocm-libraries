// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/SdpaBackwardInputs.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_map>

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

// Anything the forward applies besides the scale and the diagonal band, none of
// which the derivation reproduces.
bool hasUnmodeledForwardTerms(const fb::SdpaBackwardAttributes& a)
{
    return a.attn_mask_tensor_uid().has_value() || a.alibi_mask() || a.padding_mask()
           || a.seq_len_q_tensor_uid().has_value() || a.seq_len_kv_tensor_uid().has_value()
           || a.dropout_probability().has_value() || a.seed_tensor_uid().has_value()
           || a.offset_tensor_uid().has_value() || a.dropout_mask_tensor_uid().has_value()
           || a.dropout_scale_tensor_uid().has_value()
           || a.dropout_scale_inv_tensor_uid().has_value();
}

template <typename T>
void forwardInto(ITensor& q,
                 ITensor& k,
                 ITensor& v,
                 ITensor& o,
                 ITensor& stats,
                 std::optional<float> scale,
                 const DiagonalBandParams& band)
{
    const ShallowTensor<T> qView(q.rawHostData(), q.dims(), q.strides());
    const ShallowTensor<T> kView(k.rawHostData(), k.dims(), k.strides());
    const ShallowTensor<T> vView(v.rawHostData(), v.dims(), v.strides());
    ShallowTensor<T> oView(o.rawHostData(), o.dims(), o.strides());
    ShallowTensor<float> lseView(stats.rawHostData(), stats.dims(), stats.strides());

    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward<T, T, T, T, float>(
        qView,
        kView,
        vView,
        oView,
        scale,
        nullptr,
        band.leftBound,
        band.rightBound,
        band.topLeftAlignment,
        &lseView);

    // The views write the host copies; a device-resident tensor uploads them only
    // once told the host copy is the current one.
    o.markHostModified();
    stats.markHostModified();
}

} // anonymous namespace

void deriveSdpaBackwardForwardOutputs(const fb::Graph& graph,
                                      const fb::SdpaBackwardAttributes& attributes,
                                      InputTensorMap& inputs,
                                      const std::vector<int64_t>& ownedUids)
{
    if(hasUnmodeledForwardTerms(attributes))
    {
        return;
    }

    const auto isOwned = [&ownedUids](int64_t uid) {
        return std::find(ownedUids.begin(), ownedUids.end(), uid) != ownedUids.end();
    };

    const std::array<int64_t, 5> uids = {attributes.q_tensor_uid(),
                                         attributes.k_tensor_uid(),
                                         attributes.v_tensor_uid(),
                                         attributes.o_tensor_uid(),
                                         attributes.stats_tensor_uid()};
    for(const int64_t uid : uids)
    {
        const auto* tensor = findTensor(graph, uid);
        if(tensor == nullptr || !isOwned(uid) || tensor->ragged_offset_tensor_uid().has_value())
        {
            return;
        }
    }

    const auto ioType = findTensor(graph, uids[0])->data_type();
    for(std::size_t i = 1; i < 4; ++i)
    {
        if(findTensor(graph, uids[i])->data_type() != ioType)
        {
            return;
        }
    }
    if(findTensor(graph, uids[4])->data_type() != fb::DataType::FLOAT)
    {
        return;
    }

    // A scale tensor takes precedence over attn_scale_value, as in the CPU reference's
    // plan; neither means no scaling. The reference reads a scale tensor only as a
    // pass-by-value scalar, so any other scale tensor leaves O and stats as filled.
    std::optional<float> scale = attributes.attn_scale_value();
    if(attributes.scale_tensor_uid().has_value())
    {
        const int64_t scaleUid = *attributes.scale_tensor_uid();
        const auto* scaleTensor = findTensor(graph, scaleUid);
        if(!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scaleTensor)
           || (scaleTensor->is_runtime_pass_by_value() && !isOwned(scaleUid)))
        {
            return;
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

    auto& q = *inputs.at(uids[0]);
    auto& k = *inputs.at(uids[1]);
    auto& v = *inputs.at(uids[2]);
    auto& o = *inputs.at(uids[3]);
    auto& stats = *inputs.at(uids[4]);

    // The reference's own checks decide which shapes and band parameters can be derived,
    // including the [B, H, Sq, 1] stats layout it requires. It validates before writing
    // anything, so a rejected graph keeps its filled O and stats, and the backward
    // reference rejects the same graph when it runs.
    try
    {
        const auto band = extractDiagonalBandParams(attributes, "SDPA backward input fill");
        switch(ioType)
        {
        case fb::DataType::FLOAT:
            forwardInto<float>(q, k, v, o, stats, scale, band);
            break;
        case fb::DataType::HALF:
            forwardInto<hipdnn_data_sdk::types::half>(q, k, v, o, stats, scale, band);
            break;
        case fb::DataType::BFLOAT16:
            forwardInto<hipdnn_data_sdk::types::bfloat16>(q, k, v, o, stats, scale, band);
            break;
        default:
            return;
        }
    }
    catch(const std::invalid_argument&)
    {
        return;
    }
}

} // namespace hipdnn_integration_tests
