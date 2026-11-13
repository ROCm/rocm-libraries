// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/SdpaBackwardInputs.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>

#include "harness/ReferenceCapabilityError.hpp"

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

// Whether the process sees a HIP device. A DeviceInputFiller is built without one, so a
// fill with a filler can still have no device to run the GPU forward on.
bool deviceVisible()
{
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

// The SDPA forward graph whose outputs are the backward node's O and stats: one forward
// node over the backward node's Q, K, V and forward attributes (scale, band and causal
// flags, additive mask, ALiBi, padding with its sequence lengths, the explicit dropout
// mask and scale), and every tensor it reads, ragged offsets included. Rank-3 stats are
// declared [B, H, Sq, 1] over the same memory, the shape both forward references take.
struct ForwardGraph
{
    std::vector<uint8_t> buffer;
    std::vector<int64_t> uids; // every tensor of the graph that has a buffer
};

ForwardGraph buildForwardGraph(const fb::Graph& graph,
                               const fb::SdpaBackwardAttributes& b,
                               const std::vector<int64_t>& tensorUids)
{
    fb::GraphT fwd;
    fwd.name = "sdpa_backward_forward_derivation";
    fwd.compute_data_type = graph.compute_data_type();
    fwd.intermediate_data_type = graph.intermediate_data_type();
    fwd.io_data_type = graph.io_data_type();

    ForwardGraph out;
    for(const int64_t uid : tensorUids)
    {
        std::unique_ptr<fb::TensorAttributesT> tensor(findTensor(graph, uid)->UnPack());
        if(uid == b.stats_tensor_uid() && tensor->dims.size() == 3)
        {
            tensor->dims.push_back(1);
            tensor->strides.push_back(1);
        }
        // A pass-by-value scale that is not a runtime value lives in the graph itself.
        const bool baked
            = hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(findTensor(graph, uid))
              && !tensor->is_runtime_pass_by_value;
        if(!baked)
        {
            out.uids.push_back(uid);
        }
        fwd.tensors.push_back(std::move(tensor));
    }

    fb::SdpaAttributesT a;
    a.q_tensor_uid = b.q_tensor_uid();
    a.k_tensor_uid = b.k_tensor_uid();
    a.v_tensor_uid = b.v_tensor_uid();
    a.o_tensor_uid = b.o_tensor_uid();
    a.stats_tensor_uid = b.stats_tensor_uid();
    a.generate_stats = true;
    a.attn_mask_tensor_uid = b.attn_mask_tensor_uid();
    a.scale_tensor_uid = b.scale_tensor_uid();
    a.seq_len_q_tensor_uid = b.seq_len_q_tensor_uid();
    a.seq_len_kv_tensor_uid = b.seq_len_kv_tensor_uid();
    a.dropout_mask_tensor_uid = b.dropout_mask_tensor_uid();
    a.dropout_scale_tensor_uid = b.dropout_scale_tensor_uid();
    a.alibi_mask = b.alibi_mask();
    a.padding_mask = b.padding_mask();
    a.causal_mask = b.causal_mask();
    a.causal_mask_bottom_right = b.causal_mask_bottom_right();
    a.attn_scale_value = b.attn_scale_value();
    a.left_bound = b.left_bound();
    a.right_bound = b.right_bound();
    a.diagonal_alignment = b.diagonal_alignment();
    // dropout_probability stays unset: an explicit mask carries the dropout, and the
    // caller refuses a nonzero probability without one (Philox).

    auto node = std::make_unique<fb::NodeT>();
    node->name = "sdpa_forward";
    node->compute_data_type = graph.compute_data_type();
    node->attributes.Set(a);
    fwd.nodes.push_back(std::move(node));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::Graph::Pack(builder, &fwd));
    out.buffer.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    return out;
}

std::unordered_map<int64_t, void*> variantPack(const fb::Graph& graph,
                                               const std::vector<int64_t>& uids,
                                               InputTensorMap& inputs,
                                               bool useDevice)
{
    std::unordered_map<int64_t, void*> pack;
    for(const int64_t uid : uids)
    {
        pack.emplace(
            uid,
            hipdnn_test_sdk::utilities::selectVariantPackPointer(
                *inputs.at(uid), useDevice, findTensor(graph, uid)->is_runtime_pass_by_value()));
    }
    return pack;
}

} // anonymous namespace

std::optional<std::string>
    deriveSdpaBackwardForwardOutputs(const fb::Graph& graph,
                                     const fb::SdpaBackwardAttributes& attributes,
                                     InputTensorMap& inputs,
                                     const std::vector<int64_t>& ownedUids,
                                     IReferenceExecutors& referenceExecutors,
                                     bool onDevice)
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

    // Philox dropout draws its mask from an engine's random stream, which no reference
    // reproduces, and a backward graph has no RNG dump. A zero probability is no
    // dropout, as the engines read it; an explicit mask carries its own dropout.
    const bool philox = (!attributes.dropout_mask_tensor_uid().has_value()
                         && attributes.dropout_probability().has_value()
                         && attributes.dropout_probability().value() != 0.f)
                        || attributes.seed_tensor_uid().has_value()
                        || attributes.offset_tensor_uid().has_value();
    if(philox)
    {
        return "Philox dropout: no reference reproduces an engine's random stream";
    }

    // Every tensor the forward reads must be a graph input: its values are only known
    // once the fill wrote them.
    std::vector<std::pair<std::string, int64_t>> operands = {{"Q", attributes.q_tensor_uid()},
                                                             {"K", attributes.k_tensor_uid()},
                                                             {"V", attributes.v_tensor_uid()}};
    const auto addOptional = [&operands](const char* name, std::optional<int64_t> uid) {
        if(uid.has_value())
        {
            operands.emplace_back(name, *uid);
        }
    };
    addOptional("the attention mask", attributes.attn_mask_tensor_uid());
    addOptional("seq_len_q", attributes.seq_len_q_tensor_uid());
    addOptional("seq_len_kv", attributes.seq_len_kv_tensor_uid());
    addOptional("the dropout mask", attributes.dropout_mask_tensor_uid());
    addOptional("the dropout scale", attributes.dropout_scale_tensor_uid());
    std::vector<std::pair<std::string, int64_t>> outputs
        = {{"O", attributes.o_tensor_uid()}, {"stats", attributes.stats_tensor_uid()}};

    std::vector<int64_t> tensorUids;
    for(const auto* list : {&operands, &outputs})
    {
        for(const auto& [name, uid] : *list)
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
            tensorUids.push_back(uid);
        }
    }
    // Ragged offsets of the operands and outputs (each listed once).
    for(std::size_t i = 0, n = tensorUids.size(); i < n; ++i)
    {
        const auto offsets = findTensor(graph, tensorUids[i])->ragged_offset_tensor_uid();
        if(!offsets.has_value()
           || std::find(tensorUids.begin(), tensorUids.end(), *offsets) != tensorUids.end())
        {
            continue;
        }
        if(findTensor(graph, *offsets) == nullptr || !isOwned(*offsets))
        {
            return "the ragged offsets (uid " + std::to_string(*offsets)
                   + ") are not a graph input";
        }
        tensorUids.push_back(*offsets);
    }
    if(attributes.scale_tensor_uid().has_value())
    {
        const int64_t scaleUid = *attributes.scale_tensor_uid();
        const auto* scale = findTensor(graph, scaleUid);
        if(scale == nullptr)
        {
            return "the scale tensor (uid " + std::to_string(scaleUid) + ") is not in the graph";
        }
        const bool baked = hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scale)
                           && !scale->is_runtime_pass_by_value();
        if(!baked && !isOwned(scaleUid))
        {
            return "the scale is computed by the graph";
        }
        tensorUids.push_back(scaleUid);
    }

    auto fwd = buildForwardGraph(graph, attributes, tensorUids);
    auto& o = *inputs.at(attributes.o_tensor_uid());
    auto& stats = *inputs.at(attributes.stats_tensor_uid());

    // On the device the GPU reference executor runs the forward where the device fill
    // wrote Q, K and V. A graph it declines, or rejects before it launches, goes to the
    // CPU reference executor.
    std::string gpuReason;
    if(onDevice && deviceVisible())
    {
        auto& gpu = referenceExecutors.get(ReferenceExecutorType::GPU);
        if(gpu.isApplicable(fwd.buffer.data(), fwd.buffer.size()))
        {
            try
            {
                gpu.execute(fwd.buffer.data(),
                            fwd.buffer.size(),
                            variantPack(graph, fwd.uids, inputs, /*useDevice=*/true));
                // Download O and stats as well, so they leave the fill current on the
                // host as after the CPU forward: a const host read of a tensor whose
                // device copy is newer throws. Both copies stay valid.
                o.markDeviceModified();
                stats.markDeviceModified();
                o.rawHostData();
                stats.rawHostData();
                return std::nullopt;
            }
            catch(const ReferenceCapabilityError& e)
            {
                gpuReason = e.what();
            }
            catch(const std::invalid_argument& e)
            {
                gpuReason = e.what();
            }
            catch(const std::exception& e)
            {
                // The device failed after accepting the graph, which is no fault in the
                // graph: as for the device fill, a failure rather than a fallback.
                throw DeviceInputError(std::string("deriving O and stats on the device failed: ")
                                       + e.what());
            }
        }
    }

    auto& cpu = referenceExecutors.get(ReferenceExecutorType::CPU);
    if(!cpu.isApplicable(fwd.buffer.data(), fwd.buffer.size()))
    {
        return "no reference executor models its forward graph"
               + (gpuReason.empty() ? std::string() : " (GPU: " + gpuReason + ")");
    }
    try
    {
        cpu.execute(fwd.buffer.data(),
                    fwd.buffer.size(),
                    variantPack(graph, fwd.uids, inputs, /*useDevice=*/false));
    }
    catch(const std::exception& e)
    {
        return std::string("the forward reference rejected its forward graph: ") + e.what();
    }
    o.markHostModified();
    stats.markHostModified();
    return std::nullopt;
}

} // namespace hipdnn_integration_tests
