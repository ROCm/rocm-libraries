// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/SynthesizeInputs.hpp"

#include <algorithm>
#include <random>
#include <string>

#include <flatbuffers/flatbuffers.h>

namespace hipdnn_integration_tests
{
namespace
{

// ── Fill dispatch ───────────────────────────────────────────────────────────

SynthesisResult
    fill(hipdnn_data_sdk::utilities::ITensor& tensor, const FillSpec& spec, unsigned int seed)
{
    switch(spec.kind)
    {
    case FillSpec::Kind::FREE:
        tensor.fillTensorWithRandomValues(spec.lo, spec.hi, seed);
        return SynthesisResult::ok();
    case FillSpec::Kind::FIXED:
        tensor.fillTensorWithValue(spec.value);
        return SynthesisResult::ok();
    case FillSpec::Kind::STRUCTURED:
        return SynthesisResult::unsupported("STRUCTURED fill not yet implemented");
    case FillSpec::Kind::DERIVED:
        return SynthesisResult::unsupported("DERIVED fill not yet implemented");
    default:
        return SynthesisResult::unsupported("unknown FillSpec kind");
    }
}

// ── Per-op init defaults ────────────────────────────────────────────────────
// Each function sets defaults for one node type via config.setDefault().
// setDefault uses try_emplace — if the test already set() a uid, the default
// is silently skipped.

// ── Batchnorm ────────────────────────────────────────────────────────────────

void setBatchnormInferenceInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), FillSpec::free(-0.1f, 0.1f));
    config.setDefault(a->inv_variance_tensor_uid(), FillSpec::free(0.5f, 1.5f));
}

void setBatchnormInferenceVarianceInitDefaults(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node, SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), FillSpec::free(-0.1f, 0.1f));
    config.setDefault(a->variance_tensor_uid(), FillSpec::free(0.5f, 1.5f));
    config.setDefault(a->epsilon_tensor_uid(), FillSpec::fixed(1e-5f));
}

void setBatchnormTrainingInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->epsilon_tensor_uid(), FillSpec::fixed(1e-5f));
    config.setDefault(a->prev_running_mean_tensor_uid(), FillSpec::free(-0.1f, 0.1f));
    config.setDefault(a->prev_running_variance_tensor_uid(), FillSpec::free(0.5f, 1.5f));
    config.setDefault(a->momentum_tensor_uid(), FillSpec::free(0.0f, 1.0f));

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            config.setDefault(uid, FillSpec::structured());
        }
    }
}

void setBatchnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), FillSpec::free(-0.1f, 0.1f));
    config.setDefault(a->inv_variance_tensor_uid(), FillSpec::free(0.5f, 1.5f));

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            config.setDefault(uid, FillSpec::structured());
        }
    }
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

void setLayernormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                              SynthesisConfig& config)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->epsilon_tensor_uid(), FillSpec::fixed(1e-5f));
}

void setLayernormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      SynthesisConfig& config)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), FillSpec::derived());
    config.setDefault(a->inv_variance_tensor_uid(), FillSpec::derived());
    config.setDefault(a->epsilon_tensor_uid(), FillSpec::fixed(1e-5f));
}

// ── RMSNorm ──────────────────────────────────────────────────────────────────

void setRmsnormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                            SynthesisConfig& config)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->epsilon_tensor_uid(), FillSpec::fixed(1e-5f));
}

void setRmsnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    SynthesisConfig& config)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->inv_rms_tensor_uid(), FillSpec::derived());
}

// ── Block-scale quantization ─────────────────────────────────────────────────

void setBlockScaleDequantizeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->scale_tensor_uid(), FillSpec::structured());
}

// ── SDPA ─────────────────────────────────────────────────────────────────────

void setSdpaForwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                SynthesisConfig& config)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return;
    }

    config.setDefault(a->scale_tensor_uid(), FillSpec::free(0.1f, 1.0f));

    config.setDefault(a->descale_q_tensor_uid(), FillSpec::structured());
    config.setDefault(a->descale_k_tensor_uid(), FillSpec::structured());
    config.setDefault(a->descale_v_tensor_uid(), FillSpec::structured());
    config.setDefault(a->descale_s_tensor_uid(), FillSpec::structured());
    config.setDefault(a->scale_s_tensor_uid(), FillSpec::structured());
    config.setDefault(a->scale_o_tensor_uid(), FillSpec::structured());

    config.setDefault(a->seq_len_q_tensor_uid(), FillSpec::structured());
    config.setDefault(a->seq_len_kv_tensor_uid(), FillSpec::structured());
    config.setDefault(a->page_table_k_tensor_uid(), FillSpec::structured());
    config.setDefault(a->page_table_v_tensor_uid(), FillSpec::structured());
    config.setDefault(a->block_mask_tensor_uid(), FillSpec::structured());
    config.setDefault(a->seed_tensor_uid(), FillSpec::structured());
    config.setDefault(a->offset_tensor_uid(), FillSpec::structured());
}

void setSdpaBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 SynthesisConfig& config)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }

    config.setDefault(a->scale_tensor_uid(), FillSpec::free(0.1f, 1.0f));
    config.setDefault(a->dropout_scale_tensor_uid(), FillSpec::free(0.1f, 1.0f));
    config.setDefault(a->dropout_scale_inv_tensor_uid(), FillSpec::free(0.1f, 1.0f));

    config.setDefault(a->o_tensor_uid(), FillSpec::derived());
    config.setDefault(a->stats_tensor_uid(), FillSpec::derived());

    config.setDefault(a->seq_len_q_tensor_uid(), FillSpec::structured());
    config.setDefault(a->seq_len_kv_tensor_uid(), FillSpec::structured());
    config.setDefault(a->seed_tensor_uid(), FillSpec::structured());
    config.setDefault(a->offset_tensor_uid(), FillSpec::structured());
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

bool applyDefaultFills(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                       SynthesisConfig& config)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::BatchnormInferenceAttributes:
        setBatchnormInferenceInitDefaults(node, config);
        return true;
    case NA::BatchnormInferenceAttributesVarianceExt:
        setBatchnormInferenceVarianceInitDefaults(node, config);
        return true;
    case NA::BatchnormAttributes:
        setBatchnormTrainingInitDefaults(node, config);
        return true;
    case NA::BatchnormBackwardAttributes:
        setBatchnormBackwardInitDefaults(node, config);
        return true;
    case NA::LayernormAttributes:
        setLayernormInitDefaults(node, config);
        return true;
    case NA::LayernormBackwardAttributes:
        setLayernormBackwardInitDefaults(node, config);
        return true;
    case NA::RMSNormAttributes:
        setRmsnormInitDefaults(node, config);
        return true;
    case NA::RMSNormBackwardAttributes:
        setRmsnormBackwardInitDefaults(node, config);
        return true;
    case NA::BlockScaleDequantizeAttributes:
        setBlockScaleDequantizeInitDefaults(node, config);
        return true;
    case NA::SdpaAttributes:
        setSdpaForwardInitDefaults(node, config);
        return true;
    case NA::SdpaBackwardAttributes:
        setSdpaBackwardInitDefaults(node, config);
        return true;
    // All-FREE ops: valid ops whose inputs need no special init (all default to FREE [-1,1]).
    case NA::PointwiseAttributes:
    case NA::ConvolutionFwdAttributes:
    case NA::ConvolutionBwdAttributes:
    case NA::ConvolutionWrwAttributes:
    case NA::MatmulAttributes:
    case NA::ReductionAttributes:
    case NA::ResampleFwdAttributes:
    case NA::BlockScaleQuantizeAttributes:
    case NA::CustomOpAttributes:
    case NA::NONE:
        return true;
    default:
        return false;
    }
}

} // anonymous namespace

SynthesisResult synthesizeInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                                 InputTensorMap& inputs,
                                 const std::vector<int64_t>& ownedUids,
                                 SynthesisConfig& config)
{
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        if(!applyDefaultFills(node, config))
        {
            const auto* name = node.name();
            return SynthesisResult::unsupported(
                "no input synthesis registered for op "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)"));
        }
    }

    // Sort so the rng sequence is deterministic regardless of discovery order.
    auto sortedUids = ownedUids;
    std::sort(sortedUids.begin(), sortedUids.end());

    std::mt19937 rng(config.globalSeed());
    for(const int64_t uid : sortedUids)
    {
        const unsigned int seed
            = config.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        auto fillResult = fill(*inputs.at(uid), config.fill(uid), seed);
        if(!fillResult.filled)
        {
            return SynthesisResult::unsupported("uid " + std::to_string(uid) + ": "
                                                + fillResult.reason);
        }
    }

    return SynthesisResult::ok();
}

} // namespace hipdnn_integration_tests
