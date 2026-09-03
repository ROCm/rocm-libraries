// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/FillInputs.hpp"

#include <algorithm>
#include <random>
#include <string>

#include <flatbuffers/flatbuffers.h>

namespace hipdnn_integration_tests
{
namespace
{

// ── Fill dispatch ───────────────────────────────────────────────────────────

FillResult
    fill(hipdnn_data_sdk::utilities::ITensor& tensor, const FillRecipe& recipe, unsigned int seed)
{
    switch(recipe.kind)
    {
    case FillRecipe::Kind::FREE:
        tensor.fillTensorWithRandomValues(recipe.lo, recipe.hi, seed);
        return FillResult::ok();
    case FillRecipe::Kind::FIXED:
        tensor.fillTensorWithValue(recipe.value);
        return FillResult::ok();
    default:
        return FillResult::unsupported("unknown FillRecipe kind");
    }
}

// ── Per-op init defaults ────────────────────────────────────────────────────
// Each function sets defaults for one node type via recipes.setDefault().
// setDefault uses try_emplace — if the test already set() a uid, the default
// is silently skipped.

// ── Batchnorm ────────────────────────────────────────────────────────────────

void setBatchnormInferenceInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
}

void setBatchnormInferenceVarianceInitDefaults(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node, InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setBatchnormTrainingInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
    recipes.setDefault(a->prev_running_mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->prev_running_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
    recipes.setDefault(a->momentum_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

void setBatchnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

void setLayernormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                              InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setLayernormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

// ── RMSNorm ──────────────────────────────────────────────────────────────────

void setRmsnormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                            InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setRmsnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->inv_rms_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// ── Block-scale quantization ─────────────────────────────────────────────────

void setBlockScaleDequantizeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return;
    }
    // [0.5, 2.0]: UE8M0 scales have zero mantissa bits, so any value stored
    // discretizes to a power of two ({0.5, 1.0, 2.0}), keeping dequantized
    // products within FP16 range.
    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.5f, 2.0f));
}

// ── SDPA ─────────────────────────────────────────────────────────────────────

void setSdpaForwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return;
    }

    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
}

void setSdpaBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }

    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
    recipes.setDefault(a->dropout_scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
    recipes.setDefault(a->dropout_scale_inv_tensor_uid(), FillRecipe::free(0.1f, 1.0f));

    recipes.setDefault(a->o_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->stats_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

bool applyDefaultFills(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                       InputFillRecipes& recipes)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::BatchnormInferenceAttributes:
        setBatchnormInferenceInitDefaults(node, recipes);
        return true;
    case NA::BatchnormInferenceAttributesVarianceExt:
        setBatchnormInferenceVarianceInitDefaults(node, recipes);
        return true;
    case NA::BatchnormAttributes:
        setBatchnormTrainingInitDefaults(node, recipes);
        return true;
    case NA::BatchnormBackwardAttributes:
        setBatchnormBackwardInitDefaults(node, recipes);
        return true;
    case NA::LayernormAttributes:
        setLayernormInitDefaults(node, recipes);
        return true;
    case NA::LayernormBackwardAttributes:
        setLayernormBackwardInitDefaults(node, recipes);
        return true;
    case NA::RMSNormAttributes:
        setRmsnormInitDefaults(node, recipes);
        return true;
    case NA::RMSNormBackwardAttributes:
        setRmsnormBackwardInitDefaults(node, recipes);
        return true;
    case NA::BlockScaleDequantizeAttributes:
        setBlockScaleDequantizeInitDefaults(node, recipes);
        return true;
    case NA::SdpaAttributes:
        setSdpaForwardInitDefaults(node, recipes);
        return true;
    case NA::SdpaBackwardAttributes:
        setSdpaBackwardInitDefaults(node, recipes);
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
    case NA::MoeGroupedMatmulAttributes:
    case NA::MoeGroupedMatmulBwdAttributes:
    case NA::NONE:
        return true;
    default:
        return false;
    }
}

} // anonymous namespace

FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes)
{
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        if(!applyDefaultFills(node, recipes))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "no input fill registered for op "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)"));
        }
    }

    // Sort so the rng sequence is deterministic regardless of discovery order.
    auto sortedUids = ownedUids;
    std::sort(sortedUids.begin(), sortedUids.end());

    std::mt19937 rng(recipes.globalSeed());
    for(const int64_t uid : sortedUids)
    {
        const unsigned int seed
            = recipes.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        auto fillResult = fill(*inputs.at(uid), recipes.fill(uid), seed);
        if(!fillResult.filled)
        {
            return FillResult::unsupported("uid " + std::to_string(uid) + ": " + fillResult.reason);
        }
    }

    return FillResult::ok();
}

} // namespace hipdnn_integration_tests
