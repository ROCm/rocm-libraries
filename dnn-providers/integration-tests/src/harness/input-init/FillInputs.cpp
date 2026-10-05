// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/FillInputs.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>

namespace hipdnn_integration_tests
{
namespace
{

// ── Fill dispatch ───────────────────────────────────────────────────────────

#if defined(USE_ROCRAND)
// Smallest tensor, in allocated elements, worth generating on the device.
//
// Measured on gfx1151: a warm rocRAND fill costs about 1.4 ms whatever its size, while
// the host loop costs 0.23 ms at 4096 elements, 4.4 ms at 65536 and 1.1 s at 16 M, so
// the two cross near 2^14 to 2^15. The first device fill in a process also pays about
// 350 ms for generator and scaling-kernel setup, which a suite of tiny tensors never
// earns back. Over the full suite, test bodies took 72.8 s with no threshold, 62.4 s at
// 2^12, 64.6 s at 2^16 and 75.3 s at 2^20; 2^14 sits between the two best.
constexpr size_t DEVICE_FILL_MIN_ELEMENTS = size_t{1} << 14;

template <class T>
bool fillOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor,
                  const FillRecipe& recipe,
                  unsigned int seed)
{
    auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
    if(typed == nullptr)
    {
        return false;
    }

    hipdnn_gpu_ref::common::gpu_fp_reference_tensor::fillWithRandomValues<T>(
        *typed,
        static_cast<T>(recipe.lo),
        static_cast<T>(recipe.hi),
        seed,
        /*synchronize=*/false);
    return true;
}

// Generates a FREE fill on the device when the tensor is big enough and of a type
// rocRAND fills. Leaves the fill in flight: fillInputs() waits once for all of them.
bool tryFillOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor,
                     const FillRecipe& recipe,
                     unsigned int seed)
{
    if(tensor.elementSpace() < DEVICE_FILL_MIN_ELEMENTS)
    {
        return false;
    }

    return fillOnDevice<float>(tensor, recipe, seed)
           || fillOnDevice<hipdnn_data_sdk::types::half>(tensor, recipe, seed)
           || fillOnDevice<hipdnn_data_sdk::types::bfloat16>(tensor, recipe, seed)
           || fillOnDevice<double>(tensor, recipe, seed);
}
#endif // USE_ROCRAND

// `deviceFillPending` is set when a fill was left running on the device.
FillResult fill(hipdnn_data_sdk::utilities::ITensor& tensor,
                const FillRecipe& recipe,
                unsigned int seed,
                FillPlacement placement,
                bool& deviceFillPending)
{
    switch(recipe.kind)
    {
    case FillRecipe::Kind::FREE:
#if defined(USE_ROCRAND)
        if(placement == FillPlacement::DEVICE && tryFillOnDevice(tensor, recipe, seed))
        {
            deviceFillPending = true;
            return FillResult::ok();
        }
#else
        static_cast<void>(placement);
        static_cast<void>(deviceFillPending);
#endif
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
    case NA::ResampleBwdAttributes:
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
                      InputFillRecipes& recipes,
                      FillPlacement placement)
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
    bool deviceFillPending = false;

    // One wait for every device fill, instead of one per tensor. Also on the way out
    // of a failed fill: the tensors are about to be dropped and their fills must not
    // still be writing into the memory being freed.
    const auto waitForDeviceFills = [&deviceFillPending] {
        if(!deviceFillPending)
        {
            return;
        }
        deviceFillPending = false;

        const hipError_t status = hipDeviceSynchronize();
        if(status != hipSuccess)
        {
            throw std::runtime_error(std::string("device input fill failed: ")
                                     + hipGetErrorString(status));
        }
    };

    for(const int64_t uid : sortedUids)
    {
        const unsigned int seed
            = recipes.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        auto fillResult
            = fill(*inputs.at(uid), recipes.fill(uid), seed, placement, deviceFillPending);
        if(!fillResult.filled)
        {
            waitForDeviceFills();
            return FillResult::unsupported("uid " + std::to_string(uid) + ": " + fillResult.reason);
        }
    }

    waitForDeviceFills();
    return FillResult::ok();
}

} // namespace hipdnn_integration_tests
