// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/SdpaBackwardInputs.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>

#if defined(USE_ROCRAND)
#include <hip/hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#endif

namespace hipdnn_integration_tests
{
namespace
{

using TensorDataTypes = std::unordered_map<int64_t, hipdnn_flatbuffers_sdk::data_objects::DataType>;

TensorDataTypes collectDataTypes(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    TensorDataTypes dataTypes;
    if(graph.tensors() == nullptr)
    {
        return dataTypes;
    }
    for(const auto* tensor : *graph.tensors())
    {
        dataTypes.emplace(tensor->uid(), tensor->data_type());
    }
    return dataTypes;
}

// Symmetric range over every finite value of T, so a uniform draw rounds onto
// each representable magnitude, including the extremes.
template <typename T>
FillRecipe symmetricFullRange()
{
    const auto max = static_cast<float>(std::numeric_limits<T>::max());
    return FillRecipe::free(-max, max);
}

// ── Fill dispatch ───────────────────────────────────────────────────────────

FillResult fill(hipdnn_data_sdk::utilities::ITensor& tensor,
                const FillRecipe& recipe,
                unsigned int seed,
                DeviceInputFiller* device)
{
    switch(recipe.kind)
    {
    case FillRecipe::Kind::FREE:
        if(device != nullptr && device->tryFill(tensor, recipe, seed))
        {
            return FillResult::ok(/*deviceFilled=*/1);
        }
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

// Largest magnitude a recipe can produce.
float recipeMagnitude(const FillRecipe& recipe)
{
    if(recipe.kind == FillRecipe::Kind::FIXED)
    {
        return std::fabs(recipe.value);
    }
    return std::max(std::fabs(recipe.lo), std::fabs(recipe.hi));
}

void setBlockScaleDequantizeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return;
    }

    // Scale is normalized against the operand range in effect (its data-type
    // default or a test override), as OCP MX does: scale ~ 2^-floor(log2(amax)) for
    // amax >= 1. The dequantized block then peaks between 2 and 4 (FP4/FP6:
    // 3 to 3.75), so products stay within FP16 range; an operand with
    // amax < 1 keeps e = 0 and peaks at 2 * amax. UE8M0 has no mantissa bits,
    // so the [0.5, 2] * 2^-e draw discretizes to three powers of two. An
    // operand in [-1, 1] gives e = 0, i.e. [0.5, 2].
    const float amax = recipeMagnitude(recipes.fill(a->x_tensor_uid()));
    const int e = amax >= 1.0f ? std::ilogb(amax) : 0;
    recipes.setDefault(a->scale_tensor_uid(),
                       FillRecipe::free(std::ldexp(0.5f, -e), std::ldexp(2.0f, -e)));
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

    // Overwritten by fillInputs() with the forward of Q, K and V. The fill stands only
    // where the CPU forward rejects the shapes, and the backward reference fails those.
    recipes.setDefault(a->o_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->stats_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// ── Softmax ──────────────────────────────────────────────────────────────────

void setSoftmaxInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                            InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SoftmaxAttributes();
    if(a == nullptr)
    {
        return;
    }

    // Logits spread over [-8, 8] instead of [-1, 1]: probabilities then span about
    // seven decades, so a kernel that skips the max subtraction or drops small terms
    // from the sum diverges visibly, while exp() stays far from overflow in every dtype.
    recipes.setDefault(a->p_tensor_uid(), FillRecipe::free(-8.0f, 8.0f));
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

void setRopeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                         InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RopeAttributes();
    if(a == nullptr)
    {
        return;
    }

    // freqs holds raw angles, position * inv_freq: up to a few thousand radians at long
    // context, not the [-1, 1] default, so a kernel whose sin/cos lacks full range
    // reduction (or reads the angles as precomputed cos/sin) diverges visibly.
    recipes.setDefault(a->freqs_tensor_uid(), FillRecipe::free(0.0f, 4096.0f));
}

// ── Gather ───────────────────────────────────────────────────────────────────

void setGatherInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                           const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                           InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_GatherAttributes();
    if(a == nullptr || graph.tensors() == nullptr || a->axis() < 0)
    {
        return;
    }
    const auto axis = static_cast<flatbuffers::uoffset_t>(a->axis());

    // Every index must lie in [0, x.dims[axis]); the reference throws on any other. An
    // integer tensor casts both bounds to its type before the uniform draw over [lo, hi) and
    // truncates each draw, so FREE[0, extent] reaches every row and never extent itself.
    for(const auto* x : *graph.tensors())
    {
        if(x->uid() == a->x_tensor_uid() && x->dims() != nullptr && axis < x->dims()->size())
        {
            const auto extent = static_cast<float>(x->dims()->Get(axis));
            recipes.setDefault(a->indices_tensor_uid(), FillRecipe::free(0.0f, extent));
            return;
        }
    }
}

// ── Paged cache store ────────────────────────────────────────────────────────

void setPagedCacheStoreInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_PagedCacheStoreAttributes();
    if(a == nullptr)
    {
        return;
    }

    // A scale away from 1, so a kernel that drops it diverges visibly. slot_mapping is
    // overwritten by assignPagedCacheStoreSlots() after the fill.
    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.5f, 2.0f));
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    findTensorAttributes(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph, int64_t uid)
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

// Writes `slots` into a T tensor in row-major index order; false when it is not one.
template <class T>
bool writeSlots(hipdnn_data_sdk::utilities::ITensor& tensor, const std::vector<int64_t>& slots)
{
    auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
    if(typed == nullptr)
    {
        return false;
    }
    const auto& dims = typed->dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(const auto slot : slots)
    {
        typed->setHostValue(static_cast<T>(slot), index);
        for(auto axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
    typed->memory().markHostModified();
    return true;
}

// A uniform draw cannot make a valid slot_mapping: the reference rejects two tokens on
// one slot, and a slot at or past num_blocks * block_size. So the slots are distinct
// draws from [0, num_blocks * block_size) in shuffled order, scattering the tokens over
// blocks and in-block offsets, and about a quarter of the tokens get a distinct negative
// slot, which every implementation skips. Seeded from the uid's seed, or else from the
// global seed and the uid, so a bundle's slots are reproducible.
std::optional<std::string> assignPagedCacheStoreSlots(
    const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::PagedCacheStoreAttributes& op,
    InputTensorMap& inputs,
    const std::vector<int64_t>& ownedUids,
    const InputFillRecipes& recipes)
{
    const auto slotUid = op.slot_mapping_tensor_uid();
    if(std::find(ownedUids.begin(), ownedUids.end(), slotUid) == ownedUids.end())
    {
        return std::nullopt;
    }
    const auto* container = findTensorAttributes(graph, op.container_tensor_uid());
    if(container == nullptr || container->dims() == nullptr || container->dims()->size() != 4)
    {
        return std::string("container must be [num_blocks, H, block_size, D]");
    }
    const int64_t slotCount = container->dims()->Get(0) * container->dims()->Get(2);
    auto& slotTensor = *inputs.at(slotUid);
    const auto tokens = static_cast<int64_t>(slotTensor.elementCount());
    if(tokens > slotCount)
    {
        return std::to_string(tokens) + " tokens cannot take distinct slots among "
               + std::to_string(slotCount);
    }

    std::mt19937 rng(recipes.resolveSeed(slotUid).value_or(recipes.globalSeed()
                                                           ^ static_cast<unsigned int>(slotUid)));
    std::vector<int64_t> slots(static_cast<size_t>(slotCount));
    std::iota(slots.begin(), slots.end(), int64_t{0});
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(static_cast<size_t>(tokens));
    for(auto& slot : slots)
    {
        if(rng() % 4 == 0)
        {
            slot = -1 - slot;
        }
    }

    if(!writeSlots<int32_t>(slotTensor, slots) && !writeSlots<int64_t>(slotTensor, slots))
    {
        return std::string("slot_mapping must be INT32 or INT64");
    }
    return std::nullopt;
}

// ── Varlen index operands: cu_seqlens and state_indices ──────────────────────
// Neither can be drawn element by element: cu_seqlens is a partition of the tokens and
// state_indices names distinct pool rows. Both are written after the generic fill
// (deriveVarlenIndexOperands()), whatever recipe the tensor had, as SDPA backward's O and
// stats are. Each draws from its own generator seeded by the tensor's seed override, or
// the global seed, and its uid, so a bundle's values never depend on its other inputs.

std::mt19937 varlenGenerator(const InputFillRecipes& recipes, int64_t uid)
{
    std::seed_seq seed{recipes.resolveSeed(uid).value_or(recipes.globalSeed()),
                       static_cast<unsigned int>(uid)};
    return std::mt19937(seed);
}

// Writes `values` into a rank-1 INT32 or INT64 tensor of the same length.
std::optional<std::string> writeIndexTensor(hipdnn_data_sdk::utilities::ITensor& tensor,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                            const std::vector<int64_t>& values)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    if(dataType != DataType::INT32 && dataType != DataType::INT64)
    {
        return "an index tensor must be INT32 or INT64, not "
               + std::string(EnumNameDataType(dataType));
    }
    if(tensor.dims().size() != 1 || tensor.dims()[0] != static_cast<int64_t>(values.size()))
    {
        return "an index tensor must be rank 1 with " + std::to_string(values.size()) + " elements";
    }
    for(size_t i = 0; i < values.size(); ++i)
    {
        void* element = tensor.hostDataOffsetFromIndex(tensor.getIndex(i));
        if(dataType == DataType::INT32)
        {
            *static_cast<int32_t*>(element) = static_cast<int32_t>(values[i]);
        }
        else
        {
            *static_cast<int64_t*>(element) = values[i];
        }
    }
    tensor.markHostModified();
    return std::nullopt;
}

// Offsets of `sequences` sequences over `tokens` tokens: 0 first, `tokens` last,
// non-decreasing. With at least one token per sequence, one sequence chosen at random
// gets exactly one token, shorter than any conv window, so a kernel that mishandles a
// sequence shorter than its state is caught; the rest split the remaining tokens at
// random cut points. With fewer tokens than sequences, a random subset gets one token
// each and the others are empty.
std::vector<int64_t> cuSeqlensValues(int64_t sequences, int64_t tokens, std::mt19937& rng)
{
    std::vector<int64_t> lengths(static_cast<size_t>(sequences), 0);
    if(tokens < sequences)
    {
        std::fill_n(lengths.begin(), tokens, 1);
        for(size_t i = lengths.size() - 1; i > 0; --i)
        {
            std::swap(lengths[i], lengths[rng() % (i + 1)]);
        }
    }
    else if(sequences == 1)
    {
        lengths[0] = tokens;
    }
    else
    {
        const auto shortest = static_cast<size_t>(rng() % static_cast<uint64_t>(sequences));
        const int64_t extra = tokens - sequences;
        std::vector<int64_t> cuts;
        cuts.reserve(static_cast<size_t>(std::max<int64_t>(sequences - 2, 0)) + 1);
        for(int64_t i = 0; i < sequences - 2; ++i)
        {
            cuts.push_back(static_cast<int64_t>(rng() % static_cast<uint64_t>(extra + 1)));
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.push_back(extra);
        int64_t previous = 0;
        size_t cut = 0;
        for(size_t n = 0; n < lengths.size(); ++n)
        {
            lengths[n] = 1;
            if(n != shortest)
            {
                lengths[n] += cuts[cut] - previous;
                previous = cuts[cut++];
            }
        }
    }
    std::vector<int64_t> offsets(lengths.size() + 1, 0);
    for(size_t n = 0; n < lengths.size(); ++n)
    {
        offsets[n + 1] = offsets[n] + lengths[n];
    }
    return offsets;
}

// Pool rows of `sequences` sequences in a pool of `poolRows`: distinct and in range,
// in random order, so a kernel that ignores the indirection reads and writes the wrong
// rows. A pool smaller than the batch repeats rows, which only a graph that never writes
// the pool may use.
std::vector<int64_t> stateIndexValues(int64_t sequences, int64_t poolRows, std::mt19937& rng)
{
    std::vector<int64_t> rows(static_cast<size_t>(std::max(poolRows, int64_t{1})));
    for(size_t i = 0; i < rows.size(); ++i)
    {
        rows[i] = static_cast<int64_t>(i);
    }
    std::vector<int64_t> picked(static_cast<size_t>(sequences));
    for(size_t n = 0; n < picked.size(); ++n)
    {
        if(n < rows.size())
        {
            std::swap(rows[n], rows[n + rng() % (rows.size() - n)]);
            picked[n] = rows[n];
        }
        else
        {
            picked[n] = picked[n % rows.size()];
        }
    }
    return picked;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    findGraphTensor(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph, int64_t uid)
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

// Extent of `axis` of tensor `uid` in the graph, or nullopt when it has no such axis.
std::optional<int64_t> graphTensorExtent(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                                         int64_t uid,
                                         flatbuffers::uoffset_t axis)
{
    const auto* tensor = findGraphTensor(graph, uid);
    if(tensor == nullptr || tensor->dims() == nullptr || axis >= tensor->dims()->size())
    {
        return std::nullopt;
    }
    return tensor->dims()->Get(axis);
}

// Fills cu_seqlens `uid` ([N + 1]) with the offsets of N sequences over `tokens` tokens,
// if the bundle owns it. A cu_seqlens the graph computes is left alone.
std::optional<std::string> fillCuSeqlens(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                                         InputTensorMap& inputs,
                                         const std::vector<int64_t>& ownedUids,
                                         const InputFillRecipes& recipes,
                                         int64_t uid,
                                         int64_t tokens)
{
    if(std::find(ownedUids.begin(), ownedUids.end(), uid) == ownedUids.end())
    {
        return std::nullopt;
    }
    const auto* attributes = findGraphTensor(graph, uid);
    auto& tensor = *inputs.at(uid);
    if(attributes == nullptr || tensor.dims().size() != 1 || tensor.dims()[0] < 2)
    {
        return "cu_seqlens must be [N + 1] with N >= 1";
    }
    auto rng = varlenGenerator(recipes, uid);
    const auto why = writeIndexTensor(
        tensor, attributes->data_type(), cuSeqlensValues(tensor.dims()[0] - 1, tokens, rng));
    return why ? std::optional<std::string>("cu_seqlens: " + *why) : std::nullopt;
}

// Fills state_indices `uid` ([N]) with distinct rows of a pool of `poolRows`, if the
// bundle owns it.
std::optional<std::string>
    fillStateIndices(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                     InputTensorMap& inputs,
                     const std::vector<int64_t>& ownedUids,
                     const InputFillRecipes& recipes,
                     int64_t uid,
                     int64_t poolRows)
{
    if(std::find(ownedUids.begin(), ownedUids.end(), uid) == ownedUids.end())
    {
        return std::nullopt;
    }
    const auto* attributes = findGraphTensor(graph, uid);
    auto& tensor = *inputs.at(uid);
    if(attributes == nullptr || tensor.dims().size() != 1)
    {
        return "state_indices must be [N]";
    }
    auto rng = varlenGenerator(recipes, uid);
    const auto why = writeIndexTensor(
        tensor, attributes->data_type(), stateIndexValues(tensor.dims()[0], poolRows, rng));
    return why ? std::optional<std::string>("state_indices: " + *why) : std::nullopt;
}

// cu_seqlens over x's T tokens; state_indices over initial_state's P pool rows.
std::optional<std::string>
    deriveCausalConv1dIndices(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                              const hipdnn_flatbuffers_sdk::data_objects::CausalConv1dAttributes& a,
                              InputTensorMap& inputs,
                              const std::vector<int64_t>& ownedUids,
                              const InputFillRecipes& recipes)
{
    const auto tokens = graphTensorExtent(graph, a.x_tensor_uid(), 0);
    if(!tokens)
    {
        return "x has no token axis";
    }
    if(auto why
       = fillCuSeqlens(graph, inputs, ownedUids, recipes, a.cu_seqlens_tensor_uid(), *tokens))
    {
        return why;
    }
    if(!a.state_indices_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const auto poolRows = a.initial_state_tensor_uid().has_value()
                              ? graphTensorExtent(graph, *a.initial_state_tensor_uid(), 0)
                              : std::nullopt;
    if(!poolRows)
    {
        return "state_indices needs an initial_state pool [P, C, W - 1]";
    }
    return fillStateIndices(
        graph, inputs, ownedUids, recipes, *a.state_indices_tensor_uid(), *poolRows);
}

// Writes every varlen index operand of the graph. Returns why one cannot be written.
std::optional<std::string>
    deriveVarlenIndexOperands(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                              InputTensorMap& inputs,
                              const std::vector<int64_t>& ownedUids,
                              const InputFillRecipes& recipes)
{
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        std::optional<std::string> why;
        if(const auto* a = node.attributes_as_CausalConv1dAttributes())
        {
            why = deriveCausalConv1dIndices(graph, *a, inputs, ownedUids, recipes);
        }
        if(why)
        {
            const auto* name = node.name();
            return "node " + std::string(name != nullptr ? name->c_str() : "(unnamed)") + ": "
                   + *why;
        }
    }
    return std::nullopt;
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

bool applyDefaultFills(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                       const hipdnn_flatbuffers_sdk::data_objects::Node& node,
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
    case NA::SoftmaxAttributes:
        setSoftmaxInitDefaults(node, recipes);
        return true;
    case NA::RopeAttributes:
        setRopeInitDefaults(node, recipes);
        return true;
    case NA::GatherAttributes:
        setGatherInitDefaults(graph, node, recipes);
        return true;
    case NA::PagedCacheStoreAttributes:
        setPagedCacheStoreInitDefaults(node, recipes);
        return true;
    // x, weight, bias and the states take the [-1, 1] default; cu_seqlens and
    // state_indices are written after the fill (deriveVarlenIndexOperands()).
    case NA::CausalConv1dAttributes:
    // x takes the [-1, 1] default. Ties (frequent in bf16 / f16 rows of thousands) need no
    // avoiding: the FlyDSL kernel and the CPU reference both rank equal values by position.
    case NA::TopkAttributes:
    // Ops whose inputs need no special init; they get defaultFillFor(dtype).
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

#if defined(USE_ROCRAND)
using RocRandGenerator = hipdnn_gpu_ref::common::RocRandGenerator;
#endif

struct DeviceInputFiller::Impl
{
#if defined(USE_ROCRAND)
    std::unique_ptr<RocRandGenerator> generator;
    bool fillPending = false;

    // Starts the fill when `tensor` is a T tensor; creates the generator on first use,
    // so a tensor of a type rocRAND does not fill never costs one.
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

        if(generator == nullptr)
        {
            generator = std::make_unique<RocRandGenerator>(ROCRAND_RNG_PSEUDO_DEFAULT);
        }

        // Set before the launch: a fill that throws partway may already have queued work.
        fillPending = true;
        hipdnn_gpu_ref::common::gpu_fp_reference_tensor::gpuFillWithRandomValues<T>(
            *typed,
            static_cast<T>(recipe.lo),
            static_cast<T>(recipe.hi),
            seed,
            *generator,
            /*synchronize=*/false);
        return true;
    }
#endif
};

DeviceInputFiller::DeviceInputFiller()
    : _impl(std::make_unique<Impl>())
{
}

DeviceInputFiller::~DeviceInputFiller()
{
    // The generator is about to be destroyed; a fill must not still be using it.
    try
    {
        waitForFills();
    }
    catch(...) // NOLINT(bugprone-empty-catch): nothing useful to do about it here
    {
    }
}

bool DeviceInputFiller::isSupported()
{
#if defined(USE_ROCRAND)
    return true;
#else
    return false;
#endif
}

bool DeviceInputFiller::tryFill([[maybe_unused]] hipdnn_data_sdk::utilities::ITensor& tensor,
                                [[maybe_unused]] const FillRecipe& recipe,
                                [[maybe_unused]] unsigned int seed)
{
#if defined(USE_ROCRAND)
    if(tensor.elementSpace() < minElements())
    {
        return false;
    }

    try
    {
        auto& impl = *_impl;
        return impl.fillOnDevice<float>(tensor, recipe, seed)
               || impl.fillOnDevice<hipdnn_data_sdk::types::half>(tensor, recipe, seed)
               || impl.fillOnDevice<hipdnn_data_sdk::types::bfloat16>(tensor, recipe, seed)
               || impl.fillOnDevice<double>(tensor, recipe, seed);
    }
    catch(const std::exception& e)
    {
        // Generator creation, the scaling kernel's compile, scratch allocation and the
        // launches are all device work; none of it is a fault in the graph under test.
        throw DeviceInputError(std::string("device input fill failed: ") + e.what());
    }
#else
    return false;
#endif
}

void DeviceInputFiller::waitForFills()
{
#if defined(USE_ROCRAND)
    if(!_impl->fillPending)
    {
        return;
    }
    _impl->fillPending = false;

    const hipError_t status = hipDeviceSynchronize();
    if(status != hipSuccess)
    {
        throw DeviceInputError(std::string("device input fill failed: ")
                               + hipGetErrorString(status));
    }
#endif
}

namespace
{

// Waits for the device fills on every way out of fillInputs(), including an exception
// from a fill that threw partway: the tensors are about to be dropped, and fills
// already queued must not still be writing into memory being freed. Errors are
// swallowed here, since the one already propagating is the one worth reporting; the
// normal path waits explicitly and sees them.
class WaitForFillsOnExit
{
public:
    explicit WaitForFillsOnExit(DeviceInputFiller* device)
        : _device(device)
    {
    }

    ~WaitForFillsOnExit()
    {
        if(_device == nullptr)
        {
            return;
        }
        try
        {
            _device->waitForFills();
        }
        catch(...) // NOLINT(bugprone-empty-catch): see above
        {
        }
    }

    WaitForFillsOnExit(const WaitForFillsOnExit&) = delete;
    WaitForFillsOnExit& operator=(const WaitForFillsOnExit&) = delete;
    WaitForFillsOnExit(WaitForFillsOnExit&&) = delete;
    WaitForFillsOnExit& operator=(WaitForFillsOnExit&&) = delete;

private:
    DeviceInputFiller* _device;
};

} // anonymous namespace

FillRecipe defaultFillFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    namespace types = hipdnn_data_sdk::types;

    switch(dataType)
    {
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
        return FillRecipe{};
    // Few enough codes that a uniform draw over the full range reaches all of them;
    // [-1, 1] would only reach 3 of the 8 FP4 magnitudes.
    case DataType::FP4_E2M1:
        return symmetricFullRange<types::fp4_e2m1>();
    case DataType::FP6_E2M3:
        return symmetricFullRange<types::fp6_e2m3>();
    case DataType::FP6_E3M2:
        return symmetricFullRange<types::fp6_e3m2>();
    // FP8 codes are spread logarithmically, so no uniform range reaches most of
    // them: a wide range leaves the small exponents empty. Full coverage needs a
    // log-uniform or code-uniform distribution, which ITensor cannot draw yet.
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return FillRecipe{};
    // Power-of-two scale; [0.5, 2] discretizes to {0.5, 1, 2}.
    case DataType::FP8_E8M0:
        return FillRecipe::free(0.5f, 2.0f);
    // Integer draws from [-1, 1] collapse to a few values. A meaningful range is
    // op-specific (indices, offsets, masks), so it belongs in per-op defaults.
    case DataType::UINT8:
    case DataType::INT32:
    case DataType::INT8:
    case DataType::INT4:
    case DataType::INT64:
    case DataType::BOOLEAN:
        return FillRecipe{};
    case DataType::UNSET:
    default:
        throw std::invalid_argument("defaultFillFor: no fill default for data type "
                                    + std::string(EnumNameDataType(dataType)));
    }
}

FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes,
                      DeviceInputFiller* device)
{
    // Dtype defaults sit below test and per-op recipes. Only those that differ
    // from the generic FREE[-1, 1] are recorded, so meta.inputs stays unchanged
    // for float graphs and names the range for narrow formats. Registered
    // before the per-op pass so a default derived from another input's range
    // (block-scale dequantize's scale) sees that input's final range.
    const auto dataTypes = collectDataTypes(graph);
    for(const int64_t uid : ownedUids)
    {
        const auto dataType = dataTypes.find(uid);
        if(dataType == dataTypes.end())
        {
            continue;
        }
        const auto recipe = defaultFillFor(dataType->second);
        if(recipe != FillRecipe{})
        {
            recipes.setTypeDefault(uid, recipe);
        }
    }

    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        if(!applyDefaultFills(graph, node, recipes))
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

    const WaitForFillsOnExit waitOnExit(device);

    std::size_t deviceFilled = 0;
    for(const int64_t uid : sortedUids)
    {
        const unsigned int seed
            = recipes.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        auto fillResult = fill(*inputs.at(uid), recipes.fill(uid), seed, device);
        if(!fillResult.filled)
        {
            return FillResult::unsupported("uid " + std::to_string(uid) + ": " + fillResult.reason);
        }
        deviceFilled += fillResult.deviceFilled;
    }

    // One wait for every device fill, instead of one per tensor.
    if(device != nullptr)
    {
        device->waitForFills();
    }

    // O and the LSE of an SDPA backward node are outputs of a forward, not free inputs:
    // once Q, K and V hold their values, recompute them. Done after the waits, since
    // the forward reads Q, K and V on the host.
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        const auto* sdpaBackward = node.attributes_as_SdpaBackwardAttributes();
        if(sdpaBackward == nullptr)
        {
            continue;
        }
        if(auto why = deriveSdpaBackwardForwardOutputs(graph, *sdpaBackward, inputs, ownedUids))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "cannot derive O and stats of SDPA backward node "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)")
                + " from its Q, K and V: " + *why);
        }
    }

    // A paged cache store's slot_mapping is a set of distinct slots, not a free draw.
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        const auto* pagedCacheStore = node.attributes_as_PagedCacheStoreAttributes();
        if(pagedCacheStore == nullptr)
        {
            continue;
        }
        if(auto why
           = assignPagedCacheStoreSlots(graph, *pagedCacheStore, inputs, ownedUids, recipes))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "cannot assign slots of paged cache store node "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)") + ": " + *why);
        }
    }

    // cu_seqlens and state_indices are structured, not drawn element by element.
    if(auto why = deriveVarlenIndexOperands(graph, inputs, ownedUids, recipes))
    {
        return FillResult::unsupported("cannot fill the varlen index operands of " + *why);
    }
    return FillResult::ok(deviceFilled);
}

std::vector<int64_t> inPlaceOutputUids(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    std::vector<int64_t> uids;
    if(graph.nodes() == nullptr)
    {
        return uids;
    }
    for(const auto* node : *graph.nodes())
    {
        switch(node->attributes_type())
        {
        case NA::PagedCacheStoreAttributes:
            uids.push_back(node->attributes_as_PagedCacheStoreAttributes()->container_tensor_uid());
            break;
        // A final_state pool addressed through state_indices keeps the rows no sequence
        // names.
        case NA::CausalConv1dAttributes:
        {
            const auto* a = node->attributes_as_CausalConv1dAttributes();
            if(a->state_indices_tensor_uid().has_value() && a->final_state_tensor_uid().has_value())
            {
                uids.push_back(*a->final_state_tensor_uid());
            }
            break;
        }
        default:
            break;
        }
    }
    return uids;
}

} // namespace hipdnn_integration_tests
