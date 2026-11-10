// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp"

/**
 * @file FlydslCausalConv1dCommon.hpp
 * @brief How hipkernel:flydsl_causal_conv1d reads a graph: the problem one launch of
 *        `flydsl/kernels_src/kernels/causal_conv1d/causal_conv1d_kernel.py` computes, and
 *        every condition under which the kernel computes it exactly. Shared by the pack's
 *        matchers and dispatch, and by its host-only tests.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_causal_conv1d
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The `flags` kernarg: one bit per optional operand (the kernel's FLAG_* constants;
/// gen_causal_conv1d.py asserts the kernel still says these).
constexpr int32_t FLAG_BIAS = 1;
constexpr int32_t FLAG_INITIAL_STATE = 2;
constexpr int32_t FLAG_FINAL_STATE = 4;
constexpr int32_t FLAG_STATE_INDICES = 8;

constexpr int64_t MAX_INT32 = std::numeric_limits<int32_t>::max();
/// The kernel's token positions are int32, and a token tile's end (first + tile) must not
/// overflow: tiles are at most 64 tokens.
constexpr int64_t MAX_TOKENS = MAX_INT32 - 64;
/// cu_seqlens is read through a scalar buffer load with 32-bit byte offsets, up to two
/// words per entry (int64).
constexpr int64_t MAX_SEQUENCES = MAX_INT32 / 8 - 1;
/// The grid's y extent: the channel blocks.
constexpr int64_t MAX_GRID_Y = 65535;

/// One launch's problem: extents, element strides, the index widths and the optional
/// operands' flags. Strides of an axis of extent 1 are normalised to the packed value, which
/// is never stepped along and lets the kernel's packed fast path see a packed operand.
struct CausalConv1dProblem
{
    data_objects::DataType dataType = data_objects::DataType::UNSET;
    bool silu = false;
    int64_t tokens = 0;
    int64_t channels = 0;
    int64_t width = 0;
    int64_t sequences = 0;
    /// Rows of the state tensors: N, or the pool size P with state_indices.
    int64_t stateRows = 0;
    int64_t xStride = 0;
    int64_t yStride = 0;
    int64_t weightStrideC = 0;
    int64_t weightStrideK = 0;
    int64_t biasStride = 0;
    std::array<int64_t, 3> initialStrides{};
    std::array<int64_t, 3> finalStrides{};
    /// 32-bit words per cu_seqlens / state_indices entry: 1 for INT32, 2 for INT64.
    int32_t cuSeqlensStep = 1;
    int32_t stateIndicesStep = 1;
    int32_t flags = 0;
};

/// 32-bit words per element of an index tensor, or nullopt for a dtype the kernel does not
/// read.
inline std::optional<int32_t> indexWords(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::INT32:
        return 1;
    case data_objects::DataType::INT64:
        return 2;
    default:
        return std::nullopt;
    }
}

/// @p tensor's dims, when it has a usable rank-@p rank shape with non-negative strides.
inline std::optional<std::vector<int64_t>> dimsOf(const data_objects::TensorAttributes& tensor,
                                                  uint32_t rank)
{
    if(!flydsl_norm::hasUsableShape(tensor, rank) || tensor.dims()->size() != rank)
    {
        return std::nullopt;
    }
    std::vector<int64_t> dims(tensor.dims()->begin(), tensor.dims()->end());
    for(uint32_t axis = 0; axis < rank; ++axis)
    {
        if(dims[axis] <= 0 || tensor.strides()->Get(axis) < 0)
        {
            return std::nullopt;
        }
    }
    return dims;
}

/// @p tensor's strides with every unit-extent axis's stride replaced by its packed value.
inline std::vector<int64_t> normalisedStrides(const data_objects::TensorAttributes& tensor)
{
    const auto& dims = *tensor.dims();
    std::vector<int64_t> strides(tensor.strides()->begin(), tensor.strides()->end());
    int64_t packed = 1;
    for(auto axis = static_cast<int64_t>(dims.size()) - 1; axis >= 0; --axis)
    {
        const auto index = static_cast<size_t>(axis);
        if(dims.Get(static_cast<flatbuffers::uoffset_t>(axis)) == 1)
        {
            strides[index] = packed;
        }
        packed *= dims.Get(static_cast<flatbuffers::uoffset_t>(axis));
    }
    return strides;
}

/// Whether no two elements of a tensor of @p dims and @p strides share an address: sorted
/// by stride, each stepped axis's stride clears the extent of everything inside it. An
/// output the kernel writes must be, or two of its writes would race.
inline bool isNonOverlapping(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    std::vector<std::pair<int64_t, int64_t>> axes;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        if(dims[axis] > 1)
        {
            axes.emplace_back(strides[axis], dims[axis]);
        }
    }
    std::sort(axes.begin(), axes.end());
    int64_t reach = 1;
    for(const auto& [stride, extent] : axes)
    {
        if(stride < reach || stride > std::numeric_limits<int64_t>::max() / extent)
        {
            return false;
        }
        reach = stride * extent;
    }
    return true;
}

inline bool declaresWordAlignment(const data_objects::TensorAttributes& tensor)
{
    return tensor.alignment() >= 4 && tensor.alignment() % 4 == 0;
}

/**
 * @brief The problem a causal-conv1d node poses, or nullopt when the kernel cannot compute
 *        it exactly.
 *
 * Every operand must be a real device tensor. x, weight, y, bias and both states share one
 * dtype (bf16, f16 or f32: the CPU reference's single DataType), cu_seqlens and
 * state_indices are INT32 or INT64 and contiguous, and the shapes are the schema's. x and y
 * have a unit channel stride; y and final_state must not overlap themselves. Every other
 * stride is a kernel argument. Values that only exist at execute time -- the offsets'
 * order, the indices' range and distinctness -- are the caller's contract, as in the CPU
 * reference, which throws on a violation.
 */
inline std::optional<CausalConv1dProblem>
    resolveCausalConv1dProblem(const MatchContext& context,
                               const data_objects::CausalConv1dAttributes& attributes)
{
    const auto find = [&context](int64_t uid) { return flydsl_norm::findTensor(context, uid); };
    const auto* x = find(attributes.x_tensor_uid());
    const auto* weight = find(attributes.weight_tensor_uid());
    const auto* cuSeqlens = find(attributes.cu_seqlens_tensor_uid());
    const auto* y = find(attributes.y_tensor_uid());
    if(x == nullptr || weight == nullptr || cuSeqlens == nullptr || y == nullptr)
    {
        return std::nullopt;
    }
    const auto optional = [&find](const flatbuffers::Optional<int64_t>& uid,
                                  bool& missing) -> const data_objects::TensorAttributes* {
        if(!uid.has_value())
        {
            return nullptr;
        }
        const auto* tensor = find(*uid);
        missing = missing || tensor == nullptr;
        return tensor;
    };
    bool missing = false;
    const auto* bias = optional(attributes.bias_tensor_uid(), missing);
    const auto* initialState = optional(attributes.initial_state_tensor_uid(), missing);
    const auto* indices = optional(attributes.state_indices_tensor_uid(), missing);
    const auto* finalState = optional(attributes.final_state_tensor_uid(), missing);
    if(missing || (indices != nullptr && initialState == nullptr))
    {
        return std::nullopt;
    }

    CausalConv1dProblem problem;
    const auto activation = attributes.activation();
    if(activation != data_objects::CausalConv1dActivation::IDENTITY
       && activation != data_objects::CausalConv1dActivation::SILU)
    {
        return std::nullopt;
    }
    problem.silu = activation == data_objects::CausalConv1dActivation::SILU;

    // Every operand a real device tensor; one dtype across the floating-point ones.
    problem.dataType = x->data_type();
    if(!flydsl_norm::isSupportedDataType(problem.dataType))
    {
        return std::nullopt;
    }
    for(const auto* tensor : {x, weight, cuSeqlens, y, bias, initialState, indices, finalState})
    {
        if(tensor != nullptr && !flydsl_norm::isDeviceOperand(*tensor))
        {
            return std::nullopt;
        }
    }
    for(const auto* tensor : {weight, y, bias, initialState, finalState})
    {
        if(tensor != nullptr && tensor->data_type() != problem.dataType)
        {
            return std::nullopt;
        }
    }
    const auto cuWords = indexWords(cuSeqlens->data_type());
    const auto indexWordCount
        = indices != nullptr ? indexWords(indices->data_type()) : std::optional<int32_t>(1);
    if(!cuWords.has_value() || !indexWordCount.has_value() || !declaresWordAlignment(*cuSeqlens)
       || (indices != nullptr && !declaresWordAlignment(*indices)))
    {
        return std::nullopt;
    }
    problem.cuSeqlensStep = *cuWords;
    problem.stateIndicesStep = *indexWordCount;

    // Shapes: x and y [T, C], weight [C, W], cu_seqlens [N + 1], bias [C], state_indices
    // [N], each state [N or P, C, W - 1].
    const auto xDims = dimsOf(*x, 2);
    const auto yDims = dimsOf(*y, 2);
    const auto weightDims = dimsOf(*weight, 2);
    const auto cuDims = dimsOf(*cuSeqlens, 1);
    if(!xDims || !yDims || !weightDims || !cuDims || *yDims != *xDims)
    {
        return std::nullopt;
    }
    problem.tokens = (*xDims)[0];
    problem.channels = (*xDims)[1];
    problem.width = (*weightDims)[1];
    problem.sequences = (*cuDims)[0] - 1;
    if((*weightDims)[0] != problem.channels || problem.width < 2 || problem.sequences < 1
       || problem.tokens > MAX_TOKENS || problem.channels > MAX_INT32
       || problem.sequences > MAX_SEQUENCES)
    {
        return std::nullopt;
    }
    if(bias != nullptr)
    {
        const auto biasDims = dimsOf(*bias, 1);
        if(!biasDims || (*biasDims)[0] != problem.channels)
        {
            return std::nullopt;
        }
    }
    if(indices != nullptr)
    {
        const auto indexDims = dimsOf(*indices, 1);
        if(!indexDims || (*indexDims)[0] != problem.sequences)
        {
            return std::nullopt;
        }
    }
    problem.stateRows = problem.sequences;
    if(initialState != nullptr)
    {
        const auto dims = dimsOf(*initialState, 3);
        if(!dims || (*dims)[1] != problem.channels || (*dims)[2] != problem.width - 1
           || (indices == nullptr && (*dims)[0] != problem.sequences) || (*dims)[0] > MAX_INT32)
        {
            return std::nullopt;
        }
        problem.stateRows = (*dims)[0];
    }
    if(finalState != nullptr)
    {
        const auto dims = dimsOf(*finalState, 3);
        if(!dims || (*dims)[0] != problem.stateRows || (*dims)[1] != problem.channels
           || (*dims)[2] != problem.width - 1)
        {
            return std::nullopt;
        }
    }

    // Strides. x and y are read and written `lanes` channels at a time from a unit channel
    // stride; cu_seqlens and state_indices are read as contiguous words. The outputs must
    // not overlap themselves.
    const auto xStrides = normalisedStrides(*x);
    const auto yStrides = normalisedStrides(*y);
    if(xStrides[1] != 1 || yStrides[1] != 1 || !isNonOverlapping(*yDims, yStrides)
       || normalisedStrides(*cuSeqlens)[0] != 1
       || (indices != nullptr && normalisedStrides(*indices)[0] != 1))
    {
        return std::nullopt;
    }
    problem.xStride = xStrides[0];
    problem.yStride = yStrides[0];
    const auto weightStrides = normalisedStrides(*weight);
    problem.weightStrideC = weightStrides[0];
    problem.weightStrideK = weightStrides[1];
    if(bias != nullptr)
    {
        problem.biasStride = normalisedStrides(*bias)[0];
        problem.flags |= FLAG_BIAS;
    }
    if(initialState != nullptr)
    {
        const auto strides = normalisedStrides(*initialState);
        problem.initialStrides = {strides[0], strides[1], strides[2]};
        problem.flags |= FLAG_INITIAL_STATE;
    }
    if(indices != nullptr)
    {
        problem.flags |= FLAG_STATE_INDICES;
    }
    if(finalState != nullptr)
    {
        const auto strides = normalisedStrides(*finalState);
        if(!isNonOverlapping({problem.stateRows, problem.channels, problem.width - 1}, strides))
        {
            return std::nullopt;
        }
        problem.finalStrides = {strides[0], strides[1], strides[2]};
        problem.flags |= FLAG_FINAL_STATE;
    }

    // The kernel accumulates in f32: at least the precision any accepted compute type asks.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && computeType != data_objects::DataType::HALF
       && computeType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    return problem;
}

/// The launch geometry of an object with @p blockThreads threads of @p lanes channels and
/// @p tileTokens-token tiles: grid x is the N sequence blocks then the token tiles, grid y
/// the channel blocks. nullopt past the grid's limits.
inline std::optional<std::pair<int64_t, int64_t>> launchGrid(const CausalConv1dProblem& problem,
                                                             int64_t blockThreads,
                                                             int64_t lanes,
                                                             int64_t tileTokens)
{
    if(blockThreads <= 0 || lanes <= 0 || tileTokens <= 0)
    {
        return std::nullopt;
    }
    const int64_t gridX = problem.sequences + (problem.tokens + tileTokens - 1) / tileTokens;
    const int64_t channelsPerBlock = blockThreads * lanes;
    const int64_t gridY = (problem.channels + channelsPerBlock - 1) / channelsPerBlock;
    if(gridX > MAX_INT32 || gridY > MAX_GRID_Y)
    {
        return std::nullopt;
    }
    return std::make_pair(gridX, gridY);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_causal_conv1d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
