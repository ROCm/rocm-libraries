// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file SdpaAdditiveBias.hpp
 * @brief The SDPA attn_mask the rocKE attention kernels read as an additive fp32 bias: the
 *        graph match of the tensor and the i32 bounds its launch must respect. Shared by the
 *        dense and unified attention packs.
 */

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::sdpa_graph
{

/// The largest bias stride and bias element index a launch may use: the kernels take each
/// stride as an int32_t and address the bias with 32-bit element offsets. rocKE states the
/// same bound as ADDITIVE_BIAS_INDEX_LIMIT (an exclusive 2^31) in
/// kernels/common/_attention_additive_bias.py; test_registry_regeneration.py ties the two.
constexpr int64_t MAX_BIAS_INDEX = std::numeric_limits<int32_t>::max();

/// An additive bias the graph carries (its attn_mask): its uid and the element strides of
/// its batch, query-head and query-row axes. The key axis is unit stride, and an axis of
/// extent 1 broadcasts through stride 0.
struct AdditiveBias
{
    int64_t uid = 0;
    std::array<int32_t, 3> strides{0, 0, 0};
};

/// The largest bias element index a launch reaches: (seqlen_kv - 1) plus, for each leading
/// axis, (extent - 1) * stride, with the problem's batch, query-head and query-row extents.
/// nullopt when a product or the sum overflows int64_t.
///
/// Not checkedProduct: that one refuses a non-positive factor, and a broadcast axis has a
/// zero stride (and an extent of 1 makes the other factor zero), both legitimate here.
inline std::optional<int64_t> largestAdditiveBiasIndex(const SdpaShape& problem,
                                                       const std::array<int32_t, 3>& strides)
{
    const std::array<int64_t, 3> extents{problem.batch, problem.numQueryHeads, problem.seqLenQ};
    int64_t largest = problem.seqLenKv - 1;
    for(size_t axis = 0; axis < extents.size(); ++axis)
    {
        int64_t reach = 0;
        if(__builtin_mul_overflow(extents[axis] - 1, int64_t{strides[axis]}, &reach)
           || __builtin_add_overflow(largest, reach, &largest))
        {
            return std::nullopt;
        }
    }
    return largest;
}

/**
 * @brief The attn_mask tensor @p uid as the kernels read it, or nullopt (with the cause
 *        logged under @p engine) for one they cannot.
 *
 * FP32, rank 4 [B|1, Hq|1, Sq|1, Skv] with a unit-stride key axis, each leading axis of
 * extent 1 (broadcast, stride 0) or the problem's extent with a stride in [0, 2^31), not
 * virtual, ragged or pass-by-value, and every element a launch reads below 2^31.
 */
inline std::optional<AdditiveBias> additiveBiasFor(const MatchContext& context,
                                                   int64_t uid,
                                                   const SdpaShape& problem,
                                                   std::string_view engine)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr || !isWellFormedOperand(*tensor)
       || tensor->ragged_offset_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "attn_mask"})
                               << "the attn_mask tensor (uid " << uid
                               << ") must be a rank-4 tensor in the graph with positive "
                                  "extents, not virtual, not ragged and not pass-by-value");
        return std::nullopt;
    }
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
    if(dims->Get(HEAD_SIZE_AXIS) != problem.seqLenKv
       || (problem.seqLenKv > 1 && strides->Get(HEAD_SIZE_AXIS) != 1))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "attn_mask"})
                               << describeOperand("attn_mask", *tensor)
                               << " must be rank-4 [B|1, Hq|1, Sq|1, Skv] with Skv = "
                               << problem.seqLenKv << " at unit stride");
        return std::nullopt;
    }
    if(tensor->data_type() != data_objects::DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "attn_mask"})
                               << "attn_mask data type " << tensor->data_type()
                               << " is not supported; the kernel reads an FP32 bias");
        return std::nullopt;
    }

    const std::array<int64_t, 3> extents{problem.batch, problem.numQueryHeads, problem.seqLenQ};
    AdditiveBias bias;
    bias.uid = uid;
    for(uint32_t axis = 0; axis < extents.size(); ++axis)
    {
        const int64_t extent = dims->Get(axis);
        const int64_t stride = extent == 1 ? 0 : strides->Get(axis);
        if((extent != 1 && extent != extents[axis]) || stride < 0 || stride > MAX_BIAS_INDEX)
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "attn_mask"})
                                   << describeOperand("attn_mask", *tensor) << " axis " << axis
                                   << " must have extent 1 or " << extents[axis]
                                   << " and a stride in [0, 2^31)");
            return std::nullopt;
        }
        bias.strides[axis] = static_cast<int32_t>(stride);
    }
    const auto maxIndex = largestAdditiveBiasIndex(problem, bias.strides);
    if(!maxIndex.has_value() || *maxIndex > MAX_BIAS_INDEX)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "size_limit"})
                               << "attn_mask reaches element 2^31 or beyond; the kernel "
                                  "addresses the bias with 32-bit element offsets");
        return std::nullopt;
    }
    return bias;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::sdpa_graph
