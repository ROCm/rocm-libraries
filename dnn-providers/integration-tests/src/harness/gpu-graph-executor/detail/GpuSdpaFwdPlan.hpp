// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// A graph stats/LSE tensor is conventionally rank-4 [B, H, Sq, 1], but the SDPA
// reference requires the LSE output to be rank-3 [B, H, Sq]. Drop a single trailing
// size-1 dim; a rank-3 input passes through unchanged. Any other shape is unsupported.
inline std::vector<int64_t> squeezeTrailingUnitDim(const std::vector<int64_t>& dims)
{
    if(dims.size() == 3)
    {
        return dims;
    }
    if(dims.size() == 4 && dims.back() == 1)
    {
        return {dims.begin(), dims.end() - 1};
    }
    throw std::invalid_argument(
        "GpuSdpaFwdPlan: stats/LSE tensor must be rank-3 [B, H, Sq] or rank-4 [B, H, Sq, 1]");
}

// Truncate a stride vector to match a squeezed dims vector by dropping trailing entries.
inline std::vector<int64_t> squeezeToRank(const std::vector<int64_t>& strides, size_t rank)
{
    return {strides.begin(), strides.begin() + static_cast<std::ptrdiff_t>(rank)};
}

template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
constexpr hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode sdpaProbabilityMode()
{
    using hipdnn_data_sdk::types::bfloat16;
    // The P->bf16 cast is a property of the P@V matmul inputs, not the output: the
    // AITER BF16 forward kernel materializes the softmax probabilities in bf16 before
    // P@V whenever Q/K/V are bf16, regardless of the (fp32 or bf16) output dtype.
    if constexpr(std::is_same_v<QDataType, bfloat16> && std::is_same_v<KDataType, bfloat16>
                 && std::is_same_v<VDataType, bfloat16>)
    {
        return hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE;
    }
    else
    {
        return hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::FLOAT;
    }
}

struct GpuSdpaFwdParams
{
    GpuSdpaFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
        std::optional<float> attnScaleValue,
        int64_t leftBound,
        int64_t rightBound,
        bool topLeftAlignment,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attnMaskAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* lseAttributes = nullptr)
        : qTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(qAttributes))
        , kTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(kAttributes))
        , vTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(vAttributes))
        , oTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(oAttributes))
        , attnScaleValue(attnScaleValue)
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , attnMaskTensor(attnMaskAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *attnMaskAttributes))
                             : std::nullopt)
        , lseTensor(lseAttributes != nullptr
                        ? std::make_optional(
                              hipdnn_test_sdk::detail::unpackTensorAttributes(*lseAttributes))
                        : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    std::optional<float> attnScaleValue;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> attnMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> lseTensor;
};

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float>
class GpuSdpaFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuSdpaFwdPlan(GpuSdpaFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<QDataType> qTensor(
            variantPack.at(_params.qTensor.uid), _params.qTensor.dims, _params.qTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<KDataType> kTensor(
            variantPack.at(_params.kTensor.uid), _params.kTensor.dims, _params.kTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<VDataType> vTensor(
            variantPack.at(_params.vTensor.uid), _params.vTensor.dims, _params.vTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ODataType> oTensor(
            variantPack.at(_params.oTensor.uid), _params.oTensor.dims, _params.oTensor.strides);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> attnMaskTensor;
        if(_params.attnMaskTensor.has_value())
        {
            attnMaskTensor.emplace(variantPack.at(_params.attnMaskTensor->uid),
                                   _params.attnMaskTensor->dims,
                                   _params.attnMaskTensor->strides);
        }

        // The graph stats tensor is rank-4 [B, H, Sq, 1], but fprop requires LSE to be
        // rank-3 [B, H, Sq]. Drop the trailing size-1 dim (and its stride) so the shapes
        // line up without relaxing the reference's strict rank check.
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> lseTensor;
        if(_params.lseTensor.has_value())
        {
            const auto squeezedDims = squeezeTrailingUnitDim(_params.lseTensor->dims);
            const auto squeezedStrides
                = squeezeToRank(_params.lseTensor->strides, squeezedDims.size());
            lseTensor.emplace(
                variantPack.at(_params.lseTensor->uid), squeezedDims, squeezedStrides);
        }

        hipdnn_gpu_ref::GpuFpReferenceSdpa::
            fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                qTensor,
                kTensor,
                vTensor,
                oTensor,
                _params.attnScaleValue,
                attnMaskTensor.has_value() ? &attnMaskTensor.value() : nullptr,
                _params.leftBound,
                _params.rightBound,
                _params.topLeftAlignment,
                lseTensor.has_value() ? &lseTensor.value() : nullptr,
                sdpaProbabilityMode<QDataType, KDataType, VDataType, ODataType>());
    }

private:
    GpuSdpaFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class GpuSdpaFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using QDataType = hipdnn_test_sdk::utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = hipdnn_test_sdk::utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = hipdnn_test_sdk::utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = hipdnn_test_sdk::utilities::DataTypeToNative<ODataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->q_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->k_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->v_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->o_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);

        // Unsupported mask modes
        if(nodeAttributes->alibi_mask() || nodeAttributes->padding_mask())
        {
            return false;
        }

        // Unsupported: variable sequence lengths
        if(nodeAttributes->seq_len_q_tensor_uid().has_value()
           || nodeAttributes->seq_len_kv_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: dropout
        if(nodeAttributes->dropout_probability().has_value()
           || nodeAttributes->seed_tensor_uid().has_value()
           || nodeAttributes->offset_tensor_uid().has_value()
           || nodeAttributes->dropout_mask_tensor_uid().has_value()
           || nodeAttributes->dropout_scale_tensor_uid().has_value()
           || nodeAttributes->rng_dump_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: paged KV cache
        if(nodeAttributes->page_table_k_tensor_uid().has_value()
           || nodeAttributes->page_table_v_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: block sparse attention
        if(nodeAttributes->block_mask_tensor_uid().has_value()
           || nodeAttributes->sink_token_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: FP8 quantization / descaling
        if(nodeAttributes->descale_q_tensor_uid().has_value()
           || nodeAttributes->descale_k_tensor_uid().has_value()
           || nodeAttributes->descale_v_tensor_uid().has_value()
           || nodeAttributes->descale_s_tensor_uid().has_value()
           || nodeAttributes->scale_s_tensor_uid().has_value()
           || nodeAttributes->scale_o_tensor_uid().has_value()
           || nodeAttributes->amax_s_tensor_uid().has_value()
           || nodeAttributes->amax_o_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: max / running-sum softmax stats outputs (the reference does not
        // produce these). The log-sum-exp stats tensor IS supported and handled below.
        if(nodeAttributes->max_tensor_uid().has_value()
           || nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            return false;
        }

        // Supported: log-sum-exp output via the stats tensor. It must exist in the map and
        // be FLOAT (LSE is always float). The rank reconciliation to [B, H, Sq] is enforced
        // in execute().
        if(nodeAttributes->stats_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->stats_tensor_uid().value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        }

        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }

        const auto* attnMaskPtr = nodeAttributes->attn_mask_tensor_uid().has_value()
                                      ? tensorMap.at(nodeAttributes->attn_mask_tensor_uid().value())
                                      : nullptr;

        const auto* lsePtr = nodeAttributes->stats_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->stats_tensor_uid().value())
                                 : nullptr;

        int64_t leftBound = (nodeAttributes->left_bound().has_value())
                                ? nodeAttributes->left_bound().value()
                                : -1;
        int64_t rightBound = (nodeAttributes->right_bound().has_value())
                                 ? nodeAttributes->right_bound().value()
                                 : -1;

        if(leftBound < -1 || rightBound < -1)
        {
            throw std::invalid_argument("GpuSdpaFwdPlan: left_bound and right_bound must be >= -1 "
                                        "(got left_bound="
                                        + std::to_string(leftBound)
                                        + ", right_bound=" + std::to_string(rightBound) + ")");
        }

        bool isTopLeft = nodeAttributes->diagonal_alignment()
                         == hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;

        // Validate mutually exclusive deprecated attributes
        if(nodeAttributes->causal_mask() && nodeAttributes->causal_mask_bottom_right())
        {
            throw std::invalid_argument("Cannot set both causal_mask and causal_mask_bottom_right. "
                                        "Use diagonal_alignment={TOP_LEFT|BOTTOM_RIGHT} with "
                                        "left_bound=-1, right_bound=0 instead.");
        }

        // Check deprecated attributes
        if(nodeAttributes->causal_mask())
        {
            leftBound = -1;
            rightBound = 0;
            isTopLeft = true;
        }
        if(nodeAttributes->causal_mask_bottom_right())
        {
            leftBound = -1;
            rightBound = 0;
            isTopLeft = false;
        }

        return std::make_unique<GpuSdpaFwdPlan<QDataType, KDataType, VDataType, ODataType, float>>(
            GpuSdpaFwdParams(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                             *tensorMap.at(nodeAttributes->k_tensor_uid()),
                             *tensorMap.at(nodeAttributes->v_tensor_uid()),
                             *tensorMap.at(nodeAttributes->o_tensor_uid()),
                             attnScaleValue,
                             leftBound,
                             rightBound,
                             isTopLeft,
                             attnMaskPtr,
                             lsePtr));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
