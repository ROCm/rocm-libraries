// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceCausalConv1d.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace hipdnn_test_sdk::detail
{

struct CausalConv1dParams
{
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorAttributesT = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

    CausalConv1dParams() = default;
    CausalConv1dParams(
        const hipdnn_flatbuffers_sdk::data_objects::CausalConv1dAttributes& attributes,
        const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
        : xTensor(unpackTensorAttributes(*tensorMap.at(attributes.x_tensor_uid())))
        , weightTensor(unpackTensorAttributes(*tensorMap.at(attributes.weight_tensor_uid())))
        , cuSeqlensTensor(unpackTensorAttributes(*tensorMap.at(attributes.cu_seqlens_tensor_uid())))
        , yTensor(unpackTensorAttributes(*tensorMap.at(attributes.y_tensor_uid())))
        , biasTensor(unpackOptional(attributes.bias_tensor_uid(), tensorMap))
        , initialStateTensor(unpackOptional(attributes.initial_state_tensor_uid(), tensorMap))
        , stateIndicesTensor(unpackOptional(attributes.state_indices_tensor_uid(), tensorMap))
        , finalStateTensor(unpackOptional(attributes.final_state_tensor_uid(), tensorMap))
        , activation(attributes.activation())
    {
    }

    TensorAttributesT xTensor;
    TensorAttributesT weightTensor;
    TensorAttributesT cuSeqlensTensor;
    TensorAttributesT yTensor;
    std::optional<TensorAttributesT> biasTensor;
    std::optional<TensorAttributesT> initialStateTensor;
    std::optional<TensorAttributesT> stateIndicesTensor;
    std::optional<TensorAttributesT> finalStateTensor;
    hipdnn_flatbuffers_sdk::data_objects::CausalConv1dActivation activation
        = hipdnn_flatbuffers_sdk::data_objects::CausalConv1dActivation::IDENTITY;

private:
    static std::optional<TensorAttributesT>
        unpackOptional(const ::flatbuffers::Optional<int64_t>& uid,
                       const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
    {
        if(!uid.has_value())
        {
            return std::nullopt;
        }
        return unpackTensorAttributes(*tensorMap.at(*uid));
    }
};

template <typename DataType,
          typename SeqlensDataType,
          typename StateIndexDataType,
          typename ComputeDataType>
class CausalConv1dPlan : public IGraphNodePlanExecutor
{
public:
    explicit CausalConv1dPlan(CausalConv1dParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        if(_params.finalStateTensor.has_value())
        {
            return {_params.yTensor.uid, _params.finalStateTensor->uid};
        }
        return {_params.yTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto x = bindShallowTensor<DataType>(_params.xTensor, variantPack);
        auto weight = bindShallowTensor<DataType>(_params.weightTensor, variantPack);
        auto cuSeqlens = bindShallowTensor<SeqlensDataType>(_params.cuSeqlensTensor, variantPack);
        auto y = bindShallowTensor<DataType>(_params.yTensor, variantPack);
        auto bias = bindOptionalShallowTensor<DataType>(_params.biasTensor, variantPack);
        auto initialState
            = bindOptionalShallowTensor<DataType>(_params.initialStateTensor, variantPack);
        auto stateIndices = bindOptionalShallowTensor<StateIndexDataType>(
            _params.stateIndicesTensor, variantPack);
        auto finalState
            = bindOptionalShallowTensor<DataType>(_params.finalStateTensor, variantPack);

        utilities::CpuFpReferenceCausalConv1d::
            forward<DataType, SeqlensDataType, StateIndexDataType, ComputeDataType>(
                *x,
                *weight,
                *cuSeqlens,
                *y,
                _params.activation,
                bias.get(),
                initialState.get(),
                stateIndices.get(),
                finalState.get());
    }

private:
    CausalConv1dParams _params;
};

/// Builds a CausalConv1dPlan. StateIndicesDataTypeEnum is UNSET for graphs without
/// state_indices, else INT32 or INT64.
template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType SeqlensDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType StateIndicesDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class CausalConv1dPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DataType = utilities::DataTypeToNative<DataTypeEnum>;
    using SeqlensDataType = utilities::DataTypeToNative<SeqlensDataTypeEnum>;
    using StateIndexDataType
        = std::conditional_t<StateIndicesDataTypeEnum
                                 == hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                             int64_t,
                             int32_t>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_CausalConv1dAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->weight_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->cu_seqlens_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->weight_tensor_uid(), DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->cu_seqlens_tensor_uid(), SeqlensDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), DataTypeEnum);

        for(const auto& uid : {nodeAttributes->bias_tensor_uid(),
                               nodeAttributes->initial_state_tensor_uid(),
                               nodeAttributes->final_state_tensor_uid()})
        {
            if(uid.has_value())
            {
                CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, uid);
                CHECK_OPTIONAL_TENSOR_TYPE(tensorMap, uid, DataTypeEnum);
            }
        }

        const auto stateIndicesUid = nodeAttributes->state_indices_tensor_uid();
        if constexpr(StateIndicesDataTypeEnum == DataTypeFb::UNSET)
        {
            if(stateIndicesUid.has_value())
            {
                return false;
            }
        }
        else
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, stateIndicesUid);
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap, stateIndicesUid, StateIndicesDataTypeEnum);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_CausalConv1dAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type CausalConv1dAttributes");
        }

        return std::make_unique<
            CausalConv1dPlan<DataType, SeqlensDataType, StateIndexDataType, ComputeDataType>>(
            CausalConv1dParams(*nodeAttributes, graph.getTensorMap()));
    }

private:
    using DataTypeFb = hipdnn_flatbuffers_sdk::data_objects::DataType;
};

} // namespace hipdnn_test_sdk::detail
