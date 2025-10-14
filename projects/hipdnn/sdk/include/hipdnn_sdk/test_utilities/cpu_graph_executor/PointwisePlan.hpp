// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>

namespace hipdnn_sdk::test_utilities
{

struct PointwiseParams
{
    PointwiseParams() = default;
    PointwiseParams(const hipdnn_sdk::data_objects::PointwiseMode pointwiseMode,
                    const hipdnn_sdk::data_objects::TensorAttributes& in0Attributes,
                    const hipdnn_sdk::data_objects::TensorAttributes* optionalIn1Attributes,
                    const hipdnn_sdk::data_objects::TensorAttributes& out0Attributes)
        : in0Tensor(unpackTensorAttributes(in0Attributes))
        , out0Tensor(unpackTensorAttributes(out0Attributes))
        , mode(pointwiseMode)
    {
        if(optionalIn1Attributes != nullptr)
        {
            in1Tensor = unpackTensorAttributes(*optionalIn1Attributes);
        }
    }

    hipdnn_sdk::data_objects::TensorAttributesT in0Tensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> in1Tensor;
    hipdnn_sdk::data_objects::TensorAttributesT out0Tensor;
    hipdnn_sdk::data_objects::PointwiseMode mode;
};

template <typename DataType>
class PointwisePlan : public IGraphNodePlanExecutor
{
public:
    PointwisePlan(PointwiseParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowIn0Tensor = createShallowTensor<DataType>(
            _params.in0Tensor, variantPack.at(_params.in0Tensor.uid));

        auto shallowOut0Tensor = createShallowTensor<DataType>(
            _params.out0Tensor, variantPack.at(_params.out0Tensor.uid));

        if(isUnaryPointwiseMode(_params.mode))
        {
            CpuReferencePointwiseImpl<DataType>::pointwiseCompute(
                _params.mode, *shallowOut0Tensor, *shallowIn0Tensor);
        }
        else if(isBinaryPointwiseMode(_params.mode))
        {
            if(!_params.in1Tensor.has_value())
            {
                throw std::runtime_error("Binary pointwise operation requires in1 tensor");
            }

            auto shallowIn1Tensor = createShallowTensor<DataType>(
                _params.in1Tensor.value(), variantPack.at(_params.in1Tensor.value().uid));

            CpuReferencePointwiseImpl<DataType>::pointwiseCompute(
                _params.mode, *shallowOut0Tensor, *shallowIn0Tensor, *shallowIn1Tensor);
        }
        else
        {
            throw std::runtime_error("Unsupported pointwise operation mode");
        }
    }

private:
    PointwiseParams _params;
};

template <hipdnn_sdk::data_objects::DataType DataTypeEnum>
class PointwisePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DataType = DataTypeToNative<DataTypeEnum>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        // Check that the operation is implemented
        auto mode = nodeAttributes->operation();
        bool isImplemented = false;
        if(isUnaryPointwiseMode(mode))
        {
            isImplemented = isImplementedUnaryPointwiseMode(mode);
        }
        else if(isBinaryPointwiseMode(mode))
        {
            isImplemented = isImplementedBinaryPointwiseMode(mode);
        }

        if(!isImplemented)
        {
            return false;
        }

        // Check required tensors exist
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_0_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_0_tensor_uid());

        // Check required tensor types
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_0_tensor_uid(), DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_0_tensor_uid(), DataTypeEnum);

        // Check optional tensors based on operation mode
        if(isBinaryPointwiseMode(mode))
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->in_1_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap, nodeAttributes->in_1_tensor_uid(), DataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PointwiseAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        // Get required tensors
        const auto* in0Tensor = tensorMap.at(nodeAttributes->in_0_tensor_uid());
        const auto* out0Tensor = tensorMap.at(nodeAttributes->out_0_tensor_uid());

        // Get optional tensors
        const auto* in1Tensor
            = (nodeAttributes->in_1_tensor_uid() && *nodeAttributes->in_1_tensor_uid() != 0)
                  ? tensorMap.at(*nodeAttributes->in_1_tensor_uid())
                  : nullptr;

        PointwiseParams params(nodeAttributes->operation(), *in0Tensor, in1Tensor, *out0Tensor);

        return std::make_unique<PointwisePlan<DataType>>(std::move(params));
    }
};

}
