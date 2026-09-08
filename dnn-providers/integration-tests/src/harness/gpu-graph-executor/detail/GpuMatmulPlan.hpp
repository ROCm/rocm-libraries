// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn-gpu-ref/GpuFpReferenceMatmul.hpp"
#include "hipdnn-gpu-ref/ShallowGpuTensor.hpp"
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuMatmulParams
{
    GpuMatmulParams() = default;
    GpuMatmulParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& aAttributes,
                    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& bAttributes,
                    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& cAttributes)
        : aTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(aAttributes))
        , bTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(bAttributes))
        , cTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(cAttributes))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT aTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT bTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT cTensor;
};

template <typename ADataType, typename BDataType, typename CDataType, typename ComputeDataType>
class GpuMatmulPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuMatmulPlan(GpuMatmulParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<ADataType> aTensor(
            variantPack.at(_params.aTensor.uid), _params.aTensor.dims, _params.aTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<BDataType> bTensor(
            variantPack.at(_params.bTensor.uid), _params.bTensor.dims, _params.bTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<CDataType> cTensor(
            variantPack.at(_params.cTensor.uid), _params.cTensor.dims, _params.cTensor.strides);

        hipdnn_gpu_ref::GpuFpReferenceMatmul::
            matmul<ADataType, BDataType, CDataType, ComputeDataType>(aTensor, bTensor, cTensor);
    }

private:
    GpuMatmulParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType ADataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType BDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType CDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuMatmulPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using ADataType = hipdnn_test_sdk::utilities::DataTypeToNative<ADataTypeEnum>;
    using BDataType = hipdnn_test_sdk::utilities::DataTypeToNative<BDataTypeEnum>;
    using CDataType = hipdnn_test_sdk::utilities::DataTypeToNative<CDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

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

        const auto* nodeAttributes = node.attributes_as_MatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->a_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->b_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->c_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->a_tensor_uid(), ADataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->b_tensor_uid(), BDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->c_tensor_uid(), CDataTypeEnum);

        const std::vector<int64_t> operandUids = {nodeAttributes->a_tensor_uid(),
                                                  nodeAttributes->b_tensor_uid(),
                                                  nodeAttributes->c_tensor_uid()};
        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_MatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type MatmulAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuMatmulParams params(*tensorMap.at(nodeAttributes->a_tensor_uid()),
                               *tensorMap.at(nodeAttributes->b_tensor_uid()),
                               *tensorMap.at(nodeAttributes->c_tensor_uid()));

        return std::make_unique<GpuMatmulPlan<ADataType, BDataType, CDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
