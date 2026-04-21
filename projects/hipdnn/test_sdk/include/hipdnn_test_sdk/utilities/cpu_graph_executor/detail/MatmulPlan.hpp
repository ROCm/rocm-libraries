// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct MatmulParams
{
    MatmulParams() = default;
    MatmulParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& aAttributes,
                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& bAttributes,
                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& cAttributes)
        : aTensor(unpackTensorAttributes(aAttributes))
        , bTensor(unpackTensorAttributes(bAttributes))
        , cTensor(unpackTensorAttributes(cAttributes))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT aTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT bTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT cTensor;
};

template <typename ADataType, typename BDataType, typename CDataType, typename ComputeDataType>
class MatmulPlan : public IGraphNodePlanExecutor
{
public:
    explicit MatmulPlan(MatmulParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.cTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowATensor
            = createShallowTensor<ADataType>(_params.aTensor, variantPack.at(_params.aTensor.uid));
        auto shallowBTensor
            = createShallowTensor<BDataType>(_params.bTensor, variantPack.at(_params.bTensor.uid));
        auto shallowCTensor
            = createShallowTensor<CDataType>(_params.cTensor, variantPack.at(_params.cTensor.uid));

        utilities::CpuFpReferenceMatmul::matmul<ADataType, BDataType, CDataType, ComputeDataType>(
            *shallowATensor, *shallowBTensor, *shallowCTensor);
    }

private:
    MatmulParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType ADataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType BDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType CDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class MatmulPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using ADataType = utilities::DataTypeToNative<ADataTypeEnum>;
    using BDataType = utilities::DataTypeToNative<BDataTypeEnum>;
    using CDataType = utilities::DataTypeToNative<CDataTypeEnum>;
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

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_MatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type MatmulAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        MatmulParams params(*tensorMap.at(nodeAttributes->a_tensor_uid()),
                            *tensorMap.at(nodeAttributes->b_tensor_uid()),
                            *tensorMap.at(nodeAttributes->c_tensor_uid()));

        return std::make_unique<MatmulPlan<ADataType, BDataType, CDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
