// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/PlanUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

struct MatmulParams
{
    MatmulParams() = default;
    MatmulParams(const hipdnn_data_sdk::data_objects::TensorAttributes& aAttributes,
                 const hipdnn_data_sdk::data_objects::TensorAttributes& bAttributes,
                 const hipdnn_data_sdk::data_objects::TensorAttributes& cAttributes)
        : aTensor(unpackTensorAttributes(aAttributes))
        , bTensor(unpackTensorAttributes(bAttributes))
        , cTensor(unpackTensorAttributes(cAttributes))
    {
    }

    hipdnn_data_sdk::data_objects::TensorAttributesT aTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT bTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT cTensor;
};

template <typename ADataType, typename BDataType, typename OutputDataType, typename ComputeDataType>
class MatmulPlan : public IGraphNodePlanExecutor
{
public:
    explicit MatmulPlan(MatmulParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowATensor
            = createShallowTensor<ADataType>(_params.aTensor, variantPack.at(_params.aTensor.uid));
        auto shallowBTensor
            = createShallowTensor<BDataType>(_params.bTensor, variantPack.at(_params.bTensor.uid));
        auto shallowCTensor = createShallowTensor<OutputDataType>(
            _params.cTensor, variantPack.at(_params.cTensor.uid));

        CpuFpReferenceMatmul::matmul<ADataType, BDataType, OutputDataType, ComputeDataType>(
            *shallowATensor, *shallowBTensor, *shallowCTensor);
    }

private:
    MatmulParams _params;
};

template <hipdnn_data_sdk::data_objects::DataType ADataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType BDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType ComputeDataTypeEnum>
class MatmulPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using ADataType = DataTypeToNative<ADataTypeEnum>;
    using BDataType = DataTypeToNative<BDataTypeEnum>;
    using OutputDataType = DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_data_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
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
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->c_tensor_uid(), OutputDataTypeEnum);

        // Shape validation
        const auto* aAttr = tensorMap.at(nodeAttributes->a_tensor_uid());
        const auto* bAttr = tensorMap.at(nodeAttributes->b_tensor_uid());
        const auto* cAttr = tensorMap.at(nodeAttributes->c_tensor_uid());
        if(aAttr == nullptr || bAttr == nullptr || cAttr == nullptr)
        {
            return false;
        }

        if(aAttr->dims() == nullptr || bAttr->dims() == nullptr || cAttr->dims() == nullptr)
        {
            return false;
        }

        const auto aRank = aAttr->dims()->size();
        const auto bRank = bAttr->dims()->size();
        const auto cRank = cAttr->dims()->size();
        if(aRank != bRank || aRank != cRank || aRank < 2)
        {
            return false;
        }

        // For each batch dim: A[i] and B[i] are compatible if one divides the other
        // Output batch dim is max(A[i], B[i])
        const auto batchDims = aRank - 2;
        if(!hipdnn_test_sdk::utilities::matmul::validateBatchBroadcastDims(
               batchDims, *aAttr->dims(), *bAttr->dims(), *cAttr->dims()))
        {
            return false;
        }

        // Matrix dims: A[..., M, K] x B[..., K, N] -> C[..., M, N]
        const int64_t aM = aAttr->dims()->Get(batchDims);
        const int64_t aK = aAttr->dims()->Get(batchDims + 1);
        const int64_t bK = bAttr->dims()->Get(batchDims);
        const int64_t bN = bAttr->dims()->Get(batchDims + 1);
        const int64_t cM = cAttr->dims()->Get(batchDims);
        const int64_t cN = cAttr->dims()->Get(batchDims + 1);

        return (aK == bK) && (cM == aM) && (cN == bN);
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin_sdk::IGraph& graph,
                      const hipdnn_data_sdk::data_objects::Node& node) const override
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

        return std::make_unique<
            MatmulPlan<ADataType, BDataType, OutputDataType, ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_test_sdk::utilities
