// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

class IGenericBatchnormExecutor
{
public:
    virtual ~IGenericBatchnormExecutor() = default;

    //virtual bool isApplicable(const hipdnn_sdk::data_objects::Node& node) const = 0;

    virtual void batchnormFwdInference(std::any& input,
                                       std::any& scale,
                                       std::any& bias,
                                       std::any& mean,
                                       std::any& variance,
                                       std::any& output,
                                       double epsilon)
        = 0;
};

template <BatchnormSignatureRegistryKey Key>
class BatchnormExecutor : public IGenericBatchnormExecutor
{
public:
    using InputDataType = DataTypeToNative<Key.inputDataType>;
    using ScaleBiasDataType = DataTypeToNative<Key.scaleBiasDataType>;
    using MeanVarianceDataType = DataTypeToNative<Key.meanVarianceDataType>;

    // bool isApplicable(const hipdnn_sdk::data_objects::Node& node) const override
    // {
    //     // Implementation to check if this executor can handle the given node
    //     return true; // Placeholder
    // }

    void batchnormFwdInference(std::any& input,
                               std::any& scale,
                               std::any& bias,
                               std::any& mean,
                               std::any& variance,
                               std::any& output,
                               double epsilon) override
    {
        auto& inVar = TensorVariantUtils::unwrapToTensorVariant(input);
        auto& scVar = TensorVariantUtils::unwrapToTensorVariant(scale);
        auto& biVar = TensorVariantUtils::unwrapToTensorVariant(bias);
        auto& meVar = TensorVariantUtils::unwrapToTensorVariant(mean);
        auto& vaVar = TensorVariantUtils::unwrapToTensorVariant(variance);
        auto& outVar = TensorVariantUtils::unwrapToTensorVariant(output);

        auto& inTensor = *std::get<std::unique_ptr<TensorBase<InputDataType>>>(inVar);
        auto& scTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasDataType>>>(scVar);
        auto& biTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasDataType>>>(biVar);
        auto& meTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceDataType>>>(meVar);
        auto& vaTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceDataType>>>(vaVar);
        auto& outTensor = *std::get<std::unique_ptr<TensorBase<InputDataType>>>(outVar);

        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormFwdInference(
                inTensor, scTensor, biTensor, meTensor, vaTensor, outTensor, epsilon);
    }
};

}
}
