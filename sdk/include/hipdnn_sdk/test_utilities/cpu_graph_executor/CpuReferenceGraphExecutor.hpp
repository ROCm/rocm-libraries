// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/TensorVariant.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormRegistry.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

class CpuReferenceGraphExecutor
{
public:
    CpuReferenceGraphExecutor() = default;
    ~CpuReferenceGraphExecutor() = default;

    static void
        execute(void* graphBuffer, size_t size, std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap = hipdnn_plugin::GraphWrapper(graphBuffer, size);

        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();
            if(nodeAttributes != nullptr)
            {
                const auto& tensorMap = graphWrap.getTensorMap();
                auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
                auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
                //todo, its optional so use scale if not provided
                auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid());
                BatchnormSignatureRegistryKey key{.inputDataType = xTensorAttr->data_type(),
                                                  .scaleBiasDataType = scaleTensorAttr->data_type(),
                                                  .meanVarianceDataType
                                                  = meanTensorAttr->data_type()};

                auto it = batchnormRegistry().find(key);

                if(it != batchnormRegistry().end())
                {
                    auto shallowXTensor = TensorVariantUtils::createHostOnlyShallowTensorVariant(
                        *xTensorAttr, variantPack.at(xTensorAttr->uid()));
                    std::any input = std::ref(shallowXTensor);

                    auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
                    auto shallowYTensor = TensorVariantUtils::createHostOnlyShallowTensorVariant(
                        *yTensorAttr, variantPack.at(yTensorAttr->uid()));
                    std::any output = std::ref(shallowYTensor);

                    auto shallowScaleTensor
                        = TensorVariantUtils::createHostOnlyShallowTensorVariant(
                            *scaleTensorAttr, variantPack.at(scaleTensorAttr->uid()));
                    std::any scale = std::ref(shallowScaleTensor);

                    auto biasTensorAttr = tensorMap.at(nodeAttributes->bias_tensor_uid());
                    auto shallowBiasTensor = TensorVariantUtils::createHostOnlyShallowTensorVariant(
                        *biasTensorAttr, variantPack.at(biasTensorAttr->uid()));
                    std::any bias = std::ref(shallowBiasTensor);

                    auto shallowMeanTensor = TensorVariantUtils::createHostOnlyShallowTensorVariant(
                        *meanTensorAttr, variantPack.at(meanTensorAttr->uid()));
                    std::any mean = std::ref(shallowMeanTensor);

                    auto invVarianceTensorAttr
                        = tensorMap.at(nodeAttributes->inv_variance_tensor_uid());
                    auto shallowInvVarianceTensor
                        = TensorVariantUtils::createHostOnlyShallowTensorVariant(
                            *invVarianceTensorAttr, variantPack.at(invVarianceTensorAttr->uid()));
                    std::any variance = std::ref(shallowInvVarianceTensor);

                    it->second->batchnormFwdInference(
                        input, scale, bias, mean, variance, output, 1e-3);
                }
                else
                {
                    throw std::runtime_error("No registered function for the given signature");
                }
            }
            else
            {
                throw std::runtime_error("Unsupported node attributes type");
            }
        }
    }
};
}
}
