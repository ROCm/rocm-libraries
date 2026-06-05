// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename XType, typename ScaleType>
struct BlockScaleDequantizeTensorBundle
{
    BlockScaleDequantizeTensorBundle(const std::vector<int64_t>& xDims,
                                     const std::vector<int64_t>& scaleDims,
                                     unsigned int seed
                                     = hipdnn_test_sdk::utilities::getGlobalTestSeed())
        : xTensor(xDims)
        , scaleTensor(scaleDims)
    {
        xTensor.fillWithRandomValues(static_cast<XType>(0.0f), static_cast<XType>(1.0f), seed);
        scaleTensor.fillWithRandomValues(
            static_cast<ScaleType>(0.1f), static_cast<ScaleType>(2.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& scaleTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[scaleTensorAttr.get_uid()] = scaleTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<XType> xTensor;
    hipdnn_data_sdk::utilities::Tensor<ScaleType> scaleTensor;
};

template <typename XType, typename ScaleType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildBlockScaleDequantizeGraph(BlockScaleDequantizeTensorBundle<XType, ScaleType>& tensorBundle,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType xDataType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType yDataType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                                   const std::vector<int32_t>& blockSize)
{
    using namespace hipdnn_tests::constants;

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BlockScaleDequantizeTest");
    graph->set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(xDataType), tensorBundle.xTensor);
    xAttr.set_uid(K_BSD_TENSOR_X_UID);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType),
        tensorBundle.scaleTensor);
    scaleAttr.set_uid(K_BSD_TENSOR_SCALE_UID);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    hipdnn_frontend::graph::BlockScaleDequantizeAttributes bsdAttrs;
    bsdAttrs.set_name("block_scale_dequantize");
    bsdAttrs.set_block_size(blockSize);

    auto yTensorAttr = graph->block_scale_dequantize(xTensorAttr, scaleTensorAttr, bsdAttrs);

    yTensorAttr->set_uid(K_BSD_TENSOR_Y_UID);
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(yDataType));

    auto variantPack = tensorBundle.createVariantPack(*xTensorAttr, *scaleTensorAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
