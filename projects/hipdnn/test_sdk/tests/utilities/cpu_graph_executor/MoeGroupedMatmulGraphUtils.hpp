// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MoeGroupedMatmulTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

/// The frontend and the FlatBuffers MoE mode enums intentionally diverge
/// (NOT_SET=0/NONE=1/GATHER=2/SCATTER=3 vs NONE=0/GATHER=1/SCATTER=2 -- see the
/// keep-in-sync note on MoeGroupedMatmulNode.hpp), so graph construction needs
/// its own mapping between them.
inline hipdnn_frontend::MoeGroupedMatmulMode
    toFrontendMoeMode(hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode mode)
{
    using SdkMode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;
    using FrontendMode = hipdnn_frontend::MoeGroupedMatmulMode;
    switch(mode)
    {
    case SdkMode::NONE:
        return FrontendMode::NONE;
    case SdkMode::GATHER:
        return FrontendMode::GATHER;
    case SdkMode::SCATTER:
        return FrontendMode::SCATTER;
    default:
        return FrontendMode::NOT_SET;
    }
}

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildMoeGroupedMatmulGraph(MoeGroupedMatmulTensorBundle<InputType>& bundle,
                               hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                               hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType,
                               hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
{
    using Mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("MoeGroupedMatmulTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto tokenAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "TOKEN",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
            bundle.tokenTensor));
    tokenAttr->set_uid(1);

    auto weightAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "WEIGHT",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
            bundle.weightTensor));
    weightAttr->set_uid(2);

    auto firstTokenOffsetAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "FIRST_TOKEN_OFFSET", hipdnn_frontend::DataType::INT32, bundle.firstTokenOffsetTensor));
    firstTokenOffsetAttr->set_uid(3);

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> tokenIndexAttr;
    if(bundle.mode != Mode::NONE)
    {
        tokenIndexAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
            hipdnn_frontend::graph::makeTensorAttributes(
                "TOKEN_INDEX", hipdnn_frontend::DataType::INT32, *bundle.tokenIndexTensor));
        tokenIndexAttr->set_uid(4);
    }

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> tokenKsAttr;
    if(bundle.mode == Mode::SCATTER)
    {
        tokenKsAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
            hipdnn_frontend::graph::makeTensorAttributes(
                "TOKEN_KS", hipdnn_frontend::DataType::INT32, *bundle.tokenKsTensor));
        tokenKsAttr->set_uid(5);
    }

    hipdnn_frontend::graph::MoeGroupedMatmulAttributes moeAttrs;
    moeAttrs.set_name("MoeGroupedMatmul_test");
    moeAttrs.set_compute_data_type(graph->get_compute_data_type());
    moeAttrs.set_mode(toFrontendMoeMode(bundle.mode));
    moeAttrs.set_top_k(bundle.topK);

    auto outputAttr = graph->moe_grouped_matmul(
        tokenAttr, weightAttr, firstTokenOffsetAttr, tokenIndexAttr, tokenKsAttr, moeAttrs);
    outputAttr->set_uid(6);
    outputAttr->set_output(true);
    outputAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(outputDataType));

    auto variantPack = bundle.createVariantPack(*tokenAttr,
                                                *weightAttr,
                                                *firstTokenOffsetAttr,
                                                tokenIndexAttr.get(),
                                                tokenKsAttr.get(),
                                                *outputAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
