// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MoeGroupedMatmulBwdTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

// dweight's dims are left unset so the node infers them --
// [firstTokenOffset.dim[0], token.dim[2], doutput.dim[2]]. Its strides and data type are
// assigned here: the data type can differ from the graph's io type, and the layout is the
// caller's choice rather than a property of the operation.
template <typename InputType, typename DweightType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildMoeGroupedMatmulBwdGraph(MoeGroupedMatmulBwdTensorBundle<InputType, DweightType>& bundle,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("MoeGroupedMatmulBwdTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto doutputAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "DOUTPUT",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
            bundle.doutputTensor));
    doutputAttr->set_uid(1);

    auto tokenAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "TOKEN",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
            bundle.tokenTensor));
    tokenAttr->set_uid(2);

    auto firstTokenOffsetAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "FIRST_TOKEN_OFFSET", hipdnn_frontend::DataType::INT32, bundle.firstTokenOffsetTensor));
    firstTokenOffsetAttr->set_uid(3);

    hipdnn_frontend::graph::MoeGroupedMatmulBwdAttributes moeBwdAttrs;
    moeBwdAttrs.set_name("MoeGroupedMatmulBwd_test");
    moeBwdAttrs.set_compute_data_type(graph->get_compute_data_type());

    auto dweightAttr
        = graph->moe_grouped_matmul_bwd(doutputAttr, tokenAttr, firstTokenOffsetAttr, moeBwdAttrs);
    if(!dweightAttr->has_uid())
    {
        dweightAttr->set_uid(4);
    }
    // Derived from the bundle's dweight buffer type rather than inputDataType, so
    // the declared enum can never disagree with the memory the variant pack wires
    // in -- this is what makes the mixed-dtype registrations reachable.
    dweightAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(
        hipdnn_test_sdk::utilities::nativeTypeToDataType<DweightType>()));
    // Replicate the layout the bundle actually allocated instead of relying on the node's
    // column-major fallback -- that fallback only applies to an unset stride vector, and
    // the graph has to describe the memory the variant pack wires in.
    dweightAttr->set_stride(bundle.dweightTensor.strides());

    auto variantPack
        = bundle.createVariantPack(*doutputAttr, *tokenAttr, *firstTokenOffsetAttr, *dweightAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
