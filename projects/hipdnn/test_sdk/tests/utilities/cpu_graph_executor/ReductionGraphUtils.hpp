// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ReductionTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildReductionGraph(ReductionTensorBundle<InputType>& tensorBundle,
                        hipdnn_data_sdk::data_objects::DataType inputDataType,
                        hipdnn_data_sdk::data_objects::DataType computeDataType,
                        hipdnn_frontend::ReductionMode mode = hipdnn_frontend::ReductionMode::ADD)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("ReductionTest");
    graph->set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto yAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "Y",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.yTensor);
    yAttr.set_uid(uid++);
    auto yTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(yAttr));
    yTensorAttr->set_output(true);

    hipdnn_frontend::graph::ReductionAttributes redAttrs;
    redAttrs.set_name("Reduction_op");
    redAttrs.set_mode(mode);
    redAttrs.set_compute_data_type(graph->get_compute_data_type());

    graph->reduction(xTensorAttr, yTensorAttr, redAttrs);

    auto variantPack = tensorBundle.createVariantPack(*xTensorAttr, *yTensorAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
