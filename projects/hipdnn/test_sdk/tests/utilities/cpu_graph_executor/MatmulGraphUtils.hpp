// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MatmulTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildMatmulGraph(MatmulTensorBundle<InputType>& tensorBundle,
                     hipdnn_data_sdk::data_objects::DataType inputDataType,
                     hipdnn_data_sdk::data_objects::DataType computeDataType)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("MatmulTest");
    graph->set_compute_data_type(hipdnn_frontend::fromSdkType(computeDataType));

    int64_t uid = 1;
    auto aAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "A", hipdnn_frontend::fromSdkType(inputDataType), tensorBundle.aTensor);
    aAttr.set_uid(uid++);
    auto aTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(aAttr));

    auto bAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "B", hipdnn_frontend::fromSdkType(inputDataType), tensorBundle.bTensor);
    bAttr.set_uid(uid++);
    auto bTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(bAttr));

    hipdnn_frontend::graph::MatmulAttributes matmulAttrs;
    matmulAttrs.set_name("Matmul_inference");
    matmulAttrs.set_compute_data_type(graph->get_compute_data_type());

    auto cTensorAttr = graph->matmul(aTensorAttr, bTensorAttr, matmulAttrs);

    if(!cTensorAttr->has_uid())
    {
        cTensorAttr->set_uid(uid++);
    }

    cTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(inputDataType));

    auto variantPack = tensorBundle.createVariantPack(*aTensorAttr, *bTensorAttr, *cTensorAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
