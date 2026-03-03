// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "RMSNormTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_sdk_test_utils
{

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildRMSNormFwdGraph(hipdnn_data_sdk::data_objects::DataType inputDataType,
                         hipdnn_data_sdk::data_objects::DataType scaleDataType,
                         hipdnn_data_sdk::data_objects::DataType computeDataType,
                         const std::vector<int64_t>& dims,
                         const hipdnn_data_sdk::utilities::TensorLayout& layout)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("RMSNormFwdTest");

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::fromSdkType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale", hipdnn_frontend::fromSdkType(scaleDataType), derivedDims, derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim({1})
        .set_stride({1})
        .set_value(1e-5);

    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_fwd");
    rmsnormAttrs.set_epsilon(epsilonTensor);
    rmsnormAttrs.set_compute_data_type(hipdnn_frontend::fromSdkType(computeDataType));
    rmsnormAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

    auto outputTensorsAttr = graph->rmsnorm(xTensorAttr, scaleTensorAttr, rmsnormAttrs);

    auto& yTensorAttr = outputTensorsAttr[0];
    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(false);

    // invRms has one value per (batch, spatial) position: shape [N, 1, H, W, ...]
    auto invRmsDims = dims;
    invRmsDims[1] = 1;
    auto invRmsStrides = hipdnn_data_sdk::utilities::generateStrides(invRmsDims);

    auto& invRmsTensorAttr = outputTensorsAttr[1];
    if(!invRmsTensorAttr->has_uid())
    {
        invRmsTensorAttr->set_uid(uid++);
    }
    invRmsTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(computeDataType));
    invRmsTensorAttr->set_dim(invRmsDims);
    invRmsTensorAttr->set_stride(invRmsStrides);
    invRmsTensorAttr->set_is_virtual(false);

    return graph;
}

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildRMSNormFwdGraphWithBias(hipdnn_data_sdk::data_objects::DataType inputDataType,
                                 hipdnn_data_sdk::data_objects::DataType scaleDataType,
                                 hipdnn_data_sdk::data_objects::DataType computeDataType,
                                 const std::vector<int64_t>& dims,
                                 const hipdnn_data_sdk::utilities::TensorLayout& layout)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("RMSNormFwdWithBiasTest");

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::fromSdkType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale", hipdnn_frontend::fromSdkType(scaleDataType), derivedDims, derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias", hipdnn_frontend::fromSdkType(scaleDataType), derivedDims, derivedStrides);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim({1})
        .set_stride({1})
        .set_value(1e-5);

    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_fwd_bias");
    rmsnormAttrs.set_epsilon(epsilonTensor);
    rmsnormAttrs.set_bias(biasTensorAttr);
    rmsnormAttrs.set_compute_data_type(hipdnn_frontend::fromSdkType(computeDataType));
    rmsnormAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

    auto outputTensorsAttr = graph->rmsnorm(xTensorAttr, scaleTensorAttr, rmsnormAttrs);

    auto& yTensorAttr = outputTensorsAttr[0];
    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(false);

    // invRms has one value per (batch, spatial) position: shape [N, 1, H, W, ...]
    auto invRmsDims = dims;
    invRmsDims[1] = 1;
    auto invRmsStrides = hipdnn_data_sdk::utilities::generateStrides(invRmsDims);

    auto& invRmsTensorAttr = outputTensorsAttr[1];
    if(!invRmsTensorAttr->has_uid())
    {
        invRmsTensorAttr->set_uid(uid++);
    }
    invRmsTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(computeDataType));
    invRmsTensorAttr->set_dim(invRmsDims);
    invRmsTensorAttr->set_stride(invRmsStrides);
    invRmsTensorAttr->set_is_virtual(false);

    return graph;
}

} // namespace hipdnn_sdk_test_utils
