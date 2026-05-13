// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "RMSNormTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildRMSNormFwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                         const std::vector<int64_t>& dims,
                         const hipdnn_data_sdk::utilities::TensorLayout& layout)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("RMSNormFwdTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    // Scale/bias shape matches input except batch is broadcast. Non-1 non-batch
    // dims form a trailing suffix matching input — required by validateScaleNormalizedShape.
    auto derivedDims = dims;
    derivedDims[0] = 1;
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType),
        derivedDims,
        derivedStrides);
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
    rmsnormAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));
    rmsnormAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

    auto outputTensorsAttr = graph->rmsnorm(xTensorAttr, scaleTensorAttr, rmsnormAttrs);

    auto& yTensorAttr = outputTensorsAttr[0];
    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(false);

    // invRms derived from scale (validateNormStatsShapeIfSet): where scale is
    // non-1, inv_rms is 1; where scale is 1, inv_rms matches input.
    // With scale matching input except batch, inv_rms is [N, 1, 1, 1, ...].
    auto invRmsDims = std::vector<int64_t>(dims.size(), 1);
    invRmsDims[0] = dims[0];
    auto invRmsStrides = hipdnn_data_sdk::utilities::generateStrides(invRmsDims);

    auto& invRmsTensorAttr = outputTensorsAttr[1];
    if(!invRmsTensorAttr->has_uid())
    {
        invRmsTensorAttr->set_uid(uid++);
    }
    invRmsTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));
    invRmsTensorAttr->set_dim(invRmsDims);
    invRmsTensorAttr->set_stride(invRmsStrides);
    invRmsTensorAttr->set_is_virtual(false);

    return graph;
}

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildRMSNormFwdGraphWithBias(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                                 const std::vector<int64_t>& dims,
                                 const hipdnn_data_sdk::utilities::TensorLayout& layout)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("RMSNormFwdWithBiasTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    // Scale/bias shape matches input except batch is broadcast. Non-1 non-batch
    // dims form a trailing suffix matching input — required by validateScaleNormalizedShape.
    auto derivedDims = dims;
    derivedDims[0] = 1;
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType),
        derivedDims,
        derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType),
        derivedDims,
        derivedStrides);
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
    rmsnormAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));
    rmsnormAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

    auto outputTensorsAttr = graph->rmsnorm(xTensorAttr, scaleTensorAttr, rmsnormAttrs);

    auto& yTensorAttr = outputTensorsAttr[0];
    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(false);

    // invRms derived from scale (validateNormStatsShapeIfSet): where scale is
    // non-1, inv_rms is 1; where scale is 1, inv_rms matches input.
    // With scale matching input except batch, inv_rms is [N, 1, 1, 1, ...].
    auto invRmsDims = std::vector<int64_t>(dims.size(), 1);
    invRmsDims[0] = dims[0];
    auto invRmsStrides = hipdnn_data_sdk::utilities::generateStrides(invRmsDims);

    auto& invRmsTensorAttr = outputTensorsAttr[1];
    if(!invRmsTensorAttr->has_uid())
    {
        invRmsTensorAttr->set_uid(uid++);
    }
    invRmsTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));
    invRmsTensorAttr->set_dim(invRmsDims);
    invRmsTensorAttr->set_stride(invRmsStrides);
    invRmsTensorAttr->set_is_virtual(false);

    return graph;
}

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildRMSNormBwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                         const std::vector<int64_t>& dims,
                         const hipdnn_data_sdk::utilities::TensorLayout& layout)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("RMSNormBwdTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    // Scale/bias shape matches input except batch is broadcast
    auto scaleDims = dims;
    scaleDims[0] = 1;
    auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);

    // invRms shape matches input except non-batch dims are 1
    auto invRmsDims = std::vector<int64_t>(dims.size(), 1);
    invRmsDims[0] = dims[0];
    auto invRmsStrides = hipdnn_data_sdk::utilities::generateStrides(invRmsDims);

    int64_t uid = 1;

    auto dyAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dy", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    dyAttr.set_uid(uid++);
    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(dyAttr));

    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType),
        scaleDims,
        scaleStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensor
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto invRmsAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "inv_rms",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType),
        invRmsDims,
        invRmsStrides);
    invRmsAttr.set_uid(uid++);
    auto invRmsTensor
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(invRmsAttr));

    hipdnn_frontend::graph::RMSNormBwdAttributes rmsnormBwdAttrs;
    rmsnormBwdAttrs.set_name("rmsnorm_bwd");
    rmsnormBwdAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto outputTensorsAttr
        = graph->rmsnorm_backward(dyTensor, xTensor, scaleTensor, invRmsTensor, rmsnormBwdAttrs);

    auto& dxTensorAttr = outputTensorsAttr[0];
    if(!dxTensorAttr->has_uid())
    {
        dxTensorAttr->set_uid(uid++);
    }
    dxTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    dxTensorAttr->set_dim(dims);
    dxTensorAttr->set_stride(strides);
    dxTensorAttr->set_is_virtual(false);

    auto& dscaleTensorAttr = outputTensorsAttr[1];
    if(!dscaleTensorAttr->has_uid())
    {
        dscaleTensorAttr->set_uid(uid++);
    }
    dscaleTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType));
    dscaleTensorAttr->set_dim(scaleDims);
    dscaleTensorAttr->set_stride(scaleStrides);
    dscaleTensorAttr->set_is_virtual(false);

    auto& dbiasTensorAttr = outputTensorsAttr[2];
    if(!dbiasTensorAttr->has_uid())
    {
        dbiasTensorAttr->set_uid(uid++);
    }
    dbiasTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleDataType));
    dbiasTensorAttr->set_dim(scaleDims);
    dbiasTensorAttr->set_stride(scaleStrides);
    dbiasTensorAttr->set_is_virtual(false);

    return graph;
}

} // namespace hipdnn_sdk_test_utils
