// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdexcept>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#endif
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_test_sdk::utilities
{

namespace graph = hipdnn_frontend::graph;

/// Enum representing supported operation types for graph creation
enum class OperationType
{
    CONV_FORWARD,
    CONV_BACKWARD_DATA,
    CONV_BACKWARD_WEIGHTS,
    CONV_FWD_BIAS_ACTIV,
    BATCHNORM_TRAINING,
    BATCHNORM_INFERENCE,
    BATCHNORM_BACKWARD,
    MATMUL,
    LAYERNORM,
    RMSNORM,
    REDUCTION,
#ifdef HIPDNN_ENABLE_SDPA
    SDPA_FORWARD,
    SDPA_BACKWARD,
#endif
    POINTWISE_UNARY,
    POINTWISE_BINARY
};

/// Factory class for creating frontend Graph objects for testing
class FrontendGraphFactory
{
public:
    using DataType = hipdnn_frontend::DataType;
    using Graph = hipdnn_frontend::graph::Graph;

    /// Create a graph for the specified operation type
    static Graph create(OperationType op)
    {
        switch(op)
        {
        case OperationType::CONV_FORWARD:
            return createConvForwardGraph();
        case OperationType::CONV_BACKWARD_DATA:
            return createConvBackwardDataGraph();
        case OperationType::CONV_BACKWARD_WEIGHTS:
            return createConvBackwardWeightsGraph();
        case OperationType::CONV_FWD_BIAS_ACTIV:
            return createConvFwdBiasActivGraph();
        case OperationType::BATCHNORM_TRAINING:
            return createBatchnormTrainingGraph();
        case OperationType::BATCHNORM_INFERENCE:
            return createBatchnormInferenceGraph();
        case OperationType::BATCHNORM_BACKWARD:
            return createBatchnormBackwardGraph();
        case OperationType::MATMUL:
            return createMatmulGraph();
        case OperationType::LAYERNORM:
            return createLayernormGraph();
        case OperationType::RMSNORM:
            return createRmsnormGraph();
        case OperationType::REDUCTION:
            return createReductionGraph();
#ifdef HIPDNN_ENABLE_SDPA
        case OperationType::SDPA_FORWARD:
            return createSdpaForwardGraph();
        case OperationType::SDPA_BACKWARD:
            return createSdpaBackwardGraph();
#endif
        case OperationType::POINTWISE_UNARY:
            return createPointwiseUnaryGraph();
        case OperationType::POINTWISE_BINARY:
            return createPointwiseBinaryGraph();
        default:
            throw std::runtime_error("Unknown OperationType");
        }
    }

    /// Convolution Forward graph
    static Graph createConvForwardGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_ConvForward");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {1, 16, 16, 16};
        const std::vector<int64_t> wDims = {16, 16, 3, 3};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto wStrides = hipdnn_data_sdk::utilities::generateStrides(wDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto wAttr = graph::makeTensorAttributes("w", wDims, wStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding({1, 1}).set_post_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1});

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yAttr->set_output(true);

        return graphObj;
    }

    /// Convolution Backward Data graph
    static Graph createConvBackwardDataGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_ConvBackwardData");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> dyDims = {1, 16, 16, 16};
        const std::vector<int64_t> wDims = {16, 16, 3, 3};
        auto dyStrides = hipdnn_data_sdk::utilities::generateStrides(dyDims);
        auto wStrides = hipdnn_data_sdk::utilities::generateStrides(wDims);

        auto dyAttr = graph::makeTensorAttributes("dy", dyDims, dyStrides);
        auto wAttr = graph::makeTensorAttributes("w", wDims, wStrides);

        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvDgradAttributes convAttrs;
        convAttrs.set_pre_padding({1, 1}).set_post_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1});

        auto dxAttr = graphObj.conv_dgrad(dyTensorAttr, wTensorAttr, convAttrs);
        dxAttr->set_output(true);

        return graphObj;
    }

    /// Convolution Backward Weights graph
    static Graph createConvBackwardWeightsGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_ConvBackwardWeights");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {1, 16, 16, 16};
        const std::vector<int64_t> dyDims = {1, 16, 16, 16};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto dyStrides = hipdnn_data_sdk::utilities::generateStrides(dyDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto dyAttr = graph::makeTensorAttributes("dy", dyDims, dyStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        graph::ConvWgradAttributes convAttrs;
        convAttrs.set_pre_padding({1, 1}).set_post_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1});

        auto dwAttr = graphObj.conv_wgrad(xTensorAttr, dyTensorAttr, convAttrs);
        dwAttr->set_output(true);

        return graphObj;
    }

    /// Fused Convolution + Bias + Activation graph
    static Graph createConvFwdBiasActivGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_ConvFwdBiasActiv");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {1, 16, 16, 16};
        const std::vector<int64_t> wDims = {16, 16, 3, 3};
        const std::vector<int64_t> bDims = {1, 16, 1, 1};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto wStrides = hipdnn_data_sdk::utilities::generateStrides(wDims);
        auto bStrides = hipdnn_data_sdk::utilities::generateStrides(bDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto wAttr = graph::makeTensorAttributes("w", wDims, wStrides);
        auto bAttr = graph::makeTensorAttributes("bias", bDims, bStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding({1, 1}).set_post_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1});

        auto yConvAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        graph::PointwiseAttributes biasAttrs;
        biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
        auto yBiasAttr = graphObj.pointwise(yConvAttr, bTensorAttr, biasAttrs);

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);
        auto yActivAttr = graphObj.pointwise(yBiasAttr, activAttrs);

        yActivAttr->set_output(true);

        return graphObj;
    }

    /// Batchnorm Training graph
    static Graph createBatchnormTrainingGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_BatchnormTraining");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 16, 8, 8};
        const std::vector<int64_t> scaleDims = hipdnn_data_sdk::utilities::getDerivedShape(xDims);
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto scaleAttr = graph::makeTensorAttributes("scale", scaleDims, scaleStrides);
        auto biasAttr = graph::makeTensorAttributes("bias", scaleDims, scaleStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_value(1e-5).set_name("epsilon");

        graph::BatchnormAttributes bnAttrs;
        bnAttrs.set_epsilon(epsilonTensorAttr);

        auto [yAttr, meanAttr, invVarianceAttr, runningMeanAttr, runningVarAttr]
            = graphObj.batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        yAttr->set_output(true);
        if(meanAttr)
        {
            meanAttr->set_output(true);
        }
        if(invVarianceAttr)
        {
            invVarianceAttr->set_output(true);
        }

        return graphObj;
    }

    /// Batchnorm Inference graph
    static Graph createBatchnormInferenceGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_BatchnormInference");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 16, 8, 8};
        const std::vector<int64_t> scaleDims = hipdnn_data_sdk::utilities::getDerivedShape(xDims);
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto scaleAttr = graph::makeTensorAttributes("scale", scaleDims, scaleStrides);
        auto biasAttr = graph::makeTensorAttributes("bias", scaleDims, scaleStrides);
        auto meanAttr = graph::makeTensorAttributes("mean", scaleDims, scaleStrides);
        auto invVarAttr = graph::makeTensorAttributes("invVariance", scaleDims, scaleStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));
        auto invVarTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(invVarAttr));

        const graph::BatchnormInferenceAttributes bnAttrs;

        auto yAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                  meanTensorAttr,
                                                  invVarTensorAttr,
                                                  scaleTensorAttr,
                                                  biasTensorAttr,
                                                  bnAttrs);

        yAttr->set_output(true);

        return graphObj;
    }

    /// Batchnorm Backward graph
    static Graph createBatchnormBackwardGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_BatchnormBackward");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 16, 8, 8};
        const std::vector<int64_t> scaleDims = hipdnn_data_sdk::utilities::getDerivedShape(xDims);
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);

        auto dyAttr = graph::makeTensorAttributes("dy", xDims, xStrides);
        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto scaleAttr = graph::makeTensorAttributes("scale", scaleDims, scaleStrides);

        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        const graph::BatchnormBackwardAttributes bnAttrs;

        auto [dxAttr, dScaleAttr, dBiasAttr]
            = graphObj.batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        dxAttr->set_output(true);
        dScaleAttr->set_output(true);
        dBiasAttr->set_output(true);

        return graphObj;
    }

    /// Matmul graph
    static Graph createMatmulGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_Matmul");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> aDims = {2, 3};
        auto aStrides = hipdnn_data_sdk::utilities::generateStrides(aDims);
        auto aAttr = graph::makeTensorAttributes("A", aDims, aStrides);
        auto aTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        const std::vector<int64_t> bDims = {3, 4};
        auto bStrides = hipdnn_data_sdk::utilities::generateStrides(bDims);
        auto bAttr = graph::makeTensorAttributes("B", bDims, bStrides);
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        const graph::MatmulAttributes matmulAttrs;

        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);
        cAttr->set_output(true);

        return graphObj;
    }

    /// Layernorm graph
    static Graph createLayernormGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_Layernorm");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 8, 16};
        const std::vector<int64_t> scaleDims = {1, 8, 16};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto scaleAttr = graph::makeTensorAttributes("scale", scaleDims, scaleStrides);
        auto biasAttr = graph::makeTensorAttributes("bias", scaleDims, scaleStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_value(1e-5).set_name("epsilon");

        graph::LayernormAttributes lnAttrs;
        lnAttrs.set_epsilon(epsilonTensorAttr);
        lnAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);

        auto [yAttr, meanAttr, invVarianceAttr]
            = graphObj.layernorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, lnAttrs);

        yAttr->set_output(true);
        if(meanAttr)
        {
            meanAttr->set_output(true);
        }
        if(invVarianceAttr)
        {
            invVarianceAttr->set_output(true);
        }

        return graphObj;
    }

    /// RMSNorm graph
    static Graph createRmsnormGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_RMSNorm");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 8, 16};
        const std::vector<int64_t> scaleDims = {1, 8, 1};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto scaleAttr = graph::makeTensorAttributes("scale", scaleDims, scaleStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_value(1e-5).set_name("epsilon");

        graph::RMSNormAttributes rmsAttrs;
        rmsAttrs.set_epsilon(epsilonTensorAttr);
        rmsAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);

        auto [yAttr, invRmsAttr] = graphObj.rmsnorm(xTensorAttr, scaleTensorAttr, rmsAttrs);

        yAttr->set_output(true);
        if(invRmsAttr)
        {
            invRmsAttr->set_output(true);
        }

        return graphObj;
    }

    /// Reduction graph
    static Graph createReductionGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_Reduction");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 4, 16, 16};
        const std::vector<int64_t> yDims = {2, 4, 1, 1};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
        auto yStrides = hipdnn_data_sdk::utilities::generateStrides(yDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto yAttr = graph::makeTensorAttributes("y", yDims, yStrides);
        auto yTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(yAttr));

        graph::ReductionAttributes reductionAttrs;
        reductionAttrs.set_mode(hipdnn_frontend::ReductionMode::ADD);

        auto yResult = graphObj.reduction(xTensorAttr, yTensorAttr, reductionAttrs);
        yResult->set_output(true);

        return graphObj;
    }

#ifdef HIPDNN_ENABLE_SDPA
    /// SDPA Forward graph
    static Graph createSdpaForwardGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_SdpaForward");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> qkvDims = {2, 8, 32, 64};
        auto qkvStrides = hipdnn_data_sdk::utilities::generateStrides(qkvDims);

        auto qAttr = graph::makeTensorAttributes("Q", qkvDims, qkvStrides);
        auto kAttr = graph::makeTensorAttributes("K", qkvDims, qkvStrides);
        auto vAttr = graph::makeTensorAttributes("V", qkvDims, qkvStrides);

        auto qTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(qAttr));
        auto kTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(kAttr));
        auto vTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(vAttr));

        const graph::SdpaAttributes sdpaAttrs;

        auto [oAttr, statsAttr] = graphObj.sdpa(qTensorAttr, kTensorAttr, vTensorAttr, sdpaAttrs);
        oAttr->set_output(true);

        return graphObj;
    }

    /// SDPA Backward graph
    static Graph createSdpaBackwardGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_SdpaBackward");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> qkvDims = {2, 8, 32, 64};
        const std::vector<int64_t> statsDims = {2, 8, 32, 1};
        auto qkvStrides = hipdnn_data_sdk::utilities::generateStrides(qkvDims);
        auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(statsDims);

        auto qAttr = graph::makeTensorAttributes("Q", qkvDims, qkvStrides);
        auto kAttr = graph::makeTensorAttributes("K", qkvDims, qkvStrides);
        auto vAttr = graph::makeTensorAttributes("V", qkvDims, qkvStrides);
        auto oAttr = graph::makeTensorAttributes("O", qkvDims, qkvStrides);
        auto dOAttr = graph::makeTensorAttributes("dO", qkvDims, qkvStrides);
        auto statsAttr = graph::makeTensorAttributes("stats", statsDims, statsStrides);

        auto qTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(qAttr));
        auto kTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(kAttr));
        auto vTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(vAttr));
        auto oTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(oAttr));
        auto dOTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dOAttr));
        auto statsTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(statsAttr));

        const graph::SdpaBackwardAttributes sdpaBwdAttrs;

        auto [dqAttr, dkAttr, dvAttr] = graphObj.sdpa_backward(qTensorAttr,
                                                               kTensorAttr,
                                                               vTensorAttr,
                                                               oTensorAttr,
                                                               dOTensorAttr,
                                                               statsTensorAttr,
                                                               sdpaBwdAttrs);

        dqAttr->set_output(true);
        dkAttr->set_output(true);
        dvAttr->set_output(true);

        return graphObj;
    }
#endif // HIPDNN_ENABLE_SDPA

    /// Pointwise Unary (RELU_FWD) graph
    static Graph createPointwiseUnaryGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_PointwiseUnary");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 4, 16, 16};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);

        auto yAttr = graphObj.pointwise(xTensorAttr, pwAttrs);
        yAttr->set_output(true);

        return graphObj;
    }

    /// Pointwise Binary (ADD) graph
    static Graph createPointwiseBinaryGraph()
    {
        Graph graphObj;
        graphObj.set_name("Test_PointwiseBinary");
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        const std::vector<int64_t> xDims = {2, 4, 16, 16};
        auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);

        auto xAttr = graph::makeTensorAttributes("x", xDims, xStrides);
        auto yAttr = graph::makeTensorAttributes("y", xDims, xStrides);

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto yTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(yAttr));

        graph::PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);

        auto outAttr = graphObj.pointwise(xTensorAttr, yTensorAttr, pwAttrs);
        outAttr->set_output(true);

        return graphObj;
    }
};

/// Convert OperationType to string for test naming
inline std::string operationTypeToString(OperationType op)
{
    switch(op)
    {
    case OperationType::CONV_FORWARD:
        return "ConvForward";
    case OperationType::CONV_BACKWARD_DATA:
        return "ConvBackwardData";
    case OperationType::CONV_BACKWARD_WEIGHTS:
        return "ConvBackwardWeights";
    case OperationType::CONV_FWD_BIAS_ACTIV:
        return "ConvFwdBiasActiv";
    case OperationType::BATCHNORM_TRAINING:
        return "BatchnormTraining";
    case OperationType::BATCHNORM_INFERENCE:
        return "BatchnormInference";
    case OperationType::BATCHNORM_BACKWARD:
        return "BatchnormBackward";
    case OperationType::MATMUL:
        return "Matmul";
    case OperationType::LAYERNORM:
        return "Layernorm";
    case OperationType::RMSNORM:
        return "RMSNorm";
    case OperationType::REDUCTION:
        return "Reduction";
#ifdef HIPDNN_ENABLE_SDPA
    case OperationType::SDPA_FORWARD:
        return "SdpaForward";
    case OperationType::SDPA_BACKWARD:
        return "SdpaBackward";
#endif
    case OperationType::POINTWISE_UNARY:
        return "PointwiseUnary";
    case OperationType::POINTWISE_BINARY:
        return "PointwiseBinary";
    default:
        return "Unknown";
    }
}

} // namespace hipdnn_test_sdk::utilities
