// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/hip_mlops_engine/plans/ApplicabilityChecks.hpp"
#include <array>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>

namespace hip_kernel_provider::batchnorm
{

// True when a training graph carries a saved mean and inv_variance that are both virtual and
// requests no running-statistics update: the pair is then treated as not requested.
bool hasUnrequestedSavedPair(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace hip_kernel_provider::batchnorm

namespace hip_kernel_provider
{

class BatchnormValidator : public IValidator
{
    // --- Activation Mode Validators ---
private:
    static void checkFwdActivationModeSupported(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr);

    static void checkBwdActivationModeSupported(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr);

    // --- Validation Utilities ---

    static void validateSpatialDimensions(const std::vector<int64_t>& ioDims);

    void validateDimensionCount(size_t numDims) override;

    // --- Component Validators ---

    void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) override;

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::vector<int64_t>& affineTensorIds,
                                       const std::vector<int64_t>& statTensorIds,
                                       const std::vector<int64_t>& intermediateTensorIds);

    void checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds,
                                    bool isTraining);

    void checkTensorConfigSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds,
                                    const std::vector<int64_t>& intermediateTensorIds,
                                    bool isTraining);

public:
    BatchnormValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    // --- High-Level Configuration Validators ---

    void checkInferenceTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr);

    void checkInferenceVarianceExtTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            bnInfAttr);

    void checkInferenceActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr);

    void checkInferenceVarianceExtActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            bnInfAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr);

    void checkFwdTrainingTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr);

    void checkFwdTrainingActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr);

    void checkBwdTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr);

    void checkBwdActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr);
};

namespace batchnorm
{

/// A one-node graph reads and writes every operand through a device buffer, so none may be
/// virtual. Fused graphs carry virtual intermediates and are checked per fusion instead.
///
/// @throws HipdnnPluginException with BAD_PARAM for a virtual operand.
void checkOneNodeOperandsNotVirtual(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

/// The tensor wiring of a fused inference graph, batchnorm inference then a pointwise
/// activation on its y: the batchnorm's inputs are not virtual, y is virtual and the
/// activation's output is not.
///
/// @throws HipdnnPluginException with BAD_PARAM when the graph is not wired so.
void checkInferenceActivationTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

/// checkInferenceActivationTensors() for inference with a variance rather than its inverse.
void checkInferenceVarianceExtActivationTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

/// The first node of a batchnorm training graph: its BatchnormAttributes.
///
/// @throws HipdnnPluginException with BAD_PARAM when the node is not batchnorm training.
const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes&
    checkTrainingNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node);

/// The activation after batchnorm training: identity, ReLU, clipped ReLU or clamp (not leaky
/// ReLU), reading the batchnorm's y.
///
/// @throws HipdnnPluginException with BAD_PARAM otherwise.
const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& checkTrainingActivationNode(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr);

/// The tensor wiring of a one-node training graph as the training plan builder checks it: x,
/// scale, bias and y are not virtual, nor the saved statistics unless both are (then they are
/// not requested), nor any running statistics.
///
/// @throws HipdnnPluginException with BAD_PARAM when the graph is not wired so.
void checkTrainingTensorVirtuality1Node(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

/// The tensor wiring of a training graph fused with an activation: as the one-node graph, but
/// y is virtual and the activation's output is not.
///
/// @throws HipdnnPluginException with BAD_PARAM when the graph is not wired so.
void checkTrainingTensorVirtuality2Node(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

/// The tensor wiring of the three-node backward fusion, batchnorm inference then an
/// activation's backward then the batchnorm backward.
///
/// @throws HipdnnPluginException with BAD_PARAM when the graph is not wired so.
void checkBackwardActivationTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace batchnorm

} // namespace hip_kernel_provider
