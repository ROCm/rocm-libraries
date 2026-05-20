// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorFactory.hpp"
#include "BackendEnumStringUtils.hpp"
#include "BatchnormBackwardOperationDescriptor.hpp"
#include "BatchnormInferenceOperationDescriptor.hpp"
#include "BatchnormInferenceVarianceExtOperationDescriptor.hpp"
#include "BatchnormOperationDescriptor.hpp"
#include "BlockScaleDequantizeOperationDescriptor.hpp"
#include "BlockScaleQuantizeOperationDescriptor.hpp"
#include "ConvolutionBwdOperationDescriptor.hpp"
#include "ConvolutionFwdOperationDescriptor.hpp"
#include "ConvolutionWrwOperationDescriptor.hpp"
#include "CustomOpOperationDescriptor.hpp"
#include "EngineConfigDescriptor.hpp"
#include "EngineDescriptor.hpp"
#include "EngineHeuristicDescriptor.hpp"
#include "ExecutionPlanDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnException.hpp"
#include "KnobDescriptor.hpp"
#include "KnobSettingDescriptor.hpp"
#include "LayernormOperationDescriptor.hpp"
#include "MatmulOperationDescriptor.hpp"
#include "PointwiseOperationDescriptor.hpp"
#include "RMSNormBackwardOperationDescriptor.hpp"
#include "RMSNormOperationDescriptor.hpp"
#include "ReductionOperationDescriptor.hpp"
#include "ResampleFwdOperationDescriptor.hpp"
#include "SdpaBwdOperationDescriptor.hpp"
#include "SdpaFwdOperationDescriptor.hpp"
#include "TensorDescriptor.hpp"
#include "VariantDescriptor.hpp"
// Required: EngineHeuristicDescriptor holds std::unique_ptr<SelectionHeuristic>
// via forward declaration, so the complete type must be visible where
// make_shared<EngineHeuristicDescriptor>() instantiates the destructor.
#include "heuristics/SelectionHeuristic.hpp"
#include "logging/Logging.hpp"

namespace hipdnn_backend
{

void DescriptorFactory::create(hipdnnBackendDescriptorType_t descriptorType,
                               hipdnnBackendDescriptor_t* descriptor)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");

    HIPDNN_BACKEND_LOG_INFO("Creating descriptor of type: {}",
                            hipdnnGetBackendDescriptorTypeName(descriptorType));

    std::shared_ptr<IBackendDescriptor> privateDesc;
    switch(descriptorType)
    {
    case HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        privateDesc = std::make_shared<EngineConfigDescriptor>();
        break;
    case HIPDNN_BACKEND_ENGINE_DESCRIPTOR:
        privateDesc = std::make_shared<EngineDescriptor>();
        break;
    case HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        privateDesc = std::make_shared<ExecutionPlanDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        privateDesc = std::make_shared<GraphDescriptor>();
        break;
    case HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        privateDesc = std::make_shared<VariantDescriptor>();
        break;
    case HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        privateDesc = std::make_shared<EngineHeuristicDescriptor>();
        break;
    case HIPDNN_BACKEND_TENSOR_DESCRIPTOR:
        privateDesc = std::make_shared<TensorDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        privateDesc = std::make_shared<ConvolutionFwdOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
        privateDesc = std::make_shared<ConvolutionWrwOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<BatchnormInferenceOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
        privateDesc = std::make_shared<KnobSettingDescriptor>();
        break;
    case HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
        privateDesc = std::make_shared<KnobDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
        privateDesc = std::make_shared<PointwiseOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
        privateDesc = std::make_shared<ConvolutionBwdOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<BatchnormBackwardOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_VARIANCE_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<BatchnormInferenceVarianceExtOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
        privateDesc = std::make_shared<BlockScaleQuantizeOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
        privateDesc = std::make_shared<MatmulOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<RMSNormOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR:
        privateDesc = std::make_shared<SdpaFwdOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<LayernormOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<BatchnormOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
        privateDesc = std::make_shared<BlockScaleDequantizeOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<CustomOpOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<SdpaBwdOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
        privateDesc = std::make_shared<ReductionOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        privateDesc = std::make_shared<ResampleFwdOperationDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT:
        privateDesc = std::make_shared<RMSNormBackwardOperationDescriptor>();
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              std::string("Descriptor type ")
                                  + hipdnnGetBackendDescriptorTypeName(descriptorType)
                                  + " is not supported.");
    }

    *descriptor = HipdnnBackendDescriptor::packDescriptor(privateDesc);

    HIPDNN_BACKEND_LOG_INFO("Created descriptor: {:p}", static_cast<void*>(*descriptor));
}

void DescriptorFactory::createGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                       const uint8_t* serializedGraph,
                                       size_t graphByteSize)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");
    THROW_IF_NULL(
        serializedGraph, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "serializedGraph is null.");
    THROW_IF_TRUE(graphByteSize == 0, HIPDNN_STATUS_BAD_PARAM, "graphByteSize is 0.");

    auto graphDescriptor = std::make_shared<GraphDescriptor>();
    graphDescriptor->deserializeGraph(serializedGraph, graphByteSize);
    *descriptor = HipdnnBackendDescriptor::packDescriptor(graphDescriptor);

    HIPDNN_BACKEND_LOG_INFO("Created graph descriptor: {:p}", static_cast<void*>(*descriptor));
}

void DescriptorFactory::createGraphFromJsonExt(hipdnnBackendDescriptor_t* descriptor,
                                               const char* jsonGraph,
                                               size_t jsonByteSize)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");
    THROW_IF_NULL(jsonGraph, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "jsonGraph is null.");
    THROW_IF_TRUE(jsonByteSize == 0, HIPDNN_STATUS_BAD_PARAM, "jsonByteSize is 0.");

    auto graphDescriptor = std::make_shared<GraphDescriptor>();
    GraphDescriptor::createFromJsonGraph(*graphDescriptor, jsonGraph, jsonByteSize);
    *descriptor = HipdnnBackendDescriptor::packDescriptor(graphDescriptor);

    HIPDNN_BACKEND_LOG_INFO("Created graph descriptor from JSON: {:p}",
                            static_cast<void*>(*descriptor));
}

void DescriptorFactory::destroy(hipdnnBackendDescriptor_t descriptor)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t is null.");

    delete descriptor;

    HIPDNN_BACKEND_LOG_INFO("Destroyed descriptor: {:p}", static_cast<void*>(descriptor));
}

} // namespace hipdnn_backend
