// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormInferenceVarianceExtOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void BatchnormInferenceVarianceExtOperationDescriptor::finalize()
{
    THROW_IF_NULL(
        _xDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(
        _meanDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: MEAN tensor not set");
    THROW_IF_NULL(_varianceDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: VARIANCE "
                  "tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: SCALE "
                  "tensor not set");
    THROW_IF_NULL(
        _biasDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: BIAS tensor not set");
    THROW_IF_NULL(
        _yDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_NULL(_epsilonDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: EPSILON "
                  "tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceVarianceExtOperationDescriptor::finalize() failed: compute "
                  "data type not "
                  "set");

    HipdnnBackendDescriptorImpl<BatchnormInferenceVarianceExtOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void BatchnormInferenceVarianceExtOperationDescriptor::setAttribute(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute() failed: "
                  "Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_MEAN:
        setTensorDescriptor(_meanDesc,
                            _data.mean_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_VARIANCE:
        setTensorDescriptor(_varianceDesc,
                            _data.variance_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_SCALE:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_BIAS:
        setTensorDescriptor(_biasDesc,
                            _data.bias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_Y:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_EPSILON:
        setTensorDescriptor(_epsilonDesc,
                            _data.epsilon_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_VAR_EXT_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BatchnormInferenceVarianceExtOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void BatchnormInferenceVarianceExtOperationDescriptor::getAttribute(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements) const
{
    THROW_IF_FALSE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_MEAN:
        getTensorDescriptor(_meanDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_VARIANCE:
        getTensorDescriptor(_varianceDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_SCALE:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_BIAS:
        getTensorDescriptor(_biasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_Y:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_EPSILON:
        getTensorDescriptor(_epsilonDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_VAR_EXT_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BatchnormInferenceVarianceExtOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    BatchnormInferenceVarianceExtOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _meanDesc, _varianceDesc, _scaleDesc, _biasDesc, _yDesc, _epsilonDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    BatchnormInferenceVarianceExtOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(
        hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExtT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BatchnormInferenceVarianceExtOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EXT_DESCRIPTOR;
}

std::string BatchnormInferenceVarianceExtOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "BatchnormInferenceVarianceExtOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", mean_uid=" + std::to_string(_data.mean_tensor_uid);
    str += ", variance_uid=" + std::to_string(_data.variance_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", bias_uid=" + std::to_string(_data.bias_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", epsilon_uid=" + std::to_string(_data.epsilon_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
