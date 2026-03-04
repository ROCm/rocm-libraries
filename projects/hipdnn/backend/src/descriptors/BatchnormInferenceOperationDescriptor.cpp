// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormInferenceOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void BatchnormInferenceOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_meanDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceOperationDescriptor::finalize() failed: MEAN tensor not set");
    THROW_IF_NULL(
        _invVarianceDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceOperationDescriptor::finalize() failed: INV_VARIANCE tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_biasDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceOperationDescriptor::finalize() failed: BIAS tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormInferenceOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    HipdnnBackendDescriptorImpl<BatchnormInferenceOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void BatchnormInferenceOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                         hipdnnBackendAttributeType_t attributeType,
                                                         int64_t elementCount,
                                                         const void* arrayOfElements)
{
    THROW_IF_TRUE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "BatchnormInferenceOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_MEAN:
        setTensorDescriptor(_meanDesc,
                            _data.mean_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_INV_VARIANCE:
        setTensorDescriptor(_invVarianceDesc,
                            _data.inv_variance_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_SCALE:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_BIAS:
        setTensorDescriptor(_biasDesc,
                            _data.bias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_Y:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_EXT_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BatchnormInferenceOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void BatchnormInferenceOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                         hipdnnBackendAttributeType_t attributeType,
                                                         int64_t requestedElementCount,
                                                         int64_t* elementCount,
                                                         void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "BatchnormInferenceOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_MEAN:
        getTensorDescriptor(_meanDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_INV_VARIANCE:
        getTensorDescriptor(_invVarianceDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_SCALE:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_BIAS:
        getTensorDescriptor(_biasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_EXT_Y:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_EXT_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BatchnormInferenceOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    BatchnormInferenceOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _meanDesc, _invVarianceDesc, _scaleDesc, _biasDesc, _yDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    BatchnormInferenceOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BatchnormInferenceOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_EXT_DESCRIPTOR;
}

std::string BatchnormInferenceOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "BatchnormInferenceOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", mean_uid=" + std::to_string(_data.mean_tensor_uid);
    str += ", inv_variance_uid=" + std::to_string(_data.inv_variance_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", bias_uid=" + std::to_string(_data.bias_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
