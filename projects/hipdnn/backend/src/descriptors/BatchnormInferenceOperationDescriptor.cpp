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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y:
        setTensorDesc(attributeName, attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE:
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

void BatchnormInferenceOperationDescriptor::setTensorDesc(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void* arrayOfElements)
{
    checkSetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 arrayOfElements,
                 "BatchnormInferenceOperationDescriptor::setAttribute()");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "BatchnormInferenceOperationDescriptor::setAttribute(): elementCount is not 1");

    auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceOperationDescriptor::setAttribute(): Failed to unpack tensor "
        "descriptor");
    THROW_IF_FALSE(tensorDesc->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "BatchnormInferenceOperationDescriptor::setAttribute(): Tensor descriptor "
                   "not finalized");

    if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X)
    {
        _xDesc = tensorDesc;
        _data.x_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN)
    {
        _meanDesc = tensorDesc;
        _data.mean_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE)
    {
        _invVarianceDesc = tensorDesc;
        _data.inv_variance_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE)
    {
        _scaleDesc = tensorDesc;
        _data.scale_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS)
    {
        _biasDesc = tensorDesc;
        _data.bias_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y)
    {
        _yDesc = tensorDesc;
        _data.y_tensor_uid = tensorDesc->getData().uid;
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "BatchnormInferenceOperationDescriptor::setTensorDesc(): unsupported "
                              "attribute name");
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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS:
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y:
        getTensorDesc(
            attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE:
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

void BatchnormInferenceOperationDescriptor::getTensorDesc(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 "BatchnormInferenceOperationDescriptor::getAttribute()");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(
            elementCount,
            HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
            "BatchnormInferenceOperationDescriptor::getAttribute(): elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(
        requestedElementCount >= 1,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormInferenceOperationDescriptor::getAttribute(): requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    std::shared_ptr<TensorDescriptor> desc;
    if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X)
    {
        desc = _xDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN)
    {
        desc = _meanDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE)
    {
        desc = _invVarianceDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE)
    {
        desc = _scaleDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS)
    {
        desc = _biasDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y)
    {
        desc = _yDesc;
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "BatchnormInferenceOperationDescriptor::getTensorDesc(): unsupported "
                              "attribute name");
    }
    HipdnnBackendDescriptor::packDescriptor(desc, arrayOfElements);
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
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR;
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
