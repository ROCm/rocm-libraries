// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormInferenceOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
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
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT:
        setTensorDescriptor(_meanDesc,
                            _data.mean_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT:
        setTensorDescriptor(_invVarianceDesc,
                            _data.inv_variance_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT:
        setTensorDescriptor(_biasDesc,
                            _data.bias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BatchnormInferenceOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT:
        getTensorDescriptor(_meanDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT:
        getTensorDescriptor(_invVarianceDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT:
        getTensorDescriptor(_biasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BatchnormInferenceOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT,
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

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    BatchnormInferenceOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BatchnormInferenceOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT;
}

std::string BatchnormInferenceOperationDescriptor::toString() const
{
    std::string str = "BatchnormInferenceOperationDescriptor: {";
    str += "name=" + _name + ", ";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", mean_uid=" + std::to_string(_data.mean_tensor_uid);
    str += ", inv_variance_uid=" + std::to_string(_data.inv_variance_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", bias_uid=" + std::to_string(_data.bias_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

std::shared_ptr<BatchnormInferenceOperationDescriptor>
    BatchnormInferenceOperationDescriptor::fromNode(
        const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
        const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsBatchnormInferenceAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "BatchnormInferenceOperationDescriptor::fromNode: BatchnormInferenceAttributes "
                  "is null");

    auto desc = std::make_shared<BatchnormInferenceOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "BatchnormInferenceOperationDescriptor::fromNode: X");
    desc->_meanDesc = findTensorInMap(
        tensorMap, attrs->mean_tensor_uid, "BatchnormInferenceOperationDescriptor::fromNode: Mean");
    desc->_invVarianceDesc
        = findTensorInMap(tensorMap,
                          attrs->inv_variance_tensor_uid,
                          "BatchnormInferenceOperationDescriptor::fromNode: InvVariance");
    desc->_scaleDesc = findTensorInMap(tensorMap,
                                       attrs->scale_tensor_uid,
                                       "BatchnormInferenceOperationDescriptor::fromNode: Scale");
    desc->_biasDesc = findTensorInMap(
        tensorMap, attrs->bias_tensor_uid, "BatchnormInferenceOperationDescriptor::fromNode: Bias");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "BatchnormInferenceOperationDescriptor::fromNode: Y");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
