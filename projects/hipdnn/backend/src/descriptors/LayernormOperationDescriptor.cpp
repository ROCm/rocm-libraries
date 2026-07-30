// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "LayernormOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void LayernormOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_biasDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: BIAS tensor not set");
    THROW_IF_NULL(_epsilonDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: EPSILON tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.forward_phase
                      == hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormOperationDescriptor::finalize() failed: forward_phase not set");

    const bool hasMean = _meanDesc != nullptr;
    const bool hasInvVariance = _invVarianceDesc != nullptr;
    THROW_IF_TRUE(
        hasMean != hasInvVariance,
        HIPDNN_STATUS_BAD_PARAM,
        "LayernormOperationDescriptor::finalize() failed: mean and inverse variance tensors must "
        "both be set or both be null");

    HipdnnBackendDescriptorImpl<LayernormOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void LayernormOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t elementCount,
                                                const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "LayernormOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT:
        setTensorDescriptor(_biasDesc,
                            _data.bias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT:
        setTensorDescriptor(_epsilonDesc,
                            _data.epsilon_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT:
        setOptionalTensorDescriptor(_meanDesc,
                                    _data.mean_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT:
        setOptionalTensorDescriptor(_invVarianceDesc,
                                    _data.inv_variance_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT:
        setNormFwdPhase(_data.forward_phase,
                        attributeType,
                        elementCount,
                        arrayOfElements,
                        "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT:
        setScalar(_data.normalized_dim_count,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "LayernormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "LayernormOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "LayernormOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void LayernormOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t requestedElementCount,
                                                int64_t* elementCount,
                                                void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "LayernormOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT:
        getTensorDescriptor(_biasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT:
        getTensorDescriptor(_epsilonDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT:
        getOptionalTensorDescriptor(_meanDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT:
        getOptionalTensorDescriptor(_invVarianceDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT:
        getNormFwdPhase(_data.forward_phase,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT:
        getScalar(_data.normalized_dim_count,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "LayernormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_LAYERNORM_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "LayernormOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "LayernormOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    LayernormOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result
        = {_xDesc, _scaleDesc, _biasDesc, _epsilonDesc, _yDesc};
    if(_meanDesc && _invVarianceDesc)
    {
        result.push_back(_meanDesc);
        result.push_back(_invVarianceDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    LayernormOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::LayernormAttributesT(_data));
    return node;
}

std::shared_ptr<LayernormOperationDescriptor> LayernormOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsLayernormAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "LayernormOperationDescriptor::fromNode: LayernormAttributes is null");

    auto desc = std::make_shared<LayernormOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;

    // Required tensors
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "LayernormOperationDescriptor::fromNode: X");
    desc->_scaleDesc = findTensorInMap(
        tensorMap, attrs->scale_tensor_uid, "LayernormOperationDescriptor::fromNode: Scale");
    desc->_biasDesc = findTensorInMap(
        tensorMap, attrs->bias_tensor_uid, "LayernormOperationDescriptor::fromNode: Bias");
    desc->_epsilonDesc = findTensorInMap(
        tensorMap, attrs->epsilon_tensor_uid, "LayernormOperationDescriptor::fromNode: Epsilon");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "LayernormOperationDescriptor::fromNode: Y");

    // Optional tensors
    if(attrs->mean_tensor_uid.has_value())
    {
        desc->_meanDesc = findTensorInMap(tensorMap,
                                          attrs->mean_tensor_uid.value(),
                                          "LayernormOperationDescriptor::fromNode: Mean");
    }
    if(attrs->inv_variance_tensor_uid.has_value())
    {
        desc->_invVarianceDesc
            = findTensorInMap(tensorMap,
                              attrs->inv_variance_tensor_uid.value(),
                              "LayernormOperationDescriptor::fromNode: InvVariance");
    }

    desc->finalize();
    return desc;
}

hipdnnBackendDescriptorType_t LayernormOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT;
}

std::string LayernormOperationDescriptor::toString() const
{
    std::string str = "LayernormOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", bias_uid=" + std::to_string(_data.bias_tensor_uid);
    str += ", epsilon_uid=" + std::to_string(_data.epsilon_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", normalized_dim_count=" + std::to_string(_data.normalized_dim_count);
    if(_data.mean_tensor_uid.has_value())
    {
        str += ", mean_uid=" + std::to_string(_data.mean_tensor_uid.value());
    }
    if(_data.inv_variance_tensor_uid.has_value())
    {
        str += ", inv_variance_uid=" + std::to_string(_data.inv_variance_tensor_uid.value());
    }
    str += ", forward_phase=" + std::to_string(static_cast<int>(_data.forward_phase));
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

} // namespace hipdnn_backend
