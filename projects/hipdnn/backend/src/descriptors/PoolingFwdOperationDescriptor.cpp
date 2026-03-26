// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "PoolingFwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void PoolingFwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.window_size.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: window_size not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.pooling_mode == hipdnn_data_sdk::data_objects::PoolingMode::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingFwdOperationDescriptor::finalize() failed: pooling_mode not set");

    HipdnnBackendDescriptorImpl<PoolingFwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void PoolingFwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "PoolingFwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_PRE_PADDINGS:
        setScalarVector<int64_t>(_data.pre_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_POST_PADDINGS:
        setScalarVector<int64_t>(_data.post_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_STRIDES:
        setScalarVector<int64_t>(_data.stride,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_WINDOW_SIZE:
        setScalarVector<int64_t>(_data.window_size,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_MODE:
        setPoolingMode(_data.pooling_mode,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "PoolingFwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PoolingFwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void PoolingFwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "PoolingFwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_PRE_PADDINGS:
        getScalarVector<int64_t>(_data.pre_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_POST_PADDINGS:
        getScalarVector<int64_t>(_data.post_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_STRIDES:
        getScalarVector<int64_t>(_data.stride,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_WINDOW_SIZE:
        getScalarVector<int64_t>(_data.window_size,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_MODE:
        getPoolingMode(_data.pooling_mode,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_POOLING_FORWARD,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "PoolingFwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PoolingFwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    PoolingFwdOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _yDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    PoolingFwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::PoolingFwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t PoolingFwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR;
}

std::string PoolingFwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "PoolingFwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", window_size=" + vecToString(_data.window_size);
    str += ", pooling_mode=" + std::to_string(static_cast<int>(_data.pooling_mode));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<PoolingFwdOperationDescriptor> PoolingFwdOperationDescriptor::fromNode(
    const hipdnn_data_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsPoolingFwdAttributes();
    THROW_IF_NULL(attrs, HIPDNN_STATUS_INTERNAL_ERROR,
                  "PoolingFwdOperationDescriptor::fromNode: PoolingFwdAttributes is null");

    auto desc = std::make_shared<PoolingFwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "PoolingFwdOperationDescriptor::fromNode: X");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "PoolingFwdOperationDescriptor::fromNode: Y");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
