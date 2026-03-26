// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "PoolingBwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void PoolingBwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: DY tensor not set");
    THROW_IF_NULL(_dxDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: DX tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.window_size.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: window_size not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.pooling_mode == hipdnn_data_sdk::data_objects::PoolingMode::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PoolingBwdOperationDescriptor::finalize() failed: pooling_mode not set");

    HipdnnBackendDescriptorImpl<PoolingBwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void PoolingBwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "PoolingBwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DX:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_PRE_PADDINGS:
        setScalarVector<int64_t>(_data.pre_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_POST_PADDINGS:
        setScalarVector<int64_t>(_data.post_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_STRIDES:
        setScalarVector<int64_t>(_data.stride,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_WINDOW_SIZE:
        setScalarVector<int64_t>(_data.window_size,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_MODE:
        setPoolingMode(_data.pooling_mode,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "PoolingBwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PoolingBwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void PoolingBwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "PoolingBwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DX:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_PRE_PADDINGS:
        getScalarVector<int64_t>(_data.pre_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_POST_PADDINGS:
        getScalarVector<int64_t>(_data.post_padding,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_STRIDES:
        getScalarVector<int64_t>(_data.stride,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_WINDOW_SIZE:
        getScalarVector<int64_t>(_data.window_size,
                       HIPDNN_TYPE_INT64,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_MODE:
        getPoolingMode(_data.pooling_mode,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POOLING_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_POOLING_BACKWARD,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "PoolingBwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PoolingBwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    PoolingBwdOperationDescriptor::getTensorDescriptors() const
{
    return {_dyDesc, _dxDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    PoolingBwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::PoolingBwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t PoolingBwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_POOLING_BACKWARD_DESCRIPTOR;
}

std::string PoolingBwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "PoolingBwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", dx_uid=" + std::to_string(_data.dx_tensor_uid);
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

std::shared_ptr<PoolingBwdOperationDescriptor> PoolingBwdOperationDescriptor::fromNode(
    const hipdnn_data_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsPoolingBwdAttributes();
    THROW_IF_NULL(attrs, HIPDNN_STATUS_INTERNAL_ERROR,
                  "PoolingBwdOperationDescriptor::fromNode: PoolingBwdAttributes is null");

    auto desc = std::make_shared<PoolingBwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "PoolingBwdOperationDescriptor::fromNode: Dy");
    desc->_dxDesc = findTensorInMap(
        tensorMap, attrs->dx_tensor_uid, "PoolingBwdOperationDescriptor::fromNode: Dx");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
