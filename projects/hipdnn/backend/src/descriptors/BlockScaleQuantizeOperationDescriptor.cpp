// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BlockScaleQuantizeOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void BlockScaleQuantizeOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BlockScaleQuantizeOperationDescriptor::finalize() failed: X_EXT tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BlockScaleQuantizeOperationDescriptor::finalize() failed: Y_EXT tensor not set");
    THROW_IF_NULL(
        _scaleDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BlockScaleQuantizeOperationDescriptor::finalize() failed: SCALE_EXT tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BlockScaleQuantizeOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_FALSE(_data.block_size > 0,
                   HIPDNN_STATUS_BAD_PARAM,
                   "BlockScaleQuantizeOperationDescriptor::finalize() failed: block_size not set");

    HipdnnBackendDescriptorImpl<BlockScaleQuantizeOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void BlockScaleQuantizeOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                         hipdnnBackendAttributeType_t attributeType,
                                                         int64_t elementCount,
                                                         const void* arrayOfElements)
{
    THROW_IF_TRUE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "BlockScaleQuantizeOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE:
        setScalar(_data.block_size,
                  HIPDNN_TYPE_INT32,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT64>(
            _data.axis,
            attributeType,
            elementCount,
            arrayOfElements,
            "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT:
        setScalar(_data.transpose,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleQuantizeOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BlockScaleQuantizeOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void BlockScaleQuantizeOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                         hipdnnBackendAttributeType_t attributeType,
                                                         int64_t requestedElementCount,
                                                         int64_t* elementCount,
                                                         void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "BlockScaleQuantizeOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE:
        getScalar(_data.block_size,
                  HIPDNN_TYPE_INT32,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT64>(
            _data.axis,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT:
        getScalar(_data.transpose,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "BlockScaleQuantizeOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BlockScaleQuantizeOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    BlockScaleQuantizeOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _yDesc, _scaleDesc};
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    BlockScaleQuantizeOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(
        hipdnn_flatbuffers_sdk::data_objects::BlockScaleQuantizeAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BlockScaleQuantizeOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR;
}

std::string BlockScaleQuantizeOperationDescriptor::toString() const
{
    std::string str = "BlockScaleQuantizeOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", block_size=" + std::to_string(_data.block_size);
    str += ", axis=" + (_data.axis.has_value() ? std::to_string(_data.axis.value()) : "null");
    str += std::string(", transpose=") + (_data.transpose ? "true" : "false");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

std::shared_ptr<BlockScaleQuantizeOperationDescriptor>
    BlockScaleQuantizeOperationDescriptor::fromNode(
        const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
        const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsBlockScaleQuantizeAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "BlockScaleQuantizeOperationDescriptor::fromNode: "
                  "BlockScaleQuantizeAttributes is null");

    auto desc = std::make_shared<BlockScaleQuantizeOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;

    // Required tensors
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "BlockScaleQuantizeOperationDescriptor::fromNode: X");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "BlockScaleQuantizeOperationDescriptor::fromNode: Y");
    desc->_scaleDesc = findTensorInMap(tensorMap,
                                       attrs->scale_tensor_uid,
                                       "BlockScaleQuantizeOperationDescriptor::fromNode: Scale");

    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
