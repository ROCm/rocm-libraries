// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BlockScaleDequantizeOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void BlockScaleDequantizeOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BlockScaleDequantizeOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(
        _scaleDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "BlockScaleDequantizeOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BlockScaleDequantizeOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_blockSize.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "BlockScaleDequantizeOperationDescriptor::finalize() failed: block_size not set");
    THROW_IF_TRUE(
        _computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
        HIPDNN_STATUS_BAD_PARAM,
        "BlockScaleDequantizeOperationDescriptor::finalize() failed: compute data type not "
        "set");

    HipdnnBackendDescriptorImpl<BlockScaleDequantizeOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void BlockScaleDequantizeOperationDescriptor::setAttribute(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void* arrayOfElements)
{
    THROW_IF_TRUE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "BlockScaleDequantizeOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleDequantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleDequantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleDequantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT:
        setScalarVector(_blockSize,
                        HIPDNN_TYPE_INT32,
                        attributeType,
                        elementCount,
                        arrayOfElements,
                        "BlockScaleDequantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_IS_NEGATIVE_SCALE_EXT:
        setScalar(_data.is_negative_scale,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleDequantizeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_MATH_PREC_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BlockScaleDequantizeOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BlockScaleDequantizeOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void BlockScaleDequantizeOperationDescriptor::getAttribute(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements) const
{
    THROW_IF_FALSE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "BlockScaleDequantizeOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleDequantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleDequantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BlockScaleDequantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT:
        getScalarVector(_blockSize,
                        HIPDNN_TYPE_INT32,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "BlockScaleDequantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_IS_NEGATIVE_SCALE_EXT:
        getScalar(_data.is_negative_scale,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BlockScaleDequantizeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_MATH_PREC_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BlockScaleDequantizeOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BlockScaleDequantizeOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    BlockScaleDequantizeOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _scaleDesc, _yDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    BlockScaleDequantizeOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;

    auto attrsData = hipdnn_data_sdk::data_objects::BlockScaleDequantizeAttributesT(_data);
    attrsData.block_size = _blockSize;

    node->attributes.Set(std::move(attrsData));
    return node;
}

hipdnnBackendDescriptorType_t BlockScaleDequantizeOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR_EXT;
}

std::string BlockScaleDequantizeOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "BlockScaleDequantizeOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", block_size=" + vecToString(_blockSize);
    str += ", is_negative_scale=" + std::to_string(static_cast<int>(_data.is_negative_scale));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
