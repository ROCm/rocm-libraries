// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionFwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void ConvolutionFwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_wDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: W tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.dilation.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: dilation not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.conv_mode == hipdnn_data_sdk::data_objects::ConvMode::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: conv_mode not set");

    HipdnnBackendDescriptorImpl<ConvolutionFwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ConvolutionFwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ConvolutionFwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        setTensorDescriptor(_wDesc,
                            _data.w_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        setInt64Vector(_data.pre_padding,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        setInt64Vector(_data.post_padding,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        setInt64Vector(_data.stride,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        setInt64Vector(_data.dilation,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        setConvMode(_data.conv_mode,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "ConvolutionFwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ConvolutionFwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void ConvolutionFwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ConvolutionFwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        getTensorDescriptor(_wDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        getInt64Vector(_data.pre_padding,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        getInt64Vector(_data.post_padding,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        getInt64Vector(_data.stride,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        getInt64Vector(_data.dilation,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        getConvMode(_data.conv_mode,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "ConvolutionFwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ConvolutionFwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    ConvolutionFwdOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _wDesc, _yDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    ConvolutionFwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::ConvolutionFwdAttributesT(_data));
    return node;
}

std::shared_ptr<ConvolutionFwdOperationDescriptor> ConvolutionFwdOperationDescriptor::fromNode(
    const hipdnn_data_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsConvolutionFwdAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ConvolutionFwdOperationDescriptor::fromNode: ConvolutionFwdAttributes is null");

    auto desc = std::make_shared<ConvolutionFwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "ConvolutionFwdOperationDescriptor::fromNode: X");
    desc->_wDesc = findTensorInMap(
        tensorMap, attrs->w_tensor_uid, "ConvolutionFwdOperationDescriptor::fromNode: W");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "ConvolutionFwdOperationDescriptor::fromNode: Y");
    desc->finalize();
    return desc;
}

hipdnnBackendDescriptorType_t ConvolutionFwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
}

std::string ConvolutionFwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ConvolutionFwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", w_uid=" + std::to_string(_data.w_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", dilation=" + vecToString(_data.dilation);
    str += ", conv_mode=" + std::to_string(static_cast<int>(_data.conv_mode));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
