// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionFwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
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
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        setTensorDesc(attributeName, attributeType, elementCount, arrayOfElements);
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
        setConvMode(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        setDataType(_computeDataType,
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

void ConvolutionFwdOperationDescriptor::setTensorDesc(hipdnnBackendAttributeName_t attributeName,
                                                      hipdnnBackendAttributeType_t attributeType,
                                                      int64_t elementCount,
                                                      const void* arrayOfElements)
{
    checkSetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 arrayOfElements,
                 "ConvolutionFwdOperationDescriptor::setAttribute()");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ConvolutionFwdOperationDescriptor::setAttribute(): elementCount is not 1");

    auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM,
        "ConvolutionFwdOperationDescriptor::setAttribute(): Failed to unpack tensor "
        "descriptor");
    THROW_IF_FALSE(tensorDesc->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "ConvolutionFwdOperationDescriptor::setAttribute(): Tensor descriptor "
                   "not finalized");

    if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X)
    {
        _xDesc = tensorDesc;
        _data.x_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W)
    {
        _wDesc = tensorDesc;
        _data.w_tensor_uid = tensorDesc->getData().uid;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y)
    {
        _yDesc = tensorDesc;
        _data.y_tensor_uid = tensorDesc->getData().uid;
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "ConvolutionFwdOperationDescriptor::setTensorDesc(): unsupported "
                              "attribute name");
    }
}

void ConvolutionFwdOperationDescriptor::setConvMode(hipdnnBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    const void* arrayOfElements)
{
    checkSetArgs(HIPDNN_TYPE_INT64,
                 attributeType,
                 arrayOfElements,
                 "ConvolutionFwdOperationDescriptor::setAttribute()");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ConvolutionFwdOperationDescriptor::setAttribute(): elementCount is not 1");
    auto mode = static_cast<hipdnn_data_sdk::data_objects::ConvMode>(
        *static_cast<const int64_t*>(arrayOfElements));
    THROW_IF_TRUE(mode < hipdnn_data_sdk::data_objects::ConvMode::MIN
                      || mode > hipdnn_data_sdk::data_objects::ConvMode::MAX,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::setAttribute(): invalid ConvMode value");
    _data.conv_mode = mode;
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
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        getTensorDesc(
            attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
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
        getConvMode(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        getDataType(_computeDataType,
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

void ConvolutionFwdOperationDescriptor::getTensorDesc(hipdnnBackendAttributeName_t attributeName,
                                                      hipdnnBackendAttributeType_t attributeType,
                                                      int64_t requestedElementCount,
                                                      int64_t* elementCount,
                                                      void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 "ConvolutionFwdOperationDescriptor::getAttribute()");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "ConvolutionFwdOperationDescriptor::getAttribute(): elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ConvolutionFwdOperationDescriptor::getAttribute(): requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    std::shared_ptr<TensorDescriptor> desc;
    if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X)
    {
        desc = _xDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W)
    {
        desc = _wDesc;
    }
    else if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y)
    {
        desc = _yDesc;
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "ConvolutionFwdOperationDescriptor::getTensorDesc(): unsupported "
                              "attribute name");
    }
    HipdnnBackendDescriptor::packDescriptor(desc, arrayOfElements);
}

void ConvolutionFwdOperationDescriptor::getConvMode(hipdnnBackendAttributeType_t attributeType,
                                                    int64_t requestedElementCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements) const
{
    checkGetArgs(
        HIPDNN_TYPE_INT64, attributeType, "ConvolutionFwdOperationDescriptor::getAttribute()");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "ConvolutionFwdOperationDescriptor::getAttribute(): elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ConvolutionFwdOperationDescriptor::getAttribute(): requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<int64_t*>(arrayOfElements) = static_cast<int64_t>(_data.conv_mode);
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
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::ConvolutionFwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t ConvolutionFwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
}

std::string ConvolutionFwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ConvolutionFwdOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
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
