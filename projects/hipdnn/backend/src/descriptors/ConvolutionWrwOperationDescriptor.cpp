// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionWrwOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void ConvolutionWrwOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: DY tensor not set");
    THROW_IF_NULL(_dwDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: DW tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.dilation.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: dilation not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.conv_mode == hipdnn_data_sdk::data_objects::ConvMode::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionWrwOperationDescriptor::finalize() failed: conv_mode not set");

    HipdnnBackendDescriptorImpl<ConvolutionWrwOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ConvolutionWrwOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ConvolutionWrwOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DW:
        setTensorDescriptor(_dwDesc,
                            _data.dw_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        setInt64Vector(_data.pre_padding,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        setInt64Vector(_data.post_padding,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        setInt64Vector(_data.stride,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        setInt64Vector(_data.dilation,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        setConvMode(_data.conv_mode,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ConvolutionWrwOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void ConvolutionWrwOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ConvolutionWrwOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DW:
        getTensorDescriptor(_dwDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        getInt64Vector(_data.pre_padding,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        getInt64Vector(_data.post_padding,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        getInt64Vector(_data.stride,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        getInt64Vector(_data.dilation,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        getConvMode(_data.conv_mode,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ConvolutionWrwOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    ConvolutionWrwOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _dyDesc, _dwDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    ConvolutionWrwOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::ConvolutionWrwAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t ConvolutionWrwOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
}

std::string ConvolutionWrwOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ConvolutionWrwOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", dw_uid=" + std::to_string(_data.dw_tensor_uid);
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", dilation=" + vecToString(_data.dilation);
    str += ", conv_mode=";
    str += hipdnn_data_sdk::data_objects::EnumNameConvMode(_data.conv_mode);
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
