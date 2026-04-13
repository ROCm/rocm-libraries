// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionWrwOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
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
    validateConvolutionFinalize(_data, _computeDataType, "ConvolutionWrwOperationDescriptor");

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
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
        setTensorDescriptor(_dwDesc,
                            _data.dw_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
    default:
        setConvolutionAttribute(_data,
                                _computeDataType,
                                _name,
                                attributeName,
                                attributeType,
                                elementCount,
                                arrayOfElements,
                                "ConvolutionWrwOperationDescriptor::setAttribute()");
        break;
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
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
        getTensorDescriptor(_dwDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
    default:
        getConvolutionAttribute(_data,
                                _computeDataType,
                                _name,
                                HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_WEIGHTS_EXT,
                                attributeName,
                                attributeType,
                                requestedElementCount,
                                elementCount,
                                arrayOfElements,
                                "ConvolutionWrwOperationDescriptor::getAttribute()");
        break;
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

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    ConvolutionWrwOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributesT(_data));
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
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", dw_uid=" + std::to_string(_data.dw_tensor_uid);
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", dilation=" + vecToString(_data.dilation);
    str += ", conv_mode=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameConvMode(_data.conv_mode);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<ConvolutionWrwOperationDescriptor> ConvolutionWrwOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsConvolutionWrwAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ConvolutionWrwOperationDescriptor::fromNode: ConvolutionWrwAttributes is null");

    auto desc = std::make_shared<ConvolutionWrwOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "ConvolutionWrwOperationDescriptor::fromNode: X");
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "ConvolutionWrwOperationDescriptor::fromNode: Dy");
    desc->_dwDesc = findTensorInMap(
        tensorMap, attrs->dw_tensor_uid, "ConvolutionWrwOperationDescriptor::fromNode: Dw");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
