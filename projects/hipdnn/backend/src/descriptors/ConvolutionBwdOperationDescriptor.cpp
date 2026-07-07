// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionBwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void ConvolutionBwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionBwdOperationDescriptor::finalize() failed: DY tensor not set");
    THROW_IF_NULL(_wDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionBwdOperationDescriptor::finalize() failed: W tensor not set");
    THROW_IF_NULL(_dxDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionBwdOperationDescriptor::finalize() failed: DX tensor not set");
    validateConvolutionFinalize(_data, _computeDataType, "ConvolutionBwdOperationDescriptor");

    HipdnnBackendDescriptorImpl<ConvolutionBwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ConvolutionBwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ConvolutionBwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
        setTensorDescriptor(_wDesc,
                            _data.w_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionBwdOperationDescriptor::setAttribute()");
        break;
    default:
        setConvolutionAttribute(_data,
                                _computeDataType,
                                _name,
                                attributeName,
                                attributeType,
                                elementCount,
                                arrayOfElements,
                                "ConvolutionBwdOperationDescriptor::setAttribute()");
        break;
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void ConvolutionBwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ConvolutionBwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
        getTensorDescriptor(_wDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ConvolutionBwdOperationDescriptor::getAttribute()");
        break;
    default:
        getConvolutionAttribute(_data,
                                _computeDataType,
                                _name,
                                HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_DATA_EXT,
                                attributeName,
                                attributeType,
                                requestedElementCount,
                                elementCount,
                                arrayOfElements,
                                "ConvolutionBwdOperationDescriptor::getAttribute()");
        break;
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    ConvolutionBwdOperationDescriptor::getTensorDescriptors() const
{
    return {_dyDesc, _wDesc, _dxDesc};
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    ConvolutionBwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t ConvolutionBwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
}

std::string ConvolutionBwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ConvolutionBwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", w_uid=" + std::to_string(_data.w_tensor_uid);
    str += ", dx_uid=" + std::to_string(_data.dx_tensor_uid);
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", dilation=" + vecToString(_data.dilation);
    str += ", conv_mode=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameConvMode(_data.conv_mode);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

std::shared_ptr<ConvolutionBwdOperationDescriptor> ConvolutionBwdOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsConvolutionBwdAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ConvolutionBwdOperationDescriptor::fromNode: ConvolutionBwdAttributes is null");

    auto desc = std::make_shared<ConvolutionBwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "ConvolutionBwdOperationDescriptor::fromNode: Dy");
    desc->_wDesc = findTensorInMap(
        tensorMap, attrs->w_tensor_uid, "ConvolutionBwdOperationDescriptor::fromNode: W");
    desc->_dxDesc = findTensorInMap(
        tensorMap, attrs->dx_tensor_uid, "ConvolutionBwdOperationDescriptor::fromNode: Dx");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
