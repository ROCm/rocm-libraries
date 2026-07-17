// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "PointwiseOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void PointwiseOperationDescriptor::finalize()
{
    THROW_IF_NULL(_in0Desc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PointwiseOperationDescriptor::finalize() failed: IN_0 tensor not set");
    THROW_IF_NULL(_out0Desc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PointwiseOperationDescriptor::finalize() failed: OUT_0 tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PointwiseOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.operation == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PointwiseOperationDescriptor::finalize() failed: operation not set");

    HipdnnBackendDescriptorImpl<PointwiseOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void PointwiseOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t elementCount,
                                                const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "PointwiseOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT:
        setTensorDescriptor(_in0Desc,
                            _data.in_0_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PointwiseOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT:
        setTensorDescriptor(_out0Desc,
                            _data.out_0_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PointwiseOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT:
        setOptionalTensorDescriptor(_in1Desc,
                                    _data.in_1_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "PointwiseOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT:
        setOptionalTensorDescriptor(_in2Desc,
                                    _data.in_2_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "PointwiseOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POINTWISE_AXIS:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_data.axis_tensor_uid,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "PointwiseOperationDescriptor::setAttribute(AXIS)");
        break;
    case HIPDNN_ATTR_POINTWISE_MODE:
        setPointwiseMode(_data.operation,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         "PointwiseOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.relu_lower_clip,
            attributeType,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::setAttribute(RELU_LOWER_CLIP)");
        break;
    case HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.relu_upper_clip,
            attributeType,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::setAttribute(RELU_UPPER_CLIP)");
        break;
    case HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.relu_lower_clip_slope,
            attributeType,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::setAttribute(RELU_LOWER_CLIP_SLOPE)");
        break;
    case HIPDNN_ATTR_POINTWISE_SWISH_BETA:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.swish_beta,
            attributeType,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::setAttribute(SWISH_BETA)");
        break;
    case HIPDNN_ATTR_POINTWISE_ELU_ALPHA:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.elu_alpha,
            attributeType,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::setAttribute(ELU_ALPHA)");
        break;
    case HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.softplus_beta,
            attributeType,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::setAttribute(SOFTPLUS_BETA)");
        break;
    case HIPDNN_ATTR_POINTWISE_MATH_PREC:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "PointwiseOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "PointwiseOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PointwiseOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void PointwiseOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t requestedElementCount,
                                                int64_t* elementCount,
                                                void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "PointwiseOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT:
        getTensorDescriptor(_in0Desc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT:
        getTensorDescriptor(_out0Desc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT:
        getOptionalTensorDescriptor(_in1Desc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT:
        getOptionalTensorDescriptor(_in2Desc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POINTWISE_AXIS:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_data.axis_tensor_uid,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "PointwiseOperationDescriptor::getAttribute(AXIS)");
        break;
    case HIPDNN_ATTR_POINTWISE_MODE:
        getPointwiseMode(_data.operation,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.relu_lower_clip,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::getAttribute(RELU_LOWER_CLIP)");
        break;
    case HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.relu_upper_clip,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::getAttribute(RELU_UPPER_CLIP)");
        break;
    case HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.relu_lower_clip_slope,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::getAttribute(RELU_LOWER_CLIP_SLOPE)");
        break;
    case HIPDNN_ATTR_POINTWISE_SWISH_BETA:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.swish_beta,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::getAttribute(SWISH_BETA)");
        break;
    case HIPDNN_ATTR_POINTWISE_ELU_ALPHA:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.elu_alpha,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::getAttribute(ELU_ALPHA)");
        break;
    case HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(
            _data.softplus_beta,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
            "PointwiseOperationDescriptor::getAttribute(SOFTPLUS_BETA)");
        break;
    case HIPDNN_ATTR_POINTWISE_MATH_PREC:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "PointwiseOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_POINTWISE_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "PointwiseOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PointwiseOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    PointwiseOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result = {_in0Desc, _out0Desc};
    if(_in1Desc)
    {
        result.push_back(_in1Desc);
    }
    if(_in2Desc)
    {
        result.push_back(_in2Desc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    PointwiseOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t PointwiseOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
}

std::string PointwiseOperationDescriptor::toString() const
{
    std::string str = "PointwiseOperationDescriptor: {";
    str += "name=" + _name;
    str += ", in_0_uid=" + std::to_string(_data.in_0_tensor_uid);
    str += ", out_0_uid=" + std::to_string(_data.out_0_tensor_uid);
    str += ", in_1_uid="
           + (_data.in_1_tensor_uid ? std::to_string(*_data.in_1_tensor_uid) : "nullopt");
    str += ", in_2_uid="
           + (_data.in_2_tensor_uid ? std::to_string(*_data.in_2_tensor_uid) : "nullopt");
    str += ", axis=" + (_data.axis_tensor_uid ? std::to_string(*_data.axis_tensor_uid) : "nullopt");
    str += ", operation=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(_data.operation);
    str += ", relu_lower_clip="
           + (_data.relu_lower_clip ? std::to_string(*_data.relu_lower_clip) : "nullopt");
    str += ", relu_upper_clip="
           + (_data.relu_upper_clip ? std::to_string(*_data.relu_upper_clip) : "nullopt");
    str += ", relu_lower_clip_slope="
           + (_data.relu_lower_clip_slope ? std::to_string(*_data.relu_lower_clip_slope)
                                          : "nullopt");
    str += ", swish_beta=" + (_data.swish_beta ? std::to_string(*_data.swish_beta) : "nullopt");
    str += ", elu_alpha=" + (_data.elu_alpha ? std::to_string(*_data.elu_alpha) : "nullopt");
    str += ", softplus_beta="
           + (_data.softplus_beta ? std::to_string(*_data.softplus_beta) : "nullopt");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

std::shared_ptr<PointwiseOperationDescriptor> PointwiseOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsPointwiseAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "PointwiseOperationDescriptor::fromNode: PointwiseAttributes is null");

    auto desc = std::make_shared<PointwiseOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_in0Desc = findTensorInMap(
        tensorMap, attrs->in_0_tensor_uid, "PointwiseOperationDescriptor::fromNode: In0");
    desc->_out0Desc = findTensorInMap(
        tensorMap, attrs->out_0_tensor_uid, "PointwiseOperationDescriptor::fromNode: Out0");
    if(attrs->in_1_tensor_uid)
    {
        desc->_in1Desc = findTensorInMap(
            tensorMap, *attrs->in_1_tensor_uid, "PointwiseOperationDescriptor::fromNode: In1");
    }
    if(attrs->in_2_tensor_uid)
    {
        desc->_in2Desc = findTensorInMap(
            tensorMap, *attrs->in_2_tensor_uid, "PointwiseOperationDescriptor::fromNode: In2");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
