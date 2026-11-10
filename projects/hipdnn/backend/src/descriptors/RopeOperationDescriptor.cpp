// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RopeOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void RopeOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RopeOperationDescriptor::finalize() failed: XDESC tensor not set");
    THROW_IF_NULL(_freqsDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RopeOperationDescriptor::finalize() failed: FREQSDESC tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RopeOperationDescriptor::finalize() failed: YDESC tensor not set");
    THROW_IF_TRUE(_freqsDesc->getData().data_type
                      != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RopeOperationDescriptor::finalize() failed: FREQSDESC tensor must have "
                  "FLOAT data type");

    HipdnnBackendDescriptorImpl<RopeOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void RopeOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "RopeOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RopeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC:
        setTensorDescriptor(_freqsDesc,
                            _data.freqs_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RopeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RopeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE:
        setScalar(_data.output_scale,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "RopeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM:
        setScalar(_data.rope_dim,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "RopeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_data.rope_offset_ext,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "RopeOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "RopeOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "RopeOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void RopeOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "RopeOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC:
        getTensorDescriptor(_freqsDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE:
        getScalar(_data.output_scale,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM:
        getScalar(_data.rope_dim,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_data.rope_offset_ext,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "RopeOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_ROPE_FWD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "RopeOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "RopeOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>> RopeOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(3);
    result.push_back(_xDesc);
    result.push_back(_freqsDesc);
    result.push_back(_yDesc);
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    RopeOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::RopeAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t RopeOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR;
}

std::string RopeOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "RopeOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", freqs_uid=" + std::to_string(_data.freqs_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", output_scale=" + std::to_string(_data.output_scale);
    str += ", rope_dim=" + std::to_string(_data.rope_dim);
    str += ", rope_offset="
           + (_data.rope_offset_ext ? std::to_string(*_data.rope_offset_ext) : "nullopt");
    str += "}";
    return str;
}

std::shared_ptr<RopeOperationDescriptor> RopeOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsRopeAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "RopeOperationDescriptor::fromNode: RopeAttributes is null");

    auto desc = std::make_shared<RopeOperationDescriptor>();
    desc->_data = *attrs;
    desc->_name = nodeT.name;
    desc->_xDesc
        = findTensorInMap(tensorMap, attrs->x_tensor_uid, "RopeOperationDescriptor::fromNode: X");
    desc->_freqsDesc = findTensorInMap(
        tensorMap, attrs->freqs_tensor_uid, "RopeOperationDescriptor::fromNode: Freqs");
    desc->_yDesc
        = findTensorInMap(tensorMap, attrs->y_tensor_uid, "RopeOperationDescriptor::fromNode: Y");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
