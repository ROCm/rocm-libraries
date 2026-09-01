// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MoeGroupedMatmulBwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void MoeGroupedMatmulBwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(
        _doutputDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulBwdOperationDescriptor::finalize() failed: DOUTPUT_DESC tensor not set");
    THROW_IF_NULL(
        _tokenDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulBwdOperationDescriptor::finalize() failed: TOKEN_DESC tensor not set");
    THROW_IF_NULL(_firstTokenOffsetDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeGroupedMatmulBwdOperationDescriptor::finalize() failed: "
                  "FIRST_TOKEN_OFFSET_DESC tensor not set");
    THROW_IF_NULL(
        _dweightDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulBwdOperationDescriptor::finalize() failed: DWEIGHT_DESC tensor not set");
    THROW_IF_TRUE(
        _computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulBwdOperationDescriptor::finalize() failed: compute data type not "
        "set");
    THROW_IF_TRUE(_firstTokenOffsetDesc->getData().data_type
                      != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeGroupedMatmulBwdOperationDescriptor::finalize() failed: "
                  "FIRST_TOKEN_OFFSET_DESC tensor must have "
                  "INT32 data type");

    HipdnnBackendDescriptorImpl<MoeGroupedMatmulBwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void MoeGroupedMatmulBwdOperationDescriptor::setAttribute(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void* arrayOfElements)
{
    THROW_IF_TRUE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "MoeGroupedMatmulBwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC:
        setTensorDescriptor(_doutputDesc,
                            _data.doutput_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC:
        setTensorDescriptor(_tokenDesc,
                            _data.token_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC:
        setTensorDescriptor(_firstTokenOffsetDesc,
                            _data.first_token_offset_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC:
        setTensorDescriptor(_dweightDesc,
                            _data.dweight_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "MoeGroupedMatmulBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "MoeGroupedMatmulBwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "MoeGroupedMatmulBwdOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void MoeGroupedMatmulBwdOperationDescriptor::getAttribute(
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "MoeGroupedMatmulBwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC:
        getTensorDescriptor(_doutputDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC:
        getTensorDescriptor(_tokenDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC:
        getTensorDescriptor(_firstTokenOffsetDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC:
        getTensorDescriptor(_dweightDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_MOE_GROUPED_MATMUL_BWD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "MoeGroupedMatmulBwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "MoeGroupedMatmulBwdOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    MoeGroupedMatmulBwdOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(4);
    result.push_back(_doutputDesc);
    result.push_back(_tokenDesc);
    result.push_back(_firstTokenOffsetDesc);
    result.push_back(_dweightDesc);
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    MoeGroupedMatmulBwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(
        hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulBwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t MoeGroupedMatmulBwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR;
}

std::string MoeGroupedMatmulBwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "MoeGroupedMatmulBwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", doutput_uid=" + std::to_string(_data.doutput_tensor_uid);
    str += ", token_uid=" + std::to_string(_data.token_tensor_uid);
    str += ", first_token_offset_uid=" + std::to_string(_data.first_token_offset_tensor_uid);
    str += ", dweight_uid=" + std::to_string(_data.dweight_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

std::shared_ptr<MoeGroupedMatmulBwdOperationDescriptor>
    MoeGroupedMatmulBwdOperationDescriptor::fromNode(
        const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
        const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsMoeGroupedMatmulBwdAttributes();
    THROW_IF_NULL(
        attrs,
        HIPDNN_STATUS_INTERNAL_ERROR,
        "MoeGroupedMatmulBwdOperationDescriptor::fromNode: MoeGroupedMatmulBwdAttributes is null");

    auto desc = std::make_shared<MoeGroupedMatmulBwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_doutputDesc
        = findTensorInMap(tensorMap,
                          attrs->doutput_tensor_uid,
                          "MoeGroupedMatmulBwdOperationDescriptor::fromNode: Doutput");
    desc->_tokenDesc = findTensorInMap(tensorMap,
                                       attrs->token_tensor_uid,
                                       "MoeGroupedMatmulBwdOperationDescriptor::fromNode: Token");
    desc->_firstTokenOffsetDesc
        = findTensorInMap(tensorMap,
                          attrs->first_token_offset_tensor_uid,
                          "MoeGroupedMatmulBwdOperationDescriptor::fromNode: FirstTokenOffset");
    desc->_dweightDesc
        = findTensorInMap(tensorMap,
                          attrs->dweight_tensor_uid,
                          "MoeGroupedMatmulBwdOperationDescriptor::fromNode: Dweight");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
