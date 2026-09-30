// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CustomOpOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void CustomOpOperationDescriptor::finalize()
{
    THROW_IF_TRUE(_data.custom_op_id.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "CustomOpOperationDescriptor::finalize() failed: custom_op_id not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "CustomOpOperationDescriptor::finalize() failed: compute data type not set");

    HipdnnBackendDescriptorImpl<CustomOpOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void CustomOpOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                               hipdnnBackendAttributeType_t attributeType,
                                               int64_t elementCount,
                                               const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "CustomOpOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT:
        setTensorDescriptorArray(_inputDescs,
                                 _data.input_tensor_uids,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "CustomOpOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT:
        setTensorDescriptorArray(_outputDescs,
                                 _data.output_tensor_uids,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "CustomOpOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT:
        setString(_data.custom_op_id,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "CustomOpOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT:
        setByteArray(_data.data,
                     attributeType,
                     elementCount,
                     arrayOfElements,
                     "CustomOpOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "CustomOpOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "CustomOpOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "CustomOpOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void CustomOpOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                               hipdnnBackendAttributeType_t attributeType,
                                               int64_t requestedElementCount,
                                               int64_t* elementCount,
                                               void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "CustomOpOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT:
        getTensorDescriptorArray(_inputDescs,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "CustomOpOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT:
        getTensorDescriptorArray(_outputDescs,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "CustomOpOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT:
        getString(_data.custom_op_id,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "CustomOpOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT:
        getByteArray(_data.data,
                     attributeType,
                     requestedElementCount,
                     elementCount,
                     arrayOfElements,
                     "CustomOpOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "CustomOpOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "CustomOpOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_CUSTOM_OP_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "CustomOpOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "CustomOpOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    CustomOpOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(_inputDescs.size() + _outputDescs.size());
    result.insert(result.end(), _inputDescs.begin(), _inputDescs.end());
    result.insert(result.end(), _outputDescs.begin(), _outputDescs.end());
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    CustomOpOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::CustomOpAttributesT(_data));
    return node;
}

std::shared_ptr<CustomOpOperationDescriptor> CustomOpOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsCustomOpAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "CustomOpOperationDescriptor::fromNode: CustomOpAttributes is null");

    auto desc = std::make_shared<CustomOpOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;

    // Look up input tensors from the tensor map
    desc->_inputDescs.reserve(attrs->input_tensor_uids.size());
    for(auto uid : attrs->input_tensor_uids)
    {
        desc->_inputDescs.push_back(
            findTensorInMap(tensorMap, uid, "CustomOpOperationDescriptor::fromNode: input tensor"));
    }

    // Look up output tensors from the tensor map
    desc->_outputDescs.reserve(attrs->output_tensor_uids.size());
    for(auto uid : attrs->output_tensor_uids)
    {
        desc->_outputDescs.push_back(findTensorInMap(
            tensorMap, uid, "CustomOpOperationDescriptor::fromNode: output tensor"));
    }

    desc->finalize();
    return desc;
}

hipdnnBackendDescriptorType_t CustomOpOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT;
}

std::string CustomOpOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "CustomOpOperationDescriptor: {";
    str += "name=" + _name;
    str += ", custom_op_id=" + _data.custom_op_id;
    str += ", inputs=" + vecToString(_data.input_tensor_uids);
    str += ", outputs=" + vecToString(_data.output_tensor_uids);
    str += ", data_size=" + std::to_string(_data.data.size());
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

} // namespace hipdnn_backend
