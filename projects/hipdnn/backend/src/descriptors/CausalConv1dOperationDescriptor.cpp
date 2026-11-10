// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CausalConv1dOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void CausalConv1dOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "CausalConv1dOperationDescriptor::finalize() failed: X_EXT tensor not set");
    THROW_IF_NULL(_weightDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "CausalConv1dOperationDescriptor::finalize() failed: WEIGHT_EXT tensor not set");
    THROW_IF_NULL(
        _cuSeqlensDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "CausalConv1dOperationDescriptor::finalize() failed: CU_SEQLENS_EXT tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "CausalConv1dOperationDescriptor::finalize() failed: Y_EXT tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "CausalConv1dOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_stateIndicesDesc != nullptr && _initialStateDesc == nullptr,
                  HIPDNN_STATUS_BAD_PARAM,
                  "CausalConv1dOperationDescriptor::finalize() failed: STATE_INDICES_EXT "
                  "requires INITIAL_STATE_EXT");

    HipdnnBackendDescriptorImpl<CausalConv1dOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void CausalConv1dOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                   hipdnnBackendAttributeType_t attributeType,
                                                   int64_t elementCount,
                                                   const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "CausalConv1dOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT:
        setTensorDescriptor(_weightDesc,
                            _data.weight_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT:
        setTensorDescriptor(_cuSeqlensDesc,
                            _data.cu_seqlens_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT:
        setOptionalTensorDescriptor(_biasDesc,
                                    _data.bias_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT:
        setOptionalTensorDescriptor(_initialStateDesc,
                                    _data.initial_state_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT:
        setOptionalTensorDescriptor(_stateIndicesDesc,
                                    _data.state_indices_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT:
        setOptionalTensorDescriptor(_finalStateDesc,
                                    _data.final_state_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT:
        setCausalConv1dActivation(_data.activation,
                                  attributeType,
                                  elementCount,
                                  arrayOfElements,
                                  "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "CausalConv1dOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "CausalConv1dOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void CausalConv1dOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                   hipdnnBackendAttributeType_t attributeType,
                                                   int64_t requestedElementCount,
                                                   int64_t* elementCount,
                                                   void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "CausalConv1dOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT:
        getTensorDescriptor(_weightDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT:
        getTensorDescriptor(_cuSeqlensDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT:
        getOptionalTensorDescriptor(_biasDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT:
        getOptionalTensorDescriptor(_initialStateDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT:
        getOptionalTensorDescriptor(_stateIndicesDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT:
        getOptionalTensorDescriptor(_finalStateDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT:
        getCausalConv1dActivation(_data.activation,
                                  attributeType,
                                  requestedElementCount,
                                  elementCount,
                                  arrayOfElements,
                                  "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_CAUSAL_CONV1D_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "CausalConv1dOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "CausalConv1dOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    CausalConv1dOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(8);
    result.push_back(_xDesc);
    result.push_back(_weightDesc);
    result.push_back(_cuSeqlensDesc);
    result.push_back(_yDesc);
    if(_biasDesc)
    {
        result.push_back(_biasDesc);
    }
    if(_initialStateDesc)
    {
        result.push_back(_initialStateDesc);
    }
    if(_stateIndicesDesc)
    {
        result.push_back(_stateIndicesDesc);
    }
    if(_finalStateDesc)
    {
        result.push_back(_finalStateDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    CausalConv1dOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::CausalConv1dAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t CausalConv1dOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CAUSAL_CONV1D_DESCRIPTOR_EXT;
}

std::string CausalConv1dOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "CausalConv1dOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", weight_uid=" + std::to_string(_data.weight_tensor_uid);
    str += ", cu_seqlens_uid=" + std::to_string(_data.cu_seqlens_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", bias_uid="
           + (_data.bias_tensor_uid ? std::to_string(*_data.bias_tensor_uid) : "nullopt");
    str += ", initial_state_uid="
           + (_data.initial_state_tensor_uid ? std::to_string(*_data.initial_state_tensor_uid)
                                             : "nullopt");
    str += ", state_indices_uid="
           + (_data.state_indices_tensor_uid ? std::to_string(*_data.state_indices_tensor_uid)
                                             : "nullopt");
    str += ", final_state_uid="
           + (_data.final_state_tensor_uid ? std::to_string(*_data.final_state_tensor_uid)
                                           : "nullopt");
    str += ", activation="
           + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameCausalConv1dActivation(
               _data.activation));
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<CausalConv1dOperationDescriptor> CausalConv1dOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsCausalConv1dAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "CausalConv1dOperationDescriptor::fromNode: CausalConv1dAttributes is null");

    auto desc = std::make_shared<CausalConv1dOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "CausalConv1dOperationDescriptor::fromNode: X");
    desc->_weightDesc = findTensorInMap(
        tensorMap, attrs->weight_tensor_uid, "CausalConv1dOperationDescriptor::fromNode: Weight");
    desc->_cuSeqlensDesc = findTensorInMap(tensorMap,
                                           attrs->cu_seqlens_tensor_uid,
                                           "CausalConv1dOperationDescriptor::fromNode: CuSeqlens");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "CausalConv1dOperationDescriptor::fromNode: Y");
    if(attrs->bias_tensor_uid)
    {
        desc->_biasDesc = findTensorInMap(
            tensorMap, *attrs->bias_tensor_uid, "CausalConv1dOperationDescriptor::fromNode: Bias");
    }
    if(attrs->initial_state_tensor_uid)
    {
        desc->_initialStateDesc
            = findTensorInMap(tensorMap,
                              *attrs->initial_state_tensor_uid,
                              "CausalConv1dOperationDescriptor::fromNode: InitialState");
    }
    if(attrs->state_indices_tensor_uid)
    {
        desc->_stateIndicesDesc
            = findTensorInMap(tensorMap,
                              *attrs->state_indices_tensor_uid,
                              "CausalConv1dOperationDescriptor::fromNode: StateIndices");
    }
    if(attrs->final_state_tensor_uid)
    {
        desc->_finalStateDesc
            = findTensorInMap(tensorMap,
                              *attrs->final_state_tensor_uid,
                              "CausalConv1dOperationDescriptor::fromNode: FinalState");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
