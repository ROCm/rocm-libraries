// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void BatchnormOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_biasDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: BIAS tensor not set");
    THROW_IF_NULL(_epsilonDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: EPSILON tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    // Validate mean + inv_variance: both-or-none
    const bool hasMean = _meanDesc != nullptr;
    const bool hasInvVariance = _invVarianceDesc != nullptr;
    THROW_IF_TRUE(hasMean != hasInvVariance,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: mean and inverse variance "
                  "tensors must both be set or both be null");

    // Validate running stats: all-or-none
    const bool hasPrevRunMean = _prevRunningMeanDesc != nullptr;
    const bool hasPrevRunVar = _prevRunningVarianceDesc != nullptr;
    const bool hasMomentum = _momentumDesc != nullptr;
    const bool hasNextRunMean = _nextRunningMeanDesc != nullptr;
    const bool hasNextRunVar = _nextRunningVarianceDesc != nullptr;
    const bool anyRunning
        = hasPrevRunMean || hasPrevRunVar || hasMomentum || hasNextRunMean || hasNextRunVar;
    const bool allRunning
        = hasPrevRunMean && hasPrevRunVar && hasMomentum && hasNextRunMean && hasNextRunVar;
    THROW_IF_TRUE(anyRunning && !allRunning,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormOperationDescriptor::finalize() failed: running statistics tensors "
                  "(prev_running_mean, prev_running_variance, momentum, next_running_mean, "
                  "next_running_variance) must all be set or all be null");

    HipdnnBackendDescriptorImpl<BatchnormOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void BatchnormOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t elementCount,
                                                const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "BatchnormOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT:
        setTensorDescriptor(_biasDesc,
                            _data.bias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT:
        setTensorDescriptor(_epsilonDesc,
                            _data.epsilon_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT:
        setOptionalTensorDescriptor(_prevRunningMeanDesc,
                                    _data.prev_running_mean_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT:
        setOptionalTensorDescriptor(_prevRunningVarianceDesc,
                                    _data.prev_running_variance_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT:
        setOptionalTensorDescriptor(_momentumDesc,
                                    _data.momentum_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT:
        setOptionalTensorDescriptor(_meanDesc,
                                    _data.mean_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT:
        setOptionalTensorDescriptor(_invVarianceDesc,
                                    _data.inv_variance_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT:
        setOptionalTensorDescriptor(_nextRunningMeanDesc,
                                    _data.next_running_mean_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT:
        setOptionalTensorDescriptor(_nextRunningVarianceDesc,
                                    _data.next_running_variance_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT:
        setTensorDescriptorArray(_peerStatsDescs,
                                 _data.peer_stats_tensor_uid,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "BatchnormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "BatchnormOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "BatchnormOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void BatchnormOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t requestedElementCount,
                                                int64_t* elementCount,
                                                void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "BatchnormOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT:
        getTensorDescriptor(_biasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT:
        getTensorDescriptor(_epsilonDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT:
        getOptionalTensorDescriptor(_prevRunningMeanDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT:
        getOptionalTensorDescriptor(_prevRunningVarianceDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT:
        getOptionalTensorDescriptor(_momentumDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT:
        getOptionalTensorDescriptor(_meanDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT:
        getOptionalTensorDescriptor(_invVarianceDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT:
        getOptionalTensorDescriptor(_nextRunningMeanDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT:
        getOptionalTensorDescriptor(_nextRunningVarianceDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT:
        getTensorDescriptorArray(_peerStatsDescs,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BatchnormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_BATCHNORM_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "BatchnormOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "BatchnormOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    BatchnormOperationDescriptor::getTensorDescriptors() const
{
    // Required tensors always included
    std::vector<std::shared_ptr<TensorDescriptor>> result
        = {_xDesc, _scaleDesc, _biasDesc, _epsilonDesc, _yDesc};

    // Optional tensors: only include if set
    if(_meanDesc)
    {
        result.push_back(_meanDesc);
    }
    if(_invVarianceDesc)
    {
        result.push_back(_invVarianceDesc);
    }
    if(_prevRunningMeanDesc)
    {
        result.push_back(_prevRunningMeanDesc);
    }
    if(_prevRunningVarianceDesc)
    {
        result.push_back(_prevRunningVarianceDesc);
    }
    if(_momentumDesc)
    {
        result.push_back(_momentumDesc);
    }
    if(_nextRunningMeanDesc)
    {
        result.push_back(_nextRunningMeanDesc);
    }
    if(_nextRunningVarianceDesc)
    {
        result.push_back(_nextRunningVarianceDesc);
    }

    result.insert(result.end(), _peerStatsDescs.begin(), _peerStatsDescs.end());
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    BatchnormOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BatchnormOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT;
}

std::string BatchnormOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "BatchnormOperationDescriptor: {";
    str += "name=" + _name + ", ";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", bias_uid=" + std::to_string(_data.bias_tensor_uid);
    str += ", epsilon_uid=" + std::to_string(_data.epsilon_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", mean_uid=";
    str += _data.mean_tensor_uid.has_value() ? std::to_string(_data.mean_tensor_uid.value())
                                             : "nullopt";
    str += ", inv_variance_uid=";
    str += _data.inv_variance_tensor_uid.has_value()
               ? std::to_string(_data.inv_variance_tensor_uid.value())
               : "nullopt";
    str += ", prev_running_mean_uid=";
    str += _data.prev_running_mean_tensor_uid.has_value()
               ? std::to_string(_data.prev_running_mean_tensor_uid.value())
               : "nullopt";
    str += ", prev_running_variance_uid=";
    str += _data.prev_running_variance_tensor_uid.has_value()
               ? std::to_string(_data.prev_running_variance_tensor_uid.value())
               : "nullopt";
    str += ", momentum_uid=";
    str += _data.momentum_tensor_uid.has_value() ? std::to_string(_data.momentum_tensor_uid.value())
                                                 : "nullopt";
    str += ", next_running_mean_uid=";
    str += _data.next_running_mean_tensor_uid.has_value()
               ? std::to_string(_data.next_running_mean_tensor_uid.value())
               : "nullopt";
    str += ", next_running_variance_uid=";
    str += _data.next_running_variance_tensor_uid.has_value()
               ? std::to_string(_data.next_running_variance_tensor_uid.value())
               : "nullopt";
    str += ", peer_stats_uids=" + vecToString(_data.peer_stats_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += '}';
    return str;
}

std::shared_ptr<BatchnormOperationDescriptor> BatchnormOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsBatchnormAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "BatchnormOperationDescriptor::fromNode: BatchnormAttributes is null");

    auto desc = std::make_shared<BatchnormOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "BatchnormOperationDescriptor::fromNode: X");
    desc->_scaleDesc = findTensorInMap(
        tensorMap, attrs->scale_tensor_uid, "BatchnormOperationDescriptor::fromNode: Scale");
    desc->_biasDesc = findTensorInMap(
        tensorMap, attrs->bias_tensor_uid, "BatchnormOperationDescriptor::fromNode: Bias");
    desc->_epsilonDesc = findTensorInMap(
        tensorMap, attrs->epsilon_tensor_uid, "BatchnormOperationDescriptor::fromNode: Epsilon");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "BatchnormOperationDescriptor::fromNode: Y");
    if(attrs->prev_running_mean_tensor_uid)
    {
        desc->_prevRunningMeanDesc
            = findTensorInMap(tensorMap,
                              *attrs->prev_running_mean_tensor_uid,
                              "BatchnormOperationDescriptor::fromNode: PrevRunningMean");
    }
    if(attrs->prev_running_variance_tensor_uid)
    {
        desc->_prevRunningVarianceDesc
            = findTensorInMap(tensorMap,
                              *attrs->prev_running_variance_tensor_uid,
                              "BatchnormOperationDescriptor::fromNode: PrevRunningVariance");
    }
    if(attrs->momentum_tensor_uid)
    {
        desc->_momentumDesc = findTensorInMap(tensorMap,
                                              *attrs->momentum_tensor_uid,
                                              "BatchnormOperationDescriptor::fromNode: Momentum");
    }
    if(attrs->mean_tensor_uid)
    {
        desc->_meanDesc = findTensorInMap(
            tensorMap, *attrs->mean_tensor_uid, "BatchnormOperationDescriptor::fromNode: Mean");
    }
    if(attrs->inv_variance_tensor_uid)
    {
        desc->_invVarianceDesc
            = findTensorInMap(tensorMap,
                              *attrs->inv_variance_tensor_uid,
                              "BatchnormOperationDescriptor::fromNode: InvVariance");
    }
    if(attrs->next_running_mean_tensor_uid)
    {
        desc->_nextRunningMeanDesc
            = findTensorInMap(tensorMap,
                              *attrs->next_running_mean_tensor_uid,
                              "BatchnormOperationDescriptor::fromNode: NextRunningMean");
    }
    if(attrs->next_running_variance_tensor_uid)
    {
        desc->_nextRunningVarianceDesc
            = findTensorInMap(tensorMap,
                              *attrs->next_running_variance_tensor_uid,
                              "BatchnormOperationDescriptor::fromNode: NextRunningVariance");
    }
    for(auto uid : attrs->peer_stats_tensor_uid)
    {
        desc->_peerStatsDescs.push_back(
            findTensorInMap(tensorMap, uid, "BatchnormOperationDescriptor::fromNode: peer_stats"));
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
