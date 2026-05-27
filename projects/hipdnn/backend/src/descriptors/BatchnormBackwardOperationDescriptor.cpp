// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormBackwardOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void BatchnormBackwardOperationDescriptor::finalize()
{
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: DY tensor not set");
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_dxDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: DX tensor not set");
    THROW_IF_NULL(_dscaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: DSCALE tensor not set");
    THROW_IF_NULL(_dbiasDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: DBIAS tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    const bool hasMean = _meanDesc != nullptr;
    const bool hasInvVariance = _invVarianceDesc != nullptr;
    THROW_IF_TRUE(
        hasMean != hasInvVariance,
        HIPDNN_STATUS_BAD_PARAM,
        "BatchnormBackwardOperationDescriptor::finalize() failed: mean and inverse variance "
        "tensors must both be set or both be null");

    HipdnnBackendDescriptorImpl<BatchnormBackwardOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void BatchnormBackwardOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                        hipdnnBackendAttributeType_t attributeType,
                                                        int64_t elementCount,
                                                        const void* arrayOfElements)
{
    THROW_IF_TRUE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "BatchnormBackwardOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT:
        setTensorDescriptor(_dscaleDesc,
                            _data.dscale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT:
        setTensorDescriptor(_dbiasDesc,
                            _data.dbias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_meanDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        _data.mean_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_invVarianceDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        _data.inv_variance_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT:
        setTensorDescriptorArray(_peerStatsDescs,
                                 _data.peer_stats_tensor_uid,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BatchnormBackwardOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void BatchnormBackwardOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                        hipdnnBackendAttributeType_t attributeType,
                                                        int64_t requestedElementCount,
                                                        int64_t* elementCount,
                                                        void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "BatchnormBackwardOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT:
        getTensorDescriptor(_dscaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT:
        getTensorDescriptor(_dbiasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT:
        getOptionalTensorDescriptor(_meanDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT:
        getOptionalTensorDescriptor(_invVarianceDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT:
        getTensorDescriptorArray(_peerStatsDescs,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_BATCHNORM_BACKWARD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "BatchnormBackwardOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    BatchnormBackwardOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result
        = {_dyDesc, _xDesc, _scaleDesc, _dxDesc, _dscaleDesc, _dbiasDesc};

    if(_meanDesc && _invVarianceDesc)
    {
        result.push_back(_meanDesc);
        result.push_back(_invVarianceDesc);
    }

    result.insert(result.end(), _peerStatsDescs.begin(), _peerStatsDescs.end());
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    BatchnormBackwardOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BatchnormBackwardOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT;
}

std::string BatchnormBackwardOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "BatchnormBackwardOperationDescriptor: {";
    str += "name=" + _name + ", ";
    str += "dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", dx_uid=" + std::to_string(_data.dx_tensor_uid);
    str += ", dscale_uid=" + std::to_string(_data.dscale_tensor_uid);
    str += ", dbias_uid=" + std::to_string(_data.dbias_tensor_uid);
    str += ", mean_uid=";
    str += _data.mean_tensor_uid.has_value() ? std::to_string(_data.mean_tensor_uid.value())
                                             : "nullopt";
    str += ", inv_variance_uid=";
    str += _data.inv_variance_tensor_uid.has_value()
               ? std::to_string(_data.inv_variance_tensor_uid.value())
               : "nullopt";
    str += ", peer_stats_uids=" + vecToString(_data.peer_stats_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<BatchnormBackwardOperationDescriptor>
    BatchnormBackwardOperationDescriptor::fromNode(
        const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
        const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsBatchnormBackwardAttributes();
    THROW_IF_NULL(
        attrs,
        HIPDNN_STATUS_INTERNAL_ERROR,
        "BatchnormBackwardOperationDescriptor::fromNode: BatchnormBackwardAttributes is null");

    auto desc = std::make_shared<BatchnormBackwardOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "BatchnormBackwardOperationDescriptor::fromNode: DY");
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "BatchnormBackwardOperationDescriptor::fromNode: X");
    desc->_scaleDesc = findTensorInMap(tensorMap,
                                       attrs->scale_tensor_uid,
                                       "BatchnormBackwardOperationDescriptor::fromNode: Scale");
    desc->_dxDesc = findTensorInMap(
        tensorMap, attrs->dx_tensor_uid, "BatchnormBackwardOperationDescriptor::fromNode: DX");
    desc->_dscaleDesc = findTensorInMap(tensorMap,
                                        attrs->dscale_tensor_uid,
                                        "BatchnormBackwardOperationDescriptor::fromNode: DScale");
    desc->_dbiasDesc = findTensorInMap(tensorMap,
                                       attrs->dbias_tensor_uid,
                                       "BatchnormBackwardOperationDescriptor::fromNode: DBias");
    if(attrs->mean_tensor_uid)
    {
        desc->_meanDesc = findTensorInMap(tensorMap,
                                          *attrs->mean_tensor_uid,
                                          "BatchnormBackwardOperationDescriptor::fromNode: Mean");
    }
    if(attrs->inv_variance_tensor_uid)
    {
        desc->_invVarianceDesc
            = findTensorInMap(tensorMap,
                              *attrs->inv_variance_tensor_uid,
                              "BatchnormBackwardOperationDescriptor::fromNode: InvVariance");
    }
    for(auto uid : attrs->peer_stats_tensor_uid)
    {
        desc->_peerStatsDescs.push_back(findTensorInMap(
            tensorMap, uid, "BatchnormBackwardOperationDescriptor::fromNode: peer_stats"));
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
