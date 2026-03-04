// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormBackwardOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
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
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "BatchnormBackwardOperationDescriptor::finalize() failed: compute data type not "
                  "set");

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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_SCALE:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DX:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DSCALE:
        setTensorDescriptor(_dscaleDesc,
                            _data.dscale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DBIAS:
        setTensorDescriptor(_dbiasDesc,
                            _data.dbias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_MEAN:
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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_INV_VARIANCE:
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
    case HIPDNN_ATTR_BATCHNORM_BWD_EXT_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "BatchnormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BWD_EXT_PEER_STATS:
        setTensorDescriptorArray(_peerStatsDescs,
                                 _data.peer_stats_tensor_uid,
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
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_SCALE:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DX:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DSCALE:
        getTensorDescriptor(_dscaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_DBIAS:
        getTensorDescriptor(_dbiasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_MEAN:
        getTensorDescriptor(_meanDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_EXT_INV_VARIANCE:
        getTensorDescriptor(_invVarianceDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_BATCHNORM_BWD_EXT_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "BatchnormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_BATCHNORM_BWD_EXT_PEER_STATS:
        getTensorDescriptorArray(_peerStatsDescs,
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
    if(_meanDesc)
    {
        result.push_back(_meanDesc);
    }
    if(_invVarianceDesc)
    {
        result.push_back(_invVarianceDesc);
    }
    result.insert(result.end(), _peerStatsDescs.begin(), _peerStatsDescs.end());
    return result;
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    BatchnormBackwardOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::BatchnormBackwardAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t BatchnormBackwardOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_EXT_DESCRIPTOR;
}

std::string BatchnormBackwardOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "BatchnormBackwardOperationDescriptor: {";
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
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
