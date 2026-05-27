// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ResampleFwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void ResampleFwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.window.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: window not set");
    THROW_IF_TRUE(_data.resample_mode
                      == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: resample_mode not set");
    THROW_IF_TRUE(_data.padding_mode
                      == hipdnn_flatbuffers_sdk::data_objects::PaddingMode::PADDING_NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: padding_mode not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleFwdOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    HipdnnBackendDescriptorImpl<ResampleFwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ResampleFwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                  hipdnnBackendAttributeType_t attributeType,
                                                  int64_t elementCount,
                                                  const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ResampleFwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC:
        setOptionalTensorDescriptor(_indexDesc,
                                    _data.index_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS:
        setScalarVector<int64_t>(_data.pre_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_POST_PADDINGS:
        setScalarVector<int64_t>(_data.post_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_STRIDES:
        setScalarVector<int64_t>(_data.stride,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS:
        setScalarVector<int64_t>(_data.window,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_MODE:
        setResampleMode(_data.resample_mode,
                        attributeType,
                        elementCount,
                        arrayOfElements,
                        "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PADDING_MODE:
        setPaddingMode(_data.padding_mode,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT:
        setOptionalScalar<HIPDNN_TYPE_BOOLEAN>(_data.generate_index,
                                               attributeType,
                                               elementCount,
                                               arrayOfElements,
                                               "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "ResampleFwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ResampleFwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void ResampleFwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                  hipdnnBackendAttributeType_t attributeType,
                                                  int64_t requestedElementCount,
                                                  int64_t* elementCount,
                                                  void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ResampleFwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC:
        getOptionalTensorDescriptor(_indexDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS:
        getScalarVector<int64_t>(_data.pre_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_POST_PADDINGS:
        getScalarVector<int64_t>(_data.post_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_STRIDES:
        getScalarVector<int64_t>(_data.stride,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS:
        getScalarVector<int64_t>(_data.window,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_MODE:
        getResampleMode(_data.resample_mode,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PADDING_MODE:
        getPaddingMode(_data.padding_mode,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT:
        getOptionalScalar<HIPDNN_TYPE_BOOLEAN>(_data.generate_index,
                                               attributeType,
                                               requestedElementCount,
                                               elementCount,
                                               arrayOfElements,
                                               "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_RESAMPLE_FWD,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "ResampleFwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ResampleFwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    ResampleFwdOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result = {_xDesc, _yDesc};
    if(_indexDesc)
    {
        result.push_back(_indexDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    ResampleFwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t ResampleFwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
}

std::string ResampleFwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ResampleFwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", index_uid="
           + (_data.index_tensor_uid ? std::to_string(*_data.index_tensor_uid) : "nullopt");
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", window=" + vecToString(_data.window);
    str += ", resample_mode=" + std::to_string(static_cast<int>(_data.resample_mode));
    str += ", padding_mode=" + std::to_string(static_cast<int>(_data.padding_mode));
    str += ", generate_index="
           + (_data.generate_index ? std::to_string(static_cast<int>(*_data.generate_index))
                                   : "nullopt");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<ResampleFwdOperationDescriptor> ResampleFwdOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsResampleFwdAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ResampleFwdOperationDescriptor::fromNode: ResampleFwdAttributes is null");

    auto desc = std::make_shared<ResampleFwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "ResampleFwdOperationDescriptor::fromNode: X");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->y_tensor_uid, "ResampleFwdOperationDescriptor::fromNode: Y");
    if(attrs->index_tensor_uid)
    {
        desc->_indexDesc = findTensorInMap(
            tensorMap, *attrs->index_tensor_uid, "ResampleFwdOperationDescriptor::fromNode: Index");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
