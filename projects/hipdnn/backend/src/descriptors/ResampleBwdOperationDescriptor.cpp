// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ResampleBwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void ResampleBwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: DY tensor not set");
    THROW_IF_NULL(_dxDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: DX tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.window.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: window not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.resample_mode
                      == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: resample_mode not set");
    THROW_IF_TRUE(_data.padding_mode
                      == hipdnn_flatbuffers_sdk::data_objects::PaddingMode::PADDING_NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ResampleBwdOperationDescriptor::finalize() failed: padding_mode not set");

    HipdnnBackendDescriptorImpl<ResampleBwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ResampleBwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                  hipdnnBackendAttributeType_t attributeType,
                                                  int64_t elementCount,
                                                  const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ResampleBwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX:
        setOptionalTensorDescriptor(_indexDesc,
                                    _data.index_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS:
        setScalarVector<int64_t>(_data.pre_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_POST_PADDINGS:
        setScalarVector<int64_t>(_data.post_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_STRIDES:
        setScalarVector<int64_t>(_data.stride,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS:
        setScalarVector<int64_t>(_data.window,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_MODE:
        setResampleMode(_data.resample_mode,
                        attributeType,
                        elementCount,
                        arrayOfElements,
                        "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PADDING_MODE:
        setPaddingMode(_data.padding_mode,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT:
        setScalar(_data.generate_index,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "ResampleBwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ResampleBwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void ResampleBwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                  hipdnnBackendAttributeType_t attributeType,
                                                  int64_t requestedElementCount,
                                                  int64_t* elementCount,
                                                  void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ResampleBwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX:
        getOptionalTensorDescriptor(_indexDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS:
        getScalarVector<int64_t>(_data.pre_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_POST_PADDINGS:
        getScalarVector<int64_t>(_data.post_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_STRIDES:
        getScalarVector<int64_t>(_data.stride,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS:
        getScalarVector<int64_t>(_data.window,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_MODE:
        getResampleMode(_data.resample_mode,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_PADDING_MODE:
        getPaddingMode(_data.padding_mode,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT:
        getScalar(_data.generate_index,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RESAMPLE_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_RESAMPLE_BWD,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "ResampleBwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ResampleBwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    ResampleBwdOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result = {_dyDesc, _dxDesc};
    if(_indexDesc)
    {
        result.push_back(_indexDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    ResampleBwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t ResampleBwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
}

std::string ResampleBwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ResampleBwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", dx_uid=" + std::to_string(_data.dx_tensor_uid);
    str += ", index_uid="
           + (_data.index_tensor_uid ? std::to_string(*_data.index_tensor_uid) : "nullopt");
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", window=" + vecToString(_data.window);
    str += ", resample_mode="
           + std::string(
               hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(_data.resample_mode));
    str += ", padding_mode="
           + std::string(
               hipdnn_flatbuffers_sdk::data_objects::EnumNamePaddingMode(_data.padding_mode));
    str += ", generate_index=" + std::string(_data.generate_index ? "true" : "false");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<ResampleBwdOperationDescriptor> ResampleBwdOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsResampleBwdAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ResampleBwdOperationDescriptor::fromNode: ResampleBwdAttributes is null");

    auto desc = std::make_shared<ResampleBwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "ResampleBwdOperationDescriptor::fromNode: Dy");
    desc->_dxDesc = findTensorInMap(
        tensorMap, attrs->dx_tensor_uid, "ResampleBwdOperationDescriptor::fromNode: Dx");
    if(attrs->index_tensor_uid)
    {
        desc->_indexDesc = findTensorInMap(
            tensorMap, *attrs->index_tensor_uid, "ResampleBwdOperationDescriptor::fromNode: Index");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
