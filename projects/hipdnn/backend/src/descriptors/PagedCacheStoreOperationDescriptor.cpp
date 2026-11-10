// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "PagedCacheStoreOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void PagedCacheStoreOperationDescriptor::finalize()
{
    THROW_IF_NULL(
        _xDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "PagedCacheStoreOperationDescriptor::finalize() failed: XDESC_EXT tensor not set");
    THROW_IF_NULL(_slotMappingDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "PagedCacheStoreOperationDescriptor::finalize() failed: SLOT_MAPPING_DESC_EXT "
                  "tensor not set");
    THROW_IF_NULL(
        _containerDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "PagedCacheStoreOperationDescriptor::finalize() failed: CONTAINER_DESC_EXT tensor not set");

    HipdnnBackendDescriptorImpl<PagedCacheStoreOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void PagedCacheStoreOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                      hipdnnBackendAttributeType_t attributeType,
                                                      int64_t elementCount,
                                                      const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "PagedCacheStoreOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PagedCacheStoreOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT:
        setTensorDescriptor(_slotMappingDesc,
                            _data.slot_mapping_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PagedCacheStoreOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT:
        setTensorDescriptor(_containerDesc,
                            _data.container_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "PagedCacheStoreOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT:
        setOptionalTensorDescriptor(_scaleDesc,
                                    _data.scale_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "PagedCacheStoreOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "PagedCacheStoreOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PagedCacheStoreOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void PagedCacheStoreOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                      hipdnnBackendAttributeType_t attributeType,
                                                      int64_t requestedElementCount,
                                                      int64_t* elementCount,
                                                      void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "PagedCacheStoreOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PagedCacheStoreOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT:
        getTensorDescriptor(_slotMappingDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PagedCacheStoreOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT:
        getTensorDescriptor(_containerDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "PagedCacheStoreOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT:
        getOptionalTensorDescriptor(_scaleDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "PagedCacheStoreOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "PagedCacheStoreOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_PAGED_CACHE_STORE_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "PagedCacheStoreOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "PagedCacheStoreOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    PagedCacheStoreOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(4);
    result.push_back(_xDesc);
    result.push_back(_slotMappingDesc);
    result.push_back(_containerDesc);
    if(_scaleDesc)
    {
        result.push_back(_scaleDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    PagedCacheStoreOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::PagedCacheStoreAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t PagedCacheStoreOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_PAGED_CACHE_STORE_DESCRIPTOR_EXT;
}

std::string PagedCacheStoreOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "PagedCacheStoreOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", slot_mapping_uid=" + std::to_string(_data.slot_mapping_tensor_uid);
    str += ", container_uid=" + std::to_string(_data.container_tensor_uid);
    str += ", scale_uid="
           + (_data.scale_tensor_uid ? std::to_string(*_data.scale_tensor_uid) : "nullopt");
    str += "}";
    return str;
}

std::shared_ptr<PagedCacheStoreOperationDescriptor> PagedCacheStoreOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsPagedCacheStoreAttributes();
    THROW_IF_NULL(
        attrs,
        HIPDNN_STATUS_INTERNAL_ERROR,
        "PagedCacheStoreOperationDescriptor::fromNode: PagedCacheStoreAttributes is null");

    auto desc = std::make_shared<PagedCacheStoreOperationDescriptor>();
    desc->_data = *attrs;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "PagedCacheStoreOperationDescriptor::fromNode: X");
    desc->_slotMappingDesc
        = findTensorInMap(tensorMap,
                          attrs->slot_mapping_tensor_uid,
                          "PagedCacheStoreOperationDescriptor::fromNode: SlotMapping");
    desc->_containerDesc
        = findTensorInMap(tensorMap,
                          attrs->container_tensor_uid,
                          "PagedCacheStoreOperationDescriptor::fromNode: Container");
    if(attrs->scale_tensor_uid)
    {
        desc->_scaleDesc = findTensorInMap(tensorMap,
                                           *attrs->scale_tensor_uid,
                                           "PagedCacheStoreOperationDescriptor::fromNode: Scale");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
