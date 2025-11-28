// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BackendDescriptor.hpp"

void HipdnnBackendDescriptor::finalize()
{
    THROW_IF_TRUE(!_impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null _impl in finalize.");
    _impl->finalize();
}

bool HipdnnBackendDescriptor::isFinalized() const
{
    THROW_IF_TRUE(!_impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null _impl in isFinalized.");
    return _impl->isFinalized();
}

void HipdnnBackendDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements) const
{
    THROW_IF_TRUE(!_impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null _impl in getAttribute.");
    _impl->getAttribute(
        attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
}

void HipdnnBackendDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_TRUE(!_impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null _impl in setAttribute.");
    _impl->setAttribute(attributeName, attributeType, elementCount, arrayOfElements);
}

hipdnnBackendDescriptorType_t HipdnnBackendDescriptor::getType() const
{
    THROW_IF_TRUE(!_impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null _impl in getType.");
    return _impl->getType();
}

bool HipdnnBackendDescriptor::isValid()
{
    return _impl && getType() != hipdnnBackendDescriptorType_t::HIPDNN_INVALID_TYPE;
}

bool HipdnnBackendDescriptor::operator==(const HipdnnBackendDescriptor& other) const
{
    return _impl == other._impl;
}
