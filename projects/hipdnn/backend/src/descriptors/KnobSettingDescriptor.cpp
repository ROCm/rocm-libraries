// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KnobSettingDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

void KnobSettingDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobSettingDescriptor::finalize() failed: Already finalized.");

    THROW_IF_TRUE(_knobId.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobSettingDescriptor::finalize() failed: Knob ID is not set.");

    THROW_IF_TRUE(_value.type == hipdnn_flatbuffers_sdk::data_objects::KnobValue::NONE,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobSettingDescriptor::finalize() failed: Value is not set.");

    HipdnnBackendDescriptorImpl<KnobSettingDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void KnobSettingDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                         hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "KnobSettingDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE:
        setBoundedString(_knobId,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         "KnobSettingDescriptor::setAttribute()",
                         MAX_KNOB_ID_LENGTH,
                         1);
        break;
    case HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
        setKnobValueUnion(_value,
                          attributeType,
                          elementCount,
                          arrayOfElements,
                          "KnobSettingDescriptor::setAttribute()",
                          MAX_KNOB_STRING_VALUE_LENGTH);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobSettingDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void KnobSettingDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                         hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobSettingDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE:
        getString(_knobId,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobSettingDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
        getKnobValueUnion(_value,
                          attributeType,
                          requestedElementCount,
                          elementCount,
                          arrayOfElements,
                          "KnobSettingDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobSettingDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>
    KnobSettingDescriptor::toKnobSettingT() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobSettingDescriptor::toKnobSettingT() failed: Not finalized.");

    auto knobSetting = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>();
    knobSetting->knob_id = _knobId;

    copyKnobValueUnion(_value, knobSetting->value, "KnobSettingDescriptor::toKnobSettingT()");

    return knobSetting;
}

hipdnnBackendDescriptorType_t KnobSettingDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
}

std::string KnobSettingDescriptor::toString() const
{
    std::string str = "KnobSettingDescriptor: {knobId=" + _knobId;
    str += ", valueType=" + std::to_string(static_cast<int>(_value.type));
    str += '}';
    return str;
}

} // namespace hipdnn_backend
