// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KnobSettingDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

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

    THROW_IF_FALSE(_valueSet,
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
        setKnobId(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
        setKnobValue(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobSettingDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobSettingDescriptor::setKnobId(hipdnnBackendAttributeType_t attributeType,
                                      int64_t elementCount,
                                      const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobSettingDescriptor::setAttribute(): "
                   "attributeType is not HIPDNN_TYPE_CHAR");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobSettingDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(1),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobSettingDescriptor::setAttribute(): "
                "elementCount must be > 0 (knob ID must not be empty)");
    THROW_IF_TRUE(elementCount > MAX_KNOB_ID_LENGTH,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobSettingDescriptor::setAttribute(): "
                  "elementCount exceeds MAX_KNOB_ID_LENGTH ("
                      + std::to_string(MAX_KNOB_ID_LENGTH) + ")");

    _knobId
        = std::string(static_cast<const char*>(arrayOfElements), static_cast<size_t>(elementCount));
}

void KnobSettingDescriptor::setKnobValue(hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobSettingDescriptor::setAttribute(): arrayOfElements is null");

    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        THROW_IF_NE(
            elementCount,
            1,
            HIPDNN_STATUS_BAD_PARAM,
            "KnobSettingDescriptor::setAttribute(): elementCount must be 1 for int64 value");
        hipdnn_data_sdk::data_objects::IntValueT intVal;
        int64_t tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(int64_t));
        intVal.value = tmp;
        _value.Set(intVal);
        _valueSet = true;
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        THROW_IF_NE(
            elementCount,
            1,
            HIPDNN_STATUS_BAD_PARAM,
            "KnobSettingDescriptor::setAttribute(): elementCount must be 1 for double value");
        hipdnn_data_sdk::data_objects::FloatValueT floatVal;
        double tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(double));
        floatVal.value = tmp;
        _value.Set(floatVal);
        _valueSet = true;
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        THROW_IF_LT(
            elementCount,
            static_cast<int64_t>(0),
            HIPDNN_STATUS_BAD_PARAM,
            "KnobSettingDescriptor::setAttribute(): elementCount is negative for string value");
        THROW_IF_TRUE(elementCount > MAX_KNOB_STRING_VALUE_LENGTH,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobSettingDescriptor::setAttribute(): "
                      "elementCount exceeds MAX_KNOB_STRING_VALUE_LENGTH ("
                          + std::to_string(MAX_KNOB_STRING_VALUE_LENGTH) + ")");
        hipdnn_data_sdk::data_objects::StringValueT strVal;
        strVal.value = std::string(static_cast<const char*>(arrayOfElements),
                                   static_cast<size_t>(elementCount));
        _value.Set(std::move(strVal));
        _valueSet = true;
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobSettingDescriptor::setAttribute(): "
                                          "unsupported attribute type for KNOB_CHOICE_KNOB_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
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
        getKnobId(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
        getKnobValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobSettingDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobSettingDescriptor::getKnobId(hipdnnBackendAttributeType_t attributeType,
                                      int64_t requestedElementCount,
                                      int64_t* elementCount,
                                      void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobSettingDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobSettingDescriptor::getAttribute(): requestedElementCount is negative");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobSettingDescriptor::getAttribute(): elementCount is null");
        *elementCount = static_cast<int64_t>(_knobId.size() + 1);
        return;
    }

    auto maxSize = static_cast<size_t>(requestedElementCount);
    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
        static_cast<char*>(arrayOfElements), _knobId.c_str(), maxSize);

    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(std::min(_knobId.size() + 1, maxSize));
    }
}

void KnobSettingDescriptor::getKnobValue(hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    switch(_value.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_INT64,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobSettingDescriptor::getAttribute(): type mismatch, value is IntValue");
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobSettingDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobSettingDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<int64_t*>(arrayOfElements) = _value.AsIntValue()->value;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_DOUBLE,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobSettingDescriptor::getAttribute(): type mismatch, value is FloatValue");
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobSettingDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobSettingDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<double*>(arrayOfElements) = _value.AsFloatValue()->value;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
    {
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_CHAR,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobSettingDescriptor::getAttribute(): type mismatch, value is StringValue");

        THROW_IF_LT(requestedElementCount,
                    static_cast<int64_t>(0),
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobSettingDescriptor::getAttribute(): requestedElementCount is negative");

        const auto& str = _value.AsStringValue()->value;

        // Support the two-call pattern: first call with nullptr/0 returns required size.
        if(arrayOfElements == nullptr || requestedElementCount == 0)
        {
            THROW_IF_NULL(elementCount,
                          HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                          "KnobSettingDescriptor::getAttribute(): elementCount is null");
            *elementCount = static_cast<int64_t>(str.size() + 1);
            return;
        }

        auto maxSize = static_cast<size_t>(requestedElementCount);
        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            static_cast<char*>(arrayOfElements), str.c_str(), maxSize);

        if(elementCount != nullptr)
        {
            *elementCount = static_cast<int64_t>(std::min(str.size() + 1, maxSize));
        }
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "KnobSettingDescriptor::getAttribute(): unknown value type ("
                                  + std::to_string(static_cast<int>(_value.type)) + ")");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::unique_ptr<hipdnn_data_sdk::data_objects::KnobSettingT>
    KnobSettingDescriptor::toKnobSettingT() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobSettingDescriptor::toKnobSettingT() failed: Not finalized.");

    auto knobSetting = std::make_unique<hipdnn_data_sdk::data_objects::KnobSettingT>();
    knobSetting->knob_id = _knobId;

    // Deep-copy the KnobValueUnion
    switch(_value.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
    {
        hipdnn_data_sdk::data_objects::IntValueT intVal;
        intVal.value = _value.AsIntValue()->value;
        knobSetting->value.Set(intVal);
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
    {
        hipdnn_data_sdk::data_objects::FloatValueT floatVal;
        floatVal.value = _value.AsFloatValue()->value;
        knobSetting->value.Set(floatVal);
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
    {
        hipdnn_data_sdk::data_objects::StringValueT strVal;
        strVal.value = _value.AsStringValue()->value;
        knobSetting->value.Set(std::move(strVal));
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "KnobSettingDescriptor::toKnobSettingT(): unknown value type ("
                                  + std::to_string(static_cast<int>(_value.type)) + ")");
    }

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
    str += "}";
    return str;
}

} // namespace hipdnn_backend
