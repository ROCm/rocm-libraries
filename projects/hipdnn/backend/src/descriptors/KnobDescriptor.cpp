// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KnobDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

// ============================================================================
// finalize
// ============================================================================

void KnobDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::finalize() failed: Already finalized.");

    THROW_IF_TRUE(_knobId.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::finalize() failed: Knob ID is not set.");

    THROW_IF_FALSE(_defaultValueSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::finalize() failed: Default value is not set.");

    if(_minValueInt.has_value() && _maxValueInt.has_value())
    {
        THROW_IF_TRUE(*_minValueInt > *_maxValueInt,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "MINIMUM_VALUE (INT64) > MAXIMUM_VALUE (INT64).");
    }
    if(_minValueDouble.has_value() && _maxValueDouble.has_value())
    {
        THROW_IF_TRUE(*_minValueDouble > *_maxValueDouble,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "MINIMUM_VALUE (DOUBLE) > MAXIMUM_VALUE (DOUBLE).");
    }
    if(_stride.has_value())
    {
        THROW_IF_TRUE(*_stride <= 0,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: STRIDE must be positive (> 0).");
    }

    HipdnnBackendDescriptorImpl<KnobDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void KnobDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                  hipdnnBackendAttributeType_t attributeType,
                                  int64_t elementCount,
                                  const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "KnobDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_KNOB_INFO_TYPE:
        setKnobId(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        setMaximumValue(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        setMinimumValue(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRIDE:
        setStride(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
        setDescription(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
        setDefaultValue(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEPRECATED:
        setDeprecated(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT:
        setValidValuesInt(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING:
        setValidValuesString(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH:
        setStringMaxLength(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobDescriptor::setKnobId(hipdnnBackendAttributeType_t attributeType,
                               int64_t elementCount,
                               const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(1),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): "
                "elementCount must be > 0 (knob ID must not be empty)");
    THROW_IF_TRUE(elementCount > MAX_KNOB_ID_LENGTH,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::setAttribute(): "
                  "elementCount exceeds MAX_KNOB_ID_LENGTH ("
                      + std::to_string(MAX_KNOB_ID_LENGTH) + ")");

    _knobId
        = std::string(static_cast<const char*>(arrayOfElements), static_cast<size_t>(elementCount));
}

void KnobDescriptor::setDescription(hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount is negative");
    THROW_IF_TRUE(elementCount > MAX_DESCRIPTION_LENGTH,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::setAttribute(): "
                  "elementCount exceeds MAX_DESCRIPTION_LENGTH ("
                      + std::to_string(MAX_DESCRIPTION_LENGTH) + ")");

    if(elementCount == 0)
    {
        _description.clear();
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");

    _description
        = std::string(static_cast<const char*>(arrayOfElements), static_cast<size_t>(elementCount));
}

void KnobDescriptor::setDefaultValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t elementCount,
                                     const void* arrayOfElements)
{
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");

    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        THROW_IF_NE(elementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::setAttribute(): elementCount must be 1 for int64 default");
        hipdnn_data_sdk::data_objects::IntValueT intVal;
        int64_t tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(int64_t));
        intVal.value = tmp;
        _defaultValue.Set(intVal);
        _defaultValueSet = true;
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        THROW_IF_NE(elementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::setAttribute(): elementCount must be 1 for double default");
        hipdnn_data_sdk::data_objects::FloatValueT floatVal;
        double tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(double));
        floatVal.value = tmp;
        _defaultValue.Set(floatVal);
        _defaultValueSet = true;
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        THROW_IF_LT(elementCount,
                    static_cast<int64_t>(0),
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::setAttribute(): elementCount is negative for string default");
        THROW_IF_TRUE(elementCount > MAX_STRING_VALUE_LENGTH,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::setAttribute(): "
                      "elementCount exceeds MAX_STRING_VALUE_LENGTH ("
                          + std::to_string(MAX_STRING_VALUE_LENGTH) + ")");
        hipdnn_data_sdk::data_objects::StringValueT strVal;
        strVal.value = std::string(static_cast<const char*>(arrayOfElements),
                                   static_cast<size_t>(elementCount));
        _defaultValue.Set(std::move(strVal));
        _defaultValueSet = true;
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobDescriptor::setAttribute(): "
                                          "unsupported attribute type for DEFAULT_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void KnobDescriptor::setMaximumValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t elementCount,
                                     const void* arrayOfElements)
{
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount must be 1 for MAXIMUM_VALUE");

    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        int64_t tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(int64_t));
        _maxValueInt = tmp;
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        double tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(double));
        _maxValueDouble = tmp;
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobDescriptor::setAttribute(): "
                                          "unsupported attribute type for MAXIMUM_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void KnobDescriptor::setMinimumValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t elementCount,
                                     const void* arrayOfElements)
{
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount must be 1 for MINIMUM_VALUE");

    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        int64_t tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(int64_t));
        _minValueInt = tmp;
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        double tmp;
        std::memcpy(&tmp, arrayOfElements, sizeof(double));
        _minValueDouble = tmp;
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobDescriptor::setAttribute(): "
                                          "unsupported attribute type for MINIMUM_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void KnobDescriptor::setStride(hipdnnBackendAttributeType_t attributeType,
                               int64_t elementCount,
                               const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount must be 1 for STRIDE");

    int64_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(int64_t));
    _stride = tmp;
}

void KnobDescriptor::setDeprecated(hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BOOLEAN,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_BOOLEAN");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount must be 1 for DEPRECATED");

    bool tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(bool));
    _deprecated = tmp;
}

void KnobDescriptor::setValidValuesInt(hipdnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount is negative");

    if(elementCount == 0)
    {
        _validValuesInt.clear();
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");

    auto* values = static_cast<const int64_t*>(arrayOfElements);
    _validValuesInt.assign(values, values + static_cast<size_t>(elementCount));
}

void KnobDescriptor::setValidValuesString(hipdnnBackendAttributeType_t attributeType,
                                          int64_t elementCount,
                                          const void* arrayOfElements)
{
    // Multi-call append pattern: each call with elementCount=N appends one string of length N
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount is negative");

    if(elementCount == 0 || arrayOfElements == nullptr)
    {
        // Append an empty string
        _validValuesString.emplace_back();
        return;
    }

    _validValuesString.emplace_back(static_cast<const char*>(arrayOfElements),
                                    static_cast<size_t>(elementCount));
}

void KnobDescriptor::setStringMaxLength(hipdnnBackendAttributeType_t attributeType,
                                        int64_t elementCount,
                                        const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount must be 1 for STRING_MAX_LENGTH");

    int64_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(int64_t));
    _stringMaxLength = tmp;
}

// ============================================================================
// getAttribute
// ============================================================================

void KnobDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                  hipdnnBackendAttributeType_t attributeType,
                                  int64_t requestedElementCount,
                                  int64_t* elementCount,
                                  void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_KNOB_INFO_TYPE:
        getKnobId(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        getMaximumValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        getMinimumValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRIDE:
        getStride(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
        getDescription(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
        getDefaultValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEPRECATED:
        getDeprecated(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT:
        getValidValuesInt(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING:
        getValidValuesString(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH:
        getStringMaxLength(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE:
        getDefaultValueType(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobDescriptor::getKnobId(hipdnnBackendAttributeType_t attributeType,
                               int64_t requestedElementCount,
                               int64_t* elementCount,
                               void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount is negative");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
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

void KnobDescriptor::getDescription(hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount is negative");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
        *elementCount = static_cast<int64_t>(_description.size() + 1);
        return;
    }

    auto maxSize = static_cast<size_t>(requestedElementCount);
    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
        static_cast<char*>(arrayOfElements), _description.c_str(), maxSize);

    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(std::min(_description.size() + 1, maxSize));
    }
}

void KnobDescriptor::getDefaultValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    switch(_defaultValue.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_INT64,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): type mismatch, default is IntValue");
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<int64_t*>(arrayOfElements) = _defaultValue.AsIntValue()->value;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;

    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_DOUBLE,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): type mismatch, default is FloatValue");
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<double*>(arrayOfElements) = _defaultValue.AsFloatValue()->value;
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
                    "KnobDescriptor::getAttribute(): type mismatch, default is StringValue");

        THROW_IF_LT(requestedElementCount,
                    static_cast<int64_t>(0),
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount is negative");

        const auto& str = _defaultValue.AsStringValue()->value;

        if(arrayOfElements == nullptr || requestedElementCount == 0)
        {
            THROW_IF_NULL(elementCount,
                          HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                          "KnobDescriptor::getAttribute(): elementCount is null");
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
                              "KnobDescriptor::getAttribute(): unknown default value type ("
                                  + std::to_string(static_cast<int>(_defaultValue.type)) + ")");
    }
}

void KnobDescriptor::getMaximumValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        if(!_maxValueInt.has_value())
        {
            if(elementCount != nullptr)
            {
                *elementCount = 0;
            }
            return;
        }
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<int64_t*>(arrayOfElements) = *_maxValueInt;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        if(!_maxValueDouble.has_value())
        {
            if(elementCount != nullptr)
            {
                *elementCount = 0;
            }
            return;
        }
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<double*>(arrayOfElements) = *_maxValueDouble;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobDescriptor::getAttribute(): "
                                          "unsupported attribute type for MAXIMUM_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void KnobDescriptor::getMinimumValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        if(!_minValueInt.has_value())
        {
            if(elementCount != nullptr)
            {
                *elementCount = 0;
            }
            return;
        }
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<int64_t*>(arrayOfElements) = *_minValueInt;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        if(!_minValueDouble.has_value())
        {
            if(elementCount != nullptr)
            {
                *elementCount = 0;
            }
            return;
        }
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobDescriptor::getAttribute(): requestedElementCount must be 1");
        *static_cast<double*>(arrayOfElements) = *_minValueDouble;
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobDescriptor::getAttribute(): "
                                          "unsupported attribute type for MINIMUM_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void KnobDescriptor::getStride(hipdnnBackendAttributeType_t attributeType,
                               int64_t requestedElementCount,
                               int64_t* elementCount,
                               void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");

    if(!_stride.has_value())
    {
        if(elementCount != nullptr)
        {
            *elementCount = 0;
        }
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::getAttribute(): arrayOfElements is null");
    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount must be 1");

    *static_cast<int64_t*>(arrayOfElements) = *_stride;
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void KnobDescriptor::getDeprecated(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BOOLEAN,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_BOOLEAN");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::getAttribute(): arrayOfElements is null");
    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount must be 1");

    *static_cast<bool*>(arrayOfElements) = _deprecated;
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void KnobDescriptor::getValidValuesInt(hipdnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount is negative");

    const auto count = static_cast<int64_t>(_validValuesInt.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
        *elementCount = count;
        return;
    }

    auto copyCount = std::min(requestedElementCount, count);
    std::memcpy(
        arrayOfElements, _validValuesInt.data(), static_cast<size_t>(copyCount) * sizeof(int64_t));

    if(elementCount != nullptr)
    {
        *elementCount = copyCount;
    }
}

void KnobDescriptor::getValidValuesString(hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    // Three-step retrieval protocol:
    //   Step 1 (total count):  requestedElementCount=0, arrayOfElements=nullptr
    //                          → returns total number of strings in elementCount
    //   Step 2 (size query):   requestedElementCount=N (1-based index, N>0), arrayOfElements=nullptr
    //                          → returns byte length of string[N-1] (incl. null) in elementCount
    //                            stores N-1 as the pending copy index
    //   Step 3 (copy):         requestedElementCount=bufferSize, arrayOfElements=buffer
    //                          → copies string from pending index into buffer;
    //                            requestedElementCount is now the caller's buffer capacity,
    //                            enabling safe bounded copy via copyMaxSizeWithNullTerminator.
    //
    // Callers MUST perform step 2 immediately before step 3 for each string.

    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount is negative");

    const auto totalCount = static_cast<int64_t>(_validValuesString.size());

    // Step 1: requestedElementCount == 0 → return total number of strings
    if(requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
        *elementCount = totalCount;
        _pendingStringCopyIndex = -1;
        return;
    }

    if(arrayOfElements == nullptr)
    {
        // Step 2: requestedElementCount is 1-based index; return byte size and store index
        const int64_t index = requestedElementCount - 1;
        THROW_IF_TRUE(index >= totalCount,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::getAttribute(): index out of range for VALID_VALUES_STRING");
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
        const auto& str = _validValuesString[static_cast<size_t>(index)];
        *elementCount = static_cast<int64_t>(str.size() + 1);
        _pendingStringCopyIndex = index;
        return;
    }

    // Step 3: requestedElementCount is the caller's buffer capacity; copy pending string
    THROW_IF_TRUE(_pendingStringCopyIndex < 0 || _pendingStringCopyIndex >= totalCount,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::getAttribute(): "
                  "must query string size (null buffer) before copying VALID_VALUES_STRING");

    const auto& str = _validValuesString[static_cast<size_t>(_pendingStringCopyIndex)];
    auto maxSize = static_cast<size_t>(requestedElementCount);
    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
        static_cast<char*>(arrayOfElements), str.c_str(), maxSize);

    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(std::min(str.size() + 1, maxSize));
    }
}

void KnobDescriptor::getDefaultValueType(hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): "
                   "attributeType must be HIPDNN_TYPE_INT64 for DEFAULT_VALUE_TYPE");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::getAttribute(): arrayOfElements is null");
    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount must be 1");

    // Map the internal KnobValue discriminator to the corresponding attribute type
    // that callers should use when reading HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE.
    hipdnnBackendAttributeType_t valueType;
    switch(_defaultValue.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        valueType = HIPDNN_TYPE_INT64;
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        valueType = HIPDNN_TYPE_DOUBLE;
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        valueType = HIPDNN_TYPE_CHAR;
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "KnobDescriptor::getAttribute(): unknown default value type ("
                                  + std::to_string(static_cast<int>(_defaultValue.type)) + ")");
    }

    *static_cast<int64_t*>(arrayOfElements) = static_cast<int64_t>(valueType);
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void KnobDescriptor::getStringMaxLength(hipdnnBackendAttributeType_t attributeType,
                                        int64_t requestedElementCount,
                                        int64_t* elementCount,
                                        void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");

    if(!_stringMaxLength.has_value())
    {
        if(elementCount != nullptr)
        {
            *elementCount = 0;
        }
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::getAttribute(): arrayOfElements is null");
    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount must be 1");

    *static_cast<int64_t*>(arrayOfElements) = *_stringMaxLength;
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::unique_ptr<hipdnn_data_sdk::data_objects::KnobT> KnobDescriptor::toKnobT() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobDescriptor::toKnobT() failed: Not finalized.");

    auto knob = std::make_unique<hipdnn_data_sdk::data_objects::KnobT>();
    knob->knob_id = _knobId;
    knob->description = _description;
    knob->deprecated = _deprecated;

    // Deep-copy the default value
    switch(_defaultValue.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
    {
        hipdnn_data_sdk::data_objects::IntValueT intVal;
        intVal.value = _defaultValue.AsIntValue()->value;
        knob->default_value.Set(intVal);
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
    {
        hipdnn_data_sdk::data_objects::FloatValueT floatVal;
        floatVal.value = _defaultValue.AsFloatValue()->value;
        knob->default_value.Set(floatVal);
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
    {
        hipdnn_data_sdk::data_objects::StringValueT strVal;
        strVal.value = _defaultValue.AsStringValue()->value;
        knob->default_value.Set(std::move(strVal));
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "KnobDescriptor::toKnobT(): unknown default value type ("
                                  + std::to_string(static_cast<int>(_defaultValue.type)) + ")");
    }

    // Build constraint based on default value type and set constraint fields
    switch(_defaultValue.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        if(_minValueInt.has_value() || _maxValueInt.has_value() || _stride.has_value()
           || !_validValuesInt.empty())
        {
            hipdnn_data_sdk::data_objects::IntConstraintT intConstraint;
            intConstraint.min_value = _minValueInt.value_or(0);
            intConstraint.max_value = _maxValueInt.value_or(0);
            intConstraint.step = _stride.value_or(1);
            intConstraint.valid_values = _validValuesInt;
            knob->constraint.Set(std::move(intConstraint));
        }
        break;

    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        if(_minValueDouble.has_value() || _maxValueDouble.has_value())
        {
            hipdnn_data_sdk::data_objects::FloatConstraintT floatConstraint;
            floatConstraint.min_value = _minValueDouble.value_or(0.0);
            floatConstraint.max_value = _maxValueDouble.value_or(0.0);
            knob->constraint.Set(floatConstraint);
        }
        break;

    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        if(!_validValuesString.empty() || _stringMaxLength.has_value())
        {
            hipdnn_data_sdk::data_objects::StringConstraintT stringConstraint;
            stringConstraint.max_length = static_cast<int32_t>(_stringMaxLength.value_or(0));
            stringConstraint.valid_values = _validValuesString;
            knob->constraint.Set(std::move(stringConstraint));
        }
        break;

    default:
        break;
    }

    return knob;
}

hipdnnBackendDescriptorType_t KnobDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
}

std::string KnobDescriptor::toString() const
{
    std::string str = "KnobDescriptor: {knobId=" + _knobId;
    str += ", defaultValueType=" + std::to_string(static_cast<int>(_defaultValue.type));
    str += ", deprecated=" + std::string(_deprecated ? "true" : "false");

    if(_minValueInt.has_value() || _maxValueInt.has_value() || _stride.has_value()
       || !_validValuesInt.empty())
    {
        str += ", intConstraint:{";
        if(_minValueInt.has_value())
        {
            str += "min=" + std::to_string(*_minValueInt) + " ";
        }
        if(_maxValueInt.has_value())
        {
            str += "max=" + std::to_string(*_maxValueInt) + " ";
        }
        if(_stride.has_value())
        {
            str += "step=" + std::to_string(*_stride) + " ";
        }
        str += "validValues[" + std::to_string(_validValuesInt.size()) + "]}";
    }
    if(_minValueDouble.has_value() || _maxValueDouble.has_value())
    {
        str += ", floatConstraint:{";
        if(_minValueDouble.has_value())
        {
            str += "min=" + std::to_string(*_minValueDouble) + " ";
        }
        if(_maxValueDouble.has_value())
        {
            str += "max=" + std::to_string(*_maxValueDouble);
        }
        str += "}";
    }
    if(!_validValuesString.empty() || _stringMaxLength.has_value())
    {
        str += ", stringConstraint:{validValues[" + std::to_string(_validValuesString.size()) + "]";
        if(_stringMaxLength.has_value())
        {
            str += " maxLen=" + std::to_string(*_stringMaxLength);
        }
        str += "}";
    }

    str += "}";
    return str;
}

} // namespace hipdnn_backend
