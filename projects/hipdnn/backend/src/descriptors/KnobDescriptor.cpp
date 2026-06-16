// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KnobDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "logging/Logging.hpp"

#include <algorithm>

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

    THROW_IF_TRUE(_defaultValue.type == hipdnn_flatbuffers_sdk::data_objects::KnobValue::NONE,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::finalize() failed: Default value is not set.");

    // Validate that constraint fields match the default value type.
    // Reject mixed-type constraint sets that do not correspond to the default value type.
    switch(_defaultValue.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
        THROW_IF_TRUE(_minValueDouble.has_value() || _maxValueDouble.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "DOUBLE min/max constraints set on INT64 knob.");
        THROW_IF_FALSE(_validValuesString.empty(),
                       HIPDNN_STATUS_BAD_PARAM,
                       "KnobDescriptor::finalize() failed: "
                       "VALID_VALUES_STRING set on INT64 knob.");
        THROW_IF_TRUE(_stringMaxLength.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "STRING_MAX_LENGTH set on INT64 knob.");
        break;
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
        THROW_IF_TRUE(_minValueInt.has_value() || _maxValueInt.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "INT64 min/max constraints set on DOUBLE knob.");
        THROW_IF_TRUE(_stride.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "STRIDE set on DOUBLE knob.");
        THROW_IF_FALSE(_validValuesInt.empty(),
                       HIPDNN_STATUS_BAD_PARAM,
                       "KnobDescriptor::finalize() failed: "
                       "VALID_VALUES_INT set on DOUBLE knob.");
        THROW_IF_FALSE(_validValuesString.empty(),
                       HIPDNN_STATUS_BAD_PARAM,
                       "KnobDescriptor::finalize() failed: "
                       "VALID_VALUES_STRING set on DOUBLE knob.");
        THROW_IF_TRUE(_stringMaxLength.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "STRING_MAX_LENGTH set on DOUBLE knob.");
        break;
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
        THROW_IF_TRUE(_minValueInt.has_value() || _maxValueInt.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "INT64 min/max constraints set on STRING knob.");
        THROW_IF_TRUE(_minValueDouble.has_value() || _maxValueDouble.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "DOUBLE min/max constraints set on STRING knob.");
        THROW_IF_TRUE(_stride.has_value(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "STRIDE set on STRING knob.");
        THROW_IF_FALSE(_validValuesInt.empty(),
                       HIPDNN_STATUS_BAD_PARAM,
                       "KnobDescriptor::finalize() failed: "
                       "VALID_VALUES_INT set on STRING knob.");
        break;
    default:
        break;
    }

    // Min/max must be both set or both unset to avoid inventing default range bounds.
    THROW_IF_TRUE(
        _minValueInt.has_value() != _maxValueInt.has_value(),
        HIPDNN_STATUS_BAD_PARAM,
        "KnobDescriptor::finalize() failed: "
        "MINIMUM_VALUE (INT64) and MAXIMUM_VALUE (INT64) must both be set or both unset.");
    THROW_IF_TRUE(
        _minValueDouble.has_value() != _maxValueDouble.has_value(),
        HIPDNN_STATUS_BAD_PARAM,
        "KnobDescriptor::finalize() failed: "
        "MINIMUM_VALUE (DOUBLE) and MAXIMUM_VALUE (DOUBLE) must both be set or both unset.");

    if(_minValueInt.has_value())
    {
        THROW_IF_TRUE(*_minValueInt > *_maxValueInt,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "MINIMUM_VALUE (INT64) > MAXIMUM_VALUE (INT64).");
    }
    if(_minValueDouble.has_value())
    {
        THROW_IF_TRUE(*_minValueDouble > *_maxValueDouble,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::finalize() failed: "
                      "MINIMUM_VALUE (DOUBLE) > MAXIMUM_VALUE (DOUBLE).");
    }

    // Validate that the default value satisfies declared constraints.
    switch(_defaultValue.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
    {
        auto defaultVal = _defaultValue.AsIntValue()->value;
        if(_minValueInt.has_value())
        {
            THROW_IF_TRUE(defaultVal < *_minValueInt,
                          HIPDNN_STATUS_BAD_PARAM,
                          "KnobDescriptor::finalize() failed: "
                          "default value ("
                              + std::to_string(defaultVal) + ") < MINIMUM_VALUE ("
                              + std::to_string(*_minValueInt) + ").");
            THROW_IF_TRUE(defaultVal > *_maxValueInt,
                          HIPDNN_STATUS_BAD_PARAM,
                          "KnobDescriptor::finalize() failed: "
                          "default value ("
                              + std::to_string(defaultVal) + ") > MAXIMUM_VALUE ("
                              + std::to_string(*_maxValueInt) + ").");
            if(_stride.has_value())
            {
                THROW_IF_TRUE((defaultVal - *_minValueInt) % *_stride != 0,
                              HIPDNN_STATUS_BAD_PARAM,
                              "KnobDescriptor::finalize() failed: "
                              "default value ("
                                  + std::to_string(defaultVal) + ") is not aligned to STRIDE ("
                                  + std::to_string(*_stride) + ") from MINIMUM_VALUE ("
                                  + std::to_string(*_minValueInt) + ").");
            }
        }
        if(!_validValuesInt.empty())
        {
            THROW_IF_TRUE(std::find(_validValuesInt.begin(), _validValuesInt.end(), defaultVal)
                              == _validValuesInt.end(),
                          HIPDNN_STATUS_BAD_PARAM,
                          "KnobDescriptor::finalize() failed: "
                          "default value ("
                              + std::to_string(defaultVal) + ") is not in VALID_VALUES_INT.");
        }
        break;
    }
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
    {
        auto defaultVal = _defaultValue.AsFloatValue()->value;
        if(_minValueDouble.has_value())
        {
            THROW_IF_TRUE(defaultVal < *_minValueDouble,
                          HIPDNN_STATUS_BAD_PARAM,
                          "KnobDescriptor::finalize() failed: "
                          "default value is less than MINIMUM_VALUE (DOUBLE).");
            THROW_IF_TRUE(defaultVal > *_maxValueDouble,
                          HIPDNN_STATUS_BAD_PARAM,
                          "KnobDescriptor::finalize() failed: "
                          "default value is greater than MAXIMUM_VALUE (DOUBLE).");
        }
        break;
    }
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
    {
        const auto& defaultVal = _defaultValue.AsStringValue()->value;
        if(_stringMaxLength.has_value())
        {
            THROW_IF_TRUE(static_cast<int64_t>(defaultVal.size()) > *_stringMaxLength,
                          HIPDNN_STATUS_BAD_PARAM,
                          "KnobDescriptor::finalize() failed: "
                          "default value length ("
                              + std::to_string(defaultVal.size()) + ") exceeds STRING_MAX_LENGTH ("
                              + std::to_string(*_stringMaxLength) + ").");
        }
        if(!_validValuesString.empty())
        {
            THROW_IF_TRUE(
                std::find(_validValuesString.begin(), _validValuesString.end(), defaultVal)
                    == _validValuesString.end(),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::finalize() failed: "
                "default value is not in VALID_VALUES_STRING.");
        }
        break;
    }
    default:
        break;
    }

    HipdnnBackendDescriptorImpl<KnobDescriptor>::finalize();
}

// ============================================================================
// fromKnobT
// ============================================================================

std::shared_ptr<KnobDescriptor>
    KnobDescriptor::fromKnobT(const hipdnn_flatbuffers_sdk::data_objects::KnobT& knobNative)
{
    auto knobDesc = std::make_shared<KnobDescriptor>();

    // Set knob ID
    knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(knobNative.knob_id.size()),
                           knobNative.knob_id.c_str());

    // Set description
    if(!knobNative.description.empty())
    {
        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION,
                               HIPDNN_TYPE_CHAR,
                               static_cast<int64_t>(knobNative.description.size()),
                               knobNative.description.c_str());
    }

    // Set deprecated flag
    knobDesc->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, 1, &knobNative.deprecated);

    // Set default value and matching constraint fields based on type
    switch(knobNative.default_value.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
    {
        auto val = knobNative.default_value.AsIntValue()->value;
        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 1, &val);

        if(knobNative.constraint.type
           == hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint)
        {
            const auto* c = knobNative.constraint.AsIntConstraint();
            // Only set range bounds when non-zero: {0,0} is the plugin SDK's
            // sentinel meaning "no range constraint" (only valid_values applies).
            if(c->min_value != 0 || c->max_value != 0)
            {
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &c->min_value);
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &c->max_value);
            }
            if(c->step > 0)
            {
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &c->step);
            }
            if(!c->valid_values.empty())
            {
                knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(c->valid_values.size()),
                                       c->valid_values.data());
            }
        }
        break;
    }
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
    {
        auto val = knobNative.default_value.AsFloatValue()->value;
        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_DOUBLE, 1, &val);

        if(knobNative.constraint.type
           == hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::FloatConstraint)
        {
            const auto* c = knobNative.constraint.AsFloatConstraint();
            // Only set range bounds when non-zero: {0.0,0.0} is the plugin SDK's
            // sentinel meaning "no range constraint".
            if(c->min_value != 0.0 || c->max_value != 0.0)
            {
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &c->min_value);
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &c->max_value);
            }
        }
        break;
    }
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
    {
        const auto& val = knobNative.default_value.AsStringValue()->value;
        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                               HIPDNN_TYPE_CHAR,
                               static_cast<int64_t>(val.size()),
                               val.c_str());

        if(knobNative.constraint.type
           == hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::StringConstraint)
        {
            const auto* c = knobNative.constraint.AsStringConstraint();
            if(c->max_length > 0)
            {
                auto maxLen = static_cast<int32_t>(c->max_length);
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen);
            }
            if(!c->valid_values.empty())
            {
                // Build null-separated buffer: "str1\0str2\0str3\0"
                std::string buf;
                for(const auto& s : c->valid_values)
                {
                    buf.append(s);
                    buf.push_back('\0');
                }
                knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(buf.size()),
                                       buf.data());
            }
        }
        break;
    }
    default:
        HIPDNN_BACKEND_LOG_WARN("KnobDescriptor::fromKnobT: skipping knob '{}' "
                                "with unknown default value type {}",
                                knobNative.knob_id,
                                static_cast<int>(knobNative.default_value.type));
        return nullptr;
    }

    knobDesc->finalize();
    return knobDesc;
}

namespace
{

void setBoundValue(std::optional<int64_t>& intVal,
                   std::optional<double>& doubleVal,
                   const char* label,
                   hipdnnBackendAttributeType_t attributeType,
                   int64_t elementCount,
                   const void* arrayOfElements)
{
    if(attributeType == HIPDNN_TYPE_INT64)
    {
        setOptionalScalar<HIPDNN_TYPE_INT64>(
            intVal, attributeType, elementCount, arrayOfElements, "KnobDescriptor::setAttribute()");
    }
    else if(attributeType == HIPDNN_TYPE_DOUBLE)
    {
        setOptionalScalar<HIPDNN_TYPE_DOUBLE>(doubleVal,
                                              attributeType,
                                              elementCount,
                                              arrayOfElements,
                                              "KnobDescriptor::setAttribute()");
    }
    else
    {
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            std::string("KnobDescriptor::setAttribute(): unsupported attribute type for ") + label
                + ": " + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void getBoundValue(const std::optional<int64_t>& intVal,
                   const std::optional<double>& doubleVal,
                   const char* label,
                   hipdnnBackendAttributeType_t attributeType,
                   int64_t requestedElementCount,
                   int64_t* elementCount,
                   void* arrayOfElements)
{
    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
        getOptionalScalar<HIPDNN_TYPE_INT64>(intVal,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_TYPE_DOUBLE:
        getOptionalScalar<HIPDNN_TYPE_DOUBLE>(doubleVal,
                                              attributeType,
                                              requestedElementCount,
                                              elementCount,
                                              arrayOfElements,
                                              "KnobDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            std::string("KnobDescriptor::getAttribute(): unsupported attribute type for ") + label
                + ": " + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

} // anonymous namespace

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
        setBoundedString(_knobId,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         "KnobDescriptor::setAttribute()",
                         MAX_KNOB_ID_LENGTH,
                         1);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        setBoundValue(_maxValueInt,
                      _maxValueDouble,
                      "MAXIMUM_VALUE",
                      attributeType,
                      elementCount,
                      arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        setBoundValue(_minValueInt,
                      _minValueDouble,
                      "MINIMUM_VALUE",
                      attributeType,
                      elementCount,
                      arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRIDE:
    {
        std::optional<int64_t> temp;
        setOptionalScalar<HIPDNN_TYPE_INT64>(
            temp, attributeType, elementCount, arrayOfElements, "KnobDescriptor::setAttribute()");
        THROW_IF_TRUE(*temp <= 0,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::setAttribute(): STRIDE must be positive (> 0).");
        _stride = temp;
        break;
    }
    case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
        setBoundedString(_description,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         "KnobDescriptor::setAttribute()",
                         MAX_DESCRIPTION_LENGTH);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
        setKnobValueUnion(_defaultValue,
                          attributeType,
                          elementCount,
                          arrayOfElements,
                          "KnobDescriptor::setAttribute()",
                          MAX_STRING_VALUE_LENGTH);
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEPRECATED:
        setScalar(_deprecated,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT:
        setScalarVector(_validValuesInt,
                        HIPDNN_TYPE_INT64,
                        attributeType,
                        elementCount,
                        arrayOfElements,
                        "KnobDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING:
        setValidValuesString(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH:
    {
        std::optional<int32_t> temp;
        setOptionalScalar<HIPDNN_TYPE_INT32>(
            temp, attributeType, elementCount, arrayOfElements, "KnobDescriptor::setAttribute()");
        THROW_IF_TRUE(*temp <= 0,
                      HIPDNN_STATUS_BAD_PARAM,
                      "KnobDescriptor::setAttribute(): STRING_MAX_LENGTH must be positive (> 0).");
        _stringMaxLength = temp;
        break;
    }
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobDescriptor::setValidValuesString(hipdnnBackendAttributeType_t attributeType,
                                          int64_t elementCount,
                                          const void* arrayOfElements)
{
    checkSetArgs(
        HIPDNN_TYPE_CHAR, attributeType, arrayOfElements, "KnobDescriptor::setAttribute()");
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(1),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::setAttribute(): elementCount must be >= 1");

    // Caller passes a flat null-separated buffer: "str1\0str2\0str3\0".
    // elementCount is the total byte count of the buffer.
    // Split on embedded null terminators to populate the string list.
    const auto* data = static_cast<const char*>(arrayOfElements);
    const auto* end = data + elementCount;

    _validValuesString.clear();
    while(data < end)
    {
        // Each string runs from data to the next \0 (or end of buffer).
        const auto* strEnd
            = static_cast<const char*>(std::memchr(data, '\0', static_cast<size_t>(end - data)));
        if(strEnd == nullptr)
        {
            strEnd = end;
        }
        _validValuesString.emplace_back(data, strEnd);
        data = strEnd + 1;
    }
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
        getString(_knobId,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        getBoundValue(_maxValueInt,
                      _maxValueDouble,
                      "MAXIMUM_VALUE",
                      attributeType,
                      requestedElementCount,
                      elementCount,
                      arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        getBoundValue(_minValueInt,
                      _minValueDouble,
                      "MINIMUM_VALUE",
                      attributeType,
                      requestedElementCount,
                      elementCount,
                      arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRIDE:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_stride,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
        getString(_description,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
        getKnobValueUnion(_defaultValue,
                          attributeType,
                          requestedElementCount,
                          elementCount,
                          arrayOfElements,
                          "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEPRECATED:
        getScalar(_deprecated,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT:
        getScalarVector(_validValuesInt,
                        HIPDNN_TYPE_INT64,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING:
        getValidValuesString(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH:
        getOptionalScalar<HIPDNN_TYPE_INT32>(_stringMaxLength,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::getAttribute()");
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

void KnobDescriptor::getValidValuesString(hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    // Flat null-separated buffer retrieval:
    //   Size query: arrayOfElements=nullptr → *elementCount = total bytes needed
    //               (sum of all string lengths + 1 null per string)
    //   Copy:       arrayOfElements=char* buffer, requestedElementCount=buffer size in bytes
    //               → copies "str1\0str2\0str3\0" into buffer, truncating at buffer boundary;
    //                 *elementCount = bytes written.

    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");

    // Compute total byte size: each string contributes its length + 1 (null terminator).
    int64_t totalBytes = 0;
    for(const auto& s : _validValuesString)
    {
        totalBytes += static_cast<int64_t>(s.size()) + 1;
    }

    // Size query
    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
        *elementCount = totalBytes;
        return;
    }

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                "KnobDescriptor::getAttribute(): requestedElementCount is negative");

    // Copy "str1\0str2\0str3\0" into caller's buffer.
    auto* out = static_cast<char*>(arrayOfElements);
    int64_t remaining = requestedElementCount;
    int64_t written = 0;

    for(const auto& s : _validValuesString)
    {
        const auto needed = static_cast<int64_t>(s.size()) + 1; // string + null
        if(needed > remaining)
        {
            break; // not enough room for this string + null; stop cleanly
        }
        std::memcpy(out + written, s.c_str(), static_cast<size_t>(needed));
        written += needed;
        remaining -= needed;
    }

    if(elementCount != nullptr)
    {
        *elementCount = written;
    }
}

void KnobDescriptor::getDefaultValueType(hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    // Map the internal KnobValue discriminator to the corresponding attribute type
    // that callers should use when reading HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE.
    int64_t valueType;
    switch(_defaultValue.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
        valueType = static_cast<int64_t>(HIPDNN_TYPE_INT64);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
        valueType = static_cast<int64_t>(HIPDNN_TYPE_DOUBLE);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
        valueType = static_cast<int64_t>(HIPDNN_TYPE_CHAR);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "KnobDescriptor::getAttribute(): unknown default value type ("
                                  + std::to_string(static_cast<int>(_defaultValue.type)) + ")");
    }

    getScalar(valueType,
              HIPDNN_TYPE_INT64,
              attributeType,
              requestedElementCount,
              elementCount,
              arrayOfElements,
              "KnobDescriptor::getAttribute()");
}

// ============================================================================
// Other methods
// ============================================================================

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobT> KnobDescriptor::toKnobT() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobDescriptor::toKnobT() failed: Not finalized.");

    auto knob = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::KnobT>();
    knob->knob_id = _knobId;
    knob->description = _description;
    knob->deprecated = _deprecated;

    copyKnobValueUnion(_defaultValue, knob->default_value, "KnobDescriptor::toKnobT()");

    // Build constraint based on default value type and set constraint fields
    switch(_defaultValue.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
        if(_minValueInt.has_value() || _maxValueInt.has_value() || _stride.has_value()
           || !_validValuesInt.empty())
        {
            hipdnn_flatbuffers_sdk::data_objects::IntConstraintT intConstraint;
            intConstraint.min_value = _minValueInt.value_or(0);
            intConstraint.max_value = _maxValueInt.value_or(0);
            intConstraint.step = _stride.value_or(1);
            intConstraint.valid_values = _validValuesInt;
            knob->constraint.Set(std::move(intConstraint));
        }
        break;

    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
        if(_minValueDouble.has_value() || _maxValueDouble.has_value())
        {
            hipdnn_flatbuffers_sdk::data_objects::FloatConstraintT floatConstraint;
            floatConstraint.min_value = _minValueDouble.value_or(0.0);
            floatConstraint.max_value = _maxValueDouble.value_or(0.0);
            knob->constraint.Set(floatConstraint);
        }
        break;

    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
        if(!_validValuesString.empty() || _stringMaxLength.has_value())
        {
            hipdnn_flatbuffers_sdk::data_objects::StringConstraintT stringConstraint;
            stringConstraint.max_length = _stringMaxLength.value_or(0);
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
        str += '}';
    }
    if(!_validValuesString.empty() || _stringMaxLength.has_value())
    {
        str += ", stringConstraint:{validValues[" + std::to_string(_validValuesString.size()) + "]";
        if(_stringMaxLength.has_value())
        {
            str += " maxLen=" + std::to_string(*_stringMaxLength);
        }
        str += '}';
    }

    str += '}';
    return str;
}

} // namespace hipdnn_backend
