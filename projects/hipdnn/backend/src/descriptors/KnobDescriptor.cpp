// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KnobDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

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

    // Validate that constraint fields match the default value type.
    // Reject mixed-type constraint sets that do not correspond to the default value type.
    switch(_defaultValue.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
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
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
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
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
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
        setBoundedString(_knobId,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         MAX_KNOB_ID_LENGTH,
                         "KnobDescriptor::setAttribute()",
                         1);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        if(attributeType == HIPDNN_TYPE_INT64)
        {
            setOptionalScalar<HIPDNN_TYPE_INT64>(_maxValueInt,
                                                 attributeType,
                                                 elementCount,
                                                 arrayOfElements,
                                                 "KnobDescriptor::setAttribute()");
        }
        else if(attributeType == HIPDNN_TYPE_DOUBLE)
        {
            setOptionalScalar<HIPDNN_TYPE_DOUBLE>(_maxValueDouble,
                                                  attributeType,
                                                  elementCount,
                                                  arrayOfElements,
                                                  "KnobDescriptor::setAttribute()");
        }
        else
        {
            throw HipdnnException(
                HIPDNN_STATUS_BAD_PARAM,
                std::string("KnobDescriptor::setAttribute(): "
                            "unsupported attribute type for MAXIMUM_VALUE: ")
                    + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
        }
        break;
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        if(attributeType == HIPDNN_TYPE_INT64)
        {
            setOptionalScalar<HIPDNN_TYPE_INT64>(_minValueInt,
                                                 attributeType,
                                                 elementCount,
                                                 arrayOfElements,
                                                 "KnobDescriptor::setAttribute()");
        }
        else if(attributeType == HIPDNN_TYPE_DOUBLE)
        {
            setOptionalScalar<HIPDNN_TYPE_DOUBLE>(_minValueDouble,
                                                  attributeType,
                                                  elementCount,
                                                  arrayOfElements,
                                                  "KnobDescriptor::setAttribute()");
        }
        else
        {
            throw HipdnnException(
                HIPDNN_STATUS_BAD_PARAM,
                std::string("KnobDescriptor::setAttribute(): "
                            "unsupported attribute type for MINIMUM_VALUE: ")
                    + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
        }
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRIDE:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_stride,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
        setBoundedString(_description,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         MAX_DESCRIPTION_LENGTH,
                         "KnobDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
        setDefaultValue(attributeType, elementCount, arrayOfElements);
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
        if(elementCount == 0)
        {
            _validValuesInt.clear();
        }
        else
        {
            setScalarVector(_validValuesInt,
                            HIPDNN_TYPE_INT64,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "KnobDescriptor::setAttribute()");
        }
        break;
    case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING:
        setValidValuesString(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_stringMaxLength,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobDescriptor::setDefaultValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t elementCount,
                                     const void* arrayOfElements)
{
    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        hipdnn_data_sdk::data_objects::IntValueT intVal;
        setScalar(intVal.value,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::setAttribute()");
        _defaultValue.Set(intVal);
        _defaultValueSet = true;
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        hipdnn_data_sdk::data_objects::FloatValueT floatVal;
        setScalar(floatVal.value,
                  HIPDNN_TYPE_DOUBLE,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::setAttribute()");
        _defaultValue.Set(floatVal);
        _defaultValueSet = true;
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        hipdnn_data_sdk::data_objects::StringValueT strVal;
        setBoundedString(strVal.value,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         MAX_STRING_VALUE_LENGTH,
                         "KnobDescriptor::setAttribute()");
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

    if(elementCount == 0)
    {
        if(arrayOfElements == nullptr)
        {
            // nullptr with elementCount=0: clear the list (mirrors VALID_VALUES_INT semantics)
            _validValuesString.clear();
        }
        else
        {
            // non-null pointer with elementCount=0: append an empty string
            _validValuesString.emplace_back();
        }
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobDescriptor::setAttribute(): "
                  "arrayOfElements is null with positive elementCount for VALID_VALUES_STRING");

    _validValuesString.emplace_back(static_cast<const char*>(arrayOfElements),
                                    static_cast<size_t>(elementCount));
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
        getMaximumValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        getMinimumValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
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
        getDefaultValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
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
        getInt64Vector(_validValuesInt,
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
        getOptionalScalar<HIPDNN_TYPE_INT64>(_stringMaxLength,
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

void KnobDescriptor::getDefaultValue(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    switch(_defaultValue.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        getScalar(_defaultValue.AsIntValue()->value,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::getAttribute()");
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        getScalar(_defaultValue.AsFloatValue()->value,
                  HIPDNN_TYPE_DOUBLE,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::getAttribute()");
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        getString(_defaultValue.AsStringValue()->value,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "KnobDescriptor::getAttribute()");
        break;

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
        getOptionalScalar<HIPDNN_TYPE_INT64>(_maxValueInt,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_TYPE_DOUBLE:
        getOptionalScalar<HIPDNN_TYPE_DOUBLE>(_maxValueDouble,
                                              attributeType,
                                              requestedElementCount,
                                              elementCount,
                                              arrayOfElements,
                                              "KnobDescriptor::getAttribute()");
        break;
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
        getOptionalScalar<HIPDNN_TYPE_INT64>(_minValueInt,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "KnobDescriptor::getAttribute()");
        break;
    case HIPDNN_TYPE_DOUBLE:
        getOptionalScalar<HIPDNN_TYPE_DOUBLE>(_minValueDouble,
                                              attributeType,
                                              requestedElementCount,
                                              elementCount,
                                              arrayOfElements,
                                              "KnobDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("KnobDescriptor::getAttribute(): "
                                          "unsupported attribute type for MINIMUM_VALUE: ")
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void KnobDescriptor::getValidValuesString(hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    // Two-step retrieval protocol (stateless):
    //   Step 1 (total count):  requestedElementCount=0, arrayOfElements=nullptr
    //                          → returns total number of strings in elementCount
    //   Step 2 (size query):   requestedElementCount=N (1-based index, N>0), arrayOfElements=nullptr
    //                          → returns byte length of string[N-1] (incl. null) in elementCount
    //   Step 3 (copy):         requestedElementCount=N (same 1-based index), arrayOfElements=buffer
    //                          → copies string[N-1] into buffer; caller must have allocated
    //                            based on size from step 2.

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
        return;
    }

    // requestedElementCount is a 1-based string index for both size query and copy
    const int64_t index = requestedElementCount - 1;
    THROW_IF_TRUE(index >= totalCount,
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobDescriptor::getAttribute(): index out of range for VALID_VALUES_STRING");

    const auto& str = _validValuesString[static_cast<size_t>(index)];

    if(arrayOfElements == nullptr)
    {
        // Step 2: return byte size of string[index] (incl. null terminator)
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobDescriptor::getAttribute(): elementCount is null");
        *elementCount = static_cast<int64_t>(str.size() + 1);
        return;
    }

    // Step 3: copy string[index] into caller's buffer
    std::memcpy(arrayOfElements, str.c_str(), str.size() + 1);

    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(str.size() + 1);
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
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        valueType = static_cast<int64_t>(HIPDNN_TYPE_INT64);
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        valueType = static_cast<int64_t>(HIPDNN_TYPE_DOUBLE);
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
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
