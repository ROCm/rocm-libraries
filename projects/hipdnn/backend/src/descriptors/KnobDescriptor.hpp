// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>

#include <optional>
#include <string>
#include <vector>

namespace hipdnn_backend
{

/**
 * @brief Backend descriptor for knob metadata
 *
 * Wraps knob metadata (identifier, description, default value, constraints,
 * deprecation flag) as a standard backend descriptor, allowing knob info to
 * be passed through the C-API setAttribute/getAttribute pattern instead of
 * manual FlatBuffer serialization.
 *
 * Validation limits:
 * - Knob ID string: up to @ref MAX_KNOB_ID_LENGTH characters (must be non-empty)
 * - Description string: up to @ref MAX_DESCRIPTION_LENGTH characters
 * - String default value: up to @ref MAX_STRING_VALUE_LENGTH characters
 */
class KnobDescriptor : public HipdnnBackendDescriptorImpl<KnobDescriptor>
{
public:
    /// Maximum length of a knob ID string (characters, excluding null terminator).
    static constexpr int64_t MAX_KNOB_ID_LENGTH = 4096;

    /// Maximum length of a description string (characters, excluding null terminator).
    static constexpr int64_t MAX_DESCRIPTION_LENGTH = 65536;

    /// Maximum length of a string default value (characters, excluding null terminator).
    static constexpr int64_t MAX_STRING_VALUE_LENGTH = 65536;

    /// Construct a finalized KnobDescriptor from a deserialized KnobT.
    /// Returns nullptr if the knob has an unsupported default value type.
    static std::shared_ptr<KnobDescriptor>
        fromKnobT(const hipdnn_flatbuffers_sdk::data_objects::KnobT& knobNative);

    void finalize() override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    /// Convert to a KnobT for consumption by other components
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobT> toKnobT() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    // Core knob ID and metadata
    std::string _knobId;
    std::string _description;

    // Default value (polymorphic: int64, double, or string); NONE until set
    hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion _defaultValue;

    // Deprecation flag
    bool _deprecated = false;

    // Numeric constraint fields
    std::optional<int64_t> _maxValueInt;
    std::optional<int64_t> _minValueInt;
    std::optional<double> _maxValueDouble;
    std::optional<double> _minValueDouble;
    std::optional<int64_t> _stride;

    // Array/enum constraint fields
    std::vector<int64_t> _validValuesInt;
    std::vector<std::string> _validValuesString;
    std::optional<int32_t> _stringMaxLength;

    // Private helpers — complex logic that can't be inlined into the switch
    void setValidValuesString(hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements);

    void getValidValuesString(hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) const;

    void getDefaultValueType(hipdnnBackendAttributeType_t attributeType,
                             int64_t requestedElementCount,
                             int64_t* elementCount,
                             void* arrayOfElements) const;
};

} // namespace hipdnn_backend
