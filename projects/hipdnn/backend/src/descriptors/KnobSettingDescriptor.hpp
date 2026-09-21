// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>

namespace hipdnn_backend
{

/**
 * @brief Backend descriptor for knob setting configuration
 *
 * Wraps a knob identifier and value pair as a standard backend descriptor,
 * allowing knob settings to be passed through the C-API setAttribute/getAttribute
 * pattern instead of manual FlatBuffer serialization.
 *
 * Validation limits (to catch accidental misuse):
 * - Knob ID string: up to @ref MAX_KNOB_ID_LENGTH characters (must be non-empty)
 * - Knob string value: up to @ref MAX_KNOB_STRING_VALUE_LENGTH characters
 */
class KnobSettingDescriptor : public HipdnnBackendDescriptorImpl<KnobSettingDescriptor>
{
public:
    /// Maximum length of a knob ID string (characters, excluding null terminator).
    static constexpr int64_t MAX_KNOB_ID_LENGTH = 4096;

    /// Maximum length of a knob string value (characters, excluding null terminator).
    static constexpr int64_t MAX_KNOB_STRING_VALUE_LENGTH = 65536;
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

    /// Convert to a KnobSettingT for consumption by EngineConfigDescriptor
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT> toKnobSettingT() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    std::string _knobId;
    hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion _value;
};

} // namespace hipdnn_backend
