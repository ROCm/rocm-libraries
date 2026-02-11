// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file Knob.hpp
 * @brief Engine configuration knobs for hipDNN execution plans
 *
 * This file defines the Knob class which describes tunable configuration
 * parameters for execution engines. Knobs allow fine-grained control over
 * how operations are executed on the GPU.
 */

#pragma once

#include <HipdnnBackendFlatbufferData.h>

#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/KnobWrapper.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace hipdnn_frontend
{

/**
 * @class Knob
 * @brief Describes a tunable configuration parameter for an execution engine
 *
 * Knobs are engine-specific configuration options that can be adjusted to
 * tune performance. Each knob has:
 * - An identifier (knobId)
 * - A description of what it controls
 * - A default value
 * - Optional constraints (valid ranges, allowed values)
 *
 * Knobs are retrieved from engine descriptors and can be used to create
 * custom execution plans.
 *
 * @code{.cpp}
 * // Get available knobs for an engine
 * auto engines = graph.get_available_engines(handle);
 * auto knobs = engines[0].getKnobs();
 *
 * for(const auto& knob : knobs)
 * {
 *     std::cout << knob.knobId() << ": " << knob.description() << std::endl;
 * }
 * @endcode
 *
 * @see KnobSetting, KnobConstraint, Graph::get_available_engines()
 */
class Knob
{
public:
    /**
     * @brief Create a Knob from serialized FlatBuffer data
     * @param fbData The serialized FlatBuffer data from the backend
     * @return A Knob instance
     * @throws std::invalid_argument if the data is invalid
     */
    static Knob fromFlatbuffer(hipdnnBackendFlatbufferData_t fbData)
    {
        if(fbData.ptr == nullptr || fbData.size == 0)
        {
            throw std::invalid_argument("Flatbuffer data is nullptr or has zero size");
        }

        hipdnn_data_sdk::flatbuffer_utilities::KnobWrapper knobWrapper(fbData.ptr, fbData.size);

        if(!knobWrapper.isValid())
        {
            throw std::invalid_argument("Knob flatbuffer failed verification");
        }

        auto fbKnob = &knobWrapper.getKnob();

        // Unpack to native KnobT - all conversions done automatically by FlatBuffers
        std::unique_ptr<hipdnn_data_sdk::data_objects::KnobT> knobT(fbKnob->UnPack());

        // Extract default value from the union
        KnobValueVariant defaultValue;
        switch(knobT->default_value.type)
        {
        case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
            defaultValue = knobT->default_value.AsIntValue()->value;
            break;
        case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
            defaultValue = knobT->default_value.AsFloatValue()->value;
            break;
        case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
            defaultValue = knobT->default_value.AsStringValue()->value; // Already std::string
            break;
        default:
            throw std::invalid_argument("Unknown knob value type");
        }

        // Create the knob - strings are already std::string in KnobT
        Knob knob(std::move(knobT->knob_id),
                  std::move(knobT->description),
                  std::move(defaultValue),
                  knobT->deprecated);

        // Handle constraints using the native union types
        switch(knobT->constraint.type)
        {
        case hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint:
        {
            auto* c = knobT->constraint.AsIntConstraint();
            // c->valid_values is already std::vector<int64_t>
            std::unordered_set<int64_t> validValues(c->valid_values.begin(), c->valid_values.end());
            knob._constraint = std::make_unique<IntConstraint>(
                c->min_value, c->max_value, c->step, std::move(validValues));
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint:
        {
            auto* c = knobT->constraint.AsFloatConstraint();
            knob._constraint = std::make_unique<FloatConstraint>(c->min_value, c->max_value);
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint:
        {
            auto* c = knobT->constraint.AsStringConstraint();
            // c->valid_values is already std::vector<std::string>
            std::unordered_set<std::string> validValues(c->valid_values.begin(),
                                                        c->valid_values.end());
            knob._constraint
                = std::make_unique<StringConstraint>(c->max_length, std::move(validValues));
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::NONE:
            // No constraint
            break;
        default:
            throw std::invalid_argument("Unknown knob constraint");
            break;
        }

        return knob;
    }

    /**
     * @brief Get the knob identifier
     * @return The unique identifier string for this knob
     */
    const std::string& knobId() const
    {
        return _knobId;
    }

    /**
     * @brief Get the knob description
     * @return Human-readable description of what this knob controls
     */
    const std::string& description() const
    {
        return _description;
    }

    /**
     * @brief Check if this knob is deprecated
     * @return true if the knob is deprecated and should not be used
     */
    bool isDeprecated() const
    {
        return _deprecated;
    }

    /**
     * @brief Get the value type of this knob
     * @return The KnobValueType (INT64, FLOAT64, or STRING)
     */
    KnobValueType valueType() const
    {
        return getKnobValueTypeFromVariant(_defaultValue);
    }

    /**
     * @brief Get the default value for this knob
     * @return The default value as a variant
     */
    const KnobValueVariant& defaultValue() const
    {
        return _defaultValue;
    }

    /**
     * @brief Get the constraint for this knob
     * @return Pointer to the constraint, or nullptr if no constraint
     */
    const IConstraint* constraint() const
    {
        return _constraint.get();
    }

    /**
     * @brief Validate a knob setting against this knob's constraints
     * @param setting The KnobSetting to validate
     * @return Error indicating success or describing the validation failure
     */
    Error validate(const KnobSetting& setting) const
    {
        // Validate against constraint if present
        if(_constraint)
        {
            return _constraint->validateKnobSetting(setting);
        }

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Get a string representation of this knob
     * @return Human-readable string for debugging/logging
     */
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "Knob{knobIdStr=\"" << _knobId << "\", description=\"" << _description
            << "\", defaultValue=";

        variantToStream(oss, _defaultValue);

        oss << ", deprecated=" << (_deprecated ? "true" : "false");

        if(_constraint)
        {
            oss << ", constraint=" << _constraint->toString();
        }

        oss << "}";
        return oss.str();
    }

private:
    /// @brief Private constructor - use fromFlatbuffer() factory function
    Knob(std::string knobIdStr,
         std::string description,
         KnobValueVariant defaultValue,
         bool deprecated)
        : _knobId(std::move(knobIdStr))
        , _description(std::move(description))
        , _defaultValue(std::move(defaultValue))
        , _deprecated(deprecated)
    {
    }

    static void variantToStream(std::ostringstream& oss, const KnobValueVariant& variant)
    {
        std::visit(
            [&oss](auto&& value) {
                if constexpr(std::is_same_v<std::decay_t<decltype(value)>, std::string>)
                {
                    oss << "\"" << value << "\"";
                }
                else
                {
                    oss << value;
                }
            },
            variant);
    }

    std::string _knobId;                         ///< Unique knob identifier
    std::string _description;                    ///< Human-readable description
    KnobValueVariant _defaultValue;              ///< Default value
    bool _deprecated;                            ///< Whether this knob is deprecated

    std::shared_ptr<IConstraint> _constraint;    ///< Optional constraint
};

namespace detail
{
inline Error getKnobsForEngine(std::vector<Knob>& knobs, hipdnnBackendDescriptor_t engineDesc)
{
    int64_t knobCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineDesc,
                                             HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                             HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                             0,
                                             &knobCount,
                                             nullptr),
        "Failed to get knob count from engine descriptor.");

    if(knobCount == 0)
    {
        knobs.clear();
        return {ErrorCode::OK, ""};
    }

    std::vector<hipdnnBackendFlatbufferData_t> flatbufferDataArray(static_cast<size_t>(knobCount));

    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineDesc,
                                             HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                             HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                             knobCount,
                                             &actualCount,
                                             flatbufferDataArray.data()),
        "Failed to get knob flatbuffer data from engine descriptor.");

    if(actualCount != knobCount)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Mismatch between expected and actual knob count."};
    }

    knobs.clear();
    knobs.reserve(static_cast<size_t>(actualCount));

    std::unordered_set<std::string> usedKnobIds;

    for(size_t i = 0; i < static_cast<size_t>(actualCount); ++i)
    {
        try
        {
            knobs.emplace_back(Knob::fromFlatbuffer(flatbufferDataArray[i]));
            if(!usedKnobIds.insert(knobs.back().knobId()).second)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Engine description had knob with duplicate ID: " + knobs.back().knobId()};
            }
        }
        catch(const std::exception& e)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    std::string("Failed to create Knob from flatbuffer at index ")
                        + std::to_string(i) + ": " + e.what()};
        }
    }

    return {ErrorCode::OK, ""};
}

} // namespace detail

} // namespace hipdnn_frontend
