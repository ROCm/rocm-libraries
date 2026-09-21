// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Knob.hpp
 * @brief Engine configuration knobs for hipDNN execution plans
 *
 * This file defines the Knob class which describes tunable configuration
 * parameters for execution engines. Knobs allow fine-grained control over
 * how operations are executed on the GPU.
 */

#pragma once

#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <hipdnn_frontend/Utilities.hpp>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
 * std::vector<int64_t> engineIds;
 * graph.get_ranked_engine_ids(engineIds);
 *
 * std::vector<Knob> knobs;
 * graph.get_knobs_for_engine(engineIds[0], knobs);
 *
 * for(const auto& knob : knobs)
 * {
 *     std::cout << knob.knobId() << ": " << knob.description() << std::endl;
 * }
 * @endcode
 *
 * @see KnobSetting, IConstraint, Graph::get_knobs_for_engine()
 */
class Knob
{
public:
    /// Shared construction path for parsed knob data from any source.
    static std::pair<Error, Knob> tryCreate(std::string knobIdStr,
                                            std::string description,
                                            KnobValueVariant defaultValue,
                                            bool deprecated,
                                            std::shared_ptr<IConstraint> constraint
                                            = std::make_shared<EmptyConstraint>());

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
        if(setting.knobId() != _knobId)
        {
            return {ErrorCode::INVALID_VALUE,
                    "KnobSetting knob ID '" + setting.knobId() + "' does not match knob ID '"
                        + _knobId + "'"};
        }

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
    // Private default constructor - allows factory functions to create an empty Knob on failure.
    Knob() = default;

    // Private constructor - use tryCreate() factory function to create instances
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

    std::string _knobId; ///< Unique knob identifier
    std::string _description; ///< Human-readable description
    KnobValueVariant _defaultValue; ///< Default value
    bool _deprecated = false; ///< Whether this knob is deprecated

    std::shared_ptr<IConstraint> _constraint; ///< Optional constraint
};

inline std::pair<Error, Knob> Knob::tryCreate(std::string knobIdStr,
                                              std::string description,
                                              KnobValueVariant defaultValue,
                                              bool deprecated,
                                              std::shared_ptr<IConstraint> constraint)
{
    if(knobIdStr.empty())
    {
        return {{ErrorCode::INVALID_VALUE, "Knob ID must not be empty"}, {}};
    }

    Knob knob(std::move(knobIdStr), std::move(description), std::move(defaultValue), deprecated);
    knob._constraint
        = constraint != nullptr ? std::move(constraint) : std::make_shared<EmptyConstraint>();

    const KnobSetting defaultSetting(knob._knobId, knob._defaultValue);
    auto validationError = knob._constraint->validateKnobSetting(defaultSetting);
    if(validationError.code != ErrorCode::OK)
    {
        return {{ErrorCode::INVALID_VALUE,
                 "Knob '" + knob._knobId + "' has default_value that violates its constraint: "
                     + validationError.err_msg},
                {}};
    }

    return {{ErrorCode::OK, ""}, knob};
}

} // namespace hipdnn_frontend
