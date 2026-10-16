// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Attributes.hpp
 * @brief Base class for operation attribute classes using CRTP
 *
 * This file defines the Attributes template base class that provides
 * common functionality for all operation attribute classes. It uses
 * the Curiously Recurring Template Pattern (CRTP) to enable method
 * chaining in derived classes.
 */

#pragma once

#include "GraphAttributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace hipdnn_frontend::graph
{

/**
 * @class Attributes
 * @brief CRTP base class for operation attribute classes
 * @tparam DerivedT The derived attribute class type
 *
 * This template class provides common functionality shared by all
 * operation attribute classes:
 * - Name management for debugging and logging
 * - Compute data type configuration
 * - Automatic tensor property propagation from graph context
 *
 * Derived classes must define `inputs` and `outputs` maps with
 * TensorAttributes values. These maps are used by fill_from_context()
 * to propagate graph-level settings to tensors.
 *
 * @code{.cpp}
 * class MyOperationAttributes : public Attributes<MyOperationAttributes>
 * {
 * public:
 *     std::unordered_map<InputName, std::shared_ptr<TensorAttributes>> inputs;
 *     std::unordered_map<OutputName, std::shared_ptr<TensorAttributes>> outputs;
 * };
 * @endcode
 */
template <typename DerivedT>
class Attributes
{
    friend DerivedT;

private:
    /**
     * @brief Get mutable reference to derived class
     * @return Reference to *this cast to DerivedT&
     */
    DerivedT& self()
    {
        return static_cast<DerivedT&>(*this);
    }

    /**
     * @brief Get const reference to derived class
     * @return Reference to *this cast to const DerivedT&
     */
    const DerivedT& self() const
    {
        return static_cast<const DerivedT&>(*this);
    }

public:
    std::string name; ///< Operation name for debugging
    DataType compute_data_type = DataType::NOT_SET; ///< Compute/accumulation data type (NOLINT)

    // First-wins latch for source-compatibility setters that have no hipDNN
    // equivalent; set via recordUnsupported(), read via hasUnsupportedUsage() /
    // getUnsupportedReason(), and surfaced by the owning node at validate().
    // Public (not private) only because the attribute types are C++17 aggregates
    // brace-initialized in non-friend contexts, which a non-public base member
    // would break; prefer the accessors over touching it directly.
    std::optional<std::string> unsupported_reason; // NOLINT(readability-identifier-naming)

    /**
     * @brief Set the operation name
     * @param nameValue The name to assign
     * @return Reference to derived class for method chaining
     */
    DerivedT& set_name(const std::string& nameValue) // NOLINT(readability-identifier-naming)
    {
        name = nameValue;
        return self();
    }

    /**
     * @brief Get the operation name
     * @return The current operation name
     */
    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return name;
    }

    /**
     * @brief Set the compute data type for this operation
     * @param value The data type to use for computation/accumulation
     * @return Reference to derived class for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    DerivedT& set_compute_data_type(DataType value)
    {
        compute_data_type = value;
        return self();
    }

    /**
     * @brief Get the compute data type
     * @return The current compute data type
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_compute_data_type() const
    {
        return compute_data_type;
    }

    /**
     * @brief Whether a source-compatibility setter recorded an unsupported request
     * @return true if a setter with no hipDNN equivalent was invoked
     *
     * Setters that accept a foreign (e.g. cuDNN) spelling with no hipDNN
     * behavior record the request instead of failing at the call site; the
     * owning node surfaces it as an error at validate().
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool hasUnsupportedUsage() const
    {
        return unsupported_reason.has_value();
    }

    /**
     * @brief The reason recorded by the first unsupported setter, if any
     * @return The recorded message, or an empty string when none was recorded
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::string& getUnsupportedReason() const
    {
        static const std::string s_empty;
        return unsupported_reason ? *unsupported_reason : s_empty;
    }

    /**
     * @brief Fill unset properties from graph context
     * @param graphAttributes The graph attributes to copy from
     * @return Error indicating success or failure
     *
     * Propagates graph-level settings to all input and output tensors
     * that don't have explicit values set. Also sets the compute data
     * type if not already specified.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error fill_from_context(const GraphAttributes& graphAttributes)
    {
        for(auto& [_, tensor] : self().inputs)
        {
            if(tensor)
            {
                tensor->fill_from_context(graphAttributes);
            }
        }

        for(auto& [_, tensor] : self().outputs)
        {
            if(tensor)
            {
                tensor->fill_from_context(graphAttributes);
            }
        }

        if(get_compute_data_type() == DataType::NOT_SET)
        {
            set_compute_data_type(graphAttributes.get_compute_data_type());
        }

        return {};
    }

    /**
     * @brief High-level structural comparison of operations using CRTP.
     * Compares core attributes common to all nodes, then delegates specific
     * property evaluation to the derived node subclass.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool logicallyEquals(const Attributes<DerivedT>& other) const
    {
        // Core mathematical metadata configuration must match across all nodes
        if(this->compute_data_type != other.compute_data_type)
        {
            return false;
        }

        // Core Map logical validation handled here natively
        if(!compareMapsLogical(self().inputs, other.self().inputs)
           || !compareMapsLogical(self().outputs, other.self().outputs))
        {
            return false;
        }

        // Delegate ONLY custom extended properties to the derived hook
        return self().logicallyEqualsImpl(other.self());
    }

    /**
     * @brief Global strict equality operator inside Attributes.hpp base class.
     * Handles layout validation for ALL derived classes dynamically without redundancy.
     */
    friend bool operator==(const Attributes<DerivedT>& lhs, const Attributes<DerivedT>& rhs)
    {
        // Cast down to the actual concrete type
        const auto& derivedLhs = static_cast<const DerivedT&>(lhs);
        const auto& derivedRhs = static_cast<const DerivedT&>(rhs);

        // Check basic non-tensor base properties
        if(lhs.compute_data_type != rhs.compute_data_type || lhs.name != rhs.name)
        {
            return false;
        }

        const Attributes<DerivedT>& baseLhsView = derivedLhs;
        const Attributes<DerivedT>& baseRhsView = derivedRhs;

        if(!compareMapsStrict(baseLhsView.self().inputs, baseRhsView.self().inputs)
           || !compareMapsStrict(baseLhsView.self().outputs, baseRhsView.self().outputs))
        {
            return false;
        }

        // Delegate strict check of extended fields to derived hook
        return derivedLhs.strictEqualsImpl(derivedRhs);
    }

    friend bool operator!=(const Attributes<DerivedT>& lhs, const Attributes<DerivedT>& rhs)
    {
        return !(lhs == rhs);
    }

private:
    Attributes() = default;

protected:
    // Default fallback hooks for derived classes that do NOT have extra fields
    bool logicallyEqualsImpl([[maybe_unused]] const DerivedT& other) const
    {
        return false;
    }
    bool strictEqualsImpl([[maybe_unused]] const DerivedT& other) const
    {
        return false;
    }

    /**
     * @brief Get an input tensor by name
     * @tparam InputNameT The input name enum or type
     * @param inputName The input identifier
     * @return Shared pointer to the tensor, or nullptr if not found
     */
    template <typename InputNameT>
    std::shared_ptr<TensorAttributes> getInput(InputNameT inputName) const
    {
        auto it = self().inputs.find(inputName);
        if(it != self().inputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    /**
     * @brief Get an output tensor by name
     * @tparam OutputNameT The output name enum or type
     * @param outputName The output identifier
     * @return Shared pointer to the tensor, or nullptr if not found
     */
    template <typename OutputNameT>
    std::shared_ptr<TensorAttributes> getOutput(OutputNameT outputName) const
    {
        auto it = self().outputs.find(outputName);
        if(it != self().outputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    /**
     * @brief Set an input tensor by name (copy)
     * @tparam InputNameT The input name enum or type
     * @param inputName The input identifier
     * @param value The tensor attributes to set
     * @return Reference to derived class for method chaining
     */
    template <typename InputNameT>
    DerivedT& setInput(InputNameT inputName, const std::shared_ptr<TensorAttributes>& value)
    {
        self().inputs[inputName] = value;
        return self();
    }

    /**
     * @brief Set an input tensor by name (move)
     * @tparam InputNameT The input name enum or type
     * @param inputName The input identifier
     * @param value The tensor attributes to move
     * @return Reference to derived class for method chaining
     */
    template <typename InputNameT>
    DerivedT& setInput(InputNameT inputName, std::shared_ptr<TensorAttributes>&& value)
    {
        self().inputs[inputName] = std::move(value);
        return self();
    }

    /**
     * @brief Set an output tensor by name (copy)
     * @tparam OutputNameT The output name enum or type
     * @param outputName The output identifier
     * @param value The tensor attributes to set
     * @return Reference to derived class for method chaining
     */
    template <typename OutputNameT>
    DerivedT& setOutput(OutputNameT outputName, const std::shared_ptr<TensorAttributes>& value)
    {
        self().outputs[outputName] = value;
        return self();
    }

    /**
     * @brief Set an output tensor by name (move)
     * @tparam OutputNameT The output name enum or type
     * @param outputName The output identifier
     * @param value The tensor attributes to move
     * @return Reference to derived class for method chaining
     */
    template <typename OutputNameT>
    DerivedT& setOutput(OutputNameT outputName, std::shared_ptr<TensorAttributes>&& value)
    {
        self().outputs[outputName] = std::move(value);
        return self();
    }

    /**
     * @brief Record the first unsupported source-compatibility request (first wins)
     * @param reason Human-readable description of what is unsupported
     * @return Reference to derived class for method chaining
     *
     * Latches the first reason only; later calls are no-ops so the earliest
     * offending setter is the one surfaced at validate().
     */
    DerivedT& recordUnsupported(const char* reason)
    {
        if(!unsupported_reason.has_value())
        {
            unsupported_reason = reason;
        }
        return self();
    }

private:
    /**
     * @brief Performs a logical/semantic equality check between two maps of attribute pointers.
     * * Iterates over keys and evaluates whether their underlying values are functionally
     * equivalent by routing the evaluation down to their custom `logicallyEquals` implementation.
     * This bypasses rigid layout parameters (like memory strides) in favor of graph state matching.
     * * @tparam MapT The map collection type (e.g., std::unordered_map or std::map).
     * @param m1 The primary map instance to compare.
     * @param m2 The secondary map instance to compare against.
     * @return true If both maps represent the same functional mathematical state.
     * @return false If structural layouts or logical evaluations mismatch.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    template <typename MapT>
    static bool compareMapsLogical(const MapT& m1, const MapT& m2)
    {
        if(m1.size() != m2.size())
        {
            return false;
        }
        // NOLINTNEXTLINE(readability-identifier-naming)
        for(const auto& [key, t1] : m1)
        {
            auto it = m2.find(key);
            if(it == m2.end())
            {
                return false;
            }
            // Both are unassigned or null; equivalent semantic state
            if(!t1 && !it->second)
            {
                continue;
            }
            // Mismatched pointer presence, or structural logical validation fails
            if(!t1 || !it->second || !t1->logicallyEquals(*it->second))
            {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Performs a strict equality check between two maps of descriptor pointers.
     * * Compares sizes, keys, and dereferenced values. Two maps are strictly equal if they
     * contain the identical set of keys and their corresponding non-null pointers point
     * to objects that satisfy the binary `operator==` check.
     * * @tparam MapT The map collection type (e.g., std::unordered_map or std::map).
     * @param m1 The primary map instance to compare.
     * @param m2 The secondary map instance to compare against.
     * @return true If both maps have identical structures and strict value matching.
     * @return false If sizes, keys, nullability states, or raw evaluations mismatch.
     */
    template <typename MapT>
    static bool compareMapsStrict(const MapT& m1, const MapT& m2)
    {
        if(m1.size() != m2.size())
        {
            return false;
        }

        for(const auto& [key, t1] : m1)
        {
            auto it = m2.find(key);
            if(it == m2.end())
            {
                return false;
            }
            // Both are null pointers; they match logically and strictly
            if(!t1 && !it->second)
            {
                continue;
            }
            // One is null while the other isn't, or underlying values differ strictly
            if(!t1 || !it->second || !(*t1 == *it->second))
            {
                return false;
            }
        }

        return true;
    }
};

} // namespace hipdnn_frontend::graph
