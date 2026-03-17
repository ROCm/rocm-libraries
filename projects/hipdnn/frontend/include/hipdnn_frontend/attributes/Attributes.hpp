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

private:
    Attributes() = default;

protected:
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
};

} // namespace hipdnn_frontend::graph
