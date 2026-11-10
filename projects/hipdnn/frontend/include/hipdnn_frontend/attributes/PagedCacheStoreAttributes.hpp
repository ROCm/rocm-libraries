// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file PagedCacheStoreAttributes.hpp
 * @brief Attributes for PagedCacheStore operations
 *
 * This file defines the PagedCacheStoreAttributes class for configuring
 * PagedCacheStore operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class PagedCacheStoreAttributes
 * @brief Configuration for PagedCacheStore operations
 */
class PagedCacheStoreAttributes : public Attributes<PagedCacheStoreAttributes>
{
public:
    PagedCacheStoreAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0,
        SLOT_MAPPING = 1,
        SCALE = 2
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        CONTAINER = 0
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the slot_mapping input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_slot_mapping() const
    {
        return getInput(InputNames::SLOT_MAPPING);
    }
    /// @brief Get the scale input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    /// @brief Get the container output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_container() const
    {
        return getOutput(OutputNames::CONTAINER);
    }

    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the slot_mapping input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_slot_mapping(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SLOT_MAPPING, std::move(value));
    }
    /// @brief Set the slot_mapping input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_slot_mapping(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SLOT_MAPPING, value);
    }
    /// @brief Set the scale input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    /// @brief Set the scale input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    /// @brief Set the container output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_container(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::CONTAINER, std::move(value));
    }
    /// @brief Set the container output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    PagedCacheStoreAttributes& set_container(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::CONTAINER, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // No extra fields beyond tensors: defer entirely to the base class's
    // tensor/metadata comparison.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool logicallyEqualsImpl([[maybe_unused]] const PagedCacheStoreAttributes& other) const
    {
        return true;
    }
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool strictEqualsImpl([[maybe_unused]] const PagedCacheStoreAttributes& other) const
    {
        return true;
    }
};
} // namespace hipdnn_frontend::graph
