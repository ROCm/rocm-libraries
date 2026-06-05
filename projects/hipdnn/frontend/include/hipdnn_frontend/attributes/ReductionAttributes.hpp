// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file ReductionAttributes.hpp
 * @brief Attributes for reduction operations
 *
 * This file defines the ReductionAttributes class for configuring reduction
 * operations that reduce a tensor along all dimensions.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class ReductionAttributes
 * @brief Configuration for reduction operations
 *
 * ReductionAttributes specifies the parameters for reduction operations that
 * reduce an input tensor to a scalar or lower-dimensional tensor.
 *
 * @code{.cpp}
 * // Sum reduction
 * auto y = graph.reduction(x, ReductionAttributes()
 *              .set_mode(ReductionMode::ADD));
 *
 * // Absolute maximum reduction
 * auto y = graph.reduction(x, ReductionAttributes()
 *              .set_mode(ReductionMode::AMAX)
 *              .set_name("amax_op"));
 * @endcode
 *
 * @see Graph::reduction(), ReductionMode
 */
class ReductionAttributes : public Attributes<ReductionAttributes>
{
public:
    ReductionAttributes() = default;

    /// Input tensor identifiers
    // NOLINTNEXTLINE(readability-identifier-naming)
    enum class input_names
    {
        X = 0, ///< Input tensor
    };

    /// Output tensor identifiers
    // NOLINTNEXTLINE(readability-identifier-naming)
    enum class output_names
    {
        Y = 0, ///< Output tensor
    };

    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    std::optional<ReductionMode> mode = std::nullopt; ///< The reduction mode
    bool is_deterministic = false; ///< Whether to use deterministic algorithms
    // NOLINTEND(readability-identifier-naming)

    /// @brief Get the reduction mode
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<ReductionMode> get_mode() const
    {
        return mode;
    }

    /**
     * @brief Set the reduction operation mode
     * @param value The reduction operation to perform (see ReductionMode enum)
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ReductionAttributes& set_mode(ReductionMode value)
    {
        mode = value;
        return *this;
    }

    /// @brief Get whether deterministic algorithms are requested
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_is_deterministic() const
    {
        return is_deterministic;
    }

    /**
     * @brief Request deterministic reduction algorithms
     * @param value True to require deterministic behavior
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ReductionAttributes& set_is_deterministic(bool value)
    {
        is_deterministic = value;
        return *this;
    }

    /// @brief Get the input tensor X
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(input_names::X);
    }

    /// @brief Get the output tensor Y
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(output_names::Y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ReductionAttributes& set_x(const std::shared_ptr<TensorAttributes>& x)
    {
        return setInput(input_names::X, x);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ReductionAttributes& set_x(std::shared_ptr<TensorAttributes>&& x)
    {
        return setInput(input_names::X, std::move(x));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ReductionAttributes& set_y(const std::shared_ptr<TensorAttributes>& y)
    {
        return setOutput(output_names::Y, y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ReductionAttributes& set_y(std::shared_ptr<TensorAttributes>&& y)
    {
        return setOutput(output_names::Y, std::move(y));
    }
};

typedef ReductionAttributes Reduction_attributes; // NOLINT(readability-identifier-naming)

} // namespace hipdnn_frontend::graph
