// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file MoeGroupedMatmulBwdAttributes.hpp
 * @brief Attributes for MoeGroupedMatmulBwd operations
 *
 * This file defines the MoeGroupedMatmulBwdAttributes class for configuring
 * MoeGroupedMatmulBwd operations in hipDNN computational graphs.
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
 * @class MoeGroupedMatmulBwdAttributes
 * @brief Configuration for MoeGroupedMatmulBwd operations
 */
class MoeGroupedMatmulBwdAttributes : public Attributes<MoeGroupedMatmulBwdAttributes>
{
public:
    MoeGroupedMatmulBwdAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        DOUTPUT = 0,
        TOKEN = 1,
        FIRST_TOKEN_OFFSET = 2
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        DWEIGHT = 0
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the doutput input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_doutput() const
    {
        return getInput(InputNames::DOUTPUT);
    }
    /// @brief Get the token input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_token() const
    {
        return getInput(InputNames::TOKEN);
    }
    /// @brief Get the first_token_offset input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_first_token_offset() const
    {
        return getInput(InputNames::FIRST_TOKEN_OFFSET);
    }
    /// @brief Get the dweight output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dweight() const
    {
        return getOutput(OutputNames::DWEIGHT);
    }

    /// @brief Set the doutput input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_doutput(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DOUTPUT, std::move(value));
    }
    /// @brief Set the doutput input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_doutput(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DOUTPUT, value);
    }
    /// @brief Set the token input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_token(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::TOKEN, std::move(value));
    }
    /// @brief Set the token input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_token(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::TOKEN, value);
    }
    /// @brief Set the first_token_offset input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_first_token_offset(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::FIRST_TOKEN_OFFSET, std::move(value));
    }
    /// @brief Set the first_token_offset input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes&
        set_first_token_offset(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::FIRST_TOKEN_OFFSET, value);
    }
    /// @brief Set the dweight output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_dweight(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DWEIGHT, std::move(value));
    }
    /// @brief Set the dweight output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulBwdAttributes& set_dweight(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DWEIGHT, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // No extra fields beyond tensors: defer entirely to the base class's
    // tensor/metadata comparison.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool logicallyEqualsImpl([[maybe_unused]] const MoeGroupedMatmulBwdAttributes& other) const
    {
        return true;
    }
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool strictEqualsImpl([[maybe_unused]] const MoeGroupedMatmulBwdAttributes& other) const
    {
        return true;
    }
};
typedef MoeGroupedMatmulBwdAttributes
    Moe_grouped_matmul_bwd_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
