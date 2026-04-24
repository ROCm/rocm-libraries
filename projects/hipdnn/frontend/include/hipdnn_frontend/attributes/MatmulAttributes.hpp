// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file MatmulAttributes.hpp
 * @brief Attributes for matrix multiplication operations
 *
 * This file defines the MatmulAttributes class for configuring matrix
 * multiplication (matmul) operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class MatmulAttributes
 * @brief Configuration for matrix multiplication operations
 *
 * MatmulAttributes specifies the parameters for a matrix multiplication
 * operation: C = A @ B. Supports batched operations when tensors have
 * leading batch dimensions.
 *
 * **Tensor Shapes:**
 * - **A** (left input): `(...batch, M, K)` — leading batch dims, rows, columns
 * - **B** (right input): `(...batch, K, N)` — K must match A's last dim
 * - **C** (output): `(...batch, M, N)` — batch dims are broadcast
 *
 * Batch dimensions support broadcasting (dims must be equal or divisible).
 *
 * @code{.cpp}
 * // Matrix multiplication: C = A @ B
 * auto c = graph.matmul(a, b, MatmulAttributes());
 * @endcode
 *
 * @see Graph::matmul()
 */
class MatmulAttributes : public Attributes<MatmulAttributes>
{
public:
    MatmulAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        A = 0, ///< Left input matrix
        B = 1 ///< Right input matrix
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        C = 0 ///< Output matrix
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    /// @brief Get the left input matrix A
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_a() const
    {
        return getInput(InputNames::A);
    }
    /// @brief Get the right input matrix B
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_b() const
    {
        return getInput(InputNames::B);
    }
    /// @brief Get the output matrix C
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_c() const
    {
        return getOutput(OutputNames::C);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_a(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::A, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_a(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::A, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_b(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::B, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_b(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::B, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_c(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::C, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_c(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::C, std::move(value));
    }
};

typedef MatmulAttributes Matmul_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
