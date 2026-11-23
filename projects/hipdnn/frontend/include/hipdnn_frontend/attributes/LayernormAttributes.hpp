// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file LayernormAttributes.hpp
 * @brief Attributes for layer normalization operation
 *
 * This file defines the LayernormAttributes class used to configure
 * layer normalization operations, which normalize across the feature dimension.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class LayernormAttributes
 * @brief Configuration attributes for layer normalization
 *
 * LayernormAttributes configures a layer normalization operation.
 * Unlike batch normalization which normalizes across the batch dimension,
 * layer normalization normalizes across the last k feature dimensions.
 *
 * The number of normalized dimensions (k) is determined by:
 * 1. **Explicit**: `normalized_dim_count` k, when set (> 0), wins outright.
 * 2. **From scale shape**: a scale of lower rank than X is treated as
 *    right-aligned to X's rank by prepending 1s. k is then `rank(X) - i`
 *    for the first dimension i (0-indexed, up to but excluding X's last
 *    dimension) whose right-aligned scale value is not 1; if every such
 *    dimension is 1, k = 1. A lower-rank scale with no dimension other
 *    than 1 instead gives k equal to the scale's own rank. For example,
 *    X `(2, 3, 4, 5)` with scale `(1, 5)` right-aligns to `(1, 1, 1, 5)`,
 *    so k = 1; scale `(4, 5)` right-aligns to `(1, 1, 4, 5)`, so k = 2.
 *    A full-rank scale whose first non-1 dimension is not the batch
 *    dimension, such as `(1, C, 1, 1)` on a 4-D X, derives a k that
 *    would span non-trailing feature dimensions; no hipDNN provider
 *    normalizes non-trailing axes, so validation rejects such a graph
 *    (see "Graph validation" in docs/KNOWN_DIVERGENCES.md).
 * 3. **Default**: If scale dims are also unset, they are inferred as
 *    `[1, D1, ..., Dk]` (batch dim = 1), normalizing all non-batch dims.
 *
 * Unset Scale and Bias dims are inferred during validation as
 * `[1, D1, ..., Dk]`, as cuDNN does, or, when `normalized_dim_count` k is set
 * explicitly, as 1s followed by the last k dims of X. When only one of Scale and
 * Bias has dims set, they must equal the dims inferred for the other.
 *
 * **Tensor Shapes:**
 * - **X** (input): `(N, D1, D2, ..., Dk)` — batch first, then feature dims
 * - **Scale, Bias**: Full-rank `(1, D1, ..., Dk)` with batch dim = 1,
 *   or reduced-rank `(D1, ..., Dk)` with batch dims omitted
 * - **Y** (output): Same shape as X
 * - **Mean, InvVariance**: Batch dims from input, normalized dims = 1
 *
 * **Required inputs:**
 * - X: Input tensor to normalize
 * - Scale: Per-feature scale (gamma) tensor
 * - Bias: Per-feature bias (beta) tensor
 * - Epsilon: Small constant for numerical stability (scalar tensor)
 *
 * **Outputs:**
 * - Y: Normalized output tensor
 * - Mean: Computed mean (optional, training phase only)
 * - InvVariance: Computed inverse variance (1/sqrt(var + epsilon)) (optional, training phase only)
 *
 * @code{.cpp}
 * LayernormAttributes attr;
 * attr.set_forward_phase(NormFwdPhase::INFERENCE);
 * attr.set_epsilon(epsilon);
 *
 * auto [y, mean, inv_variance] = graph.layernorm(x, scale, bias, attr);
 * @endcode
 */
class LayernormAttributes : public Attributes<LayernormAttributes>
{
public:
    LayernormAttributes() = default;

    enum class InputNames
    {
        X = 0,
        SCALE = 1,
        BIAS = 2,
        EPSILON = 3
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        MEAN = 1,
        INV_VARIANCE = 2
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::BIAS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_epsilon() const
    {
        return getInput(InputNames::EPSILON);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getOutput(OutputNames::MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getOutput(OutputNames::INV_VARIANCE);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_forward_phase(NormFwdPhase value)
    {
        _forwardPhase = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    NormFwdPhase get_forward_phase() const
    {
        return _forwardPhase;
    }

    /// @cond INTERNAL
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_normalized_dim_count(int64_t value)
    {
        _normalizedDimCount = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_normalized_dim_count() const
    {
        return _normalizedDimCount;
    }
    /// @endcond

    /**
     * @brief Custom equality hook for layer-norm-specific attributes
     *
     * Compares the forward phase and normalized dimension count — both of
     * which define the semantics of the normalization rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const LayernormAttributes& other) const
    {
        return _forwardPhase == other._forwardPhase
               && _normalizedDimCount == other._normalizedDimCount;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const LayernormAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }

private:
    int64_t _normalizedDimCount = 0;
    NormFwdPhase _forwardPhase = NormFwdPhase::NOT_SET;
};

typedef LayernormAttributes Layernorm_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
