// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BatchnormInferenceAttributesVarianceExt.hpp
 * @brief Attributes for batch normalization inference with variance and epsilon tensors
 *
 * This file defines the BatchnormInferenceAttributesVarianceExt class for
 * configuring batch normalization during inference using raw variance and
 * epsilon as separate input tensors (rather than pre-computed inverse variance).
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class BatchnormInferenceAttributesVarianceExt
 * @brief Configuration for batch normalization inference with variance and epsilon
 *
 * Extended variant of BatchnormInferenceAttributes that accepts raw variance
 * and an epsilon tensor instead of pre-computed inverse variance:
 *
 * y = scale * (x - mean) / sqrt(variance + epsilon) + bias
 *
 * **Required inputs:**
 * - X: Input activation tensor
 * - Mean: Pre-computed running mean
 * - Variance: Pre-computed running variance
 * - Scale: Scale (gamma) parameter
 * - Bias: Bias (beta) parameter
 * - Epsilon: Small constant for numerical stability (scalar tensor)
 *
 * **Outputs:**
 * - Y: Normalized output tensor
 *
 * @code{.cpp}
 * auto y = graph.batchnorm_inference_variance_ext(x, mean, variance, scale,
 *              bias, epsilon, BatchnormInferenceAttributesVarianceExt());
 * @endcode
 *
 * @see Graph::batchnorm_inference_variance_ext(), BatchnormInferenceAttributes
 */
class BatchnormInferenceAttributesVarianceExt
    : public Attributes<BatchnormInferenceAttributesVarianceExt>
{
public:
    BatchnormInferenceAttributesVarianceExt() = default;

    enum class InputNames
    {
        X = 0,
        MEAN = 1,
        VARIANCE = 2,
        SCALE = 3,
        BIAS = 4,
        EPSILON = 5
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0
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
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(InputNames::MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_variance() const
    {
        return getInput(InputNames::VARIANCE);
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
    BatchnormInferenceAttributesVarianceExt& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    BatchnormInferenceAttributesVarianceExt&
        set_mean(const std::shared_ptr<TensorAttributes>& value) // NOLINT
    {
        return setInput(InputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MEAN, std::move(value));
    }
    BatchnormInferenceAttributesVarianceExt&
        set_variance(const std::shared_ptr<TensorAttributes>& value) // NOLINT
    {
        return setInput(InputNames::VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::VARIANCE, std::move(value));
    }
    BatchnormInferenceAttributesVarianceExt&
        set_scale(const std::shared_ptr<TensorAttributes>& value) // NOLINT
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    BatchnormInferenceAttributesVarianceExt&
        set_bias(const std::shared_ptr<TensorAttributes>& value) // NOLINT
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    BatchnormInferenceAttributesVarianceExt&
        set_epsilon(const std::shared_ptr<TensorAttributes>& value) // NOLINT
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributesVarianceExt& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }

private:
    std::shared_ptr<TensorAttributes> getInput(InputNames name) const
    {
        auto it = inputs.find(name);
        if(it != inputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    std::shared_ptr<TensorAttributes> getOutput(OutputNames name) const
    {
        auto it = outputs.find(name);
        if(it != outputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    BatchnormInferenceAttributesVarianceExt&
        setInput(InputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    BatchnormInferenceAttributesVarianceExt& setInput(InputNames name,
                                                      std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    BatchnormInferenceAttributesVarianceExt&
        setOutput(OutputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    BatchnormInferenceAttributesVarianceExt& setOutput(OutputNames name,
                                                       std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};
typedef BatchnormInferenceAttributesVarianceExt Batchnorm_inference_attributes_variance_ext;
} // namespace hipdnn_frontend::graph
