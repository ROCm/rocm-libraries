// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file LayernormFpropAttributes.hpp
 * @brief Attributes for layer normalization forward operation
 *
 * This file defines the LayernormFpropAttributes class used to configure
 * layer normalization operations, which normalize across the feature dimension.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/layernorm_fprop_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class LayernormFpropAttributes
 * @brief Configuration attributes for layer normalization forward pass
 *
 * LayernormFpropAttributes configures a layer normalization operation.
 * Unlike batch normalization which normalizes across the batch dimension,
 * layer normalization normalizes across the feature dimensions.
 *
 * **Required inputs:**
 * - X: Input tensor to normalize
 * - Epsilon: Small constant for numerical stability (scalar tensor)
 *
 * **Optional inputs:**
 * - Scale: Per-feature scale (gamma) tensor
 * - Bias: Per-feature bias (beta) tensor
 *
 * **Outputs:**
 * - Y: Normalized output tensor
 * - Mean: Computed mean (optional)
 * - Rstd: Computed reciprocal standard deviation (1/sqrt(var + epsilon)) (optional)
 *
 * @code{.cpp}
 * LayernormFpropAttributes attr;
 * attr.set_epsilon(epsilonTensor);
 *
 * auto y = graph.layernorm_fprop(x, scale, bias, attr);
 * @endcode
 */
class LayernormFpropAttributes : public Attributes<LayernormFpropAttributes>
{
public:
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
        RSTD = 2
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
    std::shared_ptr<TensorAttributes> get_rstd() const
    {
        return getOutput(OutputNames::RSTD);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_rstd(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::RSTD, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormFpropAttributes& set_rstd(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::RSTD, std::move(value));
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::LayernormFpropAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        const auto scale = get_scale();
        const auto bias = get_bias();
        const auto mean = get_mean();
        const auto rstd = get_rstd();

        return hipdnn_data_sdk::data_objects::CreateLayernormFpropAttributes(
            builder,
            get_x()->get_uid(),
            scale ? flatbuffers::Optional<int64_t>(scale->get_uid())
                  : flatbuffers::Optional<int64_t>(flatbuffers::nullopt),
            bias ? flatbuffers::Optional<int64_t>(bias->get_uid())
                 : flatbuffers::Optional<int64_t>(flatbuffers::nullopt),
            get_epsilon()->get_uid(),
            get_y()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid())
                 : flatbuffers::Optional<int64_t>(flatbuffers::nullopt),
            rstd ? flatbuffers::Optional<int64_t>(rstd->get_uid())
                 : flatbuffers::Optional<int64_t>(flatbuffers::nullopt));
    }

    static LayernormFpropAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::LayernormFpropAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        LayernormFpropAttributes attr;

        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_epsilon(tensorMap.at(fb->epsilon_tensor_uid()));
        attr.set_y(tensorMap.at(fb->y_tensor_uid()));

        if(fb->scale_tensor_uid().has_value())
        {
            attr.set_scale(tensorMap.at(fb->scale_tensor_uid().value()));
        }
        if(fb->bias_tensor_uid().has_value())
        {
            attr.set_bias(tensorMap.at(fb->bias_tensor_uid().value()));
        }
        if(fb->mean_tensor_uid().has_value())
        {
            attr.set_mean(tensorMap.at(fb->mean_tensor_uid().value()));
        }
        if(fb->rstd_tensor_uid().has_value())
        {
            attr.set_rstd(tensorMap.at(fb->rstd_tensor_uid().value()));
        }

        return attr;
    }
};

typedef LayernormFpropAttributes LayernormFprop_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
