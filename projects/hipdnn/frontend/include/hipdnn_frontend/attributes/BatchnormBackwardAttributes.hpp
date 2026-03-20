// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BatchnormBackwardAttributes.hpp
 * @brief Attributes for batch normalization backward pass operation
 *
 * This file defines the BatchnormBackwardAttributes class used to configure
 * the backward pass (gradient computation) of batch normalization.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class BatchnormBackwardAttributes
 * @brief Configuration attributes for batch normalization backward pass
 *
 * BatchnormBackwardAttributes configures the backward pass of batch normalization,
 * computing gradients with respect to the input, scale, and bias.
 *
 * **Required inputs:**
 * - DY: Gradient of the loss with respect to the output (upstream gradient)
 * - X: Original input tensor from forward pass
 * - Scale: Per-channel scale (gamma) tensor
 *
 * **Optional inputs (from forward pass):**
 * - Mean: Saved mean from forward pass
 * - Inv_variance: Saved inverse variance from forward pass
 *
 * **Outputs:**
 * - DX: Gradient with respect to input
 * - DScale: Gradient with respect to scale (gamma)
 * - DBias: Gradient with respect to bias (beta)
 *
 * @code{.cpp}
 * BatchnormBackwardAttributes attr;
 * attr.set_saved_mean_and_inv_variance(savedMean, savedInvVar);
 *
 * auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, attr);
 * @endcode
 *
 * @see BatchnormAttributes for forward pass
 */
class BatchnormBackwardAttributes : public Attributes<BatchnormBackwardAttributes>
{
public:
    BatchnormBackwardAttributes() = default;

    enum class InputNames
    {
        DY = 0,
        X = 1,
        SCALE = 2,
        MEAN = 3,
        INV_VARIANCE = 4
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        DX = 0,
        DSCALE = 1,
        DBIAS = 2
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::vector<std::shared_ptr<TensorAttributes>> peer_stats;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
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
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(InputNames::MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getInput(InputNames::INV_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dscale() const
    {
        return getOutput(OutputNames::DSCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(OutputNames::DBIAS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<std::shared_ptr<TensorAttributes>>& get_peer_stats() const
    {
        return peer_stats;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INV_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dscale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DSCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dscale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DSCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DBIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DBIAS, std::move(value));
    }
    BatchnormBackwardAttributes&
        set_peer_stats(const std::vector<std::shared_ptr<TensorAttributes>>& value) // NOLINT
    {
        peer_stats = value;
        return *this;
    }
    BatchnormBackwardAttributes&
        set_peer_stats(std::vector<std::shared_ptr<TensorAttributes>>&& value) // NOLINT
    {
        peer_stats = std::move(value);
        return *this;
    }

    BatchnormBackwardAttributes&
        set_saved_mean_and_inv_variance(const std::shared_ptr<TensorAttributes>& mean, // NOLINT
                                        const std::shared_ptr<TensorAttributes>& invVariance)
    {
        return set_mean(mean).set_inv_variance(invVariance);
    }
    BatchnormBackwardAttributes&
        set_saved_mean_and_inv_variance(std::shared_ptr<TensorAttributes>&& mean, // NOLINT
                                        std::shared_ptr<TensorAttributes>&& invVariance)
    {
        return set_mean(std::move(mean)).set_inv_variance(std::move(invVariance));
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto peerStatsVector = std::vector<int64_t>{};
        for(const auto& peerStat : peer_stats)
        {
            if(peerStat)
            {
                peerStatsVector.emplace_back(peerStat->get_uid());
            }
        }

        auto mean = get_mean();
        auto invVariance = get_inv_variance();

        return hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributesDirect(
            builder,
            get_dy()->get_uid(),
            get_x()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            invVariance ? flatbuffers::Optional<int64_t>(invVariance->get_uid())
                        : flatbuffers::nullopt,
            get_scale()->get_uid(),
            &peerStatsVector,
            get_dx()->get_uid(),
            get_dscale()->get_uid(),
            get_dbias()->get_uid());
    }

    static BatchnormBackwardAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        BatchnormBackwardAttributes attr;

        attr.set_dy(tensorMap.at(fb->dy_tensor_uid()));
        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_scale(tensorMap.at(fb->scale_tensor_uid()));

        if(fb->mean_tensor_uid().has_value())
        {
            attr.set_mean(tensorMap.at(fb->mean_tensor_uid().value()));
        }
        if(fb->inv_variance_tensor_uid().has_value())
        {
            attr.set_inv_variance(tensorMap.at(fb->inv_variance_tensor_uid().value()));
        }

        std::vector<std::shared_ptr<TensorAttributes>> peerStats;
        if(fb->peer_stats_tensor_uid() != nullptr)
        {
            for(auto uid : *fb->peer_stats_tensor_uid())
            {
                peerStats.push_back(tensorMap.at(uid));
            }
        }
        attr.set_peer_stats(peerStats);

        attr.set_dx(tensorMap.at(fb->dx_tensor_uid()));
        attr.set_dscale(tensorMap.at(fb->dscale_tensor_uid()));
        attr.set_dbias(tensorMap.at(fb->dbias_tensor_uid()));

        return attr;
    }
};
typedef BatchnormBackwardAttributes Batchnorm_backward_attributes;
} // namespace hipdnn_frontend::graph
