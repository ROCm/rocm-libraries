// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_sdk/data_objects/batchnorm_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class BatchnormAttributes : public Attributes<BatchnormAttributes>
{
public:
    enum class InputNames
    {
        X = 0,
        SCALE = 1,
        BIAS = 2,
        PREV_RUNNING_MEAN = 3,
        PREV_RUNNING_VARIANCE = 4,
        MOMENTUM = 5,
        EPSILON = 6
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        MEAN = 1,
        INV_VARIANCE = 2,
        NEXT_RUNNING_MEAN = 3,
        NEXT_RUNNING_VARIANCE = 4
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::vector<std::shared_ptr<TensorAttributes>> peer_stats;

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
    const std::vector<std::shared_ptr<TensorAttributes>>& get_peer_stats() const
    {
        return peer_stats;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_prev_running_mean() const
    {
        return getInput(InputNames::PREV_RUNNING_MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_prev_running_variance() const
    {
        return getInput(InputNames::PREV_RUNNING_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_momentum() const
    {
        return getInput(InputNames::MOMENTUM);
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
    std::shared_ptr<TensorAttributes> get_next_running_mean() const
    {
        return getOutput(OutputNames::NEXT_RUNNING_MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_next_running_variance() const
    {
        return getOutput(OutputNames::NEXT_RUNNING_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_peer_stats(const std::vector<std::shared_ptr<TensorAttributes>>& value)
    {
        peer_stats = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_peer_stats(std::vector<std::shared_ptr<TensorAttributes>>&& value)
    {
        peer_stats = std::move(value);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::PREV_RUNNING_MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::PREV_RUNNING_MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::PREV_RUNNING_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::PREV_RUNNING_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_momentum(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::MOMENTUM, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_momentum(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MOMENTUM, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_VARIANCE, std::move(value));
    }
    // NOLINTBEGIN(readability-identifier-naming)
    BatchnormAttributes&
        set_previous_running_stats(const std::shared_ptr<TensorAttributes>& mean,
                                   const std::shared_ptr<TensorAttributes>& variance,
                                   const std::shared_ptr<TensorAttributes>& momentum)
    // NOLINTEND(readability-identifier-naming)
    {
        return set_prev_running_mean(mean).set_prev_running_variance(variance).set_momentum(
            momentum);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_previous_running_stats(std::shared_ptr<TensorAttributes>&& mean,
                                                    std::shared_ptr<TensorAttributes>&& variance,
                                                    std::shared_ptr<TensorAttributes>&& momentum)
    {
        return set_prev_running_mean(std::move(mean))
            .set_prev_running_variance(std::move(variance))
            .set_momentum(std::move(momentum));
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::BatchnormAttributes>
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

        auto prevRunningMean = get_prev_running_mean();
        auto prevRunningVariance = get_prev_running_variance();
        auto momentum = get_momentum();
        auto mean = get_mean();
        auto invVariance = get_inv_variance();
        auto nextRunningMean = get_next_running_mean();
        auto nextRunningVariance = get_next_running_variance();

        return hipdnn_sdk::data_objects::CreateBatchnormAttributesDirect(
            builder,
            get_x()->get_uid(),
            get_scale()->get_uid(),
            get_bias()->get_uid(),
            get_epsilon()->get_uid(),
            &peerStatsVector,
            prevRunningMean ? flatbuffers::Optional<int64_t>(prevRunningMean->get_uid())
                            : flatbuffers::nullopt,
            prevRunningVariance ? flatbuffers::Optional<int64_t>(prevRunningVariance->get_uid())
                                : flatbuffers::nullopt,
            momentum ? flatbuffers::Optional<int64_t>(momentum->get_uid()) : flatbuffers::nullopt,
            get_y()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            invVariance ? flatbuffers::Optional<int64_t>(invVariance->get_uid())
                        : flatbuffers::nullopt,
            nextRunningMean ? flatbuffers::Optional<int64_t>(nextRunningMean->get_uid())
                            : flatbuffers::nullopt,
            nextRunningVariance ? flatbuffers::Optional<int64_t>(nextRunningVariance->get_uid())
                                : flatbuffers::nullopt);
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

    BatchnormAttributes& setInput(InputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    BatchnormAttributes& setInput(InputNames name, std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    BatchnormAttributes& setOutput(OutputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    BatchnormAttributes& setOutput(OutputNames name, std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};

typedef BatchnormAttributes Batchnorm_attributes;
}
}
