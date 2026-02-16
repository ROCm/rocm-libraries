// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <optional>

#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>

namespace test_activation_common
{

struct ActivTestCase
{
    hipdnn_data_sdk::data_objects::PointwiseMode mode;
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;

    ActivTestCase(hipdnn_data_sdk::data_objects::PointwiseMode mode_,
                  std::optional<float> reluLowerClip_ = std::nullopt,
                  std::optional<float> reluUpperClip_ = std::nullopt)
        : mode(mode_)
        , reluLowerClip(reluLowerClip_)
        , reluUpperClip(reluUpperClip_)
    {
        using PointwiseMode = hipdnn_data_sdk::data_objects::PointwiseMode;

        switch(mode)
        {
        case PointwiseMode::RELU_FWD:
        case PointwiseMode::RELU_BWD:
        case PointwiseMode::SIGMOID_FWD:
        case PointwiseMode::SIGMOID_BWD:
        case PointwiseMode::GELU_FWD:
        case PointwiseMode::GELU_BWD:
        case PointwiseMode::SWISH_FWD:
        case PointwiseMode::SWISH_BWD:
            break;
        default:
            throw std::invalid_argument("Unknown activation mode");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const ActivTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(mode:" << tc.mode;
        if(tc.reluLowerClip)
        {
            ss << " reluLowerClip:" << tc.reluLowerClip.value();
        }
        if(tc.reluUpperClip)
        {
            ss << " reluUpperClip:" << tc.reluUpperClip.value();
        }
        ss << ")";

        return ss;
    }
};

inline std::vector<ActivTestCase> createFwdActivationCases()
{
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;

    std::vector<ActivTestCase> cases;

    // RELU_FWD (standard ReLU)
    cases.emplace_back(PM::RELU_FWD,
                       0.0f, // reluLowerClip
                       std::nullopt, // reluUpperClip
    );

    // CLAMP: both lower and upper clips (e.g., clip to range [0.0, 6.0])
    cases.emplace_back(PM::RELU_FWD,
                       0.1f, // reluLowerClip
                       0.5f, // reluUpperClip
    );

    // SIGMOID
    cases.emplace_back(PM::SIGMOID_FWD);

    // GELU
    cases.emplace_back(PM::GELU_FWD);

    // SWISH
    cases.emplace_back(PM::SWISH_FWD);

    return cases;
}

} // namespace test_activation_common
