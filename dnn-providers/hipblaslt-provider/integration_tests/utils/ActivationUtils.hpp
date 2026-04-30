// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <optional>

#include <hipdnn_frontend/Types.hpp>

namespace test_activation_common
{

struct ActivTestCase
{
    hipdnn_frontend::PointwiseMode mode = hipdnn_frontend::PointwiseMode::NOT_SET;
    std::optional<float> reluLowerClip = std::nullopt;
    std::optional<float> reluUpperClip = std::nullopt;
    std::optional<float> swishBeta = std::nullopt;

    ActivTestCase(hipdnn_frontend::PointwiseMode mode_,
                  std::optional<float> reluLowerClip_ = std::nullopt,
                  std::optional<float> reluUpperClip_ = std::nullopt,
                  std::optional<float> swishBeta_ = std::nullopt)
        : mode(mode_)
        , reluLowerClip(reluLowerClip_)
        , reluUpperClip(reluUpperClip_)
        , swishBeta(swishBeta_)
    {
        using PM = hipdnn_frontend::PointwiseMode;

        switch(mode)
        {
        case PM::NOT_SET:
        case PM::RELU_FWD:
        case PM::GELU_APPROX_TANH_FWD:
        case PM::SWISH_FWD:
            break;
        default:
            throw std::invalid_argument("Unknown activation mode");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const ActivTestCase& tc)
    {
        ss << "(mode:" << hipdnn_frontend::to_string(tc.mode);
        if(tc.reluLowerClip)
        {
            ss << " reluLowerClip:" << tc.reluLowerClip.value();
        }
        if(tc.reluUpperClip)
        {
            ss << " reluUpperClip:" << tc.reluUpperClip.value();
        }
        if(tc.swishBeta)
        {
            ss << " swishBeta:" << tc.swishBeta.value();
        }
        ss << ")";

        return ss;
    }
};

inline std::vector<ActivTestCase> createFwdActivationCases()
{
    using PM = hipdnn_frontend::PointwiseMode;

    std::vector<ActivTestCase> cases;

    // RELU_FWD (standard ReLU)
    cases.emplace_back(PM::RELU_FWD, 0.0f, std::nullopt);

    // CLAMP: both lower and upper clips
    cases.emplace_back(PM::RELU_FWD, 0.1f, 0.5f);

    // GELU
    cases.emplace_back(PM::GELU_APPROX_TANH_FWD);

    // SWISH
    cases.emplace_back(PM::SWISH_FWD, std::nullopt, std::nullopt, 1.0f);

    return cases;
}

} // namespace test_activation_common
