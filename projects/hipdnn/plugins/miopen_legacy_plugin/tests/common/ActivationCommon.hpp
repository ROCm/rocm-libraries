// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <optional>

#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>

namespace test_activation_common
{

struct ActivTestCase
{
    hipdnn_sdk::data_objects::PointwiseMode mode;
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    std::optional<float> swishBeta;
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;

    ActivTestCase(hipdnn_sdk::data_objects::PointwiseMode modeLocal,
                  std::optional<float> reluLowerClipLocal,
                  std::optional<float> reluUpperClipLocal,
                  std::optional<float> reluLowerClipSlopeLocal,
                  std::optional<float> swishBetaLocal,
                  std::optional<float> eluAlphaLocal,
                  std::optional<float> softplusBetaLocal)
        : mode(modeLocal)
        , reluLowerClip(reluLowerClipLocal)
        , reluUpperClip(reluUpperClipLocal)
        , reluLowerClipSlope(reluLowerClipSlopeLocal)
        , swishBeta(swishBetaLocal)
        , eluAlpha(eluAlphaLocal)
        , softplusBeta(softplusBetaLocal)
    {
        using PointwiseMode = hipdnn_sdk::data_objects::PointwiseMode;

        switch(mode)
        {
        case PointwiseMode::RELU_FWD:
        case PointwiseMode::RELU_BWD:
        case PointwiseMode::SIGMOID_FWD:
        case PointwiseMode::SIGMOID_BWD:
        case PointwiseMode::TANH_FWD:
        case PointwiseMode::TANH_BWD:
        case PointwiseMode::ELU_FWD:
        case PointwiseMode::ELU_BWD:
        case PointwiseMode::SOFTPLUS_FWD:
        case PointwiseMode::SOFTPLUS_BWD:
        case PointwiseMode::ABS:
        case PointwiseMode::IDENTITY:
            break;
        default:
            throw std::invalid_argument("Unknown activation mode");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const ActivTestCase& tc)
    {
        using namespace hipdnn_sdk::utilities;

        ss << "(mode:" << tc.mode;
        if(tc.reluLowerClip)
        {
            ss << " reluLowerClip:" << tc.reluLowerClip.value();
        }
        if(tc.reluUpperClip)
        {
            ss << " reluUpperClip:" << tc.reluUpperClip.value();
        }
        if(tc.reluLowerClipSlope)
        {
            ss << " reluLowerClipSlope:" << tc.reluLowerClipSlope.value();
        }
        if(tc.swishBeta)
        {
            ss << " swishBeta:" << tc.swishBeta.value();
        }
        if(tc.eluAlpha)
        {
            ss << " eluAlpha:" << tc.eluAlpha.value();
        }
        if(tc.softplusBeta)
        {
            ss << " softplusBeta:" << tc.softplusBeta.value();
        }
        ss << ")";

        return ss;
    }
};

inline std::vector<ActivTestCase> createBwdActivationTestCases()
{
    using PM = hipdnn_sdk::data_objects::PointwiseMode;

    std::vector<ActivTestCase> cases;

    // RELU backward
    cases.emplace_back(PM::RELU_BWD,
                       std::nullopt, // reluLowerClip
                       std::nullopt, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    return cases;
}

} // namespace test_activation_common
