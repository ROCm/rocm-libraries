// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

#include <exception>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <optional>
#include <string>
#include <vector>

namespace test_activation_common
{

struct ActivTestCase
{
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode;
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    std::optional<float> swishBeta;
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;
    std::string note;

    ActivTestCase(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode modeLocal,
                  std::optional<float> reluLowerClipLocal = std::nullopt,
                  std::optional<float> reluUpperClipLocal = std::nullopt,
                  std::optional<float> reluLowerClipSlopeLocal = std::nullopt,
                  std::optional<float> swishBetaLocal = std::nullopt,
                  std::optional<float> eluAlphaLocal = std::nullopt,
                  std::optional<float> softplusBetaLocal = std::nullopt)
        : mode(modeLocal)
        , reluLowerClip(reluLowerClipLocal)
        , reluUpperClip(reluUpperClipLocal)
        , reluLowerClipSlope(reluLowerClipSlopeLocal)
        , swishBeta(swishBetaLocal)
        , eluAlpha(eluAlphaLocal)
        , softplusBeta(softplusBetaLocal)
        , note(generateNote())
    {
        using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

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

    std::string generateNote() const
    {
        std::vector<std::string> tags;
        if(reluLowerClip.has_value())
        {
            tags.push_back("LowerClip");
        }
        if(reluUpperClip.has_value())
        {
            tags.push_back("UpperClip");
        }
        if(reluLowerClipSlope.has_value())
        {
            tags.push_back("LowerClipSlope");
        }
        if(swishBeta.has_value())
        {
            tags.push_back("SwishBeta");
        }
        if(eluAlpha.has_value())
        {
            tags.push_back("EluAlpha");
        }
        if(softplusBeta.has_value())
        {
            tags.push_back("SoftplusBeta");
        }

        if(tags.empty())
        {
            return {};
        }

        std::string result;
        for(size_t i = 0; i < tags.size(); ++i)
        {
            if(i > 0)
            {
                result += "+";
            }
            result += tags[i];
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& ss, const ActivTestCase& tc)
    {
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
        if(!tc.note.empty())
        {
            ss << " note:" << tc.note;
        }
        ss << ")";

        return ss;
    }
};

inline std::vector<ActivTestCase> createFwdActivationSmokeCases()
{
    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    std::vector<ActivTestCase> cases;

    // RELU_FWD (standard ReLU)
    cases.emplace_back(PM::RELU_FWD,
                       std::nullopt, // reluLowerClip
                       std::nullopt, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    return cases;
}

inline std::vector<ActivTestCase> createFwdActivationFullCases()
{
    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    std::vector<ActivTestCase> cases;

    // RELU_FWD (standard ReLU)
    cases.emplace_back(PM::RELU_FWD,
                       0.0f, // reluLowerClip
                       std::nullopt, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    // ReLU6: upper clip at 6.0 (Clipped ReLU)
    cases.emplace_back(PM::RELU_FWD,
                       std::nullopt, // reluLowerClip
                       6.0f, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    // CLAMP: both lower and upper clips (e.g., clip to range [0.0, 0.5])
    cases.emplace_back(PM::RELU_FWD,
                       0.1f, // reluLowerClip
                       0.5f, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    // Leaky ReLU: ReLU but with a non-negative slope for negative inputs
    cases.emplace_back(PM::RELU_FWD,
                       std::nullopt, // reluLowerClip
                       std::nullopt, // reluUpperClip
                       0.01f, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    return cases;
}

inline std::vector<ActivTestCase> createBatchnormBwdActivationTestCases()
{
    return {
        // ReLU Backward: d/dx Max(0, x) = 1 * (x > 0)
        ActivTestCase(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD, 0.0f),
        // Clipped ReLU Backward: d/dx Clamp(x, -inf, upper)
        ActivTestCase(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD, std::nullopt, 0.5f),
        // CLAMP Backward: d/dx Clamp(x, lower, upper)
        ActivTestCase(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD, 0.1f, 0.5f)};
}

} // namespace test_activation_common
