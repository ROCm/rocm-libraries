// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>

namespace hip_kernel_provider::test_activation_common
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

inline std::vector<ActivTestCase> createFwdActivationSmokeCases()
{
    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    std::vector<ActivTestCase> cases;

    // RELU_FWD (standard ReLU) - Only activation supported by fusion ops
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
                       0.5f, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    // CLAMP: both lower and upper clips (e.g., clip to range [0.0, 6.0])
    cases.emplace_back(PM::RELU_FWD,
                       0.1f, // reluLowerClip
                       0.5f, // reluUpperClip
                       std::nullopt, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );

    // Leaky ReLU: NOT supported by batchnorm activation fusion
    // Supported activation modes for batchnorm fusion:
    //   - PASSTHRU (identity/no activation)
    //   - RELU (standard ReLU)
    //   - CLIPPED_RELU (ReLU with upper clip)
    //   - CLAMP (ReLU with both upper and lower clips)
    // Unsupported:
    //   - LEAKY_RELU (ReLU with lower clip slope)
    // This applies to both forward inference and forward training operations.
    // See: kernels/BatchnormActivation.hpp for implementation details.
#if 0
    cases.emplace_back(PM::RELU_FWD,
                       std::nullopt, // reluLowerClip
                       std::nullopt, // reluUpperClip
                       0.01f, // reluLowerClipSlope
                       std::nullopt, // swishBeta
                       std::nullopt, // eluAlpha
                       std::nullopt // softplusBeta
    );
#endif

    return cases;
}

} // namespace hip_kernel_provider::test_activation_common
