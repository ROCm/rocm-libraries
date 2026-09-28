// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ResampleTestCase.hpp"

namespace gpu_resample_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;
using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;
using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;

struct ResampleTestShape
{
    std::vector<int64_t> xDims; // Maps to dxDims for backward tests
    std::vector<int64_t> prePadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
};

// ===========================================================================
// Resample and padding modes
// ===========================================================================

inline std::vector<ResampleMode> getResampleModes()
{
    return {ResampleMode::MAXPOOL,
            ResampleMode::AVGPOOL_INCLUDE_PADDING,
            ResampleMode::AVGPOOL_EXCLUDE_PADDING};
}

inline std::vector<PaddingMode> getPaddingModes()
{
    return {PaddingMode::NEG_INF_PAD, PaddingMode::ZERO_PAD};
}

// ===========================================================================
// Layouts
// ===========================================================================

inline std::vector<TensorLayout> getResample4DLayouts()
{
    return {TensorLayout::NCHW, TensorLayout::NHWC};
}

inline std::vector<TensorLayout> getResample5DLayouts()
{
    return {TensorLayout::NCDHW, TensorLayout::NDHWC};
}

// ===========================================================================
// Small shapes - Fast binary (CI gate)
// ===========================================================================

inline std::vector<ResampleTestShape> getResampleSmall4DShapes()
{
    return {
        {{2, 3, 4, 4}, {0, 0}, {2, 2}, {2, 2}},
        {{2, 3, 4, 4}, {1, 1}, {1, 1}, {3, 3}},
        {{2, 4, 8, 6}, {0, 1}, {2, 1}, {3, 2}},
        {{1, 3, 8, 8}, {0, 0}, {1, 1}, {1, 1}},
        {{2, 8, 7, 7}, {1, 0}, {2, 2}, {3, 3}},
        {{1, 1, 14, 14}, {0, 0}, {2, 2}, {2, 2}},
    };
}

inline std::vector<ResampleTestShape> getResampleSmall5DShapes()
{
    return {
        {{2, 3, 4, 4, 4}, {0, 0, 0}, {2, 2, 2}, {2, 2, 2}},
        {{2, 3, 3, 4, 4}, {1, 1, 1}, {1, 1, 1}, {3, 3, 3}},
        {{2, 4, 8, 4, 4}, {0, 1, 1}, {1, 2, 2}, {2, 3, 3}},
        {{1, 3, 2, 4, 4}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
    };
}

// ===========================================================================
// Medium shapes - Standard tier (PR gate)
// ===========================================================================

inline std::vector<ResampleTestShape> getResampleMedium4DShapes()
{
    return {
        {{2, 32, 56, 56}, {1, 1}, {2, 2}, {3, 3}},
        {{2, 64, 28, 28}, {0, 0}, {2, 2}, {2, 2}},
        {{2, 128, 14, 14}, {0, 0}, {1, 1}, {14, 14}},
        {{4, 32, 28, 28}, {1, 1}, {2, 2}, {3, 3}},
        {{4, 128, 7, 7}, {1, 1}, {2, 2}, {3, 3}},
        {{2, 32, 16, 32}, {0, 0}, {2, 2}, {2, 2}},
    };
}

inline std::vector<ResampleTestShape> getResampleMedium5DShapes()
{
    return {
        {{1, 32, 8, 28, 28}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
        {{1, 64, 4, 14, 14}, {0, 0, 0}, {1, 2, 2}, {1, 2, 2}},
        {{2, 128, 4, 7, 7}, {0, 0, 0}, {1, 1, 1}, {4, 7, 7}},
        {{1, 32, 4, 14, 14}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
    };
}

// ===========================================================================
// Large shapes - Full/weekly tier CI execution (Comprehensive / nightly)
// ===========================================================================

inline std::vector<ResampleTestShape> getResampleLarge4DShapes()
{
    return {
        {{4, 32, 112, 112}, {1, 1}, {2, 2}, {3, 3}},
        {{4, 256, 14, 14}, {0, 0}, {2, 2}, {2, 2}},
        {{8, 64, 28, 28}, {0, 0}, {2, 2}, {2, 2}},
        {{4, 512, 14, 14}, {1, 1}, {2, 2}, {3, 3}},
        {{8, 512, 7, 7}, {0, 0}, {1, 1}, {7, 7}},
    };
}

inline std::vector<ResampleTestShape> getResampleLarge5DShapes()
{
    return {
        {{1, 16, 16, 64, 64}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
        {{2, 256, 8, 14, 14}, {0, 0, 0}, {2, 2, 2}, {2, 2, 2}},
        {{4, 64, 8, 14, 14}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
    };
}

// ==========================================================================
// Cartesian product of shapes, layouts, resample modes, and padding modes
// ==========================================================================

inline std::vector<ResampleTestCase>
    makeResampleTestCases(const std::vector<ResampleTestShape>& shapes,
                          const std::vector<TensorLayout>& layouts,
                          const std::vector<ResampleMode>& resampleModes,
                          const std::vector<PaddingMode>& paddingModes,
                          ResampleDirection direction = ResampleDirection::FORWARD)
{
    std::vector<ResampleTestCase> cases;
    cases.reserve(shapes.size() * layouts.size() * resampleModes.size() * paddingModes.size());

    for(const auto& shape : shapes)
    {
        for(const auto& layout : layouts)
        {
            for(const auto& resampleMode : resampleModes)
            {
                for(const auto& paddingMode : paddingModes)
                {
                    cases.emplace_back(shape.xDims,
                                       layout,
                                       shape.prePadding,
                                       shape.stride,
                                       shape.window,
                                       resampleMode,
                                       paddingMode,
                                       direction);
                }
            }
        }
    }

    return cases;
}

} // namespace gpu_resample_ref_test
