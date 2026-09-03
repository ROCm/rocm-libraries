// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ReductionTestCase.hpp"

namespace gpu_reduction_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

struct ReductionTestShape
{
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
};

// ===========================================================================
// Reduction modes
// ===========================================================================

inline std::vector<hipdnn_flatbuffers_sdk::data_objects::ReductionMode> getReductionModes()
{
    using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
    return {ReductionMode::ADD,
            ReductionMode::AVG,
            ReductionMode::AMAX,
            ReductionMode::NORM1,
            ReductionMode::NORM2,
            ReductionMode::MUL,
            ReductionMode::MUL_NO_ZEROS,
            ReductionMode::MIN_OP,
            ReductionMode::MAX_OP};
}

// ===========================================================================
// Layouts
// ===========================================================================

inline std::vector<hipdnn_data_sdk::utilities::TensorLayout> getReduction4DLayouts()
{
    using hipdnn_data_sdk::utilities::TensorLayout;
    return {TensorLayout::NCHW, TensorLayout::NHWC};
}

inline std::vector<hipdnn_data_sdk::utilities::TensorLayout> getReduction5DLayouts()
{
    using hipdnn_data_sdk::utilities::TensorLayout;
    return {TensorLayout::NCDHW, TensorLayout::NDHWC};
}

// ===========================================================================
// Small shapes - fast binary (CI gate)
// ===========================================================================

inline std::vector<ReductionTestShape> getReductionSmall4DShapes()
{
    return {
        {{2, 3, 4, 4}, {2, 3, 4, 1}},
        {{2, 3, 4, 4}, {2, 3, 1, 4}},
        {{2, 3, 4, 4}, {2, 1, 4, 4}},
        {{2, 3, 4, 4}, {1, 3, 4, 4}},
        {{4, 4, 4, 4}, {1, 1, 4, 4}},
        {{4, 4, 4, 4}, {4, 4, 1, 1}},
        {{2, 3, 8, 8}, {2, 3, 1, 1}},
        {{16, 8, 4, 4}, {16, 8, 1, 1}},
        {{1, 3, 14, 14}, {1, 3, 1, 1}},
        {{4, 4, 4, 4}, {1, 1, 1, 1}},
    };
}

inline std::vector<ReductionTestShape> getReductionSmall5DShapes()
{
    return {
        {{2, 3, 3, 1, 1}, {2, 3, 1, 1, 1}},
        {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}},
        {{2, 3, 4, 2, 2}, {2, 3, 4, 1, 1}},
        {{2, 3, 4, 2, 2}, {2, 1, 4, 2, 2}},
        {{2, 3, 4, 2, 2}, {2, 1, 1, 2, 2}},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}},
        {{4, 8, 2, 4, 4}, {4, 8, 2, 1, 1}},
        {{4, 8, 2, 4, 4}, {1, 1, 1, 1, 1}},
    };
}

// ===========================================================================
// Medium shapes - standard tier (PR gate)
// ===========================================================================

inline std::vector<ReductionTestShape> getReductionMedium4DShapes()
{
    return {
        {{1, 3, 14, 14}, {1, 1, 14, 14}},
        {{1, 3, 14, 14}, {1, 3, 1, 14}},
        {{1, 3, 14, 14}, {1, 1, 1, 14}},
        {{1, 256, 1, 1}, {1, 1, 1, 1}},
        {{2, 3, 1, 1}, {1, 3, 1, 1}},
        {{32, 1, 14, 14}, {32, 1, 1, 1}},
        {{32, 3, 1, 14}, {32, 3, 1, 1}},
        {{32, 3, 14, 1}, {32, 1, 14, 1}},
        {{32, 3, 14, 1}, {1, 3, 14, 1}},
        {{16, 32, 192, 128}, {1, 32, 192, 128}},
        {{16, 32, 192, 128}, {16, 1, 192, 128}},
        {{16, 64, 225, 225}, {16, 1, 1, 225}},
        {{16, 64, 225, 225}, {1, 1, 1, 1}},
        {{16, 128, 56, 56}, {16, 128, 1, 1}},
    };
}

inline std::vector<ReductionTestShape> getReductionMedium5DShapes()
{
    return {
        {{16, 3, 8, 14, 14}, {16, 3, 8, 14, 14}},
        {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}},
        {{16, 3, 8, 14, 14}, {16, 1, 8, 14, 14}},
        {{16, 3, 8, 14, 14}, {16, 3, 1, 14, 14}},
        {{16, 32, 4, 48, 32}, {1, 32, 4, 48, 32}},
        {{16, 32, 4, 48, 32}, {16, 1, 1, 48, 32}},
        {{16, 32, 4, 48, 32}, {1, 1, 1, 48, 32}},
        {{8, 64, 4, 28, 28}, {8, 64, 4, 1, 1}},
        {{8, 64, 4, 28, 28}, {8, 1, 4, 28, 28}},
        {{8, 64, 4, 28, 28}, {1, 1, 4, 28, 28}},
    };
}

// ===========================================================================
// Large shapes - Full/weekly tier CI execution (Comprehensive / nightly)
// ===========================================================================

inline std::vector<ReductionTestShape> getReductionLarge4DShapes()
{
    return {
        {{16, 288, 48, 32}, {1, 288, 48, 32}},
        {{16, 288, 48, 32}, {16, 1, 48, 32}},
        {{16, 288, 48, 32}, {16, 1, 1, 32}},
        {{16, 576, 1, 30}, {16, 576, 1, 1}},
        {{16, 576, 1, 30}, {1, 1, 1, 30}},
        {{16, 2048, 16, 32}, {1, 2048, 16, 32}},
        {{16, 2048, 16, 32}, {16, 1, 1, 1}},
        {{128, 35, 48, 32}, {128, 1, 48, 32}},
        {{128, 512, 24, 48}, {1, 512, 24, 48}},
        {{128, 512, 24, 48}, {128, 512, 1, 1}},
        {{128, 512, 24, 48}, {1, 1, 1, 1}},
    };
}

inline std::vector<ReductionTestShape> getReductionLarge5DShapes()
{
    return {
        {{16, 128, 8, 24, 16}, {1, 128, 8, 24, 16}},
        {{16, 128, 8, 24, 16}, {16, 1, 8, 24, 16}},
        {{16, 128, 8, 24, 16}, {16, 1, 1, 24, 16}},
        {{16, 256, 4, 32, 32}, {1, 256, 4, 32, 32}},
        {{16, 256, 4, 32, 32}, {16, 256, 1, 1, 1}},
        {{16, 256, 4, 32, 32}, {1, 1, 1, 32, 32}},
        {{32, 128, 8, 16, 16}, {32, 1, 8, 16, 16}},
        {{32, 128, 8, 16, 16}, {32, 128, 1, 1, 1}},
        {{32, 128, 8, 16, 16}, {1, 1, 1, 1, 1}},
    };
}

// ==========================================================================
// Cartesian product of shapes, layouts, and modes
// ==========================================================================

inline std::vector<ReductionTestCase> makeReductionTestCases(
    const std::vector<ReductionTestShape>& shapes,
    const std::vector<hipdnn_data_sdk::utilities::TensorLayout>& layouts,
    const std::vector<hipdnn_flatbuffers_sdk::data_objects::ReductionMode>& modes)
{
    std::vector<ReductionTestCase> cases;
    cases.reserve(shapes.size() * layouts.size() * modes.size());

    for(const auto& shape : shapes)
    {
        for(const auto& layout : layouts)
        {
            for(const auto& mode : modes)
            {
                cases.push_back(ReductionTestCase{shape.inputDims, shape.outputDims, layout, mode});
            }
        }
    }

    return cases;
}

} // namespace gpu_reduction_ref_test
