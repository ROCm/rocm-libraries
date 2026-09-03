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
    };
}

// ===========================================================================
// Medium shapes - standard tier (PR gate)
// ===========================================================================

inline std::vector<ReductionTestShape> getReductionMedium4DShapes()
{
    return {
        {{1, 3, 8, 8}, {1, 1, 8, 8}},
        {{1, 3, 8, 8}, {1, 3, 1, 8}},
        {{1, 3, 8, 8}, {1, 1, 1, 8}},
        {{2, 3, 1, 1}, {1, 3, 1, 1}},
        {{8, 1, 8, 8}, {8, 1, 1, 1}},
        {{8, 3, 1, 8}, {8, 3, 1, 1}},
        {{8, 3, 8, 1}, {8, 1, 8, 1}},
        {{8, 3, 8, 1}, {1, 3, 8, 1}},
        {{4, 8, 24, 16}, {1, 8, 24, 16}},
        {{4, 8, 24, 16}, {4, 1, 24, 16}},
        {{4, 16, 28, 28}, {4, 1, 1, 28}},
        {{4, 32, 14, 14}, {4, 32, 1, 1}},
    };
}

inline std::vector<ReductionTestShape> getReductionMedium5DShapes()
{
    return {
        {{4, 3, 4, 8, 8}, {1, 3, 4, 8, 8}},
        {{4, 3, 4, 8, 8}, {4, 1, 4, 8, 8}},
        {{4, 3, 4, 8, 8}, {4, 3, 1, 8, 8}},
        {{4, 8, 4, 12, 8}, {1, 8, 4, 12, 8}},
        {{4, 8, 4, 12, 8}, {4, 1, 1, 12, 8}},
        {{4, 8, 4, 12, 8}, {1, 1, 1, 12, 8}},
        {{2, 16, 4, 7, 7}, {2, 16, 4, 1, 1}},
        {{2, 16, 4, 7, 7}, {2, 1, 4, 7, 7}},
        {{2, 16, 4, 7, 7}, {1, 1, 4, 7, 7}},
    };
}

// ===========================================================================
// Large shapes - Full/weekly tier CI execution (Comprehensive / nightly)
// ===========================================================================

inline std::vector<ReductionTestShape> getReductionLarge4DShapes()
{
    return {
        {{4, 72, 12, 8}, {1, 72, 12, 8}},
        {{4, 72, 12, 8}, {4, 1, 12, 8}},
        {{4, 72, 12, 8}, {4, 1, 1, 8}},
        {{4, 144, 1, 8}, {4, 144, 1, 1}},
        {{4, 144, 1, 8}, {1, 1, 1, 8}},
        {{4, 512, 4, 8}, {1, 512, 4, 8}},
        {{4, 512, 4, 8}, {4, 1, 1, 1}},
        {{32, 9, 12, 8}, {32, 1, 12, 8}},
        {{32, 128, 8, 12}, {1, 128, 8, 12}},
        {{32, 128, 8, 12}, {32, 128, 1, 1}},
    };
}

inline std::vector<ReductionTestShape> getReductionLarge5DShapes()
{
    return {
        {{4, 32, 2, 8, 4}, {1, 32, 2, 8, 4}},
        {{4, 32, 2, 8, 4}, {4, 1, 2, 8, 4}},
        {{4, 32, 2, 8, 4}, {4, 1, 1, 8, 4}},
        {{4, 64, 1, 8, 8}, {1, 64, 1, 8, 8}},
        {{4, 64, 1, 8, 8}, {4, 64, 1, 1, 1}},
        {{4, 64, 1, 8, 8}, {1, 1, 1, 8, 8}},
        {{8, 32, 2, 4, 4}, {8, 1, 2, 4, 4}},
        {{8, 32, 2, 4, 4}, {8, 32, 1, 1, 1}},
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
