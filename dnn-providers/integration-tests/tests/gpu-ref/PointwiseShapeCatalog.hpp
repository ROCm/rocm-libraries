// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PointwiseShapeCase.hpp"
#include <vector>

namespace gpu_pointwise_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

inline const std::vector<PointwiseMode>& getUnaryOps()
{
    static const std::vector<PointwiseMode> s_unaryOps = {
        PointwiseMode::IDENTITY,
        PointwiseMode::ABS,
        PointwiseMode::NEG,
        PointwiseMode::RELU_FWD,
        PointwiseMode::SIGMOID_FWD,
        PointwiseMode::TANH_FWD,
        PointwiseMode::GELU_FWD,
        PointwiseMode::GELU_APPROX_TANH_FWD,
        PointwiseMode::SWISH_FWD,
    };
    return s_unaryOps;
}

inline const std::vector<PointwiseMode>& getBinaryOps()
{
    static const std::vector<PointwiseMode> s_binaryOps = {
        PointwiseMode::ADD,
        PointwiseMode::SUB,
        PointwiseMode::MUL,
        PointwiseMode::SIGMOID_BWD,
        PointwiseMode::TANH_BWD,
        PointwiseMode::RELU_BWD,
    };
    return s_binaryOps;
};

// ============================================================================
// Small shapes — fast binary (CI gate)
// ============================================================================

inline std::vector<std::vector<int64_t>> get4dSmallShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes
        = {{2, 3, 4, 4}, {1, 2, 14, 14}, {16, 32, 24, 16}};
    return s_shapes;
}

inline std::vector<std::vector<int64_t>> get5dSmallShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes
        = {{2, 3, 3, 1, 1}, {2, 3, 4, 2, 2}, {4, 8, 2, 4, 4}};
    return s_shapes;
}

inline std::vector<PointwiseTestCase> getSmall4dUnaryPointwiseCases()
{
    const std::vector<PointwiseMode>& unaryOps = getUnaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(unaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : unaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getSmall4dBinaryPointwiseCases()
{
    const std::vector<PointwiseMode>& binaryOps = getBinaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(binaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : binaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getSmall5dUnaryPointwiseCases()
{
    const std::vector<PointwiseMode>& unaryOps = getUnaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(unaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : unaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getSmall5dBinaryPointwiseCases()
{
    const std::vector<PointwiseMode>& binaryOps = getBinaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(binaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : binaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

// ============================================================================
// Medium shapes — Standard tier (PR gate)
// ============================================================================

inline std::vector<std::vector<int64_t>> get4dMediumShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {32, 1, 14, 14},
        {16, 32, 192, 128},
        {16, 128, 56, 56},
    };
    return s_shapes;
}

inline std::vector<std::vector<int64_t>> get5dMediumShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {16, 3, 8, 14, 14},
        {16, 32, 4, 48, 32},
        {8, 64, 4, 28, 28},
    };
    return s_shapes;
}

inline std::vector<PointwiseTestCase> getMedium4dUnaryPointwiseCases()
{
    const std::vector<PointwiseMode>& unaryOps = getUnaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(unaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : unaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getMedium4dBinaryPointwiseCases()
{
    const std::vector<PointwiseMode>& binaryOps = getBinaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(binaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : binaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getMedium5dUnaryPointwiseCases()
{
    const std::vector<PointwiseMode>& unaryOps = getUnaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(unaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : unaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getMedium5dBinaryPointwiseCases()
{
    const std::vector<PointwiseMode>& binaryOps = getBinaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(binaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : binaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

// ============================================================================
// Large shapes — split into edge cases (Comprehensive / nightly) and
// stress tests (Full / weekly) for tiered CI execution.

inline std::vector<std::vector<int64_t>> get4dLargeShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {16, 288, 48, 32},
        {128, 512, 24, 48},
    };

    return s_shapes;
}

inline std::vector<std::vector<int64_t>> get5dLargeShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes
        = {{16, 128, 8, 24, 16}, {32, 128, 8, 16, 16}};
    return s_shapes;
}

inline std::vector<PointwiseTestCase> getLarge4dUnaryPointwiseCases()
{
    const std::vector<PointwiseMode>& unaryOps = getUnaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(unaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : unaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getLarge4dBinaryPointwiseCases()
{
    const std::vector<PointwiseMode>& binaryOps = getBinaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(binaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : binaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getLarge5dUnaryPointwiseCases()
{
    const std::vector<PointwiseMode>& unaryOps = getUnaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(unaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : unaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getLarge5dBinaryPointwiseCases()
{
    const std::vector<PointwiseMode>& binaryOps = getBinaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(binaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : binaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

} // namespace gpu_pointwise_ref_test
