// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PointwiseShapeCase.hpp"
#include <vector>

namespace gpu_pointwise_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// The modes the GPU reference had before the CPU-parity work run their small fp32 cases in the quick
// tier. The extended modes run theirs in the standard tier.
inline const std::vector<PointwiseMode>& getCoreUnaryOps()
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

// Also the modes whose results on zeros, NaN, infinities and negative numbers are worth checking
// against the CPU reference.
inline const std::vector<PointwiseMode>& getExtendedUnaryOps()
{
    static const std::vector<PointwiseMode> s_ops = {
        PointwiseMode::EXP,
        PointwiseMode::LOG,
        PointwiseMode::SQRT,
        PointwiseMode::RSQRT,
        PointwiseMode::RECIPROCAL,
        PointwiseMode::SIN,
        PointwiseMode::TAN,
        PointwiseMode::ERF,
        PointwiseMode::CEIL,
        PointwiseMode::FLOOR,
        PointwiseMode::ELU_FWD,
        PointwiseMode::SOFTPLUS_FWD,
    };
    return s_ops;
}

inline const std::vector<PointwiseMode>& getUnaryOps()
{
    static const std::vector<PointwiseMode> s_unaryOps = [] {
        std::vector<PointwiseMode> ops = getCoreUnaryOps();
        ops.insert(ops.end(), getExtendedUnaryOps().begin(), getExtendedUnaryOps().end());
        return ops;
    }();
    return s_unaryOps;
}

inline const std::vector<PointwiseMode>& getCoreBinaryOps()
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
}

inline const std::vector<PointwiseMode>& getExtendedBinaryOps()
{
    static const std::vector<PointwiseMode> s_binaryOps = {
        PointwiseMode::MAX_OP,
        PointwiseMode::MIN_OP,
    };
    return s_binaryOps;
}

inline const std::vector<PointwiseMode>& getBinaryOps()
{
    static const std::vector<PointwiseMode> s_binaryOps = [] {
        std::vector<PointwiseMode> ops = getCoreBinaryOps();
        ops.insert(ops.end(), getExtendedBinaryOps().begin(), getExtendedBinaryOps().end());
        return ops;
    }();
    return s_binaryOps;
}

inline const std::vector<PointwiseMode>& getCompareOps()
{
    static const std::vector<PointwiseMode> s_compareOps = {
        PointwiseMode::CMP_GT,
    };
    return s_compareOps;
}

inline const std::vector<PointwiseMode>& getTernaryOps()
{
    static const std::vector<PointwiseMode> s_ternaryOps = {
        PointwiseMode::BINARY_SELECT,
    };
    return s_ternaryOps;
}

// ============================================================================
// Small shapes — fast binary (CI gate)
// ============================================================================

inline std::vector<std::vector<int64_t>> get4dSmallShapes()
{
    // Tests around blocksize multiple of 256
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {1, 1, 4, 4}, // 16 elements
        {2, 2, 8, 8}, // 256 elements
        {6, 50, 1, 1} // 300 elements
    };
    return s_shapes;
}

inline std::vector<std::vector<int64_t>> get5dSmallShapes()
{
    // Tests around blocksize multiple of 256
    static const std::vector<std::vector<int64_t>> s_shapes = {{3, 3, 3, 3, 3}, // 243 elements
                                                               {2, 2, 4, 2, 8}, // 256 elements
                                                               {1, 1, 7, 7, 7}}; // 343 element
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

inline std::vector<PointwiseTestCase> getSmall4dComparePointwiseCases()
{
    const std::vector<PointwiseMode>& compareOps = getCompareOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(compareOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : compareOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getSmall4dTernaryPointwiseCases()
{
    const std::vector<PointwiseMode>& ternaryOps = getTernaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ternaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : ternaryOps)
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

inline std::vector<PointwiseTestCase> getSmall5dComparePointwiseCases()
{
    const std::vector<PointwiseMode>& compareOps = getCompareOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(compareOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : compareOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getSmall5dTernaryPointwiseCases()
{
    const std::vector<PointwiseMode>& ternaryOps = getTernaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dSmallShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ternaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : ternaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase>
    makePointwiseCases(const std::vector<PointwiseMode>& ops,
                       const std::vector<std::vector<int64_t>>& shapes)
{
    std::vector<PointwiseTestCase> cases;
    cases.reserve(ops.size() * shapes.size());
    for(const auto& op : ops)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getSmall4dCoreUnaryPointwiseCases()
{
    return makePointwiseCases(getCoreUnaryOps(), get4dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall4dCoreBinaryPointwiseCases()
{
    return makePointwiseCases(getCoreBinaryOps(), get4dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall4dExtendedUnaryPointwiseCases()
{
    return makePointwiseCases(getExtendedUnaryOps(), get4dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall4dExtendedBinaryPointwiseCases()
{
    return makePointwiseCases(getExtendedBinaryOps(), get4dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall5dCoreUnaryPointwiseCases()
{
    return makePointwiseCases(getCoreUnaryOps(), get5dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall5dCoreBinaryPointwiseCases()
{
    return makePointwiseCases(getCoreBinaryOps(), get5dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall5dExtendedUnaryPointwiseCases()
{
    return makePointwiseCases(getExtendedUnaryOps(), get5dSmallShapes());
}

inline std::vector<PointwiseTestCase> getSmall5dExtendedBinaryPointwiseCases()
{
    return makePointwiseCases(getExtendedBinaryOps(), get5dSmallShapes());
}

// The special values are generated by the test, so the io dims only name the case
inline std::vector<PointwiseTestCase> getUnarySpecialValuePointwiseCases()
{
    const std::vector<PointwiseMode>& ops = getExtendedUnaryOps();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ops.size());
    for(const auto& op : ops)
    {
        cases.push_back({op, {1, 1, 1, 6}});
    }
    return cases;
}

// ============================================================================
// Medium shapes — Standard tier (PR gate)
// ============================================================================

inline std::vector<std::vector<int64_t>> get4dMediumShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {32, 2, 7, 14} // 6272 elements
    };
    return s_shapes;
}

inline std::vector<std::vector<int64_t>> get5dMediumShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {16, 3, 8, 14, 14}, // 5376 eleements
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

inline std::vector<PointwiseTestCase> getMedium4dComparePointwiseCases()
{
    const std::vector<PointwiseMode>& compareOps = getCompareOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(compareOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : compareOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getMedium4dTernaryPointwiseCases()
{
    const std::vector<PointwiseMode>& ternaryOps = getTernaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ternaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : ternaryOps)
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

inline std::vector<PointwiseTestCase> getMedium5dComparePointwiseCases()
{
    const std::vector<PointwiseMode>& compareOps = getCompareOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(compareOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : compareOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getMedium5dTernaryPointwiseCases()
{
    const std::vector<PointwiseMode>& ternaryOps = getTernaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dMediumShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ternaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : ternaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

// ============================================================================
// Large shapes — Comprehensive / nightly tier (fp32 only).

inline std::vector<std::vector<int64_t>> get4dLargeShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {16, 288, 48, 32}, // 7077888 elements
    };

    return s_shapes;
}

inline std::vector<std::vector<int64_t>> get5dLargeShapes()
{
    static const std::vector<std::vector<int64_t>> s_shapes = {
        {16, 128, 8, 24, 16}, // 6291456 elements
    };
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

inline std::vector<PointwiseTestCase> getLarge4dComparePointwiseCases()
{
    const std::vector<PointwiseMode>& compareOps = getCompareOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(compareOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : compareOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getLarge4dTernaryPointwiseCases()
{
    const std::vector<PointwiseMode>& ternaryOps = getTernaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get4dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ternaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : ternaryOps)
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

inline std::vector<PointwiseTestCase> getLarge5dComparePointwiseCases()
{
    const std::vector<PointwiseMode>& compareOps = getCompareOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(compareOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : compareOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

inline std::vector<PointwiseTestCase> getLarge5dTernaryPointwiseCases()
{
    const std::vector<PointwiseMode>& ternaryOps = getTernaryOps();
    const std::vector<std::vector<int64_t>>& shapes = get5dLargeShapes();

    std::vector<PointwiseTestCase> cases;
    cases.reserve(ternaryOps.size() * shapes.size());

    // Use std::views::cartesian_product once compiler supports C++23
    for(const auto& op : ternaryOps)
    {
        for(const auto& shape : shapes)
        {
            cases.push_back({op, shape});
        }
    }
    return cases;
}

} // namespace gpu_pointwise_ref_test
