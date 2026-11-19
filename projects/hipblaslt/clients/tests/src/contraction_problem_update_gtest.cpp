// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipBLASLt reuses a TensileLite problem for a new GEMM by resizing its tensors and calling
// updateProblem. The updated problem must match a problem built for the new GEMM.

#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>

using namespace TensileLite;

namespace
{
    struct Gemm
    {
        rocisa::DataType type;
        size_t           m, n, k;
        double           beta;
    };

    ContractionProblemGemm build(Gemm const& g)
    {
        return ContractionProblemGemm::GEMM_Strides(false,
                                                    false,
                                                    g.type,
                                                    g.type,
                                                    g.type,
                                                    g.type,
                                                    g.m,
                                                    g.n,
                                                    g.k,
                                                    1,
                                                    g.m,
                                                    g.m * g.k,
                                                    g.k,
                                                    g.k * g.n,
                                                    g.m,
                                                    g.m * g.n,
                                                    g.m,
                                                    g.m * g.n,
                                                    g.beta);
    }

    // Same steps as updateTensileProblem in hipBLASLt for a non-transposed GEMM.
    void update(ContractionProblemGemm& problem, Gemm const& g)
    {
        using Tensor = ContractionProblemGemm::TENSOR;
        problem.resetTensor(Tensor::A, g.type, {g.m, g.k, 1}, {1, g.m, g.m * g.k});
        problem.resetTensor(Tensor::B, g.type, {g.k, g.n, 1}, {1, g.k, g.k * g.n});
        problem.resetTensor(Tensor::C, g.type, {g.m, g.n, 1}, {1, g.m, g.m * g.n});
        problem.resetTensor(Tensor::D, g.type, {g.m, g.n, 1}, {1, g.m, g.m * g.n});

        auto freeIndices  = problem.freeIndices();
        auto batchIndices = problem.batchIndices();
        auto boundIndices = problem.boundIndices();
        problem.updateProblem(
            freeIndices, batchIndices, boundIndices, g.beta, problem.workspaceSize());
    }

    // A hipBLASLt matmul descriptor starts from a 1x1x1 FP32 problem with beta 1.
    Gemm const placeholder{rocisa::DataType::Float, 1, 1, 1, 1.0};

    TEST(ContractionProblemUpdate, smoke_ArithmeticIntensityFollowsNewShape)
    {
        auto problem = build(placeholder);

        for(Gemm const& g : {Gemm{rocisa::DataType::Half, 1024, 8192, 5120, 0.0},
                             Gemm{rocisa::DataType::Half, 64, 64, 64, 1.0},
                             Gemm{rocisa::DataType::Float, 4096, 16, 4096, 0.0}})
        {
            update(problem, g);
            auto fresh = build(g);
            ASSERT_EQ(problem.problemSizes(), fresh.problemSizes());
            EXPECT_DOUBLE_EQ(problem.arithmeticIntensity(), fresh.arithmeticIntensity())
                << g.m << "x" << g.n << "x" << g.k << ", beta " << g.beta;
        }
    }

    TEST(ContractionProblemUpdate, smoke_ArithmeticIntensityPredicatesSeeNewShape)
    {
        Predicates::Contraction::AIGreaterThanEqual atLeast100(100);
        Predicates::Contraction::AILessThanEqual    atMost100(100);

        auto problem = build(placeholder);
        update(problem, {rocisa::DataType::Half, 1024, 8192, 5120, 0.0});
        EXPECT_TRUE(atLeast100(problem));
        EXPECT_FALSE(atMost100(problem));

        update(problem, {rocisa::DataType::Half, 64, 64, 64, 1.0});
        EXPECT_FALSE(atLeast100(problem));
        EXPECT_TRUE(atMost100(problem));
    }
} // namespace
