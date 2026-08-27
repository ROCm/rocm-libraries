// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/client/MatmulPreparation.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{
    Arguments baseArguments()
    {
        Arguments arguments{};
        arguments.init();
        arguments.transA = 'N';
        arguments.transB = 'N';
        arguments.a_type = HIP_R_16F;
        arguments.b_type = HIP_R_16BF;
        arguments.c_type = HIP_R_32F;
        arguments.d_type = HIP_R_32F;
        arguments.M[0]   = 3;
        arguments.N[0]   = 5;
        arguments.K[0]   = 7;
        arguments.lda[0] = 4;
        arguments.ldb[0] = 8;
        arguments.ldc[0] = 6;
        arguments.ldd[0] = 9;
        arguments.lde[0] = 10;
        return arguments;
    }

    hipblaslt::client::MatmulPreparation
        prepare(const Arguments& arguments, bool swizzleA = false, bool swizzleB = false)
    {
        const auto problems  = hipblaslt::client::normalizeMatmulProblems(arguments);
        const auto dataTypes = hipblaslt::client::resolveMatmulDataTypes(arguments);
        return hipblaslt::client::prepareMatmulProblems(arguments,
                                                        problems,
                                                        arguments.a_type,
                                                        arguments.b_type,
                                                        arguments.c_type,
                                                        arguments.d_type,
                                                        dataTypes.computeScalar,
                                                        dataTypes.coefficient,
                                                        dataTypes.biasStorage,
                                                        swizzleA,
                                                        swizzleB,
                                                        false);
    }

    void expectMatrix(const hipblaslt::client::MatmulMatrix& matrix,
                      hipDataType                            type,
                      size_t                                 rows,
                      size_t                                 columns,
                      size_t                                 batches,
                      ptrdiff_t                              leadingDimension,
                      ptrdiff_t                              batchStride,
                      size_t                                 allocationElements)
    {
        EXPECT_EQ(matrix.apiType, type);
        EXPECT_EQ(matrix.hostType, hipblaslt::host_validation::scalarType(type));
        ASSERT_EQ(matrix.layout.rank(), 3);
        EXPECT_EQ(matrix.layout.extent(0), rows);
        EXPECT_EQ(matrix.layout.extent(1), columns);
        EXPECT_EQ(matrix.layout.extent(2), batches);
        EXPECT_EQ(matrix.layout.stride(0), 1);
        EXPECT_EQ(matrix.layout.stride(1), leadingDimension);
        EXPECT_EQ(matrix.layout.stride(2), batchStride);
        EXPECT_EQ(matrix.allocationElements, allocationElements);
    }
} // namespace

TEST(MatmulProblem, NormalizesLogicalMatrixGeometry)
{
    auto arguments   = baseArguments();
    arguments.transB = 'T';

    const auto problems = hipblaslt::client::normalizeMatmulProblems(arguments);

    ASSERT_EQ(problems.size(), 1);
    const auto& problem = problems.front();
    EXPECT_EQ(problem.m, 3);
    EXPECT_EQ(problem.n, 5);
    EXPECT_EQ(problem.k, 7);
    EXPECT_EQ(problem.operationA, HIPBLAS_OP_N);
    EXPECT_EQ(problem.operationB, HIPBLAS_OP_T);
    EXPECT_EQ(problem.batchMode, HIPBLASLT_BATCH_MODE_STRIDED);
    EXPECT_EQ(problem.batchCount, 1);
    expectMatrix(problem.a, HIP_R_16F, 3, 7, 1, 4, 28, 28);
    expectMatrix(problem.b, HIP_R_16BF, 5, 7, 1, 8, 56, 56);
    expectMatrix(problem.c, HIP_R_32F, 3, 5, 1, 6, 30, 30);
    expectMatrix(problem.d, HIP_R_32F, 3, 5, 1, 9, 45, 45);
    EXPECT_FALSE(problem.auxiliary);
}

TEST(MatmulProblem, KeepsDistinctStridedBatchGeometry)
{
    auto arguments        = baseArguments();
    arguments.batch_count = 4;
    arguments.stride_a[0] = 0;
    arguments.stride_b[0] = 202;
    arguments.stride_c[0] = 303;
    arguments.stride_d[0] = 404;
    arguments.stride_e[0] = 505;
    arguments.use_e       = true;
    arguments.aux_type    = HIP_R_16F;

    const auto  problems = hipblaslt::client::normalizeMatmulProblems(arguments);
    const auto& problem  = problems.front();

    expectMatrix(problem.a, HIP_R_16F, 3, 7, 4, 4, 0, 112);
    expectMatrix(problem.b, HIP_R_16BF, 7, 5, 4, 8, 202, 808);
    expectMatrix(problem.c, HIP_R_32F, 3, 5, 4, 6, 303, 1212);
    expectMatrix(problem.d, HIP_R_32F, 3, 5, 4, 9, 404, 1616);
    ASSERT_TRUE(problem.auxiliary);
    expectMatrix(*problem.auxiliary, HIP_R_16F, 3, 5, 4, 10, 505, 2020);
}

TEST(MatmulProblem, PointerArraysUseCanonicalLogicalBatchOffsets)
{
    auto arguments        = baseArguments();
    arguments.batch_mode  = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
    arguments.batch_count = 3;
    arguments.stride_a[0] = 101;
    arguments.stride_b[0] = 202;
    arguments.stride_c[0] = 303;
    arguments.stride_d[0] = 404;

    const auto  problems = hipblaslt::client::normalizeMatmulProblems(arguments);
    const auto& problem  = problems.front();

    EXPECT_EQ(problem.batchMode, HIPBLASLT_BATCH_MODE_POINTER_ARRAY);
    expectMatrix(problem.a, HIP_R_16F, 3, 7, 3, 4, 28, 28);
    expectMatrix(problem.b, HIP_R_16BF, 7, 5, 3, 8, 40, 40);
    expectMatrix(problem.c, HIP_R_32F, 3, 5, 3, 6, 30, 30);
    expectMatrix(problem.d, HIP_R_32F, 3, 5, 3, 9, 45, 45);
}

TEST(MatmulProblem, NormalizesEachGroupedProblem)
{
    auto arguments         = baseArguments();
    arguments.grouped_gemm = 2;
    arguments.batch_count  = 2;
    arguments.M[1]         = 11;
    arguments.N[1]         = 13;
    arguments.K[1]         = 17;
    arguments.lda[1]       = 12;
    arguments.ldb[1]       = 18;
    arguments.ldc[1]       = 14;
    arguments.ldd[1]       = 15;
    arguments.lde[1]       = 16;
    arguments.stride_a[1]  = 1001;
    arguments.stride_b[1]  = 1002;
    arguments.stride_c[1]  = 1003;
    arguments.stride_d[1]  = 1004;
    arguments.stride_e[1]  = 1005;

    const auto problems = hipblaslt::client::normalizeMatmulProblems(arguments);

    ASSERT_EQ(problems.size(), 2);
    EXPECT_EQ(problems[1].m, 11);
    EXPECT_EQ(problems[1].n, 13);
    EXPECT_EQ(problems[1].k, 17);
    expectMatrix(problems[1].a, HIP_R_16F, 11, 17, 2, 12, 1001, 2002);
    expectMatrix(problems[1].b, HIP_R_16BF, 17, 13, 2, 18, 1002, 2004);
    expectMatrix(problems[1].c, HIP_R_32F, 11, 13, 2, 14, 1003, 2006);
    expectMatrix(problems[1].d, HIP_R_32F, 11, 13, 2, 15, 1004, 2008);
}

TEST(MatmulProblem, MakesCEqualsDExplicit)
{
    auto arguments        = baseArguments();
    arguments.c_equal_d   = true;
    arguments.batch_count = 2;
    arguments.stride_c[0] = 303;
    arguments.stride_d[0] = 404;

    const auto  problems = hipblaslt::client::normalizeMatmulProblems(arguments);
    const auto& problem  = problems.front();

    EXPECT_TRUE(problem.cEqualsD);
    EXPECT_EQ(problem.c.layout, problem.d.layout);
    EXPECT_EQ(problem.d.layout.stride(1), arguments.ldc[0]);
    EXPECT_EQ(problem.d.layout.stride(2), arguments.stride_c[0]);
}

TEST(MatmulProblem, RejectsInvalidSerializedGeometry)
{
    auto arguments       = baseArguments();
    arguments.batch_mode = 2;
    EXPECT_THROW(hipblaslt::client::normalizeMatmulProblems(arguments), std::invalid_argument);

    arguments              = baseArguments();
    arguments.grouped_gemm = MAX_SUPPORTED_NUM_PROBLEMS + 1;
    EXPECT_THROW(hipblaslt::client::normalizeMatmulProblems(arguments), std::invalid_argument);

    arguments      = baseArguments();
    arguments.M[0] = -1;
    EXPECT_THROW(hipblaslt::client::normalizeMatmulProblems(arguments), std::invalid_argument);

    arguments           = baseArguments();
    arguments.c_equal_d = true;
    arguments.d_type    = HIP_R_16F;
    EXPECT_THROW(hipblaslt::client::normalizeMatmulProblems(arguments), std::invalid_argument);
}

TEST(MatmulDataTypes, ResolvesDefaultsOnce)
{
    const auto dataTypes = hipblaslt::client::resolveMatmulDataTypes(baseArguments());

    EXPECT_EQ(dataTypes.computeScalar, HIP_R_32F);
    EXPECT_EQ(dataTypes.computeInputA, HIP_R_32F);
    EXPECT_EQ(dataTypes.computeInputB, HIP_R_32F);
    EXPECT_EQ(dataTypes.coefficient, HIP_R_32F);
    EXPECT_EQ(dataTypes.bias, HIP_R_32F);
    EXPECT_EQ(dataTypes.biasStorage, HIP_R_32F);
    EXPECT_EQ(dataTypes.auxiliary, HIP_R_32F);
}

TEST(MatmulPreparation, ComputesLogicalStorageAndScalarState)
{
    const auto preparation = prepare(baseArguments());

    ASSERT_EQ(preparation.problems.size(), 1);
    const auto& preparedProblem = preparation.problems.front();
    EXPECT_EQ(preparedProblem.a.elements, 28);
    EXPECT_EQ(preparedProblem.b.elements, 40);
    EXPECT_EQ(preparedProblem.outputCopyElements, 45);
    EXPECT_EQ(preparedProblem.alpha.f32, 1.0f);
    EXPECT_EQ(preparedProblem.beta.f32, 0.0f);
    EXPECT_EQ(preparation.rotatingBytes, 316);
}

TEST(MatmulPreparation, CountsPointerArrayStoragePerBatch)
{
    auto arguments        = baseArguments();
    arguments.batch_mode  = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
    arguments.batch_count = 3;

    const auto preparation = prepare(arguments);

    ASSERT_EQ(preparation.problems.size(), 1);
    EXPECT_EQ(preparation.problems.front().a.elements, 28);
    EXPECT_EQ(preparation.problems.front().b.elements, 40);
    EXPECT_EQ(preparation.rotatingBytes, 948);
}

TEST(MatmulPreparation, CountsDistinctCAndDStorageTypes)
{
    auto arguments   = baseArguments();
    arguments.c_type = HIP_R_16F;
    arguments.d_type = HIP_R_32F;
    arguments.beta   = 1.0;

    const auto preparation = prepare(arguments);

    ASSERT_EQ(preparation.problems.size(), 1);
    EXPECT_EQ(preparation.rotatingBytes, 376);
}

TEST(MatmulPreparation, IsolatesSwizzledDeviceGeometry)
{
    auto arguments        = baseArguments();
    arguments.batch_count = 4;
    arguments.stride_a[0] = 29;

    const auto preparation = prepare(arguments, true);

    ASSERT_EQ(preparation.problems.size(), 1);
    const auto& preparedA = preparation.problems.front().a;
    EXPECT_EQ(preparedA.batchStride, 512);
    EXPECT_EQ(preparedA.elements, 2048);
    EXPECT_TRUE(preparedA.replacedUnsupportedBatchStride);
}

TEST(MatmulPreparation, MakesScaleAlphaVectorAUnitScalarEpilogue)
{
    auto arguments              = baseArguments();
    arguments.alpha             = 3.0;
    arguments.scaleAlpha_vector = true;

    const auto preparation = prepare(arguments);

    ASSERT_EQ(preparation.problems.size(), 1);
    const auto& preparedProblem = preparation.problems.front();
    EXPECT_EQ(preparedProblem.scaleAlphaElements, 3);
    EXPECT_EQ(preparedProblem.alpha.f32, 1.0f);
    EXPECT_TRUE(preparedProblem.epilogueEnabled);
}
