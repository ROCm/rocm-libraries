/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem.hpp>

#include <cstddef>

using namespace Tensile;

TEST(ContractionProblem, OperationDescription)
{
    EXPECT_EQ(ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2).beta(), 1.5);

    EXPECT_EQ(ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[ilk] * B[ljk]) + beta * C[ijk]");

    EXPECT_EQ(ContractionProblem::GEMM(false, true, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[ilk] * B[jlk]) + beta * C[ijk]");

    EXPECT_EQ(ContractionProblem::GEMM(true, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[lik] * B[ljk]) + beta * C[ijk]");

    EXPECT_EQ(ContractionProblem::GEMM(true, true, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[lik] * B[jlk]) + beta * C[ijk]");

    EXPECT_EQ(ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[ilk] * B[ljk])");

    EXPECT_EQ(ContractionProblem::GEMM(false, true, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[ilk] * B[jlk])");

    EXPECT_EQ(ContractionProblem::GEMM(true, false, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[lik] * B[ljk])");

    EXPECT_EQ(ContractionProblem::GEMM(true, true, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationDescription(),
              "D[ijk] = alpha * (Sum[l] A[lik] * B[jlk])");
}

TEST(ContractionProblem, OperationIdentifier)
{
    EXPECT_EQ(ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Ailk_Bljk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(false, true, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Ailk_Bjlk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(true, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Alik_Bljk_Cijk_Dijk");

    EXPECT_EQ(
        ContractionProblem::GEMM(true, true, 4, 4, 4, 4, 4, 4, 1.5, false, 2).operationIdentifier(),
        "Contraction_l_Alik_Bjlk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.0, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Ailk_Bljk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(false, true, 4, 4, 4, 4, 4, 4, 1.0, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Ailk_Bjlk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(true, false, 4, 4, 4, 4, 4, 4, 1.0, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Alik_Bljk_Cijk_Dijk");

    EXPECT_EQ(
        ContractionProblem::GEMM(true, true, 4, 4, 4, 4, 4, 4, 1.0, false, 2).operationIdentifier(),
        "Contraction_l_Alik_Bjlk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Ailk_Bljk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(false, true, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Ailk_Bjlk_Cijk_Dijk");

    EXPECT_EQ(ContractionProblem::GEMM(true, false, 4, 4, 4, 4, 4, 4, 0.0, false, 2)
                  .operationIdentifier(),
              "Contraction_l_Alik_Bljk_Cijk_Dijk");

    EXPECT_EQ(
        ContractionProblem::GEMM(true, true, 4, 4, 4, 4, 4, 4, 0.0, false, 2).operationIdentifier(),
        "Contraction_l_Alik_Bjlk_Cijk_Dijk");
}

TEST(ContractionProblem, FromOperationIdentifier)
{
    std::vector<size_t> sizes{5, 6, 4, 2};
    std::vector<size_t> empty;
    std::string         identifier = "Contraction_l_AlikC_Bjlk_Cijk_Dijk";
    auto                problem    = ContractionProblem::FromIndexSizes(identifier,
                                                      sizes,
                                                      DataType::ComplexFloat,
                                                      empty,
                                                      DataType::ComplexFloat,
                                                      empty,
                                                      DataType::ComplexFloat,
                                                      empty,
                                                      DataType::ComplexFloat,
                                                      empty,
                                                      2.0);

    EXPECT_EQ(problem.operationIdentifier(), identifier);

    identifier         = "Contraction_l_ALik_Bjlk_Cijk_Dijk";
    auto mirrorProblem = ContractionProblem::FromIndexSizes(identifier,
                                                            sizes,
                                                            DataType::Float,
                                                            empty,
                                                            DataType::Float,
                                                            empty,
                                                            DataType::Float,
                                                            empty,
                                                            DataType::Float,
                                                            empty,
                                                            2.0);
    EXPECT_TRUE(mirrorProblem.boundIndices()[0].aMirror);
    EXPECT_FALSE(mirrorProblem.boundIndices()[0].bMirror);
    EXPECT_EQ(mirrorProblem.operationIdentifier(), identifier);
}

#if 0
TEST(ContractionProblem, Simple)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    ContractionProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), false);
    EXPECT_EQ(p.blas_transB(), false);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 1534);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 2147);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(ContractionProblem, TransposeA)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {2147, 1534, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    a.transpose(0,1);

    ContractionProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), true);
    EXPECT_EQ(p.blas_transB(), false);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 2147);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 2147);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(ContractionProblem, TransposeB)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28});
    TensorDescriptor b(DataType::Float, {3481, 2147, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    b.transpose(0,1);

    ContractionProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), false);
    EXPECT_EQ(p.blas_transB(), true);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 1534);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 3481);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(ContractionProblem, TransposeAB)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {2147, 1534, 28});
    TensorDescriptor b(DataType::Float, {3481, 2147, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    a.transpose(0,1);
    b.transpose(0,1);

    ContractionProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), true);
    EXPECT_EQ(p.blas_transB(), true);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 2147);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 3481);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(ContractionProblem, Padding)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28}, {1536, 2147, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28}, {2176, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28}, {1536, 3481, 28});
    TensorDescriptor d(DataType::Float, {1534, 3481, 28}, {1568, 3481, 28});

    ContractionProblem p(a, noOps, b, noOps, c, noOps, d, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), false);
    EXPECT_EQ(p.blas_transB(), false);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 1536);
    EXPECT_EQ(p.tensile_strideA2(), 1536*2147);

    EXPECT_EQ(p.tensile_strideB1(), 2176);
    EXPECT_EQ(p.tensile_strideB2(), 2176*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1536);
    EXPECT_EQ(p.tensile_strideC2(), 1536*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1568);
    EXPECT_EQ(p.tensile_strideD2(), 1568*3481);
}

TEST(ContractionProblem, Bad)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {2147, 1534, 28});
    TensorDescriptor b(DataType::Float, {3481, 2147, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    TensorDescriptor b_batch(DataType::Float, {3481, 2147, 12});

    EXPECT_THROW(ContractionProblem(a, noOps, a, noOps, c, noOps, c, noOps, false), std::runtime_error);
    EXPECT_THROW(ContractionProblem(a, noOps, b, noOps, c, noOps, a, noOps, false), std::runtime_error);
    EXPECT_THROW(ContractionProblem(a, noOps, b, noOps, a, noOps, c, noOps, false), std::runtime_error);

    EXPECT_THROW(ContractionProblem(a, noOps, b_batch, noOps, c, noOps, c, noOps, false), std::runtime_error);

    a.transpose(0,1);
    b.transpose(0,1);

    ContractionProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), true);
    EXPECT_EQ(p.blas_transB(), true);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 2147);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 3481);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

#endif
