/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

// A mode shared by both contraction inputs and the output is a batch dimension,
// which the current M/N/K normalized layout cannot express. hiptensorCreate*
// must reject such descriptors with HIPTENSOR_STATUS_NOT_SUPPORTED rather than
// silently producing a wrong result. See ROCm/ROCm#6559.

namespace
{
    // 'b' is the shared batch mode; 'm','n','k' are the usual free/contracted modes.
    constexpr int32_t bMode = 'b';
    constexpr int32_t mMode = 'm';
    constexpr int32_t nMode = 'n';
    constexpr int32_t kMode = 'k';
} // namespace

// D[b,m,n] = sum_k A[b,m,k] * B[b,k,n] -- 'b' shared by A, B and D.
TEST(BatchedContractionTest, BinaryBatchedReturnsNotSupported)
{
    hiptensorHandle_t handle{};
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    int64_t lensA[3] = {2, 3, 4};
    int64_t lensB[3] = {2, 4, 5};
    int64_t lensD[3] = {2, 3, 5};
    int32_t modeA[3] = {bMode, mMode, kMode};
    int32_t modeB[3] = {bMode, kMode, nMode};
    int32_t modeD[3] = {bMode, mMode, nMode};

    hiptensorTensorDescriptor_t descA{}, descB{}, descD{};
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descA, 3, lensA, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descB, 3, lensB, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descD, 3, lensD, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);

    hiptensorOperationDescriptor_t opDesc{};
    EXPECT_EQ(hiptensorCreateContraction(handle,
                                         &opDesc,
                                         descA,
                                         modeA,
                                         HIPTENSOR_OP_IDENTITY,
                                         descB,
                                         modeB,
                                         HIPTENSOR_OP_IDENTITY,
                                         descD,
                                         modeD,
                                         HIPTENSOR_OP_IDENTITY,
                                         descD,
                                         modeD,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_NOT_SUPPORTED);

    hiptensorDestroyTensorDescriptor(descA);
    hiptensorDestroyTensorDescriptor(descB);
    hiptensorDestroyTensorDescriptor(descD);
    hiptensorDestroy(handle);
}

// Non-batched control D[m,n] = sum_k A[m,k] * B[k,n] must still be accepted.
TEST(BatchedContractionTest, BinaryNonBatchedIsSupported)
{
    hiptensorHandle_t handle{};
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    int64_t lensA[2] = {3, 4};
    int64_t lensB[2] = {4, 5};
    int64_t lensD[2] = {3, 5};
    int32_t modeA[2] = {mMode, kMode};
    int32_t modeB[2] = {kMode, nMode};
    int32_t modeD[2] = {mMode, nMode};

    hiptensorTensorDescriptor_t descA{}, descB{}, descD{};
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descA, 2, lensA, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descB, 2, lensB, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descD, 2, lensD, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);

    hiptensorOperationDescriptor_t opDesc{};
    EXPECT_EQ(hiptensorCreateContraction(handle,
                                         &opDesc,
                                         descA,
                                         modeA,
                                         HIPTENSOR_OP_IDENTITY,
                                         descB,
                                         modeB,
                                         HIPTENSOR_OP_IDENTITY,
                                         descD,
                                         modeD,
                                         HIPTENSOR_OP_IDENTITY,
                                         descD,
                                         modeD,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);

    if(opDesc)
        hiptensorDestroyOperationDescriptor(opDesc);
    hiptensorDestroyTensorDescriptor(descA);
    hiptensorDestroyTensorDescriptor(descB);
    hiptensorDestroyTensorDescriptor(descD);
    hiptensorDestroy(handle);
}

// E[b,m,n] = alpha * A[b,m,k] * B[b,k,n] * C[b,m,n] + ... -- 'b' shared by A, B and E.
TEST(BatchedContractionTest, TrinaryBatchedReturnsNotSupported)
{
    hiptensorHandle_t handle{};
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    int64_t lensA[3] = {2, 3, 4};
    int64_t lensB[3] = {2, 4, 5};
    int64_t lensC[3] = {2, 3, 5};
    int64_t lensE[3] = {2, 3, 5};
    int32_t modeA[3] = {bMode, mMode, kMode};
    int32_t modeB[3] = {bMode, kMode, nMode};
    int32_t modeC[3] = {bMode, mMode, nMode};
    int32_t modeE[3] = {bMode, mMode, nMode};

    hiptensorTensorDescriptor_t descA{}, descB{}, descC{}, descE{};
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descA, 3, lensA, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descB, 3, lensB, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descC, 3, lensC, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(handle, &descE, 3, lensE, nullptr, HIPTENSOR_R_32F, 4),
              HIPTENSOR_STATUS_SUCCESS);

    hiptensorOperationDescriptor_t opDesc{};
    EXPECT_EQ(hiptensorCreateContractionTrinary(handle,
                                                &opDesc,
                                                descA,
                                                modeA,
                                                HIPTENSOR_OP_IDENTITY,
                                                descB,
                                                modeB,
                                                HIPTENSOR_OP_IDENTITY,
                                                descC,
                                                modeC,
                                                HIPTENSOR_OP_IDENTITY,
                                                nullptr,
                                                nullptr,
                                                HIPTENSOR_OP_IDENTITY,
                                                descE,
                                                modeE,
                                                HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_NOT_SUPPORTED);

    hiptensorDestroyTensorDescriptor(descA);
    hiptensorDestroyTensorDescriptor(descB);
    hiptensorDestroyTensorDescriptor(descC);
    hiptensorDestroyTensorDescriptor(descE);
    hiptensorDestroy(handle);
}
