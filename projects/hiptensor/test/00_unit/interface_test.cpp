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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>

#include <contraction/contraction_solution_instances.hpp>
#include <data_types_impl.hpp>
#include <handle.hpp>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <reduction/reduction_solution.hpp>
#include <reduction/reduction_solution_instances.hpp>

TEST(hiptensorGetErrorStringTest, UtilTest)
{
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_SUCCESS), "HIPTENSOR_STATUS_SUCCESS");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_NOT_INITIALIZED),
                 "HIPTENSOR_STATUS_NOT_INITIALIZED");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_ALLOC_FAILED),
                 "HIPTENSOR_STATUS_ALLOC_FAILED");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_INVALID_VALUE),
                 "HIPTENSOR_STATUS_INVALID_VALUE");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_ARCH_MISMATCH),
                 "HIPTENSOR_STATUS_ARCH_MISMATCH");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_EXECUTION_FAILED),
                 "HIPTENSOR_STATUS_EXECUTION_FAILED");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_INTERNAL_ERROR),
                 "HIPTENSOR_STATUS_INTERNAL_ERROR");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_NOT_SUPPORTED),
                 "HIPTENSOR_STATUS_NOT_SUPPORTED");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_CK_ERROR), "HIPTENSOR_STATUS_CK_ERROR");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_HIP_ERROR), "HIPTENSOR_STATUS_HIP_ERROR");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE),
                 "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_INSUFFICIENT_DRIVER),
                 "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_IO_ERROR), "HIPTENSOR_STATUS_IO_ERROR");
}

TEST(hiptensorGetAlignmentRequirementTest, UtilTest)
{
    uint32_t alignmentRequirement = 0;

    // fail for !handle
    auto output
        = hiptensorGetAlignmentRequirement(nullptr, nullptr, nullptr, &alignmentRequirement);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED);

    // fail for !desc
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    output = hiptensorGetAlignmentRequirement(handle, nullptr, nullptr, &alignmentRequirement);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED);

    // fail for (*alignmentRequirement == 0)
    hiptensorTensorDescriptor desc;
    desc.mType = HIPTENSOR_R_32F;
    void* ptr  = reinterpret_cast<void*>(5);
    output     = hiptensorGetAlignmentRequirement(handle, ptr, desc, &alignmentRequirement);
    EXPECT_EQ(output, HIPTENSOR_STATUS_INVALID_VALUE);
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
}

TEST(hiptensorReductionTest, UtilTest)
{
    char                         buf[1];
    hiptensorHandle_t            handle;
    float                        alpha, beta;
    const void*                  A = nullptr;
    const void*                  C = &buf;
    void*                        D = &buf;
    hiptensorTensorDescriptor    descA;
    int32_t                      modeA[1];
    hiptensorTensorDescriptor    descC;
    int32_t                      modeC[1];
    hiptensorTensorDescriptor    descD;
    int32_t                      modeD[1];
    hiptensorOperator_t          opReduce;
    hiptensorComputeDescriptor_t typeCompute;
    void*                        workspace     = nullptr;
    uint64_t                     workspaceSize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    auto output = hiptensorReduction(handle,
                                     &alpha,
                                     A,
                                     descA,
                                     modeA,
                                     &beta,
                                     C,
                                     descC,
                                     modeC,
                                     D,
                                     descD,
                                     modeD,
                                     opReduce,
                                     typeCompute,
                                     workspace,
                                     workspaceSize,
                                     0);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED); // fail for A is null

    A           = &buf;
    descA.mType = HIPTENSOR_R_64F;
    descC.mType = HIPTENSOR_R_64F;
    descD.mType = HIPTENSOR_R_64F;
    typeCompute = HIPTENSOR_COMPUTE_DESC_16F;
    output      = hiptensorReduction(handle,
                                &alpha,
                                A,
                                descA,
                                modeA,
                                &beta,
                                C,
                                descC,
                                modeC,
                                D,
                                descD,
                                modeD,
                                opReduce,
                                typeCompute,
                                workspace,
                                workspaceSize,
                                0);
    EXPECT_EQ(
        output,
        HIPTENSOR_STATUS_NOT_SUPPORTED); // fail for (HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_16F) is not supported

    descA.mType    = HIPTENSOR_R_16F;
    descA.mLengths = {1};
    descC.mType    = HIPTENSOR_R_16F;
    descC.mLengths = {1, 1};
    descD.mType    = HIPTENSOR_R_16F;
    typeCompute    = HIPTENSOR_COMPUTE_DESC_16F;
    output         = hiptensorReduction(handle,
                                &alpha,
                                A,
                                descA,
                                modeA,
                                &beta,
                                C,
                                descC,
                                modeC,
                                D,
                                descD,
                                modeD,
                                opReduce,
                                typeCompute,
                                workspace,
                                workspaceSize,
                                0);
    EXPECT_EQ(
        output,
        HIPTENSOR_STATUS_NOT_SUPPORTED); // fail for descA.mLengths.size() < descC.mLengths.size()
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
}

TEST(hiptensorInitContractionDescriptorTest, UtilTest)
{
    char                             buf[1];
    hiptensorContractionDescriptor_t desc;
    hiptensorHandle_t                handle;
    hiptensorTensorDescriptor        descA;
    int32_t                          modeA[1];
    hiptensorTensorDescriptor        descB;
    int32_t                          modeB[1];
    hiptensorTensorDescriptor        descC;
    int32_t                          modeC[1];
    hiptensorTensorDescriptor        descD;
    int32_t                          modeD[1];
    const uint32_t                   alignmentRequirementA = 0;
    const uint32_t                   alignmentRequirementB = 0;
    const uint32_t                   alignmentRequirementC = 0;
    const uint32_t                   alignmentRequirementD = 0;
    hiptensorComputeDescriptor_t     typeCompute;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    auto output = hiptensorInitContractionDescriptor(handle,
                                                     nullptr,
                                                     descA,
                                                     modeA,
                                                     alignmentRequirementA,
                                                     descB,
                                                     modeB,
                                                     alignmentRequirementB,
                                                     descC,
                                                     modeC,
                                                     alignmentRequirementC,
                                                     descD,
                                                     modeD,
                                                     alignmentRequirementD,
                                                     typeCompute);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED); // fail for desc is nullptr

    // descA.mUnaryOp = HIPTENSOR_OP_NEG;
    output = hiptensorInitContractionDescriptor(handle,
                                                desc,
                                                descA,
                                                modeA,
                                                alignmentRequirementA,
                                                descB,
                                                modeB,
                                                alignmentRequirementB,
                                                descC,
                                                modeC,
                                                alignmentRequirementC,
                                                descD,
                                                modeD,
                                                alignmentRequirementD,
                                                typeCompute);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_SUPPORTED); // fail for opA != HIPTENSOR_OP_IDENTITY
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
}

TEST(hiptensorInitContractionFindTest, UtilTest)
{
    hiptensorHandle_t          handle;
    hiptensorContractionFind_t find;
    hiptensorAlgo_t            algo;
    auto                       output = hiptensorInitContractionFind(nullptr, &find, algo);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED); // fail for handle is nullptr

    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    handle->fields[0]
        = 0xFFFFFFFFFFFFFFFF; // A hack method that forcefully changes the data of the underlying handle. During subsequent comparisons, the underlying handle represented by the handle is invalid.
    output = hiptensorInitContractionFind(handle, &find, algo);
    EXPECT_EQ(
        output,
        HIPTENSOR_STATUS_ARCH_MISMATCH); // fail for currentDevice.getDeviceId() != handle->mDevice.getDeviceId()
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));

    hiptensorHandle_t handlePtr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handlePtr));
    output = hiptensorInitContractionFind(handlePtr, &find, algo);
    EXPECT_EQ(output, HIPTENSOR_STATUS_INVALID_VALUE); // fail for invalid algo

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handlePtr));
}

TEST(hiptensorContractionGetWorkspaceSizeTest, UtilTest)
{
    hiptensorContractionDescriptor_t desc;
    hiptensorContractionFind_t       find;
    hiptensorWorksizePreference_t    pref;
    uint64_t                         workspaceSize;
    auto output = hiptensorContractionGetWorkspaceSize(nullptr, &desc, &find, pref, &workspaceSize);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED); // fail for handle is null
}

TEST(hiptensorInitContractionPlanTest, UtilTest)
{
    hiptensorHandle_t                handle;
    hiptensorContractionPlan_t       plan;
    hiptensorContractionDescriptor_t desc;
    hiptensorContractionFind_t       find;
    uint64_t                         workspaceSize;
    auto output = hiptensorInitContractionPlan(nullptr, &plan, &desc, &find, workspaceSize);
    EXPECT_EQ(output, HIPTENSOR_STATUS_NOT_INITIALIZED); // fail for handle is null

    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    handle->fields[0]
        = 0xFFFFFFFFFFFFFFFF; // A hack method that forcefully changes the data of the underlying handle. During subsequent comparisons, the underlying handle represented by the handle is invalid.
    output = hiptensorInitContractionPlan(handle, &plan, &desc, &find, workspaceSize);
    EXPECT_EQ(output, HIPTENSOR_STATUS_ARCH_MISMATCH); // fail for handle is null
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
}

TEST(hiptensorContractionTest, UtilTest)
{
    char                       buf[1];
    hiptensorHandle_t          handle;
    hiptensorContractionPlan_t plan;
    float                      alpha, beta;
    void*                      A         = &buf;
    void*                      B         = &buf;
    void*                      C         = &buf;
    void*                      D         = &buf;
    void*                      workspace = &buf;
    uint64_t                   workspaceSize;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    plan.mSolution = nullptr;
    auto output    = hiptensorContraction(
        handle, &plan, &alpha, A, B, &beta, C, D, workspace, workspaceSize, 0);
    EXPECT_EQ(output, HIPTENSOR_STATUS_INTERNAL_ERROR); // fail for plan->mSolution is null

    plan.mSolution = &buf;
    handle->fields[0]
        = 0xFFFFFFFFFFFFFFFF; // A hack method that forcefully changes the data of the underlying handle. During subsequent comparisons, the underlying handle represented by the handle is invalid.
    output = hiptensorContraction(
        handle, &plan, &alpha, A, B, &beta, C, D, workspace, workspaceSize, 0);
    EXPECT_EQ(output,
              HIPTENSOR_STATUS_ARCH_MISMATCH); // fail for handle->mDevice.getDeviceId() == -1
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
}
