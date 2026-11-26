/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// The scaling factors of an operation take the scalar type its compute descriptor maps to, so
// every operation descriptor create function must reject a compute descriptor without one
// (HIPTENSOR_COMPUTE_DESC_NONE, the integer descriptors and unknown values) with
// HIPTENSOR_STATUS_INVALID_VALUE, and must not hand back a descriptor. A floating point compute
// descriptor is checked alongside them so that a failure cannot come from the tensors or modes.

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

#include <string>
#include <tuple>

namespace
{
    constexpr int32_t mMode = 'm';
    constexpr int32_t nMode = 'n';
    constexpr int32_t kMode = 'k';
    constexpr int32_t pMode = 'p';

    constexpr uint32_t alignment = 4;

    // Within the enum's value range, but not one of its enumerators.
    constexpr auto unknownComputeDesc = static_cast<hiptensorComputeDescriptor_t>(1U << 1U);

    std::string computeDescName(hiptensorComputeDescriptor_t descCompute)
    {
        switch(descCompute)
        {
        case HIPTENSOR_COMPUTE_DESC_32F:
            return "COMPUTE_DESC_32F";
        case HIPTENSOR_COMPUTE_DESC_NONE:
            return "COMPUTE_DESC_NONE";
        case HIPTENSOR_COMPUTE_DESC_8U:
            return "COMPUTE_DESC_8U";
        case HIPTENSOR_COMPUTE_DESC_8I:
            return "COMPUTE_DESC_8I";
        case HIPTENSOR_COMPUTE_DESC_32U:
            return "COMPUTE_DESC_32U";
        case HIPTENSOR_COMPUTE_DESC_32I:
            return "COMPUTE_DESC_32I";
        default:
            return "Unknown";
        }
    }
} // namespace

// Holds a handle and FP32 tensors {m,k} (A), {k,n} (B), {m,n}, {n,p}, {m,p} and {m}, enough to
// build every kind of operation descriptor. The operation descriptor under test is released in
// TearDown whatever the outcome of the case.
class ComputeDescriptorTest : public ::testing::TestWithParam<hiptensorComputeDescriptor_t>
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);

        for(auto const& [desc, rank, lengths] : {std::tuple{&mMK, 2u, mLensMK},
                                                 std::tuple{&mKN, 2u, mLensKN},
                                                 std::tuple{&mMN, 2u, mLensMN},
                                                 std::tuple{&mNP, 2u, mLensNP},
                                                 std::tuple{&mMP, 2u, mLensMP},
                                                 std::tuple{&mM, 1u, mLensM}})
        {
            ASSERT_EQ(hiptensorCreateTensorDescriptor(
                          mHandle, desc, rank, lengths, nullptr, HIPTENSOR_R_32F, alignment),
                      HIPTENSOR_STATUS_SUCCESS);
        }
    }

    void TearDown() override
    {
        if(mOpDesc)
        {
            hiptensorDestroyOperationDescriptor(mOpDesc);
        }
        for(auto desc : {mM, mMP, mNP, mMN, mKN, mMK})
        {
            if(desc)
            {
                hiptensorDestroyTensorDescriptor(desc);
            }
        }
        if(mHandle)
        {
            hiptensorDestroy(mHandle);
        }
    }

    // Checks the status of a create call for the compute descriptor under test.
    void checkCreate(hiptensorStatus_t status)
    {
        if(GetParam() == HIPTENSOR_COMPUTE_DESC_32F)
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
            EXPECT_NE(mOpDesc, nullptr);
        }
        else
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_INVALID_VALUE);
            EXPECT_EQ(mOpDesc, nullptr);
        }
    }

    hiptensorHandle_t              mHandle{};
    hiptensorTensorDescriptor_t    mMK{};
    hiptensorTensorDescriptor_t    mKN{};
    hiptensorTensorDescriptor_t    mMN{};
    hiptensorTensorDescriptor_t    mNP{};
    hiptensorTensorDescriptor_t    mMP{};
    hiptensorTensorDescriptor_t    mM{};
    hiptensorOperationDescriptor_t mOpDesc{};

    int64_t mLensMK[2] = {3, 4};
    int64_t mLensKN[2] = {4, 5};
    int64_t mLensMN[2] = {3, 5};
    int64_t mLensNP[2] = {5, 6};
    int64_t mLensMP[2] = {3, 6};
    int64_t mLensM[1]  = {3};

    int32_t mModesMK[2] = {mMode, kMode};
    int32_t mModesKN[2] = {kMode, nMode};
    int32_t mModesMN[2] = {mMode, nMode};
    int32_t mModesNP[2] = {nMode, pMode};
    int32_t mModesMP[2] = {mMode, pMode};
    int32_t mModesM[1]  = {mMode};
};

TEST_P(ComputeDescriptorTest, CreateContraction)
{
    auto status = hiptensorCreateContraction(mHandle,
                                             &mOpDesc,
                                             mMK,
                                             mModesMK,
                                             HIPTENSOR_OP_IDENTITY,
                                             mKN,
                                             mModesKN,
                                             HIPTENSOR_OP_IDENTITY,
                                             mMN,
                                             mModesMN,
                                             HIPTENSOR_OP_IDENTITY,
                                             mMN,
                                             mModesMN,
                                             GetParam());
    if(GetParam() == HIPTENSOR_COMPUTE_DESC_32F && status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "No FP32 contraction support on this device";
    }
    checkCreate(status);
}

TEST_P(ComputeDescriptorTest, CreateContractionTrinary)
{
    auto status = hiptensorCreateContractionTrinary(mHandle,
                                                    &mOpDesc,
                                                    mMK,
                                                    mModesMK,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    mKN,
                                                    mModesKN,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    mNP,
                                                    mModesNP,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    mMP,
                                                    mModesMP,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    mMP,
                                                    mModesMP,
                                                    GetParam());
    if(GetParam() == HIPTENSOR_COMPUTE_DESC_32F && status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "No FP32 contraction support on this device";
    }
    checkCreate(status);
}

TEST_P(ComputeDescriptorTest, CreatePermutation)
{
    checkCreate(hiptensorCreatePermutation(
        mHandle, &mOpDesc, mMN, mModesMN, HIPTENSOR_OP_IDENTITY, mMN, mModesMN, GetParam()));
}

TEST_P(ComputeDescriptorTest, CreateElementwiseBinary)
{
    checkCreate(hiptensorCreateElementwiseBinary(mHandle,
                                                 &mOpDesc,
                                                 mMN,
                                                 mModesMN,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 mMN,
                                                 mModesMN,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 mMN,
                                                 mModesMN,
                                                 HIPTENSOR_OP_ADD,
                                                 GetParam()));
}

TEST_P(ComputeDescriptorTest, CreateElementwiseTrinary)
{
    checkCreate(hiptensorCreateElementwiseTrinary(mHandle,
                                                  &mOpDesc,
                                                  mMN,
                                                  mModesMN,
                                                  HIPTENSOR_OP_IDENTITY,
                                                  mMN,
                                                  mModesMN,
                                                  HIPTENSOR_OP_IDENTITY,
                                                  mMN,
                                                  mModesMN,
                                                  HIPTENSOR_OP_IDENTITY,
                                                  mMN,
                                                  mModesMN,
                                                  HIPTENSOR_OP_ADD,
                                                  HIPTENSOR_OP_ADD,
                                                  GetParam()));
}

TEST_P(ComputeDescriptorTest, CreateReduction)
{
    checkCreate(hiptensorCreateReduction(mHandle,
                                         &mOpDesc,
                                         mMN,
                                         mModesMN,
                                         HIPTENSOR_OP_IDENTITY,
                                         mM,
                                         mModesM,
                                         HIPTENSOR_OP_IDENTITY,
                                         mM,
                                         mModesM,
                                         HIPTENSOR_OP_ADD,
                                         GetParam()));
}

INSTANTIATE_TEST_SUITE_P(ScalarType,
                         ComputeDescriptorTest,
                         ::testing::Values(HIPTENSOR_COMPUTE_DESC_32F),
                         [](auto const& info) { return computeDescName(info.param); });

INSTANTIATE_TEST_SUITE_P(NoScalarType,
                         ComputeDescriptorTest,
                         ::testing::Values(HIPTENSOR_COMPUTE_DESC_NONE,
                                           HIPTENSOR_COMPUTE_DESC_8U,
                                           HIPTENSOR_COMPUTE_DESC_8I,
                                           HIPTENSOR_COMPUTE_DESC_32U,
                                           HIPTENSOR_COMPUTE_DESC_32I,
                                           unknownComputeDesc),
                         [](auto const& info) { return computeDescName(info.param); });
