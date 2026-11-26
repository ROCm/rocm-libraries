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

// Every operator argument of the create-descriptor functions has a kind: the per-operand
// operators (opA, opB, opC, opD) must be unary, and the operators that combine operands (opAC,
// opAB, opABC, opReduce) must be HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX or
// HIPTENSOR_OP_MIN. An operator of the wrong kind, or a value outside hiptensorOperator_t, is
// rejected with HIPTENSOR_STATUS_INVALID_VALUE when the descriptor is created. No kernel is
// launched.

#include <tuple>

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

namespace
{
    constexpr int32_t mMode = 'm';
    constexpr int32_t nMode = 'n';
    constexpr int32_t kMode = 'k';
    constexpr int32_t pMode = 'p';

    constexpr uint32_t alignment = 256;

    // A value that lies between enumerators of hiptensorOperator_t.
    constexpr auto outOfRangeOp = static_cast<hiptensorOperator_t>(4);

    // Owns an operation descriptor so that it is released even when an assertion returns early.
    struct OpDesc
    {
        hiptensorOperationDescriptor_t desc{};
        ~OpDesc()
        {
            if(desc)
            {
                hiptensorDestroyOperationDescriptor(desc);
            }
        }
    };
} // namespace

class OperatorValidationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);

        int64_t mn[2] = {2, 3};
        int64_t m[1]  = {2};
        int64_t mk[2] = {2, 4};
        int64_t kn[2] = {4, 3};
        int64_t np[2] = {3, 5};
        int64_t mp[2] = {2, 5};

        for(auto const& [desc, rank, lengths] : {std::tuple{&mMN, 2, mn},
                                                 std::tuple{&mM, 1, m},
                                                 std::tuple{&mMK, 2, mk},
                                                 std::tuple{&mKN, 2, kn},
                                                 std::tuple{&mNP, 2, np},
                                                 std::tuple{&mMP, 2, mp}})
        {
            ASSERT_EQ(hiptensorCreateTensorDescriptor(
                          mHandle, desc, rank, lengths, nullptr, HIPTENSOR_R_32F, alignment),
                      HIPTENSOR_STATUS_SUCCESS);
        }
    }

    void TearDown() override
    {
        for(auto desc : {mMN, mM, mMK, mKN, mNP, mMP})
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

    hiptensorStatus_t createPermutation(hiptensorOperator_t opA)
    {
        OpDesc op;
        return hiptensorCreatePermutation(
            mHandle, &op.desc, mMN, mModesMN, opA, mMN, mModesMN, HIPTENSOR_COMPUTE_DESC_32F);
    }

    hiptensorStatus_t
        createBinary(hiptensorOperator_t opA, hiptensorOperator_t opC, hiptensorOperator_t opAC)
    {
        OpDesc op;
        return hiptensorCreateElementwiseBinary(mHandle,
                                                &op.desc,
                                                mMN,
                                                mModesMN,
                                                opA,
                                                mMN,
                                                mModesMN,
                                                opC,
                                                mMN,
                                                mModesMN,
                                                opAC,
                                                HIPTENSOR_COMPUTE_DESC_32F);
    }

    hiptensorStatus_t createTrinary(hiptensorOperator_t opA,
                                    hiptensorOperator_t opB,
                                    hiptensorOperator_t opC,
                                    hiptensorOperator_t opAB,
                                    hiptensorOperator_t opABC)
    {
        OpDesc op;
        return hiptensorCreateElementwiseTrinary(mHandle,
                                                 &op.desc,
                                                 mMN,
                                                 mModesMN,
                                                 opA,
                                                 mMN,
                                                 mModesMN,
                                                 opB,
                                                 mMN,
                                                 mModesMN,
                                                 opC,
                                                 mMN,
                                                 mModesMN,
                                                 opAB,
                                                 opABC,
                                                 HIPTENSOR_COMPUTE_DESC_32F);
    }

    // Reduces the {m,n} input A over n into the {m} output D.
    hiptensorStatus_t createReduction(hiptensorOperator_t opA,
                                      hiptensorOperator_t opC,
                                      hiptensorOperator_t opReduce)
    {
        OpDesc op;
        return hiptensorCreateReduction(mHandle,
                                        &op.desc,
                                        mMN,
                                        mModesMN,
                                        opA,
                                        mM,
                                        mModesM,
                                        opC,
                                        mM,
                                        mModesM,
                                        opReduce,
                                        HIPTENSOR_COMPUTE_DESC_32F);
    }

    // D{m,n} = A{m,k} * B{k,n} + C{m,n}, or D{m,n} = A{m,k} * B{k,n} without C.
    hiptensorStatus_t createContraction(hiptensorOperator_t opA,
                                        hiptensorOperator_t opB,
                                        hiptensorOperator_t opC,
                                        bool                withC = true)
    {
        OpDesc op;
        return hiptensorCreateContraction(mHandle,
                                          &op.desc,
                                          mMK,
                                          mModesMK,
                                          opA,
                                          mKN,
                                          mModesKN,
                                          opB,
                                          withC ? mMN : nullptr,
                                          withC ? mModesMN : nullptr,
                                          opC,
                                          mMN,
                                          mModesMN,
                                          HIPTENSOR_COMPUTE_DESC_32F);
    }

    // E{m,p} = A{m,k} * B{k,n} * C{n,p} + D{m,p}, or E{m,p} = A{m,k} * B{k,n} * C{n,p} without D.
    hiptensorStatus_t createContractionTrinary(hiptensorOperator_t opA,
                                               hiptensorOperator_t opB,
                                               hiptensorOperator_t opC,
                                               hiptensorOperator_t opD,
                                               bool                withD = true)
    {
        OpDesc op;
        return hiptensorCreateContractionTrinary(mHandle,
                                                 &op.desc,
                                                 mMK,
                                                 mModesMK,
                                                 opA,
                                                 mKN,
                                                 mModesKN,
                                                 opB,
                                                 mNP,
                                                 mModesNP,
                                                 opC,
                                                 withD ? mMP : nullptr,
                                                 withD ? mModesMP : nullptr,
                                                 opD,
                                                 mMP,
                                                 mModesMP,
                                                 HIPTENSOR_COMPUTE_DESC_32F);
    }

    hiptensorHandle_t           mHandle{};
    hiptensorTensorDescriptor_t mMN{};
    hiptensorTensorDescriptor_t mM{};
    hiptensorTensorDescriptor_t mMK{};
    hiptensorTensorDescriptor_t mKN{};
    hiptensorTensorDescriptor_t mNP{};
    hiptensorTensorDescriptor_t mMP{};

    int32_t mModesMN[2] = {mMode, nMode};
    int32_t mModesM[1]  = {mMode};
    int32_t mModesMK[2] = {mMode, kMode};
    int32_t mModesKN[2] = {kMode, nMode};
    int32_t mModesNP[2] = {nMode, pMode};
    int32_t mModesMP[2] = {mMode, pMode};
};

TEST_F(OperatorValidationTest, Permutation)
{
    EXPECT_EQ(createPermutation(HIPTENSOR_OP_NEG), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createPermutation(HIPTENSOR_OP_ADD), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createPermutation(outOfRangeOp), HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST_F(OperatorValidationTest, ElementwiseBinary)
{
    auto const id = HIPTENSOR_OP_IDENTITY;
    EXPECT_EQ(createBinary(id, HIPTENSOR_OP_RELU, HIPTENSOR_OP_MAX), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createBinary(HIPTENSOR_OP_MUL, id, HIPTENSOR_OP_ADD), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createBinary(id, outOfRangeOp, HIPTENSOR_OP_ADD), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createBinary(id, id, HIPTENSOR_OP_IDENTITY), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createBinary(id, id, HIPTENSOR_OP_UNKNOWN), HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST_F(OperatorValidationTest, ElementwiseTrinary)
{
    auto const id  = HIPTENSOR_OP_IDENTITY;
    auto const add = HIPTENSOR_OP_ADD;
    EXPECT_EQ(createTrinary(id, id, id, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MIN),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createTrinary(id, HIPTENSOR_OP_MAX, id, add, add), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createTrinary(id, id, id, HIPTENSOR_OP_RELU, add), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createTrinary(id, id, id, add, outOfRangeOp), HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST_F(OperatorValidationTest, Reduction)
{
    auto const id = HIPTENSOR_OP_IDENTITY;
    EXPECT_EQ(createReduction(HIPTENSOR_OP_NEG, id, HIPTENSOR_OP_MAX), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createReduction(HIPTENSOR_OP_MIN, id, HIPTENSOR_OP_ADD),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createReduction(id, id, HIPTENSOR_OP_IDENTITY), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createReduction(id, id, outOfRangeOp), HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST_F(OperatorValidationTest, Contraction)
{
    auto const id = HIPTENSOR_OP_IDENTITY;
    auto const ok = createContraction(id, HIPTENSOR_OP_RELU, id);
    if(ok == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "HIPTENSOR_STATUS_ARCH_MISMATCH: no contraction support on this device";
    }
    EXPECT_EQ(ok, HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createContraction(id, HIPTENSOR_OP_ADD, id), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createContraction(id, id, outOfRangeOp), HIPTENSOR_STATUS_INVALID_VALUE);
    // Without C, opC is unused and any value is accepted.
    EXPECT_EQ(createContraction(id, id, outOfRangeOp, false), HIPTENSOR_STATUS_SUCCESS);
}

TEST_F(OperatorValidationTest, ContractionTrinary)
{
    auto const id = HIPTENSOR_OP_IDENTITY;
    auto const ok = createContractionTrinary(id, id, HIPTENSOR_OP_RELU, id);
    if(ok == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "HIPTENSOR_STATUS_ARCH_MISMATCH: no contraction support on this device";
    }
    EXPECT_EQ(ok, HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createContractionTrinary(HIPTENSOR_OP_MUL, id, id, id),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(createContractionTrinary(id, id, id, outOfRangeOp), HIPTENSOR_STATUS_INVALID_VALUE);
    // Without D, opD is unused and any value is accepted.
    EXPECT_EQ(createContractionTrinary(id, id, id, outOfRangeOp, false), HIPTENSOR_STATUS_SUCCESS);
}
