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

// hiptensorCreatePlan returns HIPTENSOR_STATUS_NOT_SUPPORTED for tensors the kernels cannot
// address: tensors whose elements span 2^31 bytes or more (from their lengths and strides), the
// intermediate A * B tensor of a trinary contraction past that size, and f64 elementwise
// operations, including reductions that keep the rank, with an input laid out differently from
// the output past 2^29 bytes. Most cases also check that tensors just below the limit are planned.
//
// The tests create descriptors and plans only. No tensor memory is allocated: plan creation of
// permutations, elementwise operations and reductions does not allocate, and contractions are
// planned with HIPTENSOR_ALGO_ACTOR_CRITIC, which selects a kernel without running candidates.

#include <hiptensor/hiptensor.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace
{
    constexpr int64_t kTwoGiB = int64_t(1) << 31;

    // Packed column-major strides.
    std::vector<int64_t> packedStrides(std::vector<int64_t> const& lengths)
    {
        std::vector<int64_t> strides(lengths.size(), 1);
        for(std::size_t i = 1; i < lengths.size(); i++)
        {
            strides[i] = strides[i - 1] * lengths[i - 1];
        }
        return strides;
    }
} // namespace

class TensorSizeLimitTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto status = hiptensorCreate(&mHandle);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "Device not supported by hipTensor";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        for(auto plan : mPlans)
        {
            hiptensorDestroyPlan(plan);
        }
        for(auto pref : mPrefs)
        {
            hiptensorDestroyPlanPreference(pref);
        }
        for(auto op : mOps)
        {
            hiptensorDestroyOperationDescriptor(op);
        }
        for(auto tensor : mTensors)
        {
            hiptensorDestroyTensorDescriptor(tensor);
        }
        if(mHandle)
        {
            hiptensorDestroy(mHandle);
        }
    }

    // Creates a tensor descriptor with packed strides unless `strides` is given.
    hiptensorStatus_t tensor(hiptensorTensorDescriptor_t* out,
                             std::vector<int64_t> const&  lengths,
                             hiptensorDataType_t          type,
                             std::vector<int64_t> const&  strides = {})
    {
        *out              = nullptr;
        auto const packed = packedStrides(lengths);
        auto       status
            = hiptensorCreateTensorDescriptor(mHandle,
                                              out,
                                              static_cast<uint32_t>(lengths.size()),
                                              lengths.data(),
                                              strides.empty() ? packed.data() : strides.data(),
                                              type,
                                              0);
        if(*out != nullptr)
        {
            mTensors.push_back(*out);
        }
        return status;
    }

    hiptensorOperationDescriptor_t keep(hiptensorOperationDescriptor_t op)
    {
        if(op != nullptr)
        {
            mOps.push_back(op);
        }
        return op;
    }

    // Creates a plan for `op`. `planCreated` reports whether a plan object was returned.
    hiptensorStatus_t plan(hiptensorOperationDescriptor_t op,
                           hiptensorAlgo_t                algo        = HIPTENSOR_ALGO_DEFAULT,
                           bool*                          planCreated = nullptr)
    {
        hiptensorPlanPreference_t pref = nullptr;
        auto status = hiptensorCreatePlanPreference(mHandle, &pref, algo, HIPTENSOR_JIT_MODE_NONE);
        if(pref != nullptr)
        {
            mPrefs.push_back(pref);
        }
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "hiptensorCreatePlanPreference failed";
            return status;
        }
        hiptensorCacheMode_t const cacheMode = HIPTENSOR_CACHE_MODE_NONE;
        EXPECT_EQ(
            hiptensorPlanPreferenceSetAttribute(
                mHandle, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode)),
            HIPTENSOR_STATUS_SUCCESS);

        hiptensorPlan_t newPlan = nullptr;
        status                  = hiptensorCreatePlan(mHandle, &newPlan, op, pref, kWorkspace);
        if(newPlan != nullptr)
        {
            mPlans.push_back(newPlan);
        }
        if(planCreated != nullptr)
        {
            *planCreated = newPlan != nullptr;
        }
        return status;
    }

    // B = A, with B's modes reversed when `transpose` is set. Returns the status of the plan;
    // HIPTENSOR_STATUS_ARCH_MISMATCH when the device does not support `type`.
    hiptensorStatus_t permutation(std::vector<int64_t> const& lengths,
                                  hiptensorDataType_t         type,
                                  bool                        transpose,
                                  std::vector<int64_t> const& stridesA    = {},
                                  bool*                       planCreated = nullptr)
    {
        int32_t const modesA[] = {'m', 'n'};
        int32_t const modesB[] = {'n', 'm'};
        auto const    lengthsB = transpose ? std::vector<int64_t>{lengths[1], lengths[0]} : lengths;
        hiptensorTensorDescriptor_t descA = nullptr;
        hiptensorTensorDescriptor_t descB = nullptr;
        if(auto status = tensor(&descA, lengths, type, stridesA);
           status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }
        if(auto status = tensor(&descB, lengthsB, type); status != HIPTENSOR_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "hiptensorCreateTensorDescriptor failed for B";
            return status;
        }

        hiptensorOperationDescriptor_t op     = nullptr;
        auto                           status = hiptensorCreatePermutation(mHandle,
                                                 &op,
                                                 descA,
                                                 modesA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descB,
                                                 transpose ? modesB : modesA,
                                                 computeFor(type));
        keep(op);
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "hiptensorCreatePermutation failed";
            return status;
        }
        return plan(op, HIPTENSOR_ALGO_DEFAULT, planCreated);
    }

    // D = reduce(A) with C = D. Returns the status of the plan;
    // HIPTENSOR_STATUS_ARCH_MISMATCH when the device does not support `type`.
    hiptensorStatus_t reduction(std::vector<int64_t> const& lengthsA,
                                std::vector<int32_t> const& modesA,
                                std::vector<int64_t> const& lengthsD,
                                std::vector<int32_t> const& modesD,
                                hiptensorDataType_t         type)
    {
        hiptensorTensorDescriptor_t descA = nullptr;
        hiptensorTensorDescriptor_t descD = nullptr;
        if(auto status = tensor(&descA, lengthsA, type); status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }
        if(auto status = tensor(&descD, lengthsD, type); status != HIPTENSOR_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "hiptensorCreateTensorDescriptor failed for D";
            return status;
        }

        hiptensorOperationDescriptor_t op     = nullptr;
        auto                           status = hiptensorCreateReduction(mHandle,
                                               &op,
                                               descA,
                                               modesA.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               descD,
                                               modesD.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               descD,
                                               modesD.data(),
                                               HIPTENSOR_OP_ADD,
                                               computeFor(type));
        keep(op);
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "hiptensorCreateReduction failed";
            return status;
        }
        return plan(op);
    }

    static hiptensorComputeDescriptor_t computeFor(hiptensorDataType_t type)
    {
        return type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
    }

    // Plans only record the workspace limit; nothing of this size is allocated.
    static constexpr uint64_t kWorkspace = uint64_t(1) << 20;

    hiptensorHandle_t                           mHandle = nullptr;
    std::vector<hiptensorTensorDescriptor_t>    mTensors;
    std::vector<hiptensorOperationDescriptor_t> mOps;
    std::vector<hiptensorPlanPreference_t>      mPrefs;
    std::vector<hiptensorPlan_t>                mPlans;
};

// A of 32768 x 16384 floats spans exactly 2^31 bytes; one column less is planned.
TEST_F(TensorSizeLimitTest, PermutationAtLimitIsNotSupported)
{
    bool planCreated = true;
    EXPECT_EQ(permutation({32768, 16384}, HIPTENSOR_R_32F, true, {}, &planCreated),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_FALSE(planCreated);

    EXPECT_EQ(permutation({32768, 16383}, HIPTENSOR_R_32F, true), HIPTENSOR_STATUS_SUCCESS);
}

// A holds four floats, but its padded column stride makes it span more than 2^31 bytes.
TEST_F(TensorSizeLimitTest, PaddedStridesCountTowardTheLimit)
{
    EXPECT_EQ(permutation({2, 2}, HIPTENSOR_R_32F, true, {1, kTwoGiB / 4}),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// 32768 x 32768 half-precision elements per tensor span exactly 2^31 bytes; one column less is
// planned.
TEST_F(TensorSizeLimitTest, ElementwiseBinaryAtLimitIsNotSupported)
{
    int32_t const modes[] = {'m', 'n'};
    for(auto const& [lengths, expected] :
        {std::pair{std::vector<int64_t>{32768, 32768}, HIPTENSOR_STATUS_NOT_SUPPORTED},
         std::pair{std::vector<int64_t>{32768, 32767}, HIPTENSOR_STATUS_SUCCESS}})
    {
        SCOPED_TRACE(lengths[1]);
        hiptensorTensorDescriptor_t descA = nullptr, descC = nullptr, descD = nullptr;
        ASSERT_EQ(tensor(&descA, lengths, HIPTENSOR_R_16F), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(tensor(&descC, lengths, HIPTENSOR_R_16F), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(tensor(&descD, lengths, HIPTENSOR_R_16F), HIPTENSOR_STATUS_SUCCESS);

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreateElementwiseBinary(mHandle,
                                                   &op,
                                                   descA,
                                                   modes,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   descC,
                                                   modes,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   descD,
                                                   modes,
                                                   HIPTENSOR_OP_ADD,
                                                   HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(plan(keep(op)), expected);
    }
}

// A of 16384 x 32768 floats, reduced over n, spans exactly 2^31 bytes; one column less is
// planned.
TEST_F(TensorSizeLimitTest, ReductionAtLimitIsNotSupported)
{
    EXPECT_EQ(reduction({16384, 32768}, {'m', 'n'}, {16384}, {'m'}, HIPTENSOR_R_32F),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(reduction({16384, 32767}, {'m', 'n'}, {16384}, {'m'}, HIPTENSOR_R_32F),
              HIPTENSOR_STATUS_SUCCESS);
}

// f64 permutations with a transposed input are limited to 2^29 bytes per tensor; with the
// output's layout the same size is planned.
TEST_F(TensorSizeLimitTest, F64TransposedPermutationLimit)
{
    auto status = permutation({8192, 8192}, HIPTENSOR_R_64F, true);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "f64 tensors are not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(permutation({8192, 8191}, HIPTENSOR_R_64F, true), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(permutation({8192, 8192}, HIPTENSOR_R_64F, false), HIPTENSOR_STATUS_SUCCESS);
}

// A reduction whose output keeps the rank of A runs as an elementwise operation, so an f64 one
// with a transposed output is limited to 2^29 bytes per tensor as well; with the input's layout
// the same size is planned.
TEST_F(TensorSizeLimitTest, F64SameRankReductionWithTransposedOutputLimit)
{
    auto status = reduction({8192, 8192}, {'m', 'n'}, {8192, 8192}, {'n', 'm'}, HIPTENSOR_R_64F);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "f64 tensors are not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(reduction({8192, 8191}, {'m', 'n'}, {8191, 8192}, {'n', 'm'}, HIPTENSOR_R_64F),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(reduction({8192, 8192}, {'m', 'n'}, {8192, 8192}, {'m', 'n'}, HIPTENSOR_R_64F),
              HIPTENSOR_STATUS_SUCCESS);
}

// D[m,n] = A[m,k] * B[n,k] with A of 65536 x 8192 floats, spanning exactly 2^31 bytes; with
// one k less the contraction is planned.
TEST_F(TensorSizeLimitTest, ContractionAtLimitIsNotSupported)
{
    int32_t const modesA[] = {'m', 'k'};
    int32_t const modesB[] = {'n', 'k'};
    int32_t const modesD[] = {'m', 'n'};
    for(int64_t const k : {8192, 8191})
    {
        SCOPED_TRACE(k);
        hiptensorTensorDescriptor_t descA = nullptr, descB = nullptr, descD = nullptr;
        ASSERT_EQ(tensor(&descA, {65536, k}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(tensor(&descB, {2, k}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(tensor(&descD, {65536, 2}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);

        hiptensorOperationDescriptor_t op     = nullptr;
        auto                           status = hiptensorCreateContraction(mHandle,
                                                 &op,
                                                 descA,
                                                 modesA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descB,
                                                 modesB,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descD,
                                                 modesD,
                                                 HIPTENSOR_COMPUTE_DESC_32F);
        keep(op);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "Contractions are not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

        bool       planCreated = false;
        bool const atLimit     = k == 8192;
        EXPECT_EQ(plan(op, HIPTENSOR_ALGO_ACTOR_CRITIC, &planCreated),
                  atLimit ? HIPTENSOR_STATUS_NOT_SUPPORTED : HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(planCreated, !atLimit);
    }
}

// E[m,j] = A[m,k] * B[k,n] * C[n,j]: every tensor is small, but the intermediate T[m,n] = A * B
// has 2^29 floats.
TEST_F(TensorSizeLimitTest, TrinaryContractionIntermediateAtLimitIsNotSupported)
{
    int32_t const               modesA[] = {'m', 'k'};
    int32_t const               modesB[] = {'k', 'n'};
    int32_t const               modesC[] = {'n', 'j'};
    int32_t const               modesE[] = {'m', 'j'};
    hiptensorTensorDescriptor_t descA = nullptr, descB = nullptr, descC = nullptr, descE = nullptr;
    ASSERT_EQ(tensor(&descA, {32768, 2}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(tensor(&descB, {2, 16384}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(tensor(&descC, {16384, 2}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(tensor(&descE, {32768, 2}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);

    hiptensorOperationDescriptor_t op     = nullptr;
    auto                           status = hiptensorCreateContractionTrinary(mHandle,
                                                    &op,
                                                    descA,
                                                    modesA,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    descB,
                                                    modesB,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    descC,
                                                    modesC,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    nullptr,
                                                    nullptr,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    descE,
                                                    modesE,
                                                    HIPTENSOR_COMPUTE_DESC_32F);
    keep(op);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "Contractions are not supported on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    EXPECT_EQ(plan(op, HIPTENSOR_ALGO_ACTOR_CRITIC), HIPTENSOR_STATUS_NOT_SUPPORTED);
}
