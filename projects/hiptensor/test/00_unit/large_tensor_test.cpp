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

// hiptensorCreatePlan refuses, with HIPTENSOR_STATUS_NOT_SUPPORTED, tensors the kernels cannot
// address, which they used to read and write at wrong offsets without any error:
//  - a tensor whose addressed elements, from its lengths and strides, span 2^31 bytes or more
//    (for complex types, per real or imaginary plane), for every operation;
//  - for f64 elementwise operations with an input laid out differently from the output, a
//    tensor of 2^29 bytes or more;
//  - the intermediate A * B tensor of a trinary contraction.
// These are the limits of the Composable Kernel backend, so every plan here is created with that
// backend whatever HIPTENSOR_BACKEND selects (the rocKE backend addresses such tensors with
// 64-bit offsets; rocke_backend_test checks its results on them).
// Each limit is checked just below and at the threshold. Only descriptors are created: nothing
// of the refused sizes is allocated. Where planning a size just below the limit would run the
// kernels on gigabytes of scratch data (contractions), the check itself is asserted instead.

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

#include <cstdint>
#include <vector>

#include "api_test_helpers.hpp"
#include "ck_tensor_limits.hpp"
#include "data_types.hpp"

namespace
{
    constexpr int64_t twoGiB = int64_t(1) << 31;

    struct HandleGuard
    {
        hiptensorHandle_t handle = nullptr;
        ~HandleGuard()
        {
            if(handle)
            {
                static_cast<void>(hiptensorDestroy(handle));
            }
        }
    };

    struct TensorDescGuard
    {
        hiptensorTensorDescriptor_t desc = nullptr;
        ~TensorDescGuard()
        {
            if(desc)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(desc));
            }
        }
    };

    struct OpDescGuard
    {
        hiptensorOperationDescriptor_t desc = nullptr;
        ~OpDescGuard()
        {
            if(desc)
            {
                static_cast<void>(hiptensorDestroyOperationDescriptor(desc));
            }
        }
    };

    struct PlanGuard
    {
        hiptensorPlan_t plan = nullptr;
        ~PlanGuard()
        {
            if(plan)
            {
                static_cast<void>(hiptensorDestroyPlan(plan));
            }
        }
    };

    // Packed column-major strides.
    std::vector<int64_t> colMajorStrides(std::vector<int64_t> const& lengths)
    {
        std::vector<int64_t> strides(lengths.size(), 1);
        for(std::size_t i = 1; i < lengths.size(); i++)
        {
            strides[i] = strides[i - 1] * lengths[i - 1];
        }
        return strides;
    }

    hiptensorStatus_t createDesc(hiptensorHandle_t           handle,
                                 TensorDescGuard&            guard,
                                 std::vector<int64_t> const& lengths,
                                 hiptensorDataType_t         type,
                                 std::vector<int64_t> const& strides = {})
    {
        auto const packed = colMajorStrides(lengths);
        return hiptensorCreateTensorDescriptor(handle,
                                               &guard.desc,
                                               static_cast<uint32_t>(lengths.size()),
                                               lengths.data(),
                                               strides.empty() ? packed.data() : strides.data(),
                                               type,
                                               0);
    }

    hiptensorComputeDescriptor_t computeFor(hiptensorDataType_t type)
    {
        return type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
    }

    constexpr char const* noF64Support         = "f64 tensors are not supported on this device";
    constexpr char const* noContractionSupport = "This contraction is not supported on this device";
} // namespace

class LargeTensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle.handle), HIPTENSOR_STATUS_SUCCESS);
    }

    // Plans `op` with the default preference. A refused plan must not be created.
    hiptensorStatus_t plan(OpDescGuard const& op)
    {
        PlanGuard         guard;
        hiptensorStatus_t status
            = hiptensorCreatePlan(mHandle.handle, &guard.plan, op.desc, nullptr, 0);
        if(status == HIPTENSOR_STATUS_NOT_SUPPORTED)
        {
            EXPECT_EQ(guard.plan, nullptr);
        }
        return status;
    }

    // B[n,m] = A[m,n], A with packed strides unless `stridesA` is given.
    hiptensorStatus_t permutation(std::vector<int64_t> const& lengths,
                                  hiptensorDataType_t         type,
                                  std::vector<int64_t> const& stridesA = {})
    {
        int32_t const   modesA[] = {'m', 'n'};
        int32_t const   modesB[] = {'n', 'm'};
        TensorDescGuard descA, descB;
        EXPECT_EQ(createDesc(mHandle.handle, descA, lengths, type, stridesA),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, descB, {lengths[1], lengths[0]}, type),
                  HIPTENSOR_STATUS_SUCCESS);
        OpDescGuard op;
        EXPECT_EQ(hiptensorCreatePermutation(mHandle.handle,
                                             &op.desc,
                                             descA.desc,
                                             modesA,
                                             HIPTENSOR_OP_IDENTITY,
                                             descB.desc,
                                             modesB,
                                             computeFor(type)),
                  HIPTENSOR_STATUS_SUCCESS);
        return plan(op);
    }

    // D = A + C (binary) or D = A + B + C (trinary) over modes {m,n}. A carries `modesA` (a
    // subset of {m,n} in any order, so A may be transposed or broadcast); the other tensors
    // carry {m,n} in that order.
    hiptensorStatus_t elementwise(bool                        trinary,
                                  std::vector<int64_t> const& lengths,
                                  std::vector<int32_t> const& modesA,
                                  hiptensorDataType_t         type)
    {
        int32_t const        modes[] = {'m', 'n'};
        std::vector<int64_t> lengthsA;
        for(auto mode : modesA)
        {
            lengthsA.push_back(mode == 'm' ? lengths[0] : lengths[1]);
        }

        TensorDescGuard descA, descB, descC, descD;
        EXPECT_EQ(createDesc(mHandle.handle, descA, lengthsA, type), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, descB, lengths, type), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, descC, lengths, type), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, descD, lengths, type), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        if(trinary)
        {
            EXPECT_EQ(hiptensorCreateElementwiseTrinary(mHandle.handle,
                                                        &op.desc,
                                                        descA.desc,
                                                        modesA.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descB.desc,
                                                        modes,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descC.desc,
                                                        modes,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descD.desc,
                                                        modes,
                                                        HIPTENSOR_OP_ADD,
                                                        HIPTENSOR_OP_ADD,
                                                        computeFor(type)),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else
        {
            EXPECT_EQ(hiptensorCreateElementwiseBinary(mHandle.handle,
                                                       &op.desc,
                                                       descA.desc,
                                                       modesA.data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       descC.desc,
                                                       modes,
                                                       HIPTENSOR_OP_IDENTITY,
                                                       descD.desc,
                                                       modes,
                                                       HIPTENSOR_OP_ADD,
                                                       computeFor(type)),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        return plan(op);
    }

    // D[m] = sum_n A[m,n] (+ D).
    hiptensorStatus_t reduction(std::vector<int64_t> const& lengths, hiptensorDataType_t type)
    {
        int32_t const   modesA[] = {'m', 'n'};
        int32_t const   modesD[] = {'m'};
        TensorDescGuard descA, descD;
        EXPECT_EQ(createDesc(mHandle.handle, descA, lengths, type), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, descD, {lengths[0]}, type), HIPTENSOR_STATUS_SUCCESS);
        OpDescGuard op;
        EXPECT_EQ(hiptensorCreateReduction(mHandle.handle,
                                           &op.desc,
                                           descA.desc,
                                           modesA,
                                           HIPTENSOR_OP_IDENTITY,
                                           descD.desc,
                                           modesD,
                                           HIPTENSOR_OP_IDENTITY,
                                           descD.desc,
                                           modesD,
                                           HIPTENSOR_OP_ADD,
                                           computeFor(type)),
                  HIPTENSOR_STATUS_SUCCESS);
        return plan(op);
    }

    // Whether f64 tensors can be created on this device.
    bool supportsF64()
    {
        TensorDescGuard desc;
        auto            status = createDesc(mHandle.handle, desc, {1}, HIPTENSOR_R_64F);
        EXPECT_TRUE(status == HIPTENSOR_STATUS_SUCCESS || status == HIPTENSOR_STATUS_ARCH_MISMATCH);
        return status == HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensor_test::ScopedBackend mBackend{"ck"};
    HandleGuard                   mHandle;
};

// ---- The addressed range of one tensor ----

TEST(CkTensorSpanTest, FollowsLengthsAndStrides)
{
    using hiptensor::ckTensorSpanBytes;
    std::size_t const k = 1024;

    // Packed: 2^14 x (2^15 - 1) and 2^14 x 2^15 floats.
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_32F, {16 * k, 32 * k - 1}, {1, 16 * k}, 0}),
              twoGiB - 16 * 4 * int64_t(k));
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_32F, {16 * k, 32 * k}, {1, 16 * k}, 0}), twoGiB);

    // Padded: four floats whose last one is 2^29 + 1 elements after the first.
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_32F, {2, 2}, {1, 512 * k * k}, 0}),
              (int64_t(512 * k * k) + 2) * 4);

    // Rank 6 with every mode reversed in memory: the order of the strides does not matter.
    EXPECT_EQ(ckTensorSpanBytes(
                  {HIPTENSOR_R_64F,
                   {16, 16, 32, 32, 32, 16},
                   {16 * 32 * 32 * 32 * 16, 32 * 32 * 32 * 16, 32 * 32 * 16, 32 * 16, 16, 1},
                   0}),
              int64_t(1) << 30);

    // Broadcast (stride 0) and length-1 modes add nothing.
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_16F, {3, 1, 1000}, {1, 1u << 31, 0}, 0}), 6);

    // A negative stride, passed through the API as int64_t, extends the range downwards.
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_32F, {3, 2}, {static_cast<std::size_t>(-2), 1}, 0}),
              (4 + 1 + 1) * 4);

    // No elements, and rank 0.
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_32F, {0, 1u << 31}, {1, 1}, 0}), 0);
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_R_32F, {}, {}, 0}), 4);

    // Complex tensors count one real or imaginary plane.
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_C_32F, {256 * k * k}, {1}, 0}), twoGiB / 2);
    EXPECT_EQ(ckTensorSpanBytes({HIPTENSOR_C_64F, {256 * k * k}, {1}, 0}), twoGiB);

    // Lengths and strides whose products overflow 64 bits still compare as too large.
    auto const huge = std::size_t(1) << 62;
    EXPECT_GE(ckTensorSpanBytes({HIPTENSOR_R_32F, {huge, huge}, {huge, huge}, 0}), twoGiB);
    EXPECT_GE(ckTensorSpanBytes({HIPTENSOR_R_16F, {3, 1u << 31}, {1, 1}, 0}), twoGiB);
    EXPECT_GE(
        ckTensorSpanBytes({HIPTENSOR_R_16F, {4 * k, 4 * k, 64 * k}, {1, 4 * k, 16 * k * k}, 0}),
        twoGiB);
}

// ---- Permutation ----

TEST_F(LargeTensorTest, PermutationJustBelowTwoGiBIsPlanned)
{
    EXPECT_EQ(permutation({16384, 32767}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(permutation({32768, 32767}, HIPTENSOR_R_16F), HIPTENSOR_STATUS_SUCCESS);
}

TEST_F(LargeTensorTest, PermutationOfTwoGiBIsNotSupported)
{
    EXPECT_EQ(permutation({16384, 32768}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(permutation({32768, 32768}, HIPTENSOR_R_16F), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// A holds four floats, but its padded column stride makes them span 2^31 bytes, or 4 bytes less.
TEST_F(LargeTensorTest, PermutationWithPaddedStridesIsNotSupported)
{
    EXPECT_EQ(permutation({2, 2}, HIPTENSOR_R_32F, {1, (int64_t(1) << 29) - 2}),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(permutation({2, 2}, HIPTENSOR_R_32F, {1, (int64_t(1) << 29) - 3}),
              HIPTENSOR_STATUS_SUCCESS);
}

// ---- Elementwise binary and trinary ----

TEST_F(LargeTensorTest, ElementwiseJustBelowTwoGiBIsPlanned)
{
    for(bool trinary : {false, true})
    {
        EXPECT_EQ(elementwise(trinary, {16384, 32767}, {'m', 'n'}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_SUCCESS)
            << "trinary " << trinary;
    }
}

TEST_F(LargeTensorTest, ElementwiseOfTwoGiBIsNotSupported)
{
    for(bool trinary : {false, true})
    {
        EXPECT_EQ(elementwise(trinary, {16384, 32768}, {'m', 'n'}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_NOT_SUPPORTED)
            << "trinary " << trinary;
    }
}

// f64 with A transposed relative to D: correct at 512 MiB, all zeros at 1 GiB.
TEST_F(LargeTensorTest, F64ElementwiseWithTransposedInputJustBelowLimitIsPlanned)
{
    if(!supportsF64())
    {
        GTEST_SKIP() << noF64Support;
    }
    for(bool trinary : {false, true})
    {
        EXPECT_EQ(elementwise(trinary, {8192, 8191}, {'n', 'm'}, HIPTENSOR_R_64F),
                  HIPTENSOR_STATUS_SUCCESS)
            << "trinary " << trinary;
    }
}

TEST_F(LargeTensorTest, F64ElementwiseWithTransposedInputOf512MiBIsNotSupported)
{
    if(!supportsF64())
    {
        GTEST_SKIP() << noF64Support;
    }
    for(bool trinary : {false, true})
    {
        EXPECT_EQ(elementwise(trinary, {8192, 8192}, {'n', 'm'}, HIPTENSOR_R_64F),
                  HIPTENSOR_STATUS_NOT_SUPPORTED)
            << "trinary " << trinary;
        // A broadcast input is not laid out like the output either.
        EXPECT_EQ(elementwise(trinary, {8192, 8192}, {'n'}, HIPTENSOR_R_64F),
                  HIPTENSOR_STATUS_NOT_SUPPORTED)
            << "trinary " << trinary;
    }
}

// The 512 MiB limit applies neither to f64 with every tensor laid out like the output, which is
// planned up to the 2 GiB limit, nor to other types.
TEST_F(LargeTensorTest, ElementwiseOf512MiBOrMoreIsPlannedOtherwise)
{
    for(bool trinary : {false, true})
    {
        EXPECT_EQ(elementwise(trinary, {8192, 16384}, {'n', 'm'}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_SUCCESS)
            << "trinary " << trinary;
        if(supportsF64())
        {
            EXPECT_EQ(elementwise(trinary, {16384, 16383}, {'m', 'n'}, HIPTENSOR_R_64F),
                      HIPTENSOR_STATUS_SUCCESS)
                << "trinary " << trinary;
            EXPECT_EQ(elementwise(trinary, {16384, 16384}, {'m', 'n'}, HIPTENSOR_R_64F),
                      HIPTENSOR_STATUS_NOT_SUPPORTED)
                << "trinary " << trinary;
        }
    }
}

// ---- Reduction ----

TEST_F(LargeTensorTest, ReductionJustBelowTwoGiBIsPlanned)
{
    EXPECT_EQ(reduction({16384, 32767}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_SUCCESS);
    if(supportsF64())
    {
        EXPECT_EQ(reduction({16384, 16383}, HIPTENSOR_R_64F), HIPTENSOR_STATUS_SUCCESS);
    }
}

TEST_F(LargeTensorTest, ReductionOfTwoGiBIsNotSupported)
{
    EXPECT_EQ(reduction({16384, 32768}, HIPTENSOR_R_32F), HIPTENSOR_STATUS_NOT_SUPPORTED);
    if(supportsF64())
    {
        EXPECT_EQ(reduction({16384, 16384}, HIPTENSOR_R_64F), HIPTENSOR_STATUS_NOT_SUPPORTED);
    }
}

// ---- Contraction ----

// D[m,n] = A[m,k] * B[n,k], with D's column stride `strideD` (0 for packed). Returns false,
// without failing, when the device does not support the contraction.
class LargeContractionTest : public LargeTensorTest
{
protected:
    bool create(int64_t m, int64_t n, int64_t k, hiptensorDataType_t type, int64_t strideD = 0)
    {
        int32_t const modesA[] = {'m', 'k'};
        int32_t const modesB[] = {'n', 'k'};
        int32_t const modesD[] = {'m', 'n'};

        EXPECT_EQ(createDesc(mHandle.handle, mDescA, {m, k}, type), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, mDescB, {n, k}, type), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(
            createDesc(mHandle.handle,
                       mDescD,
                       {m, n},
                       type,
                       strideD == 0 ? std::vector<int64_t>{} : std::vector<int64_t>{1, strideD}),
            HIPTENSOR_STATUS_SUCCESS);

        auto compute = type == HIPTENSOR_C_32F ? HIPTENSOR_COMPUTE_DESC_C32F : computeFor(type);
        auto status  = hiptensorCreateContraction(mHandle.handle,
                                                 &mOp.desc,
                                                 mDescA.desc,
                                                 modesA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 mDescB.desc,
                                                 modesB,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 mDescD.desc,
                                                 modesD,
                                                 compute);
        if(status != HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        }
        return status == HIPTENSOR_STATUS_SUCCESS;
    }

    TensorDescGuard mDescA, mDescB, mDescD;
    OpDescGuard     mOp;
};

// Planning A of almost 2 GiB would time kernels on that much scratch data, so the size check is
// asserted directly; a small contraction shows the plan path is unchanged.
TEST_F(LargeContractionTest, JustBelowTwoGiBPassesTheSizeCheck)
{
    if(!create(65536, 2, 8191, HIPTENSOR_R_32F))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_FALSE(hiptensor::exceedsCkTensorLimits(*mOp.desc));
}

TEST_F(LargeContractionTest, SmallContractionIsPlanned)
{
    if(!create(8, 6, 4, HIPTENSOR_R_32F))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_EQ(plan(mOp), HIPTENSOR_STATUS_SUCCESS);
}

TEST_F(LargeContractionTest, TwoGiBIsNotSupported)
{
    if(!create(65536, 2, 8192, HIPTENSOR_R_32F))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_TRUE(hiptensor::exceedsCkTensorLimits(*mOp.desc));
    EXPECT_EQ(plan(mOp), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// D holds four floats, but its padded column stride reaches past 2^31 bytes.
TEST_F(LargeContractionTest, PaddedStridesReachingTwoGiBAreNotSupported)
{
    if(!create(2, 2, 2, HIPTENSOR_R_32F, int64_t(1) << 29))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_EQ(plan(mOp), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// Complex tensors are split into real and imaginary planes before they reach the kernels, so
// a complex A of 2 GiB has planes of 1 GiB.
TEST_F(LargeContractionTest, ComplexLimitAppliesPerPlane)
{
    if(!create(65536, 2, 4096, HIPTENSOR_C_32F))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_FALSE(hiptensor::exceedsCkTensorLimits(*mOp.desc));
}

TEST_F(LargeContractionTest, ComplexPlanesOfTwoGiBAreNotSupported)
{
    if(!create(65536, 2, 8192, HIPTENSOR_C_32F))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_EQ(plan(mOp), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// ---- Trinary contraction: E[m,j] = A[m,k] * B[k,n] * C[n,j], through T[m,n] = A * B ----

class LargeTrinaryContractionTest : public LargeTensorTest
{
protected:
    bool create(int64_t m, int64_t k, int64_t n, int64_t j)
    {
        int32_t const modesA[] = {'m', 'k'};
        int32_t const modesB[] = {'k', 'n'};
        int32_t const modesC[] = {'n', 'j'};
        int32_t const modesE[] = {'m', 'j'};

        EXPECT_EQ(createDesc(mHandle.handle, mDescA, {m, k}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, mDescB, {k, n}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, mDescC, {n, j}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(mHandle.handle, mDescE, {m, j}, HIPTENSOR_R_32F),
                  HIPTENSOR_STATUS_SUCCESS);

        auto status = hiptensorCreateContractionTrinary(mHandle.handle,
                                                        &mOp.desc,
                                                        mDescA.desc,
                                                        modesA,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        mDescB.desc,
                                                        modesB,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        mDescC.desc,
                                                        modesC,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        nullptr,
                                                        nullptr,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        mDescE.desc,
                                                        modesE,
                                                        HIPTENSOR_COMPUTE_DESC_32F);
        if(status != HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        }
        return status == HIPTENSOR_STATUS_SUCCESS;
    }

    // Plans with the default preference. A plan that is not created successfully must not be
    // returned. The limit on T is checked after the plan object exists, so it is released here
    // whatever the status.
    hiptensorStatus_t planTrinary()
    {
        PlanGuard         guard;
        hiptensorStatus_t status
            = hiptensorCreatePlan(mHandle.handle, &guard.plan, mOp.desc, nullptr, 0);
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            EXPECT_EQ(guard.plan, nullptr);
        }
        return status;
    }

    TensorDescGuard mDescA, mDescB, mDescC, mDescE;
    OpDescGuard     mOp;
};

TEST_F(LargeTrinaryContractionTest, SmallContractionIsPlanned)
{
    if(!create(8, 4, 6, 3))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_EQ(planTrinary(), HIPTENSOR_STATUS_SUCCESS);
}

TEST_F(LargeTrinaryContractionTest, OutputOfTwoGiBIsNotSupported)
{
    if(!create(65536, 2, 2, 8192))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_TRUE(hiptensor::exceedsCkTensorLimits(*mOp.desc));
    EXPECT_EQ(plan(mOp), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// Every tensor of the operation is small, but T[m,n] has 2^29 floats.
TEST_F(LargeTrinaryContractionTest, IntermediateOfTwoGiBIsNotSupported)
{
    if(!create(32768, 2, 16384, 1))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    EXPECT_FALSE(hiptensor::exceedsCkTensorLimits(*mOp.desc));
    EXPECT_EQ(planTrinary(), HIPTENSOR_STATUS_NOT_SUPPORTED);
}
