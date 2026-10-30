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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// API misuse and documented-legal calls that must neither crash nor corrupt state:
//  - operation descriptor attributes: buffer sizes are checked, PADDING_LEFT/RIGHT take one
//    uint32_t per output mode and only zero padding is accepted, PADDING_VALUE holds one
//    output element;
//  - compute descriptors without a scalar type are rejected when the operation is created;
//  - hiptensorCreatePlan accepts a null preference, and a preference shared by several plans
//    is not modified by any of them;
//  - a contraction kernel's workspace size can be queried after it reported insufficient
//    workspace and after it ran;
//  - complex scalars need no alignment beyond their bytes (alpha / beta at any byte offset).
// The kernels' expected outputs come from plain loops with 64-bit offsets, not from the CPU
// reference. The data are small integers, so every result is exact and compared bit for bit.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "contraction/contraction_solution.hpp"
#include "data_types.hpp"

namespace
{
    constexpr uint32_t alignment = 4;

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

    struct PlanPrefGuard
    {
        hiptensorPlanPreference_t pref = nullptr;
        ~PlanPrefGuard()
        {
            if(pref)
            {
                static_cast<void>(hiptensorDestroyPlanPreference(pref));
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

    struct DeviceBuffer
    {
        void* ptr = nullptr;
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
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
                                 std::vector<int64_t> const& lengths)
    {
        auto strides = colMajorStrides(lengths);
        return hiptensorCreateTensorDescriptor(handle,
                                               &guard.desc,
                                               static_cast<uint32_t>(lengths.size()),
                                               lengths.data(),
                                               strides.data(),
                                               HIPTENSOR_R_32F,
                                               alignment);
    }

    std::vector<float> smallIntegers(std::size_t count, int seed)
    {
        std::vector<float> values(count);
        for(std::size_t i = 0; i < count; i++)
        {
            values[i] = static_cast<float>(static_cast<int>((i * 5 + seed) % 7) - 3);
        }
        return values;
    }

    void upload(DeviceBuffer& buffer, std::vector<float> const& host)
    {
        ASSERT_EQ(hipMalloc(&buffer.ptr, host.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(buffer.ptr, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
    }

    void expectEqual(DeviceBuffer const& device, std::vector<float> const& expected)
    {
        std::vector<float> result(expected.size());
        ASSERT_EQ(
            hipMemcpy(
                result.data(), device.ptr, result.size() * sizeof(float), hipMemcpyDeviceToHost),
            hipSuccess);
        for(std::size_t i = 0; i < expected.size(); i++)
        {
            EXPECT_EQ(result[i], expected[i]) << "offset " << i;
        }
    }

    // ---- Permutation: rank 2 B[n,m] = alpha * A[m,n], rank 3 B[c,a,b] = alpha * A[a,b,c] ----

    struct PermutationProblem
    {
        std::vector<int64_t> lengthsA;
        std::vector<int32_t> modesA;
        std::vector<int64_t> lengthsB;
        std::vector<int32_t> modesB;
    };

    PermutationProblem rank2Permutation()
    {
        return {{4, 5}, {'m', 'n'}, {5, 4}, {'n', 'm'}};
    }

    PermutationProblem rank3Permutation()
    {
        return {{2, 3, 4}, {'a', 'b', 'c'}, {4, 2, 3}, {'c', 'a', 'b'}};
    }

    // Independent reference: walks B's index space and reads A at the matching index.
    std::vector<float>
        permutationReference(PermutationProblem const& p, std::vector<float> const& a, float alpha)
    {
        auto const stridesA = colMajorStrides(p.lengthsA);
        auto const stridesB = colMajorStrides(p.lengthsB);
        auto const rank     = p.lengthsB.size();

        int64_t count = 1;
        for(auto length : p.lengthsB)
        {
            count *= length;
        }

        std::vector<float> b(static_cast<std::size_t>(count));
        for(int64_t linear = 0; linear < count; linear++)
        {
            int64_t rest    = linear;
            int64_t offsetA = 0;
            int64_t offsetB = 0;
            for(std::size_t i = 0; i < rank; i++)
            {
                int64_t index = rest % p.lengthsB[i];
                rest /= p.lengthsB[i];
                offsetB += index * stridesB[i];
                for(std::size_t j = 0; j < p.modesA.size(); j++)
                {
                    if(p.modesA[j] == p.modesB[i])
                    {
                        offsetA += index * stridesA[j];
                    }
                }
            }
            b[static_cast<std::size_t>(offsetB)] = alpha * a[static_cast<std::size_t>(offsetA)];
        }
        return b;
    }

    struct PermutationRun
    {
        PermutationProblem problem;
        TensorDescGuard    descA, descB;
        OpDescGuard        op;
        PlanGuard          plan;
    };

    void createPermutationPlan(hiptensorHandle_t         handle,
                               PermutationRun&           run,
                               hiptensorPlanPreference_t pref)
    {
        ASSERT_EQ(createDesc(handle, run.descA, run.problem.lengthsA), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle, run.descB, run.problem.lengthsB), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePermutation(handle,
                                             &run.op.desc,
                                             run.descA.desc,
                                             run.problem.modesA.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             run.descB.desc,
                                             run.problem.modesB.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePlan(handle, &run.plan.plan, run.op.desc, pref, 0),
                  HIPTENSOR_STATUS_SUCCESS);
    }

    void executeAndCheckPermutation(hiptensorHandle_t handle, PermutationRun const& run)
    {
        int64_t count = 1;
        for(auto length : run.problem.lengthsA)
        {
            count *= length;
        }
        float const alpha = 2.0f;
        auto const  a     = smallIntegers(static_cast<std::size_t>(count), 1);
        auto const  b     = permutationReference(run.problem, a, alpha);

        DeviceBuffer devA, devB;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a));
        ASSERT_NO_FATAL_FAILURE(upload(devB, std::vector<float>(b.size(), -1000.0f)));

        ASSERT_EQ(hiptensorPermute(handle, run.plan.plan, &alpha, devA.ptr, devB.ptr, 0),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expectEqual(devB, b);
    }

    // ---- Scale contraction: D[m,n] = alpha * sum_k A[m,k] * B[n,k] ----

    struct ContractionProblem
    {
        int64_t m, n, k;
    };

    // Independent reference over column-major A[m,k], B[n,k] and D[m,n].
    std::vector<float> contractionReference(ContractionProblem const& p,
                                            std::vector<float> const& a,
                                            std::vector<float> const& b,
                                            float                     alpha)
    {
        std::vector<float> d(static_cast<std::size_t>(p.m * p.n));
        for(int64_t j = 0; j < p.n; j++)
        {
            for(int64_t i = 0; i < p.m; i++)
            {
                double acc = 0.0;
                for(int64_t l = 0; l < p.k; l++)
                {
                    acc += static_cast<double>(a[static_cast<std::size_t>(i + l * p.m)])
                           * static_cast<double>(b[static_cast<std::size_t>(j + l * p.n)]);
                }
                d[static_cast<std::size_t>(i + j * p.m)] = static_cast<float>(alpha * acc);
            }
        }
        return d;
    }

    struct ContractionRun
    {
        ContractionProblem problem;
        TensorDescGuard    descA, descB, descD;
        OpDescGuard        op;
        PlanGuard          plan;
    };

    // Creates the operation descriptor. Returns false, without failing, when the device has no
    // 32F contraction support.
    bool createContraction(hiptensorHandle_t handle, ContractionRun& run)
    {
        int32_t const modesA[] = {'m', 'k'};
        int32_t const modesB[] = {'n', 'k'};
        int32_t const modesD[] = {'m', 'n'};
        auto const&   p        = run.problem;

        EXPECT_EQ(createDesc(handle, run.descA, {p.m, p.k}), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle, run.descB, {p.n, p.k}), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle, run.descD, {p.m, p.n}), HIPTENSOR_STATUS_SUCCESS);

        auto status = hiptensorCreateContraction(handle,
                                                 &run.op.desc,
                                                 run.descA.desc,
                                                 modesA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 run.descB.desc,
                                                 modesB,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 run.descD.desc,
                                                 modesD,
                                                 HIPTENSOR_COMPUTE_DESC_32F);
        if(status != HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        }
        return status == HIPTENSOR_STATUS_SUCCESS;
    }

    void executeAndCheckContraction(hiptensorHandle_t handle, ContractionRun const& run)
    {
        auto const& p     = run.problem;
        float const alpha = 2.0f;
        auto const  a     = smallIntegers(static_cast<std::size_t>(p.m * p.k), 2);
        auto const  b     = smallIntegers(static_cast<std::size_t>(p.n * p.k), 5);
        auto const  d     = contractionReference(p, a, b, alpha);

        DeviceBuffer devA, devB, devD;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a));
        ASSERT_NO_FATAL_FAILURE(upload(devB, b));
        ASSERT_NO_FATAL_FAILURE(upload(devD, std::vector<float>(d.size(), -1000.0f)));

        ASSERT_EQ(hiptensorContract(handle,
                                    run.plan.plan,
                                    &alpha,
                                    devA.ptr,
                                    devB.ptr,
                                    nullptr,
                                    nullptr,
                                    devD.ptr,
                                    nullptr,
                                    0,
                                    0 /* stream */),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expectEqual(devD, d);
    }

    constexpr char const* noContractionSupport = "32F contraction is not supported on this device";
} // namespace

// ---- Operation descriptor attributes ----

// A rank-3 permutation, whose output B carries three modes, and a reduction of a rank-3 A to a
// rank-1 D, whose output carries one.
class OperationDescriptorAttributeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle.handle), HIPTENSOR_STATUS_SUCCESS);

        mPermutation.problem = rank3Permutation();
        auto& p              = mPermutation.problem;
        ASSERT_EQ(createDesc(mHandle.handle, mPermutation.descA, p.lengthsA),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(mHandle.handle, mPermutation.descB, p.lengthsB),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePermutation(mHandle.handle,
                                             &mPermutation.op.desc,
                                             mPermutation.descA.desc,
                                             p.modesA.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             mPermutation.descB.desc,
                                             p.modesB.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);

        int32_t const modesA[] = {'a', 'b', 'c'};
        int32_t const modesD[] = {'a'};
        ASSERT_EQ(createDesc(mHandle.handle, mReduceA, {2, 3, 4}), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(mHandle.handle, mReduceD, {2}), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateReduction(mHandle.handle,
                                           &mReduction.desc,
                                           mReduceA.desc,
                                           modesA,
                                           HIPTENSOR_OP_IDENTITY,
                                           mReduceD.desc,
                                           modesD,
                                           HIPTENSOR_OP_IDENTITY,
                                           mReduceD.desc,
                                           modesD,
                                           HIPTENSOR_OP_ADD,
                                           HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
    }

    hiptensorStatus_t set(hiptensorOperationDescriptor_t          desc,
                          hiptensorOperationDescriptorAttribute_t attr,
                          void const*                             buf,
                          size_t                                  size)
    {
        return hiptensorOperationDescriptorSetAttribute(mHandle.handle, desc, attr, buf, size);
    }

    hiptensorStatus_t get(hiptensorOperationDescriptor_t          desc,
                          hiptensorOperationDescriptorAttribute_t attr,
                          void*                                   buf,
                          size_t                                  size)
    {
        return hiptensorOperationDescriptorGetAttribute(mHandle.handle, desc, attr, buf, size);
    }

    // The fields next to the padding attributes keep their values.
    void expectScalarFieldsIntact(hiptensorOperationDescriptor_t desc, uint32_t tag)
    {
        uint32_t            gotTag = ~tag;
        hiptensorDataType_t scalarType{};
        EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &gotTag, sizeof(gotTag)),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(gotTag, tag);
        EXPECT_EQ(
            get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &scalarType, sizeof(scalarType)),
            HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(scalarType, HIPTENSOR_R_32F);
    }

    HandleGuard     mHandle;
    PermutationRun  mPermutation;
    TensorDescGuard mReduceA, mReduceD;
    OpDescGuard     mReduction;
};

TEST_F(OperationDescriptorAttributeTest, ScalarAttributesRoundTrip)
{
    auto           desc       = mPermutation.op.desc;
    uint32_t const tag        = 7u;
    float const    flops      = 1.5f;
    float const    movedBytes = 2.5f;
    ASSERT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, sizeof(flops)),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(
        set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &movedBytes, sizeof(movedBytes)),
        HIPTENSOR_STATUS_SUCCESS);

    float gotFlops = 0.0f, gotMovedBytes = 0.0f;
    EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS, &gotFlops, sizeof(gotFlops)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(gotFlops, flops);
    EXPECT_EQ(get(desc,
                  HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES,
                  &gotMovedBytes,
                  sizeof(gotMovedBytes)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(gotMovedBytes, movedBytes);
    expectScalarFieldsIntact(desc, tag);
}

// A buffer larger than the attribute used to be copied whole, over the next fields.
TEST_F(OperationDescriptorAttributeTest, ScalarAttributeOfWrongSizeIsRejected)
{
    auto           desc = mPermutation.op.desc;
    uint64_t const wide = (uint64_t(HIPTENSOR_R_16F) << 32) | 7u;
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &wide, sizeof(wide)),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &wide, 2),
              HIPTENSOR_STATUS_INVALID_VALUE);

    uint64_t readBack = 0;
    EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &readBack, sizeof(readBack)),
              HIPTENSOR_STATUS_INVALID_VALUE);
    expectScalarFieldsIntact(desc, 0u);
}

TEST_F(OperationDescriptorAttributeTest, NullBufferIsRejected)
{
    auto desc = mPermutation.op.desc;
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, nullptr, sizeof(uint32_t)),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, nullptr, sizeof(uint32_t)),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, nullptr, sizeof(float)),
              HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST_F(OperationDescriptorAttributeTest, NullDescriptorIsRejected)
{
    uint32_t tag = 0u;
    EXPECT_EQ(set(nullptr, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(get(nullptr, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
}

TEST_F(OperationDescriptorAttributeTest, UnknownAttributeIsNotSupported)
{
    auto       desc    = mPermutation.op.desc;
    auto const unknown = static_cast<hiptensorOperationDescriptorAttribute_t>(64);
    uint32_t   value   = 0u;
    EXPECT_EQ(set(desc, unknown, &value, sizeof(value)), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(get(desc, unknown, &value, sizeof(value)), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// One uint32_t per output mode, as cuTENSOR takes it. Zero padding leaves the output unchanged.
TEST_F(OperationDescriptorAttributeTest, ZeroPaddingPerOutputModeIsAccepted)
{
    auto           desc    = mPermutation.op.desc;
    uint32_t const zero[3] = {0u, 0u, 0u};
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, zero, sizeof(zero)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, zero, sizeof(zero)),
              HIPTENSOR_STATUS_SUCCESS);

    for(auto attr :
        {HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT})
    {
        uint32_t padding[3] = {9u, 9u, 9u};
        EXPECT_EQ(get(desc, attr, padding, sizeof(padding)), HIPTENSOR_STATUS_SUCCESS);
        for(auto p : padding)
        {
            EXPECT_EQ(p, 0u);
        }
    }
    expectScalarFieldsIntact(desc, 0u);
}

// No kernel pads its output, so non-zero padding would be silently ignored. It used to be
// copied over the fields after the single padding field.
TEST_F(OperationDescriptorAttributeTest, NonZeroPaddingIsNotSupported)
{
    auto           desc       = mPermutation.op.desc;
    uint32_t const padding[3] = {1u, 2u, 3u};
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, padding, sizeof(padding)),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, padding, sizeof(padding)),
              HIPTENSOR_STATUS_NOT_SUPPORTED);

    for(auto attr :
        {HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT})
    {
        uint32_t got[3] = {9u, 9u, 9u};
        EXPECT_EQ(get(desc, attr, got, sizeof(got)), HIPTENSOR_STATUS_SUCCESS);
        for(auto p : got)
        {
            EXPECT_EQ(p, 0u);
        }
    }
    expectScalarFieldsIntact(desc, 0u);

    // The permutation still runs.
    ASSERT_EQ(hiptensorCreatePlan(mHandle.handle, &mPermutation.plan.plan, desc, nullptr, 0),
              HIPTENSOR_STATUS_SUCCESS);
    executeAndCheckPermutation(mHandle.handle, mPermutation);
}

// The array length follows the output's modes: three for the permutation's B, one for the
// reduction's D. A single uint32_t for a rank-3 output is the wrong size.
TEST_F(OperationDescriptorAttributeTest, PaddingSizeFollowsOutputRank)
{
    uint32_t const zero[4] = {0u, 0u, 0u, 0u};
    for(auto attr :
        {HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT})
    {
        for(size_t modes : {1u, 2u, 4u})
        {
            EXPECT_EQ(set(mPermutation.op.desc, attr, zero, modes * sizeof(uint32_t)),
                      HIPTENSOR_STATUS_INVALID_VALUE)
                << "modes " << modes;
        }
        EXPECT_EQ(set(mPermutation.op.desc, attr, zero, 3 * sizeof(uint32_t)),
                  HIPTENSOR_STATUS_SUCCESS);

        EXPECT_EQ(set(mReduction.desc, attr, zero, 3 * sizeof(uint32_t)),
                  HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(set(mReduction.desc, attr, zero, sizeof(uint32_t)), HIPTENSOR_STATUS_SUCCESS);

        uint32_t got[4] = {9u, 9u, 9u, 9u};
        EXPECT_EQ(get(mReduction.desc, attr, got, 3 * sizeof(uint32_t)),
                  HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(get(mReduction.desc, attr, got, sizeof(uint32_t)), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(got[0], 0u);
        EXPECT_EQ(got[1], 9u);
    }
    expectScalarFieldsIntact(mPermutation.op.desc, 0u);
    expectScalarFieldsIntact(mReduction.desc, 0u);
}

// The padding value is one element of the output's type. It used to be copied through a null
// pointer.
TEST_F(OperationDescriptorAttributeTest, PaddingValueRoundTrips)
{
    auto  desc = mPermutation.op.desc;
    float got  = -1.0f;
    EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &got, sizeof(got)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(got, 0.0f);

    float const value = 0.5f;
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &value, sizeof(value)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &got, sizeof(got)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(got, value);

    double const wide = 0.25;
    EXPECT_EQ(set(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &wide, sizeof(wide)),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &got, sizeof(got)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(got, value);
    expectScalarFieldsIntact(desc, 0u);
}

// ---- Compute descriptors without a scalar type ----

class ComputeDescriptorTest : public ::testing::TestWithParam<hiptensorComputeDescriptor_t>
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle.handle), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(mHandle.handle, mMatrix, {2, 3}), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(mHandle.handle, mTransposed, {3, 2}), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(mHandle.handle, mVector, {2}), HIPTENSOR_STATUS_SUCCESS);
    }

    HandleGuard     mHandle;
    TensorDescGuard mMatrix, mTransposed, mVector;

    int32_t mModesMN[2] = {'m', 'n'};
    int32_t mModesNM[2] = {'n', 'm'};
    int32_t mModesNK[2] = {'n', 'k'};
    int32_t mModesMK[2] = {'m', 'k'};
    int32_t mModesM[1]  = {'m'};
};

TEST_P(ComputeDescriptorTest, ElementwiseAndReductionReturnInvalidValue)
{
    auto const compute = GetParam();
    auto const handle  = mHandle.handle;

    OpDescGuard permutation;
    EXPECT_EQ(hiptensorCreatePermutation(handle,
                                         &permutation.desc,
                                         mMatrix.desc,
                                         mModesMN,
                                         HIPTENSOR_OP_IDENTITY,
                                         mTransposed.desc,
                                         mModesNM,
                                         compute),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(permutation.desc, nullptr);

    OpDescGuard binary;
    EXPECT_EQ(hiptensorCreateElementwiseBinary(handle,
                                               &binary.desc,
                                               mMatrix.desc,
                                               mModesMN,
                                               HIPTENSOR_OP_IDENTITY,
                                               mMatrix.desc,
                                               mModesMN,
                                               HIPTENSOR_OP_IDENTITY,
                                               mMatrix.desc,
                                               mModesMN,
                                               HIPTENSOR_OP_ADD,
                                               compute),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(binary.desc, nullptr);

    OpDescGuard trinary;
    EXPECT_EQ(hiptensorCreateElementwiseTrinary(handle,
                                                &trinary.desc,
                                                mMatrix.desc,
                                                mModesMN,
                                                HIPTENSOR_OP_IDENTITY,
                                                mMatrix.desc,
                                                mModesMN,
                                                HIPTENSOR_OP_IDENTITY,
                                                mMatrix.desc,
                                                mModesMN,
                                                HIPTENSOR_OP_IDENTITY,
                                                mMatrix.desc,
                                                mModesMN,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_OP_ADD,
                                                compute),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(trinary.desc, nullptr);

    OpDescGuard reduction;
    EXPECT_EQ(hiptensorCreateReduction(handle,
                                       &reduction.desc,
                                       mMatrix.desc,
                                       mModesMN,
                                       HIPTENSOR_OP_IDENTITY,
                                       mVector.desc,
                                       mModesM,
                                       HIPTENSOR_OP_IDENTITY,
                                       mVector.desc,
                                       mModesM,
                                       HIPTENSOR_OP_ADD,
                                       compute),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(reduction.desc, nullptr);
}

// Checked before the device's matrix-core support, so the status does not depend on the device.
TEST_P(ComputeDescriptorTest, ContractionsReturnInvalidValue)
{
    auto const compute = GetParam();
    auto const handle  = mHandle.handle;

    // D[m,n] = A[m,k] * B[n,k] with every extent taken from the 2x3 matrix descriptor.
    OpDescGuard contraction;
    EXPECT_EQ(hiptensorCreateContraction(handle,
                                         &contraction.desc,
                                         mMatrix.desc,
                                         mModesMK,
                                         HIPTENSOR_OP_IDENTITY,
                                         mMatrix.desc,
                                         mModesNK,
                                         HIPTENSOR_OP_IDENTITY,
                                         nullptr,
                                         nullptr,
                                         HIPTENSOR_OP_IDENTITY,
                                         mMatrix.desc,
                                         mModesMN,
                                         compute),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(contraction.desc, nullptr);

    OpDescGuard trinary;
    EXPECT_EQ(hiptensorCreateContractionTrinary(handle,
                                                &trinary.desc,
                                                mMatrix.desc,
                                                mModesMK,
                                                HIPTENSOR_OP_IDENTITY,
                                                mMatrix.desc,
                                                mModesNK,
                                                HIPTENSOR_OP_IDENTITY,
                                                mMatrix.desc,
                                                mModesMN,
                                                HIPTENSOR_OP_IDENTITY,
                                                nullptr,
                                                nullptr,
                                                HIPTENSOR_OP_IDENTITY,
                                                mMatrix.desc,
                                                mModesMN,
                                                compute),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(trinary.desc, nullptr);
}

INSTANTIATE_TEST_SUITE_P(NoScalarType,
                         ComputeDescriptorTest,
                         ::testing::Values(HIPTENSOR_COMPUTE_DESC_NONE,
                                           HIPTENSOR_COMPUTE_DESC_8U,
                                           HIPTENSOR_COMPUTE_DESC_8I,
                                           HIPTENSOR_COMPUTE_DESC_32U,
                                           HIPTENSOR_COMPUTE_DESC_32I,
                                           // Bit 1 is not a compute descriptor.
                                           static_cast<hiptensorComputeDescriptor_t>(1U << 1U)));

// Positive control: the same calls succeed with a compute descriptor that has a scalar type.
TEST(ComputeDescriptorControlTest, ElementwiseWithScalarTypeSucceeds)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    TensorDescGuard matrix, transposed;
    ASSERT_EQ(createDesc(handle.handle, matrix, {2, 3}), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(createDesc(handle.handle, transposed, {3, 2}), HIPTENSOR_STATUS_SUCCESS);
    int32_t     modesMN[2] = {'m', 'n'};
    int32_t     modesNM[2] = {'n', 'm'};
    OpDescGuard permutation;
    EXPECT_EQ(hiptensorCreatePermutation(handle.handle,
                                         &permutation.desc,
                                         matrix.desc,
                                         modesMN,
                                         HIPTENSOR_OP_IDENTITY,
                                         transposed.desc,
                                         modesNM,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
}

// ---- hiptensorCreatePlan: null arguments and a null preference ----

TEST(CreatePlanTest, NullArgumentsReturnNotInitialized)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    PermutationRun run;
    run.problem = rank2Permutation();
    ASSERT_NO_FATAL_FAILURE(createPermutationPlan(handle.handle, run, nullptr));

    hiptensorPlan_t plan = nullptr;
    EXPECT_EQ(hiptensorCreatePlan(nullptr, &plan, run.op.desc, nullptr, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorCreatePlan(handle.handle, nullptr, run.op.desc, nullptr, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorCreatePlan(handle.handle, &plan, nullptr, nullptr, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(plan, nullptr);
}

// hiptensor.h documents pref as optional.
TEST(CreatePlanTest, NullPreferenceUsesDefaultsForPermutation)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    PermutationRun run;
    run.problem = rank3Permutation();
    ASSERT_NO_FATAL_FAILURE(createPermutationPlan(handle.handle, run, nullptr));
    executeAndCheckPermutation(handle.handle, run);
}

TEST(CreatePlanTest, NullPreferenceUsesDefaultsForContraction)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    ContractionRun run;
    run.problem = {8, 6, 4};
    if(!createContraction(handle.handle, run))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    ASSERT_EQ(hiptensorCreatePlan(handle.handle, &run.plan.plan, run.op.desc, nullptr, 0),
              HIPTENSOR_STATUS_SUCCESS);
    executeAndCheckContraction(handle.handle, run);
}

// ---- One preference shared by several plans ----

// Each plan records its selected kernel in its own copy of the preference. Sharing the caller's
// preference let the second plan overwrite the first plan's kernel.
TEST(SharedPlanPreferenceTest, PermutationPlansKeepTheirOwnKernel)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    PlanPrefGuard pref;
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  handle.handle, &pref.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);

    PermutationRun rank2, rank3;
    rank2.problem = rank2Permutation();
    rank3.problem = rank3Permutation();
    ASSERT_NO_FATAL_FAILURE(createPermutationPlan(handle.handle, rank2, pref.pref));
    ASSERT_NO_FATAL_FAILURE(createPermutationPlan(handle.handle, rank3, pref.pref));

    executeAndCheckPermutation(handle.handle, rank2);
    executeAndCheckPermutation(handle.handle, rank3);
    executeAndCheckPermutation(handle.handle, rank2);

    EXPECT_EQ(pref.pref->mSolution, nullptr);
    EXPECT_TRUE(pref.pref->mCandidates.empty());
}

TEST(SharedPlanPreferenceTest, ContractionPlansKeepTheirOwnKernel)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    PlanPrefGuard pref;
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  handle.handle, &pref.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);

    ContractionRun first, second;
    first.problem  = {8, 8, 8};
    second.problem = {16, 4, 12};
    if(!createContraction(handle.handle, first) || !createContraction(handle.handle, second))
    {
        GTEST_SKIP() << noContractionSupport;
    }
    ASSERT_EQ(hiptensorCreatePlan(handle.handle, &first.plan.plan, first.op.desc, pref.pref, 0),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreatePlan(handle.handle, &second.plan.plan, second.op.desc, pref.pref, 0),
              HIPTENSOR_STATUS_SUCCESS);

    EXPECT_EQ(pref.pref->mSolution, nullptr);
    EXPECT_TRUE(pref.pref->mCandidates.empty());

    executeAndCheckContraction(handle.handle, first);
    executeAndCheckContraction(handle.handle, second);
    executeAndCheckContraction(handle.handle, first);
}

// ---- Contraction kernel workspace size ----

namespace
{
    using ck::tensor_operation::device::BaseArgument;
    using ck::tensor_operation::device::BaseInvoker;
    using ck::tensor_operation::device::BaseOperator;

    struct WorkspaceArgument : public BaseArgument
    {
        explicit WorkspaceArgument(std::size_t bytes)
            : mBytes(bytes)
        {
        }
        std::size_t mBytes;
    };

    // Like CK operators that need a workspace, reads the size from the argument.
    struct WorkspaceOperator : public BaseOperator
    {
        std::size_t GetWorkSpaceSize(BaseArgument const* arg) const override
        {
            return static_cast<WorkspaceArgument const*>(arg)->mBytes;
        }
    };

    class WorkspaceSolution : public hiptensor::ContractionSolution
    {
    public:
        static constexpr std::size_t requiredBytes = 256;

        WorkspaceSolution()
            : hiptensor::ContractionSolution(std::make_unique<WorkspaceOperator>(), nullptr)
        {
        }

        bool initArgs(void const*,
                      void const*,
                      void const*,
                      void const*,
                      void const*,
                      void*,
                      std::vector<std::size_t>,
                      std::vector<std::size_t>,
                      std::vector<int32_t>,
                      std::vector<std::size_t>,
                      std::vector<std::size_t>,
                      std::vector<int32_t>,
                      std::vector<std::size_t>,
                      std::vector<std::size_t>,
                      std::vector<int32_t>,
                      std::vector<std::size_t>,
                      std::vector<std::size_t>,
                      std::vector<int32_t>,
                      hiptensor::ContractionUnaryOps const&,
                      void*) override
        {
            mInvokerArgPtr = std::make_unique<WorkspaceArgument>(requiredBytes);
            mInvokerPtr    = std::make_unique<BaseInvoker>();
            mValid         = true;
            return true;
        }

        std::tuple<hiptensorStatus_t, float> run(unsigned long workspaceBytes)
        {
            return (*this)(nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           {},
                           nullptr,
                           workspaceBytes);
        }
    };
} // namespace

// hiptensorContract logs the required size after an INSUFFICIENT_WORKSPACE result. The
// arguments the size is read from used to be released first.
TEST(ContractionSolutionWorkspaceTest, RequiredSizeAfterInsufficientWorkspace)
{
    WorkspaceSolution solution;
    auto [status, elapsed] = solution.run(WorkspaceSolution::requiredBytes - 1);
    EXPECT_EQ(status, HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE);
    EXPECT_EQ(solution.workspaceSize(), WorkspaceSolution::requiredBytes);
}

TEST(ContractionSolutionWorkspaceTest, SizeAfterRunIsZero)
{
    WorkspaceSolution solution;
    auto [status, elapsed] = solution.run(WorkspaceSolution::requiredBytes);
    EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(solution.workspaceSize(), 0u);
}

// The scalars of hiptensorContract are host pointers without an alignment requirement beyond the
// element bytes: a complex-double alpha stored as double[2] (8-byte aligned) or at any byte offset
// of a buffer. Reading them as hipDoubleComplex (16-byte aligned type) used to fault in readVal
// (SIGSEGV on every complex-double contraction with such scalars, Composable Kernel backend).
// Complex f64 and f32 contractions, scale and bilinear, with alpha and beta 0, 4 and 8 bytes past a
// 16-byte boundary, on each backend built into the library, against an exact loop reference.
namespace
{
    // Sets an environment variable for the life of the object.
    struct ScopedEnvironment
    {
        ScopedEnvironment(char const* name, char const* value)
            : mName(name)
        {
            if(char const* old = std::getenv(name))
            {
                mOld    = old;
                mHadOld = true;
            }
            set(value);
        }
        ~ScopedEnvironment()
        {
            set(mHadOld ? mOld.c_str() : nullptr);
        }
        void set(char const* value)
        {
#ifdef _WIN32
            _putenv_s(mName, value == nullptr ? "" : value);
#else
            if(value == nullptr)
            {
                unsetenv(mName);
            }
            else
            {
                setenv(mName, value, 1);
            }
#endif
        }
        char const* mName;
        std::string mOld;
        bool        mHadOld = false;
    };

    struct ScalarAlignmentCase
    {
        char const*         backend;
        hiptensorDataType_t type; // HIPTENSOR_C_32F or HIPTENSOR_C_64F
        bool                bilinear;
        std::size_t         offset; // bytes past a 16-byte boundary
    };

    std::string scalarAlignmentName(::testing::TestParamInfo<ScalarAlignmentCase> const& info)
    {
        auto const& c = info.param;
        return std::string(c.backend) + (c.type == HIPTENSOR_C_64F ? "_c64" : "_c32")
               + (c.bilinear ? "_bilinear" : "_scale") + "_offset" + std::to_string(c.offset);
    }

    class ScalarAlignmentTest : public ::testing::TestWithParam<ScalarAlignmentCase>
    {
    };

    // Stores complex value v as element i of `bytes` (complex f32 or f64).
    void putComplex(std::vector<uint8_t>& bytes, std::size_t i, bool f64, double re, double im)
    {
        if(f64)
        {
            double const v[2] = {re, im};
            std::memcpy(bytes.data() + i * 16, v, 16);
        }
        else
        {
            float const v[2] = {static_cast<float>(re), static_cast<float>(im)};
            std::memcpy(bytes.data() + i * 8, v, 8);
        }
    }

    std::complex<double> getComplex(std::vector<uint8_t> const& bytes, std::size_t i, bool f64)
    {
        if(f64)
        {
            double v[2];
            std::memcpy(v, bytes.data() + i * 16, 16);
            return {v[0], v[1]};
        }
        float v[2];
        std::memcpy(v, bytes.data() + i * 8, 8);
        return {v[0], v[1]};
    }
} // namespace

TEST_P(ScalarAlignmentTest, ComplexContractionWithUnalignedScalars)
{
    auto const        c   = GetParam();
    bool const        f64 = c.type == HIPTENSOR_C_64F;
    std::size_t const es  = f64 ? 16 : 8;
    ScopedEnvironment backend("HIPTENSOR_BACKEND", c.backend);
    ScopedEnvironment fallback("HIPTENSOR_BACKEND_FALLBACK", "0");

    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

    // D[c,d,e,f] = alpha * A[c,d,a,b] * B[e,f,a,b] (+ beta * C[c,d,e,f]), packed column-major
    int64_t const        a = 3, b = 2, cc = 4, d = 3, e = 5, f = 2;
    std::vector<int64_t> lA = {cc, d, a, b}, lB = {e, f, a, b}, lD = {cc, d, e, f};
    std::vector<int32_t> mA = {'c', 'd', 'a', 'b'}, mB = {'e', 'f', 'a', 'b'},
                         mD = {'c', 'd', 'e', 'f'};
    std::size_t const nA = cc * d * a * b, nB = e * f * a * b, nD = cc * d * e * f;
    auto              make = [&](TensorDescGuard& g, std::vector<int64_t> const& l) {
        auto s = colMajorStrides(l);
        return hiptensorCreateTensorDescriptor(handle.handle,
                                               &g.desc,
                                               static_cast<uint32_t>(l.size()),
                                               l.data(),
                                               s.data(),
                                               c.type,
                                               alignment);
    };
    TensorDescGuard dA, dB, dC, dD;
    ASSERT_EQ(make(dA, lA), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(make(dB, lB), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(make(dC, lD), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(make(dD, lD), HIPTENSOR_STATUS_SUCCESS);

    OpDescGuard op;
    auto const  compute = f64 ? HIPTENSOR_COMPUTE_DESC_C64F : HIPTENSOR_COMPUTE_DESC_C32F;
    auto        status  = hiptensorCreateContraction(handle.handle,
                                             &op.desc,
                                             dA.desc,
                                             mA.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB.desc,
                                             mB.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             c.bilinear ? dC.desc : nullptr,
                                             c.bilinear ? mD.data() : nullptr,
                                             HIPTENSOR_OP_IDENTITY,
                                             dD.desc,
                                             mD.data(),
                                             compute);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "no complex contraction on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    PlanPrefGuard pref;
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  handle.handle, &pref.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);
    uint64_t workspaceSize = 0;
    status                 = hiptensorEstimateWorkspaceSize(
        handle.handle, op.desc, pref.pref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize);
    PlanGuard plan;
    if(status == HIPTENSOR_STATUS_SUCCESS)
    {
        status = hiptensorCreatePlan(handle.handle, &plan.plan, op.desc, pref.pref, workspaceSize);
    }
    if(status == HIPTENSOR_STATUS_NOT_SUPPORTED || status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "backend " << c.backend << " is not built into this library or has no "
                     << "kernels for this device (" << hiptensorGetErrorString(status) << ")";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    uint64_t required = 0;
    ASSERT_EQ(hiptensorPlanGetAttribute(handle.handle,
                                        plan.plan,
                                        HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                        &required,
                                        sizeof(required)),
              HIPTENSOR_STATUS_SUCCESS);

    // small complex integers: every product and sum is exact in f32
    std::vector<uint8_t> hA(nA * es), hB(nB * es), hC(nD * es);
    auto                 value = [](std::size_t i, int seed) {
        return static_cast<double>(static_cast<int>((i * 5 + seed) % 7) - 3);
    };
    for(std::size_t i = 0; i < nA; i++)
    {
        putComplex(hA, i, f64, value(i, 1), value(i, 2));
    }
    for(std::size_t i = 0; i < nB; i++)
    {
        putComplex(hB, i, f64, value(i, 3), value(i, 4));
    }
    for(std::size_t i = 0; i < nD; i++)
    {
        putComplex(hC, i, f64, value(i, 5), value(i, 6));
    }
    DeviceBuffer pA, pB, pC, pD, ws;
    ASSERT_EQ(hipMalloc(&pA.ptr, hA.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&pB.ptr, hB.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&pC.ptr, hC.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&pD.ptr, hC.size()), hipSuccess);
    ASSERT_EQ(hipMemcpy(pA.ptr, hA.data(), hA.size(), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(pB.ptr, hB.data(), hB.size(), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(pC.ptr, hC.data(), hC.size(), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(pD.ptr, 0xFF, hC.size()), hipSuccess); // NaN until written
    if(required > 0)
    {
        ASSERT_EQ(hipMalloc(&ws.ptr, required), hipSuccess);
    }

    // alpha = 2 - i, beta = -1 + 3i, `offset` bytes past 16-byte boundaries
    std::complex<double> const alpha(2.0, -1.0),
        beta(c.bilinear ? -1.0 : 0.0, c.bilinear ? 3.0 : 0.0);
    alignas(16) uint8_t  scalars[64] = {};
    std::vector<uint8_t> sa(es), sb(es);
    putComplex(sa, 0, f64, alpha.real(), alpha.imag());
    putComplex(sb, 0, f64, beta.real(), beta.imag());
    std::memcpy(scalars + c.offset, sa.data(), es);
    std::memcpy(scalars + 32 + c.offset, sb.data(), es);

    ASSERT_EQ(hiptensorContract(handle.handle,
                                plan.plan,
                                scalars + c.offset,
                                pA.ptr,
                                pB.ptr,
                                scalars + 32 + c.offset,
                                c.bilinear ? pC.ptr : nullptr,
                                pD.ptr,
                                ws.ptr,
                                required,
                                nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<uint8_t> got(hC.size());
    ASSERT_EQ(hipMemcpy(got.data(), pD.ptr, got.size(), hipMemcpyDeviceToHost), hipSuccess);

    int bad = 0;
    for(int64_t fi = 0; fi < f; fi++)
    {
        for(int64_t ei = 0; ei < e; ei++)
        {
            for(int64_t di = 0; di < d; di++)
            {
                for(int64_t ci = 0; ci < cc; ci++)
                {
                    std::complex<double> sum = 0;
                    for(int64_t bi = 0; bi < b; bi++)
                    {
                        for(int64_t ai = 0; ai < a; ai++)
                        {
                            sum += getComplex(hA, ci + cc * (di + d * (ai + a * bi)), f64)
                                   * getComplex(hB, ei + e * (fi + f * (ai + a * bi)), f64);
                        }
                    }
                    std::size_t const i    = ci + cc * (di + d * (ei + e * fi));
                    auto const        want = alpha * sum + beta * getComplex(hC, i, f64);
                    if(getComplex(got, i, f64) != want && bad++ < 5)
                    {
                        ADD_FAILURE() << "element " << i << " got " << getComplex(got, i, f64)
                                      << " want " << want;
                    }
                }
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         ScalarAlignmentTest,
                         ::testing::ValuesIn([] {
                             std::vector<ScalarAlignmentCase> cases;
                             for(char const* backend : {"ck", "rocke"})
                             {
                                 for(auto type : {HIPTENSOR_C_64F, HIPTENSOR_C_32F})
                                 {
                                     for(bool bilinear : {false, true})
                                     {
                                         for(std::size_t offset :
                                             {std::size_t(0), std::size_t(4), std::size_t(8)})
                                         {
                                             cases.push_back({backend, type, bilinear, offset});
                                         }
                                     }
                                 }
                             }
                             return cases;
                         }()),
                         scalarAlignmentName);
