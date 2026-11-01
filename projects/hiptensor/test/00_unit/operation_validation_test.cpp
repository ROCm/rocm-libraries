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

// Regression tests for the validation of tensor descriptors, contraction modes and elementwise
// and reduction plans:
//  - contractions with more than six modes per M / N / K group, without M or N modes, with a mode
//    carried by one tensor only, a repeated mode or mismatched extents used to be accepted and
//    then write past the library's mode tables or read modes that were not there;
//  - tensor descriptors without extents, or with non-positive extents or negative strides, were
//    accepted (a null extent array was read);
//  - elementwise and reduction plans were created for problems no kernel can run, which then
//    failed with HIPTENSOR_STATUS_INTERNAL_ERROR at execution (or, for reductions with mismatched
//    extents, read past the input).
// The contract checked for plans: hiptensorCreatePlan either refuses the problem, or the plan
// executes and gives the result of a plain loop reference computed here.

#include "api_test_helpers.hpp"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using hiptensor_test::ApiObjects;
using hiptensor_test::DeviceBuffer;
using hiptensor_test::elementCount;
using hiptensor_test::packedStrides;

namespace
{
    // A binary contraction descriptor with f32 data and compute; returns the create status.
    struct ContractionCase
    {
        std::vector<int32_t> modeA, modeB, modeD;
        std::vector<int64_t> lensA, lensB, lensD;
    };

    hiptensorStatus_t createContraction(ApiObjects& api, ContractionCase const& c)
    {
        auto descA = api.tensor(c.lensA, {}, HIPTENSOR_R_32F);
        auto descB = api.tensor(c.lensB, {}, HIPTENSOR_R_32F);
        auto descD = api.tensor(c.lensD, {}, HIPTENSOR_R_32F);

        hiptensorOperationDescriptor_t op     = nullptr;
        auto                           status = hiptensorCreateContraction(api.handle,
                                                 &op,
                                                 descA,
                                                 c.modeA.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descB,
                                                 c.modeB.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descD,
                                                 c.modeD.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F);
        api.keep(op);
        if(status != HIPTENSOR_STATUS_SUCCESS && op != nullptr)
        {
            ADD_FAILURE() << "A failed create left a descriptor";
        }
        return status;
    }

    std::vector<int32_t> modeRange(int32_t first, int32_t count)
    {
        std::vector<int32_t> modes(count);
        std::iota(modes.begin(), modes.end(), first);
        return modes;
    }

    // Skips the test when the device has no f32 contraction (the create functions return
    // HIPTENSOR_STATUS_ARCH_MISMATCH before they look at the modes).
    bool hasF32Contraction(ApiObjects& api)
    {
        return createContraction(api, {{'m', 'k'}, {'n', 'k'}, {'m', 'n'}, {4, 3}, {5, 3}, {4, 5}})
               == HIPTENSOR_STATUS_SUCCESS;
    }
} // namespace

TEST(TensorDescriptorValidationTest, RejectsMissingOrInvalidExtentsAndStrides)
{
    ApiObjects                  api;
    hiptensorTensorDescriptor_t desc = nullptr;

    EXPECT_EQ(
        hiptensorCreateTensorDescriptor(api.handle, &desc, 2, nullptr, nullptr, HIPTENSOR_R_32F, 0),
        HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(desc, nullptr);

    int64_t const zeroExtent[]     = {4, 0};
    int64_t const negativeExtent[] = {-4, 2};
    for(auto const* lens : {zeroExtent, negativeExtent})
    {
        EXPECT_EQ(hiptensorCreateTensorDescriptor(
                      api.handle, &desc, 2, lens, nullptr, HIPTENSOR_R_32F, 0),
                  HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(desc, nullptr);
    }

    int64_t const lens[]            = {4, 2};
    int64_t const negativeStrides[] = {1, -4};
    EXPECT_EQ(hiptensorCreateTensorDescriptor(
                  api.handle, &desc, 2, lens, negativeStrides, HIPTENSOR_R_32F, 0),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(desc, nullptr);

    EXPECT_EQ(
        hiptensorCreateTensorDescriptor(api.handle, nullptr, 2, lens, nullptr, HIPTENSOR_R_32F, 0),
        HIPTENSOR_STATUS_NOT_INITIALIZED);

    // A rank-0 tensor needs no extents.
    EXPECT_EQ(
        hiptensorCreateTensorDescriptor(api.handle, &desc, 0, nullptr, nullptr, HIPTENSOR_R_32F, 0),
        HIPTENSOR_STATUS_SUCCESS);
    if(desc != nullptr)
    {
        hiptensorDestroyTensorDescriptor(desc);
    }
}

TEST(ContractionModeValidationTest, RejectsModesTheKernelsCannotAddress)
{
    ApiObjects api;
    if(!hasF32Contraction(api))
    {
        GTEST_SKIP() << "No f32 contraction on this device";
    }

    // Seven M, N or K modes (the kernels address six per group).
    auto m7  = modeRange('a', 7);
    auto n7  = modeRange('A', 7);
    auto k7  = modeRange('0', 7);
    auto two = [](std::size_t n) { return std::vector<int64_t>(n, 2); };

    struct Named
    {
        char const*       name;
        ContractionCase   c;
        hiptensorStatus_t expected;
    };
    std::vector<Named> cases;

    {
        auto modeA = m7;
        modeA.push_back('k');
        auto modeD = m7;
        modeD.push_back('n');
        cases.push_back({"seven M modes",
                         {modeA, {'n', 'k'}, modeD, two(8), {2, 2}, two(8)},
                         HIPTENSOR_STATUS_NOT_SUPPORTED});
    }
    {
        auto modeB = n7;
        modeB.push_back('k');
        auto modeD = std::vector<int32_t>{'m'};
        modeD.insert(modeD.end(), n7.begin(), n7.end());
        cases.push_back({"seven N modes",
                         {{'m', 'k'}, modeB, modeD, {2, 2}, two(8), two(8)},
                         HIPTENSOR_STATUS_NOT_SUPPORTED});
    }
    {
        auto modeA = std::vector<int32_t>{'m'};
        modeA.insert(modeA.end(), k7.begin(), k7.end());
        auto modeB = std::vector<int32_t>{'n'};
        modeB.insert(modeB.end(), k7.begin(), k7.end());
        cases.push_back({"seven K modes",
                         {modeA, modeB, {'m', 'n'}, two(8), two(8), {2, 2}},
                         HIPTENSOR_STATUS_NOT_SUPPORTED});
    }
    cases.push_back({"no M mode",
                     {{'k'}, {'n', 'k'}, {'n'}, {3}, {4, 3}, {4}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"no N mode",
                     {{'m', 'k'}, {'k'}, {'m'}, {4, 3}, {3}, {4}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"output mode in neither input",
                     {{'m', 'k'}, {'n', 'k'}, {'m', 'n', 'x'}, {4, 3}, {5, 3}, {4, 5, 2}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"mode carried by A only",
                     {{'m', 'k', 'z'}, {'n', 'k'}, {'m', 'n'}, {4, 3, 2}, {5, 3}, {4, 5}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"mode carried by B only",
                     {{'m', 'k'}, {'n', 'k', 'z'}, {'m', 'n'}, {4, 3}, {5, 3, 2}, {4, 5}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"mode repeated in A",
                     {{'m', 'm', 'k'}, {'n', 'k'}, {'m', 'n'}, {4, 4, 3}, {5, 3}, {4, 5}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"mode repeated in D",
                     {{'m', 'k'}, {'n', 'k'}, {'m', 'n', 'n'}, {4, 3}, {5, 3}, {4, 5, 5}},
                     HIPTENSOR_STATUS_NOT_SUPPORTED});
    cases.push_back({"K extent differs between A and B",
                     {{'m', 'k'}, {'n', 'k'}, {'m', 'n'}, {4, 3}, {5, 4}, {4, 5}},
                     HIPTENSOR_STATUS_INVALID_VALUE});
    cases.push_back({"M extent differs between A and D",
                     {{'m', 'k'}, {'n', 'k'}, {'m', 'n'}, {4, 3}, {5, 3}, {6, 5}},
                     HIPTENSOR_STATUS_INVALID_VALUE});
    // Six modes in every group is the largest problem the kernels address.
    {
        auto m6    = modeRange('a', 6);
        auto n6    = modeRange('A', 6);
        auto k6    = modeRange('0', 6);
        auto modeA = m6;
        modeA.insert(modeA.end(), k6.begin(), k6.end());
        auto modeB = n6;
        modeB.insert(modeB.end(), k6.begin(), k6.end());
        auto modeD = m6;
        modeD.insert(modeD.end(), n6.begin(), n6.end());
        cases.push_back({"six modes per group",
                         {modeA, modeB, modeD, two(12), two(12), two(12)},
                         HIPTENSOR_STATUS_SUCCESS});
    }

    for(auto const& c : cases)
    {
        EXPECT_EQ(createContraction(api, c.c), c.expected) << c.name;
    }
}

TEST(ContractionModeValidationTest, TrinaryRejectsStepsTheKernelsCannotAddress)
{
    ApiObjects api;
    if(!hasF32Contraction(api))
    {
        GTEST_SKIP() << "No f32 contraction on this device";
    }

    // E_{m,n} = A_{m,k} B_{k,j} C_{j,n}: valid. Variants put a mode on one tensor only.
    struct Case
    {
        char const*          name;
        std::vector<int32_t> modeA, modeB, modeC, modeE;
        hiptensorStatus_t    expected;
    };
    std::vector<Case> cases = {
        {"valid", {'m', 'k'}, {'k', 'j'}, {'j', 'n'}, {'m', 'n'}, HIPTENSOR_STATUS_SUCCESS},
        {"mode carried by A only",
         {'m', 'k', 'z'},
         {'k', 'j'},
         {'j', 'n'},
         {'m', 'n'},
         HIPTENSOR_STATUS_NOT_SUPPORTED},
        {"mode carried by C only",
         {'m', 'k'},
         {'k', 'j'},
         {'j', 'n', 'z'},
         {'m', 'n'},
         HIPTENSOR_STATUS_NOT_SUPPORTED},
        {"output mode in no input",
         {'m', 'k'},
         {'k', 'j'},
         {'j', 'n'},
         {'m', 'n', 'z'},
         HIPTENSOR_STATUS_NOT_SUPPORTED},
    };

    for(auto const& c : cases)
    {
        auto lens = [](std::vector<int32_t> const& modes) {
            return std::vector<int64_t>(modes.size(), 3);
        };
        auto descA = api.tensor(lens(c.modeA), {}, HIPTENSOR_R_32F);
        auto descB = api.tensor(lens(c.modeB), {}, HIPTENSOR_R_32F);
        auto descC = api.tensor(lens(c.modeC), {}, HIPTENSOR_R_32F);
        auto descE = api.tensor(lens(c.modeE), {}, HIPTENSOR_R_32F);

        hiptensorOperationDescriptor_t op = nullptr;
        EXPECT_EQ(hiptensorCreateContractionTrinary(api.handle,
                                                    &op,
                                                    descA,
                                                    c.modeA.data(),
                                                    HIPTENSOR_OP_IDENTITY,
                                                    descB,
                                                    c.modeB.data(),
                                                    HIPTENSOR_OP_IDENTITY,
                                                    descC,
                                                    c.modeC.data(),
                                                    HIPTENSOR_OP_IDENTITY,
                                                    nullptr,
                                                    nullptr,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    descE,
                                                    c.modeE.data(),
                                                    HIPTENSOR_COMPUTE_DESC_32F),
                  c.expected)
            << c.name;
        api.keep(op);
    }
}

// A contraction without K modes (an outer product) is accepted, and gives the loop result.
TEST(ContractionModeValidationTest, OuterProductRuns)
{
    ApiObjects api;
    if(!hasF32Contraction(api))
    {
        GTEST_SKIP() << "No f32 contraction on this device";
    }

    constexpr int64_t M = 12, N = 10;
    auto              descA   = api.tensor({M}, {1}, HIPTENSOR_R_32F);
    auto              descB   = api.tensor({N}, {1}, HIPTENSOR_R_32F);
    auto              descD   = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    int32_t const     modeA[] = {'m'}, modeB[] = {'n'}, modeD[] = {'m', 'n'};

    hiptensorOperationDescriptor_t op = nullptr;
    ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                         &op,
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
    api.keep(op);

    uint64_t workspaceSize = 0;
    ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, op, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
              HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t plan = nullptr;
    ASSERT_EQ(api.plan(op, plan, workspaceSize), HIPTENSOR_STATUS_SUCCESS);

    std::vector<float> a(M), b(N), c(M * N);
    for(int64_t i = 0; i < M; ++i)
    {
        a[i] = static_cast<float>(i % 5 - 2);
    }
    for(int64_t j = 0; j < N; ++j)
    {
        b[j] = static_cast<float>(j % 4 - 1);
    }
    for(int64_t i = 0; i < M * N; ++i)
    {
        c[i] = static_cast<float>(i % 7 - 3);
    }
    DeviceBuffer<float> dA(a), dB(b), dC(c);
    DeviceBuffer<float> dD(std::vector<float>(M * N, NAN));
    void*               workspace = nullptr;
    if(workspaceSize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(&workspace, workspaceSize));
    }

    float const alpha = 1.5f, beta = 0.5f;
    EXPECT_EQ(hiptensorContract(api.handle,
                                plan,
                                &alpha,
                                dA.ptr,
                                dB.ptr,
                                &beta,
                                dC.ptr,
                                dD.ptr,
                                workspace,
                                workspaceSize,
                                nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    if(workspace)
    {
        CHECK_HIP_ERROR(hipFree(workspace));
    }

    // Small integers and dyadic scalars: exact.
    auto d = dD.read();
    for(int64_t j = 0; j < N; ++j)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            double expected = 1.5 * a[i] * b[j] + 0.5 * c[i + M * j];
            EXPECT_EQ(d[i + M * j], expected) << "m = " << i << ", n = " << j;
        }
    }
}

TEST(ElementwiseOperatorValidationTest, RejectsNonUnaryOperatorsOnOperands)
{
    ApiObjects    api;
    auto          desc    = api.tensor({4, 3}, {}, HIPTENSOR_R_32F);
    int32_t const modes[] = {'m', 'n'};

    for(auto badOp : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX, HIPTENSOR_OP_UNKNOWN})
    {
        hiptensorOperationDescriptor_t op = nullptr;
        EXPECT_EQ(hiptensorCreatePermutation(
                      api.handle, &op, desc, modes, badOp, desc, modes, HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << badOp;
        api.keep(op);
        EXPECT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                   &op,
                                                   desc,
                                                   modes,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   desc,
                                                   modes,
                                                   badOp,
                                                   desc,
                                                   modes,
                                                   HIPTENSOR_OP_ADD,
                                                   HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << badOp;
        api.keep(op);
        EXPECT_EQ(hiptensorCreateReduction(api.handle,
                                           &op,
                                           desc,
                                           modes,
                                           badOp,
                                           desc,
                                           modes,
                                           HIPTENSOR_OP_IDENTITY,
                                           desc,
                                           modes,
                                           HIPTENSOR_OP_ADD,
                                           HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << badOp;
        api.keep(op);
    }

    // The reduction operator must combine two values.
    hiptensorOperationDescriptor_t op = nullptr;
    EXPECT_EQ(hiptensorCreateReduction(api.handle,
                                       &op,
                                       desc,
                                       modes,
                                       HIPTENSOR_OP_IDENTITY,
                                       desc,
                                       modes,
                                       HIPTENSOR_OP_IDENTITY,
                                       desc,
                                       modes,
                                       HIPTENSOR_OP_SQRT,
                                       HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_INVALID_VALUE);
    api.keep(op);
}

namespace
{
    // Plan contract: refused at plan time, or the plan executes with the reference result.
    template <typename Execute, typename Check>
    void expectRefusedOrCorrect(char const*                    name,
                                ApiObjects&                    api,
                                hiptensorOperationDescriptor_t op,
                                Execute                        execute,
                                Check                          check)
    {
        SCOPED_TRACE(name);
        hiptensorPlan_t plan   = nullptr;
        auto            status = api.plan(op, plan);
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED);
            EXPECT_EQ(plan, nullptr);
            std::cout << name << ": refused when the plan is created" << std::endl;
            return;
        }
        std::cout << name << ": planned and executed" << std::endl;
        ASSERT_EQ(execute(plan), HIPTENSOR_STATUS_SUCCESS)
            << "a plan was created for a problem it cannot execute";
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        check();
    }
} // namespace

TEST(ElementwisePlanValidationTest, PlansOnlyProblemsAKernelRuns)
{
    ApiObjects api;

    // Rank-1 permutation: B = 2 * A.
    {
        constexpr int64_t  n       = 16;
        auto               desc    = api.tensor({n}, {1}, HIPTENSOR_R_32F);
        int32_t const      modes[] = {'m'};
        std::vector<float> a(n);
        for(int64_t i = 0; i < n; ++i)
        {
            a[i] = static_cast<float>(i - 7);
        }
        DeviceBuffer<float> dA(a), dB(std::vector<float>(n, NAN));

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &op,
                                             desc,
                                             modes,
                                             HIPTENSOR_OP_IDENTITY,
                                             desc,
                                             modes,
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        float const alpha = 2.0f;
        expectRefusedOrCorrect(
            "rank-1 permutation",
            api,
            op,
            [&](hiptensorPlan_t plan) {
                return hiptensorPermute(api.handle, plan, &alpha, dA.ptr, dB.ptr, nullptr);
            },
            [&] {
                auto b = dB.read();
                for(int64_t i = 0; i < n; ++i)
                {
                    EXPECT_EQ(b[i], 2.0f * a[i]) << i;
                }
            });
    }

    // Rank-7 permutation: B = A.
    {
        std::vector<int64_t> lens(7, 2);
        auto                 desc  = api.tensor(lens, {}, HIPTENSOR_R_32F);
        auto                 modes = std::vector<int32_t>{'a', 'b', 'c', 'd', 'e', 'f', 'g'};
        auto                 count = elementCount(lens);
        std::vector<float>   a(count);
        for(int64_t i = 0; i < count; ++i)
        {
            a[i] = static_cast<float>(i % 11 - 5);
        }
        DeviceBuffer<float> dA(a), dB(std::vector<float>(count, NAN));

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &op,
                                             desc,
                                             modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             desc,
                                             modes.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        float const alpha = 1.0f;
        expectRefusedOrCorrect(
            "rank-7 permutation",
            api,
            op,
            [&](hiptensorPlan_t plan) {
                return hiptensorPermute(api.handle, plan, &alpha, dA.ptr, dB.ptr, nullptr);
            },
            [&] { EXPECT_EQ(dB.read(), a); });
    }

    // BF16 binary operation with f32 scalars (a data type combination the API lists): D = A + C.
    {
        constexpr int64_t         M = 8, N = 4;
        auto                      desc    = api.tensor({M, N}, {}, HIPTENSOR_R_16BF);
        int32_t const             modes[] = {'m', 'n'};
        std::vector<hip_bfloat16> a, c;
        for(int64_t i = 0; i < M * N; ++i)
        {
            a.push_back(static_cast<hip_bfloat16>(static_cast<float>(i % 7 - 3)));
            c.push_back(static_cast<hip_bfloat16>(static_cast<float>(i % 5 - 2)));
        }
        DeviceBuffer<hip_bfloat16> dA(a), dC(c),
            dD(std::vector<hip_bfloat16>(M * N, static_cast<hip_bfloat16>(-100.0f)));

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                   &op,
                                                   desc,
                                                   modes,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   desc,
                                                   modes,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   desc,
                                                   modes,
                                                   HIPTENSOR_OP_ADD,
                                                   HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        float const one = 1.0f;
        expectRefusedOrCorrect(
            "bf16 binary",
            api,
            op,
            [&](hiptensorPlan_t plan) {
                return hiptensorElementwiseBinaryExecute(
                    api.handle, plan, &one, dA.ptr, &one, dC.ptr, dD.ptr, nullptr);
            },
            [&] {
                auto d = dD.read();
                for(int64_t i = 0; i < M * N; ++i)
                {
                    EXPECT_EQ(static_cast<float>(d[i]),
                              static_cast<float>(a[i]) + static_cast<float>(c[i]))
                        << i;
                }
            });
    }

    // Permutation into a tensor of a different, unimplemented type (f16 into f64).
    {
        constexpr int64_t     M = 8, N = 4;
        auto                  descA   = api.tensor({M, N}, {}, HIPTENSOR_R_16F);
        auto                  descB   = api.tensor({M, N}, {}, HIPTENSOR_R_64F);
        int32_t const         modes[] = {'m', 'n'};
        std::vector<_Float16> a;
        for(int64_t i = 0; i < M * N; ++i)
        {
            a.push_back(static_cast<_Float16>(i % 9 - 4));
        }
        DeviceBuffer<_Float16> dA(a);
        DeviceBuffer<double>   dB(std::vector<double>(M * N, NAN));

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &op,
                                             descA,
                                             modes,
                                             HIPTENSOR_OP_IDENTITY,
                                             descB,
                                             modes,
                                             HIPTENSOR_COMPUTE_DESC_16F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        _Float16 const alpha = 1.0f;
        expectRefusedOrCorrect(
            "f16 to f64 permutation",
            api,
            op,
            [&](hiptensorPlan_t plan) {
                return hiptensorPermute(api.handle, plan, &alpha, dA.ptr, dB.ptr, nullptr);
            },
            [&] {
                auto b = dB.read();
                for(int64_t i = 0; i < M * N; ++i)
                {
                    EXPECT_EQ(b[i], static_cast<double>(a[i])) << i;
                }
            });
    }
}

TEST(ReductionPlanValidationTest, RefusesMalformedReductionsAtPlanTime)
{
    ApiObjects api;

    auto descA  = api.tensor({4, 3}, {}, HIPTENSOR_R_32F);
    auto descC  = api.tensor({4}, {}, HIPTENSOR_R_32F);
    auto descC5 = api.tensor({5}, {}, HIPTENSOR_R_32F);
    auto descA3 = api.tensor({4, 4, 3}, {}, HIPTENSOR_R_32F);
    auto descCK = api.tensor({3}, {}, HIPTENSOR_R_32F);

    struct Case
    {
        char const*                  name;
        hiptensorTensorDescriptor_t  a;
        std::vector<int32_t>         modeA;
        hiptensorTensorDescriptor_t  c;
        std::vector<int32_t>         modeC, modeD;
        hiptensorComputeDescriptor_t compute;
        hiptensorStatus_t            expected;
    };
    std::vector<Case> cases = {
        {"valid",
         descA,
         {'m', 'k'},
         descC,
         {'m'},
         {'m'},
         HIPTENSOR_COMPUTE_DESC_32F,
         HIPTENSOR_STATUS_SUCCESS},
        {"kept mode with another extent",
         descA,
         {'m', 'k'},
         descC5,
         {'m'},
         {'m'},
         HIPTENSOR_COMPUTE_DESC_32F,
         HIPTENSOR_STATUS_INVALID_VALUE},
        {"mode repeated in A",
         descA3,
         {'m', 'm', 'k'},
         descC,
         {'m'},
         {'m'},
         HIPTENSOR_COMPUTE_DESC_32F,
         HIPTENSOR_STATUS_NOT_SUPPORTED},
        {"D with modes other than C's",
         descA,
         {'m', 'k'},
         descC,
         {'m'},
         {'x'},
         HIPTENSOR_COMPUTE_DESC_32F,
         HIPTENSOR_STATUS_NOT_SUPPORTED},
        {"C mode not in A",
         descA,
         {'m', 'k'},
         descCK,
         {'x'},
         {'x'},
         HIPTENSOR_COMPUTE_DESC_32F,
         HIPTENSOR_STATUS_NOT_SUPPORTED},
        {"f32 data with f64 compute",
         descA,
         {'m', 'k'},
         descC,
         {'m'},
         {'m'},
         HIPTENSOR_COMPUTE_DESC_64F,
         HIPTENSOR_STATUS_NOT_SUPPORTED},
    };

    for(auto const& c : cases)
    {
        hiptensorOperationDescriptor_t op     = nullptr;
        auto                           status = hiptensorCreateReduction(api.handle,
                                               &op,
                                               c.a,
                                               c.modeA.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               c.c,
                                               c.modeC.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               c.c,
                                               c.modeD.data(),
                                               HIPTENSOR_OP_ADD,
                                               c.compute);
        api.keep(op);
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << c.name;
        hiptensorPlan_t plan = nullptr;
        EXPECT_EQ(api.plan(op, plan), c.expected) << c.name;
        if(c.expected != HIPTENSOR_STATUS_SUCCESS)
        {
            EXPECT_EQ(plan, nullptr) << c.name;
        }
    }
}

namespace
{
    // The elementwise data type rows of cuTENSOR 2.x that the API accepts beyond the original
    // Composable Kernel rows: the rocKE backend serves them all. The Composable Kernel backend
    // serves the rows whose tensors share one data type (bf16 permutations, f16 trinary operations
    // with f32 scalars, bf16 trinary operations: `ck`), and declines the others when the plan is
    // created.
    struct ExtendedRow
    {
        char const*         name;
        int                 arity; // 1 permutation, 2 binary, 3 trinary
        hiptensorDataType_t a, c, d; // C: B and C of a trinary operation, C of a binary one
        hiptensorComputeDescriptor_t compute;
        bool                         ck; // served by the Composable Kernel backend too
    };

    std::vector<ExtendedRow> extendedRows()
    {
        auto const f16 = HIPTENSOR_R_16F, bf16 = HIPTENSOR_R_16BF;
        auto const f32 = HIPTENSOR_R_32F, f64 = HIPTENSOR_R_64F;
        return {
            {"bf16 permutation, bf16 scalars",
             1,
             bf16,
             bf16,
             bf16,
             HIPTENSOR_COMPUTE_DESC_16BF,
             true},
            {"bf16 permutation, f32 scalars",
             1,
             bf16,
             bf16,
             bf16,
             HIPTENSOR_COMPUTE_DESC_32F,
             true},
            {"f32 into f64 permutation, f64 scalars",
             1,
             f32,
             f32,
             f64,
             HIPTENSOR_COMPUTE_DESC_64F,
             false},
            {"binary f32 A, f16 C, f32 scalars",
             2,
             f32,
             f16,
             f16,
             HIPTENSOR_COMPUTE_DESC_32F,
             false},
            {"binary f64 A, f32 C, f64 scalars",
             2,
             f64,
             f32,
             f32,
             HIPTENSOR_COMPUTE_DESC_64F,
             false},
            {"f16 trinary, f32 scalars", 3, f16, f16, f16, HIPTENSOR_COMPUTE_DESC_32F, true},
            {"bf16 trinary, bf16 scalars", 3, bf16, bf16, bf16, HIPTENSOR_COMPUTE_DESC_16BF, true},
            {"bf16 trinary, f32 scalars", 3, bf16, bf16, bf16, HIPTENSOR_COMPUTE_DESC_32F, true}};
    }

    size_t bytesOf(hiptensorDataType_t t)
    {
        return t == HIPTENSOR_R_64F ? 8 : t == HIPTENSOR_R_32F ? 4 : 2;
    }

    // Values of type `t` (exactly representable ones) as bytes, and back.
    std::vector<uint8_t> encode(std::vector<double> const& values, hiptensorDataType_t t)
    {
        std::vector<uint8_t> out(values.size() * bytesOf(t));
        for(size_t i = 0; i < values.size(); ++i)
        {
            auto* p = out.data() + i * bytesOf(t);
            if(t == HIPTENSOR_R_16F)
            {
                auto v = static_cast<_Float16>(values[i]);
                std::memcpy(p, &v, 2);
            }
            else if(t == HIPTENSOR_R_16BF)
            {
                auto v = static_cast<hip_bfloat16>(static_cast<float>(values[i]));
                std::memcpy(p, &v, 2);
            }
            else if(t == HIPTENSOR_R_32F)
            {
                auto v = static_cast<float>(values[i]);
                std::memcpy(p, &v, 4);
            }
            else
            {
                std::memcpy(p, &values[i], 8);
            }
        }
        return out;
    }

    std::vector<double> decode(std::vector<uint8_t> const& bytes, hiptensorDataType_t t)
    {
        std::vector<double> out(bytes.size() / bytesOf(t));
        for(size_t i = 0; i < out.size(); ++i)
        {
            auto const* p = bytes.data() + i * bytesOf(t);
            if(t == HIPTENSOR_R_16F)
            {
                _Float16 v;
                std::memcpy(&v, p, 2);
                out[i] = static_cast<double>(v);
            }
            else if(t == HIPTENSOR_R_16BF)
            {
                hip_bfloat16 v;
                std::memcpy(&v, p, 2);
                out[i] = static_cast<double>(static_cast<float>(v));
            }
            else if(t == HIPTENSOR_R_32F)
            {
                float v;
                std::memcpy(&v, p, 4);
                out[i] = v;
            }
            else
            {
                std::memcpy(&out[i], p, 8);
            }
        }
        return out;
    }

    struct RowOutcome
    {
        hiptensorStatus_t estimate = HIPTENSOR_STATUS_SUCCESS;
        hiptensorStatus_t plan     = HIPTENSOR_STATUS_SUCCESS;
    };

    // Estimates and plans `row` on a 6 x 5 problem (A stored transposed, the other inputs and D
    // column-major) with the backends the environment selects. A plan is executed with
    // alpha = 2, beta = -1, gamma = 0.5 and small integers and D is compared exactly with a loop:
    // D = alpha A, alpha A + gamma C, or alpha A + beta B + gamma C.
    RowOutcome planAndRun(ExtendedRow const& row)
    {
        RowOutcome out;
        ApiObjects api;
        if(!api.supports(row.a) || !api.supports(row.c) || !api.supports(row.d))
        {
            out.estimate = out.plan = HIPTENSOR_STATUS_NOT_SUPPORTED;
            return out;
        }
        constexpr int64_t M = 6, N = 5;
        int32_t const     modeAt[] = {'n', 'm'}, modes[] = {'m', 'n'};
        auto              dA = api.tensor({N, M}, {}, row.a);
        auto              dC = api.tensor({M, N}, {}, row.c);
        auto              dD = api.tensor({M, N}, {}, row.d);

        hiptensorOperationDescriptor_t op = nullptr;
        auto const                     id = HIPTENSOR_OP_IDENTITY;
        hiptensorStatus_t              created;
        if(row.arity == 1)
        {
            created = hiptensorCreatePermutation(
                api.handle, &op, dA, modeAt, id, dD, modes, row.compute);
        }
        else if(row.arity == 2)
        {
            created = hiptensorCreateElementwiseBinary(api.handle,
                                                       &op,
                                                       dA,
                                                       modeAt,
                                                       id,
                                                       dC,
                                                       modes,
                                                       id,
                                                       dD,
                                                       modes,
                                                       HIPTENSOR_OP_ADD,
                                                       row.compute);
        }
        else
        {
            created = hiptensorCreateElementwiseTrinary(api.handle,
                                                        &op,
                                                        dA,
                                                        modeAt,
                                                        id,
                                                        dC,
                                                        modes,
                                                        id,
                                                        dC,
                                                        modes,
                                                        id,
                                                        dD,
                                                        modes,
                                                        HIPTENSOR_OP_ADD,
                                                        HIPTENSOR_OP_ADD,
                                                        row.compute);
        }
        EXPECT_EQ(created, HIPTENSOR_STATUS_SUCCESS) << row.name;
        api.keep(op);

        uint64_t workspace = 0;
        out.estimate       = hiptensorEstimateWorkspaceSize(
            api.handle, op, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &workspace);
        hiptensorPlan_t plan = nullptr;
        out.plan             = api.plan(op, plan, workspace);
        if(out.plan != HIPTENSOR_STATUS_SUCCESS)
        {
            EXPECT_EQ(plan, nullptr) << row.name;
            return out;
        }
        EXPECT_EQ(out.estimate, HIPTENSOR_STATUS_SUCCESS) << row.name;

        // A(n, m), B(m, n), C(m, n); D(m, n) starts as a value no result has
        std::vector<double> a(M * N), b(M * N), c(M * N);
        for(int64_t i = 0; i < M * N; ++i)
        {
            a[i] = double(i % 7 - 3);
            b[i] = double(i % 5 - 2);
            c[i] = double(i % 9 - 4);
        }
        DeviceBuffer<uint8_t> devA(encode(a, row.a)), devB(encode(b, row.c)),
            devC(encode(c, row.c)), devD(encode(std::vector<double>(M * N, -77.0), row.d));
        uint8_t      scalars[3][8] = {};
        double const values[3]     = {2.0, -1.0, 0.5};
        auto const   scalarType    = row.compute == HIPTENSOR_COMPUTE_DESC_64F    ? HIPTENSOR_R_64F
                                     : row.compute == HIPTENSOR_COMPUTE_DESC_16BF ? HIPTENSOR_R_16BF
                                     : row.compute == HIPTENSOR_COMPUTE_DESC_16F  ? HIPTENSOR_R_16F
                                                                                  : HIPTENSOR_R_32F;
        for(int s = 0; s < 3; ++s)
        {
            auto bytes = encode({values[s]}, scalarType);
            std::memcpy(scalars[s], bytes.data(), bytes.size());
        }
        hiptensorStatus_t executed;
        if(row.arity == 1)
        {
            executed = hiptensorPermute(api.handle, plan, scalars[0], devA.ptr, devD.ptr, nullptr);
        }
        else if(row.arity == 2)
        {
            executed = hiptensorElementwiseBinaryExecute(
                api.handle, plan, scalars[0], devA.ptr, scalars[2], devC.ptr, devD.ptr, nullptr);
        }
        else
        {
            executed = hiptensorElementwiseTrinaryExecute(api.handle,
                                                          plan,
                                                          scalars[0],
                                                          devA.ptr,
                                                          scalars[1],
                                                          devB.ptr,
                                                          scalars[2],
                                                          devC.ptr,
                                                          devD.ptr,
                                                          nullptr);
        }
        EXPECT_EQ(executed, HIPTENSOR_STATUS_SUCCESS) << row.name;
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        auto d = decode(devD.read(), row.d);
        for(int64_t m = 0; m < M; ++m)
        {
            for(int64_t n = 0; n < N; ++n)
            {
                double want = values[0] * a[n + N * m];
                if(row.arity == 3)
                {
                    want += values[1] * b[m + M * n];
                }
                if(row.arity >= 2)
                {
                    want += values[2] * c[m + M * n];
                }
                EXPECT_EQ(d[m + M * n], want) << row.name << " at (" << m << ", " << n << ")";
            }
        }
        return out;
    }
} // namespace

// The Composable Kernel backend declines the rows that only the rocKE backend serves when the plan
// is created, and serves the rows of one data type with the loop's result: with the default
// selection and with HIPTENSOR_BACKEND=ck, with and without the fallback (which only falls back to
// the Composable Kernel backend).
TEST(ElementwiseTypeRowsTest, CkBackendDeclinesTheRocKERows)
{
    for(char const* selection : {static_cast<char const*>(nullptr), "ck"})
    {
        for(char const* fallback : {static_cast<char const*>(nullptr), "0"})
        {
            hiptensor_test::ScopedBackend             backend(selection);
            hiptensor_test::ScopedEnvironmentVariable noFallback("HIPTENSOR_BACKEND_FALLBACK",
                                                                 fallback);
            for(auto const& row : extendedRows())
            {
                SCOPED_TRACE(std::string(selection == nullptr ? "default selection" : selection)
                             + (fallback == nullptr ? ", fallback" : ", no fallback"));
                auto out = planAndRun(row);
                EXPECT_EQ(out.plan,
                          row.ck ? HIPTENSOR_STATUS_SUCCESS : HIPTENSOR_STATUS_NOT_SUPPORTED)
                    << row.name;
            }
        }
    }
}

// With HIPTENSOR_BACKEND=rocke, the rows are either served with the loop's result (a library built
// with the rocKE backend, with a kernel pack for the device) or declined when the plan is created
// (no rocKE backend, or no pack for the device; the fallback backend declines them too); never a
// plan that fails or computes another result.
TEST(ElementwiseTypeRowsTest, RocKERowsServedExactlyOrDeclined)
{
    for(char const* fallback : {static_cast<char const*>(nullptr), "0"})
    {
        hiptensor_test::ScopedBackend             backend("rocke");
        hiptensor_test::ScopedEnvironmentVariable noFallback("HIPTENSOR_BACKEND_FALLBACK",
                                                             fallback);
        for(auto const& row : extendedRows())
        {
            SCOPED_TRACE(fallback == nullptr ? "fallback" : "no fallback");
            auto out = planAndRun(row);
            if(out.plan == HIPTENSOR_STATUS_SUCCESS)
            {
                EXPECT_EQ(out.estimate, HIPTENSOR_STATUS_SUCCESS) << row.name;
                std::cout << row.name << ": planned and executed" << std::endl;
                continue;
            }
            EXPECT_TRUE(out.plan == HIPTENSOR_STATUS_NOT_SUPPORTED
                        || out.plan == HIPTENSOR_STATUS_ARCH_MISMATCH)
                << row.name << ": " << hiptensorGetErrorString(out.plan);
            std::cout << row.name << ": refused when the plan is created" << std::endl;
        }
    }
}
