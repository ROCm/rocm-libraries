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

// Regression tests for zero scaling factors in the elementwise operations. As in cuTENSOR, an
// input whose scaling factor is zero is not read and its operator is not applied, so NaN or Inf
// in it, or produced by its operator (LOG(0), LOG(-1)), must not reach the output; it may even be
// nullptr. The library used to compute 0 * op(x), which is NaN for such inputs.
// The expected results come from plain loops here; the finite inputs are small integers and the
// scalars dyadic, so the results are exact.

#include "api_test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using hiptensor_test::ApiObjects;
using hiptensor_test::DeviceBuffer;

namespace
{
    constexpr int64_t M = 16, N = 12, count = M * N;

    double binaryOp(hiptensorOperator_t op, double x, double y)
    {
        switch(op)
        {
        case HIPTENSOR_OP_ADD:
            return x + y;
        case HIPTENSOR_OP_MUL:
            return x * y;
        case HIPTENSOR_OP_MAX:
            return std::max(x, y);
        default:
            return std::min(x, y);
        }
    }

    // Zero, negative and non-finite values: LOG gives -Inf and NaN, and the values themselves
    // are NaN and Inf.
    template <typename T>
    std::vector<T> poisoned()
    {
        std::vector<T> v(count);
        for(int64_t i = 0; i < count; ++i)
        {
            switch(i % 4)
            {
            case 0:
                v[i] = T(0);
                break;
            case 1:
                v[i] = T(-1);
                break;
            case 2:
                v[i] = std::numeric_limits<T>::quiet_NaN();
                break;
            default:
                v[i] = std::numeric_limits<T>::infinity();
                break;
            }
        }
        return v;
    }

    template <typename T>
    std::vector<T> integers(int period, int offset)
    {
        std::vector<T> v(count);
        for(int64_t i = 0; i < count; ++i)
        {
            v[i] = static_cast<T>(i % period - offset);
        }
        return v;
    }

    template <typename T>
    void expectEqual(std::vector<T> const& actual, std::vector<double> const& expected)
    {
        for(int64_t i = 0; i < count; ++i)
        {
            EXPECT_EQ(static_cast<double>(actual[i]), expected[i]) << "element " << i;
        }
    }

    template <typename T>
    constexpr hiptensorDataType_t dataType()
    {
        return std::is_same_v<T, double> ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
    }
    template <typename T>
    constexpr hiptensorComputeDescriptor_t computeType()
    {
        return std::is_same_v<T, double> ? HIPTENSOR_COMPUTE_DESC_64F
                                         : HIPTENSOR_COMPUTE_DESC_32F;
    }

    template <typename T>
    void binaryZeroScales()
    {
        ApiObjects api;
        if(!api.supports(dataType<T>()))
        {
            GTEST_SKIP() << "Data type not supported on this device";
        }
        auto          desc    = api.tensor({M, N}, {1, M}, dataType<T>());
        int32_t const modes[] = {'m', 'n'};

        auto const goodC = integers<T>(5, 2);
        auto const bad   = poisoned<T>();

        for(auto opAC : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX})
        {
            hiptensorOperationDescriptor_t op = nullptr;
            ASSERT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                       &op,
                                                       desc,
                                                       modes,
                                                       HIPTENSOR_OP_LOG,
                                                       desc,
                                                       modes,
                                                       HIPTENSOR_OP_IDENTITY,
                                                       desc,
                                                       modes,
                                                       opAC,
                                                       computeType<T>()),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(op);
            hiptensorPlan_t plan = nullptr;
            ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

            DeviceBuffer<T> dGoodC(goodC), dBad(bad);
            DeviceBuffer<T> dD(std::vector<T>(count, std::numeric_limits<T>::quiet_NaN()));

            T const zero = 0, scale = 0.75, one = 1;

            // alpha == 0: D = opAC(0, 0.75 * C) with A poisoned (LOG(A) is -Inf or NaN), then
            // with A = nullptr.
            for(T const* aPtr : {static_cast<T const*>(dBad.ptr), static_cast<T const*>(nullptr)})
            {
                dD.write(std::vector<T>(count, std::numeric_limits<T>::quiet_NaN()));
                ASSERT_EQ(hiptensorElementwiseBinaryExecute(api.handle,
                                                            plan,
                                                            &zero,
                                                            aPtr,
                                                            &scale,
                                                            dGoodC.ptr,
                                                            dD.ptr,
                                                            nullptr),
                          HIPTENSOR_STATUS_SUCCESS)
                    << "opAC " << opAC;
                CHECK_HIP_ERROR(hipDeviceSynchronize());
                std::vector<double> expected(count);
                for(int64_t i = 0; i < count; ++i)
                {
                    expected[i] = binaryOp(opAC, 0.0, 0.75 * double(goodC[i]));
                }
                SCOPED_TRACE(aPtr == nullptr ? "A = nullptr" : "A poisoned");
                expectEqual(dD.read(), expected);
            }

            // gamma == 0: D = opAC(alpha * LOG(A), 0) with C poisoned, then without C. A holds
            // powers of two so LOG is finite.
            std::vector<T> powers(count);
            for(int64_t i = 0; i < count; ++i)
            {
                powers[i] = static_cast<T>(1 << (i % 5));
            }
            DeviceBuffer<T> dPowers(powers);
            for(T const* cPtr : {static_cast<T const*>(dBad.ptr), static_cast<T const*>(nullptr)})
            {
                dD.write(std::vector<T>(count, std::numeric_limits<T>::quiet_NaN()));
                ASSERT_EQ(hiptensorElementwiseBinaryExecute(
                              api.handle, plan, &one, dPowers.ptr, &zero, cPtr, dD.ptr, nullptr),
                          HIPTENSOR_STATUS_SUCCESS);
                CHECK_HIP_ERROR(hipDeviceSynchronize());
                auto actual = dD.read();
                for(int64_t i = 0; i < count; ++i)
                {
                    double expected = binaryOp(opAC, std::log(double(powers[i])), 0.0);
                    EXPECT_NEAR(double(actual[i]), expected, 1e-6) << i;
                }
            }

            // Both zero: D = opAC(0, 0) = 0, nothing read; D held NaN before.
            dD.write(std::vector<T>(count, std::numeric_limits<T>::quiet_NaN()));
            ASSERT_EQ(hiptensorElementwiseBinaryExecute(
                          api.handle, plan, &zero, nullptr, &zero, nullptr, dD.ptr, nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            expectEqual(dD.read(), std::vector<double>(count, 0.0));

            // A non-zero scale with a null input is still an error.
            EXPECT_EQ(hiptensorElementwiseBinaryExecute(
                          api.handle, plan, &one, nullptr, &one, dGoodC.ptr, dD.ptr, nullptr),
                      HIPTENSOR_STATUS_NOT_INITIALIZED);
        }
    }
} // namespace

TEST(ElementwiseZeroScaleTest, BinaryF32)
{
    binaryZeroScales<float>();
}

TEST(ElementwiseZeroScaleTest, BinaryF64)
{
    binaryZeroScales<double>();
}

// D = alpha * A + beta * B + gamma * C with each scale zero in turn and that input poisoned;
// the same plan is reused across the calls, also with every scale non-zero.
TEST(ElementwiseZeroScaleTest, TrinaryF32ReusedPlan)
{
    ApiObjects    api;
    auto          desc    = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    int32_t const modes[] = {'m', 'n'};

    hiptensorOperationDescriptor_t op = nullptr;
    ASSERT_EQ(hiptensorCreateElementwiseTrinary(api.handle,
                                                &op,
                                                desc,
                                                modes,
                                                HIPTENSOR_OP_IDENTITY,
                                                desc,
                                                modes,
                                                HIPTENSOR_OP_IDENTITY,
                                                desc,
                                                modes,
                                                HIPTENSOR_OP_IDENTITY,
                                                desc,
                                                modes,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    api.keep(op);
    hiptensorPlan_t plan = nullptr;
    ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

    std::vector<std::vector<float>> good
        = {integers<float>(7, 3), integers<float>(5, 2), integers<float>(9, 4)};
    DeviceBuffer<float> d0(good[0]), d1(good[1]), d2(good[2]), dBad(poisoned<float>());
    DeviceBuffer<float> dD(std::vector<float>(count, NAN));
    float const         scales[] = {1.25f, 0.5f, 0.75f};

    for(int zeroed = -1; zeroed < 3; ++zeroed)
    {
        float s[3];
        void const* in[3] = {d0.ptr, d1.ptr, d2.ptr};
        for(int k = 0; k < 3; ++k)
        {
            s[k] = (k == zeroed) ? 0.0f : scales[k];
            if(k == zeroed)
            {
                in[k] = dBad.ptr;
            }
        }
        dD.write(std::vector<float>(count, NAN));
        ASSERT_EQ(hiptensorElementwiseTrinaryExecute(api.handle,
                                                     plan,
                                                     &s[0],
                                                     in[0],
                                                     &s[1],
                                                     in[1],
                                                     &s[2],
                                                     in[2],
                                                     dD.ptr,
                                                     nullptr),
                  HIPTENSOR_STATUS_SUCCESS)
            << "zeroed " << zeroed;
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        std::vector<double> expected(count);
        for(int64_t i = 0; i < count; ++i)
        {
            expected[i] = 0;
            for(int k = 0; k < 3; ++k)
            {
                expected[i] += (k == zeroed) ? 0.0 : double(scales[k]) * good[k][i];
            }
        }
        SCOPED_TRACE(zeroed);
        expectEqual(dD.read(), expected);
    }
}

// B = alpha * opA(A): alpha == 0 gives zeros without reading A, also after the plan ran with
// alpha == 1 (where a pass-through kernel may be selected) and before it runs with 2 again.
TEST(ElementwiseZeroScaleTest, PermutationReusedPlan)
{
    for(auto type : {HIPTENSOR_R_32F, HIPTENSOR_R_64F})
    {
        ApiObjects api;
        if(!api.supports(type))
        {
            continue;
        }
        auto          descA   = api.tensor({M, N}, {1, M}, type);
        auto          descB   = api.tensor({N, M}, {1, N}, type);
        int32_t const modesA[] = {'m', 'n'}, modesB[] = {'n', 'm'};
        auto          compute
            = type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &op,
                                             descA,
                                             modesA,
                                             HIPTENSOR_OP_IDENTITY,
                                             descB,
                                             modesB,
                                             compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        hiptensorPlan_t plan = nullptr;
        ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

        auto run = [&](auto tag) {
            using T            = decltype(tag);
            auto const      a  = integers<T>(7, 3);
            DeviceBuffer<T> dA(a), dBad(poisoned<T>());
            DeviceBuffer<T> dB(std::vector<T>(count, std::numeric_limits<T>::quiet_NaN()));
            // With alpha == 0, the first call passes the poisoned A and the second none.
            int zeroCalls = 0;
            for(T alpha : {T(1), T(0), T(2), T(0)})
            {
                T const* aPtr = dA.ptr;
                if(alpha == T(0))
                {
                    aPtr = (zeroCalls++ == 0) ? dBad.ptr : nullptr;
                }
                dB.write(std::vector<T>(count, std::numeric_limits<T>::quiet_NaN()));
                ASSERT_EQ(hiptensorPermute(api.handle, plan, &alpha, aPtr, dB.ptr, nullptr),
                          HIPTENSOR_STATUS_SUCCESS)
                    << "alpha " << double(alpha);
                CHECK_HIP_ERROR(hipDeviceSynchronize());
                std::vector<double> expected(count);
                for(int64_t n = 0; n < N; ++n)
                {
                    for(int64_t m = 0; m < M; ++m)
                    {
                        expected[n + N * m] = double(alpha) * double(a[m + M * n]);
                    }
                }
                SCOPED_TRACE(double(alpha));
                expectEqual(dB.read(), expected);
            }
        };
        if(type == HIPTENSOR_R_64F)
        {
            run(double{});
        }
        else
        {
            run(float{});
        }
    }
}

// A carries fewer modes than D (it is broadcast along n): with alpha == 0 it is not read.
TEST(ElementwiseZeroScaleTest, BroadcastInputWithZeroScale)
{
    ApiObjects    api;
    auto          descA    = api.tensor({M}, {1}, HIPTENSOR_R_32F);
    auto          descCD   = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    int32_t const modesA[] = {'m'}, modesCD[] = {'m', 'n'};

    hiptensorOperationDescriptor_t op = nullptr;
    ASSERT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                               &op,
                                               descA,
                                               modesA,
                                               HIPTENSOR_OP_SQRT,
                                               descCD,
                                               modesCD,
                                               HIPTENSOR_OP_IDENTITY,
                                               descCD,
                                               modesCD,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    api.keep(op);
    hiptensorPlan_t plan = nullptr;
    ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

    auto const          c = integers<float>(5, 2);
    DeviceBuffer<float> dA(std::vector<float>(M, -1.0f)), dC(c), dD(std::vector<float>(count, NAN));
    float const         zero = 0.0f, gamma = 0.5f;
    ASSERT_EQ(hiptensorElementwiseBinaryExecute(
                  api.handle, plan, &zero, dA.ptr, &gamma, dC.ptr, dD.ptr, nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    std::vector<double> expected(count);
    for(int64_t i = 0; i < count; ++i)
    {
        expected[i] = 0.5 * c[i];
    }
    expectEqual(dD.read(), expected);
}
