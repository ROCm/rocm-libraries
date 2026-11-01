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

// A trinary contraction E = alpha * A * B * C + beta * D runs as T = A * B, then
// E = alpha * T * C + beta * D, with T held in the caller's workspace. These tests fill the
// workspace with NaN before every execution: its initial contents must not reach E.
//
// The expected values come from a plain loop over every index with 64-bit offsets, not from
// hipTensor's CPU reference. All inputs are small integers and alpha/beta are integers, so
// every product and partial sum is exact in f32 and the comparison is bit-exact.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    constexpr float    alphaValue = 2.0f;
    constexpr float    betaValue  = -1.0f;
    constexpr float    sentinel   = -1000.0f;
    constexpr uint32_t alignment  = 4;

    // E[m0,m1,s0,s1] = sum over k0, k1, n0, n1 of
    //                  A[m0,m1,k0,k1] * B[n0,n1,k0,k1] * C[s0,s1,n0,n1]
    constexpr int64_t extentM0 = 5;
    constexpr int64_t extentM1 = 3;
    constexpr int64_t extentK0 = 3;
    constexpr int64_t extentK1 = 4;
    constexpr int64_t extentN0 = 4;
    constexpr int64_t extentN1 = 2;
    constexpr int64_t extentS0 = 3;
    constexpr int64_t extentS1 = 5;

    // Packed column-major (first mode fastest) tensor of rank 4.
    struct HostTensor
    {
        std::vector<int32_t> modes;
        std::vector<int64_t> extents;
        std::vector<int64_t> strides;
        std::vector<float>   data;
    };

    HostTensor makeTensor(std::vector<int32_t> const& modes, std::vector<int64_t> const& extents)
    {
        HostTensor t;
        t.modes          = modes;
        t.extents        = extents;
        int64_t elements = 1;
        for(auto extent : extents)
        {
            t.strides.push_back(elements);
            elements *= extent;
        }
        t.data.assign(static_cast<std::size_t>(elements), sentinel);
        return t;
    }

    void fillRandomIntegers(HostTensor& t, int lo, int hi, unsigned seed)
    {
        std::mt19937                       gen(seed);
        std::uniform_int_distribution<int> dist(lo, hi);
        for(auto& v : t.data)
        {
            v = static_cast<float>(dist(gen));
        }
    }

    int64_t offsetOf(HostTensor const& t, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
    {
        return i0 * t.strides[0] + i1 * t.strides[1] + i2 * t.strides[2] + i3 * t.strides[3];
    }

    double at(HostTensor const& t, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
    {
        return static_cast<double>(t.data[static_cast<std::size_t>(offsetOf(t, i0, i1, i2, i3))]);
    }

    // Independent reference for E. D is ignored when hasD is false.
    std::vector<float> reference(HostTensor const& a,
                                 HostTensor const& b,
                                 HostTensor const& c,
                                 HostTensor const& d,
                                 HostTensor const& e,
                                 bool              hasD)
    {
        std::vector<float> expected(e.data.size(), sentinel);
        for(int64_t s1 = 0; s1 < extentS1; s1++)
        {
            for(int64_t s0 = 0; s0 < extentS0; s0++)
            {
                for(int64_t m1 = 0; m1 < extentM1; m1++)
                {
                    for(int64_t m0 = 0; m0 < extentM0; m0++)
                    {
                        double acc = 0.0;
                        for(int64_t n1 = 0; n1 < extentN1; n1++)
                        {
                            for(int64_t n0 = 0; n0 < extentN0; n0++)
                            {
                                for(int64_t k1 = 0; k1 < extentK1; k1++)
                                {
                                    for(int64_t k0 = 0; k0 < extentK0; k0++)
                                    {
                                        acc += at(a, m0, m1, k0, k1) * at(b, n0, n1, k0, k1)
                                               * at(c, s0, s1, n0, n1);
                                    }
                                }
                            }
                        }
                        double value = alphaValue * acc;
                        if(hasD)
                        {
                            value += betaValue * at(d, m0, m1, s0, s1);
                        }
                        auto offset      = static_cast<std::size_t>(offsetOf(e, m0, m1, s0, s1));
                        expected[offset] = static_cast<float>(value);
                    }
                }
            }
        }
        return expected;
    }

    // ---- RAII guards: cleanup runs on every exit path, including a failed ASSERT --

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

    hiptensorStatus_t
        createDesc(hiptensorHandle_t handle, TensorDescGuard& guard, HostTensor const& t)
    {
        return hiptensorCreateTensorDescriptor(handle,
                                               &guard.desc,
                                               t.modes.size(),
                                               t.extents.data(),
                                               t.strides.data(),
                                               HIPTENSOR_R_32F,
                                               alignment);
    }

    void upload(DeviceBuffer& buffer, std::vector<float> const& host)
    {
        ASSERT_EQ(hipMalloc(&buffer.ptr, host.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(buffer.ptr, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
    }

    // Bit-exact comparison of every element. A NaN never compares equal, so it is a mismatch.
    void expectMatches(DeviceBuffer const&       device,
                       std::vector<float> const& expected,
                       std::string const&        name)
    {
        std::vector<float> result(expected.size());
        ASSERT_EQ(
            hipMemcpy(
                result.data(), device.ptr, result.size() * sizeof(float), hipMemcpyDeviceToHost),
            hipSuccess);

        std::size_t        mismatches = 0;
        std::ostringstream firstMismatches;
        for(std::size_t i = 0; i < result.size(); i++)
        {
            if(!(result[i] == expected[i]))
            {
                if(mismatches < 4)
                {
                    firstMismatches << " [offset " << i << ": got " << result[i] << ", expected "
                                    << expected[i] << "]";
                }
                mismatches++;
            }
        }
        EXPECT_EQ(mismatches, 0u) << name << ":" << firstMismatches.str();
    }

    struct WorkspaceCase
    {
        bool                hasD; // bilinear (with D) or scale (without D)
        hiptensorAlgo_t     algorithm;
        hiptensorOperator_t opA;
    };

    std::string caseName(WorkspaceCase const& c)
    {
        std::string name = c.hasD ? "Bilinear" : "Scale";
        name += (c.algorithm == HIPTENSOR_ALGO_ACTOR_CRITIC) ? "_ActorCritic" : "_Default";
        if(c.opA != HIPTENSOR_OP_IDENTITY)
        {
            name += "_UnaryOp";
        }
        return name;
    }

    void runAndCheck(WorkspaceCase const& tc)
    {
        int32_t const m0 = 'a', m1 = 'b', k0 = 'x', k1 = 'y';
        int32_t const n0 = 'p', n1 = 'q', s0 = 's', s1 = 't';

        auto a = makeTensor({m0, m1, k0, k1}, {extentM0, extentM1, extentK0, extentK1});
        auto b = makeTensor({n0, n1, k0, k1}, {extentN0, extentN1, extentK0, extentK1});
        auto c = makeTensor({s0, s1, n0, n1}, {extentS0, extentS1, extentN0, extentN1});
        auto d = makeTensor({m0, m1, s0, s1}, {extentM0, extentM1, extentS0, extentS1});
        auto e = makeTensor({m0, m1, s0, s1}, {extentM0, extentM1, extentS0, extentS1});
        // A is non-negative, so HIPTENSOR_OP_RELU on A leaves the expected values unchanged.
        fillRandomIntegers(a, 0, 3, 11u);
        fillRandomIntegers(b, -3, 3, 12u);
        fillRandomIntegers(c, -3, 3, 13u);
        fillRandomIntegers(d, -4, 4, 14u);

        auto const expected = reference(a, b, c, d, e, tc.hasD);

        HandleGuard handle;
        ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descC, descD, descE;
        ASSERT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descC, c), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descD, d), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descE, e), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        auto        status = hiptensorCreateContractionTrinary(handle.handle,
                                                        &op.desc,
                                                        descA.desc,
                                                        a.modes.data(),
                                                        tc.opA,
                                                        descB.desc,
                                                        b.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descC.desc,
                                                        c.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        tc.hasD ? descD.desc : nullptr,
                                                        tc.hasD ? d.modes.data() : nullptr,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descE.desc,
                                                        e.modes.data(),
                                                        HIPTENSOR_COMPUTE_DESC_32F);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "This contraction is not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

        PlanPrefGuard pref;
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      handle.handle, &pref.pref, tc.algorithm, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t workspaceSize = 0;
        ASSERT_EQ(
            hiptensorEstimateWorkspaceSize(
                handle.handle, op.desc, pref.pref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
            HIPTENSOR_STATUS_SUCCESS);

        PlanGuard plan;
        ASSERT_EQ(hiptensorCreatePlan(handle.handle, &plan.plan, op.desc, pref.pref, workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS);

        DeviceBuffer devA, devB, devC, devD, devE, workspace;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a.data));
        ASSERT_NO_FATAL_FAILURE(upload(devB, b.data));
        ASSERT_NO_FATAL_FAILURE(upload(devC, c.data));
        ASSERT_NO_FATAL_FAILURE(upload(devD, d.data));
        ASSERT_NO_FATAL_FAILURE(upload(devE, e.data)); // all sentinel
        ASSERT_GT(workspaceSize, 0u); // holds the intermediate T
        ASSERT_EQ(hipMalloc(&workspace.ptr, workspaceSize), hipSuccess);

        // Twice on the same plan and workspace, refilled with NaN before each execution.
        for(int run = 0; run < 2; run++)
        {
            // All bits set is a NaN in every floating-point type.
            ASSERT_EQ(hipMemset(workspace.ptr, 0xFF, workspaceSize), hipSuccess);

            ASSERT_EQ(hiptensorContractTrinary(handle.handle,
                                               plan.plan,
                                               &alphaValue,
                                               devA.ptr,
                                               devB.ptr,
                                               devC.ptr,
                                               tc.hasD ? &betaValue : nullptr,
                                               tc.hasD ? devD.ptr : nullptr,
                                               devE.ptr,
                                               workspace.ptr,
                                               workspaceSize,
                                               0 /* stream */),
                      HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            ASSERT_NO_FATAL_FAILURE(
                expectMatches(devE, expected, caseName(tc) + " run " + std::to_string(run)));
        }
    }

    class TrinaryContractionWorkspaceTest : public ::testing::TestWithParam<WorkspaceCase>
    {
    };

} // namespace

TEST_P(TrinaryContractionWorkspaceTest, NaNWorkspaceDoesNotReachResult)
{
    runAndCheck(GetParam());
}

// Bilinear trinary contractions run their first step with the bilinear kernel, which reads
// the workspace. The scale case reads no bias and is a control.
INSTANTIATE_TEST_SUITE_P(
    TrinaryContractionWorkspace,
    TrinaryContractionWorkspaceTest,
    ::testing::Values(WorkspaceCase{true, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_OP_IDENTITY},
                      WorkspaceCase{true, HIPTENSOR_ALGO_ACTOR_CRITIC, HIPTENSOR_OP_IDENTITY},
                      WorkspaceCase{true, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_OP_RELU},
                      WorkspaceCase{false, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_OP_IDENTITY}),
    [](::testing::TestParamInfo<WorkspaceCase> const& info) { return caseName(info.param); });
