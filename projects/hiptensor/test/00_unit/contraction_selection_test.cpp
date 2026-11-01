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

// Kernel selection with HIPTENSOR_ALGO_ACTOR_CRITIC:
//  - f32 contractions with f32 and with bf16 compute are planned and run correctly;
//  - a problem the actor-critic table names no kernel for is planned among the kernels that can
//    run it, instead of failing;
//  - the kernel the actor-critic table names is only kept if it can run the problem within the
//    workspace limit, and, for a trinary contraction, both of its steps; otherwise the fastest
//    kernel that can is selected.
// The ActorCriticRowMajorTableTest cases need the tables for row-major strides, which name
// kernels that cannot run the column-major problems below. They run in the ctest entry
// contraction_selection_test_row_major_tables, which sets HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=OFF,
// and are skipped otherwise. The ActorCriticWinnerTest cases use mock kernels, so that which
// kernels can run the problem is known.
// Expected outputs come from a plain loop with 64-bit offsets, not from the CPU reference. The
// data are small integers and every partial sum is an integer of magnitude at most 2^11, so the
// results are exact, also with bf16 compute, and are compared bit for bit.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ck_tensor_limits.hpp"
#include "contraction/contraction_selection.hpp"
#include "data_types.hpp"
#include "hiptensor_options.hpp"

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

    char const* const rowMajorTablesOnly = "Runs only with the row-major actor-critic tables "
                                           "(HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=OFF)";

    bool usesRowMajorTables()
    {
        return !hiptensor::HiptensorOptions::instance()->isColMajorStrides();
    }

    // A packed column-major f32 tensor whose modes are stored in the given order.
    struct Tensor
    {
        std::vector<int32_t> modes;
        std::vector<int64_t> lengths;
        std::vector<int64_t> strides;
        std::vector<float>   data;
    };

    // Values in [-3, 3].
    Tensor
        makeTensor(std::vector<int32_t> const& modes, std::vector<int64_t> const& lengths, int seed)
    {
        Tensor t;
        t.modes   = modes;
        t.lengths = lengths;
        t.strides.assign(lengths.size(), 1);
        for(std::size_t i = 1; i < lengths.size(); i++)
        {
            t.strides[i] = t.strides[i - 1] * lengths[i - 1];
        }

        int64_t elements = 1;
        for(auto length : lengths)
        {
            elements *= length;
        }
        t.data.resize(static_cast<std::size_t>(elements));
        for(std::size_t i = 0; i < t.data.size(); i++)
        {
            t.data[i] = static_cast<float>(static_cast<int>((i * 5 + seed) % 7) - 3);
        }
        return t;
    }

    int64_t lengthOf(std::vector<Tensor const*> const& tensors, int32_t mode)
    {
        for(auto const* t : tensors)
        {
            auto it = std::find(t->modes.cbegin(), t->modes.cend(), mode);
            if(it != t->modes.cend())
            {
                return t->lengths[std::distance(t->modes.cbegin(), it)];
            }
        }
        return 1;
    }

    int64_t offsetOf(Tensor const&               t,
                     std::vector<int32_t> const& modes,
                     std::vector<int64_t> const& idx)
    {
        int64_t offset = 0;
        for(std::size_t i = 0; i < t.modes.size(); i++)
        {
            auto it  = std::find(modes.cbegin(), modes.cend(), t.modes[i]);
            auto pos = static_cast<std::size_t>(std::distance(modes.cbegin(), it));
            offset += idx[pos] * t.strides[i];
        }
        return offset;
    }

    // out = alpha * (product of the inputs, summed over every mode out does not carry).
    std::vector<float> contractionReference(std::vector<Tensor const*> const& inputs,
                                            Tensor const&                     out,
                                            float                             alpha)
    {
        // Every mode once, the output's first.
        std::vector<int32_t> modes = out.modes;
        for(auto const* t : inputs)
        {
            for(auto mode : t->modes)
            {
                if(std::find(modes.cbegin(), modes.cend(), mode) == modes.cend())
                {
                    modes.push_back(mode);
                }
            }
        }

        std::vector<Tensor const*> all = inputs;
        all.push_back(&out);
        std::vector<int64_t> lengths;
        for(auto mode : modes)
        {
            lengths.push_back(lengthOf(all, mode));
        }

        std::vector<double>  acc(out.data.size(), 0.0);
        std::vector<int64_t> idx(modes.size(), 0);
        while(true)
        {
            double product = 1.0;
            for(auto const* t : inputs)
            {
                product *= t->data[static_cast<std::size_t>(offsetOf(*t, modes, idx))];
            }
            acc[static_cast<std::size_t>(offsetOf(out, modes, idx))] += product;

            std::size_t i = 0;
            while(i < idx.size() && ++idx[i] == lengths[i])
            {
                idx[i++] = 0;
            }
            if(i == idx.size())
            {
                break;
            }
        }

        std::vector<float> result(acc.size());
        for(std::size_t i = 0; i < acc.size(); i++)
        {
            result[i] = static_cast<float>(alpha * acc[i]);
        }
        return result;
    }

    hiptensorStatus_t createDesc(hiptensorHandle_t handle, TensorDescGuard& guard, Tensor const& t)
    {
        return hiptensorCreateTensorDescriptor(handle,
                                               &guard.desc,
                                               static_cast<uint32_t>(t.lengths.size()),
                                               t.lengths.data(),
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

    // Creates a plan for the operation with the given algorithm, with the estimated workspace as
    // limit, and returns the first status that is not HIPTENSOR_STATUS_SUCCESS.
    hiptensorStatus_t createPlan(hiptensorHandle_t              handle,
                                 hiptensorOperationDescriptor_t op,
                                 hiptensorAlgo_t                algo,
                                 PlanPrefGuard&                 pref,
                                 PlanGuard&                     plan,
                                 uint64_t&                      workspaceSize)
    {
        EXPECT_EQ(hiptensorCreatePlanPreference(handle, &pref.pref, algo, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        auto status = hiptensorEstimateWorkspaceSize(
            handle, op, pref.pref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize);
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }

        status = hiptensorCreatePlan(handle, &plan.plan, op, pref.pref, workspaceSize);
        if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }

        return hiptensorPlanGetAttribute(handle,
                                         plan.plan,
                                         HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                         &workspaceSize,
                                         sizeof(workspaceSize));
    }

    void checkResult(DeviceBuffer const& device, std::vector<float> const& expected)
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

    // Plans D = alpha * A * B with algo, runs it with alpha = 2 and compares D with the
    // reference.
    void planRunAndCheck(Tensor const&                a,
                         Tensor const&                b,
                         Tensor const&                d,
                         hiptensorComputeDescriptor_t compute,
                         hiptensorAlgo_t              algo)
    {
        HandleGuard handle;
        ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descD;
        ASSERT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descD, d), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        auto        status = hiptensorCreateContraction(handle.handle,
                                                 &op.desc,
                                                 descA.desc,
                                                 a.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descB.desc,
                                                 b.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descD.desc,
                                                 d.modes.data(),
                                                 compute);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "This contraction is not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

        // A problem the kernels can address, so that the plan is decided by kernel selection.
        ASSERT_FALSE(hiptensor::exceedsCkTensorLimits(*op.desc));

        PlanPrefGuard pref;
        PlanGuard     plan;
        uint64_t      workspaceSize = 0;
        ASSERT_EQ(createPlan(handle.handle, op.desc, algo, pref, plan, workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS);

        // The kernels read alpha as their compute type.
        float const           alpha = 2.0f;
        hiptensor::ScalarData alphaValue;
        hiptensor::writeVal(&alphaValue, compute, hiptensor::ScalarData(compute, alpha));

        auto const expected = contractionReference({&a, &b}, d, alpha);

        DeviceBuffer devA, devB, devD, workspace;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a.data));
        ASSERT_NO_FATAL_FAILURE(upload(devB, b.data));
        ASSERT_NO_FATAL_FAILURE(upload(devD, std::vector<float>(expected.size(), -1000.0f)));
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace.ptr, workspaceSize), hipSuccess);
        }

        ASSERT_EQ(hiptensorContract(handle.handle,
                                    plan.plan,
                                    &alphaValue,
                                    devA.ptr,
                                    devB.ptr,
                                    nullptr,
                                    nullptr,
                                    devD.ptr,
                                    workspace.ptr,
                                    workspaceSize,
                                    0 /* stream */),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_NO_FATAL_FAILURE(checkResult(devD, expected));
    }

    // Plans E = alpha * A * B * C with algo, runs it with alpha = 2 and compares E with the
    // reference.
    void planRunAndCheckTrinary(
        Tensor const& a, Tensor const& b, Tensor const& c, Tensor const& e, hiptensorAlgo_t algo)
    {
        HandleGuard handle;
        ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descC, descE;
        ASSERT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descC, c), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descE, e), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        auto        status = hiptensorCreateContractionTrinary(handle.handle,
                                                        &op.desc,
                                                        descA.desc,
                                                        a.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descB.desc,
                                                        b.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descC.desc,
                                                        c.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        nullptr,
                                                        nullptr,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descE.desc,
                                                        e.modes.data(),
                                                        HIPTENSOR_COMPUTE_DESC_32F);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "This contraction is not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        ASSERT_FALSE(hiptensor::exceedsCkTensorLimits(*op.desc));

        PlanPrefGuard pref;
        PlanGuard     plan;
        uint64_t      workspaceSize = 0;
        ASSERT_EQ(createPlan(handle.handle, op.desc, algo, pref, plan, workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS);

        float const           alpha = 2.0f;
        hiptensor::ScalarData alphaValue;
        hiptensor::writeVal(&alphaValue,
                            HIPTENSOR_COMPUTE_DESC_32F,
                            hiptensor::ScalarData(HIPTENSOR_COMPUTE_DESC_32F, alpha));

        auto const expected = contractionReference({&a, &b, &c}, e, alpha);

        DeviceBuffer devA, devB, devC, devE, workspace;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a.data));
        ASSERT_NO_FATAL_FAILURE(upload(devB, b.data));
        ASSERT_NO_FATAL_FAILURE(upload(devC, c.data));
        ASSERT_NO_FATAL_FAILURE(upload(devE, std::vector<float>(expected.size(), -1000.0f)));
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace.ptr, workspaceSize), hipSuccess);
        }

        ASSERT_EQ(hiptensorContractTrinary(handle.handle,
                                           plan.plan,
                                           &alphaValue,
                                           devA.ptr,
                                           devB.ptr,
                                           devC.ptr,
                                           nullptr,
                                           nullptr,
                                           devE.ptr,
                                           workspace.ptr,
                                           workspaceSize,
                                           0 /* stream */),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_NO_FATAL_FAILURE(checkResult(devE, expected));
    }

    // D[m,n] = A[m,k] * B[n,k]. |partial sums| <= 8 * 9 = 72.
    void planRunAndCheckMatrixProduct(hiptensorComputeDescriptor_t compute, hiptensorAlgo_t algo)
    {
        auto a = makeTensor({'m', 'k'}, {32, 8}, 2);
        auto b = makeTensor({'n', 'k'}, {32, 8}, 5);
        auto d = makeTensor({'m', 'n'}, {32, 32}, 0);
        planRunAndCheck(a, b, d, compute, algo);
    }

    // E[m,n] = A[m,k] * B[k,j] * C[j,n], so T = A * B is T[m,j]. Column-major T is
    // contiguous along m, which is not contracted in step 2 (T * C).
    // |partial sums| <= 8 * 8 * 27 = 1728.
    void planRunAndCheckTrinaryOutputContiguousT(hiptensorAlgo_t algo)
    {
        auto a = makeTensor({'m', 'k'}, {16, 8}, 1);
        auto b = makeTensor({'k', 'j'}, {8, 8}, 3);
        auto c = makeTensor({'j', 'n'}, {8, 8}, 4);
        auto e = makeTensor({'m', 'n'}, {16, 8}, 0);
        planRunAndCheckTrinary(a, b, c, e, algo);
    }

    // E[n,m] = A[j,k] * B[k,m] * C[j,n], so T = A * B is T[j,m]. Column-major T is contiguous
    // along j, which step 2 (T * C) contracts, and C and E are contiguous along the modes step 2
    // contracts and keeps from C, so kernels that read T, C and E in vectors can run step 2. In
    // step 1, A and T are contiguous along j, which step 1 keeps from A, so those kernels cannot
    // run step 1.
    void planRunAndCheckTrinaryContractedContiguousT(hiptensorAlgo_t algo)
    {
        auto a = makeTensor({'j', 'k'}, {8, 8}, 1);
        auto b = makeTensor({'k', 'm'}, {8, 16}, 3);
        auto c = makeTensor({'j', 'n'}, {8, 8}, 4);
        auto e = makeTensor({'n', 'm'}, {8, 16}, 0);
        planRunAndCheckTrinary(a, b, c, e, algo);
    }
} // namespace

TEST(ActorCriticSelectionTest, PlansAndRunsF32)
{
    planRunAndCheckMatrixProduct(HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_ALGO_ACTOR_CRITIC);
}

// The f32 data / bf16 compute branches of the actor-critic table.
TEST(ActorCriticSelectionTest, PlansAndRunsF32WithBf16Compute)
{
    planRunAndCheckMatrixProduct(HIPTENSOR_COMPUTE_DESC_16BF, HIPTENSOR_ALGO_ACTOR_CRITIC);
}

// D[m,n] = A[m] * B[n] contracts no mode. The actor-critic table has no entry for an A with a
// single mode, which used to fail the plan with HIPTENSOR_STATUS_EXECUTION_FAILED. A kernel is
// now selected among those that can run the problem.
TEST(ActorCriticSelectionTest, ProblemWithoutTableEntryIsPlannedAndRun)
{
    auto a = makeTensor({'m'}, {32}, 2);
    auto b = makeTensor({'n'}, {32}, 5);
    auto d = makeTensor({'m', 'n'}, {32, 32}, 0);
    planRunAndCheck(a, b, d, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_ALGO_ACTOR_CRITIC);
}

// The trinary problems of ActorCriticRowMajorTableTest, with the tables this build uses and with
// brute force, which also selects among the kernels that can run both steps.
TEST(ActorCriticSelectionTest, TrinaryPlansAndRuns)
{
    for(auto algo : {HIPTENSOR_ALGO_ACTOR_CRITIC, HIPTENSOR_ALGO_DEFAULT})
    {
        SCOPED_TRACE(algo == HIPTENSOR_ALGO_DEFAULT ? "ALGO_DEFAULT" : "ALGO_ACTOR_CRITIC");
        ASSERT_NO_FATAL_FAILURE(planRunAndCheckTrinaryOutputContiguousT(algo));
        ASSERT_NO_FATAL_FAILURE(planRunAndCheckTrinaryContractedContiguousT(algo));
    }
}

// For two-mode A, the row-major table names a kernel that reads A and B in vectors along the
// contracted mode and writes D in vectors along n. Column-major A and D are contiguous along m,
// so that kernel cannot run the problem. Its plan used to succeed, and the contraction then
// failed with HIPTENSOR_STATUS_INTERNAL_ERROR.
TEST(ActorCriticRowMajorTableTest, TableKernelThatCannotRunTheProblemIsReplaced)
{
    if(!usesRowMajorTables())
    {
        GTEST_SKIP() << rowMajorTablesOnly;
    }
    planRunAndCheckMatrixProduct(HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_ALGO_ACTOR_CRITIC);
}

// The same table kernel cannot run step 2 of this trinary contraction, which the table is
// consulted for.
TEST(ActorCriticRowMajorTableTest, TrinaryTableKernelThatCannotRunStep2IsReplaced)
{
    if(!usesRowMajorTables())
    {
        GTEST_SKIP() << rowMajorTablesOnly;
    }
    planRunAndCheckTrinaryOutputContiguousT(HIPTENSOR_ALGO_ACTOR_CRITIC);
}

// The same table kernel can run step 2 of this trinary contraction but not step 1, which runs
// the same kernel. The plan used to succeed, and the contraction then failed in step 1.
TEST(ActorCriticRowMajorTableTest, TrinaryTableKernelThatCannotRunStep1IsReplaced)
{
    if(!usesRowMajorTables())
    {
        GTEST_SKIP() << rowMajorTablesOnly;
    }
    planRunAndCheckTrinaryContractedContiguousT(HIPTENSOR_ALGO_ACTOR_CRITIC);
}

// ---- Checking the actor-critic winner ----

namespace
{
    using ck::tensor_operation::device::BaseArgument;
    using ck::tensor_operation::device::BaseInvoker;
    using ck::tensor_operation::device::BaseOperator;
    using hiptensor::ContractionSolution;

    struct MockArgument : public BaseArgument
    {
        explicit MockArgument(std::size_t workspaceBytes)
            : mWorkspaceBytes(workspaceBytes)
        {
        }
        std::size_t mWorkspaceBytes;
    };

    struct MockOperator : public BaseOperator
    {
        explicit MockOperator(std::string name)
            : mName(std::move(name))
        {
        }
        std::string GetTypeString() const override
        {
            return mName;
        }
        std::size_t GetWorkSpaceSize(BaseArgument const* arg) const override
        {
            return static_cast<MockArgument const*>(arg)->mWorkspaceBytes;
        }
        std::string mName;
    };

    // Counts its launches and reports a fixed time, so that brute force can rank the kernels.
    struct MockInvoker : public BaseInvoker
    {
        MockInvoker(int& runs, float timeMs)
            : mRuns(runs)
            , mTimeMs(timeMs)
        {
        }
        float Run(BaseArgument const*, StreamConfig const&) override
        {
            mRuns++;
            return mTimeMs;
        }
        int&  mRuns;
        float mTimeMs;
    };

    struct MockParams : public hiptensor::ContractionSolutionParams
    {
        int32_t dimsM() const override
        {
            return 1;
        }
        int32_t dimsN() const override
        {
            return 1;
        }
        int32_t dimsK() const override
        {
            return 1;
        }
        hiptensorDataType_t typeA() const override
        {
            return HIPTENSOR_R_32F;
        }
        hiptensorDataType_t typeB() const override
        {
            return HIPTENSOR_R_32F;
        }
        hiptensorDataType_t typeC() const override
        {
            return hiptensor::NONE_TYPE;
        }
        hiptensorDataType_t typeD() const override
        {
            return HIPTENSOR_R_32F;
        }
        hiptensorComputeDescriptor_t typeCompute() const override
        {
            return HIPTENSOR_COMPUTE_DESC_32F;
        }
        hiptensorOperator_t opA() const override
        {
            return HIPTENSOR_OP_IDENTITY;
        }
        hiptensorOperator_t opB() const override
        {
            return HIPTENSOR_OP_IDENTITY;
        }
        hiptensor::ContractionOpId_t opCDE() const override
        {
            return hiptensor::ContractionOpId_t::SCALE;
        }
    };

    // A kernel that either can or cannot run every problem, needs workspaceBytes of workspace,
    // and takes timeMs.
    class MockSolution : public ContractionSolution
    {
    public:
        MockSolution(std::string name, bool supported, std::size_t workspaceBytes, float timeMs)
            : ContractionSolution(std::make_unique<MockOperator>(std::move(name)),
                                  std::make_unique<MockParams>())
            , mSupported(supported)
            , mWorkspaceBytes(workspaceBytes)
            , mTimeMs(timeMs)
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
            resetArgs();
            if(!mSupported)
            {
                return false;
            }
            mInvokerArgPtr = std::make_unique<MockArgument>(mWorkspaceBytes);
            mInvokerPtr    = std::make_unique<MockInvoker>(mRuns, mTimeMs);
            mM             = 4;
            mN             = 4;
            mK             = 4;
            mBytes         = 256;
            mValid         = true;
            return true;
        }

        int runs() const
        {
            return mRuns;
        }

    private:
        bool        mSupported;
        std::size_t mWorkspaceBytes;
        float       mTimeMs;
        int         mRuns = 0;
    };

    constexpr uint64_t workspaceLimit = 256;

    // D[m,n] = A[m,k] * B[n,k] with m = n = k = 4.
    hiptensorStatus_t checkWinner(ContractionSolution**                    winner,
                                  std::vector<ContractionSolution*> const& candidates)
    {
        std::vector<std::size_t> const lengths = {4, 4};
        std::vector<std::size_t> const strides = {1, 4};
        return hiptensor::ensureSupportedWinner(winner,
                                                candidates,
                                                HIPTENSOR_R_32F,
                                                lengths,
                                                strides,
                                                {'m', 'k'},
                                                HIPTENSOR_R_32F,
                                                lengths,
                                                strides,
                                                {'n', 'k'},
                                                hiptensor::NONE_TYPE,
                                                {},
                                                {},
                                                {},
                                                HIPTENSOR_R_32F,
                                                lengths,
                                                strides,
                                                {'m', 'n'},
                                                HIPTENSOR_COMPUTE_DESC_32F,
                                                {},
                                                workspaceLimit);
    }
} // namespace

TEST(ActorCriticWinnerTest, WinnerThatCanRunTheProblemIsKept)
{
    MockSolution named("named", true, 0, 2.0f);
    MockSolution faster("faster", true, 0, 1.0f);

    ContractionSolution* winner = &named;
    EXPECT_EQ(checkWinner(&winner, {&named, &faster}), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(winner, &named);

    // Nothing was timed.
    EXPECT_EQ(named.runs(), 0);
    EXPECT_EQ(faster.runs(), 0);
}

TEST(ActorCriticWinnerTest, WinnerThatCannotRunTheProblemIsReplaced)
{
    MockSolution named("named", false, 0, 1.0f);
    MockSolution slow("slow", true, 0, 3.0f);
    MockSolution fast("fast", true, 0, 2.0f);

    ContractionSolution* winner = &named;
    EXPECT_EQ(checkWinner(&winner, {&named, &slow, &fast}), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(winner, &fast);
    EXPECT_EQ(named.runs(), 0);
}

TEST(ActorCriticWinnerTest, WinnerNeedingMoreWorkspaceThanTheLimitIsReplaced)
{
    MockSolution named("named", true, workspaceLimit + 1, 1.0f);
    MockSolution other("other", true, workspaceLimit, 2.0f);

    ContractionSolution* winner = &named;
    EXPECT_EQ(checkWinner(&winner, {&named, &other}), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(winner, &other);
    EXPECT_EQ(named.runs(), 0);
}

// The table names no kernel for the problem, or a kernel that was not built.
TEST(ActorCriticWinnerTest, MissingWinnerIsSelected)
{
    MockSolution unsupported("unsupported", false, 0, 1.0f);
    MockSolution supported("supported", true, 0, 2.0f);

    ContractionSolution* winner = nullptr;
    EXPECT_EQ(checkWinner(&winner, {&unsupported, &supported}), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(winner, &supported);
}

TEST(ActorCriticWinnerTest, NotSupportedWhenNoKernelCanRunTheProblem)
{
    MockSolution named("named", false, 0, 1.0f);
    MockSolution other("other", false, 0, 1.0f);
    MockSolution tooMuchWorkspace("too_much_workspace", true, workspaceLimit + 1, 1.0f);

    ContractionSolution* winner = &named;
    EXPECT_EQ(checkWinner(&winner, {&named, &other, &tooMuchWorkspace}),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(winner, nullptr);
    EXPECT_EQ(tooMuchWorkspace.runs(), 0);
}
