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

// Concurrent execution tests.
//
// Kernel solutions live in process-wide registries and are shared by every plan
// (and thread) that selects them. These tests run several threads that execute
// the same problem at the same time, each with its own inputs, output and stream,
// and compare every result bit for bit with the result of a single-threaded call.
//
// Inputs are small integers so every result is exact in fp32 regardless of the
// accumulation order, which keeps the bitwise comparison valid even when threads
// end up on different kernels.
//
// Two controls guard the harness itself:
//   Serial: every call is made under one mutex; it must report no mismatches.
//   Broken: thread 1 is given thread 0's inputs; every one of its calls must be
//           reported as a mismatch.
//
// HIPTENSOR_CONCURRENCY_ITERS overrides the number of calls per thread.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "util.hpp"

namespace
{
    constexpr int kThreads = 4;

    long iterations()
    {
        if(auto* env = std::getenv("HIPTENSOR_CONCURRENCY_ITERS"))
        {
            return std::max(1L, std::atol(env));
        }
        return 300;
    }

    // Small integer pattern, distinct per scale.
    std::vector<float> pattern(std::size_t n, int scale)
    {
        std::vector<float> v(n);
        for(std::size_t i = 0; i < n; i++)
        {
            v[i] = static_cast<float>(((i * 37) % 7) * scale);
        }
        return v;
    }

    struct DeviceBuffer
    {
        DeviceBuffer() = default;
        explicit DeviceBuffer(std::size_t bytes)
        {
            if(bytes > 0 && hipMalloc(&mPtr, bytes) != hipSuccess)
            {
                mPtr = nullptr;
            }
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : mPtr(std::exchange(other.mPtr, nullptr))
        {
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
        {
            std::swap(mPtr, other.mPtr);
            return *this;
        }
        ~DeviceBuffer()
        {
            if(mPtr != nullptr)
            {
                (void)hipFree(mPtr);
            }
        }
        void* mPtr = nullptr;
    };

    DeviceBuffer upload(std::vector<float> const& host)
    {
        DeviceBuffer buf(host.size() * sizeof(float));
        if(buf.mPtr != nullptr)
        {
            (void)hipMemcpy(
                buf.mPtr, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice);
        }
        return buf;
    }

    // One operation under test: how to describe it, what each thread's inputs are,
    // and how to execute it.
    struct OpCase
    {
        virtual ~OpCase() = default;

        // Create the operation descriptor. Return a non-success status to skip.
        virtual hiptensorStatus_t create(hiptensorHandle_t handle) = 0;

        virtual std::size_t outputElements() const = 0;

        // Host inputs for thread t, in the order execute() expects them.
        virtual std::vector<std::vector<float>> inputs(int t) const = 0;

        virtual hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                          hiptensorPlan_t           plan,
                                          std::vector<void*> const& in,
                                          void*                     out,
                                          void*                     workspace,
                                          uint64_t                  workspaceSize,
                                          hipStream_t               stream) const
            = 0;

        hiptensorOperationDescriptor_t           mOp = nullptr;
        std::vector<hiptensorTensorDescriptor_t> mDescs;

        hiptensorTensorDescriptor_t desc(hiptensorHandle_t           handle,
                                         std::vector<int64_t> const& lengths)
        {
            hiptensorTensorDescriptor_t d = nullptr;
            hiptensorCreateTensorDescriptor(
                handle, &d, lengths.size(), lengths.data(), nullptr, HIPTENSOR_R_32F, 128);
            mDescs.push_back(d);
            return d;
        }

        void destroy()
        {
            if(mOp != nullptr)
            {
                hiptensorDestroyOperationDescriptor(mOp);
                mOp = nullptr;
            }
            for(auto d : mDescs)
            {
                hiptensorDestroyTensorDescriptor(d);
            }
            mDescs.clear();
        }
    };

    // D_{m,n,u,v} = alpha * A_{m,n,h,k} * B_{u,v,h,k}
    struct ScaleContraction : OpCase
    {
        static constexpr int64_t E = 8;
        std::vector<int32_t> mA{'m', 'n', 'h', 'k'}, mB{'u', 'v', 'h', 'k'}, mD{'m', 'n', 'u', 'v'};
        hiptensorOperator_t      mOpA = HIPTENSOR_OP_IDENTITY;

        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            std::vector<int64_t> l(4, E);
            auto                 a = desc(handle, l), b = desc(handle, l), d = desc(handle, l);
            return hiptensorCreateContraction(handle,
                                              &mOp,
                                              a,
                                              mA.data(),
                                              mOpA,
                                              b,
                                              mB.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              nullptr,
                                              nullptr,
                                              HIPTENSOR_OP_IDENTITY,
                                              d,
                                              mD.data(),
                                              HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::size_t outputElements() const override
        {
            return E * E * E * E;
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(outputElements(), 1 + t), pattern(outputElements(), 1)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*                     workspace,
                                  uint64_t                  workspaceSize,
                                  hipStream_t               stream) const override
        {
            float alpha = 1.0f;
            return hiptensorContract(handle,
                                     plan,
                                     &alpha,
                                     in[0],
                                     in[1],
                                     nullptr,
                                     nullptr,
                                     out,
                                     workspace,
                                     workspaceSize,
                                     stream);
        }
    };

    // D_{m,n,u,v} = alpha * A_{m,n,h,k} * B_{u,v,h,k} + beta * C_{m,n,u,v}
    struct BilinearContraction : ScaleContraction
    {
        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            std::vector<int64_t> l(4, E);
            auto a = desc(handle, l), b = desc(handle, l), c = desc(handle, l), d = desc(handle, l);
            return hiptensorCreateContraction(handle,
                                              &mOp,
                                              a,
                                              mA.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              b,
                                              mB.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              c,
                                              mD.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              d,
                                              mD.data(),
                                              HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(outputElements(), 1 + t),
                    pattern(outputElements(), 1),
                    pattern(outputElements(), 2)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*                     workspace,
                                  uint64_t                  workspaceSize,
                                  hipStream_t               stream) const override
        {
            float alpha = 1.0f, beta = 1.0f;
            return hiptensorContract(handle,
                                     plan,
                                     &alpha,
                                     in[0],
                                     in[1],
                                     &beta,
                                     in[2],
                                     out,
                                     workspace,
                                     workspaceSize,
                                     stream);
        }
    };

    // E_{m,j} = alpha * A_{m,k} * B_{k,n} * C_{n,j}
    struct TrinaryContraction : OpCase
    {
        static constexpr int64_t M = 16, K = 8, N = 8, J = 16;
        std::vector<int32_t>     mA{'m', 'k'}, mB{'k', 'n'}, mC{'n', 'j'}, mE{'m', 'j'};

        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            auto a = desc(handle, {M, K}), b = desc(handle, {K, N}), c = desc(handle, {N, J}),
                 e = desc(handle, {M, J});
            return hiptensorCreateContractionTrinary(handle,
                                                     &mOp,
                                                     a,
                                                     mA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     b,
                                                     mB.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c,
                                                     mC.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     nullptr,
                                                     nullptr,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     e,
                                                     mE.data(),
                                                     HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::size_t outputElements() const override
        {
            return M * J;
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(M * K, 1 + t), pattern(K * N, 1), pattern(N * J, 1)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*                     workspace,
                                  uint64_t                  workspaceSize,
                                  hipStream_t               stream) const override
        {
            float alpha = 1.0f;
            return hiptensorContractTrinary(handle,
                                            plan,
                                            &alpha,
                                            in[0],
                                            in[1],
                                            in[2],
                                            nullptr,
                                            nullptr,
                                            out,
                                            workspace,
                                            workspaceSize,
                                            stream);
        }
    };

    // B_{c,a,b} = alpha * A_{a,b,c}
    struct Permute : OpCase
    {
        static constexpr int64_t E = 32;
        std::vector<int32_t>     mA{'a', 'b', 'c'}, mB{'c', 'a', 'b'};

        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            auto a = desc(handle, {E, E, E}), b = desc(handle, {E, E, E});
            return hiptensorCreatePermutation(handle,
                                              &mOp,
                                              a,
                                              mA.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              b,
                                              mB.data(),
                                              HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::size_t outputElements() const override
        {
            return E * E * E;
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(outputElements(), 1 + t)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*,
                                  uint64_t,
                                  hipStream_t stream) const override
        {
            float alpha = 1.0f;
            return hiptensorPermute(handle, plan, &alpha, in[0], out, stream);
        }
    };

    // D_{n,m} = alpha * A_{m,n} + gamma * C_{m,n}
    struct ElementwiseBinary : OpCase
    {
        static constexpr int64_t E = 64;
        std::vector<int32_t>     mA{'m', 'n'}, mD{'n', 'm'};

        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            auto a = desc(handle, {E, E}), c = desc(handle, {E, E}), d = desc(handle, {E, E});
            return hiptensorCreateElementwiseBinary(handle,
                                                    &mOp,
                                                    a,
                                                    mA.data(),
                                                    HIPTENSOR_OP_IDENTITY,
                                                    c,
                                                    mA.data(),
                                                    HIPTENSOR_OP_IDENTITY,
                                                    d,
                                                    mD.data(),
                                                    HIPTENSOR_OP_ADD,
                                                    HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::size_t outputElements() const override
        {
            return E * E;
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(outputElements(), 1 + t), pattern(outputElements(), 1)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*,
                                  uint64_t,
                                  hipStream_t stream) const override
        {
            float alpha = 1.0f, gamma = 1.0f;
            return hiptensorElementwiseBinaryExecute(
                handle, plan, &alpha, in[0], &gamma, in[1], out, stream);
        }
    };

    // D_{n,m} = (alpha * A_{m,n} + beta * B_{m,n}) + gamma * C_{m,n}
    struct ElementwiseTrinary : OpCase
    {
        static constexpr int64_t E = 64;
        std::vector<int32_t>     mA{'m', 'n'}, mD{'n', 'm'};

        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            auto a = desc(handle, {E, E}), b = desc(handle, {E, E}), c = desc(handle, {E, E}),
                 d = desc(handle, {E, E});
            return hiptensorCreateElementwiseTrinary(handle,
                                                     &mOp,
                                                     a,
                                                     mA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     b,
                                                     mA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c,
                                                     mA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     d,
                                                     mD.data(),
                                                     HIPTENSOR_OP_ADD,
                                                     HIPTENSOR_OP_ADD,
                                                     HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::size_t outputElements() const override
        {
            return E * E;
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(outputElements(), 1 + t),
                    pattern(outputElements(), 1),
                    pattern(outputElements(), 2)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*,
                                  uint64_t,
                                  hipStream_t stream) const override
        {
            float alpha = 1.0f, beta = 1.0f, gamma = 1.0f;
            return hiptensorElementwiseTrinaryExecute(
                handle, plan, &alpha, in[0], &beta, in[1], &gamma, in[2], out, stream);
        }
    };

    // D_{m} = alpha * sum_k A_{m,k}
    struct Reduction : OpCase
    {
        static constexpr int64_t M = 64, K = 64;
        std::vector<int32_t>     mA{'m', 'k'}, mD{'m'};

        hiptensorStatus_t create(hiptensorHandle_t handle) override
        {
            auto a = desc(handle, {M, K}), d = desc(handle, {M});
            return hiptensorCreateReduction(handle,
                                            &mOp,
                                            a,
                                            mA.data(),
                                            HIPTENSOR_OP_IDENTITY,
                                            d,
                                            mD.data(),
                                            HIPTENSOR_OP_IDENTITY,
                                            d,
                                            mD.data(),
                                            HIPTENSOR_OP_ADD,
                                            HIPTENSOR_COMPUTE_DESC_32F);
        }

        std::size_t outputElements() const override
        {
            return M;
        }

        std::vector<std::vector<float>> inputs(int t) const override
        {
            return {pattern(M * K, 1 + t)};
        }

        hiptensorStatus_t execute(hiptensorHandle_t         handle,
                                  hiptensorPlan_t           plan,
                                  std::vector<void*> const& in,
                                  void*                     out,
                                  void*                     workspace,
                                  uint64_t                  workspaceSize,
                                  hipStream_t               stream) const override
        {
            float alpha = 1.0f, beta = 0.0f;
            return hiptensorReduce(
                handle, plan, &alpha, in[0], &beta, out, out, workspace, workspaceSize, stream);
        }
    };

    enum class Mode
    {
        Concurrent,
        Serial,
        Broken
    };

    enum class Plans
    {
        PerThread,
        Shared
    };

    std::string toString(Mode m)
    {
        switch(m)
        {
        case Mode::Concurrent:
            return "Concurrent";
        case Mode::Serial:
            return "Serial";
        default:
            return "Broken";
        }
    }

    std::unique_ptr<OpCase> makeCase(std::string const& name)
    {
        if(name == "ScaleContraction")
            return std::make_unique<ScaleContraction>();
        if(name == "BilinearContraction")
            return std::make_unique<BilinearContraction>();
        if(name == "TrinaryContraction")
            return std::make_unique<TrinaryContraction>();
        if(name == "Permute")
            return std::make_unique<Permute>();
        if(name == "ElementwiseBinary")
            return std::make_unique<ElementwiseBinary>();
        if(name == "ElementwiseTrinary")
            return std::make_unique<ElementwiseTrinary>();
        return std::make_unique<Reduction>();
    }

    // Runs a cleanup function when the scope exits, including through a failed
    // ASSERT_* or GTEST_SKIP, which return from the test body early.
    template <typename F>
    struct ScopeExit
    {
        F mFn;
        ~ScopeExit()
        {
            mFn();
        }
    };
    template <typename F>
    ScopeExit(F) -> ScopeExit<F>;

    struct ThreadState
    {
        std::vector<DeviceBuffer> inputs;
        std::vector<void*>        inputPtrs;
        DeviceBuffer              out;
        DeviceBuffer              workspace;
        uint64_t                  workspaceSize = 0;
        hiptensorPlan_t           plan          = nullptr;
        hipStream_t               stream        = nullptr;
        std::vector<float>        reference;

        long              calls = 0, wrong = 0, errors = 0;
        hiptensorStatus_t lastError = HIPTENSOR_STATUS_SUCCESS;
    };

    // Creates a plan, falling back to a skip when the device does not support the case.
    hiptensorStatus_t createPlan(hiptensorHandle_t              handle,
                                 hiptensorOperationDescriptor_t op,
                                 hiptensorCacheMode_t           cacheMode,
                                 hiptensorPlan_t*               plan,
                                 uint64_t*                      workspaceSize)
    {
        hiptensorPlanPreference_t pref   = nullptr;
        auto                      status = hiptensorCreatePlanPreference(
            handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE);
        if(status != HIPTENSOR_STATUS_SUCCESS)
            return status;
        status = hiptensorPlanPreferenceSetAttribute(
            handle, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode));
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = hiptensorEstimateWorkspaceSize(
                handle, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, workspaceSize);
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = hiptensorCreatePlan(handle, plan, op, pref, *workspaceSize);
        hiptensorDestroyPlanPreference(pref);
        return status;
    }

    bool isSkipStatus(hiptensorStatus_t status)
    {
        return status == HIPTENSOR_STATUS_ARCH_MISMATCH || status == HIPTENSOR_STATUS_NOT_SUPPORTED;
    }

    // Deliberately avoids printing the status name: CTest treats any output containing
    // HIPTENSOR_STATUS_ARCH_MISMATCH as a skip of the whole binary, which would hide
    // failures in the cases that do run on this device.
    char const* skipReason(hiptensorStatus_t status)
    {
        return status == HIPTENSOR_STATUS_ARCH_MISMATCH ? "operation not supported on this device"
                                                        : "operation not supported";
    }

    // A status for failure messages. Numeric for the same reason as skipReason: the
    // name of HIPTENSOR_STATUS_ARCH_MISMATCH would turn the failure into a skip.
    int statusCode(hiptensorStatus_t status)
    {
        return static_cast<int>(status);
    }


    // A plan that autotunes incrementally: the first count calls each try a candidate
    // kernel, then the fastest one is added to the plan cache.
    hiptensorStatus_t createIncrementalPlan(hiptensorHandle_t              handle,
                                            hiptensorOperationDescriptor_t op,
                                            int32_t                        count,
                                            hiptensorPlan_t*               plan,
                                            uint64_t*                      workspaceSize)
    {
        hiptensorPlanPreference_t pref   = nullptr;
        auto                      status = hiptensorCreatePlanPreference(
            handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE);
        if(status != HIPTENSOR_STATUS_SUCCESS)
            return status;
        auto cacheMode    = HIPTENSOR_CACHE_MODE_PEDANTIC;
        auto autotuneMode = HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;
        status            = hiptensorPlanPreferenceSetAttribute(
            handle, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode));
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = hiptensorPlanPreferenceSetAttribute(handle,
                                                         pref,
                                                         HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
                                                         &autotuneMode,
                                                         sizeof(autotuneMode));
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = hiptensorPlanPreferenceSetAttribute(
                handle, pref, HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT, &count, sizeof(count));
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = hiptensorEstimateWorkspaceSize(
                handle, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, workspaceSize);
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = hiptensorCreatePlan(handle, plan, op, pref, *workspaceSize);
        hiptensorDestroyPlanPreference(pref);
        return status;
    }

    // Collects the kernel id of every contraction call, from the performance trace the
    // library logs for each call. Installed for the lifetime of the object.
    class KernelLog
    {
    public:
        KernelLog()
        {
            std::scoped_lock lock(sMutex);
            sIds.clear();
            hiptensorLoggerSetCallback(&KernelLog::callback);
            hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE);
        }
        ~KernelLog()
        {
            hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_OFF);
            hiptensorLoggerSetCallback(nullptr);
        }
        KernelLog(KernelLog const&)            = delete;
        KernelLog& operator=(KernelLog const&) = delete;

        // The kernel ids logged so far, in call order, and clears them.
        std::vector<std::size_t> take()
        {
            std::scoped_lock lock(sMutex);
            return std::exchange(sIds, {});
        }

    private:
        static void callback(int32_t, const char* funcName, const char* msg)
        {
            std::string_view const func(funcName);
            if(func != "hiptensorContraction" && func != "hiptensorContractTrinary")
                return;
            if(auto* id = std::strstr(msg, "KernelId: "))
            {
                std::scoped_lock lock(sMutex);
                sIds.push_back(std::strtoull(id + std::strlen("KernelId: "), nullptr, 10));
            }
        }

        static inline std::mutex               sMutex;
        static inline std::vector<std::size_t> sIds;
    };

    // The (problem hash, kernel id) lines of the handle's plan cache, read back from the
    // file hiptensorHandleWritePlanCacheToFile writes.
    std::vector<std::pair<std::size_t, std::size_t>> planCacheLines(hiptensorHandle_t handle)
    {
        std::ostringstream name;
        name << ::testing::TempDir() << "hiptensor_"
             << ::testing::UnitTest::GetInstance()->current_test_info()->name() << "_" << std::hex
             << std::random_device{}() << std::random_device{}() << ".bin";
        auto const path = name.str();

        std::vector<std::pair<std::size_t, std::size_t>> lines;
        EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(handle, path.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);
        {
            std::ifstream in(path, std::ios::in | std::ios::binary);
            uint32_t      maxCachelines = 0;
            std::size_t   size          = 0;
            in.read(reinterpret_cast<char*>(&maxCachelines), sizeof(maxCachelines));
            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            lines.resize(in.good() ? size : 0);
            for(auto& [hash, uid] : lines)
            {
                in.read(reinterpret_cast<char*>(&hash), sizeof(hash));
                in.read(reinterpret_cast<char*>(&uid), sizeof(uid));
            }
            EXPECT_TRUE(in.good()) << "truncated plan cache file " << path;
        }
        std::remove(path.c_str());
        return lines;
    }

    // Device buffers for one call of an operation case.
    struct CallBuffers
    {
        std::vector<DeviceBuffer> inputs;
        std::vector<void*>        inputPtrs;
        DeviceBuffer              out;
        DeviceBuffer              workspace;
    };

    CallBuffers makeBuffers(OpCase const& opCase, int t, uint64_t workspaceSize)
    {
        CallBuffers b;
        for(auto const& host : opCase.inputs(t))
        {
            b.inputs.push_back(upload(host));
            b.inputPtrs.push_back(b.inputs.back().mPtr);
        }
        b.out       = DeviceBuffer(opCase.outputElements() * sizeof(float));
        b.workspace = DeviceBuffer(workspaceSize);
        return b;
    }

    // Runs one call on the null stream and returns its status and output.
    std::pair<hiptensorStatus_t, std::vector<float>> runCall(OpCase const&      opCase,
                                                             hiptensorHandle_t  handle,
                                                             hiptensorPlan_t    plan,
                                                             CallBuffers const& b,
                                                             uint64_t           workspaceSize)
    {
        std::vector<float> result(opCase.outputElements());
        auto               status = opCase.execute(
            handle, plan, b.inputPtrs, b.out.mPtr, b.workspace.mPtr, workspaceSize, nullptr);
        if(status == HIPTENSOR_STATUS_SUCCESS
           && hipMemcpy(result.data(),
                        b.out.mPtr,
                        result.size() * sizeof(float),
                        hipMemcpyDeviceToHost)
                  != hipSuccess)
        {
            status = HIPTENSOR_STATUS_HIP_ERROR;
        }
        return {status, std::move(result)};
    }
} // namespace

class ConcurrencyTest : public ::testing::TestWithParam<std::tuple<std::string, Plans, Mode>>
{
};

TEST_P(ConcurrencyTest, MatchesSingleThreadedResult)
{
    auto const&       param    = GetParam();
    std::string const caseName = std::get<0>(param);
    Plans const       plans    = std::get<1>(param);
    Mode const        mode     = std::get<2>(param);
    auto              opCase   = makeCase(caseName);
    long const        iters    = iterations();

    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    std::vector<ThreadState> threads(kThreads);
    ScopeExit                cleanup{[&] {
        for(auto& ts : threads)
        {
            if(ts.plan)
                hiptensorDestroyPlan(ts.plan);
            if(ts.stream)
                (void)hipStreamDestroy(ts.stream);
        }
        opCase->destroy();
        hiptensorDestroy(handle);
    }};

    auto status = opCase->create(handle);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << caseName << ": " << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    std::size_t const outBytes = opCase->outputElements() * sizeof(float);

    // Set up each thread in turn. Each plan after the first is created after the
    // first call has run, so every plan adopts the same cached kernel.
    for(int t = 0; t < kThreads; t++)
    {
        auto& ts = threads[t];
        if(plans == Plans::PerThread || t == 0)
        {
            status = createPlan(
                handle, opCase->mOp, HIPTENSOR_CACHE_MODE_PEDANTIC, &ts.plan, &ts.workspaceSize);
            if(isSkipStatus(status))
            {
                GTEST_SKIP() << caseName << ": " << skipReason(status);
            }
            ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        }
        else
        {
            ts.workspaceSize = threads[0].workspaceSize;
        }

        for(auto const& host : opCase->inputs(t))
        {
            ts.inputs.push_back(upload(host));
            ts.inputPtrs.push_back(ts.inputs.back().mPtr);
            ASSERT_NE(ts.inputPtrs.back(), nullptr);
        }
        ts.out       = DeviceBuffer(outBytes);
        ts.workspace = DeviceBuffer(ts.workspaceSize);
        ASSERT_NE(ts.out.mPtr, nullptr);
        ASSERT_EQ(hipStreamCreateWithFlags(&ts.stream, hipStreamNonBlocking), hipSuccess);

        auto plan = ts.plan ? ts.plan : threads[0].plan;
        ASSERT_EQ(hipMemsetAsync(ts.out.mPtr, 0xff, outBytes, ts.stream), hipSuccess);
        ASSERT_EQ(opCase->execute(handle,
                                  plan,
                                  ts.inputPtrs,
                                  ts.out.mPtr,
                                  ts.workspace.mPtr,
                                  ts.workspaceSize,
                                  ts.stream),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(ts.stream), hipSuccess);
        ts.reference.resize(opCase->outputElements());
        ASSERT_EQ(hipMemcpy(ts.reference.data(), ts.out.mPtr, outBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
    }

    // A swapped result can only be detected if every thread expects something different.
    for(int t = 1; t < kThreads; t++)
    {
        ASSERT_NE(std::memcmp(threads[t].reference.data(), threads[0].reference.data(), outBytes),
                  0)
            << "threads 0 and " << t << " have equal references";
    }

    if(mode == Mode::Broken)
    {
        // Thread 1 computes thread 0's problem; the check must flag every call.
        threads[1].inputPtrs = threads[0].inputPtrs;
    }

    std::mutex       serial;
    std::atomic<int> ready{0};
    auto             worker = [&](int t) {
        auto&              ts   = threads[t];
        auto               plan = ts.plan ? ts.plan : threads[0].plan;
        std::vector<float> result(opCase->outputElements());

        ready++;
        while(ready.load() < kThreads)
        {
            std::this_thread::yield();
        }

        for(long i = 0; i < iters; i++)
        {
            ts.calls++;
            if(hipMemsetAsync(ts.out.mPtr, 0xff, outBytes, ts.stream) != hipSuccess)
            {
                ts.errors++;
                continue;
            }

            hiptensorStatus_t callStatus;
            auto              call = [&]() {
                return opCase->execute(handle,
                                       plan,
                                       ts.inputPtrs,
                                       ts.out.mPtr,
                                       ts.workspace.mPtr,
                                       ts.workspaceSize,
                                       ts.stream);
            };
            if(mode == Mode::Concurrent)
            {
                callStatus = call();
            }
            else
            {
                std::scoped_lock lock(serial);
                callStatus = call();
            }

            if(callStatus != HIPTENSOR_STATUS_SUCCESS)
            {
                ts.errors++;
                ts.lastError = callStatus;
                (void)hipStreamSynchronize(ts.stream);
                continue;
            }

            if(hipMemcpyAsync(
                   result.data(), ts.out.mPtr, outBytes, hipMemcpyDeviceToHost, ts.stream)
                   != hipSuccess
               || hipStreamSynchronize(ts.stream) != hipSuccess)
            {
                ts.errors++;
                continue;
            }

            if(std::memcmp(result.data(), ts.reference.data(), outBytes) != 0)
            {
                ts.wrong++;
            }
        }
    };

    std::vector<std::thread> workers;
    for(int t = 0; t < kThreads; t++)
    {
        workers.emplace_back(worker, t);
    }
    for(auto& w : workers)
    {
        w.join();
    }

    for(int t = 0; t < kThreads; t++)
    {
        auto const& ts = threads[t];
        SCOPED_TRACE("thread " + std::to_string(t));
        EXPECT_EQ(ts.calls, iters);
        EXPECT_EQ(ts.errors, 0) << "last error status: " << statusCode(ts.lastError);
        if(mode == Mode::Broken && t == 1)
        {
            EXPECT_EQ(ts.wrong, iters) << "the harness failed to detect a swapped problem";
        }
        else
        {
            EXPECT_EQ(ts.wrong, 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllOps,
    ConcurrencyTest,
    ::testing::Combine(::testing::Values("ScaleContraction",
                                         "BilinearContraction",
                                         "TrinaryContraction",
                                         "Permute",
                                         "ElementwiseBinary",
                                         "ElementwiseTrinary",
                                         "Reduction"),
                       ::testing::Values(Plans::PerThread, Plans::Shared),
                       ::testing::Values(Mode::Concurrent, Mode::Serial, Mode::Broken)),
    [](auto const& info) {
        return std::get<0>(info.param)
               + (std::get<1>(info.param) == Plans::Shared ? "_SharedPlan_" : "_PlanPerThread_")
               + toString(std::get<2>(info.param));
    });

// Incremental autotuning, checked through the plan cache. Without a plan cache there is
// nothing to autotune into, and writing the cache file succeeds without doing anything.
class ConcurrencyAutotuneTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(hiptensor::checkEnvironmentVariableEnabled("HIPTENSOR_DISABLE_PLAN_CACHE"))
        {
            GTEST_SKIP() << "plan cache disabled by HIPTENSOR_DISABLE_PLAN_CACHE";
        }
    }
};

// Incremental autotuning with one plan shared by all threads. Each thread makes one call
// at the same time, so every call of the round is in flight before any is recorded. Each
// call must run its own candidate, and the round must cache one of the kernels it ran.
TEST_F(ConcurrencyAutotuneTest, SharedPlanTriesEachCandidateOnce)
{
    ScaleContraction  opCase;
    hiptensorHandle_t handle = nullptr, refHandle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreate(&refHandle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t plan = nullptr, refPlan = nullptr;
    ScopeExit       cleanup{[&] {
        if(plan)
            hiptensorDestroyPlan(plan);
        if(refPlan)
            hiptensorDestroyPlan(refPlan);
        opCase.destroy();
        hiptensorDestroy(refHandle);
        hiptensorDestroy(handle);
    }};

    auto status = opCase.create(handle);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    uint64_t workspaceSize = 0, refWorkspaceSize = 0;
    status = createIncrementalPlan(handle, opCase.mOp, kThreads, &plan, &workspaceSize);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    // The reference results come from another handle, so they do not take part in the round.
    ASSERT_EQ(createPlan(refHandle,
                         opCase.mOp,
                         HIPTENSOR_CACHE_MODE_NONE,
                         &refPlan,
                         &refWorkspaceSize),
              HIPTENSOR_STATUS_SUCCESS);

    std::vector<CallBuffers>        buffers;
    std::vector<std::vector<float>> references;
    for(int t = 0; t < kThreads; t++)
    {
        buffers.push_back(makeBuffers(opCase, t, std::max(workspaceSize, refWorkspaceSize)));
        auto [refStatus, reference]
            = runCall(opCase, refHandle, refPlan, buffers.back(), refWorkspaceSize);
        ASSERT_EQ(refStatus, HIPTENSOR_STATUS_SUCCESS) << "status " << statusCode(refStatus);
        references.push_back(std::move(reference));
    }

    KernelLog                      log;
    std::vector<hiptensorStatus_t> callStatus(kThreads, HIPTENSOR_STATUS_SUCCESS);
    std::vector<char>              matches(kThreads, false);
    std::atomic<int>               ready{0};
    auto                           worker = [&](int t) {
        ready++;
        while(ready.load() < kThreads)
        {
            std::this_thread::yield();
        }
        auto [s, result] = runCall(opCase, handle, plan, buffers[t], workspaceSize);
        callStatus[t]    = s;
        matches[t]       = s == HIPTENSOR_STATUS_SUCCESS && result == references[t];
    };
    std::vector<std::thread> workers;
    for(int t = 0; t < kThreads; t++)
    {
        workers.emplace_back(worker, t);
    }
    for(auto& w : workers)
    {
        w.join();
    }

    // A candidate kernel may not support the problem: that call fails and is skipped.
    int successes = 0;
    for(int t = 0; t < kThreads; t++)
    {
        SCOPED_TRACE("thread " + std::to_string(t));
        if(callStatus[t] == HIPTENSOR_STATUS_SUCCESS)
        {
            successes++;
            EXPECT_TRUE(matches[t]);
        }
    }
    ASSERT_GT(successes, 0);

    auto const tried = log.take();
    ASSERT_EQ(tried.size(), static_cast<std::size_t>(successes));
    EXPECT_EQ(std::set<std::size_t>(tried.begin(), tried.end()).size(), tried.size())
        << "two calls of the round ran the same candidate";

    auto const lines = planCacheLines(handle);
    ASSERT_EQ(lines.size(), 1u) << "the round did not cache a kernel";
    EXPECT_NE(std::find(tried.begin(), tried.end(), lines[0].second), tried.end())
        << "the cached kernel was not tried in the round";

    // The next call runs the cached kernel.
    auto [s, result] = runCall(opCase, handle, plan, buffers[0], workspaceSize);
    ASSERT_EQ(s, HIPTENSOR_STATUS_SUCCESS) << "status " << statusCode(s);
    EXPECT_EQ(result, references[0]);
    EXPECT_EQ(log.take(), std::vector<std::size_t>{lines[0].second});
}

// A call that reserves a candidate and then returns early, here for a missing workspace,
// counts as a failed run of that candidate: the round still completes.
TEST_F(ConcurrencyAutotuneTest, AbandonedCandidateDoesNotStallRound)
{
    TrinaryContraction opCase;
    hiptensorHandle_t  handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t plan = nullptr;
    ScopeExit       cleanup{[&] {
        if(plan)
            hiptensorDestroyPlan(plan);
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    auto status = opCase.create(handle);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    uint64_t workspaceSize = 0;
    status                 = createIncrementalPlan(handle, opCase.mOp, 2, &plan, &workspaceSize);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    ASSERT_GT(workspaceSize, 0u) << "the intermediate tensor needs a workspace";

    auto const b = makeBuffers(opCase, 0, workspaceSize);
    KernelLog  log;

    auto [abandoned, unused] = runCall(opCase, handle, plan, b, 0);
    EXPECT_EQ(abandoned, HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
        << "status " << statusCode(abandoned);
    EXPECT_TRUE(planCacheLines(handle).empty());

    auto [s, result] = runCall(opCase, handle, plan, b, workspaceSize);
    ASSERT_EQ(s, HIPTENSOR_STATUS_SUCCESS) << "status " << statusCode(s);

    auto const lines = planCacheLines(handle);
    ASSERT_EQ(lines.size(), 1u) << "the round did not complete";
    EXPECT_EQ(log.take(), std::vector<std::size_t>{lines[0].second});
}

// Calls of different problems on one handle, interleaved, are tuned separately: each
// problem caches its kernel after its own count of calls. Permute has no candidate list,
// so its round counts calls of its one solution.
TEST_F(ConcurrencyAutotuneTest, InterleavedProblemsAreTunedSeparately)
{
    constexpr int32_t kCount = 3;

    std::vector<std::unique_ptr<OpCase>> cases;
    cases.push_back(std::make_unique<ScaleContraction>());
    cases.push_back(std::make_unique<BilinearContraction>());
    cases.push_back(std::make_unique<Permute>());

    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    std::vector<hiptensorPlan_t> plans(cases.size(), nullptr);
    ScopeExit                    cleanup{[&] {
        for(auto plan : plans)
        {
            if(plan)
                hiptensorDestroyPlan(plan);
        }
        for(auto& opCase : cases)
        {
            opCase->destroy();
        }
        hiptensorDestroy(handle);
    }};

    std::vector<uint64_t>    workspaceSizes(cases.size(), 0);
    std::vector<CallBuffers> buffers;
    for(std::size_t i = 0; i < cases.size(); i++)
    {
        auto status = cases[i]->create(handle);
        if(status == HIPTENSOR_STATUS_SUCCESS)
            status = createIncrementalPlan(
                handle, cases[i]->mOp, kCount, &plans[i], &workspaceSizes[i]);
        if(isSkipStatus(status))
        {
            GTEST_SKIP() << skipReason(status);
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        buffers.push_back(makeBuffers(*cases[i], 0, workspaceSizes[i]));
    }

    for(int32_t call = 0; call < kCount; call++)
    {
        SCOPED_TRACE("call " + std::to_string(call));
        EXPECT_EQ(planCacheLines(handle).size(), 0u);
        for(std::size_t i = 0; i < cases.size(); i++)
        {
            // A candidate that cannot run the problem fails its call; the round moves on.
            (void)runCall(*cases[i], handle, plans[i], buffers[i], workspaceSizes[i]);
        }
    }

    auto const lines = planCacheLines(handle);
    EXPECT_EQ(lines.size(), cases.size()) << "every problem caches its own kernel";
    std::set<std::size_t> hashes;
    for(auto const& line : lines)
    {
        hashes.insert(line.first);
    }
    EXPECT_EQ(hashes.size(), lines.size());
}

// An incremental count of 0 (the default) tries no candidate: the first call's kernel is
// cached, as without incremental autotuning.
TEST_F(ConcurrencyAutotuneTest, CountZeroCachesFirstCall)
{
    ScaleContraction  opCase;
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t plan = nullptr;
    ScopeExit       cleanup{[&] {
        if(plan)
            hiptensorDestroyPlan(plan);
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    auto     status        = opCase.create(handle);
    uint64_t workspaceSize = 0;
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, opCase.mOp, 0, &plan, &workspaceSize);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    auto const b = makeBuffers(opCase, 0, workspaceSize);
    KernelLog  log;
    auto [s, result] = runCall(opCase, handle, plan, b, workspaceSize);
    ASSERT_EQ(s, HIPTENSOR_STATUS_SUCCESS) << "status " << statusCode(s);

    auto const lines = planCacheLines(handle);
    ASSERT_EQ(lines.size(), 1u) << "the first call was not cached";
    EXPECT_EQ(log.take(), std::vector<std::size_t>{lines[0].second});
}

// Two plans of the same problem with different incremental counts share one round. The
// round keeps the count of the plan that started it, so calls through either plan
// complete it.
TEST_F(ConcurrencyAutotuneTest, PlansWithDifferentCountsShareRound)
{
    ScaleContraction  opCase;
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t countTwo = nullptr, countOne = nullptr;
    ScopeExit       cleanup{[&] {
        if(countTwo)
            hiptensorDestroyPlan(countTwo);
        if(countOne)
            hiptensorDestroyPlan(countOne);
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    auto     status        = opCase.create(handle);
    uint64_t workspaceSize = 0, unused = 0;
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, opCase.mOp, 2, &countTwo, &workspaceSize);
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, opCase.mOp, 1, &countOne, &unused);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    auto const b = makeBuffers(opCase, 0, std::max(workspaceSize, unused));
    KernelLog  log;

    // A candidate kernel may not support the problem: that call fails, which still
    // counts toward the round.
    (void)runCall(opCase, handle, countTwo, b, workspaceSize);
    EXPECT_TRUE(planCacheLines(handle).empty()) << "a count-2 round cached after one call";

    (void)runCall(opCase, handle, countOne, b, unused);
    auto const tried = log.take();
    EXPECT_EQ(std::set<std::size_t>(tried.begin(), tried.end()).size(), tried.size())
        << "two calls of the round ran the same candidate";

    auto const lines = planCacheLines(handle);
    ASSERT_EQ(lines.size(), 1u) << "the round did not complete";
    EXPECT_NE(std::find(tried.begin(), tried.end(), lines[0].second), tried.end());
}

// A count-0 plan joins the open round of another plan of the same problem instead of
// caching its own kernel, which would leave the round unfinished.
TEST_F(ConcurrencyAutotuneTest, CountZeroPlanJoinsOpenRound)
{
    ScaleContraction  opCase;
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t countThree = nullptr, countZero = nullptr;
    ScopeExit       cleanup{[&] {
        if(countThree)
            hiptensorDestroyPlan(countThree);
        if(countZero)
            hiptensorDestroyPlan(countZero);
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    auto     status        = opCase.create(handle);
    uint64_t workspaceSize = 0, unused = 0;
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, opCase.mOp, 3, &countThree, &workspaceSize);
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, opCase.mOp, 0, &countZero, &unused);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    auto const b = makeBuffers(opCase, 0, std::max(workspaceSize, unused));
    KernelLog  log;

    (void)runCall(opCase, handle, countThree, b, workspaceSize);

    // The count-0 call runs the round's second candidate: the round stays open.
    (void)runCall(opCase, handle, countZero, b, unused);
    EXPECT_TRUE(planCacheLines(handle).empty())
        << "the count-0 call cached its own kernel instead of joining the round";

    // The third candidate completes the round.
    (void)runCall(opCase, handle, countThree, b, workspaceSize);
    auto const tried = log.take();
    EXPECT_EQ(std::set<std::size_t>(tried.begin(), tried.end()).size(), tried.size())
        << "two calls of the round ran the same candidate";

    auto const lines = planCacheLines(handle);
    ASSERT_EQ(lines.size(), 1u) << "the round did not complete";
    EXPECT_NE(std::find(tried.begin(), tried.end(), lines[0].second), tried.end());
}

// The problem hash leaves out unary operators, so an identity plan and a plan that negates
// A describe the same problem but select from different kernels. Each must autotune with
// its own candidates: an identity kernel would drop the negation and still succeed.
TEST_F(ConcurrencyAutotuneTest, PlansWithDifferentOperatorsDoNotShareRound)
{
    ScaleContraction identity, negate;
    negate.mOpA = HIPTENSOR_OP_NEG;

    hiptensorHandle_t handle = nullptr, refHandle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreate(&refHandle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t identityPlan = nullptr, negatePlan = nullptr, refPlan = nullptr;
    ScopeExit       cleanup{[&] {
        for(auto plan : {identityPlan, negatePlan, refPlan})
        {
            if(plan)
                hiptensorDestroyPlan(plan);
        }
        identity.destroy();
        negate.destroy();
        hiptensorDestroy(refHandle);
        hiptensorDestroy(handle);
    }};

    uint64_t identityWs = 0, negateWs = 0, refWs = 0;
    auto     status = identity.create(handle);
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = negate.create(handle);
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, identity.mOp, 2, &identityPlan, &identityWs);
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createIncrementalPlan(handle, negate.mOp, 2, &negatePlan, &negateWs);
    // The reference comes from another handle, so it takes no part in the rounds.
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status = createPlan(refHandle, negate.mOp, HIPTENSOR_CACHE_MODE_NONE, &refPlan, &refWs);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    auto const b                = makeBuffers(negate, 0, std::max({identityWs, negateWs, refWs}));
    auto [refStatus, reference] = runCall(negate, refHandle, refPlan, b, refWs);
    ASSERT_EQ(refStatus, HIPTENSOR_STATUS_SUCCESS) << "status " << statusCode(refStatus);

    // The identity plan starts its round, then every call of the negating plan must negate.
    // A candidate kernel may not support the problem: that call fails and is skipped.
    (void)runCall(identity, handle, identityPlan, b, identityWs);
    int successes = 0;
    for(int call = 0; call < 3; call++)
    {
        SCOPED_TRACE("call " + std::to_string(call));
        auto [s, result] = runCall(negate, handle, negatePlan, b, negateWs);
        if(s == HIPTENSOR_STATUS_SUCCESS)
        {
            successes++;
            EXPECT_EQ(result, reference);
        }
    }
    EXPECT_GT(successes, 0) << "no call of the negating plan succeeded";
}

// The pass-through permutation kernel, selected when alpha is 1, ignores alpha. A plan keeps
// the kernel of its first call, so later calls with another alpha, from any thread sharing
// the plan, must not run it.
TEST(ConcurrencyPermuteTest, SharedPlanWithDifferentAlphas)
{
    Permute           opCase;
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    hiptensorPlan_t plan = nullptr;
    ScopeExit       cleanup{[&] {
        if(plan)
            hiptensorDestroyPlan(plan);
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    uint64_t workspaceSize = 0;
    auto     status        = opCase.create(handle);
    if(status == HIPTENSOR_STATUS_SUCCESS)
        status
            = createPlan(handle, opCase.mOp, HIPTENSOR_CACHE_MODE_PEDANTIC, &plan, &workspaceSize);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    std::size_t const               bytes = opCase.outputElements() * sizeof(float);
    std::vector<CallBuffers>        buffers;
    std::vector<std::vector<float>> references; // alpha 1
    for(int t = 0; t < kThreads; t++)
    {
        // The first call has alpha 1, so the plan keeps the pass-through kernel.
        buffers.push_back(makeBuffers(opCase, t, workspaceSize));
        auto [s, reference] = runCall(opCase, handle, plan, buffers.back(), workspaceSize);
        ASSERT_EQ(s, HIPTENSOR_STATUS_SUCCESS) << "status " << statusCode(s);
        references.push_back(std::move(reference));
    }

    long const        iters = std::min(iterations(), 50L);
    std::vector<long> wrong(kThreads, 0), errors(kThreads, 0);
    std::atomic<int>  ready{0};
    auto              worker = [&](int t) {
        std::vector<float> result(opCase.outputElements()), expected(result.size());
        ready++;
        while(ready.load() < kThreads)
        {
            std::this_thread::yield();
        }
        for(long i = 0; i < iters; i++)
        {
            float const alpha = (t + i) % 2 == 0 ? 0.5f : 1.0f;
            if(hiptensorPermute(
                   handle, plan, &alpha, buffers[t].inputPtrs[0], buffers[t].out.mPtr, nullptr)
                   != HIPTENSOR_STATUS_SUCCESS
               || hipMemcpy(result.data(), buffers[t].out.mPtr, bytes, hipMemcpyDeviceToHost)
                      != hipSuccess)
            {
                errors[t]++;
                continue;
            }
            // Integer inputs: alpha 0.5 scales them exactly.
            std::transform(references[t].begin(),
                           references[t].end(),
                           expected.begin(),
                           [alpha](float v) { return alpha * v; });
            if(result != expected)
            {
                wrong[t]++;
            }
        }
    };
    std::vector<std::thread> workers;
    for(int t = 0; t < kThreads; t++)
    {
        workers.emplace_back(worker, t);
    }
    for(auto& w : workers)
    {
        w.join();
    }

    for(int t = 0; t < kThreads; t++)
    {
        SCOPED_TRACE("thread " + std::to_string(t));
        EXPECT_EQ(errors[t], 0);
        EXPECT_EQ(wrong[t], 0);
    }
}

// Several threads create plans at the same time. Plan cache lookup is disabled so
// every thread runs the brute-force selection, which times every candidate kernel.
TEST(ConcurrencyPlanCreationTest, ConcurrentBruteForceSelection)
{
    ScaleContraction  opCase;
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit cleanup{[&] {
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    auto status = opCase.create(handle);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    std::size_t const outBytes = opCase.outputElements() * sizeof(float);

    // Single-threaded reference per thread.
    std::vector<std::vector<float>> references(kThreads);
    {
        hiptensorPlan_t plan = nullptr;
        uint64_t        ws   = 0;
        status = createPlan(handle, opCase.mOp, HIPTENSOR_CACHE_MODE_NONE, &plan, &ws);
        ScopeExit destroyPlan{[&] {
            if(plan)
                hiptensorDestroyPlan(plan);
        }};
        if(isSkipStatus(status))
        {
            GTEST_SKIP() << skipReason(status);
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        DeviceBuffer workspace(ws), out(outBytes);
        for(int t = 0; t < kThreads; t++)
        {
            std::vector<DeviceBuffer> in;
            std::vector<void*>        inPtrs;
            for(auto const& host : opCase.inputs(t))
            {
                in.push_back(upload(host));
                inPtrs.push_back(in.back().mPtr);
            }
            ASSERT_EQ(opCase.execute(handle, plan, inPtrs, out.mPtr, workspace.mPtr, ws, nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            references[t].resize(opCase.outputElements());
            ASSERT_EQ(hipMemcpy(references[t].data(), out.mPtr, outBytes, hipMemcpyDeviceToHost),
                      hipSuccess);
        }
    }

    std::vector<hiptensorStatus_t> createStatus(kThreads, HIPTENSOR_STATUS_SUCCESS);
    std::vector<hiptensorStatus_t> execStatus(kThreads, HIPTENSOR_STATUS_SUCCESS);
    std::vector<int>               matches(kThreads, 0);
    std::atomic<int>               ready{0};

    auto worker = [&](int t) {
        ready++;
        while(ready.load() < kThreads)
        {
            std::this_thread::yield();
        }

        hiptensorPlan_t plan = nullptr;
        uint64_t        ws   = 0;
        createStatus[t] = createPlan(handle, opCase.mOp, HIPTENSOR_CACHE_MODE_NONE, &plan, &ws);
        if(createStatus[t] != HIPTENSOR_STATUS_SUCCESS)
            return;

        std::vector<DeviceBuffer> in;
        std::vector<void*>        inPtrs;
        for(auto const& host : opCase.inputs(t))
        {
            in.push_back(upload(host));
            inPtrs.push_back(in.back().mPtr);
        }
        DeviceBuffer workspace(ws), out(outBytes);
        hipStream_t  stream = nullptr;
        (void)hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        execStatus[t] = opCase.execute(handle, plan, inPtrs, out.mPtr, workspace.mPtr, ws, stream);
        std::vector<float> result(opCase.outputElements());
        (void)hipMemcpyAsync(result.data(), out.mPtr, outBytes, hipMemcpyDeviceToHost, stream);
        (void)hipStreamSynchronize(stream);
        matches[t] = std::memcmp(result.data(), references[t].data(), outBytes) == 0;
        (void)hipStreamDestroy(stream);
        hiptensorDestroyPlan(plan);
    };

    std::vector<std::thread> workers;
    for(int t = 0; t < kThreads; t++)
    {
        workers.emplace_back(worker, t);
    }
    for(auto& w : workers)
    {
        w.join();
    }

    for(int t = 0; t < kThreads; t++)
    {
        SCOPED_TRACE("thread " + std::to_string(t));
        EXPECT_EQ(createStatus[t], HIPTENSOR_STATUS_SUCCESS)
            << "status " << statusCode(createStatus[t]);
        EXPECT_EQ(execStatus[t], HIPTENSOR_STATUS_SUCCESS)
            << "status " << statusCode(execStatus[t]);
        EXPECT_TRUE(matches[t]);
    }
}

// Several threads estimate the workspace at the same time. A preference owned by a
// plan holds the full candidate list, so every estimate builds kernel arguments for
// every candidate.
TEST(ConcurrencyWorkspaceTest, ConcurrentWorkspaceEstimate)
{
    ScaleContraction  opCase;
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit cleanup{[&] {
        opCase.destroy();
        hiptensorDestroy(handle);
    }};

    auto status = opCase.create(handle);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    hiptensorPlan_t plan = nullptr;
    uint64_t        ws   = 0;
    status = createPlan(handle, opCase.mOp, HIPTENSOR_CACHE_MODE_PEDANTIC, &plan, &ws);
    ScopeExit destroyPlan{[&] {
        if(plan)
            hiptensorDestroyPlan(plan);
    }};
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << skipReason(status);
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    ASSERT_FALSE(plan->mPref->mCandidates.empty());

    auto const prefs
        = {HIPTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_DEFAULT, HIPTENSOR_WORKSPACE_MAX};
    std::vector<uint64_t> expected;
    for(auto pref : prefs)
    {
        uint64_t size = 0;
        ASSERT_EQ(hiptensorEstimateWorkspaceSize(handle, opCase.mOp, plan->mPref, pref, &size),
                  HIPTENSOR_STATUS_SUCCESS);
        expected.push_back(size);
    }

    long const        iters = iterations();
    std::atomic<int>  ready{0};
    std::atomic<long> mismatches{0};
    auto              worker = [&]() {
        ready++;
        while(ready.load() < kThreads)
        {
            std::this_thread::yield();
        }
        for(long i = 0; i < iters; i++)
        {
            std::size_t j = 0;
            for(auto pref : prefs)
            {
                uint64_t size = 0;
                if(hiptensorEstimateWorkspaceSize(handle, opCase.mOp, plan->mPref, pref, &size)
                       != HIPTENSOR_STATUS_SUCCESS
                   || size != expected[j])
                {
                    mismatches++;
                }
                j++;
            }
        }
    };

    std::vector<std::thread> workers;
    for(int t = 0; t < kThreads; t++)
    {
        workers.emplace_back(worker);
    }
    for(auto& w : workers)
    {
        w.join();
    }

    EXPECT_EQ(mismatches.load(), 0);
}
