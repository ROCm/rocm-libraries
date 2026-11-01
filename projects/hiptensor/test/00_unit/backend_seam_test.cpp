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

// Backend selection, through the public API, for every operation (contraction, trinary
// contraction, permutation, elementwise binary and trinary, reduction):
//  - by default the Composable Kernel backend serves every plan;
//  - HIPTENSOR_BACKEND selects the backend; an unknown value is refused when the plan is
//    created and when the workspace is estimated, as are bad HIPTENSOR_BACKEND_FALLBACK and
//    HIPTENSOR_BACKEND_AB values;
//  - a backend that declines a problem (or is not built) passes it to the fallback backend,
//    unless HIPTENSOR_BACKEND_FALLBACK=0, which makes plan creation report
//    HIPTENSOR_STATUS_NOT_SUPPORTED;
//  - HIPTENSOR_BACKEND_AB runs both backends and logs their difference, without changing the
//    result, also for a contraction whose output is its input C;
//  - every execution runs on the caller's stream;
//  - plan-cache files record the backends.
// Results are compared with plain loops with 64-bit offsets, not with the CPU reference. The
// data are small integers, so every result is exact and compared bit for bit.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include "api_test_helpers.hpp"

namespace
{
    using hiptensor_test::ApiObjects;
    using hiptensor_test::DeviceBuffer;
    using hiptensor_test::packedStrides;
    using hiptensor_test::ScopedEnvironmentVariable;

    // Collects the heuristics-trace and error log lines while it lives.
    class LogCapture
    {
    public:
        LogCapture()
        {
            std::scoped_lock lock(sMutex);
            sLines.clear();
            hiptensorLoggerSetCallback(&LogCapture::callback);
            hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE
                                   | HIPTENSOR_LOG_LEVEL_ERROR);
        }

        ~LogCapture()
        {
            hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_OFF);
            hiptensorLoggerSetCallback(nullptr);
        }

        bool contains(std::string const& text) const
        {
            std::scoped_lock lock(sMutex);
            return std::any_of(sLines.begin(), sLines.end(), [&text](auto const& line) {
                return line.find(text) != std::string::npos;
            });
        }

        std::string all() const
        {
            std::scoped_lock lock(sMutex);
            std::string      result;
            for(auto const& line : sLines)
            {
                result += line + "\n";
            }
            return result;
        }

    private:
        static void callback(int32_t, char const* funcName, char const* msg)
        {
            std::scoped_lock lock(sMutex);
            sLines.push_back(std::string(funcName) + ": " + msg);
        }

        static inline std::mutex               sMutex;
        static inline std::vector<std::string> sLines;
    };

    enum class Op
    {
        Contraction,
        ContractionTrinary,
        Permute,
        ElementwiseBinary,
        ElementwiseTrinary,
        Reduction,
    };

    constexpr Op allOps[] = {Op::Contraction,
                             Op::ContractionTrinary,
                             Op::Permute,
                             Op::ElementwiseBinary,
                             Op::ElementwiseTrinary,
                             Op::Reduction};

    // The operation names of the backend selection log.
    char const* opName(Op op)
    {
        switch(op)
        {
        case Op::Contraction:
            return "contraction";
        case Op::ContractionTrinary:
            return "trinary contraction";
        case Op::Permute:
            return "permutation";
        case Op::ElementwiseBinary:
            return "elementwise binary";
        case Op::ElementwiseTrinary:
            return "elementwise trinary";
        case Op::Reduction:
            return "reduction";
        }
        return "";
    }

    // Small integers in [-3, 3].
    std::vector<float> data(int64_t count, int seed)
    {
        std::vector<float> values(count);
        for(int64_t i = 0; i < count; i++)
        {
            values[i] = static_cast<float>((i * 7 + seed * 5) % 7 - 3);
        }
        return values;
    }

    struct Outcome
    {
        hiptensorStatus_t planStatus = HIPTENSOR_STATUS_SUCCESS;
        hiptensorStatus_t execStatus = HIPTENSOR_STATUS_SUCCESS;
        bool              skipped    = false;
    };

    constexpr float alphaValue = 2.0f;
    constexpr float betaValue  = 3.0f;
    constexpr float gammaValue = -1.0f;

    // Plans `op` with the default preference, runs it on `stream` and compares the output with
    // the loop reference. With `inPlace` the contraction's output is its input C.
    Outcome run(Op op, hipStream_t stream, bool inPlace = false)
    {
        Outcome    outcome;
        ApiObjects api;
        auto const f32     = HIPTENSOR_R_32F;
        auto const compute = HIPTENSOR_COMPUTE_DESC_32F;
        auto const id      = HIPTENSOR_OP_IDENTITY;

        hiptensorOperationDescriptor_t  desc = nullptr;
        std::vector<float>              expected;
        std::vector<float>              initialOutput;
        std::vector<std::vector<float>> inputs;

        int64_t const M = 8, N = 6, K = 5, U = 4;
        switch(op)
        {
        case Op::Contraction:
        {
            // D[m,n] = alpha * sum_k A[m,k] * B[k,n] + beta * C[m,n]
            int32_t const modeA[] = {'m', 'k'}, modeB[] = {'k', 'n'}, modeD[] = {'m', 'n'};
            auto          dA   = api.tensor({M, K}, packedStrides({M, K}), f32);
            auto          dB   = api.tensor({K, N}, packedStrides({K, N}), f32);
            auto          dD   = api.tensor({M, N}, packedStrides({M, N}), f32);
            outcome.planStatus = hiptensorCreateContraction(
                api.handle, &desc, dA, modeA, id, dB, modeB, id, dD, modeD, id, dD, modeD, compute);
            api.keep(desc);
            inputs = {data(M * K, 1), data(K * N, 2), data(M * N, 3)};
            expected.resize(M * N);
            for(int64_t n = 0; n < N; n++)
            {
                for(int64_t m = 0; m < M; m++)
                {
                    float sum = 0.0f;
                    for(int64_t k = 0; k < K; k++)
                    {
                        sum += inputs[0][m + M * k] * inputs[1][k + K * n];
                    }
                    expected[m + M * n] = alphaValue * sum + betaValue * inputs[2][m + M * n];
                }
            }
            initialOutput = inPlace ? inputs[2] : std::vector<float>(M * N, 0.0f);
            break;
        }
        case Op::ContractionTrinary:
        {
            // E[m,n] = alpha * sum_{k,u} A[m,k] * B[k,u] * C[u,n] + beta * D[m,n]
            int32_t const modeA[] = {'m', 'k'}, modeB[] = {'k', 'u'}, modeC[] = {'u', 'n'},
                          modeE[] = {'m', 'n'};
            auto dA               = api.tensor({M, K}, packedStrides({M, K}), f32);
            auto dB               = api.tensor({K, U}, packedStrides({K, U}), f32);
            auto dC               = api.tensor({U, N}, packedStrides({U, N}), f32);
            auto dE               = api.tensor({M, N}, packedStrides({M, N}), f32);
            outcome.planStatus    = hiptensorCreateContractionTrinary(api.handle,
                                                                   &desc,
                                                                   dA,
                                                                   modeA,
                                                                   id,
                                                                   dB,
                                                                   modeB,
                                                                   id,
                                                                   dC,
                                                                   modeC,
                                                                   id,
                                                                   dE,
                                                                   modeE,
                                                                   id,
                                                                   dE,
                                                                   modeE,
                                                                   compute);
            api.keep(desc);
            inputs = {data(M * K, 1), data(K * U, 2), data(U * N, 3), data(M * N, 4)};
            expected.resize(M * N);
            for(int64_t n = 0; n < N; n++)
            {
                for(int64_t m = 0; m < M; m++)
                {
                    float sum = 0.0f;
                    for(int64_t u = 0; u < U; u++)
                    {
                        float ab = 0.0f;
                        for(int64_t k = 0; k < K; k++)
                        {
                            ab += inputs[0][m + M * k] * inputs[1][k + K * u];
                        }
                        sum += ab * inputs[2][u + U * n];
                    }
                    expected[m + M * n] = alphaValue * sum + betaValue * inputs[3][m + M * n];
                }
            }
            initialOutput = std::vector<float>(M * N, 0.0f);
            break;
        }
        case Op::Permute:
        {
            // B[n,m] = alpha * A[m,n]
            int32_t const modeA[] = {'m', 'n'}, modeB[] = {'n', 'm'};
            auto          dA = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dB = api.tensor({N, M}, packedStrides({N, M}), f32);
            outcome.planStatus
                = hiptensorCreatePermutation(api.handle, &desc, dA, modeA, id, dB, modeB, compute);
            api.keep(desc);
            inputs = {data(M * N, 1)};
            expected.resize(M * N);
            for(int64_t n = 0; n < N; n++)
            {
                for(int64_t m = 0; m < M; m++)
                {
                    expected[n + N * m] = alphaValue * inputs[0][m + M * n];
                }
            }
            initialOutput = std::vector<float>(M * N, 0.0f);
            break;
        }
        case Op::ElementwiseBinary:
        {
            // D[m,n] = alpha * A[m,n] + gamma * C[m,n]
            int32_t const modes[] = {'m', 'n'};
            auto          dA      = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dC      = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dD      = api.tensor({M, N}, packedStrides({M, N}), f32);
            outcome.planStatus    = hiptensorCreateElementwiseBinary(api.handle,
                                                                  &desc,
                                                                  dA,
                                                                  modes,
                                                                  id,
                                                                  dC,
                                                                  modes,
                                                                  id,
                                                                  dD,
                                                                  modes,
                                                                  HIPTENSOR_OP_ADD,
                                                                  compute);
            api.keep(desc);
            inputs = {data(M * N, 1), data(M * N, 2)};
            expected.resize(M * N);
            for(int64_t i = 0; i < M * N; i++)
            {
                expected[i] = alphaValue * inputs[0][i] + gammaValue * inputs[1][i];
            }
            initialOutput = std::vector<float>(M * N, 0.0f);
            break;
        }
        case Op::ElementwiseTrinary:
        {
            // D[m,n] = (alpha * A[m,n] + beta * B[m,n]) + gamma * C[m,n]
            int32_t const modes[] = {'m', 'n'};
            auto          dA      = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dB      = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dC      = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dD      = api.tensor({M, N}, packedStrides({M, N}), f32);
            outcome.planStatus    = hiptensorCreateElementwiseTrinary(api.handle,
                                                                   &desc,
                                                                   dA,
                                                                   modes,
                                                                   id,
                                                                   dB,
                                                                   modes,
                                                                   id,
                                                                   dC,
                                                                   modes,
                                                                   id,
                                                                   dD,
                                                                   modes,
                                                                   HIPTENSOR_OP_ADD,
                                                                   HIPTENSOR_OP_ADD,
                                                                   compute);
            api.keep(desc);
            inputs = {data(M * N, 1), data(M * N, 2), data(M * N, 3)};
            expected.resize(M * N);
            for(int64_t i = 0; i < M * N; i++)
            {
                expected[i] = alphaValue * inputs[0][i] + betaValue * inputs[1][i]
                              + gammaValue * inputs[2][i];
            }
            initialOutput = std::vector<float>(M * N, 0.0f);
            break;
        }
        case Op::Reduction:
        {
            // D[m] = alpha * sum_n A[m,n] + beta * C[m]
            int32_t const modeA[] = {'m', 'n'}, modeD[] = {'m'};
            auto          dA   = api.tensor({M, N}, packedStrides({M, N}), f32);
            auto          dD   = api.tensor({M}, packedStrides({M}), f32);
            outcome.planStatus = hiptensorCreateReduction(api.handle,
                                                          &desc,
                                                          dA,
                                                          modeA,
                                                          id,
                                                          dD,
                                                          modeD,
                                                          id,
                                                          dD,
                                                          modeD,
                                                          HIPTENSOR_OP_ADD,
                                                          compute);
            api.keep(desc);
            inputs = {data(M * N, 1), data(M, 2)};
            expected.resize(M);
            for(int64_t m = 0; m < M; m++)
            {
                float sum = 0.0f;
                for(int64_t n = 0; n < N; n++)
                {
                    sum += inputs[0][m + M * n];
                }
                expected[m] = alphaValue * sum + betaValue * inputs[1][m];
            }
            initialOutput = std::vector<float>(M, 0.0f);
            break;
        }
        }

        if(outcome.planStatus == HIPTENSOR_STATUS_ARCH_MISMATCH
           || outcome.planStatus == HIPTENSOR_STATUS_NOT_SUPPORTED)
        {
            outcome.skipped = true;
            return outcome;
        }
        EXPECT_EQ(outcome.planStatus, HIPTENSOR_STATUS_SUCCESS);
        if(outcome.planStatus != HIPTENSOR_STATUS_SUCCESS)
        {
            return outcome;
        }

        // A declined problem has no estimate; its plan creation reports why.
        uint64_t workspaceSize = 0;
        auto     estimate      = hiptensorEstimateWorkspaceSize(
            api.handle, desc, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize);
        EXPECT_TRUE(estimate == HIPTENSOR_STATUS_SUCCESS
                    || estimate == HIPTENSOR_STATUS_NOT_SUPPORTED
                    || estimate == HIPTENSOR_STATUS_ARCH_MISMATCH)
            << opName(op);

        hiptensorPlan_t plan = nullptr;
        outcome.planStatus   = api.plan(desc, plan, workspaceSize);
        if(outcome.planStatus == HIPTENSOR_STATUS_ARCH_MISMATCH
           || outcome.planStatus == HIPTENSOR_STATUS_NOT_SUPPORTED)
        {
            outcome.skipped = true;
            return outcome;
        }
        EXPECT_EQ(outcome.planStatus, HIPTENSOR_STATUS_SUCCESS) << opName(op);
        EXPECT_EQ(estimate, HIPTENSOR_STATUS_SUCCESS) << opName(op);
        if(outcome.planStatus != HIPTENSOR_STATUS_SUCCESS)
        {
            return outcome;
        }
        uint64_t required = 0;
        EXPECT_EQ(
            hiptensorPlanGetAttribute(
                api.handle, plan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &required, sizeof(required)),
            HIPTENSOR_STATUS_SUCCESS);

        std::vector<std::unique_ptr<DeviceBuffer<float>>> devInputs;
        for(auto const& in : inputs)
        {
            devInputs.push_back(std::make_unique<DeviceBuffer<float>>(in));
        }
        auto  devOut = std::make_unique<DeviceBuffer<float>>(initialOutput);
        void* out    = devOut->ptr;
        if(inPlace)
        {
            // The contraction's C is its output.
            devInputs[2].reset();
        }
        auto input
            = [&](std::size_t i) -> void const* { return devInputs[i] ? devInputs[i]->ptr : out; };

        void* workspace = nullptr;
        if(required > 0)
        {
            EXPECT_EQ(hipMalloc(&workspace, required), hipSuccess);
        }

        float const alpha = alphaValue, beta = betaValue, gamma = gammaValue;
        switch(op)
        {
        case Op::Contraction:
            outcome.execStatus = hiptensorContract(api.handle,
                                                   plan,
                                                   &alpha,
                                                   input(0),
                                                   input(1),
                                                   &beta,
                                                   input(2),
                                                   out,
                                                   workspace,
                                                   required,
                                                   stream);
            break;
        case Op::ContractionTrinary:
            outcome.execStatus = hiptensorContractTrinary(api.handle,
                                                          plan,
                                                          &alpha,
                                                          input(0),
                                                          input(1),
                                                          input(2),
                                                          &beta,
                                                          input(3),
                                                          out,
                                                          workspace,
                                                          required,
                                                          stream);
            break;
        case Op::Permute:
            outcome.execStatus = hiptensorPermute(api.handle, plan, &alpha, input(0), out, stream);
            break;
        case Op::ElementwiseBinary:
            outcome.execStatus = hiptensorElementwiseBinaryExecute(
                api.handle, plan, &alpha, input(0), &gamma, input(1), out, stream);
            break;
        case Op::ElementwiseTrinary:
            outcome.execStatus = hiptensorElementwiseTrinaryExecute(
                api.handle, plan, &alpha, input(0), &beta, input(1), &gamma, input(2), out, stream);
            break;
        case Op::Reduction:
            outcome.execStatus = hiptensorReduce(api.handle,
                                                 plan,
                                                 &alpha,
                                                 input(0),
                                                 &beta,
                                                 input(1),
                                                 out,
                                                 workspace,
                                                 required,
                                                 stream);
            break;
        }
        EXPECT_EQ(outcome.execStatus, HIPTENSOR_STATUS_SUCCESS) << opName(op);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        auto got = devOut->read();
        EXPECT_EQ(got, expected) << opName(op);
        if(workspace != nullptr)
        {
            static_cast<void>(hipFree(workspace));
        }
        return outcome;
    }

    std::string servedBy(char const* backend, Op op)
    {
        return std::string("Backend: ") + backend + ", Op: " + opName(op) + ",";
    }
} // namespace

TEST(BackendSeamTest, DefaultBackendServesEveryOperation)
{
    ScopedEnvironmentVariable backend("HIPTENSOR_BACKEND", nullptr);
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", nullptr);
    for(auto op : allOps)
    {
        LogCapture log;
        auto       outcome = run(op, nullptr);
        if(outcome.skipped)
        {
            // The f32 elementwise and reduction kernels exist for every device; only the
            // contractions may be unsupported.
            EXPECT_TRUE(op == Op::Contraction || op == Op::ContractionTrinary) << opName(op);
            continue;
        }
        EXPECT_TRUE(log.contains(servedBy("ck", op))) << opName(op) << "\n" << log.all();
    }
}

TEST(BackendSeamTest, SelectedBackendServesEveryOperation)
{
    ScopedEnvironmentVariable backend("HIPTENSOR_BACKEND", " CK ");
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", nullptr);
    for(auto op : allOps)
    {
        LogCapture log;
        auto       outcome = run(op, nullptr);
        if(outcome.skipped)
        {
            // The f32 elementwise and reduction kernels exist for every device; only the
            // contractions may be unsupported.
            EXPECT_TRUE(op == Op::Contraction || op == Op::ContractionTrinary) << opName(op);
            continue;
        }
        EXPECT_TRUE(log.contains(servedBy("ck", op))) << opName(op) << "\n" << log.all();
    }
}

TEST(BackendSeamTest, UnknownSelectionIsRefused)
{
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", nullptr);

    char const* const settings[][2] = {{"HIPTENSOR_BACKEND", "nonesuch"},
                                       {"HIPTENSOR_BACKEND_FALLBACK", "sometimes"},
                                       {"HIPTENSOR_BACKEND_AB", "ck"},
                                       {"HIPTENSOR_BACKEND_AB", "ck,nonesuch"}};
    for(auto const& setting : settings)
    {
        ScopedEnvironmentVariable variable(setting[0], setting[1]);

        ApiObjects                     api;
        int32_t const                  modeA[] = {'m', 'n'}, modeB[] = {'n', 'm'};
        auto                           dA   = api.tensor({4, 3}, {}, HIPTENSOR_R_32F);
        auto                           dB   = api.tensor({3, 4}, {}, HIPTENSOR_R_32F);
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &desc,
                                             dA,
                                             modeA,
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             modeB,
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);

        hiptensorPlan_t plan = nullptr;
        EXPECT_EQ(api.plan(desc, plan), HIPTENSOR_STATUS_INVALID_VALUE)
            << setting[0] << "=" << setting[1];
        EXPECT_EQ(plan, nullptr);

        uint64_t estimate = 0;
        EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                      api.handle, desc, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << setting[0] << "=" << setting[1];
    }
}

// The rocKE backend serves the problem, or declines it (or is not built into the library) and
// the fallback backend serves it.
TEST(BackendSeamTest, DeclinedProblemsFallBack)
{
    ScopedEnvironmentVariable backend("HIPTENSOR_BACKEND", "rocke");
    ScopedEnvironmentVariable fallback("HIPTENSOR_BACKEND_FALLBACK", nullptr);
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", nullptr);
    for(auto op : allOps)
    {
        LogCapture log;
        auto       outcome = run(op, nullptr);
        if(outcome.skipped)
        {
            // The f32 elementwise and reduction kernels exist for every device; only the
            // contractions may be unsupported.
            EXPECT_TRUE(op == Op::Contraction || op == Op::ContractionTrinary) << opName(op);
            continue;
        }
        bool const rocke = log.contains(servedBy("rocke", op));
        bool const fellBack
            = log.contains(std::string("Backend rocke declined the ") + opName(op) + " problem")
              && log.contains(servedBy("ck", op));
        EXPECT_TRUE(rocke || fellBack) << opName(op) << "\n" << log.all();
    }
}

TEST(BackendSeamTest, DeclinedProblemsAreNotSupportedWithoutFallback)
{
    ScopedEnvironmentVariable backend("HIPTENSOR_BACKEND", "rocke");
    ScopedEnvironmentVariable fallback("HIPTENSOR_BACKEND_FALLBACK", "0");
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", nullptr);
    for(auto op : allOps)
    {
        LogCapture log;
        auto       outcome = run(op, nullptr);
        if(log.contains(servedBy("rocke", op)))
        {
            // Served, and checked by run().
            continue;
        }
        EXPECT_TRUE(outcome.skipped) << opName(op);
        EXPECT_EQ(outcome.planStatus, HIPTENSOR_STATUS_NOT_SUPPORTED) << opName(op);
        EXPECT_TRUE(
            log.contains(std::string("Backend rocke declined the ") + opName(op) + " problem"))
            << opName(op) << "\n"
            << log.all();
        EXPECT_FALSE(log.contains(servedBy("ck", op))) << opName(op) << "\n" << log.all();
    }
}

TEST(BackendSeamTest, ABRunsBothBackendsWithoutChangingTheResult)
{
    ScopedEnvironmentVariable backend("HIPTENSOR_BACKEND", nullptr);
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", "ck,ck");
    for(auto op : allOps)
    {
        LogCapture log;
        auto       outcome = run(op, nullptr);
        if(outcome.skipped)
        {
            // The f32 elementwise and reduction kernels exist for every device; only the
            // contractions may be unsupported.
            EXPECT_TRUE(op == Op::Contraction || op == Op::ContractionTrinary) << opName(op);
            continue;
        }
        EXPECT_TRUE(log.contains("A/B ck vs ck:")) << opName(op) << "\n" << log.all();
        EXPECT_TRUE(log.contains("max |diff| 0,")) << opName(op) << "\n" << log.all();
    }

    // The shadow backend runs first, so the primary backend still reads the original C.
    LogCapture log;
    auto       outcome = run(Op::Contraction, nullptr, true);
    if(!outcome.skipped)
    {
        EXPECT_TRUE(log.contains("max |diff| 0,")) << log.all();
    }
}

TEST(BackendSeamTest, ExecutionsRunOnTheCallerStream)
{
    ScopedEnvironmentVariable backend("HIPTENSOR_BACKEND", nullptr);
    ScopedEnvironmentVariable ab("HIPTENSOR_BACKEND_AB", nullptr);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    for(auto op : allOps)
    {
        run(op, stream);
    }
    run(Op::Contraction, stream, true);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(BackendSeamTest, KernelCacheCalls)
{
    ApiObjects api;
    auto       fileName = ::testing::TempDir() + "hiptensor_backend_seam_test_kernel_cache.bin";
    EXPECT_EQ(hiptensorWriteKernelCacheToFile(nullptr, fileName.c_str()),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorWriteKernelCacheToFile(api.handle, nullptr), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorReadKernelCacheFromFile(nullptr, fileName.c_str()),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorReadKernelCacheFromFile(api.handle, nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorWriteKernelCacheToFile(api.handle, fileName.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    std::remove(fileName.c_str());
}

// The plan-cache file is format version 2 and names the Composable Kernel backend.
TEST(BackendSeamTest, PlanCacheFileRecordsTheBackends)
{
    char const* disabled = std::getenv("HIPTENSOR_DISABLE_PLAN_CACHE");
    if(disabled != nullptr
       && (std::string(disabled) == "1" || std::string(disabled) == "ON"
           || std::string(disabled) == "on"))
    {
        GTEST_SKIP() << "the plan cache is disabled";
    }

    ApiObjects api;
    auto       fileName = ::testing::TempDir() + "hiptensor_backend_seam_test_plan_cache.bin";
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(api.handle, fileName.c_str()),
              HIPTENSOR_STATUS_SUCCESS);

    std::ifstream     file(fileName, std::ios::in | std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    std::remove(fileName.c_str());

    ASSERT_GT(bytes.size(), 8u);
    EXPECT_EQ(std::string(bytes.data(), 4), "HTPC");
    uint32_t version = 0;
    std::memcpy(&version, bytes.data() + 4, sizeof(version));
    EXPECT_EQ(version, 2u);

    // u32 length 2 followed by "ck".
    std::string const ck = std::string("\x02\x00\x00\x00", 4) + "ck";
    EXPECT_NE(std::string(bytes.begin(), bytes.end()).find(ck), std::string::npos);
}
