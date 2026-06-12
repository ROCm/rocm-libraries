/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

//
// Light, fast unit tests for the FP64 (Ozaki Scheme II) emulation path.
//
//   ./hipblaslt-test --gtest_filter='Fp64Emulation*'
//
// Two fixtures keep setup isolated and reusable as more emulation entry points
// gain coverage:
//   * Fp64EmulationHostTest - pure host helpers, no GPU/handle required.
//   * Fp64EmulationTest      - owns a hipblasLtHandle_t for API-driven tests.
//
// The internal entry points (declared in the rocblaslt-private fp64_emulation.hpp)
// are linkable here because hipblaslt-test privately links the
// hipblaslt-fp64-emulation OBJECT library - the same object files the hipblaslt
// shared library is built from - so no symbols are exported from the release ABI.
//

#include "fp64_emulation.hpp" // internal: functions under test
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h> // public API + emulation setters

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
    bool has_device()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    // -----------------------------------------------------------------------
    // Host-only fixture: pure host helpers, no GPU or handle required.
    // -----------------------------------------------------------------------
    class Fp64EmulationHostTest : public ::testing::Test
    {
    };

    // Default/env-derived moduli count must stay within the supported range.
    TEST_F(Fp64EmulationHostTest, NumModuliInValidRange)
    {
        const unsigned s = fp64EmulationNumModuli();
        EXPECT_GE(s, 2u);
        EXPECT_LE(s, 18u); // OZ2_S_MAX
    }

    // Workspace must be non-empty and not shrink as the moduli count grows.
    TEST_F(Fp64EmulationHostTest, WorkspaceSizePositiveAndMonotonic)
    {
        const int64_t m = 1024, n = 1024, k = 1024;
        const size_t  ws8  = fp64EmulationWorkspaceSize(m, n, k, 8);
        const size_t  ws16 = fp64EmulationWorkspaceSize(m, n, k, 16);
        EXPECT_GT(ws8, 0u);
        EXPECT_GE(ws16, ws8);
    }

    TEST_F(Fp64EmulationHostTest, ParseEnabledEnv)
    {
        EXPECT_EQ(fp64EmulationParseEnabledEnv(nullptr).state, FP64_EMULATION_ENV_UNSET);

        auto on = fp64EmulationParseEnabledEnv("1");
        EXPECT_EQ(on.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(on.value, 1u);

        auto off = fp64EmulationParseEnabledEnv("0");
        EXPECT_EQ(off.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(off.value, 0u);

        EXPECT_EQ(fp64EmulationParseEnabledEnv("true").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseEnabledEnv("").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, ParseStrategyEnv)
    {
        EXPECT_EQ(fp64EmulationParseStrategyEnv(nullptr).state, FP64_EMULATION_ENV_UNSET);

        auto performant = fp64EmulationParseStrategyEnv("performant");
        EXPECT_EQ(performant.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(performant.value,
                  static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT));

        auto eager = fp64EmulationParseStrategyEnv("eager");
        EXPECT_EQ(eager.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(eager.value, static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER));

        EXPECT_EQ(fp64EmulationParseStrategyEnv("default").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseStrategyEnv("EAGER").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, ParseSpecialValuesMaskEnv)
    {
        auto unset = fp64EmulationParseSpecialValuesMaskEnv(nullptr);
        EXPECT_EQ(unset.state, FP64_EMULATION_ENV_UNSET);
        EXPECT_EQ(unset.value, 0x3u);

        auto hex = fp64EmulationParseSpecialValuesMaskEnv("0x3");
        EXPECT_EQ(hex.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(hex.value, 0x3u);

        auto zero = fp64EmulationParseSpecialValuesMaskEnv("0");
        EXPECT_EQ(zero.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(zero.value, 0u);

        EXPECT_EQ(fp64EmulationParseSpecialValuesMaskEnv("-1").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseSpecialValuesMaskEnv("3x").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, ParseMantissaBitCountEnv)
    {
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv(nullptr).state, FP64_EMULATION_ENV_UNSET);

        auto bits55 = fp64EmulationParseMantissaBitCountEnv("55");
        EXPECT_EQ(bits55.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(bits55.value, 55u);

        auto bits140 = fp64EmulationParseMantissaBitCountEnv("140");
        EXPECT_EQ(bits140.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(bits140.value, 140u);

        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("141").state,
                  FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("-1").state,
                  FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("55.0").state,
                  FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, MantissaBitCountRange)
    {
        EXPECT_TRUE(fp64EmulationIsValidMantissaBitCount(-1));
        EXPECT_TRUE(fp64EmulationIsValidMantissaBitCount(0));
        EXPECT_TRUE(fp64EmulationIsValidMantissaBitCount(140));
        EXPECT_FALSE(fp64EmulationIsValidMantissaBitCount(-2));
        EXPECT_FALSE(fp64EmulationIsValidMantissaBitCount(141));
    }

    // -----------------------------------------------------------------------
    // Handle fixture: isolated setup/teardown, reusable by future tests.
    // -----------------------------------------------------------------------
    class Fp64EmulationTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(!has_device())
                GTEST_SKIP() << "No HIP device available";
            ASSERT_EQ(hipblasLtCreate(&m_handle), HIPBLAS_STATUS_SUCCESS);
            m_roc = reinterpret_cast<const _rocblaslt_handle*>(m_handle);
        }

        void TearDown() override
        {
            if(m_handle)
                hipblasLtDestroy(m_handle);
        }

        void set_enabled(bool on)
        {
            ASSERT_EQ(hipblasLtSetEmulationEnabled(m_handle, on), HIPBLAS_STATUS_SUCCESS);
        }

        void set_strategy(hipblasLtEmulationStrategy_t s)
        {
            ASSERT_EQ(hipblasLtSetEmulationStrategy(m_handle, s), HIPBLAS_STATUS_SUCCESS);
        }

        bool would_apply(hipDataType t, int64_t m, int64_t n, int64_t k, int batch)
        {
            const Fp64EmulationDecision decision =
                fp64EmulationDecision(m_roc, t, m, n, k, batch);
            EXPECT_EQ(decision.status, rocblaslt_status_success);
            return decision.apply;
        }

        hipblasLtHandle_t        m_handle = nullptr;
        const _rocblaslt_handle* m_roc    = nullptr;
    };

    // Explicit "off" must win regardless of the environment variable.
    TEST_F(Fp64EmulationTest, WouldApply_ForcedOffReturnsFalse)
    {
        set_enabled(false);
        EXPECT_FALSE(would_apply(HIP_R_64F, 4096, 4096, 4096, 1));
    }

    // Enabled + EAGER intercepts small DGEMMs.
    TEST_F(Fp64EmulationTest, WouldApply_EnabledEagerSmallF64)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_TRUE(would_apply(HIP_R_64F, 16, 16, 16, 1));
    }

    // Enabled + EAGER bypasses the cost model, so a large FP64 GEMM is intercepted.
    TEST_F(Fp64EmulationTest, WouldApply_EnabledEagerLargeF64)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_TRUE(would_apply(HIP_R_64F, 4096, 4096, 4096, 1));
    }

    // Only FP64 inputs are eligible for emulation.
    TEST_F(Fp64EmulationTest, WouldApply_RejectsNonF64)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_FALSE(would_apply(HIP_R_32F, 4096, 4096, 4096, 1));
    }

    // Batched GEMM is not supported by the emulation path.
    TEST_F(Fp64EmulationTest, WouldApply_RejectsBatched)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_FALSE(would_apply(HIP_R_64F, 4096, 4096, 4096, 2));
    }

    TEST_F(Fp64EmulationTest, ApiValidationRejectsInvalidMantissaControl)
    {
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, static_cast<hipblasEmulationMantissaControl_t>(-1)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, static_cast<hipblasEmulationMantissaControl_t>(2)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(Fp64EmulationTest, ApiValidationRejectsInvalidMantissaBitCount)
    {
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, -2),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, 141),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, -1),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, 0),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, 140),
                  HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(Fp64EmulationTest, DefaultDynamicDecisionUsesCurrentSixteenModuliPath)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        ASSERT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);

        const Fp64EmulationDecision decision =
            fp64EmulationDecision(m_roc, HIP_R_64F, 4096, 4096, 4096, 1);
        ASSERT_EQ(decision.status, rocblaslt_status_success);
        ASSERT_TRUE(decision.apply);
        EXPECT_TRUE(decision.dynamic_mode);
        EXPECT_EQ(decision.num_moduli, 16u);
    }

    // End-to-end smoke test of the emulated GEMM. Column-major
    // D = alpha*op(A)*op(B) + beta*C, with A = I and beta = 0, so D == B.
    // Skips when the INT8 device library is unavailable for the running arch.
    TEST_F(Fp64EmulationTest, EmulatedGemm_SmokeIdentity)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);

        constexpr int64_t N     = 64;
        const size_t      bytes = static_cast<size_t>(N) * N * sizeof(double);

        std::vector<double> hA(N * N, 0.0), hB(N * N, 0.0), hD(N * N, 0.0);
        for(int64_t i = 0; i < N; ++i)
            hA[i * N + i] = 1.0; // identity
        for(int64_t i = 0; i < N * N; ++i)
            hB[i] = static_cast<double>((i % 7) - 3);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        const double          alpha = 1.0, beta = 0.0;
        Fp64EmulationSettings settings{};
        settings.num_moduli      = 0; // derive from env/default
        settings.sv_mask         = 0; // skip Inf/NaN check (faster, inputs are finite)
        settings.workspace       = nullptr; // library allocates
        settings.workspace_bytes = 0;
        settings.handle          = m_handle;

        const rocblaslt_status st = fp64EmulatedGemm(HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     N,
                                                     N,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     N,
                                                     dB,
                                                     N,
                                                     &beta,
                                                     dC,
                                                     N,
                                                     dD,
                                                     N,
                                                     /*stream=*/nullptr,
                                                     settings);

        if(st != rocblaslt_status_success)
        {
            (void)hipFree(dA);
            (void)hipFree(dB);
            (void)hipFree(dC);
            (void)hipFree(dD);
            GTEST_SKIP() << "fp64EmulatedGemm returned non-success (" << static_cast<int>(st)
                         << "); INT8 device library may be unavailable on this arch";
        }

        ASSERT_EQ(hipMemcpy(hD.data(), dD, bytes, hipMemcpyDeviceToHost), hipSuccess);

        double max_abs_err = 0.0;
        for(int64_t i = 0; i < N * N; ++i)
            max_abs_err = std::max(max_abs_err, std::abs(hD[i] - hB[i]));
        EXPECT_LT(max_abs_err, 1e-9);

        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
    }

} // namespace
