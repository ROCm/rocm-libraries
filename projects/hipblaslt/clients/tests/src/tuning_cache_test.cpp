/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// User-driven tuning cache tests.
//
// Everything here drives the public API with HIPBLASLT_TUNING_MODE and
// HIPBLASLT_TUNING_CACHE_PATH, and asserts on the library's own hit / miss /
// invalidated / tuned counters rather than on log text.
//
// The mode switch is latched on first use and the loaded-path set lives for the
// process, so each test starts by calling hipblaslt_tuning_reset_for_test().
// Without it, only the first mode set in the binary would ever take effect.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

extern "C" void hipblaslt_tuning_reset_for_test();
extern "C" void hipblaslt_tuning_counters_for_test(uint64_t* loaded,
                                                   uint64_t* hits,
                                                   uint64_t* misses,
                                                   uint64_t* invalidated,
                                                   uint64_t* tuned,
                                                   uint64_t* skipped);

#ifdef WIN32
static int setenv(const char* name, const char* value, int overwrite)
{
    return _putenv_s(name, value);
}
static int unsetenv(const char* name)
{
    return _putenv_s(name, "");
}
#endif

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    struct Counters
    {
        uint64_t loaded = 0, hits = 0, misses = 0, invalidated = 0, tuned = 0, skipped = 0;
    };

    Counters counters()
    {
        Counters c;
        hipblaslt_tuning_counters_for_test(
            &c.loaded, &c.hits, &c.misses, &c.invalidated, &c.tuned, &c.skipped);
        return c;
    }

    std::string tempCachePath(const char* stem)
    {
        std::ostringstream oss;
#ifdef WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        oss << "hipblaslt_" << stem << "_" << static_cast<long long>(pid) << ".tuning";
        return oss.str();
    }

    void enterMode(const char* mode, const std::string& path)
    {
        if(mode)
            setenv("HIPBLASLT_TUNING_MODE", mode, 1);
        else
            unsetenv("HIPBLASLT_TUNING_MODE");

        if(path.empty())
            unsetenv("HIPBLASLT_TUNING_CACHE_PATH");
        else
            setenv("HIPBLASLT_TUNING_CACHE_PATH", path.c_str(), 1);

        // These tests check the cache mechanism, not tuning quality, so keep the
        // search small. The shipping defaults measure every kernel over 2000
        // launches apiece, which costs minutes across the suite for no extra
        // coverage here.
        setenv("HIPBLASLT_TUNING_ALL_KERNELS", "0", 1);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "16", 1);
        setenv("HIPBLASLT_TUNING_COLD_ITERS", "2", 1);
        setenv("HIPBLASLT_TUNING_HOT_ITERS", "5", 1);
        setenv("HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE", "0", 1);
        setenv("HIPBLASLT_TUNING_SCRATCH_MAX_BYTES", "1073741824", 1);

        // Left at the shipping default so the flush path is exercised, but the
        // per-device calibration burst is the dominant cost at these tiny
        // iteration counts, so it is measured once and reused across tests.
        setenv("HIPBLASLT_TUNING_FLUSH_ICACHE", "1", 1);

        // Large enough to rotate over several blocks rather than collapsing to
        // one. runGemm asks for a 32 MiB workspace, and the workspace is part of
        // the per-block footprint, so a budget below that silently disables
        // rotation and leaves the seeding and per-block pointer arithmetic
        // untested. Still far below the 512 MiB default.
        setenv("HIPBLASLT_TUNING_ROTATING_MB", "160", 1);

        hipblaslt_tuning_reset_for_test();
    }

    /** fp16 1.0 and the exact fp16 encoding of a small whole number. */
    uint16_t halfBits(float value)
    {
        const _Float16 h = static_cast<_Float16>(value);
        uint16_t       bits;
        std::memcpy(&bits, &h, sizeof(bits));
        return bits;
    }

    /**
     * One fp16 GEMM through the public C API, heuristic then matmul.
     *
     * verifyProduct fills A and B with ones so the result is a real product
     * rather than zeros, and checks it. Without it these tests pass even if the
     * tuned winner is never launched, because zero inputs make every possible
     * answer identical.
     */
    bool runGemm(int64_t m,
                 int64_t n,
                 int64_t k,
                 float   betaValue     = 0.0f,
                 bool    inPlace       = true,
                 bool    verifyFirst   = false,
                 bool    verifyProduct = false)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dC      = nullptr;
        void*        dD      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, k * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, m * n * sizeof(uint16_t)) == hipSuccess
                  && (inPlace || hipMalloc(&dD, m * n * sizeof(uint16_t)) == hipSuccess)
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;

        if(ok)
        {
            ok = hipMemset(dA, 0, m * k * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dB, 0, k * n * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dC, 0, m * n * sizeof(uint16_t)) == hipSuccess
                 && (inPlace || hipMemset(dD, 0, m * n * sizeof(uint16_t)) == hipSuccess);
            if(ok && verifyFirst)
            {
                // IEEE fp16 1.0. A and B are zero, so beta=2 must turn this
                // element into fp16 2.0 (0x4000) in the caller's final launch.
                const uint16_t one = 0x3c00;
                ok = hipMemcpy(dC, &one, sizeof(one), hipMemcpyHostToDevice) == hipSuccess;
            }
            if(ok && verifyProduct)
            {
                const std::vector<uint16_t> onesA(static_cast<size_t>(m * k), halfBits(1.0f));
                const std::vector<uint16_t> onesB(static_cast<size_t>(k * n), halfBits(1.0f));
                ok = hipMemcpy(
                         dA, onesA.data(), onesA.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                         == hipSuccess
                     && hipMemcpy(dB,
                                  onesB.data(),
                                  onesB.size() * sizeof(uint16_t),
                                  hipMemcpyHostToDevice)
                            == hipSuccess;
            }
        }

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        if(ok)
        {
            ok = hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            const uint64_t maxWs = wsBytes;
            ok                   = hipblasLtMatmulPreferenceSetAttribute(
                     pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                 == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            hipblasLtMatmulHeuristicResult_t heuristic[1];
            int                              returned = 0;
            ok                                        = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                 desc,
                                                 layoutA,
                                                 layoutB,
                                                 layoutC,
                                                 layoutC,
                                                 pref,
                                                 1,
                                                 heuristic,
                                                 &returned)
                     == HIPBLAS_STATUS_SUCCESS
                 && returned > 0;

            if(ok)
            {
                const float alpha = 1.0f;
                ok                = hipblasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     dA,
                                     layoutA,
                                     dB,
                                     layoutB,
                                     &betaValue,
                                     dC,
                                     layoutC,
                                     inPlace ? dC : dD,
                                     layoutC,
                                     &heuristic[0].algo,
                                     dWs,
                                     wsBytes,
                                     nullptr)
                     == HIPBLAS_STATUS_SUCCESS;
                ok = ok && hipDeviceSynchronize() == hipSuccess;
                if(ok && verifyFirst)
                {
                    uint16_t first = 0;
                    ok = hipMemcpy(&first, inPlace ? dC : dD, sizeof(first), hipMemcpyDeviceToHost)
                             == hipSuccess
                         && first == 0x4000;
                }
                if(ok && verifyProduct)
                {
                    // Every element is a sum of k products of one, accumulated
                    // in fp32, so the fp16 result is exactly k for the small k
                    // used here.
                    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
                    ok = hipMemcpy(out.data(),
                                   inPlace ? dC : dD,
                                   out.size() * sizeof(uint16_t),
                                   hipMemcpyDeviceToHost)
                         == hipSuccess;

                    const uint16_t expected = halfBits(static_cast<float>(k));
                    for(size_t i = 0; ok && i < out.size(); i++)
                        ok = out[i] == expected;
                }
            }
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        if(layoutC)
            hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);

        return ok;
    }

    /**
     * One FP8 GEMM, or false if this device has no FP8 solution for it.
     *
     * Separate from runGemm because FP8 needs one-byte inputs, an fp16 output
     * and a transposed A, and because it exists only to prove that the datatype
     * a row records is the datatype it parses back as.
     */
    bool runFp8Gemm(int64_t m, int64_t n, int64_t k, bool* unsupported)
    {
        *unsupported = false;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dD      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k) == hipSuccess && hipMalloc(&dB, k * n) == hipSuccess
                  && hipMalloc(&dD, m * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;
        if(ok)
            ok = hipMemset(dA, 0, m * k) == hipSuccess && hipMemset(dB, 0, k * n) == hipSuccess
                 && hipMemset(dD, 0, m * n * sizeof(uint16_t)) == hipSuccess;

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutD = nullptr;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        if(ok)
        {
            // transA = T, so A is stored k by m.
            const hipblasOperation_t opT = HIPBLAS_OP_T;
            const hipblasOperation_t opN = HIPBLAS_OP_N;
            ok = hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_8F_E4M3_FNUZ, k, m, k)
                     == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_8F_E4M3_FNUZ, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            const uint64_t maxWs = wsBytes;
            ok                   = hipblasLtMatmulPreferenceSetAttribute(
                     pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                 == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            hipblasLtMatmulHeuristicResult_t heuristic[1];
            int                              returned = 0;
            const bool                       gotAlgo  = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                 desc,
                                                                 layoutA,
                                                                 layoutB,
                                                                 layoutD,
                                                                 layoutD,
                                                                 pref,
                                                                 1,
                                                                 heuristic,
                                                                 &returned)
                                     == HIPBLAS_STATUS_SUCCESS
                                 && returned > 0;

            if(!gotAlgo)
            {
                *unsupported = true;
            }
            else
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                ok                = hipblasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     dA,
                                     layoutA,
                                     dB,
                                     layoutB,
                                     &beta,
                                     dD,
                                     layoutD,
                                     dD,
                                     layoutD,
                                     &heuristic[0].algo,
                                     dWs,
                                     wsBytes,
                                     nullptr)
                     == HIPBLAS_STATUS_SUCCESS;
                ok = ok && hipDeviceSynchronize() == hipSuccess;
            }
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        if(layoutD)
            hipblasLtMatrixLayoutDestroy(layoutD);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);

        return ok;
    }

    std::vector<std::string> readLines(const std::string& path)
    {
        std::vector<std::string> lines;
        std::ifstream            in(path);
        std::string              line;
        while(std::getline(in, line))
            lines.push_back(line);
        return lines;
    }

    /** Overwrite one named column in every value row. */
    bool rewriteColumn(const std::string& path, const std::string& column, const std::string& value)
    {
        auto lines = readLines(path);
        if(lines.empty())
            return false;

        auto split = [](const std::string& s) {
            std::vector<std::string> out;
            std::stringstream        ss(s);
            std::string              cell;
            while(std::getline(ss, cell, ','))
            {
                const auto b = cell.find_first_not_of(" \t");
                const auto e = cell.find_last_not_of(" \t");
                out.push_back(b == std::string::npos ? "" : cell.substr(b, e - b + 1));
            }
            return out;
        };

        bool changed = false;
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            const auto names  = split(lines[i]);
            auto       values = split(lines[i + 1]);
            for(size_t c = 0; c < names.size() && c < values.size(); c++)
            {
                if(names[c] == column)
                {
                    values[c] = value;
                    changed   = true;
                }
            }

            std::ostringstream rebuilt;
            for(size_t c = 0; c < values.size(); c++)
                rebuilt << (c ? "," : "") << values[c];
            lines[i + 1] = rebuilt.str();
            i++;
        }

        if(!changed)
            return false;

        std::ofstream out(path, std::ios::trunc);
        for(const auto& l : lines)
            out << l << "\n";
        return true;
    }

    bool fileHasColumn(const std::string& path, const std::string& column)
    {
        for(const auto& line : readLines(path))
            if(line.find("transA") != std::string::npos && line.find(column) != std::string::npos)
                return true;
        return false;
    }

    size_t valueRowCount(const std::string& path)
    {
        size_t     n     = 0;
        const auto lines = readLines(path);
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") != std::string::npos)
            {
                n++;
                i++;
            }
        }
        return n;
    }

    class TuningCache : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            static const char* const names[] = {"HIPBLASLT_TUNING_MODE",
                                                "HIPBLASLT_TUNING_CACHE_PATH",
                                                "HIPBLASLT_TUNING_OVERRIDE_FILE",
                                                "HIPBLASLT_TUNING_ALL_KERNELS",
                                                "HIPBLASLT_TUNING_MAX_CANDIDATES",
                                                "HIPBLASLT_TUNING_COLD_ITERS",
                                                "HIPBLASLT_TUNING_HOT_ITERS",
                                                "HIPBLASLT_TUNING_ROTATING_MB",
                                                "HIPBLASLT_TUNING_FLUSH_ICACHE",
                                                "HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE",
                                                "HIPBLASLT_TUNING_SCRATCH_MAX_BYTES"};
            for(const char* name : names)
            {
                const char* value = getenv(name);
                m_savedEnv.emplace_back(name,
                                        value ? std::optional<std::string>(value) : std::nullopt);
            }

            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";

            // A user's legacy override must not leak into a cache-mode test.
            unsetenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
            m_path = tempCachePath(::testing::UnitTest::GetInstance()->current_test_info()->name());
            std::remove(m_path.c_str());
        }

        void TearDown() override
        {
            std::remove(m_path.c_str());
            for(const auto& [name, value] : m_savedEnv)
            {
                if(value)
                    setenv(name.c_str(), value->c_str(), 1);
                else
                    unsetenv(name.c_str());
            }
            hipblaslt_tuning_reset_for_test();
        }

        std::string m_path;
        std::vector<std::pair<std::string, std::optional<std::string>>> m_savedEnv;
    };

    // Nothing is written and nothing is consulted when tuning is not asked for.
    TEST_F(TuningCache, OffModeWritesNothing)
    {
        enterMode("off", m_path);
        ASSERT_TRUE(runGemm(256, 256, 256));

        std::ifstream probe(m_path);
        EXPECT_FALSE(probe.good()) << "off mode created " << m_path;

        const auto c = counters();
        EXPECT_EQ(c.tuned, 0u);
        EXPECT_EQ(c.hits, 0u);
    }

    // cache/tune do nothing without somewhere to keep results, and must not
    // fall back to the legacy override variable.
    TEST_F(TuningCache, CacheModeWithoutPathIsInert)
    {
        enterMode("cache", "");
        EXPECT_TRUE(runGemm(256, 256, 256));

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u);
        EXPECT_EQ(c.tuned, 0u);
    }

    // A tuned row must carry everything a later lookup rebuilds the key from,
    // plus the identity and baseline fields.
    TEST_F(TuningCache, TuneWritesRowWithIdentityAndBaseline)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        // Asserted rather than skipped. Selection takes the fastest candidate
        // outright and records even when that is the default pick, so on a
        // working GPU tune mode always writes a row for this shape; no row means
        // the benchmarker or the writer is broken, which is exactly what these
        // tests exist to catch. Skipping instead let every one of them go green
        // on a build where tuning silently did nothing.
        ASSERT_TRUE(std::ifstream(m_path).good()) << "tune mode recorded nothing";

        EXPECT_TRUE(fileHasColumn(m_path, "kernel_name"));
        EXPECT_TRUE(fileHasColumn(m_path, "schema_version"));
        EXPECT_TRUE(fileHasColumn(m_path, "baseline_index"));
        EXPECT_TRUE(fileHasColumn(m_path, "compute_input_type_a"));
        EXPECT_TRUE(fileHasColumn(m_path, "gcnArchName"));
    }

    // Exercise the shipping-default exhaustive enumeration independently from
    // the fast ranked-prefix settings used by the rest of this suite.
    TEST_F(TuningCache, ExhaustiveEnumerationWritesEntry)
    {
        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_ALL_KERNELS", "1", 1);
        setenv("HIPBLASLT_TUNING_COLD_ITERS", "0", 1);
        setenv("HIPBLASLT_TUNING_HOT_ITERS", "1", 1);
        setenv("HIPBLASLT_TUNING_ROTATING_MB", "0", 1);
        hipblaslt_tuning_reset_for_test();

        ASSERT_TRUE(runGemm(64, 64, 64));
        ASSERT_GT(valueRowCount(m_path), 0u) << "exhaustive tuning recorded nothing";
    }

    // The winner has to actually run on the caller's buffers and be right.
    //
    // Separate C and D, and nonzero inputs, because the rest of the suite tunes
    // in-place with zeroed operands: there every candidate and every bug
    // produces the same all-zero output, so a tuning path that reported success
    // without ever launching the winner would still pass.
    TEST_F(TuningCache, TunedLaunchProducesCorrectResult)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(128, 128, 128, 0.0f, false, false, true));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(128, 128, 128, 0.0f, false, false, true));
        EXPECT_GE(counters().hits, 1u) << "tuned entry did not replay";
    }

    // A recorded datatype must parse back as the datatype it was written from.
    //
    // The FNUZ and OCP FP8 types share one bench spelling, "f8_r", which the
    // parser resolves to the OCP type. gfx942 FP8 is FNUZ, so every FP8 row it
    // wrote described a different type than the one measured: the row never
    // matched its own key, the shape re-tuned on every process start, and the
    // cache file grew a duplicate each time.
    TEST_F(TuningCache, Fp8FnuzEntryReplays)
    {
        bool unsupported = false;

        enterMode("tune", m_path);
        ASSERT_TRUE(runFp8Gemm(512, 512, 512, &unsupported));
        if(unsupported)
            GTEST_SKIP() << "no FP8 FNUZ solution on this device";
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runFp8Gemm(512, 512, 512, &unsupported));
        EXPECT_GE(counters().hits, 1u) << "FP8 entry did not replay";

        // The shape is already tuned, so a second tune run must not append.
        const size_t rows = valueRowCount(m_path);
        enterMode("tune", m_path);
        ASSERT_TRUE(runFp8Gemm(512, 512, 512, &unsupported));
        EXPECT_EQ(valueRowCount(m_path), rows) << "FP8 shape re-tuned and appended a duplicate";
    }

    // In-place nonzero-beta timing repeatedly feeds D back as C. Until the
    // timing loop can reset each reused block outside its event span, tuning
    // must decline while still running the caller's real GEMM correctly.
    TEST_F(TuningCache, InPlaceNonzeroBetaRunsWithoutTuning)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(384, 256, 128, 2.0f, true, true));
        EXPECT_EQ(valueRowCount(m_path), 0u);
        EXPECT_GE(counters().skipped, 1u);
    }

    // The round trip the whole feature exists for.
    TEST_F(TuningCache, TunedEntryReplaysInCacheMode)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "tuned entry did not replay";
        EXPECT_EQ(c.invalidated, 0u);
    }

    // An entry whose recorded kernel is not what the index resolves to must be
    // refused, and the shape must still run.
    TEST_F(TuningCache, TamperedKernelNameIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));

        enterMode("cache", m_path);
        EXPECT_TRUE(runGemm(1024, 512, 1024)) << "rejecting an entry must not fail the call";

        const auto c = counters();
        EXPECT_GE(c.invalidated, 1u);
        EXPECT_EQ(c.hits, 0u);
    }

    // Re-tuning must be possible once every entry for a shape has gone stale,
    // and the replacement must survive a reload of the file.
    TEST_F(TuningCache, InvalidatedEntryIsRetunedAndReplays)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const size_t before = valueRowCount(m_path);
        ASSERT_GT(before, 0u) << "tune mode recorded nothing";
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GT(valueRowCount(m_path), before) << "stale entry was never replaced";

        // Fresh load, as a later process would see it.
        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "replacement entry did not replay after reload";
    }

    // A stale row must not suppress the fresh row that supersedes it when the two
    // share a solution index.
    //
    // InvalidatedEntryIsRetunedAndReplays exercises the same append-and-reload
    // path, but only catches this when re-tuning happens to land on a different
    // index, which on a noisy shape it usually does. Here the two rows are the
    // same recorded row with only the name changed, so the collision is
    // guaranteed and the load order is the one the file format produces:
    // superseded row first, replacement after.
    TEST_F(TuningCache, StaleRowDoesNotSuppressItsReplacement)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        auto lines = readLines(m_path);
        ASSERT_GE(lines.size(), 2u) << "tune mode recorded nothing";

        // Prepend a copy carrying a name no kernel can have. Same index, same
        // problem key, so the only thing separating the two rows is the name.
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));
        auto stale = readLines(m_path);

        std::ofstream out(m_path, std::ios::trunc);
        for(const auto& l : stale)
            out << l << "\n";
        for(const auto& l : lines)
            out << l << "\n";
        out.close();

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "stale row hid the valid row that followed it";
    }

    // A row written by a newer hipBLASLt must be ignored, not read as if it were
    // this build's format.
    //
    // A future schema keys on fields this parser cannot see, so interpreting one
    // as the current version matches on the subset it happens to recognise and
    // applies a kernel chosen for a problem that differs in the rest. The
    // version was previously accepted as current whenever it was at least the
    // current number, so this row replayed.
    TEST_F(TuningCache, UnknownSchemaVersionIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        ASSERT_TRUE(rewriteColumn(m_path, "schema_version", "99"));

        enterMode("cache", m_path);
        EXPECT_TRUE(runGemm(1024, 512, 1024)) << "ignoring a row must not fail the call";

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u) << "a row from an unknown schema was replayed";
    }

    // Some strings are recognized as hipDataType values but are not legal GEMM
    // tensor types. They must be rejected before the assert-based Tensile
    // converter is called.
    TEST_F(TuningCache, UnsupportedCurrentDatatypeIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(rewriteColumn(m_path, "a_type", "e8_r"));

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(counters().hits, 0u);
    }

    // A current-schema row missing a value for a key field must be dropped.
    //
    // stride_a is blanked because this shape's real batch stride is 0, so the
    // empty cell parses to a value that matches and the row replays. That is the
    // hazard: an empty cell is indistinguishable from a real zero, and a row
    // truncated in one field is not trustworthy in the others either.
    TEST_F(TuningCache, CurrentSchemaRowWithBlankKeyFieldIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        ASSERT_TRUE(rewriteColumn(m_path, "stride_a", ""));

        enterMode("cache", m_path);
        EXPECT_TRUE(runGemm(1024, 512, 1024)) << "ignoring a row must not fail the call";

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u) << "a row with no value for a key field was replayed";
    }

    TEST_F(TuningCache, CurrentSchemaNumericPrefixIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(rewriteColumn(m_path, "m", "1024junk"));

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(counters().hits, 0u);
    }

    // Recover after a process dies between writing a header and its value row.
    // The orphan must not consume the next complete record's header.
    TEST_F(TuningCache, TornHeaderDoesNotHideFollowingEntry)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        const auto complete = readLines(m_path);
        ASSERT_GE(complete.size(), 3u);

        {
            std::ofstream out(m_path, std::ios::trunc);
            out << complete[0] << "\n"
                << complete[1] << "\n"
                << complete[1] << "\n"
                << complete[2] << "\n";
        }

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GE(counters().hits, 1u);
    }

    // Sequential tuning on two devices must allocate and reuse scratch in each
    // device's address space rather than handing device 1 the pointer allocated
    // while device 0 was current.
    TEST_F(TuningCache, ScratchIsDeviceLocal)
    {
        int deviceCount = 0;
        ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
        if(deviceCount < 2)
            GTEST_SKIP() << "A second GPU is required";

        int originalDevice = 0;
        ASSERT_EQ(hipGetDevice(&originalDevice), hipSuccess);
        struct RestoreDevice
        {
            int device;
            ~RestoreDevice()
            {
                static_cast<void>(hipSetDevice(device));
            }
        } restore{originalDevice};

        enterMode("tune", m_path);
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_TRUE(runGemm(320, 256, 128));
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        ASSERT_TRUE(runGemm(384, 256, 128));

        EXPECT_EQ(valueRowCount(m_path), 2u);
    }

    // Files written before the widened key must keep matching on the fields the
    // old format recorded.
    TEST_F(TuningCache, LegacyRowStillMatches)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        auto lines = readLines(m_path);
        ASSERT_GE(lines.size(), 3u) << "tune mode recorded nothing";

        // Reduce the row to the historical column set: the ten problem columns
        // plus solution_index, with no schema_version and no name of either kind.
        static const char* kLegacy[] = {"transA",
                                        "transB",
                                        "batch_count",
                                        "m",
                                        "n",
                                        "k",
                                        "a_type",
                                        "b_type",
                                        "c_type",
                                        "compute_type",
                                        "solution_index"};

        auto split = [](const std::string& s) {
            std::vector<std::string> out;
            std::stringstream        ss(s);
            std::string              cell;
            while(std::getline(ss, cell, ','))
            {
                const auto b = cell.find_first_not_of(" \t");
                const auto e = cell.find_last_not_of(" \t");
                out.push_back(b == std::string::npos ? "" : cell.substr(b, e - b + 1));
            }
            return out;
        };

        std::string header, values;
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            const auto names = split(lines[i]);
            const auto vals  = split(lines[i + 1]);

            std::ostringstream h, v;
            bool               first = true;
            for(const char* want : kLegacy)
            {
                for(size_t c = 0; c < names.size() && c < vals.size(); c++)
                {
                    if(names[c] != want)
                        continue;
                    h << (first ? "" : ",") << names[c];
                    v << (first ? "" : ",") << vals[c];
                    first = false;
                }
            }
            header = h.str();
            values = v.str();
            break;
        }
        ASSERT_FALSE(header.empty());

        {
            std::ofstream out(m_path, std::ios::trunc);
            out << lines[0] << "\n    " << header << "\n" << values << "\n";
        }

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.loaded, 1u) << "legacy row was not loaded";
        EXPECT_GE(c.hits, 1u) << "legacy row did not match";
    }
} // namespace
