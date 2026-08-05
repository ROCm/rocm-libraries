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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
        oss << "hipblaslt_" << stem << "_" << static_cast<long long>(::getpid()) << ".tuning";
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

        hipblaslt_tuning_reset_for_test();
    }

    /** One fp16 GEMM through the public C API, heuristic then matmul. */
    bool runGemm(int64_t m, int64_t n, int64_t k)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dC      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, k * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, m * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;

        if(ok)
        {
            ok = hipMemset(dA, 0, m * k * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dB, 0, k * n * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dC, 0, m * n * sizeof(uint16_t)) == hipSuccess;
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
                const float beta  = 0.0f;
                ok                = hipblasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     dA,
                                     layoutA,
                                     dB,
                                     layoutB,
                                     &beta,
                                     dC,
                                     layoutC,
                                     dC,
                                     layoutC,
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
        if(layoutC)
            hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dC));
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
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";
            m_path = tempCachePath(::testing::UnitTest::GetInstance()->current_test_info()->name());
            std::remove(m_path.c_str());
        }

        void TearDown() override
        {
            std::remove(m_path.c_str());
            enterMode(nullptr, "");
        }

        std::string m_path;
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

        std::ifstream probe(m_path);
        if(!probe.good())
            GTEST_SKIP() << "no winner cleared the improvement margin for this shape";

        EXPECT_TRUE(fileHasColumn(m_path, "solution_name"));
        EXPECT_TRUE(fileHasColumn(m_path, "schema_version"));
        EXPECT_TRUE(fileHasColumn(m_path, "baseline_index"));
        EXPECT_TRUE(fileHasColumn(m_path, "compute_input_type_a"));
        EXPECT_TRUE(fileHasColumn(m_path, "gcnArchName"));
    }

    // The round trip the whole feature exists for.
    TEST_F(TuningCache, TunedEntryReplaysInCacheMode)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        if(valueRowCount(m_path) == 0)
            GTEST_SKIP() << "no winner cleared the improvement margin for this shape";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "tuned entry did not replay";
        EXPECT_EQ(c.invalidated, 0u);
    }

    // An entry whose recorded kernel is not what the index resolves to must be
    // refused, and the shape must still run.
    TEST_F(TuningCache, TamperedSolutionNameIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        if(valueRowCount(m_path) == 0)
            GTEST_SKIP() << "no winner cleared the improvement margin for this shape";
        ASSERT_TRUE(rewriteColumn(m_path, "solution_name", "NotARealKernelName"));

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
        if(before == 0)
            GTEST_SKIP() << "no winner cleared the improvement margin for this shape";
        ASSERT_TRUE(rewriteColumn(m_path, "solution_name", "NotARealKernelName"));

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GT(valueRowCount(m_path), before) << "stale entry was never replaced";

        // Fresh load, as a later process would see it.
        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "replacement entry did not replay after reload";
    }

    // Files written before the widened key must keep matching on the fields the
    // old format recorded.
    TEST_F(TuningCache, LegacyRowStillMatches)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        auto lines = readLines(m_path);
        if(lines.size() < 3)
            GTEST_SKIP() << "no winner cleared the improvement margin for this shape";

        // Reduce the row to the historical column set: the ten problem columns
        // plus solution_index, with no schema_version and no solution_name.
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
