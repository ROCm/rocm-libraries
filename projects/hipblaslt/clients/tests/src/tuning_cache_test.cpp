// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tuning file tests.
//
// Each case writes a tuning file the way hipblaslt-bench does, points
// HIPBLASLT_TUNING_OVERRIDE_FILE or cache mode at it, and asserts on the
// solution index the heuristic hands back or the matmul launches, and on the
// library's own counters. Rows are built from solutions the heuristic itself
// offers for the problem, so every recorded index is real and only the
// recorded name or build decides whether it is used.
//
// The tuning variables are read on first use and each file is loaded once per
// process, so each case starts from hipblaslt_tuning_reset_for_test().

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#ifdef WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

extern "C" void   hipblaslt_tuning_reset_for_test();
extern "C" int    hipblaslt_tuning_last_launch_for_test();
extern "C" size_t hipblaslt_tuning_stats_for_test(uint64_t* values, size_t count);

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
    // One fp16 NN problem throughout. Large enough that the heuristic offers
    // several distinct solutions, which the stale-entry cases need.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 512;
    constexpr int64_t kK = 1024;

    constexpr uint64_t kWorkspaceBytes = 32 * 1024 * 1024;

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    std::string tempTuningPath(const char* stem)
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

    /** The descriptors of the one problem these cases run, for a given handle. */
    struct Problem
    {
        hipblasLtMatrixLayout_t     layoutA = nullptr;
        hipblasLtMatrixLayout_t     layoutB = nullptr;
        hipblasLtMatrixLayout_t     layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc    = nullptr;
        hipblasLtMatmulPreference_t pref    = nullptr;

        bool create()
        {
            if(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, kM, kK, kM)
                   != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, kK, kN, kK)
                      != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, kM, kN, kM)
                      != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                      != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
                return false;

            const uint64_t maxWs = kWorkspaceBytes;
            return hipblasLtMatmulPreferenceSetAttribute(
                       pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                   == HIPBLAS_STATUS_SUCCESS;
        }

        ~Problem()
        {
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
        }
    };

    struct Identity
    {
        int         index = -1;
        std::string kernelName;
    };

    /**
     * Distinct solutions the heuristic offers for the problem, best first, with
     * the name each one's index resolves to. Read with no tuning file in play.
     */
    std::vector<Identity> candidateIdentities(int want)
    {
        std::vector<Identity> found;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return found;

        {
            Problem                                       problem;
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(want);
            int                                           returned = 0;
            if(problem.create()
               && hipblasLtMatmulAlgoGetHeuristic(handle,
                                                  problem.desc,
                                                  problem.layoutA,
                                                  problem.layoutB,
                                                  problem.layoutC,
                                                  problem.layoutC,
                                                  problem.pref,
                                                  want,
                                                  heuristic.data(),
                                                  &returned)
                      == HIPBLAS_STATUS_SUCCESS)
            {
                for(int i = 0; i < returned; i++)
                {
                    Identity id;
                    id.index      = hipblaslt_ext::getIndexFromAlgo(heuristic[i].algo);
                    id.kernelName = hipblaslt_ext::getKernelNameFromAlgo(handle, heuristic[i].algo);
                    if(id.kernelName.empty())
                        continue;

                    bool seen = false;
                    for(const auto& other : found)
                        seen = seen || other.index == id.index;
                    if(!seen)
                        found.push_back(id);
                }
            }
        }

        hipblasLtDestroy(handle);
        return found;
    }

    /** The build this library reports, as hipblaslt-bench records it. */
    std::string buildStamp()
    {
        std::string       stamp;
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS)
        {
            char rev[128] = {};
            if(hipblasLtGetGitRevision(handle, rev) == HIPBLAS_STATUS_SUCCESS)
                stamp = rev;
            hipblasLtDestroy(handle);
        }
        return stamp;
    }

    /**
     * One heuristic query and matmul through the C API.
     *
     * selectedIndex is the solution the heuristic returned first, which with a
     * tuning file in play is the entry replay chose; returnedCount is how many
     * it returned for the requested count.
     */
    bool runGemm(int* selectedIndex, int requested = 1, int* returnedCount = nullptr)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void* dA  = nullptr;
        void* dB  = nullptr;
        void* dC  = nullptr;
        void* dWs = nullptr;

        bool ok = hipMalloc(&dA, kM * kK * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, kK * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, kM * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, kWorkspaceBytes) == hipSuccess
                  && hipMemset(dA, 0, kM * kK * sizeof(uint16_t)) == hipSuccess
                  && hipMemset(dB, 0, kK * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMemset(dC, 0, kM * kN * sizeof(uint16_t)) == hipSuccess;

        {
            Problem problem;
            ok = ok && problem.create();

            std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(requested);
            int                                           returned = 0;
            ok                                                     = ok
                 && hipblasLtMatmulAlgoGetHeuristic(handle,
                                                    problem.desc,
                                                    problem.layoutA,
                                                    problem.layoutB,
                                                    problem.layoutC,
                                                    problem.layoutC,
                                                    problem.pref,
                                                    requested,
                                                    heuristic.data(),
                                                    &returned)
                        == HIPBLAS_STATUS_SUCCESS
                 && returned > 0;

            if(ok)
            {
                if(selectedIndex)
                    *selectedIndex = hipblaslt_ext::getIndexFromAlgo(heuristic[0].algo);
                if(returnedCount)
                    *returnedCount = returned;

                const float alpha = 1.0f;
                const float beta  = 0.0f;
                ok                = hipblasLtMatmul(handle,
                                     problem.desc,
                                     &alpha,
                                     dA,
                                     problem.layoutA,
                                     dB,
                                     problem.layoutB,
                                     &beta,
                                     dC,
                                     problem.layoutC,
                                     dC,
                                     problem.layoutC,
                                     &heuristic[0].algo,
                                     dWs,
                                     kWorkspaceBytes,
                                     nullptr)
                         == HIPBLAS_STATUS_SUCCESS
                     && hipDeviceSynchronize() == hipSuccess;
            }
        }

        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);
        return ok;
    }

    /**
     * The heuristic query through the C++ extension API, for the same shape
     * with every matrix of `type`.
     */
    bool extHeuristicIndex(int*                 selectedIndex,
                           hipDataType          type    = HIP_R_16F,
                           hipblasComputeType_t compute = HIPBLAS_COMPUTE_32F)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        const size_t elementBytes = type == HIP_R_32F ? sizeof(float) : sizeof(uint16_t);

        void* dA = nullptr;
        void* dB = nullptr;
        void* dC = nullptr;
        bool  ok = hipMalloc(&dA, kM * kK * elementBytes) == hipSuccess
                  && hipMalloc(&dB, kK * kN * elementBytes) == hipSuccess
                  && hipMalloc(&dC, kM * kN * elementBytes) == hipSuccess;

        if(ok)
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;

            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(kWorkspaceBytes);

            hipblaslt_ext::Gemm gemm(
                handle, HIPBLAS_OP_N, HIPBLAS_OP_N, type, type, type, type, compute);

            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   inputs;
            inputs.setA(dA);
            inputs.setB(dB);
            inputs.setC(dC);
            inputs.setD(dC);
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            gemm.setProblem(kM, kN, kK, 1, epilogue, inputs);

            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            ok = gemm.algoGetHeuristic(1, pref, results) == HIPBLAS_STATUS_SUCCESS
                 && !results.empty();
            if(ok)
                *selectedIndex = hipblaslt_ext::getIndexFromAlgo(results[0].algo);
        }

        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);
        return ok;
    }

    /** The heuristic query for a grouped GEMM of two copies of the problem. */
    bool groupedHeuristicIndex(int* selectedIndex)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void* dA = nullptr;
        void* dB = nullptr;
        void* dC = nullptr;
        bool  ok = hipMalloc(&dA, kM * kK * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, kK * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, kM * kN * sizeof(uint16_t)) == hipSuccess;

        if(ok)
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;

            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(kWorkspaceBytes);

            hipblaslt_ext::GroupedGemm grouped(handle,
                                               HIPBLAS_OP_N,
                                               HIPBLAS_OP_N,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIPBLAS_COMPUTE_32F);

            std::vector<int64_t>                     m(2, kM), n(2, kN), k(2, kK), batch(2, 1);
            std::vector<hipblaslt_ext::GemmEpilogue> epilogue(2);
            std::vector<hipblaslt_ext::GemmInputs>   inputs(2);
            for(auto& in : inputs)
            {
                in.setA(dA);
                in.setB(dB);
                in.setC(dC);
                in.setD(dC);
                in.setAlpha(&alpha);
                in.setBeta(&beta);
            }

            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            ok = grouped.setProblem(m, n, k, batch, epilogue, inputs) == HIPBLAS_STATUS_SUCCESS
                 && grouped.algoGetHeuristic(1, pref, results) == HIPBLAS_STATUS_SUCCESS
                 && !results.empty();
            if(ok)
                *selectedIndex = hipblaslt_ext::getIndexFromAlgo(results[0].algo);
        }

        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);
        return ok;
    }

    /** The library's tallies, in the order hipblaslt_tuning_stats_for_test documents. */
    struct Counters
    {
        uint64_t loaded = 0, hits = 0, misses = 0, invalidated = 0;
        uint64_t shapes = 0, matched = 0, fellback = 0;
    };

    Counters counters()
    {
        uint64_t     v[7]  = {};
        const size_t known = hipblaslt_tuning_stats_for_test(v, 7);
        EXPECT_GE(known, 7u) << "the library reports fewer tallies than this reads";

        Counters c;
        c.loaded      = v[0];
        c.hits        = v[1];
        c.misses      = v[2];
        c.invalidated = v[3];
        c.shapes      = v[4];
        c.matched     = v[5];
        c.fellback    = v[6];
        return c;
    }

    uint16_t halfBits(float value)
    {
        const _Float16 h = static_cast<_Float16>(value);
        uint16_t       bits;
        std::memcpy(&bits, &h, sizeof(bits));
        return bits;
    }

    /** Where launchGemm takes the algo it hands hipblasLtMatmul from. */
    enum class AlgoFrom
    {
        Heuristic, // the heuristic's first result, queried just before the call
        Index, // a given solution index, the way a caller reuses one algo
        Null, // no algo, which leaves the choice to the library
    };

    /**
     * One matmul of the problem with separate C and D, reporting the solution
     * the library actually launched.
     *
     * verifyProduct fills A and B with ones and checks every element of D is k,
     * which proves the launched kernel ran on the caller's buffers.
     */
    bool launchGemm(AlgoFrom from, int index, int* launched, bool verifyProduct = false)
    {
        *launched = -1;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void* dA  = nullptr;
        void* dB  = nullptr;
        void* dC  = nullptr;
        void* dD  = nullptr;
        void* dWs = nullptr;

        bool ok = hipMalloc(&dA, kM * kK * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, kK * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, kM * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dD, kM * kN * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, kWorkspaceBytes) == hipSuccess;

        if(ok)
        {
            const uint16_t              fill = verifyProduct ? halfBits(1.0f) : 0;
            const std::vector<uint16_t> a(static_cast<size_t>(kM * kK), fill);
            const std::vector<uint16_t> b(static_cast<size_t>(kK * kN), fill);
            ok = hipMemcpy(dA, a.data(), a.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                     == hipSuccess
                 && hipMemcpy(dB, b.data(), b.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                        == hipSuccess
                 && hipMemset(dC, 0, kM * kN * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dD, 0, kM * kN * sizeof(uint16_t)) == hipSuccess;
        }

        {
            Problem problem;
            ok = ok && problem.create();

            const float                  alpha   = 1.0f;
            const float                  beta    = 0.0f;
            hipblasLtMatmulAlgo_t        algo    = {};
            const hipblasLtMatmulAlgo_t* algoPtr = nullptr;

            if(ok && from == AlgoFrom::Heuristic)
            {
                hipblasLtMatmulHeuristicResult_t heuristic[1];
                int                              returned = 0;
                ok                                        = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                     problem.desc,
                                                     problem.layoutA,
                                                     problem.layoutB,
                                                     problem.layoutC,
                                                     problem.layoutC,
                                                     problem.pref,
                                                     1,
                                                     heuristic,
                                                     &returned)
                         == HIPBLAS_STATUS_SUCCESS
                     && returned > 0;
                algo    = heuristic[0].algo;
                algoPtr = &algo;
            }
            else if(ok && from == AlgoFrom::Index)
            {
                std::vector<int>                              indexes{index};
                std::vector<hipblasLtMatmulHeuristicResult_t> results;
                ok = hipblaslt_ext::getAlgosFromIndex(handle, indexes, results)
                         == HIPBLAS_STATUS_SUCCESS
                     && !results.empty();
                if(ok)
                {
                    algo    = results[0].algo;
                    algoPtr = &algo;
                }
            }

            if(ok)
            {
                ok = hipblasLtMatmul(handle,
                                     problem.desc,
                                     &alpha,
                                     dA,
                                     problem.layoutA,
                                     dB,
                                     problem.layoutB,
                                     &beta,
                                     dC,
                                     problem.layoutC,
                                     dD,
                                     problem.layoutC,
                                     algoPtr,
                                     dWs,
                                     kWorkspaceBytes,
                                     nullptr)
                         == HIPBLAS_STATUS_SUCCESS
                     && hipDeviceSynchronize() == hipSuccess;
                *launched = hipblaslt_tuning_last_launch_for_test();
            }
        }

        if(ok && verifyProduct)
        {
            std::vector<uint16_t> out(static_cast<size_t>(kM * kN), 0);
            ok = hipMemcpy(out.data(), dD, out.size() * sizeof(uint16_t), hipMemcpyDeviceToHost)
                 == hipSuccess;

            const uint16_t expected = halfBits(static_cast<float>(kK));
            for(size_t i = 0; ok && i < out.size(); i++)
                ok = out[i] == expected;
        }

        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);
        return ok;
    }

    /**
     * Collects what the library writes to std::cerr while it is alive. It sees
     * nothing sent to a log file or written straight to file descriptor 2.
     */
    class CerrCapture
    {
    public:
        CerrCapture()
            : m_saved(std::cerr.rdbuf(m_buffer.rdbuf()))
        {
        }

        ~CerrCapture()
        {
            std::cerr.rdbuf(m_saved);
        }

        CerrCapture(const CerrCapture&)            = delete;
        CerrCapture& operator=(const CerrCapture&) = delete;

        std::string str() const
        {
            return m_buffer.str();
        }

    private:
        std::ostringstream m_buffer;
        std::streambuf*    m_saved;
    };

    /**
     * The logger reads its environment once per process, so a run with a level
     * or log file set sends tuning notices somewhere CerrCapture cannot see.
     */
    bool loggingEnvIsClean()
    {
        return getenv("HIPBLASLT_LOG_LEVEL") == nullptr && getenv("HIPBLASLT_LOG_MASK") == nullptr
               && getenv("HIPBLASLT_LOG_FILE") == nullptr;
    }

    struct Row
    {
        int                        index = -1;
        std::optional<std::string> kernelName;
        // a_type, b_type, c_type and compute_type, in the file's spelling.
        std::string types = "f16_r,f16_r,f16_r,f32_r";
    };

    /**
     * A tuning file in hipblaslt-bench's layout: a version line, then a header
     * and a value row per entry, all for the problem these cases run. An empty
     * stamp writes no version line at all.
     */
    void writeTuningFile(const std::string&      path,
                         const std::string&      stamp,
                         const std::vector<Row>& rows)
    {
        std::ofstream out(path, std::ios::trunc);
        if(!stamp.empty())
            out << "Git Version: " << stamp << "\n";

        for(const auto& row : rows)
        {
            out << "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,solution_"
                   "index";
            if(row.kernelName)
                out << ",kernel_name";
            out << "\n";

            out << "N,N,1," << kM << "," << kN << "," << kK << "," << row.types << "," << row.index;
            if(row.kernelName)
                out << "," << *row.kernelName;
            out << "\n";
        }
    }

    class TuningCache_pre_checkin : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            for(const char* name : kTuningVariables)
            {
                const char* value = getenv(name);
                m_savedEnv.emplace_back(name,
                                        value ? std::optional<std::string>(value) : std::nullopt);
            }

            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";

            // Solutions are read with no file in play, so a user's own tuning
            // settings cannot decide what the cases below take as the default.
            for(const char* name : kTuningVariables)
                unsetenv(name);
            hipblaslt_tuning_reset_for_test();

            m_path
                = tempTuningPath(::testing::UnitTest::GetInstance()->current_test_info()->name());
            std::remove(m_path.c_str());

            m_identities = candidateIdentities(8);
            m_stamp      = buildStamp();
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

        /** Point the override variable at the file just written, starting clean. */
        void useTuningFile()
        {
            setenv("HIPBLASLT_TUNING_OVERRIDE_FILE", m_path.c_str(), 1);
            hipblaslt_tuning_reset_for_test();
        }

        /** Cache mode on a path, which may be empty, starting clean. */
        void useCache(const std::string& path)
        {
            setenv("HIPBLASLT_TUNING_MODE", "cache", 1);
            if(path.empty())
                unsetenv("HIPBLASLT_TUNING_CACHE_PATH");
            else
                setenv("HIPBLASLT_TUNING_CACHE_PATH", path.c_str(), 1);
            hipblaslt_tuning_reset_for_test();
        }

        /**
         * Skip unless the heuristic offers at least `count` solutions. A case
         * records a non-default one so that replaying it is distinguishable from
         * default selection.
         */
        bool haveSolutions(size_t count)
        {
            return m_identities.size() >= count;
        }

        static constexpr const char* kTuningVariables[] = {"HIPBLASLT_TUNING_OVERRIDE_FILE",
                                                           "HIPBLASLT_TUNING_MODE",
                                                           "HIPBLASLT_TUNING_CACHE_PATH"};

        std::string                                                     m_path;
        std::string                                                     m_stamp;
        std::vector<Identity>                                           m_identities;
        std::vector<std::pair<std::string, std::optional<std::string>>> m_savedEnv;
    };

    // A row whose recorded name still matches what its index resolves to is
    // replayed.
    TEST_F(TuningCache_pre_checkin, NamedEntryReplays)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{recorded.index, recorded.kernelName}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, recorded.index);
    }

    // A row whose index now resolves to a different kernel is not launched: the
    // problem falls back to default selection.
    TEST_F(TuningCache_pre_checkin, EntryWhoseNameNoLongerMatchesIsRejected)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, std::string("NotARealKernel")}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_NE(selected, m_identities[1].index) << "a row naming another kernel was launched";
        EXPECT_EQ(selected, m_identities[0].index) << "did not fall back to default selection";
    }

    // The second row for a problem is judged on its own solution, even when the
    // first row is stale.
    TEST_F(TuningCache_pre_checkin, StaleFirstEntryDoesNotHideAValidSecond)
    {
        if(!haveSolutions(3))
            GTEST_SKIP() << "the heuristic offers fewer than three solutions for this problem";

        const auto& valid = m_identities[2];
        writeTuningFile(m_path,
                        m_stamp,
                        {{m_identities[1].index, std::string("NotARealKernel")},
                         {valid.index, valid.kernelName}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, valid.index);
    }

    // A named row is trusted on its name, not on the file's version line.
    TEST_F(TuningCache_pre_checkin, NamedEntrySurvivesAnotherBuildsVersionLine)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, "not-this-build", {{recorded.index, recorded.kernelName}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, recorded.index);
    }

    // A row without a name has nothing to be checked against, so it is used
    // only when the file was written by the running build.
    TEST_F(TuningCache_pre_checkin, UnnamedEntryFromThisBuildIsUsed)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";
        if(m_stamp.empty())
            GTEST_SKIP() << "this build reports no revision to write";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, std::nullopt}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[1].index);
    }

    TEST_F(TuningCache_pre_checkin, UnnamedEntryFromAnotherBuildIsIgnored)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, "not-this-build", {{m_identities[1].index, std::nullopt}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);
    }

    // A file with no version line vouches for no build, including one built
    // outside a git checkout, which has no revision of its own.
    TEST_F(TuningCache_pre_checkin, UnnamedEntryWithoutAVersionLineIsIgnored)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, "", {{m_identities[1].index, std::nullopt}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);
    }

    // The C++ extension API applies the same per-row rule as the C API.
    TEST_F(TuningCache_pre_checkin, ExtApiIgnoresUnnamedEntryFromAnotherBuild)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, "not-this-build", {{m_identities[1].index, std::nullopt}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(extHeuristicIndex(&selected));
        EXPECT_NE(selected, m_identities[1].index)
            << "the C++ API applied an unnamed row written by another build";
    }

    TEST_F(TuningCache_pre_checkin, ExtApiReplaysNamedEntry)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, "not-this-build", {{recorded.index, recorded.kernelName}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(extHeuristicIndex(&selected));
        EXPECT_EQ(selected, recorded.index);
    }

    // An XF32 problem whose entry cannot run, as XF32 or as the FP32 fallback,
    // is still XF32 when default selection takes over.
    TEST_F(TuningCache_pre_checkin, ExtApiXf32ProblemStaysXf32AfterAnUnusableEntry)
    {
        if(!haveSolutions(1))
            GTEST_SKIP() << "the heuristic offers no solution for this problem";

        int xf32Default = -1;
        int fp32Default = -1;
        ASSERT_TRUE(extHeuristicIndex(&xf32Default, HIP_R_32F, HIPBLAS_COMPUTE_32F_FAST_TF32));
        ASSERT_TRUE(extHeuristicIndex(&fp32Default, HIP_R_32F, HIPBLAS_COMPUTE_32F));
        if(xf32Default == fp32Default)
            GTEST_SKIP() << "default selection picks the same solution for XF32 and FP32";

        // An fp16 solution: its index still names its kernel, so it passes the
        // name check, but an fp32 problem cannot run it.
        const auto& fp16 = m_identities[0];
        writeTuningFile(
            m_path, m_stamp, {{fp16.index, fp16.kernelName, "f32_r,f32_r,f32_r,xf32_r"}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(extHeuristicIndex(&selected, HIP_R_32F, HIPBLAS_COMPUTE_32F_FAST_TF32));
        EXPECT_EQ(selected, xf32Default);
    }

    // Tuning file rows describe single GEMMs. A grouped GEMM whose groups match
    // a row still uses default selection.
    TEST_F(TuningCache_pre_checkin, GroupedGemmUsesDefaultSelection)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        int defaultIndex = -1;
        ASSERT_TRUE(groupedHeuristicIndex(&defaultIndex));

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{recorded.index, recorded.kernelName}});
        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(groupedHeuristicIndex(&selected));
        EXPECT_EQ(selected, defaultIndex);
    }

    // When the file satisfies a single-algo request, the heuristic's own search
    // is skipped; the returned count must still be exactly one.
    TEST_F(TuningCache_pre_checkin, SingleAlgoRequestServedByTheFileReturnsOne)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{recorded.index, recorded.kernelName}});
        useTuningFile();

        int selected = -1;
        int returned = -1;
        ASSERT_TRUE(runGemm(&selected, 1, &returned));
        EXPECT_EQ(selected, recorded.index);
        EXPECT_EQ(returned, 1);
    }

    // A file that is not there yet when the first query looks for it is read
    // once it appears.
    TEST_F(TuningCache_pre_checkin, FileCreatedAfterTheFirstQueryIsRead)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        useTuningFile();

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{recorded.index, recorded.kernelName}});

        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, recorded.index);
    }

    // Cache mode replays a named row through the heuristic, on both APIs.
    TEST_F(TuningCache_pre_checkin, CacheModeReplaysANamedRow)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{recorded.index, recorded.kernelName}});
        useCache(m_path);

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, recorded.index);

        const auto c = counters();
        EXPECT_EQ(c.loaded, 1u);
        EXPECT_EQ(c.hits, 1u) << "the matmul with the heuristic's algo was counted again";

        int extSelected = -1;
        ASSERT_TRUE(extHeuristicIndex(&extSelected));
        EXPECT_EQ(extSelected, recorded.index);
    }

    // A cache file that is not there yet when the first query looks for it is
    // read once it appears.
    TEST_F(TuningCache_pre_checkin, CacheModeReadsAFileCreatedAfterTheFirstQuery)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        useCache(m_path);

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);

        const auto& recorded = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{recorded.index, recorded.kernelName}});

        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, recorded.index);
        EXPECT_EQ(counters().loaded, 1u);
    }

    // A mode with nowhere to read from does nothing, and does not fall back to
    // the override file.
    TEST_F(TuningCache_pre_checkin, CacheModeWithoutPathIsInert)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, m_identities[1].kernelName}});
        setenv("HIPBLASLT_TUNING_OVERRIDE_FILE", m_path.c_str(), 1);
        useCache("");

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);
        EXPECT_EQ(counters().hits, 0u);
    }

    // With a tuning mode set only the cache is consulted, never the override
    // file.
    TEST_F(TuningCache_pre_checkin, CacheModeIgnoresTheOverrideFile)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, m_identities[1].kernelName}});
        setenv("HIPBLASLT_TUNING_OVERRIDE_FILE", m_path.c_str(), 1);
        useCache(m_path + ".absent");

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);
        EXPECT_EQ(counters().loaded, 0u);
    }

    TEST_F(TuningCache_pre_checkin, CachedEntryWhoseNameNoLongerMatchesIsRejected)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, std::string("NotARealKernel")}});
        useCache(m_path);

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));
        EXPECT_EQ(selected, m_identities[0].index);

        const auto c = counters();
        EXPECT_EQ(c.invalidated, 1u);
        EXPECT_EQ(c.misses, 1u);
        EXPECT_EQ(c.hits, 0u);
    }

    // The heuristic lookup and the execution path both meet a stale row, and it
    // is still one rejected entry.
    TEST_F(TuningCache_pre_checkin, StaleEntryIsCountedAsOneInvalidation)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, std::string("NotARealKernel")}});
        useCache(m_path);

        int selected = -1;
        int launched = -1;
        ASSERT_TRUE(runGemm(&selected));
        ASSERT_TRUE(launchGemm(AlgoFrom::Null, -1, &launched));
        EXPECT_EQ(counters().invalidated, 1u);
    }

    // A caller that passes an algo gets that algo, whatever the cache holds.
    TEST_F(TuningCache_pre_checkin, ExplicitAlgoLaunchesAsGivenInCacheMode)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& given  = m_identities[0];
        const auto& cached = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{cached.index, cached.kernelName}});
        useCache(m_path);

        int launched = -1;
        ASSERT_TRUE(launchGemm(AlgoFrom::Index, given.index, &launched));
        EXPECT_EQ(launched, given.index) << "the cache replaced the caller's algo";
        EXPECT_EQ(counters().hits, 0u) << "a hit was counted for a launch the cache did not choose";
    }

    // A matmul given no algo leaves the choice to the library, so a usable
    // entry is what it launches, and it computes the right result.
    TEST_F(TuningCache_pre_checkin, NullAlgoLaunchesTheCachedKernel)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        const auto& cached = m_identities[1];
        writeTuningFile(m_path, m_stamp, {{cached.index, cached.kernelName}});
        useCache(m_path);

        int launched = -1;
        ASSERT_TRUE(launchGemm(AlgoFrom::Null, -1, &launched, true));
        EXPECT_EQ(launched, cached.index) << "a matmul with no algo ignored the cached entry";
        EXPECT_EQ(counters().hits, 1u);
    }

    // With no usable entry a null algo still runs, on default selection's pick,
    // and the lookup counts as the fallback it is.
    TEST_F(TuningCache_pre_checkin, NullAlgoWithoutAnEntryLaunchesDefaultSelection)
    {
        if(!haveSolutions(1))
            GTEST_SKIP() << "the heuristic offers no solution for this problem";

        useCache(m_path);

        int launched = -1;
        ASSERT_TRUE(launchGemm(AlgoFrom::Null, -1, &launched, true));
        EXPECT_EQ(launched, m_identities[0].index);

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u);
        EXPECT_EQ(c.misses, 1u);
    }

    // A cache that served nothing must not read like a process that asked
    // nothing.
    TEST_F(TuningCache_pre_checkin, ShapesThatMissAnEmptyCacheAreStillCounted)
    {
        useCache(m_path);

        int selected = -1;
        ASSERT_TRUE(runGemm(&selected));

        const auto c = counters();
        EXPECT_EQ(c.shapes, 1u);
        EXPECT_EQ(c.matched, 0u);
        EXPECT_EQ(c.fellback, 1u);
    }

    // The summary is read against loaded=N, so it counts problems rather than
    // lookups.
    TEST_F(TuningCache_pre_checkin, LookupTallyCountsShapesNotCalls)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, m_identities[1].kernelName}});
        useCache(m_path);

        for(int i = 0; i < 4; i++)
        {
            int selected = -1;
            ASSERT_TRUE(runGemm(&selected));
        }

        const auto c = counters();
        EXPECT_EQ(c.shapes, 1u);
        EXPECT_EQ(c.matched, 1u);
        EXPECT_EQ(c.fellback, 0u);
    }

    // The first lookup says what mode is on and what loaded, without any log
    // level set, and names an override file it is ignoring.
    TEST_F(TuningCache_pre_checkin, StartupLineReportsWhatLoaded)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, m_identities[1].kernelName}});
        useCache(m_path);

        std::string loaded;
        {
            CerrCapture capture;
            int         selected = -1;
            ASSERT_TRUE(runGemm(&selected));
            loaded = capture.str();
        }
        EXPECT_NE(loaded.find("tuning-cache: mode=cache path=" + m_path + " load=ok loaded=1"),
                  std::string::npos)
            << loaded;

        setenv("HIPBLASLT_TUNING_OVERRIDE_FILE", m_path.c_str(), 1);
        useCache(m_path + ".absent");

        std::string missing;
        {
            CerrCapture capture;
            int         selected = -1;
            ASSERT_TRUE(runGemm(&selected));
            missing = capture.str();
        }
        EXPECT_NE(missing.find("load=not-found loaded=0"), std::string::npos) << missing;
        EXPECT_NE(missing.find("ignoring HIPBLASLT_TUNING_OVERRIDE_FILE"), std::string::npos)
            << missing;
    }

    // Replay is the steady state, so it must be silent per call.
    TEST_F(TuningCache_pre_checkin, RepeatedCacheHitsAddNoOutput)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, m_identities[1].kernelName}});
        useCache(m_path);

        {
            CerrCapture capture;
            int         selected = -1;
            ASSERT_TRUE(runGemm(&selected));
        }

        std::string later;
        {
            CerrCapture capture;
            int         selected = -1;
            int         launched = -1;
            ASSERT_TRUE(runGemm(&selected));
            ASSERT_TRUE(launchGemm(AlgoFrom::Null, -1, &launched));
            later = capture.str();
        }
        EXPECT_EQ(later, "");
    }

    // With no tuning mode set, the override file behaves as it always has and
    // prints no tuning notices.
    TEST_F(TuningCache_pre_checkin, OverrideFilePrintsNoTuningNotices)
    {
        if(!haveSolutions(2))
            GTEST_SKIP() << "the heuristic offers one solution for this problem";
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        writeTuningFile(m_path, m_stamp, {{m_identities[1].index, m_identities[1].kernelName}});
        useTuningFile();

        std::string out;
        int         selected = -1;
        {
            CerrCapture capture;
            ASSERT_TRUE(runGemm(&selected));
            out = capture.str();
        }
        ASSERT_EQ(selected, m_identities[1].index) << "the override file was not consulted";
        EXPECT_EQ(out.find("tuning-cache:"), std::string::npos) << out;
    }
} // namespace
