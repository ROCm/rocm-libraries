// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Every hipBLASLt handle in a process shares one TensileHost and the libraries
// it loads. These tests run heuristic selection and launches from several
// threads at once, each thread with its own handle and non-blocking stream,
// and check every result. A and B are all ones, so every element of D is K.
// GEMM threads use bf16 and grouped GEMM threads use fp16, the grouped type
// every target ships.
//
// The cold case runs the workload in a fresh process. Libraries are loaded
// lazily, one shard per problem type, so the first lookups there load the
// shards while other threads are already looking up and launching from them.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// Re-reads TENSILE_DB inside libhipblaslt.so; see streamk5_test.cpp.
extern "C" void hipblaslt_debug_reload();

namespace
{
    enum class Kind
    {
        Gemm,
        GroupedGemm,
    };

    constexpr int kIterations = 40;
    constexpr int kAlgos      = 4;

    // Covers any solution the heuristic picks for these shapes.
    constexpr uint64_t kWsBudgetBytes = 128ull << 20;

    // Wall-clock budget for a whole run. Past it the queue is treated as wedged.
    constexpr int kDeadlineSeconds = 120;

    // Exit codes of the cold child.
    constexpr int kChildPassed     = 0;
    constexpr int kChildFailed     = 1;
    constexpr int kChildNoSolution = 2;
    constexpr int kChildNotCold    = 3;

    // K stays in {256, 512, 768, 1024}: sums of ones up to there are exact in
    // bf16 and fp16.
    struct Shape
    {
        int64_t m, n, k;
    };
    constexpr Shape   kShapes[] = {{512, 512, 256}, {1024, 256, 768}, {256, 1024, 512}, {768, 768, 1024}};
    constexpr int     kShapeCount = 4;
    constexpr int64_t kMaxElems   = 1 << 20;

    // A grouped GEMM runs two of the shapes above as its two groups.
    constexpr int kGroups = 2;

    hipDataType elementType(Kind kind)
    {
        return kind == Kind::Gemm ? HIP_R_16BF : HIP_R_16F;
    }

    // Printed by PlaceholderLibrary when TENSILE_DB has 0x20 set.
    constexpr char kPlaceholderLoadBit[]    = "0x20";
    constexpr char kPlaceholderLoadMarker[] = "load placeholder library";

    bool gpuAvailable()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    // Bits of a small positive integer, which both 16-bit types hold exactly.
    uint16_t bitsOfExactInt(Kind kind, int64_t v)
    {
        const float f   = static_cast<float>(v);
        uint32_t    raw = 0;
        static_assert(sizeof(raw) == sizeof(f), "float is 32 bits");
        std::memcpy(&raw, &f, sizeof(raw));
        if(kind == Kind::Gemm)
            return static_cast<uint16_t>(raw >> 16);
        const uint32_t exponent = ((raw >> 23) & 0xFF) - 127 + 15;
        return static_cast<uint16_t>((exponent << 10) | ((raw >> 13) & 0x3FF));
    }

    std::string shapeText(Shape const& sh)
    {
        return std::to_string(sh.m) + "x" + std::to_string(sh.n) + "x" + std::to_string(sh.k);
    }

    struct Worker
    {
        int         id         = 0;
        Kind        kind       = Kind::Gemm;
        bool        noSolution = false;
        int         completed  = 0;
        std::string error; // first failure; gtest assertions stay on the main thread
    };

    struct Start
    {
        std::atomic<int>                      ready{0};
        std::atomic<bool>                     go{false};
        std::atomic<bool>                     abortAll{false};
        std::chrono::steady_clock::time_point deadline;
    };

    // One thread's handle, stream, and buffers. A and B hold ones; D has one
    // buffer per group.
    class Session
    {
    public:
        Session(Worker& w, Start& start)
            : m_w(w)
            , m_start(start)
        {
        }

        ~Session()
        {
            if(m_stream)
                static_cast<void>(hipStreamSynchronize(m_stream));
            for(void* p : {m_a, m_b, m_d[0], m_d[1], m_ws})
                static_cast<void>(hipFree(p));
            if(m_pref)
                hipblasLtMatmulPreferenceDestroy(m_pref);
            if(m_stream)
                static_cast<void>(hipStreamDestroy(m_stream));
            if(m_handle)
                hipblasLtDestroy(m_handle);
        }

        Session(Session const&)            = delete;
        Session& operator=(Session const&) = delete;

        void fail(std::string const& what)
        {
            if(m_w.error.empty())
                m_w.error = what;
            m_start.abortAll = true;
        }

        bool hipOk(hipError_t e, char const* what)
        {
            if(e == hipSuccess)
                return true;
            fail(std::string(what) + ": " + hipGetErrorString(e));
            return false;
        }

        bool blasOk(hipblasStatus_t s, char const* what)
        {
            if(s == HIPBLAS_STATUS_SUCCESS)
                return true;
            fail(std::string(what) + ": hipblasStatus " + std::to_string(static_cast<int>(s)));
            return false;
        }

        bool setUp()
        {
            const size_t bytes = static_cast<size_t>(kMaxElems) * sizeof(uint16_t);
            bool         ok    = hipOk(hipSetDevice(0), "hipSetDevice")
                      && blasOk(hipblasLtCreate(&m_handle), "hipblasLtCreate")
                      && hipOk(hipStreamCreateWithFlags(&m_stream, hipStreamNonBlocking),
                               "hipStreamCreateWithFlags")
                      && blasOk(hipblasLtMatmulPreferenceCreate(&m_pref), "PreferenceCreate")
                      && hipOk(hipMalloc(&m_a, bytes), "hipMalloc A")
                      && hipOk(hipMalloc(&m_b, bytes), "hipMalloc B")
                      && hipOk(hipMalloc(&m_d[0], bytes), "hipMalloc D0")
                      && hipOk(hipMalloc(&m_d[1], bytes), "hipMalloc D1");
            if(!ok)
                return false;

            const uint64_t budget = kWsBudgetBytes;
            ok = blasOk(hipblasLtMatmulPreferenceSetAttribute(
                            m_pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budget, sizeof(budget)),
                        "PreferenceSetAttribute");

            std::vector<uint16_t> ones(static_cast<size_t>(kMaxElems), bitsOfExactInt(m_w.kind, 1));
            return ok && hipOk(hipMemcpy(m_a, ones.data(), bytes, hipMemcpyHostToDevice), "memcpy A")
                   && hipOk(hipMemcpy(m_b, ones.data(), bytes, hipMemcpyHostToDevice), "memcpy B");
        }

        bool ensureWorkspace(size_t bytes)
        {
            if(bytes <= m_wsBytes)
                return true;
            static_cast<void>(hipFree(m_ws));
            m_ws      = nullptr;
            m_wsBytes = bytes;
            return hipOk(hipMalloc(&m_ws, bytes), "hipMalloc workspace");
        }

        // Waits for the stream against the deadline, so a wedged queue fails
        // the test with a message instead of hanging the job.
        bool drain(int iter)
        {
            hipError_t q = hipStreamQuery(m_stream);
            while(q == hipErrorNotReady)
            {
                if(std::chrono::steady_clock::now() > m_start.deadline)
                {
                    std::fprintf(stderr,
                                 "\n  thread %d made no progress at iteration %d of %d.\n"
                                 "  Exiting without cleanup because the queue is wedged.\n\n",
                                 m_w.id,
                                 iter,
                                 kIterations);
                    std::fflush(stderr);
                    std::_Exit(1);
                }
                std::this_thread::yield();
                q = hipStreamQuery(m_stream);
            }
            return hipOk(q, "hipStreamQuery");
        }

        bool checkD(int group, Shape const& sh, int iter)
        {
            std::vector<uint16_t> out(static_cast<size_t>(sh.m * sh.n), 0);
            if(!hipOk(hipMemcpyAsync(out.data(),
                                     m_d[group],
                                     out.size() * sizeof(uint16_t),
                                     hipMemcpyDeviceToHost,
                                     m_stream),
                      "memcpy D")
               || !drain(iter))
                return false;

            const uint16_t want = bitsOfExactInt(m_w.kind, sh.k);
            for(size_t i = 0; i < out.size(); ++i)
            {
                if(out[i] != want)
                {
                    char bits[64];
                    std::snprintf(bits, sizeof(bits), "got 0x%04x, want 0x%04x", out[i], want);
                    fail("wrong result at iteration " + std::to_string(iter) + ", group "
                         + std::to_string(group) + ", shape " + shapeText(sh) + ", element "
                         + std::to_string(i) + ": " + bits);
                    return false;
                }
            }
            return true;
        }

        bool clearD()
        {
            const size_t bytes = static_cast<size_t>(kMaxElems) * sizeof(uint16_t);
            return hipOk(hipMemsetAsync(m_d[0], 0, bytes, m_stream), "memset D0")
                   && hipOk(hipMemsetAsync(m_d[1], 0, bytes, m_stream), "memset D1");
        }

        bool runGemm(Shape const& sh, int iter)
        {
            hipblasLtMatmulDesc_t   desc = nullptr;
            hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layD = nullptr;

            bool ok = blasOk(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, sh.m, sh.k, sh.m),
                             "LayoutCreate A")
                      && blasOk(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, sh.k, sh.n, sh.k),
                                "LayoutCreate B")
                      && blasOk(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, sh.m, sh.n, sh.m),
                                "LayoutCreate D")
                      && blasOk(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                                "DescCreate");

            hipblasLtMatmulHeuristicResult_t heuristic[kAlgos]{};
            int                              returned = 0;
            if(ok)
                ok = blasOk(hipblasLtMatmulAlgoGetHeuristic(m_handle,
                                                            desc,
                                                            layA,
                                                            layB,
                                                            layD,
                                                            layD,
                                                            m_pref,
                                                            kAlgos,
                                                            heuristic,
                                                            &returned),
                            "AlgoGetHeuristic");
            if(ok && returned == 0)
            {
                m_w.noSolution = true;
                ok             = false;
            }

            const float alpha = 1.0f, beta = 0.0f;
            ok = ok && ensureWorkspace(heuristic[0].workspaceSize) && clearD()
                 && blasOk(hipblasLtMatmul(m_handle,
                                           desc,
                                           &alpha,
                                           m_a,
                                           layA,
                                           m_b,
                                           layB,
                                           &beta,
                                           m_d[0],
                                           layD,
                                           m_d[0],
                                           layD,
                                           &heuristic[0].algo,
                                           m_ws,
                                           heuristic[0].workspaceSize,
                                           m_stream),
                           "hipblasLtMatmul")
                 && checkD(0, sh, iter);

            for(auto layout : {layA, layB, layD})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            return ok;
        }

        bool runGroupedGemm(Shape const (&sh)[kGroups], int iter)
        {
            const hipDataType          type = elementType(Kind::GroupedGemm);
            hipblaslt_ext::GroupedGemm gemm(
                m_handle, HIPBLAS_OP_N, HIPBLAS_OP_N, type, type, type, type, HIPBLAS_COMPUTE_32F);

            const float                              alpha = 1.0f, beta = 0.0f;
            std::vector<int64_t>                     m, n, k, batch;
            std::vector<hipblaslt_ext::GemmEpilogue> epilogue(kGroups);
            std::vector<hipblaslt_ext::GemmInputs>   inputs(kGroups);
            for(int g = 0; g < kGroups; ++g)
            {
                m.push_back(sh[g].m);
                n.push_back(sh[g].n);
                k.push_back(sh[g].k);
                batch.push_back(1);
                inputs[g].setA(m_a);
                inputs[g].setB(m_b);
                inputs[g].setC(m_d[g]);
                inputs[g].setD(m_d[g]);
                inputs[g].setAlpha(&alpha);
                inputs[g].setBeta(&beta);
            }
            if(!blasOk(gemm.setProblem(m, n, k, batch, epilogue, inputs), "GroupedGemm setProblem"))
                return false;

            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(kWsBudgetBytes);
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
            if(!blasOk(gemm.algoGetHeuristic(kAlgos, pref, heuristic), "GroupedGemm algoGetHeuristic"))
                return false;

            hipblasLtMatmulAlgo_t* chosen = nullptr;
            size_t                 wsSize = 0;
            for(auto& result : heuristic)
            {
                size_t need = 0;
                if(gemm.isAlgoSupported(result.algo, need) == HIPBLAS_STATUS_SUCCESS
                   && need <= kWsBudgetBytes)
                {
                    chosen = &result.algo;
                    wsSize = need;
                    break;
                }
            }
            if(!chosen)
            {
                m_w.noSolution = true;
                return false;
            }

            gemm.setMaxWorkspaceBytes(kWsBudgetBytes);
            return ensureWorkspace(wsSize) && clearD()
                   && blasOk(gemm.initialize(*chosen, m_ws, false, m_stream), "GroupedGemm initialize")
                   && blasOk(gemm.run(m_stream), "GroupedGemm run") && checkD(0, sh[0], iter)
                   && checkD(1, sh[1], iter);
        }

    private:
        Worker&                     m_w;
        Start&                      m_start;
        hipblasLtHandle_t           m_handle  = nullptr;
        hipblasLtMatmulPreference_t m_pref    = nullptr;
        hipStream_t                 m_stream  = nullptr;
        void*                       m_a       = nullptr;
        void*                       m_b       = nullptr;
        void*                       m_d[kGroups]{};
        void*                       m_ws      = nullptr;
        size_t                      m_wsBytes = 0;
    };

    void runWorker(Worker& w, Start& start)
    {
        Session    session(w, start);
        const bool setup = session.setUp();

        // Every thread arrives, even after a setup failure, or the others
        // would wait on it forever. They all start the first lookup together.
        ++start.ready;
        while(!start.go)
            std::this_thread::yield();
        if(!setup || start.abortAll)
            return;

        for(int iter = 0; iter < kIterations && !start.abortAll; ++iter)
        {
            const int   first = (w.id + iter) % kShapeCount;
            const Shape pair[kGroups] = {kShapes[first], kShapes[(first + 1) % kShapeCount]};
            const bool  ok = w.kind == Kind::Gemm ? session.runGemm(pair[0], iter)
                                                  : session.runGroupedGemm(pair, iter);
            if(!ok)
                break;
            ++w.completed;
        }
    }

    // Runs `threads` workers, alternating GEMM and grouped GEMM when `mixed`.
    std::vector<Worker> runWorkers(int threads, bool mixed)
    {
        Start start;
        start.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kDeadlineSeconds);

        std::vector<Worker>      workers(threads);
        std::vector<std::thread> pool;
        for(int i = 0; i < threads; ++i)
        {
            workers[i].id   = i;
            workers[i].kind = (mixed && i % 2 == 1) ? Kind::GroupedGemm : Kind::Gemm;
            pool.emplace_back(runWorker, std::ref(workers[i]), std::ref(start));
        }
        while(start.ready < threads)
            std::this_thread::yield();
        start.go = true;
        for(auto& t : pool)
            t.join();
        return workers;
    }

    char const* kindName(Kind kind)
    {
        return kind == Kind::Gemm ? "GEMM" : "grouped GEMM";
    }

    // Returns true when the results are usable; false means skip.
    bool expectAllCompleted(std::vector<Worker> const& workers)
    {
        bool anySkip = false;
        for(auto const& w : workers)
        {
            anySkip = anySkip || w.noSolution;
            EXPECT_TRUE(w.error.empty()) << kindName(w.kind) << " thread " << w.id << ": " << w.error;
        }
        if(anySkip)
            return false;
        for(auto const& w : workers)
            EXPECT_EQ(w.completed, kIterations) << kindName(w.kind) << " thread " << w.id;
        EXPECT_EQ(hipGetLastError(), hipSuccess);
        return true;
    }

    void setEnv(char const* name, char const* value)
    {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

#ifdef _WIN32
    int fdOf(FILE* f)
    {
        return _fileno(f);
    }
    int dupFd(int fd)
    {
        return _dup(fd);
    }
    int dup2Fd(int from, int to)
    {
        return _dup2(from, to);
    }
    int closeFd(int fd)
    {
        return _close(fd);
    }
    std::string makeCapturePath()
    {
        char const* dir = std::getenv("TEMP");
        return std::string(dir ? dir : ".") + "\\hipblaslt_cold_capture.txt";
    }
#else
    int fdOf(FILE* f)
    {
        return fileno(f);
    }
    int dupFd(int fd)
    {
        return dup(fd);
    }
    int dup2Fd(int from, int to)
    {
        return dup2(from, to);
    }
    int closeFd(int fd)
    {
        return close(fd);
    }
    std::string makeCapturePath()
    {
        char      path[] = "/tmp/hipblaslt_cold_XXXXXX";
        const int fd     = mkstemp(path);
        if(fd < 0)
            return {};
        close(fd);
        return path;
    }
#endif

    // Body of the cold case, run in the death-test child. It exits with one of
    // the kChild* codes.
    [[noreturn]] void runColdChild()
    {
        setEnv("TENSILE_DB", kPlaceholderLoadBit);
        hipblaslt_debug_reload();

        // Capture what the library prints, to count the shard loads.
        const std::string capture = makeCapturePath();
        std::fflush(stdout);
        const int savedStdout = dupFd(fdOf(stdout));
        FILE*     sink        = capture.empty() ? nullptr : std::fopen(capture.c_str(), "w");
        if(!sink || savedStdout < 0)
        {
            std::fprintf(stderr, "could not capture stdout\n");
            std::_Exit(kChildFailed);
        }
        dup2Fd(fdOf(sink), fdOf(stdout));

        constexpr int kThreads = 8;
        auto          workers  = runWorkers(kThreads, true);

        std::fflush(stdout);
        dup2Fd(savedStdout, fdOf(stdout));
        closeFd(savedStdout);
        std::fclose(sink);

        std::ifstream     in(capture);
        const std::string printed((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        std::remove(capture.c_str());

        int loads = 0;
        for(auto at = printed.find(kPlaceholderLoadMarker); at != std::string::npos;
            at      = printed.find(kPlaceholderLoadMarker, at + 1))
            ++loads;

        bool failed = false, noSolution = false;
        for(auto const& w : workers)
        {
            if(!w.error.empty())
            {
                std::fprintf(stderr, "%s thread %d: %s\n", kindName(w.kind), w.id, w.error.c_str());
                failed = true;
            }
            else if(w.noSolution)
                noSolution = true;
            else if(w.completed != kIterations)
            {
                std::fprintf(stderr,
                             "%s thread %d completed %d of %d iterations\n",
                             kindName(w.kind),
                             w.id,
                             w.completed,
                             kIterations);
                failed = true;
            }
        }
        std::fprintf(stderr, "cold child: %d lazy shard loads\n", loads);
        if(!failed && !noSolution && loads == 0)
            std::fprintf(stderr, "no shard was loaded lazily, so the run did not start cold\n");
        std::fflush(stderr);

        if(failed)
            std::_Exit(kChildFailed);
        if(noSolution)
            std::_Exit(kChildNoSolution);
        std::_Exit(loads == 0 ? kChildNotCold : kChildPassed);
    }

    TEST(ConcurrentHandles, smoke_Matmul)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";
        if(!expectAllCompleted(runWorkers(4, false)))
            GTEST_SKIP() << "No solution for one of the test shapes on this device";
    }

    TEST(ConcurrentHandles, smoke_MixedGemmAndGroupedGemm)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";
        if(!expectAllCompleted(runWorkers(8, true)))
            GTEST_SKIP() << "No solution for one of the test shapes on this device";
    }

    // The same mixed workload in a fresh process, where no library shard has
    // been loaded yet. The child counts the lazy shard loads it saw, so a run
    // that was not cold fails instead of passing on warm libraries.
    TEST(ConcurrentHandles, smoke_ColdLibrariesMixedGemmAndGroupedGemm)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";
        GTEST_FLAG_SET(death_test_style, "threadsafe");

        // Accepting a no-solution exit keeps gtest from failing the case, so
        // it can be reported as a skip. Any other non-zero exit fails and
        // prints the child's stderr.
        bool noSolution  = false;
        auto passOrSkip  = [&noSolution](int status) {
            noSolution = ::testing::ExitedWithCode(kChildNoSolution)(status);
            return noSolution || ::testing::ExitedWithCode(kChildPassed)(status);
        };
        EXPECT_EXIT(runColdChild(), passOrSkip, "");
        if(noSolution)
            GTEST_SKIP() << "No solution for one of the test shapes on this device";
    }
} // namespace
