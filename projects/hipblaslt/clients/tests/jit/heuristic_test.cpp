// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Queries hipblasLtMatmulAlgoGetHeuristic and GemmInstance::algoGetHeuristic for
// one FP16 GEMM, prints each query as a JSON line, and runs and checks every
// returned algorithm. --from-index resolves algorithm indices instead, --tuned
// reports hipblaslt_ext::matmulIsTuned, and --threads with --barrier issues the
// same queries from several threads that start together, also across
// processes. Uses only the public API, so it builds
// with and without HIPBLASLT_ENABLE_JIT; test_heuristic.py checks what
// HIPBLASLT_JIT should return.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
    // Serializes the JSON lines and PASS lines of concurrent threads.
    std::mutex outputMutex;

    void check(hipError_t status, const char* what)
    {
        if(status != hipSuccess)
            throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
    void check(hipblasStatus_t status, const std::string& what)
    {
        if(status != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error(what + ": status " + std::to_string(status));
    }

    struct Settings
    {
        std::string api       = "both"; // c, cpp or both
        int         requested = 1;
        int64_t     m = 256, n = 128, k = 512;
        int         handles = 1, queries = 1;
        size_t           workspace = 32 << 20;
        bool             nullAlgo  = false;
        bool             run       = true;
        bool             tuned     = false;
        std::vector<int> fromIndex;
        int              threads = 1;
        std::string      barrier;
    };

    // A and B are multiples of 1/8 and C of 1/4, so the FP32 sums are exact and
    // the reference only rounds the result to FP16.
    struct Problem
    {
        const Settings&         s;
        int                     thread;
        hipblasLtHandle_t       handle = nullptr;
        hipblasLtMatmulDesc_t   desc   = nullptr;
        hipblasLtMatrixLayout_t aLayout = nullptr, bLayout = nullptr, cLayout = nullptr,
                                dLayout = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        hipStream_t                 stream = nullptr;
        __half *                    a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
        void*                       workspace = nullptr;
        float                       alpha = 1.25f, beta = 0.5f;
        std::vector<__half>         hostA, hostB, hostC, hostD;
        std::vector<float>          expected;

        Problem(const Settings& settings, int thread)
            : s(settings)
            , thread(thread)
        {
            const auto m = s.m, n = s.n, k = s.k;
            check(hipblasLtCreate(&handle), "Create handle");
            check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  "Create matmul descriptor");
            check(hipblasLtMatrixLayoutCreate(&aLayout, HIP_R_16F, m, k, m), "Create A layout");
            check(hipblasLtMatrixLayoutCreate(&bLayout, HIP_R_16F, k, n, k), "Create B layout");
            check(hipblasLtMatrixLayoutCreate(&cLayout, HIP_R_16F, m, n, m), "Create C layout");
            check(hipblasLtMatrixLayoutCreate(&dLayout, HIP_R_16F, m, n, m), "Create D layout");
            check(hipblasLtMatmulPreferenceCreate(&pref), "Create preference");
            check(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &s.workspace,
                                                        sizeof(s.workspace)),
                  "Set workspace limit");
            check(hipStreamCreate(&stream), "Create stream");
            hostA.resize(m * k);
            hostB.resize(k * n);
            hostC.resize(m * n);
            hostD.resize(m * n);
            expected.resize(m * n);
            for(int64_t i = 0; i < m * k; ++i)
                hostA[i] = __float2half(((i % m) * 3 + (i / m) * 5) % 13 / 8.0f - 0.75f);
            for(int64_t i = 0; i < k * n; ++i)
                hostB[i] = __float2half(((i % k) * 7 + (i / k) * 2) % 11 / 8.0f - 0.625f);
            for(int64_t i = 0; i < m * n; ++i)
                hostC[i] = __float2half(((i % m) + (i / m) * 3) % 7 / 4.0f - 0.75f);
            for(int64_t col = 0; col < n; ++col)
                for(int64_t row = 0; row < m; ++row)
                {
                    float sum = 0;
                    for(int64_t i = 0; i < k; ++i)
                        sum += __half2float(hostA[row + i * m]) * __half2float(hostB[i + col * k]);
                    expected[row + col * m] = __half2float(
                        __float2half(alpha * sum + beta * __half2float(hostC[row + col * m])));
                }
            // Gemm rejects a null A or B when alpha is nonzero, even when K is zero.
            check(hipMalloc(&a, std::max<size_t>(hostA.size(), 1) * sizeof(__half)), "Allocate A");
            check(hipMalloc(&b, std::max<size_t>(hostB.size(), 1) * sizeof(__half)), "Allocate B");
            check(hipMalloc(&c, hostC.size() * sizeof(__half)), "Allocate C");
            check(hipMalloc(&d, hostD.size() * sizeof(__half)), "Allocate D");
            check(hipMalloc(&workspace, s.workspace), "Allocate workspace");
            check(hipMemcpy(a, hostA.data(), hostA.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy A");
            check(hipMemcpy(b, hostB.data(), hostB.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy B");
            check(hipMemcpy(c, hostC.data(), hostC.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy C");
        }
        ~Problem()
        {
            for(void* buffer : {workspace, static_cast<void*>(d), static_cast<void*>(c),
                                static_cast<void*>(b), static_cast<void*>(a)})
                static_cast<void>(hipFree(buffer));
            static_cast<void>(hipStreamDestroy(stream));
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(dLayout);
            hipblasLtMatrixLayoutDestroy(cLayout);
            hipblasLtMatrixLayoutDestroy(bLayout);
            hipblasLtMatrixLayoutDestroy(aLayout);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }

        void reset()
        {
            // FP16 NaN, so an unwritten D fails verification.
            std::fill(hostD.begin(), hostD.end(), __float2half(NAN));
            check(hipMemcpy(d, hostD.data(), hostD.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Reset D");
        }
        void verify(const std::string& label)
        {
            check(hipStreamSynchronize(stream), "Synchronize GEMM");
            check(hipMemcpy(hostD.data(), d, hostD.size() * sizeof(__half), hipMemcpyDeviceToHost),
                  "Copy D");
            for(size_t i = 0; i < hostD.size(); ++i)
            {
                const float actual = __half2float(hostD[i]);
                if(!std::isfinite(actual)
                   || std::abs(actual - expected[i]) > 0.0005f + 0.001f * std::abs(expected[i]))
                    throw std::runtime_error(label + ": mismatch at " + std::to_string(i)
                                             + ", expected " + std::to_string(expected[i])
                                             + ", actual " + std::to_string(actual));
            }
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cerr << label << " PASS\n";
        }
        hipblasStatus_t matmul(const hipblasLtMatmulAlgo_t* algo)
        {
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   a,
                                   aLayout,
                                   b,
                                   bLayout,
                                   &beta,
                                   c,
                                   cLayout,
                                   d,
                                   dLayout,
                                   algo,
                                   workspace,
                                   s.workspace,
                                   stream);
        }
    };

    void print(Problem&                                       p,
               const char*                                    api,
               int                                            handle,
               int                                            query,
               hipblasStatus_t                                status,
               std::vector<hipblasLtMatmulHeuristicResult_t>& results)
    {
        std::ostringstream indices, workspaces, kernels;
        for(size_t i = 0; i < results.size(); ++i)
        {
            indices << (i ? "," : "") << hipblaslt_ext::getIndexFromAlgo(results[i].algo);
            workspaces << (i ? "," : "") << results[i].workspaceSize;
            kernels << (i ? ",\"" : "\"")
                    << hipblaslt_ext::getKernelNameFromAlgo(p.handle, results[i].algo) << '"';
        }
        std::lock_guard<std::mutex> lock(outputMutex);
        std::cout << "{\"api\":\"" << api << "\",\"thread\":" << p.thread
                  << ",\"handle\":" << handle << ",\"query\":" << query
                  << ",\"status\":" << status << ",\"count\":" << results.size()
                  << ",\"indices\":[" << indices.str() << "],\"workspace\":["
                  << workspaces.str() << "],\"kernels\":[" << kernels.str() << "]}"
                  << std::endl;
    }

    void queryC(Problem& p, int handle, int query)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> results(p.s.requested);
        int                                           count  = -1;
        const auto                                    status = hipblasLtMatmulAlgoGetHeuristic(
            p.handle,
            p.desc,
            p.aLayout,
            p.bLayout,
            p.cLayout,
            p.dLayout,
            p.pref,
            p.s.requested,
            results.data(),
            &count);
        if(count < 0 || count > p.s.requested)
            throw std::runtime_error("C heuristic returned count " + std::to_string(count));
        results.resize(count);
        print(p, "c", handle, query, status, results);
        for(int i = 0; p.s.run && i < count; ++i)
        {
            p.reset();
            const auto label = "C result " + std::to_string(i);
            check(p.matmul(&results[i].algo), label + " hipblasLtMatmul");
            p.verify(label);
        }
    }

    void queryCpp(Problem& p, int handle, int query)
    {
        hipblaslt_ext::Gemm gemm(p.handle,
                                 p.desc,
                                 &p.alpha,
                                 p.a,
                                 p.aLayout,
                                 p.b,
                                 p.bLayout,
                                 &p.beta,
                                 p.c,
                                 p.cLayout,
                                 p.d,
                                 p.dLayout);
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(p.s.workspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        const auto status = gemm.algoGetHeuristic(p.s.requested, pref, results);
        print(p, "cpp", handle, query, status, results);
        for(size_t i = 0; p.s.run && i < results.size(); ++i)
        {
            p.reset();
            const auto label = "C++ result " + std::to_string(i);
            check(gemm.initialize(results[i].algo, p.workspace), label + " initialize");
            check(gemm.run(p.stream), label + " run");
            p.verify(label);
        }
    }

    void queryIndices(Problem& p, int handle)
    {
        auto                                          indices = p.s.fromIndex;
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        const auto status = hipblaslt_ext::getAlgosFromIndex(p.handle, indices, results);
        print(p, "from-index", handle, 0, status, results);
        for(size_t i = 0; p.s.run && i < results.size(); ++i)
        {
            p.reset();
            const auto label = "Index result " + std::to_string(i);
            check(p.matmul(&results[i].algo), label + " hipblasLtMatmul");
            p.verify(label);
        }
    }

    // Claims a ready-N file in dir, then waits for dir/go, so the threads of
    // several processes issue their first query together.
    void waitAtBarrier(const std::filesystem::path& dir)
    {
        for(int slot = 0;; ++slot)
        {
            if(!std::filesystem::is_directory(dir) || slot == 4096)
                throw std::runtime_error("Cannot join the barrier in " + dir.string());
            const auto ready = dir / ("ready-" + std::to_string(slot));
            if(auto* file = std::fopen(ready.string().c_str(), "wx"))
            {
                std::fclose(file);
                break;
            }
        }
        while(!std::filesystem::exists(dir / "go"))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::vector<int> parseIndices(const std::string& list)
    {
        std::vector<int>  indices;
        std::stringstream stream(list);
        for(std::string item; std::getline(stream, item, ',');)
            indices.push_back(std::stoi(item));
        return indices;
    }

    Settings parse(int argc, char** argv)
    {
        Settings s;
        for(int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            auto              value = [&]() -> std::string {
                if(++i == argc)
                    throw std::invalid_argument(arg + " needs a value");
                return argv[i];
            };
            if(arg == "--api")
                s.api = value();
            else if(arg == "--requested")
                s.requested = std::stoi(value());
            else if(arg == "--m")
                s.m = std::stoll(value());
            else if(arg == "--n")
                s.n = std::stoll(value());
            else if(arg == "--k")
                s.k = std::stoll(value());
            else if(arg == "--handles")
                s.handles = std::stoi(value());
            else if(arg == "--queries")
                s.queries = std::stoi(value());
            else if(arg == "--workspace")
                s.workspace = std::stoull(value());
            else if(arg == "--null-algo")
                s.nullAlgo = true;
            else if(arg == "--no-run")
                s.run = false;
            else if(arg == "--tuned")
                s.tuned = true;
            else if(arg == "--from-index")
                s.fromIndex = parseIndices(value());
            else if(arg == "--threads")
                s.threads = std::stoi(value());
            else if(arg == "--barrier")
                s.barrier = value();
            else
                throw std::invalid_argument("Unknown argument " + arg);
        }
        if((s.api != "c" && s.api != "cpp" && s.api != "both" && s.api != "none")
           || s.requested < 1 || s.m < 1 || s.n < 1 || s.k < 0 || s.handles < 1
           || s.queries < 1 || s.threads < 1)
            throw std::invalid_argument("Invalid arguments");
        return s;
    }

    void runThread(const Settings& s, int thread)
    {
        for(int handle = 0; handle < s.handles; ++handle)
        {
            Problem problem(s, thread);
            if(handle == 0 && !s.barrier.empty())
                waitAtBarrier(s.barrier);
            for(int query = 0; query < s.queries; ++query)
            {
                if(s.api == "c" || s.api == "both")
                    queryC(problem, handle, query);
                if(s.api == "cpp" || s.api == "both")
                    queryCpp(problem, handle, query);
            }
            if(!s.fromIndex.empty())
                queryIndices(problem, handle);
            if(s.tuned)
            {
                const auto tuned = hipblaslt_ext::matmulIsTuned(problem.handle,
                                                                problem.desc,
                                                                problem.aLayout,
                                                                problem.bLayout,
                                                                problem.cLayout,
                                                                problem.dLayout);
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cout << "{\"api\":\"tuned\",\"thread\":" << thread << ",\"handle\":" << handle
                          << ",\"tuned\":" << tuned << "}" << std::endl;
            }
            if(s.nullAlgo)
            {
                problem.reset();
                const auto status = problem.matmul(nullptr);
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    std::cout << "{\"api\":\"null-algo\",\"thread\":" << thread
                              << ",\"handle\":" << handle << ",\"status\":" << status << "}"
                              << std::endl;
                }
                if(status == HIPBLAS_STATUS_SUCCESS)
                    problem.verify("hipblasLtMatmul without an algorithm");
            }
        }
    }
}

int main(int argc, char** argv)
try
{
    const auto               settings = parse(argc, argv);
    std::vector<std::string> failures(settings.threads);
    std::vector<std::thread> threads;
    for(int thread = 0; thread < settings.threads; ++thread)
        threads.emplace_back([&, thread] {
            try
            {
                runThread(settings, thread);
            }
            catch(const std::exception& e)
            {
                failures[thread] = e.what();
            }
        });
    for(auto& thread : threads)
        thread.join();
    int exitCode = 0;
    for(const auto& failure : failures)
        if(!failure.empty())
        {
            std::cerr << "FAIL: " << failure << '\n';
            exitCode = 1;
        }
    return exitCode;
}
catch(const std::exception& e)
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
