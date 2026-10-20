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

#include "AllDevicesClient.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "DataInitialization.hpp"
#include "ResultReporter.hpp"
#include "MetaResultReporter.hpp"
#include "SolutionIterator.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            // Device-side release gate. Every device's timed stream enqueues one
            // instance of this kernel BEFORE its timing-start event, so the stream
            // blocks here until the host flips *flag. Because the flag is flipped
            // once for all devices, every device's first timed kernel begins within
            // a few hundred nanoseconds of the others regardless of host enqueue
            // skew. maxSpins bounds the wait so a lost release cannot hang the GPU.
            __global__ void allDevicesReleaseGate(uint32_t*          flag,
                                                  unsigned long long maxSpins)
            {
                unsigned long long i = 0;
                while(__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0u)
                {
                    if(++i >= maxSpins)
                        break;
                }
            }

            // Reusable host thread barrier (so the barrier launch-sync mode does not
            // need a C++20 std::barrier). All N worker threads block in wait() until
            // the Nth arrives, then all are released together.
            class ThreadBarrier
            {
            public:
                explicit ThreadBarrier(unsigned count)
                    : m_threshold(count)
                    , m_count(count)
                    , m_generation(0)
                {
                }

                void wait()
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    unsigned                     gen = m_generation;
                    if(--m_count == 0)
                    {
                        m_generation++;
                        m_count = m_threshold;
                        m_cv.notify_all();
                    }
                    else
                    {
                        m_cv.wait(lock, [this, gen] { return gen != m_generation; });
                    }
                }

            private:
                std::mutex              m_mutex;
                std::condition_variable m_cv;
                unsigned                m_threshold;
                unsigned                m_count;
                unsigned                m_generation;
            };

            double median(std::vector<double> v)
            {
                if(v.empty())
                    return std::numeric_limits<double>::quiet_NaN();
                std::sort(v.begin(), v.end());
                size_t n = v.size();
                return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
            }

            double mean(std::vector<double> const& v)
            {
                if(v.empty())
                    return std::numeric_limits<double>::quiet_NaN();
                return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
            }

            // Per-GPU GFLOPS from a per-enqueue time, matching BenchmarkTimer's
            // flopCount / time_us / 1000 convention.
            double gflopsFromUs(double flopCount, double timeUs)
            {
                if(!(timeUs > 0.0))
                    return std::numeric_limits<double>::quiet_NaN();
                return flopCount / timeUs / 1000.0;
            }
        } // namespace

        int runAllDevices(po::variables_map const&                                       args,
                          std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
                          std::shared_ptr<Hardware>                                      hardware,
                          ClientProblemFactory const&                                    problemFactory)
        {
            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
            if(deviceCount < 1)
            {
                std::cerr << "[all-devices] no HIP devices visible" << std::endl;
                return 1;
            }
            // "all" == every HIP-visible device (already filtered by
            // ROCR/HIP_VISIBLE_DEVICES by the runtime).
            std::vector<int> devices(deviceCount);
            std::iota(devices.begin(), devices.end(), 0);
            const int W = (int)devices.size();

            const std::string syncMode = args["launch-sync-mode"].as<std::string>();
            if(syncMode != "gate" && syncMode != "barrier")
            {
                std::cerr << "[all-devices] --launch-sync-mode must be 'gate' or 'barrier', got '"
                          << syncMode << "'" << std::endl;
                return 1;
            }

            const int warmups     = std::max(0, args["num-warmups"].as<int>());
            const int enq         = std::max(1, args["num-enqueues-per-sync"].as<int>());
            const int syncs       = std::max(1, args["num-syncs-per-benchmark"].as<int>());
            const int benchmarks  = std::max(1, args["num-benchmarks"].as<int>());
            const int launches    = enq * syncs; // timed launches per benchmark
            const bool exitOnError = args["exit-on-error"].as<bool>();

            int firstProblemIdx = args["problem-start-idx"].as<int>();
            int numProblems     = args["num-problems"].as<int>();
            auto problems        = problemFactory.problems();
            if(numProblems < 0)
                numProblems = (int)problems.size();
            int lastProblemIdx = firstProblemIdx + numProblems - 1;

            // Library directory for per-device lazy loading (same derivation as main).
            std::string libraryFile = args["library-file"].as<std::string>();
            size_t      dirPos      = libraryFile.rfind('/');
            std::string libraryDirectory
                = (dirPos != std::string::npos) ? libraryFile.substr(0, dirPos + 1) : std::string(".");
            auto const& codeObjectFiles = args["code-object"].as<std::vector<std::string>>();

            std::cout << "[all-devices] single-process run on " << W
                      << " GPU(s), launch-sync=" << syncMode << ", warmups=" << warmups
                      << ", launches/benchmark=" << launches << " (" << enq << " enq x " << syncs
                      << " syncs), benchmarks=" << benchmarks
                      << " (validation is disabled in this mode)" << std::endl;

            // --- Per-device, device-persistent setup (adapters/streams/events). ---
            std::vector<std::shared_ptr<hip::SolutionAdapter>> adapters(W);
            std::vector<hipStream_t>                           streams(W, nullptr);
            std::vector<hipEvent_t>                            startEv(W, nullptr), stopEv(W, nullptr);
            std::vector<std::shared_ptr<DataInitialization>>   dataInit(W);

            for(int i = 0; i < W; i++)
            {
                int d = devices[i];
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipStreamCreate(&streams[i]));
                HIP_CHECK_EXC(hipEventCreate(&startEv[i]));
                HIP_CHECK_EXC(hipEventCreate(&stopEv[i]));

                adapters[i] = std::make_shared<hip::SolutionAdapter>();
                for(auto const& co : codeObjectFiles)
                    (void)adapters[i]->loadCodeObjectFile(co);
                (void)adapters[i]->initializeLazyLoading(hardware->archName(), libraryDirectory);

                // Each device owns a DataInitialization bound to its own allocations.
                dataInit[i] = std::make_shared<DataInitialization>(args, problemFactory);
            }

            // Host-coherent release flag shared by all devices' gate kernels.
            uint32_t* flagHost = nullptr;
            uint32_t* flagDev  = nullptr;
            if(syncMode == "gate")
            {
                HIP_CHECK_EXC(hipSetDevice(devices[0]));
                HIP_CHECK_EXC(hipHostMalloc((void**)&flagHost,
                                            sizeof(uint32_t),
                                            hipHostMallocCoherent | hipHostMallocMapped));
                HIP_CHECK_EXC(hipHostGetDevicePointer((void**)&flagDev, flagHost, 0));
            }
            // ~tens of billions of spins is multiple seconds of timeout; the host
            // flips the flag within milliseconds in the happy path.
            const unsigned long long kGateMaxSpins = 50000000000ull;

            // --- Result CSV (summary, one row per problem/solution). ---
            // Do NOT overwrite the pipeline's standard results-file: the tuning
            // LibraryLogic step parses it and cannot read this mode's schema. Write
            // to a sidecar CSV (<results-file>_alldevices.csv) derived from it.
            std::string resultsFile = args["results-file"].as<std::string>();
            std::string sidecarFile;
            {
                size_t slash = resultsFile.find_last_of('/');
                size_t dot   = resultsFile.find_last_of('.');
                if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
                    sidecarFile
                        = resultsFile.substr(0, dot) + "_alldevices" + resultsFile.substr(dot);
                else
                    sidecarFile = resultsFile + "_alldevices.csv";
            }
            std::ofstream csv(sidecarFile.c_str());
            bool          csvOk = csv.is_open();
            if(csvOk)
                std::cout << "[all-devices] writing aggregate results CSV to " << sidecarFile
                          << std::endl;
            else
                std::cerr << "[all-devices] could not open sidecar results CSV '" << sidecarFile
                          << "'" << std::endl;
            if(csvOk)
                csv << "problem_idx,M,N,K,batch,solution_idx,solution_name,num_devices,launch_sync,"
                       "samples,time_us_median,time_us_mean,per_gpu_pflops_median,"
                       "per_gpu_pflops_mean,total_pflops_median,total_pflops_mean\n";

            auto solutionIterator = SolutionIterator::Default(library, hardware, args);
            // The solution iterator's preSolution()/checkSolution() hooks report to
            // a ResultReporter (m_reporter). Outside the normal listener chain that
            // pointer is null, so attach an empty MetaResultReporter (all report*()
            // calls become safe no-ops) to avoid a null-deref crash.
            solutionIterator->setReporter(std::make_shared<MetaResultReporter>());

            int rc = 0;

            for(int problemIdx = firstProblemIdx; problemIdx <= lastProblemIdx; problemIdx++)
            {
                auto problem = problems[problemIdx].get();
                auto gemm    = dynamic_cast<ContractionProblemGemm*>(problem);
                if(!gemm)
                {
                    std::cerr << "[all-devices] problem " << problemIdx
                              << " is not a plain GEMM; grouped GEMM is not supported in this mode"
                              << std::endl;
                    rc = 1;
                    if(exitOnError)
                        break;
                    continue;
                }

                const double flopCount = (double)gemm->flopCount();
                const size_t M         = gemm->freeSizeA(0);
                const size_t N         = gemm->freeSizeB(0);
                const size_t K         = gemm->boundSize(0);
                size_t       batch     = 1;
                for(size_t b = 0; b < gemm->batchIndices().size(); ++b)
                    batch *= gemm->batchSize(b);

                solutionIterator->preProblem(problem);

                int solutionCount = 0;
                while(solutionIterator->moreSolutionsInProblem())
                {
                    auto solution = solutionIterator->getSolution();
                    if(!solution)
                    {
                        std::cerr << "[all-devices] getSolution returned null" << std::endl;
                        rc = 1;
                        break;
                    }
                    solutionIterator->preSolution(solution.get());

                    if(!solutionIterator->runCurrentSolution())
                    {
                        solutionIterator->postSolution();
                        continue;
                    }

                    // Per-device: refresh inputs and solve this solution.
                    std::vector<std::shared_ptr<ProblemInputs>>  inputs(W);
                    std::vector<std::vector<KernelInvocation>>   kernels(W);
                    bool                                         solveOk = true;
                    for(int i = 0; i < W && solveOk; i++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(devices[i]));
                        inputs[i]  = dataInit[i]->prepareGPUInputs(problem);
                        try
                        {
                            kernels[i] = solution->solve(
                                (*problem), *inputs[i], *hardware, nullptr, 0, streams[i]);
                        }
                        catch(std::exception const& e)
                        {
                            std::cerr << "[all-devices] solve failed on device " << devices[i]
                                      << ": " << e.what() << std::endl;
                            solveOk = false;
                        }
                    }
                    if(!solveOk)
                    {
                        rc = 1;
                        solutionIterator->postSolution();
                        if(exitOnError)
                            break;
                        continue;
                    }

                    // Warmups (sequential; not timed).
                    for(int w = 0; w < warmups; w++)
                    {
                        for(int i = 0; i < W; i++)
                        {
                            HIP_CHECK_EXC(hipSetDevice(devices[i]));
                            HIP_CHECK_EXC(adapters[i]->launchKernels(
                                kernels[i], streams[i], (hipEvent_t) nullptr, (hipEvent_t) nullptr));
                        }
                    }
                    for(int i = 0; i < W; i++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(devices[i]));
                        HIP_CHECK_EXC(hipStreamSynchronize(streams[i]));
                    }

                    // Per-device time-per-enqueue samples, one per benchmark.
                    std::vector<std::vector<double>> perDeviceUs(W);
                    bool                             timingOk = true;

                    for(int b = 0; b < benchmarks && timingOk; b++)
                    {
                        std::vector<float> elapsedMs(W, 0.0f);

                        if(syncMode == "gate")
                        {
                            __atomic_store_n(flagHost, 0u, __ATOMIC_RELEASE);
                            for(int i = 0; i < W; i++)
                            {
                                HIP_CHECK_EXC(hipSetDevice(devices[i]));
                                // Plain ints for the launch config: TensileLite's
                                // geom.hpp defines a `dim3` alias that would otherwise
                                // shadow HIP's ::dim3 here.
                                hipLaunchKernelGGL(allDevicesReleaseGate,
                                                   ::dim3(1),
                                                   ::dim3(1),
                                                   0,
                                                   streams[i],
                                                   flagDev,
                                                   kGateMaxSpins);
                                HIP_CHECK_EXC(hipEventRecord(startEv[i], streams[i]));
                                for(int s = 0; s < syncs; s++)
                                    for(int e = 0; e < enq; e++)
                                        HIP_CHECK_EXC(
                                            adapters[i]->launchKernels(kernels[i],
                                                                       streams[i],
                                                                       (hipEvent_t) nullptr,
                                                                       (hipEvent_t) nullptr));
                                HIP_CHECK_EXC(hipEventRecord(stopEv[i], streams[i]));
                            }
                            // Give every stream time to reach its gate kernel, then
                            // release all devices at once.
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                            __atomic_store_n(flagHost, 1u, __ATOMIC_RELEASE);

                            for(int i = 0; i < W; i++)
                            {
                                HIP_CHECK_EXC(hipSetDevice(devices[i]));
                                HIP_CHECK_EXC(hipEventSynchronize(stopEv[i]));
                                HIP_CHECK_EXC(
                                    hipEventElapsedTime(&elapsedMs[i], startEv[i], stopEv[i]));
                            }
                        }
                        else // barrier
                        {
                            ThreadBarrier          barrier((unsigned)W);
                            std::vector<std::thread> threads;
                            std::vector<hipError_t>  threadErr(W, hipSuccess);
                            threads.reserve(W);
                            for(int i = 0; i < W; i++)
                            {
                                threads.emplace_back([&, i]() {
                                    hipError_t err = hipSetDevice(devices[i]);
                                    // Align the host-side enqueue across all threads.
                                    barrier.wait();
                                    if(err == hipSuccess)
                                        err = hipEventRecord(startEv[i], streams[i]);
                                    for(int s = 0; s < syncs && err == hipSuccess; s++)
                                        for(int e = 0; e < enq && err == hipSuccess; e++)
                                            err = adapters[i]->launchKernels(kernels[i],
                                                                             streams[i],
                                                                             (hipEvent_t) nullptr,
                                                                             (hipEvent_t) nullptr);
                                    if(err == hipSuccess)
                                        err = hipEventRecord(stopEv[i], streams[i]);
                                    if(err == hipSuccess)
                                        err = hipStreamSynchronize(streams[i]);
                                    if(err == hipSuccess)
                                        err = hipEventElapsedTime(
                                            &elapsedMs[i], startEv[i], stopEv[i]);
                                    threadErr[i] = err;
                                });
                            }
                            for(auto& t : threads)
                                t.join();
                            for(int i = 0; i < W; i++)
                                if(threadErr[i] != hipSuccess)
                                {
                                    std::cerr << "[all-devices] device " << devices[i]
                                              << " failed during timed run: "
                                              << hipGetErrorString(threadErr[i]) << std::endl;
                                    timingOk = false;
                                }
                        }

                        if(!timingOk)
                            break;
                        for(int i = 0; i < W; i++)
                            perDeviceUs[i].push_back((double)elapsedMs[i] * 1000.0
                                                     / (double)launches);
                    }

                    if(!timingOk)
                    {
                        rc = 1;
                        solutionIterator->postSolution();
                        if(exitOnError)
                            break;
                        continue;
                    }

                    // Aggregate: pool every (device, benchmark) per-enqueue sample.
                    std::vector<double> allSamples;
                    for(int i = 0; i < W; i++)
                        allSamples.insert(
                            allSamples.end(), perDeviceUs[i].begin(), perDeviceUs[i].end());

                    const double medUs  = median(allSamples);
                    const double meanUs = mean(allSamples);
                    const double perGpuPflopsMedian = gflopsFromUs(flopCount, medUs) / 1e6;
                    const double perGpuPflopsMean   = gflopsFromUs(flopCount, meanUs) / 1e6;
                    const double totalPflopsMedian  = perGpuPflopsMedian * (double)W;
                    const double totalPflopsMean    = perGpuPflopsMean * (double)W;

                    std::cout << std::fixed << std::setprecision(3);
                    std::cout << "[all-devices] problem " << problemIdx << " [" << M << "x" << N
                              << "x" << K << " batch " << batch << "] solution #" << solutionCount
                              << " " << solution->name() << "\n";
                    for(int i = 0; i < W; i++)
                        std::cout << "[all-devices]   dev " << devices[i]
                                  << ": time_us median=" << median(perDeviceUs[i])
                                  << " mean=" << mean(perDeviceUs[i]) << "\n";
                    std::cout << "[all-devices]   across " << W << " GPU(s) (" << allSamples.size()
                              << " samples): time_us median=" << medUs << " mean=" << meanUs
                              << "\n";
                    std::cout << "[all-devices]   per-GPU PFLOPS: median=" << perGpuPflopsMedian
                              << " mean=" << perGpuPflopsMean
                              << " | total-node PFLOPS: median=" << totalPflopsMedian
                              << " mean=" << totalPflopsMean << std::endl;

                    if(csvOk)
                    {
                        csv << problemIdx << "," << M << "," << N << "," << K << "," << batch << ","
                            << solutionCount << ",\"" << solution->name() << "\"," << W << ","
                            << syncMode << "," << allSamples.size() << "," << medUs << "," << meanUs
                            << "," << perGpuPflopsMedian << "," << perGpuPflopsMean << ","
                            << totalPflopsMedian << "," << totalPflopsMean << "\n";
                        csv.flush();
                    }

                    solutionIterator->postSolution();
                    solutionCount++;
                }

                solutionIterator->postProblem();
                if(rc != 0 && exitOnError)
                    break;
            }

            // Cleanup.
            for(int i = 0; i < W; i++)
            {
                HIP_CHECK_EXC(hipSetDevice(devices[i]));
                if(startEv[i])
                    (void)hipEventDestroy(startEv[i]);
                if(stopEv[i])
                    (void)hipEventDestroy(stopEv[i]);
                if(streams[i])
                    (void)hipStreamDestroy(streams[i]);
            }
            if(flagHost)
                (void)hipHostFree(flagHost);

            if(csvOk)
                csv.close();

            return rc;
        }
    } // namespace Client
} // namespace TensileLite
