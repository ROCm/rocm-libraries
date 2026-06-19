// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientRunScheduler.hpp"

#include "TimingInstrumentation.hpp"

#include <Tensile/hip/HipUtils.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            std::string problemProgressString(int problemIdx, int lastProblemIdx)
            {
                return std::to_string(problemIdx) + "/" + std::to_string(lastProblemIdx);
            }
        } // namespace

        ClientRunScheduler::ClientRunScheduler(ClientRunSchedulerConfig config,
                                               ClientRunSchedulerDependencies dependencies)
            : m_config(std::move(config))
            , m_deps(std::move(dependencies))
        {
            if(!m_deps.problems || !m_deps.listeners || !m_deps.reporter || !m_deps.dataCoordinator
               || !m_deps.solutionSource || !m_deps.kernelLauncher || !m_deps.hardware)
            {
                throw std::invalid_argument("ClientRunScheduler requires non-null dependencies");
            }

            if(!m_deps.callbacks.kernelHotPathSizeFn)
                m_deps.callbacks.kernelHotPathSizeFn = IcacheRotationPolicy::defaultKernelHotPathSizeFn();
        }

        void ClientRunScheduler::maybeLoadAutoIcacheRotation(size_t inputSlotCount)
        {
            if(m_config.icacheRotateCopies != -1)
                return;

            int nRotationModules = m_deps.kernelLauncher->numRotationModules();
            if(!m_icacheRotationPolicy.shouldLoadAutoCopies(
                   m_config.icacheRotateCopies, nRotationModules))
                return;

            auto plan = m_icacheRotationPolicy.computeAutoPlan(inputSlotCount,
                                                               m_config.codeObjectFilenames,
                                                               m_config.icacheRotateSizeKB,
                                                               m_deps.callbacks.kernelHotPathSizeFn);

            if(plan.extras > 0)
            {
                ScopedTimer timer("icache_rotate_extra_copies_loading");
                for(auto const& filename : m_config.codeObjectFilenames)
                {
                    HIP_CHECK_EXC(m_deps.kernelLauncher->loadCodeObjectFileExtraCopies(
                        filename, plan.extras));
                }
            }

#if defined(__linux__)
            if(plan.kernelHotPathSize == 0)
            {
                std::cerr << "[icache-rotate] warning: no label_GW_End found in any --code-object; "
                             "cache-based term contributes 0"
                          << std::endl;
            }

            std::cout << "[icache-rotate] auto extras = max("
                      << plan.extrasFromDataInit << " from inputSlotCount-1, "
                      << plan.extrasFromCache << " from " << plan.cacheBudgetBytes << "/"
                      << plan.kernelHotPathSize << ") = " << plan.extras << " (total = "
                      << (plan.extras + 1) << " modules)" << std::endl;
#else
            std::cout << "[icache-rotate] auto extras = max("
                      << plan.extrasFromDataInit << " from inputSlotCount-1, "
                      << plan.extrasFromCache << " from --icache-rotate-size) = "
                      << plan.extras << " (total = " << (plan.extras + 1) << " modules)"
                      << std::endl;
#endif
        }

        ClientRunSchedulerResult ClientRunScheduler::run(void*& dUA, void*& dUAHost)
        {
            while(m_deps.listeners->needMoreBenchmarkRuns())
            {
                m_deps.listeners->preBenchmarkRun();
                auto const flushGridSize = m_deps.callbacks.flushGridSizeFn();

                for(auto icacheFlush : m_config.icacheFlushArgs)
                {
                    m_deps.callbacks.setIcacheFlushTimeUsFn(
                        icacheFlush ? m_config.icacheFlushTimeUs : 0.f);

                    IcacheRotationCursor rotationCursor;

                    for(int problemIdx = m_config.firstProblemIdx; problemIdx <= m_config.lastProblemIdx;
                        ++problemIdx)
                    {
                        auto problem = (*m_deps.problems)[problemIdx].get();

                        m_deps.reporter->reportProblemIndex(problemIdx);
                        m_deps.reporter->reportProblemProgress(
                            problemProgressString(problemIdx, m_config.lastProblemIdx));

                        {
                            ScopedTimer timer("pre_problem");
                            m_deps.listeners->preProblem(problem);
                        }

                        {
                            ScopedTimer timer("cancel_async_reset");
                            m_deps.dataCoordinator->cancelAsyncReset();
                        }

                        std::shared_ptr<ProblemInputs> inputs;
                        {
                            ScopedTimer timer("gpu_input_preparation");
                            inputs = m_deps.dataCoordinator->prepareGPUInputs(problem);
                        }

                        size_t warmupInvocationsForSizing = m_deps.listeners->numWarmupRuns();
                        size_t syncsForSizing              = m_deps.listeners->numSyncs();
                        size_t enqForSizing                = m_deps.listeners->numEnqueuesPerSync();
                        auto rotatingOutputPlan = m_rotatingOutputPolicy.plan(
                            warmupInvocationsForSizing, syncsForSizing, enqForSizing);

                        std::vector<std::shared_ptr<ProblemInputs>> inputArr;
                        {
                            ScopedTimer timer("rotating_buffer_preparation");
                            inputArr = m_deps.dataCoordinator->prepareRotatingGPUOutput(
                                rotatingOutputPlan.maxRotatingBufferNum,
                                problem,
                                inputs,
                                m_deps.stream);
                            m_deps.callbacks.deviceSynchronizeFn();
                        }

                        maybeLoadAutoIcacheRotation(inputArr.size());

                        bool resetInput = true;
                        while(m_deps.solutionSource->moreSolutionsInProblem())
                        {
                            std::shared_ptr<ContractionSolution> solution;
                            {
                                ScopedTimer timer("solution_selection");
                                solution = m_deps.solutionSource->getSolution();
                            }

                            if(solution == nullptr)
                                throw std::runtime_error("Could not find a solution");

                            {
                                ScopedTimer timer("pre_solution");
                                m_deps.listeners->preSolution(solution.get());
                            }

                            if(m_deps.solutionSource->runCurrentSolution() && m_config.runKernels)
                            {
                                try
                                {
                                    while(m_deps.listeners->needMoreRunsInSolution())
                                    {
                                        if(resetInput)
                                        {
                                            ScopedTimer timer("gpu_input_reset");
                                            inputs = m_deps.dataCoordinator->prepareGPUInputs(
                                                problem);
                                            inputArr[0] = inputs;
                                        }
                                        resetInput = true;

                                        std::vector<std::vector<KernelInvocation>> kernels;
                                        {
                                            ScopedTimer timer("kernel_solving");
                                            for(size_t r = 0; r < inputArr.size(); ++r)
                                            {
                                                auto kernel = m_config.useUserArgs
                                                                  ? solution->solveTensileGPU(
                                                                        *problem,
                                                                        *inputArr[r],
                                                                        *m_deps.hardware,
                                                                        &dUA,
                                                                        &dUAHost,
                                                                        nullptr,
                                                                        0,
                                                                        m_deps.stream)
                                                                  : solution->solve(*problem,
                                                                                    *inputArr[r],
                                                                                    *m_deps.hardware,
                                                                                    nullptr,
                                                                                    0,
                                                                                    m_deps.stream);
                                                kernels.push_back(kernel);
                                            }
                                        }

                                        size_t warmupInvocations
                                            = m_deps.listeners->numWarmupRuns();
                                        size_t warmupEventCount = kernels[0].size();
                                        TimingEvents warmupStartEvents(warmupInvocations,
                                                                       warmupEventCount);
                                        TimingEvents warmupStopEvents(warmupInvocations,
                                                                      warmupEventCount);

                                        int nRotationModules = m_deps.kernelLauncher->numRotationModules();
                                        assert(nRotationModules > 0);
                                        auto rotateAndSelect = [&]() {
                                            m_deps.kernelLauncher->selectRotationCopy(
                                                rotationCursor.nextIndex(nRotationModules));
                                        };

                                        {
                                            ScopedTimer timer("wait_copy_done");
                                            m_deps.dataCoordinator->waitCopyDone(m_deps.stream);
                                        }

                                        if(warmupInvocations > 0)
                                        {
                                            {
                                                ScopedTimer timer("warmup_runs");
                                                m_deps.listeners->preWarmup();
                                                HIP_CHECK_EXC(m_deps.kernelLauncher->launchKernels(
                                                    kernels[0],
                                                    m_deps.stream,
                                                    warmupStartEvents[0],
                                                    warmupStopEvents[0]));
                                            }

                                            {
                                                ScopedTimer timer("validate_warmups");
                                                m_deps.listeners->validateWarmups(
                                                    inputs, warmupStartEvents, warmupStopEvents);
                                            }

                                            {
                                                ScopedTimer timer("warmup_runs");
                                                for(int i = 1; i < static_cast<int>(warmupInvocations);
                                                    ++i)
                                                {
                                                    size_t kIdx = static_cast<size_t>(i)
                                                                  % kernels.size();
                                                    HIP_CHECK_EXC(m_deps.kernelLauncher->launchKernels(
                                                        kernels[kIdx],
                                                        m_deps.stream,
                                                        warmupStartEvents[static_cast<size_t>(i)],
                                                        warmupStopEvents[static_cast<size_t>(i)]));
                                                }
                                                m_deps.listeners->postWarmup(
                                                    warmupStartEvents, warmupStopEvents, m_deps.stream);
                                            }
                                        }

#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
                                        TimingEvents ProfilerStartEvents(1, warmupEventCount);
                                        TimingEvents ProfilerStopEvents(1, warmupEventCount);
                                        m_deps.listeners->preProfiler();
                                        rotateAndSelect();
                                        HIP_CHECK_EXC(m_deps.kernelLauncher->launchKernels(
                                            kernels[warmupInvocations % kernels.size()],
                                            m_deps.stream,
                                            ProfilerStartEvents[0],
                                            ProfilerStopEvents[0]));
                                        m_deps.listeners->postProfiler();
#endif

                                        size_t eventCount = m_config.gpuTimer ? kernels[0].size() : 0;
                                        size_t syncs = m_deps.listeners->numSyncs();
                                        size_t enq   = m_deps.listeners->numEnqueuesPerSync();
                                        bool   ranBenchmarkEnqueue = false;

                                        {
                                            ScopedTimer timer("benchmark_runs");
                                            m_deps.listeners->preSyncs();
                                            if(enq)
                                            {
                                                for(size_t i = 0; i < syncs; ++i)
                                                {
                                                    TimingEvents startEvents(enq, eventCount);
                                                    TimingEvents stopEvents(enq, eventCount);

                                                    m_deps.listeners->preEnqueues(m_deps.stream);

                                                    for(size_t j = 0; j < enq; ++j)
                                                    {
                                                        size_t kIdx = ((i * enq) + j) % kernels.size();
                                                        rotateAndSelect();
                                                        HIP_CHECK_EXC(
                                                            m_deps.kernelLauncher->launchKernels(
                                                                kernels[kIdx],
                                                                m_deps.stream,
                                                                nullptr,
                                                                nullptr));
                                                        ranBenchmarkEnqueue = true;

                                                        if(icacheFlush)
                                                        {
                                                            m_deps.callbacks.flushIcacheFn(
                                                                flushGridSize, m_deps.stream);
                                                        }
                                                    }

                                                    m_deps.listeners->postEnqueues(
                                                        startEvents, stopEvents, m_deps.stream);
                                                    m_deps.listeners->validateEnqueues(
                                                        inputs, startEvents, stopEvents);
                                                }
                                            }

                                            m_deps.listeners->postSyncs();
                                        }

                                        if(m_config.useUserArgs)
                                        {
                                            solution->relaseDeviceUserArgs(dUA, dUAHost);
                                        }

                                        if(!ranBenchmarkEnqueue)
                                        {
                                            ScopedTimer timer("async_reset_submit");
                                            m_deps.dataCoordinator->beginAsyncReset(problem);
                                            m_deps.dataCoordinator->beginAsyncReset(problem);
                                        }
                                    }
                                }
                                catch(std::runtime_error const& err)
                                {
                                    m_deps.reporter->reportInvalid();
                                    std::string message = "Exception occurred: ";
                                    message += err.what();
                                    message += '\n';
                                    m_deps.reporter->logError(message);
                                }
                            }

                            {
                                ScopedTimer timer("post_solution");
                                m_deps.listeners->postSolution();
                            }

                            if(m_config.exitOnError && m_deps.listeners->error() > 0)
                            {
                                return {true, std::min(m_deps.listeners->error(), 255)};
                            }
                        }

                        {
                            ScopedTimer timer("post_problem");
                            m_deps.listeners->postProblem();
                        }
                    }
                }

                m_deps.listeners->postBenchmarkRun();
            }

            return {false, std::min(m_deps.listeners->error(), 255)};
        }

    } // namespace Client
} // namespace TensileLite
