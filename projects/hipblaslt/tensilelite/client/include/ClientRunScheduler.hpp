// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/Tensile.hpp>

#include "ClientRunPolicies.hpp"
#include "RunListener.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace TensileLite
{
    namespace Client
    {
        using FlushGridSizeFn = std::function<uint32_t()>;
        using FlushIcacheFn   = std::function<void(uint32_t, hipStream_t)>;
        using DeviceSynchronizeFn = std::function<void()>;
        using SetIcacheFlushTimeUsFn = std::function<void(float)>;

        struct ClientRunSchedulerConfig
        {
            int                 firstProblemIdx   = 0;
            int                 lastProblemIdx    = -1;
            bool                runKernels        = true;
            bool                exitOnError       = false;
            bool                gpuTimer          = true;
            bool                useUserArgs       = false;
            std::vector<bool>   icacheFlushArgs;
            float               icacheFlushTimeUs = 0.f;
            int                 icacheRotateCopies = 0;
            int                 icacheRotateSizeKB  = 64;
            std::vector<std::string> codeObjectFilenames;
        };

        class RunReporter
        {
        public:
            virtual ~RunReporter() = default;

            virtual void reportProblemIndex(int idx)                    = 0;
            virtual void reportProblemProgress(std::string const& text) = 0;
            virtual void reportInvalid()                                = 0;
            virtual void logError(std::string const& message)           = 0;
        };

        class RunDataCoordinator
        {
        public:
            virtual ~RunDataCoordinator() = default;

            virtual void cancelAsyncReset() = 0;
            virtual std::shared_ptr<ProblemInputs>
                prepareGPUInputs(ContractionProblem const* problem) = 0;
            virtual std::vector<std::shared_ptr<ProblemInputs>>
                prepareRotatingGPUOutput(int32_t                        maxRotatingBufferNum,
                                         ContractionProblem const*      problem,
                                         std::shared_ptr<ProblemInputs> inputs,
                                         hipStream_t                    stream) = 0;
            virtual void waitCopyDone(hipStream_t stream) = 0;
            virtual void beginAsyncReset(ContractionProblem const* problem) = 0;
        };

        class RunSolutionSource
        {
        public:
            virtual ~RunSolutionSource() = default;

            virtual bool moreSolutionsInProblem() const                     = 0;
            virtual std::shared_ptr<ContractionSolution> getSolution()     = 0;
            virtual bool runCurrentSolution()                               = 0;
        };

        class RunKernelLauncher
        {
        public:
            virtual ~RunKernelLauncher() = default;

            virtual int numRotationModules()                                 = 0;
            virtual void selectRotationCopy(int idx)                         = 0;
            virtual hipError_t loadCodeObjectFileExtraCopies(std::string const& path,
                                                            int               extraCopies) = 0;
            virtual hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                             hipStream_t                          stream,
                                             std::vector<hipEvent_t> const&       startEvents,
                                             std::vector<hipEvent_t> const&       stopEvents) = 0;
            virtual hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                             hipStream_t                          stream,
                                             hipEvent_t                           startEvent,
                                             hipEvent_t                           stopEvent) = 0;
        };

        struct ClientRunSchedulerCallbacks
        {
            FlushGridSizeFn       flushGridSizeFn       = [] { return uint32_t{0}; };
            FlushIcacheFn         flushIcacheFn         = [](uint32_t, hipStream_t) {};
            DeviceSynchronizeFn   deviceSynchronizeFn   = [] {};
            SetIcacheFlushTimeUsFn setIcacheFlushTimeUsFn = [](float) {};
            KernelHotPathSizeFn   kernelHotPathSizeFn;
        };

        struct ClientRunSchedulerDependencies
        {
            std::vector<std::shared_ptr<ContractionProblem>> const* problems = nullptr;
            RunListener*         listeners     = nullptr;
            RunReporter*         reporter      = nullptr;
            RunDataCoordinator*  dataCoordinator = nullptr;
            RunSolutionSource*   solutionSource = nullptr;
            RunKernelLauncher*   kernelLauncher = nullptr;
            Hardware const*      hardware      = nullptr;
            hipStream_t          stream        = nullptr;
            ClientRunSchedulerCallbacks callbacks;
        };

        struct ClientRunSchedulerResult
        {
            bool exitedEarly = false;
            int  returnCode  = 0;
        };

        class ClientRunScheduler
        {
        public:
            ClientRunScheduler(ClientRunSchedulerConfig config,
                               ClientRunSchedulerDependencies dependencies);

            ClientRunSchedulerResult run(void*& dUA, void*& dUAHost);

        private:
            void maybeLoadAutoIcacheRotation(size_t inputSlotCount);

            ClientRunSchedulerConfig       m_config;
            ClientRunSchedulerDependencies m_deps;
            RotatingOutputPolicy           m_rotatingOutputPolicy;
            IcacheRotationPolicy          m_icacheRotationPolicy;
        };

    } // namespace Client
} // namespace TensileLite
