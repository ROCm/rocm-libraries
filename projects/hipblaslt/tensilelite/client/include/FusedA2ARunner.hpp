// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DeviceContext.hpp"
#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Rank r runs on device r.
        class FusedA2ARunner
        {
        public:
            static constexpr size_t kChannels       = 2;
            static constexpr size_t kFlagAllocBytes = kChannels * FUSED_A2A_FLAG_BLOCK_BYTES;

            static int  worldSize(po::variables_map const& args);
            static void configureProblem(po::variables_map const& args,
                                         ContractionProblemGemm&  problem,
                                         int                      world,
                                         int                      runIdx);

            FusedA2ARunner(Hardware const&                                    hardware,
                           ContractionProblemGemm const&                      problem,
                           ContractionSolution const&                         solution,
                           std::vector<std::shared_ptr<DeviceContext>> const& devices);
            ~FusedA2ARunner();

            FusedA2ARunner(FusedA2ARunner const&)            = delete;
            FusedA2ARunner& operator=(FusedA2ARunner const&) = delete;

            // Writes the fused fields into every card's inputArr. Entry k runs data
            // slot k % R on flag channel k % kChannels.
            std::vector<std::vector<std::vector<KernelInvocation>>> solve();

        private:
            void release();

            Hardware const&                             m_hardware;
            ContractionProblemGemm const&               m_problem;
            ContractionSolution const&                  m_solution;
            std::vector<std::shared_ptr<DeviceContext>> m_devices;

            size_t                          m_recvBytes;
            std::vector<void*>              m_recvBase;
            std::vector<std::vector<void*>> m_recv;
        };
    } // namespace Client
} // namespace TensileLite
