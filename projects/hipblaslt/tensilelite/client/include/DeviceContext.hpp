// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        class DataInitialization;
        class SdmaQueue;

        class ScopedDevice
        {
        public:
            explicit ScopedDevice(int deviceId)
            {
                HIP_CHECK_EXC(hipGetDevice(&m_prevDeviceId));
                HIP_CHECK_EXC(hipSetDevice(deviceId));
            }

            ~ScopedDevice()
            {
                static_cast<void>(hipSetDevice(m_prevDeviceId));
            }

            ScopedDevice(ScopedDevice const&)            = delete;
            ScopedDevice& operator=(ScopedDevice const&) = delete;

        private:
            int m_prevDeviceId = 0;
        };

        struct DeviceContext
        {
            // sdmaPeers: device ids to enable P2P to and open one SDMA queue each for,
            // indexed by rank; the entry equal to deviceId is a loopback queue.
            DeviceContext(int                     deviceId,
                          bool                    useDefaultStream,
                          size_t                  commFlagsBytes = 0,
                          std::vector<int> const& sdmaPeers      = {});
            ~DeviceContext();

            DeviceContext(DeviceContext const&)            = delete;
            DeviceContext& operator=(DeviceContext const&) = delete;

            int                                         deviceId;
            hipStream_t                                 stream = nullptr;
            std::shared_ptr<hip::SolutionAdapter>       adapter;
            std::shared_ptr<DataInitialization>         dataInit;
            std::shared_ptr<ProblemInputs>              inputs;
            std::vector<std::shared_ptr<ProblemInputs>> inputArr;
            size_t                                      commFlagsBytes;
            void*                                       commFlags = nullptr;
            std::vector<std::shared_ptr<SdmaQueue>>     sdmaQueues;

        private:
            void release();
        };

        constexpr size_t kCommChannels   = 2;
        constexpr size_t kCommFlagsBytes = kCommChannels * FUSED_A2A_FLAG_BLOCK_BYTES;

        template <typename F>
        void forEachDevice(std::vector<std::shared_ptr<DeviceContext>> const& devices, F&& f)
        {
            for(auto const& ctx : devices)
            {
                ScopedDevice guard(ctx->deviceId);
                f(*ctx);
            }
        }

        // devices: the problem's ranks, rank r on devices[r]. Both are no-ops unless the
        // problem is a fused GEMM+A2A.
        void resetCommBuffers(ContractionProblem const&                          problem,
                              std::vector<std::shared_ptr<DeviceContext>> const& devices);

        // Copies devices[rank]'s inputs into scratch and adds the peers of the launch's slot
        // and channel.
        ProblemInputs const* rankInputs(ContractionInputs&                                 scratch,
                                        ContractionProblem const&                          problem,
                                        ContractionSolution const&                         solution,
                                        Hardware const&                                    hardware,
                                        std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                        size_t                                             rank,
                                        size_t                                             launch);
    } // namespace Client
} // namespace TensileLite
