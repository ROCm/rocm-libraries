// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
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
            // sdmaRotate: steps each remote peer queue takes along the even xGMI engines
            // (sdmaRotateEngine) from KFD's recommendation; the loopback queue keeps it.
            DeviceContext(int                     deviceId,
                          bool                    useDefaultStream,
                          size_t                  commFlagsBytes = 0,
                          std::vector<int> const& sdmaPeers      = {},
                          int                     sdmaRotate     = 0);
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

        constexpr int kFusedA2ASdmaRotateDefault = -1;

        // Moves engine `steps` places around the ring 2, 4, ..., 14 (gfx950 xGMI engines).
        inline uint32_t sdmaRotateEngine(uint32_t engine, int steps)
        {
            if(steps == 0)
                return engine;
            if(engine < 2 || engine > 14 || engine % 2 != 0)
                throw std::runtime_error("SDMA rotation needs an even xGMI engine in 2..14; got "
                                         + std::to_string(engine));
            return 2 + 2 * ((((int)engine - 2) / 2 + steps % 7 + 7) % 7);
        }

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
        // problem is fused.
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
