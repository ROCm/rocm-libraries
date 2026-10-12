// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DeviceContext.hpp"

#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <stdexcept>

namespace TensileLite
{
    namespace Client
    {
        DeviceContext::DeviceContext(int                     deviceId,
                                     bool                    useDefaultStream,
                                     size_t                  commFlagsBytes,
                                     std::vector<int> const& sdmaPeers)
            : deviceId(deviceId)
            , adapter(std::make_shared<hip::SolutionAdapter>())
            , commFlagsBytes(commFlagsBytes)
        {
            if(!useDefaultStream)
            {
                ScopedDevice guard(deviceId);
                HIP_CHECK_EXC(hipStreamCreate(&stream));
            }
            if(commFlagsBytes != 0)
            {
                ScopedDevice guard(deviceId);
                // Fine-grained: written by remote peers, must bypass stale L2.
                HIP_CHECK_EXC(
                    hipExtMallocWithFlags(&commFlags, commFlagsBytes, hipDeviceMallocFinegrained));
                HIP_CHECK_EXC(hipMemset(commFlags, 0, commFlagsBytes));
            }
            if(!sdmaPeers.empty())
            {
#ifdef TENSILELITE_ENABLE_SDMA
                ScopedDevice guard(deviceId);
                for(int peer : sdmaPeers)
                {
                    if(peer == deviceId)
                        continue;
                    hipError_t pe = hipDeviceEnablePeerAccess(peer, 0);
                    if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                        HIP_CHECK_EXC(pe);
                }
                const uint32_t node = sdmaNodeIdForDevice(deviceId);
                for(int peer : sdmaPeers)
                    sdmaQueues.push_back(std::make_shared<SdmaQueue>(
                        node, sdmaSelectEngine(node, sdmaNodeIdForDevice(peer))));
#else
                throw std::runtime_error(
                    "DeviceContext: SDMA queues requested, but this client was built without "
                    "TENSILELITE_ENABLE_SDMA.");
#endif
            }
        }

        DeviceContext::~DeviceContext()
        {
            if(commFlags != nullptr)
                static_cast<void>(hipFree(commFlags));
            if(stream != nullptr)
                static_cast<void>(hipStreamDestroy(stream));
        }
    } // namespace Client
} // namespace TensileLite
