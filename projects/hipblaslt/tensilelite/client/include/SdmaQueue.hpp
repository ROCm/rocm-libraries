// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side SDMA queue management for the fused GEMM+AllToAll SDMA offload
// route: allocates the ring, creates the KFD SDMA queue, and exports the ring
// base and HsaQueueResource that the kernarg packer reads.
//
// Header-only, and therefore hsakmt-DEPENDENT: including it requires the
// hsakmt/hsa headers on the include path. Only TUs in tensilelite-client-common
// have that (the dependency is PRIVATE to that target), and the sole includer
// gates itself on TENSILELITE_ENABLE_SDMA.

#pragma once

#include <hip/hip_runtime.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // 256KB SDMA ring. wptr/rptr/doorbell are monotonically increasing
        // BYTE counts; wrap happens only when indexing into the ring
        // (index % SDMA_QUEUE_SIZE).
        constexpr uint32_t SDMA_QUEUE_SIZE = 256 * 1024;

        // A NAMED namespace, not an anonymous one: the state below must be one
        // object per PROCESS, and an anonymous namespace in a header gives each
        // TU its own copy -- `inline` would not merge them either, since each
        // TU's entity is distinct.
        namespace detail
        {
            inline void checkHip(hipError_t e, const char* what, const char* file, int line)
            {
                if(e != hipSuccess)
                    throw std::runtime_error(std::string("HIP error at ") + file + ":"
                                             + std::to_string(line) + " - " + what + " ("
                                             + hipGetErrorString(e) + ")");
            }

            inline void checkHsakmt(HSAKMT_STATUS s, const char* what, const char* file, int line)
            {
                if(s != HSAKMT_STATUS_SUCCESS)
                    throw std::runtime_error(std::string("HSAKMT error ") + std::to_string((int)s)
                                             + " at " + file + ":" + std::to_string(line) + " - "
                                             + what);
            }

            inline void checkHsa(hsa_status_t s, const char* what, const char* file, int line)
            {
                if(s != HSA_STATUS_SUCCESS && s != HSA_STATUS_INFO_BREAK)
                {
                    const char* msg = nullptr;
                    hsa_status_string(s, &msg);
                    throw std::runtime_error(std::string("HSA error at ") + file + ":"
                                             + std::to_string(line) + " - " + what + " ("
                                             + (msg ? msg : "?") + ")");
                }
            }

            // HSA + KFD are process-global; initialize once.
            //
            // Indexed by HIP DEVICE ORDINAL, which is what every caller has and
            // is not what hsa_iterate_agents hands back: HSA enumerates in its
            // own order and ignores HIP_VISIBLE_DEVICES, so under it the two
            // orderings name different cards. ensureHsaKfd() reindexes once, at
            // the only place the vector is built, rather than leaving raw
            // enumeration order around for callers to get right.
            inline std::once_flag           gHsaInitFlag;
            inline std::vector<hsa_agent_t> gGpuAgentsByHipDevice;

            inline hsa_status_t gpuAgentCb(hsa_agent_t agent, void* data)
            {
                auto*             agents = static_cast<std::vector<hsa_agent_t>*>(data);
                hsa_device_type_t type{};
                hsa_status_t      st = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
                if(st != HSA_STATUS_SUCCESS)
                    return st;
                if(type == HSA_DEVICE_TYPE_GPU)
                    agents->push_back(agent);
                return HSA_STATUS_SUCCESS;
            }
        } // namespace detail

// Scoped to this header: #undef'd at the bottom, after the last use, so a
// header-only include does not leak three very generic names.
#define CHK_HIP(cmd) ::TensileLite::Client::detail::checkHip((cmd), #cmd, __FILE__, __LINE__)
#define CHK_KMT(cmd) ::TensileLite::Client::detail::checkHsakmt((cmd), #cmd, __FILE__, __LINE__)
#define CHK_HSA(cmd) ::TensileLite::Client::detail::checkHsa((cmd), #cmd, __FILE__, __LINE__)

        namespace detail
        {
            inline void ensureHsaKfd()
            {
                std::call_once(gHsaInitFlag, [] {
                    CHK_HSA(hsa_init());
                    std::vector<hsa_agent_t> hsaOrder;
                    CHK_HSA(hsa_iterate_agents(&gpuAgentCb, &hsaOrder));
                    CHK_KMT(hsaKmtOpenKFD());
                    HsaSystemProperties props{};
                    CHK_KMT(hsaKmtAcquireSystemProperties(&props));

                    // Reindex HSA's enumeration into HIP device order, keyed on
                    // PCI domain+bus+device so the result does not depend on how
                    // either runtime filters. Agents HIP cannot see are dropped.
                    int hipCount = 0;
                    CHK_HIP(hipGetDeviceCount(&hipCount));
                    gGpuAgentsByHipDevice.reserve(hipCount);
                    for(int d = 0; d < hipCount; ++d)
                    {
                        hipDeviceProp_t prop{};
                        CHK_HIP(hipGetDeviceProperties(&prop, d));

                        const hsa_agent_t* match = nullptr;
                        for(const hsa_agent_t& agent : hsaOrder)
                        {
                            uint32_t bdfid = 0, domain = 0;
                            CHK_HSA(hsa_agent_get_info(
                                agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdfid));
                            CHK_HSA(hsa_agent_get_info(
                                agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &domain));
                            // BDFID packs bus into [15:8] and device into [7:3].
                            if((int)domain == prop.pciDomainID
                               && (int)((bdfid >> 8) & 0xFF) == prop.pciBusID
                               && (int)((bdfid >> 3) & 0x1F) == prop.pciDeviceID)
                            {
                                match = &agent;
                                break;
                            }
                        }
                        if(!match)
                            throw std::runtime_error(
                                "ensureHsaKfd: no HSA GPU agent matches HIP device "
                                + std::to_string(d) + " at PCI domain "
                                + std::to_string(prop.pciDomainID) + " bus "
                                + std::to_string(prop.pciBusID) + " device "
                                + std::to_string(prop.pciDeviceID) + " ("
                                + std::to_string(hsaOrder.size()) + " GPU agents visible to HSA)");
                        gGpuAgentsByHipDevice.push_back(*match);
                    }
                });
            }
        } // namespace detail

        // ---- Topology helpers ---------------------------------------------
        // KFD topology node id for a HIP device ordinal (via the HSA agent's
        // NODE info). Initializes HSA + KFD on first call, which is also what
        // puts the agent vector into HIP device order.
        inline uint32_t sdmaNodeIdForDevice(int hipDeviceId)
        {
            detail::ensureHsaKfd();
            if(hipDeviceId < 0 || hipDeviceId >= (int)detail::gGpuAgentsByHipDevice.size())
                throw std::runtime_error("sdmaNodeIdForDevice: HIP device "
                                         + std::to_string(hipDeviceId) + " out of range ("
                                         + std::to_string(detail::gGpuAgentsByHipDevice.size())
                                         + " GPU agents)");
            uint32_t node = 0;
            CHK_HSA(hsa_agent_get_info(
                detail::gGpuAgentsByHipDevice[hipDeviceId], HSA_AGENT_INFO_NODE, &node));
            return node;
        }

        // SDMA engine id to use for the srcNode->dstNode link: the first engine
        // in KFD's RecSdmaEngIdMask for that io-link. Everything else -- loopback
        // (no io-link), an empty mask, or a failed KFD query -- yields the general
        // engine 0.
        inline uint32_t sdmaSelectEngine(uint32_t srcNode, uint32_t dstNode)
        {
            detail::ensureHsaKfd();
            if(srcNode == dstNode)
                return 0;

            HsaNodeProperties props{};
            if(hsaKmtGetNodeProperties(srcNode, &props) != HSAKMT_STATUS_SUCCESS
               || props.NumIOLinks == 0)
                return 0;

            std::vector<HsaIoLinkProperties> links(props.NumIOLinks);
            if(hsaKmtGetNodeIoLinkProperties(srcNode, props.NumIOLinks, links.data())
               != HSAKMT_STATUS_SUCCESS)
                return 0;

            for(const auto& link : links)
            {
                if(link.NodeTo == dstNode)
                {
                    uint32_t mask = link.RecSdmaEngIdMask;
                    // First engine set in the recommended mask (one queue per
                    // peer -- no fan-out over multiple engines).
                    for(uint32_t b = 0; b < 32; ++b)
                        if(mask & (1u << b))
                            return b;
                    break;
                }
            }
            return 0;
        }

        // One SDMA queue: owns a 256KB Uncached ring and the KFD queue resource.
        // Non-copyable (owns HW resources). The two software cursors are not
        // here: they are a u64 pair in the client's counter buffer
        // (FusedA2ACounterSentinel.hpp).
        //
        // localNode / engineId are KFD topology ids; use sdmaNodeIdForDevice()
        // and sdmaSelectEngine() above to derive them from a HIP device id.
        class SdmaQueue
        {
        public:
            SdmaQueue(uint32_t localNode, uint32_t engineId)
            {
                detail::ensureHsaKfd();

                // Ring: NonPaged + HostAccess + ExecuteAccess + Uncached, 4KB pages.
                // Uncached is load-bearing (packet writes bypass L2 -> no flush).
                HsaMemFlags memFlags{};
                memFlags.ui32.NonPaged      = 1;
                memFlags.ui32.HostAccess    = 1;
                memFlags.ui32.PageSize      = HSA_PAGE_SIZE_4KB;
                memFlags.ui32.NoNUMABind    = 1;
                memFlags.ui32.ExecuteAccess = 1;
                memFlags.ui32.Uncached      = 1;

                // ~SdmaQueue() will not run on a throw here, so run the same
                // teardown on any exception before rethrowing.
                try
                {
                    CHK_KMT(hsaKmtAllocMemory(localNode, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_));
                    CHK_KMT(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, nullptr));

                    std::memset(&queue_, 0, sizeof(HsaQueueResource));
                    CHK_KMT(hsaKmtCreateQueueExt(localNode,
                                                 HSA_QUEUE_SDMA_BY_ENG_ID,
                                                 100, // queue percentage
                                                 HSA_QUEUE_PRIORITY_MAXIMUM,
                                                 engineId,
                                                 queueBuffer_,
                                                 SDMA_QUEUE_SIZE,
                                                 nullptr,
                                                 &queue_));
                }
                catch(...)
                {
                    teardown();
                    throw;
                }
            }

            ~SdmaQueue()
            {
                teardown();
            }

            SdmaQueue(const SdmaQueue&)            = delete;
            SdmaQueue& operator=(const SdmaQueue&) = delete;

            // Not part of HsaQueueResource: we pass it INTO hsaKmtCreateQueueExt.
            void* ringBase() const
            {
                return queueBuffer_;
            }

            const HsaQueueResource& queueResource() const
            {
                return queue_;
            }

        private:
            // Best-effort release, shared by the destructor and the ctor's failure
            // path, so it must stay safe after a partial construction.
            void teardown() noexcept
            {
                if(queue_.QueueId)
                {
                    (void)hsaKmtDestroyQueue(queue_.QueueId);
                    queue_.QueueId = 0;
                }
                if(queueBuffer_)
                {
                    (void)hsaKmtUnmapMemoryToGPU(queueBuffer_);
                    (void)hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE);
                    queueBuffer_ = nullptr;
                }
            }

            void*            queueBuffer_ = nullptr; // ring (Uncached)
            HsaQueueResource queue_{}; // KFD queue resource
        };

    } // namespace Client
} // namespace TensileLite

// Last use is above; do not let these escape to includers.
#undef CHK_HIP
#undef CHK_KMT
#undef CHK_HSA
