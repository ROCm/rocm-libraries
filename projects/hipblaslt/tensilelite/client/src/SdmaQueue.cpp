// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of the host-side SDMA queue manager (see SdmaQueue.hpp).
// Ported closely from MORI's anvil (SdmaQueue ctor + AnvilLib topology/engine
// selection). Only host bring-up code lives here; the production packet
// producer is a GPU kernel added by a later task.

#include "SdmaQueue.hpp"

#include <hip/hip_runtime.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            void checkHip(hipError_t e, const char* what, const char* file, int line)
            {
                if(e != hipSuccess)
                    throw std::runtime_error(std::string("HIP error at ") + file + ":"
                                             + std::to_string(line) + " - " + what + " ("
                                             + hipGetErrorString(e) + ")");
            }

            void checkHsakmt(HSAKMT_STATUS s, const char* what, const char* file, int line)
            {
                if(s != HSAKMT_STATUS_SUCCESS)
                    throw std::runtime_error(std::string("HSAKMT error ") + std::to_string((int)s)
                                             + " at " + file + ":" + std::to_string(line) + " - "
                                             + what);
            }

            void checkHsa(hsa_status_t s, const char* what, const char* file, int line)
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

#define CHK_HIP(cmd) checkHip((cmd), #cmd, __FILE__, __LINE__)
#define CHK_KMT(cmd) checkHsakmt((cmd), #cmd, __FILE__, __LINE__)
#define CHK_HSA(cmd) checkHsa((cmd), #cmd, __FILE__, __LINE__)

            // HSA + KFD are process-global; initialize once. GPU agents are
            // captured in HIP-device order via the iterate-agents callback.
            std::once_flag           gHsaInitFlag;
            std::vector<hsa_agent_t> gGpuAgents;

            hsa_status_t gpuAgentCb(hsa_agent_t agent, void* data)
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

            void ensureHsaKfd()
            {
                std::call_once(gHsaInitFlag, [] {
                    CHK_HSA(hsa_init());
                    CHK_HSA(hsa_iterate_agents(&gpuAgentCb, &gGpuAgents));
                    CHK_KMT(hsaKmtOpenKFD());
                    HsaSystemProperties props{};
                    CHK_KMT(hsaKmtAcquireSystemProperties(&props));
                });
            }
        } // namespace

        uint32_t sdmaNodeIdForDevice(int hipDeviceId)
        {
            ensureHsaKfd();
            if(hipDeviceId < 0 || hipDeviceId >= (int)gGpuAgents.size())
                throw std::runtime_error("sdmaNodeIdForDevice: HIP device "
                                         + std::to_string(hipDeviceId) + " out of range ("
                                         + std::to_string(gGpuAgents.size()) + " GPU agents)");
            uint32_t node = 0;
            CHK_HSA(hsa_agent_get_info(gGpuAgents[hipDeviceId], HSA_AGENT_INFO_NODE, &node));
            return node;
        }

        uint32_t sdmaSelectEngine(uint32_t srcNode, uint32_t dstNode)
        {
            ensureHsaKfd();
            // Loopback (self) has no io-link and no recommended engine; use a
            // general (non-xGMI) SDMA engine, matching MORI's loopback path.
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
            return 0; // fall back to a general engine if KFD reports no mask
        }

        // Pimpl: holds the hsakmt types kept out of the header (the KFD queue
        // resource + the ring pointer). All KFD resource lifetime lives here.
        struct SdmaQueue::Impl
        {
            void*            queueBuffer = nullptr; // ring (Uncached)
            HsaQueueResource queue{}; // KFD queue resource
        };

        SdmaQueue::SdmaQueue(uint32_t localNode, uint32_t engineId)
            : impl_(std::make_unique<Impl>())
        {
            ensureHsaKfd();

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
                CHK_KMT(
                    hsaKmtAllocMemory(localNode, SDMA_QUEUE_SIZE, memFlags, &impl_->queueBuffer));
                CHK_KMT(hsaKmtMapMemoryToGPU(impl_->queueBuffer, SDMA_QUEUE_SIZE, nullptr));

                std::memset(&impl_->queue, 0, sizeof(HsaQueueResource));
                CHK_KMT(hsaKmtCreateQueueExt(localNode,
                                             HSA_QUEUE_SDMA_BY_ENG_ID,
                                             100, // queue percentage
                                             HSA_QUEUE_PRIORITY_MAXIMUM,
                                             engineId,
                                             impl_->queueBuffer,
                                             SDMA_QUEUE_SIZE,
                                             nullptr,
                                             &impl_->queue));

                // Software cursors in uncached device memory (shared producer state).
                CHK_HIP(hipMalloc(&deviceHandle_, sizeof(SdmaQueueDeviceHandle)));
                CHK_HIP(hipExtMallocWithFlags(
                    (void**)&cachedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
                CHK_HIP(hipExtMallocWithFlags(
                    (void**)&committedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));

                // Seed the cursors to the current HARDWARE write pointer so the
                // first reserved index is contiguous with whatever the queue was
                // created at (MORI does exactly this).
                const uint64_t hwWptr = (uint64_t) * (impl_->queue.Queue_write_ptr_aql);
                const uint64_t hwRptr = (uint64_t) * (impl_->queue.Queue_read_ptr_aql);
                hostWptr_             = hwWptr;

                hostHandle_ = SdmaQueueDeviceHandle{
                    /*queueBuf*/ static_cast<uint32_t*>(impl_->queueBuffer),
                    /*rptr*/ (uint64_t*)impl_->queue.Queue_read_ptr_aql,
                    /*wptr*/ (uint64_t*)impl_->queue.Queue_write_ptr_aql,
                    /*doorbell*/ (uint64_t*)impl_->queue.Queue_DoorBell_aql,
                    /*cachedWptr*/ cachedWptr_,
                    /*committedWptr*/ committedWptr_,
                    // Per-producer private cache seed, see SdmaQueue.hpp.
                    /*cachedHwReadIndex*/ hwRptr,
                };

                CHK_HIP(hipMemcpy(deviceHandle_,
                                  &hostHandle_,
                                  sizeof(SdmaQueueDeviceHandle),
                                  hipMemcpyHostToDevice));
                CHK_HIP(hipMemcpy(cachedWptr_, &hwWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
                CHK_HIP(
                    hipMemcpy(committedWptr_, &hwWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
            }
            catch(...)
            {
                teardown();
                throw;
            }
        }

        void SdmaQueue::teardown() noexcept
        {
            // Best-effort release, shared by the destructor and the ctor's
            // failure path; safe to call after a partial construction.
            if(impl_ && impl_->queue.QueueId)
            {
                (void)hsaKmtDestroyQueue(impl_->queue.QueueId);
                impl_->queue.QueueId = 0;
            }
            if(deviceHandle_)
            {
                (void)hipFree(deviceHandle_);
                deviceHandle_ = nullptr;
            }
            if(cachedWptr_)
            {
                (void)hipFree(cachedWptr_);
                cachedWptr_ = nullptr;
            }
            if(committedWptr_)
            {
                (void)hipFree(committedWptr_);
                committedWptr_ = nullptr;
            }
            if(impl_ && impl_->queueBuffer)
            {
                (void)hsaKmtUnmapMemoryToGPU(impl_->queueBuffer);
                (void)hsaKmtFreeMemory(impl_->queueBuffer, SDMA_QUEUE_SIZE);
                impl_->queueBuffer = nullptr;
            }
        }

        SdmaQueue::~SdmaQueue()
        {
            teardown();
        }

        uint64_t SdmaQueue::submitPacketHost(const void* pkt, size_t bytes)
        {
            if(bytes == 0 || (bytes % sizeof(uint32_t)) != 0)
                throw std::runtime_error("submitPacketHost: bytes must be a nonzero multiple of 4");
            if(bytes > SDMA_QUEUE_SIZE)
                throw std::runtime_error("submitPacketHost: packet larger than ring");

            // Byte offset into the ring for the current write cursor.
            const uint64_t offset = hostWptr_ % SDMA_QUEUE_SIZE;
            if(offset + bytes > SDMA_QUEUE_SIZE)
                throw std::runtime_error("submitPacketHost: packet would wrap the ring "
                                         "(host smoke path does not implement wrap)");

            // Ring is Uncached, so this is visible to the engine with no flush.
            std::memcpy(static_cast<uint8_t*>(impl_->queueBuffer) + offset, pkt, bytes);

            hostWptr_ += bytes;

            // Publish the new write pointer, then ring the doorbell.
            *(impl_->queue.Queue_write_ptr_aql) = hostWptr_;
            // Ensure the wptr store lands before the doorbell store.
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            *(impl_->queue.Queue_DoorBell_aql) = hostWptr_;

            return hostWptr_;
        }

        bool SdmaQueue::waitIdleHost(uint64_t timeoutSpins)
        {
            for(uint64_t i = 0; i < timeoutSpins; ++i)
            {
                const uint64_t rp
                    = (uint64_t) * (volatile HSAuint64*)(impl_->queue.Queue_read_ptr_aql);
                if(rp >= hostWptr_)
                    return true;
            }
            return false;
        }

        SdmaQueueSet::SdmaQueueSet(uint32_t localNode, const std::vector<uint32_t>& targetNodes)
        {
            ensureHsaKfd();

            std::vector<SdmaQueueDeviceHandle> handles;
            handles.reserve(targetNodes.size());
            for(uint32_t dstNode : targetNodes)
            {
                uint32_t engine = sdmaSelectEngine(localNode, dstNode);
                queues_.emplace_back(std::make_unique<SdmaQueue>(localNode, engine));
                handles.push_back(queues_.back()->hostHandle());
            }

            const size_t bytes = handles.size() * sizeof(SdmaQueueDeviceHandle);

            // Hold the allocation in a local owner until the copy succeeds, so a
            // failing hipMemcpy doesn't leak it (CHK_HIP throws, and a throw here
            // means ~SdmaQueueSet never runs).
            SdmaQueueDeviceHandle* raw = nullptr;
            CHK_HIP(hipMalloc(&raw, bytes));
            auto hipFreeDeleter = [](SdmaQueueDeviceHandle* p) { (void)hipFree(p); };
            std::unique_ptr<SdmaQueueDeviceHandle, decltype(hipFreeDeleter)> owned(raw,
                                                                                   hipFreeDeleter);
            CHK_HIP(hipMemcpy(owned.get(), handles.data(), bytes, hipMemcpyHostToDevice));
            dHandles_ = owned.release();
        }

        SdmaQueueSet::~SdmaQueueSet()
        {
            if(dHandles_)
                (void)hipFree(dHandles_);
        }

    } // namespace Client
} // namespace TensileLite
