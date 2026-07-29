// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side SDMA queue management for the fused GEMM+AllToAll SDMA offload
// route (ROCM-27524). This is the host half of "have a GPU kernel push SDMA
// packets instead of storing over xGMI from CUs": it allocates the ring,
// creates the KFD SDMA queue, and exports the device-visible handle(s) that
// later GPU-assembly tasks read to fill packets and ring the doorbell.
//
// Ported from MORI's anvil (src/application/transport/sdma/anvil.cpp
// SdmaQueue::SdmaQueue and include/mori/core/transport/sdma/anvil_device.hpp
// SdmaQueueDeviceHandle). Kept deliberately close to that known-good code:
// same allocation flags, same queue-creation call, same handle layout.
//
// Ring is Uncached (HsaMemFlags.ui32.Uncached=1): packet writes bypass L2, so
// a producer never has to flush to make a packet visible to the SDMA engine.
// Changing this to cached would reintroduce a hard-to-find visibility bug.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// NOTE: this header deliberately does NOT include any hsakmt/hsa headers. The
// device-visible handle below is the cross-task contract and must be includable
// by downstream code that has no hsakmt headers on its include path. All hsakmt
// state is confined to SdmaQueue.cpp (behind the pimpl Impl declared there).

namespace TensileLite
{
    namespace Client
    {
        // 256KB SDMA ring, matching MORI's SDMA_QUEUE_SIZE. wptr/rptr/doorbell
        // are monotonically increasing BYTE counts; wrap happens only when
        // indexing into the ring (index % SDMA_QUEUE_SIZE).
        constexpr uint32_t SDMA_QUEUE_SIZE = 256 * 1024;

        // -------------------------------------------------------------------
        // Device-visible handle. THE FIELD LAYOUT IS A CONTRACT: later
        // GPU-assembly tasks read these by fixed offset, so field order/type
        // must not change (the static_asserts below lock every offset).
        //
        // Two DIFFERENT kinds of member live here:
        //   * The first six are POINTERS into producer-SHARED memory (the ring
        //     and the four cursor/register words). Every producer sees the same
        //     memory through them.
        //   * cachedHwReadIndex is a VALUE, not a pointer, and it is NOT shared
        //     state. MORI's device code copies the whole handle into a private
        //     local (handle = *deviceHandle[q]) and mutates cachedHwReadIndex on
        //     that private copy; the copy in this struct is only the seed value
        //     (= the hardware read pointer at construction). Its job is to avoid
        //     a SYSTEM-scope read of rptr on every reserve: a producer checks
        //     the local cache first and only re-reads the hardware rptr when the
        //     cache says the ring is full (see MORI CanWriteUpto). When later
        //     tasks port the producer to assembly it will live in an SGPR, each
        //     producer maintaining its own, and will NEVER be written back to
        //     this memory. Treating it as shared state (e.g. atomically updating
        //     it) would add an unnecessary, hard-to-diagnose race whose only
        //     symptom is "occasionally spins a few extra times".
        // -------------------------------------------------------------------
        struct SdmaQueueDeviceHandle
        {
            // Producer-shared resources (pointers into shared memory). These
            // use plain uint64_t* rather than hsakmt's HSAuint64* so the header
            // stays hsakmt-free; HSAuint64 is itself a uint64_t typedef, so the
            // layout (and every static_assert below) is unchanged.
            uint32_t* queueBuf;      // ring base (Uncached)
            uint64_t* rptr;          // hardware read pointer (byte count)
            uint64_t* wptr;          // hardware write pointer (byte count)
            uint64_t* doorbell;      // doorbell (byte count)
            uint64_t* cachedWptr;    // software producer cursor (shared, uncached)
            uint64_t* committedWptr; // software committed cursor (shared, uncached)

            // Per-producer PRIVATE cache seed (a value, not shared state).
            uint64_t cachedHwReadIndex;
        };

        // Lock every field offset so a reorder / type change / reserved-gap
        // edit is caught at compile time (the layout is read by assembly later).
        static_assert(sizeof(SdmaQueueDeviceHandle) == 7 * sizeof(uint64_t),
                      "SdmaQueueDeviceHandle must be exactly 7 x 8 bytes");
        static_assert(offsetof(SdmaQueueDeviceHandle, queueBuf) == 0 * 8, "queueBuf @ 0");
        static_assert(offsetof(SdmaQueueDeviceHandle, rptr) == 1 * 8, "rptr @ 8");
        static_assert(offsetof(SdmaQueueDeviceHandle, wptr) == 2 * 8, "wptr @ 16");
        static_assert(offsetof(SdmaQueueDeviceHandle, doorbell) == 3 * 8, "doorbell @ 24");
        static_assert(offsetof(SdmaQueueDeviceHandle, cachedWptr) == 4 * 8, "cachedWptr @ 32");
        static_assert(offsetof(SdmaQueueDeviceHandle, committedWptr) == 5 * 8, "committedWptr @ 40");
        static_assert(offsetof(SdmaQueueDeviceHandle, cachedHwReadIndex) == 6 * 8,
                      "cachedHwReadIndex @ 48");

        // -------------------------------------------------------------------
        // One SDMA queue: owns a 256KB Uncached ring, the KFD queue resource,
        // the two software cursors (uncached device memory), and a device copy
        // of its SdmaQueueDeviceHandle. Non-copyable (owns HW resources).
        //
        // localNode / engineId are KFD topology ids; use sdmaNodeIdForDevice()
        // and sdmaSelectEngine() (below) to derive them from a HIP device id.
        // -------------------------------------------------------------------
        class SdmaQueue
        {
        public:
            SdmaQueue(uint32_t localNode, uint32_t engineId);
            ~SdmaQueue();

            SdmaQueue(const SdmaQueue&)            = delete;
            SdmaQueue& operator=(const SdmaQueue&) = delete;

            // Device pointer to this queue's handle (for single-queue device use).
            SdmaQueueDeviceHandle* deviceHandle() const { return deviceHandle_; }
            // Host-visible copy of the same handle (for host-side driving/packing).
            const SdmaQueueDeviceHandle& hostHandle() const { return hostHandle_; }

            // ---- Host-side driving (smoke / bring-up only) -----------------
            // The production producer is the GPU kernel (later task). These
            // helpers let the host enqueue a packet and drive the doorbell so a
            // queue can be exercised end-to-end without any device code.
            //
            // submitPacketHost: copy `bytes` of `pkt` into the ring at the
            // current write cursor, advance wptr, and ring the doorbell. `bytes`
            // must be a multiple of 4 and fit in the ring without wrapping (the
            // smoke path enqueues a handful of small packets from an empty ring;
            // wrap/NOP-pad handling is the device producer's job). Returns the
            // submitted (post-increment) wptr byte count.
            uint64_t submitPacketHost(const void* pkt, size_t bytes);

            // Spin until the engine's read pointer catches up to the last
            // submitted write pointer (queue fully drained). Returns false on
            // timeout.
            bool waitIdleHost(uint64_t timeoutSpins = (1ull << 34));

        private:
            // Best-effort release of every owned resource, shared by the
            // destructor and the constructor's failure path (a throw mid-ctor
            // leaves the object partially constructed, so ~SdmaQueue never runs).
            void teardown() noexcept;

            // Impl holds the KFD queue resource (HsaQueueResource) and the ring
            // pointer. It is defined only in SdmaQueue.cpp so the hsakmt type
            // never appears in this header; unique_ptr keeps the class movable-
            // free and its destruction ordered after the members below.
            struct Impl;
            std::unique_ptr<Impl> impl_;

            uint64_t*              cachedWptr_    = nullptr; // uncached device mem
            uint64_t*              committedWptr_ = nullptr; // uncached device mem
            SdmaQueueDeviceHandle* deviceHandle_  = nullptr; // device copy
            SdmaQueueDeviceHandle  hostHandle_{};            // host copy
            uint64_t               hostWptr_ = 0; // host-side write cursor
        };

        // -------------------------------------------------------------------
        // A set of W queues for one local device -- one queue per peer (each
        // peer gets its own queue; we deliberately do NOT fan a peer out over
        // multiple queues, measured worse). The W device handles are packed
        // contiguously into a single device array so a kernel can index them by
        // destination rank.
        // -------------------------------------------------------------------
        class SdmaQueueSet
        {
        public:
            // localNode is this device's KFD node; targetNodes[j] is peer j's
            // KFD node (use localNode for a loopback/self entry). One queue is
            // created per target, with its engine chosen by sdmaSelectEngine().
            SdmaQueueSet(uint32_t localNode, const std::vector<uint32_t>& targetNodes);
            ~SdmaQueueSet();

            SdmaQueueSet(const SdmaQueueSet&)            = delete;
            SdmaQueueSet& operator=(const SdmaQueueSet&) = delete;

            size_t size() const { return queues_.size(); }
            SdmaQueue& queue(size_t i) { return *queues_[i]; }

            // Device pointer to the W-element SdmaQueueDeviceHandle array.
            SdmaQueueDeviceHandle* deviceHandles() const { return dHandles_; }

        private:
            std::vector<std::unique_ptr<SdmaQueue>> queues_;
            SdmaQueueDeviceHandle*                  dHandles_ = nullptr;
        };

        // ---- Topology helpers ---------------------------------------------
        // KFD topology node id for a HIP device ordinal (via the HSA agent's
        // NODE info, mirroring MORI). Initializes HSA + KFD on first call.
        uint32_t sdmaNodeIdForDevice(int hipDeviceId);

        // SDMA engine id to use for the srcNode->dstNode link. Prefers the
        // first engine in KFD's RecSdmaEngIdMask for that io-link; for a
        // loopback (srcNode==dstNode, no io-link) returns a general engine (0),
        // matching MORI's loopback handling.
        uint32_t sdmaSelectEngine(uint32_t srcNode, uint32_t dstNode);

    } // namespace Client
} // namespace TensileLite
