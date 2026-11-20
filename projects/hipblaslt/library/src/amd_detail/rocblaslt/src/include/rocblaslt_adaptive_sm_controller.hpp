// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The HIP side of adaptive sm_count_target (see
// rocblaslt_adaptive_sm_count.hpp): a handle's probe slot arrays, one
// estimator per stream slot and the hints they publish.

#include "rocblaslt_adaptive_sm_count.hpp"

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

// Adaptive sm_count_target settings, read from the environment once.
const rocblaslt::adaptive_sm::Config& rocblaslt_adaptive_sm_config();

// Writes one adaptive sm_count_target log line to the library log stream.
void rocblaslt_adaptive_sm_log(const std::string& line);

namespace rocblaslt
{
    namespace adaptive_sm
    {
        class Launch;

        // Whether `stream` is capturing into a graph. A launch on a capturing
        // stream is not adaptive: its graph replays it with frozen kernargs.
        bool streamCapturing(hipStream_t stream);

        // Snapshot of one stream slot's state, for tests.
        struct StreamStateForTest
        {
            uint32_t published; // published hint, 0: none
            uint32_t epoch; // last probe epoch
            bool     pending; // a recorded probe is unread
            uint32_t missed;
            uint32_t sinceTick;
            bool     due;
            uint32_t probesRead;
            uint32_t lastCount;

            bool operator==(const StreamStateForTest& o) const
            {
                return published == o.published && epoch == o.epoch && pending == o.pending
                       && missed == o.missed && sinceTick == o.sinceTick && due == o.due
                       && probesRead == o.probesRead && lastCount == o.lastCount;
            }
        };

        // A handle's adaptive state. Each stream slot has a probe slot array
        // in host-mapped memory, written by the GPU with plain stores and read
        // by the host without a sync, plus an estimator, a completion event
        // and the hint it published. HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE is
        // applied here only.
        class Controller
        {
        public:
            // For `streams` stream slots on the current device, which has `nCu`
            // CUs. Null when adaptive mode is off or setup fails.
            static std::unique_ptr<Controller> create(size_t streams, uint32_t nCu);

            ~Controller();

            // First phase of a C API launch on `stream`, in stream slot `slot`
            // (-1: none): its hint, and the probe target when it should probe.
            // A tiny launch never probes; a launch finding the slot's lock held
            // leaves the slot's state alone. Not for a capturing stream.
            Launch begin(hipStream_t stream, int slot, bool tiny);

            // Hint for the heuristic query, which has no stream: the latest
            // published on any stream slot.
            uint32_t queryHint() const;

            StreamStateForTest stateForTest(int slot);

            // Whether another thread holds `slot`'s lock.
            bool lockedForTest(int slot);

        private:
            friend class Launch;
            struct Stream;

            Controller(const Config& cfg, size_t streams, uint32_t nCu);

            // Publishes `st`'s estimator hint if it changed. Called once per
            // tracked launch, by begin().
            void publish(Stream& st, hipStream_t stream);

            const Config&             m_cfg;
            const uint32_t            m_nCu;
            std::unique_ptr<Stream[]> m_streams;
            uint32_t*                 m_probeHost = nullptr;
            void*                     m_probeDev  = nullptr;
            std::atomic<uint32_t>     m_queryHint{0};
        };

        // One C API launch, from Controller::begin() to its destruction, which
        // records the completion event of a probe the launch ran. A probe
        // launch holds its stream slot's lock throughout, so probes enqueue
        // in epoch order; other launches on the slot meanwhile leave its
        // state alone.
        class Launch
        {
        public:
            Launch(Launch&&)            = default;
            Launch& operator=(Launch&&) = delete;
            ~Launch();

            uint32_t hint      = 0; // sm_count_target, 0: none
            void*    probeAddr = nullptr; // null: no probe
            uint32_t epoch     = 0;
            uint32_t minGrid   = 0; // workgroups a probe covers
            bool     probed    = false; // set by the launch: a kernel carried the probe

            bool holdsLockForTest() const
            {
                return m_lock.owns_lock();
            }

        private:
            friend class Controller;

            Launch() = default;

            // Second phase of a probe launch: records its probe, if it probed.
            void finish();

            hipStream_t                  m_stream = nullptr;
            Controller::Stream*          m_state  = nullptr;
            std::unique_lock<std::mutex> m_lock; // held exactly when probing
        };
    } // namespace adaptive_sm
} // namespace rocblaslt
