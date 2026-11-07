// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocblaslt_adaptive_sm_controller.hpp"
#include "check_numerics_matrix.hpp"
#include "utility.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

// Gated by HIPBLASLT_ADAPTIVE_SM_COUNT_LOG rather than the log mask, written
// where the library log goes, timestamped like CHECK_NUMERICS lines.
void rocblaslt_adaptive_sm_log(const std::string& line)
{
    std::lock_guard<std::mutex> lk(log_mutex);
    std::ostream*               sink = get_logger_os();
    if(!sink)
        sink = &std::cerr;
    *sink << hipblaslt_check_numerics_ts() << "[hipBLASLt ADAPTIVE_SM] " << line << std::endl;
}

const rocblaslt::adaptive_sm::Config& rocblaslt_adaptive_sm_config()
{
    static const rocblaslt::adaptive_sm::Config cfg = [] {
        std::vector<std::string> warnings;
        const auto               c = rocblaslt::adaptive_sm::Config::fromEnv(&warnings);
        for(auto const& w : warnings)
            rocblaslt_adaptive_sm_log(w);
        if(c.log)
            rocblaslt_adaptive_sm_log(c.describe());
        return c;
    }();
    return cfg;
}

namespace rocblaslt
{
    namespace adaptive_sm
    {
        bool streamCapturing(hipStream_t stream)
        {
            hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
            return hipStreamIsCapturing(stream, &cap) == hipSuccess
                   && cap != hipStreamCaptureStatusNone;
        }

        namespace
        {
            // Logs, under _LOG, a probe event call `what` that failed with
            // `err` on `stream`, and clears the error: the probe gives no
            // reading.
            void eventFailed(const char* what, hipError_t err, hipStream_t stream)
            {
                if(rocblaslt_adaptive_sm_config().log)
                {
                    std::ostringstream msg;
                    msg << "probe event " << what
                        << " failed stream=" << static_cast<const void*>(stream) << " ("
                        << hipGetErrorName(err) << ")";
                    rocblaslt_adaptive_sm_log(msg.str());
                }
                static_cast<void>(hipGetLastError());
            }
        } // namespace

        struct Controller::Stream
        {
            ~Stream()
            {
                if(event)
                    static_cast<void>(hipEventDestroy(event));
            }

            size_t                slot = 0; // index in the handle's stream slots
            std::atomic<uint32_t> hint{0}; // published, 0: none; read without the lock
            std::mutex            lock; // guards the members below
            hipEvent_t            event = nullptr; // the pending probe's completion
            Estimator             estimator;
        };

        Launch::~Launch()
        {
            if(m_lock.owns_lock())
                finish();
        }

        std::unique_ptr<Controller> Controller::create(size_t streams, uint32_t nCu)
        {
            const Config& cfg = rocblaslt_adaptive_sm_config();
            if(!cfg.enabled)
                return nullptr;

            // Warns once per process, for the first device it exceeds.
            static std::atomic<bool> warnedForce{false};
            if(cfg.force > nCu && !warnedForce.exchange(true))
                rocblaslt_adaptive_sm_log(std::string(kEnvForce) + "=" + std::to_string(cfg.force)
                                          + " exceeds the device's " + std::to_string(nCu)
                                          + " CUs");

            std::unique_ptr<Controller> c(new Controller(cfg, streams, nCu));
            const size_t                bytes = streams * c_probeSlots * sizeof(uint32_t);
            void*                       host  = nullptr;
            hipError_t                  err
                = hipHostMalloc(&host, bytes, hipHostMallocCoherent | hipHostMallocMapped);
            if(err == hipSuccess)
            {
                c->m_probeHost = static_cast<uint32_t*>(host);
                std::memset(host, 0, bytes);
                err = hipHostGetDevicePointer(&c->m_probeDev, host, 0);
            }
            if(err != hipSuccess)
            {
                rocblaslt_adaptive_sm_log("disabled: probe allocation failed (hipError="
                                          + std::to_string(static_cast<int>(err)) + ")");
                return nullptr;
            }
            for(size_t i = 0; i < streams; ++i)
            {
                Stream& st   = c->m_streams[i];
                st.slot      = i;
                st.estimator = Estimator(cfg, c->m_probeHost + i * c_probeSlots, nCu);
                err          = hipEventCreateWithFlags(&st.event, hipEventDisableTiming);
                if(err != hipSuccess)
                {
                    st.event = nullptr;
                    static_cast<void>(hipGetLastError());
                    rocblaslt_adaptive_sm_log("disabled: probe event creation failed (hipError="
                                              + std::to_string(static_cast<int>(err)) + ")");
                    return nullptr;
                }
            }
            return c;
        }

        Controller::Controller(const Config& cfg, size_t streams, uint32_t nCu)
            : m_cfg(cfg)
            , m_nCu(nCu)
            , m_streams(std::make_unique<Stream[]>(streams))
        {
        }

        Controller::~Controller()
        {
            if(m_probeHost)
                static_cast<void>(hipHostFree(m_probeHost));
        }

        Launch Controller::begin(hipStream_t stream, int slot, bool tiny)
        {
            Launch launch;
            launch.minGrid = m_nCu;
            Stream* st     = slot < 0 ? nullptr : &m_streams[slot];
            // Under force the state is kept only to probe under forceProbe.
            if(st && m_cfg.tracksLaunches())
            {
                // Another thread on this stream holds its state: launch untracked.
                std::unique_lock<std::mutex> lock(st->lock, std::try_to_lock);
                if(lock.owns_lock())
                {
                    Estimator& est   = st->estimator;
                    ProbeState state = ProbeState::Pending;
                    if(est.pending())
                    {
                        // hipErrorNotReady does not set the last error; other failures do.
                        const hipError_t q = hipEventQuery(st->event);
                        if(q == hipSuccess)
                            state = ProbeState::Done;
                        else if(q != hipErrorNotReady)
                        {
                            eventFailed("query", q, stream);
                            state = ProbeState::Failed;
                        }
                    }
                    const uint32_t read = est.probesRead();
                    launch.epoch        = est.beforeLaunch(!tiny, state);
                    if(m_cfg.log >= 2 && est.probesRead() != read)
                    {
                        std::ostringstream msg;
                        msg << "reading stream=" << static_cast<const void*>(stream)
                            << " slot=" << slot << " count=" << est.lastCount()
                            << (m_cfg.force ? " (discarded)" : "");
                        rocblaslt_adaptive_sm_log(msg.str());
                    }
                    publish(*st, stream);
                    if(launch.epoch)
                    {
                        launch.probeAddr = static_cast<char*>(m_probeDev)
                                           + slot * c_probeSlots * sizeof(uint32_t);
                        launch.m_stream = stream;
                        launch.m_state  = st;
                        launch.m_lock   = std::move(lock);
                    }
                }
            }
            if(m_cfg.force)
                launch.hint = m_cfg.force;
            else if(st)
                launch.hint = st->hint.load(std::memory_order_relaxed);
            return launch;
        }

        void Launch::finish()
        {
            bool recorded = false;
            // A capture begun since begin() would take the record into its
            // graph: the probe then stays due.
            if(probed && !streamCapturing(m_stream))
            {
                const hipError_t err = hipEventRecord(m_state->event, m_stream);
                recorded             = err == hipSuccess;
                if(!recorded)
                    eventFailed("record", err, m_stream);
            }
            m_state->estimator.afterProbe(recorded);
        }

        void Controller::publish(Stream& st, hipStream_t stream)
        {
            const uint32_t hint = st.estimator.hint();
            const uint32_t old  = st.hint.load(std::memory_order_relaxed);
            // Readings under force move no published hint.
            if(m_cfg.force || hint == old)
                return;
            st.hint.store(hint, std::memory_order_relaxed);
            m_queryHint.store(hint, std::memory_order_relaxed);
            if(!m_cfg.log)
                return;
            std::ostringstream msg;
            msg << "hint stream=" << static_cast<const void*>(stream) << " slot=" << st.slot
                << " hint=" << old << "->" << hint;
            const auto& w = st.estimator.window();
            if(w.size == 0)
                msg << " missed=" << st.estimator.missed();
            else
                msg << " readings=";
            for(uint32_t i = 0; i < w.size; ++i)
                msg << (i ? "," : "") << w.values[i];
            rocblaslt_adaptive_sm_log(msg.str());
        }

        uint32_t Controller::queryHint() const
        {
            return m_cfg.force ? m_cfg.force : m_queryHint.load(std::memory_order_relaxed);
        }

        StreamStateForTest Controller::stateForTest(int slot)
        {
            Stream&                     st  = m_streams[slot];
            const Estimator&            est = st.estimator;
            std::lock_guard<std::mutex> guard(st.lock);
            return {st.hint.load(std::memory_order_relaxed),
                    est.epoch(),
                    est.pending(),
                    est.missed(),
                    est.sinceTick(),
                    est.due(),
                    est.probesRead(),
                    est.lastCount()};
        }

        bool Controller::lockedForTest(int slot)
        {
            std::mutex& lock = m_streams[slot].lock;
            if(!lock.try_lock())
                return true;
            lock.unlock();
            return false;
        }
    } // namespace adaptive_sm
} // namespace rocblaslt
