// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Adaptive sm_count_target: estimates the CUs available to a stream from the
// occupancy probe that StreamK hybrid kernels write into a host-mapped slot
// array, and turns it into an sm_count_target hint. Holds the environment
// config, the per-stream Estimator and the pure helpers around them. HIP-free,
// so the per-launch logic is unit-tested GPU-free and clients parse the
// environment exactly as the library does.
//
// Probe protocol: a probe launch with epoch e (per-stream, never 0) runs at
// least N_CU workgroups, padded with probe-only workgroups if its grid is
// smaller, and stores e into slots[XCC<<6 | SE<<4 | CU] once per workgroup.
// Other launches do not mark. A completed probe e reads as the number of
// slots holding e or a later epoch.

#include "rocblaslt_secure_env.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rocblaslt
{
    namespace adaptive_sm
    {
        // Largest probe window.
        constexpr uint32_t c_maxProbeWindow = 16;

        // Environment variables, read once per process.
        constexpr const char* kEnvEnable      = "HIPBLASLT_ADAPTIVE_SM_COUNT";
        constexpr const char* kEnvForce       = "HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE";
        constexpr const char* kEnvForceProbe  = "HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE_PROBE";
        constexpr const char* kEnvLog         = "HIPBLASLT_ADAPTIVE_SM_COUNT_LOG";
        constexpr const char* kEnvProbeEvery  = "HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_EVERY";
        constexpr const char* kEnvTinyMflops  = "HIPBLASLT_ADAPTIVE_SM_COUNT_TINY_MFLOPS";
        constexpr const char* kEnvProbeWindow = "HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_WINDOW";

        // Flag `name` with value `raw`: 1/on/true or 0/off/false. Unset or
        // empty gives `def`; anything else gives `def` with a warning.
        inline bool
            envFlag(const char* name, const char* raw, bool def, std::vector<std::string>* warnings)
        {
            if(!raw)
                return def;
            const std::string v = normalizeEnv(raw);
            if(v.empty())
                return def;
            if(v == "1" || v == "on" || v == "true")
                return true;
            if(v == "0" || v == "off" || v == "false")
                return false;
            if(warnings)
                warnings->push_back(std::string(name) + "=" + raw + " is not 0/1; using "
                                    + (def ? "1" : "0"));
            return def;
        }

        // Unsigned `name` with value `raw`, in [lo, hi]. Unset or empty gives
        // `def`; anything else gives `def` with a warning.
        inline uint32_t envU32(const char*               name,
                               const char*               raw,
                               uint32_t                  def,
                               uint32_t                  lo,
                               uint32_t                  hi,
                               std::vector<std::string>* warnings)
        {
            if(!raw)
                return def;
            const std::string v = normalizeEnv(raw);
            if(v.empty())
                return def;
            char* end                  = nullptr;
            errno                      = 0;
            const unsigned long long x = std::strtoull(v.c_str(), &end, 10);
            if(v[0] == '-' || *end || errno == ERANGE || x < lo || x > hi)
            {
                if(warnings)
                    warnings->push_back(std::string(name) + "=" + raw + " is not in ["
                                        + std::to_string(lo) + ", " + std::to_string(hi)
                                        + "]; using " + std::to_string(def));
                return def;
            }
            return static_cast<uint32_t>(x);
        }

        struct Config
        {
            bool     enabled     = false; // HIPBLASLT_ADAPTIVE_SM_COUNT=1, or _FORCE set
            uint32_t log         = 0; // _LOG: 1 config, hint changes; 2 + readings
            uint32_t force       = 0; // _FORCE: fixed hint for query and launch
            bool     forceProbe  = false; // _FORCE_PROBE: probe under _FORCE, readings discarded
            uint32_t probeEvery  = 8; // _PROBE_EVERY: launches between probes
            uint32_t tinyMflops  = 256; // _TINY_MFLOPS: 2*M*N*K*batch below this never probes
            uint32_t probeWindow = 3; // _PROBE_WINDOW: latest readings that decide the hint

            // Reads the variables through `get` (getenv's signature). An unset
            // or empty variable keeps its default; an invalid value keeps it
            // too and appends one line to `warnings`.
            static Config fromEnv(const char* (*get)(const char*),
                                  std::vector<std::string>* warnings)
            {
                const uint32_t u32Max = 0xffffffffu;
                Config         c;
                c.force      = envU32(kEnvForce, get(kEnvForce), 0, 0, 0xffff, warnings);
                c.enabled    = envFlag(kEnvEnable, get(kEnvEnable), false, warnings) || c.force > 0;
                c.forceProbe = envFlag(kEnvForceProbe, get(kEnvForceProbe), false, warnings);
                c.log        = envU32(kEnvLog, get(kEnvLog), 0, 0, 2, warnings);
                c.probeEvery = envU32(
                    kEnvProbeEvery, get(kEnvProbeEvery), c.probeEvery, 1, u32Max, warnings);
                c.tinyMflops = envU32(
                    kEnvTinyMflops, get(kEnvTinyMflops), c.tinyMflops, 0, u32Max, warnings);
                c.probeWindow = envU32(kEnvProbeWindow,
                                       get(kEnvProbeWindow),
                                       c.probeWindow,
                                       1,
                                       c_maxProbeWindow,
                                       warnings);
                if(c.forceProbe && !c.force)
                {
                    if(warnings)
                        warnings->push_back(std::string(kEnvForceProbe) + "=" + get(kEnvForceProbe)
                                            + " needs " + kEnvForce + "; ignored");
                    c.forceProbe = false;
                }
                return c;
            }

            static Config fromEnv(std::vector<std::string>* warnings = nullptr)
            {
                return fromEnv([](const char* name) -> const char* { return std::getenv(name); },
                               warnings);
            }

            // Whether timing under this config can change the timed solution.
            bool hintVaries() const
            {
                return enabled && !force;
            }

            // Whether launches update a stream's state: always, except under
            // force without forceProbe.
            bool tracksLaunches() const
            {
                return !force || forceProbe;
            }

            // One line of the resolved settings.
            std::string describe() const
            {
                return "config enabled=" + std::to_string(enabled) + " force="
                       + std::to_string(force) + " force_probe=" + std::to_string(forceProbe)
                       + " probe_every=" + std::to_string(probeEvery) + " tiny_mflops="
                       + std::to_string(tinyMflops) + " probe_window=" + std::to_string(probeWindow)
                       + " log=" + std::to_string(log);
            }
        };

        // How many probe slots a stream has, sized for gfx950 and its hardware id register.
        constexpr uint32_t c_probeSlots = 512;
        // Misses in a row before the hint drops to 0.
        constexpr uint32_t c_resetMisses = 2;

        // The adaptive hint applies in StreamK AUTO or ON mode when neither the
        // desc nor the handle sets an explicit sm_count_target.
        inline bool applies(bool enabled, bool modeAutoOrOn, int32_t descHint, int32_t handleHint)
        {
            return enabled && modeAutoOrOn && descHint <= 0 && handleHint <= 0;
        }

        // Probe epochs between zeroings of a stream's slots, so a mark stays
        // well under 2^31 epochs old and countFrom() never takes it for a new one.
        constexpr uint32_t c_epochClearInterval = uint32_t(1) << 30;

        inline void clearSlots(uint32_t* slots)
        {
            std::memset(slots, 0, c_probeSlots * sizeof(uint32_t));
        }

        // One stream's probe epochs. After a zeroing, every mark a probe since
        // then wrote is after `cleared`; any other mark is older.
        struct ProbeEpoch
        {
            uint32_t epoch   = 0; // last probe epoch, never 0 once taken
            uint32_t cleared = 0; // last epoch before the latest zeroing

            // Returns the epoch of a new probe, zeroing `slots` first every
            // c_epochClearInterval epochs. A new probe implies none unread; a
            // dropped probe's late mark is still an older epoch.
            uint32_t advance(uint32_t* slots)
            {
                if(epoch - cleared >= c_epochClearInterval)
                {
                    clearSlots(slots);
                    cleared = epoch;
                }
                epoch = epoch + 1 ? epoch + 1 : 1;
                return epoch;
            }
        };

        // A launch below cfg.tinyMflops (2*M*N*K*batch) neither probes nor
        // pads.
        inline bool tinyProblem(int64_t m, int64_t n, int64_t k, int64_t batch, const Config& cfg)
        {
            const double flops
                = 2.0 * double(m) * double(n) * double(k) * double(std::max<int64_t>(batch, 1));
            return flops < 1e6 * double(cfg.tinyMflops);
        }

        // State of a recorded probe's completion event.
        enum class ProbeState
        {
            Pending,
            Done,
            Failed, // query error: dropped without a reading, due again
        };

        // CUs marked by probe `epoch` or a later one, at most nCu.
        inline uint32_t countFrom(const volatile uint32_t* slots, uint32_t epoch, uint32_t nCu)
        {
            uint32_t count = 0;
            for(uint32_t i = 0; i < c_probeSlots; ++i)
            {
                const uint32_t v = slots[i];
                count += v != 0 && static_cast<int32_t>(v - epoch) >= 0;
            }
            return std::min(count, nCu);
        }

        // The last n of up to c_maxProbeWindow readings, the oldest
        // overwritten.
        struct ReadingWindow
        {
            uint32_t values[c_maxProbeWindow] = {};
            uint32_t size                     = 0;
            uint32_t next                     = 0; // slot the next reading takes

            // Adds `v` to a window of `n`; returns whether it is full.
            bool add(uint32_t v, uint32_t n)
            {
                values[next] = v;
                next         = (next + 1) % n;
                size         = std::min(size + 1, n);
                return size == n;
            }

            uint32_t lo() const
            {
                return *std::min_element(values, values + size);
            }

            uint32_t hi() const
            {
                return *std::max_element(values, values + size);
            }
        };

        // Hint for a window of readings from `lo` to `hi` under hint `hint`
        // (0: none, standing for nCu); returns `hint` for no change. All at
        // nCu or more: 0. All below or all above the hint: the smallest
        // reading. Otherwise no change. The Estimator drops a hint on a single
        // reading at nCu, without the window.
        inline uint32_t candidate(uint32_t lo, uint32_t hi, uint32_t hint, uint32_t nCu)
        {
            const uint32_t from = hint ? hint : nCu;
            if(lo >= nCu)
                return 0;
            if(hi < from || lo > from)
                return lo;
            return hint;
        }

        // One stream's adaptive state: probe ticks and epochs, the probe
        // awaiting completion, the readings and the hint. Not thread-safe; the
        // caller serialises it. Every tracked launch calls beforeLaunch(), and
        // afterProbe() when given an epoch.
        //
        // A probe is due from the start, and falls due again every
        // cfg.probeEvery-th launch (a tick); the first launch that can probe
        // takes it. A probe that did not run or whose event failed stays due.
        // A tick that finds the probe still due is a miss unless a pending
        // probe defers it; c_resetMisses misses in a row drop the hint to 0
        // and clear the window, and a recorded probe restarts the count.
        // At most one probe is pending: a due probe waits until it is read, so
        // a later probe never overwrites the marks of an unread one. Each
        // completed probe yields at most one reading; the last
        // cfg.probeWindow of them decide the hint (see candidate()), except
        // that a reading of nCu under a hint drops it to 0 at once and
        // restarts the window from that reading.
        class Estimator
        {
        public:
            Estimator() = default;

            // `slots`: the stream's c_probeSlots probe slots; `cfg` outlives
            // the estimator.
            Estimator(const Config& cfg, uint32_t* slots, uint32_t nCu)
                : m_cfg(&cfg)
                , m_slots(slots)
                , m_nCu(nCu)
            {
            }

            // A probe awaits its reading: the caller passes its state to the
            // next beforeLaunch().
            bool pending() const
            {
                return m_pending;
            }

            // Before a tracked launch: counts it, and at a tick counts a miss
            // if the probe is still due; reads the pending probe if
            // `pendingState` says it completed; then plans the probe. A due
            // probe waits while one is pending, else a launch that `canProbe`
            // takes it. Returns the epoch it probes with, or 0 for none.
            uint32_t beforeLaunch(bool canProbe, ProbeState pendingState)
            {
                if(++m_sinceTick >= m_cfg->probeEvery)
                {
                    m_sinceTick = 0;
                    if(m_due && !m_pending)
                        miss();
                    m_due = true;
                }
                if(m_pending && pendingState != ProbeState::Pending)
                {
                    m_pending = false;
                    if(pendingState == ProbeState::Done)
                        takeReading(countFrom(m_slots, m_epoch.epoch, m_nCu));
                    else
                        m_due = true; // dropped without a reading
                }
                if(!m_due || m_pending || !canProbe)
                    return 0;
                // Assumed recorded until afterProbe() says otherwise.
                m_pending = true;
                m_due     = false;
                return m_epoch.advance(m_slots);
            }

            // After a launch beforeLaunch() gave an epoch: whether it probed
            // and its completion event was recorded. If not, the probe stays
            // due.
            void afterProbe(bool recorded)
            {
                if(recorded)
                {
                    m_missed = 0;
                    return;
                }
                m_pending = false;
                m_due     = true;
            }

            uint32_t hint() const
            {
                return m_hint;
            }

            // Last probe epoch.
            uint32_t epoch() const
            {
                return m_epoch.epoch;
            }

            // The readings deciding the hint, in no particular order.
            const ReadingWindow& window() const
            {
                return m_window;
            }

            uint32_t missed() const
            {
                return m_missed;
            }

            // Count of the last completed probe, a reading or not.
            uint32_t lastCount() const
            {
                return m_lastCount;
            }

            // Completed probes read so far, readings or not.
            uint32_t probesRead() const
            {
                return m_probesRead;
            }

            uint32_t sinceTick() const
            {
                return m_sinceTick;
            }

            bool due() const
            {
                return m_due;
            }

        private:
            // A completed probe's count. A count of 0 is not a reading.
            // cfg.probeWindow is in [1, c_maxProbeWindow].
            void takeReading(uint32_t count)
            {
                m_lastCount = count;
                ++m_probesRead;
                if(count == 0)
                    return;
                if(m_hint && count >= m_nCu)
                {
                    m_window = {};
                    m_hint   = 0;
                }
                if(m_window.add(count, m_cfg->probeWindow))
                    m_hint = candidate(m_window.lo(), m_window.hi(), m_hint, m_nCu);
            }

            void miss()
            {
                if(++m_missed < c_resetMisses)
                    return;
                m_window = {};
                m_hint   = 0;
            }

            const Config* m_cfg   = nullptr;
            uint32_t*     m_slots = nullptr;
            uint32_t      m_nCu   = 0;
            ProbeEpoch    m_epoch;
            ReadingWindow m_window;
            uint32_t      m_sinceTick  = 0;
            uint32_t      m_missed     = 0;
            uint32_t      m_lastCount  = 0;
            uint32_t      m_probesRead = 0;
            uint32_t      m_hint       = 0;
            bool          m_due        = true; // the first probe is due at once
            bool          m_pending    = false; // probe m_epoch.epoch is unread
        };
    } // namespace adaptive_sm
} // namespace rocblaslt
