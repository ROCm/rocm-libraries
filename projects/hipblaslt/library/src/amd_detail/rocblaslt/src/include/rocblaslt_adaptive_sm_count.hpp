// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Adaptive sm_count_target: estimates the CUs available to a stream from the
// occupancy probe that SK5 kernels write into a host-mapped slot array, and
// turns it into an sm_count_target hint. HIP-free, so the per-launch logic
// is unit-tested GPU-free and clients parse the environment exactly as the
// library does.
//
// Probe protocol: a probe launch with epoch e (per-stream, never 0) runs at
// least N_CU workgroups, padded with probe-only workgroups if its grid is
// smaller, and stores e into slots[XCC<<6 | SE<<4 | CU] once per workgroup.
// Other launches do not mark. A completed probe e reads as the number of
// slots holding e or a later epoch.

#include "rocblaslt_secure_env.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
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

        struct Config
        {
            bool     enabled     = false; // HIPBLASLT_ADAPTIVE_SM_COUNT=1, or _FORCE set
            uint32_t log         = 0; // _LOG: 1 config, hint changes, re-selections; 2 + readings
            uint32_t force       = 0; // _FORCE: fixed hint for query and launch
            bool     forceProbe  = false; // _FORCE_PROBE: probe under _FORCE, readings discarded
            uint32_t probeEvery  = 8; // _PROBE_EVERY: launches between probes
            uint32_t tinyMflops  = 4096; // _TINY_MFLOPS: 2*M*N*K*batch below this never probes
            uint32_t probeWindow = 2; // _PROBE_WINDOW: latest readings that decide the hint

            // Reads the variables through `get` (getenv's signature). An unset
            // or empty variable keeps its default; an invalid value keeps it
            // too and appends one line to `warnings`.
            template <typename Getenv>
            static Config fromEnv(Getenv get, std::vector<std::string>* warnings)
            {
                auto warn = [&](const char* name, const char* raw, const std::string& why) {
                    if(warnings)
                        warnings->push_back(std::string(name) + "=" + raw + " " + why);
                };
                auto flag = [&](const char* name, bool def) {
                    const char* raw = get(name);
                    if(!raw)
                        return def;
                    const std::string v = rocblaslt::normalizeEnv(raw);
                    if(v.empty())
                        return def;
                    if(v == "1" || v == "on" || v == "true")
                        return true;
                    if(v == "0" || v == "off" || v == "false")
                        return false;
                    warn(name, raw, std::string("is not 0/1; using ") + (def ? "1" : "0"));
                    return def;
                };
                auto u32 = [&](const char* name, uint32_t def, uint32_t lo, uint32_t hi) {
                    const char* raw = get(name);
                    if(!raw)
                        return def;
                    const std::string v = rocblaslt::normalizeEnv(raw);
                    if(v.empty())
                        return def;
                    char*               end = nullptr;
                    const unsigned long x   = std::strtoul(v.c_str(), &end, 10);
                    if(v[0] == '-' || *end || x < lo || x > hi)
                    {
                        warn(name,
                             raw,
                             "is not in [" + std::to_string(lo) + ", " + std::to_string(hi)
                                 + "]; using " + std::to_string(def));
                        return def;
                    }
                    return static_cast<uint32_t>(x);
                };

                Config c;
                c.force       = u32(kEnvForce, 0, 0, 0xffff);
                c.enabled     = flag(kEnvEnable, false) || c.force > 0;
                c.forceProbe  = flag(kEnvForceProbe, false);
                c.log         = u32(kEnvLog, 0, 0, 2);
                c.probeEvery  = u32(kEnvProbeEvery, c.probeEvery, 1, 0xffffffffu);
                c.tinyMflops  = u32(kEnvTinyMflops, c.tinyMflops, 0, 0xffffffffu);
                c.probeWindow = u32(kEnvProbeWindow, c.probeWindow, 1, c_maxProbeWindow);
                if(c.forceProbe && !c.force)
                {
                    warn(kEnvForceProbe,
                         get(kEnvForceProbe),
                         std::string("needs ") + kEnvForce + "; ignored");
                    c.forceProbe = false;
                }
                return c;
            }

            static Config fromEnv(std::vector<std::string>* warnings = nullptr)
            {
                return fromEnv([](const char* name) { return std::getenv(name); }, warnings);
            }

            // Whether timing under this config can change the timed solution.
            bool hintVaries() const
            {
                return enabled && !force;
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

        constexpr uint32_t c_probeSlots = 512;
        // Missed probe opportunities in a row before the hint drops to 0.
        constexpr uint32_t c_resetMisses = 2;
        // Failed probe event creates or records in a row that count as one
        // miss.
        constexpr uint32_t c_eventFailuresPerMiss = 4;
        // Probes per stream awaiting completion. A due probe waits while the
        // ring is full. With one, a probe is read before the next one runs,
        // so a later probe never overwrites the marks of an unread one.
        constexpr uint32_t c_probeRing = 1;
        // Candidates examined by launch-time re-selection.
        constexpr int c_reselectTopK = 8;
        // Least workspace reported for the top heuristic result in adaptive
        // mode, capped by the caller's limit: origami's static-grid
        // partial-tile limit, enough for SK5 re-grids and re-selections under
        // any hint.
        constexpr size_t c_adaptiveWorkspace = size_t(128) << 20;

        // The adaptive hint applies in StreamK AUTO or ON mode when neither the
        // desc nor the handle sets an explicit sm_count_target. Under ON the
        // dynamic grid ignores it; non-dynamic picks and origami WGM still
        // use it.
        inline bool applies(bool enabled, bool modeAutoOrOn, int32_t descHint, int32_t handleHint)
        {
            return enabled && modeAutoOrOn && descHint <= 0 && handleHint <= 0;
        }

        // Whether a launch updates its stream's estimator. A captured launch
        // replays with frozen kernargs, so it leaves the state alone; under
        // cfg.force the state is kept only to probe under cfg.forceProbe.
        inline bool tracksLaunch(const Config& cfg, bool capturing)
        {
            return !capturing && (!cfg.force || cfg.forceProbe);
        }

        // Hint a launch takes before its stream's estimator is consulted: the
        // forced one, none on a capturing stream, else the `published` one.
        inline uint32_t launchHint(const Config& cfg, bool capturing, uint32_t published)
        {
            if(cfg.force)
                return cfg.force;
            return capturing ? 0 : published;
        }

        // Probe epoch after `epoch`, skipping 0.
        inline uint32_t nextEpoch(uint32_t epoch)
        {
            return epoch + 1 ? epoch + 1 : 1;
        }

        // Probe epochs between zeroings of a stream's slots, so a mark stays
        // well under 2^31 epochs old and countFrom() never takes it for a new one.
        constexpr uint32_t c_epochClearInterval = uint32_t(1) << 30;

        // One stream's probe epochs. After a zeroing, every mark a probe since
        // then wrote is after `cleared`; any other mark is older.
        struct ProbeEpoch
        {
            uint32_t epoch   = 0; // last probe epoch
            uint32_t cleared = 0; // last epoch before the latest zeroing

            // Takes the epoch of a new probe with `outstanding` probes unread.
            // Returns true when the caller must zero the slots before enqueueing
            // it: due every c_epochClearInterval epochs, held back while a probe
            // is unread. A dropped probe's late mark is still an older epoch.
            bool advance(uint32_t outstanding)
            {
                const bool clear = outstanding == 0 && epoch - cleared >= c_epochClearInterval;
                if(clear)
                    cleared = epoch;
                epoch = nextEpoch(epoch);
                return clear;
            }
        };

        inline void clearSlots(uint32_t* slots)
        {
            std::memset(slots, 0, c_probeSlots * sizeof(uint32_t));
        }

        // A launch below cfg.tinyMflops (2*M*N*K*batch) neither probes nor
        // pads.
        inline bool tinyProblem(int64_t m, int64_t n, int64_t k, int64_t batch, const Config& cfg)
        {
            const double flops
                = 2.0 * double(m) * double(n) * double(k) * double(std::max<int64_t>(batch, 1));
            return flops < 1e6 * double(cfg.tinyMflops);
        }

        // Whether a launch given a probe address probed: its kernel supports
        // the probe (`kernelProbes`, never for grouped GEMM) and was dispatched
        // with at least minGrid workgroups. TensileLite pads a 1-D grid below
        // minGrid to it and drops the address for any other grid below it.
        inline bool launchProbed(bool     hasAddr,
                                 bool     kernelProbes,
                                 bool     groupedGemm,
                                 uint32_t gridX,
                                 uint32_t gridY,
                                 uint32_t gridZ,
                                 uint32_t minGrid)
        {
            return hasAddr && kernelProbes && !groupedGemm
                   && uint64_t(gridX) * gridY * gridZ >= minGrid;
        }

        // State of a recorded probe's completion event.
        enum class ProbeState
        {
            Pending,
            Done,
            Failed, // query error: dropped without a reading
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

        // Hint for the probe window `counts` (`n` readings) under hint `hint`
        // (0: none, standing for nCu); returns `hint` for no change. All at
        // nCu or more: 0. All below or all above the hint: the smallest
        // reading. Otherwise no change.
        inline uint32_t candidate(const uint32_t* counts, uint32_t n, uint32_t hint, uint32_t nCu)
        {
            uint32_t lo = counts[0], hi = counts[0];
            for(uint32_t i = 1; i < n; ++i)
            {
                lo = std::min(lo, counts[i]);
                hi = std::max(hi, counts[i]);
            }
            const uint32_t from = hint ? hint : nCu;
            if(lo >= nCu)
                return 0;
            if(hi < from || lo > from)
                return lo;
            return hint;
        }

        // Heuristic-query tag in rocblaslt_matmul_algo::data[4..7] (data[0..3] is
        // the solution index): data[4] bit 0 = eligible for launch-time
        // re-selection, data[5] = tag version, data[6..7] = hint at query.
        constexpr uint8_t c_algoTagVersion = 1;

        inline void tagAlgo(uint8_t* data, uint32_t queryHint)
        {
            const uint16_t h = static_cast<uint16_t>(std::min<uint32_t>(queryHint, 0xffff));
            data[4] |= 1;
            data[5] = c_algoTagVersion;
            std::memcpy(data + 6, &h, sizeof(h));
        }

        inline bool algoTag(const uint8_t* data, uint32_t* queryHint)
        {
            if(!(data[4] & 1) || data[5] != c_algoTagVersion)
                return false;
            uint16_t h;
            std::memcpy(&h, data + 6, sizeof(h));
            *queryHint = h;
            return true;
        }

        // Launch-time re-selection over the top-K solutions ranked under the
        // launch hint, given whether each fits the caller's workspace.
        struct ReselectChoice
        {
            int  index;
            bool keepQueryHint; // launch the original under the query hint
        };

        inline ReselectChoice pickReselection(int                      original,
                                              const std::vector<int>&  candidates,
                                              const std::vector<bool>& candidateFits,
                                              bool                     originalFits)
        {
            for(size_t i = 0; i < candidates.size(); ++i)
                if(candidateFits[i])
                    return {candidates[i], false};
            return {original, !originalFits};
        }

        // Small memo with round-robin eviction once it holds N entries.
        template <typename Entry, size_t N>
        class SmallMemo
        {
        public:
            template <typename Match>
            const Entry* find(Match match) const
            {
                for(auto const& e : m_entries)
                    if(match(e))
                        return &e;
                return nullptr;
            }

            const Entry* insert(Entry e)
            {
                if(m_entries.size() < N)
                {
                    m_entries.push_back(std::move(e));
                    return &m_entries.back();
                }
                Entry& slot = m_entries[m_next++ % N];
                slot        = std::move(e);
                return &slot;
            }

            size_t size() const
            {
                return m_entries.size();
            }

            void clear()
            {
                m_entries.clear();
                m_next = 0;
            }

        private:
            std::vector<Entry> m_entries;
            size_t             m_next = 0;
        };

        // Result of Estimator::startLaunch().
        struct LaunchPlan
        {
            bool     cadence  = false; // a cadence point
            bool     deferred = false; // due probe held back by a full ring
            bool     probe    = false; // this launch takes the due probe
            bool     read     = false; // a completed probe was read
            uint32_t oldHint  = 0; // hint() on entry
        };

        // Per-stream estimator: probe cadence, probes awaiting completion and
        // the hint. Not thread-safe; the caller serialises it. A tracked
        // launch (tracksLaunch()) calls startLaunch() before and finishLaunch()
        // after it.
        //
        // Every cfg.probeEvery launches a probe falls due; the first launch
        // that can probe takes it. A cadence point passing with the probe
        // still due is a miss unless the probe was deferred (full ring or
        // failed event); c_eventFailuresPerMiss failed events in a row are a
        // miss. c_resetMisses misses in a row drop the hint to 0. Each
        // completed probe yields at most one reading; the last
        // cfg.probeWindow of them decide the hint (see candidate()). Under
        // cfg.force readings are discarded.
        class Estimator
        {
        public:
            // First phase of a tracked launch: counts it; at a cadence point
            // binds the stream, `streamId(&id)` returning false when the id is
            // unknown; reads the newest completed probe (see poll()); then
            // plans the probe. A due probe waits while the ring is full
            // (deferred), else a launch that `canProbe` takes it.
            template <typename StreamId, typename State>
            LaunchPlan startLaunch(const Config&            cfg,
                                   bool                     canProbe,
                                   StreamId                 streamId,
                                   const volatile uint32_t* slots,
                                   uint32_t                 nCu,
                                   State                    state)
            {
                LaunchPlan plan;
                plan.oldHint   = m_hint;
                const bool due = beginLaunch(cfg, &plan.cadence);
                // A reused stream slot can carry a different stream, and with
                // it a different CU mask.
                unsigned long long id = 0;
                if(plan.cadence && streamId(&id))
                    bindStream(id);
                const uint32_t read = m_probesRead;
                poll(slots, nCu, cfg, state);
                plan.read     = m_probesRead != read;
                plan.deferred = due && ringFull();
                plan.probe    = due && canProbe && !plan.deferred;
                return plan;
            }

            // Second phase of a tracked launch, `plan` from startLaunch().
            // `recordable`: the launch probed with `epoch` and its completion
            // event may be recorded; `record(i)` records it for ring position
            // i and returns false on failure (see eventFailed()). A recordable
            // probe whose record fails stays due, deferred; the full-ring check
            // is defensive, as the caller planned the probe with a free slot.
            // Returns true when hint() changed.
            template <typename Record>
            bool
                finishLaunch(const LaunchPlan& plan, bool recordable, uint32_t epoch, Record record)
            {
                bool changed = false;
                if(recordable && !ringFull())
                {
                    if(record(nextRingSlot()))
                        addProbe(epoch);
                    else
                        changed = eventFailed();
                }
                const bool missed = endLaunch(plan.cadence, plan.deferred || recordable);
                return missed || changed;
            }

            uint32_t hint() const
            {
                return m_hint;
            }

            // Newest-first readings behind the current state.
            const uint32_t* readings() const
            {
                return m_readings;
            }

            uint32_t readingCount() const
            {
                return m_readingCount;
            }

            // Readings taken so far.
            uint32_t readingsTaken() const
            {
                return m_readingsTaken;
            }

            uint32_t missed() const
            {
                return m_missed;
            }

            uint32_t outstanding() const
            {
                return m_ringSize;
            }

            // Count of the last completed probe, discarded or not.
            uint32_t lastCount() const
            {
                return m_lastCount;
            }

            // Completed probes read so far, discarded or not.
            uint32_t probesRead() const
            {
                return m_probesRead;
            }

            uint32_t sinceCadence() const
            {
                return m_sinceCadence;
            }

            bool due() const
            {
                return m_due;
            }

        private:
            // Counts a launch. Returns whether a probe is due; `*cadence`
            // tells whether this launch is a cadence point.
            bool beginLaunch(const Config& cfg, bool* cadence)
            {
                *cadence = ++m_sinceCadence >= cfg.probeEvery;
                if(*cadence)
                {
                    m_sinceCadence = 0;
                    m_due          = true;
                }
                return m_due;
            }

            // Ends a launch, `cadence` as from beginLaunch(); `deferred`: a
            // due probe was held back by a full ring or a failed record.
            // Returns true when hint() changed.
            bool endLaunch(bool cadence, bool deferred)
            {
                if(!cadence || !m_due || deferred)
                    return false;
                return miss();
            }

            // A due probe whose event could not be created or recorded.
            // Returns true when hint() changed.
            bool eventFailed()
            {
                if(++m_eventFailures < c_eventFailuresPerMiss)
                    return false;
                m_eventFailures = 0;
                return miss();
            }

            bool ringFull() const
            {
                return m_ringSize == c_probeRing;
            }

            // Ring position the next probe takes.
            uint32_t nextRingSlot() const
            {
                return (m_ringHead + m_ringSize) % c_probeRing;
            }

            // Records probe `epoch` at nextRingSlot(); needs !ringFull().
            void addProbe(uint32_t epoch)
            {
                m_ring[nextRingSlot()] = epoch;
                ++m_ringSize;
                m_due           = false;
                m_missed        = 0;
                m_eventFailures = 0;
            }

            // Takes one reading from the newest completed probe, `state(i)`
            // giving the ProbeState of the probe at ring position i, and
            // drops it and any older one. A failed oldest probe is dropped
            // alone, without a reading. Returns true when hint() changed.
            template <typename State>
            bool poll(const volatile uint32_t* slots, uint32_t nCu, const Config& cfg, State state)
            {
                if(m_ringSize == 0)
                    return false;
                const ProbeState head = state(m_ringHead);
                if(head == ProbeState::Failed)
                {
                    m_ringHead = (m_ringHead + 1) % c_probeRing;
                    --m_ringSize;
                    return false;
                }
                if(head == ProbeState::Pending)
                    return false;
                // Probes complete in order: a done newer one supersedes it.
                uint32_t n = 1;
                while(n < m_ringSize && state((m_ringHead + n) % c_probeRing) == ProbeState::Done)
                    ++n;
                const uint32_t epoch = m_ring[(m_ringHead + n - 1) % c_probeRing];
                m_ringHead           = (m_ringHead + n) % c_probeRing;
                m_ringSize -= n;
                m_lastCount = countFrom(slots, epoch, nCu);
                ++m_probesRead;
                return !cfg.force && addReading(m_lastCount, nCu, cfg);
            }

            // Adds reading `count`. Returns true when hint() changed.
            bool addReading(uint32_t count, uint32_t nCu, const Config& cfg)
            {
                if(count == 0)
                    return false;
                ++m_readingsTaken;
                const uint32_t n = std::clamp(cfg.probeWindow, 1u, c_maxProbeWindow);
                for(uint32_t j = std::min(m_readingCount, n - 1); j > 0; --j)
                    m_readings[j] = m_readings[j - 1];
                m_readings[0]  = count;
                m_readingCount = std::min(m_readingCount + 1, n);
                if(m_readingCount < n)
                    return false;
                const uint32_t h = candidate(m_readings, n, m_hint, nCu);
                return std::exchange(m_hint, h) != h;
            }

            // Binds to stream `id`; a different stream than the last one
            // (a reused slot) resets the estimator. Returns true on reset.
            bool bindStream(unsigned long long id)
            {
                if(id == m_streamId)
                    return false;
                const bool rebound = m_streamId != 0;
                if(rebound)
                    *this = Estimator{};
                m_streamId = id;
                return rebound;
            }

            bool miss()
            {
                if(++m_missed < c_resetMisses)
                    return false;
                m_readingCount = 0;
                return std::exchange(m_hint, 0) != 0;
            }

            uint32_t           m_ring[c_probeRing]          = {}; // probe epochs
            uint32_t           m_ringHead                   = 0; // oldest
            uint32_t           m_ringSize                   = 0;
            uint32_t           m_readings[c_maxProbeWindow] = {}; // newest first
            uint32_t           m_readingCount               = 0;
            uint32_t           m_readingsTaken              = 0;
            uint32_t           m_sinceCadence               = 0;
            uint32_t           m_missed                     = 0;
            uint32_t           m_eventFailures              = 0;
            uint32_t           m_lastCount                  = 0;
            uint32_t           m_probesRead                 = 0;
            uint32_t           m_hint                       = 0;
            bool               m_due                        = false;
            unsigned long long m_streamId                   = 0;
        };

        // Snapshot of one stream's state, for tests.
        struct StreamStateForTest
        {
            uint32_t hint; // taken by the next uncaptured launch
            uint32_t epoch; // last probe epoch
            uint32_t outstanding;
            uint32_t missed;
            uint32_t sinceCadence;
            bool     due;
            uint32_t readingsTaken;
            uint32_t probesRead;
            uint32_t lastCount;

            bool operator==(const StreamStateForTest& o) const
            {
                return hint == o.hint && epoch == o.epoch && outstanding == o.outstanding
                       && missed == o.missed && sinceCadence == o.sinceCadence && due == o.due
                       && readingsTaken == o.readingsTaken && probesRead == o.probesRead
                       && lastCount == o.lastCount;
            }
        };
    } // namespace adaptive_sm
} // namespace rocblaslt
