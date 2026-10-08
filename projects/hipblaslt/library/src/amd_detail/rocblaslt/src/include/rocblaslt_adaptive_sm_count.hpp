// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Adaptive sm_count_target: estimates the CUs available to a stream from the
// occupancy probe that SK5 kernels write into a host-mapped slot array, and
// turns it into an sm_count_target hint. Dependency-free (no HIP) so the
// estimator can be unit-tested GPU-free.
//
// Probe protocol: launch e (per-stream epoch, never 0) with a grid of at least
// N_CU workgroups, or padded to N_CU with probe-only workgroups, stores e into
// slots[XCC<<8 | SE<<5 | SH<<4 | CU] once per workgroup. The host runs ahead
// of the GPU, so the window is anchored on the newest epoch found in the array
// (E_max), not on the host's epoch.

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace rocblaslt
{
    namespace adaptive_sm
    {
        constexpr uint32_t c_probeSlots  = 4096;
        constexpr uint32_t c_hintQuantum = 32;
        constexpr uint32_t c_maxWindow   = 16;
        // Downward moves larger than this, or to a hint at or below
        // c_lowHint, need Config::confirmDown samples.
        constexpr uint32_t c_bigDownMove = 64;
        constexpr uint32_t c_lowHint     = 64;
        // An epoch up to this far ahead of the reference is a later launch
        // from another thread (age 0); one further ahead was issued over 2^31
        // epochs ago (stale). A slot left untouched for 2^32 - 2^16 epochs or
        // more reads as current again; nothing bounds that, but it takes
        // billions of launches on one stream.
        constexpr uint32_t c_maxAhead = 1u << 16;
        constexpr uint32_t c_staleAge = 0xffffffffu;

        // Wrap-safe age of epoch `v` relative to `ref`.
        inline uint32_t epochAge(uint32_t v, uint32_t ref)
        {
            const int32_t d = static_cast<int32_t>(ref - v);
            if(d >= 0)
                return static_cast<uint32_t>(d);
            return d > -static_cast<int32_t>(c_maxAhead) ? 0u : c_staleAge;
        }

        struct Config
        {
            bool     enabled       = false; // HIPBLASLT_ADAPTIVE_SM_COUNT=1
            bool     log           = false; // HIPBLASLT_ADAPTIVE_SM_COUNT_LOG=1
            uint32_t force         = 0; // _FORCE: fixed hint
            uint32_t period        = 8; // _PERIOD: launches between reads
            uint32_t window        = 4; // _WINDOW: newest distinct probe epochs counted
            uint32_t tolerance     = 8; // _TOL: count >= N_CU - tol publishes 0
            uint32_t probePeriod   = 64; // _PROBE_PERIOD: launches without a sample before a probe
            uint32_t fullScanEvery = 64; // _FULL_SCAN: reads between full slot scans
            uint32_t confirm       = 2; // _CONFIRM: equal consecutive samples before publishing
            uint32_t confirmDown   = 4; // _CONFIRM_DOWN: same, for large or low downward moves
            uint32_t confirmUp     = 3; // _CONFIRM_UP: same, for moves from a hint to 0
            uint32_t reselectTopK  = 8; // _TOPK: candidates examined by launch-time re-selection
            uint32_t probeLosses   = 4; // _PROBE_LOSSES: lost probes in a row before hint 0
            uint32_t probeStride   = 16; // _PROBE_STRIDE: launches between padded probes, 0 = off
            uint32_t tinyMflops    = 4096; // _TINY_MFLOPS: 2*M*N*K*batch below this never probes

            // A probe launch that never shows up as the anchor (its kernel
            // could not probe) stops blocking the next one after this many
            // launches.
            uint32_t probeTimeout() const
            {
                return 8 * probePeriod;
            }

            // Tiny launches in a row, with no other launch on the stream, after
            // which a published hint is dropped: nothing re-measures it.
            uint32_t tinyStale() const
            {
                return probeLosses * probeTimeout();
            }

            static uint32_t envU32(const char* name, uint32_t def, uint32_t minValue)
            {
                const char* v = std::getenv(name);
                if(!v || !*v)
                    return def;
                char*               end = nullptr;
                const unsigned long x   = std::strtoul(v, &end, 10);
                if(end == v || x < minValue || x > 0xffffffffUL)
                    return def;
                return static_cast<uint32_t>(x);
            }

            static Config fromEnv()
            {
                Config c;
                c.force     = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE", 0, 0);
                c.enabled   = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT", 0, 0) == 1 || c.force > 0;
                c.log       = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_LOG", 0, 0) == 1;
                c.period    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_PERIOD", c.period, 1);
                c.window    = std::min(envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_WINDOW", c.window, 1),
                                    c_maxWindow);
                c.tolerance = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_TOL", c.tolerance, 0);
                c.probePeriod
                    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_PERIOD", c.probePeriod, 1);
                c.fullScanEvery
                    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_FULL_SCAN", c.fullScanEvery, 1);
                c.confirm = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_CONFIRM", c.confirm, 1);
                c.confirmDown
                    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_CONFIRM_DOWN", c.confirmDown, 1);
                c.confirmUp    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_CONFIRM_UP", c.confirmUp, 1);
                c.reselectTopK = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_TOPK", c.reselectTopK, 1);
                c.probeLosses
                    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_LOSSES", c.probeLosses, 1);
                c.probeStride
                    = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_STRIDE", c.probeStride, 0);
                c.tinyMflops = envU32("HIPBLASLT_ADAPTIVE_SM_COUNT_TINY_MFLOPS", c.tinyMflops, 0);
                return c;
            }
        };

        // The adaptive hint applies in StreamK AUTO mode when neither the desc
        // nor the handle sets an explicit sm_count_target.
        inline bool applies(bool enabled, bool modeAuto, int32_t descHint, int32_t handleHint)
        {
            return enabled && modeAuto && descHint <= 0 && handleHint <= 0;
        }

        // A launch this small neither probes nor pads: the probe would cost
        // more than 1% of it.
        inline bool tinyProblem(int64_t m, int64_t n, int64_t k, int64_t batch, const Config& cfg)
        {
            const double flops
                = 2.0 * double(m) * double(n) * double(k) * double(std::max<int64_t>(batch, 1));
            return flops < 1e6 * double(cfg.tinyMflops);
        }

        // Per-launch probe decisions for a non-tiny launch with epoch `epoch`.
        // A tiny launch takes no epoch and none of these.
        struct LaunchPlan
        {
            bool probeLaunch; // the scheduled probe launch
            bool pad; // pad a grid below N_CU with probe-only workgroups
            bool read; // sample the probe array
        };

        // Whether `epoch` is at or past the scheduled probe epoch `probeAt`
        // (0: none). The first such launch claims it.
        inline bool probeDueAt(uint32_t epoch, uint32_t probeAt)
        {
            return probeAt != 0 && static_cast<int32_t>(epoch - probeAt) >= 0;
        }

        // `claimed`: this launch claimed the scheduled probe. Every
        // probeStride-th launch is padded too, also under FORCE; FORCE never
        // reads.
        inline LaunchPlan planLaunch(uint32_t epoch, bool claimed, const Config& cfg)
        {
            return {claimed,
                    claimed || (cfg.probeStride && epoch % cfg.probeStride == 0),
                    !cfg.force && epoch % cfg.period == 0};
        }

        // A run of `tinyRun` tiny launches drops a published hint.
        inline bool tinyRunDecays(uint32_t tinyRun, uint32_t hint, const Config& cfg)
        {
            return hint != 0 && tinyRun >= cfg.tinyStale();
        }

        // A probe launch under a hint whose kernel cannot pad runs at hint 0
        // instead, but only if the kernel picked there can probe.
        inline bool probeFallbackToHintZero(bool     probeLaunch,
                                            uint32_t hint,
                                            bool     pickedCanProbe,
                                            bool     hintZeroPickCanProbe)
        {
            return probeLaunch && hint != 0 && !pickedCanProbe && hintZeroPickCanProbe;
        }

        // Hint for `count` available CUs: 0 (no hint) within `tolerance` of nCu,
        // else count rounded to the nearest multiple of the 32-CU grid
        // setSmCountTarget uses, at least 32 and 0 if that reaches nCu.
        inline uint32_t quantise(uint32_t count, uint32_t nCu, uint32_t tolerance)
        {
            if(count + tolerance >= nCu)
                return 0;
            const uint32_t q = (count + c_hintQuantum / 2) / c_hintQuantum * c_hintQuantum;
            if(q >= nCu)
                return 0;
            return std::max(c_hintQuantum, q);
        }

        // Samples a candidate hint needs before it replaces `current`.
        inline uint32_t
            confirmations(uint32_t current, uint32_t candidate, uint32_t nCu, const Config& cfg)
        {
            const uint32_t from = current ? current : nCu;
            const uint32_t to   = candidate ? candidate : nCu;
            const bool     down = to < from && (from - to > c_bigDownMove || to <= c_lowHint);
            if(down)
                return cfg.confirmDown;
            return current && !candidate ? cfg.confirmUp : cfg.confirm;
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

        // Probe-launch scheduling. Launches with a grid below N_CU do not
        // probe (a small problem, or one under a published hint); once no
        // sample has arrived for probePeriod launches, one launch keeps the
        // hint and is padded to N_CU with probe-only workgroups (or runs at
        // hint 0 if its kernel cannot pad) to re-measure. The next waits until
        // the anchor reaches that launch's epoch, or a timeout if it never
        // probed. Each lost probe moves the next one a launch later, so a
        // periodic workload does not keep landing on the same GEMM.
        class ProbeSchedule
        {
        public:
            // Returns the epoch of the launch to run as a probe (always after
            // hostEpoch), or 0 for none.
            uint32_t due(uint32_t      hostEpoch,
                         uint32_t      lastSampleEpoch,
                         bool          haveAnchor,
                         uint32_t      anchor,
                         const Config& cfg)
            {
                if(m_pending)
                {
                    if(haveAnchor && static_cast<int32_t>(anchor - m_epoch) >= 0)
                        m_lost = 0;
                    else if(static_cast<int32_t>(hostEpoch - m_epoch)
                            >= static_cast<int32_t>(m_timeout))
                        ++m_lost;
                    else
                        return 0;
                    m_pending = false;
                }
                if(hostEpoch - lastSampleEpoch < cfg.probePeriod)
                    return 0;
                // The probe runs after the launches already queued ahead of
                // it: allow twice the host's lead over the GPU at the last
                // sample on top of the base timeout.
                const uint32_t lead = haveAnchor ? epochAge(anchor, lastSampleEpoch) : 0;
                m_pending           = true;
                m_epoch             = hostEpoch + 1 + m_lost % cfg.period;
                if(m_epoch == 0)
                    m_epoch = 1;
                m_timeout = cfg.probeTimeout() + 2 * std::min(lead, c_maxAhead);
                return m_epoch;
            }

            void reset()
            {
                m_pending = false;
                m_epoch   = 0;
                m_timeout = 0;
                m_lost    = 0;
            }

            bool pending() const
            {
                return m_pending;
            }

            // Epoch of the last scheduled probe launch, or 0.
            uint32_t scheduled() const
            {
                return m_epoch;
            }

            // Probes in a row that never showed up.
            uint32_t lost() const
            {
                return m_lost;
            }

        private:
            bool     m_pending = false;
            uint32_t m_epoch   = 0;
            uint32_t m_timeout = 0;
            uint32_t m_lost    = 0;
        };

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

        private:
            std::vector<Entry> m_entries;
            size_t             m_next = 0;
        };

        // Per-stream estimator. Not thread-safe; the caller serialises it.
        class Estimator
        {
        public:
            // Samples `slots` at host epoch `hostEpoch` (the newest epoch issued
            // on the stream). Returns true when hint() changed.
            bool read(const volatile uint32_t* slots,
                      uint32_t                 hostEpoch,
                      uint32_t                 nCu,
                      const Config&            cfg)
            {
                if(m_known.empty() || m_scanNext || m_reads % cfg.fullScanEvery == 0)
                {
                    const size_t known = m_known.size();
                    fullScan(slots);
                    // A discovery scan that finds nothing (e.g. a CU-masked
                    // stream) backs off exponentially.
                    if(m_scanNext)
                    {
                        m_discoveryDelay = m_known.size() > known
                                               ? 1
                                               : std::min(2 * m_discoveryDelay, cfg.fullScanEvery);
                        m_nextDiscovery  = m_reads + m_discoveryDelay;
                    }
                }
                ++m_reads;
                m_scanNext = false;

                // Snapshot so the anchor and the count see the same values.
                m_values.resize(m_known.size());
                bool     found  = false;
                uint32_t anchor = 0;
                uint32_t minAge = 0;
                for(size_t i = 0; i < m_known.size(); ++i)
                {
                    const uint32_t v = slots[m_known[i]];
                    m_values[i]      = v;
                    if(v == 0)
                        continue;
                    // Ages are relative to the newest issued epoch.
                    const uint32_t age = epochAge(v, hostEpoch);
                    if(age == c_staleAge)
                        continue;
                    if(!found || age < minAge)
                    {
                        found  = true;
                        minAge = age;
                        anchor = v;
                    }
                }

                if(!found || (m_haveAnchor && anchor == m_anchor))
                    return false;

                // Window: slots holding one of the `window` newest distinct
                // probe epochs seen so far, so it spans `window` probing
                // launches however sparse they are; epochs since overwritten
                // still count towards it. Scheduled probe launches are at
                // least probePeriod apart, so a closer anchor that is neither
                // one nor a padded (probeStride) launch means the launches in
                // between probed too (a stream mixing probing and non-probing
                // launches gets a narrower window). A stride epoch is taken as
                // padded whether or not its kernel could pad: if the launches
                // in between did probe, the window only spans the epochs seen,
                // which is wider, never miscounted.
                const uint32_t window = std::clamp(cfg.window, 1u, c_maxWindow);
                forgetStale(anchor);
                const uint32_t step = anchor - m_anchor;
                const bool     probeAnchor = (cfg.probeStride && anchor % cfg.probeStride == 0)
                                         || (m_probe.scheduled() && anchor == m_probe.scheduled());
                if(m_haveAnchor && step < cfg.probePeriod && !probeAnchor)
                    for(uint32_t d = std::min(step, window) - 1; d > 0; --d)
                        if(anchor - d != 0)
                            remember(anchor - d, anchor, window);
                // Neighbouring slots mostly hold the same epoch.
                uint32_t last = 0;
                for(uint32_t v : m_values)
                {
                    if(v != 0 && v != last)
                        remember(v, anchor, window);
                    last = v;
                }

                m_haveAnchor          = true;
                m_anchor              = anchor;
                m_lastSampleEpoch     = hostEpoch;
                const uint32_t maxAge = epochAge(m_recent[m_recentCount - 1], anchor);

                uint32_t count   = 0;
                bool     allSeen = true;
                for(uint32_t v : m_values)
                {
                    const bool in = v != 0 && epochAge(v, anchor) <= maxAge;
                    count += in;
                    allSeen = allSeen && in;
                }
                m_count = count;
                // Every known slot is busy yet the GPU looks short: there may be
                // slots the sparse list has not discovered.
                m_scanNext = allSeen && count + cfg.tolerance < nCu
                             && static_cast<int32_t>(m_reads - m_nextDiscovery) >= 0;

                const uint32_t q = quantise(count, nCu, cfg.tolerance);
                if(m_streak > 0 && q == m_candidate)
                    ++m_streak;
                else
                {
                    m_candidate = q;
                    m_streak    = 1;
                }
                if(q != m_hint && m_streak >= confirmations(m_hint, q, nCu, cfg))
                {
                    m_hint = q;
                    return true;
                }
                return false;
            }

            // Epoch of the next launch to run as a probe launch, or 0.
            // After cfg.probeLosses lost probes in a row the hint drops to 0
            // instead, so every launch runs its full grid again.
            uint32_t probeDue(uint32_t hostEpoch, const Config& cfg)
            {
                const uint32_t at
                    = m_probe.due(hostEpoch, m_lastSampleEpoch, m_haveAnchor, m_anchor, cfg);
                if(m_probe.lost() < cfg.probeLosses)
                    return at;
                m_probe.reset();
                m_hint      = 0;
                m_candidate = 0;
                m_streak    = 0;
                return 0;
            }

            // Binds to stream `id`; a different stream than the last one
            // (a reused slot) resets the estimate. Returns true on reset.
            bool bindStream(unsigned long long id)
            {
                if(id == m_streamId)
                    return false;
                const bool rebound = m_streamId != 0;
                m_streamId         = id;
                if(rebound)
                    reset();
                return rebound;
            }

            // Forgets the published hint, anchor and window; known slots are
            // kept.
            void reset()
            {
                m_haveAnchor      = false;
                m_anchor          = 0;
                m_lastSampleEpoch = 0;
                m_candidate       = 0;
                m_streak          = 0;
                m_hint            = 0;
                m_count           = 0;
                m_recentCount     = 0;
                m_discoveryDelay  = 1;
                m_nextDiscovery   = m_reads;
                m_probe.reset();
            }

            uint32_t hint() const
            {
                return m_hint;
            }

            // CUs counted by the last sample.
            uint32_t lastCount() const
            {
                return m_count;
            }

            // E_max, and the host epoch it was first seen at.
            uint32_t anchor() const
            {
                return m_anchor;
            }

            uint32_t lastSampleEpoch() const
            {
                return m_lastSampleEpoch;
            }

            size_t knownSlots() const
            {
                return m_known.size();
            }

            uint32_t fullScans() const
            {
                return m_fullScans;
            }

        private:
            void fullScan(const volatile uint32_t* slots)
            {
                ++m_fullScans;
                for(uint32_t k = 0; k < c_probeSlots; ++k)
                {
                    if(slots[k] != 0 && !m_isKnown[k])
                    {
                        m_isKnown[k] = true;
                        m_known.push_back(static_cast<uint16_t>(k));
                    }
                }
            }

            // Inserts `v` into m_recent, the newest-first list of the newest
            // `window` distinct epochs, ages taken from `anchor`.
            void remember(uint32_t v, uint32_t anchor, uint32_t window)
            {
                const uint32_t a = epochAge(v, anchor);
                if(a == c_staleAge)
                    return;
                uint32_t i = 0;
                for(uint32_t j = 0; j < m_recentCount; ++j)
                {
                    if(m_recent[j] == v)
                        return;
                    i += epochAge(m_recent[j], anchor) < a;
                }
                if(i >= window)
                    return;
                const uint32_t n = std::min(m_recentCount + 1, window);
                for(uint32_t j = n - 1; j > i; --j)
                    m_recent[j] = m_recent[j - 1];
                m_recent[i]   = v;
                m_recentCount = n;
            }

            // Drops window epochs that `anchor` has left more than 2^31 behind.
            void forgetStale(uint32_t anchor)
            {
                uint32_t n = 0;
                for(uint32_t j = 0; j < m_recentCount; ++j)
                    if(epochAge(m_recent[j], anchor) != c_staleAge)
                        m_recent[n++] = m_recent[j];
                m_recentCount = n;
            }

            std::vector<uint16_t>     m_known;
            std::bitset<c_probeSlots> m_isKnown;
            std::vector<uint32_t>     m_values;
            uint32_t                  m_recent[c_maxWindow] = {};
            uint32_t                  m_recentCount         = 0;
            ProbeSchedule             m_probe;
            unsigned long long        m_streamId        = 0;
            uint32_t                  m_reads           = 0;
            bool                      m_scanNext        = false;
            uint32_t                  m_nextDiscovery   = 0;
            uint32_t                  m_discoveryDelay  = 1;
            uint32_t                  m_fullScans       = 0;
            bool                      m_haveAnchor      = false;
            uint32_t                  m_anchor          = 0;
            uint32_t                  m_lastSampleEpoch = 0;
            uint32_t                  m_candidate       = 0;
            uint32_t                  m_streak          = 0;
            uint32_t                  m_hint            = 0;
            uint32_t                  m_count           = 0;
        };
    } // namespace adaptive_sm
} // namespace rocblaslt
