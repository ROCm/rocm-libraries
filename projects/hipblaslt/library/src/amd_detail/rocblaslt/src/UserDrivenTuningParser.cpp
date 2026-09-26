/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "UserDrivenTuningParser.hpp"
#include "rocblaslt_secure_env.hpp"
#include "utility.hpp"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#ifndef TO_STR2
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#endif

#ifndef HIPBLASLT_VERSION_TWEAK
#error "HIPBLASLT_VERSION_TWEAK (hipblaslt-version.h) decides which tuning file rows are trusted"
#endif

namespace TensileLite
{
    const std::string& currentBuildStamp()
    {
        static const std::string stamp = TO_STR(HIPBLASLT_VERSION_TWEAK);
        return stamp;
    }

    void TuningModeSingleton::load()
    {
        m_config = TuningModeConfig::fromEnvironment(rocblaslt_process_is_privileged());

        if(m_config.suppressedForSecurity)
            log_error(__func__,
                      "Ignoring HIPBLASLT_TUNING_MODE and HIPBLASLT_TUNING_CACHE_PATH because the "
                      "process is running in a secure execution context (set-uid/set-gid or "
                      "another credential-changing exec, such as file capabilities); tuning "
                      "stays off.");
    }

    TuningFileSelection selectTuningFile()
    {
        const auto& tuning = TuningModeSingleton::getInstance();

        if(tuning.mode() == TuningMode::Off)
        {
            const OverrideSingleton& legacy = OverrideSingleton::getInstance();
            if(!legacy.env_mode)
                return {};
            return {true, legacy.file_path};
        }

        if(tuning.cachePath().empty())
        {
            // Without a path nothing is ever loaded, so this is the only place
            // that can tell a misconfigured process why nothing happens.
            announceTuningModeOnce(TuningLoadStatus::NoPath);
            return {};
        }

        return {true, tuning.cachePath()};
    }

    bool tuningAttemptIsSkip(TuningAttempt result)
    {
        switch(result)
        {
        case TuningAttempt::SkippedInPlaceBeta:
        case TuningAttempt::SkippedExtentUnknown:
        case TuningAttempt::SkippedScratchCap:
        case TuningAttempt::SkippedBudget:
            return true;
        default:
            return false;
        }
    }

    const char* tuningAttemptReason(TuningAttempt result)
    {
        switch(result)
        {
        case TuningAttempt::Tuned:
            return "tuned";
        case TuningAttempt::TunedPartial:
            return "tuned from a search the time budget stopped early";
        case TuningAttempt::SkippedInPlaceBeta:
            return "in-place C==D with nonzero beta cannot be measured without mutating its input";
        case TuningAttempt::SkippedExtentUnknown:
            return "output extent could not be established";
        case TuningAttempt::SkippedScratchCap:
            return "scratch for this shape exceeds the cap; raise or unset "
                   "HIPBLASLT_TUNING_SCRATCH_MAX_BYTES";
        case TuningAttempt::SkippedBudget:
            return "time budget stopped the search; raise or unset "
                   "HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE";
        case TuningAttempt::FallbackScratchAlloc:
            return "scratch could not be allocated on this device";
        case TuningAttempt::FallbackSetup:
            return "scratch could not be prepared";
        case TuningAttempt::FallbackEnumeration:
            return "no candidate solution could be enumerated";
        case TuningAttempt::FallbackNoWinner:
            return "no candidate completed a measurement";
        case TuningAttempt::FallbackException:
            return "benchmarking threw";
        }
        return "unknown";
    }

    namespace
    {
        /**
         * Process-lifetime state behind the tuning diagnostics. In the source
         * rather than the header so its containers can use
         * std::hash<ProblemOverride>.
         */
        struct DiagnosticsState
        {
            static DiagnosticsState& instance()
            {
                static DiagnosticsState gInstance;
                return gInstance;
            }

            bool infoOn() const
            {
                return (get_logger_layer_mode() & rocblaslt_layer_mode_log_info) != 0;
            }

            std::atomic<bool> announced{false};

            mutable std::mutex                        mutex;
            std::unordered_set<ProblemOverride>       startKeys;
            std::unordered_set<ProblemOverride>       doneKeys;
            std::unordered_set<ProblemOverride>       failedKeys;
            std::set<uint32_t>                        reasonsSeen;
            std::unordered_set<ProblemOverride>       hitKeys;
            std::unordered_set<ProblemOverride>       missKeys;
            std::unordered_set<ProblemOverride>       invalidKeys;
            std::unordered_set<ProblemOverride>       tunedKeys;
            std::unordered_map<ProblemOverride, bool> lookups;

            // Not a diagnostic: shapes this process already spent a search on,
            // kept here because the per-key process-lifetime state and its
            // mutex already live here.
            std::unordered_set<ProblemOverride> attemptedKeys;

            // Indexes already counted as rejected, per key. A key can hold
            // several entries, and each is its own rejected entry.
            std::unordered_map<ProblemOverride, std::set<int>> invalidEntries;
        };

        void summaryTally(uint64_t* shapes, uint64_t* matched, uint64_t* fellback, uint64_t* tuned)
        {
            DiagnosticsState&           state = DiagnosticsState::instance();
            std::lock_guard<std::mutex> lock(state.mutex);

            uint64_t served = 0;
            for(const auto& lookup : state.lookups)
                if(lookup.second)
                    served++;

            *shapes   = state.lookups.size();
            *matched  = served;
            *fellback = state.lookups.size() - served;
            *tuned    = state.tunedKeys.size();
        }

        /**
         * Writes the closing summary when the process unwinds its statics.
         *
         * The sink and layer mode are captured at registration rather than read
         * back here, so nothing queries the logger while statics are being torn
         * down.
         */
        struct SummaryEmitter
        {
            SummaryEmitter(std::ostream* sink, uint32_t layerMode)
                : m_sink(sink)
                , m_layerMode(layerMode)
            {
            }

            ~SummaryEmitter()
            {
                // Runs during static destruction, where an exception terminates
                // the process, and the sink throws on failure.
                try
                {
                    emit();
                }
                catch(...)
                {
                }
            }

            void emit() const
            {
                uint64_t shapes = 0, matched = 0, fellback = 0, tuned = 0;
                summaryTally(&shapes, &matched, &fellback, &tuned);

                const auto&    counters    = TuningCounters::instance();
                const uint64_t invalidated = counters.invalidated.load();

                // All zeroes would read as "the cache served nothing" when the
                // cache was never reached at all.
                if(shapes == 0 && tuned == 0 && invalidated == 0
                   && counters.entriesLoaded.load() == 0)
                    return;

                std::ostringstream msg;
                msg << "tuning-cache: summary shapes=" << shapes << " matched=" << matched
                    << " fellback=" << fellback << " tuned=" << tuned
                    << " invalidated=" << invalidated;

                std::ostream*               os = m_sink ? m_sink : &std::cerr;
                std::lock_guard<std::mutex> lock(log_mutex);

                // Laid out by hand rather than through log_info, which would
                // query the logger during static destruction, with the spacing
                // log_arguments would have used.
                if(m_layerMode & rocblaslt_layer_mode_log_info)
                    *os << prefix(rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_info), kFunc)
                        << " " << msg.str() << std::endl;
                else
                    *os << msg.str() << std::endl;
            }

            // Stands in for __func__, which here would name a destructor that
            // runs at exit.
            static constexpr const char* kFunc = "tuning_summary";

            std::ostream* m_sink;
            uint32_t      m_layerMode;
        };
    } // namespace

    void announceTuningModeOnce(TuningLoadStatus status)
    {
        const auto& tuning = TuningModeSingleton::getInstance();
        if(tuning.mode() == TuningMode::Off)
            return;

        // A plain load first: selectTuningFile calls this on every heuristic
        // call when the path is missing.
        DiagnosticsState& state = DiagnosticsState::instance();
        if(state.announced.load(std::memory_order_relaxed) || state.announced.exchange(true))
            return;

        const char* statusName = "ok";
        switch(status)
        {
        case TuningLoadStatus::Ok:
            statusName = "ok";
            break;
        case TuningLoadStatus::NotFound:
            statusName = "not-found";
            break;
        case TuningLoadStatus::ReadError:
            statusName = "read-error";
            break;
        case TuningLoadStatus::NoPath:
            statusName = "no-path";
            break;
        }

        const char* modeName = tuning.mode() == TuningMode::Tune ? "tune" : "cache";

        std::ostringstream msg;
        msg << "mode=" << modeName << " path=" << tuning.cachePath() << " load=" << statusName
            << " loaded=" << TuningCounters::instance().entriesLoaded.load();

        if(status == TuningLoadStatus::NoPath)
            msg << "; set HIPBLASLT_TUNING_CACHE_PATH or tuning does nothing";

        if(status == TuningLoadStatus::ReadError)
            msg << "; the file is there but could not be read";

        if(OverrideSingleton::getInstance().env_mode)
            msg << "; ignoring HIPBLASLT_TUNING_OVERRIDE_FILE because a tuning mode is set";

        log_tuning_lifecycle(__func__, msg.str());

        // Registered only after the line above has gone through the logger, so
        // LoggerSingleton is already constructed and outlives this emitter:
        // statics are destroyed in reverse order of construction.
        static SummaryEmitter emitter(get_logger_os(), get_logger_layer_mode());
        static_cast<void>(emitter);
    }

    void recordTuningLookup(const ProblemOverride& key, bool matched)
    {
        if(TuningModeSingleton::getInstance().mode() == TuningMode::Off)
            return;

        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);

        auto inserted = state.lookups.try_emplace(key, matched);
        if(!inserted.second && matched)
            inserted.first->second = true;
    }

    bool recordTuningInvalidation(const ProblemOverride& key, int solutionIndex)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.invalidEntries[key].insert(solutionIndex).second;
    }

    void recordTuningAttempt(const ProblemOverride& key)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.attemptedKeys.insert(key);
    }

    bool tuningAlreadyAttempted(const ProblemOverride& key)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.attemptedKeys.count(key) != 0;
    }

    void recordTuningWinner(const ProblemOverride& key)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.tunedKeys.insert(key);
    }

    bool shouldLogTuningStart(const ProblemOverride& key)
    {
        DiagnosticsState& state = DiagnosticsState::instance();
        if(state.infoOn())
            return true;

        std::lock_guard<std::mutex> lock(state.mutex);
        return state.startKeys.insert(key).second;
    }

    bool shouldLogTuningTerminal(const ProblemOverride& key, TuningAttempt result)
    {
        DiagnosticsState& state = DiagnosticsState::instance();
        if(state.infoOn())
            return true;

        std::lock_guard<std::mutex> lock(state.mutex);

        // A partial tune counts as a success here: it recorded a winner, and the
        // run that finishes the search is a later process.
        if(result == TuningAttempt::Tuned || result == TuningAttempt::TunedPartial)
            return state.doneKeys.insert(key).second;

        if(state.startKeys.count(key) != 0)
            return state.failedKeys.insert(key).second;

        return state.reasonsSeen.insert(static_cast<uint32_t>(result)).second;
    }

    bool shouldLogTuningKeyEvent(TuningKeyEvent kind, const ProblemOverride& key)
    {
        DiagnosticsState& state = DiagnosticsState::instance();
        if(!state.infoOn())
            return false;

        std::lock_guard<std::mutex> lock(state.mutex);
        switch(kind)
        {
        case TuningKeyEvent::Hit:
            return state.hitKeys.insert(key).second;
        case TuningKeyEvent::Miss:
            return state.missKeys.insert(key).second;
        case TuningKeyEvent::Invalid:
            return state.invalidKeys.insert(key).second;
        }
        return false;
    }

    void resetTuningDiagnosticsForTest()
    {
        DiagnosticsState& state = DiagnosticsState::instance();
        state.announced         = false;

        std::lock_guard<std::mutex> lock(state.mutex);
        state.startKeys.clear();
        state.doneKeys.clear();
        state.failedKeys.clear();
        state.reasonsSeen.clear();
        state.hitKeys.clear();
        state.missKeys.clear();
        state.invalidKeys.clear();
        state.tunedKeys.clear();
        state.invalidEntries.clear();
        state.attemptedKeys.clear();
        state.lookups.clear();
    }

    void tuningLookupTallyForTest(uint64_t* shapes,
                                  uint64_t* matched,
                                  uint64_t* fellback,
                                  uint64_t* tuned)
    {
        summaryTally(shapes, matched, fellback, tuned);
    }

    namespace
    {
        /**
         * The spelling string_to_hip_datatype parses back to the type it was
         * written from. hipDataType_to_bench_string spells both FP8 variants
         * of each pair the same way, which parses back as the OCP type, so an
         * FNUZ row would never match its own problem.
         */
        const char* tuningDataTypeToString(hipDataType type)
        {
            switch(type)
            {
            case HIP_R_8F_E4M3_FNUZ:
                return "f8_fnuz_r";
            case HIP_R_8F_E5M2_FNUZ:
                return "bf8_fnuz_r";
            default:
                return hipDataType_to_bench_string(type);
            }
        }

        /**
         * The spelling string_to_hipblas_computetype accepts.
         * rocblaslt_compute_type_to_string produces "COMPUTE_32F" style names,
         * which it does not.
         */
        const char* computeTypeToBenchString(rocblaslt_compute_type type)
        {
            switch(type)
            {
            case rocblaslt_compute_f32:
                return "f32_r";
            case rocblaslt_compute_f32_fast_xf32:
                return "xf32_r";
            case rocblaslt_compute_f64:
                return "f64_r";
            case rocblaslt_compute_i32:
                return "i32_r";
            case rocblaslt_compute_f32_fast_f16:
                return "f32_f16_r";
            case rocblaslt_compute_f32_fast_bf16:
                return "f32_bf16_r";
            default:
                return "f32_r";
            }
        }
    } // namespace

    bool appendTunedEntry(const std::string&                 path,
                          const RocblasltContractionProblem& problem,
                          const TunedEntry&                  entry)
    {
        if(path.empty() || entry.solutionIndex < 0)
            return false;

        const TuningRowTypes types{tuningDataTypeToString(problem.a_type),
                                   tuningDataTypeToString(problem.b_type),
                                   tuningDataTypeToString(problem.c_type),
                                   tuningDataTypeToString(problem.d_type),
                                   computeTypeToBenchString(problem.compute_type)};

        return appendTuningRow(path,
                               formatTuningRow(RocblasltContractionProblem2ProblemOverride(problem),
                                               types,
                                               entry,
                                               currentBuildStamp()),
                               currentBuildStamp());
    }

    void getContractionProblemsFromFile(const std::string& path)
    {
        if(path.empty())
            return;

        OverrideMap& map = OverrideMap::getMap();

        // Runs on every heuristic query. Once the file is loaded, the read lock
        // isLoaded takes is all a query needs.
        if(map.isLoaded(path))
            return;

        std::lock_guard<std::mutex> lock(map.getLock());
        if(map.isLoaded(path))
            return;

        const auto& tuning  = TuningModeSingleton::getInstance();
        const bool  managed = tuning.mode() != TuningMode::Off && path == tuning.cachePath();

        TuningLoadStatus status = TuningLoadStatus::Ok;
        std::ifstream    file(path);
        if(!file.is_open())
        {
            // A missing file is a status rather than an error, and is looked
            // for again on the next query, since tune mode creates it. A file
            // that is there and still will not open is a different problem,
            // usually permissions, and reporting it as missing would send the
            // user looking for a file they already have.
            std::error_code ec;
            status = std::filesystem::exists(path, ec) ? TuningLoadStatus::ReadError
                                                       : TuningLoadStatus::NotFound;
        }
        else
        {
            const auto loaded = loadTuningRows(file, map, currentBuildStamp());
            TuningCounters::instance().entriesLoaded += loaded.accepted;

            if(loaded.skippedUnnamed > 0)
                log_error(__func__,
                          "Ignored " + std::to_string(loaded.skippedUnnamed)
                              + " entries without a kernel name in " + path
                              + ": its Git Version line does not match this build. Re-run the "
                                "tuning with this build to use them.");
            else if(currentBuildStamp().empty() || loaded.fileBuildStamp != currentBuildStamp())
                log_info(__func__,
                         path
                             + " has no Git Version line matching this build; each entry is "
                               "used only while its kernel_name still matches.");

            // Only a clean read counts as loaded. A read that stopped on an I/O
            // error partway through would otherwise leave a partial map that is
            // never completed.
            if(loaded.readError)
                status = TuningLoadStatus::ReadError;
            else
                map.markLoaded(path);
        }

        if(managed)
            announceTuningModeOnce(status);
    }
} // namespace TensileLite
