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
#include "utility.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef TO_STR2
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#endif

namespace TensileLite
{
    const std::string& currentBuildStamp()
    {
#ifdef HIPBLASLT_VERSION_TWEAK
        static const std::string stamp = TO_STR(HIPBLASLT_VERSION_TWEAK);
#else
        static const std::string stamp = "";
#endif
        return stamp;
    }

#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
    TuningModeSingleton::TuningModeSingleton()
    {
        if(const char* env = getenv("HIPBLASLT_TUNING_MODE"))
        {
            const std::string value(env);
            if(value == "cache")
                m_mode = TuningMode::Cache;
            else if(value == "tune")
                m_mode = TuningMode::Tune;
            else
                m_mode = TuningMode::Off;
        }

        if(const char* path = getenv("HIPBLASLT_TUNING_CACHE_PATH"))
            m_cachePath = path;
    }

    void TuningModeSingleton::reloadForTest()
    {
        m_mode = TuningMode::Off;
        m_cachePath.clear();

        if(const char* env = getenv("HIPBLASLT_TUNING_MODE"))
        {
            const std::string value(env);
            if(value == "cache")
                m_mode = TuningMode::Cache;
            else if(value == "tune")
                m_mode = TuningMode::Tune;
        }

        if(const char* path = getenv("HIPBLASLT_TUNING_CACHE_PATH"))
            m_cachePath = path;
    }

    TuningFileSelection selectTuningFile()
    {
        const auto& tuning = TuningModeSingleton::getInstance();

        if(tuning.mode() == TuningMode::Off)
        {
            OverrideSingleton& legacy = OverrideSingleton::getInstance();
            if(!legacy.env_mode)
                return {};
            return {true, false, legacy.file_path};
        }

        if(tuning.cachePath().empty())
        {
            // A mode with nowhere to keep results does nothing at all. Say so
            // rather than leaving a misconfigured process silent: the load path
            // below never runs, so this is the only chance to announce.
            announceTuningModeOnce(TuningLoadStatus::NoPath);
            return {};
        }

        return {true, tuning.writes(), tuning.cachePath()};
    }
#else
    /**
     * Without the runtime cache there is only the offline override file, so the
     * selection collapses to what it was before the feature existed.
     */
    TuningFileSelection selectTuningFile()
    {
        OverrideSingleton& legacy = OverrideSingleton::getInstance();
        if(!legacy.env_mode)
            return {};
        return {true, false, legacy.file_path};
    }
#endif // HIPBLASLT_ENABLE_TUNING_CACHE

#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
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
         * Process-lifetime state behind the tuning diagnostics.
         *
         * Kept in the source rather than the header so the key containers can
         * use std::hash<ProblemOverride>, which is only specialised at the end
         * of the header.
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

            // Not a diagnostic: shapes whose search already ran and produced
            // nothing, so the tuner declines to repeat it. Kept here because
            // this is where the per-key process-lifetime state and its mutex
            // already live.
            std::unordered_set<ProblemOverride> fruitlessKeys;

            // Solution indexes already counted as rejected, per key. A key can
            // hold several entries and each is its own rejected entry, so the
            // index is part of the identity rather than the key alone.
            std::unordered_map<ProblemOverride, std::set<int>> invalidEntries;
        };

        void summaryTally(uint64_t* shapes, uint64_t* matched, uint64_t* fellback, uint64_t* tuned)
        {
            DiagnosticsState&           state = DiagnosticsState::instance();
            std::lock_guard<std::mutex> lock(state.mutex);

            uint64_t served = 0;
            for(const auto& [key, everMatched] : state.lookups)
            {
                static_cast<void>(key);
                if(everMatched)
                    served++;
            }

            *shapes   = state.lookups.size();
            *matched  = served;
            *fellback = state.lookups.size() - served;
            *tuned    = state.tunedKeys.size();
        }

        /**
         * Writes the closing summary when the process unwinds its statics.
         *
         * The sink and layer mode are captured at registration instead of being
         * read back here, so nothing queries the logger while statics are being
         * torn down.
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
                // Nothing may escape: this runs during static destruction, where
                // an exception terminates the process, and the sink it is about
                // to write to throws on failure. See log_tuning_lifecycle.
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

                // A process that loaded nothing and looked nothing up has no
                // summary to give, and printing one is worse than silence: all
                // zeroes reads as "the cache served nothing" when the truth is
                // that the cache was never reached. The test binary lands here
                // too, because its reset hook clears both the tally and the
                // counters between cases.
                if(shapes == 0 && tuned == 0 && invalidated == 0
                   && counters.entriesLoaded.load() == 0)
                    return;

                std::ostringstream msg;
                msg << "tuning-cache: summary shapes=" << shapes << " matched=" << matched
                    << " fellback=" << fellback << " tuned=" << tuned
                    << " invalidated=" << invalidated;

                std::ostream*               os = m_sink ? m_sink : &std::cerr;
                std::lock_guard<std::mutex> lock(log_mutex);

                // Laid out by hand rather than through log_info, because the
                // logger must not be queried while statics are being destroyed.
                // The spacing has to match what log_arguments would have written
                // so this line is not the one odd entry in the file.
                if(m_layerMode & rocblaslt_layer_mode_log_info)
                    *os << prefix(rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_info), kFunc)
                        << " " << msg.str() << std::endl;
                else
                    *os << msg.str() << std::endl;
            }

            // Stands in for __func__: the enclosing function is a destructor
            // running at exit, and naming it that in the log would tell the
            // reader nothing about which subsystem spoke.
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

        // Plain load first: selectTuningFile calls this on every heuristic call,
        // and after the first one this should cost a predictable branch rather
        // than a read-modify-write.
        DiagnosticsState& state = DiagnosticsState::instance();
        if(state.announced.load(std::memory_order_relaxed) || state.announced.exchange(true))
            return;

        const char* modeName = tuning.mode() == TuningMode::Tune ? "tune" : "cache";

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

        std::ostringstream msg;
        msg << "mode=" << modeName << " path=" << tuning.cachePath() << " load=" << statusName
            << " loaded=" << TuningCounters::instance().entriesLoaded.load();

        if(status == TuningLoadStatus::NoPath)
            msg << "; set HIPBLASLT_TUNING_CACHE_PATH or tuning does nothing";

        if(status == TuningLoadStatus::ReadError)
            msg << "; the file is there but could not be read";

        // The two file variables are mutually exclusive rather than merged, and
        // until now the losing one was dropped without a word.
        if(OverrideSingleton::getInstance().env_mode)
            msg << "; ignoring HIPBLASLT_TUNING_OVERRIDE_FILE because a tuning mode is set";

        log_tuning_lifecycle(__func__, msg.str());

        // Registered only after the line above has gone through the logger, so
        // LoggerSingleton is already constructed and therefore outlives this
        // emitter. Static destruction runs in reverse order of construction.
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

    void recordFruitlessTuning(const ProblemOverride& key)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.fruitlessKeys.insert(key);
    }

    bool tuningWasFruitless(const ProblemOverride& key)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.fruitlessKeys.count(key) != 0;
    }

    void recordTuningWinner(const ProblemOverride& key)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.tunedKeys.insert(key);
    }

    bool recordTuningInvalidation(const ProblemOverride& key, int solutionIndex)
    {
        DiagnosticsState&           state = DiagnosticsState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.invalidEntries[key].insert(solutionIndex).second;
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

        // Success and failure are bounded separately, not against one shared
        // per-key slot. A failed attempt records no winner, so the same key is
        // attempted again on the next matmul and can succeed, and that later
        // tuning-done is the line explaining the minutes the caller just spent.
        if(result == TuningAttempt::Tuned)
            return state.doneKeys.insert(key).second;

        // Announcing a start is a promise that something will say how it ended,
        // so a key that got one is answered per key. Only the declines made
        // before the start line, which are the ones that repeat across thousands
        // of un-tunable shapes, collapse to the first occurrence of each reason.
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
        state.fruitlessKeys.clear();
        state.lookups.clear();
    }

    void tuningLookupTallyForTest(uint64_t* shapes,
                                  uint64_t* matched,
                                  uint64_t* fellback,
                                  uint64_t* tuned)
    {
        summaryTally(shapes, matched, fellback, tuned);
    }
#endif // HIPBLASLT_ENABLE_TUNING_CACHE

    namespace
    {
        const char* const kGitVersionHeader = "Git Version: ";

        // A line is a header row if it names the first problem column. Value
        // rows never do.
        const char* const kHeaderSentinel = "transA";

        std::string trimmed(const std::string& s)
        {
            const char* ws    = " \t\n\r\f\v";
            auto        first = s.find_first_not_of(ws);
            if(first == std::string::npos)
                return {};
            auto last = s.find_last_not_of(ws);
            return s.substr(first, last - first + 1);
        }

        std::vector<std::string> splitCsv(const std::string& line)
        {
            std::vector<std::string> out;
            std::stringstream        ss(line);
            std::string              cell;
            while(std::getline(ss, cell, ','))
                out.push_back(trimmed(cell));
            return out;
        }

        bool isHeaderRow(const std::string& line)
        {
            const auto cells = splitCsv(line);
            return std::find(cells.begin(), cells.end(), kHeaderSentinel) != cells.end()
                   && std::find(cells.begin(), cells.end(), "solution_index") != cells.end();
        }

        /**
         * Zip a header row and a value row into a name-keyed row.
         *
         * Name-keyed rather than positional on purpose. The previous parser
         * walked a fixed HeaderFields sequence and only advanced when a column
         * matched the next expected name, then rejected any row whose collected
         * count was not exactly the enum size. That made adding a key field a
         * breaking format change and silently discarded rows.
         */
        std::map<std::string, std::string> zipRow(const std::vector<std::string>& names,
                                                  const std::vector<std::string>& values)
        {
            std::map<std::string, std::string> row;
            const size_t                       count = std::min(names.size(), values.size());
            for(size_t i = 0; i < count; i++)
                if(!names[i].empty())
                    row.emplace(names[i], values[i]);
            return row;
        }

        bool has(const std::map<std::string, std::string>& row, const char* name)
        {
            return row.find(name) != row.end();
        }

        std::string str(const std::map<std::string, std::string>& row,
                        const char*                               name,
                        const std::string&                        fallback = {})
        {
            auto it = row.find(name);
            return it == row.end() ? fallback : it->second;
        }

        // Absent or unparseable columns take the caller's default. That is safe
        // for mode and presence fields, where the default narrows what an entry
        // matches. It is not safe for strides and leading dimensions, which have
        // no meaningful default; rows missing those are rejected as legacy, see
        // problemFromEntries.
        int64_t num(const std::map<std::string, std::string>& row,
                    const char*                               name,
                    int64_t                                   fallback = 0)
        {
            auto it = row.find(name);
            if(it == row.end() || it->second.empty())
                return fallback;
            try
            {
                return std::stoll(it->second);
            }
            catch(...)
            {
                return fallback;
            }
        }

        double real(const std::map<std::string, std::string>& row,
                    const char*                               name,
                    double                                    fallback = 0.0)
        {
            auto it = row.find(name);
            if(it == row.end() || it->second.empty())
                return fallback;
            try
            {
                return std::stod(it->second);
            }
            catch(...)
            {
                return fallback;
            }
        }

        bool flag(const std::map<std::string, std::string>& row,
                  const char*                               name,
                  bool                                      fallback = false)
        {
            auto it = row.find(name);
            if(it == row.end() || it->second.empty())
                return fallback;

            const std::string& v = it->second;
            if(v == "true" || v == "True" || v == "TRUE")
                return true;
            if(v == "false" || v == "False" || v == "FALSE")
                return false;
            return num(row, name, fallback ? 1 : 0) != 0;
        }

        std::optional<rocisa::DataType> dataType(const std::map<std::string, std::string>& row,
                                                 const char*                               name)
        {
            auto it = row.find(name);
            if(it == row.end() || it->second.empty())
                return std::nullopt;

            const hipDataType type = string_to_hip_datatype(it->second);
            if(type == HIPBLASLT_DATATYPE_INVALID)
                return std::nullopt;

            // hipDataType_to_tensile_type asserts on unsupported enum values.
            // The public string parser also recognizes scale-only/newer types
            // such as e8_r and e5m3_r which that converter does not support as
            // GEMM tensor types, so whitelist before calling it.
            switch(type)
            {
            case HIP_R_16F:
            case HIP_R_32F:
            case HIP_R_64F:
            case HIP_R_16BF:
            case HIP_R_8F_E4M3_FNUZ:
            case HIP_R_8F_E5M2_FNUZ:
            case HIP_R_8F_E4M3:
            case HIP_R_8F_E5M2:
            case HIP_R_8I:
            case HIP_R_32I:
            case HIP_C_32F:
            case HIP_C_64F:
            case HIP_R_6F_E2M3:
            case HIP_R_6F_E3M2:
            case HIP_R_4F_E2M1:
                return hipDataType_to_tensile_type(type);
            default:
                return std::nullopt;
            }
        }

        /**
         * A cell that is a number and nothing else, or nothing.
         *
         * num() above folds "absent", "empty" and "not a number" all into the
         * caller's default, which is right for optional fields and wrong for
         * ones being validated: an empty lda would validate as the perfectly
         * plausible leading dimension 0. splitCsv has already trimmed the cell,
         * so any leftover character means the cell is not a number.
         */
        std::optional<int64_t> exactNum(const std::map<std::string, std::string>& row,
                                        const char*                               name)
        {
            auto it = row.find(name);
            if(it == row.end() || it->second.empty())
                return std::nullopt;
            try
            {
                size_t        consumed = 0;
                const int64_t value    = std::stoll(it->second, &consumed);
                if(consumed != it->second.size())
                    return std::nullopt;
                return value;
            }
            catch(...)
            {
                return std::nullopt;
            }
        }

        bool isInt32(const std::map<std::string, std::string>& row, const char* name)
        {
            const auto value = exactNum(row, name);
            return value && *value >= std::numeric_limits<int32_t>::min()
                   && *value <= std::numeric_limits<int32_t>::max();
        }

        // Every field in a current row's semantic key is mandatory and strict.
        // Permissive defaults remain solely for schema-less v0 rows.
        bool hasRequiredCurrentColumns(const std::map<std::string, std::string>& row)
        {
            if((str(row, "transA") != "N" && str(row, "transA") != "T")
               || (str(row, "transB") != "N" && str(row, "transB") != "T"))
                return false;

            static const char* const requiredText[]
                = {"a_type", "b_type", "c_type", "d_type", "compute_type", "gcnArchName"};
            for(const char* name : requiredText)
                if(!has(row, name) || str(row, name).empty())
                    return false;

            static const char* const nonNegative[] = {"m",
                                                      "n",
                                                      "k",
                                                      "batch_count",
                                                      "lda",
                                                      "ldb",
                                                      "ldc",
                                                      "ldd",
                                                      "stride_a",
                                                      "stride_b",
                                                      "stride_c",
                                                      "stride_d",
                                                      "required_workspace"};
            for(const char* name : nonNegative)
            {
                const auto value = exactNum(row, name);
                if(!value || *value < 0)
                    return false;
            }
            if(*exactNum(row, "batch_count") == 0)
                return false;

            static const char* const int32Fields[] = {"compute_input_type_a",
                                                      "compute_input_type_b",
                                                      "batch_mode",
                                                      "epilogue",
                                                      "bias_type",
                                                      "bias_stride",
                                                      "aux_type",
                                                      "scaleA_format",
                                                      "scaleB_format",
                                                      "streamk_tile_scheduling",
                                                      "sm_count_target",
                                                      "CUs"};
            for(const char* name : int32Fields)
                if(!isInt32(row, name))
                    return false;

            static const char* const boolFields[] = {"gradient",
                                                     "bias_vector",
                                                     "scaleA",
                                                     "scaleB",
                                                     "scaleC",
                                                     "scaleD",
                                                     "scaleE",
                                                     "scaleAlphaVec",
                                                     "amaxD",
                                                     "swizzle_a",
                                                     "swizzle_b"};
            for(const char* name : boolFields)
            {
                const auto value = exactNum(row, name);
                if(!value || (*value != 0 && *value != 1))
                    return false;
            }

            const bool hasName
                = (has(row, "kernel_name") && !str(row, "kernel_name").empty())
                  || (has(row, "solution_name") && !str(row, "solution_name").empty());
            if(!hasName)
                return false;

            return true;
        }
    } // namespace

    std::optional<std::pair<ProblemOverride, TunedEntry>>
        problemFromEntries(const std::map<std::string, std::string>& row)
    {
        // Reject a malformed or unknown schema before interpreting any other
        // cell. A newer schema can introduce datatype spellings this build does
        // not know; parsing those first reaches assert-based enum converters and
        // aborts a debug build before the version check gets a chance to reject
        // the row.
        TuningSchemaVersion schemaVersion = TuningSchemaVersion::Legacy;
        if(has(row, "schema_version"))
        {
            const auto version = exactNum(row, "schema_version");
            if(!version || *version != static_cast<int64_t>(TuningSchemaVersion::Current)
               || !hasRequiredCurrentColumns(row))
                return std::nullopt;
            schemaVersion = TuningSchemaVersion::Current;
        }

        // A solution index of 0 is legitimate: getSolutionsFromIndex accepts it
        // and the shipped logic YAML contains SolutionIndex: 0. The previous
        // parser required > 0 and silently dropped those rows.
        const auto solutionIndexValue = exactNum(row, "solution_index");
        if(!solutionIndexValue || *solutionIndexValue < 0
           || *solutionIndexValue > std::numeric_limits<int32_t>::max())
            return std::nullopt;

        const int32_t solutionIndex = static_cast<int32_t>(*solutionIndexValue);

        ProblemOverride po;

        po.transA = str(row, "transA", "N") != "N";
        po.transB = str(row, "transB", "N") != "N";

        po.m         = static_cast<size_t>(num(row, "m"));
        po.n         = static_cast<size_t>(num(row, "n"));
        po.k         = static_cast<size_t>(num(row, "k"));
        po.batchSize = static_cast<size_t>(num(row, "batch_count", 1));

        const auto inputTypeA  = dataType(row, "a_type");
        const auto inputTypeB  = dataType(row, "b_type");
        const auto outputTypeC = dataType(row, "c_type");
        if(!inputTypeA || !inputTypeB || !outputTypeC)
            return std::nullopt;

        po.inputTypeA  = *inputTypeA;
        po.inputTypeB  = *inputTypeB;
        po.outputTypeC = *outputTypeC;
        // Historical rows key only c_type; treat d as equal to c when absent so
        // such a row matches the non-mixed-precision problems it described.
        if(has(row, "d_type"))
        {
            const auto outputTypeD = dataType(row, "d_type");
            if(!outputTypeD)
                return std::nullopt;
            po.outputTypeD = *outputTypeD;
        }
        else
        {
            po.outputTypeD = po.outputTypeC;
        }

        {
            const std::string ct = str(row, "compute_type");
            if(ct.empty())
                return std::nullopt;

            const auto computeType = string_to_hipblas_computetype(ct);
            if(computeType == HIPBLASLT_COMPUTE_TYPE_INVALID)
                return std::nullopt;

            po.computeType
                = rocComputeType_to_tensile_type(static_cast<rocblaslt_compute_type>(computeType));
        }
        po.computeInputTypeA = static_cast<int32_t>(num(row, "compute_input_type_a"));
        po.computeInputTypeB = static_cast<int32_t>(num(row, "compute_input_type_b"));

        po.colStrideA   = static_cast<size_t>(num(row, "lda"));
        po.colStrideB   = static_cast<size_t>(num(row, "ldb"));
        po.colStrideC   = static_cast<size_t>(num(row, "ldc"));
        po.colStrideD   = static_cast<size_t>(num(row, "ldd"));
        po.batchStrideA = static_cast<size_t>(num(row, "stride_a"));
        po.batchStrideB = static_cast<size_t>(num(row, "stride_b"));
        po.batchStrideC = static_cast<size_t>(num(row, "stride_c"));
        po.batchStrideD = static_cast<size_t>(num(row, "stride_d"));
        po.batchMode    = static_cast<int32_t>(num(row, "batch_mode"));

        po.epilogue   = static_cast<int32_t>(num(row, "epilogue"));
        po.gradient   = flag(row, "gradient");
        po.biasType   = static_cast<int32_t>(num(row, "bias_type"));
        po.biasStride = static_cast<int32_t>(num(row, "bias_stride"));
        po.hasBias    = flag(row, "bias_vector");
        po.auxType    = static_cast<int32_t>(num(row, "aux_type"));

        po.scaleAFormat     = static_cast<int32_t>(num(row, "scaleA_format"));
        po.scaleBFormat     = static_cast<int32_t>(num(row, "scaleB_format"));
        po.hasScaleA        = flag(row, "scaleA");
        po.hasScaleB        = flag(row, "scaleB");
        po.hasScaleC        = flag(row, "scaleC");
        po.hasScaleD        = flag(row, "scaleD");
        po.hasScaleE        = flag(row, "scaleE");
        po.hasScaleAlphaVec = flag(row, "scaleAlphaVec");
        po.hasAmaxD         = flag(row, "amaxD");

        po.swizzleA              = flag(row, "swizzle_a");
        po.swizzleB              = flag(row, "swizzle_b");
        po.streamkTileScheduling = static_cast<int32_t>(num(row, "streamk_tile_scheduling"));
        po.smCountTarget         = static_cast<int32_t>(num(row, "sm_count_target"));

        // Kept unstripped. The library's own rocblaslt_internal_get_arch_name()
        // truncates at the first colon, which would discard sramecc and xnack;
        // those can affect which kernels apply, and the bench writer already
        // records the full string, so the full string is what both sides use.
        po.archName = str(row, "gcnArchName");
        po.cuCount  = static_cast<int32_t>(num(row, "CUs"));

        TunedEntry entry;
        entry.solutionIndex = solutionIndex;

        if(has(row, "kernel_name") && !str(row, "kernel_name").empty())
            entry.kernelName = str(row, "kernel_name");
        if(has(row, "solution_name") && !str(row, "solution_name").empty())
            entry.solutionName = str(row, "solution_name");

        entry.schemaVersion = schemaVersion;

        entry.buildStamp             = str(row, "git_version");
        if(schemaVersion == TuningSchemaVersion::Current)
            entry.requiredWorkspaceBytes
                = static_cast<size_t>(*exactNum(row, "required_workspace"));
        else
            entry.requiredWorkspaceBytes
                = static_cast<size_t>(std::max<int64_t>(0, num(row, "required_workspace")));
        entry.winnerTimeUs = real(row, "us");

        return std::make_pair(po, entry);
    }

    namespace
    {
        /**
         * The spelling string_to_hipblas_computetype accepts, so a row this
         * writer emits parses back to the compute type it was written from.
         * rocblaslt_compute_type_to_string produces "COMPUTE_32F" style names,
         * which the parser does not recognise.
         */
        /**
         * The spelling string_to_hip_datatype parses back to the type it was
         * written from.
         *
         * hipDataType_to_bench_string collapses each FP8 pair onto one spelling,
         * so both HIP_R_8F_E4M3_FNUZ and HIP_R_8F_E4M3 emit "f8_r", and the
         * parser resolves that to the OCP type. gfx942 FP8 is FNUZ, so on that
         * architecture every FP8 row came back describing a different type than
         * the one measured, never matched its own key, and made the shape retune
         * and append another row on every process start. Only the two FNUZ types
         * are ambiguous; everything else already round-trips.
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

#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
    bool appendTunedEntry(const std::string&                 path,
                          const RocblasltContractionProblem& problem,
                          const TunedEntry&                  entry)
    {
        if(path.empty() || entry.solutionIndex < 0)
            return false;

        static std::mutex          writeMutex;
        std::lock_guard<std::mutex> lock(writeMutex);

        const bool needHeader = [&] {
            std::ifstream probe(path);
            return !probe || probe.peek() == std::ifstream::traits_type::eof();
        }();

        std::ofstream out(path, std::ios::app);
        if(!out)
            return false;

        // Built as one payload and written once, so a row cannot be split by
        // another writer in this process.
        std::ostringstream names;
        std::ostringstream values;
        bool               first = true;

        auto column = [&](const char* name, auto&& value) {
            if(!first)
            {
                names << ',';
                values << ',';
            }
            first = false;
            names << name;
            values << value;
        };

        // Built once and used for every column the key derives rather than the
        // problem, so what is written is exactly what a later lookup rebuilds.
        // Any key field missing from this row parses back as its default and
        // silently fails to match.
        const ProblemOverride key = RocblasltContractionProblem2ProblemOverride(problem);

        column("schema_version", static_cast<uint32_t>(TuningSchemaVersion::Current));
        column("git_version", currentBuildStamp());

        column("transA", problem.trans_a == HIPBLAS_OP_N ? "N" : "T");
        column("transB", problem.trans_b == HIPBLAS_OP_N ? "N" : "T");
        column("m", problem.m);
        column("n", problem.n);
        column("k", problem.k);
        column("batch_count", problem.batch_count);

        column("a_type", tuningDataTypeToString(problem.a_type));
        column("b_type", tuningDataTypeToString(problem.b_type));
        column("c_type", tuningDataTypeToString(problem.c_type));
        column("d_type", tuningDataTypeToString(problem.d_type));
        column("compute_type", computeTypeToBenchString(problem.compute_type));
        column("compute_input_type_a", key.computeInputTypeA);
        column("compute_input_type_b", key.computeInputTypeB);

        column("lda", problem.col_stride_a);
        column("ldb", problem.col_stride_b);
        column("ldc", problem.col_stride_c);
        column("ldd", problem.col_stride_d);
        column("stride_a", problem.batch_stride_a);
        column("stride_b", problem.batch_stride_b);
        column("stride_c", problem.batch_stride_c);
        column("stride_d", problem.batch_stride_d);
        column("batch_mode", static_cast<int32_t>(problem.batchMode));

        column("epilogue", static_cast<int32_t>(problem.epilogue));
        column("gradient", problem.gradient ? 1 : 0);
        column("bias_type", static_cast<int32_t>(problem.bias_type));
        column("bias_stride", problem.bias_stride);
        column("bias_vector", problem.bias != nullptr ? 1 : 0);
        column("aux_type", static_cast<int32_t>(problem.aux_type));

        column("scaleA_format", static_cast<int32_t>(problem.scaleAType));
        column("scaleB_format", static_cast<int32_t>(problem.scaleBType));
        column("scaleA", problem.scaleA != nullptr ? 1 : 0);
        column("scaleB", problem.scaleB != nullptr ? 1 : 0);
        column("scaleC", problem.scaleC != nullptr ? 1 : 0);
        column("scaleD", problem.scaleD != nullptr ? 1 : 0);
        column("scaleE", problem.scaleE != nullptr ? 1 : 0);
        column("scaleAlphaVec", problem.scaleAlphaVec != nullptr ? 1 : 0);
        column("amaxD", problem.amaxD != nullptr ? 1 : 0);

        column("swizzle_a", problem.swizzleA ? 1 : 0);
        column("swizzle_b", problem.swizzleB ? 1 : 0);
        column("streamk_tile_scheduling", problem.streamk_tile_scheduling_ext);
        column("sm_count_target", problem.sm_count_target);

        // From the canonical builder rather than a fresh device query, so the
        // arch recorded here is byte-identical to what a later lookup compares.
        column("gcnArchName", key.archName);
        column("CUs", key.cuCount);

        column("solution_index", entry.solutionIndex);
        column("kernel_name", entry.kernelName.value_or(std::string{}));
        column("required_workspace", entry.requiredWorkspaceBytes);
        column("us", entry.winnerTimeUs);

        // One write for the header and the row together, rather than a flush
        // apiece. The process mutex above orders writers inside this process,
        // but nothing orders a second process appending to the same file, and
        // there the unit of tearing is the write: a single one that the
        // filesystem appends atomically cannot be split down the middle by
        // another rank, whereas two can interleave. This narrows that window on
        // a local filesystem; it is not a guarantee, least of all on the network
        // filesystems multi-node jobs use, which is why concurrent writers stay
        // unsupported.
        std::ostringstream payload;
        if(needHeader)
            payload << kGitVersionHeader << currentBuildStamp() << "\n";
        payload << "    " << names.str() << "\n" << values.str() << "\n";

        const std::string bytes = payload.str();
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();

        // Reported rather than swallowed: a full disk or a read-only path
        // otherwise looks like a successful tune whose results silently vanish
        // at process exit.
        return out.good();
    }
#endif // HIPBLASLT_ENABLE_TUNING_CACHE

    void getContractionProblemsFromFile(const std::string& path)
    {
        if(path.empty())
            return;

        OverrideMap& m_override = OverrideMap::getMap();

#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
        const bool fromManagedCache = (TuningModeSingleton::getInstance().mode() != TuningMode::Off)
                                      && (path == TuningModeSingleton::getInstance().cachePath());
#else
        // Only the offline override file exists here, and it is never announced.
        const bool fromManagedCache = false;
#endif

        // Claimed before the file is opened. Opening first meant every lookup on
        // an already-loaded path paid a filesystem open just to discover the
        // load latch was already set, on the replay hot path.
        if(!m_override.claimLoad(path))
            return;

#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
        // Announced from a scope guard because the load has several exits and
        // the user is owed the same startup line whichever one it takes.
        // Declared before the claim so it runs after it, once the accepted count
        // is final.
        TuningLoadStatus loadStatus = TuningLoadStatus::Ok;

        struct LoadAnnounce
        {
            bool                    managed;
            const TuningLoadStatus* status;
            ~LoadAnnounce()
            {
                if(managed)
                    announceTuningModeOnce(*status);
            }
        } announce{fromManagedCache, &loadStatus};
#else
        static_cast<void>(fromManagedCache);
#endif

        struct LoadClaim
        {
            OverrideMap& map;
            std::string  path;
            bool         success = false;
            ~LoadClaim()
            {
                map.finishLoad(path, success);
            }
        } claim{m_override, path};

        std::ifstream file_read(path);
        if(!file_read)
        {
            // A missing file is expected on the first tune run, so it is a
            // status rather than an error: the file is about to be created by
            // the first winner. A file that is there and still would not open is
            // a different problem, usually permissions, and reporting it as
            // missing sends the user looking for a file they already have.
#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
            std::error_code ec;
            loadStatus = std::filesystem::exists(path, ec) ? TuningLoadStatus::ReadError
                                                           : TuningLoadStatus::NotFound;
#endif
            return;
        }

        std::string fileBuildStamp;
        std::string line;
        std::string pendingHeader;

        while(true)
        {
            if(!pendingHeader.empty())
            {
                line = std::move(pendingHeader);
                pendingHeader.clear();
            }
            else if(!std::getline(file_read, line))
            {
                break;
            }

            const std::string header = trimmed(line);
            if(header.empty())
                continue;

            if(fileBuildStamp.empty())
            {
                const auto pos = header.find(kGitVersionHeader);
                if(pos != std::string::npos)
                {
                    fileBuildStamp = header.substr(pos + std::strlen(kGitVersionHeader));
                    continue;
                }
            }

            if(!isHeaderRow(header))
                continue;

            std::string valueLine;
            if(!std::getline(file_read, valueLine))
                break;

            const std::string value = trimmed(valueLine);
            if(isHeaderRow(value))
            {
                // A crashed/torn append can leave only its header. Do not
                // consume the next valid header as that record's values; put it
                // back through the loop so its following value row survives.
                pendingHeader = value;
                continue;
            }

            auto parsed = problemFromEntries(zipRow(splitCsv(header), splitCsv(value)));
            if(!parsed)
                continue;

            auto& [key, entry] = *parsed;

            entry.source = fromManagedCache ? TuningEntrySource::ManagedCacheFile
                                            : TuningEntrySource::LegacyOverrideFile;

            if(entry.buildStamp.empty())
                entry.buildStamp = fileBuildStamp;

            // A row carrying a name validates itself at replay: the recorded
            // index is resolved in the current library and the name must still
            // match. A row without one has no such check, so the build stamp is
            // the only trust signal it has.
            if(!entry.kernelName.has_value() && !entry.solutionName.has_value()
               && entry.buildStamp != currentBuildStamp())
                continue;

            // v0 rows go into the narrow map. They cannot populate the widened
            // key, so filing them under it would mean they match nothing, which
            // is how existing hipblaslt-bench files stopped replaying.
            const bool inserted = (entry.schemaVersion == TuningSchemaVersion::Legacy)
                                      ? m_override.addLegacyIfAbsent(key, entry)
                                      : m_override.addIfAbsent(key, entry);
            if(inserted)
                TuningCounters::instance().entriesLoaded++;
        }

        // Only a clean read counts as loaded. A read that stopped on badbit,
        // such as an I/O error partway through, would otherwise latch a
        // partially populated map as complete and never be retried.
        claim.success = !file_read.bad();
#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
        if(!claim.success)
            loadStatus = TuningLoadStatus::ReadError;
#endif
    }
} // namespace TensileLite
