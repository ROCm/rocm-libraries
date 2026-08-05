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
#include <fstream>
#include <shared_mutex>
#include <sstream>
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
            return {};

        return {true, tuning.writes(), tuning.cachePath()};
    }

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

        rocisa::DataType dataType(const std::map<std::string, std::string>& row,
                                  const char*                               name)
        {
            auto it = row.find(name);
            if(it == row.end() || it->second.empty())
                return rocisa::DataType::None;
            return hipDataType_to_tensile_type(string_to_hip_datatype(it->second));
        }

        // Columns that a current-schema row must carry. These are the fields
        // with no safe default, so a row lacking any of them cannot be treated
        // as current no matter what schema_version claims.
        bool hasRequiredCurrentColumns(const std::map<std::string, std::string>& row)
        {
            static const char* const required[]
                = {"lda", "ldb", "ldc", "ldd", "stride_a", "stride_b", "stride_c", "stride_d"};
            for(const char* name : required)
                if(!has(row, name))
                    return false;
            return true;
        }
    } // namespace

    std::optional<std::pair<ProblemOverride, TunedEntry>>
        problemFromEntries(const std::map<std::string, std::string>& row)
    {
        // A solution index of 0 is legitimate: getSolutionsFromIndex accepts it
        // and the shipped logic YAML contains SolutionIndex: 0. The previous
        // parser required > 0 and silently dropped those rows.
        if(!has(row, "solution_index"))
            return std::nullopt;

        const int64_t solutionIndex = num(row, "solution_index", -1);
        if(solutionIndex < 0)
            return std::nullopt;

        ProblemOverride po;

        po.transA = str(row, "transA", "N") != "N";
        po.transB = str(row, "transB", "N") != "N";

        po.m         = static_cast<size_t>(num(row, "m"));
        po.n         = static_cast<size_t>(num(row, "n"));
        po.k         = static_cast<size_t>(num(row, "k"));
        po.batchSize = static_cast<size_t>(num(row, "batch_count", 1));

        po.inputTypeA  = dataType(row, "a_type");
        po.inputTypeB  = dataType(row, "b_type");
        po.outputTypeC = dataType(row, "c_type");
        // Historical rows key only c_type; treat d as equal to c when absent so
        // such a row matches the non-mixed-precision problems it described.
        po.outputTypeD = has(row, "d_type") ? dataType(row, "d_type") : po.outputTypeC;

        {
            const std::string ct = str(row, "compute_type");
            po.computeType
                = ct.empty() ? rocisa::DataType::None
                             : rocComputeType_to_tensile_type(static_cast<rocblaslt_compute_type>(
                                   string_to_hipblas_computetype(ct)));
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
        entry.solutionIndex = static_cast<int32_t>(solutionIndex);

        if(has(row, "solution_name") && !str(row, "solution_name").empty())
            entry.solutionName = str(row, "solution_name");

        const bool claimsCurrent = has(row, "schema_version")
                                   && num(row, "schema_version")
                                          >= static_cast<int64_t>(TuningSchemaVersion::Current);

        entry.schemaVersion = (claimsCurrent && hasRequiredCurrentColumns(row))
                                  ? TuningSchemaVersion::Current
                                  : TuningSchemaVersion::Legacy;

        entry.buildStamp             = str(row, "git_version");
        entry.requiredWorkspaceBytes = static_cast<size_t>(num(row, "required_workspace"));
        entry.winnerTimeUs           = real(row, "us");
        entry.baselineIndex          = static_cast<int32_t>(num(row, "baseline_index", -1));
        entry.baselineTimeUs         = real(row, "baseline_us");

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

        static std::mutex          writeMutex;
        std::lock_guard<std::mutex> lock(writeMutex);

        const bool needHeader = [&] {
            std::ifstream probe(path);
            return !probe || probe.peek() == std::ifstream::traits_type::eof();
        }();

        std::ofstream out(path, std::ios::app);
        if(!out)
            return false;

        if(needHeader)
            out << kGitVersionHeader << currentBuildStamp() << std::endl;

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

        column("a_type", hipDataType_to_bench_string(problem.a_type));
        column("b_type", hipDataType_to_bench_string(problem.b_type));
        column("c_type", hipDataType_to_bench_string(problem.c_type));
        column("d_type", hipDataType_to_bench_string(problem.d_type));
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
        column("solution_name", entry.solutionName.value_or(std::string{}));
        column("required_workspace", entry.requiredWorkspaceBytes);
        column("us", entry.winnerTimeUs);

        // Recorded but not read back in this milestone. What default selection
        // would have chosen, and how fast it was, cannot be reconstructed after
        // the fact, and without it the only answer to "has the default caught
        // up with my tuned entry" is to re-tune everything.
        column("baseline_index", entry.baselineIndex);
        column("baseline_us", entry.baselineTimeUs);

        out << "    " << names.str() << "\n" << values.str() << std::endl;

        // Reported rather than swallowed: a full disk or a read-only path
        // otherwise looks like a successful tune whose results silently vanish
        // at process exit.
        return out.good();
    }

    void getContractionProblemsFromFile(const std::string& path)
    {
        if(path.empty())
            return;

        OverrideMap& m_override = OverrideMap::getMap();

        // Each path is parsed at most once. The previous gate was
        // m_override.size() == 0, which re-read the file on every heuristic call
        // whenever it yielded no usable rows, and which would also have blocked
        // loading once online tuning had inserted anything.
        if(!m_override.claimLoad(path))
            return;

        std::ifstream file_read(path);
        if(!file_read)
            return;

        const bool fromManagedCache
            = (TuningModeSingleton::getInstance().mode() != TuningMode::Off)
              && (path == TuningModeSingleton::getInstance().cachePath());

        std::string fileBuildStamp;
        std::string line;

        while(std::getline(file_read, line))
        {
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

            if(header.find(kHeaderSentinel) == std::string::npos)
                continue;

            std::string valueLine;
            if(!std::getline(file_read, valueLine))
                break;

            auto parsed = problemFromEntries(zipRow(splitCsv(header), splitCsv(trimmed(valueLine))));
            if(!parsed)
                continue;

            auto& [key, entry] = *parsed;

            entry.source = fromManagedCache ? TuningEntrySource::ManagedCacheFile
                                            : TuningEntrySource::LegacyOverrideFile;

            if(entry.buildStamp.empty())
                entry.buildStamp = fileBuildStamp;

            // A row carrying a solution name validates itself at replay: the
            // recorded index is resolved in the current library and the name
            // must still match. A row without one has no such check, so the
            // build stamp is the only trust signal it has.
            if(!entry.solutionName.has_value() && entry.buildStamp != currentBuildStamp())
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
    }
} // namespace TensileLite
