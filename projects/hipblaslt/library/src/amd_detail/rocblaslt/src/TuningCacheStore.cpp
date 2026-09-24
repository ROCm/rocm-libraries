// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TuningCacheStore.hpp"
#include "hipblaslt_type_strings.hpp"
#include "rocblaslt_secure_env.hpp"
#include "tensile_type_map.hpp"

#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>

namespace TensileLite
{
    TuningModeConfig TuningModeConfig::fromEnvironment(bool isPrivileged)
    {
        TuningModeConfig config;

        if(const char* env = rocblaslt_secure_getenv_impl("HIPBLASLT_TUNING_MODE", isPrivileged))
        {
            const std::string value(env);
            if(value == "cache")
                config.mode = TuningMode::Cache;
            else if(value == "tune")
                config.mode = TuningMode::Tune;
        }

        if(const char* path
           = rocblaslt_secure_getenv_impl("HIPBLASLT_TUNING_CACHE_PATH", isPrivileged))
            config.cachePath = path;

        config.suppressedForSecurity
            = rocblaslt_env_suppressed_for_security_impl("HIPBLASLT_TUNING_MODE", isPrivileged)
              || rocblaslt_env_suppressed_for_security_impl("HIPBLASLT_TUNING_CACHE_PATH",
                                                            isPrivileged);

        return config;
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
        bool isTransposeCell(const std::string& value)
        {
            return value == "N" || value == "T" || value == "C";
        }

        // Every column of a current row is mandatory and strict, the search and
        // completion columns included: those decide whether tune mode revisits
        // the row, so a row that lost them is not trustworthy either.
        bool hasRequiredCurrentColumns(const std::map<std::string, std::string>& row)
        {
            if(!isTransposeCell(str(row, "transA")) || !isTransposeCell(str(row, "transB")))
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
                                                      "lde",
                                                      "stride_e",
                                                      "required_workspace",
                                                      "budget_ms",
                                                      "search_workspace"};
            for(const char* name : nonNegative)
            {
                const auto value = exactNum(row, name);
                if(!value || *value < 0)
                    return false;
            }
            if(*exactNum(row, "batch_count") == 0)
                return false;

            static const char* const nonNegativeInt32[]
                = {"search_max_candidates", "cold_iters", "hot_iters", "rotating_mb"};
            for(const char* name : nonNegativeInt32)
                if(!isInt32(row, name) || *exactNum(row, name) < 0)
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
                                                     "swizzle_b",
                                                     "uniform_summation_order",
                                                     "complete",
                                                     "search_all_kernels",
                                                     "flush_icache"};
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

        po.transA     = str(row, "transA", "N") != "N";
        po.transB     = str(row, "transB", "N") != "N";
        po.conjugateA = str(row, "transA") == "C";
        po.conjugateB = str(row, "transB") == "C";

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
        po.colStrideE   = static_cast<size_t>(num(row, "lde"));
        po.batchStrideE = static_cast<size_t>(num(row, "stride_e"));
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
        po.uniformSummationOrder = flag(row, "uniform_summation_order");

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

        if(schemaVersion == TuningSchemaVersion::Current)
        {
            // Validated above, so these are present and well-formed.
            entry.complete = *exactNum(row, "complete") != 0;
            entry.budgetMs = *exactNum(row, "budget_ms");

            TuningSearch search;
            search.allKernels     = *exactNum(row, "search_all_kernels") != 0;
            search.maxCandidates  = static_cast<int32_t>(*exactNum(row, "search_max_candidates"));
            search.workspaceBytes = static_cast<size_t>(*exactNum(row, "search_workspace"));
            search.coldIters      = static_cast<int32_t>(*exactNum(row, "cold_iters"));
            search.hotIters       = static_cast<int32_t>(*exactNum(row, "hot_iters"));
            search.flushICache    = *exactNum(row, "flush_icache") != 0;
            search.rotatingMb     = static_cast<int32_t>(*exactNum(row, "rotating_mb"));
            entry.search          = search;
        }
        else
        {
            // Legacy rows come from hipblaslt-bench or a hand-written override
            // file, which record no search at all, so they read as complete with
            // an unknown ceiling.
            entry.complete = true;
            entry.budgetMs = -1;
        }

        return std::make_pair(po, entry);
    }

    std::string formatTuningRow(const ProblemOverride& key,
                                const TuningRowTypes&  types,
                                const TunedEntry&      entry,
                                const std::string&     buildStamp)
    {
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

        column("schema_version", static_cast<uint32_t>(TuningSchemaVersion::Current));
        column("git_version", buildStamp);

        auto transposeCell = [](bool transposed, bool conjugated) {
            return !transposed ? "N" : conjugated ? "C" : "T";
        };
        column("transA", transposeCell(key.transA, key.conjugateA));
        column("transB", transposeCell(key.transB, key.conjugateB));
        column("m", key.m);
        column("n", key.n);
        column("k", key.k);
        column("batch_count", key.batchSize);

        column("a_type", types.a);
        column("b_type", types.b);
        column("c_type", types.c);
        column("d_type", types.d);
        column("compute_type", types.compute);
        column("compute_input_type_a", key.computeInputTypeA);
        column("compute_input_type_b", key.computeInputTypeB);

        column("lda", key.colStrideA);
        column("ldb", key.colStrideB);
        column("ldc", key.colStrideC);
        column("ldd", key.colStrideD);
        column("stride_a", key.batchStrideA);
        column("stride_b", key.batchStrideB);
        column("stride_c", key.batchStrideC);
        column("stride_d", key.batchStrideD);
        column("lde", key.colStrideE);
        column("stride_e", key.batchStrideE);
        column("batch_mode", key.batchMode);

        column("epilogue", key.epilogue);
        column("gradient", key.gradient ? 1 : 0);
        column("bias_type", key.biasType);
        column("bias_stride", key.biasStride);
        column("bias_vector", key.hasBias ? 1 : 0);
        column("aux_type", key.auxType);

        column("scaleA_format", key.scaleAFormat);
        column("scaleB_format", key.scaleBFormat);
        column("scaleA", key.hasScaleA ? 1 : 0);
        column("scaleB", key.hasScaleB ? 1 : 0);
        column("scaleC", key.hasScaleC ? 1 : 0);
        column("scaleD", key.hasScaleD ? 1 : 0);
        column("scaleE", key.hasScaleE ? 1 : 0);
        column("scaleAlphaVec", key.hasScaleAlphaVec ? 1 : 0);
        column("amaxD", key.hasAmaxD ? 1 : 0);

        column("swizzle_a", key.swizzleA ? 1 : 0);
        column("swizzle_b", key.swizzleB ? 1 : 0);
        column("streamk_tile_scheduling", key.streamkTileScheduling);
        column("sm_count_target", key.smCountTarget);
        column("uniform_summation_order", key.uniformSummationOrder ? 1 : 0);

        column("gcnArchName", key.archName);
        column("CUs", key.cuCount);

        column("solution_index", entry.solutionIndex);
        column("kernel_name", entry.kernelName.value_or(std::string{}));
        column("required_workspace", entry.requiredWorkspaceBytes);
        column("us", entry.winnerTimeUs);

        // Every row emits its own header line, so a reader that does not know
        // one of these columns simply never looks for it.
        column("complete", entry.complete ? 1 : 0);
        column("budget_ms", entry.budgetMs);

        const TuningSearch search = entry.search.value_or(TuningSearch{});
        column("search_all_kernels", search.allKernels ? 1 : 0);
        column("search_max_candidates", search.maxCandidates);
        column("search_workspace", search.workspaceBytes);
        column("cold_iters", search.coldIters);
        column("hot_iters", search.hotIters);
        column("flush_icache", search.flushICache ? 1 : 0);
        column("rotating_mb", search.rotatingMb);

        return "    " + names.str() + "\n" + values.str() + "\n";
    }

    bool appendTuningRow(const std::string& path, const std::string& row, const std::string& buildStamp)
    {
        if(path.empty())
            return false;

        static std::mutex           writeMutex;
        std::lock_guard<std::mutex> lock(writeMutex);

        const bool needHeader = [&] {
            std::ifstream probe(path);
            return !probe || probe.peek() == std::ifstream::traits_type::eof();
        }();

        std::ofstream out(path, std::ios::app);
        if(!out)
            return false;

        // One write for the header and the row together, rather than a flush
        // apiece. The process mutex above orders writers inside this process,
        // but nothing orders a second process appending to the same file, and
        // there the unit of tearing is the write: a single one that the
        // filesystem appends atomically cannot be split down the middle by
        // another rank, whereas two can interleave. This narrows that window on
        // a local filesystem; it is not a guarantee, least of all on the network
        // filesystems multi-node jobs use, which is why concurrent writers stay
        // unsupported.
        std::string bytes;
        if(needHeader)
            bytes = kGitVersionHeader + buildStamp + "\n";
        bytes += row;

        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();

        // Reported rather than swallowed: a full disk or a read-only path
        // otherwise looks like a successful tune whose results silently vanish
        // at process exit.
        return out.good();
    }

    TuningRowsLoaded loadTuningRows(std::istream&      in,
                                    OverrideMap&       map,
                                    TuningEntrySource  source,
                                    const std::string& currentBuildStamp)
    {
        TuningRowsLoaded loaded;

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
            else if(!std::getline(in, line))
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
            if(!std::getline(in, valueLine))
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

            const auto names  = splitCsv(header);
            const auto values = splitCsv(value);

            // A current-schema row carries exactly one value per column. zipRow
            // pairs only up to the shorter of the two, so a value row that a crash
            // cut short would otherwise parse as whatever prefix it kept, and with
            // its trailing completion cells missing that could read as final.
            if(names.size() != values.size()
               && std::find(names.begin(), names.end(), "schema_version") != names.end())
                continue;

            auto parsed = problemFromEntries(zipRow(names, values));
            if(!parsed)
                continue;

            auto& [key, entry] = *parsed;

            entry.source = source;

            if(entry.buildStamp.empty())
                entry.buildStamp = fileBuildStamp;

            // A row carrying a name validates itself at replay: the recorded
            // index is resolved in the current library and the name must still
            // match. A row without one has no such check, so the build stamp is
            // the only trust signal it has.
            if(!entry.kernelName.has_value() && !entry.solutionName.has_value()
               && entry.buildStamp != currentBuildStamp)
                continue;

            // v0 rows go into the narrow map. They cannot populate the widened
            // key, so filing them under it would mean they match nothing, which
            // is how existing hipblaslt-bench files stopped replaying.
            const bool inserted = (entry.schemaVersion == TuningSchemaVersion::Legacy)
                                      ? map.addLegacyIfAbsent(key, entry)
                                      : map.addIfAbsent(key, entry);
            if(inserted)
                loaded.accepted++;
        }

        // Only a clean read counts as loaded. A read that stopped on badbit,
        // such as an I/O error partway through, would otherwise latch a
        // partially populated map as complete and never be retried.
        loaded.readError = in.bad();
        return loaded;
    }
} // namespace TensileLite
