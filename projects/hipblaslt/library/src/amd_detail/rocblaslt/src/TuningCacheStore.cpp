// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TuningCacheStore.hpp"
#include "hipblaslt_type_strings.hpp"
#include "rocblaslt_secure_env.hpp"
#include "tensile_type_map.hpp"

#include <algorithm>
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
            if(std::string(env) == "cache")
                config.mode = TuningMode::Cache;
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
        const char* const kGitVersionHeader = "Git Version:";

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

        // A row header names at least the columns an entry is keyed and resolved
        // on. Value rows never do, and anything else in the file is ignored.
        bool isHeaderRow(const std::string& line)
        {
            const auto cells = splitCsv(line);
            return std::find(cells.begin(), cells.end(), "transA") != cells.end()
                   && std::find(cells.begin(), cells.end(), "solution_index") != cells.end();
        }

        // Pairs cells by column name rather than by position, so a column can be
        // added to the format without breaking readers that do not know it.
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

        // Absent or unparseable cells take the caller's default. That is only
        // safe where the default narrows what an entry matches; current rows
        // validate every cell up front instead, see hasRequiredCurrentColumns.
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

            // hipDataType_to_tensile_type asserts on types it does not support,
            // and string_to_hip_datatype also recognises scale-only types such
            // as e8_r and e5m3_r, which are not GEMM tensor types.
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
         * num() folds "absent", "empty" and "not a number" into the caller's
         * default, which would let an empty lda validate as the plausible
         * leading dimension 0. splitCsv has already trimmed the cell, so any
         * leftover character means it is not a number.
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

        bool isTransposeCell(const std::string& value)
        {
            return value == "N" || value == "T" || value == "C";
        }

        // Every column of a current row is mandatory and strict. Defaults would
        // turn a damaged row into a well-formed row for some other problem.
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
                                                     "swizzle_b",
                                                     "uniform_summation_order"};
            for(const char* name : boolFields)
            {
                const auto value = exactNum(row, name);
                if(!value || (*value != 0 && *value != 1))
                    return false;
            }

            // A current row is always validated by name at replay.
            return (has(row, "kernel_name") && !str(row, "kernel_name").empty())
                   || (has(row, "solution_name") && !str(row, "solution_name").empty());
        }
    } // namespace

    std::optional<std::pair<ProblemOverride, TunedEntry>>
        problemFromEntries(const std::map<std::string, std::string>& row)
    {
        // The schema is checked before any other cell is interpreted. A newer
        // schema can use datatype spellings this build does not know, and the
        // Tensile type converters assert on those.
        TuningSchemaVersion schemaVersion = TuningSchemaVersion::Legacy;
        if(has(row, "schema_version"))
        {
            const auto version = exactNum(row, "schema_version");
            if(!version || *version != static_cast<int64_t>(TuningSchemaVersion::Current)
               || !hasRequiredCurrentColumns(row))
                return std::nullopt;
            schemaVersion = TuningSchemaVersion::Current;
        }

        // Index 0 is a real solution in the shipped logic; only a negative
        // index is meaningless.
        const auto solutionIndexValue = exactNum(row, "solution_index");
        if(!solutionIndexValue || *solutionIndexValue < 0
           || *solutionIndexValue > std::numeric_limits<int32_t>::max())
            return std::nullopt;

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

        // Legacy rows record only c_type, and describe problems where d
        // matches it.
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

        // The full gcnArchName, sramecc and xnack included, since those can
        // change which kernels apply.
        po.archName = str(row, "gcnArchName");
        po.cuCount  = static_cast<int32_t>(num(row, "CUs"));

        TunedEntry entry;
        entry.solutionIndex = static_cast<int32_t>(*solutionIndexValue);

        if(has(row, "kernel_name") && !str(row, "kernel_name").empty())
            entry.kernelName = str(row, "kernel_name");
        if(has(row, "solution_name") && !str(row, "solution_name").empty())
            entry.solutionName = str(row, "solution_name");

        entry.schemaVersion = schemaVersion;
        entry.buildStamp    = str(row, "git_version");
        entry.requiredWorkspaceBytes
            = static_cast<size_t>(std::max<int64_t>(0, num(row, "required_workspace")));
        entry.winnerTimeUs = real(row, "us");

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

        return "    " + names.str() + "\n" + values.str() + "\n";
    }

    bool appendTuningRow(const std::string& path,
                         const std::string& row,
                         const std::string& buildStamp)
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

        // One write for the header and the row together: another process
        // appending to the same file can split two writes apart, while a
        // single write to a local filesystem usually lands whole.
        std::string bytes;
        if(needHeader)
            bytes = std::string(kGitVersionHeader) + " " + buildStamp + "\n";
        bytes += row;

        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        return out.good();
    }

    TuningRowsLoaded
        loadTuningRows(std::istream& in, OverrideMap& map, const std::string& currentBuildStamp)
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
                    fileBuildStamp = trimmed(header.substr(pos + std::strlen(kGitVersionHeader)));
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
                // An interrupted append can leave a header with nothing under it.
                // Hand the next header back to the loop rather than reading it as
                // this row's values, so the row that follows it survives.
                pendingHeader = value;
                continue;
            }

            const auto names  = splitCsv(header);
            const auto values = splitCsv(value);

            // A current row has exactly one value per column. zipRow pairs only
            // up to the shorter list, so a value line cut short by an
            // interrupted append would otherwise read as whatever prefix it
            // kept, a kernel name cut partway included.
            if(names.size() != values.size()
               && std::find(names.begin(), names.end(), "schema_version") != names.end())
                continue;

            auto parsed = problemFromEntries(zipRow(names, values));
            if(!parsed)
                continue;

            auto& [key, entry] = *parsed;

            if(entry.buildStamp.empty())
                entry.buildStamp = fileBuildStamp;

            // A row that records a name is checked at replay, where its index is
            // resolved in the running library and the name must still match. A
            // row without one has nothing to check it against, so it is trusted
            // only when it was written by this build, which a build with no
            // stamp of its own cannot establish.
            if(!entry.kernelName && !entry.solutionName
               && (currentBuildStamp.empty() || entry.buildStamp != currentBuildStamp))
            {
                loaded.skippedUnnamed++;
                continue;
            }

            const bool inserted = entry.schemaVersion == TuningSchemaVersion::Legacy
                                      ? map.addLegacyIfAbsent(key, entry)
                                      : map.addIfAbsent(key, entry);
            if(inserted)
                loaded.accepted++;
        }

        loaded.fileBuildStamp = fileBuildStamp;
        loaded.readError      = in.bad();
        return loaded;
    }
} // namespace TensileLite
