// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/DeclaredOracle.hpp>
#include <hipdnn_corpus_gen/GraphSize.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/PoolAssembly.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

/// @file KernelCatalogSource.hpp
/// @brief The `kernel` pool: the geometries a descriptor pack actually carries.
/// Only geometries over `maxBytes` are dropped; kernels per geometry is reported, not filtered,
/// because pack density is not the matcher's runtime candidate count.
namespace hipdnn_corpus_gen
{

/// What one pack contributed. Counts are geometries (kernels folded by shape), not kernels.
struct PackReport
{
    std::string pack;

    /// Descriptors read, before folding onto geometries.
    int64_t kernels = 0;
    int64_t geometries = 0;

    /// Kernels claiming the densest geometry. Reported only; never used to drop geometries.
    int64_t maxCandidates = 0;
    bool deterministic = false;

    /// Descriptors missing a mapped field; all of them fold into a single geometry.
    int64_t noGeometry = 0;

    /// Geometries with a value the declaration does not map, e.g. a dtype outside
    /// @ref KernelCatalog::enums.
    int64_t unmappedValue = 0;

    /// Geometries the declaration could not build (e.g. causal cross attention, whose FLOP
    /// count is non-positive).
    int64_t unbuildable = 0;
    std::string firstBuildError;

    /// Geometries whose graph exceeds the benchmarking ceiling.
    int64_t overByteBudget = 0;

    int64_t eligible = 0;

    /// Why the pack contributed nothing; empty if it contributed or no operator action helps.
    std::string shutOut;

    nlohmann::json asJson() const
    {
        nlohmann::json report{{"pack", pack},
                              {"kernels", kernels},
                              {"geometries", geometries},
                              {"max_candidates", maxCandidates},
                              {"deterministic", deterministic},
                              {"no_geometry", noGeometry},
                              {"unmapped_value", unmappedValue},
                              {"unbuildable", unbuildable},
                              {"over_byte_budget", overByteBudget},
                              {"eligible", eligible}};
        if(!firstBuildError.empty())
        {
            report["first_build_error"] = firstBuildError;
        }
        if(!shutOut.empty())
        {
            report["shut_out"] = shutOut;
        }
        return report;
    }
};

struct PackHarvest
{
    std::vector<PoolEntry> entries;
    PackReport report;
};

namespace detail
{

/// @brief Reads one descriptor's @p fields as a point, or nullopt if it yields none.
/// @p unmapped is set when a field is present but its value is not mapped (vs. absent).
inline std::optional<ProblemPoint> pointFromDescriptor(const OperationMetadata& metadata,
                                                       const nlohmann::json& fields,
                                                       bool& unmapped)
{
    unmapped = false;
    ProblemPoint point;

    for(const auto& mapped : metadata.kernelCatalog.fields)
    {
        const auto found = fields.find(mapped.second);
        if(found == fields.end() || found->is_null())
        {
            return std::nullopt;
        }

        const auto* parameter = metadata.find(mapped.first);
        if(parameter == nullptr)
        {
            return std::nullopt;
        }

        switch(parameter->type)
        {
        // The build rejects a switch without default; a new ParameterType with no arm is
        // counted as unmapped rather than misread.
        default:
            unmapped = true;
            return std::nullopt;

        case ParameterType::INT64:
            if(!found->is_number_integer() && !found->is_number_unsigned())
            {
                unmapped = true;
                return std::nullopt;
            }
            point[mapped.first] = found->get<int64_t>();
            break;

        case ParameterType::FLOAT64:
            if(!found->is_number())
            {
                unmapped = true;
                return std::nullopt;
            }
            point[mapped.first] = found->get<double>();
            break;

        case ParameterType::BOOL:
            // Packs spell a flag as both `true` and `1`.
            if(found->is_boolean())
            {
                point[mapped.first] = found->get<bool>();
            }
            else if(found->is_number_integer() || found->is_number_unsigned())
            {
                point[mapped.first] = found->get<int64_t>() != 0;
            }
            else
            {
                unmapped = true;
                return std::nullopt;
            }
            break;

        case ParameterType::ENUM:
        {
            if(!found->is_string())
            {
                unmapped = true;
                return std::nullopt;
            }
            const auto spelling = found->get<std::string>();

            const auto table = metadata.kernelCatalog.enums.find(mapped.first);
            if(table != metadata.kernelCatalog.enums.end())
            {
                const auto translated = table->second.find(spelling);
                if(translated == table->second.end())
                {
                    unmapped = true;
                    return std::nullopt;
                }
                point[mapped.first] = translated->second;
                break;
            }

            // No translation table: the value must already be one the declaration lists.
            if(std::find(parameter->values.begin(), parameter->values.end(), spelling)
               == parameter->values.end())
            {
                unmapped = true;
                return std::nullopt;
            }
            point[mapped.first] = spelling;
            break;
        }
        }
    }

    // Cannot overwrite a descriptor's value: the parser rejects constants that collide with a
    // mapping.
    for(const auto& constant : metadata.kernelCatalog.constants)
    {
        point[constant.first] = constant.second;
    }
    return point;
}

/// Stable ordering key, so a pack read twice yields the same corpus.
inline std::string geometryKey(const ProblemPoint& point)
{
    return describe(point);
}

} // namespace detail

/// @brief Every `*.kdp.json` under @p roots, in a stable order. A root naming a file is taken
/// as that file.
inline std::vector<std::filesystem::path>
    discoverPacks(const std::vector<std::filesystem::path>& roots)
{
    std::set<std::filesystem::path> found;
    std::error_code ignored;

    for(const auto& root : roots)
    {
        if(std::filesystem::is_regular_file(root, ignored))
        {
            found.insert(std::filesystem::absolute(root, ignored).lexically_normal());
        }
        else if(std::filesystem::is_directory(root, ignored))
        {
            for(const auto& entry : std::filesystem::recursive_directory_iterator(root, ignored))
            {
                const auto& path = entry.path();
                if(path.extension() == ".json" && path.stem().extension() == ".kdp")
                {
                    found.insert(std::filesystem::absolute(path, ignored).lexically_normal());
                }
            }
        }
    }
    return {found.begin(), found.end()};
}

/// @brief Why @p report's pack contributed nothing despite having geometries. Empty when it
/// contributed, or when no operator action would help (e.g. descriptors with no shape).
inline std::string shutOut(const PackReport& report, const std::string& operation)
{
    if(report.eligible > 0 || report.geometries == 0)
    {
        return {};
    }

    const auto all = std::to_string(report.geometries);
    if(report.noGeometry == report.geometries)
    {
        return {};
    }
    if(report.unmappedValue == report.geometries)
    {
        return "all " + all + " of its geometries name values '" + operation
               + "' does not map; see its `kernel_catalog.enums` block for the ones it does.";
    }
    if(report.overByteBudget == report.geometries)
    {
        return "all " + all
               + " of its geometries exceed --max-bytes and so cannot be measured; raise it to "
                 "admit them.";
    }
    if(report.unbuildable == report.geometries)
    {
        return "none of its " + all + " geometries builds a graph: " + report.firstBuildError;
    }
    return {};
}

/// @brief The geometries one pack carries, minus those that cannot be measured. Empty for an
/// operation that declares no @ref KernelCatalog.
inline PackHarvest fromPack(const OperationMetadata& metadata,
                            const std::filesystem::path& path,
                            int64_t maxBytes = 0)
{
    PackHarvest harvest;
    harvest.report.pack = path.string();
    if(metadata.kernelCatalog.empty())
    {
        return harvest;
    }

    nlohmann::json pack;
    {
        std::ifstream file(path);
        if(!file.is_open())
        {
            harvest.report.shutOut = "could not be opened.";
            return harvest;
        }
        try
        {
            file >> pack;
        }
        catch(const std::exception& error)
        {
            harvest.report.shutOut = "could not be read: " + std::string(error.what());
            return harvest;
        }
    }

    /// One geometry and the kernels claiming it, keyed on the parsed point so `causal: 1` and
    /// `causal: true` fold together.
    struct Bucket
    {
        std::optional<ProblemPoint> point;
        bool unmapped = false;
        int64_t kernels = 0;
    };
    std::map<std::string, Bucket> buckets;
    Bucket shapeless;

    const auto descriptors = pack.value("kernelDescriptors", nlohmann::json::array());
    harvest.report.kernels = static_cast<int64_t>(descriptors.size());
    for(const auto& descriptor : descriptors)
    {
        const auto fields = descriptor.value("metadata", nlohmann::json::object());

        bool unmapped = false;
        auto point = detail::pointFromDescriptor(metadata, fields, unmapped);
        if(!point.has_value() && !unmapped)
        {
            ++shapeless.kernels;
            continue;
        }

        // An unmapped descriptor still counts as a geometry, keyed on its raw fields.
        auto& bucket = buckets[point.has_value() ? detail::geometryKey(*point)
                                                 : "unmapped:" + fields.dump()];
        bucket.point = point;
        bucket.unmapped = unmapped;
        ++bucket.kernels;
    }

    auto& report = harvest.report;
    report.geometries = static_cast<int64_t>(buckets.size()) + (shapeless.kernels > 0 ? 1 : 0);
    report.noGeometry = shapeless.kernels > 0 ? 1 : 0;
    for(const auto& bucket : buckets)
    {
        report.maxCandidates = std::max(report.maxCandidates, bucket.second.kernels);
    }
    report.maxCandidates = std::max(report.maxCandidates, shapeless.kernels);
    report.deterministic = !buckets.empty() && report.maxCandidates <= 1;

    const BuildTally tally{&report.unbuildable, &report.firstBuildError};
    for(const auto& bucket : buckets)
    {
        if(bucket.second.unmapped || !bucket.second.point.has_value())
        {
            ++report.unmappedValue;
            continue;
        }

        const auto& point = *bucket.second.point;
        const auto built = buildAdmissible(metadata, point, 0, tally);
        if(!built.has_value())
        {
            continue;
        }
        if(maxBytes > 0 && graphBytes(*built) > maxBytes)
        {
            ++report.overByteBudget;
            continue;
        }

        ++report.eligible;
        PoolEntry entry;
        entry.point = point;
        entry.source = "kernel";
        entry.origin
            = path.filename().string() + ":" + std::to_string(bucket.second.kernels) + " kernels";
        entry.regime = regimeLabel(metadata, point);
        harvest.entries.push_back(std::move(entry));
    }

    report.shutOut = shutOut(report, metadata.operation);
    return harvest;
}

/// @brief Every pack's eligible geometries, ordered by regime (@ref spread) so any prefix cut
/// to a budget is not just the first pack's listings.
inline std::pair<std::vector<PoolEntry>, std::vector<PackReport>>
    collectPacks(const OperationMetadata& metadata,
                 const std::vector<std::filesystem::path>& paths,
                 int64_t maxBytes = 0)
{
    std::vector<PoolEntry> entries;
    std::vector<PackReport> reports;

    for(const auto& path : paths)
    {
        auto harvest = fromPack(metadata, path, maxBytes);
        entries.insert(entries.end(),
                       std::make_move_iterator(harvest.entries.begin()),
                       std::make_move_iterator(harvest.entries.end()));
        reports.push_back(std::move(harvest.report));
    }
    return {detail::spread(entries), std::move(reports)};
}

} // namespace hipdnn_corpus_gen
