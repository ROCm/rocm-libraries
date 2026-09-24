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
///
/// A descriptor engine's kernels bake their geometry in, so its matcher pins the shape fields
/// before ranking starts. That makes the pack the authority on the one thing nothing else can
/// answer offline: which problems the engine will have a candidate for at all. The declaration
/// sweep can propose any point the operation can express; only the pack knows which of them
/// this engine was compiled for.
///
/// Packs are read structurally, not by file name -- a descriptor contributes a geometry if its
/// metadata carries every mapped field -- so a pack landing anywhere under the given roots is
/// picked up, and one carrying no geometry at all is read and contributes nothing, which the
/// report records rather than hides.
///
/// **What is deliberately not asked here** is which of those geometries are worth *ranking*.
/// That used to be asked, through a floor on how many kernels claimed a geometry, and it was
/// wrong twice over: pack density is an upper bound on what the matcher offers at runtime and
/// never a count of it -- a pack may carry two block_m variants the matcher collapses to one
/// candidate -- and no single floor fits a pack that runs one to six kernels deep across its
/// geometries. Such a floor both drops problems the engine serves and admits problems L2 will
/// exclude anyway. The question is answered downstream where it is cheap and exact, from the
/// candidates that were actually timed. Those measurements are not wasted when a catalog turns
/// out to be deterministic -- they are exactly the labels `predict_engine_tflops` needs, which
/// is the whole reason a deterministic engine gets a corpus.
///
/// `maxBytes` remains, because it is a different kind of rule: a geometry whose tensors do not
/// fit cannot be measured at all, and a corpus entry that cannot be measured is a hole in the
/// training set rather than a member of it.
namespace hipdnn_corpus_gen
{

/// What one pack contributed, and what it lost on the way.
///
/// Every count here is a geometry, not a kernel: the pack's descriptors are folded onto their
/// geometry first, because two kernels compiled for the same shape are one problem to measure.
struct PackReport
{
    std::string pack;

    /// Descriptors read, before folding onto geometries.
    int64_t kernels = 0;
    int64_t geometries = 0;

    /// Kernels claiming the densest geometry, and whether any geometry has a choice at all.
    ///
    /// Reported, never enforced. A pack whose every geometry is claimed by one kernel is a sign
    /// that `sort_kernel_catalog` will have nothing to rank, and an operator is better off
    /// knowing that here than eight hours into a sweep -- but it is only a sign, for the reason
    /// the file comment gives, so it is written to the manifest and no geometry is dropped.
    int64_t maxCandidates = 0;
    bool deterministic = false;

    /// Descriptors missing one or more mapped fields. Folded onto a single geometry, exactly as
    /// a real geometry is, because "the descriptors with no shape" is one population.
    int64_t noGeometry = 0;

    /// Geometries naming a value the declaration does not map -- a dtype outside
    /// @ref KernelCatalog::enums, most often. Skipped and counted, never guessed.
    int64_t unmappedValue = 0;

    /// Geometries the declaration could not build. A pack may carry a geometry no corpus can
    /// label: causal cross attention, whose declared FLOP count is non-positive. The kernel
    /// exists and the engine will run it; there is simply no training row to be made of it.
    int64_t unbuildable = 0;
    std::string firstBuildError;

    /// Geometries whose graph exceeds the benchmarking ceiling.
    int64_t overByteBudget = 0;

    int64_t eligible = 0;

    /// Why this pack contributed nothing, when it had geometries to contribute. Empty when it
    /// contributed something, or when nothing an operator can do would change the outcome.
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

/// One pack's contribution.
struct PackHarvest
{
    std::vector<PoolEntry> entries;
    PackReport report;
};

namespace detail
{

/// @brief One descriptor's metadata read through @p catalog, or nullopt if it carries no
/// geometry.
///
/// Absent is not the same as unmapped: a descriptor with no `seqlen_q` field at all is one the
/// pack simply does not describe shapes in, while a `dtype` of `FP8` is a shape this tool was
/// not taught to build. The first is nobody's fault and the second is a gap in the declaration,
/// so they are counted apart. @p unmapped distinguishes them on return.
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
        // Unreachable while every ParameterType has an arm below, and kept because the build
        // treats a switch with no default as an error. A type added without an arm here drops
        // the geometry and counts it, rather than reading the field as whatever it happens to
        // be.
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
            // Packs spell a flag both ways -- `true` and `1` -- and which one a kernel author
            // wrote is not a fact about the problem.
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

            // No translation table: the pack must already be speaking the declaration's
            // vocabulary, and a value outside it is unmapped rather than passed through.
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

    // Applied after the pack's own fields, and only for parameters it maps none of -- the
    // parser refuses a constant that collides with a mapping, so this cannot overwrite
    // anything a descriptor actually said.
    for(const auto& constant : metadata.kernelCatalog.constants)
    {
        point[constant.first] = constant.second;
    }
    return point;
}

/// A stable ordering key for a geometry, so a pack read twice yields the same corpus.
inline std::string geometryKey(const ProblemPoint& point)
{
    return describe(point);
}

} // namespace detail

/// @brief Every `*.kdp.json` under @p roots, in a stable order.
///
/// A path naming a file is taken as that file, so one pack can be pointed at directly without
/// arranging a directory around it.
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

/// @brief Why @p report's pack contributed nothing, when it had geometries to contribute.
///
/// A pack that contributes nothing is invisible in the corpus that results -- the graphs simply
/// come from the other sources -- and the operator finds out an eight-hour sweep later, when
/// the held-out slice has nothing the engine serves. Saying it here costs one line and saves
/// that.
///
/// Only causes an operator can act on are named. A pack whose descriptors carry no shape at all
/// is silent: there is no flag that admits it, and naming one that would not help is worse than
/// saying nothing.
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

/// @brief The geometries one pack carries, with the ones that cannot be measured dropped.
///
/// An operation declaring no @ref KernelCatalog has no kernel pool and reads no pack -- that is
/// how coverage stays "whatever has a declaration", with no operation list to keep in step.
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

    /// One geometry and the kernels claiming it. Keyed on the point rather than on the raw
    /// metadata, so two descriptors spelling the same shape differently -- `causal: 1` and
    /// `causal: true` -- are one problem to measure rather than two.
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
            // The descriptors with no shape are one population, not one each.
            ++shapeless.kernels;
            continue;
        }

        // An unmapped descriptor still occupies a geometry -- it is a shape the pack carries
        // and this tool cannot build -- so it is counted as one and keyed on what it did read.
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

/// @brief Every pack's eligible geometries, ordered so that any prefix stays spread out.
///
/// @ref spread does the ordering, by the regime each geometry falls in. A pool left in pack
/// order and then cut to a budget is a corpus of whatever the first pack happened to list --
/// and packs are written one arch, one dtype, one head size at a time, so that prefix is not a
/// sample of anything.
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
