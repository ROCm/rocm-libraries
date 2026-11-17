// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvCompileCensus.cpp
 * @brief Opt-in census of the conv engines' runtime compiles, for the code-size gate
 *        (conv/tools/code_gate).
 *
 * Every conv catalog of a descriptor source tree (HKP_CONV_CENSUS_TREE, the descriptors/conv
 * folder) is matched, on each architecture of HKP_CONV_CENSUS_ARCHS (a device described by its
 * name, wave size and compute units; no GPU is used), against the graphs of a coverage map
 * (HKP_CONV_CENSUS_MAP, hip_kernel_provider_conv_coverage_map.json), each row in its own
 * direction, dtype, layout and memory form, and of a sweep along what the configs bake (filter
 * geometry, stride, group width, dtype, layout, direction; HKP_CONV_CENSUS_SWEEP=0 leaves it
 * out); each forward graph also with a fused bias and clipped ReLU. Every config a graph admits is prepared through its pack's dispatch, which
 * builds the config's compile options; at most HKP_CONV_CENSUS_PER_CONFIG graphs a config and
 * architecture (default 8). Run it under conv/tools/rdna_generic/rtc_hook.c with HKP_RTC_DRY=1
 * and HKP_RTC_LOG=<file>, so each distinct compile is logged and nothing is compiled or loaded,
 * and with HIPDNN_DISABLE_CACHE=1.
 *
 * HKP_CONV_CENSUS_OUT receives one JSON line per catalog config and architecture its pack
 * serves: the engine, the catalog, the config, how many graphs admitted it, how many prepares
 * succeeded, and the first failure. A config no graph admits is listed with zero admissions, so
 * the gate reports what the census did not reach.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

std::string envOr(const char* name, const std::string& fallback = {})
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::string(value) : fallback;
}

nlohmann::json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    return nlohmann::json::parse(stream);
}

/// One catalog file: its engine's graph match, its pack's kernel match and dispatch, the
/// architectures it is packed for (empty: every one), and its configs.
struct Catalog
{
    std::string file;
    std::string engine;
    ConvDirection direction = ConvDirection::FORWARD;
    std::string graphMatch;
    std::string kernelMatch;
    std::string dispatch;
    std::vector<std::string> arch;
    std::vector<KernelDefinition> kernels;
};

std::vector<Catalog> loadCatalogs(const std::filesystem::path& tree)
{
    std::map<std::string, std::pair<std::string, std::string>> engines; // id: name, graph match
    std::map<std::string, std::string> kernelMatches; // id: symbol
    std::map<std::string, std::string> dispatches; // id: symbol
    std::vector<std::filesystem::path> catalogFiles;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(tree))
    {
        const auto name = entry.path().filename().string();
        const auto endsWith = [&](const std::string& suffix) {
            return name.size() > suffix.size()
                   && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        if(endsWith(".ued.json"))
        {
            const auto json = readJson(entry.path());
            engines[json.at("id").get<std::string>()]
                = {json.at("name").get<std::string>(),
                   json.at("graph_match").at("native").get<std::string>()};
        }
        else if(endsWith(".umd.json"))
        {
            const auto json = readJson(entry.path());
            kernelMatches[json.at("id").get<std::string>()]
                = json.at("match_symbol").get<std::string>();
        }
        else if(endsWith(".udd.json"))
        {
            const auto json = readJson(entry.path());
            dispatches[json.at("id").get<std::string>()]
                = json.at("dispatch_symbol").get<std::string>();
        }
        else if(endsWith(".kdp.json"))
        {
            catalogFiles.push_back(entry.path());
        }
    }
    std::sort(catalogFiles.begin(), catalogFiles.end());

    std::vector<Catalog> catalogs;
    for(const auto& path : catalogFiles)
    {
        const auto json = readJson(path);
        Catalog catalog;
        catalog.file = path.lexically_relative(tree).generic_string();
        const auto& engine = engines.at(json.at("engine").get<std::string>());
        catalog.engine = engine.first;
        catalog.graphMatch = engine.second;
        if(catalog.engine.find("_BWD") != std::string::npos)
        {
            catalog.direction = ConvDirection::BACKWARD_DATA;
        }
        else if(catalog.engine.find("_WRW") != std::string::npos)
        {
            catalog.direction = ConvDirection::BACKWARD_WEIGHTS;
        }
        for(const auto& matcher : json.at("matchers"))
        {
            const auto found = kernelMatches.find(matcher.get<std::string>());
            if(found != kernelMatches.end())
            {
                catalog.kernelMatch = found->second;
            }
        }
        catalog.dispatch = dispatches.at(json.at("dispatch").get<std::string>());
        catalog.arch = json.at("arch").get<std::vector<std::string>>();
        for(const auto& descriptor : json.at("kernelDescriptors"))
        {
            KernelDefinition kernel;
            kernel.name = descriptor.at("name").get<std::string>();
            kernel.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
            kernel.source.sourceFile
                = descriptor.at("kernel_source").at("source_file").get<std::string>();
            kernel.source.entryPoint
                = descriptor.at("kernel_source").at("entry_point").get<std::string>();
            for(const auto& [key, value] : descriptor.at("metadata").items())
            {
                if(value.is_string())
                {
                    kernel.metadata[key] = value.get<std::string>();
                }
                else
                {
                    kernel.metadata[key] = value.get<int64_t>();
                }
            }
            kernel.priority = descriptor.at("priority").get<int64_t>();
            catalog.kernels.push_back(std::move(kernel));
        }
        catalogs.push_back(std::move(catalog));
    }
    return catalogs;
}

/// A device of @p arch as the matchers read it: wave size and compute units of its usual part.
hipdnn_plugin_sdk::ingestor::DeviceProperties deviceFor(const std::string& arch)
{
    static const std::map<std::string, int> s_units{{"gfx906", 60},
                                                    {"gfx908", 120},
                                                    {"gfx90a", 104},
                                                    {"gfx942", 304},
                                                    {"gfx950", 256},
                                                    {"gfx1100", 96},
                                                    {"gfx1101", 60},
                                                    {"gfx1151", 40},
                                                    {"gfx1201", 64}};
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = arch.rfind("gfx1", 0) == 0 && arch.size() > 5 ? 32 : 64;
    const auto units = s_units.find(arch);
    properties.multiProcessorCount = units != s_units.end() ? units->second : 64;
    properties.sharedMemPerBlock = arch == "gfx950" ? 160 * 1024 : 64 * 1024;
    return properties;
}

/// A coverage-map row as a shape and direction; nullopt for a row this census cannot build.
std::optional<std::pair<conv_naive_test::ConvShape, ConvDirection>>
    shapeOf(const nlohmann::json& row)
{
    conv_naive_test::ConvShape shape;
    const auto x = row.at("x").get<std::vector<int64_t>>();
    const auto w = row.at("w").get<std::vector<int64_t>>();
    if(x.size() < 3 || w.size() != x.size() || w[1] <= 0)
    {
        return std::nullopt;
    }
    shape.n = x[0];
    shape.c = x[1];
    shape.k = w[0];
    shape.groups = x[1] / w[1];
    shape.input.assign(x.begin() + 2, x.end());
    shape.filter.assign(w.begin() + 2, w.end());
    shape.prePadding
        = (row.contains("pad") ? row.at("pad") : row.at("pre")).get<std::vector<int64_t>>();
    shape.postPadding
        = (row.contains("pad") ? row.at("pad") : row.at("post")).get<std::vector<int64_t>>();
    shape.stride = row.at("stride").get<std::vector<int64_t>>();
    shape.dilation = row.at("dil").get<std::vector<int64_t>>();
    shape.channelsLast = row.at("layout").get<std::string>() == "nhwc";
    const auto dtype = row.at("dtype").get<std::string>();
    shape.dataType = dtype == "half"       ? DataType::HALF
                     : dtype == "bfloat16" ? DataType::BFLOAT16
                                           : DataType::FLOAT;
    shape.mode = row.value("mode", std::string()) == "CONVOLUTION" ? ConvMode::CONVOLUTION
                                                                   : ConvMode::CROSS_CORRELATION;
    const auto direction = row.at("dir").get<std::string>();
    ConvDirection dir = ConvDirection::FORWARD;
    if(direction == "bwd")
    {
        dir = ConvDirection::BACKWARD_DATA;
    }
    else if(direction == "wrw")
    {
        dir = ConvDirection::BACKWARD_WEIGHTS;
    }
    return std::make_pair(shape, dir);
}

/// One graph of the pool, built for each architecture.
struct PoolGraph
{
    std::string label;
    ConvDirection direction = ConvDirection::FORWARD;
    std::function<flatbuffers::FlatBufferBuilder()> build;
};

/// Adds @p spec, and for a forward graph the same convolution with a fused bias and clipped ReLU.
void addGraph(std::vector<PoolGraph>& pool,
              const std::string& label,
              const conv_naive_test::ConvShape& shape,
              const conv_naive_test::ConvGraphSpec& spec)
{
    pool.push_back({label, spec.direction, [spec] { return conv_naive_test::buildGraph(spec); }});
    if(spec.direction == ConvDirection::FORWARD)
    {
        const std::vector<conv_epilogue_test::Step> steps{conv_epilogue_test::bias(),
                                                          conv_epilogue_test::relu(0.0F, 6.0F)};
        const auto fused = conv_epilogue_test::fusedSpecFor(shape, steps);
        pool.push_back({label + "+bias+relu6", spec.direction, [fused] {
                            return conv_epilogue_test::buildFusedGraph(fused);
                        }});
    }
}

/// The coverage map's graphs, each in its own memory form where the row records one.
void addMapGraphs(std::vector<PoolGraph>& pool, const nlohmann::json& map)
{
    for(const auto& row : map.at("rows"))
    {
        const auto parsed = shapeOf(row);
        if(!parsed.has_value())
        {
            continue;
        }
        const auto& [shape, direction] = *parsed;
        auto spec = conv_naive_test::specFor(shape, direction);
        if(row.contains("xs"))
        {
            spec.x.strides = row.at("xs").get<std::vector<int64_t>>();
        }
        if(row.contains("ws"))
        {
            spec.w.strides = row.at("ws").get<std::vector<int64_t>>();
        }
        if(row.contains("ys"))
        {
            spec.y.strides = row.at("ys").get<std::vector<int64_t>>();
        }
        addGraph(pool, row.at("id").get<std::string>(), shape, spec);
    }
}

/// A sweep along what the configs bake: filter geometry (2-D and 3-D, square and separable, the
/// depthwise-rows widths), stride, group width (dense, depthwise, 2 to 64 channels a group,
/// small-channel stems), dtype, layout and direction, at extents every config's tile admits.
void addSweepGraphs(std::vector<PoolGraph>& pool)
{
    const std::vector<std::vector<int64_t>> filters{
        {1, 1},    {2, 2},    {3, 3},    {4, 4},    {5, 5},    {7, 7},  {9, 9},
        {11, 11},  {1, 3},    {3, 1},    {1, 7},    {7, 1},    {1, 31}, {1, 43},
        {3, 3, 3}, {1, 3, 3}, {3, 1, 1}, {5, 5, 5}, {3, 7, 7}, {3},     {5}};
    // Channels, groups: dense, depthwise, groups of 2 to 64, and small-channel stems.
    const std::vector<std::pair<int64_t, int64_t>> channels{{256, 1},
                                                            {256, 256},
                                                            {256, 128},
                                                            {256, 64},
                                                            {256, 32},
                                                            {256, 16},
                                                            {384, 16},
                                                            {256, 8},
                                                            {256, 4},
                                                            {3, 1},
                                                            {4, 1},
                                                            {1, 1}};
    const std::vector<DataType> types{DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};
    for(const auto& filter : filters)
    {
        const size_t rank = filter.size();
        for(const int64_t stride : {1, 2, 3, 4})
        {
            for(const auto& [c, groups] : channels)
            {
                for(const auto type : types)
                {
                    for(const bool channelsLast : {false, true})
                    {
                        for(const auto direction : {ConvDirection::FORWARD,
                                                    ConvDirection::BACKWARD_DATA,
                                                    ConvDirection::BACKWARD_WEIGHTS})
                        {
                            conv_naive_test::ConvShape shape;
                            shape.n = 2;
                            shape.c = c;
                            shape.k = c < 16 ? 64 : c;
                            shape.groups = groups;
                            shape.filter = filter;
                            shape.input.assign(rank, rank == 3 ? 12 : (rank == 1 ? 256 : 40));
                            shape.input.back()
                                = std::max<int64_t>(shape.input.back(), 2 * filter.back() + 8);
                            shape.prePadding.clear();
                            shape.postPadding.clear();
                            for(const auto f : filter)
                            {
                                shape.prePadding.push_back(f / 2);
                                shape.postPadding.push_back(f / 2);
                            }
                            shape.stride.assign(rank, stride);
                            shape.dilation.assign(rank, 1);
                            shape.channelsLast = channelsLast;
                            shape.dataType = type;
                            std::ostringstream label;
                            label << "sweep f" << nlohmann::json(filter).dump() << " s" << stride
                                  << " c" << c << " g" << groups << " t" << static_cast<int>(type)
                                  << (channelsLast ? " cl" : " cf") << " d"
                                  << static_cast<int>(direction);
                            addGraph(pool,
                                     label.str(),
                                     shape,
                                     conv_naive_test::specFor(shape, direction));
                        }
                    }
                }
            }
        }
    }
}

TEST(TestConvCompileCensus, PreparesEveryAdmittedConfig)
{
    const auto tree = envOr("HKP_CONV_CENSUS_TREE");
    const auto mapPath = envOr("HKP_CONV_CENSUS_MAP");
    const auto outPath = envOr("HKP_CONV_CENSUS_OUT");
    if(tree.empty() || mapPath.empty() || outPath.empty())
    {
        GTEST_SKIP() << "opt-in: set HKP_CONV_CENSUS_TREE, HKP_CONV_CENSUS_MAP and "
                        "HKP_CONV_CENSUS_OUT (conv/tools/code_gate/check_code_size.py)";
    }
    const int perConfig = std::stoi(envOr("HKP_CONV_CENSUS_PER_CONFIG", "8"));
    std::vector<std::string> archs;
    {
        std::stringstream list(
            envOr("HKP_CONV_CENSUS_ARCHS",
                  "gfx950,gfx942,gfx90a,gfx908,gfx906,gfx1100,gfx1101,gfx1151,gfx1201"));
        for(std::string arch; std::getline(list, arch, ',');)
        {
            archs.push_back(arch);
        }
    }

    const auto catalogs = loadCatalogs(tree);
    ASSERT_FALSE(catalogs.empty()) << "no *.kdp.json under " << tree;
    std::vector<PoolGraph> pool;
    addMapGraphs(pool, readJson(mapPath));
    if(envOr("HKP_CONV_CENSUS_SWEEP", "1") == "1")
    {
        addSweepGraphs(pool);
    }
    ASSERT_FALSE(pool.empty()) << "no graphs in " << mapPath;
    registerNativeIngestorSymbols();

    std::ofstream out(outPath);
    for(const auto& arch : archs)
    {
        const auto device = deviceFor(arch);
        // Per catalog config: graphs that admitted it, prepares that succeeded, first failure.
        std::map<std::pair<size_t, size_t>, std::tuple<int, int, std::string>> tally;
        for(const auto& graph : pool)
        {
            const GraphFixture fixture(graph.build(), device);
            const auto context = fixture.context();
            for(size_t c = 0; c < catalogs.size(); ++c)
            {
                const auto& catalog = catalogs[c];
                if(catalog.direction != graph.direction
                   || (!catalog.arch.empty()
                       && std::find(catalog.arch.begin(), catalog.arch.end(), arch)
                              == catalog.arch.end()))
                {
                    continue;
                }
                std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens> bound;
                try
                {
                    bound = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
                        catalog.graphMatch)(context);
                }
                catch(const std::exception&)
                {
                    continue;
                }
                if(!bound.has_value())
                {
                    continue;
                }
                const auto matcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
                    catalog.kernelMatch);
                const auto* handler
                    = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
                        catalog.dispatch);
                for(size_t k = 0; k < catalog.kernels.size(); ++k)
                {
                    auto& [admitted, prepared, failure] = tally[{c, k}];
                    if(admitted >= perConfig)
                    {
                        continue;
                    }
                    const auto& kernel = catalog.kernels[k];
                    try
                    {
                        if(!matcher(context, *bound, kernel))
                        {
                            continue;
                        }
                        ++admitted;
                        if(handler->prepare(context, *bound, kernel) != nullptr)
                        {
                            ++prepared;
                        }
                    }
                    catch(const std::exception& error)
                    {
                        if(failure.empty())
                        {
                            failure = graph.label + ": " + error.what();
                        }
                    }
                }
            }
        }
        for(size_t c = 0; c < catalogs.size(); ++c)
        {
            const auto& catalog = catalogs[c];
            if(!catalog.arch.empty()
               && std::find(catalog.arch.begin(), catalog.arch.end(), arch) == catalog.arch.end())
            {
                continue;
            }
            for(size_t k = 0; k < catalog.kernels.size(); ++k)
            {
                const auto found = tally.find({c, k});
                const auto [admitted, prepared, failure]
                    = found != tally.end() ? found->second : std::tuple<int, int, std::string>{};
                out << nlohmann::json{{"arch", arch},
                                      {"engine", catalog.engine},
                                      {"catalog", catalog.file},
                                      {"config", catalog.kernels[k].name},
                                      {"admitted", admitted},
                                      {"prepared", prepared},
                                      {"failure", failure}}
                           .dump()
                    << "\n";
            }
        }
        out.flush();
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
