// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <nlohmann/json.hpp>

/**
 * @file IntegrationConvCoverageCensus.cpp
 * @brief Discovery-only coverage census of the convolution engines.
 *
 * A checked-in map (`hip_kernel_provider_conv_coverage_map.json`, installed beside this
 * binary) lists about 1.8K convolution graphs chosen to span the parameter space: every
 * direction, rank, filter, stride, dilation, padding form, group kind, channel class, dtype,
 * layout, conv mode and memory form, the boundary graphs of every audited host gate, and
 * images and tensors past 2 GiB. For each graph the map records, one letter per entry of its
 * "archs" list, the tier that served it when the map was made:
 *   h  a hipkernel:CONV_* engine other than naive;
 *   b  only the explicit-GEMM engines (ROCBLAS_CONV_ENGINE, HIPBLASLT_CONV_ENGINE);
 *   n  only hipkernel:CONV_NAIVE_*;
 *   m  only MIOpen; u  no engine at all; -  not recorded for that architecture.
 * Every graph is built and its engines are asked for (get_ranked_engine_ids); nothing is
 * compiled or executed, so the whole map takes seconds. The checks:
 *   (a) no graph is served below its mapped tier on this architecture (a regression);
 *       a graph served above it is an intended upgrade, listed for the map refresh. An
 *       architecture the map has no column for skips this check;
 *   (b) the generic-tier engines a row names (those that served it on every architecture the
 *       row records) serve it here, on the architectures the row records;
 *   (c) every fwd/bwd graph some engine serves has a served wrw twin (same tensors), except
 *       twins the map records as known gaps for this architecture, or, on an architecture
 *       without a column, as known gaps on any architecture; and naive wrw serves
 *       every scaled pair whose fwd/bwd original naive serves (the naive engines are the
 *       structural floor);
 *   (d) each gated engine serves the graph just below its limit on the architectures the
 *       map records.
 * Engine names come from the backend, so every engine a provider loads is classified.
 * A build without the explicit-GEMM conv engines (HIPBLASLTPROVIDER_ENABLE_CONV=OFF, the
 * default) cannot check b rows or wrw twins only those engines serve; the test says so and
 * checks the rest.
 * Known holes are rows with a b (or lower) tier: a fix that serves them with a HIP engine
 * shows as an upgrade. Refresh the map for this architecture with
 * HIPKERNELPROVIDER_CONV_COVERAGE_MAP_OUT=<file>, which writes the map with this run's tiers,
 * known gaps, generic-tier requirements and gate acceptances (run it with the explicit-GEMM
 * engines built, so b tiers are seen).
 */
namespace
{

constexpr const char* MAP_FILE = "hip_kernel_provider_conv_coverage_map.json";
constexpr const char* MAP_OUT_ENV = "HIPKERNELPROVIDER_CONV_COVERAGE_MAP_OUT";
constexpr std::string_view HIP_PREFIX = "hipkernel:CONV_";
constexpr std::string_view NAIVE_PREFIX = "hipkernel:CONV_NAIVE_";
// The explicit-GEMM conv engines (ROCBLAS_CONV_*, HIPBLASLT_CONV_*, any variant).
constexpr std::array<std::string_view, 2> BLAS_PREFIXES = {"ROCBLAS_CONV_", "HIPBLASLT_CONV_"};

enum class Tier
{
    HIP = 0,
    BLAS = 1,
    NAIVE = 2,
    NONE = 3,
};

// Map tier letters: h hip, b blas, n naive, m MIOpen only, u unserved, - not recorded.
Tier parseTier(char letter)
{
    if(letter == 'h')
    {
        return Tier::HIP;
    }
    if(letter == 'b')
    {
        return Tier::BLAS;
    }
    if(letter == 'n')
    {
        return Tier::NAIVE;
    }
    return Tier::NONE;
}

const char* tierName(Tier tier)
{
    constexpr std::array<const char*, 4> NAMES = {"hip", "blas", "naive", "unserved"};
    return NAMES.at(static_cast<size_t>(tier));
}

char tierLetter(Tier tier)
{
    constexpr std::array<char, 4> LETTERS = {'h', 'b', 'n', 'u'};
    return LETTERS.at(static_cast<size_t>(tier));
}

/// A row in words, for failure messages.
std::string describe(const nlohmann::json& row)
{
    std::ostringstream text;
    text << row.at("id").get<std::string>() << " (" << row.at("dir").get<std::string>() << " "
         << row.at("dtype").get<std::string>() << " " << row.at("layout").get<std::string>() << " x"
         << row.at("x").dump() << " w" << row.at("w").dump() << " stride" << row.at("stride").dump()
         << " dil" << row.at("dil").dump();
    if(row.contains("pad"))
    {
        text << " pad" << row.at("pad").dump();
    }
    else
    {
        text << " pre" << row.at("pre").dump() << " post" << row.at("post").dump();
    }
    for(const auto* key : {"mode", "xs", "ws", "ys"})
    {
        if(row.contains(key))
        {
            text << " " << key << row.at(key).dump();
        }
    }
    text << ")";
    return text.str();
}

bool isBlas(const std::string& engine)
{
    return std::any_of(BLAS_PREFIXES.begin(), BLAS_PREFIXES.end(), [&](std::string_view prefix) {
        return engine.rfind(prefix, 0) == 0;
    });
}

Tier tierOf(const std::vector<std::string>& engines)
{
    auto tier = Tier::NONE;
    for(const auto& engine : engines)
    {
        if(engine.rfind(HIP_PREFIX, 0) == 0 && engine.rfind(NAIVE_PREFIX, 0) != 0)
        {
            return Tier::HIP;
        }
        if(isBlas(engine))
        {
            tier = Tier::BLAS;
        }
        else if(engine.rfind(NAIVE_PREFIX, 0) == 0 && tier == Tier::NONE)
        {
            tier = Tier::NAIVE;
        }
    }
    return tier;
}

/// The base arch of the current device (features dropped), empty without one.
std::string deviceArch()
{
    int device = 0;
    hipDeviceProp_t properties{};
    if(hipGetDevice(&device) != hipSuccess
       || hipGetDeviceProperties(&properties, device) != hipSuccess)
    {
        return {};
    }
    const std::string name = properties.gcnArchName;
    return name.substr(0, name.find(':'));
}

std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, bool channelsLast)
{
    std::vector<size_t> order = {0};
    if(!channelsLast)
    {
        order.push_back(1);
    }
    for(size_t axis = 2; axis < dims.size(); ++axis)
    {
        order.push_back(axis);
    }
    if(channelsLast)
    {
        order.push_back(1);
    }
    std::vector<int64_t> strides(dims.size(), 0);
    int64_t running = 1;
    for(auto it = order.rbegin(); it != order.rend(); ++it)
    {
        strides[*it] = running;
        running *= dims[*it];
    }
    return strides;
}

/// The hipDNN graph JSON of a compact map row.
std::string graphJson(const nlohmann::json& row)
{
    const auto dtype = row.at("dtype").get<std::string>();
    const bool channelsLast = row.at("layout").get<std::string>() == "nhwc";
    const auto tensor
        = [&](int64_t uid, const char* name, const char* dimsKey, const char* stridesKey) {
              const auto dims = row.at(dimsKey).get<std::vector<int64_t>>();
              const auto strides = row.contains(stridesKey)
                                       ? row.at(stridesKey).get<std::vector<int64_t>>()
                                       : packedStrides(dims, channelsLast);
              return nlohmann::json{{"uid", uid},
                                    {"name", name},
                                    {"dims", dims},
                                    {"strides", strides},
                                    {"data_type", dtype},
                                    {"virtual", false}};
          };
    nlohmann::json node;
    nlohmann::json tensors;
    const auto direction = row.at("dir").get<std::string>();
    if(direction == "fwd")
    {
        tensors = {tensor(1, "x", "x", "xs"), tensor(2, "w", "w", "ws"), tensor(0, "y", "y", "ys")};
        node = {{"name", "conv_fprop"},
                {"type", "ConvolutionFwdAttributes"},
                {"inputs", {{"x_tensor_uid", 1}, {"w_tensor_uid", 2}}},
                {"outputs", {{"y_tensor_uid", 0}}}};
    }
    else if(direction == "bwd")
    {
        tensors
            = {tensor(1, "dy", "y", "ys"), tensor(2, "w", "w", "ws"), tensor(0, "dx", "x", "xs")};
        node = {{"name", "conv_dgrad"},
                {"type", "ConvolutionBwdAttributes"},
                {"inputs", {{"dy_tensor_uid", 1}, {"w_tensor_uid", 2}}},
                {"outputs", {{"dx_tensor_uid", 0}}}};
    }
    else
    {
        tensors
            = {tensor(1, "x", "x", "xs"), tensor(2, "dy", "y", "ys"), tensor(0, "dw", "w", "ws")};
        node = {{"name", "conv_wgrad"},
                {"type", "ConvolutionWrwAttributes"},
                {"inputs", {{"x_tensor_uid", 1}, {"dy_tensor_uid", 2}}},
                {"outputs", {{"dw_tensor_uid", 0}}}};
    }
    node["compute_data_type"] = "float";
    node["parameters"] = {{"conv_mode", row.value("mode", std::string("CROSS_CORRELATION"))},
                          {"pre_padding", row.contains("pad") ? row.at("pad") : row.at("pre")},
                          {"post_padding", row.contains("pad") ? row.at("pad") : row.at("post")},
                          {"stride", row.at("stride")},
                          {"dilation", row.at("dil")}};
    const nlohmann::json graph = {{"name", row.at("id")},
                                  {"compute_data_type", "float"},
                                  {"io_data_type", dtype},
                                  {"intermediate_data_type", dtype},
                                  {"tensors", tensors},
                                  {"nodes", nlohmann::json::array({node})}};
    return graph.dump();
}

struct Observed
{
    std::vector<std::string> engines;
    Tier tier = Tier::NONE;
    std::string error;
};

struct Census
{
    std::string skip;
    std::string arch;
    nlohmann::json map;
    std::map<std::string, Observed> observed;
    bool blasLoaded = false;
};

std::filesystem::path pluginDirectory()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    return std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path());
}

/// The name of a loaded engine, resolved through the backend (so engines this test does not
/// know by name are classified too).
std::string engineName(hipdnnHandle_t handle, int64_t id, std::map<int64_t, std::string>& names)
{
    const auto it = names.find(id);
    if(it != names.end())
    {
        return it->second;
    }
    std::string name = "id:" + std::to_string(id);
    size_t length = 0;
    if(hipdnnGetEngineNameById_ext(handle, id, nullptr, &length) == HIPDNN_STATUS_SUCCESS
       && length > 0)
    {
        std::vector<char> buffer(length);
        if(hipdnnGetEngineNameById_ext(handle, id, buffer.data(), &length) == HIPDNN_STATUS_SUCCESS)
        {
            name = buffer.data();
        }
    }
    names.emplace(id, name);
    return name;
}

Observed discover(hipdnnHandle_t handle,
                  const nlohmann::json& row,
                  std::map<int64_t, std::string>& names)
{
    Observed result;
    hipdnn_frontend::graph::Graph graph;
    auto error = graph.from_json(graphJson(row));
    if(!error.is_bad())
    {
        error = graph.validate();
    }
    if(!error.is_bad())
    {
        error = graph.build_operation_graph(handle);
    }
    std::vector<int64_t> ids;
    if(!error.is_bad())
    {
        error = graph.get_ranked_engine_ids(ids);
    }
    if(error.is_bad())
    {
        result.error = error.get_message();
    }
    for(const auto id : ids)
    {
        result.engines.push_back(engineName(handle, id, names));
    }
    result.tier = tierOf(result.engines);
    return result;
}

const Census& census()
{
    static const Census RESULT = [] {
        Census c;
        c.arch = deviceArch();
        if(c.arch.empty())
        {
            c.skip = "no device";
            return c;
        }
        const auto mapPath = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / MAP_FILE;
        std::ifstream stream(mapPath);
        c.map = nlohmann::json::parse(stream, nullptr, false);
        if(!c.map.is_object() || !c.map.contains("rows"))
        {
            c.skip = "no readable coverage map at " + mapPath.string();
            return c;
        }

        // The conv group's provider and, when this build ships it, the explicit-GEMM provider.
        // MIOpen is not loaded: the census measures what these providers serve.
        const auto directory = pluginDirectory();
        std::vector<std::string> plugins
            = {(directory / std::filesystem::path(PLUGIN_PATH).filename()).string()};
        const auto blas = directory / "hipblaslt_plugin";
        std::error_code absent;
        if(std::filesystem::exists(directory
                                       / (hipdnn_data_sdk::utilities::LIB_PREFIX
                                          + std::string("hipblaslt_plugin")
                                          + hipdnn_data_sdk::utilities::SHARED_LIB_EXT),
                                   absent))
        {
            plugins.push_back(blas.string());
        }
        std::vector<const char*> paths;
        paths.reserve(plugins.size());
        for(const auto& plugin : plugins)
        {
            paths.push_back(plugin.c_str());
        }
        if(hipdnnSetEnginePluginPaths_ext(
               paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE)
           != HIPDNN_STATUS_SUCCESS)
        {
            c.skip = "could not set the engine plugin paths";
            return c;
        }
        hipdnnHandle_t handle = nullptr;
        if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
        {
            c.skip = "could not create a hipDNN handle";
            return c;
        }

        std::map<int64_t, std::string> names;
        for(const auto& row : c.map.at("rows"))
        {
            auto observed = discover(handle, row, names);
            c.blasLoaded = c.blasLoaded
                           || std::any_of(observed.engines.begin(), observed.engines.end(), isBlas);
            c.observed.emplace(row.at("id").get<std::string>(), std::move(observed));
        }
        hipdnnDestroy(handle);

        // Refresh mode: the map with this architecture's tiers set to what this run saw.
        const auto out = hipdnn_data_sdk::utilities::getEnv(MAP_OUT_ENV);
        if(!out.empty())
        {
            auto refreshed = c.map;
            auto& archs = refreshed["archs"];
            auto slot = static_cast<size_t>(
                std::distance(archs.begin(), std::find(archs.begin(), archs.end(), c.arch)));
            if(slot == archs.size())
            {
                archs.push_back(c.arch);
            }
            for(auto& row : refreshed["rows"])
            {
                const auto& seen = c.observed.at(row.at("id").get<std::string>());
                // This census loads no MIOpen, so a graph it finds unserved keeps the MIOpen
                // label the map gave it.
                auto tiers = row.at("tier").get<std::string>();
                tiers.resize(archs.size(), '-');
                if(seen.tier != Tier::NONE || (tiers[slot] != 'm' && tiers[slot] != 'u'))
                {
                    tiers[slot] = tierLetter(seen.tier);
                }
                row["tier"] = tiers;
                // Requirements and gates hold where they were seen: a generic-tier engine this
                // architecture does not offer for the row leaves the row's requirement, and the
                // gated engine's acceptance just below its limit is recorded per architecture.
                if(row.contains("require"))
                {
                    auto required = nlohmann::json::array();
                    for(const auto& engine : row.at("require"))
                    {
                        if(std::find(
                               seen.engines.begin(), seen.engines.end(), engine.get<std::string>())
                           != seen.engines.end())
                        {
                            required.push_back(engine);
                        }
                    }
                    if(required.empty())
                    {
                        row.erase("require");
                    }
                    else
                    {
                        row["require"] = required;
                    }
                }
                if(row.contains("gate"))
                {
                    const auto prefix = row["gate"]["prefix"].get<std::string>();
                    auto& gateArchs = row["gate"]["archs"];
                    const bool accepted = std::any_of(
                        seen.engines.begin(), seen.engines.end(), [&](const std::string& engine) {
                            return engine.rfind(prefix, 0) == 0;
                        });
                    const bool listed
                        = std::find(gateArchs.begin(), gateArchs.end(), c.arch) != gateArchs.end();
                    if(accepted && !listed)
                    {
                        gateArchs.push_back(c.arch);
                    }
                    else if(!accepted && listed)
                    {
                        gateArchs.erase(std::remove(gateArchs.begin(), gateArchs.end(), c.arch),
                                        gateArchs.end());
                    }
                }
                if(row.value("pair_kind", std::string()) == "twin")
                {
                    auto gaps = row.value("known_gap", nlohmann::json::array());
                    const bool listed = std::find(gaps.begin(), gaps.end(), c.arch) != gaps.end();
                    const bool gap
                        = seen.tier == Tier::NONE
                          && c.observed.at(row.at("pair").get<std::string>()).tier != Tier::NONE;
                    if(listed && !gap)
                    {
                        gaps.erase(std::remove(gaps.begin(), gaps.end(), c.arch), gaps.end());
                    }
                    else if(!listed && gap)
                    {
                        gaps.push_back(c.arch);
                    }
                    if(gaps.empty())
                    {
                        row.erase("known_gap");
                    }
                    else
                    {
                        row["known_gap"] = gaps;
                    }
                }
            }
            // The checked-in layout: header keys, then one compact row per line.
            std::ofstream file(out);
            file << "{\n";
            for(const auto* key : {"archs", "revision", "tiers", "version"})
            {
                file << "\"" << key << "\":" << refreshed[key].dump() << ",\n";
            }
            file << "\"rows\":[\n";
            const auto& rows = refreshed["rows"];
            for(size_t i = 0; i < rows.size(); ++i)
            {
                file << rows[i].dump() << (i + 1 < rows.size() ? ",\n" : "\n");
            }
            file << "]\n}\n";
        }
        return c;
    }();
    return RESULT;
}

/// The tier a row is mapped to on this architecture, or nothing when the map has none.
std::optional<Tier>
    mappedTier(const nlohmann::json& map, const nlohmann::json& row, const std::string& arch)
{
    const auto& archs = map.at("archs");
    const auto slot = static_cast<size_t>(
        std::distance(archs.begin(), std::find(archs.begin(), archs.end(), arch)));
    const auto& tiers = row.at("tier").get_ref<const std::string&>();
    if(slot >= tiers.size() || tiers[slot] == '-')
    {
        return std::nullopt;
    }
    return parseTier(tiers[slot]);
}

/// Whether the twin row is a recorded known gap on this architecture: listed for it, or, on
/// an architecture the row records no tier for, listed for any architecture. Whether a twin is
/// served can depend on the device (the naive engines' budget scales with its compute units),
/// so an architecture without a column cannot tell a known gap from a new one.
bool knownGap(const nlohmann::json& map, const nlohmann::json& row, const std::string& arch)
{
    if(!row.contains("known_gap"))
    {
        return false;
    }
    const auto& gaps = row.at("known_gap");
    if(std::find(gaps.begin(), gaps.end(), arch) != gaps.end())
    {
        return true;
    }
    if(mappedTier(map, row, arch))
    {
        return false;
    }
    return !gaps.empty();
}

/// Whether a twin's service may rest on the explicit-GEMM engines alone: the map records
/// it as served only by them on this architecture, or, on an architecture without a column,
/// on any architecture, or records no HIP or naive service for it anywhere.
bool mayNeedBlas(const nlohmann::json& map, const nlohmann::json& row, const std::string& arch)
{
    const auto& tiers = row.at("tier").get_ref<const std::string&>();
    if(tiers.find_first_of("hn") == std::string::npos)
    {
        return true;
    }
    const auto& archs = map.at("archs");
    const auto slot = static_cast<size_t>(
        std::distance(archs.begin(), std::find(archs.begin(), archs.end(), arch)));
    if(slot < tiers.size() && tiers[slot] != '-')
    {
        return tiers[slot] == 'b';
    }
    return tiers.find('b') != std::string::npos;
}

constexpr const char* NO_BLAS_NOTE
    = "this build serves no explicit-GEMM conv engine (ROCBLAS_CONV_*, HIPBLASLT_CONV_*); the "
      "map's b rows and the wrw twins only those engines serve are not checked. Build with "
      "HIPBLASLTPROVIDER_ENABLE_CONV=ON to check them.";

#define SKIP_WITHOUT_CENSUS()              \
    do                                     \
    {                                      \
        if(!census().skip.empty())         \
        {                                  \
            GTEST_SKIP() << census().skip; \
        }                                  \
    } while(false)

TEST(IntegrationConvCoverageCensus, NoGraphIsServedBelowItsMappedTier)
{
    SKIP_WITHOUT_CENSUS();
    const auto& c = census();
    std::ostringstream regressions;
    std::ostringstream upgrades;
    int worse = 0;
    int better = 0;
    int checked = 0;
    int unchecked = 0;
    for(const auto& row : c.map.at("rows"))
    {
        const auto expected = mappedTier(c.map, row, c.arch);
        if(!expected)
        {
            ++unchecked;
            continue;
        }
        const auto& id = row.at("id").get_ref<const std::string&>();
        const auto& seen = c.observed.at(id);
        // A blas row cannot be checked in a build without the explicit-GEMM engines.
        if(*expected == Tier::BLAS && !c.blasLoaded && seen.tier != Tier::HIP)
        {
            ++unchecked;
            continue;
        }
        ++checked;
        if(seen.tier > *expected)
        {
            ++worse;
            regressions << "\n  " << describe(row) << ": mapped " << tierName(*expected)
                        << ", served " << tierName(seen.tier)
                        << (seen.error.empty() ? "" : " (" + seen.error + ")");
        }
        else if(seen.tier < *expected)
        {
            ++better;
            upgrades << "\n  " << id << ": " << tierName(*expected) << " -> "
                     << tierName(seen.tier);
        }
    }
    if(!c.blasLoaded)
    {
        std::cout << "Note: " << NO_BLAS_NOTE << "\n";
    }
    if(better > 0)
    {
        std::cout << better << " graphs are served above their mapped tier on " << c.arch
                  << " (refresh the map with " << MAP_OUT_ENV << "):" << upgrades.str() << "\n";
    }
    if(checked == 0)
    {
        GTEST_SKIP() << "the coverage map records no tier for " << c.arch;
    }
    EXPECT_EQ(worse, 0) << worse << " of " << checked << " graphs are served below their mapped "
                        << "tier on " << c.arch << " (" << unchecked
                        << " unchecked):" << regressions.str();
}

TEST(IntegrationConvCoverageCensus, GenericTierEnginesServeTheirRegion)
{
    SKIP_WITHOUT_CENSUS();
    const auto& c = census();
    std::ostringstream missing;
    int failures = 0;
    for(const auto& row : c.map.at("rows"))
    {
        // The generic-tier engines a row names are those that served it on every architecture
        // the map records; on others the requirement is unknown.
        if(!row.contains("require") || !mappedTier(c.map, row, c.arch))
        {
            continue;
        }
        const auto& id = row.at("id").get_ref<const std::string&>();
        const auto& engines = c.observed.at(id).engines;
        for(const auto& engine : row["require"])
        {
            if(std::find(engines.begin(), engines.end(), engine.get<std::string>())
               == engines.end())
            {
                ++failures;
                missing << "\n  " << describe(row) << ": " << engine.get<std::string>();
            }
        }
    }
    EXPECT_EQ(failures, 0) << "generic-tier engines decline graphs in their region on " << c.arch
                           << ":" << missing.str();
}

TEST(IntegrationConvCoverageCensus, EveryServedFwdBwdGraphHasAServedWrwTwin)
{
    SKIP_WITHOUT_CENSUS();
    const auto& c = census();
    std::ostringstream holes;
    int failures = 0;
    int pairs = 0;
    int unchecked = 0;
    for(const auto& row : c.map.at("rows"))
    {
        if(row.value("pair_kind", std::string()) != "twin")
        {
            continue;
        }
        const auto& original = c.observed.at(row.at("pair").get<std::string>());
        if(original.tier == Tier::NONE)
        {
            continue;
        }
        ++pairs;
        const auto& id = row.at("id").get_ref<const std::string&>();
        if(c.observed.at(id).tier != Tier::NONE || knownGap(c.map, row, c.arch))
        {
            continue;
        }
        if(!c.blasLoaded && mayNeedBlas(c.map, row, c.arch))
        {
            ++unchecked;
            continue;
        }
        ++failures;
        holes << "\n  " << describe(row);
    }
    if(!c.blasLoaded)
    {
        std::cout << "Note: " << NO_BLAS_NOTE << " (" << unchecked << " twins unchecked)\n";
    }
    EXPECT_EQ(failures, 0) << failures << " of " << pairs << " served fwd/bwd graphs on " << c.arch
                           << " have a wrw twin no engine serves:" << holes.str();
}

TEST(IntegrationConvCoverageCensus, NaiveWrwAcceptsWhatNaiveFwdBwdAccept)
{
    SKIP_WITHOUT_CENSUS();
    const auto& c = census();
    std::ostringstream violations;
    int failures = 0;
    for(const auto& row : c.map.at("rows"))
    {
        if(row.value("pair_kind", std::string()) != "scaled")
        {
            continue;
        }
        const auto& original = c.observed.at(row.at("pair").get<std::string>()).engines;
        const bool naiveOriginal = std::any_of(
            original.begin(), original.end(), [](const std::string& engine) {
                return engine == "hipkernel:CONV_NAIVE_FWD" || engine == "hipkernel:CONV_NAIVE_BWD";
            });
        const auto& id = row.at("id").get_ref<const std::string&>();
        const auto& twin = c.observed.at(id).engines;
        if(naiveOriginal
           && std::find(twin.begin(), twin.end(), "hipkernel:CONV_NAIVE_WRW") == twin.end())
        {
            ++failures;
            violations << "\n  " << describe(row);
        }
    }
    EXPECT_EQ(failures, 0) << "naive wrw declines structure naive fwd/bwd accept on " << c.arch
                           << ":" << violations.str();
}

TEST(IntegrationConvCoverageCensus, GatedEnginesServeTheGraphJustBelowTheirLimit)
{
    SKIP_WITHOUT_CENSUS();
    const auto& c = census();
    std::ostringstream declined;
    int failures = 0;
    for(const auto& row : c.map.at("rows"))
    {
        if(!row.contains("gate"))
        {
            continue;
        }
        const auto& archs = row["gate"]["archs"];
        if(std::find(archs.begin(), archs.end(), c.arch) == archs.end())
        {
            continue;
        }
        const auto prefix = row["gate"]["prefix"].get<std::string>();
        const auto& id = row.at("id").get_ref<const std::string&>();
        const auto& engines = c.observed.at(id).engines;
        if(std::none_of(engines.begin(), engines.end(), [&](const std::string& engine) {
               return engine.rfind(prefix, 0) == 0;
           }))
        {
            ++failures;
            declined << "\n  " << describe(row) << ": no " << prefix << "* engine";
        }
    }
    EXPECT_EQ(failures, 0) << "gated engines decline the graph just below their limit on " << c.arch
                           << ":" << declined.str();
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
