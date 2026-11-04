// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX950_CONV_WGRAD_AOT)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <nlohmann/json.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

// ---- hipkernel:Gfx950ConvWgradAot: the shipped wgrad catalog on the GPU ----
//
// The wgrad sibling of IntegrationGpuGfx950ConvDgrad.cpp. The engine ships a curated list
// of rocKE wgrad sweep kernels (IngestorGenerator/configs/gfx950_conv_wgrad_aot_{fp16,
// bf16}.instances.json): shape-generic implicit-GEMM instances (kernel_family 0, which bake
// no geometry at all and therefore admit every problem of their dtype -- unlike dgrad there
// is no tilde/halo/patch capability to gate on) and every packaged direct wgrad pipeline
// (kernel_family 1: direct_wgrad, direct_wgrad_tiled, direct_wgrad_dw, direct_wgrad_dw_cols,
// direct_wgrad_dw_mfma, direct_wgrad_gather, direct_wgrad_grouped_mfma), plus the
// direct_wgrad_fold helper (kernel_family 2) the engine never offers as a candidate.
//
// Unlike dgrad's suite, this one does not hardcode a per-graph instance census against the
// Python builder: because an implicit-GEMM wgrad instance bakes nothing, "how many catalog
// instances serve graph G" is computed here directly from the shipped catalog's own baked
// fields (the same predicate the native kernel matcher applies), not restated as a literal.
// This is deliberately a representative graph set, not an exhaustive one (see the note on
// IntegrationGpuGfx950ConvWgradAotCatalog::EveryInstanceMatchesCpuReference below) --
// widening it to reach every catalog instance, as dgrad's suite does, is left to whoever
// runs this on a GPU with the real packaged bundle (Phase 6).
//
// The UED knobs make every per-graph candidate tuple unique, so pinning all of a kernel's
// knobs selects exactly that kernel. Each pinned instance runs with dW poisoned (NaN)
// beforehand and is checked against CpuReferenceGraphExecutor.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx950ConvWgradAot";
constexpr const char* DESCRIPTOR_PREFIX = "gfx950_conv_wgrad_aot";
constexpr const char* KERNEL_FAMILY_KNOB = "kernel_family";
constexpr const char* DIRECT_VARIANT_KNOB = "direct_variant";
constexpr int64_t IMPLICIT_GEMM_FAMILY = 0;
constexpr int64_t DIRECT_FAMILY = 1;
constexpr int64_t HELPER_FAMILY = 2;
/// Every direct_variant a selectable kernel carries, "none" being implicit GEMM: the
/// packaged wgrad direct pipelines of builders/common/convolution_backward_filter.py
/// (AOT_WGRAD_DIRECT_ALGORITHMS), less the fold helper, which never serves a problem.
constexpr std::array<const char*, 8> VARIANTS{"none",
                                              "direct_wgrad",
                                              "direct_wgrad_tiled",
                                              "direct_wgrad_dw",
                                              "direct_wgrad_dw_cols",
                                              "direct_wgrad_dw_mfma",
                                              "direct_wgrad_gather",
                                              "direct_wgrad_grouped_mfma"};
constexpr size_t BENCHMARK_LIMIT = 32;

template <size_t Count>
bool among(const std::array<const char*, Count>& names, const std::string& name)
{
    return std::find_if(
               names.begin(), names.end(), [&](const char* entry) { return name == entry; })
           != names.end();
}

/// Whether a plan build failed because no candidate carries the requested knob values: the
/// frontend checks each value against the values the engine advertises for the graph, and
/// the engine checks the combination.
bool knobsUnsatisfiable(const std::string& error, const LogRecorderBase& recorder)
{
    for(const auto* marker :
        {"no kernel satisfying the requested knob setting", "is not in the list of valid values"})
    {
        if(error.find(marker) != std::string::npos || recorder.hasLogContaining(marker))
        {
            return true;
        }
    }
    return false;
}

constexpr int64_t X_UID = 1;
constexpr int64_t DY_UID = 2;
constexpr int64_t DW_UID = 3;

struct WgradProblem
{
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t hi = 14;
    int64_t wi = 14;
    int64_t y = 3;
    int64_t x = 3;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t padH = 1;
    int64_t padW = 1;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    // hipDNN carries no group attribute: groups = X channels / dW dims[1].
    int64_t groups = 1;

    int64_t cpg() const
    {
        return c / groups;
    }
    int64_t kpg() const
    {
        return k / groups;
    }
    int64_t ho() const
    {
        return (hi + 2 * padH - dilationH * (y - 1) - 1) / strideH + 1;
    }
    int64_t wo() const
    {
        return (wi + 2 * padW - dilationW * (x - 1) - 1) / strideW + 1;
    }
};

struct WgradGraph
{
    std::string name;
    WgradProblem problem;
};

// N is 2 wherever the kernel's grid allows it, so a write past the last dW row lands in the
// next one and fails. A deliberately representative, not exhaustive, set: dense unit
// stride, strided, dilated, a nonsquare filter, and several group/channel shapes the direct
// pipelines' per-variant knobs (cpg/kpg/taps) could plausibly serve. See the file header.
// clang-format off
const std::vector<WgradGraph>& wgradGraphs()
{
    static const std::vector<WgradGraph> s_graphs{
        {"Dense", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}},
        {"Strided", {2, 32, 48, 15, 13, 3, 3, 2, 2, 1, 1, 1, 1, 1}},
        {"Dilated2", {2, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2, 1}},
        {"NonsquareFilterStride1x2", {2, 24, 40, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1, 1}},
        {"PointwiseStride2", {2, 32, 64, 14, 13, 1, 1, 2, 2, 0, 0, 1, 1, 1}},
        {"GroupedG4", {2, 64, 96, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 4}},
        {"DepthwiseLikeG32", {2, 32, 32, 14, 15, 3, 3, 1, 1, 1, 1, 1, 1, 32}},
        {"C16K20", {2, 16, 20, 11, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}},
        {"Filter1x7", {2, 128, 128, 9, 17, 1, 7, 1, 1, 0, 3, 1, 1, 1}},
        {"HaloWo32", {2, 128, 128, 12, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}},
    };
    return s_graphs;
}
// clang-format on

struct WgradVariant
{
    std::string name;
    DataType dtype;
    /// Index into wgradGraphs(); the catalog suite runs every graph and leaves it empty.
    std::optional<size_t> graph;
};

// gtest prints a parameter on failure; the name says everything about it.
void PrintTo(const WgradVariant& variant, std::ostream* stream)
{
    *stream << variant.name;
}

std::string dtypeSuffix(DataType dtype)
{
    return dtype == DataType::HALF ? "Fp16" : "Bf16";
}

std::string dtypeToken(DataType dtype)
{
    return dtype == DataType::HALF ? "fp16" : "bf16";
}

std::vector<WgradVariant> graphVariants()
{
    std::vector<WgradVariant> variants;
    for(size_t index = 0; index < wgradGraphs().size(); ++index)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            variants.push_back({wgradGraphs()[index].name + dtypeSuffix(dtype), dtype, index});
        }
    }
    return variants;
}

std::shared_ptr<Graph> buildWgradGraph(DataType dtype, const WgradProblem& problem)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("gfx950_conv_wgrad")
        .set_io_data_type(dtype)
        .set_intermediate_data_type(dtype)
        .set_compute_data_type(DataType::FLOAT);

    // X [N, C, Hi, Wi] stored NHWC.
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID)
        .set_name("X")
        .set_dim({problem.n, problem.c, problem.hi, problem.wi})
        .set_stride({problem.c * problem.hi * problem.wi, 1, problem.wi * problem.c, problem.c})
        .set_data_type(dtype);
    const auto ho = problem.ho();
    const auto wo = problem.wo();
    // dY [N, K, Ho, Wo] stored NHWK.
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(DY_UID)
        .set_name("DY")
        .set_dim({problem.n, problem.k, ho, wo})
        .set_stride({problem.k * ho * wo, 1, wo * problem.k, problem.k})
        .set_data_type(dtype);

    ConvWgradAttributes attributes;
    attributes.set_name("gfx950_conv_wgrad")
        .set_padding({problem.padH, problem.padW})
        .set_stride({problem.strideH, problem.strideW})
        .set_dilation({problem.dilationH, problem.dilationW});
    auto dw = graph->conv_wgrad(dy, x, attributes);
    // dW [K, C/groups, Y, X] stored KYXC over the group's channels.
    const auto cpg = problem.cpg();
    dw->set_uid(DW_UID)
        .set_name("DW")
        .set_dim({problem.k, cpg, problem.y, problem.x})
        .set_stride({cpg * problem.y * problem.x, 1, problem.x * cpg, cpg})
        .set_output(true)
        .set_data_type(dtype);
    return graph;
}

KnobSetting benchmarkingKnob(bool enabled)
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{enabled ? 1 : 0});
}

std::string candidateCount(size_t candidates)
{
    return std::to_string(candidates) + " candidate(s)";
}

/// The kernel id following @p marker in the last log that carries it, or empty.
std::string loggedKernel(const LogRecorderBase& recorder, const std::string& marker)
{
    std::string kernel;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at == std::string::npos)
        {
            continue;
        }
        const auto begin = at + marker.size();
        kernel = log.message.substr(begin, log.message.find(' ', begin) - begin);
    }
    return kernel;
}

std::string selectedKernel(const LogRecorderBase& recorder)
{
    return loggedKernel(recorder, "' selected kernel ");
}

std::string servedKernel(const LogRecorderBase& recorder)
{
    return loggedKernel(recorder, "' served kernel ");
}

std::string canonicalUuid(const std::string& text)
{
    return ingestor::toString(hipdnn_flatbuffers_sdk::utilities::parseUuid(text));
}

// ---- The shipped catalog, read from the product tree main() loads ----

/// One kernel of the shipped wgrad bundle: what the test needs to pin it and to know which
/// graphs it could serve. An implicit-GEMM kernel bakes none of the geometry fields below
/// (they read 0, the KMD's builder_default for a non-direct kernel), so it admits every
/// problem of its dtype; a direct or helper kernel bakes every one of them.
struct CatalogKernel
{
    std::string id;
    std::string name;
    std::string dtype;
    int64_t family = 0;
    std::string variant;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t cpg = 0;
    int64_t kpg = 0;
    /// Every UED knob, at the kernel's value (an ordinal for a string field).
    std::vector<KnobSetting> knobs;

    /// Whether the kernel's pinned build-time capability is the graph's.
    bool capabilityAdmits(const WgradProblem& problem) const
    {
        if(family == IMPLICIT_GEMM_FAMILY)
        {
            return true;
        }
        const bool geometry = filterH == problem.y && filterW == problem.x
                              && strideH == problem.strideH && strideW == problem.strideW
                              && padH == problem.padH && padW == problem.padW
                              && dilationH == problem.dilationH && dilationW == problem.dilationW;
        return geometry && (cpg == 0 || cpg == problem.cpg()) && (kpg == 0 || kpg == problem.kpg());
    }
};

struct WgradCatalog
{
    std::filesystem::path directory;
    std::vector<CatalogKernel> kernels;
    /// The ordinal of every value of each non-integer KMD field, as the engine numbers
    /// them: the sorted distinct values over the engine's whole catalog, defaults applied.
    std::map<std::string, std::map<std::string, int64_t>> ordinals;

    std::optional<int64_t> ordinal(const std::string& field, const std::string& value) const
    {
        const auto domain = ordinals.find(field);
        if(domain == ordinals.end())
        {
            return std::nullopt;
        }
        const auto found = domain->second.find(value);
        return found == domain->second.end() ? std::nullopt : std::optional{found->second};
    }

    const CatalogKernel* find(const std::string& id) const
    {
        const auto found = std::find_if(
            kernels.begin(), kernels.end(), [&](const auto& kernel) { return kernel.id == id; });
        return found == kernels.end() ? nullptr : &*found;
    }

    /// How many kernels of @p dtype admit @p problem, per direct_variant (VARIANTS order).
    std::array<size_t, VARIANTS.size()> counts(const std::string& dtype,
                                               const WgradProblem& problem) const
    {
        std::array<size_t, VARIANTS.size()> result{};
        for(const auto& kernel : kernels)
        {
            if(kernel.dtype != dtype || kernel.family == HELPER_FAMILY
               || !kernel.capabilityAdmits(problem))
            {
                continue;
            }
            for(size_t i = 0; i < VARIANTS.size(); ++i)
            {
                if(kernel.variant == VARIANTS[i])
                {
                    ++result[i];
                    break;
                }
            }
        }
        return result;
    }

    size_t total(const std::string& dtype, const WgradProblem& problem) const
    {
        size_t sum = 0;
        for(const auto count : counts(dtype, problem))
        {
            sum += count;
        }
        return sum;
    }
};

nlohmann::json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    return nlohmann::json::parse(stream);
}

/// The shipped wgrad bundle under HIPDNN_DESCRIPTOR_RUNTIME_DIR, read once. Empty when it
/// is not there (e.g. no kdp.json/compiled artifacts were packaged for this build, which is
/// expected until the Phase 6 GPU session produces and DVC-registers them).
const WgradCatalog& shippedCatalog()
{
    static const WgradCatalog s_catalog = [] {
        WgradCatalog catalog;
        const std::filesystem::path root
            = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
        std::error_code error;
        if(root.empty() || !std::filesystem::is_directory(root, error))
        {
            return catalog;
        }
        const std::string kmdName = std::string(DESCRIPTOR_PREFIX) + ".kmd.json";
        for(auto it = std::filesystem::recursive_directory_iterator(root, error);
            !error && it != std::filesystem::recursive_directory_iterator();
            it.increment(error))
        {
            if(it->path().filename() == kmdName)
            {
                catalog.directory = it->path().parent_path();
                break;
            }
        }
        if(catalog.directory.empty())
        {
            return catalog;
        }

        const auto kmd = readJson(catalog.directory / kmdName);
        const auto ued
            = readJson(catalog.directory / (std::string(DESCRIPTOR_PREFIX) + ".ued.json"));
        std::vector<std::pair<std::string, nlohmann::json>> fields;
        for(const auto& field : kmd.at("fields"))
        {
            fields.emplace_back(field.at("name").get<std::string>(),
                                field.contains("default_value") ? field.at("default_value")
                                                                : nlohmann::json());
        }

        // Every kernel's metadata with the KMD defaults filled in, in a fixed order.
        std::vector<std::pair<std::string, nlohmann::json>> documents;
        const std::string prefix = std::string(DESCRIPTOR_PREFIX) + "_";
        const std::string suffix = ".kdp.json";
        std::vector<std::filesystem::path> shards;
        // Recursive: the kdp.json shards live under the bundle's artifacts/ (DVC-pulled),
        // not directly beside the kmd.json.
        for(const auto& file :
            std::filesystem::recursive_directory_iterator(catalog.directory, error))
        {
            const auto name = file.path().filename().string();
            if(name.size() > prefix.size() + suffix.size() && name.rfind(prefix, 0) == 0
               && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                shards.push_back(file.path());
            }
        }
        std::sort(shards.begin(), shards.end());
        for(const auto& shard : shards)
        {
            // Held by name: a range over a member of a temporary would dangle.
            const auto document = readJson(shard);
            for(const auto& kernel : document.at("kernelDescriptors"))
            {
                auto metadata = kernel.at("metadata");
                for(const auto& [field, defaultValue] : fields)
                {
                    if(!metadata.contains(field) && !defaultValue.is_null())
                    {
                        metadata[field] = defaultValue;
                    }
                }
                metadata["__id"] = kernel.at("id");
                metadata["__name"] = kernel.at("name");
                documents.emplace_back(kernel.at("id").get<std::string>(), std::move(metadata));
            }
        }

        for(const auto& [field, unused] : fields)
        {
            std::set<std::string> values;
            for(const auto& [id, metadata] : documents)
            {
                if(metadata.contains(field) && metadata.at(field).is_string())
                {
                    values.insert(metadata.at(field).get<std::string>());
                }
            }
            int64_t index = 0;
            for(const auto& value : values)
            {
                catalog.ordinals[field][value] = index++;
            }
        }

        for(const auto& [id, metadata] : documents)
        {
            CatalogKernel kernel;
            kernel.id = canonicalUuid(id);
            kernel.name = metadata.at("__name").get<std::string>();
            kernel.dtype = metadata.at("dtype").get<std::string>();
            kernel.family = metadata.at("kernel_family").get<int64_t>();
            kernel.variant = metadata.at("direct_variant").get<std::string>();
            kernel.filterH = metadata.at("filter_h").get<int64_t>();
            kernel.filterW = metadata.at("filter_w").get<int64_t>();
            kernel.strideH = metadata.at("stride_h").get<int64_t>();
            kernel.strideW = metadata.at("stride_w").get<int64_t>();
            kernel.padH = metadata.at("pad_h").get<int64_t>();
            kernel.padW = metadata.at("pad_w").get<int64_t>();
            kernel.dilationH = metadata.at("dilation_h").get<int64_t>();
            kernel.dilationW = metadata.at("dilation_w").get<int64_t>();
            kernel.cpg = metadata.at("cpg").get<int64_t>();
            kernel.kpg = metadata.at("kpg").get<int64_t>();
            for(const auto& knob : ued.at("knobs"))
            {
                const auto name = knob.get<std::string>();
                const auto& value = metadata.at(name);
                if(value.is_string())
                {
                    kernel.knobs.emplace_back(
                        name, catalog.ordinals.at(name).at(value.get<std::string>()));
                }
                else
                {
                    kernel.knobs.emplace_back(name, value.get<int64_t>());
                }
            }
            catalog.kernels.push_back(std::move(kernel));
        }
        return catalog;
    }();
    return s_catalog;
}

/// X and dY randomized once, dW computed once on the host; the GPU side is reused by every
/// plan of the graph, with dW poisoned before each execution.
struct Reference
{
    std::unique_ptr<GraphTensorBundle> gpu;
    std::unique_ptr<GraphTensorBundle> cpu;
    std::vector<uint8_t> serialized;
};

enum class PinOutcome
{
    SERVED,
    REFUSED,
    FAILED
};

} // namespace

class IntegrationGpuGfx950ConvWgrad
    : public test_utilities::IntegrationGraphVerificationHarness<float, WgradVariant>
{
protected:
    using Base = test_utilities::IntegrationGraphVerificationHarness<float, WgradVariant>;

    void SetUp() override
    {
        Base::SetUp();
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, _deviceId), hipSuccess);
        _arch = properties.gcnArchName;
        if(!hipdnn_plugin_sdk::archMatches(
               _arch, "gfx950", hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        {
            GTEST_SKIP() << "The packaged convolution family requires gfx950";
        }
    }

    void TearDown() override
    {
        if(_workspace != nullptr)
        {
            EXPECT_EQ(hipFree(_workspace), hipSuccess);
            _workspace = nullptr;
        }
        Base::TearDown();
    }

    int64_t engineId() const
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    const WgradGraph& graphOf(const WgradVariant& variant) const
    {
        return wgradGraphs().at(variant.graph.value());
    }

    /// Builds @p graph's plan for this engine with @p knobs. Returns the error text of the
    /// first failing stage, empty on success.
    std::string tryBuild(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        auto result = graph.build_operation_graph(_handle);
        if(result.code != ErrorCode::OK)
        {
            return "build_operation_graph: " + result.err_msg;
        }
        std::vector<int64_t> engineIds;
        result = graph.get_ranked_engine_ids(engineIds);
        if(result.code != ErrorCode::OK)
        {
            return "get_ranked_engine_ids: " + result.err_msg;
        }
        if(std::find(engineIds.begin(), engineIds.end(), engineId()) == engineIds.end())
        {
            return std::string("the plugin did not offer ") + ENGINE_NAME;
        }
        // Hard selection: a preferred-engine hint can fall back silently to another
        // provider.
        result = graph.create_execution_plan_ext(engineId(), knobs);
        if(result.code != ErrorCode::OK)
        {
            return "create_execution_plan_ext: " + result.err_msg;
        }
        result = graph.check_support();
        if(result.code != ErrorCode::OK)
        {
            return "check_support: " + result.err_msg;
        }
        result = graph.build_plans();
        if(result.code != ErrorCode::OK)
        {
            return "build_plans: " + result.err_msg;
        }
        int64_t selected = 0;
        result = graph.get_execution_plan_engine_id(selected);
        if(result.code != ErrorCode::OK || selected != engineId())
        {
            return "the plan's engine is not " + std::string(ENGINE_NAME);
        }
        return {};
    }

    void buildAndCompile(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        const auto error = tryBuild(graph, knobs);
        ASSERT_TRUE(error.empty()) << error;
    }

    Reference makeReference(const WgradVariant& variant, const WgradProblem& problem, unsigned seed)
    {
        // Any built graph of the problem has the same tensors as every plan's graph.
        auto graph = buildWgradGraph(variant.dtype, problem);
        const auto built = graph->build_operation_graph(_handle);
        EXPECT_EQ(built.code, ErrorCode::OK) << built.err_msg;
        Reference reference;
        auto [serialized, serializationError] = graph->to_binary();
        EXPECT_TRUE(serializationError.is_good()) << serializationError.get_message();
        reference.serialized.assign(serialized.begin(), serialized.end());
        const auto wrapper
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                reference.serialized.data(), reference.serialized.size());
        reference.gpu = std::make_unique<GraphTensorBundle>(wrapper.getTensorMap());
        reference.cpu = std::make_unique<GraphTensorBundle>(wrapper.getTensorMap());
        for(const auto uid : {X_UID, DY_UID})
        {
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            reference.gpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
            reference.cpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
        }
        reference.cpu->getTensor(DW_UID).fillWithSentinelValue();
        CpuReferenceGraphExecutor().execute(reference.serialized.data(),
                                            reference.serialized.size(),
                                            reference.cpu->toHostVariantPack());
        return reference;
    }

    /// Executes @p graph's plan with dW poisoned to NaN and compares dW with the reference.
    void executeAndVerify(Graph& graph, DataType dtype, Reference& reference)
    {
        int64_t workspaceSize = -1;
        auto result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0);
        if(static_cast<size_t>(workspaceSize) > _workspaceSize)
        {
            if(_workspace != nullptr)
            {
                ASSERT_EQ(hipFree(_workspace), hipSuccess);
                _workspace = nullptr;
            }
            ASSERT_EQ(hipMalloc(&_workspace, static_cast<size_t>(workspaceSize)), hipSuccess);
            _workspaceSize = static_cast<size_t>(workspaceSize);
        }
        if(_workspace != nullptr)
        {
            // 0xFF bytes are NaN in both fp16 and bf16; a direct pipeline that folds
            // partial slots through this scratch must not read a previous instance's.
            ASSERT_EQ(hipMemsetAsync(_workspace, 0xFF, _workspaceSize, _stream), hipSuccess);
        }
        // Every dW element the kernel does not write stays NaN and fails the comparison.
        reference.gpu->getTensor(DW_UID).fillWithSentinelValue();
        auto buffers = reference.gpu->toDeviceVariantPack();
        result = graph.execute(_handle, buffers, workspaceSize > 0 ? _workspace : nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        reference.gpu->getTensor(DW_UID).markDeviceModified();

        // FP32 accumulation followed by a low-precision store; inputs bounded to [-1,1].
        const float relativeTolerance = dtype == DataType::HALF ? 0.002F : 0.016F;
        const auto validator
            = createAllCloseValidator(frontendToSdkDataType(dtype), 0.001F, relativeTolerance);
        EXPECT_TRUE(validator->allClose(reference.cpu->getTensor(DW_UID),
                                        reference.gpu->getTensor(DW_UID)));
    }

    /// Pins every knob of @p kernel on a fresh graph of @p problem. SERVED: the engine
    /// selected exactly that kernel and its dW matched the reference; REFUSED: no candidate
    /// satisfied the knobs; FAILED: anything else (recorded as a test failure).
    PinOutcome pinAndVerify(const WgradVariant& variant,
                            const WgradProblem& problem,
                            const CatalogKernel& kernel,
                            Reference& reference,
                            IsolatedLogRecorder& recorder,
                            size_t expectedBeforeFiltering)
    {
        recorder.clearLogs();
        auto graph = buildWgradGraph(variant.dtype, problem);
        auto knobs = kernel.knobs;
        knobs.push_back(benchmarkingKnob(false));
        const auto error = tryBuild(*graph, knobs);
        if(!error.empty())
        {
            if(knobsUnsatisfiable(error, recorder))
            {
                return PinOutcome::REFUSED;
            }
            ADD_FAILURE() << kernel.name << ": " << error << "\n"
                          << recorder.getRecordedLogsAsString();
            return PinOutcome::FAILED;
        }
        const auto expected = "at rank 0 from 1 candidate(s) ("
                              + std::to_string(expectedBeforeFiltering) + " before knob filtering)";
        if(selectedKernel(recorder) != kernel.id || !recorder.hasLogContaining(expected))
        {
            ADD_FAILURE() << kernel.name << " pinned, but expected '" << kernel.id << "' "
                          << expected << ": " << recorder.getRecordedLogsAsString();
            return PinOutcome::FAILED;
        }
        const auto failuresBefore
            = ::testing::UnitTest::GetInstance()->current_test_info()->result()->total_part_count();
        executeAndVerify(*graph, variant.dtype, reference);
        const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
        for(int part = failuresBefore; part < result->total_part_count(); ++part)
        {
            if(result->GetTestPartResult(part).failed())
            {
                ADD_FAILURE() << "wrong dW from " << kernel.name << " (" << kernel.id << ")";
                return PinOutcome::FAILED;
            }
        }
        return PinOutcome::SERVED;
    }

    // ---- Winner-record plumbing for the benchmarked plan, as the dgrad suite ----

    std::filesystem::path winnerShardPath() const
    {
        const auto engineDir = hipdnn_data_sdk::utilities::cacheRoot() / "ingestor-winners"
                               / std::string(ingestor::winnerCacheVersion())
                               / hipdnn_data_sdk::utilities::sanitizeForPath(ENGINE_NAME);
        const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(_arch));
        std::vector<std::filesystem::path> shards;
        std::error_code error;
        for(auto it = std::filesystem::recursive_directory_iterator(engineDir, error);
            !error && it != std::filesystem::recursive_directory_iterator();
            it.increment(error))
        {
            const auto& path = it->path();
            if(path.filename() == "winners.jsonl" && path.parent_path().filename() == arch)
            {
                shards.push_back(path);
            }
        }
        EXPECT_LE(shards.size(), 1U) << "more than one winner shard under " << engineDir;
        return shards.empty() ? std::filesystem::path{} : shards.front();
    }

    std::optional<ingestor::WinnerRecord> persistedRanking() const
    {
        std::ifstream stream(winnerShardPath());
        std::string line;
        std::optional<ingestor::WinnerRecord> ranking;
        while(std::getline(stream, line))
        {
            if(const auto entry = ingestor::decodeWinnerRecordLine(line))
            {
                EXPECT_FALSE(ranking.has_value());
                ranking = entry->second;
            }
        }
        return ranking;
    }

    ScopedTestCacheDir _cacheDir{"gfx950-conv-wgrad", ScopedTestCacheDir::Scope::TEST};
    ScopedEnvironmentVariableSetter _forceBenchmarking{"HIPDNN_FORCE_BENCHMARKING", ""};
    ScopedEnvironmentVariableSetter _cacheEnabled{"HIPDNN_DISABLE_CACHE", ""};
    ScopedEnvironmentVariableSetter _shortlistLimit{
        hipdnn_plugin_sdk::INGESTOR_SHORTLIST_LIMIT_ENV_NAME, ""};
    std::string _arch;
    void* _workspace = nullptr;
    size_t _workspaceSize = 0;
};

// ---- Per graph: the unbenchmarked plan and a capped benchmark ----

class IntegrationGpuGfx950ConvWgradAot : public IntegrationGpuGfx950ConvWgrad
{
};

// The whole catalog's count (computed from the shipped catalog's own baked fields, not a
// literal) and the family the fallback score prefers when deciding, plus each
// kernel_family/direct_variant knob slice.
TEST_P(IntegrationGpuGfx950ConvWgradAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    const auto& variant = GetParam();
    const auto& graphCase = graphOf(variant);
    const auto& catalog = shippedCatalog();
    ASSERT_FALSE(catalog.kernels.empty()) << "the shipped wgrad bundle was not found";
    const auto dtype = dtypeToken(variant.dtype);
    const auto total = catalog.total(dtype, graphCase.problem);
    if(total == 0)
    {
        GTEST_SKIP() << "no catalog instance serves " << graphCase.name;
    }
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    auto reference = makeReference(variant, graphCase.problem, 53);

    auto graph = buildWgradGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(false)}));
    EXPECT_TRUE(recorder.hasLogContaining("(" + std::to_string(total) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto* selected = catalog.find(selectedKernel(recorder));
    ASSERT_NE(selected, nullptr) << recorder.getRecordedLogsAsString();
    EXPECT_EQ(selected->dtype, dtype);
    EXPECT_NE(selected->family, HELPER_FAMILY);
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
    graph.reset();

    const auto counts = catalog.counts(dtype, graphCase.problem);
    const auto implicitGemm = counts[0];
    std::vector<std::pair<KnobSetting, size_t>> forced{
        {KnobSetting(KERNEL_FAMILY_KNOB, IMPLICIT_GEMM_FAMILY), implicitGemm},
        {KnobSetting(KERNEL_FAMILY_KNOB, DIRECT_FAMILY), total - implicitGemm},
        // Helpers are never candidates.
        {KnobSetting(KERNEL_FAMILY_KNOB, HELPER_FAMILY), 0}};
    for(size_t i = 1; i < VARIANTS.size(); ++i)
    {
        const auto ordinal = catalog.ordinal(DIRECT_VARIANT_KNOB, VARIANTS[i]);
        ASSERT_TRUE(ordinal) << VARIANTS[i];
        forced.emplace_back(KnobSetting(DIRECT_VARIANT_KNOB, *ordinal), counts[i]);
    }
    for(const auto& [knob, count] : forced)
    {
        recorder.clearLogs();
        auto pinned = buildWgradGraph(variant.dtype, graphCase.problem);
        const auto error = tryBuild(*pinned, {knob, benchmarkingKnob(false)});
        if(count == 0)
        {
            EXPECT_TRUE(knobsUnsatisfiable(error, recorder)) << knob.knobId() << ": " << error;
            continue;
        }
        ASSERT_TRUE(error.empty()) << knob.knobId() << ": " << error;
        EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(count) + " ("
                                              + std::to_string(total) + " before knob filtering)"))
            << knob.knobId() << ": " << recorder.getRecordedLogsAsString();
        ASSERT_NO_FATAL_FAILURE(executeAndVerify(*pinned, variant.dtype, reference));
    }
}

// Benchmarking under HIPDNN_INGESTOR_SHORTLIST_LIMIT measures only the best-ranked
// candidates, and its record serves the next build of the graph.
TEST_P(IntegrationGpuGfx950ConvWgradAot, CappedBenchmarkMatchesCpuReference)
{
    const auto& variant = GetParam();
    const auto& graphCase = graphOf(variant);
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto total = shippedCatalog().total(dtypeToken(variant.dtype), graphCase.problem);
    if(total == 0)
    {
        GTEST_SKIP() << "no catalog instance serves " << graphCase.name;
    }
    auto reference = makeReference(variant, graphCase.problem, 71);
    const auto capped = std::min(BENCHMARK_LIMIT, total);
    _shortlistLimit.setValue(std::to_string(BENCHMARK_LIMIT));

    auto graph = buildWgradGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(true)}));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(capped) + " ("
                                          + std::to_string(total) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
    graph.reset();

    const auto ranking = persistedRanking();
    ASSERT_TRUE(ranking) << "Benchmarking did not persist its measured candidate ranking";
    ASSERT_EQ(ranking->size(), capped);
    for(const auto& candidate : *ranking)
    {
        EXPECT_TRUE(std::isfinite(candidate.timeMs));
        EXPECT_GT(candidate.timeMs, 0.0);
    }
    const std::string winner = ingestor::toString(ranking->front().kernelId);
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel " + winner))
        << recorder.getRecordedLogsAsString();

    recorder.clearLogs();
    auto recreated = buildWgradGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {benchmarkingKnob(true)}));
    EXPECT_EQ(servedKernel(recorder), winner) << recorder.getRecordedLogsAsString();
    const auto count = std::to_string(capped);
    EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + count + " entry(s) for "
                                          + count + " candidate(s)"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*recreated, variant.dtype, reference));
}

// ---- Per dtype: every catalog instance admitted by at least one of wgradGraphs() ----

class IntegrationGpuGfx950ConvWgradAotCatalog : public IntegrationGpuGfx950ConvWgrad
{
};

// Pins every catalog instance of the dtype that at least one of wgradGraphs() admits, on
// every such graph, and checks it against the CPU reference. Unlike the dgrad suite this
// does not assert that every selectable instance ran: wgradGraphs() is a representative,
// not exhaustive, set (see the file header), so an instance whose baked direct geometry
// none of these graphs happens to match is reported, not failed -- reaching every instance
// is Phase 6's job, once the real packaged bundle is available to widen this set against.
TEST_P(IntegrationGpuGfx950ConvWgradAotCatalog, EveryAdmittedInstanceMatchesCpuReference)
{
    const auto& variant = GetParam();
    const auto& catalog = shippedCatalog();
    ASSERT_FALSE(catalog.kernels.empty()) << "the shipped wgrad bundle was not found";
    const auto dtype = dtypeToken(variant.dtype);
    std::vector<const CatalogKernel*> selectable;
    size_t helpers = 0;
    for(const auto& kernel : catalog.kernels)
    {
        if(kernel.dtype != dtype)
        {
            continue;
        }
        if(kernel.family == HELPER_FAMILY)
        {
            ++helpers;
            continue;
        }
        selectable.push_back(&kernel);
    }
    ASSERT_FALSE(selectable.empty());

    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    std::set<std::string> covered;
    size_t executions = 0;
    unsigned int seed = 101;
    for(const auto& graphCase : wgradGraphs())
    {
        SCOPED_TRACE(graphCase.name);
        const auto total = catalog.total(dtype, graphCase.problem);
        if(total == 0)
        {
            continue;
        }
        auto reference = makeReference(variant, graphCase.problem, seed++);
        for(const auto* kernel : selectable)
        {
            if(!kernel->capabilityAdmits(graphCase.problem))
            {
                continue;
            }
            const auto outcome
                = pinAndVerify(variant, graphCase.problem, *kernel, reference, recorder, total);
            if(outcome == PinOutcome::SERVED)
            {
                ++executions;
                covered.insert(kernel->id);
            }
        }
    }

    std::vector<std::string> uncovered;
    for(const auto* kernel : selectable)
    {
        if(covered.count(kernel->id) == 0)
        {
            uncovered.push_back(kernel->name);
        }
    }
    std::cout << "[ wgrad    ] " << variant.name << ": " << covered.size() << " of "
              << selectable.size() << " instances (" << helpers << " helper(s) excluded) ran, "
              << executions << " pinned executions checked against the CPU reference; "
              << uncovered.size() << " instance(s) not reached by this representative graph "
                 "set\n";
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx950ConvWgradAot,
                         ::testing::ValuesIn(graphVariants()),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AotInstances,
                         IntegrationGpuGfx950ConvWgradAotCatalog,
                         ::testing::Values(WgradVariant{"Fp16", DataType::HALF, std::nullopt},
                                           WgradVariant{"Bf16", DataType::BFLOAT16, std::nullopt}),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX950_CONV_WGRAD_AOT
