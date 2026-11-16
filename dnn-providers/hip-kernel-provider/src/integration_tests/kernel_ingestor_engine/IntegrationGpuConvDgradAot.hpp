// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

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
#include <string_view>
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
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
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

#include "engines/kernel_ingestor_engine/packs/ConvDgradAotGeometry.hpp"

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

/// @file IntegrationGpuConvDgradAot.hpp
/// The GPU suites of the dgrad AOT-instance engines (hipkernel:Gfx950ConvDgradAot,
/// hipkernel:Gfx1250ConvDgradAot), shared: a backward-data graph, the shipped catalog read
/// from the product tree main() loads, the CPU reference, and ConvDgradAotHarness, whose
/// members are the suites' test bodies. Each engine's file supplies its Traits -- the
/// engine, its architecture, its direct_variant families and its graphs, each with how many
/// catalog instances of every family serve it as conv_dgrad_aot_plan in
/// rocke/library/builders/common/convolution_backward_data.py counts them -- and
/// instantiates the suites.
///
/// The UED knobs make every per-graph candidate tuple unique, so pinning all of a kernel's
/// knobs selects exactly that kernel. The catalog suite pins every selectable instance of a
/// dtype on every graph whose capability it could serve, executes each one the engine
/// accepts with dX poisoned (NaN) beforehand, checks it against CpuReferenceGraphExecutor,
/// and requires every instance to have run at least once. The per-graph suite runs an
/// unbenchmarked plan (whole-catalog and per-family counts) and a benchmark under
/// HIPDNN_INGESTOR_SHORTLIST_LIMIT.
namespace hip_kernel_provider::kernel_ingestor_engine::integration::conv_dgrad_aot_test
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

inline constexpr const char* KERNEL_FAMILY_KNOB = "kernel_family";
inline constexpr const char* DIRECT_VARIANT_KNOB = "direct_variant";
inline constexpr int64_t IMPLICIT_GEMM_FAMILY = 0;
inline constexpr int64_t DIRECT_FAMILY = 1;
inline constexpr int64_t HELPER_FAMILY = 2;
inline constexpr size_t BENCHMARK_LIMIT = 64;
/// The NaN tail executeAndVerify places after dX: a kernel's store past dX's end lands in
/// it. Wider than any one workgroup's dX tile row.
inline constexpr size_t DX_GUARD_BYTES = 64 * 1024;
inline constexpr int64_t DY_UID = 1;
inline constexpr int64_t W_UID = 2;
inline constexpr int64_t DX_UID = 3;

using Names = std::vector<std::string>;

inline bool among(const Names& names, const std::string& name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

/// Whether a plan build failed because no candidate carries the requested knob values: the
/// frontend checks each value against the values the engine advertises for the graph, and
/// the engine checks the combination.
inline bool knobsUnsatisfiable(const std::string& error, const LogRecorderBase& recorder)
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

struct DgradProblem
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
    // hipDNN carries no group attribute: groups = dX channels / W dims[1].
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

/// One graph and how many catalog instances of each direct_variant (its engine's
/// Traits::variants() order) serve it in each dtype.
struct DgradGraph
{
    std::string name;
    DgradProblem problem;
    std::vector<size_t> fp16;
    std::vector<size_t> bf16;

    const std::vector<size_t>& counts(DataType dtype) const
    {
        return dtype == DataType::HALF ? fp16 : bf16;
    }
};

struct DgradVariant
{
    std::string name;
    DataType dtype;
    /// Index into the engine's graphs; the catalog suite runs every graph and leaves it
    /// empty.
    std::optional<size_t> graph;
};

// gtest prints a parameter on failure; the name says everything about it.
// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
inline void PrintTo(const DgradVariant& variant, std::ostream* stream)
{
    *stream << variant.name;
}

inline std::string dtypeToken(DataType dtype)
{
    return dtype == DataType::HALF ? "fp16" : "bf16";
}

/// Every graph of @p graphs in each dtype, for the per-graph suite.
inline std::vector<DgradVariant> graphVariants(const std::vector<DgradGraph>& graphs)
{
    std::vector<DgradVariant> variants;
    for(size_t index = 0; index < graphs.size(); ++index)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            variants.push_back({graphs[index].name + dtypeSuffix(dtype), dtype, index});
        }
    }
    return variants;
}

/// The two dtypes, for the catalog suite.
inline std::vector<DgradVariant> dtypeVariants()
{
    return {DgradVariant{"Fp16", DataType::HALF, std::nullopt},
            DgradVariant{"Bf16", DataType::BFLOAT16, std::nullopt}};
}

/// @p alignment is each tensor's declared byte alignment; anything but 16 makes the graph
/// need graph schema 1.3.0.
inline std::shared_ptr<Graph> buildDgradGraph(DataType dtype,
                                              const DgradProblem& problem,
                                              const std::string& name,
                                              int64_t alignment = 16)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(name)
        .set_io_data_type(dtype)
        .set_intermediate_data_type(dtype)
        .set_compute_data_type(DataType::FLOAT);

    const auto ho = problem.ho();
    const auto wo = problem.wo();
    // dY [N, K, Ho, Wo] stored NHWK.
    const auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(DY_UID)
        .set_name("DY")
        .set_dim({problem.n, problem.k, ho, wo})
        .set_stride({problem.k * ho * wo, 1, wo * problem.k, problem.k})
        .set_data_type(dtype)
        .set_alignment(alignment);
    // W [K, C/groups, Y, X] stored KYXC over the group's channels.
    const auto cpg = problem.cpg();
    const auto w = std::make_shared<TensorAttributes>();
    w->set_uid(W_UID)
        .set_name("W")
        .set_dim({problem.k, cpg, problem.y, problem.x})
        .set_stride({cpg * problem.y * problem.x, 1, problem.x * cpg, cpg})
        .set_data_type(dtype)
        .set_alignment(alignment);

    ConvDgradAttributes attributes;
    attributes.set_name(name)
        .set_padding({problem.padH, problem.padW})
        .set_stride({problem.strideH, problem.strideW})
        .set_dilation({problem.dilationH, problem.dilationW});
    const auto dx = graph->conv_dgrad(dy, w, attributes);
    // dX [N, C, Hi, Wi] stored NHWC.
    dx->set_uid(DX_UID)
        .set_name("DX")
        .set_dim({problem.n, problem.c, problem.hi, problem.wi})
        .set_stride({problem.c * problem.hi * problem.wi, 1, problem.wi * problem.c, problem.c})
        .set_output(true)
        .set_data_type(dtype)
        .set_alignment(alignment);
    return graph;
}

inline KnobSetting benchmarkingKnob(bool enabled)
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{enabled ? 1 : 0});
}

/// The kernel id following @p marker (one of detail::KERNEL_SELECTION_LOG_MARKERS) in the
/// last log that carries it, or empty.
inline std::string loggedKernel(const LogRecorderBase& recorder, std::string_view marker)
{
    std::string kernel;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at != std::string::npos)
        {
            kernel = log.message.substr(at + marker.size(), detail::KERNEL_ID_LENGTH);
        }
    }
    return kernel;
}

inline std::string selectedKernel(const LogRecorderBase& recorder)
{
    return loggedKernel(recorder, detail::SELECTED_KERNEL_LOG);
}

inline std::string servedKernel(const LogRecorderBase& recorder)
{
    return loggedKernel(recorder, detail::KERNEL_SELECTION_LOG_MARKERS[1]);
}

inline std::string canonicalUuid(const std::string& text)
{
    return ingestor::toString(hipdnn_flatbuffers_sdk::utilities::parseUuid(text));
}

// ---- The shipped catalog, read from the product tree main() loads ----

/// One kernel of a shipped dgrad bundle: what the test needs to pin it and to know which
/// graphs it could serve.
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

    /// Whether the kernel's pinned build-time capability is the graph's. A direct kernel
    /// bakes its whole geometry (0 channel counts are runtime); an implicit GEMM kernel at
    /// most its filter and stride (0 is runtime), and the engine judges the rest. Kernels
    /// that differ only in baked geometry share every knob, so pinning one on a graph it
    /// cannot serve would select its sibling.
    bool capabilityAdmits(const DgradProblem& problem) const
    {
        if(family == IMPLICIT_GEMM_FAMILY)
        {
            const auto admits
                = [](int64_t baked, int64_t actual) { return baked == 0 || baked == actual; };
            return admits(filterH, problem.y) && admits(filterW, problem.x)
                   && admits(strideH, problem.strideH) && admits(strideW, problem.strideW);
        }
        const bool geometry = filterH == problem.y && filterW == problem.x
                              && strideH == problem.strideH && strideW == problem.strideW
                              && padH == problem.padH && padW == problem.padW
                              && dilationH == problem.dilationH && dilationW == problem.dilationW;
        return geometry && (cpg == 0 || cpg == problem.cpg()) && (kpg == 0 || kpg == problem.kpg());
    }
};

struct DgradCatalog
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
};

inline nlohmann::json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    return nlohmann::json::parse(stream);
}

/// The shipped bundle whose descriptors are named @p prefix (its slug), under
/// HIPDNN_DESCRIPTOR_RUNTIME_DIR. Empty when it is not there.
inline DgradCatalog readShippedCatalog(const std::string& prefix)
{
    DgradCatalog catalog;
    const std::filesystem::path root
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
    std::error_code error;
    if(root.empty() || !std::filesystem::is_directory(root, error))
    {
        return catalog;
    }
    const std::string kmdName = prefix + ".kmd.json";
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
    const auto ued = readJson(catalog.directory / (prefix + ".ued.json"));
    std::vector<std::pair<std::string, nlohmann::json>> fields;
    for(const auto& field : kmd.at("fields"))
    {
        fields.emplace_back(field.at("name").get<std::string>(),
                            field.contains("default_value") ? field.at("default_value")
                                                            : nlohmann::json());
    }

    // Every kernel's metadata with the KMD defaults filled in, in a fixed order.
    std::vector<std::pair<std::string, nlohmann::json>> documents;
    const std::string shardPrefix = prefix + "_";
    const std::string suffix = ".kdp.json";
    std::vector<std::filesystem::path> shards;
    // Recursive: the kdp.json shards live under the bundle's artifacts/ (DVC-pulled), not
    // directly beside the kmd.json.
    for(const auto& file : std::filesystem::recursive_directory_iterator(catalog.directory, error))
    {
        const auto name = file.path().filename().string();
        if(name.size() > shardPrefix.size() + suffix.size() && name.rfind(shardPrefix, 0) == 0
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
                kernel.knobs.emplace_back(name,
                                          catalog.ordinals.at(name).at(value.get<std::string>()));
            }
            else
            {
                kernel.knobs.emplace_back(name, value.get<int64_t>());
            }
        }
        catalog.kernels.push_back(std::move(kernel));
    }
    return catalog;
}

/// dY and W randomized once, dX computed once on the host; the GPU side is reused by every
/// plan of the graph, with dX poisoned before each execution.
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

/// RAII: lowers the loaded provider's own batch-chunk byte limit so a small batch
/// exercises the chunked-launch device path (DeviceTables, views into dY/dX), which no
/// test otherwise reaches -- the kernels' real 32-bit ABI limit needs a multi-GiB image
/// per batch element to trigger naturally.
///
/// Goes through the plugin's exported
/// hipdnnEnginePluginSetConvDgradAotBatchChunkByteLimitForTesting symbol
/// (setProviderDgradBatchChunkByteLimit; both dgrad engines share the limit), not the
/// header's inline batchChunkByteLimitForTesting() directly: this TU and the loaded
/// provider .so are separate binaries, each with its own copy of that function-local
/// static, so setting this TU's copy would leave the provider's own limit at I32_MAX and
/// the chunk path untested.
class ScopedBatchChunkLimit
{
public:
    explicit ScopedBatchChunkLimit(int64_t limit)
    {
        // EXPECT_, not ASSERT_: gtest's ASSERT_* macros expand to a bare `return;`, which
        // a constructor (no return type, not even void) cannot contain.
        EXPECT_EQ(detail::setProviderDgradBatchChunkByteLimit(limit), "");
    }
    ~ScopedBatchChunkLimit()
    {
        // The documented production default (ConvDgradAotGeometry.hpp): production
        // code never touches the limit, so restoring it here is restoring the only value
        // any other test can observe.
        EXPECT_EQ(detail::setProviderDgradBatchChunkByteLimit(conv_dgrad_aot::I32_MAX), "");
    }
    ScopedBatchChunkLimit(const ScopedBatchChunkLimit&) = delete;
    ScopedBatchChunkLimit& operator=(const ScopedBatchChunkLimit&) = delete;
};

/// The harness of one engine's suites. @p Traits supplies:
///   ENGINE_NAME, DESCRIPTOR_PREFIX (the bundle slug), ARCH, GRAPH_NAME, CACHE_TAG,
///   SELECTABLE_INSTANCES and HELPER_INSTANCES per dtype, variants() (every direct_variant
///   a selectable kernel carries, "none" first), singleKernelVariants() and mfmaVariants()
///   (the fallback score's tiers: single-kernel direct pipelines above the MFMA ones above
///   implicit GEMM), and graphs().
template <typename Traits>
class ConvDgradAotHarness
    : public test_utilities::IntegrationGraphVerificationHarness<float, DgradVariant>
{
protected:
    using Base = test_utilities::IntegrationGraphVerificationHarness<float, DgradVariant>;

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
               _arch, Traits::ARCH, hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        {
            GTEST_SKIP() << "The packaged convolution family requires " << Traits::ARCH;
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

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(Traits::ENGINE_NAME);
    }

    /// The engine's shipped bundle, read once.
    static const DgradCatalog& shippedCatalog()
    {
        static const DgradCatalog s_catalog = readShippedCatalog(Traits::DESCRIPTOR_PREFIX);
        return s_catalog;
    }

    static const DgradGraph& graphOf(const DgradVariant& variant)
    {
        return Traits::graphs().at(variant.graph.value());
    }

    static std::shared_ptr<Graph>
        buildGraph(DataType dtype, const DgradProblem& problem, int64_t alignment = 16)
    {
        return buildDgradGraph(dtype, problem, Traits::GRAPH_NAME, alignment);
    }

    /// How many catalog instances of @p variant serve @p graph in @p dtype.
    static size_t count(const DgradGraph& graph, DataType dtype, const std::string& variant)
    {
        const auto& variants = Traits::variants();
        for(size_t i = 0; i < variants.size(); ++i)
        {
            if(variant == variants[i])
            {
                return graph.counts(dtype).at(i);
            }
        }
        return 0;
    }

    static size_t count(const DgradGraph& graph, DataType dtype, const Names& variants)
    {
        size_t total = 0;
        for(const auto& variant : variants)
        {
            total += count(graph, dtype, variant);
        }
        return total;
    }

    static size_t total(const DgradGraph& graph, DataType dtype)
    {
        return count(graph, dtype, Traits::variants());
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
            return std::string("the plugin did not offer ") + Traits::ENGINE_NAME;
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
            return "the plan's engine is not " + std::string(Traits::ENGINE_NAME);
        }
        return {};
    }

    void buildAndCompile(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        const auto error = tryBuild(graph, knobs);
        ASSERT_TRUE(error.empty()) << error;
    }

    Reference makeReference(const DgradVariant& variant, const DgradProblem& problem, unsigned seed)
    {
        // Any built graph of the problem has the same tensors as every plan's graph.
        const auto graph = buildGraph(variant.dtype, problem);
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
        for(const auto uid : {DY_UID, W_UID})
        {
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            reference.gpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
            reference.cpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
        }
        reference.cpu->getTensor(DX_UID).fillWithSentinelValue();
        CpuReferenceGraphExecutor().execute(reference.serialized.data(),
                                            reference.serialized.size(),
                                            reference.cpu->toHostVariantPack());
        return reference;
    }

    /// Executes @p graph's plan with dX poisoned to NaN and compares dX with the reference.
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
            // _workspace (ws_t/ws_coa) is reused, never reallocated, across every instance
            // this harness runs in a test process: an MFMA instance's transpose/reorganize
            // under-writing it would otherwise read a previous sibling's correct weights
            // and pass. 0xFF bytes are NaN in both fp16 and bf16.
            ASSERT_EQ(hipMemsetAsync(_workspace, 0xFF, _workspaceSize, _stream), hipSuccess);
        }
        // Every dX element the kernel does not write stays NaN and fails the comparison.
        auto& dx = reference.gpu->getTensor(DX_UID);
        dx.fillWithSentinelValue();
        auto buffers = reference.gpu->toDeviceVariantPack();
        // dX runs in a copy followed by a NaN guard tail, which a write past its end would
        // change: the comparison below sees only dX itself.
        const auto dxBytes = dx.elementSpace() * dx.elementSize();
        void* guarded = nullptr;
        ASSERT_EQ(hipMalloc(&guarded, dxBytes + DX_GUARD_BYTES), hipSuccess);
        const std::unique_ptr<void, hipError_t (*)(void*)> guardedOwner(guarded, hipFree);
        auto* tail = static_cast<char*>(guarded) + dxBytes;
        ASSERT_EQ(
            hipMemcpyAsync(guarded, buffers.at(DX_UID), dxBytes, hipMemcpyDeviceToDevice, _stream),
            hipSuccess);
        ASSERT_EQ(hipMemsetAsync(tail, 0xFF, DX_GUARD_BYTES, _stream), hipSuccess);
        buffers[DX_UID] = guarded;
        result = graph.execute(_handle, buffers, workspaceSize > 0 ? _workspace : nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(
            hipMemcpyAsync(dx.rawDeviceData(), guarded, dxBytes, hipMemcpyDeviceToDevice, _stream),
            hipSuccess);
        std::vector<unsigned char> guard(DX_GUARD_BYTES);
        ASSERT_EQ(
            hipMemcpyAsync(guard.data(), tail, DX_GUARD_BYTES, hipMemcpyDeviceToHost, _stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        dx.markDeviceModified();
        const auto written
            = std::find_if(guard.begin(), guard.end(), [](unsigned char b) { return b != 0xFF; });
        EXPECT_EQ(written, guard.end())
            << "the plan wrote " << (written - guard.begin()) << " byte(s) past the end of dX";

        // FP32 accumulation followed by a low-precision store; inputs bounded to [-1,1].
        const float relativeTolerance = dtype == DataType::HALF ? 0.002F : 0.016F;
        const auto validator
            = createAllCloseValidator(frontendToSdkDataType(dtype), 0.001F, relativeTolerance);
        EXPECT_TRUE(validator->allClose(reference.cpu->getTensor(DX_UID),
                                        reference.gpu->getTensor(DX_UID)));
    }

    /// Pins every knob of @p kernel on a fresh graph of @p problem. SERVED: the engine
    /// selected exactly that kernel and its dX matched the reference; REFUSED: no candidate
    /// satisfied the knobs; FAILED: anything else (recorded as a test failure).
    PinOutcome pinAndVerify(const DgradVariant& variant,
                            const DgradProblem& problem,
                            const CatalogKernel& kernel,
                            Reference& reference,
                            IsolatedLogRecorder& recorder,
                            size_t expectedBeforeFiltering)
    {
        recorder.clearLogs();
        const auto graph = buildGraph(variant.dtype, problem);
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
                ADD_FAILURE() << "wrong dX from " << kernel.name << " (" << kernel.id << ")";
                return PinOutcome::FAILED;
            }
        }
        return PinOutcome::SERVED;
    }

    // ---- Winner-record plumbing for the benchmarked plan, as the forward suite ----

    std::filesystem::path winnerShardPath() const
    {
        return integration::winnerShardPath(Traits::ENGINE_NAME, _arch);
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

    // ---- The suites' bodies ----

    /// The whole catalog's count, each family's and each pipeline's count through the
    /// kernel_family and direct_variant knobs, and the family an unbenchmarked plan prefers
    /// when the fallback score decides: a single-kernel direct pipeline, then an MFMA
    /// pipeline, then implicit GEMM (planScore).
    void unbenchmarkedPlanServesFromTheWholeCatalog(const DgradVariant& variant)
    {
        const auto& graphCase = graphOf(variant);
        const auto& catalog = shippedCatalog();
        ASSERT_FALSE(catalog.kernels.empty()) << "the shipped dgrad bundle was not found";
        const auto all = total(graphCase, variant.dtype);
        if(all == 0)
        {
            GTEST_SKIP() << "no catalog instance serves " << graphCase.name;
        }
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        auto reference = makeReference(variant, graphCase.problem, 53);

        auto graph = buildGraph(variant.dtype, graphCase.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(false)}));
        // Only "before knob filtering" (the pack-admitted total, independent of ranking) is
        // pinned here: with no knob requested, shadowByArchTier can rank fewer than `all`
        // candidates (one per arch tier), so "at rank 0 from N" does not have to equal it.
        EXPECT_TRUE(
            recorder.hasLogContaining("(" + std::to_string(all) + " before knob filtering)"))
            << recorder.getRecordedLogsAsString();
        const auto* selected = catalog.find(selectedKernel(recorder));
        ASSERT_NE(selected, nullptr) << recorder.getRecordedLogsAsString();
        EXPECT_EQ(selected->dtype, dtypeToken(variant.dtype));
        EXPECT_NE(selected->family, HELPER_FAMILY);
        EXPECT_GT(count(graphCase, variant.dtype, selected->variant), 0U) << selected->name;
        // The family order is the fallback score's (planScore); a trained model installed
        // in the arch root decides by its own ranking, so then only require that the
        // selected kernel comes from a pipeline that serves this graph (above).
        if(recorder.hasLogContaining("decided_by=native"))
        {
            if(count(graphCase, variant.dtype, Traits::singleKernelVariants()) > 0)
            {
                EXPECT_TRUE(selected->family == DIRECT_FAMILY
                            && among(Traits::singleKernelVariants(), selected->variant))
                    << selected->name;
            }
            else if(count(graphCase, variant.dtype, Traits::mfmaVariants()) > 0)
            {
                EXPECT_TRUE(selected->family == DIRECT_FAMILY
                            && among(Traits::mfmaVariants(), selected->variant))
                    << selected->name;
            }
            else
            {
                EXPECT_EQ(selected->family, IMPLICIT_GEMM_FAMILY) << selected->name;
            }
        }
        ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
        graph.reset();

        const auto implicitGemm = count(graphCase, variant.dtype, "none");
        std::vector<std::pair<KnobSetting, size_t>> forced{
            {KnobSetting(KERNEL_FAMILY_KNOB, IMPLICIT_GEMM_FAMILY), implicitGemm},
            {KnobSetting(KERNEL_FAMILY_KNOB, DIRECT_FAMILY), all - implicitGemm},
            // Helpers are never candidates.
            {KnobSetting(KERNEL_FAMILY_KNOB, HELPER_FAMILY), 0}};
        const auto& variants = Traits::variants();
        for(size_t i = 1; i < variants.size(); ++i)
        {
            const auto ordinal = catalog.ordinal(DIRECT_VARIANT_KNOB, variants[i]);
            ASSERT_TRUE(ordinal) << variants[i];
            forced.emplace_back(KnobSetting(DIRECT_VARIANT_KNOB, *ordinal),
                                graphCase.counts(variant.dtype).at(i));
        }
        for(const auto& [knob, served] : forced)
        {
            recorder.clearLogs();
            const auto pinned = buildGraph(variant.dtype, graphCase.problem);
            const auto error = tryBuild(*pinned, {knob, benchmarkingKnob(false)});
            if(served == 0)
            {
                EXPECT_TRUE(knobsUnsatisfiable(error, recorder)) << knob.knobId() << ": " << error;
                continue;
            }
            ASSERT_TRUE(error.empty()) << knob.knobId() << ": " << error;
            EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(served) + " ("
                                                  + std::to_string(all)
                                                  + " before knob filtering)"))
                << knob.knobId() << ": " << recorder.getRecordedLogsAsString();
            ASSERT_NO_FATAL_FAILURE(executeAndVerify(*pinned, variant.dtype, reference));
        }
    }

    /// Benchmarking under HIPDNN_INGESTOR_SHORTLIST_LIMIT measures only the best-ranked
    /// candidates, and its record serves the next build of the graph.
    void cappedBenchmarkMatchesCpuReference(const DgradVariant& variant)
    {
        const auto& graphCase = graphOf(variant);
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        const auto all = total(graphCase, variant.dtype);
        if(all == 0)
        {
            GTEST_SKIP() << "no catalog instance serves " << graphCase.name;
        }
        auto reference = makeReference(variant, graphCase.problem, 71);
        const auto capped = std::min(BENCHMARK_LIMIT, all);
        _shortlistLimit.setValue(std::to_string(BENCHMARK_LIMIT));

        auto graph = buildGraph(variant.dtype, graphCase.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(true)}));
        EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(capped) + " ("
                                              + std::to_string(all) + " before knob filtering)"))
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
        const auto recreated = buildGraph(variant.dtype, graphCase.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {benchmarkingKnob(true)}));
        EXPECT_EQ(servedKernel(recorder), winner) << recorder.getRecordedLogsAsString();
        const auto entries = std::to_string(capped);
        EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + entries
                                              + " entry(s) for " + entries + " candidate(s)"))
            << recorder.getRecordedLogsAsString();
        EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
        ASSERT_NO_FATAL_FAILURE(executeAndVerify(*recreated, variant.dtype, reference));
    }

    /// The batch-chunk device path (DeviceTables, step.views into dY/dX at an n0/n offset)
    /// never otherwise runs: every other case's images fit the kernels' 32-bit ABI whole.
    /// With the chunk limit lowered for the test alone, N=3 of a 64x64x64 fp16 image (one
    /// image 524288 bytes; the limit below fits one but not two) splits into three
    /// single-image launches, each checked against the CPU reference. Only the
    /// implicit-GEMM plan slices, so the family is pinned, and the provider's launch count
    /// must grow by three: an unsliced launch of the graph fails it.
    void aBatchPastTheChunkLimitRunsAsSlicesOnDevice()
    {
        constexpr int64_t CHUNK_LIMIT = 600000;
        const DgradVariant variant{"BatchChunk", DataType::HALF, std::nullopt};
        const DgradProblem problem{3, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1};
        {
            // The geometry's own slicing at this limit, in this binary's copy of it.
            auto& limit = ::hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot::
                batchChunkByteLimitForTesting();
            const auto saved = limit;
            limit = CHUNK_LIMIT;
            const ::hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot::Problem p{
                problem.n,
                problem.c,
                problem.k,
                problem.hi,
                problem.wi,
                problem.y,
                problem.x,
                problem.strideH,
                problem.strideW,
                problem.padH,
                problem.padW,
                problem.dilationH,
                problem.dilationW,
                problem.groups};
            const auto geometry
                = ::hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot::deriveGeometry(p);
            const auto chunks
                = geometry
                      ? ::hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot::batchChunks(
                            p, *geometry)
                      : std::nullopt;
            limit = saved;
            ASSERT_TRUE(chunks);
            ASSERT_EQ(chunks->size(), 3U);
        }
        const ScopedBatchChunkLimit chunkLimit(CHUNK_LIMIT);
        auto reference = makeReference(variant, problem, 97);

        const auto graph = buildGraph(variant.dtype, problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(
            *graph,
            {KnobSetting(KERNEL_FAMILY_KNOB, IMPLICIT_GEMM_FAMILY), benchmarkingKnob(false)}));
        std::string error;
        const auto before = detail::providerDgradImplicitLaunchCount(error);
        ASSERT_GE(before, 0) << error;
        ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
        EXPECT_EQ(detail::providerDgradImplicitLaunchCount(error) - before, 3)
            << "the plan did not run as three slices";
    }

    /// A graph whose tensors declare 32- or 64-byte alignment needs graph schema 1.3.0. The
    /// engine's graph matcher gates alignment (multiples of 16 bytes) and ragged tensors,
    /// and its UED claims 1.3.0, so neither the provider's schema guard
    /// (ApplicabilityManager) nor the plan builder holds it back: the plan is built on this
    /// engine and matches the reference.
    void servesGraphsDeclaringWiderAlignments()
    {
        const DgradProblem problem{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1};
        for(const int64_t alignment : {32, 64})
        {
            SCOPED_TRACE("alignment " + std::to_string(alignment));
            const DgradVariant variant{"Aligned", DataType::HALF, std::nullopt};
            auto reference = makeReference(variant, problem, 61);
            const auto graph = buildGraph(variant.dtype, problem, alignment);
            ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(false)}));
            ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
        }
    }

    /// Every selectable instance of the dtype is pinned on every graph whose capability it
    /// could serve; the engine must accept exactly the counts of the graphs, each accepted
    /// instance must match the reference, and every instance must run at least once.
    void everyInstanceMatchesCpuReference(const DgradVariant& variant)
    {
        const auto& catalog = shippedCatalog();
        ASSERT_FALSE(catalog.kernels.empty()) << "the shipped dgrad bundle was not found";
        std::vector<const CatalogKernel*> selectable;
        size_t helpers = 0;
        for(const auto& kernel : catalog.kernels)
        {
            if(kernel.dtype != dtypeToken(variant.dtype))
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
        EXPECT_EQ(selectable.size(), Traits::SELECTABLE_INSTANCES);
        EXPECT_EQ(helpers, Traits::HELPER_INSTANCES);

        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        std::set<std::string> covered;
        size_t executions = 0;
        unsigned int seed = 101;
        for(const auto& graphCase : Traits::graphs())
        {
            SCOPED_TRACE(graphCase.name);
            auto reference = makeReference(variant, graphCase.problem, seed++);
            std::map<std::string, size_t> served;
            for(const auto* kernel : selectable)
            {
                if(!kernel->capabilityAdmits(graphCase.problem))
                {
                    continue;
                }
                const auto outcome = pinAndVerify(variant,
                                                  graphCase.problem,
                                                  *kernel,
                                                  reference,
                                                  recorder,
                                                  total(graphCase, variant.dtype));
                if(outcome != PinOutcome::SERVED)
                {
                    continue;
                }
                ++executions;
                covered.insert(kernel->id);
                ++served[kernel->variant];
            }
            for(const auto& name : Traits::variants())
            {
                EXPECT_EQ(served[name], count(graphCase, variant.dtype, name)) << name;
            }
        }

        std::vector<std::string> missing;
        for(const auto* kernel : selectable)
        {
            if(covered.count(kernel->id) == 0)
            {
                missing.push_back(kernel->name);
            }
        }
        EXPECT_TRUE(missing.empty()) << missing.size() << " instance(s) never ran, first "
                                     << (missing.empty() ? "" : missing.front());
        std::cout << "[ dgrad    ] " << variant.name << ": " << covered.size() << " of "
                  << selectable.size() << " instances ran, " << executions
                  << " pinned executions checked against the CPU reference\n";
    }

    ScopedTestCacheDir _cacheDir{Traits::CACHE_TAG, ScopedTestCacheDir::Scope::TEST};
    ScopedEnvironmentVariableSetter _forceBenchmarking{"HIPDNN_FORCE_BENCHMARKING", ""};
    ScopedEnvironmentVariableSetter _cacheEnabled{"HIPDNN_DISABLE_CACHE", ""};
    ScopedEnvironmentVariableSetter _shortlistLimit{
        hipdnn_plugin_sdk::INGESTOR_SHORTLIST_LIMIT_ENV_NAME, ""};
    std::string _arch;
    void* _workspace = nullptr;
    size_t _workspaceSize = 0;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration::conv_dgrad_aot_test
