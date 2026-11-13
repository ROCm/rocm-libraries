// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#endif

#include "../IntegrationGraphVerificationHarness.hpp"
#include "ScopedPluginLogCapture.hpp"
#include "engines/kernel_ingestor_engine/packs/SdpaLayoutTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

// A test name's dtype token: "Fp16" for HALF, "Bf16" otherwise (the 16-bit conv engines).
inline std::string dtypeSuffix(hipdnn_frontend::DataType dtype)
{
    return dtype == hipdnn_frontend::DataType::HALF ? "Fp16" : "Bf16";
}

// The engine log's count of the candidates a plan build ranked.
inline std::string candidateCount(size_t candidates)
{
    return std::to_string(candidates) + " candidate(s)";
}

// The two forms in which the frontend saves a built plan.
enum class SaveForm
{
    GRAPH_AND_PLAN,
    PLAN_ONLY
};

// An ingestor engine and the one architecture its packs serve.
struct ServedEngine
{
    const char* name;
    const char* arch;
};

namespace detail
{

// The plugin log line of a kernel that plan build selects without benchmarking.
inline constexpr const char* SELECTED_KERNEL_LOG = "' selected kernel ";

// The plugin log line of each path that selects or benchmarks a kernel at plan build.
inline constexpr std::array<const char*, 4> KERNEL_SELECTION_LOG_MARKERS = {
    SELECTED_KERNEL_LOG, "' served kernel ", "' will benchmark ", "benchmarking selected kernel"};

// The plugin log line of a new provider container.
inline constexpr const char* CREATING_CONTAINER_LOG = "Creating Container";

// The selection line spells the kernel id in lowercase with hyphens, in 36 characters.
inline constexpr std::size_t KERNEL_ID_LENGTH = 36;

// The already-loaded provider's own shared-object path. Does not load a second copy.
inline std::filesystem::path loadedProviderPluginPath()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                            + pluginTarget.filename().string()
                            + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
    return std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path()
        / pluginFile);
}

// Clears the process-wide kpack module cache of the loaded provider.
// Makes the loaded provider read kpack code bytes from disk again.
// The provider keeps one module for each archive, entry and architecture for the life of
// the process. A case that must read code bytes again calls this function first.
// The function finds the provider that the harness loaded. It does not load a second copy.
// Returns a description of the failure, or an empty string.
inline std::string resetProviderModuleCaches()
{
    const auto pluginPath = loadedProviderPluginPath();

    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        return "the provider at " + pluginPath.string()
               + " is not loaded, so there are no resident modules to reset";
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* reset = reinterpret_cast<void (*)()>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginResetKpackModuleCacheForTesting"));
    if(reset == nullptr)
    {
        hipdnn_data_sdk::utilities::closeLibrary(library);
        return "the provider at " + pluginPath.string()
               + " exports no hipdnnEnginePluginResetKpackModuleCacheForTesting; it was "
                 "built without HIPDNN_ENABLE_KERNEL_INGESTOR, or the test-only reset "
                 "hook was removed";
    }

    reset();

    // Releases only the reference that this function took. The harness keeps the provider loaded.
    hipdnn_data_sdk::utilities::closeLibrary(library);
    return {};
}

// Sets the loaded provider's own per-image byte limit for the dgrad AOT engines' shared
// batch chunker (ConvDgradAotGeometry.hpp::batchChunkByteLimitForTesting()), so a
// small batch exercises the chunked-launch device path without a multi-GiB image. The
// limit lives in a function-local static: a test in another TU that includes the header
// and mutates it directly holds its own copy, never the plugin's, which is why this goes
// through the plugin's exported hipdnnEnginePluginSetConvDgradAotBatchChunkByteLimit
// ForTesting symbol instead. Returns a description of the failure, or an empty string.
inline std::string setProviderDgradBatchChunkByteLimit(int64_t limit)
{
    const auto pluginPath = loadedProviderPluginPath();

    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        return "the provider at " + pluginPath.string()
               + " is not loaded, so there is no batch chunk limit to set";
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* setLimit = reinterpret_cast<void (*)(int64_t)>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginSetConvDgradAotBatchChunkByteLimitForTesting"));
    if(setLimit == nullptr)
    {
        hipdnn_data_sdk::utilities::closeLibrary(library);
        return "the provider at " + pluginPath.string()
               + " exports no "
                 "hipdnnEnginePluginSetConvDgradAotBatchChunkByteLimitForTesting; it was "
                 "built without HIPDNN_ENABLE_KERNEL_INGESTOR, or the test-only hook was removed";
    }

    setLimit(limit);

    hipdnn_data_sdk::utilities::closeLibrary(library);
    return {};
}

// The loaded provider's count of hipkernel:Gfx950ConvDgradAot implicit-GEMM launches, one
// per batch slice (exported as
// hipdnnEnginePluginConvDgradAotImplicitLaunchCountForTesting). Writes a
// description of the failure to error and returns -1 if the hook is not there.
inline int64_t providerDgradImplicitLaunchCount(std::string& error)
{
    const auto pluginPath = loadedProviderPluginPath();
    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        error = "the provider at " + pluginPath.string() + " is not loaded";
        return -1;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* count = reinterpret_cast<int64_t (*)()>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginConvDgradAotImplicitLaunchCountForTesting"));
    const auto value = count == nullptr ? -1 : count();
    if(count == nullptr)
    {
        error = "the provider at " + pluginPath.string()
                + " exports no hipdnnEnginePluginConvDgradAotImplicitLaunchCountForTesting";
    }
    hipdnn_data_sdk::utilities::closeLibrary(library);
    return value;
}

// Replaces handle with a new hipDNN handle that uses stream.
// Destroy every graph built with the old handle before you call this function.
// When no other handle is alive, the next plugin handle gets a new provider container.
// Wrap the call in ASSERT_NO_FATAL_FAILURE.
inline void replaceHandleWithAFreshOne(hipdnnHandle_t& handle, hipStream_t stream)
{
    ASSERT_EQ(resetProviderModuleCaches(), "");

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
}

// Saves the built plan of graph in form.
inline std::pair<std::vector<uint8_t>, hipdnn_frontend::Error>
    saveInForm(const hipdnn_frontend::graph::Graph& graph, SaveForm form)
{
    return form == SaveForm::GRAPH_AND_PLAN ? graph.to_binary() : graph.to_compiled_plan_binary();
}

// Loads bytes that saveInForm wrote in form into graph.
inline hipdnn_frontend::Error loadInForm(hipdnn_frontend::graph::Graph& graph,
                                         hipdnnHandle_t handle,
                                         const std::vector<uint8_t>& bytes,
                                         SaveForm form)
{
    return form == SaveForm::GRAPH_AND_PLAN ? graph.from_binary(handle, bytes)
                                            : graph.from_compiled_plan_binary(handle, bytes);
}

// Fails the test for each kernel selection or benchmarking line in recorder.
inline void
    expectNoKernelSelectionLogged(const hipdnn_test_sdk::utilities::LogRecorderBase& recorder)
{
    for(const char* marker : KERNEL_SELECTION_LOG_MARKERS)
    {
        EXPECT_FALSE(recorder.hasLogContaining(marker))
            << "found '" << marker << "'. Captured logs:\n"
            << recorder.getRecordedLogsAsString();
    }
}

// The production descriptors that HIPDNN_DESCRIPTOR_RUNTIME_DIR names, loaded once. Every
// case reads the same root, which main() sets before any case runs.
inline const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& runtimeCatalog()
{
    static const auto s_catalog = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(
        std::filesystem::path(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")));
    return s_catalog;
}

// Calls visit for each kernel that runtimeCatalog() serves engine with: each kernel of each
// pack that is not conflicted, belongs to engine.name and lists engine.arch.
template <typename Visit>
void forEachServedKernel(const ServedEngine& engine, Visit&& visit)
{
    const auto& catalog = runtimeCatalog();
    for(const auto& entry : catalog.packs)
    {
        const auto& pack = entry.second;
        const auto& arch = pack.descriptor.arch;
        const auto owner = catalog.engines.find(pack.descriptor.engineId);
        if(pack.conflicted || owner == catalog.engines.end()
           || owner->second.descriptor.name != engine.name
           || std::find(arch.begin(), arch.end(), engine.arch) == arch.end())
        {
            continue;
        }
        for(const auto& kernel : pack.descriptor.kernels)
        {
            visit(kernel);
        }
    }
}

// A kernel metadata field and the value a case expects it to carry.
using MetadataKey = std::vector<std::pair<std::string, hipdnn_plugin_sdk::ingestor::MetadataValue>>;

inline std::string describeKey(const MetadataKey& key)
{
    std::string text;
    for(const auto& [field, value] : key)
    {
        text += (text.empty() ? "" : " ") + field + "=";
        std::visit(
            [&text](const auto& alternative) {
                using Alternative = std::decay_t<decltype(alternative)>;
                if constexpr(std::is_same_v<Alternative, std::string>)
                {
                    text += alternative;
                }
                else if constexpr(std::is_arithmetic_v<Alternative>)
                {
                    text += std::to_string(alternative);
                }
            },
            value);
    }
    return text;
}

// Whether kernel carries every field of key at the key's value.
inline bool carriesKey(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel,
                       const MetadataKey& key)
{
    return std::all_of(key.begin(), key.end(), [&kernel](const auto& field) {
        const auto value = kernel.metadata.find(field.first);
        return value != kernel.metadata.end() && value->second == field.second;
    });
}

// The id of the one kernel engine serves that carries key, as the selection line spells it.
// Records a failure naming key and the number of kernels carrying it, and returns nullopt,
// when that is not exactly one.
inline std::optional<std::string> onlyServedKernelCarrying(const ServedEngine& engine,
                                                           const MetadataKey& key)
{
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
    {
        ADD_FAILURE() << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production "
                         "descriptors to read the expected kernel id from";
        return std::nullopt;
    }

    std::vector<std::string> ids;
    forEachServedKernel(engine,
                        [&key, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
                            if(carriesKey(kernel, key))
                            {
                                ids.push_back(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
                            }
                        });

    if(ids.size() != 1)
    {
        ADD_FAILURE() << ids.size() << " " << engine.name << " kernel(s) for " << engine.arch
                      << " carry " << describeKey(key) << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                      << "'; exactly one must";
        return std::nullopt;
    }
    return ids.front();
}

// Whether knobs advertise value for the integer knob knobId.
inline bool advertises(const std::vector<hipdnn_frontend::Knob>& knobs,
                       const std::string& knobId,
                       int64_t value)
{
    const auto knob
        = std::find_if(knobs.begin(), knobs.end(), [&](const hipdnn_frontend::Knob& candidate) {
              return candidate.knobId() == knobId;
          });
    if(knob == knobs.end())
    {
        return false;
    }
    const auto* constraint
        = dynamic_cast<const hipdnn_frontend::IntConstraint*>(knob->constraint());
    return constraint != nullptr && constraint->getValidValues().count(value) > 0;
}

// The kernel id that engine's selection line names, or nullopt when no plan was built.
inline std::optional<std::string>
    selectedKernelId(const ServedEngine& engine,
                     const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::string marker = std::string("engine '") + engine.name + SELECTED_KERNEL_LOG;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at != std::string::npos)
        {
            return log.message.substr(at + marker.size(), KERNEL_ID_LENGTH);
        }
    }
    return std::nullopt;
}

// Prints `KNOB_SELECTED <case> <kernelId>` when the case ends, however it ends, so a run's
// log carries the selected kernel of every case, including a failing one.
class SelectedKernelReport
{
public:
    SelectedKernelReport(const ServedEngine& engine,
                         std::string caseName,
                         const ScopedPluginLogCapture& capture)
        : _engine(engine)
        , _caseName(std::move(caseName))
        , _capture(capture)
    {
    }

    ~SelectedKernelReport()
    {
        std::cout << "KNOB_SELECTED " << _caseName << " "
                  << selectedKernelId(_engine, _capture.recorder()).value_or("none") << '\n';
    }

    SelectedKernelReport(const SelectedKernelReport&) = delete;
    SelectedKernelReport& operator=(const SelectedKernelReport&) = delete;
    SelectedKernelReport(SelectedKernelReport&&) = delete;
    SelectedKernelReport& operator=(SelectedKernelReport&&) = delete;

private:
    ServedEngine _engine;
    std::string _caseName;
    const ScopedPluginLogCapture& _capture;
};

} // namespace detail

// A fixture that runs only on engine.arch. There the engine, its packs and its kernels are
// all expected present, and any of them missing fails the case.
template <typename TestCaseType>
class ServedEngineHarness
    : public test_utilities::IntegrationGraphVerificationHarness<float, TestCaseType>
{
protected:
    using Harness = test_utilities::IntegrationGraphVerificationHarness<float, TestCaseType>;

    // cacheTag names a cache root private to ONE test case. A winner record another case
    // measured would replace the heuristic order, and the selected kernel with it.
    ServedEngineHarness(const ServedEngine& engine, const std::string& cacheTag)
        : _engine(engine)
        , _cacheDir(cacheTag, hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST)
    {
    }

    void SetUp() override
    {
        Harness::SetUp();
        if(this->IsSkipped() || this->HasFatalFailure())
        {
            return;
        }

        const auto arch = hip_kernel_provider_common::getDeviceString(this->_stream);
        if(arch != _engine.arch)
        {
            GTEST_SKIP() << _engine.name << " ships for " << _engine.arch
                         << " only; this device is " << arch;
        }
    }

    // Offsets the seed by UID so Q, K and V carry different data: K and V share a shape, and
    // an operand swap between them is invisible when their contents agree.
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first,
                                   Harness::DEFAULT_MIN,
                                   Harness::DEFAULT_MAX,
                                   seed + static_cast<unsigned int>(tensorPair.first));
        }
    }

    int64_t engineId() const
    {
        return hipdnn_data_sdk::utilities::engineNameToId(_engine.name);
    }

    // On engine.arch the engine's absence from the ranked list is a missing engine or pack,
    // not a skip.
    void buildOperationGraphTheEngineOffersToServe(hipdnn_frontend::graph::Graph& graph)
    {
        auto result = graph.build_operation_graph(this->_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        const bool offered
            = result.code == hipdnn_frontend::ErrorCode::OK
              && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId())
                     != rankedEngineIds.end();
        ASSERT_TRUE(offered) << _engine.name << " did not offer to serve this graph on "
                             << _engine.arch << " (" << result.err_msg
                             << "). Its packed descriptors load through "
                             << "HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                             << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                             << "'; this configuration must pack the production descriptors for "
                             << _engine.arch << ".";
    }

private:
    ServedEngine _engine;
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir;
};

#ifdef HIPDNN_ENABLE_SDPA

// Every spelling of the masks the dense attention engines serve.
enum class SdpaMask
{
    NO_MASK,
    BOUNDS_TOP_LEFT,
    BOUNDS_BOTTOM_RIGHT,
    CAUSAL_MASK_FLAG,
    CAUSAL_MASK_BOTTOM_RIGHT_FLAG,
    // left_bound W, right_bound 0: a causal window of W keys.
    WINDOW_TOP_LEFT,
    WINDOW_BOTTOM_RIGHT
};

// The memory layout of a [B, H, S, D] SDPA tensor, shared with the unit suites.
using kernel_ingestor_engine::testing::SdpaLayout;

// One SDPA forward graph: its tensors' shapes and the attributes a case varies.
struct IngestorSdpaGraphSpec
{
    hipdnn_frontend::DataType dataType;
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
    SdpaMask mask;
    int64_t batch;
    int64_t seqQ;
    int64_t seqKv;
    // The window in keys for the WINDOW_* masks; 0 otherwise.
    int64_t window = 0;
    // NOT_SET leaves mma_core_mode at its default.
    hipdnn_frontend::DataType mmaCoreMode = hipdnn_frontend::DataType::NOT_SET;
    // Leaves attn_scale_value unset, so the engine must apply 1.0 (no scaling).
    bool omitScale = false;
    // Sets attn_scale_value to this instead of 1/sqrt(headSize), so a kernel that hard-wires
    // the default scale fails the comparison. Ignored when omitScale is set.
    std::optional<float> scaleOverride = std::nullopt;
    // Binds the stats (LSE) output.
    bool withStats = false;
    // The layouts of K and V. Q and O are BSHD.
    SdpaLayout kLayout = SdpaLayout::BSHD;
    SdpaLayout vLayout = SdpaLayout::BSHD;
};

struct SdpaGraph
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> output;
    // The stats (LSE) output, or nullptr for a graph that binds none.
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> stats;
};

namespace detail
{

// The catalog's `dtype` spelling of a graph dtype; the catalog carries only these two.
inline std::string catalogDtypeName(hipdnn_frontend::DataType dataType)
{
    return dataType == hipdnn_frontend::DataType::HALF ? "FP16" : "BF16";
}

// The catalog's `causal` value of a mask: every spelling but NO_MASK is causal.
inline int64_t catalogCausal(SdpaMask mask)
{
    return mask == SdpaMask::NO_MASK ? 0 : 1;
}

// The forward SDPA tolerance a SdpaFwd bundle is graded at when its engine's config names no
// override.
inline float sdpaForwardTolerance(hipdnn_frontend::DataType dataType)
{
    return dataType == hipdnn_frontend::DataType::HALF
               ? hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::half>()
               : hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<
                     hipdnn_data_sdk::types::bfloat16>();
}

using kernel_ingestor_engine::testing::sdpaStrides;

inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    makeSdpaTensor(int64_t uid,
                   const std::string& name,
                   hipdnn_frontend::DataType dataType,
                   int64_t batch,
                   int64_t heads,
                   int64_t seq,
                   int64_t headSize,
                   SdpaLayout layout)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({batch, heads, seq, headSize})
        .set_stride(sdpaStrides(layout, heads, seq, headSize))
        .set_data_type(dataType);
    return tensor;
}

// A single SDPA forward node with BSHD Q and O, K and V in the shape's layouts, and the
// softmax scale 1/sqrt(D), set explicitly unless the shape omits it. O is laid out
// explicitly: the frontend's default for an unset output is packed BHSD, which the engines
// decline. A shape with stats binds the stats output as the frontend does,
// FLOAT [B, Hq, Sq, 1], packed.
inline SdpaGraph buildSdpaGraph(const std::string& name, const IngestorSdpaGraphSpec& shape)
{
    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    constexpr int64_t STATS_UID = 5;

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name(name)
        .set_io_data_type(shape.dataType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto q = makeSdpaTensor(Q_UID,
                            "Q",
                            shape.dataType,
                            shape.batch,
                            shape.queryHeads,
                            shape.seqQ,
                            shape.headSize,
                            SdpaLayout::BSHD);
    auto k = makeSdpaTensor(K_UID,
                            "K",
                            shape.dataType,
                            shape.batch,
                            shape.kvHeads,
                            shape.seqKv,
                            shape.headSize,
                            shape.kLayout);
    auto v = makeSdpaTensor(V_UID,
                            "V",
                            shape.dataType,
                            shape.batch,
                            shape.kvHeads,
                            shape.seqKv,
                            shape.headSize,
                            shape.vLayout);

    hipdnn_frontend::graph::SdpaAttributes attributes;
    attributes.set_name(name);
    if(!shape.omitScale)
    {
        attributes.set_attn_scale(
            shape.scaleOverride.value_or(1.0f / std::sqrt(static_cast<float>(shape.headSize))));
    }
    switch(shape.mask)
    {
    case SdpaMask::NO_MASK:
        break;
    case SdpaMask::BOUNDS_TOP_LEFT:
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
        break;
    case SdpaMask::BOUNDS_BOTTOM_RIGHT:
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
        break;
    case SdpaMask::CAUSAL_MASK_FLAG:
        attributes.causal_mask = true;
        break;
    case SdpaMask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG:
        attributes.causal_mask_bottom_right = true;
        break;
    case SdpaMask::WINDOW_TOP_LEFT:
        attributes.set_diagonal_band_left_bound(shape.window);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
        break;
    case SdpaMask::WINDOW_BOTTOM_RIGHT:
        attributes.set_diagonal_band_left_bound(shape.window);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
        break;
    default:
        throw std::invalid_argument("buildSdpaGraph: unhandled SdpaMask value");
    }
    if(shape.mmaCoreMode != hipdnn_frontend::DataType::NOT_SET)
    {
        attributes.set_mma_core_mode(shape.mmaCoreMode);
    }
    if(shape.withStats)
    {
        attributes.set_generate_stats(true);
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    auto o = outputs[0];
    o->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(shape.dataType)
        .set_dim({shape.batch, shape.queryHeads, shape.seqQ, shape.headSize})
        .set_stride(sdpaStrides(SdpaLayout::BSHD, shape.queryHeads, shape.seqQ, shape.headSize));

    auto stats = outputs[1];
    if(stats != nullptr)
    {
        stats->set_uid(STATS_UID)
            .set_name("Stats")
            .set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim({shape.batch, shape.queryHeads, shape.seqQ, 1})
            .set_stride({shape.queryHeads * shape.seqQ, shape.seqQ, 1, 1});
    }
    return {graph, o, stats};
}

} // namespace detail

#endif // HIPDNN_ENABLE_SDPA

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration
