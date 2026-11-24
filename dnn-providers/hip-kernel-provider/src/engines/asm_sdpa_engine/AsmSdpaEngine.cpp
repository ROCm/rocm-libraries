// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaEngine.hpp"
#include "core/Utils.hpp"

#include <exception>
#include <map>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/HipEngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/RankingMetric.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#endif

namespace asm_sdpa_engine
{

namespace
{

/// The full revision, prefix and digest, as AsmSdpaSelectorRevision.cmake computes it at
/// configure time over everything that decides which forward or backward kernel runs and
/// how. A deployed L1 model must record this exact string as
/// `trained_against.selector_revision` (RFC 0019 §4.1) or the loader refuses it.
#ifndef HKP_ASM_SDPA_REVISION
// Without a revision, report one no shipped model can match.
#define HKP_ASM_SDPA_REVISION "undetermined"
#endif
constexpr const char* SELECTOR_REVISION = HKP_ASM_SDPA_REVISION;

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// Resolves AsmSdpaEngine::L1_MODEL_IDS through the provider's descriptor catalog. Never
/// throws: with no descriptor tree nothing resolves and the engine reports UNAVAILABLE.
/// Each model binds under the metric its own `score.metric` declares.
void bindDeclaredL1Models(hipdnn_plugin_sdk::uhd::EngineModelBinding& binding,
                          const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& catalog)
{
    namespace ingestor = hipdnn_plugin_sdk::ingestor;
    std::map<std::string, std::vector<ingestor::DescriptorId>> declared;
    for(const auto& [arch, id] : AsmSdpaEngine::L1_MODEL_IDS)
    {
        try
        {
            declared[std::string(arch)].push_back(hipdnn_flatbuffers_sdk::utilities::parseUuid(id));
        }
        catch(const std::exception& error)
        {
            // A bad compiled-in literal is an authoring bug; log and skip rather than
            // take the whole provider down.
            HIPDNN_PLUGIN_LOG_ERROR("asm sdpa: declared L1 model id '"
                                    << id << "' for arch '" << arch
                                    << "' is not a UUID: " << error.what());
        }
    }

    const auto resolved = ingestor::resolveDeclaredEnginePredictions(
        catalog, AsmSdpaEngine::engineName(), SELECTOR_REVISION, declared);
    for(const auto& [metric, byArch] : resolved.byMetric)
    {
        for(const auto& [arch, model] : byArch)
        {
            binding.bind(metric, arch, ingestor::UhdKernelHeuristic::configFrom(model));
        }
    }
    for(const auto& [metric, byArch] : resolved.refused)
    {
        for(const auto& [arch, refusal] : byArch)
        {
            binding.markUnusable(metric, arch, refusal.status, refusal.reason);
        }
    }
}
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

/// Refuses a plan that needs @p bytes of workspace when the configuration sets
/// `global.workspace_size_limit` below it. The limit is a property of the engine
/// configuration, not of the graph, so it cannot be an applicability decline: the plan is
/// refused with the status the SDK's ingestor engines use for the same condition, and is
/// never built.
/// @throws HipdnnPluginException INVALID_VALUE when the plan needs more than the limit, or
///         when the limit itself is not a nonnegative integer.
void declineOverWorkspaceLimit(
    uint64_t bytes, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config)
{
    const auto limit = hipdnn_plugin_sdk::heuristics::workspaceLimit(config);
    if(limit && bytes > static_cast<uint64_t>(*limit))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "ASM SDPA plan needs " + std::to_string(bytes)
                + " bytes of workspace, over the workspace limit of " + std::to_string(*limit)
                + " bytes");
    }
}

} // namespace

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
AsmSdpaEngine::AsmSdpaEngine()
    : AsmSdpaEngine(hip_kernel_provider::kernel_ingestor_engine::descriptorCatalog())
{
}

AsmSdpaEngine::AsmSdpaEngine(const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& catalog)
{
    bindDeclaredL1Models(_l1Models, catalog);
}
#else
AsmSdpaEngine::AsmSdpaEngine() = default;
#endif

void AsmSdpaEngine::addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder)
{
    _planBuilders.emplace_back(std::move(planBuilder));
}

int64_t AsmSdpaEngine::id() const
{
    return staticId();
}

const char* AsmSdpaEngine::selectorRevision()
{
    return SELECTOR_REVISION;
}

int64_t AsmSdpaEngine::staticId()
{
    return hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_ID;
}

const IPlanBuilder* AsmSdpaEngine::findApplicablePlanBuilder(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            return pb.get();
        }
    }
    return nullptr;
}

bool AsmSdpaEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return findApplicablePlanBuilder(handle, opGraph) != nullptr;
}

void AsmSdpaEngine::getDetails(Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               hipdnnPluginConstData_t& detailsOut) const
{
    // Knobs come from the builder that serves the graph, so a forward graph never
    // advertises the backward builder's knobs.
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;
    if(const auto* pb = findApplicablePlanBuilder(handle, opGraph))
    {
        knobs = pb->getCustomKnobs(handle, opGraph);
    }
    const std::vector<int32_t> behaviorNotes{
        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)};
    hip_kernel_provider::core::utils::storeEngineDetails(
        handle, id(), knobs, behaviorNotes, detailsOut);
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
const hipDeviceProp_t& AsmSdpaEngine::predictionDevice(Handle& handle) const
{
    return hipdnn_plugin_sdk::heuristics::predictionDevice(handle.getStream());
}
#endif

hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT AsmSdpaEngine::getPrediction(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
    hipdnnEnginePredictionKind_t kind,
    bool evaluate) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    EnginePredictionT result;
    result.engine_id = id();
    result.kind = kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION ? PredictionKind::CONFIGURATION
                                                                 : PredictionKind::ENGINE;
    result.status = PredictionStatus::UNAVAILABLE;
    // Outside the try below: an unregistered metric is a bad request (BAD_PARAM), not a
    // missing answer, and must not be reported as one.
    result.metric = std::string(hipdnn_plugin_sdk::heuristics::rankingMetric(config).name);
    if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
    {
        // RFC 0019 §11.2's "A only (opaque)" row: this engine exposes no catalog and no
        // knobs, so it selects its own kernel and has no exact configuration to name.
        result.reason = "ASM SDPA selects its own kernel and predicts no exact configuration";
        return result;
    }
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    try
    {
        // The models describe graphs the builders serve in the three dense layouts the
        // corpus covers (bshd, bhsd and kv_bhsd), so a graph no builder serves gets no
        // estimate rather than an extrapolated one. A graph a builder serves in any other
        // stride pattern (a padded view, a fused-QKV slice) is still estimated, by
        // extrapolation: the models never saw it.
        if(findApplicablePlanBuilder(handle, graph) == nullptr)
        {
            result.reason = "ASM SDPA serves no plan for this graph";
            return result;
        }
        const auto& device = predictionDevice(handle);
        // The gfx950 model reads the device's memory clock and bandwidth. A device that
        // reports no clock feeds it a value far outside anything it trained on and gets a
        // number that is not a rank, so it gets no estimate.
        if(device.memoryClockRate <= 0)
        {
            result.reason = "ASM SDPA needs the device's memory clock, which this device "
                            "reports as "
                            + std::to_string(device.memoryClockRate) + " kHz";
            return result;
        }
        const auto features = hipdnn_plugin_sdk::heuristics::engineFeatures(graph, config, device);
        return _l1Models.predict(id(),
                                 engineName(),
                                 SELECTOR_REVISION,
                                 result.metric,
                                 device.gcnArchName,
                                 features,
                                 evaluate);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        // A malformed request is a fault, not a missing estimate: a bad
        // `global.workspace_size_limit` is rejected here as it is where the plan is sized.
        if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
        {
            throw;
        }
        result.reason = error.what();
        return result;
    }
    catch(const std::exception& error)
    {
        // A missing answer, not a claim of bad performance; applicability is untouched
        // (§11.2).
        result.reason = error.what();
        return result;
    }
#else
    static_cast<void>(handle);
    static_cast<void>(graph);
    static_cast<void>(evaluate);
    result.reason = "UHD engine prediction requires a build with HIPDNN_ENABLE_KERNEL_INGESTOR";
    return result;
#endif
}

size_t AsmSdpaEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    const auto* pb = findApplicablePlanBuilder(handle, opGraph);
    if(pb == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "AsmSdpaEngine::getMaxWorkspaceSize: no plan builder applies to the graph");
    }

    // Size with the same knob settings initializeExecutionContext builds the plan
    // with, so the workspace matches the plan the engine config selects.
    Settings executionSettings;
    pb->initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
    const auto bytes = pb->getMaxWorkspaceSize(handle, opGraph, executionSettings);
    declineOverWorkspaceLimit(bytes, engineConfig);
    return bytes;
}

void AsmSdpaEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    const auto* pb = findApplicablePlanBuilder(handle, opGraph);
    if(pb == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "AsmSdpaEngine::initializeExecutionContext: no plan builder applies to the graph");
    }

    // buildPlan reads the knob settings from the context, so they are stored first.
    Settings executionSettings;
    pb->initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
    declineOverWorkspaceLimit(pb->getMaxWorkspaceSize(handle, opGraph, executionSettings),
                              engineConfig);
    executionContext.setExecutionSettings(executionSettings);
    pb->buildPlan(handle, opGraph, engineConfig, executionContext);
}

} // namespace asm_sdpa_engine
