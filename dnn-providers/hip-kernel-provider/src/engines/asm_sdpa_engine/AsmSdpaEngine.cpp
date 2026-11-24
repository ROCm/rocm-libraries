// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaEngine.hpp"
#include "core/Utils.hpp"

#include <exception>
#include <map>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/HipEngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/RankingMetric.hpp>

#include "version.h"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#endif

namespace asm_sdpa_engine
{

namespace
{

/// `<provider>/asm-sdpa/<digest>`: a configure-time digest of everything that decides
/// which forward or backward kernel runs and how (see AsmSdpaSelectorRevision.cmake). A
/// deployed L1 model must record this exact string as `trained_against.selector_revision`
/// (RFC 0019 §4.1) or the loader refuses it.
#ifndef HKP_ASM_SDPA_REVISION
// Without a digest, report a revision no shipped model can match.
#define HKP_ASM_SDPA_REVISION "undetermined"
#endif
constexpr const char* SELECTOR_REVISION = "hip-kernel-provider/asm-sdpa/" HKP_ASM_SDPA_REVISION;

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// Resolves AsmSdpaEngine::L1_MODEL_IDS through the provider's descriptor catalog. Never
/// throws: with no descriptor tree nothing resolves and the engine reports UNAVAILABLE.
/// Each model binds under the metric its own `score.metric` declares.
void bindDeclaredL1Models(hipdnn_plugin_sdk::uhd::EngineModelBinding& binding)
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
        hip_kernel_provider::kernel_ingestor_engine::descriptorCatalog(),
        AsmSdpaEngine::engineName(),
        SELECTOR_REVISION,
        declared);
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

} // namespace

AsmSdpaEngine::AsmSdpaEngine()
{
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    bindDeclaredL1Models(_l1Models);
#endif
}

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

hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT AsmSdpaEngine::getPrediction(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
    hipdnnEnginePredictionKind_t kind,
    bool evaluate) const
{
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    const DeviceQuery device = [&handle]() -> const hipDeviceProp_t& {
        return hipdnn_plugin_sdk::heuristics::predictionDevice(handle.getStream());
    };
#else
    // Without the ingestor getPredictionOn answers UNAVAILABLE before it asks for a device.
    const DeviceQuery device;
#endif
    return getPredictionOn(handle, graph, config, kind, evaluate, device);
}

hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT AsmSdpaEngine::getPredictionOn(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
    hipdnnEnginePredictionKind_t kind,
    bool evaluate,
    const DeviceQuery& deviceOf) const
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
        // The models were trained on graphs the builders serve; any other graph is outside
        // what they describe, so it gets no estimate rather than an extrapolated one.
        if(findApplicablePlanBuilder(handle, graph) == nullptr)
        {
            result.reason = "ASM SDPA serves no plan for this graph";
            return result;
        }
        const auto& device = deviceOf();
        const auto features = hipdnn_plugin_sdk::heuristics::engineFeatures(graph, config, device);
        return _l1Models.predict(id(),
                                 engineName(),
                                 SELECTOR_REVISION,
                                 result.metric,
                                 device.gcnArchName,
                                 features,
                                 evaluate);
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
    static_cast<void>(deviceOf);
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
    if(const auto limit = hipdnn_plugin_sdk::heuristics::workspaceLimit(engineConfig);
       limit && bytes > static_cast<uint64_t>(*limit))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                                       "ASM SDPA exceeds the workspace limit");
    }
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
    if(const auto limit = hipdnn_plugin_sdk::heuristics::workspaceLimit(engineConfig);
       limit
       && pb->getMaxWorkspaceSize(handle, opGraph, executionSettings)
              > static_cast<uint64_t>(*limit))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                                       "ASM SDPA exceeds the workspace limit");
    }
    executionContext.setExecutionSettings(executionSettings);
    pb->buildPlan(handle, opGraph, engineConfig, executionContext);
}

} // namespace asm_sdpa_engine
