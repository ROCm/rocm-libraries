// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipMlopsEngine.hpp"
#include "HipMlopsL1Models.hpp"
#include "core/Utils.hpp"
#include "hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

#include <exception>
#include <map>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/HipEngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/RankingMetric.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#endif

namespace hip_kernel_provider
{

namespace
{

/// `hip-kernel-provider/hip-mlops/<digest>`: a configure-time digest of everything that
/// decides which kernel a graph gets and the code it runs (HipMlopsSelectorRevision.cmake).
/// A deployed L1 model must record this exact string as `trained_against.selector_revision`
/// (RFC 0019 §4.1) or the loader refuses it.
#ifndef HKP_HIP_MLOPS_REVISION
// Without a digest, report a revision no shipped model can match.
#define HKP_HIP_MLOPS_REVISION "undetermined"
#endif
constexpr const char* SELECTOR_REVISION = "hip-kernel-provider/hip-mlops/" HKP_HIP_MLOPS_REVISION;

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// Resolves HIP_MLOPS_L1_MODEL_IDS through the provider's descriptor catalog. Never
/// throws: with no descriptor tree nothing resolves and the engine reports UNAVAILABLE.
/// Each model binds under the metric its own `score.metric` declares.
void bindDeclaredL1Models(hipdnn_plugin_sdk::uhd::EngineModelBinding& binding)
{
    namespace ingestor = hipdnn_plugin_sdk::ingestor;
    std::map<std::string, std::vector<ingestor::DescriptorId>> declared;
    for(const auto& [arch, id] : HIP_MLOPS_L1_MODEL_IDS)
    {
        try
        {
            declared[std::string(arch)].push_back(hipdnn_flatbuffers_sdk::utilities::parseUuid(id));
        }
        catch(const std::exception& error)
        {
            // A bad compiled-in literal is an authoring bug; log and skip rather than take
            // the whole provider down.
            HIPDNN_PLUGIN_LOG_ERROR("hip mlops: declared L1 model id '"
                                    << id << "' for arch '" << arch
                                    << "' is not a UUID: " << error.what());
        }
    }

    const auto resolved = ingestor::resolveDeclaredEnginePredictions(
        kernel_ingestor_engine::descriptorCatalog(),
        hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_NAME,
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

HipMlopsEngine::HipMlopsEngine(int64_t id)
    : _id(id)
{
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    bindDeclaredL1Models(_l1Models);
#endif
}

const char* HipMlopsEngine::selectorRevision()
{
    return SELECTOR_REVISION;
}

int64_t HipMlopsEngine::id() const
{
    return _id;
}

static void initializeHipKernelSettings(
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings)
{
}

bool HipMlopsEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[HipMlopsEngine::isApplicable] ";

    const auto& tensorMap = opGraph.getTensorMap();
    HIP_KERNEL_RETURN_FALSE_IF(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorMap),
                               "Ragged tensors are not supported");
    HIP_KERNEL_RETURN_FALSE_IF(
        !hip_kernel_provider::core::utils::hasOnly16ByteAlignedTensors(tensorMap),
        "All tensors must be aligned to a multiple of 16 bytes");

    // This is wrong if we ever have more than 1 plan builder thats applicable.
    // If this is the case, we should split plan builders accross multiple engines.
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void HipMlopsEngine::getDetails(Handle& handle,
                                const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                                hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobsVector;

    // Collect custom knobs from plan builders
    for(const auto& planBuilder : _planBuilders)
    {
        auto customKnobs = planBuilder->getCustomKnobs(handle, opGraph);

        if(customKnobs.empty())
        {
            continue;
        }

        for(const auto& knobT : customKnobs)
        {
            auto knobOffset = hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knobT);
            knobsVector.push_back(knobOffset);
        }

        // Only one plan builder should be applicable for a given graph and return custom knobs.
        // Stop after finding the first one to avoid duplicates.
        break;
    }

    auto knobs = builder.CreateVector(knobsVector);

    const std::vector<int32_t> behaviorNotesVector{
        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)};
    auto behaviorNotes = builder.CreateVector(behaviorNotesVector);

    auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
        builder, _id, knobs, behaviorNotes);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t HipMlopsEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    Settings baseExecutionSettings;
    initializeHipKernelSettings(engineConfig, baseExecutionSettings);
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            Settings executionSettings = baseExecutionSettings;
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            workspaceSize
                = std::max(workspaceSize,
                           planBuilder->getMaxWorkspaceSize(handle, opGraph, executionSettings));
        }
    }
    return workspaceSize;
}

hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT HipMlopsEngine::getPrediction(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
    hipdnnEnginePredictionKind_t kind,
    bool evaluate) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    EnginePredictionT result;
    result.engine_id = _id;
    result.kind = kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION ? PredictionKind::CONFIGURATION
                                                                 : PredictionKind::ENGINE;
    result.status = PredictionStatus::UNAVAILABLE;
    // Outside the try below: an unregistered metric is a bad request (BAD_PARAM), not a
    // missing answer, and must not be reported as one.
    result.metric = std::string(hipdnn_plugin_sdk::heuristics::rankingMetric(config).name);
    if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
    {
        // RFC 0019 §11.2's "A only (opaque)" row: one kernel per graph and no knobs, so
        // there is no exact configuration to name.
        result.reason = "HIP_MLOPS_ENGINE has no configurations and predicts no exact one";
        return result;
    }
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    try
    {
        const auto& device = hipdnn_plugin_sdk::heuristics::predictionDevice(handle.getStream());
        const auto features = hipdnn_plugin_sdk::heuristics::engineFeatures(graph, config, device);
        return _l1Models.predict(_id,
                                 hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_NAME,
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
    result.reason = "UHD engine prediction requires a build with HIPDNN_ENABLE_KERNEL_INGESTOR";
    return result;
#endif
}

void HipMlopsEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    Settings executionSettings;
    initializeHipKernelSettings(engineConfig, executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            break;
        }
    }

    executionContext.setExecutionSettings(executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
            break;
        }
    }
}

void HipMlopsEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>> planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}
