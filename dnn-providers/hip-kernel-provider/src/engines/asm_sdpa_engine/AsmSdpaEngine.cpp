// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaEngine.hpp"

#include <exception>
#include <map>
#include <string>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/HipEngineFeatures.hpp>

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

/// What this build of the engine was, for a model that claims to have measured it.
///
/// `<provider>/<provider release>/<selector>`. There is no separate library version to
/// name: the ASM kernels are vendored inside this provider rather than linked from an
/// independently versioned library, so the provider version IS the kernel-snapshot
/// version. Bump the trailing selector segment when what this engine does with those
/// kernels changes without the provider version moving -- a different dispatch choice
/// invalidates a trained estimate just as a different kernel would.
///
/// The RELEASE version, deliberately, and not HIP_KERNEL_PROVIDER_VERSION_STRING, which
/// appends the build's git hash. A deployed L1 model records this exact string as RFC 0019
/// §4.1's `trained_against.selector_revision` and the loader refuses one that does not
/// match -- so naming the hash would expire every model on every commit to the repository,
/// including commits that cannot touch this engine, and no model could ever be shipped
/// alongside the source that produced it. What changes these measurements is the vendored
/// kernel snapshot and this engine's dispatch, and both move the version or the selector
/// segment; a hash names the build, which is not the same claim.
///
/// The refusal itself stays: L1 is the score compared across engines, so a stale estimate
/// changes which engine is selected rather than merely misreporting a number.
#define HKP_STRINGIFY_INNER(value) #value
#define HKP_STRINGIFY(value) HKP_STRINGIFY_INNER(value)
constexpr const char* SELECTOR_REVISION
    = "hip-kernel-provider/" HKP_STRINGIFY(HIP_KERNEL_PROVIDER_VERSION_MAJOR) "." HKP_STRINGIFY(
        HIP_KERNEL_PROVIDER_VERSION_MINOR) "." HKP_STRINGIFY(HIP_KERNEL_PROVIDER_VERSION_PATCH)
      "/asm-sdpa-untuned-v1";

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// Resolves AsmSdpaEngine::L1_MODEL_IDS through the descriptor catalog the provider has
/// already parsed (RFC 0019 Open Question 7, RESOLVED). Never throws: with no descriptor
/// tree installed the catalog is empty, nothing resolves, and the engine reports
/// UNAVAILABLE exactly as it did before any model existed.
void bindDeclaredL1Models(hipdnn_plugin_sdk::uhd::EngineModelBinding& binding)
{
    namespace ingestor = hipdnn_plugin_sdk::ingestor;
    std::map<std::string, ingestor::DescriptorId> declared;
    for(const auto& [arch, id] : AsmSdpaEngine::L1_MODEL_IDS)
    {
        try
        {
            declared.emplace(arch, hipdnn_flatbuffers_sdk::utilities::parseUuid(id));
        }
        catch(const std::exception& error)
        {
            // A compiled-in literal, so this is an authoring bug in this file rather than
            // anything a deployment can cause. Logged and skipped rather than thrown: a
            // throw here would take the whole provider down over one unusable model.
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
    for(const auto& [arch, model] : resolved.byArch)
    {
        binding.bind(arch, ingestor::UhdKernelHeuristic::configFrom(model));
    }
    for(const auto& [arch, refusal] : resolved.refused)
    {
        binding.markUnusable(arch, refusal.status, refusal.reason);
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

int64_t AsmSdpaEngine::staticId()
{
    return hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_ID;
}

bool AsmSdpaEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void AsmSdpaEngine::getDetails(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetailsDirect(builder, id(), nullptr);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    auto* dataPtr = detachedBuffer->data();
    handle.storeEngineDetailsDetachedBuffer(dataPtr, std::move(detachedBuffer));
}

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
    if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
    {
        // RFC 0019 §11.2's "A only (opaque)" row: this engine exposes no catalog and no
        // knobs, so it selects its own kernel and has no exact configuration to name.
        result.reason = "ASM SDPA selects its own kernel and predicts no exact configuration";
        return result;
    }
    try
    {
        const auto& device = hipdnn_plugin_sdk::heuristics::predictionDevice(handle.getStream());
        const auto features = hipdnn_plugin_sdk::heuristics::engineFeatures(graph, config, device);
        return _l1Models.predict(
            id(), engineName(), SELECTOR_REVISION, device.gcnArchName, features, evaluate);
    }
    catch(const std::exception& error)
    {
        // An unreadable device or an unbuildable feature row is a missing answer, not a
        // claim of bad performance: applicability is untouched (§11.2).
        result.reason = error.what();
        return result;
    }
}

size_t AsmSdpaEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            const auto bytes = pb->getMaxWorkspaceSize(handle, opGraph, Settings{});
            if(const auto limit = hipdnn_plugin_sdk::heuristics::workspaceLimit(engineConfig);
               limit && bytes > static_cast<uint64_t>(*limit))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "ASM SDPA exceeds the workspace limit");
            }
            return bytes;
        }
    }

    HIPDNN_PLUGIN_LOG_ERROR("AsmSdpaEngine::getMaxWorkspaceSize: no supporting engine found");
    return 0;
}

void AsmSdpaEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    executionContext.setExecutionSettings(Settings{});

    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            if(const auto limit = hipdnn_plugin_sdk::heuristics::workspaceLimit(engineConfig);
               limit
               && pb->getMaxWorkspaceSize(handle, opGraph, Settings{})
                      > static_cast<uint64_t>(*limit))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "ASM SDPA exceeds the workspace limit");
            }
            pb->buildPlan(handle, opGraph, engineConfig, executionContext);
            return;
        }
    }

    HIPDNN_PLUGIN_LOG_ERROR(
        "AsmSdpaEngine::initializeExecutionContext: no supporting engine found");
}

} // namespace asm_sdpa_engine
