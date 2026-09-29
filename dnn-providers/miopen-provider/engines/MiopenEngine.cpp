// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenEngine.hpp"
#include "plans/MiopenBatchnormPlanBuilder.hpp"

#include <exception>
#include <filesystem>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/HipEngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/RankingMetric.hpp>

#include "version.h"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
// The ONLY thing this provider takes from the ingestor: the by-UUID lookup that resolves
// a declared L1 model (RFC 0019 Open Question 7, RESOLVED). Nothing in miopen-provider is
// descriptor-backed -- no UED, no kernel packs, no catalog to rank -- and nothing else
// here may grow a dependency on this header. The loader is header-only, so the cost is a
// compile-time include and no link-time coupling; when the ingestor is not built the
// declaration simply resolves to nothing and every engine reports UNAVAILABLE.
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <nlohmann/json.hpp>
#endif

namespace miopen_plugin
{

namespace
{

auto createBenchmarkingKnob(flatbuffers::FlatBufferBuilder& builder)
{
    return hipdnn_plugin_sdk::KnobFactory::createIntKnob(
        builder, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, "Enable benchmarking", 0, 0, 1, 1, {});
}

void handleBenchmarkingKnobSetting(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenSettings& executionSettings)
{
    if(!engineConfig.hasKnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME))
    {
        return;
    }

    const auto& knobSetting
        = engineConfig.getKnobSettingByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);

    if(knobSetting.valueType() != hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Benchmarking knob setting value is not an integer. Type: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameKnobValue(
                    knobSetting.valueType())));
    }

    auto value = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>().value();
    executionSettings.setBenchmarkingEnabled(value != 0);
}

void initializeMiopenSettings(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenSettings& executionSettings)
{
    if(engineConfig.isValid())
    {
        handleBenchmarkingKnobSetting(engineConfig, executionSettings);
    }
    else
    {
        HIPDNN_PLUGIN_LOG_WARN("Engine config is invalid");
    }

    // Applied outside the isValid() branch and after the knob: an unset override leaves
    // the above untouched, =1 forces on even for an invalid config (the plain-execute
    // path), and =0 forces off a knob-enabled run.
    if(const auto forced = hipdnn_plugin_sdk::benchmarkingOverrideFromEnv())
    {
        executionSettings.setBenchmarkingEnabled(*forced);
    }
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// What this provider asks MIOpen for, as a revision. Bump it whenever that changes
/// without the provider's semantic version or MIOpen's own moving -- a different find or
/// tuning mode, a different solution-selection call, a changed DB policy -- because every
/// deployed L1 model was measured under the old request and must stop matching.
constexpr const char* MIOPEN_SELECTOR_POLICY_REVISION = "untuned-v1";

/// What this build of @p engineName was, for a model that claims to have measured it.
///
/// `miopen-provider/<major.minor.patch>/<engine>-<policy revision>/miopen-<x.y.z>`.
/// Every segment is deliberately versioned, and nothing in it changes merely because the
/// provider was rebuilt from another commit: a model trained at one commit must keep
/// loading at the next, or committing the model would invalidate it. The commit stays
/// provenance only (MIOPEN_PROVIDER_VERSION_STRING, the plugin's reported version).
///   - The provider's semantic version, from version.json.
///   - The engine name: MIOPEN_ENGINE and MIOPEN_ENGINE_DETERMINISTIC are different
///     selectors over different solvers, so a model measured on one says nothing about
///     the other.
///   - MIOPEN_SELECTOR_POLICY_REVISION, for a change in what we ask MIOpen.
///   - The MIOpen library version, which is what actually picks the solution and so
///     decides this engine's throughput. Queried at run time rather than compiled in,
///     because the provider links whatever MIOpen the machine installed.
///
/// A deployed L1 model records this exact string as RFC 0019 §4.1's
/// `trained_against.selector_revision`, and the loader refuses one that does not match:
/// L1 is the score compared across engines, so a stale estimate changes which engine is
/// selected rather than merely misreporting a number.
std::string selectorRevision(const std::string& engineName)
{
    size_t major = 0;
    size_t minor = 0;
    size_t patch = 0;
    std::string library = "unknown";
    if(miopenGetVersion(&major, &minor, &patch) == miopenStatusSuccess)
    {
        library = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
    else
    {
        // Not fatal: the revision stays well-formed and simply will not match a model
        // recorded on a machine that could answer, which is the safe direction.
        HIPDNN_PLUGIN_LOG_WARN("miopen: cannot read the MIOpen library version; no L1 model "
                               "will match this build");
    }
    return "miopen-provider/" MIOPEN_PROVIDER_VERSION "/" + engineName + "-"
           + MIOPEN_SELECTOR_POLICY_REVISION + "/miopen-" + library;
}

/// Every descriptor file this provider reads, parsed once for the process.
///
/// MIOpen ships no descriptor tree of its own, so the only roots are the ones an operator
/// names: HIPDNN_DESCRIPTOR_DIR, then HIPDNN_DESCRIPTOR_RUNTIME_DIR and every
/// HIPDNN_DESCRIPTOR_PATH entry. With none named the catalog is empty, nothing resolves,
/// and every engine reports UNAVAILABLE -- absence is not an error (RFC 0019 §11.2).
const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& descriptorCatalog()
{
    static const hipdnn_plugin_sdk::ingestor::DescriptorCatalog s_catalog = [] {
        auto named = hipdnn_plugin_sdk::ingestor::environmentDescriptorRoots();
        std::vector<std::filesystem::path> roots;
        if(!named.replacement.empty())
        {
            roots.push_back(std::move(named.replacement));
        }
        for(auto& additional : named.additional)
        {
            roots.push_back(std::move(additional));
        }
        return hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(roots);
    }();

    return s_catalog;
}

/// Resolves the ids this engine's container declared for it. Never throws: an id nothing
/// deploys, a model whose provenance fails, and no descriptor tree at all are all
/// "no estimate", never a failure to construct the engine.
///
/// Every id is bound under `default`: one model may cover several architectures (its
/// artifact's `training_arches` says which, checked per query), so the declaration names
/// a model per metric and never per architecture. The loader files each resolved model
/// under the metric its own `score.metric` declares; a model deployed under the id
/// declared for another metric is refused rather than bound, since answering in a metric
/// its id was never declared for would make the declaration mean nothing.
void bindDeclaredL1Models(hipdnn_plugin_sdk::uhd::EngineModelBinding& binding,
                          const std::string& engineName,
                          const std::string& selectorRevision,
                          const std::map<std::string, std::string>& l1ModelIds)
{
    namespace ingestor = hipdnn_plugin_sdk::ingestor;
    std::vector<ingestor::DescriptorId> ids;
    std::map<ingestor::DescriptorId, std::string> declaredMetric;
    for(const auto& [metric, id] : l1ModelIds)
    {
        // Both checks guard compiled-in literals, so a failure is an authoring bug in
        // MiopenContainer rather than anything a deployment can cause. Logged and skipped
        // rather than thrown: a throw here would cost the whole provider over one
        // unusable model.
        if(hipdnn_data_sdk::utilities::findRankingMetric(metric) == nullptr)
        {
            HIPDNN_PLUGIN_LOG_ERROR("miopen: engine '" << engineName << "' declares L1 model id '"
                                                       << id << "' for unregistered metric '"
                                                       << metric << "'");
            continue;
        }
        try
        {
            const auto parsed = hipdnn_flatbuffers_sdk::utilities::parseUuid(id);
            ids.push_back(parsed);
            declaredMetric.emplace(parsed, metric);
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_ERROR("miopen: engine '" << engineName << "' declared L1 model id '"
                                                       << id << "' for metric '" << metric
                                                       << "' is not a UUID: " << error.what());
        }
    }
    if(ids.empty())
    {
        return;
    }

    const auto resolved = ingestor::resolveDeclaredEnginePredictions(
        descriptorCatalog(), engineName, selectorRevision, {{"default", ids}});
    for(const auto& [metric, byArch] : resolved.byMetric)
    {
        for(const auto& [arch, model] : byArch)
        {
            if(const auto& declared = declaredMetric.at(model.id); declared != metric)
            {
                binding.markUnusable(
                    metric,
                    arch,
                    hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::INVALID,
                    "model " + ingestor::toString(model.id)
                        + " is deployed under the id declared for metric '" + declared
                        + "' but its score.metric is '" + metric + "'");
                continue;
            }
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

MiopenEngine::MiopenEngine(int64_t id,
                           std::string name,
                           std::map<std::string, std::string> l1ModelIds)
    : _id(id)
    , _name(std::move(name))
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    , _selectorRevision(selectorRevision(_name))
    , _l1ModelIds(std::move(l1ModelIds))
#endif
{
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    bindDeclaredL1Models(_l1Models, _name, _selectorRevision, _l1ModelIds);
#else
    // Without the ingestor there is no UHD runtime to bind a declared model into, so the
    // engine reports no estimate -- the same outcome as an id nothing deploys.
    static_cast<void>(l1ModelIds);
#endif
}

int64_t MiopenEngine::id() const
{
    return _id;
}

bool MiopenEngine::isApplicable(
    HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
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

void MiopenEngine::getDetails(HipdnnMiopenHandle& handle,
                              const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                              hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto benchmarkingKnob = createBenchmarkingKnob(builder);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobsVector;
    knobsVector.push_back(benchmarkingKnob);

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

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT MiopenEngine::getPrediction(
    HipdnnMiopenHandle& handle,
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
        // RFC 0019 §11.2's "A only (opaque)" row: MIOpen runs its own solver, so it has
        // no exact configuration of ours to name.
        result.reason = "MIOpen selects its own solution and predicts no exact configuration";
        return result;
    }
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    try
    {
        const auto& device = hipdnn_plugin_sdk::heuristics::predictionDevice(handle.getStream());
        const auto features = hipdnn_plugin_sdk::heuristics::engineFeatures(graph, config, device);
        auto prediction = _l1Models.predict(
            _id, _name, _selectorRevision, result.metric, device.gcnArchName, features, evaluate);
        // A description names the model an author must publish, deployed or not: the
        // id this engine declares for the requested metric is what a first model is
        // promoted under, so collection records it even before anything is installed.
        // The binding only carries `uhd_id` itself once a model resolved (and then it is
        // this same id).
        if(const auto declared = _l1ModelIds.find(result.metric);
           !evaluate && declared != _l1ModelIds.end() && !prediction.binding_json.empty())
        {
            auto binding = nlohmann::json::parse(prediction.binding_json);
            if(!binding.contains("uhd_id"))
            {
                binding["uhd_id"] = declared->second;
                prediction.binding_json = binding.dump();
            }
        }
        return prediction;
    }
    catch(const std::exception& error)
    {
        // An unreadable device or an unbuildable feature row is a missing answer, not a
        // claim of bad performance: applicability is untouched (§11.2).
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

size_t MiopenEngine::getMaxWorkspaceSize(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    HipdnnMiopenSettings baseExecutionSettings;
    initializeMiopenSettings(engineConfig, baseExecutionSettings);

    size_t workspaceSize = 0;

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            HipdnnMiopenSettings executionSettings = baseExecutionSettings;
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            workspaceSize
                = std::max(workspaceSize,
                           planBuilder->getMaxWorkspaceSize(handle, opGraph, executionSettings));
        }
    }

    return workspaceSize;
}

void MiopenEngine::initializeExecutionContext(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    HipdnnMiopenSettings executionSettings;
    initializeMiopenSettings(engineConfig, executionSettings);

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

void MiopenEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::
                        IPlanBuilder<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>>
        planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}
