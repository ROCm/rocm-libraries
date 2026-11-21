// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/UnpulledModelsRecord.hpp"

// The build's HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS, set by CMake on every census file
// that includes this header (src/tests/CMakeLists.txt).
#ifndef HKP_REQUIRE_DVC_DESCRIPTORS
#error "add the including file to the HKP_REQUIRE_DVC_DESCRIPTORS list in src/tests/CMakeLists.txt"
#endif

/**
 * @file ModelBindingTestSupport.hpp
 * @brief What the ModelBinding census suites share: the trained models a UED binds, and
 *        whether the packed tree the census reads must hold them.
 *
 * A model whose `model.bin` was never pulled (`dvc pull` not run) is packed without its
 * UHD, with a warning, unless the build sets HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS;
 * hkp_pack then also writes hkp-unpulled-models.record at the packed arch root, naming each
 * UHD it left out. A census accepts a bound model whose UHD the packed tree lacks only in
 * a non-strict build and only when that record names it. Any other absence (a packer that
 * lost the UHD or its model.bin) fails in every build, as does an absence in a strict
 * build. A bound model whose UHD the tree holds must load and its model file must be
 * present (the loader only warns about an absent one), whatever the mode.
 *
 * Loading is not enough: the loader's pre-flight reads the UHD, not the weights, and every
 * UHD records `<adapter>.hash`, the SHA-256 of its `model.bin`, which the adapter verifies
 * only when the scorer is built. A foreign or corrupted `model.bin` therefore loads and is
 * refused at run time with a log line, selection falling back silently. Every bound model
 * also builds its scorer here the way the runtime does (L2: UhdKernelHeuristic::tryCreate,
 * as makeEngine()'s ranker does on its first multi-candidate selection; L1:
 * UhdKernelHeuristic::configFrom and uhd::prediction_detail::model(), as GenericEngine's
 * EngineModelBinding does), and a census that passes ProbeGraphs also asks each model once
 * about a graph its engine serves (L1: an AVAILABLE, finite prediction through
 * EngineModelBinding::predict(); L2: a ranking of the packs' kernels that leads with a
 * positive score), host-only.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

using ModelMap = std::map<std::string,
                          std::map<std::string, hipdnn_plugin_sdk::ingestor::HeuristicDescriptor>>;

/// The model @p models holds for @p metric under arch key @p archKey, or null.
inline const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor*
    findModel(const ModelMap& models, const std::string& metric, const std::string& archKey)
{
    const auto byArch = models.find(metric);
    if(byArch == models.end())
    {
        return nullptr;
    }
    const auto model = byArch->second.find(archKey);
    return model == byArch->second.end() ? nullptr : &model->second;
}

/// The ids a UED role map binds for arch key @p archKey.
inline std::vector<std::string> boundIds(
    const std::map<std::string, std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId>>& role,
    const std::string& archKey)
{
    std::vector<std::string> ids;
    const auto byArch = role.find(archKey);
    if(byArch != role.end())
    {
        for(const auto& id : byArch->second)
        {
            ids.push_back(hipdnn_flatbuffers_sdk::utilities::formatUuid(id));
        }
    }
    return ids;
}

/// Whether the packed tree this census reads holds a UHD file naming `id`. A model left
/// out of the pack (its `model.bin` never pulled) has no UHD there.
inline bool packedTreeHoldsUhd(const std::string& id)
{
    const std::string root = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
    if(root.empty())
    {
        return true; // no explicit root to inspect: treat the model as shipped
    }
    for(const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if(!entry.is_regular_file() || entry.path().string().find(".uhd.json") == std::string::npos)
        {
            continue;
        }
        const std::ifstream file(entry.path());
        std::stringstream text;
        text << file.rdbuf();
        if(text.str().find(id) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

/// Whether hkp_pack's skip record in the packed tree names `id` as a UHD it left out for
/// an un-pulled model. The packer writes the record only when it skips, so a UHD or model
/// the packer lost for any other reason is in no record.
inline bool packedTreeRecordsUnpulled(const std::string& id)
{
    const std::string root = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
    return !root.empty() && unpulledRecordNames(root, id);
}

/// Whether the case must check that the models `ids` name loaded. False only when the
/// packed tree lacks a UHD that the pack's skip record names: a model left out because
/// its `model.bin` was never pulled, which the pack warned about. A UHD absent for any
/// other reason (nothing in the record), or absent in a strict build (where the pack
/// should have failed), is a failure.
inline bool packedTreeObligesLoad(const std::vector<std::string>& ids, const char* role)
{
    for(const auto& id : ids)
    {
        if(packedTreeHoldsUhd(id))
        {
            continue;
        }
        // The macro is a compile-time 0 or 1, so one arm is dead in each build.
        // NOLINTNEXTLINE(misc-redundant-expression)
        if(HKP_REQUIRE_DVC_DESCRIPTORS != 0)
        {
            ADD_FAILURE() << "the " << role << " model " << id
                          << " has no UHD in the packed tree, and "
                             "HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS is ON; the pack "
                             "should have failed";
        }
        else if(!packedTreeRecordsUnpulled(id))
        {
            ADD_FAILURE() << "the " << role << " model " << id
                          << " has no UHD in the packed tree, and " << UNPULLED_RECORD
                          << " does not name it, so its model.bin was not left out as "
                             "un-pulled: the packer lost the UHD or its model";
        }
        else
        {
            std::cout << "[  NOTE    ] the " << role << " model " << id
                      << " has no UHD in the packed tree (its model.bin was not pulled, as "
                      << UNPULLED_RECORD
                      << " records); the pack warned and the engine runs with that role "
                         "off\n";
        }
        return false;
    }
    return true;
}

/// The loader only warns about a UHD whose model file is absent and keeps the model, so
/// "it loaded" does not mean its `model.bin` shipped. A packer that drops the file from the
/// packed tree would otherwise pass: check the file the loaded descriptor names.
inline void
    expectModelArtifactPresent(const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                               const char* role)
{
    std::error_code error;
    const auto path = std::filesystem::path(model.baseDir) / model.modelArtifactPath;
    EXPECT_TRUE(std::filesystem::is_regular_file(path, error))
        << "the " << role << " model " << hipdnn_flatbuffers_sdk::utilities::formatUuid(model.id)
        << " loaded, but its model file " << path.string()
        << " is absent from the packed tree: the packer lost it";
}

/// `<id> (<role>, arch <arch key>)`, for a failure message.
inline std::string describeModel(const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                                 const char* role,
                                 const std::string& archKey)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(model.id) + " (" + role + ", arch "
           + archKey + ")";
}

/// The L2 scorer of @p model, built the way makeEngine()'s ranker builds it on the first
/// multi-candidate selection (UhdKernelHeuristic::tryCreate, which reads `model.bin`, checks
/// its SHA-256 against the digest the UHD records, and checks the feature contract against
/// the UED's knobs and the KMD's fields). The loader's pre-flight reads only the descriptor,
/// so a `model.bin` whose bytes differ from what the UHD recorded loads and is refused only
/// here, with a log line and a fall back to priority order. Null, after a failure, when it
/// cannot be built.
inline std::shared_ptr<hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic>
    expectRankerScorerBuilds(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                             const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                             const std::string& archKey)
{
    auto scorer = hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic::tryCreate(
        model, set.engine.name, set.engine.knobs, set.schema.fieldNames());
    EXPECT_NE(scorer, nullptr)
        << "the model " << describeModel(model, "L2 sort_kernel_catalog", archKey)
        << " cannot be brought up as a scorer (the log line above says why: a model.bin that "
           "does not hash to the UHD's recorded digest, malformed weights, or a feature "
           "contract the UED or KMD does not meet); kernels rank by priority";
    return scorer;
}

/// The L1 model of @p model, bound and compiled the way GenericEngine does (validateBinding,
/// then prediction_detail::model(), which builds the adapter and so checks `model.bin`
/// against the SHA-256 the UHD records). Null, after a failure, unless it is AVAILABLE.
inline std::shared_ptr<const hipdnn_plugin_sdk::uhd::prediction_detail::Model>
    expectPredictorScorerBuilds(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                                const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                                const std::string& metric,
                                const std::string& archKey)
{
    const auto name = describeModel(model, "L1 predict_engine", archKey);
    const auto config = hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic::configFrom(model);
    try
    {
        hipdnn_plugin_sdk::uhd::prediction_detail::validateBinding(
            config, set.engine.name, archKey, metric);
    }
    catch(const std::exception& error)
    {
        ADD_FAILURE() << "the model " << name << " does not match its binding: " << error.what();
        return nullptr;
    }
    auto compiled = hipdnn_plugin_sdk::uhd::prediction_detail::model(config);
    if(compiled->status != hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::AVAILABLE)
    {
        ADD_FAILURE() << "the model " << name
                      << " cannot be brought up as a scorer: " << compiled->reason
                      << " (the engine reports no prediction)";
        return nullptr;
    }
    return compiled;
}

/// A graph a census asks a bound model about, built with the test graph helpers; nothing runs
/// it. A census passes the graphs its engines can serve, and each model is asked about the
/// first one its engine's graph match accepts on the model's architecture.
struct ProbeGraph
{
    std::string name;
    std::function<flatbuffers::FlatBufferBuilder()> build;
};

using ProbeGraphs = std::vector<ProbeGraph>;

/// A device of architecture @p archKey, with the properties the models' `device.*` features
/// read set to plausible values: the probe checks that a model answers, not what a board does.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties probeDevice(const std::string& archKey)
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties device;
    device.gcnArchName = archKey;
    device.warpSize = archKey.rfind("gfx9", 0) == 0 ? 64 : 32;
    device.multiProcessorCount = 64;
    device.totalGlobalMem = size_t{16} << 30;
    device.sharedMemPerBlock = size_t{64} << 10;
    return device;
}

/// A probe graph the engine of @p set serves on an architecture, with what its graph match
/// bound.
struct ServedProbe
{
    std::string name;
    std::unique_ptr<GraphFixture> fixture;
    hipdnn_plugin_sdk::ingestor::BoundTokens bound;
};

/// The @p probes whose graph the engine's graph match accepts on @p archKey, in order (an
/// engine declaring no graph match serves every graph). Fails when there are none: a bound
/// model no probe reaches is one no census decides, so the census gets a graph the engine
/// serves.
inline std::vector<ServedProbe>
    expectServedProbes(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                       const std::string& archKey,
                       const ProbeGraphs& probes)
{
    std::vector<ServedProbe> served;
    for(const auto& probe : probes)
    {
        auto fixture = std::make_unique<GraphFixture>(probe.build(), probeDevice(archKey));
        hipdnn_plugin_sdk::ingestor::BoundTokens bound;
        if(!set.engine.graphMatchNativeSymbol.empty())
        {
            auto matched = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
                set.engine.graphMatchNativeSymbol)(fixture->context());
            if(!matched.has_value())
            {
                continue;
            }
            bound = std::move(*matched);
        }
        served.push_back({probe.name, std::move(fixture), std::move(bound)});
    }
    if(served.empty())
    {
        ADD_FAILURE() << "the " << set.engine.name << " engine serves none of this census's "
                      << probes.size() << " probe graphs on " << archKey
                      << ", so its bound models cannot be asked once; add a graph it serves";
    }
    return served;
}

/// Asks @p model, the engine's L1 for @p archKey, about the probe graphs the engine serves,
/// through the binding GenericEngine::getPrediction() asks: the same problem features plus
/// the graph match's bindings, the same compile and the same scorer. The model must answer
/// AVAILABLE with a finite value for at least one of them (a model answers only for the graphs
/// its features cover: UNAVAILABLE for the others is the engine's normal report); when none,
/// each graph's reason is named.
inline void expectPredictorAnswers(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                                   const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                                   const std::string& metric,
                                   const std::string& archKey,
                                   const ProbeGraphs& probes)
{
    const auto served = expectServedProbes(set, archKey, probes);
    if(served.empty())
    {
        return;
    }
    hipdnn_plugin_sdk::uhd::EngineModelBinding binding;
    binding.bind(
        metric, archKey, hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic::configFrom(model));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);
    std::ostringstream refusals;
    for(const auto& probe : served)
    {
        const auto context = probe.fixture->context();
        auto features = hipdnn_plugin_sdk::heuristics::engineFeatures(
            context.graph, config, context.deviceProperties);
        hipdnn_plugin_sdk::ingestor::detail::bindGraphMatchBindings(
            features, probe.bound, !set.slots.empty());
        const auto prediction
            = binding.predict(0,
                              set.engine.name,
                              hipdnn_plugin_sdk::ingestor::engineSelectorRevision(set),
                              metric,
                              context.deviceProperties.gcnArchName,
                              features,
                              /*evaluate=*/true);
        if(prediction.status == hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::AVAILABLE
           && std::isfinite(prediction.value))
        {
            return;
        }
        refusals << "\n  " << probe.name << ": " << prediction.reason;
    }
    ADD_FAILURE() << "the model " << describeModel(model, "L1 predict_engine", archKey)
                  << " gives no finite prediction for any of the " << served.size()
                  << " probe graphs the engine serves:" << refusals.str();
}

/// Every kernel @p set's packs ship for @p archKey, as the engine's catalog holds it: a field
/// the descriptor omits takes the KMD default.
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>
    shippedKernels(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                   const std::string& archKey)
{
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    for(const auto& pack : set.packs)
    {
        if(!pack.arch.empty()
           && std::find(pack.arch.begin(), pack.arch.end(), archKey) == pack.arch.end())
        {
            continue;
        }
        for(const auto& kernel : pack.kernels)
        {
            hipdnn_plugin_sdk::ingestor::KernelDefinition definition;
            definition.kernelId = kernel.id;
            definition.packId = pack.id;
            definition.dispatchId = pack.dispatchId;
            definition.metadata = kernel.metadata;
            for(const auto& field : set.schema.fields)
            {
                if(field.defaultValue.has_value())
                {
                    definition.metadata.emplace(field.name, *field.defaultValue);
                }
            }
            definition.priority = kernel.priority;
            definition.name = kernel.name;
            kernels.push_back(std::move(definition));
        }
    }
    return kernels;
}

/// Ranks every kernel the packs ship for @p archKey with @p scorer, the engine's L2 for it, on
/// the probe graphs the engine serves. For at least one of them the ranking must hold the same
/// candidates and lead with a finite positive score: a ranking that fails (a feature the
/// graph does not give, a kernel the model cannot encode) falls back to declared order,
/// whose every score is 0. When none does, each graph's outcome is named.
inline void expectRankerRanks(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                              const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                              const hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic& scorer,
                              const std::string& metric,
                              const std::string& archKey,
                              const ProbeGraphs& probes)
{
    const auto served = expectServedProbes(set, archKey, probes);
    if(served.empty())
    {
        return;
    }
    const auto kernels = shippedKernels(set, archKey);
    const auto name = describeModel(model, "L2 sort_kernel_catalog", archKey);
    if(kernels.empty())
    {
        ADD_FAILURE() << "the packs ship no kernel for " << archKey << ", so the model " << name
                      << " ranks nothing";
        return;
    }
    std::ostringstream failures;
    for(const auto& probe : served)
    {
        hipdnn_plugin_sdk::ingestor::Catalog catalog;
        catalog.bound = probe.bound;
        for(const auto& kernel : kernels)
        {
            catalog.entries.push_back(&kernel);
        }
        auto context = probe.fixture->context();
        context.rankingMetric = metric;
        const auto ranking = scorer.rankScored(catalog, context);
        if(ranking.size() == kernels.size() && std::isfinite(ranking.front().score)
           && ranking.front().score > 0.0)
        {
            return;
        }
        failures << "\n  " << probe.name << ": ranked " << ranking.size() << " of "
                 << kernels.size() << " candidates"
                 << (ranking.empty() ? ""
                                     : ", leading score " + std::to_string(ranking.front().score));
    }
    ADD_FAILURE() << "the model " << name << " ranks none of the " << served.size()
                  << " probe graphs the engine serves (a ranking that fails falls back to "
                     "declared order, scored 0):"
                  << failures.str();
}

/// The L2 sort_kernel_catalog ranker @p set's UED binds under @p archKey, checked the way
/// every ModelBinding census checks it: the UED binds one, and when the packed tree obliges
/// it to load (packedTreeObligesLoad), one for @p metric survived the loader's pre-flight,
/// it is one the UED binds, its model file is present, and its scorer builds as the runtime
/// builds it (expectRankerScorerBuilds: the model file hashes to the UHD's recorded digest
/// and its weights parse). With @p probes, the scorer also ranks the kernels the packs ship
/// for a graph the engine serves (expectRankerRanks). Null when none loaded (a failure) or
/// none had to.
inline const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor*
    expectBoundRankerLoads(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                           const std::string& metric,
                           const std::string& archKey,
                           const ProbeGraphs& probes = {})
{
    constexpr const char* ROLE = "L2 sort_kernel_catalog";
    const auto ids = boundIds(set.engine.sortKernelCatalog, archKey);
    if(ids.empty())
    {
        ADD_FAILURE() << "the UED binds no " << archKey << " sort_kernel_catalog ranker";
        return nullptr;
    }
    if(!packedTreeObligesLoad(ids, ROLE))
    {
        return nullptr;
    }

    // The loader prunes a ranker it refuses into unavailableHeuristicArches; a pruned one
    // is absent from the map, so presence is the load check.
    const auto* model = findModel(set.heuristicsByMetric, metric, archKey);
    if(model == nullptr)
    {
        ADD_FAILURE() << "no " << metric << " sort_kernel_catalog ranker under " << archKey
                      << " survived the loader's pre-flight; kernels rank by priority";
        return nullptr;
    }
    EXPECT_NE(
        std::find(ids.begin(), ids.end(), hipdnn_flatbuffers_sdk::utilities::formatUuid(model->id)),
        ids.end())
        << "the loaded " << metric << " ranker is not one the UED binds";
    expectModelArtifactPresent(*model, ROLE);
    if(const auto scorer = expectRankerScorerBuilds(set, *model, archKey);
       scorer != nullptr && !probes.empty())
    {
        expectRankerRanks(set, *model, *scorer, metric, archKey, probes);
    }
    return model;
}

/// The L1 predict_engine model @p set's UED binds under @p archKey, checked the way every
/// ModelBinding census checks it: with none bound, none loaded; with one bound and the packed
/// tree obliging it to load (packedTreeObligesLoad), one for @p metric survived the loader's
/// pre-flight, its model file is present, its scorer builds as GenericEngine builds it
/// (expectPredictorScorerBuilds: the model file hashes to the UHD's recorded digest, its
/// weights parse and its feature contract holds), and it records the selector revision this
/// build computes. With @p probes, the model also answers for a graph the engine serves
/// (expectPredictorAnswers). The engine selector's revision digests every launch's
/// code-object hash and the L2 rankers, so a compiler bump, a catalog change or a retrained
/// L2 makes GenericEngine refuse the L1 (PredictionStatus::UNAVAILABLE) with nothing but a
/// log line; this fails the build instead.
inline void
    expectBoundPredictorMatchesThisBuild(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                                         const std::string& metric,
                                         const std::string& archKey,
                                         const ProbeGraphs& probes = {})
{
    constexpr const char* ROLE = "L1 predict_engine";
    const auto ids = boundIds(set.engine.predictEngine, archKey);
    if(ids.empty())
    {
        // No L1 ships. The check stays live: the day a UED binds one, the branches below
        // hold it to the revision this build computes.
        EXPECT_EQ(findModel(set.enginePredictionsByMetric, metric, archKey), nullptr)
            << "the UED binds no " << archKey << " predict_engine model, yet one loaded";
        return;
    }
    if(!packedTreeObligesLoad(ids, ROLE))
    {
        return;
    }

    const auto* model = findModel(set.enginePredictionsByMetric, metric, archKey);
    ASSERT_NE(model, nullptr) << "no " << metric << " predict_engine model under " << archKey
                              << " survived the loader's pre-flight; the engine reports no "
                                 "prediction";
    expectModelArtifactPresent(*model, ROLE);
    if(expectPredictorScorerBuilds(set, *model, metric, archKey) != nullptr && !probes.empty())
    {
        expectPredictorAnswers(set, *model, metric, archKey, probes);
    }

    // The revision digests the L2 too. With a bound L2 left out of the pack as un-pulled
    // (the L2 case reports that), the runtime refuses the L1 for that reason, not because the
    // L1 is stale, so the comparison below would blame the wrong model.
    for(const auto& ranker : boundIds(set.engine.sortKernelCatalog, archKey))
    {
        // The macro is a compile-time 0 or 1, so one arm is dead in each build.
        // NOLINTNEXTLINE(misc-redundant-expression)
        if(HKP_REQUIRE_DVC_DESCRIPTORS == 0 && !packedTreeHoldsUhd(ranker)
           && packedTreeRecordsUnpulled(ranker))
        {
            std::cout << "[  NOTE    ] the L2 sort_kernel_catalog model " << ranker
                      << " was not pulled, so this build's selector revision cannot match the "
                         "L1's and the engine runs with the L1 off too\n";
            return;
        }
    }

    // The same call makeEngine() makes, on the same loaded set.
    EXPECT_EQ(model->trainedAgainstSelectorRevision,
              hipdnn_plugin_sdk::ingestor::engineSelectorRevision(set))
        << "GenericEngine refuses this model and reports no prediction: the shipped UHD was "
           "trained against another selector revision than this build computes. Retrain the "
           "L1 against this build, or find what changed the code objects.";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
