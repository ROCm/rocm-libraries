// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdParser.hpp>

namespace hipdnn_plugin_sdk::uhd
{
namespace prediction_detail
{
using PredictionStatus = hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;
inline constexpr const char* ENGINE_ROLE = "predict_engine";
inline constexpr size_t MAX_ARTIFACT_BYTES = size_t{256} * 1024 * 1024;

struct Model
{
    std::unique_ptr<const FeatureExtractor> extractor;
    std::shared_ptr<const IUhdAdapter> adapter;
    PredictionStatus status = PredictionStatus::INVALID;
    std::string reason;
};

/// @brief Compile and validate the L1 model a resolved UED role names.
/// Touches the filesystem; callers cache the result rather than calling per query.
inline std::shared_ptr<const Model> model(const UhdConfig& config)
{
    auto loaded = std::make_shared<Model>();
    try
    {
        // trained_against is required only with a feature signature; native and
        // custom_library models may omit it.
        if(config.trainedAgainst.is_object())
        {
            parser_detail::provenance(config.trainedAgainst, "L1 UHD trained_against");
        }
        // Features from another revision: UNAVAILABLE, like a stale selector revision
        // (RFC 0019 §11.2). Covers configs that bypassed the loader's check.
        if(auto mismatch
           = featureSemanticsMismatch(config.featuresSignature, config.trainedAgainst);
           !mismatch.empty())
        {
            loaded->status = PredictionStatus::UNAVAILABLE;
            loaded->reason = std::move(mismatch);
            return loaded;
        }
        loaded->extractor = std::make_unique<const FeatureExtractor>(config.featuresSignature,
                                                                     config.categoricalEncoding);
        if(loaded->extractor->kernelDependentCount() != 0)
        {
            throw std::invalid_argument("L1 UHD cannot consume kernel features");
        }
        for(const auto& entry : config.categoricalEncoding)
        {
            if(entry.first.rfind("$kernel.", 0) == 0)
            {
                throw std::invalid_argument("L1 UHD cannot encode kernel features");
            }
        }
        if((!config.featuresSignature.empty() || !config.featuresHash.empty())
           && loaded->extractor->getSignatureHash() != config.featuresHash)
        {
            throw std::invalid_argument("L1 UHD feature signature hash mismatch");
        }
        if(config.adapterType != "native")
        {
            if(!std::filesystem::path(config.modelArtifactPath).is_absolute())
            {
                throw std::invalid_argument("L1 UHD artifact path is not resolved absolutely");
            }
            std::error_code error;
            if(!std::filesystem::exists(config.modelArtifactPath, error) && !error)
            {
                loaded->status = PredictionStatus::UNAVAILABLE;
                loaded->reason = "UHD model artifact is not deployed";
                return loaded;
            }
            const auto size = std::filesystem::file_size(config.modelArtifactPath, error);
            if(error || size == 0 || size > MAX_ARTIFACT_BYTES)
            {
                throw std::invalid_argument(
                    "UHD model artifact is unreadable or exceeds size bound");
            }
            // custom_library digest is verified by CustomLibraryAdapter::load for every
            // role; do not duplicate it here.
        }
        else if(config.nativeSymbol.empty())
        {
            throw std::invalid_argument("L1 native UHD requires a symbol");
        }
        loaded->adapter = makeUhdAdapter(config);
        if(!loaded->adapter)
        {
            loaded->status = config.adapterType == "native" ? PredictionStatus::UNAVAILABLE
                                                            : PredictionStatus::INVALID;
            loaded->reason = "UHD adapter is unavailable or its model failed validation";
        }
        // Grouped tree_data models are INVALID here: an L1 row has no group, so score()
        // would use only the root ensemble, which is not what was trained.
        else if(config.adapterType == "tree_data" && loaded->adapter->groupFeatureIndex() >= 0)
        {
            loaded->reason = "grouped tree_data artifact cannot be bound to the predict_engine "
                             "role: grouped L1 models have no per-row contract";
        }
        else if(loaded->adapter->expectedFeatureCount() != loaded->extractor->featureCount()
                || loaded->adapter->getFeaturesHash() != config.featuresHash)
        {
            loaded->reason = "UHD model feature contract does not match its signature";
        }
        else
        {
            loaded->status = PredictionStatus::AVAILABLE;
        }
    }
    catch(const std::exception& error)
    {
        loaded->reason = error.what();
    }
    return loaded;
}

/// @brief Check that a resolved role model agrees with the engine asking for it.
/// Descriptor provenance (RFC 0019 §8.1) is checked by the loader, not here.
/// @param metric The requested metric; the model must estimate exactly it (RFC 0019 §4.4).
inline void validateBinding(const UhdConfig& config,
                            const std::string& engine,
                            const std::string& arch,
                            const std::string& metric)
{
    if((!config.engineName.empty() && config.engineName != engine)
       || (!config.role.empty() && config.role != ENGINE_ROLE)
       || (!config.arch.empty() && config.arch != "default" && config.arch != arch))
    {
        throw std::invalid_argument("UHD attachment does not match engine, role, or architecture");
    }
    // L1 estimates are compared across engines (RFC 0019 §11.1), so require a calibrated
    // estimate of the requested metric, in its objective, with an invertible transform.
    const auto* registered = hipdnn_data_sdk::utilities::findRankingMetric(config.scoreMetric);
    if(registered == nullptr || config.scoreMetric != metric || !config.scoreCalibrated
       || config.objective != hipdnn_data_sdk::utilities::objectiveOf(*registered)
       || !score_transform::isSupported(config.scoreTransform))
    {
        throw std::invalid_argument("L1 UHD requires a calibrated '" + metric
                                    + "' score with that metric's objective and an invertible "
                                      "transform (one of "
                                    + score_transform::supportedTransformList() + ")");
    }
    if(config.adapterType != "tree_data" && config.adapterType != "native"
       && config.adapterType != "custom_library")
    {
        throw std::invalid_argument("L1 UHD requires tree_data, native, or custom_library adapter");
    }
    if(config.adapterType == "tree_data" && config.featuresSignature.empty())
    {
        throw std::invalid_argument("L1 tree_data UHD requires a feature signature");
    }
}
} // namespace prediction_detail

/// @brief Describe or evaluate a graph-only engine estimate in one ranking metric.
/// Description never loads or evaluates a model; binding/features JSON is emitted only
/// for description. Missing coverage or a bad model never changes engine applicability.
/// @param metric The requested metric; models of any other metric are never asked.
/// @param config The bound model, or a default config when none is bound (description
///               still names the binding to train against).
/// @param compiled @p config compiled by prediction_detail::model(); null when nothing is
///                 deployed. Read only when @p evaluate.
/// @returns A physical value in @p metric's registered units only for AVAILABLE predictions.
inline hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
    predictEngine(int64_t engineId,
                  const std::string& engineName,
                  const std::string& selectorRevision,
                  const std::string& metric,
                  const std::string& arch,
                  const FeatureExtractionContext& features,
                  bool evaluate,
                  const UhdConfig& config,
                  const std::shared_ptr<const prediction_detail::Model>& compiled)
{
    using namespace prediction_detail;
    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT result;
    result.engine_id = engineId;
    result.kind = hipdnn_flatbuffers_sdk::data_objects::PredictionKind::ENGINE;
    result.status = PredictionStatus::UNAVAILABLE;
    result.metric = metric;
    const auto targetArch = arch.substr(0, arch.find(':'));
    try
    {
        if(!evaluate)
        {
            nlohmann::json binding
                = {{"engine", engineName},
                   {"role", ENGINE_ROLE},
                   {"metric", metric},
                   {"arch", targetArch},
                   {"selector_revision", selectorRevision},
                   // Always emitted: a description without trained_against
                   // cannot be turned into a UHD.
                   {"trained_against", {{"selector_revision", selectorRevision}}}};
            if(!config.uhdId.empty())
            {
                binding["uhd_id"] = config.uhdId;
            }
            // Descriptor-backed engines add their descriptor set to trained_against
            // (GenericEngine::getPrediction).
            result.uhd_id = config.uhdId;
            result.binding_json = binding.dump();
            result.features_json = features.toJson().dump();
            result.reason = "engine prediction binding description";
            return result;
        }
        if(!compiled)
        {
            result.reason = std::string("no ") + ENGINE_ROLE + " UHD for metric '" + metric
                            + "' on arch '" + targetArch + "'";
            return result;
        }
        result.uhd_id = config.uhdId;
        validateBinding(config, engineName, targetArch, metric);
        if(compiled->status != PredictionStatus::AVAILABLE)
        {
            result.status = compiled->status;
            result.reason = compiled->reason;
            return result;
        }
        if(!compiled->adapter->isTrainedForArch(targetArch))
        {
            result.reason = "UHD model has no coverage for this architecture";
            return result;
        }
        std::vector<double> row;
        try
        {
            // A missing variable in an unselected branch is not a coverage failure, so
            // let the evaluator decide.
            row = compiled->extractor->extract(features);
        }
        catch(const JsonLogicError& error)
        {
            result.reason = std::string("UHD feature coverage unavailable: ") + error.what();
            return result;
        }
        const double raw = compiled->adapter->score(row);
        const double physical = score_transform::applyInverse(raw, config.scoreTransform);
        // validateBinding proved the metric registered and equal to the model's.
        if(!std::isfinite(raw)
           || !hipdnn_data_sdk::utilities::isValidMetricValue(
               *hipdnn_data_sdk::utilities::findRankingMetric(metric), physical))
        {
            throw std::invalid_argument("UHD prediction is not a valid '" + metric + "' value");
        }
        result.value = physical;
        result.status = PredictionStatus::AVAILABLE;
    }
    catch(const std::exception& error)
    {
        result.status = PredictionStatus::INVALID;
        result.reason = error.what();
    }
    return result;
}

/// @brief One engine's L1 models per (metric, architecture), refused pairs, and a shared
/// compiled-model cache.
/// Bindings come only from compiled code (UED role map or provider engine definition),
/// never from a UHD document. Thread-safe for concurrent predict() after binding.
class EngineModelBinding
{
public:
    /// @param metric The registered metric @p config declares in `score.metric`.
    /// @param arch `default`, or a gcnArchName prefix, exactly as the binding spelled it.
    void bind(const std::string& metric, const std::string& arch, UhdConfig config)
    {
        _byMetric[metric].insert_or_assign(arch, std::move(config));
    }

    /// @brief Record a (metric, architecture) whose bound model this build will not use.
    /// @param status UNAVAILABLE for a model built for another revision; INVALID for one
    ///               that failed its contract (RFC 0019 §11.2).
    /// @param reason Surfaced verbatim to the caller; name what was compared.
    void markUnusable(const std::string& metric,
                      const std::string& arch,
                      hipdnn_flatbuffers_sdk::data_objects::PredictionStatus status,
                      std::string reason)
    {
        _refused[metric].insert_or_assign(arch, Refusal{status, std::move(reason)});
    }

    /// @brief This engine's L1 prediction in @p metric for @p arch, described or evaluated.
    /// @param metric The requested metric; arch fallback never crosses to another metric.
    /// @param arch The device `gcnArchName`, feature suffix included.
    /// @param evaluate False describes the binding without touching a model.
    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        predict(int64_t engineId,
                const std::string& engineName,
                const std::string& selectorRevision,
                const std::string& metric,
                const std::string& arch,
                const FeatureExtractionContext& features,
                bool evaluate) const
    {
        // Longest matching architecture wins; `default` only as fallback (RFC 0019 §8.3).
        std::string selectedArch;
        const UhdConfig* selected = nullptr;
        const Refusal* refused = nullptr;
        if(const auto models = _byMetric.find(metric); models != _byMetric.end())
        {
            for(const auto& [target, model] : models->second)
            {
                if((target == "default" && selectedArch.empty())
                   || (target != "default" && archMatches(arch, target, ArchMatchMode::PREFIX)
                       && (selectedArch == "default" || target.size() > selectedArch.size())))
                {
                    selected = &model;
                    selectedArch = target;
                }
            }
        }
        // `>=`, not `>`: a refusal at the same specificity as a bound model wins.
        if(const auto refusals = _refused.find(metric); refusals != _refused.end())
        {
            for(const auto& [target, refusal] : refusals->second)
            {
                if((target == "default" && selectedArch.empty())
                   || (target != "default" && archMatches(arch, target, ArchMatchMode::PREFIX)
                       && (selectedArch == "default" || target.size() >= selectedArch.size())))
                {
                    refused = &refusal;
                }
            }
        }
        static const UhdConfig s_unbound;
        const bool evaluateModel = evaluate && refused == nullptr;
        std::shared_ptr<const prediction_detail::Model> compiled;
        if(evaluateModel && selected != nullptr)
        {
            compiled = compiledModel(metric, selectedArch, *selected);
        }
        auto result = predictEngine(engineId,
                                    engineName,
                                    selectorRevision,
                                    metric,
                                    arch,
                                    features,
                                    evaluateModel,
                                    selected != nullptr ? *selected : s_unbound,
                                    compiled);
        if(refused != nullptr)
        {
            result.status = refused->status;
            result.reason = refused->reason;
            // Refused models took the description branch; drop its payload when ranking.
            if(evaluate)
            {
                result.binding_json.clear();
                result.features_json.clear();
            }
        }
        return result;
    }

private:
    struct Refusal
    {
        hipdnn_flatbuffers_sdk::data_objects::PredictionStatus status;
        std::string reason;
    };

    /// Caches only successful compiles, so an artifact still being deployed can recover
    /// (RFC 0019 §5). Keyed by UUID (arch key if none), so one UUID bound under several
    /// arches compiles once.
    std::shared_ptr<const prediction_detail::Model> compiledModel(const std::string& metric,
                                                                  const std::string& arch,
                                                                  const UhdConfig& config) const
    {
        const std::lock_guard<std::mutex> lock(_modelMutex);
        const auto key
            = std::make_pair(metric, config.uhdId.empty() ? "arch:" + arch : config.uhdId);
        if(const auto cached = _modelCache.find(key); cached != _modelCache.end())
        {
            return cached->second;
        }
        auto compiled = prediction_detail::model(config);
        if(compiled != nullptr
           && compiled->status == hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::AVAILABLE)
        {
            _modelCache.emplace(key, compiled);
        }
        return compiled;
    }

    std::map<std::string, std::map<std::string, UhdConfig>> _byMetric;
    std::map<std::string, std::map<std::string, Refusal>> _refused;
    mutable std::mutex _modelMutex;
    mutable std::map<std::pair<std::string, std::string>,
                     std::shared_ptr<const prediction_detail::Model>>
        _modelCache;
};
} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
