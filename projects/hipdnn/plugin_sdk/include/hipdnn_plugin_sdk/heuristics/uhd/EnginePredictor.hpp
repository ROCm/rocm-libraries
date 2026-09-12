// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

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
inline constexpr const char* ENGINE_ROLE = "predict_engine_tflops";
inline constexpr size_t MAX_ARTIFACT_BYTES = 256 * 1024 * 1024;

struct Model
{
    std::unique_ptr<const FeatureExtractor> extractor;
    std::shared_ptr<const IUhdAdapter> adapter;
    PredictionStatus status = PredictionStatus::INVALID;
    std::string reason;
};

/// @brief Compile and validate the L1 model one resolved UED role names.
/// Touches the filesystem and rebuilds the feature contract, so the owning engine
/// caches the result for its lifetime rather than repeating it per query.
inline std::shared_ptr<const Model> model(const UhdConfig& config)
{
    auto loaded = std::make_shared<Model>();
    try
    {
        // Optional by construction: parseUhdConfig and the schema require trained_against
        // only for a model carrying a feature signature, so a native or custom_library
        // model that featurizes from its own bindings legally omits it.
        if(config.trainedAgainst.is_object())
        {
            parser_detail::provenance(config.trainedAgainst, "L1 UHD trained_against");
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
            if(config.adapterType == "custom_library" && !config.modelHash.empty())
            {
                std::ifstream stream(config.modelArtifactPath, std::ios::binary);
                std::vector<uint8_t> contents(static_cast<size_t>(size));
                if(!stream.read(reinterpret_cast<char*>(contents.data()),
                                static_cast<std::streamsize>(contents.size()))
                   || sha256(contents.data(), contents.size()) != config.modelHash)
                {
                    throw std::invalid_argument("UHD custom library artifact hash mismatch");
                }
            }
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
/// Descriptor provenance is not rechecked here: the UUID and major/minor rule of
/// RFC 0019 §8.1 already ran in the loader, which is what binds this UHD to this UED.
inline void
    validateBinding(const UhdConfig& config, const std::string& engine, const std::string& arch)
{
    if((!config.engineName.empty() && config.engineName != engine)
       || (!config.role.empty() && config.role != ENGINE_ROLE)
       || (!config.arch.empty() && config.arch != "default" && config.arch != arch))
    {
        throw std::invalid_argument("UHD attachment does not match engine, role, or architecture");
    }
    // RFC 0019 §11.3: an L1 estimate is cross-engine TFLOPS, so the transform has to be one
    // this runtime can invert back into the declared units. That is score_transform's
    // vocabulary, asked rather than restated: the inline list here accepted only identity and
    // log1p, which made a second, narrower spelling of a closed set the parser already gates
    // (parseUhdConfig) -- two vocabularies to keep in step, and a descriptor an author could
    // load and then not bind. What keeps the estimate *physical* is not the list but the
    // finite-and-non-negative check at the evaluation site below, which is transform-agnostic.
    if(config.scoreUnits != "tflops" || !config.scoreCalibrated || config.objective != "max"
       || !score_transform::isSupported(config.scoreTransform))
    {
        throw std::invalid_argument("L1 UHD requires calibrated TFLOPS with an invertible "
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

/// @brief Describe or evaluate a graph-only engine throughput model.
/// A UHD reaches an engine only through a binding that lives in compiled code -- the
/// UED's `predict_engine_tflops` role map for a descriptor-backed engine (RFC 0019
/// §3.1), or the UUID a provider names in its own engine definition for an engine that
/// ships no UED (RFC 0019 Open Question 7, RESOLVED). There is no discovery here and no
/// document claims an engine. Description never loads or evaluates a model. Missing
/// coverage and malformed models leave engine applicability unchanged. Binding/features
/// JSON is emitted only for description, not policy evaluation.
/// @param config The bound model, or a default-constructed config when nothing binds
///               one: description still names the binding an author must train
///               against, which is how the first model is bootstrapped.
/// @param compiled @p config compiled by prediction_detail::model() and cached by the
///                 caller; null whenever nothing is deployed for this architecture.
///                 Only read when @p evaluate, so description cannot touch a model.
/// @returns Physical calibrated TFLOPS only for AVAILABLE predictions.
inline hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
    predictEngine(int64_t engineId,
                  const std::string& engineName,
                  const std::string& selectorRevision,
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
    const auto targetArch = arch.substr(0, arch.find(':'));
    try
    {
        if(!evaluate)
        {
            nlohmann::json binding = {{"engine", engineName},
                                      {"role", ENGINE_ROLE},
                                      {"arch", targetArch},
                                      {"selector_revision", selectorRevision},
                                      // What a model collected from this description would
                                      // be trained against. An engine with no descriptors
                                      // has exactly this and nothing else (§4.1, Open
                                      // Question 7), so it is written here rather than
                                      // left to a caller that has nothing to add: a
                                      // description carrying no trained_against at all
                                      // cannot be turned into a UHD, which is where every
                                      // opaque L1 collection stopped (run 67929509).
                                      {"trained_against",
                                       {{"selector_revision", selectorRevision}}}};
            if(!config.uhdId.empty())
            {
                binding["uhd_id"] = config.uhdId;
            }
            // A descriptor-backed engine ADDS its set to trained_against on top of this
            // (GenericEngine.hpp:211-221): it is trained against both the descriptors it
            // loaded and the provider build that ran them.
            result.uhd_id = config.uhdId;
            result.binding_json = binding.dump();
            result.features_json = features.toJson().dump();
            result.reason = "engine prediction binding description";
            return result;
        }
        if(!compiled)
        {
            result.reason = "no engine UHD is deployed";
            return result;
        }
        result.uhd_id = config.uhdId;
        validateBinding(config, engineName, targetArch);
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
            // Let the expression evaluator enforce lazy defaults and branches: a
            // missing variable in an unselected branch is not a coverage failure.
            row = compiled->extractor->extract(features);
        }
        catch(const JsonLogicError& error)
        {
            result.reason = std::string("UHD feature coverage unavailable: ") + error.what();
            return result;
        }
        const double raw = compiled->adapter->score(row);
        const double physical = score_transform::applyInverse(raw, config.scoreTransform);
        if(!std::isfinite(raw) || !std::isfinite(physical) || physical < 0.0)
        {
            throw std::invalid_argument("UHD prediction is not finite nonnegative TFLOPS");
        }
        result.tflops = physical;
        result.status = PredictionStatus::AVAILABLE;
    }
    catch(const std::exception& error)
    {
        result.status = PredictionStatus::INVALID;
        result.reason = error.what();
    }
    return result;
}

/// @brief One engine's L1 throughput binding: the models bound to it per architecture,
/// the architectures whose bound model this build refused, and the compiled-model cache
/// those queries share.
///
/// A binding always comes from compiled code, never from the document. A
/// descriptor-backed engine fills it from its UED's `predict_engine_tflops` role map,
/// which the loader resolves (RFC 0019 §3.1); an engine that ships no UED fills it from
/// the UUID its provider names in its own engine definition (RFC 0019 Open Question 7,
/// RESOLVED). Either way the UHD keeps §4.1's shape -- no `engine`, `role` or `arch`
/// member -- so no document can attach itself to an engine by claiming one.
///
/// Held by the engine, so a model is compiled once and shared by every later query.
class EngineModelBinding
{
public:
    /// @param arch `default`, or a gcnArchName prefix, exactly as the binding spelled it.
    void bind(const std::string& arch, UhdConfig config)
    {
        _byArch.insert_or_assign(arch, std::move(config));
    }

    /// @brief Record an architecture whose bound model this build will not use.
    ///
    /// RFC 0019 §11.2 separates two refusals, and @p status is which one this is:
    ///   - UNAVAILABLE -- "I do not answer this question". The model is fine, it just is
    ///     not this build's: a model trained against another provider revision (§4.1
    ///     `trained_against.selector_revision`) says nothing about this one.
    ///   - INVALID -- "I answer, and the answer is bad". A model that is present and
    ///     failed its contract is a claim: do not pick me.
    /// @param reason Surfaced verbatim to the caller, so it must name what was compared.
    void markUnusable(const std::string& arch,
                      hipdnn_flatbuffers_sdk::data_objects::PredictionStatus status,
                      std::string reason)
    {
        _refused.insert_or_assign(arch, Refusal{status, std::move(reason)});
    }

    /// @brief This engine's L1 prediction for @p arch, described or evaluated.
    /// @param arch The device `gcnArchName`, feature suffix included.
    /// @param evaluate False describes the binding without touching a model.
    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        predict(int64_t engineId,
                const std::string& engineName,
                const std::string& selectorRevision,
                const std::string& arch,
                const FeatureExtractionContext& features,
                bool evaluate) const
    {
        // RFC 0019 §8.3: the longest matching architecture wins, with `default` used
        // only when nothing more specific matched.
        std::string selectedArch;
        const UhdConfig* selected = nullptr;
        const Refusal* refused = nullptr;
        for(const auto& [target, model] : _byArch)
        {
            if((target == "default" && selectedArch.empty())
               || (target != "default" && archMatches(arch, target, ArchMatchMode::PREFIX)
                   && (selectedArch == "default" || target.size() > selectedArch.size())))
            {
                selected = &model;
                selectedArch = target;
            }
        }
        // `>=`, not `>`: a refusal at the same specificity as a bound model wins, so an
        // exact-arch failure can never silently fall through to another arch's model.
        for(const auto& [target, refusal] : _refused)
        {
            if((target == "default" && selectedArch.empty())
               || (target != "default" && archMatches(arch, target, ArchMatchMode::PREFIX)
                   && (selectedArch == "default" || target.size() >= selectedArch.size())))
            {
                refused = &refusal;
            }
        }
        // Nothing outside this binding can attach a model to the engine, so an engine
        // with no bound model for this architecture simply has none.
        static const UhdConfig UNBOUND;
        const bool evaluateModel = evaluate && refused == nullptr;
        std::shared_ptr<const prediction_detail::Model> compiled;
        if(evaluateModel && selected != nullptr)
        {
            compiled = compiledModel(selectedArch, *selected);
        }
        auto result = predictEngine(engineId,
                                    engineName,
                                    selectorRevision,
                                    arch,
                                    features,
                                    evaluateModel,
                                    selected != nullptr ? *selected : UNBOUND,
                                    compiled);
        if(refused != nullptr)
        {
            result.status = refused->status;
            result.reason = refused->reason;
            // A disabled model still took the description branch to get here. Ranking is
            // not a description request, so the payload it built is dropped rather than
            // serialized across the plugin ABI for every candidate engine.
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

    /// Compiling a UHD rebuilds its feature contract and reads its artifact off disk.
    /// A usable model is compiled once and shared by every later query on this engine.
    /// A failed compile is NOT cached: deployment is separate from load (RFC 0019 §5), so
    /// an artifact that is still being installed, or a transient read error, must not
    /// disable the model for the rest of the provider's lifetime.
    std::shared_ptr<const prediction_detail::Model> compiledModel(const std::string& arch,
                                                                  const UhdConfig& config) const
    {
        const std::lock_guard<std::mutex> lock(_modelMutex);
        if(const auto cached = _modelCache.find(arch); cached != _modelCache.end())
        {
            return cached->second;
        }
        auto compiled = prediction_detail::model(config);
        if(compiled != nullptr
           && compiled->status == hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::AVAILABLE)
        {
            _modelCache.emplace(arch, compiled);
        }
        return compiled;
    }

    std::map<std::string, UhdConfig> _byArch;
    std::map<std::string, Refusal> _refused;
    mutable std::mutex _modelMutex;
    mutable std::map<std::string, std::shared_ptr<const prediction_detail::Model>> _modelCache;
};
} // namespace hipdnn_plugin_sdk::uhd
