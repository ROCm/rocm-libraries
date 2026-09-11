// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
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
/// A UHD reaches an engine only through that engine's UED `predict_engine_tflops`
/// role map, which the descriptor loader resolves (RFC 0019 §3.1); there is no
/// discovery here and an engine without a UED contributes no prediction. Description
/// never loads or evaluates a model. Missing coverage and malformed models leave engine
/// applicability unchanged. Binding/features JSON is emitted only for description,
/// not policy evaluation.
/// @param config The resolved role model, or a default-constructed config when the
///               UED binds none: description still names the binding an author must
///               train against, which is how the first model is bootstrapped.
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
                                      {"selector_revision", selectorRevision}};
            if(!config.uhdId.empty())
            {
                binding["uhd_id"] = config.uhdId;
            }
            // trained_against is left to the caller: the engine knows the descriptor set
            // the model is being compared against, which is what a staleness check needs.
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
} // namespace hipdnn_plugin_sdk::uhd
