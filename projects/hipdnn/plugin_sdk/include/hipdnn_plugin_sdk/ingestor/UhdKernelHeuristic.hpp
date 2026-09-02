// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/UhdLoader.hpp>

/// @file UhdKernelHeuristic.hpp
/// @brief Ranks an engine's kernels with a trained UHD (RFC 0019 §5).
namespace hipdnn_plugin_sdk::ingestor
{

namespace detail
{

/// A metadata value as the feature extractor's variable context spells it.
///
/// Four of the five alternatives map across directly. `vector<int64_t>` has no
/// counterpart, so it yields nullopt and the name is left unbound: an expression guarding
/// it with `value_or_default` still evaluates, and a bare reference fails closed, which is
/// the behaviour RFC 0019 §5 wants from a feature the runtime cannot supply.
inline std::optional<uhd::VariableContext::ValueType> toValueType(const MetadataValue& value)
{
    if(const auto* v = std::get_if<bool>(&value))
    {
        return uhd::VariableContext::ValueType{*v};
    }
    if(const auto* v = std::get_if<int64_t>(&value))
    {
        return uhd::VariableContext::ValueType{*v};
    }
    if(const auto* v = std::get_if<double>(&value))
    {
        return uhd::VariableContext::ValueType{*v};
    }
    if(const auto* v = std::get_if<std::string>(&value))
    {
        return uhd::VariableContext::ValueType{*v};
    }
    return std::nullopt;
}

inline uhd::FeatureExtractionContext::ValueMap
    deviceVarsFrom(const DeviceProperties& deviceProperties)
{
    // `cu_count` and `multi_processor_count` name the same quantity; both spellings are
    // bound so a signature authored against either resolves.
    //
    // `arch` is deliberately absent. It selects which UHD runs (RFC 0019 §3.1) and is
    // passed separately to isTrainedForArch(); admitting it as a feature would let a model
    // split on the architecture it was chosen for.
    return {{"cu_count", static_cast<int64_t>(deviceProperties.multiProcessorCount)},
            {"multi_processor_count", static_cast<int64_t>(deviceProperties.multiProcessorCount)},
            {"warp_size", static_cast<int64_t>(deviceProperties.warpSize)}};
}

inline uhd::FeatureExtractionContext::ValueMap queryVarsFrom(const BoundTokens& bound)
{
    uhd::FeatureExtractionContext::ValueMap vars;
    for(const auto& [name, value] : bound)
    {
        if(auto converted = toValueType(value))
        {
            vars.emplace(name, std::move(*converted));
        }
    }
    return vars;
}

inline uhd::FeatureExtractionContext::ValueMap kernelVarsFrom(const KernelDefinition& kernel)
{
    uhd::FeatureExtractionContext::ValueMap vars;
    for(const auto& [name, value] : kernel.metadata)
    {
        if(auto converted = toValueType(value))
        {
            vars.emplace(name, std::move(*converted));
        }
    }
    vars.emplace("priority", kernel.priority);

    // `id` is not bound: a kernel's id is a UUID, and a feature that compared UUID bytes
    // would be ordering by authoring accident. The id already decides ties in rank().
    return vars;
}

/// `priority` descending, then id ascending -- the order rank() falls back to, and the
/// same one UnrankedKernelHeuristic produces.
inline std::vector<KernelDefinition> declaredOrder(const std::vector<KernelDefinition>& entries)
{
    std::vector<KernelDefinition> ordered(entries);
    std::stable_sort(ordered.begin(),
                     ordered.end(),
                     [](const KernelDefinition& a, const KernelDefinition& b) {
                         if(a.priority != b.priority)
                         {
                             return a.priority > b.priority;
                         }
                         return a.kernelId < b.kernelId;
                     });
    return ordered;
}

} // namespace detail

/// @brief Ranks a catalog with the model a MODEL heuristic descriptor names.
///
/// @brief The `$kernel.*` axes a feature signature actually reads.
///
/// RFC 0019 §6.3 check 2 requires `F == set(UED.knobs)`, where F is this set: the knobs an
/// engine exposes must be exactly the axes its model ranks on. Both directions matter and both
/// fail silently. A knob the model does not read is a dial the caller can turn while the
/// heuristic ignores it; an axis with no knob is a model ranking on something the API never
/// lets anyone vary.
inline std::unordered_set<std::string> kernelAxesOf(const std::vector<std::string>& signature)
{
    std::unordered_set<std::string> axes;
    for(const auto& entry : signature)
    {
        // An entry is a JsonLogic expression; a bare reference is not valid JSON on its own,
        // so it arrives quoted and parses to a string.
        try
        {
            for(const auto& variable :
                uhd::JsonLogicEvaluator::extractVariables(nlohmann::json::parse(entry)))
            {
                constexpr std::string_view PREFIX = "$kernel.";
                if(variable.rfind(PREFIX, 0) == 0)
                {
                    axes.insert(variable.substr(PREFIX.size()));
                }
            }
        }
        catch(const std::exception&)
        {
            // A signature entry that will not parse is already a broken contract; the load
            // below reports it. Skipping here keeps this helper from being the messenger.
            continue;
        }
    }
    return axes;
}

/// Holds no mutable state: rank() keeps its shared feature row on the stack, so one
/// instance serves concurrent selections without a lock.
///
/// Two feature-vocabulary limits are worth knowing, because a UHD authored against the
/// backend's policy path and dropped in here will evaluate differently rather than fail:
///  - `$device.*` offers only what DeviceProperties carries. `device_id` and
///    `total_global_mem` are not available.
///  - `$kernel.id` is unbound, the id being a UUID here rather than an integer.
/// Either reference degrades the whole ranking to declared order, with a log line.
class UhdKernelHeuristic : public IKernelHeuristic
{
public:
    /// @returns nullptr when the UHD cannot be brought up, so the caller can substitute
    ///          declared-order ranking. Never throws.
    static std::shared_ptr<UhdKernelHeuristic> tryCreate(const HeuristicDescriptor& descriptor,
                                                         const std::string& describedBy,
                                                         const std::vector<std::string>& knobs = {})
    {
        try
        {
            const auto path = descriptor.baseDir / descriptor.payload;
            auto config = uhd::UhdLoader::load(path);
            if(!config.has_value())
            {
                return nullptr;
            }

            // RFC 0019 §6.3 check 2: the exposed knobs *are* the model's feature axes.
            // Returning nullptr degrades to declared order, which is what §5 step 7 asks for
            // on a broken feature contract -- the model's inputs are not the ones it was
            // trained on, so its scores would be wrong.
            const auto axes = kernelAxesOf(config->featuresSignature);
            const std::unordered_set<std::string> exposed(knobs.begin(), knobs.end());
            if(axes != exposed)
            {
                const auto join = [](const auto& names) {
                    std::string text;
                    for(const auto& name : names)
                    {
                        text += (text.empty() ? "" : ", ") + name;
                    }
                    return text.empty() ? std::string("<none>") : text;
                };
                HIPDNN_PLUGIN_LOG_ERROR(
                    "uhd: " << describedBy << " exposes knobs [" << join(exposed)
                            << "] but its model ranks on [" << join(axes)
                            << "]; RFC 0019 §6.3 requires these to be the same set, so the "
                               "model is not used and kernels rank by priority, then id");
                return nullptr;
            }

            auto adapter = uhd::makeUhdAdapter(*config);
            if(adapter == nullptr)
            {
                HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " names adapter '"
                                                << config->adapterType
                                                << "', which built no scorer");
                return nullptr;
            }

            auto extractor = std::make_shared<const uhd::FeatureExtractor>(
                config->featuresSignature, config->derived);

            // RFC 0019 §6.3: the signature the model was trained against must be the one
            // the extractor will produce. Both sides carry the hash; disagreeing means the
            // pair was assembled from two different training runs.
            if(extractor->getSignatureHash() != config->featuresHash)
            {
                HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " signature hashes disagree -- "
                                                << "descriptor declares '" << config->featuresHash
                                                << "', signature computes '"
                                                << extractor->getSignatureHash() << "'");
                return nullptr;
            }

            return std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(
                std::move(*config), std::move(adapter), std::move(extractor), describedBy));
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " failed to load: " << e.what());
            return nullptr;
        }
    }

    /// Scores one kernel, extracting the whole row. rank() does not use this -- it shares
    /// the problem and device half across candidates -- so this is the path for a direct
    /// caller or a test.
    double score(const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel) const override
    {
        uhd::FeatureExtractionContext ctx;
        ctx.bindDeviceVars(detail::deviceVarsFrom(context.deviceProperties));
        ctx.bindQueryVars(detail::queryVarsFrom(bound));
        ctx.bindKernelVars(detail::kernelVarsFrom(kernel));
        return orientedScore(_extractor->extract(ctx));
    }

    std::vector<KernelDefinition> rank(const Catalog& catalog,
                                       const MatchContext& context) const override
    {
        try
        {
            uhd::FeatureExtractionContext ctx;
            ctx.bindDeviceVars(detail::deviceVarsFrom(context.deviceProperties));
            ctx.bindQueryVars(detail::queryVarsFrom(catalog.bound));

            if(!_adapter->isTrainedForArch(context.deviceProperties.gcnArchName))
            {
                // A warning, not a refusal: RFC 0019 §9.3 treats an unseen architecture as
                // out-of-distribution, where the model is still better than no model.
                HIPDNN_PLUGIN_LOG_WARN("uhd: " << _describedBy << " was not trained for '"
                                               << context.deviceProperties.gcnArchName
                                               << "'; ranking anyway");
            }

            // RFC 0019 §6 step 2: the problem and device slots are the same for every
            // candidate, so they are evaluated once and the kernel slots overwritten.
            const std::vector<double> sharedRow = _extractor->extractSharedRow(ctx);

            std::vector<std::pair<double, const KernelDefinition*>> scored;
            scored.reserve(catalog.entries.size());
            for(const auto& entry : catalog.entries)
            {
                ctx.clearKernelVars();
                ctx.bindKernelVars(detail::kernelVarsFrom(entry));

                std::vector<double> row = sharedRow;
                _extractor->extractKernelInto(ctx, row);
                scored.emplace_back(orientedScore(row), &entry);
            }

            std::stable_sort(scored.begin(),
                             scored.end(),
                             [](const auto& a, const auto& b) {
                                 if(a.first != b.first)
                                 {
                                     return a.first > b.first;
                                 }
                                 if(a.second->priority != b.second->priority)
                                 {
                                     return a.second->priority > b.second->priority;
                                 }
                                 return a.second->kernelId < b.second->kernelId;
                             });

            std::vector<KernelDefinition> ordered;
            ordered.reserve(scored.size());
            for(const auto& [_, entry] : scored)
            {
                ordered.push_back(*entry);
            }

            traceSelection(scored, context);
            return ordered;
        }
        catch(const std::exception& e)
        {
            // The whole ranking falls back, not the kernels that happened to fail. A mix
            // of model scores and sentinels is neither order, and RFC 0019 §5 asks for a
            // degraded ranking rather than a partial one.
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << _describedBy << " failed while ranking: "
                                            << e.what()
                                            << "; kernels rank by priority, then descriptor id");
            // RFC 0019 §12 wants the trace to say *whether the model or a fallback decided*,
            // so the degraded path is traced too. A trace that only ever appears on success
            // cannot answer the question it exists for.
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: " << _describedBy << " decided_by=fallback"
                                                 << " reason=ranking_failed"
                                                 << " candidates=" << catalog.entries.size()
                                                 << " uhd=" << _config.uhdId
                                                 << " adapter=" << _config.adapterType
                                                 << " features_hash=" << _config.featuresHash);
            return detail::declaredOrder(catalog.entries);
        }
    }

    /// RFC 0019 §12: the selection trace -- candidates, scores, the ranked order, the winner,
    /// and whether the model or a fallback decided -- plus the model provenance that says
    /// which model produced them.
    ///
    /// Logged rather than returned. The removed backend implementation kept an in-memory trace
    /// map with a retrieval path that had no public API, so nothing outside its own test could
    /// read it; a log line is what an operator can actually see, and §12 exists so selection is
    /// inspectable rather than queryable.
    ///
    /// At INFO because it is per-graph and verbose: a build ranking thousands of graphs should
    /// not pay for it by default, and §12's error-level requirements are the contract
    /// diagnostics, which are logged where they occur.
    void traceSelection(const std::vector<std::pair<double, const KernelDefinition*>>& scored,
                        const MatchContext& context) const
    {
        if(scored.empty())
        {
            return;
        }

        std::ostringstream candidates;
        for(size_t i = 0; i < scored.size(); ++i)
        {
            candidates << (i == 0 ? "" : " ") << toString(scored[i].second->kernelId)
                       << "=" << scored[i].first;
        }

        HIPDNN_PLUGIN_LOG_INFO("uhd trace: "
                               << _describedBy << " decided_by=model"
                               << " winner=" << toString(scored.front().second->kernelId)
                               << " candidates=" << scored.size()
                               << " arch=" << context.deviceProperties.gcnArchName
                               << " uhd=" << _config.uhdId
                               << " adapter=" << _config.adapterType
                               << " objective=" << _config.objective
                               << " features_hash=" << _config.featuresHash
                               << " ranked=[" << candidates.str() << "]");
    }

private:
    UhdKernelHeuristic(uhd::UhdConfig config,
                       std::shared_ptr<const uhd::IUhdAdapter> adapter,
                       std::shared_ptr<const uhd::FeatureExtractor> extractor,
                       std::string describedBy)
        : _config(std::move(config))
        , _adapter(std::move(adapter))
        , _extractor(std::move(extractor))
        // rank() sorts descending, so a model predicting a cost rather than a rate has to
        // be negated. Omitting this silently inverts every latency-trained UHD.
        , _objectiveSign(_config.objective == "min" ? -1.0 : 1.0)
        , _describedBy(std::move(describedBy))
    {
    }

    /// The model's raw output, returned to its declared units and oriented so that larger
    /// is better.
    double orientedScore(const std::vector<double>& row) const
    {
        const double raw = _adapter->score(row);
        return _objectiveSign * uhd::score_transform::applyInverse(raw, _config.scoreTransform);
    }

    uhd::UhdConfig _config;
    std::shared_ptr<const uhd::IUhdAdapter> _adapter;
    std::shared_ptr<const uhd::FeatureExtractor> _extractor;
    double _objectiveSign;
    std::string _describedBy;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
