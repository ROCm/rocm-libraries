// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdParser.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// @file UhdKernelHeuristic.hpp
/// @brief Ranks an engine's kernels with a trained UHD (RFC 0019 §5).
namespace hipdnn_plugin_sdk::ingestor
{

namespace detail
{

/// Converts a scalar metadata value; `vector<int64_t>` yields nullopt and stays unbound,
/// so a bare reference to it fails closed (RFC 0019 §5).
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

inline void appendFeatureValue(uhd::FeatureExtractionContext::ValueMap& vars,
                               const std::string& name,
                               const MetadataValue& value)
{
    if(const auto* array = std::get_if<std::vector<int64_t>>(&value))
    {
        for(size_t i = 0; i < array->size(); ++i)
        {
            vars.emplace(name + "[" + std::to_string(i) + "]", (*array)[i]);
        }
    }
    else if(auto scalar = toValueType(value))
    {
        vars.insert_or_assign(name, std::move(*scalar));
    }
}

inline uhd::FeatureExtractionContext::ValueMap queryVarsFrom(const BoundTokens& bound)
{
    uhd::FeatureExtractionContext::ValueMap vars;
    for(const auto& [name, value] : bound)
    {
        appendFeatureValue(vars, name, value);
    }
    return vars;
}

/// True for a name in a namespace the engine publishes itself: `graph.*` and `device.*`
/// (problemFeatures), `constraint.*` (engineFeatures) and `kernel.*` (kernel metadata).
inline bool isReservedFeatureName(const std::string& name)
{
    const std::string_view bare
        = !name.empty() && name.front() == '$' ? std::string_view(name).substr(1) : name;
    for(const std::string_view reserved : {"graph.", "device.", "constraint.", "kernel."})
    {
        if(bare.substr(0, reserved.size()) == reserved)
        {
            return true;
        }
    }
    return false;
}

/// Binds graph-match tokens as features. Reserved names are skipped so a matcher cannot
/// override a canonical feature such as `graph.flops`.
inline void bindGraphMatchBindings(uhd::FeatureExtractionContext& features,
                                   const BoundTokens& bound)
{
    for(const auto& entry : queryVarsFrom(bound))
    {
        if(!isReservedFeatureName(entry.first))
        {
            features.bind(entry.first, entry.second);
        }
    }
}

/// The problem half of every `sort_kernel_catalog` feature row: `graph.*`, `device.*` and
/// graph-match bindings. Shared by runtime ranking and training data so both see the same
/// features. Excludes `constraint.*` because ranked catalogs are cached per CatalogKey.
/// Compute once per graph, not per candidate.
inline uhd::FeatureExtractionContext catalogProblemFeatures(const MatchContext& context,
                                                            const BoundTokens& bound)
{
    auto features = heuristics::problemFeatures(context.graph, context.deviceProperties);
    bindGraphMatchBindings(features, bound);
    return features;
}

inline uhd::FeatureExtractionContext::ValueMap kernelVarsFrom(const KernelDefinition& kernel)
{
    uhd::FeatureExtractionContext::ValueMap vars;
    for(const auto& [name, value] : kernel.metadata)
    {
        appendFeatureValue(vars, name, value);
    }
    vars.emplace("priority", kernel.priority);

    // `id` is deliberately unbound: it is a UUID, and it already breaks ties in rank().
    return vars;
}

} // namespace detail

/// @brief The `$kernel.*` axes a feature signature reads.
///
/// RFC 0019 §6.3 check 2 requires this set to be a subset of `UED.knobs`, not equal to it:
/// a knob the model ignores is only a warning.
inline std::unordered_set<std::string> kernelAxesOf(const uhd::FeatureExtractor& extractor)
{
    std::unordered_set<std::string> axes;
    for(const auto& variable : extractor.getVariableRefs())
    {
        // `$kernel.tile[0]` ranks on knob `tile`.
        if(auto field = uhd::FeatureExtractor::kernelFieldOf(variable))
        {
            axes.insert(std::move(*field));
        }
    }
    return axes;
}

/// @brief Ranks an engine's kernels with a UHD model. Safe for concurrent selections.
///
/// `$device.*` offers only DeviceProperties fields (no `device_id`, `total_global_mem`) and
/// `$kernel.id` is unbound; referencing either degrades ranking to declared order.
class UhdKernelHeuristic : public IKernelHeuristic
{
public:
    /// Builds an instance with no model of its own that lazily resolves a per-(metric, arch)
    /// model: exact arch, then `default`, within the metric (RFC 0019 §3.1, §8.3).
    /// @param byMetric Metric (`""` for the metric-less ranker) to arch key to descriptor.
    /// @param unavailable Metric to arch keys whose named model was refused: such a key never
    ///        falls back to the `default` key's model for the same metric.
    static std::shared_ptr<UhdKernelHeuristic> makeResolver(
        const std::map<std::string, std::map<std::string, HeuristicDescriptor>>& byMetric,
        const std::string& describedBy,
        const std::vector<std::string>& knobs,
        const std::unordered_set<std::string>& kmdFields = {},
        const std::map<std::string, std::set<std::string>>& unavailable = {})
    {
        auto built = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(describedBy));
        built->_byMetric = byMetric;
        built->_knobs = knobs;
        built->_kmdFields = kmdFields;
        built->_unavailable = unavailable;
        return built;
    }

    /// @returns nullptr when the UHD cannot be brought up, so the caller can substitute
    ///          declared-order ranking. Never throws.
    /// @param knobs The UED's declared knob names; @p kmdFields the KMD's declared field
    ///        names. If either is empty, a model reading any `$kernel.*` axis is refused.
    static std::shared_ptr<UhdKernelHeuristic>
        tryCreate(const HeuristicDescriptor& descriptor,
                  const std::string& describedBy,
                  const std::vector<std::string>& knobs = {},
                  const std::unordered_set<std::string>& kmdFields = {})
    {
        std::ostringstream failure;
        auto built = create(descriptor, describedBy, knobs, kmdFields, failure);
        if(built == nullptr)
        {
            HIPDNN_PLUGIN_LOG_ERROR(failure.str());
        }
        return built;
    }

    /// The descriptor's fields, with the artifact path resolved against the file it was
    /// declared in.
    static uhd::UhdConfig configFrom(const HeuristicDescriptor& descriptor)
    {
        uhd::UhdConfig config;
        config.uhdId = toString(descriptor.id);
        config.name = descriptor.name;
        config.featuresSignature = descriptor.featuresSignature;
        config.featuresHash = descriptor.featuresHash;
        config.categoricalEncoding = descriptor.categoricalEncoding;
        config.objective = descriptor.objective;
        config.scoreMetric = descriptor.score.metric;
        config.scoreCalibrated = descriptor.score.calibrated;
        config.scoreTransform = descriptor.score.transform;
        config.nativeSymbol = descriptor.nativeSymbol;
        config.customLibrarySymbol = descriptor.customLibrarySymbol;
        config.modelHash = descriptor.modelHash;
        config.engineName = descriptor.engineName;
        config.role = descriptor.role;
        config.arch = descriptor.arch;
        config.trainedAgainst = descriptor.trainedAgainstJson;

        switch(descriptor.adapter)
        {
        case UhdAdapter::STATIC_ORDER:
            config.adapterType = "static_order";
            break;
        case UhdAdapter::NATIVE:
            config.adapterType = "native";
            break;
        case UhdAdapter::TREE_DATA:
            config.adapterType = "tree_data";
            break;
        case UhdAdapter::TABLE:
            config.adapterType = "table";
            break;
        case UhdAdapter::CUSTOM_LIBRARY:
            config.adapterType = "custom_library";
            break;
        // Required by -Wswitch-default; every enum member is handled above.
        default:
            config.adapterType = "static_order";
            break;
        }

        if(!descriptor.modelArtifactPath.empty())
        {
            config.modelArtifactPath = (descriptor.baseDir / descriptor.modelArtifactPath).string();
        }
        return config;
    }

    /// Scores one kernel by extracting a full row; rank() shares the problem half instead.
    double score(const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel) const override
    {
        if(isResolver())
        {
            const auto choice = chooseRanker(std::string(context.rankingMetric),
                                             context.deviceProperties.gcnArchName);
            return choice.model ? choice.model->score(context, bound, kernel) : 0.0;
        }
        if(_direct)
        {
            return _direct->score(context, bound, kernel);
        }
        if(!_extractor)
        {
            return 0.0;
        }
        auto ctx = detail::catalogProblemFeatures(context, bound);
        ctx.bindKernelVars(detail::kernelVarsFrom(kernel));
        // The reported form; ranking goes through rankScored instead.
        return scoreCandidate(_extractor->extract(ctx)).reported;
    }

    /// @brief Scores the matching architecture in physical units of `context.rankingMetric`,
    ///        best first in that metric's direction, including a singleton catalog.
    ///
    /// Empty unless the requested metric's own calibrated model answers; the default ranker
    /// never substitutes for it (RFC 0019 §11.4).
    std::vector<ScoredKernel> calibratedRanking(const Catalog& catalog,
                                                const MatchContext& context,
                                                std::string& modelId) const override
    {
        if(isResolver())
        {
            const auto resolved = resolveFor(std::string(context.rankingMetric),
                                             context.deviceProperties.gcnArchName);
            return resolved ? resolved->calibratedRanking(catalog, context, modelId)
                            : std::vector<ScoredKernel>{};
        }
        const auto* metric = hipdnn_data_sdk::utilities::findRankingMetric(context.rankingMetric);
        if(!_hasDefaultModel || metric == nullptr
           || !answersCalibrated(_config.scoreMetric,
                                 _config.scoreCalibrated,
                                 _config.objective,
                                 context.rankingMetric)
           || (!_direct
               && (!_adapter || !_extractor
                   || !_adapter->isTrainedForArch(context.deviceProperties.gcnArchName))))
        {
            return {};
        }
        auto ranking = rankWith(catalog, context);
        for(auto& candidate : ranking)
        {
            // Undo orientation to recover the physical value; the 0 no-measurement sentinel
            // stays +0 under a `min` objective.
            candidate.score = candidate.score == 0.0 ? 0.0 : _objectiveSign * candidate.score;
        }
        // Degraded rankings use zero sentinels, never available physical estimates.
        if(ranking.empty() || ranking.front().score == 0.0
           || !hipdnn_data_sdk::utilities::isValidMetricValue(*metric, ranking.front().score))
        {
            return {};
        }
        modelId = _config.uhdId;
        return ranking;
    }

    /// RFC 0019 §12: what decides the ranking, spelled as in the trace line.
    std::string traceDecidedBy() const override
    {
        // Context-free: a resolver says declared_order though it may rank by model on a named
        // architecture; the per-ranking trace line is authoritative.
        return _hasDefaultModel ? "model" : "declared_order";
    }

    /// The signature entry at the adapter's group slot, `$` stripped, or nothing when the
    /// model decides in one layer.
    std::optional<std::string> groupFeature() const override
    {
        // No device here: answer for the default metric's `default` entry.
        if(isResolver())
        {
            const auto choice
                = chooseRanker(std::string(hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC), {});
            return choice.model ? choice.model->groupFeature() : std::nullopt;
        }
        if(_adapter == nullptr)
        {
            return std::nullopt;
        }
        const int slot = _adapter->groupFeatureIndex();
        if(slot < 0 || static_cast<size_t>(slot) >= _config.featuresSignature.size())
        {
            return std::nullopt;
        }
        // A bare reference is unwrapped to its field; an expression is labelled by its text.
        const auto& entry = _config.featuresSignature[static_cast<size_t>(slot)];
        if(!entry.is_string())
        {
            return entry.dump();
        }
        std::string name = entry.get<std::string>();
        if(!name.empty() && name.front() == '$')
        {
            name.erase(name.begin());
        }
        return name;
    }

    std::vector<ScoredKernel> rankScored(const Catalog& catalog,
                                         const MatchContext& context) const override
    {
        // Only an empty catalog short-circuits. A sole candidate is still scored, since a 0
        // score means "no measurement" (§5 step 7), not "only one candidate".
        if(catalog.entries.empty())
        {
            return {};
        }
        if(!isResolver() && _hasDefaultModel)
        {
            return rankWith(catalog, context);
        }
        const auto& arch = context.deviceProperties.gcnArchName;
        const std::string metric(context.rankingMetric);
        // RFC 0019 §11.4 ranker choice, resolved lazily (§9.2) because descriptor discovery
        // runs before any device exists.
        if(const auto choice = chooseRanker(metric, arch); choice.model)
        {
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: "
                                   << _describedBy << " metric=" << metric
                                   << " ranker=" << choice.source << " ranker_metric="
                                   << (choice.metric.empty() ? "(none)" : choice.metric)
                                   << " arch=" << arch);
            return choice.model->rankWith(catalog, context);
        }

        // §8.3: no exact or `default` model; never borrow another architecture's model.
        reportNoModelForArchOnce(arch, metric);

        // §12: degraded paths are traced too.
        HIPDNN_PLUGIN_LOG_INFO("uhd trace: " << _describedBy << " decided_by=declared_order"
                                             << " reason=no_model_for_arch" << " metric=" << metric
                                             << " arch=" << arch
                                             << " candidates=" << catalog.entries.size());
        return detail::asScored(detail::declaredOrder(catalog.entries));
    }

    /// The model @p metric's own entries name for @p arch, loaded on first success per
    /// (metric, arch key) and cached. A failed load is retried on the next call so an
    /// artifact still being deployed can recover (RFC 0019 §5); it is reported once per key.
    /// Fallback stays inside the metric (RFC 0019 §3.1).
    std::shared_ptr<const UhdKernelHeuristic> resolveFor(const std::string& metric,
                                                         const std::string& arch) const
    {
        std::string key;
        const auto* chosen = boundFor(metric, arch, key);
        if(chosen == nullptr)
        {
            return nullptr;
        }
        const std::lock_guard<std::mutex> lock(_archMutex);
        const auto cacheKey = std::make_pair(metric, key);
        if(const auto cached = _archCache.find(cacheKey); cached != _archCache.end())
        {
            return cached->second;
        }
        // Lazily resolved models face the same knob/field checks (RFC 0019 §8.3).
        std::ostringstream failure;
        auto loaded = create(*chosen, _describedBy, _knobs, _kmdFields, failure);
        if(loaded)
        {
            _archCache.emplace(cacheKey, loaded);
        }
        else if(_archLoadFailuresReported.insert(cacheKey).second)
        {
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << _describedBy << " model for metric "
                                            << (metric.empty() ? "(none)" : "'" + metric + "'")
                                            << " on '" << key
                                            << "' failed to load: " << failure.str());
        }
        return loaded;
    }

    /// @brief The id of the model calibratedRanking() answers with for @p metric on @p arch.
    ///
    /// Loads no model and ranks nothing, so artifact arch coverage is not checked.
    std::string calibratedModelId(const std::string& metric, const std::string& arch) const override
    {
        if(isResolver())
        {
            std::string key;
            const auto* bound = boundFor(metric, arch, key);
            if(bound == nullptr
               || !answersCalibrated(
                   bound->score.metric, bound->score.calibrated, bound->objective, metric))
            {
                return {};
            }
            return toString(bound->id);
        }
        if(!_hasDefaultModel
           || !answersCalibrated(
               _config.scoreMetric, _config.scoreCalibrated, _config.objective, metric))
        {
            return {};
        }
        return _config.uhdId;
    }

private:
    /// tryCreate() without the ERROR: why the UHD was refused goes to @p failure, so a caller
    /// that retries can report it once.
    static std::shared_ptr<UhdKernelHeuristic>
        create(const HeuristicDescriptor& descriptor,
               const std::string& describedBy,
               const std::vector<std::string>& knobs,
               const std::unordered_set<std::string>& kmdFields,
               std::ostream& failure)
    {
        // RFC 0019 §9.4 load time: covers config, signature compilation and artifact read.
        const auto loadStart = Clock::now();
        try
        {
            auto config = configFrom(descriptor);
            // Also checked by the loader; repeated for descriptors that arrive another way.
            if(const auto mismatch
               = uhd::featureSemanticsMismatch(config.featuresSignature, config.trainedAgainst);
               !mismatch.empty())
            {
                failure << "uhd: " << describedBy << " " << mismatch
                        << "; the model is not used and kernels rank by priority, then id";
                return nullptr;
            }
            if(descriptor.adapter == UhdAdapter::STATIC_ORDER
               || (descriptor.adapter == UhdAdapter::NATIVE && config.featuresSignature.empty()))
            {
                auto built
                    = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(describedBy));
                built->_objectiveSign = objectiveSignOf(config.objective);
                if(descriptor.adapter == UhdAdapter::STATIC_ORDER)
                {
                    built->_direct = std::make_shared<UnrankedKernelHeuristic>();
                }
                else
                {
                    // rankScored orders higher-first, so a cost scorer needs the objective.
                    built->_direct
                        = std::make_shared<NativeKernelHeuristic>(descriptor.nativeSymbol,
                                                                  describedBy,
                                                                  config.objective,
                                                                  config.scoreTransform,
                                                                  config.scoreMetric);
                }
                built->_config = std::move(config);
                built->_hasDefaultModel = true;
                return built;
            }

            auto extractor = std::make_shared<const uhd::FeatureExtractor>(
                config.featuresSignature, config.categoricalEncoding);
            // RFC 0019 §6.3 check 2: every axis the model ranks on must be an exposed knob;
            // otherwise degrade to declared order (§5 step 7).
            const auto axes = kernelAxesOf(*extractor);
            const std::unordered_set<std::string> exposed(knobs.begin(), knobs.end());
            const auto join = [](const auto& names) {
                std::string text;
                for(const auto& name : names)
                {
                    text += (text.empty() ? "" : ", ") + name;
                }
                return text.empty() ? std::string("<none>") : text;
            };

            // RFC 0019 §6.3 check 2, first assertion: `F ⊆ KMD.fields`. Re-checked here
            // because descriptor sets are drop-in and may be edited after generation.
            const auto undeclared = extractor->getMissingKmdFields(kmdFields);
            if(!undeclared.empty())
            {
                std::vector<std::string> missing = undeclared;
                std::sort(missing.begin(), missing.end());
                std::vector<std::string> declared(kmdFields.begin(), kmdFields.end());
                std::sort(declared.begin(), declared.end());
                failure << "uhd: "
                        << describeDescriptor("heuristic", descriptor.name, descriptor.id) << " on "
                        << describedBy << " reads [" << join(missing)
                        << "], which its KMD does not declare as fields [" << join(declared)
                        << "]; RFC 0019 §6.3 requires the model's axes to be declared, so the "
                           "model is not used and kernels rank by priority, then id";
                return nullptr;
            }

            std::vector<std::string> unexposed;
            for(const auto& axis : axes)
            {
                if(exposed.count(axis) == 0)
                {
                    unexposed.push_back(axis);
                }
            }
            if(!unexposed.empty())
            {
                std::sort(unexposed.begin(), unexposed.end());
                failure << "uhd: " << describedBy << " ranks on [" << join(unexposed)
                        << "], which its UED does not expose as knobs [" << join(exposed)
                        << "]; RFC 0019 §6.3 requires the model's axes to be exposed, so the "
                           "model is not used and kernels rank by priority, then id";
                return nullptr;
            }

            auto adapter = uhd::makeUhdAdapter(config);
            if(adapter == nullptr)
            {
                failure << "uhd: " << describedBy << " names adapter '" << config.adapterType
                        << "', which built no scorer";
                return nullptr;
            }

            // RFC 0019 §6.3: the descriptor's features hash must match the signature.
            if(!config.featuresSignature.empty()
               && extractor->getSignatureHash() != config.featuresHash)
            {
                failure << "uhd: " << describedBy << " signature hashes disagree -- "
                        << "descriptor declares '" << config.featuresHash
                        << "', signature computes '" << extractor->getSignatureHash() << "'";
                return nullptr;
            }

            // RFC 0019 §6.3 check 4: the hash does not cover artifact arity, and
            // TreeDataAdapter silently treats a short row as missing values.
            if(adapter->expectedFeatureCount() != extractor->featureCount())
            {
                failure << "uhd: " << describedBy << " model expects "
                        << adapter->expectedFeatureCount() << " features, its signature "
                        << "produces " << extractor->featureCount()
                        << "; the model is not used and kernels rank by priority, then id";
                return nullptr;
            }

            // Knobs the model does not read are legal (e.g. constant in training); warn only.
            std::vector<std::string> unread;
            for(const auto& knob : exposed)
            {
                if(axes.count(knob) == 0)
                {
                    unread.push_back(knob);
                }
            }
            if(!unread.empty())
            {
                std::sort(unread.begin(), unread.end());
                HIPDNN_PLUGIN_LOG_WARN("uhd: " << describedBy << " exposes knobs [" << join(unread)
                                               << "] its model does not rank on; selection "
                                                  "ignores them");
            }

            auto built = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(
                std::move(config), std::move(adapter), std::move(extractor), describedBy));
            built->_timing.loadNs.store(elapsedNs(loadStart), std::memory_order_relaxed);
            return built;
        }
        catch(const std::exception& e)
        {
            failure << "uhd: " << describedBy << " failed to load: " << e.what();
            return nullptr;
        }
    }

    /// The descriptor @p metric's own entries bind for @p arch, and its arch key, or null.
    /// A refused entry (exact or `default`) never falls through to a model it would shadow.
    const HeuristicDescriptor*
        boundFor(const std::string& metric, const std::string& arch, std::string& key) const
    {
        static const std::set<std::string> s_noneUnavailable;
        const auto refusedIt = _unavailable.find(metric);
        const auto& refused
            = refusedIt == _unavailable.end() ? s_noneUnavailable : refusedIt->second;
        for(const auto& unavailable : refused)
        {
            if(unavailable != "default" && archMatches(arch, unavailable, ArchMatchMode::PREFIX))
            {
                return nullptr;
            }
        }
        const auto models = _byMetric.find(metric);
        if(models == _byMetric.end())
        {
            return nullptr;
        }
        const HeuristicDescriptor* chosen = nullptr;
        key.clear();
        for(const auto& [candidate, descriptor] : models->second)
        {
            if(candidate != "default" && archMatches(arch, candidate, ArchMatchMode::PREFIX)
               && candidate.size() > key.size())
            {
                chosen = &descriptor;
                key = candidate;
            }
        }
        if(chosen == nullptr && refused.count("default") == 0)
        {
            if(const auto fallback = models->second.find("default");
               fallback != models->second.end())
            {
                chosen = &fallback->second;
                key = "default";
            }
        }
        return chosen;
    }

    /// Whether a model with this score answers calibratedRanking() in @p metric: the same
    /// registered metric (RFC 0019 §11.4), calibrated, ordered in the metric's direction.
    static bool answersCalibrated(const std::string& scoreMetric,
                                  bool calibrated,
                                  const std::string& objective,
                                  std::string_view metric)
    {
        const auto* registered = hipdnn_data_sdk::utilities::findRankingMetric(metric);
        return registered != nullptr && calibrated && scoreMetric == registered->name
               && objective == hipdnn_data_sdk::utilities::objectiveOf(*registered);
    }

    /// Which ranker decides a kernel choice, and why (§11.4).
    struct RankerChoice
    {
        std::shared_ptr<const UhdKernelHeuristic> model;
        /// `metric` for the requested metric's own ranker, `default` for the stand-in.
        const char* source = "declared_order";
        /// The metric the deciding UHD declares; empty for the metric-less ranker.
        std::string metric;
    };

    /// True for an instance built by makeResolver, which ranks through per-(metric, arch)
    /// children.
    bool isResolver() const
    {
        return !_byMetric.empty() || !_unavailable.empty();
    }

    /// RFC 0019 §3.1, §11.4: @p metric's ranker, else the metric-less ranker, else the
    /// DEFAULT_RANKING_METRIC ranker, else none (static order).
    RankerChoice chooseRanker(const std::string& metric, const std::string& arch) const
    {
        if(auto own = resolveFor(metric, arch))
        {
            return {std::move(own), "metric", metric};
        }
        for(const std::string& fallback :
            {std::string(), std::string(hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC)})
        {
            if(fallback == metric)
            {
                continue;
            }
            if(auto standIn = resolveFor(fallback, arch))
            {
                return {std::move(standIn), "default", fallback};
            }
        }
        return {};
    }

    /// One candidate's number, in the two forms that must not be conflated.
    struct CandidateScore
    {
        /// Higher wins; -infinity when there is no measurement, so it always sorts last.
        double ordering;
        /// RFC 0019.13 §15.2's figure of merit; 0 when there is no measurement (§5 step 7).
        double reported;
    };

    /// One scored candidate, as the ranking holds it before it becomes a ScoredKernel.
    struct Ranked
    {
        CandidateScore score;
        const KernelDefinition* entry;
        /// The grouping feature's value; NaN for a single-layer model (see ScoredKernel::group).
        double group = std::numeric_limits<double>::quiet_NaN();
    };

    std::vector<ScoredKernel> rankWith(const Catalog& catalog, const MatchContext& context) const
    {
        if(_direct)
        {
            return _direct->rankScored(catalog, context);
        }
        try
        {
            if(!_adapter->isTrainedForArch(context.deviceProperties.gcnArchName))
            {
                // RFC 0019 §9.3: an unseen architecture is out-of-distribution, not refused.
                HIPDNN_PLUGIN_LOG_WARN("uhd: " << _describedBy << " was not trained for '"
                                               << context.deviceProperties.gcnArchName
                                               << "'; ranking anyway");
            }

            // RFC 0019 §6 step 2: problem/device slots are evaluated once per selection;
            // only kernel slots vary. §9.4 times the prefix and per-candidate tail apart.
            const auto prefixStart = Clock::now();
            auto ctx = detail::catalogProblemFeatures(context, catalog.bound);
            auto features = _extractor->prepare(ctx);
            _timing.prefixNs.fetch_add(elapsedNs(prefixStart), std::memory_order_relaxed);
            _timing.selections.fetch_add(1, std::memory_order_relaxed);

            // One batched scoring call: a grouped model compares candidates against each other.
            uint64_t tailNs = 0;
            std::vector<std::vector<double>> rows;
            rows.reserve(catalog.entries.size());
            for(const auto& entry : catalog.entries)
            {
                const auto tailStart = Clock::now();
                ctx.clearKernelVars();
                ctx.bindKernelVars(detail::kernelVarsFrom(entry));

                _extractor->extractKernelInto(ctx, features);
                tailNs += elapsedNs(tailStart);
                rows.push_back(features.values);
            }

            const auto scoreStart = Clock::now();
            const auto raw = _adapter->scoreBatch(rows);
            const auto scoreNs = elapsedNs(scoreStart);

            // Read the group from the scored row, not metadata, which derived or categorical
            // features could make disagree.
            const int groupSlot = _adapter->groupFeatureIndex();
            const auto groupOf = [&](const std::vector<double>& row) {
                return groupSlot >= 0 && static_cast<size_t>(groupSlot) < row.size()
                           ? row[static_cast<size_t>(groupSlot)]
                           : std::numeric_limits<double>::quiet_NaN();
            };

            std::vector<Ranked> scored;
            scored.reserve(catalog.entries.size());
            for(size_t index = 0; index < catalog.entries.size(); ++index)
            {
                scored.push_back(
                    {scoreFromRaw(raw[index]), &catalog.entries[index], groupOf(rows[index])});
            }

            _timing.tailNs.fetch_add(tailNs, std::memory_order_relaxed);
            _timing.scoreNs.fetch_add(scoreNs, std::memory_order_relaxed);
            _timing.candidates.fetch_add(scored.size(), std::memory_order_relaxed);

            // A raw -infinity is a candidate the adapter declined (a grouped model's losing
            // groups), not an out-of-range prediction; only the latter are reported, out of
            // the candidates actually scored.
            size_t declined = 0;
            size_t outOfRange = 0;
            for(size_t index = 0; index < scored.size(); ++index)
            {
                if(raw[index] == -std::numeric_limits<double>::infinity())
                {
                    ++declined;
                }
                else if(!std::isfinite(scored[index].score.ordering))
                {
                    ++outOfRange;
                }
            }
            reportOutOfRangeOnce(outOfRange, scored.size() - declined);

            // Scores are never NaN here (scoreFromRaw maps them to -infinity); NaN would break
            // the strict weak ordering std::stable_sort requires.
            std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
                if(a.score.ordering != b.score.ordering)
                {
                    return a.score.ordering > b.score.ordering;
                }
                if(a.entry->priority != b.entry->priority)
                {
                    return a.entry->priority > b.entry->priority;
                }
                return a.entry->kernelId < b.entry->kernelId;
            });

            std::vector<ScoredKernel> ordered;
            ordered.reserve(scored.size());
            for(const auto& candidate : scored)
            {
                ordered.push_back(
                    {candidate.entry->kernelId, candidate.score.reported, candidate.group});
            }

            traceSelection(scored, context);
            return ordered;
        }
        catch(const std::exception& e)
        {
            // The whole ranking degrades, never a partial mix (RFC 0019 §5).
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << _describedBy << " failed while ranking: " << e.what()
                                            << "; kernels rank by priority, then descriptor id");
            // RFC 0019 §12: trace the fallback too, with the same `declared_order` spelling
            // as UnrankedKernelHeuristic; `reason` says why.
            HIPDNN_PLUGIN_LOG_INFO(
                "uhd trace: " << _describedBy << " decided_by=declared_order"
                              << " reason=ranking_failed" << " metric=" << context.rankingMetric
                              << " candidates=" << catalog.entries.size()
                              << " uhd=" << _config.uhdId << " adapter=" << _config.adapterType
                              << " features_hash=" << _config.featuresHash);
            // Scores are 0, RFC 0019 §5 step 7's "no measurement".
            return detail::asScored(detail::declaredOrder(catalog.entries));
        }
    }

    /// RFC 0019 §12 selection trace and model provenance, logged at INFO because the plugin
    /// ABI has no trace-retrieval entry point.
    void traceSelection(const std::vector<Ranked>& scored, const MatchContext& context) const
    {
        // Check the level first: building the trace is O(candidates) per selection.
        if(scored.empty() || !::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))
        {
            return;
        }

        std::ostringstream candidates;
        for(size_t i = 0; i < scored.size(); ++i)
        {
            candidates << (i == 0 ? "" : " ") << toString(scored[i].entry->kernelId) << "="
                       << scored[i].score.reported;
        }

        // A two-layer model also decides the group, so the trace records it.
        std::ostringstream group;
        if(const auto feature = groupFeature())
        {
            group << " group=" << *feature << "=" << scored.front().group;
        }

        HIPDNN_PLUGIN_LOG_INFO("uhd trace: "
                               << _describedBy << " decided_by=" << traceDecidedBy()
                               << " metric=" << context.rankingMetric
                               << " winner=" << toString(scored.front().entry->kernelId)
                               << group.str() << " candidates=" << scored.size() << " arch="
                               << context.deviceProperties.gcnArchName << " uhd=" << _config.uhdId
                               << " adapter=" << _config.adapterType << " score_metric="
                               << (_config.scoreMetric.empty() ? "(none)" : _config.scoreMetric)
                               << " objective=" << _config.objective
                               << " features_hash=" << _config.featuresHash << " " << timingTrace()
                               << " ranked=[" << candidates.str() << "]");
    }

    /// RFC 0019 §9.4 timings as running averages: load per instance, prefix per selection,
    /// tail and score per candidate.
    std::string timingTrace() const
    {
        const auto selections = _timing.selections.load(std::memory_order_relaxed);
        const auto candidates = _timing.candidates.load(std::memory_order_relaxed);
        const auto per
            = [](uint64_t total, uint64_t count) { return count == 0 ? 0 : total / count; };
        std::ostringstream out;
        out << "load_ns=" << _timing.loadNs.load(std::memory_order_relaxed)
            << " prefix_ns=" << per(_timing.prefixNs.load(std::memory_order_relaxed), selections)
            << " tail_ns=" << per(_timing.tailNs.load(std::memory_order_relaxed), candidates)
            << " score_ns=" << per(_timing.scoreNs.load(std::memory_order_relaxed), candidates)
            << " over=" << selections << "sel/" << candidates << "cand";
        return out.str();
    }

    explicit UhdKernelHeuristic(std::string describedBy)
        : _describedBy(std::move(describedBy))
    {
    }

    UhdKernelHeuristic(uhd::UhdConfig config,
                       std::shared_ptr<const uhd::IUhdAdapter> adapter,
                       std::shared_ptr<const uhd::FeatureExtractor> extractor,
                       std::string describedBy)
        : _config(std::move(config))
        , _adapter(std::move(adapter))
        , _extractor(std::move(extractor))
        , _objectiveSign(objectiveSignOf(_config.objective))
        , _positiveRequired(
              uhd::score_transform::isPhysicalScore(_config.scoreMetric, _config.scoreTransform))
        , _describedBy(std::move(describedBy))
        , _hasDefaultModel(true)
    {
    }

    /// rank() sorts descending, so a `min` (cost) objective is negated.
    static double objectiveSignOf(const std::string& objective)
    {
        return objective == "min" ? -1.0 : 1.0;
    }

    /// The model's score in its metric's units, oriented so higher wins. An unusable value
    /// (reachable from legal descriptors) reports 0, RFC 0019 §5 step 7's "no measurement".
    CandidateScore scoreCandidate(const std::vector<double>& row) const
    {
        return scoreFromRaw(_adapter->score(row));
    }

    /// Transform inversion, range check and orientation of one raw adapter score.
    CandidateScore scoreFromRaw(const double raw) const
    {
        const double recovered = uhd::score_transform::applyInverse(raw, _config.scoreTransform);

        // RFC 0019 §8.3: `recovered` must be finite and, when physical, strictly positive
        // (log1p's inverse can yield a finite negative). Checked here because only this layer
        // knows the score's units.
        if(!uhd::score_transform::isRankableScore(recovered, _positiveRequired))
        {
            // rankWith counts these and reports once per ranking.
            _lastOutOfRangeRaw = raw;
            _lastOutOfRangeRecovered = recovered;
            return {-std::numeric_limits<double>::infinity(), 0.0};
        }

        // Orient only after the range check: oriented scores are negative under `min`.
        const double oriented = _objectiveSign * recovered;
        return {oriented, oriented};
    }

    /// Reports that no model covers the running architecture, once per heuristic.
    void reportNoModelForArchOnce(const std::string& arch, const std::string& metric) const
    {
        if(_reportedNoModelForArch.exchange(true))
        {
            return;
        }

        std::ostringstream named;
        for(const auto& [modelMetric, byArch] : _byMetric)
        {
            for(const auto& [candidate, descriptor] : byArch)
            {
                named << (named.tellp() == std::streampos(0) ? "" : ", ")
                      << (modelMetric.empty() ? "(none)" : modelMetric) << "@" << candidate;
            }
        }
        HIPDNN_PLUGIN_LOG_WARN(
            "uhd: " << _describedBy << " names no model for '" << arch << "' in metric '" << metric
                    << "' and no default ranker or 'default' entry (it names: " << named.str()
                    << "); kernels rank by priority, then descriptor id. "
                       "Further occurrences are not logged.");
    }

    /// Reports a model predicting outside the range its target can occupy, once per heuristic.
    /// ERROR because the score is wrong, not merely uncertain, and is discarded.
    void reportOutOfRangeOnce(size_t affected, size_t total) const
    {
        if(affected == 0 || _reportedScoreOutOfRange.exchange(true))
        {
            return;
        }
        HIPDNN_PLUGIN_LOG_ERROR(
            "uhd: " << _describedBy << " predicted a score its target cannot take for " << affected
                    << " of " << total << " candidates (raw=" << _lastOutOfRangeRaw
                    << ", recovered=" << _lastOutOfRangeRecovered << ", transform='"
                    << _config.scoreTransform
                    << "'). A metric value cannot be negative, so those scores are discarded and "
                       "those candidates rank last. "
                    << (affected == total
                            ? "Every candidate was affected, so this ranking is declared order "
                              "and the model contributed nothing."
                            : "The remaining candidates ranked on the model.")
                    << " Further occurrences for this heuristic are not logged.");
    }

    /// Authored (metric, architecture) entries and their lazily loaded per-engine models.
    std::map<std::string, std::map<std::string, HeuristicDescriptor>> _byMetric;
    std::map<std::string, std::set<std::string>> _unavailable;
    std::vector<std::string> _knobs;

    /// The KMD's declared field names, for checking lazily resolved models (§6.3 check 2).
    std::unordered_set<std::string> _kmdFields;
    mutable std::mutex _archMutex;
    /// Keyed by (metric, arch key): one UHD per metric per key, so the pair names a model.
    /// Holds successful loads only.
    mutable std::map<std::pair<std::string, std::string>, std::shared_ptr<const UhdKernelHeuristic>>
        _archCache;
    /// Keys whose load failure was already logged; guarded by _archMutex.
    mutable std::set<std::pair<std::string, std::string>> _archLoadFailuresReported;

    uhd::UhdConfig _config;
    std::shared_ptr<const IKernelHeuristic> _direct;
    std::shared_ptr<const uhd::IUhdAdapter> _adapter;
    std::shared_ptr<const uhd::FeatureExtractor> _extractor;
    double _objectiveSign = 1.0;
    /// Whether §8.3 requires this model's recovered score to be positive as well as finite.
    bool _positiveRequired = true;

    /// Mutable and atomic: ranking runs through a shared_ptr<const> from any thread.
    mutable std::atomic<bool> _reportedScoreOutOfRange{false};

    mutable std::atomic<bool> _reportedNoModelForArch{false};

    mutable std::atomic<double> _lastOutOfRangeRaw{0.0};
    mutable std::atomic<double> _lastOutOfRangeRecovered{0.0};
    std::string _describedBy;

    /// steady_clock: durations must not be affected by wall-clock adjustments.
    using Clock = std::chrono::steady_clock;

    static uint64_t elapsedNs(const Clock::time_point& start)
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }

    /// RFC 0019 §9.4 component timings in nanoseconds, accumulated per instance (so per
    /// architecture model). Relaxed atomics: read only for reporting.
    struct SelectionTiming
    {
        std::atomic<uint64_t> loadNs{0}; ///< Model parse + adapter build, once per instance.
        std::atomic<uint64_t> prefixNs{0}; ///< Shared problem/device slots, once per selection.
        std::atomic<uint64_t> tailNs{0}; ///< Per-candidate `$kernel.*` slots.
        std::atomic<uint64_t> scoreNs{0}; ///< Adapter inference plus the inverse transform.
        std::atomic<uint64_t> selections{0}; ///< Denominator for prefixNs.
        std::atomic<uint64_t> candidates{0}; ///< Denominator for tailNs and scoreNs.
    };
    mutable SelectionTiming _timing;

    /// False for an instance built by makeResolver, which has no model of its own.
    bool _hasDefaultModel = false;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
