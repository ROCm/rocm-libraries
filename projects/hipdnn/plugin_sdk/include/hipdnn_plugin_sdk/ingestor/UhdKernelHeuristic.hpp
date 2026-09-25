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
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>
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
    // Through deviceFeatureValues, never a second list: the benchmark recorder writes
    // the same names as `device.*` columns, and a vocabulary maintained twice drifts
    // into a model trained on a column the runtime cannot bind.
    uhd::FeatureExtractionContext::ValueMap vars;
    for(const auto& entry : deviceFeatureValues(deviceProperties))
    {
        // entry.first, not a captured structured binding: those are C++20.
        std::visit([&vars, &entry](auto held) { vars.emplace(entry.first, held); }, entry.second);
    }
    return vars;
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

inline uhd::FeatureExtractionContext::ValueMap kernelVarsFrom(const KernelDefinition& kernel)
{
    uhd::FeatureExtractionContext::ValueMap vars;
    for(const auto& [name, value] : kernel.metadata)
    {
        appendFeatureValue(vars, name, value);
    }
    vars.emplace("priority", kernel.priority);

    // `id` is not bound: a kernel's id is a UUID, and a feature that compared UUID bytes
    // would be ordering by authoring accident. The id already decides ties in rank().
    return vars;
}

} // namespace detail

/// @brief Ranks a catalog with the model a MODEL heuristic descriptor names.
///
/// @brief The `$kernel.*` axes a feature signature actually reads.
///
/// RFC 0019 §6.3 check 2 requires `F ⊆ set(UED.knobs)`, where F is this set. The two
/// directions are not symmetric, and the RFC originally asked for equality on the
/// assumption that the generation tool derives the knob list *from* the trained feature
/// set -- which would make a constant knob vanish from the UED the moment training
/// dropped it as unsplittable.
///
/// It cannot: `UED.knobs` is the engine's public knob surface, read by its UMDs and by
/// callers who tune, so a training run must not reshape it. A model that ignores a knob
/// is a dial the heuristic does not read -- worth a warning, and normal for a knob whose
/// column carried one value. A model ranking on an axis the UED never exposed is still a
/// load error: the caller cannot vary what selection depends on.
inline std::unordered_set<std::string> kernelAxesOf(const uhd::FeatureExtractor& extractor)
{
    std::unordered_set<std::string> axes;
    for(const auto& variable : extractor.getVariableRefs())
    {
        constexpr std::string_view PREFIX = "$kernel.";
        if(variable.rfind(PREFIX, 0) == 0)
        {
            axes.insert(variable.substr(PREFIX.size()));
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
    /// Builds an instance that holds no model of its own, only the per-(metric, arch)
    /// candidates.
    ///
    /// RFC 0019 §3.1 and §8.3: a role maps each architecture to one UHD per metric, resolved
    /// "exact gcnArchName, then `default`" within the requested metric. A UED may name
    /// per-arch models and no `default`, and that engine must still rank by model on the
    /// architectures it does name.
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
    ///        names. RFC 0019 §6.3 check 2 is two assertions over the same set, so both
    ///        halves arrive the same way: a caller that cannot supply one supplies an empty
    ///        list, and a model reading any `$kernel.*` axis is then refused rather than
    ///        passing a check nothing was given to check against.
    static std::shared_ptr<UhdKernelHeuristic>
        tryCreate(const HeuristicDescriptor& descriptor,
                  const std::string& describedBy,
                  const std::vector<std::string>& knobs = {},
                  const std::unordered_set<std::string>& kmdFields = {})
    {
        // RFC 0019 §9.4's first component: "descriptor load and model parse". Timed from
        // the top of the build so it covers configFrom, the extractor's signature
        // compilation and the adapter's artifact read -- everything paid once per
        // architecture before a single candidate is scored.
        const auto loadStart = Clock::now();
        try
        {
            // The descriptor IS the UHD -- no second file to open, so nothing here can
            // fail to load. It used to be a stub naming a FlatBuffer that held these
            // fields, which made the descriptor unreadable to save 134 bytes on a file
            // read once per engine.
            auto config = configFrom(descriptor);
            if(descriptor.adapter == UhdAdapter::STATIC_ORDER
               || (descriptor.adapter == UhdAdapter::NATIVE && config.featuresSignature.empty()))
            {
                auto built
                    = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(describedBy));
                built->_config = std::move(config);
                if(descriptor.adapter == UhdAdapter::STATIC_ORDER)
                {
                    built->_direct = std::make_shared<UnrankedKernelHeuristic>();
                }
                else
                {
                    built->_direct = std::make_shared<NativeKernelHeuristic>(
                        descriptor.nativeSymbol, describedBy);
                }
                built->_hasDefaultModel = true;
                return built;
            }

            auto extractor = std::make_shared<const uhd::FeatureExtractor>(
                config.featuresSignature, config.categoricalEncoding);
            // RFC 0019 §6.3 check 2: every axis the model ranks on must be a knob the
            // engine exposes. Returning nullptr degrades to declared order, which is what
            // §5 step 7 asks for on a broken feature contract -- the model would be
            // ranking on something the caller has no way to vary.
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

            // RFC 0019 §6.3 check 2, first assertion: `F ⊆ KMD.fields`. A feature can never
            // read a variant field the kernels do not carry -- an unbound `$kernel.*`
            // reference extracts nothing and the row the model scores is not the row it was
            // fitted on. §6.3 puts the check in both places on purpose ("the pipeline
            // enforces it when it emits the engine and the loader re-checks"), because a
            // descriptor set is drop-in: ValidateDescriptors catches a set built here, and
            // this catches a UED and KMD regenerated out of step or hand-edited after the
            // tool ran. Same routine on both sides, so the two cannot disagree about what
            // "reachable" means -- a `$kernel.*` nested inside a computed entry counts.
            const auto undeclared = extractor->getMissingKmdFields(kmdFields);
            if(!undeclared.empty())
            {
                std::vector<std::string> missing = undeclared;
                std::sort(missing.begin(), missing.end());
                std::vector<std::string> declared(kmdFields.begin(), kmdFields.end());
                std::sort(declared.begin(), declared.end());
                HIPDNN_PLUGIN_LOG_ERROR(
                    "uhd: " << describeDescriptor("heuristic", descriptor.name, descriptor.id)
                            << " on " << describedBy << " reads [" << join(missing)
                            << "], which its KMD does not declare as fields [" << join(declared)
                            << "]; RFC 0019 §6.3 requires the model's axes to be declared, so "
                               "the model is not used and kernels rank by priority, then id");
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
                HIPDNN_PLUGIN_LOG_ERROR(
                    "uhd: " << describedBy << " ranks on [" << join(unexposed)
                            << "], which its UED does not expose as knobs [" << join(exposed)
                            << "]; RFC 0019 §6.3 requires the model's axes to be exposed, so "
                               "the model is not used and kernels rank by priority, then id");
                return nullptr;
            }

            // The other direction is legal and expected. A knob whose column carried one
            // value cannot separate candidates, so training drops it from the signature --
            // and the UED still has to declare it, because it is the engine's public knob
            // surface and its UMDs and callers read it. Reported so an unread dial is
            // visible, never fatal: `uhd_gen knobs` is what tells the kernel author
            // whether to remove it, and that call is theirs.
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

            auto adapter = uhd::makeUhdAdapter(config);
            if(adapter == nullptr)
            {
                HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " names adapter '"
                                                << config.adapterType
                                                << "', which built no scorer");
                return nullptr;
            }

            // RFC 0019 §6.3: the signature the model was trained against must be the one
            // the extractor will produce. Both sides carry the hash; disagreeing means the
            // pair was assembled from two different training runs.
            if(!config.featuresSignature.empty()
               && extractor->getSignatureHash() != config.featuresHash)
            {
                HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " signature hashes disagree -- "
                                                << "descriptor declares '" << config.featuresHash
                                                << "', signature computes '"
                                                << extractor->getSignatureHash() << "'");
                return nullptr;
            }

            // RFC 0019 §6.3 check 4: the adapter must accept the row the extractor will hand
            // it. The hash above proves the *contract* is the one the model was trained
            // against; it says nothing about the artifact's own arity, and the two can
            // disagree -- a tree table carrying `num_features: 3` under a two-slot signature
            // has a matching features_hash and a column count that does not.
            //
            // Without this the short row is not even an error: TreeDataAdapter dispatches a
            // row shorter than `num_features` to its missing-value branch, so every split on
            // an absent column silently takes the default direction and the model scores --
            // wrongly, and quietly. EnginePredictor has always made this comparison on the L1
            // path; the kernel path did not, which is the asymmetry this closes.
            if(adapter->expectedFeatureCount() != extractor->featureCount())
            {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "uhd: " << describedBy << " model expects " << adapter->expectedFeatureCount()
                            << " features, its signature "
                            << "produces " << extractor->featureCount()
                            << "; the model is not used and kernels rank by priority, then id");
                return nullptr;
            }

            auto built = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(
                std::move(config), std::move(adapter), std::move(extractor), describedBy));
            built->_timing.loadNs.store(elapsedNs(loadStart), std::memory_order_relaxed);
            return built;
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " failed to load: " << e.what());
            return nullptr;
        }
    }

    /// The descriptor's own fields, with the artifact path resolved against the file it
    /// was declared in.
    ///
    /// A straight copy, because the descriptor IS the UHD -- there is no second document
    /// to reconcile it against. It used to be a four-field stub naming a FlatBuffer that
    /// held these fields, which made the descriptor unreadable to save 134 bytes on a
    /// file read once per engine.
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
        config.staticOrderFields = descriptor.staticOrderFields;
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
        // -Wswitch-default. The enum is closed and every member is handled above.
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

    /// Scores one kernel, extracting the whole row. rank() does not use this -- it shares
    /// the problem and device half across candidates -- so this is the path for a direct
    /// caller or a test.
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
        uhd::FeatureExtractionContext ctx;
        ctx.bindDeviceVars(detail::deviceVarsFrom(context.deviceProperties));
        ctx.bindQueryVars(detail::queryVarsFrom(bound));
        ctx.bindKernelVars(detail::kernelVarsFrom(kernel));
        // The reported form: this entry point answers "what is this kernel worth", and
        // ranking does not go through it -- rankScored is overridden and uses both forms.
        return scoreCandidate(_extractor->extract(ctx)).reported;
    }

    /// @brief Scores the matching architecture in physical units of `context.rankingMetric`,
    ///        best first in that metric's direction, including a singleton catalog.
    ///
    /// The single answer to "is this engine's score comparable against another engine's". It
    /// replaced a `scoreIsCalibrated()` accessor that read `_config.scoreCalibrated` off *this*
    /// object: on a resolver that config is nobody's, while the ranking comes from the
    /// architecture's own model -- so the flag answered for a descriptor that was not the one
    /// ranking, in both directions.
    ///
    /// Only the requested metric's own ranker answers (RFC 0019 §11.4): the default ranker
    /// picks kernels when a metric has none, but its number is in another metric, and
    /// reporting it would be the substitution §4.4 forbids. The model must be calibrated, and
    /// its objective is the metric's registered direction -- the parser already refuses any
    /// other -- so the order rankWith produced is already best-first in that direction.
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
        if(!_hasDefaultModel || metric == nullptr || _config.scoreMetric != metric->name
           || !_config.scoreCalibrated
           || _config.objective != hipdnn_data_sdk::utilities::objectiveOf(*metric)
           || (!_direct
               && (!_adapter || !_extractor
                   || !_adapter->isTrainedForArch(context.deviceProperties.gcnArchName))))
        {
            return {};
        }
        // A direct scorer is ordered higher-first by IKernelHeuristic::rankScored and never
        // sees the objective, so its order is best-first only for a metric where higher wins.
        if(_direct
           && metric->direction != hipdnn_data_sdk::utilities::MetricDirection::HIGHER_IS_BETTER)
        {
            return {};
        }
        auto ranking = rankWith(catalog, context);
        for(auto& candidate : ranking)
        {
            // A model's reported score is oriented so higher wins; undoing the orientation
            // recovers the physical value. 0 is the no-measurement sentinel either way, and
            // stays +0 rather than becoming -0 under a `min` objective.
            candidate.score = _direct ? uhd::score_transform::applyInverse(candidate.score,
                                                                           _config.scoreTransform)
                              : candidate.score == 0.0 ? 0.0
                                                       : _objectiveSign * candidate.score;
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

    /// RFC 0019 §12 asks which of the three decided. The base reports "native", which is right
    /// for a scorer compiled into the engine and wrong for a loaded model -- and this class's
    /// own trace line already said "model", so the two spellings disagreed. One source now.
    std::string traceDecidedBy() const override
    {
        // What this object decides by *absent an architecture*, which is all a context-free
        // accessor can honestly answer. A resolver holds candidates and no model, so its
        // default is declared order even though it will rank by model on the architectures
        // it does name. The per-ranking answer is the trace line, which is emitted where the
        // device and the metric are known -- and that is the one §12 actually specifies.
        return _hasDefaultModel ? "model" : "declared_order";
    }

    /// The signature entry the model groups on, `$` stripped, or nothing when it decides in one
    /// layer. Taken from the adapter's slot rather than from the descriptor's text, so it names
    /// the field the model is actually reading.
    std::optional<std::string> groupFeature() const override
    {
        // A resolver holds no model of its own. There is no device or request here to resolve
        // against, so it answers for what a default-metric request would rank with on the
        // `default` entry -- the same context-free reading traceDecidedBy() takes.
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
        // A signature entry is either a bare reference -- `"$kernel.solver_id"` -- or an inline
        // expression object. A caller wants the field, so the reference is unwrapped; an
        // expression has no field to name, and its own text is the closest honest label.
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
        // An empty catalog has nothing to order and nothing to score, and saying so needs no
        // architecture -- the one case that still short-circuits.
        //
        // A *sole* candidate no longer does. Its order is settled whatever it scores, but the
        // score is RFC 0019.13 §15.2's figure of merit and `detail::asScored` stamps the 0 that
        // §5 step 7 reserves for "no measurement" -- a claim about the model, not about how many
        // kernels survived filtering. Stamping it for a healthy model made a one-candidate
        // catalog indistinguishable from a degraded ranking. Resolving and scoring costs one
        // model load per engine per (metric, architecture), cached by resolveFor, which is what
        // the number meaning what it says is worth.
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
        // RFC 0019 §11.4: the requested metric's ranker when it resolves for this device,
        // else the engine's default ranker, recording which. Resolved here rather than at
        // load because descriptor discovery is a process-wide static that runs before any
        // device exists -- and §9.2 asks for load-on-demand with a per-engine cache anyway,
        // which is what this is.
        if(const auto choice = chooseRanker(metric, arch); choice.model)
        {
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: "
                                   << _describedBy << " metric=" << metric
                                   << " ranker=" << choice.source << " ranker_metric="
                                   << (choice.metric.empty() ? "(none)" : choice.metric)
                                   << " arch=" << arch);
            return choice.model->rankWith(catalog, context);
        }

        // §8.3's third step: exact, then `default`, then unavailable -- for the requested
        // metric and for the default ranker alike. A UED naming models only for other
        // architectures has nothing to say about this one, and using one of them anyway would
        // rank this device on a model trained for different hardware -- silently, since the
        // ranking would look normal.
        reportNoModelForArchOnce(arch, metric);

        // §12's trace, on this path too. Every other degraded path emits one; this branch was
        // added without it, so a selection that fell through for want of an architecture was
        // the one degradation the trace could not account for.
        HIPDNN_PLUGIN_LOG_INFO("uhd trace: " << _describedBy << " decided_by=declared_order"
                                             << " reason=no_model_for_arch"
                                             << " metric=" << metric << " arch=" << arch
                                             << " candidates=" << catalog.entries.size());
        return detail::asScored(detail::declaredOrder(catalog.entries));
    }

    /// The ranker @p metric's own UHD names for @p arch, resolved once per authored
    /// (metric, architecture key) and shared across all devices that use it. RFC 0019 §3.1's
    /// arch fallback stays inside the metric: (gfx942, time) falls back to (default, time),
    /// never to another metric. An explicitly unavailable/failed exact entry must never turn
    /// into a successful default model selection.
    std::shared_ptr<const UhdKernelHeuristic> resolveFor(const std::string& metric,
                                                         const std::string& arch) const
    {
        static const std::set<std::string> NONE_UNAVAILABLE;
        const auto refusedIt = _unavailable.find(metric);
        const auto& refused
            = refusedIt == _unavailable.end() ? NONE_UNAVAILABLE : refusedIt->second;
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
        std::string key;
        for(const auto& [candidate, descriptor] : models->second)
        {
            if(candidate != "default" && archMatches(arch, candidate, ArchMatchMode::PREFIX)
               && candidate.size() > key.size())
            {
                chosen = &descriptor;
                key = candidate;
            }
        }
        if(chosen == nullptr)
        {
            if(refused.count("default") != 0)
            {
                return nullptr;
            }
            if(const auto fallback = models->second.find("default");
               fallback != models->second.end())
            {
                chosen = &fallback->second;
                key = "default";
            }
        }
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
        // The per-arch model gets the same contract the `default` one was checked against:
        // RFC 0019 §8.3 resolves a different artifact per architecture, never a different
        // UED or KMD, so a check that ran only on the eagerly built model would leave every
        // lazily resolved arch unverified.
        auto loaded = tryCreate(*chosen, _describedBy, _knobs, _kmdFields);
        if(!loaded)
        {
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << _describedBy << " model for metric "
                                            << (metric.empty() ? "(none)" : "'" + metric + "'")
                                            << " on '" << key << "' failed to load");
        }
        _archCache.emplace(cacheKey, loaded);
        return loaded;
    }

private:
    /// Which ranker decides a kernel choice, and why -- the answer §11.4 asks the trace for.
    struct RankerChoice
    {
        std::shared_ptr<const UhdKernelHeuristic> model;
        /// `metric` when the requested metric's own ranker decided, `default` when the
        /// engine's default ranker stood in for it.
        const char* source = "declared_order";
        /// The metric the deciding UHD declares; empty for the metric-less ranker.
        std::string metric;
    };

    /// True for an instance built by makeResolver, which ranks through per-(metric, arch)
    /// children rather than a model of its own.
    bool isResolver() const
    {
        return !_byMetric.empty() || !_unavailable.empty();
    }

    /// RFC 0019 §3.1 and §11.4: kernel choice for @p metric uses that metric's ranker when it
    /// resolves for @p arch, else the engine's default ranker -- the metric-less
    /// `sort_kernel_catalog` UHD if any, else the one for DEFAULT_RANKING_METRIC -- else
    /// nothing, and the caller falls back to static order. Each candidate resolves with its
    /// own metric's arch fallback, so no step borrows another metric's arch entry.
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
        /// Higher wins. -infinity when there is no measurement, so an unmeasured candidate
        /// sorts last whichever way the objective points.
        double ordering;
        /// RFC 0019.13 §15.2's figure of merit. 0 when there is no measurement, matching the
        /// value §5 step 7 gives that condition at the engine level.
        double reported;
    };

    /// One scored candidate, as the ranking holds it before it becomes a ScoredKernel.
    struct Ranked
    {
        CandidateScore score;
        const KernelDefinition* entry;
        /// The grouping feature's value for this candidate, NaN where the model decides in
        /// one layer. See ScoredKernel::group for why NaN rather than a sentinel.
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
            // §9.4 asks for the two halves to be timed apart, because the prefix is paid
            // once per graph while the tail is the O(N) term the RFC calls "the main lever
            // on selection cost" -- one aggregate number cannot tell them apart.
            const auto prefixStart = Clock::now();
            auto features = _extractor->prepare(ctx);
            _timing.prefixNs.fetch_add(elapsedNs(prefixStart), std::memory_order_relaxed);
            _timing.selections.fetch_add(1, std::memory_order_relaxed);

            // Rows first, then one scoring call. A grouped model decides which group wins by
            // comparing candidates against each other, which no per-row call can express; for
            // a single-layer model the adapter's default scoreBatch is the loop this replaces,
            // so the per-candidate cost §9.4 budgets is unchanged either way.
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

            // The slot the model grouped on, read from the row the model was handed rather
            // than re-derived from the candidate's metadata: a derived value or a categorical
            // encoding could make the two disagree, and the reported answer would then name a
            // group that did not decide.
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

            // Two different things arrive as -infinity here, and reporting them alike turns
            // §12's loudest diagnostic into noise.
            //
            // The adapter returns -infinity for a candidate it declines to score: a grouped
            // model excludes every group but the one layer 1 chose. That is a decision the
            // model made, and counting it would report "predicted a score its target cannot
            // take" on every grouped ranking -- an error message about a training defect,
            // emitted for the design working exactly as intended.
            //
            // A finite raw score that fails the range check is the real thing that error is
            // for. Only those are counted, and only the candidates the model actually scored
            // are the population it is counted against, so "every candidate was affected"
            // keeps meaning "the model contributed nothing".
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

            // scoreCandidate already replaced any non-finite value with -infinity, so the
            // comparator sees only real numbers. That matters beyond tidiness: NaN compares
            // false both ways, so it reads as "equivalent" to every element while real scores
            // stay ordered among themselves, which violates the strict weak ordering
            // std::stable_sort requires -- undefined behaviour, not merely a wrong order.
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
            // The whole ranking falls back, not the kernels that happened to fail. A mix
            // of model scores and sentinels is neither order, and RFC 0019 §5 asks for a
            // degraded ranking rather than a partial one.
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << _describedBy << " failed while ranking: " << e.what()
                                            << "; kernels rank by priority, then descriptor id");
            // RFC 0019 §12 wants the trace to say *whether the model or a fallback decided*,
            // so the degraded path is traced too. A trace that only ever appears on success
            // cannot answer the question it exists for.
            // "declared_order", the same word UnrankedKernelHeuristic reports, not a fourth
            // synonym. A degraded ranking is a degraded ranking however it got there; `reason`
            // carries the difference. Two spellings for one condition is what makes a trace
            // unassertable, and unassertable observability is the thing §12 is trying to avoid.
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: " << _describedBy << " decided_by=declared_order"
                                                 << " reason=ranking_failed"
                                                 << " metric=" << context.rankingMetric
                                                 << " candidates=" << catalog.entries.size()
                                                 << " uhd=" << _config.uhdId
                                                 << " adapter=" << _config.adapterType
                                                 << " features_hash=" << _config.featuresHash);
            // Declared order carries no model score. It reports 0 -- RFC 0019 §5 step 7's
            // value for "no measurement" -- so a degraded ranking and a model that scored zero
            // describe themselves the same way, which is what lets calibratedRanking apply one
            // rule. traceDecidedBy() is where the two are told apart.
            return detail::asScored(detail::declaredOrder(catalog.entries));
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
    void traceSelection(const std::vector<Ranked>& scored, const MatchContext& context) const
    {
        // The level check precedes the work. Building the candidate string walks every kernel
        // and formats a double per entry, on every graph -- so doing it before asking whether
        // anyone is listening put a per-selection cost on builds that log nothing.
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

        // §12 asks for "whether the model or a fallback decided". Where the model decides in two
        // layers, half of what it decided is the group, and a trace naming only the winning
        // kernel would not record it.
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

    /// RFC 0019 §9.4's three components, as the per-unit figures the 2x-of-native budget is
    /// stated in: load is per architecture, prefix is per selection, and tail and score are
    /// per candidate -- which is the only form in which a regression is attributable, since
    /// a sum over a run conflates a bigger catalog with a slower model.
    ///
    /// Cumulative, so a single line is a running average rather than one noisy sample. The
    /// `native` adapter measures through this same path (only a `native` UHD with no
    /// features_signature bypasses it), so it remains the baseline §9.4 compares against.
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
        // rank() sorts descending, so a model predicting a cost rather than a rate has to
        // be negated. Omitting this silently inverts every latency-trained UHD.
        , _objectiveSign(_config.objective == "min" ? -1.0 : 1.0)
        , _describedBy(std::move(describedBy))
        , _hasDefaultModel(true)
    {
    }

    /// The model's score, returned to its metric's units and oriented so higher always wins,
    /// or 0 when there is no usable number.
    ///
    /// Non-finite is reachable without anything malformed: `applyInverse` reports out-of-domain
    /// as NaN, and a GBDT raw score is unbounded, so a `log`/`exp`/`sqrt`-transformed model
    /// predicting a negative value lands here on a legal descriptor. A native or custom_library
    /// scorer can return anything at all.
    ///
    /// Zero, not NaN, because RFC 0019 §5 step 7 already fixed what "no measurement" looks like
    /// one layer up -- the engine reports no estimate and "loses on merit rather than by
    /// exception" -- and a per-kernel score that means the same thing should say it the same
    /// way. Nothing needs the two distinguished: §15.2's callers use the order, and the caller
    /// that reads the value is calibratedRanking, which withholds a ranking whose top score is
    /// not a valid measurement.
    CandidateScore scoreCandidate(const std::vector<double>& row) const
    {
        return scoreFromRaw(_adapter->score(row));
    }

    /// The half of scoring that does not touch the adapter: transform inversion, the range
    /// check, orientation. Split out so a batched ranking applies exactly the same rules to
    /// a score the adapter produced for a whole catalog at once.
    CandidateScore scoreFromRaw(const double raw) const
    {
        const double recovered = uhd::score_transform::applyInverse(raw, _config.scoreTransform);

        // `recovered` is a physical quantity before any orientation is applied: throughput for
        // a calibrated model, and a cost -- a time -- for the `min` targets §15.1 permits.
        // RFC 0019.13 §8.4 names no target that can be negative, so a negative value here is
        // the model predicting outside the range it was fitted to. That is a training defect,
        // not a slow kernel, and it is refused whatever the model declares.
        //
        // Only some transforms make it loud. log's inverse yields NaN, but log1p's yields a
        // finite negative, and log1p is what uhd_gen emits by default -- so the most common
        // configuration is the one a finite-only check lets through.
        //
        // Bounded here rather than in the adapter because this is the only layer that knows
        // what the number means: TreeDataAdapter sums leaves and has no transform and no units.
        if(!std::isfinite(recovered) || recovered < 0.0)
        {
            // Not reported here. One ranking can trip this for a single candidate or for all
            // of them, and those mean different things -- a bad extrapolation versus a model
            // that is useless on this problem. rankWith counts them and reports which.
            _lastOutOfRangeRaw = raw;
            _lastOutOfRangeRecovered = recovered;
            return {-std::numeric_limits<double>::infinity(), 0.0};
        }

        // Orientation is applied only to a value that survived the range check, which is what
        // keeps the two ideas apart. A negative *oriented* score is ordinary -- `objective: min`
        // negates a cost, so every real candidate scores below zero -- while a negative
        // *recovered* value is never meaningful. Reporting 0 as the ordering key would have
        // made an unmeasured candidate outrank every measured one under that objective.
        const double oriented = _objectiveSign * recovered;
        return {oriented, oriented};
    }

    /// Reports that no model covers the running architecture, once per heuristic.
    ///
    /// The engine still selects, by declared order, which RFC 0019 §5 step 7 makes a legal
    /// ranking -- so without this the only symptom is that a UHD-carrying engine quietly stops
    /// using its UHD on some machines and not others.
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
    ///
    /// ERROR, not WARN. The one WARN peer on this path is "not trained for this architecture",
    /// which RFC 0019 §9.3 treats as still-useful -- the model is outside its distribution but
    /// its ordering may hold. This is the other kind: a negative throughput is not a value the
    /// target can take, so the number is wrong rather than uncertain and the score is
    /// discarded. That puts it with the hash-mismatch and load-failure peers, which are ERROR.
    ///
    /// Once per heuristic, because the condition is a property of the model and so recurs for
    /// every graph; a per-ranking message would bury what it is trying to report. The counts
    /// are what make it diagnosable -- @p affected of @p total says whether this was a single
    /// bad extrapolation or a model that cannot rank this problem at all.
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

    /// The KMD's declared field names, carried so a per-arch model resolved later faces
    /// RFC 0019 §6.3 check 2's first assertion as well.
    std::unordered_set<std::string> _kmdFields;
    mutable std::mutex _archMutex;
    /// Keyed by (metric, arch key): one UHD per metric per key, so the pair names a model.
    mutable std::map<std::pair<std::string, std::string>, std::shared_ptr<const UhdKernelHeuristic>>
        _archCache;

    uhd::UhdConfig _config;
    std::shared_ptr<const IKernelHeuristic> _direct;
    std::shared_ptr<const uhd::IUhdAdapter> _adapter;
    std::shared_ptr<const uhd::FeatureExtractor> _extractor;
    double _objectiveSign = 1.0;

    /// Set the first time an out-of-range score is reported. Mutable and atomic because
    /// ranking runs through a shared_ptr<const> from any thread.
    mutable std::atomic<bool> _reportedScoreOutOfRange{false};

    /// Set the first time an architecture resolves to no model at all.
    mutable std::atomic<bool> _reportedNoModelForArch{false};

    /// Atomic diagnostic samples avoid data races between concurrent selections.
    mutable std::atomic<double> _lastOutOfRangeRaw{0.0};
    mutable std::atomic<double> _lastOutOfRangeRecovered{0.0};
    std::string _describedBy;

    /// steady_clock, not system_clock: these are durations, and a wall-clock adjustment
    /// mid-selection must not show up as negative feature-extraction time.
    using Clock = std::chrono::steady_clock;

    static uint64_t elapsedNs(const Clock::time_point& start)
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }

    /// RFC 0019 §9.4: "The components should be **wall-clocked separately** -- descriptor
    /// load and model parse, feature extraction (shared prefix vs. per-candidate tail), and
    /// scoring -- so a regression is attributable and so the cost of different adapters can
    /// be compared directly". These are the numbers that requirement asks to exist; §12's
    /// trace line reports them, so the `native` baseline and a `tree_data` model can be
    /// compared on the same graph without a separate harness.
    ///
    /// Nanoseconds, accumulated over every selection this instance served. The counters sit
    /// on the per-arch child that actually ranks, so each architecture's model is measured
    /// separately rather than averaged with its siblings. Relaxed ordering throughout: these
    /// are monotonic accumulators read for reporting, never for synchronisation.
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

    /// False for an instance built by makeResolver: it carries candidates but no model of
    /// its own, so §8.3's `default` step has nothing to fall back to.
    bool _hasDefaultModel = false;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
