// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <atomic>
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
    /// Builds an instance that holds no model of its own, only the per-arch candidates.
    ///
    /// RFC 0019 §8.3 resolves "exact gcnArchName, then `default`". A UED may name per-arch
    /// models and no `default`, and that engine must still rank by model on the architectures
    /// it does name. Previously such a UED produced no object at all -- there was nothing to
    /// build eagerly -- so the whole map was discarded and the engine ranked by declared order
    /// everywhere, including on the architectures it had a model for.
    static std::shared_ptr<UhdKernelHeuristic>
        makeArchResolver(const std::map<std::string, HeuristicDescriptor>& byArch,
                         const std::string& describedBy,
                         const std::vector<std::string>& knobs)
    {
        auto built = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(describedBy));
        built->_byArch = byArch;
        built->_knobs = knobs;
        return built;
    }

    static std::shared_ptr<UhdKernelHeuristic> tryCreate(const HeuristicDescriptor& descriptor,
                                                         const std::string& describedBy,
                                                         const std::vector<std::string>& knobs = {},
                                                         const std::map<std::string,
                                                                        HeuristicDescriptor>&
                                                             byArch = {})
    {
        try
        {
            // The descriptor IS the UHD -- no second file to open, so nothing here can
            // fail to load. It used to be a stub naming a FlatBuffer that held these
            // fields, which made the descriptor unreadable to save 134 bytes on a file
            // read once per engine.
            auto config = configFrom(descriptor);

            // RFC 0019 §6.3 check 2: the exposed knobs *are* the model's feature axes.
            // Returning nullptr degrades to declared order, which is what §5 step 7 asks for
            // on a broken feature contract -- the model's inputs are not the ones it was
            // trained on, so its scores would be wrong.
            const auto axes = kernelAxesOf(config.featuresSignature);
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

            auto adapter = uhd::makeUhdAdapter(config);
            if(adapter == nullptr)
            {
                HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " names adapter '"
                                                << config.adapterType
                                                << "', which built no scorer");
                return nullptr;
            }

            auto extractor = std::make_shared<const uhd::FeatureExtractor>(
                config.featuresSignature, config.derived, config.categoricalEncoding);

            // RFC 0019 §6.3: the signature the model was trained against must be the one
            // the extractor will produce. Both sides carry the hash; disagreeing means the
            // pair was assembled from two different training runs.
            if(extractor->getSignatureHash() != config.featuresHash)
            {
                HIPDNN_PLUGIN_LOG_ERROR("uhd: " << describedBy << " signature hashes disagree -- "
                                                << "descriptor declares '" << config.featuresHash
                                                << "', signature computes '"
                                                << extractor->getSignatureHash() << "'");
                return nullptr;
            }

            auto built = std::shared_ptr<UhdKernelHeuristic>(new UhdKernelHeuristic(
                std::move(config), std::move(adapter), std::move(extractor), describedBy));
            // Kept for RFC 0019 §8.3: the arch this was built from is whatever the loader
            // resolved (the `default` entry), and rank() re-resolves against the running
            // device the first time it sees one that these do not describe.
            built->_byArch = byArch;
            built->_knobs = knobs;
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
        config.objective = descriptor.objective;
        config.scoreUnits = descriptor.score.units;
        config.scoreCalibrated = descriptor.score.calibrated;
        config.scoreTransform = descriptor.score.transform;
        config.staticOrderFields = descriptor.staticOrderFields;
        config.nativeSymbol = descriptor.nativeSymbol;

        for(const auto& entry : descriptor.derived)
        {
            config.derived.emplace_back(entry.name, entry.expression);
        }

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
        // -Wswitch-default. The enum is closed and every member is handled above.
        default:
            config.adapterType = "static_order";
            break;
        }

        if(!descriptor.modelArtifactPath.empty())
        {
            config.modelArtifactPath
                = (descriptor.baseDir / descriptor.modelArtifactPath).string();
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
        uhd::FeatureExtractionContext ctx;
        ctx.bindDeviceVars(detail::deviceVarsFrom(context.deviceProperties));
        ctx.bindQueryVars(detail::queryVarsFrom(bound));
        ctx.bindKernelVars(detail::kernelVarsFrom(kernel));
        // The reported form: this entry point answers "what is this kernel worth", and
        // ranking does not go through it -- rankScored is overridden and uses both forms.
        return scoreCandidate(_extractor->extract(ctx)).reported;
    }

    /// Whatever the UHD declared. §15.1 already refused the one combination that would make
    /// this incoherent -- calibrated together with a descending objective -- so a calibrated
    /// score reaching here is ascending TFLOPS, which is what §11.3 asks a cross-engine
    /// comparison to be given.
    bool scoreIsCalibrated() const override
    {
        return _config.scoreCalibrated;
    }

    /// RFC 0019 §12 asks which of the three decided. The base reports "native", which is right
    /// for a scorer compiled into the engine and wrong for a loaded model -- and this class's
    /// own trace line already said "model", so the two spellings disagreed. One source now.
    std::string traceDecidedBy() const override
    {
        // What this object decides by *absent an architecture*, which is all a context-free
        // accessor can honestly answer. A resolver built by makeArchResolver holds candidates
        // and no model, so its default is declared order even though it will rank by model on
        // the architectures it does name. The per-ranking answer is the trace line, which is
        // emitted where the device is known -- and that is the one §12 actually specifies.
        return _hasDefaultModel ? "model" : "declared_order";
    }

    std::vector<ScoredKernel> rankScored(const Catalog& catalog,
                                         const MatchContext& context) const override
    {
        // RFC 0019 §8.3: exact gcnArchName, then `default`. Resolved here rather than at
        // load because descriptor discovery is a process-wide static that runs before any
        // device exists -- and §9.2 asks for load-on-demand with a per-engine cache anyway,
        // which is what this is.
        if(const auto forArch = resolveForArch(context.deviceProperties.gcnArchName))
        {
            return forArch->rankWith(catalog, context);
        }
        if(_hasDefaultModel)
        {
            return rankWith(catalog, context);
        }

        // §8.3's third step, which had no implementation: exact, then `default`, then
        // unavailable. A UED naming models only for other architectures has nothing to say
        // about this one, and using one of them anyway would rank this device on a model
        // trained for different hardware -- silently, since the ranking would look normal.
        reportNoModelForArchOnce(context.deviceProperties.gcnArchName);

        // §12's trace, on this path too. Every other degraded path emits one; this branch was
        // added without it, so a selection that fell through for want of an architecture was
        // the one degradation the trace could not account for.
        HIPDNN_PLUGIN_LOG_INFO("uhd trace: " << _describedBy << " decided_by=declared_order"
                                             << " reason=no_model_for_arch"
                                             << " arch=" << context.deviceProperties.gcnArchName
                                             << " candidates=" << catalog.entries.size());
        return detail::asScored(detail::declaredOrder(catalog.entries));
    }

    /// @returns the model for @p arch when the UED named a different one for it, else
    ///          nullptr, meaning "the model this object already holds applies".
    std::shared_ptr<const UhdKernelHeuristic> resolveForArch(const std::string& arch) const
    {
        if(arch.empty() || _byArch.empty())
        {
            return nullptr;
        }
        if(_byArch.size() == 1 && _byArch.count("default") == 1)
        {
            // The only entry is the `default` one, which is what was built eagerly. Every
            // architecture gets it, so there is nothing to look up: no lock, no cache.
            //
            // Keyed on the entry being `default`, not on the count. A single *arch-named*
            // entry is not a universal model -- treating it as one is how a gfx950-only UHD
            // came to rank every device, including the ones it says nothing about.
            return nullptr;
        }

        const std::lock_guard<std::mutex> lock(_archMutex);
        if(const auto cached = _archCache.find(arch); cached != _archCache.end())
        {
            return cached->second;
        }

        const HeuristicDescriptor* chosen = nullptr;
        for(const auto& [candidate, descriptor] : _byArch)
        {
            // archMatches, not ==: a device reports its features (`gfx942:sramecc+:xnack-`)
            // while an authored entry carries a bare id, so equality would miss every real
            // device.
            if(candidate != "default" && archMatches(arch, candidate, ArchMatchMode::PREFIX))
            {
                chosen = &descriptor;
                break;
            }
        }
        if(chosen == nullptr)
        {
            // The `default` entry is what was built eagerly, so falling back to it means
            // using this object. Caching the null keeps the miss from re-scanning per graph.
            _archCache.emplace(arch, nullptr);
            return nullptr;
        }

        auto loaded = tryCreate(*chosen, _describedBy, _knobs);
        if(loaded == nullptr)
        {
            HIPDNN_PLUGIN_LOG_ERROR("uhd: " << _describedBy << " names a model for '" << arch
                                            << "' that could not be brought up; the default "
                                               "model ranks instead");
        }
        _archCache.emplace(arch, loaded);
        return loaded;
    }

private:
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
    };

    std::vector<ScoredKernel> rankWith(const Catalog& catalog,
                                       const MatchContext& context) const
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

            std::vector<Ranked> scored;
            scored.reserve(catalog.entries.size());
            for(const auto& entry : catalog.entries)
            {
                ctx.clearKernelVars();
                ctx.bindKernelVars(detail::kernelVarsFrom(entry));

                std::vector<double> row = sharedRow;
                _extractor->extractKernelInto(ctx, row);
                scored.push_back({scoreCandidate(row), &entry});
            }

            const auto outOfRange = static_cast<size_t>(
                std::count_if(scored.begin(), scored.end(), [](const Ranked& candidate) {
                    return !std::isfinite(candidate.score.ordering);
                }));
            reportOutOfRangeOnce(outOfRange, scored.size());

            // scoreCandidate already replaced any non-finite value with -infinity, so the
            // comparator sees only real numbers. That matters beyond tidiness: NaN compares
            // false both ways, so it reads as "equivalent" to every element while real scores
            // stay ordered among themselves, which violates the strict weak ordering
            // std::stable_sort requires -- undefined behaviour, not merely a wrong order.
            std::stable_sort(scored.begin(),
                             scored.end(),
                             [](const auto& a, const auto& b) {
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
                ordered.push_back({candidate.entry->kernelId, candidate.score.reported});
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
            // "declared_order", the same word UnrankedKernelHeuristic reports, not a fourth
            // synonym. A degraded ranking is a degraded ranking however it got there; `reason`
            // carries the difference. Two spellings for one condition is what makes a trace
            // unassertable, and unassertable observability is the thing §12 is trying to avoid.
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: " << _describedBy << " decided_by=declared_order"
                                                 << " reason=ranking_failed"
                                                 << " candidates=" << catalog.entries.size()
                                                 << " uhd=" << _config.uhdId
                                                 << " adapter=" << _config.adapterType
                                                 << " features_hash=" << _config.featuresHash);
            // Declared order carries no model score. It reports 0 -- RFC 0019 §5 step 7's
            // value for "no measurement" -- so a degraded ranking and a model that scored zero
            // describe themselves the same way, which is what lets estimateTflops apply one
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

        HIPDNN_PLUGIN_LOG_INFO("uhd trace: "
                               << _describedBy << " decided_by=" << traceDecidedBy()
                               << " winner=" << toString(scored.front().entry->kernelId)
                               << " candidates=" << scored.size()
                               << " arch=" << context.deviceProperties.gcnArchName
                               << " uhd=" << _config.uhdId
                               << " adapter=" << _config.adapterType
                               << " objective=" << _config.objective
                               << " features_hash=" << _config.featuresHash
                               << " ranked=[" << candidates.str() << "]");
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

    /// The model's raw output, returned to its declared units and oriented so that larger
    /// is better.
    /// The model's score, oriented so higher always wins, or 0 when there is no usable number.
    ///
    /// Non-finite is reachable without anything malformed: `applyInverse` reports out-of-domain
    /// as NaN, and a GBDT raw score is unbounded, so a `log`/`exp`/`sqrt`-transformed model
    /// predicting a negative value lands here on a legal descriptor. A native or custom_library
    /// scorer can return anything at all.
    ///
    /// Zero, not NaN, because RFC 0019 §5 step 7 already fixed what "no measurement" looks like
    /// one layer up -- "the engine reports an estimated throughput of 0... and loses on merit
    /// rather than by exception" -- and a per-kernel score that means the same thing should say
    /// it the same way. Nothing needs the two distinguished: §15.2's callers use the order, and
    /// the one caller that reads the value is estimateTflops, which reports 0 for this case too.
    CandidateScore scoreCandidate(const std::vector<double>& row) const
    {
        const double raw = _adapter->score(row);
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
    void reportNoModelForArchOnce(const std::string& arch) const
    {
        if(_reportedNoModelForArch.exchange(true))
        {
            return;
        }

        std::ostringstream named;
        for(const auto& [candidate, descriptor] : _byArch)
        {
            named << (named.tellp() == std::streampos(0) ? "" : ", ") << candidate;
        }
        HIPDNN_PLUGIN_LOG_WARN("uhd: " << _describedBy << " names no model for '" << arch
                                       << "' and no 'default' (it names: " << named.str()
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
                    << "'). A throughput cannot be negative, so those scores are discarded and "
                       "those candidates rank last. "
                    << (affected == total
                            ? "Every candidate was affected, so this ranking is declared order "
                              "and the model contributed nothing."
                            : "The remaining candidates ranked on the model.")
                    << " Further occurrences for this heuristic are not logged.");
    }

    /// RFC 0019 §3.1's arch -> UHD map, and §9.2's per-engine cache of what has been
    /// loaded from it. Empty when the UED named a bare id, in which case the eagerly built
    /// model above serves every architecture.
    std::map<std::string, HeuristicDescriptor> _byArch;
    std::vector<std::string> _knobs;
    mutable std::mutex _archMutex;
    mutable std::map<std::string, std::shared_ptr<const UhdKernelHeuristic>> _archCache;

    uhd::UhdConfig _config;
    std::shared_ptr<const uhd::IUhdAdapter> _adapter;
    std::shared_ptr<const uhd::FeatureExtractor> _extractor;
    double _objectiveSign;

    /// Set the first time an out-of-range score is reported. Mutable and atomic because
    /// ranking runs through a shared_ptr<const> from any thread.
    mutable std::atomic<bool> _reportedScoreOutOfRange{false};

    /// Set the first time an architecture resolves to no model at all.
    mutable std::atomic<bool> _reportedNoModelForArch{false};

    /// The most recent offending pair, carried to the report so it names a concrete value.
    /// Racy under concurrent ranking, which is acceptable: it is diagnostic detail on a message
    /// that fires once, and any offending pair illustrates the condition as well as another.
    mutable double _lastOutOfRangeRaw = 0.0;
    mutable double _lastOutOfRangeRecovered = 0.0;
    std::string _describedBy;

    /// False for an instance built by makeArchResolver: it carries candidates but no model of
    /// its own, so §8.3's `default` step has nothing to fall back to.
    bool _hasDefaultModel = false;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
