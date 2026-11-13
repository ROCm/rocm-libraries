// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

/// @file KernelFeatureRows.hpp
/// @brief The kernel half of a UHD feature row: how a KMD becomes `$kernel.*` bindings, and
///        the per-catalog builder that turns a catalog's kernels into rows (RFC 0019 §9.3).
namespace hipdnn_plugin_sdk::ingestor::detail
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

/// Builds the feature rows of a catalog's kernels for one signature, with the per-kernel work
/// done once per kernel instead of once per candidate.
///
/// A signature entry that reads only `$kernel.*` is a function of the kernel alone, so its
/// value is cached per kernel. An entry that mixes `$kernel.*` with problem or device symbols
/// is a function of the selection's problem and of the kernel values it reads, so a selection
/// evaluates it once per distinct set of those values, against a context in which only the
/// `kernel.*` names the mixed entries read are rebound, in place. Rows are bit-identical to
/// binding kernelVarsFrom() and calling FeatureExtractor::extract() per kernel.
///
/// Safe for concurrent build() calls (RFC 0019 §9.3: rankWith is const and concurrent).
class KernelRowBuilder
{
public:
    explicit KernelRowBuilder(std::shared_ptr<const uhd::FeatureExtractor> extractor)
        : _extractor(std::move(extractor))
    {
    }

    /// Rows of @p kernels in order, row-major: `kernels.size() * featureCount()` doubles.
    /// @param ctx The selection's problem bindings. build() binds the mixed entries'
    ///        `kernel.*` names into it, so it must not be shared with another thread; it
    ///        unbinds them first, so a context an earlier build() used is safe to pass.
    /// @param prefix Output of FeatureExtractor::prepare() on @p ctx.
    /// @throws uhd::JsonLogicError where extract() would, for the first such kernel.
    std::vector<double> build(uhd::FeatureExtractionContext& ctx,
                              const uhd::FeatureExtractor::Workspace& prefix,
                              const std::vector<const KernelDefinition*>& kernels) const
    {
        const size_t width = _extractor->featureCount();
        if(prefix.owner != _extractor.get() || prefix.values.size() != width)
        {
            throw uhd::JsonLogicError("Feature workspace belongs to a different signature");
        }

        const auto [facts, keyCount] = factsFor(kernels);
        const size_t mixedCount = _extractor->mixedCount();
        const auto& names = _extractor->mixedKernelVariables();
        // Start from no kernel bound: a name left by an earlier build() on this context would
        // otherwise stand in for a field the first kernel lacks.
        for(const auto& name : names)
        {
            ctx.unbindSlot(name);
        }
        // Per selection: the value of each distinct (entry, kernel values) key once computed.
        std::vector<double> memo(keyCount);
        std::vector<uint8_t> computed(keyCount, 0);
        // Mapped values of the names currently bound in ctx; null while a name is unbound.
        std::vector<uhd::VariableContext::ValueType*> bound(names.size(), nullptr);

        std::vector<double> rows;
        rows.reserve(kernels.size() * width);
        for(size_t index = 0; index < kernels.size(); ++index)
        {
            // Append, not resize: the row is overwritten with the prefix at once.
            rows.insert(rows.end(), prefix.values.begin(), prefix.values.end());
            double* row = rows.data() + (index * width);
            const KernelFacts& fact = *facts[index];
            _extractor->applyKernelOnly(row, fact.kernelOnly);

            bool bindingsCurrent = false;
            for(size_t j = 0; j < mixedCount; ++j)
            {
                const uint32_t key = fact.mixedKeys[j];
                if(computed[key] == 0)
                {
                    if(!bindingsCurrent)
                    {
                        bindKernel(ctx, names, fact, bound);
                        bindingsCurrent = true;
                    }
                    memo[key] = _extractor->evaluateMixed(j, ctx);
                    computed[key] = 1;
                }
                row[_extractor->mixedFeatureIndex(j)] = memo[key];
            }
        }
        return rows;
    }

    /// Kernels whose facts are cached.
    size_t cachedKernelCount() const
    {
        const std::shared_lock<std::shared_mutex> lock(_mutex);
        return _facts.size();
    }

private:
    static constexpr std::string_view KERNEL_PREFIX = "kernel.";

    using Slot = std::optional<uhd::VariableContext::ValueType>;

    /// What the signature needs from one kernel, derived once.
    struct KernelFacts
    {
        /// FeatureExtractor::kernelOnlyValues() for the kernel.
        std::vector<double> kernelOnly;
        /// The value of each mixedKernelVariables() name in order; nullopt when the kernel
        /// has no such field, so the name stays unbound for it.
        std::vector<Slot> slots;
        /// Per mixed entry, the id of its (entry, values of the names it reads) key. Kernels
        /// that agree on those values share an id, and so share the entry's value.
        std::vector<uint32_t> mixedKeys;
    };

    /// One character naming the alternative a key part holds, so equal bytes of different
    /// types never collide.
    template <typename T>
    static constexpr char typeTag()
    {
        if constexpr(std::is_same_v<T, double>)
        {
            return 'd';
        }
        else if constexpr(std::is_same_v<T, int64_t>)
        {
            return 'i';
        }
        else
        {
            return 'b';
        }
    }

    /// @p value in a form two keys share only when the entries cannot tell the values apart.
    static void appendKeyPart(std::string& key, const Slot& value)
    {
        if(!value)
        {
            key.push_back('u');
            return;
        }
        std::visit(
            [&key](const auto& held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr(std::is_same_v<T, std::string>)
                {
                    key.push_back('s');
                    const auto size = held.size();
                    key.append(reinterpret_cast<const char*>(&size), sizeof(size));
                    key += held;
                }
                else
                {
                    key.push_back(typeTag<T>());
                    key.append(reinterpret_cast<const char*>(&held), sizeof(held));
                }
            },
            *value);
    }

    /// Facts before their keys are numbered; numbering needs the lock.
    struct Derived
    {
        KernelFacts facts;
        std::vector<std::string> keys;
    };

    Derived deriveFacts(const KernelDefinition& kernel) const
    {
        const auto vars = kernelVarsFrom(kernel);
        uhd::FeatureExtractionContext kernelCtx;
        kernelCtx.bindKernelVars(vars);

        Derived derived;
        derived.facts.kernelOnly = _extractor->kernelOnlyValues(kernelCtx);
        const auto& names = _extractor->mixedKernelVariables();
        derived.facts.slots.reserve(names.size());
        for(const auto& name : names)
        {
            const auto found = vars.find(name.substr(KERNEL_PREFIX.size()));
            derived.facts.slots.push_back(found == vars.end() ? std::nullopt : Slot(found->second));
        }
        for(size_t j = 0; j < _extractor->mixedCount(); ++j)
        {
            // The entry number first, so two entries never share a key.
            std::string key = std::to_string(j) + ':';
            for(const auto slot : _extractor->mixedEntrySlots(j))
            {
                appendKeyPart(key, derived.facts.slots[slot]);
            }
            derived.keys.push_back(std::move(key));
        }
        return derived;
    }

    /// Makes ctx hold exactly @p fact's values for the mixed entries' names; a name the
    /// kernel lacks must read as unbound (null), as when the whole namespace was cleared
    /// between candidates.
    static void bindKernel(uhd::FeatureExtractionContext& ctx,
                           const std::vector<std::string>& names,
                           const KernelFacts& fact,
                           std::vector<uhd::VariableContext::ValueType*>& bound)
    {
        for(size_t slot = 0; slot < names.size(); ++slot)
        {
            const auto& value = fact.slots[slot];
            if(!value)
            {
                if(bound[slot] != nullptr)
                {
                    ctx.unbindSlot(names[slot]);
                    bound[slot] = nullptr;
                }
            }
            else if(bound[slot] != nullptr)
            {
                *bound[slot] = *value;
            }
            else
            {
                bound[slot] = ctx.bindSlot(names[slot], *value);
            }
        }
    }

    struct CatalogFacts
    {
        std::vector<const KernelFacts*> facts;
        /// Key ids are below this.
        size_t keyCount = 0;
    };

    /// The facts of every kernel, in order. Pointers stay valid for the builder's lifetime:
    /// nothing is ever evicted and unordered_map elements do not move on rehash.
    CatalogFacts factsFor(const std::vector<const KernelDefinition*>& kernels) const
    {
        CatalogFacts result;
        result.facts.assign(kernels.size(), nullptr);
        bool missing = false;
        {
            const std::shared_lock<std::shared_mutex> lock(_mutex);
            for(size_t i = 0; i < kernels.size(); ++i)
            {
                const auto found = _facts.find(keyOf(*kernels[i]));
                if(found == _facts.end())
                {
                    missing = true;
                }
                else
                {
                    result.facts[i] = &found->second;
                }
            }
            if(!missing)
            {
                result.keyCount = _keyIds.size();
                return result;
            }
        }
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            if(result.facts[i] == nullptr)
            {
                // Derived outside the lock; when two threads race on one kernel the first
                // insertion wins and both read the same values.
                auto derived = deriveFacts(*kernels[i]);
                const std::unique_lock<std::shared_mutex> lock(_mutex);
                const auto [entry, inserted] = _facts.try_emplace(keyOf(*kernels[i]));
                if(inserted)
                {
                    entry->second = std::move(derived.facts);
                    for(auto& key : derived.keys)
                    {
                        const auto id = static_cast<uint32_t>(_keyIds.size());
                        entry->second.mixedKeys.push_back(
                            _keyIds.try_emplace(std::move(key), id).first->second);
                    }
                }
                result.facts[i] = &entry->second;
            }
        }
        const std::shared_lock<std::shared_mutex> lock(_mutex);
        result.keyCount = _keyIds.size();
        return result;
    }

    std::shared_ptr<const uhd::FeatureExtractor> _extractor;

    /// Keyed by (KernelDefinition::kernelId, the definition's address), so a cached value is
    /// reused only for the very definition it was derived from. Neither half is enough alone:
    ///  - An id is not unique to one definition. KernelIngestorStateManager.hpp:875-883
    ///    refuses a repeated id only where the two declarations' arches overlap, so one id can
    ///    name differently-parameterized definitions (metadata, priority) on disjoint arches,
    ///    and one heuristic serves several arches (an arch-prefix model entry covers more than
    ///    one device).
    ///  - An address alone could be recycled by a different definition.
    /// What makes the pair exact is the lifetime of definitions, which this relies on:
    ///  - Every definition a production heuristic ranks lives in its manager's `_definitions`
    ///    (KernelIngestorStateManager.hpp:1578-1581, :1175-1176 - "never mutated after"
    ///    construction, catalogs point into it) and is never moved, because the manager is
    ///    pinned (:248-252) and KernelDefinition.hpp:20-22 gives pointers that lifetime.
    ///  - The heuristic is built for exactly one manager: MakeEngine.hpp:294 creates it and
    ///    :300-305 moves it into the manager, the only non-test construction of either. It cannot
    ///    outlive its use of those definitions, so while it exists no address is freed and
    ///    reused, and an (id, address) pair always names the same definition content.
    /// A caller that ranks catalogs over definitions it frees and re-creates at the same
    /// address with the same id but other metadata, using one heuristic throughout, is outside
    /// that contract and would read stale values.
    /// Lives as long as the owning heuristic, so a model reload (a new heuristic, a new
    /// signature) starts empty. Never evicted: one small entry per definition the engine owns,
    /// which the engine's kernel count bounds.
    struct FactsKey
    {
        DescriptorId id;
        const KernelDefinition* definition;

        bool operator==(const FactsKey& other) const
        {
            return definition == other.definition && id == other.id;
        }
    };

    struct FactsKeyHash
    {
        size_t operator()(const FactsKey& key) const noexcept
        {
            // Boost-style combine of the id hash and the address.
            const size_t address = std::hash<const KernelDefinition*>{}(key.definition);
            return DescriptorIdHash{}(key.id)
                   ^ (address + 0x9e3779b97f4a7c15ULL + (DescriptorIdHash{}(key.id) << 6)
                      + (DescriptorIdHash{}(key.id) >> 2));
        }
    };

    static FactsKey keyOf(const KernelDefinition& kernel)
    {
        return {kernel.kernelId, &kernel};
    }

    mutable std::shared_mutex _mutex;
    mutable std::unordered_map<FactsKey, KernelFacts, FactsKeyHash> _facts;
    /// Numbers the distinct (mixed entry, kernel values) keys; at most one per entry per
    /// kernel, so bounded like _facts. Guarded by _mutex.
    mutable std::unordered_map<std::string, uint32_t> _keyIds;
};

} // namespace hipdnn_plugin_sdk::ingestor::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
