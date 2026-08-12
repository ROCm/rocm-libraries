// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

/**
 * @file Descriptors.hpp
 * @brief The universal descriptor set from RFC 0017, as parsed in-memory data.
 *
 * Each descriptor kind maps 1:1 onto a concept hipDNN already has:
 *
 * | Descriptor | Describes                                                    |
 * |------------|--------------------------------------------------------------|
 * | KMD        | the variant fields every kernel in an engine carries          |
 * | UHD        | how to rank the kernels that fit a graph                      |
 * | UED        | the engine: its identity, its one UHD, its one KMD, its knobs |
 * | UMD        | one applicability check over the graph and kernel metadata    |
 * | UDD        | how to invoke a kernel: workspace and launch                  |
 * | UKD        | one launchable kernel: a source plus its KMD values           |
 * | KDP        | binds a matcher set, one engine, and one UDD over N kernels   |
 *
 * Descriptors reference each other by `id` only, never by pointer or index, so a pack
 * authored independently can name a matcher or an engine it does not own.
 *
 * These are the *parsed* form; nothing here parses, loads, or validates a file. The
 * loader that produces them from disk is ALMIOPEN-2401; the expression language that
 * replaces the native symbol fields below is the UMD/UDD follow-up RFC.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// Stable, globally unique id every descriptor carries; descriptors reference each
/// other only by this id. A 128-bit UUID, matching hipDNN's graph identity (RFC 0017 §4).
using DescriptorId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// @brief A descriptor id in its canonical text form, for diagnostics.
inline std::string toString(const DescriptorId& id)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(id);
}

/// Hash for keying maps on a descriptor id (std::array has no std::hash).
struct DescriptorIdHash
{
    size_t operator()(const DescriptorId& id) const noexcept
    {
        // FNV-1a fold over the UUID bytes; not persisted or exposed outside this process.
        size_t hash = 1469598103934665603ULL;
        for(const uint8_t byte : id)
        {
            hash ^= static_cast<size_t>(byte);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

/// Value type for a KMD field or a bound `$graph.*` fact (MatchContext.hpp's
/// BoundTokens). `bool` is distinct from `int64_t` so a flag doesn't compare as 1;
/// `std::vector<int64_t>` covers list-valued facts like `stride_order`.
using MetadataValue = std::variant<bool, int64_t, double, std::string, std::vector<int64_t>>;

/// The type a KMD field or bound graph fact holds. Ordered to match MetadataValue's
/// alternatives.
enum class MetadataType
{
    BOOL,
    INT,
    FLOAT,
    STRING,
    INT_LIST,
};

/// @brief The type of a value, for checking one against its field's declaration.
inline MetadataType metadataTypeOf(const MetadataValue& value)
{
    return static_cast<MetadataType>(value.index());
}

/// A kernel's complete metadata tuple: every KMD field mapped to this kernel's value.
/// Must be unique per kernel within an engine (RFC 0017 §4); ordered so equal tuples
/// compare equal regardless of insertion order.
using MetadataValues = std::map<std::string, MetadataValue>;

/// One field a kernel may vary along, as declared by an engine's KMD.
struct MetadataField
{
    std::string name;
    MetadataType type = MetadataType::INT;
    /// Value used when a kernel omits this field; nullopt means the field is mandatory.
    std::optional<MetadataValue> defaultValue;
};

/// KMD: the metadata schema, one per engine, shared by every kernel it owns. The
/// field set is the engine's kernel key; matchers read fields as `$kernel.<field>`.
struct MetadataSchema
{
    DescriptorId id;
    std::string name;
    std::vector<MetadataField> fields;
};

/// Which adapter builds an engine's IKernelHeuristic from a UHD's `payload` (RFC 0017
/// §9.1).
enum class HeuristicKind
{
    /// `payload` is a NativeRegistry score symbol (IKernelHeuristic.hpp's
    /// makeKernelHeuristic()). Only kind with an adapter today.
    NATIVE,
    /// Trained model artifact plus feature signature (UHD follow-up RFC). No adapter yet.
    MODEL,
};

/// UHD: the kernel-selection model for one engine — chooses which kernel within an
/// engine to run, given the kernels that fit the graph.
struct HeuristicDescriptor
{
    DescriptorId id;
    std::string name;
    HeuristicKind kind = HeuristicKind::NATIVE;
    /// `kind`'s payload: a NativeRegistry score symbol when kind == NATIVE.
    std::string payload;
};

/// UED: the engine itself, carrying no logic of its own. `name` hashes into hipDNN's
/// engine-id space (EngineNames.hpp); must be globally unique, e.g. "rocke:SDPA".
struct EngineDescriptor
{
    DescriptorId id;
    std::string name;
    DescriptorId heuristicId;
    DescriptorId metadataSchemaId;
    /// KMD field names this engine exposes to the caller. Unmatched name is a load error.
    std::vector<std::string> knobs;
    /// `hipdnnBackendBehaviorNote_t` values (BehaviorNote.h), reported via EngineDetails.
    /// Stored as int32 so an unrecognized newer note value isn't truncated.
    std::vector<int32_t> behaviorNotes;
    /**
     * @brief The hipDNN graph schema version this engine's descriptors understand
     *        (RFC 0017 §4's `sdk_version`).
     *
     * Graph-level matching and token binding belong to the engine: every descriptor
     * under it reads the tokens that binding produces, so they must all agree on the
     * schema those tokens were derived from. One version on the UED is that shared
     * agreement, and it lets a graph the engine cannot understand bail before any
     * pack, matcher, or kernel is looked at.
     *
     * A graph reports the version its own contents require via
     * `min_required_engine_api_version` (graph.fbs), computed from the optional
     * fields it sets. An engine declaring less than that floor declines the graph
     * rather than matching it: it would otherwise bind the fields it knows and
     * silently ignore one that changes what the graph means.
     *
     * Defaults to the baseline, which every graph's floor is at least equal to, so
     * an engine that sets nothing behaves exactly as before.
     *
     * This is a match-time gate, not a load-time one: the floor is a property of
     * each graph, not of the runtime, so it cannot be resolved when the descriptor
     * is read. The per-file-type `major.minor` of RFC 0017 §4's version table is a
     * separate, load-time check and belongs to the loader (ALMIOPEN-2401).
     */
    hipdnn_data_sdk::utilities::Version sdkVersion{K_ENGINE_PLUGIN_API_VERSION_BASELINE};
};

/// Which inputs a matcher reads; determines how often it runs and what its failure
/// prunes (RFC 0017 §5). Provisional until the criteria language lands, when this
/// becomes derived from whether the expression references `$kernel.*`.
enum class MatchScope
{
    /// Runs once per (graph, device); failure disqualifies every kernel in every pack
    /// that lists it.
    GRAPH,
    /// Also reads `$kernel.*`; re-evaluated per candidate kernel, disqualifying only
    /// that kernel.
    KERNEL,
};

/// UMD: one applicability check, shared by id across packs.
struct MatchDescriptor
{
    DescriptorId id;
    std::string name;
    MatchScope scope;
    /// Resolved through NativeRegistry; a data-driven form (structural node pattern plus
    /// declarative criteria) is the UMD follow-up RFC.
    std::string matchSymbol;
};

/// UDD: how to invoke a kernel — the dispatch ABI, shared by every kernel in a pack.
/// A kernel with a different argument list or launch-formula shape belongs in
/// another pack with its own UDD (RFC 0017 §6).
struct DispatchDescriptor
{
    DescriptorId id;
    std::string name;
    /// Resolved through NativeRegistry to a handler supplying workspace and launch. A
    /// data-driven form (symbolic grid/block/shared-memory/workspace/args) is the UDD
    /// follow-up RFC.
    std::string dispatchSymbol;
};

/// Which adapter loads and prepares one kernel's code, given its source's payload (RFC
/// 0017 §9.1's adapter dispatch point).
enum class KernelSourceKind
{
    /// A named source file plus an entry point, compiled at plan-build time. The only
    /// kind this POC implements.
    EMBEDDED_SOURCE,
    /// A prebuilt kpack library plus the symbol to resolve inside it. No adapter yet.
    KPACK_SYMBOL,
    /// A standalone `.hsaco` code-object file. No adapter yet.
    HSACO_FILE,
    /// A rocke builder name plus its build values. No adapter yet.
    ROCKE_BUILDER,
};

/// UKD's source: where a kernel's code comes from, a tagged union over RFC 0017 §7's
/// source kinds. Only `EMBEDDED_SOURCE` is implemented.
struct KernelSource
{
    KernelSourceKind kind = KernelSourceKind::EMBEDDED_SOURCE;
    /// `EMBEDDED_SOURCE`: source file name. Unused by other kinds.
    std::string sourceFile;
    /// `EMBEDDED_SOURCE`: entry point within `sourceFile`. Unused by other kinds.
    std::string entryPoint;
};

/// UKD: one launchable kernel — a source plus concrete values for its engine KMD's
/// fields. Matchers, engine, and dispatch are inherited from its pack.
struct KernelDescriptor
{
    DescriptorId id;
    std::string name;
    KernelSource source;
    /// This kernel's values for the KMD's fields; omitted fields take the KMD default.
    /// The completed tuple is this kernel's catalog key, unique engine-wide.
    MetadataValues metadata;
    /// Tie-break when the heuristic is not decisive. Higher wins.
    int64_t priority = 0;
};

/// KDP: one pack binding a matcher set, one engine, and one dispatch descriptor over a
/// vector of child kernels. Everything except the child kernels is shared by id.
struct KernelDescriptorPack
{
    DescriptorId id;
    std::string name;
    /// A child kernel applies only when every one of these matchers passes.
    std::vector<DescriptorId> matcherIds;
    DescriptorId engineId;
    DescriptorId dispatchId;
    std::vector<KernelDescriptor> kernels;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
