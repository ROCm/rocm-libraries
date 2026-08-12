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

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

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

/// A descriptor cross-reference: the stable, globally unique id every descriptor
/// carries, and the only way descriptors name one another.
///
/// A 128-bit UUID, matching the graph identity hipDNN mints at finalization, so the
/// two id spaces in this system are one type (RFC 0017 §4). `parseUuid` and
/// `formatUuid` in the same header convert to and from the text form descriptor files
/// carry.
using DescriptorId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// @brief A descriptor id in its canonical text form, for diagnostics.
inline std::string toString(const DescriptorId& id)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(id);
}

/// Hash for keying maps on a descriptor id. std::array has no std::hash, so this is
/// passed explicitly to the containers that need it.
struct DescriptorIdHash
{
    size_t operator()(const DescriptorId& id) const noexcept
    {
        // The id is a UUID, already well distributed, so an FNV-1a fold over its bytes
        // is sufficient for an in-process map and is never persisted or exposed.
        size_t hash = 1469598103934665603ULL;
        for(const uint8_t byte : id)
        {
            hash ^= static_cast<size_t>(byte);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

/// A value a kernel supplies for a KMD-declared field, and also the value type
/// `$graph.*` binds (see MatchContext.hpp's BoundTokens): both sides of a criteria
/// comparison need one shared value type.
///
/// `int64_t` and `double` are the widest signed integer and floating-point forms, so a
/// narrower authored value converts into one without loss; `bool` is distinct because a
/// flag compares and prints differently from the integer 1. `std::vector<int64_t>` is
/// for list-valued facts like `stride_order`.
using MetadataValue = std::variant<bool, int64_t, double, std::string, std::vector<int64_t>>;

/// The type a KMD field holds, or a bound graph fact's type. Ordered to match
/// MetadataValue's alternatives, so a field can declare its type without also having to
/// supply a value of it.
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

/// A kernel's complete metadata tuple: every KMD field name mapped to this kernel's
/// value for it. This tuple is the kernel's identity to the catalog (RFC 0017 §4), so
/// it must be unique across every kernel in an engine. Ordered, so two kernels with the
/// same fields compare and hash identically regardless of insertion order.
using MetadataValues = std::map<std::string, MetadataValue>;

/// One field a kernel may vary along, as declared by an engine's KMD.
struct MetadataField
{
    std::string name;
    /// What this field holds.
    MetadataType type = MetadataType::INT;
    /// The value a kernel that omits this field is taken to have supplied. Optional,
    /// because a field can be mandatory; a kernel omitting a mandatory field is a load
    /// error rather than a silent fallback.
    std::optional<MetadataValue> defaultValue;
};

/// KMD: the metadata schema, one per engine, shared by every kernel the engine owns.
///
/// The field set is the engine's kernel key: the heuristic ranks on these values and
/// matchers read them as `$kernel.<field>`, so any quantity that distinguishes two
/// kernels must appear here or the two collide.
struct MetadataSchema
{
    DescriptorId id;
    std::string name;
    std::vector<MetadataField> fields;
};

/// Which adapter builds an engine's IKernelHeuristic from a UHD's `payload` (RFC 0017
/// §9.1's adapter dispatch point).
enum class HeuristicKind
{
    /// `payload` is a NativeRegistry score symbol, resolved into a NativeKernelHeuristic
    /// (see IKernelHeuristic.hpp's makeKernelHeuristic()). The only kind with an adapter
    /// today.
    NATIVE,
    /// A trained model artifact plus its feature signature (the UHD follow-up RFC). No
    /// adapter yet.
    MODEL,
};

/// UHD: the kernel-selection model for one engine.
///
/// One level below hipDNN's engine-selection heuristic: this chooses *which kernel
/// within an engine* to run, given the kernels that fit the graph.
struct HeuristicDescriptor
{
    DescriptorId id;
    std::string name;
    HeuristicKind kind = HeuristicKind::NATIVE;
    /// `kind`'s payload: a NativeRegistry score symbol (kind == NATIVE), or, for a kind
    /// with no adapter yet, that kind's analogous single identifier (a model artifact
    /// path).
    std::string payload;
};

/// UED: the engine itself, carrying no logic of its own.
///
/// Names its one heuristic and one metadata schema by id, because a single selector
/// ranks all of the engine's kernels over one feature space. `name` hashes into
/// hipDNN's engine-id space (see EngineNames.hpp), so it must be globally unique and
/// should be scoped, e.g. "rocke:SDPA".
struct EngineDescriptor
{
    DescriptorId id;
    std::string name;
    DescriptorId heuristicId;
    DescriptorId metadataSchemaId;
    /// KMD field names this engine exposes for the caller to control. A name no KMD
    /// field matches is a load error.
    std::vector<std::string> knobs;
    /// Advisory execution behavior this engine's kernels exhibit, as
    /// `hipdnnBackendBehaviorNote_t` values (BehaviorNote.h), reported through
    /// EngineDetails like any other engine's.
    ///
    /// Held as int32 rather than a typed enum because the transport is int32, so a
    /// newer descriptor can carry a note value an older backend does not know without
    /// truncating it.
    std::vector<int32_t> behaviorNotes;
};

/// Which inputs a matcher reads, which decides how often it runs and what its failure
/// prunes (RFC 0017 §5).
///
/// Authored per matcher here, which is provisional: once the criteria language lands
/// this becomes a derived property, computed from whether a matcher's expression
/// references `$kernel.*`.
enum class MatchScope
{
    /// Reads only graph and device facts, so it runs once per (graph, device) and its
    /// failure disqualifies every kernel in every pack that lists it.
    GRAPH,
    /// Also reads `$kernel.*`, so it is re-evaluated per candidate kernel and
    /// disqualifies that kernel alone.
    KERNEL,
};

/// UMD: one applicability check, shared by id across packs.
struct MatchDescriptor
{
    DescriptorId id;
    std::string name;
    MatchScope scope;
    /// Resolved through NativeRegistry. The data-driven form (a structural node pattern
    /// plus a declarative criteria expression) is the UMD follow-up RFC.
    std::string matchSymbol;
};

/// UDD: how to invoke a kernel — the dispatch ABI, shared by every kernel in a pack.
///
/// A kernel whose argument list or launch-formula shape differs belongs in another pack
/// with its own UDD (RFC 0017 §6).
struct DispatchDescriptor
{
    DescriptorId id;
    std::string name;
    /// Resolved through NativeRegistry to a dispatch handler supplying both the
    /// workspace requirement and the launch. The data-driven form (symbolic grid, block,
    /// shared memory, workspace, and argument signature) is the UDD follow-up RFC.
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

/// UKD's source: where a kernel's code comes from, as a tagged union over RFC 0017
/// §7's source kinds. Only `EMBEDDED_SOURCE` is implemented; the others exist as enum
/// values, since a descriptor states which kind it carries independently of whether
/// this provider can load that kind yet.
struct KernelSource
{
    KernelSourceKind kind = KernelSourceKind::EMBEDDED_SOURCE;
    /// `EMBEDDED_SOURCE`: the source file name. Unused, and left empty, by every other
    /// kind until each grows its own payload shape.
    std::string sourceFile;
    /// `EMBEDDED_SOURCE`: the entry point within `sourceFile`. Unused by every other
    /// kind, for the same reason as `sourceFile` above.
    std::string entryPoint;
};

/// UKD: one launchable kernel — a source plus concrete values for the fields its
/// engine's KMD declares. Its matchers, engine, and dispatch are all its pack's, and
/// its heuristic and metadata schema are that engine's, so it names none of them.
struct KernelDescriptor
{
    DescriptorId id;
    std::string name;
    /// Where this kernel's code comes from and how to load it. See KernelSource for why
    /// this is a tagged union rather than two bare strings.
    KernelSource source;
    /// This kernel's values for the KMD's fields. Omitted fields take the KMD default;
    /// the completed tuple is this kernel's catalog key and must be unique engine-wide.
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
