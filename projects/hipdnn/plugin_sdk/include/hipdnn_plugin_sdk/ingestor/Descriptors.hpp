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
 * Each descriptor kind maps 1:1 onto a concept hipDNN already has, expressed as data
 * rather than hand-written C++:
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
 * These types are the *parsed* form. Nothing here parses, loads, or validates a file:
 * an ingestor built on this header is handed already-constructed descriptors. The
 * loader that produces them from disk is ALMIOPEN-2401 and its follow-ups; the
 * expression language that replaces the native symbol fields below is the UMD/UDD
 * follow-up RFC.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// A descriptor cross-reference: the stable, globally unique id every descriptor
/// carries, and the only way descriptors name one another.
///
/// A 128-bit UUID rather than a string, matching the graph identity hipDNN mints at
/// finalization, so the two id spaces in this system are one type. RFC 0017 §4 chooses
/// GUIDs so any author can mint an id locally that never collides with another's, with
/// no central allocation authority; `parseUuid` and `formatUuid` in the same header
/// convert to and from the text form descriptor files carry.
using DescriptorId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// @brief A descriptor id in its canonical text form, for diagnostics.
inline std::string toString(const DescriptorId& id)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(id);
}

/// Hash for keying maps on a descriptor id. std::array has no std::hash, and a
/// specialization for it cannot be added without opening namespace std over a standard
/// type, so this is passed explicitly to the containers that need it.
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

/// A value a kernel supplies for a KMD-declared field.
///
/// Spans the primitive types a descriptor field can hold. `int64_t` and `double` are the
/// widest signed integer and floating-point forms, so a narrower authored value converts
/// into one without loss; `bool` is distinct because a flag compares and prints
/// differently from the integer 1.
using MetadataValue = std::variant<bool, int64_t, double, std::string>;

/// The type a KMD field holds. Kept as an explicit enum, and ordered to match
/// MetadataValue's alternatives, so a field can declare its type without also having to
/// supply a value of it.
enum class MetadataType
{
    BOOL,
    INT,
    FLOAT,
    STRING,
};

/// @brief The type of a value, for checking one against its field's declaration.
inline MetadataType metadataTypeOf(const MetadataValue& value)
{
    return static_cast<MetadataType>(value.index());
}

/// A kernel's complete metadata tuple: every KMD field name mapped to this kernel's
/// value for it. RFC 0017 §4 makes this tuple the kernel's identity to the catalog, so
/// it must be unique across every kernel in an engine. Ordered, so two kernels with the
/// same fields compare and hash identically regardless of insertion order.
using MetadataValues = std::map<std::string, MetadataValue>;

/// One field a kernel may vary along, as declared by an engine's KMD.
struct MetadataField
{
    std::string name;
    /// What this field holds. Declared separately from the default because a field need
    /// not have one, and a kernel supplying the wrong alternative is a load error the
    /// validating loader reports against this.
    MetadataType type = MetadataType::INT;
    /// The value a kernel that omits this field is taken to have supplied.
    ///
    /// Optional, because a field can be mandatory. RFC 0017 §4's KMD example carries
    /// `optional` and `default` as separate attributes on the field; those collapse here,
    /// since a field that is mandatory has no default to fall back on. A kernel omitting
    /// such a field is a load error rather than a silent fallback to a catalog key its
    /// author never wrote.
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

/// UHD: the kernel-selection model for one engine.
///
/// One level below hipDNN's engine-selection heuristic: this chooses *which kernel
/// within an engine* to run, given the kernels that fit the graph.
struct HeuristicDescriptor
{
    DescriptorId id;
    std::string name;
    /// Resolved through NativeRegistry to a per-kernel scorer. The data-driven form
    /// (a model artifact plus its feature signature) is the UHD follow-up RFC.
    std::string scoreSymbol;
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
};

/// Which inputs a matcher reads, which decides how often it runs and what its failure
/// prunes (RFC 0017 §5, "applicability is a cheap, shared-matcher pass").
///
/// Authored per matcher here, which is provisional. RFC 0017 §5 puts the split inside a
/// single matcher's criteria tree rather than across matchers: `conv.tile_fit` mixes
/// graph-bound dims and `$kernel.*` in one `and`, and the `$kernel.*` clauses are the
/// ones re-evaluated per candidate. Once the criteria language lands this becomes a
/// derived property, computed from whether a matcher's expression references `$kernel.*`,
/// and a matcher with mixed clauses is evaluated per clause rather than classified whole.
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

/// UKD: one launchable kernel — a source plus concrete values for the fields its
/// engine's KMD declares. Its matchers, engine, and dispatch are all its pack's, and
/// its heuristic and metadata schema are that engine's, so it names none of them.
struct KernelDescriptor
{
    DescriptorId id;
    std::string name;
    /// Where this kernel's code comes from.
    ///
    /// A named source file plus an entry point, which is what the embedded-source path
    /// this engine currently compiles through needs. The real form is a tagged union
    /// over the source kinds of RFC 0017 §7, one alternative per supported kind, so a
    /// descriptor states which kind it carries rather than every kind sharing two
    /// strings.
    ///
    /// Shipped AOT kernels converge on one of those kinds: they are always packed into
    /// a kpack and loaded dynamically from it, so the field these two become is a kpack
    /// library plus the symbol to resolve inside it. The other kinds (`hsaco`, `hip`,
    /// `rocke`) describe how a kernel is *built*, and the build-time packaging step
    /// lowers each of them to that same kpack form before install.
    std::string sourceFile;
    std::string entryPoint;
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
