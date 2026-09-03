// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

/// @file Descriptors.hpp
/// @brief The universal descriptor set, as parsed in-memory data: KMD (kernel
/// metadata fields), UHD (ranking model), UED (engine identity), UMD (applicability
/// check), UDD (invocation ABI), UKD (one kernel), KDP (pack binding the above over N
/// kernels). Descriptors reference each other by `id` only. This is the parsed form;
/// nothing here parses, loads, or validates a file.
namespace hipdnn_plugin_sdk::ingestor
{

/// Stable, globally unique id every descriptor carries and is referenced by.
using DescriptorId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

inline std::string toString(const DescriptorId& id)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(id);
}

/// How a descriptor is named in a diagnostic: its kind, name, and id.
inline std::string
    describeDescriptor(std::string_view kind, const std::string& name, const DescriptorId& id)
{
    return std::string(kind) + " '" + name + "' (" + toString(id) + ")";
}

/// Hash for keying maps on a descriptor id (std::array has no std::hash).
struct DescriptorIdHash
{
    size_t operator()(const DescriptorId& id) const noexcept
    {
        // FNV-1a fold over the UUID bytes.
        size_t hash = 1469598103934665603ULL;
        for(const uint8_t byte : id)
        {
            hash ^= static_cast<size_t>(byte);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

/// Value type for a KMD field or a bound `$graph.*` fact.
using MetadataValue = std::variant<bool, int64_t, double, std::string, std::vector<int64_t>>;

/// The type a KMD field or bound graph fact holds.
enum class MetadataType
{
    BOOL,
    INT,
    FLOAT,
    STRING,
    INT_LIST,
};

inline MetadataType metadataTypeOf(const MetadataValue& value)
{
    return static_cast<MetadataType>(value.index());
}

// MetadataType's order must match MetadataValue's alternative order; kept in sync by
// the asserts below.
static_assert(std::variant_size_v<MetadataValue> == 5,
              "MetadataValue gained or lost an alternative; add the matching MetadataType "
              "enumerator and extend the assertions below.");
static_assert(std::is_same_v<std::variant_alternative_t<static_cast<size_t>(MetadataType::BOOL),
                                                        MetadataValue>,
                             bool>,
              "MetadataType::BOOL no longer indexes MetadataValue's bool alternative.");
static_assert(std::is_same_v<
                  std::variant_alternative_t<static_cast<size_t>(MetadataType::INT), MetadataValue>,
                  int64_t>,
              "MetadataType::INT no longer indexes MetadataValue's int64_t alternative.");
static_assert(std::is_same_v<std::variant_alternative_t<static_cast<size_t>(MetadataType::FLOAT),
                                                        MetadataValue>,
                             double>,
              "MetadataType::FLOAT no longer indexes MetadataValue's double alternative.");
static_assert(std::is_same_v<std::variant_alternative_t<static_cast<size_t>(MetadataType::STRING),
                                                        MetadataValue>,
                             std::string>,
              "MetadataType::STRING no longer indexes MetadataValue's std::string alternative.");
static_assert(
    std::is_same_v<
        std::variant_alternative_t<static_cast<size_t>(MetadataType::INT_LIST), MetadataValue>,
        std::vector<int64_t>>,
    "MetadataType::INT_LIST no longer indexes MetadataValue's vector<int64_t> alternative.");

/// A kernel's complete metadata tuple; must be unique per kernel within an engine.
using MetadataValues = std::map<std::string, MetadataValue>;

/// One field a kernel may vary along, as declared by an engine's KMD.
struct MetadataField
{
    std::string name;
    MetadataType type = MetadataType::INT;
    /// nullopt means the field is mandatory.
    std::optional<MetadataValue> defaultValue;
};

/// KMD: the metadata schema, one per engine. Field set is the kernel key; matchers
/// read fields as `$kernel.<field>`.
struct MetadataSchema
{
    DescriptorId id;
    std::string name;
    std::vector<MetadataField> fields;
};

/// How a UHD ranks a catalog. RFC 0019 §4.2: one discriminant, which also selects the
/// adapter-scoped body's key.
enum class UhdAdapter
{
    STATIC_ORDER, ///< No model. `priority`, then descriptor id.
    NATIVE, ///< A scorer compiled into the engine, resolved by symbol.
    TREE_DATA, ///< GBDT tree table shipped as a data artifact. The default (§7.2).
    TABLE, ///< Bucketed lookup table shipped as a data artifact.
};

/// Units and calibration of a UHD's score, for cross-engine comparison (RFC 0019 §11.3).
struct UhdScore
{
    std::string units; ///< e.g. "tflops", "ms".
    bool calibrated = false; ///< True iff comparable across engines.
    std::string transform; ///< Applied to raw model output: "identity", "log1p".
};

/// One `$derived.*` entry: a name and the JsonLogic expression producing it
/// (RFC 0019 §6.4). Evaluated in declaration order; a later entry may reference an
/// earlier one.
struct UhdDerivedValue
{
    std::string name;
    std::string expression;
};

/// UHD: the kernel-selection model for one engine.
///
/// The whole descriptor, not a pointer to one. An earlier design put these fields in a
/// FlatBuffer that a four-field JSON stub named, which made the UHD the only descriptor
/// in the family a human could not read, diff or hand-write -- for 134 bytes, on a file
/// read once per engine. RFC 0019 §4 always specified JSON; the binary is reserved for
/// the model artifact, which earns it by being read once per candidate score.
struct HeuristicDescriptor
{
    DescriptorId id;
    std::string name;
    UhdAdapter adapter = UhdAdapter::STATIC_ORDER;

    /// Ordered model inputs, each a JsonLogic expression over `$device.*`, `$kernel.*`,
    /// `$q.*` and `$derived.*`. Order is part of the contract: it is the order the model
    /// was trained on. Empty for static_order, which consumes no features.
    std::vector<std::string> featuresSignature;
    /// Guards @ref featuresSignature against the model that was trained on it. The
    /// extractor recomputes it and refuses to load on a mismatch (RFC 0019 §6.3).
    std::string featuresHash;
    /// Evaluated before the signature, forming the `$derived.*` namespace.
    std::vector<UhdDerivedValue> derived;

    /// "max" or "min". A model trained on a cost rather than a rate ranks ascending, and
    /// getting this wrong silently inverts every ranking it produces.
    std::string objective = "max";
    UhdScore score;

    /// NATIVE: the symbol the engine registered its scorer under.
    std::string nativeSymbol;
    /// TREE_DATA / TABLE: the artifact path, relative to @ref baseDir.
    std::string modelArtifactPath;
    /// STATIC_ORDER: ordering criteria, e.g. {"priority", "id"}.
    std::vector<std::string> staticOrderFields;

    /// Directory of the `.uhd.json` that declared this descriptor. @ref modelArtifactPath
    /// resolves against it, so a descriptor set relocates as a unit.
    ///
    /// Empty for descriptors built in memory rather than parsed from disk.
    std::filesystem::path baseDir;
    /// The descriptor tree @ref baseDir was found under -- the root the loader was
    /// pointed at, not the file's own folder.
    ///
    /// Same split as KernelDescriptor's originDirectory/treeRoot pair, and for the same
    /// reason: resolution and CONTAINMENT are different questions. An artifact resolves
    /// against baseDir, but the boundary it may not cross is the tree, so a descriptor
    /// nested in an arch shard can legitimately name `../shared/model.bin` while nothing
    /// may climb out of the tree entirely. The artifact is author-controlled input
    /// (RFC 0019 §16, "Drop-in trust").
    ///
    /// Empty for descriptors built in memory. Filled by the loader, never authored.
    std::filesystem::path treeRoot;

    /// RFC 0019 §8.1: the descriptor versions this heuristic was generated against, keyed
    /// by kind -- "ued", "umd", "kmd". Empty when the UHD declares none, which §4's field
    /// table permits for an adapter carrying no features.
    ///
    /// The coupling exists because a model reads its inputs *through* those descriptors: a
    /// KMD that gains a field, or a UED whose knob list moves, changes what `$kernel.*`
    /// resolves to without changing the model. Nothing else detects it -- features_hash
    /// covers the signature, not the descriptors the signature resolves against -- so a UHD
    /// regenerated out of step otherwise loads clean and ranks on stale meaning.
    std::map<std::string, hipdnn_data_sdk::utilities::Version> trainedAgainst;
};

/// UED: the engine itself, carrying no logic of its own. `name` hashes into hipDNN's
/// engine-id space; must be globally unique, e.g. "rocke:SDPA".
struct EngineDescriptor
{
    DescriptorId id;
    std::string name;
    /// The engine's catalog-ranking UHD, resolved for the running architecture.
    ///
    /// nullopt when the engine ships no UHD; selection then falls back to the
    /// descriptor-declared order. Must equal `DescriptorSet::heuristic`'s id when set.
    /// This is the `default` entry of @ref sortKernelCatalog, or the legacy `heuristic` key.
    std::optional<DescriptorId> heuristicId;

    /// RFC 0019 §3.1: an engine names up to three role-scoped UHDs, each mapped by
    /// architecture, and each independently optional.
    ///
    ///   sort_kernel_catalog        ranks the catalog and picks the kernel
    ///   predict_engine_tflops      cheap f(graph) -> expected perf, for engine selection
    ///   predict_applicable_kernels generates the candidate set (future, JIT case)
    ///
    /// Keyed by `gcnArchName` with a `default` fallback, matching how the backend's
    /// EngineRegistry stores them. A bare id in the UED is read as a `default`-only map, so
    /// the legacy single-reference form is one of these with one entry rather than a
    /// separate concept.
    ///
    /// RFC 0020 §4.4 still describes `heuristic` as an engine's *one* UHD id; that sentence
    /// predates the multiple-reference model and the two RFCs disagree. This follows
    /// RFC 0019 while continuing to load the older form.
    std::map<std::string, DescriptorId> sortKernelCatalog;
    std::map<std::string, DescriptorId> predictEngineTflops;
    std::map<std::string, DescriptorId> predictApplicableKernels;
    DescriptorId metadataSchemaId;
    std::vector<std::string> knobs;
    /// `hipdnnBackendBehaviorNote_t` values; int32 so a newer note isn't truncated.
    std::vector<int32_t> behaviorNotes;
    /// Graph schema version this engine understands; a graph below this floor is
    /// declined rather than matched with an ignored field. Baseline by default.
    hipdnn_data_sdk::utilities::Version sdkVersion{K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    /// RFC 0020 §4.2 numerical notes, held as authored; no hipDNN enum exists for them
    /// yet, so nothing consumes the strings.
    std::vector<std::string> numericalNotes;
    /// Resolved through GraphMatchRegistry; empty means this engine declares no
    /// graph-topology match.
    std::string graphMatchNativeSymbol;
};

/// Which inputs a matcher reads, and so what its failure prunes.
enum class MatchScope
{
    GRAPH, ///< Once per (graph, device); failure disqualifies every kernel in the pack.
    KERNEL, ///< Reads `$kernel.*` too; disqualifies only the one kernel.
};

/// UMD: one applicability check, shared by id across packs.
struct MatchDescriptor
{
    DescriptorId id;
    std::string name;
    MatchScope scope = MatchScope::GRAPH;
    std::string matchSymbol; ///< Resolved through NativeRegistry.
};

/// UDD: how to invoke a kernel, shared by every kernel in a pack.
struct DispatchDescriptor
{
    DescriptorId id;
    std::string name;
    std::string dispatchSymbol; ///< Resolved through NativeRegistry.
};

/// Which adapter loads and prepares one kernel's code.
enum class KernelSourceKind
{
    EMBEDDED_SOURCE, ///< Source file plus entry point, compiled at plan-build time.
    KPACK, ///< Prebuilt kpack archive plus toc key and symbol.
    HSACO_FILE, ///< Standalone `.hsaco` code-object file. No adapter yet.
    ROCKE_BUILDER, ///< rocke builder name plus build values. No adapter yet.
};

/// UKD's source. `EMBEDDED_SOURCE` and `KPACK` are implemented; a kind fills only its own
/// fields and leaves the rest empty.
struct KernelSource
{
    KernelSourceKind kind = KernelSourceKind::EMBEDDED_SOURCE;
    std::string sourceFile; ///< EMBEDDED_SOURCE.
    std::string entryPoint; ///< EMBEDDED_SOURCE.
    /// KPACK: archive path, relative to the directory of the descriptor that declared it.
    /// Relative because the installed tree is relocatable and an absolute build-machine
    /// path would not survive packaging.
    std::string library;
    /// KPACK: the archive's own key for this code object. Opaque -- the hip packager
    /// content-addresses it on (source, build) and the rocKE producer uses a different
    /// scheme entirely, so nothing here may parse it. Not unique per kernel: two kernels
    /// differing only by entry point share one key, one blob, and one loaded module.
    std::string tocKey;
    /// KPACK: the undecorated extern "C" name to resolve inside the loaded module. The
    /// only field that separates two kernels sharing a tocKey.
    std::string symbol;
    /// KPACK: digest of the raw decompressed code object, as the packager recorded it.
    ///
    /// Identification only. Nothing verifies it, on this path or any other, and it is
    /// not a security control: a descriptor and the archive it names travel together, so
    /// whoever can rewrite one can rewrite the other. Treat a match as evidence the two
    /// came from the same pack run, nothing more. The loader's defence against a wrong
    /// or corrupt payload is KpackArchive's container check, not this field.
    std::string sha256;
};

namespace detail
{

/// True iff `T{Args...}` is a valid brace initialization. `std::is_constructible` cannot
/// answer this under C++17: it probes parenthesized direct-initialization, which does not
/// perform aggregate initialization until C++20.
template <typename T, typename = void, typename... Args>
struct IsBraceInitializable : std::false_type
{
};

template <typename T, typename... Args>
struct IsBraceInitializable<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...>
    : std::true_type
{
};

template <typename T, typename... Args>
inline constexpr bool IS_BRACE_INITIALIZABLE_V = IsBraceInitializable<T, void, Args...>::value;

} // namespace detail

// KernelSource's field count is pinned here: accepting exactly seven initializers and no
// more makes an inserted field ill-formed at this assertion, rather than silently
// rebinding every value after it at a positional initialization site. Only the count --
// two same-typed members swapped past each other still brace-initialize.
static_assert(detail::IS_BRACE_INITIALIZABLE_V<KernelSource,
                                               KernelSourceKind,
                                               std::string,
                                               std::string,
                                               std::string,
                                               std::string,
                                               std::string,
                                               std::string>
                  && !detail::IS_BRACE_INITIALIZABLE_V<KernelSource,
                                                       KernelSourceKind,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string>,
              "KernelSource gained or lost a field; append only, then extend this "
              "assertion.");

/// UKD: one launchable kernel. Matchers, engine, and dispatch come from its pack.
struct KernelDescriptor
{
    DescriptorId id;
    std::string name;
    KernelSource source;
    /// Omitted fields take the KMD default; completed tuple is the catalog key.
    MetadataValues metadata;
    int64_t priority = 0; ///< Tie-break when the heuristic is not decisive.
    /// GFX base targets this kernel runs on; empty inherits the pack's list. A kernel may
    /// narrow to part of what its pack claims, never reach outside it, and the resolved
    /// list reaches dispatch as KernelDefinition::arch. Authored on either form: inline in
    /// a KDP, or in a standalone `.ukd.json`.
    ///
    /// For a standalone kernel it is also half the catalog key. A per-arch shard ships the
    /// same kernel id in every shard, differing only in which code object it names, so the
    /// id alone cannot say which copy a pack means. An inline kernel is not keyed at all --
    /// it lives inside a pack document that the pack's own (id, arch) key already separates
    /// per shard -- so there the field is applicability only.
    std::vector<std::string> arch;
    /// Directory of the descriptor file that defined this kernel. Any path a descriptor
    /// names is resolved against it, so a relocated, DESTDIR-staged or drop-in install
    /// resolves from where the file actually is rather than from a loader root. Empty for a
    /// kernel built in memory, which names no file. Filled by the loader, never authored.
    std::filesystem::path originDirectory;
    /// The descriptor tree @c originDirectory was found under -- the root the loader was
    /// pointed at, not the file's own folder.
    ///
    /// Carried because resolution and CONTAINMENT are different questions. A path is
    /// resolved against originDirectory (above), but the boundary it may not cross is the
    /// tree, not the individual descriptor's folder: one archive is shipped per arch shard
    /// at the shard root, so a descriptor nested inside that shard legitimately climbs out
    /// of its own directory to reach it. Anchoring containment on originDirectory rejected
    /// every nested descriptor and made production-packaged kernels unloadable.
    ///
    /// The tree root rather than the arch shard root: it is what the loader actually
    /// walked, so it needs no probing and no assumption about how deep a shard sits, and
    /// it stays correct for a flat tree where the two coincide. Empty for a kernel built
    /// in memory. Filled by the loader, never authored.
    std::filesystem::path treeRoot;
};

/// KDP: one pack binding a matcher set, one engine, and one dispatch descriptor over
/// a vector of child kernels.
struct KernelDescriptorPack
{
    DescriptorId id;
    std::string name;
    std::vector<DescriptorId> matcherIds;
    DescriptorId engineId;
    DescriptorId dispatchId;
    /// GFX targets, e.g. `{"gfx942", "gfx950"}`; empty means arch-independent, matched
    /// exactly against the device's base target id. Part of the pack's catalog identity
    /// -- packs are keyed by (id, arch), so per-arch shards may ship one pack id many
    /// times -- as well as a match-time filter. The filter itself is enforced at catalog
    /// build, not load time, so an excluded pack still builds and simply declines per
    /// call -- an expected decline like a matcher returning false, not a malformed load,
    /// so it must not be reported as one.
    std::vector<std::string> arch;
    /// Every kernel this pack binds, whether authored inline or referenced by id.
    std::vector<KernelDescriptor> kernels;
    /// Kernels named by id rather than spelled inline; each is a standalone `.ukd.json`.
    /// resolveDescriptorSets() looks them up and appends them to `kernels`, which is the
    /// resolved truth every consumer reads -- this stays as the authored record. Packs
    /// built in memory leave it empty and fill `kernels` directly.
    std::vector<DescriptorId> kernelIds;
};

/// One engine and every descriptor it references by id; self-contained.
struct DescriptorSet
{
    EngineDescriptor engine;
    MetadataSchema schema;
    /// The UHD for the `default` arch, or the only one when the UED named a bare id.
    /// nullopt when this engine ships no ranking model; the generic engine then ranks on
    /// `priority` then descriptor id. See makeKernelHeuristic().
    std::optional<HeuristicDescriptor> heuristic;

    /// RFC 0019 §3.1: the engine's catalog-ranking UHD per architecture, keyed as the UED
    /// wrote it, `default` included.
    ///
    /// Resolution cannot happen at load: descriptor discovery is a process-wide memoized
    /// static that runs before any device exists. §8.3's "exact gcnArchName, then default"
    /// therefore happens at first rank(), where the device is known -- which is also what
    /// §9.2 asks for, load-on-demand with a per-engine cache.
    std::map<std::string, HeuristicDescriptor> heuristicsByArch;
    std::vector<MatchDescriptor> matchers;
    std::vector<DispatchDescriptor> dispatches;
    std::vector<KernelDescriptorPack> packs;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
