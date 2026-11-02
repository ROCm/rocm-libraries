// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>

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

/// The descriptor spelling of @p type, which `metadataTypeFromString` parses. A value
/// outside the enum gives "unknown (<value>)".
inline std::string toString(MetadataType type)
{
    switch(type)
    {
    case MetadataType::BOOL:
        return "bool";
    case MetadataType::INT:
        return "int";
    case MetadataType::FLOAT:
        return "float";
    case MetadataType::STRING:
        return "string";
    case MetadataType::INT_LIST:
        return "int_list";
    default:
        break;
    }
    return "unknown (" + std::to_string(static_cast<int>(type)) + ")";
}

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

/// A kernel's complete metadata tuple. Kernels of one engine sharing a tuple must not compete at
/// the same arch tier on any device; a better tier shadows a worse one.
using MetadataValues = std::map<std::string, MetadataValue>;

namespace detail
{

/// Equality for two metadata values that treats an integer and a floating-point alternative
/// holding the same number as equal, as JSON does: a kernel's metadata is typed by its KMD only
/// at resolve time, so `1` and `1.0` are one authored value at parse. A bool never equals a
/// number.
inline bool metadataValueEquivalent(const MetadataValue& lhs, const MetadataValue& rhs)
{
    if(lhs.index() == rhs.index())
    {
        return lhs == rhs;
    }
    if(const auto* integer = std::get_if<int64_t>(&lhs))
    {
        const auto* real = std::get_if<double>(&rhs);
        return real != nullptr && static_cast<double>(*integer) == *real;
    }
    if(const auto* real = std::get_if<double>(&lhs))
    {
        const auto* integer = std::get_if<int64_t>(&rhs);
        return integer != nullptr && static_cast<double>(*integer) == *real;
    }
    return false;
}

inline bool metadataValuesEquivalent(const MetadataValues& lhs, const MetadataValues& rhs)
{
    return lhs.size() == rhs.size()
           && std::equal(
               lhs.begin(), lhs.end(), rhs.begin(), [](const auto& left, const auto& right) {
                   return left.first == right.first
                          && metadataValueEquivalent(left.second, right.second);
               });
}

} // namespace detail

/// One field a kernel may vary along, as declared by an engine's KMD.
struct MetadataField
{
    std::string name;
    MetadataType type = MetadataType::INT;
    /// nullopt means the field is mandatory.
    std::optional<MetadataValue> defaultValue;
};

inline bool operator==(const MetadataField& lhs, const MetadataField& rhs)
{
    return std::tie(lhs.name, lhs.type, lhs.defaultValue)
           == std::tie(rhs.name, rhs.type, rhs.defaultValue);
}

inline bool operator!=(const MetadataField& lhs, const MetadataField& rhs)
{
    return !(lhs == rhs);
}

/// KMD: the metadata schema, one per engine. Field set is the kernel key; matchers
/// read fields as `$kernel.<field>`.
struct MetadataSchema
{
    DescriptorId id;
    std::string name;
    std::vector<MetadataField> fields;
    /// Authored semantics; independent of the JSON file-format version.
    hipdnn_data_sdk::utilities::Version revision{1, 0, 0};
};

inline bool operator==(const MetadataSchema& lhs, const MetadataSchema& rhs)
{
    return std::tie(lhs.id, lhs.name, lhs.fields, lhs.revision)
           == std::tie(rhs.id, rhs.name, rhs.fields, rhs.revision);
}

inline bool operator!=(const MetadataSchema& lhs, const MetadataSchema& rhs)
{
    return !(lhs == rhs);
}

/// How a UHD ranks a catalog. RFC 0019 §4.2: one discriminant, which also selects the
/// adapter-scoped body's key.
enum class UhdAdapter
{
    STATIC_ORDER, ///< No model. `priority`, then descriptor id.
    NATIVE, ///< A scorer compiled into the engine, resolved by symbol.
    TREE_DATA, ///< GBDT tree table shipped as a data artifact. The default (§7.2).
    TABLE, ///< Bucketed lookup table shipped as a data artifact.
    CUSTOM_LIBRARY, ///< An author-supplied shared library, loaded and called by symbol (§7.3).
};

/// What a UHD's score measures, and whether it is comparable across engines (RFC 0019
/// §4.4, §11.3).
struct UhdScore
{
    /// A registered ranking metric (RankingMetrics.hpp), which fixes the units and the
    /// direction; empty for a ranker that orders its catalog without predicting a comparable
    /// number (a `native` comparator, a `static_order`).
    std::string metric;
    bool calibrated = false; ///< True iff comparable across engines.
    std::string transform; ///< Applied to raw model output: "identity", "log1p".
};

inline bool operator==(const UhdScore& lhs, const UhdScore& rhs)
{
    return std::tie(lhs.metric, lhs.calibrated, lhs.transform)
           == std::tie(rhs.metric, rhs.calibrated, rhs.transform);
}

inline bool operator!=(const UhdScore& lhs, const UhdScore& rhs)
{
    return !(lhs == rhs);
}

/// A semantic dependency, independent of the descriptor file-format version.
struct DescriptorDependency
{
    DescriptorId id;
    hipdnn_data_sdk::utilities::Version revision{1, 0, 0};
};

inline bool operator==(const DescriptorDependency& lhs, const DescriptorDependency& rhs)
{
    return std::tie(lhs.id, lhs.revision) == std::tie(rhs.id, rhs.revision);
}

inline bool operator!=(const DescriptorDependency& lhs, const DescriptorDependency& rhs)
{
    return !(lhs == rhs);
}

/// The resolved descriptor definitions used to collect a model's training corpus.
struct HeuristicProvenance
{
    DescriptorDependency ued;
    DescriptorDependency kmd;
    std::vector<DescriptorDependency> umd;
};

inline bool operator==(const HeuristicProvenance& lhs, const HeuristicProvenance& rhs)
{
    return std::tie(lhs.ued, lhs.kmd, lhs.umd) == std::tie(rhs.ued, rhs.kmd, rhs.umd);
}

inline bool operator!=(const HeuristicProvenance& lhs, const HeuristicProvenance& rhs)
{
    return !(lhs == rhs);
}

/// UHD: the kernel-selection model for one engine.
///
/// `operator==` is authored-content equality: it ignores @ref modelArtifactPath, @ref baseDir,
/// @ref treeRoot, @ref engineName, @ref role and @ref arch, which the loader derives from the
/// file's location or fills after parsing, so the copies of one UHD shipped in per-arch shards
/// compare equal.
struct HeuristicDescriptor
{
    DescriptorId id;
    std::string name;
    UhdAdapter adapter = UhdAdapter::STATIC_ORDER;

    /// Ordered bare references or inline expression objects over published symbols,
    /// `$device.*` and `$kernel.*`. Order is the model's input order.
    std::vector<nlohmann::json> featuresSignature;
    /// Guards @ref featuresSignature against the model that was trained on it. The
    /// extractor recomputes it and refuses to load on a mismatch (RFC 0019 §6.3).
    std::string featuresHash;

    /// RFC 0019 §6.5 string-to-code map, folded into @ref featuresHash. Keyed by the whole
    /// `$`-reference, not the field name: `$kernel.dtype` ("BF16") and
    /// `$q.attention_dense.dtype` ("bf16") are separate vocabularies. Empty when the signature
    /// reads no string field.
    std::map<std::string, std::map<std::string, int32_t>> categoricalEncoding;

    /// "max" or "min". A model trained on a cost rather than a rate ranks ascending, and
    /// getting this wrong silently inverts every ranking it produces.
    std::string objective = "max";
    UhdScore score;

    /// NATIVE: the symbol the engine registered its scorer under.
    std::string nativeSymbol;
    /// TREE_DATA / TABLE / CUSTOM_LIBRARY: the artifact path the adapter opens. The loader
    /// stores it resolved and absolute; a descriptor built in memory may give it relative to
    /// @ref baseDir. Derived from the file's location; not part of operator==.
    std::string modelArtifactPath;
    /// The authored `artifact` (or `library`) value, relative to the declaring `.uhd.json`.
    /// Compared by operator== in place of @ref modelArtifactPath.
    std::string authoredArtifactPath;
    /// SHA-256 of the artifact (lowercase hex): the declared digest, else that of the bytes
    /// present at parse. Verifies the artifact at load; empty when no artifact exists yet.
    std::string modelHash;
    /// CUSTOM_LIBRARY: the scorer function's symbol name inside the `.so`.
    std::string customLibrarySymbol;

    /// Directory of the `.uhd.json` that declared this; @ref modelArtifactPath resolves
    /// against it. Empty for descriptors built in memory. Derived from the file's location; not
    /// part of operator==.
    std::filesystem::path baseDir;
    /// The tree root the loader walked: the containment boundary for the author-controlled
    /// artifact path (RFC 0019 §16), which may leave @ref baseDir but never the tree. Filled
    /// by the loader; empty for descriptors built in memory.
    /// Not part of operator==.
    std::filesystem::path treeRoot;

    /// Required for feature-consuming models. Semantic revisions are checked against
    /// the engine, metadata schema and matcher identities before a model is used.
    std::optional<HeuristicProvenance> trainedAgainst;
    /// RFC 0019 §4.1 `trained_against.selector_revision`: the provider build this model was
    /// measured on, for engines with no UED. Empty unless authored. A mismatch is refused, not
    /// warned about, because a stale L1 estimate changes which engine is selected.
    std::string trainedAgainstSelectorRevision;
    /// The role-map entry that resolved this model: backfilled by
    /// DescriptorLoader::resolveRole from the owning UED, never authored in the UHD
    /// (RFC 0019 §3.1). Empty for a model no role map has resolved yet.
    /// Backfilled, not authored. Not part of operator==.
    std::string engineName;
    std::string role;
    std::string arch;
    /// The authored `trained_against` object, kept verbatim for UhdConfig.
    nlohmann::json trainedAgainstJson;
};

inline bool operator==(const HeuristicDescriptor& lhs, const HeuristicDescriptor& rhs)
{
    return std::tie(lhs.id,
                    lhs.name,
                    lhs.adapter,
                    lhs.featuresSignature,
                    lhs.featuresHash,
                    lhs.categoricalEncoding,
                    lhs.objective,
                    lhs.score,
                    lhs.nativeSymbol,
                    lhs.authoredArtifactPath,
                    lhs.modelHash,
                    lhs.customLibrarySymbol,
                    lhs.trainedAgainst,
                    lhs.trainedAgainstSelectorRevision,
                    lhs.trainedAgainstJson)
           == std::tie(rhs.id,
                       rhs.name,
                       rhs.adapter,
                       rhs.featuresSignature,
                       rhs.featuresHash,
                       rhs.categoricalEncoding,
                       rhs.objective,
                       rhs.score,
                       rhs.nativeSymbol,
                       rhs.authoredArtifactPath,
                       rhs.modelHash,
                       rhs.customLibrarySymbol,
                       rhs.trainedAgainst,
                       rhs.trainedAgainstSelectorRevision,
                       rhs.trainedAgainstJson);
}

inline bool operator!=(const HeuristicDescriptor& lhs, const HeuristicDescriptor& rhs)
{
    return !(lhs == rhs);
}

/// The `native` arm of a UED's `graph_match`: a GraphMatchRegistry symbol. The function it
/// names decides the graph topology and publishes the binding.
struct NativeGraphMatch
{
    std::string symbol;
};

inline bool operator==(const NativeGraphMatch& lhs, const NativeGraphMatch& rhs)
{
    return lhs.symbol == rhs.symbol;
}

inline bool operator!=(const NativeGraphMatch& lhs, const NativeGraphMatch& rhs)
{
    return !(lhs == rhs);
}

/// How an engine binds a graph, as declared by its UED's `graph_match`.
///  - std::monostate: no `graph_match`. The engine binds nothing, and every graph reaches
///    the UMDs with an empty binding.
///  - NativeGraphMatch: the `native` arm, resolved through GraphMatchRegistry.
///  - CompiledGraphPattern: the declarative `nodes` arm, compiled against the op-schema
///    registry when the UED is read (see CompiledGraphPattern.hpp). Never null. Immutable
///    and shared by every copy of the descriptor.
using GraphMatch
    = std::variant<std::monostate, NativeGraphMatch, std::shared_ptr<const CompiledGraphPattern>>;

/// True when @p lhs and @p rhs declare the same arm with the same content. A compiled
/// pattern compares by structure, not by pointer, because every parse of a UED compiles
/// its own instance.
inline bool sameGraphMatch(const GraphMatch& lhs, const GraphMatch& rhs)
{
    if(lhs.index() != rhs.index())
    {
        return false;
    }
    if(const auto* native = std::get_if<NativeGraphMatch>(&lhs))
    {
        return *native == std::get<NativeGraphMatch>(rhs);
    }
    if(const auto* pattern = std::get_if<std::shared_ptr<const CompiledGraphPattern>>(&lhs))
    {
        return **pattern == *std::get<std::shared_ptr<const CompiledGraphPattern>>(rhs);
    }
    return true;
}

/// UED: the engine itself, carrying no logic of its own. `name` hashes into hipDNN's
/// engine-id space; must be globally unique, e.g. "rocke:SDPA".
///
/// `operator==` is authored-content equality: it ignores @ref heuristicId, which the loader
/// fills after parsing.
struct EngineDescriptor
{
    DescriptorId id;
    std::string name;
    /// The default catalog ranker for the `default` arch key (see DescriptorSet::heuristic).
    /// nullopt when the engine ships no UHD; selection then falls back to the
    /// descriptor-declared order. Filled by resolveDescriptorSets(), not at parse, because the
    /// default depends on the metric each loaded UHD declares.
    /// Not part of operator==.
    std::optional<DescriptorId> heuristicId;

    /// RFC 0019 §3.1 role-scoped UHDs, each optional, keyed by `gcnArchName` with an optional
    /// `default`. The scoring roles list UHDs per arch; the loader indexes them by each UHD's
    /// own `score.metric` rather than restating the metric here.
    std::map<std::string, std::vector<DescriptorId>> sortKernelCatalog;
    std::map<std::string, std::vector<DescriptorId>> predictEngine;
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
    /// The graph-topology match; std::monostate when the UED declares none.
    GraphMatch graphMatch;
    /// Authored knob and binding semantics, not the JSON file-format version.
    hipdnn_data_sdk::utilities::Version revision{1, 0, 0};
};

inline bool operator==(const EngineDescriptor& lhs, const EngineDescriptor& rhs)
{
    return sameGraphMatch(lhs.graphMatch, rhs.graphMatch)
           && std::tie(lhs.id,
                       lhs.name,
                       lhs.sortKernelCatalog,
                       lhs.predictEngine,
                       lhs.predictApplicableKernels,
                       lhs.metadataSchemaId,
                       lhs.knobs,
                       lhs.behaviorNotes,
                       lhs.sdkVersion,
                       lhs.numericalNotes,
                       lhs.revision)
                  == std::tie(rhs.id,
                              rhs.name,
                              rhs.sortKernelCatalog,
                              rhs.predictEngine,
                              rhs.predictApplicableKernels,
                              rhs.metadataSchemaId,
                              rhs.knobs,
                              rhs.behaviorNotes,
                              rhs.sdkVersion,
                              rhs.numericalNotes,
                              rhs.revision);
}

inline bool operator!=(const EngineDescriptor& lhs, const EngineDescriptor& rhs)
{
    return !(lhs == rhs);
}

/// The engine's native graph_match symbol, or an empty string when its `graph_match` is not
/// the `native` arm (a `nodes` pattern, or none).
inline std::string nativeGraphMatchSymbol(const EngineDescriptor& engine)
{
    const auto* native = std::get_if<NativeGraphMatch>(&engine.graphMatch);
    return native != nullptr ? native->symbol : std::string{};
}

namespace detail
{

/// The GraphMatch a native symbol spells: std::monostate for an empty symbol, otherwise the
/// `native` arm. Backs the overloads that take the symbol as a string.
inline GraphMatch graphMatchFromNativeSymbol(const std::string& symbol)
{
    if(symbol.empty())
    {
        return std::monostate{};
    }
    return NativeGraphMatch{symbol};
}

} // namespace detail

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
    hipdnn_data_sdk::utilities::Version revision{1, 0, 0};
};

inline bool operator==(const MatchDescriptor& lhs, const MatchDescriptor& rhs)
{
    return std::tie(lhs.id, lhs.name, lhs.scope, lhs.matchSymbol, lhs.revision)
           == std::tie(rhs.id, rhs.name, rhs.scope, rhs.matchSymbol, rhs.revision);
}

inline bool operator!=(const MatchDescriptor& lhs, const MatchDescriptor& rhs)
{
    return !(lhs == rhs);
}

/// UDD: how to invoke a kernel, shared by every kernel in a pack.
struct DispatchDescriptor
{
    DescriptorId id;
    std::string name;
    std::string dispatchSymbol; ///< Resolved through NativeRegistry.
};

inline bool operator==(const DispatchDescriptor& lhs, const DispatchDescriptor& rhs)
{
    return std::tie(lhs.id, lhs.name, lhs.dispatchSymbol)
           == std::tie(rhs.id, rhs.name, rhs.dispatchSymbol);
}

inline bool operator!=(const DispatchDescriptor& lhs, const DispatchDescriptor& rhs)
{
    return !(lhs == rhs);
}

/// Which adapter loads and prepares one kernel's code.
enum class KernelSourceKind
{
    EMBEDDED_SOURCE, ///< Source file plus entry point, compiled at plan-build time.
    KPACK, ///< Prebuilt kpack archive plus toc key and symbol.
    HSACO_FILE, ///< Standalone `.hsaco` code-object file. No adapter yet.
    ROCKE_BUILDER, ///< rocke builder name plus build values. No adapter yet.
};

/// The descriptor spelling of @p kind, which `kernelSourceKindFromString` parses. A value
/// outside the enum gives "unknown (<value>)".
inline std::string toString(KernelSourceKind kind)
{
    switch(kind)
    {
    case KernelSourceKind::EMBEDDED_SOURCE:
        return "embedded_source";
    case KernelSourceKind::KPACK:
        return "kpack";
    case KernelSourceKind::HSACO_FILE:
        return "hsaco_file";
    case KernelSourceKind::ROCKE_BUILDER:
        return "rocke_builder";
    default:
        break;
    }
    return "unknown (" + std::to_string(static_cast<int>(kind)) + ")";
}

/// One argument a kernel expects the host to marshal, as the compiled code object's AMDGPU
/// metadata note declares it -- read out of the object by the packager, never authored.
///
/// Compiler-appended hidden arguments are excluded: the host does not marshal them, so
/// keeping them would make this list incomparable with the one a pack passes to `launch`.
struct KernelArgument
{
    /// `.value_kind` verbatim: `global_buffer`, `by_value`, and so on. Kept as the string
    /// the toolchain emits rather than mapped to an enum, so a kind this build has never
    /// heard of still parses, still compares, and still prints in a diagnostic.
    std::string kind;
    uint32_t size = 0; ///< Bytes it occupies in the kernarg segment.
    uint32_t offset = 0; ///< Where it starts within that segment.
    /// `.name`, empty when the producer emitted none -- clang omits names for HIP
    /// `extern "C" __global__` kernels. Empty means "this producer records no names",
    /// never "named the empty string".
    std::string name;
};

inline bool operator==(const KernelArgument& lhs, const KernelArgument& rhs)
{
    return std::tie(lhs.kind, lhs.size, lhs.offset, lhs.name)
           == std::tie(rhs.kind, rhs.size, rhs.offset, rhs.name);
}

inline bool operator!=(const KernelArgument& lhs, const KernelArgument& rhs)
{
    return !(lhs == rhs);
}

/// One argument list rendered for a diagnostic, as `kind:size@offset` per argument with the
/// name appended when there is one.
inline std::string describeKernelSignature(const std::vector<KernelArgument>& signature)
{
    if(signature.empty())
    {
        return "(no arguments)";
    }
    std::string text;
    for(const auto& argument : signature)
    {
        if(!text.empty())
        {
            text += ", ";
        }
        text += argument.kind + ":" + std::to_string(argument.size) + "@"
                + std::to_string(argument.offset);
        if(!argument.name.empty())
        {
            text += " '" + argument.name + "'";
        }
    }
    return text;
}

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
    /// scheme entirely, so nothing here may parse it. An authored hsaco's key hashes the
    /// normalized root-relative file path, so one file serving several symbols is one entry.
    /// Not unique per kernel: two kernels differing only by entry point share one key, one
    /// blob, and one loaded module.
    std::string tocKey;
    /// KPACK: the undecorated extern "C" name to resolve inside the loaded module. The
    /// only field that separates two kernels sharing a tocKey.
    std::string symbol;
    /// KPACK: digest of the raw decompressed code object, as the packager recorded it.
    ///
    /// Checked before the code object reaches the driver: a TOC entry pointing at the
    /// wrong offset decompresses cleanly and yields another entry's blob, which only this
    /// field can catch. An integrity check, not a security control -- a descriptor travels
    /// with the archive it names, so whoever can rewrite one can rewrite the other.
    std::string sha256;
    /// KPACK: the argument list `symbol` declares, read out of the code object at pack
    /// time. Not a description of the call -- the list a pack marshals is fixed in its
    /// C++; this is the copy that comparison checks it against, in requireSignatureMatch.
    ///
    /// Empty is a kernel that takes no arguments, which is why the parser rejects the key
    /// being absent rather than reading absence as empty.
    std::vector<KernelArgument> signature;
};

inline bool operator==(const KernelSource& lhs, const KernelSource& rhs)
{
    return std::tie(lhs.kind,
                    lhs.sourceFile,
                    lhs.entryPoint,
                    lhs.library,
                    lhs.tocKey,
                    lhs.symbol,
                    lhs.sha256,
                    lhs.signature)
           == std::tie(rhs.kind,
                       rhs.sourceFile,
                       rhs.entryPoint,
                       rhs.library,
                       rhs.tocKey,
                       rhs.symbol,
                       rhs.sha256,
                       rhs.signature);
}

inline bool operator!=(const KernelSource& lhs, const KernelSource& rhs)
{
    return !(lhs == rhs);
}

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

// KernelSource's field count is pinned here: accepting exactly eight initializers and no
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
                                               std::string,
                                               std::vector<KernelArgument>>
                  && !detail::IS_BRACE_INITIALIZABLE_V<KernelSource,
                                                       KernelSourceKind,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::string,
                                                       std::vector<KernelArgument>,
                                                       std::string>,
              "KernelSource gained or lost a field; append only, then extend this "
              "assertion.");

/// UKD: one launchable kernel. Matchers, engine, and dispatch come from its pack.
///
/// `operator==` is authored-content equality: it ignores @ref originDirectory and @ref treeRoot,
/// which the loader stamps from the file's location after parsing, so a kernel shipped
/// identically in several per-arch shards compares equal. `metadata` values compare
/// numerically across the integer and floating-point alternatives (`1` equals `1.0`).
struct KernelDescriptor
{
    DescriptorId id;
    std::string name;
    KernelSource source;
    /// RFC 0017 §15.1: the launches of a multi-launch kernel, keyed by launch name, in place
    /// of @c source. Empty for every kernel that ships one launch, which keeps `source`;
    /// non-empty always holds at least two entries and leaves `source` default. The
    /// descriptor only lists the entries: launch order, workspace and the hand-off between
    /// stages belong to the native dispatch handler of the pack, which also declares the
    /// names it launches.
    std::map<std::string, KernelSource> sources;
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
    /// kernel built in memory, which names no file. Filled by the loader, never authored. Not part
    /// of operator==.
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
    /// in memory. Filled by the loader, never authored. Not part of operator==.
    std::filesystem::path treeRoot;
};

inline bool operator==(const KernelDescriptor& lhs, const KernelDescriptor& rhs)
{
    return detail::metadataValuesEquivalent(lhs.metadata, rhs.metadata)
           && std::tie(lhs.id, lhs.name, lhs.source, lhs.sources, lhs.priority, lhs.arch)
                  == std::tie(rhs.id, rhs.name, rhs.source, rhs.sources, rhs.priority, rhs.arch);
}

inline bool operator!=(const KernelDescriptor& lhs, const KernelDescriptor& rhs)
{
    return !(lhs == rhs);
}

/// The launch names of a kernel's @c sources map: the set a native dispatch handler must
/// declare to launch the kernel. Empty for a single-launch kernel, which keeps `source`.
inline std::set<std::string> launchNamesOf(const std::map<std::string, KernelSource>& sources)
{
    std::set<std::string> names;
    for(const auto& launch : sources)
    {
        names.insert(launch.first);
    }
    return names;
}

// A multi-launch kernel leaves `source` default, and a default KernelSource reads as an
// EMBEDDED_SOURCE kernel with no file. Read a kernel's sources through the two helpers
// below, which look at `sources` first, rather than through `source` directly. Each takes
// a KernelDescriptor or a KernelDefinition; both carry `source` and `sources`.

/// The one source of a single-launch @p kernel, or nullptr for a multi-launch kernel. For a
/// reader that needs exactly one code object, which must refuse a multi-launch kernel. The
/// pointer refers into @p kernel and is valid only while @p kernel is, so never pass a
/// temporary.
template <typename Kernel>
const KernelSource* singleLaunchSource(const Kernel& kernel)
{
    return kernel.sources.empty() ? &kernel.source : nullptr;
}

/// True iff @p predicate holds for every launch of @p kernel: each `sources` entry of a
/// multi-launch kernel, or the one `source` of a single-launch kernel.
template <typename Kernel, typename Predicate>
bool allLaunchSources(const Kernel& kernel, Predicate predicate)
{
    if(kernel.sources.empty())
    {
        return predicate(kernel.source);
    }
    return std::all_of(kernel.sources.begin(), kernel.sources.end(), [&](const auto& launch) {
        return predicate(launch.second);
    });
}

/// KDP: one pack binding a matcher set, one engine, and one dispatch descriptor over
/// a vector of child kernels.
struct KernelDescriptorPack
{
    DescriptorId id;
    std::string name;
    std::vector<DescriptorId> matcherIds;
    DescriptorId engineId;
    DescriptorId dispatchId;
    /// GFX targets, e.g. `{"gfx942", "gfx950"}`; empty means arch-independent. A concrete
    /// entry is matched exactly against the device's base target id; a generic entry
    /// matches by membership in the generic target table. Part of the pack's catalog identity
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

inline bool operator==(const KernelDescriptorPack& lhs, const KernelDescriptorPack& rhs)
{
    return std::tie(lhs.id,
                    lhs.name,
                    lhs.matcherIds,
                    lhs.engineId,
                    lhs.dispatchId,
                    lhs.arch,
                    lhs.kernels,
                    lhs.kernelIds)
           == std::tie(rhs.id,
                       rhs.name,
                       rhs.matcherIds,
                       rhs.engineId,
                       rhs.dispatchId,
                       rhs.arch,
                       rhs.kernels,
                       rhs.kernelIds);
}

inline bool operator!=(const KernelDescriptorPack& lhs, const KernelDescriptorPack& rhs)
{
    return !(lhs == rhs);
}

/// One engine and every descriptor it references by id; self-contained.
struct DescriptorSet
{
    EngineDescriptor engine;
    MetadataSchema schema;
    /// The default ranker for the `default` architecture key (RFC 0019 §3.1): the
    /// metric-less `sort_kernel_catalog` UHD if there is one, else the one for
    /// `DEFAULT_RANKING_METRIC`. nullopt when neither exists; the generic engine then ranks
    /// on `priority` then descriptor id. See makeKernelHeuristic().
    std::optional<HeuristicDescriptor> heuristic;

    /// RFC 0019 §3.1 catalog-ranking UHDs by declared metric (`""` for the metric-less
    /// ranker), then by arch key as the UED wrote it. Arch resolution (exact, then `default`,
    /// within one metric) happens at first rank(): discovery runs before any device exists.
    std::map<std::string, std::map<std::string, HeuristicDescriptor>> heuristicsByMetric;
    /// Resolved cheap graph/device models by metric, then architecture; never consumed by
    /// catalog ranking. Every `predict_engine` UHD declares a metric, so there is no `""`.
    std::map<std::string, std::map<std::string, HeuristicDescriptor>> enginePredictionsByMetric;
    std::vector<MatchDescriptor> matchers;
    std::vector<DispatchDescriptor> dispatches;
    std::vector<KernelDescriptorPack> packs;
    /// Explicitly named models that could not be used, by metric (`""` for the metric-less
    /// ranker) and architecture key. An exact-arch failure must not silently select another
    /// architecture's model for the same metric.
    std::map<std::string, std::set<std::string>> unavailableHeuristicArches;
    std::map<std::string, std::set<std::string>> unavailableEnginePredictionArches;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
