// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
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

/// Which adapter builds an engine's IKernelHeuristic from a UHD's `payload`.
enum class HeuristicKind
{
    NATIVE, ///< NativeRegistry score symbol. Only kind with an adapter today.
    MODEL, ///< Trained model artifact plus feature signature. No adapter yet.
};

/// UHD: the kernel-selection model for one engine.
struct HeuristicDescriptor
{
    DescriptorId id;
    std::string name;
    HeuristicKind kind = HeuristicKind::NATIVE;
    std::string payload;
};

/// UED: the engine itself, carrying no logic of its own. `name` hashes into hipDNN's
/// engine-id space; must be globally unique, e.g. "rocke:SDPA".
struct EngineDescriptor
{
    DescriptorId id;
    std::string name;
    /// nullopt when the engine ships no UHD; selection then falls back to the
    /// descriptor-declared order. Must equal `DescriptorSet::heuristic`'s id when set.
    std::optional<DescriptorId> heuristicId;
    DescriptorId metadataSchemaId;
    std::vector<std::string> knobs;
    /// `hipdnnBackendBehaviorNote_t` values; int32 so a newer note isn't truncated.
    std::vector<int32_t> behaviorNotes;
    /// Graph schema version this engine understands; a graph below this floor is
    /// declined rather than matched with an ignored field. Baseline by default.
    hipdnn_data_sdk::utilities::Version sdkVersion{K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    /// RFC 0020 §4.2 numerical notes, held as authored. Unlike `behaviorNotes` these map
    /// to no hipDNN enum -- there is no numerical-note vocabulary yet -- so the loader
    /// carries the strings and nothing consumes them.
    std::vector<std::string> numericalNotes;
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
    KPACK_SYMBOL, ///< Prebuilt kpack library plus symbol. No adapter yet.
    HSACO_FILE, ///< Standalone `.hsaco` code-object file. No adapter yet.
    ROCKE_BUILDER, ///< rocke builder name plus build values. No adapter yet.
};

/// UKD's source. Only `EMBEDDED_SOURCE` is implemented.
struct KernelSource
{
    KernelSourceKind kind = KernelSourceKind::EMBEDDED_SOURCE;
    std::string sourceFile;
    std::string entryPoint;
};

/// UKD: one launchable kernel. Matchers, engine, and dispatch come from its pack.
struct KernelDescriptor
{
    DescriptorId id;
    std::string name;
    KernelSource source;
    /// Omitted fields take the KMD default; completed tuple is the catalog key.
    MetadataValues metadata;
    int64_t priority = 0; ///< Tie-break when the heuristic is not decisive.
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
    /// GFX targets, e.g. `{"gfx942", "gfx950"}`; empty means arch-independent.
    /// Matches the base target id exactly, so `gfx942` never accepts `gfx950`.
    ///
    /// Enforced only at catalog build (KernelIngestorStateManager::buildCatalog), against
    /// the device that call targets. There is no load-time gate, so a pack every local
    /// device excludes is still built and simply declines per call.
    ///
    /// An arch-excluded pack is a correct, expected decline, the same category as a matcher
    /// returning false -- reporting it as malformed makes a healthy cross-arch install read
    /// as a pile of failures.
    std::vector<std::string> arch;
    std::vector<KernelDescriptor> kernels;
};

/// One engine and every descriptor it references by id; self-contained.
struct DescriptorSet
{
    EngineDescriptor engine;
    MetadataSchema schema;
    /// nullopt when this engine ships no ranking model; the generic engine then ranks
    /// on `priority` then descriptor id. See makeKernelHeuristic().
    std::optional<HeuristicDescriptor> heuristic;
    std::vector<MatchDescriptor> matchers;
    std::vector<DispatchDescriptor> dispatches;
    std::vector<KernelDescriptorPack> packs;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
