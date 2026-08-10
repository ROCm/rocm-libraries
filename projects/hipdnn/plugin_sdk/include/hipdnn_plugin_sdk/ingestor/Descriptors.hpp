// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

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

/// A descriptor cross-reference. Stable and globally unique in the real system (a GUID);
/// any unique string here, since nothing is loaded from a file yet.
using DescriptorId = std::string;

/// A value a kernel supplies for a KMD-declared field. The KMD declares which type each
/// field holds; a UKD carrying the wrong alternative for a field is a load error the
/// validating loader reports (not modelled here, since nothing is loaded).
using MetadataValue = std::variant<int64_t, std::string>;

/// A kernel's complete metadata tuple: every KMD field name mapped to this kernel's
/// value for it. RFC 0017 §4 makes this tuple the kernel's identity to the catalog, so
/// it must be unique across every kernel in an engine. Ordered, so two kernels with the
/// same fields compare and hash identically regardless of insertion order.
using MetadataValues = std::map<std::string, MetadataValue>;

/// One field a kernel may vary along, as declared by an engine's KMD.
struct MetadataField
{
    std::string name;
    /// The value a kernel that omits this field is taken to have supplied.
    MetadataValue defaultValue;
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
    /// The kernel source. Real sources are a tagged union over kpack, hsaco, hip, and
    /// rocke (RFC 0017 §7); this skeleton carries only what the embedded-source path
    /// needs, and a real source kind is the packaging follow-up.
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
