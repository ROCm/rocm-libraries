// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

/**
 * @file DescriptorLoader.hpp
 * @brief Reads descriptor files from disk into the types Descriptors.hpp models.
 *
 * The UED follows RFC 0020, which is the source of truth for that file; the other five
 * follow RFC 0017 §4 until their own follow-ups land.
 *
 * One descriptor per file, named `<id>.json`; the file's own `id` field is authoritative
 * and the filename is never parsed. Because neither the filename nor the directory carries
 * a type, each file declares its own with a `schema` string:
 *
 * | `schema`          | Struct               |
 * |-------------------|----------------------|
 * | `hipdnn.kmd/v1`   | MetadataSchema       |
 * | `hipdnn.uhd/v1`   | HeuristicDescriptor  |
 * | `hipdnn.ued/v1`   | EngineDescriptor     |
 * | `hipdnn.umd/v1`   | MatchDescriptor      |
 * | `hipdnn.udd/v1`   | DispatchDescriptor   |
 * | `hipdnn.kdp/v1`   | KernelDescriptorPack |
 *
 * The tag is matched exactly, so an unrecognised one is skipped with its path logged.
 * RFC 0017 §4 versions each type independently as `major.minor` and gates it at load, but
 * carries that in a sibling `version` field this loader does not parse yet; the minor half
 * of that rule arrives with the field. A tag naming a major this build has no reader for
 * is already refused, since it matches nothing here.
 *
 * Apart from the UED, whose keys RFC 0020 §4.2 fixes, every JSON key is the `snake_case`
 * spelling of the C++ field name, and any key not spelled by the struct is a parse error
 * rather than a silent no-op, so a typo is reported instead of ignored. Optional keys,
 * with their defaults: `default_value` (absent -> std::nullopt), `priority` (absent -> 0),
 * and `knobs` / `behavior_notes` / `numerical_notes` / `metadata` (absent -> empty).
 * Everything else is required.
 *
 * The authored format is deliberately a subset of what the RFCs describe, carrying only
 * what Descriptors.hpp models: the declarative `nodes`/`criteria`, `grid`/`block`/
 * `args_signature`, `features_signature`, per-file `sdk_version` and KDP `arch` have no
 * parsed representation yet, so they are rejected as unknown keys. They arrive with the
 * follow-up RFCs that add the fields; because this loader is a mechanical mirror of the
 * structs, adding a field is a change in both places and nowhere else.
 *
 * Nothing here throws to its caller. A malformed file, an unresolvable cross-reference,
 * or an unregistered native symbol is logged at ERROR naming the file, id and name, and
 * skipped, so one bad descriptor never costs a working engine.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// One parsed descriptor plus the provenance duplicate resolution needs.
template <typename T>
struct CatalogEntry
{
    T descriptor;
    /// The parsed document, kept so a second file claiming the same id can be compared
    /// against this one by content. Comparing the parsed JSON ignores key order and
    /// whitespace; comparing the parsed structs instead would need an operator== on all
    /// seven descriptor types that nothing else in the system wants.
    nlohmann::json source;
    std::filesystem::path path; ///< first file that defined this id
    bool conflicted = false;    ///< two files disagreed; treat as absent
};

template <typename T>
using DescriptorMap = std::unordered_map<DescriptorId, CatalogEntry<T>, DescriptorIdHash>;

/// Every descriptor found under a root, keyed by id, one map per type. Identity is
/// (type, id): the same GUID naming a UED in one file and a KMD in another is legal
/// and invisible here.
struct DescriptorCatalog
{
    DescriptorMap<MetadataSchema> schemas;
    DescriptorMap<HeuristicDescriptor> heuristics;
    DescriptorMap<EngineDescriptor> engines;
    DescriptorMap<MatchDescriptor> matchers;
    DescriptorMap<DispatchDescriptor> dispatches;
    DescriptorMap<KernelDescriptorPack> packs;
};

namespace detail
{

/// A file's declared type. RFC 0017 §4 versions each type independently as `major.minor`
/// but carries that in a sibling `version` field, not in this tag, so the tag is matched
/// exactly and a `/v2` file is an unrecognised type -- which is the right answer for a
/// major bump anyway, since no reader for it exists here. The minor half of the accept
/// rule arrives with `version` when the loader parses it.
inline constexpr std::string_view SCHEMA_KMD = "hipdnn.kmd/v1";
inline constexpr std::string_view SCHEMA_UHD = "hipdnn.uhd/v1";
inline constexpr std::string_view SCHEMA_UED = "hipdnn.ued/v1";
inline constexpr std::string_view SCHEMA_UMD = "hipdnn.umd/v1";
inline constexpr std::string_view SCHEMA_UDD = "hipdnn.udd/v1";
inline constexpr std::string_view SCHEMA_KDP = "hipdnn.kdp/v1";

/// Every parse violation leaves through here, so the caller catches one type and the
/// message never carries the path: the caller already has it and adds it to the log.
[[noreturn]] inline void fail(const std::string& message)
{
    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, message);
}

inline void requireObject(const nlohmann::json& value, const std::string& where)
{
    if(!value.is_object())
    {
        fail(where + " must be a JSON object");
    }
}

/// Rejects any key the struct does not spell, so an authoring typo is reported rather
/// than dropped on the floor.
inline void requireOnlyKeys(const nlohmann::json& object,
                            std::initializer_list<std::string_view> allowed,
                            const std::string& where)
{
    for(const auto& item : object.items())
    {
        if(std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
        {
            fail("unknown key '" + item.key() + "' in " + where);
        }
    }
}

inline const nlohmann::json&
    requireKey(const nlohmann::json& object, std::string_view key, const std::string& where)
{
    const auto it = object.find(std::string(key));
    if(it == object.end())
    {
        fail("missing required key '" + std::string(key) + "' in " + where);
    }
    return *it;
}

/// Every string in the format names something -- a descriptor, a symbol, a field, a file
/// -- so an empty one is an authoring mistake, not a value: it would reach EngineRegistrar
/// or the runtime compiler as a blank identifier.
inline std::string
    requireString(const nlohmann::json& object, std::string_view key, const std::string& where)
{
    const auto& value = requireKey(object, key, where);
    if(!value.is_string())
    {
        fail("key '" + std::string(key) + "' in " + where + " must be a string");
    }
    auto text = value.get<std::string>();
    if(text.empty())
    {
        fail("key '" + std::string(key) + "' in " + where + " must not be empty");
    }
    return text;
}

inline DescriptorId
    requireId(const nlohmann::json& object, std::string_view key, const std::string& where)
{
    const auto text = requireString(object, key, where);
    try
    {
        return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
    }
    catch(const std::invalid_argument& error)
    {
        fail("key '" + std::string(key) + "' in " + where + " is not a UUID: " + error.what());
    }
}

/// RFC 0020 §4.2: the engine name is a scoped `namespace:local` identifier, because it is
/// hashed into the 64-bit engine-id space and must be globally unique -- an unscoped
/// "pointwise" is exactly the name two vendors both pick.
inline void requireScopedName(const std::string& name, const std::string& where)
{
    const auto colon = name.find(':');
    const auto isNameChar = [](unsigned char c) {
        return std::isalnum(c) != 0 || c == '_' || c == '.' || c == '-';
    };
    if(colon == std::string::npos || colon == 0 || colon + 1 == name.size()
       || !std::all_of(name.begin(), name.end(), [&](unsigned char c) {
              return c == ':' || isNameChar(c);
          })
       || name.find(':', colon + 1) != std::string::npos)
    {
        fail("engine name '" + name + "' in " + where
             + " must be a scoped 'namespace:local' name matching"
               " ^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$");
    }
}

/// An absent array key means an empty list, which is what every optional list-valued
/// descriptor field defaults to.
inline std::vector<std::string> optionalStringArray(const nlohmann::json& object,
                                                    std::string_view key,
                                                    const std::string& where)
{
    std::vector<std::string> values;
    const auto it = object.find(std::string(key));
    if(it == object.end())
    {
        return values;
    }
    if(!it->is_array())
    {
        fail("key '" + std::string(key) + "' in " + where + " must be an array of strings");
    }
    for(const auto& element : *it)
    {
        if(!element.is_string())
        {
            fail("key '" + std::string(key) + "' in " + where + " must be an array of strings");
        }
        values.push_back(element.get<std::string>());
    }
    return values;
}

inline MetadataType metadataTypeFromString(const std::string& text, const std::string& where)
{
    if(text == "bool")
    {
        return MetadataType::BOOL;
    }
    if(text == "int")
    {
        return MetadataType::INT;
    }
    if(text == "float")
    {
        return MetadataType::FLOAT;
    }
    if(text == "string")
    {
        return MetadataType::STRING;
    }
    if(text == "int_list")
    {
        return MetadataType::INT_LIST;
    }
    fail("unknown metadata type '" + text + "' in " + where);
}

inline HeuristicKind heuristicKindFromString(const std::string& text, const std::string& where)
{
    if(text == "native")
    {
        return HeuristicKind::NATIVE;
    }
    if(text == "model")
    {
        return HeuristicKind::MODEL;
    }
    fail("unknown heuristic kind '" + text + "' in " + where);
}

inline MatchScope matchScopeFromString(const std::string& text, const std::string& where)
{
    if(text == "graph")
    {
        return MatchScope::GRAPH;
    }
    if(text == "kernel")
    {
        return MatchScope::KERNEL;
    }
    fail("unknown match scope '" + text + "' in " + where);
}

inline KernelSourceKind kernelSourceKindFromString(const std::string& text,
                                                   const std::string& where)
{
    if(text == "embedded_source")
    {
        return KernelSourceKind::EMBEDDED_SOURCE;
    }
    if(text == "kpack_symbol")
    {
        return KernelSourceKind::KPACK_SYMBOL;
    }
    if(text == "hsaco_file")
    {
        return KernelSourceKind::HSACO_FILE;
    }
    if(text == "rocke_builder")
    {
        return KernelSourceKind::ROCKE_BUILDER;
    }
    fail("unknown kernel source kind '" + text + "' in " + where);
}

/// Behavior notes are authored as names and mapped here to their transport values. An
/// unknown name is a parse error: reject, never reinterpret. One entry is all this
/// change needs; add further ones when a descriptor needs them.
inline int32_t behaviorNoteFromString(const std::string& text, const std::string& where)
{
    if(text == "runtime_compilation")
    {
        return HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
    }
    fail("unknown behavior note '" + text + "' in " + where);
}

/// nlohmann reports an unsigned literal above INT64_MAX as an integer, and get<int64_t>()
/// would static_cast it to a negative value. Every integer the format carries is an
/// int64_t, so one that does not fit is rejected rather than silently reinterpreted.
inline int64_t requireInt64(const nlohmann::json& value, const std::string& where)
{
    if(value.is_number_unsigned()
       && value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
        fail(where + " is too large for a 64-bit signed integer");
    }
    return value.get<int64_t>();
}

/// A value's JSON kind decides its alternative: boolean -> bool, integer -> int64_t,
/// real -> double, string -> std::string, array of integers -> std::vector<int64_t>.
inline MetadataValue metadataValueFromJson(const nlohmann::json& value, const std::string& where)
{
    if(value.is_boolean())
    {
        return MetadataValue{value.get<bool>()};
    }
    if(value.is_number_integer())
    {
        return MetadataValue{requireInt64(value, where)};
    }
    if(value.is_number_float())
    {
        return MetadataValue{value.get<double>()};
    }
    if(value.is_string())
    {
        return MetadataValue{value.get<std::string>()};
    }
    if(value.is_array())
    {
        std::vector<int64_t> list;
        for(const auto& element : value)
        {
            if(!element.is_number_integer())
            {
                fail(where + " must be an array of integers");
            }
            list.push_back(requireInt64(element, where));
        }
        return MetadataValue{std::move(list)};
    }
    fail(where + " must be a boolean, a number, a string, or an array of integers");
}

/// Widens an authored integer to the double a FLOAT field declares, because JSON writes
/// 1 and 1.0 identically and an author who wrote the former meant the field's type.
/// Every other mismatch is a genuine type error and is left for the caller to report.
inline bool coerceToDeclaredType(MetadataValue& value, MetadataType declared)
{
    if(metadataTypeOf(value) == declared)
    {
        return true;
    }
    if(declared == MetadataType::FLOAT && std::holds_alternative<int64_t>(value))
    {
        value = MetadataValue{static_cast<double>(std::get<int64_t>(value))};
        return true;
    }
    return false;
}

inline MetadataSchema parseMetadataSchema(const nlohmann::json& root)
{
    const std::string where{SCHEMA_KMD};
    requireOnlyKeys(root, {"schema", "id", "name", "fields"}, where);

    MetadataSchema schema;
    schema.id = requireId(root, "id", where);
    schema.name = requireString(root, "name", where);

    const auto& fields = requireKey(root, "fields", where);
    if(!fields.is_array())
    {
        fail("key 'fields' in " + where + " must be an array");
    }
    for(const auto& fieldJson : fields)
    {
        requireObject(fieldJson, "a 'fields' entry");
        requireOnlyKeys(fieldJson, {"name", "type", "default_value"}, "a 'fields' entry");

        MetadataField field;
        field.name = requireString(fieldJson, "name", "a 'fields' entry");
        const std::string fieldWhere = "field '" + field.name + "'";
        field.type = metadataTypeFromString(requireString(fieldJson, "type", fieldWhere),
                                            fieldWhere);

        if(const auto it = fieldJson.find("default_value"); it != fieldJson.end())
        {
            auto value = metadataValueFromJson(*it, fieldWhere + " default_value");
            if(!coerceToDeclaredType(value, field.type))
            {
                fail(fieldWhere + " has a default_value whose type contradicts its declared type");
            }
            field.defaultValue = std::move(value);
        }
        schema.fields.push_back(std::move(field));
    }
    return schema;
}

inline HeuristicDescriptor parseHeuristicDescriptor(const nlohmann::json& root)
{
    const std::string where{SCHEMA_UHD};
    requireOnlyKeys(root, {"schema", "id", "name", "kind", "payload"}, where);

    HeuristicDescriptor heuristic;
    heuristic.id = requireId(root, "id", where);
    heuristic.name = requireString(root, "name", where);
    heuristic.kind = heuristicKindFromString(requireString(root, "kind", where), where);
    heuristic.payload = requireString(root, "payload", where);
    return heuristic;
}

/// Duplicates in any of the UED's three lists are authoring mistakes rather than
/// redundancies to collapse: a repeated knob is reported twice in EngineDetails, and a
/// repeated note twice in diagnostics (RFC 0020 §4.2, `uniqueItems`).
inline void requireNoDuplicates(const std::vector<std::string>& values,
                                std::string_view what,
                                const std::string& where)
{
    for(auto value = values.begin(); value != values.end(); ++value)
    {
        if(std::find(values.begin(), value, *value) != value)
        {
            fail(std::string(what) + " '" + *value + "' is listed twice in " + where);
        }
    }
}

inline EngineDescriptor parseEngineDescriptor(const nlohmann::json& root)
{
    const std::string where{SCHEMA_UED};
    requireOnlyKeys(root,
                    {"schema",
                     "id",
                     "name",
                     "sdk_version",
                     "heuristic",
                     "metadata",
                     "knobs",
                     "behavior_notes",
                     "numerical_notes"},
                    where);

    EngineDescriptor engine;
    engine.id = requireId(root, "id", where);
    engine.name = requireString(root, "name", where);
    requireScopedName(engine.name, where);
    engine.heuristicId = requireId(root, "heuristic", where);
    engine.metadataSchemaId = requireId(root, "metadata", where);
    engine.knobs = optionalStringArray(root, "knobs", where);
    requireNoDuplicates(engine.knobs, "knob", where);

    const auto behaviorNotes = optionalStringArray(root, "behavior_notes", where);
    requireNoDuplicates(behaviorNotes, "behavior note", where);
    for(const auto& note : behaviorNotes)
    {
        engine.behaviorNotes.push_back(behaviorNoteFromString(note, where));
    }

    // Carried as authored text, not mapped to an enum like the behavior notes above:
    // hipDNN has no numerical-note vocabulary yet, so there is nothing to map onto and
    // nothing reads these. They are parsed because RFC 0020 §4.2 makes them a legal field
    // and rejecting a conforming UED is worse than holding a string nobody asks for.
    engine.numericalNotes = optionalStringArray(root, "numerical_notes", where);
    requireNoDuplicates(engine.numericalNotes, "numerical note", where);

    // The graph schema this engine's descriptors were authored against. Absent leaves
    // the struct's baseline default, which every graph's floor is at least equal to, so
    // an engine that declares nothing keeps behaving as it did before the field existed.
    // Gating on it is match-time and belongs to the engine (Descriptors.hpp); the loader
    // only carries the value.
    if(const auto it = root.find("sdk_version"); it != root.end())
    {
        try
        {
            engine.sdkVersion = hipdnn_data_sdk::utilities::Version{
                requireString(root, "sdk_version", where)};
        }
        catch(const std::invalid_argument& error)
        {
            fail("key 'sdk_version' in " + where + " is not a version: " + error.what());
        }
    }
    return engine;
}

inline MatchDescriptor parseMatchDescriptor(const nlohmann::json& root)
{
    const std::string where{SCHEMA_UMD};
    requireOnlyKeys(root, {"schema", "id", "name", "scope", "match_symbol"}, where);

    MatchDescriptor matcher;
    matcher.id = requireId(root, "id", where);
    matcher.name = requireString(root, "name", where);
    matcher.scope = matchScopeFromString(requireString(root, "scope", where), where);
    matcher.matchSymbol = requireString(root, "match_symbol", where);
    return matcher;
}

inline DispatchDescriptor parseDispatchDescriptor(const nlohmann::json& root)
{
    const std::string where{SCHEMA_UDD};
    requireOnlyKeys(root, {"schema", "id", "name", "dispatch_symbol"}, where);

    DispatchDescriptor dispatch;
    dispatch.id = requireId(root, "id", where);
    dispatch.name = requireString(root, "name", where);
    dispatch.dispatchSymbol = requireString(root, "dispatch_symbol", where);
    return dispatch;
}

inline KernelSource parseKernelSource(const nlohmann::json& root, const std::string& where)
{
    requireObject(root, where);
    requireOnlyKeys(root, {"kind", "source_file", "entry_point"}, where);

    KernelSource source;
    source.kind = kernelSourceKindFromString(requireString(root, "kind", where), where);
    // Only an embedded-source kernel is compiled from a file at an entry point; every
    // other kind leaves both empty per Descriptors.hpp, so requiring them there would
    // mean authoring values the runtime never reads.
    if(source.kind == KernelSourceKind::EMBEDDED_SOURCE)
    {
        source.sourceFile = requireString(root, "source_file", where);
        source.entryPoint = requireString(root, "entry_point", where);
    }
    else
    {
        requireOnlyKeys(root, {"kind"}, where);
    }
    return source;
}

inline KernelDescriptor parseKernelDescriptor(const nlohmann::json& root)
{
    requireObject(root, "a 'kernels' entry");
    requireOnlyKeys(root, {"id", "name", "source", "metadata", "priority"}, "a 'kernels' entry");

    KernelDescriptor kernel;
    kernel.id = requireId(root, "id", "a 'kernels' entry");
    kernel.name = requireString(root, "name", "a 'kernels' entry");
    const std::string where = "kernel '" + kernel.name + "'";
    kernel.source = parseKernelSource(requireKey(root, "source", where), where + " source");

    if(const auto it = root.find("metadata"); it != root.end())
    {
        requireObject(*it, where + " metadata");
        for(const auto& item : it->items())
        {
            // Parsed by JSON kind only. The values are checked and coerced against the
            // engine's KMD during set resolution, which is the first point the schema is
            // known.
            kernel.metadata.emplace(
                item.key(),
                metadataValueFromJson(item.value(), where + " metadata '" + item.key() + "'"));
        }
    }

    if(const auto it = root.find("priority"); it != root.end())
    {
        if(!it->is_number_integer())
        {
            fail(where + " priority must be an integer");
        }
        kernel.priority = requireInt64(*it, where + " priority");
    }
    return kernel;
}

inline KernelDescriptorPack parseKernelDescriptorPack(const nlohmann::json& root)
{
    const std::string where{SCHEMA_KDP};
    requireOnlyKeys(
        root,
        {"schema", "id", "name", "matcher_ids", "engine_id", "dispatch_id", "kernels"},
        where);

    KernelDescriptorPack pack;
    pack.id = requireId(root, "id", where);
    pack.name = requireString(root, "name", where);
    pack.engineId = requireId(root, "engine_id", where);
    pack.dispatchId = requireId(root, "dispatch_id", where);

    const auto& matcherIds = requireKey(root, "matcher_ids", where);
    if(!matcherIds.is_array())
    {
        fail("key 'matcher_ids' in " + where + " must be an array of UUID strings");
    }
    for(const auto& matcherId : matcherIds)
    {
        if(!matcherId.is_string())
        {
            fail("key 'matcher_ids' in " + where + " must be an array of UUID strings");
        }
        try
        {
            pack.matcherIds.push_back(
                hipdnn_flatbuffers_sdk::utilities::parseUuid(matcherId.get<std::string>()));
        }
        catch(const std::invalid_argument& error)
        {
            fail("key 'matcher_ids' in " + where + " holds a value that is not a UUID: "
                 + error.what());
        }
    }

    const auto& kernels = requireKey(root, "kernels", where);
    if(!kernels.is_array())
    {
        fail("key 'kernels' in " + where + " must be an array");
    }
    for(const auto& kernelJson : kernels)
    {
        pack.kernels.push_back(parseKernelDescriptor(kernelJson));
    }
    return pack;
}

/// Inserts a freshly parsed descriptor, resolving a repeated id against what is already
/// held: identical content is a duplicate shard and is dropped, differing content
/// poisons the entry so neither definition is used.
template <typename T>
inline void insertCatalogEntry(DescriptorMap<T>& map,
                               T descriptor,
                               const nlohmann::json& source,
                               const std::filesystem::path& path)
{
    const DescriptorId id = descriptor.id;
    const std::string name = descriptor.name;

    auto [it, inserted]
        = map.try_emplace(id, CatalogEntry<T>{std::move(descriptor), source, path, false});
    if(inserted)
    {
        HIPDNN_PLUGIN_LOG_INFO("descriptor loader: loaded " << path << " id=" << toString(id)
                                                            << " name='" << name << "'");
        return;
    }
    // Byte-identical content under one id is the same descriptor reached twice by the
    // walk -- a per-arch layout shipping one shared UED is the case -- so it collapses to
    // one rather than tripping the drop-all rule. RFC 0020 §10.2.1 says drop every UED in
    // an id collision, but its reason is that keep-the-first leaves which definition won
    // up to load order; with identical bytes there is no second definition to choose
    // between. Differing content under one id is a real collision and drops both, below.
    if(it->second.source == source)
    {
        HIPDNN_PLUGIN_LOG_INFO("descriptor loader: duplicate identical descriptor "
                               << path << " id=" << toString(id) << " name='" << name
                               << "', already loaded from " << it->second.path << "; skipping");
        return;
    }
    HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: " << path << " and " << it->second.path
                                                  << " both define id=" << toString(id)
                                                  << " name='" << name
                                                  << "' with different contents; ignoring both");
    // Never cleared by a later file: once two files disagree about what an id means,
    // no third file can decide which of them was right.
    it->second.conflicted = true;
}

/// The descriptor a cross-reference names, or nullptr when it is missing or conflicted.
template <typename T>
inline const T* findDescriptor(const DescriptorMap<T>& map, const DescriptorId& id)
{
    const auto it = map.find(id);
    if(it == map.end() || it->second.conflicted)
    {
        return nullptr;
    }
    return &it->second.descriptor;
}

/// Checks and completes one kernel's metadata against its engine's KMD, mirroring the
/// rules KernelIngestorStateManager::completeMetadata enforces so a violation drops one
/// pack here rather than throwing out of the state manager and taking the whole engine.
inline bool coerceKernelMetadata(KernelDescriptor& kernel,
                                 const MetadataSchema& schema,
                                 std::string& error)
{
    for(const auto& field : schema.fields)
    {
        const auto it = kernel.metadata.find(field.name);
        if(it == kernel.metadata.end())
        {
            if(!field.defaultValue.has_value())
            {
                error = "kernel '" + kernel.name + "' omits metadata field '" + field.name
                        + "', which declares no default";
                return false;
            }
            continue;
        }
        if(!coerceToDeclaredType(it->second, field.type))
        {
            error = "kernel '" + kernel.name + "' supplies metadata field '" + field.name
                    + "' with a value of the wrong type";
            return false;
        }
    }

    for(const auto& entry : kernel.metadata)
    {
        const std::string& name = entry.first;
        const auto declared = std::find_if(schema.fields.begin(),
                                           schema.fields.end(),
                                           [&name](const MetadataField& field) {
                                               return field.name == name;
                                           });
        if(declared == schema.fields.end())
        {
            error = "kernel '" + kernel.name + "' supplies metadata field '" + name
                    + "', which schema '" + schema.name + "' does not declare";
            return false;
        }
    }
    return true;
}

/// RFC 0020 §12: an engine named in HIPDNN_DISABLE_ENGINES is skipped before registration,
/// so it never loads and never claims its name or id. An entry may be the UED `name`, the
/// 64-bit engine id that name hashes to (decimal or `0x` hex), or the UED's own UUID; an
/// entry matching nothing is ignored, because one list is expected to span providers.
///
/// Read per load rather than cached: loads are rare, and a cached first read would make
/// the variable order-dependent inside a shared test binary.
inline bool isEngineDisabled(const EngineDescriptor& engine)
{
    const auto list = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DISABLE_ENGINES", "");
    if(list.empty())
    {
        return false;
    }

    const auto equalsIgnoringCase = [](std::string_view lhs, std::string_view rhs) {
        return lhs.size() == rhs.size()
               && std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](char a, char b) {
                      return std::tolower(static_cast<unsigned char>(a))
                             == std::tolower(static_cast<unsigned char>(b));
                  });
    };

    const int64_t engineId = hipdnn_data_sdk::utilities::engineNameToId(engine.name);
    const auto hex = hipdnn_data_sdk::utilities::formatEngineIdHex(engineId);
    const auto decimal = std::to_string(engineId);
    const auto uuid = toString(engine.id);

    for(size_t begin = 0; begin <= list.size();)
    {
        const auto comma = std::min(list.find(',', begin), list.size());
        auto entry = std::string_view{list}.substr(begin, comma - begin);
        begin = comma + 1;

        while(!entry.empty() && std::isspace(static_cast<unsigned char>(entry.front())) != 0)
        {
            entry.remove_prefix(1);
        }
        while(!entry.empty() && std::isspace(static_cast<unsigned char>(entry.back())) != 0)
        {
            entry.remove_suffix(1);
        }
        if(entry.empty())
        {
            continue;
        }

        if(entry == engine.name || entry == decimal || equalsIgnoringCase(entry, hex)
           || equalsIgnoringCase(entry, uuid))
        {
            HIPDNN_PLUGIN_LOG_INFO("descriptor loader: engine '"
                                   << engine.name << "' id=" << hex
                                   << " is disabled by HIPDNN_DISABLE_ENGINES; skipping it");
            return true;
        }
    }
    return false;
}

} // namespace detail

/**
 * @brief Every descriptor file under @p root, parsed and keyed by (type, id).
 *
 * Walks the tree in sorted path order and takes every `*.json` file. Never throws: a file
 * that fails to open, fails to parse, declares no recognised `schema`, or violates the
 * authored format is logged at ERROR with its path and the reason, and skipped.
 */
inline DescriptorCatalog loadDescriptorCatalog(const std::filesystem::path& root)
{
    DescriptorCatalog catalog;

    std::error_code error;
    if(!std::filesystem::is_directory(root, error))
    {
        HIPDNN_PLUGIN_LOG_INFO("descriptor loader: no descriptor directory at "
                               << root << "; no descriptor-backed engines loaded");
        return catalog;
    }

    // Every architecture directory under the root is unioned, and nothing prunes a
    // foreign-architecture pack: KernelDescriptorPack carries no arch field, and the
    // calling device is unknown at load time because IDeviceResolver needs a handle while
    // the provider's container is built before one exists. Matchers are the only thing
    // standing between a gfx950 pack and a gfx942 device today. The trigger to revisit is
    // the first pair of arch directories shipping *different* packs, at which point arch
    // belongs on the pack and pruning belongs at match time.
    //
    // Iterated with the error_code increment rather than a range-for, whose operator++ is
    // the throwing overload: an unreadable subdirectory under the root would otherwise
    // throw filesystem_error out of a loader that promises never to throw.
    std::vector<std::filesystem::path> files;
    auto walk = std::filesystem::recursive_directory_iterator(root, error);
    if(error)
    {
        HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: cannot read " << root << ": "
                                                                  << error.message());
    }
    for(; walk != std::filesystem::recursive_directory_iterator(); walk.increment(error))
    {
        if(error)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: stopped walking " << root << ": "
                                                                          << error.message());
            break;
        }
        std::error_code entryError;
        if(walk->is_regular_file(entryError) && walk->path().extension() == ".json")
        {
            files.push_back(walk->path());
        }
        else if(entryError)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: skipping " << walk->path() << ": "
                                                                   << entryError.message());
        }
    }
    // Sorted before parsing so which file of a conflicting pair is reported as the
    // incumbent, and the order the load lines appear in, never depend on the filesystem.
    std::sort(files.begin(), files.end());

    for(const auto& path : files)
    {
        nlohmann::json document;
        try
        {
            std::ifstream file(path, std::ios::binary);
            if(!file.is_open())
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: failed to open " << path);
                continue;
            }
            document = nlohmann::json::parse(file);
        }
        catch(const std::exception& parseError)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: failed to parse " << path << ": "
                                                                          << parseError.what());
            continue;
        }

        try
        {
            detail::requireObject(document, "the document root");
            const auto schema = detail::requireString(document, "schema", "the document root");
            if(schema == detail::SCHEMA_KMD)
            {
                detail::insertCatalogEntry(
                    catalog.schemas, detail::parseMetadataSchema(document), document, path);
            }
            else if(schema == detail::SCHEMA_UHD)
            {
                detail::insertCatalogEntry(
                    catalog.heuristics, detail::parseHeuristicDescriptor(document), document, path);
            }
            else if(schema == detail::SCHEMA_UED)
            {
                detail::insertCatalogEntry(
                    catalog.engines, detail::parseEngineDescriptor(document), document, path);
            }
            else if(schema == detail::SCHEMA_UMD)
            {
                detail::insertCatalogEntry(
                    catalog.matchers, detail::parseMatchDescriptor(document), document, path);
            }
            else if(schema == detail::SCHEMA_UDD)
            {
                detail::insertCatalogEntry(
                    catalog.dispatches, detail::parseDispatchDescriptor(document), document, path);
            }
            else if(schema == detail::SCHEMA_KDP)
            {
                detail::insertCatalogEntry(
                    catalog.packs, detail::parseKernelDescriptorPack(document), document, path);
            }
            else
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: " << path << " declares unknown schema '"
                                                              << schema << "'; skipping");
            }
        }
        catch(const std::exception& formatError)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: " << path << " is not a valid descriptor: "
                                                          << formatError.what());
        }
    }

    return catalog;
}

/**
 * @brief Groups @p catalog into one DescriptorSet per engine whose references all resolve.
 *
 * Engines are walked in ascending id order, and each set's matchers, dispatches and packs
 * are deduplicated and sorted by id, so a DescriptorSet is a deterministic function of the
 * file contents rather than of hash-map or filesystem order. Container::copyEngineIds and
 * the container constructor both walk the resulting vector and must agree index for index.
 */
inline std::vector<DescriptorSet> resolveDescriptorSets(const DescriptorCatalog& catalog)
{
    std::vector<const CatalogEntry<EngineDescriptor>*> engineEntries;
    engineEntries.reserve(catalog.engines.size());
    for(const auto& [id, entry] : catalog.engines)
    {
        engineEntries.push_back(&entry);
    }
    // DescriptorId is std::array<uint8_t, 16>, so its operator< is the byte-lexicographic
    // order toString() would render -- same ordering, without formatting a UUID per
    // comparison.
    std::sort(engineEntries.begin(),
              engineEntries.end(),
              [](const auto* lhs, const auto* rhs) {
                  return lhs->descriptor.id < rhs->descriptor.id;
              });

    // RFC 0020 §12: disabled engines leave before anything claims a name, which is what
    // makes the variable the recovery lever for the collision rule below -- disabling one
    // of two same-name UEDs lets the other load.
    engineEntries.erase(std::remove_if(engineEntries.begin(),
                                       engineEntries.end(),
                                       [](const auto* entry) {
                                           return detail::isEngineDisabled(entry->descriptor);
                                       }),
                        engineEntries.end());

    // RFC 0020 §10.2.1: every UED in a name collision is dropped, not just the ones after
    // the first. Load order is filesystem order, so keep-the-first would leave which
    // definition won up to the directory walk. Keyed by hash rather than by the name
    // itself because the hash is what the engine-id space collides in.
    std::map<int64_t, int> nameClaims;
    for(const auto* engineEntry : engineEntries)
    {
        ++nameClaims[hipdnn_data_sdk::utilities::engineNameToId(engineEntry->descriptor.name)];
    }

    std::vector<DescriptorSet> sets;

    for(const auto* engineEntry : engineEntries)
    {
        const auto& engine = engineEntry->descriptor;
        if(engineEntry->conflicted)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << engine.name << "' id=" << toString(engine.id)
                                    << " is defined by conflicting files; dropping it");
            continue;
        }
        if(nameClaims[hipdnn_data_sdk::utilities::engineNameToId(engine.name)] > 1)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine name '"
                                    << engine.name << "' is claimed by more than one UED; "
                                    << "dropping every one of them, including id="
                                    << toString(engine.id)
                                    << ". Disable all but one with HIPDNN_DISABLE_ENGINES");
            continue;
        }

        const auto* heuristic = detail::findDescriptor(catalog.heuristics, engine.heuristicId);
        if(heuristic == nullptr)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << engine.name << "' names heuristic "
                                    << toString(engine.heuristicId)
                                    << ", which no descriptor defines; dropping it");
            continue;
        }
        const auto* schema = detail::findDescriptor(catalog.schemas, engine.metadataSchemaId);
        if(schema == nullptr)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << engine.name << "' names metadata schema "
                                    << toString(engine.metadataSchemaId)
                                    << ", which no descriptor defines; dropping it");
            continue;
        }
        if(const auto* undeclared = findUndeclaredKnob(engine, schema->fields))
        {
            // Rejected here rather than left to GenericEngine's constructor, which throws
            // on it: by then copyEngineIds has already advertised the id, so the throw
            // takes the whole provider down instead of costing one engine.
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << engine.name << "' exposes knob '" << *undeclared
                                    << "', which metadata schema '" << schema->name
                                    << "' does not declare; dropping it");
            continue;
        }

        std::vector<const KernelDescriptorPack*> packEntries;
        for(const auto& [id, entry] : catalog.packs)
        {
            if(!entry.conflicted && entry.descriptor.engineId == engine.id)
            {
                packEntries.push_back(&entry.descriptor);
            }
        }
        std::sort(packEntries.begin(), packEntries.end(), [](const auto* lhs, const auto* rhs) {
            return lhs->id < rhs->id;
        });

        DescriptorSet set;
        set.engine = engine;
        set.schema = *schema;
        set.heuristic = *heuristic;

        // Keyed by id: deduplicates descriptors two packs share and orders them in one
        // step.
        std::map<DescriptorId, MatchDescriptor> matchers;
        std::map<DescriptorId, DispatchDescriptor> dispatches;

        for(const auto* packEntry : packEntries)
        {
            // Failure granularity is the pack: a pack whose cross-references dangle or
            // whose kernels contradict the KMD is dropped while the engine keeps its other
            // packs. A duplicate kernel metadata tuple is the exception -- the state
            // manager's constructor throws on it, taking the whole engine. RFC 0017 §10
            // wants only the colliding kernel dropped; the upgrade is making that
            // constructor log and drop rather than throw, in one place, so hand-built packs
            // get the same behavior.
            KernelDescriptorPack pack = *packEntry;
            std::vector<const MatchDescriptor*> packMatchers;
            std::string reason;

            for(const auto& matcherId : pack.matcherIds)
            {
                const auto* matcher = detail::findDescriptor(catalog.matchers, matcherId);
                if(matcher == nullptr)
                {
                    reason = "names matcher " + toString(matcherId)
                             + ", which no descriptor defines";
                    break;
                }
                packMatchers.push_back(matcher);
            }

            const auto* dispatch = reason.empty()
                                       ? detail::findDescriptor(catalog.dispatches, pack.dispatchId)
                                       : nullptr;
            if(reason.empty() && dispatch == nullptr)
            {
                reason = "names dispatch descriptor " + toString(pack.dispatchId)
                         + ", which no descriptor defines";
            }

            if(reason.empty())
            {
                for(auto& kernel : pack.kernels)
                {
                    if(!detail::coerceKernelMetadata(kernel, set.schema, reason))
                    {
                        break;
                    }
                }
            }

            if(reason.empty() && pack.kernels.empty())
            {
                reason = "declares no kernels";
            }

            if(!reason.empty())
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: pack '"
                                        << pack.name << "' id=" << toString(pack.id) << " "
                                        << reason << "; dropping the pack");
                continue;
            }

            for(const auto* matcher : packMatchers)
            {
                matchers.emplace(matcher->id, *matcher);
            }
            dispatches.emplace(dispatch->id, *dispatch);
            set.packs.push_back(std::move(pack));
        }

        if(set.packs.empty())
        {
            // An engine with no kernels can never match, so advertising it is noise.
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << engine.name
                                    << "' has no loadable kernel pack; dropping it");
            continue;
        }

        for(auto& [key, matcher] : matchers)
        {
            set.matchers.push_back(std::move(matcher));
        }
        for(auto& [key, dispatch] : dispatches)
        {
            set.dispatches.push_back(std::move(dispatch));
        }

        sets.push_back(std::move(set));
    }

    return sets;
}

/// @brief Builds the state manager one DescriptorSet's engine selects over.
///
/// @p set's UED is ignored: a UED is 1:1 with a hipDNN engine and is owned by the engine,
/// not by the state manager.
template <typename THandle>
inline std::unique_ptr<KernelIngestorStateManager<THandle>> makeStateManager(DescriptorSet set)
{
    return std::make_unique<KernelIngestorStateManager<THandle>>(std::move(set.schema),
                                                                 std::move(set.matchers),
                                                                 std::move(set.dispatches),
                                                                 std::move(set.packs),
                                                                 makeKernelHeuristic(set.heuristic));
}

/// @brief Builds the engine one DescriptorSet describes.
///
/// @p deviceResolver is held by reference by the engine, so the provider must keep it
/// alive for the engine's lifetime.
template <typename THandle, typename TSettings, typename TContext>
inline std::unique_ptr<IEngine<THandle, TSettings, TContext>>
    makeDescriptorEngine(DescriptorSet set, const IDeviceResolver<THandle>& deviceResolver)
{
    // Moves the UED out of `set` in its own statement, fully sequenced before the move of
    // the (now engine-less) remainder below -- not two moves racing inside one call's
    // argument list. makeStateManager() never reads set.engine, so its moved-from state
    // here is inert.
    auto engine = std::move(set.engine);
    return std::make_unique<GenericEngine<THandle, TSettings, TContext>>(
        std::move(engine), makeStateManager<THandle>(std::move(set)), deviceResolver);
}

namespace detail
{

/// The engine names this loader has registered, held in a deque because EngineRegistrar
/// stores a string_view into whatever it was handed: the referenced storage must outlive
/// the process-wide map, and a deque never relocates the elements already in it. Also what
/// tells a name this loader registered on an earlier call apart from a foreign engine's.
inline std::deque<std::string>& registeredEngineNames()
{
    static std::deque<std::string> s_names;
    return s_names;
}

} // namespace detail

/**
 * @brief Every descriptor set under @p root that this provider can actually construct.
 *
 * The provider-facing entry point, and the only place validation happens. A set survives
 * only if every native symbol it names is registered, its name claims an engine id no
 * already-registered engine holds, and a state manager built from it constructs without
 * throwing, so an engine this returns is one the provider can advertise and then serve.
 *
 * @warning Native symbols must already be registered when this is called; a set naming an
 *          unregistered symbol is dropped.
 */
template <typename THandle>
inline std::vector<DescriptorSet> loadValidatedDescriptorSets(const std::filesystem::path& root)
{
    std::vector<DescriptorSet> validated;

    for(auto& set : resolveDescriptorSets(loadDescriptorCatalog(root)))
    {
        // Symbols are checked here rather than inside KernelIngestorStateManager's
        // constructor, where the other cross-reference validation lives. Moving them there
        // would cover hand-built packs too, but that constructor is handed an
        // already-built heuristic rather than the UHD, so the score symbol would still
        // have to be checked out here. Checking at load is what the ticket requires:
        // getDispatchDetails() throws only *after* applicability has told hipDNN the engine
        // will serve the graph, which is far too late to skip a descriptor.
        bool resolvable = true;
        for(const auto& matcher : set.matchers)
        {
            const bool registered = matcher.scope == MatchScope::GRAPH
                                        ? GraphMatcherRegistry::isRegistered(matcher.matchSymbol)
                                        : KernelMatcherRegistry::isRegistered(matcher.matchSymbol);
            if(!registered)
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                        << set.engine.name << "' names unregistered match symbol '"
                                        << matcher.matchSymbol << "'; dropping it");
                resolvable = false;
            }
        }
        for(const auto& dispatch : set.dispatches)
        {
            if(!DispatchRegistry<THandle>::isRegistered(dispatch.dispatchSymbol))
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                        << set.engine.name
                                        << "' names unregistered dispatch symbol '"
                                        << dispatch.dispatchSymbol << "'; dropping it");
                resolvable = false;
            }
        }
        if(set.heuristic.kind == HeuristicKind::NATIVE
           && !ScoreRegistry::isRegistered(set.heuristic.payload))
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << set.engine.name << "' names unregistered score symbol '"
                                    << set.heuristic.payload << "'; dropping it");
            resolvable = false;
        }

        // A name hashing onto an engine someone else already registered is dropped:
        // EngineManager::addEngine emplace-drops the loser silently while
        // Container::copyEngineIds still advertises its id. Skipped for a name this loader
        // itself registered, because reloading the same directory must be idempotent.
        const auto& registered = hipdnn_data_sdk::utilities::getEngineIdToNameMap();
        const auto claimed
            = registered.find(hipdnn_data_sdk::utilities::engineNameToId(set.engine.name));
        const auto& ours = detail::registeredEngineNames();
        if(claimed != registered.end()
           && std::find(ours.begin(), ours.end(), set.engine.name) == ours.end())
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << set.engine.name << "' collides with already-registered '"
                                    << claimed->second << "' on engine id "
                                    << hipdnn_data_sdk::utilities::formatEngineIdHex(claimed->first)
                                    << "; dropping it");
            resolvable = false;
        }
        if(!resolvable)
        {
            continue;
        }

        try
        {
            // Built only to prove the set validates, then thrown away: the state manager's
            // constructor is where duplicate-metadata-tuple and cross-reference validation
            // lives, and Container::copyEngineIds is static -- it advertises ids with no
            // container in existence, so an engine that parses but fails to construct would
            // make the advertised count exceed the constructed one. Extracting
            // validateAndIndexPacks() into a predicate both this and that constructor call
            // would remove this discarded second walk.
            auto probe = makeStateManager<THandle>(set);
            static_cast<void>(probe);
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << set.engine.name << "' does not validate: " << error.what()
                                    << "; dropping it");
            continue;
        }

        // Best-effort, and a throw here is logged and ignored: registration only improves
        // plugin-side diagnostics, because the registry is process-local to a plugin built
        // with hidden visibility and --exclude-libs=ALL, so the backend's copy never sees it
        // and hipdnn_list_engines still renders these engines as hex (AICK-1901). Real name
        // collisions are already rejected above, so a throw here means the name was
        // registered by an earlier call and is not a reason to drop a working engine.
        auto& registeredNames = detail::registeredEngineNames();
        try
        {
            registeredNames.push_back(set.engine.name);
            const hipdnn_data_sdk::utilities::EngineRegistrar registrar{registeredNames.back()};
            static_cast<void>(registrar);
        }
        catch(const std::exception& error)
        {
            registeredNames.pop_back();
            HIPDNN_PLUGIN_LOG_INFO("descriptor loader: engine name '"
                                   << set.engine.name
                                   << "' was not registered for diagnostics: " << error.what());
        }

        validated.push_back(std::move(set));
    }

    HIPDNN_PLUGIN_LOG_INFO("descriptor loader: " << validated.size()
                                                 << " descriptor-backed engine(s) loaded from "
                                                 << root);
    return validated;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
