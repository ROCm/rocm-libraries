// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
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
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

/**
 * @file DescriptorLoader.hpp
 * @brief Reads descriptor files from disk into the types Descriptors.hpp models.
 *
 * One descriptor per file; the filename suffix is the only thing that types it. The `id`
 * field inside is authoritative for identity, the stem is never parsed:
 *
 * | Filename          | Struct               |
 * |-------------------|----------------------|
 * | `<name>.kmd.json` | MetadataSchema       |
 * | `<name>.uhd.json` | HeuristicDescriptor  |
 * | `<name>.ued.json` | EngineDescriptor     |
 * | `<name>.umd.json` | MatchDescriptor      |
 * | `<name>.udd.json` | DispatchDescriptor   |
 * | `<name>.kdp.json` | KernelDescriptorPack |
 * | `<name>.ukd.json` | KernelDescriptor     |
 *
 * Directories under the root are organizational only -- the walk is recursive and a
 * file's folder means nothing here. A `.json`/`.jsonc` matching no suffix is a WARN and
 * skip, not an error. Every file carries a required `version`, gated per type (FILE_TYPES
 * below) by RFC 0017 §4: accept iff the major matches and the minor is no newer.
 *
 * A KDP names its kernels either inline or by id: an entry of `kernelDescriptors` is an
 * object (the kernel itself) or a bare UUID string naming a `.ukd.json`. The two are
 * equivalent once loaded -- references resolve during set building and append to the same
 * `kernels` vector, so nothing downstream can tell them apart. A file exists so a kernel
 * can ship separately from the pack binding it, or be shared by packs of different
 * engines; two packs of one engine sharing a kernel is not legal, since the duplicate
 * completed metadata tuples collide on the catalog key.
 *
 * The UED follows RFC 0020 (source of truth); the other six follow RFC 0017 §4 until
 * their own follow-ups land. Two deliberate divergences from RFC 0020 §4.2, pending an
 * amendment: no `schema` member -- the filename already carries that fact, and a file
 * whose name and body disagree has no correct reading; and `version` required on every
 * type, not just the UED -- a type with no version can't be gated by §11.1 at all.
 * `sdk_version` sits on the UED rather than the UMD as RFC 0017 §4 has it; see the note at
 * parseEngineDescriptor().
 *
 * Apart from the UED and KDP, whose keys the RFCs fix, every JSON key is the snake_case
 * spelling of its C++ field, and an unrecognized key is a parse error, not a silent
 * no-op -- so a typo is reported rather than ignored. The KDP's `kernelDescriptors` key
 * is camelCase, the RFCs' own inconsistency, kept as-is rather than silently "fixed".
 *
 * Only fields Descriptors.hpp models are parsed; the RFCs describe more (declarative
 * `nodes`/`criteria`, `features_signature`) that arrive with follow-up RFCs. This loader
 * mirrors the structs exactly, so a new field is a change in both places and nowhere else.
 *
 * Nothing here throws to its caller: a malformed file, unresolved cross-reference, or
 * unregistered native symbol is logged at ERROR naming the file, id and name, and
 * skipped, so one bad descriptor never costs a working engine.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// One parsed descriptor plus the provenance duplicate resolution needs.
template <typename T>
struct CatalogEntry
{
    T descriptor;
    /// Kept so a second file claiming the same id can be compared by content: parsed JSON
    /// ignores key order/whitespace, unlike adding operator== to all seven struct types.
    nlohmann::json source;
    std::filesystem::path path; ///< first file that defined this id
    bool conflicted = false; ///< two files disagreed; treat as absent
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
    DescriptorMap<KernelDescriptor> kernels;
};

namespace detail
{

/// A descriptor file's type, taken from the suffix of its filename. The stem is free-form
/// documentation and is never parsed: `pointwise_add.kdp.json` and `a.kdp.json` are read
/// identically.
inline constexpr std::string_view SUFFIX_KMD = ".kmd.json";
inline constexpr std::string_view SUFFIX_UHD = ".uhd.json";
inline constexpr std::string_view SUFFIX_UED = ".ued.json";
inline constexpr std::string_view SUFFIX_UMD = ".umd.json";
inline constexpr std::string_view SUFFIX_UDD = ".udd.json";
inline constexpr std::string_view SUFFIX_KDP = ".kdp.json";
inline constexpr std::string_view SUFFIX_UKD = ".ukd.json";

/// One row per descriptor file type: the suffix that selects it, the `major.minor` this
/// build accepts per RFC 0017 §4 (per type, not a build-wide pair, so one type reaching
/// 1.1 can't widen what the others accept), and the parse-and-insert function. Declared
/// ahead of the parse functions so versionIsSupported() below can take a row by
/// reference; FILE_TYPES itself is assembled further down.
struct FileType
{
    std::string_view suffix;
    int major;
    int minor;
    void (*insert)(DescriptorCatalog&, const nlohmann::json&, const std::filesystem::path&);
};

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

/// A descriptor's declared `major.minor`. Deliberately not `hipdnn_data_sdk::Version`
/// (`major.minor.patch`): RFC 0020 §4.2 spells this field with exactly two components,
/// and the two types gate different things -- this one a file at load, `Version` a graph
/// against an engine at match time.
struct DescriptorVersion
{
    int major = 0;
    int minor = 0;
};

/// Parses `<major>.<minor>` as two separate integers, not a decimal fraction: RFC 0020
/// §11.1 compares them as integers, so `1.10` is newer than `1.9` -- reading the field as
/// a float would order those two backwards.
inline DescriptorVersion parseDescriptorVersion(const std::string& text, const std::string& where)
{
    const std::string_view all{text};
    const auto dot = all.find('.');
    // Nine digits keeps the stoi calls below inside int without a range check of their
    // own; a version component that long is a malformed file, not a real generation.
    const auto isDigits = [](std::string_view part) {
        return !part.empty() && part.size() <= 9
               && std::all_of(
                   part.begin(), part.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    };
    if(dot == std::string_view::npos || !isDigits(all.substr(0, dot))
       || !isDigits(all.substr(dot + 1)))
    {
        fail("key 'version' in " + where + " must be '<major>.<minor>' with numeric halves, not '"
             + text + "'");
    }

    DescriptorVersion version;
    version.major = std::stoi(text.substr(0, dot));
    version.minor = std::stoi(text.substr(dot + 1));
    return version;
}

/// RFC 0017 §4's accept rule, run for every descriptor before its body is parsed and
/// ahead of the catalog insert: RFC 0020 §10.2.1 requires an unsupported-version UED to
/// drop for its version alone and leave the descriptors it would have collided with
/// standing.
inline bool versionIsSupported(const nlohmann::json& document,
                               const FileType& fileType,
                               const std::filesystem::path& path)
{
    const std::string where{fileType.suffix};
    if(document.find("version") == document.end())
    {
        fail("missing required key 'version' in " + where);
    }

    const auto version = parseDescriptorVersion(requireString(document, "version", where), where);
    if(version.major != fileType.major || version.minor > fileType.minor)
    {
        // Warning, not error: a descriptor from a newer toolchain landing beside an older
        // provider is a version skew the operator can act on, not a malformed file.
        HIPDNN_PLUGIN_LOG_WARN("descriptor loader: "
                               << path << " declares " << where << " version " << version.major
                               << "." << version.minor << "; this build reads " << fileType.major
                               << "." << fileType.minor << " and earlier minors; skipping");
        return false;
    }
    return true;
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
    const auto isNameChar
        = [](unsigned char c) { return std::isalnum(c) != 0 || c == '_' || c == '.' || c == '-'; };
    if(colon == std::string::npos || colon == 0 || colon + 1 == name.size()
       || !std::all_of(
           name.begin(), name.end(), [&](unsigned char c) { return c == ':' || isNameChar(c); })
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

/// Deviates from RFC 0017 §4: the RFC's example field also carries `optional` and spells
/// the default `default`, but MetadataField has neither, so a conforming field is
/// rejected as an unknown key until the struct grows one.
inline MetadataSchema parseMetadataSchema(const nlohmann::json& root)
{
    const std::string where{SUFFIX_KMD};
    requireOnlyKeys(root, {"version", "id", "name", "fields"}, where);

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
        field.type
            = metadataTypeFromString(requireString(fieldJson, "type", fieldWhere), fieldWhere);

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
    const std::string where{SUFFIX_UHD};
    requireOnlyKeys(root, {"version", "id", "name", "kind", "payload"}, where);

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

/// A gfx target id in the shape archSupports (DeviceProperties.hpp) will compare: "gfx",
/// a base id, then any number of `:feature` groups. Deliberately no stricter than
/// archMatches in PREFIX mode, which terminates the candidate on ':' or end-of-string --
/// so `gfx942:sramecc+` legitimately matches a device reporting `gfx942:sramecc+:xnack-`,
/// and LLVM generic targets like `gfx9-4-generic` are real gcnArchName values. The check
/// catches an authoring typo, which would otherwise disable the pack on every device with
/// nothing louder than an INFO decline; it is not an existence check, so an unheard-of but
/// well-formed id still parses.
inline bool isPlausibleArchBaseId(std::string_view value)
{
    constexpr std::string_view PREFIX = "gfx";
    if(value.size() <= PREFIX.size() || value.compare(0, PREFIX.size(), PREFIX) != 0)
    {
        return false;
    }

    // The base id, then one group per ':'. Lowercase only: a device reports its arch
    // lowercased and the compare is case-sensitive, so `gfx942:SRAMECC+` is a typo.
    for(std::size_t start = PREFIX.size();;)
    {
        const auto colon = value.find(':', start);
        const auto end = colon == std::string_view::npos ? value.size() : colon;
        auto group = value.substr(start, end - start);

        // A feature carries a trailing '+' or '-' (sramecc+, xnack-); a base id does not.
        if(start > PREFIX.size() && !group.empty() && (group.back() == '+' || group.back() == '-'))
        {
            group.remove_suffix(1);
        }
        if(group.empty() || !std::all_of(group.begin(), group.end(), [](unsigned char c) {
               return (c >= 'a' && c <= 'z') || std::isdigit(c) != 0 || c == '-' || c == '_';
           }))
        {
            return false;
        }

        if(colon == std::string_view::npos)
        {
            return true;
        }
        start = colon + 1;
    }
}

/// `arch`: every entry must be non-empty, non-repeated, and a plausible gfx base id.
/// archSupports is a case-sensitive exact compare, so `""`, `" gfx942"`, or `"gfx94"`
/// would otherwise silently disable the pack everywhere with nothing louder than an
/// INFO decline line to say why. Empty stays legal for the list itself -- that is what
/// "arch-independent" parses as.
inline std::vector<std::string> requireArchList(const nlohmann::json& object,
                                                const std::string& where)
{
    auto values = optionalStringArray(object, "arch", where);
    for(const auto& value : values)
    {
        if(value.empty())
        {
            fail("key 'arch' in " + where + " must not contain an empty string");
        }
        if(!isPlausibleArchBaseId(value))
        {
            std::string message = "key 'arch' in ";
            message += where;
            message += " has '";
            message += value;
            message += "', which is not a bare gfx target id (e.g. 'gfx942')";
            fail(message);
        }
    }
    requireNoDuplicates(values, "arch entry", where);
    return values;
}

inline EngineDescriptor parseEngineDescriptor(const nlohmann::json& root)
{
    const std::string where{SUFFIX_UED};
    // `sdk_version` deviates from RFC 0020 §4.2, whose field table and schema don't list
    // it: RFC 0017 §4 puts the graph schema version on the UMD, but every descriptor
    // under an engine reads tokens that engine's binding produced, so it belongs on the
    // engine instead. Accepted here pending the RFC amendment that moves the field.
    requireOnlyKeys(root,
                    {"version",
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
    // Optional: a UED naming no UHD ranks by declared order, and makeKernelHeuristic()
    // warns and substitutes DeclaredOrderKernelHeuristic. Absence is the only way out --
    // a `heuristic` key present but naming nothing is still a parse error below.
    if(root.find("heuristic") != root.end())
    {
        engine.heuristicId = requireId(root, "heuristic", where);
    }
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
    // the struct's baseline default, so an engine declaring nothing behaves as it did
    // before the field existed. Gating is match-time and belongs to the engine
    // (Descriptors.hpp); the loader only carries the value.
    if(const auto it = root.find("sdk_version"); it != root.end())
    {
        try
        {
            engine.sdkVersion
                = hipdnn_data_sdk::utilities::Version{requireString(root, "sdk_version", where)};
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
    const std::string where{SUFFIX_UMD};
    requireOnlyKeys(root, {"version", "id", "name", "scope", "match_symbol"}, where);

    MatchDescriptor matcher;
    matcher.id = requireId(root, "id", where);
    matcher.name = requireString(root, "name", where);
    matcher.scope = matchScopeFromString(requireString(root, "scope", where), where);
    matcher.matchSymbol = requireString(root, "match_symbol", where);
    return matcher;
}

inline DispatchDescriptor parseDispatchDescriptor(const nlohmann::json& root)
{
    const std::string where{SUFFIX_UDD};
    requireOnlyKeys(root, {"version", "id", "name", "dispatch_symbol"}, where);

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
    const std::string kindText = requireString(root, "kind", where);
    source.kind = kernelSourceKindFromString(kindText, where);
    // Only EMBEDDED_SOURCE has an implementation the dispatch handler can call, and that
    // handler never inspects source.kind -- so accepting another kind here would let
    // applicability advertise a kernel that throws inside getKernelSrc("") at
    // plan-build time instead of failing cleanly at load.
    if(source.kind == KernelSourceKind::EMBEDDED_SOURCE)
    {
        // Not cross-checked against the provider's embedded kernel map: that map is
        // provider-specific with no plugin_sdk-level registry to check against. A
        // typo'd source_file/entry_point reaches getKernelSrc() and throws at
        // plan-build time -- the same late-failure mode the match/dispatch/score
        // pre-flight in loadValidatedDescriptorSets() closes for those three. Not
        // closed here without a provider-populated registry to query.
        source.sourceFile = requireString(root, "source_file", where);
        source.entryPoint = requireString(root, "entry_point", where);
    }
    else
    {
        fail("kernel source kind '" + kindText + "' in " + where
             + " has no implementation yet; only 'embedded_source' can be dispatched");
    }
    return source;
}

/// The fields a kernel carries in either form: inline in a KDP's `kernelDescriptors`, or
/// alone in a `.ukd.json`. The caller owns the allow-list, because the two forms differ by
/// exactly one key -- a file carries `version`, an inline entry has no file to version.
inline KernelDescriptor parseKernelDescriptorBody(const nlohmann::json& root,
                                                  const std::string& entryLabel)
{
    KernelDescriptor kernel;
    kernel.id = requireId(root, "id", entryLabel);
    kernel.name = requireString(root, "name", entryLabel);
    const std::string where = "kernel '" + kernel.name + "'";
    kernel.source
        = parseKernelSource(requireKey(root, "kernel_source", where), where + " kernel_source");

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

/// A kernel spelled inside its pack.
inline KernelDescriptor parseKernelDescriptor(const nlohmann::json& root)
{
    const std::string where = "a 'kernelDescriptors' entry";
    requireObject(root, where);
    requireOnlyKeys(root, {"id", "name", "kernel_source", "metadata", "priority"}, where);
    return parseKernelDescriptorBody(root, where);
}

/// UKD: a kernel shipped in its own file, named from a pack by bare id. Identical to an
/// inline one once loaded; the file exists so a kernel can ship separately from the pack
/// that binds it, or be shared by packs of different engines.
inline KernelDescriptor parseStandaloneKernelDescriptor(const nlohmann::json& root)
{
    const std::string where{SUFFIX_UKD};
    requireOnlyKeys(
        root, {"version", "id", "name", "kernel_source", "metadata", "priority"}, where);
    return parseKernelDescriptorBody(root, where);
}

/// A UUID string from an array-valued key, with the key named in any failure. Shared by
/// `matchers` and the bare-id form of `kernelDescriptors`.
inline DescriptorId
    requireUuidEntry(const nlohmann::json& value, const char* key, const std::string& where)
{
    if(!value.is_string())
    {
        fail("key '" + std::string(key) + "' in " + where + " must be an array of UUID strings");
    }
    try
    {
        return hipdnn_flatbuffers_sdk::utilities::parseUuid(value.get<std::string>());
    }
    catch(const std::invalid_argument& error)
    {
        fail("key '" + std::string(key) + "' in " + where
             + " holds a value that is not a UUID: " + error.what());
    }
}

inline KernelDescriptorPack parseKernelDescriptorPack(const nlohmann::json& root)
{
    const std::string where{SUFFIX_KDP};
    requireOnlyKeys(
        root,
        {"version", "id", "name", "arch", "matchers", "engine", "dispatch", "kernelDescriptors"},
        where);

    KernelDescriptorPack pack;
    pack.id = requireId(root, "id", where);
    pack.name = requireString(root, "name", where);
    pack.engineId = requireId(root, "engine", where);
    pack.dispatchId = requireId(root, "dispatch", where);
    pack.arch = requireArchList(root, where);

    const auto& matcherIds = requireKey(root, "matchers", where);
    if(!matcherIds.is_array())
    {
        fail("key 'matchers' in " + where + " must be an array of UUID strings");
    }
    for(const auto& matcherId : matcherIds)
    {
        pack.matcherIds.push_back(requireUuidEntry(matcherId, "matchers", where));
    }

    const auto& kernels = requireKey(root, "kernelDescriptors", where);
    if(!kernels.is_array())
    {
        fail("key 'kernelDescriptors' in " + where + " must be an array");
    }
    for(const auto& entry : kernels)
    {
        // A string references a standalone `.ukd.json` by id, resolved once the whole tree
        // is read; an object is the kernel itself. Anything else fails as a malformed
        // entry, since only these two spellings name a kernel.
        if(entry.is_string())
        {
            pack.kernelIds.push_back(requireUuidEntry(entry, "kernelDescriptors", where));
        }
        else
        {
            pack.kernels.push_back(parseKernelDescriptor(entry));
        }
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
    // Parsed-JSON equality, not byte equality: whitespace, key order, or int-vs-float
    // spelling never survive into the parsed descriptor either, so treating them as a
    // real collision would fail a duplicate shard over a formatting choice -- e.g. a
    // per-arch layout shipping one shared UED. RFC 0020 §10.2.1's drop-all rule exists
    // because keep-the-first leaves which definition won up to load order; with
    // identical content there is no second definition to choose between.
    if(it->second.source == source)
    {
        HIPDNN_PLUGIN_LOG_INFO("descriptor loader: duplicate identical descriptor "
                               << path << " id=" << toString(id) << " name='" << name
                               << "', already loaded from " << it->second.path << "; skipping");
        return;
    }
    HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: " << path << " and " << it->second.path
                                                  << " both define id=" << toString(id) << " name='"
                                                  << name
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
inline bool
    coerceKernelMetadata(KernelDescriptor& kernel, const MetadataSchema& schema, std::string& error)
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
        const auto declared
            = std::find_if(schema.fields.begin(),
                           schema.fields.end(),
                           [&name](const MetadataField& field) { return field.name == name; });
        if(declared == schema.fields.end())
        {
            error = "kernel '" + kernel.name + "' supplies metadata field '" + name
                    + "', which schema '" + schema.name + "' does not declare";
            return false;
        }
    }
    return true;
}

/// RFC 0020 §12: an engine named in HIPDNN_DISABLE_ENGINES is skipped before
/// registration. An entry may be the UED `name`, the 64-bit engine id it hashes to
/// (decimal or `0x` hex), or the UED's own UUID; an entry matching nothing is ignored,
/// since one list is expected to span providers. Read per load, not cached, since a
/// cached first read would make this order-dependent inside a shared test binary.
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

/// Each row names the suffix it matches and the `major`/`minor` this build accepts for
/// it -- six independent pairs rather than one shared pair, so raising one type's
/// version cannot silently widen what the other five accept.
inline constexpr std::array FILE_TYPES{
    FileType{SUFFIX_KMD,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.schemas, parseMetadataSchema(d), d, p);
             }},
    FileType{SUFFIX_UHD,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.heuristics, parseHeuristicDescriptor(d), d, p);
             }},
    FileType{SUFFIX_UED,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.engines, parseEngineDescriptor(d), d, p);
             }},
    FileType{SUFFIX_UMD,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.matchers, parseMatchDescriptor(d), d, p);
             }},
    FileType{SUFFIX_UDD,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.dispatches, parseDispatchDescriptor(d), d, p);
             }},
    FileType{SUFFIX_KDP,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.packs, parseKernelDescriptorPack(d), d, p);
             }},
    FileType{SUFFIX_UKD,
             1,
             0,
             [](DescriptorCatalog& c, const nlohmann::json& d, const std::filesystem::path& p) {
                 insertCatalogEntry(c.kernels, parseStandaloneKernelDescriptor(d), d, p);
             }},
};
static_assert(FILE_TYPES.size() == 7, "one row per descriptor file type");

/// The row @p filename's suffix selects, or nullptr if it names no descriptor type.
///
/// Requires a non-empty stem, so a bare `.ued.json` is not a descriptor. C++17: no
/// std::string_view::ends_with (the project is set to 17 in projects/hipdnn/CMakeLists.txt).
inline const FileType* findFileType(std::string_view filename)
{
    for(const auto& candidate : FILE_TYPES)
    {
        if(filename.size() > candidate.suffix.size()
           && filename.compare(filename.size() - candidate.suffix.size(),
                               candidate.suffix.size(),
                               candidate.suffix)
                  == 0)
        {
            return &candidate;
        }
    }
    return nullptr;
}

/// @brief Appends every descriptor file under @p root to @p files, sorted by path.
///
/// Sorted per root rather than across all of them, so an earlier root's files stay ahead
/// of a later one's and the incumbent of a conflicting pair never depends on how the two
/// roots happen to be spelled. Never throws.
inline void
    collectDescriptorFiles(const std::filesystem::path& root,
                           std::vector<std::pair<std::filesystem::path, const FileType*>>& files)
{
    // Collected locally so this root's group can be sorted before it joins @p files.
    std::vector<std::pair<std::filesystem::path, const FileType*>> found;
    std::error_code error;
    if(!std::filesystem::is_directory(root, error))
    {
        // INFO, not WARN: a root that does not exist is the normal state of an optional
        // drop-in tree, and only the caller knows whether ending with nothing is a fault.
        HIPDNN_PLUGIN_LOG_INFO("descriptor loader: no descriptor directory at " << root);
        return;
    }

    // `arch` prunes at match time only, not here -- the calling device is unknown at
    // load, and folders are purely organizational (the walk is recursive; a file's
    // directory means nothing to the loader). skip_permission_denied keeps one
    // unreadable subdirectory from turning the whole iterator into end() and silently
    // losing every engine after it; iterated with error_code overloads throughout since
    // this loader promises never to throw.
    auto walk = std::filesystem::recursive_directory_iterator(
        root, std::filesystem::directory_options::skip_permission_denied, error);
    if(error)
    {
        HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: cannot read " << root << ": "
                                                                  << error.message());
    }
    // A failed increment is recovered from rather than ending the walk, so one bad entry
    // does not cost every file sorting after it. Bounded because recovery assumes the
    // increment still advanced: libstdc++ moves past the offending entry, but an
    // implementation that reports an error without advancing would spin here forever.
    // Consecutive failures only -- any successful step resets the budget.
    constexpr int MAX_CONSECUTIVE_WALK_ERRORS = 64;
    int consecutiveErrors = 0;
    for(; walk != std::filesystem::recursive_directory_iterator(); walk.increment(error))
    {
        if(error)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: could not continue walking "
                                    << root << ": " << error.message());
            error.clear();
            if(++consecutiveErrors >= MAX_CONSECUTIVE_WALK_ERRORS)
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: giving up on "
                                        << root << " after " << MAX_CONSECUTIVE_WALK_ERRORS
                                        << " consecutive errors; the walk is not advancing");
                break;
            }
            continue;
        }
        consecutiveErrors = 0;
        try
        {
            std::error_code entryError;
            if(walk->is_regular_file(entryError))
            {
                // filename()/extension() can throw std::system_error on a name not
                // representable in the native encoding (e.g. an unpaired UTF-16
                // surrogate on Windows) -- from a function that promises never to throw,
                // hence the try around this whole entry rather than an error_code
                // overload that does not exist for these two calls.
                const std::string entryName = walk->path().filename().string();
                if(const auto* fileType = detail::findFileType(entryName))
                {
                    found.emplace_back(walk->path(), fileType);
                }
                else
                {
                    // Lowercased so `pointwise.KDP.JSON` still warns instead of vanishing
                    // in silence: findFileType() stays case-sensitive on purpose, only
                    // this "did the author mean a descriptor" check widens. `.jsonc`
                    // warns too -- it is never a loadable extension, so it always lands
                    // here.
                    std::string extension = walk->path().extension().string();
                    std::transform(extension.begin(),
                                   extension.end(),
                                   extension.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if(extension == ".json" || extension == ".jsonc")
                    {
                        // A .json naming no descriptor type is skipped before it is
                        // opened. WARN, not ERROR: an unrelated JSON file under the root
                        // is legitimate, but a misspelled suffix silently costs an
                        // engine, and this is the only place that can say so.
                        HIPDNN_PLUGIN_LOG_WARN(
                            "descriptor loader: "
                            << walk->path()
                            << " is not a descriptor filename (expected "
                               "<name>.{kmd,uhd,ued,umd,udd,kdp,ukd}.json); skipping");
                    }
                }
            }
            else if(entryError)
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: skipping " << walk->path() << ": "
                                                                       << entryError.message());
            }
        }
        catch(const std::exception& filenameError)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: skipping an entry under "
                                    << root << ": " << filenameError.what());
        }
    }
    // Sorted before parsing so which file of a conflicting pair is reported as the
    // incumbent, and the order the load lines appear in, never depend on the filesystem.
    std::sort(found.begin(), found.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    files.insert(files.end(), found.begin(), found.end());
}

} // namespace detail

/**
 * @brief Every descriptor file under @p roots, parsed and keyed by (type, id).
 *
 * Walks each root in turn, taking every file whose name ends in one of the seven type
 * suffixes. The roots feed one catalog, so two files claiming one id resolve by the same
 * rules whether they sit in one root or in two: identical content collapses, and content
 * that disagrees drops both. A later root therefore extends the earlier ones and cannot
 * quietly redefine them.
 *
 * Never throws: a file that fails to open, fails to parse, or violates the authored
 * format is logged at ERROR with its path and the reason, and skipped; a `.json` naming
 * no type is logged at WARN and skipped before it is opened.
 */
inline DescriptorCatalog loadDescriptorCatalog(const std::vector<std::filesystem::path>& roots)
{
    DescriptorCatalog catalog;

    std::vector<std::pair<std::filesystem::path, const detail::FileType*>> files;
    for(const auto& root : roots)
    {
        detail::collectDescriptorFiles(root, files);
    }

    for(const auto& [path, fileType] : files)
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
            // Comments only, no trailing commas: RFC 0020 §4.3's authored form strips
            // `//` and `/* */` before validation, narrower than what "JSONC" commonly
            // implies (VS Code, tsconfig) -- a trailing comma is still a hard nlohmann
            // parse_error.101. Only the parser ever sees the comments --
            // `insertCatalogEntry` compares the parsed documents, so a comment cannot
            // make two copies of one descriptor look like a collision.
            document = nlohmann::json::parse(file,
                                             nullptr,
                                             /*allow_exceptions=*/true,
                                             /*ignore_comments=*/true);
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
            // Version before schema tag: a file from a newer toolchain bumps both, and
            // "unsupported version 2.0" names the actionable mismatch where "must be
            // 'hipdnn.ued/v1'" would blame the wrong field. Both still precede the insert,
            // so RFC 0020 §10.2.1's version-before-duplicate ordering is unaffected.
            if(!detail::versionIsSupported(document, *fileType, path))
            {
                continue;
            }
            fileType->insert(catalog, document, path);
        }
        catch(const std::exception& formatError)
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: " << path << " is not a valid descriptor: "
                                                          << formatError.what());
        }
    }

    return catalog;
}

/// @brief The one-root form: every descriptor file under @p root.
inline DescriptorCatalog loadDescriptorCatalog(const std::filesystem::path& root)
{
    return loadDescriptorCatalog(std::vector<std::filesystem::path>{root});
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
    std::sort(engineEntries.begin(), engineEntries.end(), [](const auto* lhs, const auto* rhs) {
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
    // Conflicted entries are dropped on their own below and must not count toward a
    // name's claim total -- an engine already doomed by disagreeing files would
    // otherwise take a healthy same-named engine down with it via the > 1 rule below.
    std::map<int64_t, int> nameClaims;
    for(const auto* engineEntry : engineEntries)
    {
        if(engineEntry->conflicted)
        {
            continue;
        }
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

        // Only resolved when the UED names one. Naming a UHD no file defines still drops
        // the engine: the author asked for a model that did not ship, which is a broken
        // install rather than a deliberate declared-order ranking.
        const HeuristicDescriptor* heuristic = nullptr;
        if(engine.heuristicId.has_value())
        {
            heuristic = detail::findDescriptor(catalog.heuristics, *engine.heuristicId);
            if(heuristic == nullptr)
            {
                HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                        << engine.name << "' names heuristic "
                                        << toString(*engine.heuristicId)
                                        << ", which no descriptor defines; dropping it");
                continue;
            }
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
        if(heuristic != nullptr)
        {
            set.heuristic = *heuristic;
        }

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
                    reason
                        = "names matcher " + toString(matcherId) + ", which no descriptor defines";
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

            // Referenced kernels join the inline ones before either check below, so a
            // `.ukd.json` is validated against the KMD exactly like a kernel spelled in
            // place, and a pack whose kernels are all references is not "empty". Appended
            // rather than interleaved: rank() orders by (score, priority, id), so position
            // never decides selection.
            if(reason.empty())
            {
                for(const auto& kernelId : pack.kernelIds)
                {
                    const auto* kernel = detail::findDescriptor(catalog.kernels, kernelId);
                    if(kernel == nullptr)
                    {
                        reason = "names kernel " + toString(kernelId)
                                 + ", which no descriptor defines";
                        break;
                    }
                    pack.kernels.push_back(*kernel);
                }
            }

            // One id twice -- referenced twice, or referenced and also inline -- reaches
            // the state manager as two identical kernels, which collide on the completed
            // metadata key and throw, costing the whole engine. Dropping the pack keeps
            // the failure at pack granularity, where every other malformed pack lands.
            if(reason.empty())
            {
                std::vector<DescriptorId> seen;
                seen.reserve(pack.kernels.size());
                for(const auto& kernel : pack.kernels)
                {
                    if(std::find(seen.begin(), seen.end(), kernel.id) != seen.end())
                    {
                        reason = "names kernel " + toString(kernel.id) + " more than once";
                        break;
                    }
                    seen.push_back(kernel.id);
                }
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
                                    << engine.name << "' has no loadable kernel pack; dropping it");
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

    // Packs are only ever reached through the per-engine scan above, so a pack naming an
    // id no UED defines would otherwise vanish with no diagnostic -- the one silent
    // failure in a loader where every other rejection is logged. Diagnostics only; never
    // changes what gets loaded.
    std::vector<const CatalogEntry<KernelDescriptorPack>*> orphans;
    for(const auto& [id, entry] : catalog.packs)
    {
        if(!entry.conflicted
           && catalog.engines.find(entry.descriptor.engineId) == catalog.engines.end())
        {
            orphans.push_back(&entry);
        }
    }
    std::sort(orphans.begin(), orphans.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->descriptor.id < rhs->descriptor.id;
    });
    for(const auto* entry : orphans)
    {
        HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: pack '"
                                << entry->descriptor.name
                                << "' id=" << toString(entry->descriptor.id) << " names engine "
                                << toString(entry->descriptor.engineId)
                                << ", which no descriptor defines; dropping it");
    }

    // Same argument one level down: a standalone kernel is reachable only through a pack's
    // reference list, so one nobody names does nothing and says nothing. WARN, not ERROR --
    // an unreferenced kernel costs no engine, and a typo'd reference already dropped its
    // pack loudly above; this is the other half of that story.
    std::vector<const CatalogEntry<KernelDescriptor>*> unreferenced;
    for(const auto& kernelEntry : catalog.kernels)
    {
        if(kernelEntry.second.conflicted)
        {
            continue;
        }
        const DescriptorId& kernelId = kernelEntry.first;
        const bool named = std::any_of(
            catalog.packs.begin(), catalog.packs.end(), [&kernelId](const auto& packEntry) {
                const auto& ids = packEntry.second.descriptor.kernelIds;
                return !packEntry.second.conflicted
                       && std::find(ids.begin(), ids.end(), kernelId) != ids.end();
            });
        if(!named)
        {
            unreferenced.push_back(&kernelEntry.second);
        }
    }
    std::sort(unreferenced.begin(), unreferenced.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->descriptor.id < rhs->descriptor.id;
    });
    for(const auto* entry : unreferenced)
    {
        HIPDNN_PLUGIN_LOG_WARN("descriptor loader: kernel '"
                               << entry->descriptor.name << "' id="
                               << toString(entry->descriptor.id) << " loaded from " << entry->path
                               << ", but no pack references it; it will never be dispatched");
    }

    return sets;
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
 * @brief Every descriptor set under @p roots that this provider can actually construct.
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
inline std::vector<DescriptorSet>
    loadValidatedDescriptorSets(const std::vector<std::filesystem::path>& roots)
{
    std::vector<DescriptorSet> validated;

    for(auto& set : resolveDescriptorSets(loadDescriptorCatalog(roots)))
    {
        // Checked here rather than inside KernelIngestorStateManager's constructor, where
        // the other cross-reference validation lives, because getDispatchDetails() throws
        // only *after* applicability has told hipDNN the engine will serve the graph --
        // far too late to skip a descriptor.
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
        // Nothing to pre-flight when the engine ships no UHD: declared-order ranking
        // resolves no symbol.
        if(set.heuristic.has_value() && set.heuristic->kind == HeuristicKind::NATIVE
           && !ScoreRegistry::isRegistered(set.heuristic->payload))
        {
            HIPDNN_PLUGIN_LOG_ERROR("descriptor loader: engine '"
                                    << set.engine.name << "' names unregistered score symbol '"
                                    << set.heuristic->payload << "'; dropping it");
            resolvable = false;
        }

        // A name hashing onto an engine someone else already registered is dropped and the
        // incumbent stands (RFC 0020 §10.2.1's drop-all applies to UEDs in a collision, not
        // to a hand-written engine); skipped for a name this loader itself registered, so
        // reloading a directory is idempotent. Rarely reachable by design -- §4.2 requires
        // a scoped `namespace:local` name and every built-in is unscoped, so this needs an
        // FNV-1a collision between the two spellings, not a literal name clash.
        //
        // What this check can't see: getEngineIdToNameMap()'s registry is private to one
        // plugin (hidden visibility, --exclude-libs=ALL), so a name colliding with a
        // DIFFERENT plugin's engine is invisible to this local lookup. That collision is
        // still caught, just elsewhere and more harshly: EnginePluginManager::
        // validateBeforeAdding tracks every id already loaded and rejects a later plugin
        // outright on overlap, so the whole plugin fails to load -- every engine it ships,
        // not just the one that collided. In-process pairs neither check can see (two
        // hand-written engines, or anything registering after load) are caught by
        // EngineManager::addEngine, which logs the duplicate instead of letting the map
        // discard it in silence.
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
            // Built only to prove the set validates, then thrown away: Container::copyEngineIds
            // is static and would otherwise advertise an id for a set that fails to
            // construct. Extracting validateAndIndexPacks() into a shared predicate would
            // remove this discarded second walk, and with it the duplicate warning an
            // engine shipping no heuristic gets: once here, once at real construction.
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

        // Best-effort; a throw here is logged and ignored. Registration only improves
        // plugin-side diagnostics (the registry is process-local, hidden visibility, so
        // hipdnn_list_engines still renders these as hex -- AICK-1901); real name
        // collisions are already rejected above, so a throw here just means an earlier
        // call already registered this name.
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

    std::string from;
    for(const auto& root : roots)
    {
        from += (from.empty() ? "" : ", ") + root.string();
    }
    HIPDNN_PLUGIN_LOG_INFO("descriptor loader: " << validated.size()
                                                 << " descriptor-backed engine(s) loaded from "
                                                 << from);
    return validated;
}

/// @brief The one-root form: every constructible descriptor set under @p root.
template <typename THandle>
inline std::vector<DescriptorSet> loadValidatedDescriptorSets(const std::filesystem::path& root)
{
    return loadValidatedDescriptorSets<THandle>(std::vector<std::filesystem::path>{root});
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
