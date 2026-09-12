// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>

namespace hipdnn_plugin_sdk::uhd
{
namespace parser_detail
{
inline constexpr size_t MAX_DOCUMENT_BYTES = 8 * 1024 * 1024;
inline constexpr size_t MAX_DOCUMENT_NODES = 131072;
inline constexpr size_t MAX_DOCUMENT_DEPTH = 2 * expression::Program::MAX_EXPRESSION_DEPTH + 8;

[[noreturn]] inline void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

inline void object(const nlohmann::json& value, const std::string& where)
{
    if(!value.is_object())
    {
        fail(where + " must be a JSON object");
    }
}

inline void keys(const nlohmann::json& value,
                 std::initializer_list<std::string_view> allowed,
                 const std::string& where)
{
    object(value, where);
    for(const auto& item : value.items())
    {
        if(std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end()
           && item.key().rfind("x-", 0) != 0 && item.key().rfind('_', 0) != 0
           && item.key() != "provenance")
        {
            fail("unknown key '" + item.key() + "' in " + where);
        }
    }
}

inline const nlohmann::json&
    required(const nlohmann::json& value, const std::string& key, const std::string& where)
{
    const auto found = value.find(key);
    if(found == value.end())
    {
        fail("missing required key '" + key + "' in " + where);
    }
    return *found;
}

inline std::string
    text(const nlohmann::json& value, const std::string& key, const std::string& where)
{
    const auto& entry = required(value, key, where);
    if(!entry.is_string() || entry.get_ref<const std::string&>().empty())
    {
        fail("key '" + key + "' must be a nonempty string in " + where);
    }
    return entry.get<std::string>();
}

inline void bounds(const nlohmann::json& value, size_t depth, size_t& count, size_t& bytes)
{
    if(depth > MAX_DOCUMENT_DEPTH || ++count > MAX_DOCUMENT_NODES)
    {
        fail("UHD document exceeds depth or node bound");
    }
    if(value.is_string())
    {
        bytes += value.get_ref<const std::string&>().size();
    }
    if(value.is_object())
    {
        for(const auto& item : value.items())
        {
            bytes += item.key().size();
            bounds(item.value(), depth + 1, count, bytes);
        }
    }
    else if(value.is_array())
    {
        for(const auto& item : value)
        {
            bounds(item, depth + 1, count, bytes);
        }
    }
    if(bytes > MAX_DOCUMENT_BYTES)
    {
        fail("UHD document exceeds input-size bound");
    }
}

inline void revision(const std::string& value, const std::string& where)
{
    const auto dot = value.find('.');
    const auto digits = [](std::string_view part) {
        return !part.empty() && part.size() <= 9
               && std::all_of(
                   part.begin(), part.end(), [](unsigned char c) { return c >= '0' && c <= '9'; });
    };
    if(dot == std::string::npos || !digits(std::string_view(value).substr(0, dot))
       || !digits(std::string_view(value).substr(dot + 1)))
    {
        fail("revision must be numeric major.minor in " + where);
    }
}

inline void dependency(const nlohmann::json& value, const std::string& where)
{
    keys(value, {"id", "revision"}, where);
    (void)hipdnn_flatbuffers_sdk::utilities::parseUuid(text(value, "id", where));
    revision(text(value, "revision", where), where);
}

/// RFC 0019 §4.1: a UHD names what it was generated against, and only that. Two things
/// can be named, and a model names whichever one applies to the engine that will bind it:
///
///   - the descriptor set -- `ued`/`kmd`/`umd`, all three or none -- for a model a UED
///     role map binds. The loader's UUID and major/minor rule (§8.1) checks it.
///   - `selector_revision`, the provider build whose behaviour was actually measured, for
///     a model an engine with no UED binds by declared UUID (Open Question 7, RESOLVED).
///     That engine has no UED, KMD or UMD to name, and its behaviour is decided by the
///     vendor library it wraps, so this is the only thing there is to be trained against.
///     The loader refuses a model whose recorded revision is not the one the provider
///     reports: L1 is the one score compared ACROSS engines, so a stale estimate does not
///     merely misreport a number, it changes which engine is selected.
///
/// Neither names an engine. A UHD still cannot say what it attaches to -- the binding is
/// the UED role map or the provider-declared UUID, both of which live in compiled code.
inline void provenance(const nlohmann::json& value, const std::string& where)
{
    keys(value, {"ued", "kmd", "umd", "selector_revision"}, where);
    const bool namesDescriptorSet
        = value.contains("ued") || value.contains("kmd") || value.contains("umd");
    const bool namesSelector = value.contains("selector_revision");
    if(!namesDescriptorSet && !namesSelector)
    {
        fail("trained_against must name a descriptor set or a selector_revision in " + where);
    }
    if(namesSelector)
    {
        // text() rejects a non-string and an empty one; the value itself is opaque here,
        // since only the provider that produced it can say what it means.
        (void)text(value, "selector_revision", where);
    }
    if(!namesDescriptorSet)
    {
        return;
    }
    // All three or none: two thirds of a descriptor set is not a weaker claim, it is an
    // unverifiable one, and required() below is what says which third is missing.
    dependency(required(value, "ued", where), where + " ued");
    dependency(required(value, "kmd", where), where + " kmd");
    const auto& matchers = required(value, "umd", where);
    if(!matchers.is_array())
    {
        fail("trained_against.umd must be an array in " + where);
    }
    std::set<std::string> ids;
    for(const auto& matcher : matchers)
    {
        dependency(matcher, where + " umd");
        auto id = text(matcher, "id", where);
        std::transform(id.begin(), id.end(), id.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if(!ids.insert(id).second)
        {
            fail("duplicate matcher dependency in " + where);
        }
    }
}
} // namespace parser_detail

/// @brief Read a bounded UHD JSON document, rejecting duplicate keys before interpretation.
inline nlohmann::json readUhdDocument(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file)
    {
        parser_detail::fail("cannot read UHD " + path.string());
    }
    const auto length = file.tellg();
    if(length <= 0 || length > static_cast<std::streamoff>(parser_detail::MAX_DOCUMENT_BYTES))
    {
        parser_detail::fail("UHD exceeds input-size bound: " + path.string());
    }
    std::string contents(static_cast<size_t>(length), '\0');
    file.seekg(0);
    if(!file.read(contents.data(), static_cast<std::streamsize>(length)))
    {
        parser_detail::fail("cannot read complete UHD " + path.string());
    }
    std::vector<std::set<std::string>> objects;
    size_t events = 0;
    return nlohmann::json::parse(
        contents, [&](int depth, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
            if(depth > static_cast<int>(parser_detail::MAX_DOCUMENT_DEPTH)
               || ++events > 4 * parser_detail::MAX_DOCUMENT_NODES)
            {
                parser_detail::fail("UHD exceeds depth or node bound: " + path.string());
            }
            if(event == nlohmann::json::parse_event_t::object_start)
            {
                objects.emplace_back();
            }
            else if(event == nlohmann::json::parse_event_t::object_end)
            {
                objects.pop_back();
            }
            else if(event == nlohmann::json::parse_event_t::key
                    && !objects.back().insert(parsed.get<std::string>()).second)
            {
                parser_detail::fail("duplicate UHD key in " + path.string());
            }
            return true;
        });
}

/// @brief Parse the common UHD format independently of any descriptor catalog.
/// @param root Already-decoded document; structural size/depth bounds still apply.
/// @param path Descriptor filename, used to resolve artifact paths absolutely.
/// @throws std::invalid_argument or nlohmann::json::exception for malformed input.
inline UhdConfig parseUhdConfig(const nlohmann::json& root, const std::filesystem::path& path)
{
    using namespace parser_detail;
    const auto where = path.string();
    size_t count = 0;
    size_t bytes = 0;
    bounds(root, 0, count, bytes);
    keys(root,
         {"version",
          "id",
          "name",
          "adapter",
          "features_signature",
          "features_hash",
          "categorical_encoding",
          "objective",
          "score",
          "static_order",
          "native",
          "tree_data",
          "table",
          "onnx",
          "custom_library",
          "trained_against"},
         where);
    if(text(root, "version", where) != "1.0")
    {
        fail("unsupported UHD version in " + where);
    }
    UhdConfig result;
    result.uhdId = text(root, "id", where);
    (void)hipdnn_flatbuffers_sdk::utilities::parseUuid(result.uhdId);
    result.name = text(root, "name", where);
    result.adapterType = text(root, "adapter", where);
    size_t bodies = 0;
    for(const auto* adapter :
        {"static_order", "native", "tree_data", "table", "onnx", "custom_library"})
    {
        bodies += root.contains(adapter) ? size_t{1} : size_t{0};
    }
    if(bodies != 1 || !root.contains(result.adapterType)
       || (result.adapterType != "static_order" && result.adapterType != "native"
           && result.adapterType != "tree_data" && result.adapterType != "table"
           && result.adapterType != "onnx" && result.adapterType != "custom_library"))
    {
        fail("UHD requires exactly one body matching its adapter in " + where);
    }
    if(root.contains("features_signature"))
    {
        const auto& signature = root.at("features_signature");
        if(!signature.is_array() || signature.empty())
        {
            fail("features_signature must be a nonempty array in " + where);
        }
        result.featuresSignature = signature.get<std::vector<nlohmann::json>>();
        for(const auto& entry : result.featuresSignature)
        {
            if(!(entry.is_string() && !entry.get_ref<const std::string&>().empty()
                 && entry.get_ref<const std::string&>().front() == '$')
               && !(entry.is_object() && entry.size() == 1))
            {
                fail("features_signature requires references or inline expressions in " + where);
            }
        }
    }
    if(root.contains("features_hash"))
    {
        result.featuresHash = text(root, "features_hash", where);
        if(result.featuresHash.size() != 23 || result.featuresHash.compare(0, 7, "sha256:") != 0
           || !std::all_of(
               result.featuresHash.begin() + 7, result.featuresHash.end(), [](unsigned char c) {
                   return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
               }))
        {
            fail("features_hash requires sha256: and 16 lowercase hexadecimal digits in " + where);
        }
    }
    if(root.contains("categorical_encoding"))
    {
        const auto& encoding = root.at("categorical_encoding");
        object(encoding, where + " categorical_encoding");
        for(const auto& field : encoding.items())
        {
            object(field.value(), where + " categorical field");
            if(field.key().empty() || field.key().front() != '$' || field.value().empty())
            {
                fail("categorical_encoding requires full references and nonempty vocabularies");
            }
            auto& codes = result.categoricalEncoding[field.key()];
            for(const auto& entry : field.value().items())
            {
                if(!entry.value().is_number_integer()
                   || entry.value() < std::numeric_limits<int32_t>::min()
                   || entry.value() > std::numeric_limits<int32_t>::max())
                {
                    fail("categorical_encoding code must fit int32 in " + where);
                }
                codes[entry.key()] = entry.value().get<int32_t>();
            }
        }
    }
    if(result.adapterType != "static_order" || root.contains("objective"))
    {
        result.objective = text(root, "objective", where);
        if(result.objective != "max" && result.objective != "min")
        {
            fail("UHD objective must be max or min in " + where);
        }
    }
    if(root.contains("score"))
    {
        const auto& score = root.at("score");
        keys(score, {"units", "calibrated", "transform"}, where + " score");
        if(score.contains("units"))
        {
            result.scoreUnits = text(score, "units", where);
        }
        if(score.contains("transform"))
        {
            result.scoreTransform = text(score, "transform", where);
            // RFC 0019 §4 and §11.3: `score.transform` exists so a consumer can invert it and
            // recover `score.units`, which is what makes the number comparable. The vocabulary
            // is therefore closed, and this is where it closes -- an unsupported name reaching
            // applyInverse falls through its identity branch and reports a transformed number
            // as if it were in the declared units: still positive, still ordered, and wrong by
            // whatever the transform was.
            if(!score_transform::isSupported(result.scoreTransform))
            {
                fail("UHD score.transform must be one of "
                     + score_transform::supportedTransformList() + " in " + where);
            }
        }
        if(score.contains("calibrated"))
        {
            if(!score.at("calibrated").is_boolean())
            {
                fail("score.calibrated must be boolean in " + where);
            }
            result.scoreCalibrated = score.at("calibrated").get<bool>();
        }
    }
    if(result.scoreCalibrated && result.objective != "max")
    {
        fail("calibrated UHD score requires objective max in " + where);
    }
    if(root.contains("trained_against"))
    {
        result.trainedAgainst = root.at("trained_against");
        provenance(result.trainedAgainst, where + " trained_against");
    }
    if(!result.featuresSignature.empty()
       && (result.featuresHash.empty() || result.trainedAgainst.is_null()))
    {
        fail("feature-consuming UHD requires features_hash and trained_against in " + where);
    }
    const auto& body = root.at(result.adapterType);
    if(result.adapterType == "static_order")
    {
        keys(body, {"order"}, where);
        if(body.contains("order"))
        {
            result.staticOrderFields = body.at("order").get<std::vector<std::string>>();
            if(result.staticOrderFields.empty())
            {
                result.staticOrderFields = {"priority", "id"};
            }
        }
    }
    else if(result.adapterType == "native")
    {
        keys(body, {"symbol"}, where);
        result.nativeSymbol = text(body, "symbol", where);
    }
    else
    {
        const bool custom = result.adapterType == "custom_library";
        if(custom)
        {
            keys(body, {"library", "hash", "symbol", "config"}, where);
            result.customLibrarySymbol = text(body, "symbol", where);
            if(body.contains("config")
               && (!body.at("config").is_object() || !body.at("config").empty()))
            {
                fail("custom_library configuration is not supported in " + where);
            }
        }
        else
        {
            keys(body, {"artifact", "hash"}, where);
            if(result.featuresSignature.empty())
            {
                fail("model UHD requires features_signature in " + where);
            }
        }
        result.modelArtifactPath
            = std::filesystem::absolute(path.parent_path()
                                        / text(body, custom ? "library" : "artifact", where))
                  .lexically_normal()
                  .string();
        if(body.contains("hash"))
        {
            result.modelHash = text(body, "hash", where);
        }
    }
    return result;
}
} // namespace hipdnn_plugin_sdk::uhd
