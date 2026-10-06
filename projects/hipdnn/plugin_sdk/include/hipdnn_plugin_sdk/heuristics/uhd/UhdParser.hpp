// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/heuristics/FeatureSemantics.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>

namespace hipdnn_plugin_sdk::uhd
{
namespace parser_detail
{
inline constexpr size_t MAX_DOCUMENT_BYTES = size_t{8} * 1024 * 1024;
inline constexpr size_t MAX_DOCUMENT_NODES = 131072;
inline constexpr size_t MAX_DOCUMENT_DEPTH = 2 * ExpressionSet::MAX_EXPRESSION_DEPTH + 8;

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
           && item.key().rfind("x-", 0) != 0 && item.key().rfind('_', 0) != 0)
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

/// Whether @p value is a SHA-256 digest as a model body's `hash` spells it: exactly 64
/// lowercase hexadecimal digits, no prefix.
inline bool isSha256Digest(std::string_view value)
{
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

/// Portable descriptor-relative asset names use '/' on every platform. Reject parent
/// segments and native-only spellings before passing the same spelling to filesystem::path.
/// In particular, '\' is a filename character on Linux, not a directory separator.
inline bool isContainedRelativePath(std::string_view value)
{
    if(value.empty() || value.front() == '/' || value.find_first_of("\\:") != std::string_view::npos
       || value.find('\0') != std::string_view::npos)
    {
        return false;
    }
    size_t start = 0;
    while(true)
    {
        const auto end = value.find('/', start);
        const auto segment = value.substr(start, end - start);
        if(segment == "..")
        {
            return false;
        }
        if(end == std::string_view::npos)
        {
            return !segment.empty() && segment != ".";
        }
        start = end + 1;
    }
}

/// Resolve existing symlinks too, including a symlinked ancestor of a missing asset.
/// This is an admission-time containment check, not protection from later filesystem
/// mutation; deployment must keep descriptors and artifacts immutable while in use.
inline std::filesystem::path containedArtifactPath(const std::filesystem::path& descriptor,
                                                   const std::string& relativePath)
{
    const auto directory
        = std::filesystem::weakly_canonical(std::filesystem::absolute(descriptor).parent_path());
    const auto resolved = std::filesystem::weakly_canonical(directory / relativePath);
    const auto relative = resolved.lexically_relative(directory);
    if(relative.empty() || relative == "." || relative.is_absolute() || *relative.begin() == "..")
    {
        fail("model path must remain inside the descriptor's directory after resolving symlinks: "
             + relativePath + " in " + descriptor.string());
    }
    return resolved;
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

/// Validate `trained_against` (RFC 0019 §4.1): a descriptor set (`ued`/`kmd`/`umd`, all
/// three) for UED-bound models, and/or a `selector_revision` for engines without a UED.
/// `feature_semantics_revision` is optional and does not satisfy either form on its own.
inline void provenance(const nlohmann::json& value, const std::string& where)
{
    keys(value, {"ued", "kmd", "umd", "selector_revision", "feature_semantics_revision"}, where);
    if(value.contains("feature_semantics_revision"))
    {
        // Integer only (not 1.0 or a bool), and must fit int64_t. Parsed JSON is unsigned;
        // documents built in memory may be signed.
        const auto& recorded = value.at("feature_semantics_revision");
        const bool valid = recorded.is_number_unsigned()
                               ? recorded.get<uint64_t>() >= 1
                                     && recorded.get<uint64_t>() <= static_cast<uint64_t>(
                                            std::numeric_limits<int64_t>::max())
                               : recorded.is_number_integer() && recorded.get<int64_t>() >= 1;
        if(!valid)
        {
            fail("trained_against.feature_semantics_revision must be an integer >= 1 in " + where);
        }
    }
    const bool namesDescriptorSet
        = value.contains("ued") || value.contains("kmd") || value.contains("umd");
    const bool namesSelector = value.contains("selector_revision");
    if(!namesDescriptorSet && !namesSelector)
    {
        fail("trained_against must name a descriptor set or a selector_revision in " + where);
    }
    if(namesSelector)
    {
        // Opaque: only the provider that produced it can interpret it.
        (void)text(value, "selector_revision", where);
    }
    if(!namesDescriptorSet)
    {
        return;
    }
    // All three or none: a partial descriptor set is unverifiable.
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

/// @brief Why a model cannot read this build's features, or "" when it can.
/// Applies only to models with a @p featuresSignature. A missing revision means 1; any
/// mismatch refuses. Callers report it as UNAVAILABLE. Shared by every binding path.
inline std::string featureSemanticsMismatch(const std::vector<nlohmann::json>& featuresSignature,
                                            const nlohmann::json& trainedAgainst)
{
    if(featuresSignature.empty())
    {
        return {};
    }
    int64_t recorded = 1;
    if(trainedAgainst.is_object())
    {
        if(const auto found = trainedAgainst.find("feature_semantics_revision");
           found != trainedAgainst.end())
        {
            recorded = found->get<int64_t>();
        }
    }
    if(recorded == heuristics::FEATURE_SEMANTICS_REVISION)
    {
        return {};
    }
    return "model was trained against feature semantics revision " + std::to_string(recorded)
           + ", this build computes revision "
           + std::to_string(heuristics::FEATURE_SEMANTICS_REVISION);
}

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

/// @brief Lowercase SHA-256 hex of the artifact at @p path, or "" when it is absent, not a
/// regular file, empty, over 256 MiB, or unreadable.
inline std::string artifactDigest(const std::filesystem::path& path)
{
    constexpr std::uintmax_t MAX_ARTIFACT_BYTES = std::uintmax_t{256} * 1024 * 1024;
    std::error_code error;
    if(!std::filesystem::is_regular_file(path, error))
    {
        return {};
    }
    const auto size = std::filesystem::file_size(path, error);
    if(error || size == 0 || size > MAX_ARTIFACT_BYTES)
    {
        return {};
    }
    std::ifstream file(path, std::ios::binary);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if(!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)))
    {
        return {};
    }
    return sha256(bytes.data(), bytes.size());
}

/// @brief Parse the common UHD format independently of any descriptor catalog.
/// @param root Already-decoded document; structural size/depth bounds still apply.
/// @param path Descriptor filename. A model body's `artifact` or `library` must resolve
///        inside this file's directory and is made absolute against it. A `custom_library`
///        body must declare its `hash`; any other model body declaring none has its artifact
///        read and digested (artifactDigest()).
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
          "trained_against",
          // Free-form authoring notes, never read; root only (RFC 0019 §4.1).
          "provenance"},
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
            if((!entry.is_string() || entry.get_ref<const std::string&>().empty()
                || entry.get_ref<const std::string&>().front() != '$')
               && (!entry.is_object() || entry.size() != 1))
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
        keys(score, {"metric", "calibrated", "transform"}, where + " score");
        if(score.contains("metric"))
        {
            result.scoreMetric = text(score, "metric", where);
            // The metric registry is closed (RFC 0019 §4.4): units and direction come from
            // hipDNN, not the model.
            if(hipdnn_data_sdk::utilities::findRankingMetric(result.scoreMetric) == nullptr)
            {
                fail("UHD score.metric '" + result.scoreMetric
                     + "' is not a registered ranking metric in " + where);
            }
        }
        if(score.contains("transform"))
        {
            result.scoreTransform = text(score, "transform", where);
            // Closed vocabulary (RFC 0019 §11.3): applyInverse treats unknown names as
            // identity, which would silently report transformed values in metric units.
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
    // Calibrated means comparable across engines, which needs a named metric (§4.4).
    if(result.scoreCalibrated && result.scoreMetric.empty())
    {
        fail("calibrated UHD score requires score.metric in " + where);
    }
    // The metric fixes the direction; `objective` must agree with it.
    if(!result.scoreMetric.empty())
    {
        const auto expected = std::string(hipdnn_data_sdk::utilities::objectiveOf(
            *hipdnn_data_sdk::utilities::findRankingMetric(result.scoreMetric)));
        if(result.objective != expected)
        {
            fail("UHD score.metric '" + result.scoreMetric + "' requires objective " + expected
                 + " in " + where);
        }
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
        // static_order ranks by UKD priority then descriptor id; refuse criteria it would
        // silently ignore.
        if(body.contains("order"))
        {
            fail("static_order.order is not supported in " + where
                 + ": declared ordering criteria are not implemented; static_order ranks by "
                   "priority, then descriptor id");
        }
        keys(body, {}, where);
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
        const std::string pathKey = custom ? "library" : "artifact";
        const auto relativePath = text(body, pathKey, where);
        if(!isContainedRelativePath(relativePath))
        {
            fail("key '" + pathKey + "' must be a relative path inside the descriptor's "
                 + "directory, got '" + relativePath + "' in " + where);
        }
        result.modelArtifactPath = containedArtifactPath(path, relativePath).string();
        // Model identity is its content digest (versions the winner cache). A library must
        // declare its expected bytes, but a descriptor-supplied digest is not authentication:
        // whoever can replace both files can replace the digest too. Other artifacts without
        // a declared hash are digested now and verified by the adapter later; empty when absent.
        if(custom || body.contains("hash"))
        {
            result.modelHash = text(body, "hash", where);
            if(!isSha256Digest(result.modelHash))
            {
                fail("key 'hash' must be the SHA-256 of the " + pathKey
                     + " as 64 lowercase hexadecimal digits in " + where);
            }
        }
        else
        {
            result.modelHash = artifactDigest(result.modelArtifactPath);
        }
    }
    return result;
}
} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
