// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/cachekey_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
#include <nlohmann/json.hpp>
#endif

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

/// The graph half of a winner-cache key: content, never identity. Two graphs are equal
/// when a kernel measurement taken on one is valid for the other.
///
/// `hash` narrows the lookup and may be lossy; `logicallyEqual` decides the match. Both
/// are generated from `graph.fbs` into `cachekey_generated.h`, so a new schema field
/// participates automatically, reading the buffer in place with no `UnPack` or
/// allocation.
///
/// Field policy lives in `graph.fbs`: `(cache_ignore)` drops a field, `(cache_uid)`
/// folds a tensor reference as its ordinal in `Graph.tensors`, not its caller-assigned
/// uid -- renumbering keys the same, rewiring an operand does not.
///
/// Tensors and nodes compare in vector order, fixed by `IGraph`'s topological-order
/// precondition: two construction orders of one logical DAG miss rather than mismatch.
class GraphContentKey
{
public:
    /// Shared because a record outlives the plan that produced it.
    using Content = std::shared_ptr<const std::vector<uint8_t>>;

    GraphContentKey() = default;

    /// Retained before `fold()` reads it, so hash and comparison walk the same bytes.
    explicit GraphContentKey(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
        : _content(retain(graph))
        , _hash(fold())
    {
    }

    uint64_t hash() const
    {
        return _hash;
    }

    const Content& content() const
    {
        return _content;
    }

    /// An unkeyable graph matches nothing, including another unkeyable graph: absence
    /// of content is a permanent miss, not a wildcard.
    bool operator==(const GraphContentKey& other) const
    {
        if(_hash != other._hash)
        {
            return false;
        }
        const auto* left = root();
        const auto* right = other.root();
        if(left == nullptr || right == nullptr)
        {
            return false;
        }
        return hipdnn_flatbuffers_sdk::data_objects::cachekey::logicallyEqual(left, right);
    }

    /// False when the graph was invalid or supplied no `bytes()`; callers must not
    /// cache under it.
    bool isUsable() const
    {
        return root() != nullptr;
    }

    bool operator!=(const GraphContentKey& other) const
    {
        return !(*this == other);
    }

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
    /// This key's JSON representation: one field, the retained graph bytes as base64.
    /// An unusable key still serializes, as an empty base64 field; see `fromJson()`.
    nlohmann::json toJson() const
    {
        nlohmann::json json;
        json[CONTENT_FIELD] = _content != nullptr ? encodeBase64(*_content) : std::string{};
        return json;
    }

    /// Parses `toJson()`'s output back into a key, fail-soft: a missing or mistyped
    /// field, invalid base64, or content that fails a `flatbuffers::Verifier` recheck
    /// (base64 that decodes cleanly can still not be a valid `Graph`) all return
    /// `std::nullopt` rather than throwing or handing back a mismatched key.
    ///
    /// The catch is unrestricted because this is `noexcept`: the payload comes from a
    /// cache line with no size bound, so decoding one can throw `std::bad_alloc` or
    /// `std::length_error` as readily as a JSON error, and any of them escaping calls
    /// `std::terminate` in the host process.
    static std::optional<GraphContentKey> fromJson(const nlohmann::json& json) noexcept
    {
        try
        {
            if(!json.is_object())
            {
                return std::nullopt;
            }
            const auto contentField = json.find(CONTENT_FIELD);
            if(contentField == json.end() || !contentField->is_string())
            {
                return std::nullopt;
            }
            auto decoded = decodeBase64(contentField->get<std::string>());
            if(!decoded.has_value())
            {
                return std::nullopt;
            }
            if(decoded->empty())
            {
                return GraphContentKey{};
            }
            flatbuffers::Verifier verifier(decoded->data(), decoded->size());
            if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>())
            {
                return std::nullopt;
            }
            return GraphContentKey{
                std::make_shared<const std::vector<uint8_t>>(std::move(*decoded))};
        }
        catch(...)
        {
            return std::nullopt;
        }
    }
#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

protected:
    /// Test seam: forces a hash collision so a test can reach the structural
    /// comparison, which `operator==` otherwise short-circuits before.
    void forceHash(uint64_t hash)
    {
        _hash = hash;
    }

private:
#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
    static constexpr const char* CONTENT_FIELD = "content_base64";

    /// Takes already-verified bytes; every other caller goes through the `IGraph`
    /// constructor instead.
    explicit GraphContentKey(Content verifiedContent)
        : _content(std::move(verifiedContent))
        , _hash(fold())
    {
    }

    static std::string encodeBase64(const std::vector<uint8_t>& bytes)
    {
        static constexpr std::string_view TABLE
            = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(((bytes.size() + 2) / 3) * 4);
        size_t i = 0;
        for(; i + 2 < bytes.size(); i += 3)
        {
            const uint32_t chunk = (static_cast<uint32_t>(bytes[i]) << 16U)
                                   | (static_cast<uint32_t>(bytes[i + 1]) << 8U)
                                   | static_cast<uint32_t>(bytes[i + 2]);
            out.push_back(TABLE[(chunk >> 18U) & 0x3FU]);
            out.push_back(TABLE[(chunk >> 12U) & 0x3FU]);
            out.push_back(TABLE[(chunk >> 6U) & 0x3FU]);
            out.push_back(TABLE[chunk & 0x3FU]);
        }
        const size_t remaining = bytes.size() - i;
        if(remaining == 1)
        {
            const uint32_t chunk = static_cast<uint32_t>(bytes[i]) << 16U;
            out.push_back(TABLE[(chunk >> 18U) & 0x3FU]);
            out.push_back(TABLE[(chunk >> 12U) & 0x3FU]);
            out.append("==");
        }
        else if(remaining == 2)
        {
            const uint32_t chunk = (static_cast<uint32_t>(bytes[i]) << 16U)
                                   | (static_cast<uint32_t>(bytes[i + 1]) << 8U);
            out.push_back(TABLE[(chunk >> 18U) & 0x3FU]);
            out.push_back(TABLE[(chunk >> 12U) & 0x3FU]);
            out.push_back(TABLE[(chunk >> 6U) & 0x3FU]);
            out.push_back('=');
        }
        return out;
    }

    static int decodeBase64Char(char c)
    {
        if(c >= 'A' && c <= 'Z')
        {
            return c - 'A';
        }
        if(c >= 'a' && c <= 'z')
        {
            return c - 'a' + 26;
        }
        if(c >= '0' && c <= '9')
        {
            return c - '0' + 52;
        }
        if(c == '+')
        {
            return 62;
        }
        if(c == '/')
        {
            return 63;
        }
        return -1;
    }

    /// `std::nullopt` for a wrong length, a non-alphabet character, or a '=' anywhere
    /// but the last one or two positions of the final group. Never throws.
    static std::optional<std::vector<uint8_t>> decodeBase64(const std::string& text)
    {
        if(text.size() % 4 != 0)
        {
            return std::nullopt;
        }
        std::vector<uint8_t> out;
        out.reserve((text.size() / 4) * 3);
        for(size_t i = 0; i < text.size(); i += 4)
        {
            int pad = 0;
            std::array<uint32_t, 4> vals{0, 0, 0, 0};
            for(size_t k = 0; k < 4; ++k)
            {
                const char c = text[i + k];
                if(c == '=')
                {
                    if(i + 4 != text.size() || (k != 2U && k != 3U))
                    {
                        return std::nullopt;
                    }
                    ++pad;
                    continue;
                }
                if(pad > 0)
                {
                    return std::nullopt;
                }
                const int decoded = decodeBase64Char(c);
                if(decoded < 0)
                {
                    return std::nullopt;
                }
                vals[k] = static_cast<uint32_t>(decoded);
            }
            const uint32_t chunk = (vals[0] << 18U) | (vals[1] << 12U) | (vals[2] << 6U) | vals[3];
            out.push_back(static_cast<uint8_t>((chunk >> 16U) & 0xFFU));
            if(pad < 2)
            {
                out.push_back(static_cast<uint8_t>((chunk >> 8U) & 0xFFU));
            }
            if(pad < 1)
            {
                out.push_back(static_cast<uint8_t>(chunk & 0xFFU));
            }
        }
        return out;
    }
#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

    /// Copies the verified buffer: `IGraph` is a view this key does not own.
    static Content retain(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        const auto bytes = graph.bytes();
        if(bytes.data == nullptr || bytes.size == 0)
        {
            return nullptr;
        }
        return std::make_shared<const std::vector<uint8_t>>(bytes.data, bytes.data + bytes.size);
    }

    const hipdnn_flatbuffers_sdk::data_objects::Graph* root() const
    {
        if(_content == nullptr)
        {
            return nullptr;
        }
        // Verified by GraphWrapper before bytes() would hand them over.
        return ::flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            _content->data());
    }

    /// Hash 0 means unkeyable, agreeing with `isUsable()` and `operator==`.
    uint64_t fold() const
    {
        const auto* graph = root();
        if(graph == nullptr)
        {
            return 0;
        }
        hipdnn_flatbuffers_sdk::data_objects::cachekey::Hasher hasher;
        hipdnn_flatbuffers_sdk::data_objects::cachekey::hashAppend(hasher, graph);
        return hasher.value();
    }

    Content _content;
    uint64_t _hash = 0;
};

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities

namespace std
{

template <>
struct hash<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey>
{
    size_t operator()(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey& key) const noexcept
    {
        return static_cast<size_t>(key.hash());
    }
};

} // namespace std
