// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// JsonDataSource.hpp - a sample data source backed by nlohmann::json.
//
// All names below live in namespace hipdnn_plugin_sdk::ingestor::jsonexpr;
// these examples assume `namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;`.
//
// JsonDataSource wraps an nlohmann::json document and satisfies the
// data-source contract (Value getData(const std::string&) const), so a compiled
// jexpr::Expression can evaluate directly against a JSON document:
//
//     jexpr::JsonDataSource src(nlohmann::json{{"q", {{"dims", {8, 16}}}}});
//     auto expr = jexpr::compile<jexpr::JsonDataSource>(rule);
//     jexpr::Value r = expr(src);
//
// Path syntax:
//   - dotted keys:            a.b.c
//   - [N] array subscripts:   arr[0], rows[2].name, grid[0][1]
//   - dot-form array indices: arr.1 (resolves as an index only against an
//                             existing array)
//   - an optional leading variable sigil (VARIABLE_SIGIL) is stripped, so both
//     "q.dims[0]" and "$q.dims[0]" address the same location
//   - the empty path (or a bare variable sigil) names nothing
//
// getData follows the language convention: an unresolved path, including a
// malformed one, reads as null (Value()).
//
// This is a sample accessor. Objects and null in the document both convert to
// a null jexpr::Value, since Value models scalars and arrays only and has no
// object alternative.
//
// Full reference: docs/JsonExpression.md.

#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Syntax.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
class JsonDataSource
{
public:
    JsonDataSource() = default;
    explicit JsonDataSource(nlohmann::json doc)
        : _doc(std::move(doc))
    {
    }

    /// Access the backing document, for inspection or bulk replacement.
    const nlohmann::json& document() const
    {
        return _doc;
    }
    nlohmann::json& document()
    {
        return _doc;
    }

    /// Data-source contract: resolve a variable path to a Value. Any of the
    /// following is unresolved and returns null:
    ///   - a missing key
    ///   - an out-of-range index
    ///   - a non-numeric index
    ///   - a subscript applied to a non-array
    Value getData(const std::string& path) const
    {
        std::vector<Segment> segs;
        if(!tokenize(path, segs))
        {
            return {}; // malformed path -> not found
        }
        const nlohmann::json* cur = &_doc;
        for(const auto& seg : segs)
        {
            if(seg.subscript || cur->is_array())
            {
                if(!cur->is_array())
                {
                    return {}; // subscript on a non-array
                }
                std::size_t idx = 0;
                if(!parseIndex(seg.text, idx) || idx >= cur->size())
                {
                    return {};
                }
                cur = &(*cur)[idx];
            }
            else if(cur->is_object())
            {
                const auto it = cur->find(seg.text);
                if(it == cur->end())
                {
                    return {};
                }
                cur = &*it;
            }
            else
            {
                return {};
            }
        }
        return toValue(*cur);
    }

private:
    struct Segment
    {
        bool subscript; // true for [N] forms, false for dotted/leading keys
        std::string text;
    };

    /// Split a path into segments, stripping one optional leading sigil.
    /// Returns false for a malformed path:
    ///   - an empty path, or a bare sigil, which names no location
    ///   - an unterminated subscript
    ///   - a leading dot after the optional sigil
    ///   - an empty segment (`a..b`, or a trailing `.`)
    ///   - text between a `]` and the next separator (`q.dims[0]bogus`)
    ///
    /// A malformed path resolves as not found rather than as a guess at the
    /// location the caller meant.
    static bool tokenize(const std::string& raw, std::vector<Segment>& out)
    {
        std::size_t pos = 0;
        if(!raw.empty() && raw[0] == VARIABLE_SIGIL)
        {
            ++pos; // strip the variable sigil
        }
        if(pos == raw.size())
        {
            return false; // no location named
        }
        if(raw[pos] == '.')
        {
            return false; // leading empty segment
        }
        bool afterSubscript = false;
        while(pos < raw.size())
        {
            if(raw[pos] == '[')
            {
                const std::size_t close = raw.find(']', pos + 1);
                if(close == std::string::npos)
                {
                    return false;
                }
                out.push_back({true, raw.substr(pos + 1, close - pos - 1)});
                pos = close + 1;
                afterSubscript = true;
            }
            else
            {
                if(raw[pos] == '.')
                {
                    ++pos; // skip the key separator
                }
                else if(afterSubscript)
                {
                    // A `]` must be followed by `.`, `[`, or the end of the
                    // path; anything else is trailing garbage, not a key.
                    return false;
                }
                const std::size_t next = raw.find_first_of(".[", pos);
                const std::size_t end = (next == std::string::npos) ? raw.size() : next;
                if(end == pos)
                {
                    return false; // empty segment
                }
                out.push_back({false, raw.substr(pos, end - pos)});
                pos = end;
                afterSubscript = false;
            }
        }
        return true;
    }

    /// Parse a decimal index. Rejects empty text, any non-digit character, and
    /// any text too long to index a document this addresses. A rejected index
    /// resolves as not found, the same as an index past the end of an array.
    static bool parseIndex(const std::string& s, std::size_t& idx)
    {
        if(s.empty())
        {
            return false;
        }
        // Digits only, checked by hand rather than with strtol. strtol accepts
        // leading whitespace and a leading '+', so `[ 3]` and `[+3]` would both
        // resolve as index 3, and it saturates at LONG_MAX on overflow instead
        // of failing.
        for(const char c : s)
        {
            if(c < '0' || c > '9')
            {
                return false;
            }
        }
        // Bound the digit count, so an absurdly long string never reaches a
        // conversion that would overflow. No document this addresses holds an
        // array anywhere near seven digits long.
        constexpr std::size_t MAX_INDEX_DIGITS = 7;
        if(s.size() > MAX_INDEX_DIGITS)
        {
            return false;
        }
        std::size_t val = 0;
        for(const char c : s)
        {
            val = (val * 10U) + static_cast<std::size_t>(c - '0');
        }
        idx = val;
        return true;
    }

    static Value toValue(const nlohmann::json& j)
    {
        if(j.is_boolean())
        {
            return {j.get<bool>()};
        }
        if(j.is_number_unsigned())
        {
            const auto u = j.get<std::uint64_t>();
            if(u > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            {
                return {};
            }
            return {static_cast<std::int64_t>(u)};
        }
        if(j.is_number_integer())
        {
            return {j.get<std::int64_t>()};
        }
        if(j.is_number_float())
        {
            return {j.get<double>()};
        }
        if(j.is_string())
        {
            return {j.get<std::string>()};
        }
        if(j.is_array())
        {
            Value::Array a;
            a.reserve(j.size());
            for(const auto& e : j)
            {
                a.push_back(toValue(e));
            }
            return {std::move(a)};
        }
        return {}; // null or object -> not representable, treated as null
    }

    nlohmann::json _doc;
};

} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
