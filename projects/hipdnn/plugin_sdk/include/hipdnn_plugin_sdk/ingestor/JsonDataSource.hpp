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
// It also offers the inverse, setData, which writes a jexpr::Value back into
// the document at a path, creating intermediate objects and arrays as needed:
//
//     src.setData("$q.dims[0]", 2);   // -> {"q":{"dims":[2, 16]}}
//
// Path syntax (shared by getData and setData):
//   - dotted keys:            a.b.c
//   - [N] array subscripts:   arr[0], rows[2].name, grid[0][1]
//   - dot-form array indices: arr.1 (resolves as an index only against an
//                             existing array; use arr[1] to force array creation)
//   - an optional leading variable sigil ('$' by default) is stripped, so both
//     "q.dims[0]" and "$q.dims[0]" address the same location
//   - the empty path (or a bare sigil) addresses the whole document
//
// getData follows the language convention: an unresolved path reads as null
// (Value()). setData is a mutation and reports a malformed path or an
// incompatible index by throwing std::invalid_argument.
//
// This is a *sample* accessor: objects and null in the document convert to
// jexpr::Value null (Value has no object alternative), matching Value's
// scalar/array-only model.
//
// Full reference: docs/JsonExpression.md.

#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
class JsonDataSource
{
public:
    JsonDataSource() = default;
    explicit JsonDataSource(nlohmann::json doc, char sigil = '$')
        : _doc(std::move(doc))
        , _sigil(sigil)
    {
    }

    /// The backing document, for inspection or bulk replacement.
    const nlohmann::json& document() const
    {
        return _doc;
    }
    nlohmann::json& document()
    {
        return _doc;
    }

    /// Data-source contract: resolve a variable path to a Value.
    /// An empty path returns the whole document; an unresolved path (missing
    /// key, out-of-range or non-numeric index, subscript on a non-array)
    /// returns null so a `var` default, if any, takes over.
    Value getData(const std::string& path) const
    {
        std::vector<Segment> segs;
        if(!tokenize(path, _sigil, segs))
        {
            return {}; // malformed subscript -> not found
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

    /// Write a Value into the document at a path, creating intermediate
    /// containers as needed: a `[N]` subscript (or a dot-form index against an
    /// existing array) grows an array, filling gaps with null; any other key
    /// creates or descends into an object. An empty path replaces the whole
    /// document. Throws std::invalid_argument on a malformed path, a
    /// non-numeric index applied to an array, or an index at or above
    /// MAX_ARRAY_INDEX.
    ///
    /// All-or-nothing: the path is fully validated before anything is written,
    /// so a throwing call leaves the document exactly as it was. Validating
    /// inline instead would let a rejected write destroy data it had already
    /// passed over -- `setData("q.dims[999999999]")` on `{"q":5}` would replace
    /// the 5 with `{"dims":null}` on its way to the throw.
    void setData(const std::string& path, const Value& value)
    {
        std::vector<Segment> segs;
        if(!tokenize(path, _sigil, segs))
        {
            throw std::invalid_argument("JsonDataSource::setData: malformed path '" + path + "'");
        }
        if(segs.empty())
        {
            _doc = toJson(value); // whole-document assignment
            return;
        }
        validatePath(segs, path);

        // Every failure mode is now ruled out, so this loop cannot throw and
        // the document cannot be left half-written.
        nlohmann::json* cur = &_doc;
        for(std::size_t i = 0; i < segs.size(); ++i)
        {
            const Segment& seg = segs[i];
            nlohmann::json* target = nullptr;
            std::size_t idx = 0;
            if(seg.subscript)
            {
                parseIndex(seg.text, idx);
                if(!cur->is_array())
                {
                    *cur = nlohmann::json::array();
                }
                target = &(*cur)[idx]; // grows the array, filling gaps with null
            }
            else if(cur->is_array())
            {
                parseIndex(seg.text, idx);
                target = &(*cur)[idx];
            }
            else
            {
                if(!cur->is_object())
                {
                    *cur = nlohmann::json::object();
                }
                target = &(*cur)[seg.text];
            }
            if(i + 1 == segs.size())
            {
                *target = toJson(value);
            }
            else
            {
                cur = target;
            }
        }
    }

private:
    struct Segment
    {
        bool subscript; // true for [N] forms, false for dotted/leading keys
        std::string text;
    };

    /// Reject every index setData's write walk could reject, without touching
    /// the document. Mirrors that walk's branching exactly: a subscript always
    /// indexes, and a dotted segment indexes only where the document already
    /// holds an array -- so it has to descend the existing structure to know
    /// which segments are index positions. Descent stops at the first segment
    /// that is not already present, because from there on the write creates
    /// objects and only explicit subscripts can be index positions.
    void validatePath(const std::vector<Segment>& segs, const std::string& path) const
    {
        const nlohmann::json* cur = &_doc;
        for(const Segment& seg : segs)
        {
            std::size_t idx = 0;
            if(seg.subscript)
            {
                if(!parseIndex(seg.text, idx))
                {
                    throw std::invalid_argument("JsonDataSource::setData: bad array index '"
                                                + seg.text + "' in path '" + path + "'");
                }
                cur = (cur != nullptr && cur->is_array() && idx < cur->size()) ? &(*cur)[idx]
                                                                               : nullptr;
            }
            else if(cur != nullptr && cur->is_array())
            {
                if(!parseIndex(seg.text, idx))
                {
                    throw std::invalid_argument("JsonDataSource::setData: non-numeric key '"
                                                + seg.text + "' on array in path '" + path + "'");
                }
                cur = idx < cur->size() ? &(*cur)[idx] : nullptr;
            }
            else if(cur != nullptr && cur->is_object() && cur->contains(seg.text))
            {
                cur = &(*cur)[seg.text];
            }
            else
            {
                // Not present: the write creates objects from here down, so no
                // later segment can land on an array unless it says so itself.
                cur = nullptr;
            }
        }
    }

    /// Split a path into segments. Strips one optional leading sigil. Returns
    /// false on a malformed path: an unterminated subscript, an empty segment
    /// (`a..b`, or a trailing `.`), or text wedged between a `]` and the next
    /// separator (`q.dims[0]bogus`). Accepting those would let setData create
    /// a key nobody wrote, under a contract that says it throws instead.
    static bool tokenize(const std::string& raw, char sigil, std::vector<Segment>& out)
    {
        std::size_t pos = 0;
        if(!raw.empty() && raw[0] == sigil)
        {
            ++pos; // strip the variable sigil
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

    /// The largest index a path may name. A `[N]` subscript in setData grows
    /// the array to N, so an unbounded index turns a one-character typo in a
    /// descriptor path into an allocation of arbitrary size. No document this
    /// addresses is anywhere near this long, and getData resolves an index at
    /// or above the bound to null exactly as it already did for any index past
    /// the end.
    static constexpr std::size_t MAX_ARRAY_INDEX = (1U << 20U);

    /// Parse a decimal index below MAX_ARRAY_INDEX. Rejects empty text, any
    /// non-digit character, and an out-of-bounds value.
    static bool parseIndex(const std::string& s, std::size_t& idx)
    {
        if(s.empty())
        {
            return false;
        }
        // Digits only, checked by hand rather than with strtol: strtol accepts
        // leading whitespace and a leading '+', so `[ 3]` and `[+3]` would both
        // resolve as index 3, and it saturates at LONG_MAX on overflow instead
        // of failing. Requiring digits only removes all three.
        for(const char c : s)
        {
            if(c < '0' || c > '9')
            {
                return false;
            }
        }
        // Bounds-check on the digit count first, so an absurdly long string
        // never reaches a conversion that would saturate.
        constexpr std::size_t MAX_INDEX_DIGITS = 7; // MAX_ARRAY_INDEX is 7 digits
        if(s.size() > MAX_INDEX_DIGITS)
        {
            return false;
        }
        std::size_t val = 0;
        for(const char c : s)
        {
            val = (val * 10U) + static_cast<std::size_t>(c - '0');
        }
        if(val >= MAX_ARRAY_INDEX)
        {
            return false;
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
        if(j.is_number_integer() || j.is_number_unsigned())
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

    static nlohmann::json toJson(const Value& v)
    {
        if(v.isBool())
        {
            return v.asBool();
        }
        if(v.isInt())
        {
            return v.asInt();
        }
        if(v.isDouble())
        {
            return v.asDouble();
        }
        if(v.isString())
        {
            return v.asString();
        }
        if(v.isArray())
        {
            nlohmann::json a = nlohmann::json::array();
            for(const auto& e : v.asArray())
            {
                a.push_back(toJson(e));
            }
            return a;
        }
        return nullptr; // null
    }

    nlohmann::json _doc;
    char _sigil = '$';
};

} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
