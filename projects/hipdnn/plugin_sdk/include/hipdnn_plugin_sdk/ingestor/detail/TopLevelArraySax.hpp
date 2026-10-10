// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace hipdnn_plugin_sdk::ingestor::detail
{

/// Builds a document from SAX events the way nlohmann's own DOM parser does -- a repeated
/// key overwrites, and each number keeps the integer, unsigned or float kind the lexer
/// reported -- through the public json API only.
class JsonBuilder
{
public:
    explicit JsonBuilder(nlohmann::json& root)
        : _root(&root)
    {
    }

    /// Open containers; 0 once the value started at the root is complete.
    size_t depth() const
    {
        return _open.size();
    }
    void value(nlohmann::json value)
    {
        *slot() = std::move(value);
    }
    void startObject()
    {
        auto* opened = slot();
        *opened = nlohmann::json::object();
        _open.push_back(opened);
    }
    void startArray()
    {
        auto* opened = slot();
        *opened = nlohmann::json::array();
        _open.push_back(opened);
    }
    void key(std::string name)
    {
        _key = std::move(name);
    }
    void end()
    {
        _open.pop_back();
    }

private:
    /// Where the next value goes. An array element's address is taken only while that
    /// element is the last one, and no sibling is appended until it closes, so every
    /// pointer on the open stack stays valid.
    nlohmann::json* slot()
    {
        if(_open.empty())
        {
            return _root;
        }
        auto& parent = *_open.back();
        if(parent.is_array())
        {
            parent.push_back(nullptr);
            return &parent.back();
        }
        return &parent[_key];
    }

    nlohmann::json* _root;
    std::vector<nlohmann::json*> _open;
    std::string _key;
};

/// SAX consumer for nlohmann::json::sax_parse that streams the elements of every array
/// stored under one key of the root object, without ever holding such an array whole.
///
/// The whole document is lexed, so every syntax error surfaces as a DOM parse reports it.
/// Everything outside those arrays can be built into @p rest, where each such array is
/// left in place and empty, so @p rest is the document minus the arrays' elements. Each
/// array is announced to @p handler as it opens -- after @p rest already holds its key --
/// and the handler decides whether its elements stream. A streamed element is built as
/// its own small document, handed over, and discarded; elements arrive in document order,
/// so the handler sees exactly what a loop over the DOM array would.
///
/// @p Handler provides:
///  - `bool openArray(size_t ordinal)`: the ordinal-th (1-based) such array has opened;
///    true streams its elements.
///  - `bool element(const nlohmann::json& entry)`: one element of a streamed array; false
///    stops streaming the rest of that array, which is then only lexed. An exception it
///    throws propagates out of sax_parse.
///
/// No `nlohmann::json_sax` base: sax_parse takes any type with these members, and the
/// base's virtual template members fail clang-tidy.
template <typename Handler>
class TopLevelArraySax final
{
public:
    /// @p rest may be null when nothing outside the arrays is wanted.
    TopLevelArraySax(std::string_view key, nlohmann::json* rest, Handler& handler)
        : _key(key)
        , _handler(handler)
        , _elementBuilder(_element)
    {
        if(rest != nullptr)
        {
            _rest.emplace(*rest);
        }
    }

    /// How many arrays were found under the key so far.
    size_t arrays() const
    {
        return _arrays;
    }
    /// Whether a key of the root object followed the first such array.
    bool keysAfterArray() const
    {
        return _keysAfterArray;
    }

    // NOLINTBEGIN(readability-identifier-naming) - nlohmann's SAX interface names these
    bool null()
    {
        return scalar(nullptr);
    }
    bool boolean(bool value)
    {
        return scalar(value);
    }
    bool number_integer(nlohmann::json::number_integer_t value)
    {
        return scalar(value);
    }
    bool number_unsigned(nlohmann::json::number_unsigned_t value)
    {
        return scalar(value);
    }
    bool number_float(nlohmann::json::number_float_t value,
                      const nlohmann::json::string_t& /*text*/)
    {
        return scalar(value);
    }
    bool string(nlohmann::json::string_t& value)
    {
        return scalar(std::move(value));
    }
    bool binary(nlohmann::json::binary_t& value)
    {
        return scalar(nlohmann::json::binary(value));
    }
    bool start_object(std::size_t /*elements*/)
    {
        if(_arrayDepth == 0)
        {
            ++_depth;
            _keyNext = false;
            if(_rest)
            {
                _rest->startObject();
            }
            return true;
        }
        beginElement();
        ++_depth;
        if(_building)
        {
            _elementBuilder.startObject();
        }
        return true;
    }
    bool key(nlohmann::json::string_t& name)
    {
        if(_arrayDepth == 0)
        {
            if(_depth == 1 && _arrays != 0)
            {
                _keysAfterArray = true;
            }
            _keyNext = _depth == 1 && name == _key;
            if(_rest)
            {
                _rest->key(name);
            }
        }
        else if(_building)
        {
            _elementBuilder.key(name);
        }
        return true;
    }
    bool end_object()
    {
        --_depth;
        if(_arrayDepth == 0)
        {
            if(_rest)
            {
                _rest->end();
            }
        }
        else
        {
            endContainer();
        }
        return true;
    }
    bool start_array(std::size_t /*elements*/)
    {
        if(_arrayDepth == 0)
        {
            ++_depth;
            if(_rest)
            {
                _rest->startArray();
            }
            if(_keyNext)
            {
                // The array stays in the rest document, empty; its elements stream.
                _keyNext = false;
                if(_rest)
                {
                    _rest->end();
                }
                _arrayDepth = _depth;
                _streaming = _handler.openArray(++_arrays);
            }
            return true;
        }
        beginElement();
        ++_depth;
        if(_building)
        {
            _elementBuilder.startArray();
        }
        return true;
    }
    bool end_array()
    {
        if(_arrayDepth == 0)
        {
            --_depth;
            if(_rest)
            {
                _rest->end();
            }
            return true;
        }
        if(!_building && _depth == _arrayDepth)
        {
            _arrayDepth = 0;
            _streaming = false;
            --_depth;
            return true;
        }
        --_depth;
        endContainer();
        return true;
    }
    /// Rethrows the concrete type nlohmann reports (`parse_error`, or `out_of_range` for a
    /// number overflow), as its DOM parser does, so a caller sees the same exception.
    template <typename Exception>
    static bool parse_error(std::size_t /*position*/,
                            const std::string& /*lastToken*/,
                            const Exception& error)
    {
        throw error;
    }
    // NOLINTEND(readability-identifier-naming)

private:
    /// Starts a new element when a container opens directly inside a streamed array.
    void beginElement()
    {
        if(_streaming && !_building && _depth == _arrayDepth)
        {
            _element = nlohmann::json();
            _building = true;
        }
    }

    void endContainer()
    {
        if(!_building)
        {
            return;
        }
        _elementBuilder.end();
        if(_elementBuilder.depth() == 0)
        {
            _building = false;
            _streaming = _handler.element(_element);
        }
    }

    template <typename Value>
    bool scalar(Value&& value)
    {
        if(_arrayDepth == 0)
        {
            _keyNext = false;
            if(_rest)
            {
                _rest->value(nlohmann::json(std::forward<Value>(value)));
            }
        }
        else if(_building)
        {
            _elementBuilder.value(nlohmann::json(std::forward<Value>(value)));
        }
        else if(_streaming && _depth == _arrayDepth)
        {
            _streaming = _handler.element(nlohmann::json(std::forward<Value>(value)));
        }
        return true;
    }

    std::string_view _key;
    Handler& _handler;
    std::optional<JsonBuilder> _rest;
    nlohmann::json _element;
    JsonBuilder _elementBuilder;
    bool _building = false;
    bool _streaming = false; ///< inside an array the handler chose to stream
    bool _keyNext = false; ///< the next value belongs to the key
    bool _keysAfterArray = false;
    size_t _arrays = 0;
    size_t _depth = 0;
    size_t _arrayDepth = 0; ///< depth of the array being read; 0 outside one
};

} // namespace hipdnn_plugin_sdk::ingestor::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
