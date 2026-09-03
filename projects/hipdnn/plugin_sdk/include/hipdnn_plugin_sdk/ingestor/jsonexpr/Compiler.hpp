// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Compiler.hpp - the json -> node tree lowering.
//
// Runs after the layout-alias pre-pass, so a `stride_order` here is always an
// integer array.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Error.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Node.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/OperatorTable.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
inline Value jsonScalarToValue(const nlohmann::json& j)
{
    if(j.is_boolean())
    {
        return {j.get<bool>()};
    }
    if(j.is_number_unsigned())
    {
        const auto raw = j.get<nlohmann::json::number_unsigned_t>();
        if(raw > static_cast<nlohmann::json::number_unsigned_t>(
               std::numeric_limits<std::int64_t>::max()))
        {
            throw JsonExpressionCompileError("unsigned integer literal exceeds int64_t range");
        }
        return {static_cast<std::int64_t>(raw)};
    }
    if(j.is_number_integer())
    {
        return {j.get<std::int64_t>()};
    }
    if(j.is_number_float())
    {
        return {j.get<double>()};
    }
    return {}; // null
}

inline NodePtr compileNode(const nlohmann::json& j, std::size_t depth = 0);

inline NodePtr compileObject(const nlohmann::json& j, std::size_t depth)
{
    if(j.size() != 1)
    {
        throw JsonExpressionCompileError("expression object must have exactly one operator key");
    }
    const auto it = j.begin();
    const std::string& key = it.key();
    const nlohmann::json& val = it.value();
    if(key == "var")
    {
        throw JsonExpressionCompileError(
            "the 'var' operator is not supported; write a variable as a sigil-prefixed "
            "string (\"$path\"), and use 'value_or_default' for a fallback");
    }

    const OpSpec* spec = lookupOp(key);
    if(spec == nullptr)
    {
        throw JsonExpressionCompileError("unrecognized operation: " + key);
    }

    auto node = std::make_unique<OpNode>();
    node->spec = spec;
    if(val.is_array())
    {
        node->args.reserve(val.size());
        for(const auto& e : val)
        {
            node->args.push_back(compileNode(e, depth + 1));
        }
    }
    else
    {
        node->args.push_back(compileNode(val, depth + 1));
    }
    checkArity(*spec, node->args.size(), key);
    return node;
}

inline NodePtr compileNode(const nlohmann::json& j, std::size_t depth)
{
    checkExpressionDepth(depth);
    if(j.is_object())
    {
        return compileObject(j, depth);
    }
    if(j.is_array())
    {
        auto n = std::make_unique<ArrayNode>();
        n->items.reserve(j.size());
        for(const auto& e : j)
        {
            n->items.push_back(compileNode(e, depth + 1));
        }
        return n;
    }
    if(j.is_string())
    {
        const auto& s = j.get_ref<const nlohmann::json::string_t&>();
        if(s.empty() || s[0] != VARIABLE_SIGIL)
        {
            return std::make_unique<LiteralNode>(Value(s));
        }
        if(s.size() >= 2 && s[1] == VARIABLE_SIGIL)
        {
            return std::make_unique<LiteralNode>(Value(s.substr(1))); // "$$x" -> "$x"
        }
        if(s.size() == 1)
        {
            throw JsonExpressionCompileError("whole-document variable reference is not supported");
        }
        auto n = std::make_unique<VarNode>();
        n->path = s.substr(1);
        return n;
    }
    return std::make_unique<LiteralNode>(jsonScalarToValue(j));
}
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
