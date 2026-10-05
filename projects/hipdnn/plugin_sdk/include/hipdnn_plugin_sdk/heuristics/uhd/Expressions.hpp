// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>
#include <nlohmann/json.hpp>

/// @file Expressions.hpp
/// @brief UHD's binding to the descriptor expression language (RFC 0019 §6.2).
///
/// Operators that cannot answer (unbound variable, zero divisor, non-finite result) yield
/// null rather than throwing; use number() when a feature row needs a strict number.
namespace hipdnn_plugin_sdk::uhd
{

namespace jsonexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

/// A descriptor expression that cannot be compiled, or a feature that cannot be evaluated.
class JsonLogicError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/// Per-reference vocabularies of a categorical feature, keyed by the full `$` reference.
using CategoricalEncoding = std::map<std::string, std::map<std::string, int32_t>>;

/// The symbols an expression reads, keyed without the `$` sigil (accepted either way).
/// An unbound name reads as null.
class VariableContext
{
public:
    using ValueType = std::variant<double, int64_t, std::string, bool>;

    void bind(const std::string& name, ValueType value)
    {
        _bindings.insert_or_assign(unsigil(name), std::move(value));
    }

    void bindNamespace(const std::string& ns,
                       const std::unordered_map<std::string, ValueType>& values)
    {
        const std::string prefix = ns + ".";
        for(const auto& [name, value] : values)
        {
            _bindings.insert_or_assign(prefix + name, value);
        }
    }

    void clearNamespace(const std::string& ns)
    {
        const std::string prefix = ns + ".";
        for(auto it = _bindings.begin(); it != _bindings.end();)
        {
            if(it->first.rfind(prefix, 0) == 0)
            {
                it = _bindings.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    const ValueType* find(const std::string& name) const
    {
        const auto it = !name.empty() && name.front() == '$' ? _bindings.find(name.substr(1))
                                                             : _bindings.find(name);
        return it == _bindings.end() ? nullptr : &it->second;
    }

    bool has(const std::string& name) const
    {
        return find(name) != nullptr;
    }

    void clear()
    {
        _bindings.clear();
    }

    /// @brief Every binding, keyed without the sigil.
    const std::unordered_map<std::string, ValueType>& bindings() const
    {
        return _bindings;
    }

    /// @brief The jsonexpr data-source contract: @p path arrives without the sigil.
    jsonexpr::Value getData(const std::string& path) const
    {
        const auto it = _bindings.find(path);
        if(it == _bindings.end())
        {
            return {};
        }
        return std::visit([](const auto& held) { return jsonexpr::Value(held); }, it->second);
    }

private:
    static std::string unsigil(const std::string& name)
    {
        return !name.empty() && name.front() == '$' ? name.substr(1) : name;
    }

    std::unordered_map<std::string, ValueType> _bindings;
};

/// Descriptor expressions compiled once and evaluated many times.
/// Immutable after construction: safe for concurrent evaluation; copies share the trees.
class ExpressionSet
{
public:
    /// Operator nesting bound per expression; tighter than the language's own limit.
    static constexpr size_t MAX_EXPRESSION_DEPTH = 64;
    /// JSON nodes read across the whole set, before anything is compiled.
    static constexpr size_t MAX_INPUT_NODES = 65536;
    static constexpr size_t MAX_EXPRESSIONS = 16384;
    static constexpr size_t MAX_STRING_BYTES = 65536;

    ExpressionSet() = default;

    /// @throws JsonLogicError when an expression exceeds a bound or does not compile.
    explicit ExpressionSet(const std::vector<nlohmann::json>& expressions)
    {
        if(expressions.size() > MAX_EXPRESSIONS)
        {
            throw JsonLogicError("Too many descriptor expressions");
        }
        auto compiled = std::make_shared<Compiled>();
        size_t visited = 0;
        compiled->entries.reserve(expressions.size());
        for(const auto& expression : expressions)
        {
            if(expression.is_null())
            {
                // A null literal would compile to an expression that never resolves.
                throw JsonLogicError("Unsupported descriptor expression type");
            }
            checkBounds(expression, 0, visited);
            const auto lowered = lower(expression);
            Entry entry;
            if(isReference(lowered))
            {
                entry.reference = lowered.get<std::string>();
            }
            try
            {
                entry.compiled = jsonexpr::compile<VariableContext>(lowered);
            }
            catch(const std::exception& error)
            {
                throw JsonLogicError("Invalid descriptor expression: " + std::string(error.what()));
            }
            for(const auto& path : entry.compiled.variables())
            {
                entry.variables.push_back("$" + path);
                compiled->variables.insert("$" + path);
            }
            entry.kernel = jsonexpr::referencesVariableRoot(entry.compiled, "kernel");
            compiled->entries.push_back(std::move(entry));
        }
        _compiled = std::move(compiled);
    }

    size_t size() const
    {
        return _compiled->entries.size();
    }

    /// @brief The language's answer for expression @p index; null when it did not resolve.
    jsonexpr::Value evaluate(size_t index, const VariableContext& context) const
    {
        return _compiled->entries.at(index).compiled(context);
    }

    /// @brief Expression @p index's answer, which must resolve.
    /// @throws JsonLogicError naming the first unbound symbol, or a generic error when all
    ///         symbols were bound but the expression still did not resolve.
    jsonexpr::Value resolve(size_t index, const VariableContext& context) const
    {
        auto value = evaluate(index, context);
        if(value.containsUnresolved())
        {
            for(const auto& variable : variables(index))
            {
                if(!context.has(variable))
                {
                    throw JsonLogicError("Undefined variable: " + variable);
                }
            }
            throw JsonLogicError("Descriptor expression did not resolve");
        }
        return value;
    }

    /// @brief Whether expression @p index reads any `$kernel` symbol.
    bool referencesKernel(size_t index) const
    {
        return _compiled->entries.at(index).kernel;
    }

    /// @brief The `$` references expression @p index reads, in order of appearance.
    const std::vector<std::string>& variables(size_t index) const
    {
        return _compiled->entries.at(index).variables;
    }

    /// @brief The `$` reference expression @p index is, once lowered; empty when it is an
    ///        expression rather than a single reference.
    const std::string& reference(size_t index) const
    {
        return _compiled->entries.at(index).reference;
    }

    /// @brief Every `$` reference the set reads.
    const std::unordered_set<std::string>& variables() const
    {
        return _compiled->variables;
    }

    /// @brief A resolved value as a finite number: booleans read as 0/1.
    /// @throws JsonLogicError for null, a string, an array, or a non-finite number.
    static double number(const jsonexpr::Value& value)
    {
        if(value.containsUnresolved())
        {
            throw JsonLogicError("Descriptor expression did not resolve");
        }
        if(value.isString())
        {
            throw JsonLogicError("Type error: string used where a number is required");
        }
        if(!value.isNumber() && !value.isBool())
        {
            throw JsonLogicError("Type error: array used where a number is required");
        }
        const double result = value.toNumber();
        if(!std::isfinite(result))
        {
            throw JsonLogicError("Non-finite expression value");
        }
        return result;
    }

    /// @brief A resolved value's truth, in the language's own reading.
    /// @throws JsonLogicError for null.
    static bool truth(const jsonexpr::Value& value)
    {
        if(value.containsUnresolved())
        {
            throw JsonLogicError("Descriptor expression did not resolve");
        }
        return value.truthy();
    }

private:
    struct Entry
    {
        jsonexpr::Expression<VariableContext> compiled;
        std::vector<std::string> variables;
        std::string reference;
        bool kernel = false;
    };

    struct Compiled
    {
        std::vector<Entry> entries;
        std::unordered_set<std::string> variables;
    };

    static void checkBounds(const nlohmann::json& node, size_t depth, size_t& visited)
    {
        // Each operator adds two JSON levels (object + argument array).
        if(depth > 2 * MAX_EXPRESSION_DEPTH + 2 || ++visited > MAX_INPUT_NODES)
        {
            throw JsonLogicError("Descriptor expression exceeds the depth or input-size bound");
        }
        if(node.is_string() && node.get_ref<const std::string&>().size() > MAX_STRING_BYTES)
        {
            throw JsonLogicError("Expression string exceeds size bound");
        }
        if(node.is_array() || node.is_object())
        {
            for(const auto& child : node)
            {
                checkBounds(child, depth + 1, visited);
            }
        }
    }

    static bool isReference(const nlohmann::json& node)
    {
        if(!node.is_string())
        {
            return false;
        }
        const auto& text = node.get_ref<const std::string&>();
        return text.size() > 1 && text[0] == '$' && text[1] != '$';
    }

    /// Lowers §6.2 short-hands: `{"shape": ["$t", k]}` -> `"$t.dims[k]"`,
    /// `{"rank": "$t"}` -> `"$t.rank"`.
    static nlohmann::json lower(const nlohmann::json& node)
    {
        if(node.is_array())
        {
            auto result = nlohmann::json::array();
            for(const auto& child : node)
            {
                result.push_back(lower(child));
            }
            return result;
        }
        if(!node.is_object())
        {
            return node;
        }
        if(node.size() == 1)
        {
            const auto& name = node.begin().key();
            const auto& body = node.begin().value();
            if(name == "shape")
            {
                if(!body.is_array() || body.size() != 2)
                {
                    throw JsonLogicError("Invalid argument count for shape");
                }
                if(!isReference(body[0]))
                {
                    throw JsonLogicError("shape requires a tensor reference");
                }
                if(!body[1].is_number_integer() || body[1].get<int64_t>() < 0
                   || body[1].get<int64_t>() > 1024)
                {
                    throw JsonLogicError("shape requires a constant nonnegative dimension index");
                }
                return body[0].get<std::string>() + ".dims["
                       + std::to_string(body[1].get<int64_t>()) + "]";
            }
            if(name == "rank")
            {
                const bool wrapped = body.is_array();
                if(wrapped && body.size() != 1)
                {
                    throw JsonLogicError("Invalid argument count for rank");
                }
                const auto& tensor = wrapped ? body[0] : body;
                if(!isReference(tensor))
                {
                    throw JsonLogicError("rank requires a tensor reference");
                }
                return tensor.get<std::string>() + ".rank";
            }
        }
        auto result = nlohmann::json::object();
        for(const auto& [key, value] : node.items())
        {
            result[key] = lower(value);
        }
        return result;
    }

    std::shared_ptr<const Compiled> _compiled = std::make_shared<const Compiled>();
};

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
