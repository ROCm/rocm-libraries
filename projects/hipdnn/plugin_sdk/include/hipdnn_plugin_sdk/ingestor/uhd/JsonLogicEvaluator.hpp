// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/CategoricalEncoding.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief Exception thrown when JsonLogic evaluation fails.
class JsonLogicError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/// @brief Thrown specifically when a `$ref` names a variable that is not bound.
///
/// Distinguished from the other JsonLogicError cases so `value_or_default` can supply
/// its default for an absent binding without also swallowing a type error or an
/// invalid operation — RFC 0019 §7.2 requires those to fail closed.
class UndefinedVariableError : public JsonLogicError
{
public:
    using JsonLogicError::JsonLogicError;
};

/// @brief Variable resolution context for JsonLogic evaluation.
///
/// Provides bindings for $device.*, $kernel.*, and $q.* namespaces.
/// Variables are resolved as doubles for feature extraction.
class VariableContext
{
public:
    using ValueType = std::variant<double, int64_t, std::string, bool>;

    /// Bind a variable value by full name (e.g., "$device.cu_count").
    void bind(const std::string& name, ValueType value);

    /// Bind all variables from a namespace map (e.g., "device" -> {"cu_count": 64}).
    void bindNamespace(const std::string& ns,
                       const std::unordered_map<std::string, ValueType>& values);

    /// Erase every binding in a namespace (e.g., "kernel" drops all $kernel.* entries).
    ///
    /// Rebinding a namespace does not remove keys the previous binding set but the
    /// new one omits. Callers that reuse a context across candidates must clear the
    /// namespace first, or a candidate missing a field silently inherits the previous
    /// candidate's value.
    void clearNamespace(const std::string& ns);

    /// Resolve a variable to a double. Returns nullopt if not found.
    std::optional<double> resolveDouble(const std::string& name) const;

    /// Resolve a variable to its raw value. Returns nullopt if not found.
    std::optional<ValueType> resolve(const std::string& name) const;

    /// Check if a variable is bound.
    bool has(const std::string& name) const;

    /// Clear all bindings.
    void clear();

private:
    std::unordered_map<std::string, ValueType> _bindings;
};

/// @brief JsonLogic expression evaluator.
///
/// Evaluates JsonLogic expressions per RFC 0018 format:
/// - Operators: {"op": [args]} where op is the operation name
/// - Variables: "$namespace.field" strings (no {"var": ...} wrapper)
/// - Literals: numbers, strings, booleans
///
/// Supported operators (RFC 0018 set):
/// - Arithmetic: +, -, *, /, %, ceil_div
/// - Math: min, max, abs, pow, log2, rsqrt
/// - Comparison: ==, !=, <, <=, >, >=
/// - Logical: and, or, !
/// - Control: if, value_or_default
class JsonLogicEvaluator
{
public:
    using Value = std::variant<double, bool, std::string>;

    /// Parse a JsonLogic expression from JSON string.
    /// @throws JsonLogicError on parse failure.
    static nlohmann::json parse(const std::string& jsonStr);

    /// Evaluate an expression against a variable context.
    ///
    /// When `expr` is a bare `$namespace.field` reference whose binding is a string,
    /// the field's category is resolved through CategoricalEncoding.hpp (RFC 0019 §6.5)
    /// so `dtype`, `layout` and their kin can be features at all. Every other numeric
    /// context still refuses a string.
    ///
    /// @returns The result as a double (for feature extraction).
    /// @throws JsonLogicError on evaluation failure, on a string that is not an encoded
    ///         category, and on a value a known category has no code for.
    double evaluateDouble(const nlohmann::json& expr, const VariableContext& ctx) const;

    /// Evaluate an expression to a generic value.
    /// Evaluate an expression to a generic value.
    /// @param depth Current recursion depth; the interpreter is bounded per RFC 0019
    ///        §7.2/§16, since a descriptor is author-controlled input and a deeply
    ///        nested expression would otherwise overflow the stack.
    Value evaluate(const nlohmann::json& expr, const VariableContext& ctx, size_t depth = 0) const;

    /// Maximum expression nesting the interpreter will descend.
    static constexpr size_t MAX_EXPRESSION_DEPTH = 64;

    /// Extract all variable references from an expression.
    /// Returns variable names (e.g., "$device.cu_count", "$kernel.tile_m").
    static std::unordered_set<std::string> extractVariables(const nlohmann::json& expr);

private:
    Value evaluateOp(const std::string& op,
                     const nlohmann::json& args,
                     const VariableContext& ctx,
                     size_t depth) const;

    /// Coerce to a number for a numeric context.
    /// @throws JsonLogicError if the value is a string — RFC 0019 §7.2 requires
    ///         failing closed on a type error rather than yielding NaN.
    static double toDouble(const Value& v);

    /// Structural equality: strings compare as strings, numbers as numbers.
    /// @throws JsonLogicError when comparing a string against a number.
    static bool valuesEqual(const Value& a, const Value& b);
    static bool toBool(const Value& v);
};


// ============================================================================
// VariableContext
// ============================================================================

inline void VariableContext::bind(const std::string& name, ValueType value)
{
    _bindings[name] = std::move(value);
}

inline void VariableContext::bindNamespace(const std::string& ns,
                                    const std::unordered_map<std::string, ValueType>& values)
{
    std::string prefix = "$";
    prefix += ns;
    prefix += ".";
    for(const auto& [key, val] : values)
    {
        _bindings[prefix + key] = val;
    }
}

inline void VariableContext::clearNamespace(const std::string& ns)
{
    std::string prefix = "$";
    prefix += ns;
    prefix += ".";

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

inline std::optional<double> VariableContext::resolveDouble(const std::string& name) const
{
    auto it = _bindings.find(name);
    if(it == _bindings.end())
    {
        return std::nullopt;
    }

    return std::visit(
        [&name](const auto& v) -> double {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, double>)
            {
                return v;
            }
            else if constexpr(std::is_same_v<T, int64_t>)
            {
                return static_cast<double>(v);
            }
            else if constexpr(std::is_same_v<T, bool>)
            {
                return v ? 1.0 : 0.0;
            }
            else
            {
                // Fails closed, for the same reason JsonLogicEvaluator::toDouble does
                // (RFC 0019 §7.2). A NaN here is not inert: it flows through `shape` or
                // `rank` into the feature row, and a GBDT reads NaN as a *missing* value
                // and takes default_left -- so a string bound where an extent belongs is
                // scored as an ordinary absent feature and never surfaces. The two
                // conversions must agree; one throwing and the other returning NaN is how
                // the same error becomes visible on one path and silent on the other.
                throw JsonLogicError("Type error: variable '" + name + "' is the string \""
                                     + v + "\", which cannot be used where a number is "
                                           "required");
            }
        },
        it->second);
}

inline std::optional<VariableContext::ValueType> VariableContext::resolve(const std::string& name) const
{
    auto it = _bindings.find(name);
    if(it == _bindings.end())
    {
        return std::nullopt;
    }
    return it->second;
}

inline bool VariableContext::has(const std::string& name) const
{
    return _bindings.find(name) != _bindings.end();
}

inline void VariableContext::clear()
{
    _bindings.clear();
}

// ============================================================================
// JsonLogicEvaluator
// ============================================================================

inline nlohmann::json JsonLogicEvaluator::parse(const std::string& jsonStr)
{
    try
    {
        return nlohmann::json::parse(jsonStr);
    }
    catch(const nlohmann::json::exception& e)
    {
        // Catch the whole nlohmann hierarchy, not just parse_error. Numeric overflow
        // ("1e400") raises out_of_range.406, which would otherwise escape as a raw
        // nlohmann type past every contract that promises JsonLogicError.
        throw JsonLogicError("Failed to parse JsonLogic expression: " + std::string(e.what()));
    }
}

inline double JsonLogicEvaluator::evaluateDouble(const nlohmann::json& expr,
                                          const VariableContext& ctx) const
{
    const Value value = evaluate(expr, ctx);

    // RFC 0019 §6.5: a features_signature entry naming a string-valued field is the one
    // numeric context where a string is not a type error. It is a category, and
    // CategoricalEncoding.hpp says which number it is -- globally, so `dtype="fp16"`
    // is the same feature value whichever engine asked (§11.3).
    //
    // Deliberately here and not in toDouble. toDouble is *every* operator's numeric
    // context, so encoding there would also make {"+": ["$kernel.dtype", 1]} succeed --
    // arithmetic on a category, which has no meaning and must keep failing. Only the
    // signature entry itself is a place a category legitimately becomes a number.
    if(const auto* text = std::get_if<std::string>(&value); text != nullptr && expr.is_string())
    {
        const std::string_view category
            = categoryOfReference(expr.get_ref<const nlohmann::json::string_t&>());
        if(const auto code = encodeCategorical(category, *text); code.has_value())
        {
            return *code;
        }
        if(isKnownCategory(category))
        {
            // Distinct from toDouble's blanket type error on purpose. The category is
            // one we encode, so this is a value the fixed table has never been told
            // about -- a catalog that moved past what any model was trained on. Scoring
            // it would put an unseen kernel on the same axis as seen ones; refusing
            // makes the gap visible at the signature entry that has the problem.
            throw JsonLogicError("Categorical value \"" + *text + "\" has no code in category '"
                                 + std::string(category)
                                 + "'. Append it to CATEGORICAL_ENCODING_TABLE in "
                                   "CategoricalEncoding.hpp and mirror it in "
                                   "tools/uhd_gen/features.py; existing codes must not move.");
        }
    }

    return toDouble(value);
}

inline JsonLogicEvaluator::Value JsonLogicEvaluator::evaluate(const nlohmann::json& expr,
                                                       const VariableContext& ctx,
                                                       size_t depth) const
{
    // Bound the descent (RFC 0019 §7.2 "safe, bounded interpreter"; §16 lists the
    // descriptor as author-controlled input). Without this a deeply nested expression
    // is a stack overflow, which the tree-walk bound in TreeDataAdapter does not cover.
    if(depth > MAX_EXPRESSION_DEPTH)
    {
        throw JsonLogicError("JsonLogic expression exceeds the maximum nesting depth of "
                             + std::to_string(MAX_EXPRESSION_DEPTH));
    }

    // Literal number
    if(expr.is_number())
    {
        return expr.get<double>();
    }

    // Literal boolean
    if(expr.is_boolean())
    {
        return expr.get<bool>();
    }

    // String: either a literal or a variable reference ($...)
    if(expr.is_string())
    {
        std::string s = expr.get<std::string>();
        if(!s.empty() && s[0] == '$')
        {
            auto raw = ctx.resolve(s);
            if(!raw.has_value())
            {
                throw UndefinedVariableError("Undefined variable: " + s);
            }

            // Return the binding's own type. A string stays a string so `==` and `in`
            // can compare it; using one where a number is required is caught in
            // toDouble, which is the actual numeric context. Checking here instead
            // would make every string-valued property uncomparable.
            return std::visit(
                [](const auto& v) -> Value {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, bool>)
                    {
                        return v;
                    }
                    else
                    {
                        return static_cast<double>(v);
                    }
                },
                *raw);
        }
        return s;
    }

    // Object: {"op": [args]} format
    if(expr.is_object() && expr.size() == 1)
    {
        auto it = expr.begin();
        const std::string& op = it.key();
        const nlohmann::json& args = it.value();
        return evaluateOp(op, args, ctx, depth);
    }

    // Array: evaluate and return first element (or error)
    if(expr.is_array())
    {
        if(expr.empty())
        {
            throw JsonLogicError("Empty array in expression");
        }
        return evaluate(expr[0], ctx, depth + 1);
    }

    throw JsonLogicError("Unsupported expression type");
}

inline bool JsonLogicEvaluator::valuesEqual(const Value& a, const Value& b)
{
    const bool aIsString = std::holds_alternative<std::string>(a);
    const bool bIsString = std::holds_alternative<std::string>(b);

    if(aIsString != bIsString)
    {
        throw JsonLogicError("Type error: cannot compare a string against a number");
    }
    if(aIsString)
    {
        return std::get<std::string>(a) == std::get<std::string>(b);
    }
    return toDouble(a) == toDouble(b);
}

inline JsonLogicEvaluator::Value JsonLogicEvaluator::evaluateOp(const std::string& op,
                                                         const nlohmann::json& args,
                                                         const VariableContext& ctx,
                                                         size_t depth) const
{
    // Ensure args is an array for most operations
    auto getArgs = [&]() -> std::vector<Value> {
        if(!args.is_array())
        {
            return {evaluate(args, ctx, depth + 1)};
        }
        std::vector<Value> result;
        result.reserve(args.size());
        for(const auto& arg : args)
        {
            result.push_back(evaluate(arg, ctx, depth + 1));
        }
        return result;
    };

    // Arithmetic operators
    if(op == "+")
    {
        auto vals = getArgs();
        double sum = 0.0;
        for(const auto& v : vals)
        {
            sum += toDouble(v);
        }
        return sum;
    }

    if(op == "-")
    {
        auto vals = getArgs();
        if(vals.empty())
        {
            return 0.0;
        }
        if(vals.size() == 1)
        {
            return -toDouble(vals[0]);
        }
        double result = toDouble(vals[0]);
        for(size_t i = 1; i < vals.size(); ++i)
        {
            result -= toDouble(vals[i]);
        }
        return result;
    }

    if(op == "*")
    {
        auto vals = getArgs();
        double product = 1.0;
        for(const auto& v : vals)
        {
            product *= toDouble(v);
        }
        return product;
    }

    if(op == "/")
    {
        auto vals = getArgs();
        if(vals.size() < 2)
        {
            throw JsonLogicError("Division requires at least 2 arguments");
        }
        double result = toDouble(vals[0]);
        for(size_t i = 1; i < vals.size(); ++i)
        {
            const double divisor = toDouble(vals[i]);
            if(divisor == 0.0)
            {
                throw JsonLogicError("Division by zero");
            }
            result /= divisor;
        }
        return result;
    }

    if(op == "%")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("Modulo requires exactly 2 arguments");
        }
        const double a = toDouble(vals[0]);
        const double b = toDouble(vals[1]);
        if(b == 0.0)
        {
            throw JsonLogicError("Modulo by zero");
        }
        return std::fmod(a, b);
    }

    if(op == "ceil_div")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("ceil_div requires exactly 2 arguments");
        }
        const double a = toDouble(vals[0]);
        const double b = toDouble(vals[1]);
        if(b == 0.0)
        {
            throw JsonLogicError("Division by zero in ceil_div");
        }
        return std::ceil(a / b);
    }

    // Math operators
    if(op == "min")
    {
        auto vals = getArgs();
        if(vals.empty())
        {
            throw JsonLogicError("min requires at least 1 argument");
        }
        double result = toDouble(vals[0]);
        for(size_t i = 1; i < vals.size(); ++i)
        {
            result = std::min(result, toDouble(vals[i]));
        }
        return result;
    }

    if(op == "max")
    {
        auto vals = getArgs();
        if(vals.empty())
        {
            throw JsonLogicError("max requires at least 1 argument");
        }
        double result = toDouble(vals[0]);
        for(size_t i = 1; i < vals.size(); ++i)
        {
            result = std::max(result, toDouble(vals[i]));
        }
        return result;
    }

    if(op == "abs")
    {
        auto vals = getArgs();
        if(vals.size() != 1)
        {
            throw JsonLogicError("abs requires exactly 1 argument");
        }
        return std::abs(toDouble(vals[0]));
    }

    if(op == "pow")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("pow requires exactly 2 arguments");
        }
        return std::pow(toDouble(vals[0]), toDouble(vals[1]));
    }

    if(op == "log2")
    {
        auto vals = getArgs();
        if(vals.size() != 1)
        {
            throw JsonLogicError("log2 requires exactly 1 argument");
        }
        const double v = toDouble(vals[0]);
        if(v <= 0.0)
        {
            throw JsonLogicError("log2 of non-positive number");
        }
        return std::log2(v);
    }

    if(op == "rsqrt")
    {
        auto vals = getArgs();
        if(vals.size() != 1)
        {
            throw JsonLogicError("rsqrt requires exactly 1 argument");
        }
        const double v = toDouble(vals[0]);
        if(v <= 0.0)
        {
            throw JsonLogicError("rsqrt of non-positive number");
        }
        return 1.0 / std::sqrt(v);
    }

    // Comparison operators
    if(op == "==")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("== requires exactly 2 arguments");
        }
        return valuesEqual(vals[0], vals[1]);
    }

    if(op == "!=")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("!= requires exactly 2 arguments");
        }
        return !valuesEqual(vals[0], vals[1]);
    }

    if(op == "<")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("< requires exactly 2 arguments");
        }
        return toDouble(vals[0]) < toDouble(vals[1]);
    }

    if(op == "<=")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("<= requires exactly 2 arguments");
        }
        return toDouble(vals[0]) <= toDouble(vals[1]);
    }

    if(op == ">")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("> requires exactly 2 arguments");
        }
        return toDouble(vals[0]) > toDouble(vals[1]);
    }

    if(op == ">=")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError(">= requires exactly 2 arguments");
        }
        return toDouble(vals[0]) >= toDouble(vals[1]);
    }

    // Logical operators
    if(op == "and")
    {
        if(!args.is_array())
        {
            return toBool(evaluate(args, ctx, depth + 1));
        }
        for(const auto& arg : args)
        {
            if(!toBool(evaluate(arg, ctx, depth + 1)))
            {
                return false;
            }
        }
        return true;
    }

    if(op == "or")
    {
        if(!args.is_array())
        {
            return toBool(evaluate(args, ctx, depth + 1));
        }
        for(const auto& arg : args)
        {
            if(toBool(evaluate(arg, ctx, depth + 1)))
            {
                return true;
            }
        }
        return false;
    }

    if(op == "!")
    {
        auto vals = getArgs();
        if(vals.size() != 1)
        {
            throw JsonLogicError("! requires exactly 1 argument");
        }
        return !toBool(vals[0]);
    }

    // Control operators
    if(op == "if")
    {
        if(!args.is_array() || args.size() < 2)
        {
            throw JsonLogicError("if requires at least 2 arguments");
        }
        // if/then/else chains: [cond1, val1, cond2, val2, ..., default]
        for(size_t i = 0; i + 1 < args.size(); i += 2)
        {
            if(toBool(evaluate(args[i], ctx, depth + 1)))
            {
                return evaluate(args[i + 1], ctx, depth + 1);
            }
        }
        // If odd number of args, last is the default
        if(args.size() % 2 == 1)
        {
            return evaluate(args[args.size() - 1], ctx, depth + 1);
        }
        return false; // No default, all conditions false
    }

    if(op == "value_or_default")
    {
        if(!args.is_array() || args.size() != 2)
        {
            throw JsonLogicError("value_or_default requires exactly 2 arguments");
        }
        // Only an absent binding falls through to the default. Catching every
        // JsonLogicError here would turn a divide-by-zero, an arity mistake or a type
        // error into a silent default — RFC 0019 §7.2 requires those to fail closed.
        try
        {
            return evaluate(args[0], ctx, depth + 1);
        }
        catch(const UndefinedVariableError&)
        {
            return evaluate(args[1], ctx, depth + 1);
        }
    }

    // Membership operator: {"in": [value, [array]]}
    if(op == "in")
    {
        if(!args.is_array() || args.size() != 2)
        {
            throw JsonLogicError("in requires exactly 2 arguments");
        }
        const Value needle = evaluate(args[0], ctx, depth + 1);
        const auto& haystack = args[1];
        if(!haystack.is_array())
        {
            throw JsonLogicError("in requires array as second argument");
        }
        for(const auto& item : haystack)
        {
            if(valuesEqual(evaluate(item, ctx, depth + 1), needle))
            {
                return true;
            }
        }
        return false;
    }

    // Array predicate: {"all": [[array], {predicate using "$current"}]}
    if(op == "all")
    {
        if(!args.is_array() || args.size() != 2)
        {
            throw JsonLogicError("all requires exactly 2 arguments");
        }
        const auto& arr = args[0];
        const auto& predicate = args[1];
        if(!arr.is_array())
        {
            throw JsonLogicError("all requires array as first argument");
        }
        if(arr.empty())
        {
            return true; // Empty array satisfies "all"
        }
        for(const auto& item : arr)
        {
            // Evaluate the predicate in a copy of the context with $current bound to
            // this element. Both descents carry depth + 1: the predicate is
            // author-controlled and may itself nest "all", so omitting it here would
            // reset the counter at every level and leave the recursion unbounded.
            VariableContext tempCtx = ctx;
            tempCtx.bind("$current", toDouble(evaluate(item, ctx, depth + 1)));
            if(!toBool(evaluate(predicate, tempCtx, depth + 1)))
            {
                return false;
            }
        }
        return true;
    }

    // Shape accessor: {"shape": ["$tensor", dimension_index]}
    // Returns the size of a tensor dimension. Currently returns 0 if not bound.
    if(op == "shape")
    {
        if(!args.is_array() || args.size() != 2)
        {
            throw JsonLogicError("shape requires exactly 2 arguments");
        }
        // Shape access requires tensor metadata binding (not yet implemented)
        // For now, attempt to resolve as $tensor.shape[dim]
        //
        // Do NOT evaluate args[0] here. It names a tensor, not a scalar, so resolving
        // it as a variable throws "Undefined variable" for every reference that is not
        // separately bound as a number — which made the synthesis below unreachable and
        // the operator unusable for its stated purpose.
        const auto dim = static_cast<int>(toDouble(evaluate(args[1], ctx, depth + 1)));

        // Try to resolve $tensor.shape_N pattern
        if(args[0].is_string())
        {
            std::string tensorName = args[0].get<std::string>();
            if(!tensorName.empty() && tensorName[0] == '$')
            {
                const std::string shapeVar = tensorName + ".shape_" + std::to_string(dim);
                auto val = ctx.resolveDouble(shapeVar);
                if(val.has_value())
                {
                    return val.value();
                }
            }
        }
        throw JsonLogicError("shape: tensor shape not bound for dimension " + std::to_string(dim));
    }

    // Rank accessor: {"rank": "$tensor"}
    // Returns the number of dimensions of a tensor.
    if(op == "rank")
    {
        auto vals = getArgs();
        if(vals.size() != 1)
        {
            throw JsonLogicError("rank requires exactly 1 argument");
        }
        // Try to resolve $tensor.rank pattern
        if(args.is_array() && args.size() == 1 && args[0].is_string())
        {
            std::string tensorName = args[0].get<std::string>();
            if(!tensorName.empty() && tensorName[0] == '$')
            {
                const std::string rankVar = tensorName + ".rank";
                auto val = ctx.resolveDouble(rankVar);
                if(val.has_value())
                {
                    return val.value();
                }
            }
        }
        throw JsonLogicError("rank: tensor rank not bound");
    }

    // Divisibility check: {"divisible": [dividend, divisor]}
    // Returns true if dividend is evenly divisible by divisor.
    if(op == "divisible")
    {
        auto vals = getArgs();
        if(vals.size() != 2)
        {
            throw JsonLogicError("divisible requires exactly 2 arguments");
        }
        const double dividend = toDouble(vals[0]);
        const double divisor = toDouble(vals[1]);
        if(divisor == 0.0)
        {
            throw JsonLogicError("divisible: division by zero");
        }
        return std::fmod(dividend, divisor) == 0.0;
    }

    throw JsonLogicError("Unknown operator: " + op);
}

inline std::unordered_set<std::string> JsonLogicEvaluator::extractVariables(const nlohmann::json& expr)
{
    std::unordered_set<std::string> vars;

    std::function<void(const nlohmann::json&)> extract = [&](const nlohmann::json& e) {
        if(e.is_string())
        {
            std::string s = e.get<std::string>();
            // $current is bound by `all` for the duration of its predicate, so it is
            // never supplied by the caller. Reporting it would make it look like a
            // missing binding to getMissingVariables and the KMD coverage check.
            if(!s.empty() && s[0] == '$' && s != "$current")
            {
                vars.insert(s);
            }
        }
        else if(e.is_object())
        {
            for(auto it = e.begin(); it != e.end(); ++it)
            {
                extract(it.value());
            }
        }
        else if(e.is_array())
        {
            for(const auto& item : e)
            {
                extract(item);
            }
        }
    };

    extract(expr);
    return vars;
}

inline double JsonLogicEvaluator::toDouble(const Value& v)
{
    return std::visit(
        [](const auto& val) -> double {
            using T = std::decay_t<decltype(val)>;
            if constexpr(std::is_same_v<T, double>)
            {
                return val;
            }
            else if constexpr(std::is_same_v<T, bool>)
            {
                return val ? 1.0 : 0.0;
            }
            else
            {
                // RFC 0019 §7.2: fail closed on a type error. Returning NaN here was
                // silently wrong — a GBDT treats NaN as a missing value, routes it
                // down default_left and returns an ordinary leaf, so a string used as
                // a number was scored as data and never surfaced. Strings do not
                // implicitly convert, even when they happen to parse.
                throw JsonLogicError("Type error: string \"" + val
                                     + "\" cannot be used where a number is required");
            }
        },
        v);
}

inline bool JsonLogicEvaluator::toBool(const Value& v)
{
    return std::visit(
        [](const auto& val) -> bool {
            using T = std::decay_t<decltype(val)>;
            if constexpr(std::is_same_v<T, double>)
            {
                return val != 0.0 && !std::isnan(val);
            }
            else if constexpr(std::is_same_v<T, bool>)
            {
                return val;
            }
            else
            {
                return !val.empty();
            }
        },
        v);
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
