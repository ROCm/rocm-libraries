// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "JsonLogicEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace hipdnn_backend::heuristics::uhd
{

// ============================================================================
// VariableContext
// ============================================================================

void VariableContext::bind(const std::string& name, ValueType value)
{
    _bindings[name] = std::move(value);
}

void VariableContext::bindNamespace(const std::string& ns,
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

void VariableContext::clearNamespace(const std::string& ns)
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

std::optional<double> VariableContext::resolveDouble(const std::string& name) const
{
    auto it = _bindings.find(name);
    if(it == _bindings.end())
    {
        return std::nullopt;
    }

    return std::visit(
        [](const auto& v) -> double {
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
                return std::numeric_limits<double>::quiet_NaN();
            }
        },
        it->second);
}

std::optional<VariableContext::ValueType> VariableContext::resolve(const std::string& name) const
{
    auto it = _bindings.find(name);
    if(it == _bindings.end())
    {
        return std::nullopt;
    }
    return it->second;
}

bool VariableContext::has(const std::string& name) const
{
    return _bindings.find(name) != _bindings.end();
}

void VariableContext::clear()
{
    _bindings.clear();
}

// ============================================================================
// JsonLogicEvaluator
// ============================================================================

nlohmann::json JsonLogicEvaluator::parse(const std::string& jsonStr)
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

double JsonLogicEvaluator::evaluateDouble(const nlohmann::json& expr,
                                          const VariableContext& ctx) const
{
    return toDouble(evaluate(expr, ctx));
}

JsonLogicEvaluator::Value JsonLogicEvaluator::evaluate(const nlohmann::json& expr,
                                                       const VariableContext& ctx,
                                                       size_t depth) const
{
    // Bound the descent (RFC 0019 §7.2 "safe, bounded interpreter"; §16 lists the
    // descriptor as author-controlled input). Without this a deeply nested expression
    // is a stack overflow, which the tree-walk bound in TreeDataAdapter does not cover.
    if(depth > kMaxExpressionDepth)
    {
        throw JsonLogicError("JsonLogic expression exceeds the maximum nesting depth of " +
                             std::to_string(kMaxExpressionDepth));
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
                    if constexpr(std::is_same_v<T, std::string>)
                    {
                        return v;
                    }
                    else if constexpr(std::is_same_v<T, bool>)
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

bool JsonLogicEvaluator::valuesEqual(const Value& a, const Value& b)
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

JsonLogicEvaluator::Value JsonLogicEvaluator::evaluateOp(const std::string& op,
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
        // Note: $current binding would require context mutation;
        // for now, we support literal arrays only
        for(const auto& item : arr)
        {
            // Create temporary context with $current bound
            VariableContext tempCtx = ctx;
            tempCtx.bind("$current", toDouble(evaluate(item, ctx, depth + 1)));
            if(!toBool(evaluate(predicate, tempCtx)))
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

std::unordered_set<std::string> JsonLogicEvaluator::extractVariables(const nlohmann::json& expr)
{
    std::unordered_set<std::string> vars;

    std::function<void(const nlohmann::json&)> extract = [&](const nlohmann::json& e) {
        if(e.is_string())
        {
            std::string s = e.get<std::string>();
            if(!s.empty() && s[0] == '$')
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

double JsonLogicEvaluator::toDouble(const Value& v)
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
                throw JsonLogicError("Type error: string \"" + val +
                                     "\" cannot be used where a number is required");
            }
        },
        v);
}

bool JsonLogicEvaluator::toBool(const Value& v)
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

} // namespace hipdnn_backend::heuristics::uhd
