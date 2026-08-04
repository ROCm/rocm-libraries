// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Exception thrown when JsonLogic evaluation fails.
class JsonLogicError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
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
    /// @returns The result as a double (for feature extraction).
    /// @throws JsonLogicError on evaluation failure.
    double evaluateDouble(const nlohmann::json& expr, const VariableContext& ctx) const;

    /// Evaluate an expression to a generic value.
    Value evaluate(const nlohmann::json& expr, const VariableContext& ctx) const;

    /// Extract all variable references from an expression.
    /// Returns variable names (e.g., "$device.cu_count", "$kernel.tile_m").
    static std::unordered_set<std::string> extractVariables(const nlohmann::json& expr);

private:
    Value evaluateOp(const std::string& op,
                     const nlohmann::json& args,
                     const VariableContext& ctx) const;

    static double toDouble(const Value& v);
    static bool toBool(const Value& v);
};

} // namespace hipdnn_backend::heuristics::uhd
