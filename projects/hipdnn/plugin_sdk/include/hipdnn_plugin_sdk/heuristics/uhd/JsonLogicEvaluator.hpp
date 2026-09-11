// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/heuristics/uhd/DescriptorExpression.hpp>

namespace hipdnn_plugin_sdk::uhd
{

using JsonLogicError = expression::Error;
using UndefinedVariableError = expression::UndefinedVariableError;
using VariableContext = expression::VariableContext;

/// One-shot convenience API over the descriptor compiler. Repeated consumers compile
/// an expression::Program once and keep a workspace, as FeatureExtractor does.
class JsonLogicEvaluator
{
public:
    using Value = std::variant<double, bool, std::string>;
    static constexpr size_t MAX_EXPRESSION_DEPTH = expression::Program::MAX_EXPRESSION_DEPTH;

    explicit JsonLogicEvaluator(expression::CategoricalEncoding categoricalEncoding = {})
        : _encoding(std::move(categoricalEncoding))
    {
    }
    static nlohmann::json parse(const std::string& text)
    {
        if(text.size() > 8 * 1024 * 1024)
        {
            throw JsonLogicError("Expression exceeds input-size bound");
        }
        try
        {
            return nlohmann::json::parse(
                text, [](int depth, nlohmann::json::parse_event_t, nlohmann::json&) {
                    if(depth > static_cast<int>(2 * MAX_EXPRESSION_DEPTH + 2))
                    {
                        throw JsonLogicError("Expression exceeds nesting bound");
                    }
                    return true;
                });
        }
        catch(const nlohmann::json::exception& error)
        {
            throw JsonLogicError("Failed to parse descriptor expression: "
                                 + std::string(error.what()));
        }
    }

    double evaluateDouble(const nlohmann::json& expr, const VariableContext& context) const
    {
        const expression::Program program({expr}, _encoding);
        auto work = program.workspace();
        return expression::Program::number(program.evaluate(0, context, work));
    }

    Value evaluate(const nlohmann::json& expr, const VariableContext& context) const
    {
        const expression::Program program({expr}, _encoding);
        auto work = program.workspace();
        const auto& value = program.evaluate(0, context, work);
        if(const auto* text = std::get_if<std::string_view>(&value.raw))
        {
            return std::string(*text);
        }
        if(const auto* boolean = std::get_if<bool>(&value.raw))
        {
            return *boolean;
        }
        return expression::Program::number(value);
    }

    static std::unordered_set<std::string> extractVariables(const nlohmann::json& expr)
    {
        return expression::Program({expr}).variables();
    }

private:
    expression::CategoricalEncoding _encoding;
};

} // namespace hipdnn_plugin_sdk::uhd
