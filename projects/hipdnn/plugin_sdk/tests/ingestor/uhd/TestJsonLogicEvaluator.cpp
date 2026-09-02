// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestJsonLogicEvaluator.cpp
 * @brief Tests for the JsonLogic expression evaluator used by UHD.
 */

#include <hipdnn_plugin_sdk/ingestor/uhd/JsonLogicEvaluator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using hipdnn_plugin_sdk::ingestor::uhd::JsonLogicError;
using hipdnn_plugin_sdk::ingestor::uhd::JsonLogicEvaluator;
using hipdnn_plugin_sdk::ingestor::uhd::VariableContext;

namespace
{

class TestJsonLogicEvaluator : public ::testing::Test
{
protected:
    JsonLogicEvaluator _evaluator;
    VariableContext _ctx;
};

// ========== Literal values ==========

TEST_F(TestJsonLogicEvaluator, EvaluatesNumericLiteral)
{
    auto expr = JsonLogicEvaluator::parse("42.5");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 42.5);
}

TEST_F(TestJsonLogicEvaluator, EvaluatesBooleanLiteralTrue)
{
    auto expr = JsonLogicEvaluator::parse("true");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, EvaluatesBooleanLiteralFalse)
{
    auto expr = JsonLogicEvaluator::parse("false");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 0.0);
}

// ========== Variable resolution ==========

TEST_F(TestJsonLogicEvaluator, ResolvesDeviceVariable)
{
    _ctx.bind("$device.cu_count", 120.0);
    auto expr = JsonLogicEvaluator::parse("\"$device.cu_count\"");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 120.0);
}

TEST_F(TestJsonLogicEvaluator, ResolvesKernelVariable)
{
    _ctx.bind("$kernel.tile_m", 64.0);
    auto expr = JsonLogicEvaluator::parse("\"$kernel.tile_m\"");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 64.0);
}

TEST_F(TestJsonLogicEvaluator, ResolvesQueryVariable)
{
    _ctx.bind("$q.batch", 32.0);
    auto expr = JsonLogicEvaluator::parse("\"$q.batch\"");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 32.0);
}

TEST_F(TestJsonLogicEvaluator, ThrowsOnUndefinedVariable)
{
    auto expr = JsonLogicEvaluator::parse("\"$device.unknown\"");
    EXPECT_THROW(_evaluator.evaluateDouble(expr, _ctx), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, BindNamespacePopulatesVariables)
{
    const std::unordered_map<std::string, VariableContext::ValueType> deviceVars = {
        {"cu_count", 120.0},
        {"warp_size", int64_t{64}},
    };
    _ctx.bindNamespace("device", deviceVars);

    auto expr1 = JsonLogicEvaluator::parse("\"$device.cu_count\"");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr1, _ctx), 120.0);

    auto expr2 = JsonLogicEvaluator::parse("\"$device.warp_size\"");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr2, _ctx), 64.0);
}

// ========== Arithmetic operators ==========

TEST_F(TestJsonLogicEvaluator, AdditionTwoOperands)
{
    auto expr = JsonLogicEvaluator::parse(R"({"+": [10, 20]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 30.0);
}

TEST_F(TestJsonLogicEvaluator, AdditionMultipleOperands)
{
    auto expr = JsonLogicEvaluator::parse(R"({"+": [1, 2, 3, 4]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 10.0);
}

TEST_F(TestJsonLogicEvaluator, SubtractionTwoOperands)
{
    auto expr = JsonLogicEvaluator::parse(R"({"-": [100, 30]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 70.0);
}

TEST_F(TestJsonLogicEvaluator, SubtractionUnary)
{
    auto expr = JsonLogicEvaluator::parse(R"({"-": [42]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), -42.0);
}

TEST_F(TestJsonLogicEvaluator, Multiplication)
{
    auto expr = JsonLogicEvaluator::parse(R"({"*": [3, 4, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 60.0);
}

TEST_F(TestJsonLogicEvaluator, Division)
{
    auto expr = JsonLogicEvaluator::parse(R"({"/": [100, 4]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 25.0);
}

TEST_F(TestJsonLogicEvaluator, DivisionByZeroThrows)
{
    auto expr = JsonLogicEvaluator::parse(R"({"/": [100, 0]})");
    EXPECT_THROW(_evaluator.evaluateDouble(expr, _ctx), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, Modulo)
{
    auto expr = JsonLogicEvaluator::parse(R"({"%": [17, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 2.0);
}

TEST_F(TestJsonLogicEvaluator, CeilDiv)
{
    auto expr = JsonLogicEvaluator::parse(R"({"ceil_div": [17, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 4.0);
}

TEST_F(TestJsonLogicEvaluator, CeilDivExact)
{
    auto expr = JsonLogicEvaluator::parse(R"({"ceil_div": [20, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 4.0);
}

// ========== Math operators ==========

TEST_F(TestJsonLogicEvaluator, Min)
{
    auto expr = JsonLogicEvaluator::parse(R"({"min": [5, 3, 8, 1, 9]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, Max)
{
    auto expr = JsonLogicEvaluator::parse(R"({"max": [5, 3, 8, 1, 9]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 9.0);
}

TEST_F(TestJsonLogicEvaluator, Abs)
{
    auto expr = JsonLogicEvaluator::parse(R"({"abs": [-42]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 42.0);
}

TEST_F(TestJsonLogicEvaluator, Pow)
{
    auto expr = JsonLogicEvaluator::parse(R"({"pow": [2, 10]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1024.0);
}

TEST_F(TestJsonLogicEvaluator, Log2)
{
    auto expr = JsonLogicEvaluator::parse(R"({"log2": [1024]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 10.0);
}

TEST_F(TestJsonLogicEvaluator, Rsqrt)
{
    auto expr = JsonLogicEvaluator::parse(R"({"rsqrt": [4]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 0.5);
}

// ========== Comparison operators ==========

TEST_F(TestJsonLogicEvaluator, EqualTrue)
{
    auto expr = JsonLogicEvaluator::parse(R"({"==": [5, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, EqualFalse)
{
    auto expr = JsonLogicEvaluator::parse(R"({"==": [5, 3]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 0.0);
}

TEST_F(TestJsonLogicEvaluator, NotEqual)
{
    auto expr = JsonLogicEvaluator::parse(R"({"!=": [5, 3]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, LessThan)
{
    auto expr = JsonLogicEvaluator::parse(R"({"<": [3, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, LessThanOrEqual)
{
    auto expr = JsonLogicEvaluator::parse(R"({"<=": [5, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, GreaterThan)
{
    auto expr = JsonLogicEvaluator::parse(R"({">": [5, 3]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, GreaterThanOrEqual)
{
    auto expr = JsonLogicEvaluator::parse(R"({">=": [5, 5]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

// ========== Logical operators ==========

TEST_F(TestJsonLogicEvaluator, AndAllTrue)
{
    auto expr = JsonLogicEvaluator::parse(R"({"and": [true, true, true]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, AndOneFalse)
{
    auto expr = JsonLogicEvaluator::parse(R"({"and": [true, false, true]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 0.0);
}

TEST_F(TestJsonLogicEvaluator, OrAllFalse)
{
    auto expr = JsonLogicEvaluator::parse(R"({"or": [false, false, false]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 0.0);
}

TEST_F(TestJsonLogicEvaluator, OrOneTrue)
{
    auto expr = JsonLogicEvaluator::parse(R"({"or": [false, true, false]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

TEST_F(TestJsonLogicEvaluator, NotTrue)
{
    auto expr = JsonLogicEvaluator::parse(R"({"!": [true]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 0.0);
}

TEST_F(TestJsonLogicEvaluator, NotFalse)
{
    auto expr = JsonLogicEvaluator::parse(R"({"!": [false]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 1.0);
}

// ========== Control operators ==========

TEST_F(TestJsonLogicEvaluator, IfThenElse)
{
    auto expr = JsonLogicEvaluator::parse(R"({"if": [true, 42, 0]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 42.0);
}

TEST_F(TestJsonLogicEvaluator, IfThenElseFalseCondition)
{
    auto expr = JsonLogicEvaluator::parse(R"({"if": [false, 42, 99]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 99.0);
}

TEST_F(TestJsonLogicEvaluator, IfChain)
{
    // if cond1 then val1 elif cond2 then val2 else default
    auto expr = JsonLogicEvaluator::parse(R"({"if": [false, 1, false, 2, 3]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 3.0);
}

TEST_F(TestJsonLogicEvaluator, ValueOrDefaultResolved)
{
    _ctx.bind("$device.cu_count", 120.0);
    auto expr = JsonLogicEvaluator::parse(R"({"value_or_default": ["$device.cu_count", 64]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 120.0);
}

TEST_F(TestJsonLogicEvaluator, ValueOrDefaultFallback)
{
    auto expr = JsonLogicEvaluator::parse(R"({"value_or_default": ["$device.missing", 64]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 64.0);
}

// ========== Nested expressions ==========

TEST_F(TestJsonLogicEvaluator, NestedArithmetic)
{
    // (10 + 20) * 2 = 60
    auto expr = JsonLogicEvaluator::parse(R"({"*": [{"+": [10, 20]}, 2]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 60.0);
}

TEST_F(TestJsonLogicEvaluator, NestedWithVariables)
{
    _ctx.bind("$device.cu_count", 120.0);
    _ctx.bind("$kernel.tile_m", 64.0);
    // ceil_div($device.cu_count, $kernel.tile_m) = ceil(120/64) = 2
    auto expr
        = JsonLogicEvaluator::parse(R"({"ceil_div": ["$device.cu_count", "$kernel.tile_m"]})");
    EXPECT_DOUBLE_EQ(_evaluator.evaluateDouble(expr, _ctx), 2.0);
}

// ========== Variable extraction ==========

TEST_F(TestJsonLogicEvaluator, ExtractVariablesSimple)
{
    const auto expr = JsonLogicEvaluator::parse("\"$device.cu_count\"");
    const auto vars = JsonLogicEvaluator::extractVariables(expr);
    EXPECT_EQ(vars.size(), 1u);
    EXPECT_TRUE(vars.count("$device.cu_count") > 0);
}

TEST_F(TestJsonLogicEvaluator, ExtractVariablesNested)
{
    const auto expr = JsonLogicEvaluator::parse(
        R"({"*": ["$device.cu_count", {"+": ["$kernel.tile_m", "$q.batch"]}]})");
    const auto vars = JsonLogicEvaluator::extractVariables(expr);
    EXPECT_EQ(vars.size(), 3u);
    EXPECT_TRUE(vars.count("$device.cu_count") > 0);
    EXPECT_TRUE(vars.count("$kernel.tile_m") > 0);
    EXPECT_TRUE(vars.count("$q.batch") > 0);
}

// ========== Error handling ==========

TEST_F(TestJsonLogicEvaluator, InvalidJsonThrows)
{
    EXPECT_THROW(JsonLogicEvaluator::parse("{invalid json}"), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, UnknownOperatorThrows)
{
    auto expr = JsonLogicEvaluator::parse(R"({"unknown_op": [1, 2]})");
    EXPECT_THROW(_evaluator.evaluateDouble(expr, _ctx), JsonLogicError);
}

// ========== Bounded interpreter (RFC 0019 §7.2, §16) ==========

TEST_F(TestJsonLogicEvaluator, DeeplyNestedExpressionIsRejected)
{
    // A descriptor is author-controlled input, so the interpreter is bounded. Without
    // the depth limit this recurses until the stack gives out.
    std::string expr = "1";
    for(int i = 0; i < 512; ++i)
    {
        expr.insert(0, R"({"+": [)");
        expr.append(", 0]}");
    }

    const VariableContext ctx;
    const JsonLogicEvaluator eval;
    EXPECT_THROW(eval.evaluateDouble(JsonLogicEvaluator::parse(expr), ctx), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, DeeplyNestedAllPredicateIsRejected)
{
    // Regression: "all" evaluates its predicate in a rebuilt context, and that descent
    // used to be made with the default depth of 0 rather than depth + 1. Because the
    // counter reset at every level, nesting "all" inside its own predicate walked past
    // MAX_EXPRESSION_DEPTH unbounded — the one operator that escaped the §16 bound.
    // Nesting through the predicate position is what makes this fail before the fix;
    // DeeplyNestedExpressionIsRejected nests only "+" and passes either way.
    std::string expr = "true";
    for(int i = 0; i < 512; ++i)
    {
        expr.insert(0, R"({"all": [[1], )");
        expr.append("]}");
    }

    const VariableContext ctx;
    const JsonLogicEvaluator eval;
    EXPECT_THROW(eval.evaluateDouble(JsonLogicEvaluator::parse(expr), ctx), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, ModeratelyNestedAllPredicateStillEvaluates)
{
    // The tightened bound must not reject a legitimately nested "all".
    std::string expr = "true";
    for(int i = 0; i < 8; ++i)
    {
        expr.insert(0, R"({"all": [[1], )");
        expr.append("]}");
    }

    const VariableContext ctx;
    const JsonLogicEvaluator eval;
    EXPECT_NO_THROW(eval.evaluateDouble(JsonLogicEvaluator::parse(expr), ctx));
}

TEST_F(TestJsonLogicEvaluator, ModeratelyNestedExpressionStillEvaluates)
{
    // The bound must not reject expressions a real derived feature would use.
    std::string expr = "1";
    for(int i = 0; i < 16; ++i)
    {
        expr.insert(0, R"({"+": [)");
        expr.append(", 1]}");
    }

    const VariableContext ctx;
    const JsonLogicEvaluator eval;
    EXPECT_DOUBLE_EQ(eval.evaluateDouble(JsonLogicEvaluator::parse(expr), ctx), 17.0);
}

// ========== value_or_default only covers absent bindings (RFC 0019 §7.2) ==========

TEST_F(TestJsonLogicEvaluator, ValueOrDefaultSuppliesDefaultForUnboundVariable)
{
    const VariableContext ctx;
    const JsonLogicEvaluator eval;
    const auto expr = JsonLogicEvaluator::parse(R"({"value_or_default": ["$q.missing", 7]})");
    EXPECT_DOUBLE_EQ(eval.evaluateDouble(expr, ctx), 7.0);
}

TEST_F(TestJsonLogicEvaluator, ValueOrDefaultDoesNotSwallowDivideByZero)
{
    // §7.2 requires failing closed on an invalid operation. Catching every
    // JsonLogicError here would turn a genuine expression bug into a silent default.
    VariableContext ctx;
    ctx.bind("$q.n", 1.0);
    const JsonLogicEvaluator eval;
    const auto expr = JsonLogicEvaluator::parse(R"({"value_or_default": [{"/": ["$q.n", 0]}, 7]})");
    EXPECT_THROW(eval.evaluateDouble(expr, ctx), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, ValueOrDefaultDoesNotSwallowArityErrors)
{
    const VariableContext ctx;
    const JsonLogicEvaluator eval;
    const auto expr = JsonLogicEvaluator::parse(R"({"value_or_default": [{"pow": [2]}, 7]})");
    EXPECT_THROW(eval.evaluateDouble(expr, ctx), JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, ValueOrDefaultDoesNotSwallowTypeErrors)
{
    VariableContext ctx;
    ctx.bind("$q.name", std::string("gfx942"));
    const JsonLogicEvaluator eval;
    const auto expr = JsonLogicEvaluator::parse(R"({"value_or_default": ["$q.name", 7]})");
    EXPECT_THROW(eval.evaluateDouble(expr, ctx), JsonLogicError);
}

// ========== Comparisons are structural, not numeric coercion (RFC 0019 §7.2) ==========

TEST_F(TestJsonLogicEvaluator, EqualityComparesStringsAsStrings)
{
    // Routing both sides through toDouble made every string comparison NaN == NaN,
    // i.e. always false — so {"==": ["$kernel.dtype", "fp16"]} could never match.
    VariableContext ctx;
    ctx.bind("$kernel.dtype", std::string("fp16"));
    const JsonLogicEvaluator eval;

    EXPECT_TRUE(std::get<bool>(
        eval.evaluate(JsonLogicEvaluator::parse(R"({"==": ["$kernel.dtype", "fp16"]})"), ctx)));
    EXPECT_FALSE(std::get<bool>(
        eval.evaluate(JsonLogicEvaluator::parse(R"({"==": ["$kernel.dtype", "bf16"]})"), ctx)));
}

TEST_F(TestJsonLogicEvaluator, InequalityComparesStringsAsStrings)
{
    VariableContext ctx;
    ctx.bind("$kernel.dtype", std::string("fp16"));
    const JsonLogicEvaluator eval;

    EXPECT_TRUE(std::get<bool>(
        eval.evaluate(JsonLogicEvaluator::parse(R"({"!=": ["$kernel.dtype", "bf16"]})"), ctx)));
}

TEST_F(TestJsonLogicEvaluator, InMatchesStringMembership)
{
    VariableContext ctx;
    ctx.bind("$kernel.dtype", std::string("bf16"));
    const JsonLogicEvaluator eval;

    EXPECT_TRUE(std::get<bool>(eval.evaluate(
        JsonLogicEvaluator::parse(R"({"in": ["$kernel.dtype", ["fp16", "bf16", "fp32"]]})"), ctx)));
    EXPECT_FALSE(std::get<bool>(eval.evaluate(
        JsonLogicEvaluator::parse(R"({"in": ["$kernel.dtype", ["fp16", "fp32"]]})"), ctx)));
}

TEST_F(TestJsonLogicEvaluator, NumericComparisonStillWorks)
{
    VariableContext ctx;
    ctx.bind("$kernel.tile_m", 64.0);
    const JsonLogicEvaluator eval;

    EXPECT_TRUE(std::get<bool>(
        eval.evaluate(JsonLogicEvaluator::parse(R"({"==": ["$kernel.tile_m", 64]})"), ctx)));
    EXPECT_TRUE(std::get<bool>(eval.evaluate(
        JsonLogicEvaluator::parse(R"({"in": ["$kernel.tile_m", [32, 64, 128]]})"), ctx)));
}

TEST_F(TestJsonLogicEvaluator, ComparingStringAgainstNumberIsATypeError)
{
    VariableContext ctx;
    ctx.bind("$kernel.dtype", std::string("fp16"));
    const JsonLogicEvaluator eval;

    EXPECT_THROW(eval.evaluate(JsonLogicEvaluator::parse(R"({"==": ["$kernel.dtype", 64]})"), ctx),
                 JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, StringInArithmeticIsATypeError)
{
    // The type check lives in the numeric context, so a string is fine to compare but
    // never to compute with — and it does not silently become NaN.
    VariableContext ctx;
    ctx.bind("$kernel.dtype", std::string("fp16"));
    const JsonLogicEvaluator eval;

    EXPECT_THROW(
        eval.evaluateDouble(JsonLogicEvaluator::parse(R"({"+": ["$kernel.dtype", 1]})"), ctx),
        JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, NumericLookingStringDoesNotImplicitlyConvert)
{
    // "64" is a string. Parsing it would make the type system advisory.
    VariableContext ctx;
    ctx.bind("$kernel.label", std::string("64"));
    const JsonLogicEvaluator eval;

    EXPECT_THROW(
        eval.evaluateDouble(JsonLogicEvaluator::parse(R"({"+": ["$kernel.label", 1]})"), ctx),
        JsonLogicError);
}

TEST_F(TestJsonLogicEvaluator, CurrentIsNotReportedAsAnExternalVariable)
{
    // `all` binds $current for its predicate, so it is never caller-supplied.
    // Reporting it would look like a missing binding to getMissingVariables.
    const auto expr
        = JsonLogicEvaluator::parse(R"({"all": [[1, 2, 3], {">": ["$current", "$q.threshold"]}]})");
    const auto vars = JsonLogicEvaluator::extractVariables(expr);

    EXPECT_EQ(vars.count("$current"), 0u);
    EXPECT_EQ(vars.count("$q.threshold"), 1u);
}

} // namespace
