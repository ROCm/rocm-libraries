// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestJsonLogicEvaluator.cpp
 * @brief Tests for the JsonLogic expression evaluator used by UHD.
 */

#include "heuristics/uhd/JsonLogicEvaluator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using hipdnn_backend::heuristics::uhd::JsonLogicError;
using hipdnn_backend::heuristics::uhd::JsonLogicEvaluator;
using hipdnn_backend::heuristics::uhd::VariableContext;

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
    auto expr = JsonLogicEvaluator::parse(R"({"ceil_div": ["$device.cu_count", "$kernel.tile_m"]})");
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

} // namespace
