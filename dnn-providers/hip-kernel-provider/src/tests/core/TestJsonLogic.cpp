// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "hip_kernel_provider_common/JsonDataSource.hpp"
#include "hip_kernel_provider_common/JsonLogic.hpp"

namespace jlogic = hip_kernel_provider_common::jsonlogic;

using json = nlohmann::json;
using V = jlogic::Value;

namespace
{
const jlogic::JsonDataSource D{json{{"x", 41},
                                    {"y", 8},
                                    {"name", "amd"},
                                    {"nested", {{"a", {{"b", 7}}}}},
                                    {"arr", {10, 20, 30}},
                                    {"grid", {{1, 2}, {3, 4}}},
                                    {"rows", {{{"name", "a0"}}, {{"name", "a1"}}}},
                                    {"flag", true},
                                    {"zero", 0}}};

// Compile a rule and evaluate it against the shared document D.
V eval(const json& rule)
{
    return jlogic::compile<jlogic::JsonDataSource>(rule)(D);
}
} // namespace

TEST(TestJsonLogic, Literals)
{
    EXPECT_EQ(eval(42), V(42));
    EXPECT_EQ(eval("hello"), V("hello"));
    EXPECT_EQ(eval(true), V(true));
    EXPECT_EQ(eval(json::array({1, 2, 3})), V(V::Array{V(1), V(2), V(3)}));
}

TEST(TestJsonLogic, VarStockForm)
{
    EXPECT_EQ(eval(json({{"var", "x"}})), V(41));
    EXPECT_EQ(eval(json({{"var", "nested.a.b"}})), V(7));
    EXPECT_EQ(eval(json({{"var", "arr.1"}})), V(20));
    EXPECT_EQ(eval(json({{"var", "nope"}})), V());
    EXPECT_EQ(eval(json({{"var", json::array({"nope", 99})}})), V(99));
    EXPECT_EQ(eval(json({{"var", json::array({"x", 99})}})), V(41));
    EXPECT_EQ(eval(json({{"var", "arr"}})), V(V::Array{V(10), V(20), V(30)}));
}

TEST(TestJsonLogic, InlineVariables)
{
    EXPECT_EQ(eval("$x"), V(41));
    EXPECT_EQ(eval("$nested.a.b"), V(7));
    EXPECT_EQ(eval("$arr.2"), V(30));
    EXPECT_EQ(eval("$nope"), V());
    EXPECT_EQ(eval("$$literal"), V("$literal"));
    EXPECT_EQ(eval(json({{"+", json::array({"$x", "$y"})}})), V(49));
    EXPECT_EQ(eval(json({{"+", json::array({"$x", 1})}})), V(42));
    EXPECT_EQ(eval(json({{"==", json::array({"amd", "$name"})}})), V(true));
}

TEST(TestJsonLogic, SubscriptIndex)
{
    // [N] subscript, stock and inline forms
    EXPECT_EQ(eval(json({{"var", "arr[0]"}})), V(10));
    EXPECT_EQ(eval("$arr[2]"), V(30));
    // subscript and dotted keys mixed
    EXPECT_EQ(eval("$rows[1].name"), V("a1"));
    // chained subscripts into a nested array
    EXPECT_EQ(eval("$grid[0][1]"), V(2));
    EXPECT_EQ(eval("$grid[1][0]"), V(3));
    // dot-form array index still resolves
    EXPECT_EQ(eval("$arr.1"), V(20));
    // out-of-range, non-numeric, and subscript-on-non-array resolve to null
    EXPECT_EQ(eval("$arr[9]"), V());
    EXPECT_EQ(eval("$arr[x]"), V());
    EXPECT_EQ(eval("$nested[0]"), V());
}

TEST(TestJsonLogic, Arithmetic)
{
    EXPECT_EQ(eval(json({{"+", json::array({1, 2, 3})}})), V(6));
    EXPECT_EQ(eval(json({{"+", json::array({"2", "3"})}})), V(5));
    EXPECT_EQ(eval(json({{"-", json::array({10, 3})}})), V(7));
    EXPECT_EQ(eval(json({{"-", json::array({5})}})), V(-5));
    EXPECT_EQ(eval(json({{"*", json::array({2, 3, 4})}})), V(24));
    EXPECT_EQ(eval(json({{"/", json::array({12, 4})}})), V(3));
    EXPECT_EQ(eval(json({{"%", json::array({10, 3})}})), V(1));
    EXPECT_EQ(eval(json({{"min", json::array({3, 1, 2})}})), V(1));
    EXPECT_EQ(eval(json({{"max", json::array({"$x", "$y", 100})}})), V(100));
    EXPECT_EQ(eval(json({{"+", json::array({{{"var", "x"}}, 1})}})), V(42));
}

TEST(TestJsonLogic, ComparisonIsStrict)
{
    EXPECT_EQ(eval(json({{"==", json::array({1, 1})}})), V(true));
    EXPECT_EQ(eval(json({{"==", json::array({1, 1.0})}})), V(true));
    EXPECT_EQ(eval(json({{"==", json::array({1, "1"})}})), V(false)); // no coercion
    EXPECT_EQ(eval(json({{"==", json::array({"amd", "amd"})}})), V(true));
    EXPECT_EQ(eval(json({{"!=", json::array({1, 2})}})), V(true));
    EXPECT_EQ(eval(json({{"!=", json::array({1, "1"})}})), V(true)); // no coercion
    EXPECT_EQ(eval(json({{">", json::array({"$x", "$y"})}})), V(true));
    EXPECT_EQ(eval(json({{">=", json::array({5, 5})}})), V(true));
    EXPECT_EQ(eval(json({{"<", json::array({1, 2})}})), V(true));
    EXPECT_EQ(eval(json({{"<=", json::array({2, 2})}})), V(true));
    EXPECT_EQ(eval(json({{"<", json::array({1, 2, 3})}})), V(true));
    EXPECT_EQ(eval(json({{"<", json::array({1, 5, 3})}})), V(false));
    EXPECT_EQ(eval(json({{"<=", json::array({1, 1, 3})}})), V(true));
    EXPECT_EQ(eval(json({{"<", json::array({"abc", "abd"})}})), V(true));
}

TEST(TestJsonLogic, TruthinessAndLogic)
{
    EXPECT_EQ(eval(json({{"!", 0}})), V(true));
    EXPECT_EQ(eval(json({{"!", 1}})), V(false));
    EXPECT_EQ(eval(json({{"!", json::array({json::array()})}})), V(true));
    EXPECT_EQ(eval(json({{"!!", "hello"}})), V(true));
    EXPECT_EQ(eval(json({{"and", json::array({true, 1, "x"})}})), V("x"));
    EXPECT_EQ(eval(json({{"and", json::array({true, 0, "x"})}})), V(0));
    EXPECT_EQ(eval(json({{"or", json::array({0, "", "hit"})}})), V("hit"));
    EXPECT_EQ(eval(json({{"or", json::array({0, ""})}})), V(""));
}

TEST(TestJsonLogic, IfTernary)
{
    EXPECT_EQ(eval(json({{"if", json::array({true, "yes", "no"})}})), V("yes"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "yes", "no"})}})), V("no"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "a", true, "b", "c"})}})), V("b"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "a"})}})), V());
    EXPECT_EQ(eval(json({{"if", json::array({{{">", json::array({"$x", "$y"})}}, "$x", "$y"})}})),
              V(41));
}

TEST(TestJsonLogic, Composed)
{
    EXPECT_EQ(
        eval(json({{"if",
                    json::array({{{"<", json::array({{{"+", json::array({"$x", "$y"})}}, 50})}},
                                 "small",
                                 "big"})}})),
        V("small"));
}

TEST(TestJsonLogic, Membership)
{
    EXPECT_EQ(eval(json({{"in", json::array({20, "$arr"})}})), V(true));
    EXPECT_EQ(eval(json({{"in", json::array({99, "$arr"})}})), V(false));
    EXPECT_EQ(eval(json({{"in", json::array({"$x", json::array({40, 41, 42})})}})), V(true));
    // strict element equality: "41" != 41
    EXPECT_EQ(eval(json({{"in", json::array({"41", json::array({40, 41, 42})})}})), V(false));
    EXPECT_EQ(eval(json({{"in", json::array({"m", "$name"})}})), V(true));
    EXPECT_EQ(eval(json({{"in", json::array({"z", "$name"})}})), V(false));
}

TEST(TestJsonLogic, MathExtensions)
{
    EXPECT_EQ(eval(json({{"ceil_div", json::array({100, 16})}})), V(7));
    EXPECT_EQ(eval(json({{"ceil_div", json::array({32, 16})}})), V(2));
    EXPECT_EQ(eval(json({{"abs", -5}})), V(5));
    EXPECT_EQ(eval(json({{"abs", 5}})), V(5));
    EXPECT_EQ(eval(json({{"pow", json::array({2, 10})}})), V(1024));
    EXPECT_EQ(eval(json({{"log2", 8}})), V(3));
    EXPECT_EQ(eval(json({{"rsqrt", 4}})), V(0.5));
}

TEST(TestJsonLogic, ValueOrDefault)
{
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$x", 99})}})), V(41));
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$nope", 99})}})), V(99));
    // keys on existence, not truthiness: present 0 is returned
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$zero", 99})}})), V(0));
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$nested.a.b", -1})}})), V(7));
    // default is itself an expression, evaluated lazily
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$nope", "$x"})}})), V(41));
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$nope", "fallback"})}})),
              V("fallback"));
}

TEST(TestJsonLogic, PresenceOperators)
{
    // A resolving path is present; an unresolved one is not.
    EXPECT_EQ(eval(json({{"present", json::array({"$x"})}})), V(true));
    EXPECT_EQ(eval(json({{"present", json::array({"$nope"})}})), V(false));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$nope"})}})), V(true));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$x"})}})), V(false));
    // Presence keys on existence, not truthiness: a present 0 / false is present.
    EXPECT_EQ(eval(json({{"present", json::array({"$zero"})}})), V(true));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$zero"})}})), V(false));
    // Unlike every other operator, an unresolved path yields a real boolean
    // rather than propagating null, so the criterion decides instead of declining.
    EXPECT_TRUE(eval(json({{"present", json::array({"$nope"})}})).isBool());
    EXPECT_TRUE(eval(json({{"not_present", json::array({"$nope"})}})).isBool());
    // n-ary: both fold with `and` over every argument.
    EXPECT_EQ(eval(json({{"present", json::array({"$x", "$y", "$name"})}})), V(true));
    EXPECT_EQ(eval(json({{"present", json::array({"$x", "$nope"})}})), V(false));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$nope", "$missing"})}})), V(true));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$nope", "$x"})}})), V(false));
    // unary sugar (a bare argument, not an array)
    EXPECT_EQ(eval(json({{"present", "$x"}})), V(true));
    // at least one argument is required
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"present", json::array()}})),
                 jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"not_present", json::array()}})),
                 jlogic::JsonLogicCompileError);
}

TEST(TestJsonLogic, NullPropagatesThroughEveryOtherOperator)
{
    // An unresolved reference is "unknown", not a value. Every operator except
    // the presence pair and value_or_default must yield null rather than
    // coerce, because a coerced null reads as false / 0 / not-equal and would
    // make a narrowing check silently PASS on data it never saw.
    for(const json& rule : {json({{"!", "$nope"}}),
                            json({{"!!", "$nope"}}),
                            json({{"!=", json::array({"$nope", 1})}}),
                            json({{"==", json::array({"$nope", 1})}}),
                            json({{"<", json::array({"$nope", 5})}}),
                            json({{"<=", json::array({"$nope", 5})}}),
                            json({{">", json::array({"$nope", 5})}}),
                            json({{">=", json::array({"$nope", 0})}}),
                            json({{"+", json::array({"$nope", 1})}}),
                            json({{"*", json::array({"$nope", 1})}}),
                            json({{"-", json::array({"$nope", 1})}}),
                            json({{"in", json::array({"$nope", json::array({1, 2})})}}),
                            json({{"ceil_div", json::array({"$nope", 4})}}),
                            json({{"abs", "$nope"}}),
                            json({{"min", json::array({"$nope", 1})}}),
                            json({{"max", json::array({"$nope", 1})}}),
                            json({{"if", json::array({"$nope", 1, 2})}})})
    {
        EXPECT_TRUE(eval(rule).isNull()) << rule.dump();
    }
    // Two unresolved references are not "equal": the question is unanswerable.
    EXPECT_TRUE(eval(json({{"==", json::array({"$nope", "$missing"})}})).isNull());
    EXPECT_TRUE(eval(json({{"!=", json::array({"$nope", "$missing"})}})).isNull());
}

TEST(TestJsonLogic, KleeneAndOrShortCircuitPastUnknown)
{
    // A definite false decides an `and` even beside an unknown, and a definite
    // true decides an `or`. This is what lets "absent, or present and
    // constrained" accept an absent operand whose field checks cannot run.
    EXPECT_EQ(eval(json({{"and", json::array({false, "$nope"})}})), V(false));
    EXPECT_EQ(eval(json({{"or", json::array({true, "$nope"})}})), V(true));
    // Without a decisive argument the result stays unknown rather than
    // collapsing to true/false.
    EXPECT_TRUE(eval(json({{"and", json::array({true, "$nope"})}})).isNull());
    EXPECT_TRUE(eval(json({{"or", json::array({false, "$nope"})}})).isNull());
    // A fully-resolved expression is unaffected.
    EXPECT_EQ(eval(json({{"and", json::array({true, true})}})), V(true));
    EXPECT_EQ(eval(json({{"or", json::array({false, false})}})), V(false));
}

TEST(TestJsonLogic, DivisionAndDomainErrorsFailClosed)
{
    // A zero divisor declines instead of yielding inf/NaN, giving uniform
    // fail-closed zero-guarding (RFC 0018 A.7).
    EXPECT_TRUE(eval(json({{"/", json::array({"$x", 0})}})).isNull());
    EXPECT_TRUE(eval(json({{"%", json::array({"$x", 0})}})).isNull());
    EXPECT_TRUE(eval(json({{"ceil_div", json::array({"$x", 0})}})).isNull());
    // log2/rsqrt decline on a non-positive argument rather than returning
    // -inf/NaN.
    EXPECT_TRUE(eval(json({{"log2", 0}})).isNull());
    EXPECT_TRUE(eval(json({{"rsqrt", 0}})).isNull());
    EXPECT_TRUE(eval(json({{"rsqrt", -4}})).isNull());
    // The well-behaved cases still compute.
    EXPECT_EQ(eval(json({{"/", json::array({8, 2})}})), V(4));
    EXPECT_EQ(eval(json({{"log2", 8}})), V(3));
}

TEST(TestJsonLogic, Umd0018ConstraintShapes)
{
    EXPECT_EQ(eval(json({{"in", json::array({"$name", json::array({"amd", "xilinx"})})}})),
              V(true));
    EXPECT_EQ(eval(json({{"==", json::array({"$arr", json::array({10, 20, 30})})}})), V(true));
    // 41 % 8 == 1
    EXPECT_EQ(eval(json({{"==", json::array({{{"%", json::array({"$x", "$y"})}}, 1})}})), V(true));
    // ceil(41/16) == 3
    EXPECT_EQ(eval(json({{"ceil_div", json::array({"$x", 16})}})), V(3));
}

TEST(TestJsonLogic, CompileOnceReuseAcrossData)
{
    const auto expr
        = jlogic::compile<jlogic::JsonDataSource>(json({{"*", json::array({"$x", 2})}}));
    const jlogic::JsonDataSource a{json{{"x", 3}}};
    const jlogic::JsonDataSource b{json{{"x", 10}}};
    EXPECT_EQ(expr(a), V(6));
    EXPECT_EQ(expr(b), V(20));
}

TEST(TestJsonLogic, MalformedRulesThrowAtCompileTime)
{
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"nope", json::array({1, 2})}})),
                 jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"/", json::array({1, 2, 3})}})),
                 jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"abs", json::array({1, 2})}})),
                 jlogic::JsonLogicCompileError);
}

TEST(TestJsonLogic, VariablesCollectsReferencedPaths)
{
    using S = std::set<std::string>;
    // Keep the Expression alive while draining the borrowed range into a set.
    const auto vars = [](const json& rule) {
        const auto expr = jlogic::compile<jlogic::JsonDataSource>(rule);
        return S(expr.variables().begin(), expr.variables().end());
    };
    // inline vars across an op
    EXPECT_EQ(vars(json({{"+", json::array({"$x", "$y"})}})), (S{"x", "y"}));
    // stock var form keeps the dotted path verbatim
    EXPECT_EQ(vars(json({{"var", "nested.a.b"}})), (S{"nested.a.b"}));
    // a var default subtree contributes its own referenced vars
    EXPECT_EQ(vars(json({{"var", json::array({"nope", "$y"})}})), (S{"nope", "y"}));
    // nested composition reaches every leaf var
    EXPECT_EQ(
        vars(json({{"if", json::array({{{">", json::array({"$x", "$y"})}}, "$name", "$zero"})}})),
        (S{"x", "y", "name", "zero"}));
    // literal-only expressions reference nothing
    EXPECT_EQ(vars(json({{"+", json::array({1, 2})}})), (S{}));
    EXPECT_TRUE(vars(json(42)).empty());
}

TEST(TestJsonLogic, ReferencesVariableRootMatchesFirstToken)
{
    const auto refs = [](const json& rule, std::string_view root) {
        const auto expr = jlogic::compile<jlogic::JsonDataSource>(rule);
        return jlogic::referencesVariableRoot(expr, root);
    };
    // matches on the first path token, before any '.' or '[' separator
    EXPECT_TRUE(refs(json({{"<", json::array({"$kernel.tile_m", "$device.lds_size"})}}), "kernel"));
    EXPECT_TRUE(refs(json({{"<", json::array({"$kernel.tile_m", "$device.lds_size"})}}), "device"));
    EXPECT_TRUE(refs(json({{"==", json::array({"$kernel.vec[0]", 1})}}), "kernel"));
    // a bare root (no field) still matches its own token
    EXPECT_TRUE(refs(json("$kernel"), "kernel"));
    // no variable has that root
    EXPECT_FALSE(refs(json({{"==", json::array({"$q.head_size", 128})}}), "kernel"));
    // a root is a whole-token match, not a prefix
    EXPECT_FALSE(refs(json("$kernelish.x"), "kernel"));
    // literal-only expression references nothing
    EXPECT_FALSE(refs(json({{"+", json::array({1, 2})}}), "kernel"));
}

TEST(TestJsonLogic, VariablesRangeIsLazyAndKeepsDuplicates)
{
    // range-for yields every occurrence, duplicates included
    const auto expr
        = jlogic::compile<jlogic::JsonDataSource>(json({{"+", json::array({"$x", "$x", "$y"})}}));
    std::vector<std::string> seen;
    for(const std::string& v : expr.variables())
    {
        seen.push_back(v);
    }
    EXPECT_EQ(seen.size(), 3u);
    EXPECT_EQ(std::count(seen.begin(), seen.end(), std::string("x")), 2);
    EXPECT_EQ(std::count(seen.begin(), seen.end(), std::string("y")), 1);

    // empty range: begin() == end(), the body never runs
    const auto lit = jlogic::compile<jlogic::JsonDataSource>(json(42));
    EXPECT_TRUE(lit.variables().begin() == lit.variables().end());

    // composes with STL algorithms over the borrowed references
    const auto r = expr.variables();
    EXPECT_EQ(std::distance(r.begin(), r.end()), 3);
    EXPECT_TRUE(std::any_of(r.begin(), r.end(), [](const std::string& s) { return s == "y"; }));
}

TEST(TestJsonLogic, WholeDocumentAndDynamicKeysRejected)
{
    // whole-document references: inline sigil, empty var, empty array, null
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json("$")), jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"var", ""}})),
                 jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"var", json::array()}})),
                 jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(json({{"var", nullptr}})),
                 jlogic::JsonLogicCompileError);
    // computed / dynamic keys (object key, or computed first array element)
    EXPECT_THROW(
        jlogic::compile<jlogic::JsonDataSource>(json({{"var", {{"+", json::array({1, 2})}}}})),
        jlogic::JsonLogicCompileError);
    EXPECT_THROW(jlogic::compile<jlogic::JsonDataSource>(
                     json({{"var", json::array({{{"+", json::array({1, 2})}}, 0})}})),
                 jlogic::JsonLogicCompileError);
}
