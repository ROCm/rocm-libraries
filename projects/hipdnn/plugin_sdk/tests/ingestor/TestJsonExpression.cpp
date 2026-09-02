// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

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

#include <hipdnn_plugin_sdk/ingestor/JsonDataSource.hpp>
#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>

namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

using json = nlohmann::json;
using V = jexpr::Value;

namespace
{
const jexpr::JsonDataSource D{json{{"x", 41},
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
    return jexpr::compile<jexpr::JsonDataSource>(rule)(D);
}
} // namespace

TEST(TestJsonExpression, Literals)
{
    EXPECT_EQ(eval(42), V(42));
    EXPECT_EQ(eval("hello"), V("hello"));
    EXPECT_EQ(eval(true), V(true));
    EXPECT_EQ(eval(json::array({1, 2, 3})), V(V::Array{V(1), V(2), V(3)}));
}

TEST(TestJsonExpression, VarOperatorRejected)
{
    // `var` is not an operator: a variable is only ever a sigil-prefixed
    // string. Rejecting it at compile time keeps a rule written against stock
    // JsonLogic from silently reading as an unknown-operator error, and stops
    // two spellings of the same thing from coexisting.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"var", "x"}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"var", json::array({"nope", 99})}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(
        jexpr::compile<jexpr::JsonDataSource>(json({{"+", json::array({{{"var", "x"}}, 1})}})),
        jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, InlineVariables)
{
    EXPECT_EQ(eval("$x"), V(41));
    EXPECT_EQ(eval("$nested.a.b"), V(7));
    EXPECT_EQ(eval("$arr.2"), V(30));
    EXPECT_EQ(eval("$arr"), V(V::Array{V(10), V(20), V(30)}));
    EXPECT_EQ(eval("$nope"), V());
    EXPECT_EQ(eval("$$literal"), V("$literal"));
    EXPECT_EQ(eval(json({{"+", json::array({"$x", "$y"})}})), V(49));
    EXPECT_EQ(eval(json({{"+", json::array({"$x", 1})}})), V(42));
    EXPECT_EQ(eval(json({{"==", json::array({"amd", "$name"})}})), V(true));
}

TEST(TestJsonExpression, SubscriptIndex)
{
    EXPECT_EQ(eval("$arr[0]"), V(10));
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

TEST(TestJsonExpression, Arithmetic)
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
}

TEST(TestJsonExpression, ComparisonIsStrict)
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

TEST(TestJsonExpression, TruthinessAndLogic)
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

TEST(TestJsonExpression, IfTernary)
{
    EXPECT_EQ(eval(json({{"if", json::array({true, "yes", "no"})}})), V("yes"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "yes", "no"})}})), V("no"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "a", true, "b", "c"})}})), V("b"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "a"})}})), V());
    EXPECT_EQ(eval(json({{"if", json::array({{{">", json::array({"$x", "$y"})}}, "$x", "$y"})}})),
              V(41));
}

TEST(TestJsonExpression, Composed)
{
    EXPECT_EQ(
        eval(json({{"if",
                    json::array({{{"<", json::array({{{"+", json::array({"$x", "$y"})}}, 50})}},
                                 "small",
                                 "big"})}})),
        V("small"));
}

TEST(TestJsonExpression, Membership)
{
    EXPECT_EQ(eval(json({{"in", json::array({20, "$arr"})}})), V(true));
    EXPECT_EQ(eval(json({{"in", json::array({99, "$arr"})}})), V(false));
    EXPECT_EQ(eval(json({{"in", json::array({"$x", json::array({40, 41, 42})})}})), V(true));
    // strict element equality: "41" != 41
    EXPECT_EQ(eval(json({{"in", json::array({"41", json::array({40, 41, 42})})}})), V(false));
    EXPECT_EQ(eval(json({{"in", json::array({"m", "$name"})}})), V(true));
    EXPECT_EQ(eval(json({{"in", json::array({"z", "$name"})}})), V(false));
}

TEST(TestJsonExpression, MathExtensions)
{
    EXPECT_EQ(eval(json({{"ceil_div", json::array({100, 16})}})), V(7));
    EXPECT_EQ(eval(json({{"ceil_div", json::array({32, 16})}})), V(2));
    EXPECT_EQ(eval(json({{"abs", -5}})), V(5));
    EXPECT_EQ(eval(json({{"abs", 5}})), V(5));
    EXPECT_EQ(eval(json({{"pow", json::array({2, 10})}})), V(1024));
    EXPECT_EQ(eval(json({{"log2", 8}})), V(3));
    EXPECT_EQ(eval(json({{"rsqrt", 4}})), V(0.5));
}

TEST(TestJsonExpression, Divisible)
{
    EXPECT_EQ(eval(json({{"divisible", json::array({32, 16})}})), V(true));
    EXPECT_EQ(eval(json({{"divisible", json::array({100, 16})}})), V(false));
    EXPECT_EQ(eval(json({{"divisible", json::array({0, 16})}})), V(true));
    // negative operands divide the same way fmod does
    EXPECT_EQ(eval(json({{"divisible", json::array({-32, 16})}})), V(true));
    EXPECT_EQ(eval(json({{"divisible", json::array({32, -16})}})), V(true));
    // the tile-fit shape the RFCs write: a product against a kernel constant
    EXPECT_EQ(eval(json({{"divisible", json::array({{{"*", json::array({"$y", 4})}}, 16})}})),
              V(true));
    // a zero divisor declines, exactly as `%` does
    EXPECT_EQ(eval(json({{"divisible", json::array({32, 0})}})), V());
    // an unresolved operand propagates rather than reading as divisible
    EXPECT_EQ(eval(json({{"divisible", json::array({"$nope", 16})}})), V());
    EXPECT_EQ(eval(json({{"divisible", json::array({32, "$nope"})}})), V());
}

TEST(TestJsonExpression, DivisibleMatchesModuloLonghand)
{
    // `divisible` is a short-hand for {"==": [{"%": [a, b]}, 0]}; the RFCs use
    // both spellings for the same check, so they must agree on every input.
    for(const int a : {0, 1, 7, 16, 32, 100, -32, -7})
    {
        for(const int b : {1, 2, 16, -16, 0})
        {
            const json args = json::array({a, b});
            const V shorthand = eval(json({{"divisible", args}}));
            const V longhand = eval(json({{"==", json::array({{{"%", args}}, 0})}}));
            EXPECT_EQ(shorthand, longhand) << "a=" << a << " b=" << b;
        }
    }
}

TEST(TestJsonExpression, LayoutAliasesExpandToCanonicalArrays)
{
    const jexpr::JsonDataSource src{
        json{{"x", {{"rank", 4}, {"stride_order", {3, 0, 2, 1}}}},
             {"y", {{"rank", 4}, {"stride_order", {3, 2, 1, 0}}}},
             {"vol", {{"rank", 5}, {"stride_order", {4, 0, 3, 2, 1}}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    // Every alias in RFC 0018 A.4, against a tensor that has that layout.
    EXPECT_EQ(ev(json({{"==", json::array({"$x.stride_order", "nhwc"})}})), V(true));
    EXPECT_EQ(ev(json({{"==", json::array({"$y.stride_order", "nchw"})}})), V(true));
    EXPECT_EQ(ev(json({{"==", json::array({"$y.stride_order", "bhsd"})}})), V(true));
    EXPECT_EQ(ev(json({{"==", json::array({"$vol.stride_order", "ndhwc"})}})), V(true));
    // ...and against one that does not.
    EXPECT_EQ(ev(json({{"==", json::array({"$x.stride_order", "nchw"})}})), V(false));
    EXPECT_EQ(ev(json({{"!=", json::array({"$x.stride_order", "nchw"})}})), V(true));

    // The alias is exactly its array: both spellings agree.
    EXPECT_EQ(ev(json({{"==", json::array({"$x.stride_order", json::array({3, 0, 2, 1})})}})),
              ev(json({{"==", json::array({"$x.stride_order", "nhwc"})}})));

    // The alias may sit on either side.
    EXPECT_EQ(ev(json({{"==", json::array({"nhwc", "$x.stride_order"})}})), V(true));

    // The array remains the canonical form and still works untouched.
    EXPECT_EQ(ev(json({{"==", json::array({"$vol.stride_order", json::array({4, 0, 3, 2, 1})})}})),
              V(true));
}

TEST(TestJsonExpression, LayoutAliasesOnlyExpandOppositeStrideOrder)
{
    // "nhwc" is an ordinary string literal anywhere else -- expanding it
    // wherever it appeared would silently rewrite unrelated data.
    const jexpr::JsonDataSource src{json{{"layout", "nhwc"}, {"q", {{"dtype", "BFLOAT16"}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };
    EXPECT_EQ(ev(json({{"==", json::array({"$layout", "nhwc"})}})), V(true));
    EXPECT_EQ(ev(json({{"in", json::array({"$layout", json::array({"nhwc", "nchw"})})}})), V(true));
    EXPECT_EQ(ev("nhwc"), V("nhwc"));
}

TEST(TestJsonExpression, UnknownLayoutAliasRejected)
{
    // A stride_order is an IntArray, so a string opposite one can only be an
    // alias. A typo would otherwise compare unequal forever and decline
    // silently on every graph.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(
                     json({{"==", json::array({"$x.stride_order", "nhcw"})}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(
                     json({{"!=", json::array({"nhcw", "$x.stride_order"})}})),
                 jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, LayoutAliasContradictingARankPinRejected)
{
    // RFC 0018 A.4: every alias is fixed-rank, so an alias compared against a
    // tensor the criteria pin to a different rank is refused at compile rather
    // than declining silently at match time.
    const json rankFour = json({{"==", json::array({"$x.rank", 4})}});
    const json aliasRank5 = json({{"==", json::array({"$x.stride_order", "ndhwc"})}});
    EXPECT_THROW(
        jexpr::compile<jexpr::JsonDataSource>(json({{"and", json::array({rankFour, aliasRank5})}})),
        jexpr::JsonExpressionCompileError);

    // The rank-agreeing alias compiles.
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(json(
        {{"and", json::array({rankFour, {{"==", json::array({"$x.stride_order", "nhwc"})}}})}})));

    // The pin may appear after the alias, with either operand order.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json(
                     {{"and", json::array({aliasRank5, {{"==", json::array({4, "$x.rank"})}}})}})),
                 jexpr::JsonExpressionCompileError);

    // A pin on a *different* tensor does not constrain this alias.
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(
        json({{"and",
               json::array({{{"==", json::array({"$other.rank", 4})}},
                            {{"==", json::array({"$vol.stride_order", "ndhwc"})}}})}})));

    // A conditional pin cannot contradict the alias, so it is not collected.
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(
        json({{"and", json::array({{{"or", json::array({rankFour, false})}}, aliasRank5})}})));
}

TEST(TestJsonExpression, LayoutAliasPrePassLeavesVariableReferencesAlone)
{
    // A variable reference is a string too, so the alias pre-pass must key on
    // the sigil, not on "is a string": "do these two tensors share a layout"
    // is the most ordinary cross-tensor layout criterion there is, and reading
    // the second reference as a typo'd alias made it uncompilable.
    const jexpr::JsonDataSource src{json{{"q", {{"rank", 4}, {"stride_order", {3, 0, 2, 1}}}},
                                         {"k", {{"rank", 4}, {"stride_order", {3, 0, 2, 1}}}},
                                         {"v", {{"rank", 4}, {"stride_order", {3, 2, 1, 0}}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    // Reference against reference, both operand orders.
    EXPECT_EQ(ev(json({{"==", json::array({"$q.stride_order", "$k.stride_order"})}})), V(true));
    EXPECT_EQ(ev(json({{"==", json::array({"$q.stride_order", "$v.stride_order"})}})), V(false));
    EXPECT_EQ(ev(json({{"!=", json::array({"$q.stride_order", "$v.stride_order"})}})), V(true));
    EXPECT_EQ(ev(json({{"==", json::array({"$k.stride_order", "$q.stride_order"})}})), V(true));

    // A reference among a membership set's elements is a value to compare
    // against, not an alias to expand.
    EXPECT_EQ(
        ev(json(
            {{"in", json::array({"$q.stride_order", json::array({"$v.stride_order", "nhwc"})})}})),
        V(true));
    EXPECT_EQ(
        ev(json(
            {{"in", json::array({"$q.stride_order", json::array({"$v.stride_order", "ncdhw"})})}})),
        V(false));

    // An escaped literal stays a literal here, exactly as it does everywhere
    // else, rather than being read as an unknown alias.
    EXPECT_EQ(ev(json({{"==", json::array({"$q.stride_order", "$$nhwc"})}})), V(false));

    // An unresolved reference still propagates null rather than throwing.
    EXPECT_TRUE(
        ev(json({{"==", json::array({"$q.stride_order", "$nope.stride_order"})}})).isNull());

    // A genuine typo opposite a reference is still refused.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(
                     json({{"==", json::array({"$q.stride_order", "nhcw"})}})),
                 jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, LayoutAliasesInAMembershipSet)
{
    // RFC 0018 section 5: a family accepting either of two layouts anchors an
    // `in` over the set. An alias there must expand too, or it would compare
    // unequal against every array in the haystack and never match.
    const jexpr::JsonDataSource src{json{{"q", {{"rank", 4}, {"stride_order", {3, 1, 2, 0}}}},
                                         {"x", {{"rank", 4}, {"stride_order", {3, 0, 2, 1}}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    // BHSD or BSHD, the worked example's set: $q is BSHD, so an alias-only set
    // of the two must accept it via the array arm.
    const json bshd = json::array({3, 1, 2, 0});
    EXPECT_EQ(ev(json({{"in", json::array({"$q.stride_order", json::array({"bhsd", bshd})})}})),
              V(true));
    // $x is NHWC and is accepted by an all-alias set.
    EXPECT_EQ(ev(json({{"in", json::array({"$x.stride_order", json::array({"nchw", "nhwc"})})}})),
              V(true));
    // ...and declined by one that omits its layout.
    EXPECT_EQ(ev(json({{"in", json::array({"$x.stride_order", json::array({"nchw", "bhsd"})})}})),
              V(false));

    // A typo in the set is refused, not silently unmatchable.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json(
                     {{"in", json::array({"$x.stride_order", json::array({"nchw", "nhcw"})})}})),
                 jexpr::JsonExpressionCompileError);

    // A rank pin still applies to every alias in the set.
    EXPECT_THROW(
        jexpr::compile<jexpr::JsonDataSource>(json(
            {{"and",
              json::array(
                  {{{"==", json::array({"$x.rank", 4})}},
                   {{"in", json::array({"$x.stride_order", json::array({"nhwc", "ndhwc"})})}}})}})),
        jexpr::JsonExpressionCompileError);

    // A string set NOT anchored on a stride_order is untouched.
    EXPECT_EQ(ev(json({{"in", json::array({"nhwc", json::array({"nhwc", "nchw"})})}})), V(true));
}

TEST(TestJsonExpression, ValueOrDefault)
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

TEST(TestJsonExpression, PresenceOperators)
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
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"present", json::array()}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"not_present", json::array()}})),
                 jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, NullPropagatesThroughEveryOtherOperator)
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

TEST(TestJsonExpression, KleeneAndOrShortCircuitPastUnknown)
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

TEST(TestJsonExpression, DivisionAndDomainErrorsFailClosed)
{
    // A zero divisor declines instead of yielding inf/NaN, giving uniform
    // fail-closed zero-guarding.
    EXPECT_TRUE(eval(json({{"/", json::array({"$x", 0})}})).isNull());
    EXPECT_TRUE(eval(json({{"%", json::array({"$x", 0})}})).isNull());
    EXPECT_TRUE(eval(json({{"ceil_div", json::array({"$x", 0})}})).isNull());
    // log2/rsqrt decline on a non-positive argument rather than returning
    // -inf/NaN.
    EXPECT_TRUE(eval(json({{"log2", 0}})).isNull());
    EXPECT_TRUE(eval(json({{"rsqrt", 0}})).isNull());
    EXPECT_TRUE(eval(json({{"rsqrt", -4}})).isNull());
    // pow declines on a domain error or an overflow for the same reason: a
    // NaN/inf result compares UNORDERED, so every ordering test on it is
    // false and its negation is true -- a criterion would ACCEPT input it
    // never meaningfully evaluated.
    EXPECT_TRUE(eval(json({{"pow", json::array({-8, 0.5})}})).isNull());
    EXPECT_TRUE(eval(json({{"pow", json::array({10, 400})}})).isNull());
    // ...so the surrounding predicate declines rather than passing.
    const json domainError = json({{"pow", json::array({-8, 0.5})}});
    const json narrowing = json({{"<", json::array({domainError, 1})}});
    EXPECT_TRUE(eval(narrowing).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({narrowing})}})).isNull());
    EXPECT_EQ(eval(json({{"pow", json::array({2, 10})}})), V(1024));
    // The well-behaved cases still compute.
    EXPECT_EQ(eval(json({{"/", json::array({8, 2})}})), V(4));
    EXPECT_EQ(eval(json({{"log2", 8}})), V(3));
}

TEST(TestJsonExpression, UnresolvableNumericOperandsDecline)
{
    // A NaN arrives as an OPERAND, not just as a result: Value::toNumber
    // yields NaN for a non-numeric string ($name is "amd"). A domain guard
    // written `n <= 0.0` is FALSE for NaN and lets it straight through, so
    // every numeric operator has to reject a non-finite result, not just the
    // ones with an obvious domain error.
    for(const char* op : {"log2", "rsqrt", "abs"})
    {
        EXPECT_TRUE(eval(json({{op, "$name"}})).isNull()) << op << " admitted a NaN operand";
    }
    for(const char* op : {"+", "-", "*", "/", "%", "ceil_div", "pow"})
    {
        EXPECT_TRUE(eval(json({{op, json::array({"$x", "$name"})}})).isNull())
            << op << " admitted a NaN operand";
    }
    // min/max must decline rather than SKIP the operand: a NaN sentinel for
    // "nothing chosen yet" is indistinguishable from a NaN argument, so the
    // operator would silently answer from fewer operands than were authored.
    for(const char* op : {"min", "max"})
    {
        EXPECT_TRUE(eval(json({{op, json::array({"$y", "$name"})}})).isNull())
            << op << " dropped an unresolvable operand instead of declining";
    }
    // The whole point: an unresolvable operand must not let a narrowing
    // predicate's NEGATION pass. Both sides decline.
    const json narrowing = json({{"<", json::array({json({{"log2", "$name"}}), 8})}});
    EXPECT_TRUE(eval(narrowing).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({narrowing})}})).isNull());
    // Well-behaved arithmetic is untouched.
    EXPECT_EQ(eval(json({{"+", json::array({2, 3})}})), V(5));
    EXPECT_EQ(eval(json({{"min", json::array({3, 9})}})), V(3));
    EXPECT_EQ(eval(json({{"max", json::array({3, 9})}})), V(9));
    EXPECT_EQ(eval(json({{"abs", -5}})), V(5));
}

TEST(TestJsonExpression, IntegersCompareExactlyAboveTwoToThe53)
{
    // Routed through double, 2^53 and 2^53+1 are the same value. This language
    // gates dispatch on sizes, strides and byte offsets, so that is a wrong
    // decision rather than a rounding error.
    const std::int64_t big = 9007199254740992LL; // 2^53
    EXPECT_NE(V(big), V(big + 1));
    EXPECT_EQ(V::compare(V(big), V(big + 1)), V::Ordering::LESS);
    EXPECT_EQ(V::compare(V(big + 1), V(big)), V::Ordering::GREATER);
    EXPECT_EQ(V::compare(V(big), V(big)), V::Ordering::EQUAL);

    const jexpr::JsonDataSource wide{json{{"bytes", big}}};
    const auto evalWide
        = [&wide](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(wide); };
    EXPECT_EQ(evalWide(json({{"==", json::array({"$bytes", big + 1})}})), V(false));
    EXPECT_EQ(evalWide(json({{"<", json::array({"$bytes", big + 1})}})), V(true));
    EXPECT_EQ(evalWide(json({{">=", json::array({"$bytes", big + 1})}})), V(false));
    EXPECT_EQ(evalWide(json({{"==", json::array({"$bytes", big})}})), V(true));
    // Cross-kind numeric equality still coerces, as documented.
    EXPECT_EQ(eval(json({{"==", json::array({4, 4.0})}})), V(true));
}

TEST(TestJsonExpression, DeeplyNestedRulesAreRejectedNotFatal)
{
    // Rules are read from descriptor files on disk, and compilation and
    // evaluation both recurse per nesting level, so an over-deep rule must
    // report a bad rule rather than overflow the stack.
    //
    // Bracketed at the exact boundary, not loosely around it: compilation runs
    // three recursive passes (rank pins, alias expansion, lowering) that share
    // one MAX_EXPRESSION_DEPTH, and if they charge depth at different rates the
    // strictest silently becomes the real limit while the diagnostic still
    // names the documented one. A test sized at MAX/2 cannot see that.
    const auto nest = [](std::size_t depth) {
        json rule = json("$x");
        for(std::size_t i = 0; i < depth; ++i)
        {
            rule = json({{"+", json::array({rule, 1})}});
        }
        return rule;
    };
    // Comfortably inside the limit: compiles and evaluates.
    EXPECT_EQ(eval(nest(16)), V(41 + 16));
    // At the limit: still compiles, and -- the claim the bound actually makes --
    // still EVALUATES, since evaluation recurses per level too and is bounded
    // only through compilation.
    EXPECT_EQ(eval(nest(jexpr::MAX_EXPRESSION_DEPTH - 1)),
              V(41 + static_cast<std::int64_t>(jexpr::MAX_EXPRESSION_DEPTH) - 1));
    // One past it: a diagnostic, not a crash.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(nest(jexpr::MAX_EXPRESSION_DEPTH + 1)),
                 jexpr::JsonExpressionCompileError);

    // The alias pre-pass runs before lowering and recurses too, so it must
    // enforce the same bound -- neither a looser one (it would hand an
    // over-deep document to lowering) nor a tighter one (it would reject a rule
    // the documented limit admits, citing a limit that is not the real one).
    const auto aliasNest = [](std::size_t depth) {
        json rule = json({{"==", json::array({"$q.stride_order", "nhwc"})}});
        for(std::size_t i = 0; i < depth; ++i)
        {
            rule = json({{"and", json::array({rule})}});
        }
        return rule;
    };
    // Expanding the alias to its 4-element array adds one tree level, so the
    // deepest accepted alias rule sits one below the plain bound.
    EXPECT_NO_THROW(
        jexpr::compile<jexpr::JsonDataSource>(aliasNest(jexpr::MAX_EXPRESSION_DEPTH - 2)));
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(aliasNest(jexpr::MAX_EXPRESSION_DEPTH + 1)),
                 jexpr::JsonExpressionCompileError);

    // The rank-pin walk is the third pass over the same document; an `and`
    // chain is what it descends, so it must agree on the bound as well.
    json pinned = json({{"and",
                         json::array({json({{"==", json::array({"$q.rank", 4})}}),
                                      json({{"==", json::array({"$q.stride_order", "nhwc"})}})})}});
    for(std::size_t i = 0; i < jexpr::MAX_EXPRESSION_DEPTH + 1; ++i)
    {
        pinned = json({{"and", json::array({pinned})}});
    }
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(pinned), jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, ConstraintShapes)
{
    EXPECT_EQ(eval(json({{"in", json::array({"$name", json::array({"amd", "xilinx"})})}})),
              V(true));
    EXPECT_EQ(eval(json({{"==", json::array({"$arr", json::array({10, 20, 30})})}})), V(true));
    // 41 % 8 == 1
    EXPECT_EQ(eval(json({{"==", json::array({{{"%", json::array({"$x", "$y"})}}, 1})}})), V(true));
    // ceil(41/16) == 3
    EXPECT_EQ(eval(json({{"ceil_div", json::array({"$x", 16})}})), V(3));
}

TEST(TestJsonExpression, CompileOnceReuseAcrossData)
{
    const auto expr = jexpr::compile<jexpr::JsonDataSource>(json({{"*", json::array({"$x", 2})}}));
    const jexpr::JsonDataSource a{json{{"x", 3}}};
    const jexpr::JsonDataSource b{json{{"x", 10}}};
    EXPECT_EQ(expr(a), V(6));
    EXPECT_EQ(expr(b), V(20));
}

TEST(TestJsonExpression, MalformedRulesThrowAtCompileTime)
{
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"nope", json::array({1, 2})}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"/", json::array({1, 2, 3})}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"abs", json::array({1, 2})}})),
                 jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, VariablesCollectsReferencedPaths)
{
    using S = std::set<std::string>;
    // Keep the Expression alive while draining the borrowed range into a set.
    const auto vars = [](const json& rule) {
        const auto expr = jexpr::compile<jexpr::JsonDataSource>(rule);
        return S(expr.variables().begin(), expr.variables().end());
    };
    // inline vars across an op
    EXPECT_EQ(vars(json({{"+", json::array({"$x", "$y"})}})), (S{"x", "y"}));
    // a dotted path is kept verbatim
    EXPECT_EQ(vars(json("$nested.a.b")), (S{"nested.a.b"}));
    // a value_or_default fallback subtree contributes its own referenced vars
    EXPECT_EQ(vars(json({{"value_or_default", json::array({"$nope", "$y"})}})), (S{"nope", "y"}));
    // nested composition reaches every leaf var
    EXPECT_EQ(
        vars(json({{"if", json::array({{{">", json::array({"$x", "$y"})}}, "$name", "$zero"})}})),
        (S{"x", "y", "name", "zero"}));
    // literal-only expressions reference nothing
    EXPECT_EQ(vars(json({{"+", json::array({1, 2})}})), (S{}));
    EXPECT_TRUE(vars(json(42)).empty());
}

TEST(TestJsonExpression, ReferencesVariableRootMatchesFirstToken)
{
    const auto refs = [](const json& rule, std::string_view root) {
        const auto expr = jexpr::compile<jexpr::JsonDataSource>(rule);
        return jexpr::referencesVariableRoot(expr, root);
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

TEST(TestJsonExpression, VariablesRangeIsLazyAndKeepsDuplicates)
{
    // range-for yields every occurrence, duplicates included
    const auto expr
        = jexpr::compile<jexpr::JsonDataSource>(json({{"+", json::array({"$x", "$x", "$y"})}}));
    std::vector<std::string> seen;
    for(const std::string& v : expr.variables())
    {
        seen.push_back(v);
    }
    EXPECT_EQ(seen.size(), 3u);
    EXPECT_EQ(std::count(seen.begin(), seen.end(), std::string("x")), 2);
    EXPECT_EQ(std::count(seen.begin(), seen.end(), std::string("y")), 1);

    // empty range: begin() == end(), the body never runs
    const auto lit = jexpr::compile<jexpr::JsonDataSource>(json(42));
    EXPECT_TRUE(lit.variables().begin() == lit.variables().end());

    // composes with STL algorithms over the borrowed references
    const auto r = expr.variables();
    EXPECT_EQ(std::distance(r.begin(), r.end()), 3);
    EXPECT_TRUE(std::any_of(r.begin(), r.end(), [](const std::string& s) { return s == "y"; }));
}

TEST(TestJsonExpression, VariablesIteratorEqualityComparesPositions)
{
    // The range advertises input_iterator_tag, so equality must compare
    // positions. Comparing only "both at end" makes an iterator unequal to
    // itself, which quietly breaks any algorithm comparing two positions --
    // and the end-only cases above would not notice.
    const auto expr
        = jexpr::compile<jexpr::JsonDataSource>(json({{"+", json::array({"$x", "$y"})}}));
    const auto r = expr.variables();

    auto first = r.begin();
    EXPECT_TRUE(first == first); // reflexive
    EXPECT_FALSE(first != first);

    auto copy = first;
    EXPECT_TRUE(copy == first); // a copy sits at the same position

    auto second = first;
    ++second;
    EXPECT_FALSE(second == first); // different positions differ
    EXPECT_TRUE(second != first);
    EXPECT_FALSE(second == r.end()); // ...and neither is the end yet

    ++second;
    EXPECT_TRUE(second == r.end()); // exhausted compares equal to end
    EXPECT_TRUE(r.end() == r.end());
}

TEST(TestJsonExpression, VariablesRangeSurvivesBeingATemporary)
{
    // variables() returns a VarRange by value, so `expr.variables().begin()`
    // is the natural spelling -- and it is only safe because begin()/end()
    // return an iterator BY VALUE. Returning a reference into the range's own
    // members leaves that binding dangling once the temporary dies at the
    // semicolon; ASAN reports stack-use-after-scope, and hipDNN CI runs it.
    const auto expr
        = jexpr::compile<jexpr::JsonDataSource>(json({{"+", json::array({"$x", "$y"})}}));

    auto it = expr.variables().begin();
    const auto stop = expr.variables().end();
    std::set<std::string> seen;
    for(; it != stop; ++it)
    {
        seen.insert(*it);
    }
    EXPECT_EQ(seen, (std::set<std::string>{"x", "y"}));

    // A second range over the same expression is independent of the first.
    EXPECT_EQ(std::distance(expr.variables().begin(), expr.variables().end()), 2);
}

TEST(TestJsonExpression, WholeDocumentReferenceRejected)
{
    // A bare sigil would address the whole document, which the data-source
    // contract does not offer -- getData never receives an empty path.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json("$")),
                 jexpr::JsonExpressionCompileError);
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
