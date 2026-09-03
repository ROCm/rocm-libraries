// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
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
    // `var` is not an operator; a variable is always a sigil-prefixed string.
    // Rejecting `var` by name gives a rule written against stock JsonLogic a
    // clear error instead of a generic unknown-operator one, and keeps two
    // spellings of the same thing from coexisting.
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
    // Subscript and dotted keys mixed.
    EXPECT_EQ(eval("$rows[1].name"), V("a1"));
    // Chained subscripts into a nested array.
    EXPECT_EQ(eval("$grid[0][1]"), V(2));
    EXPECT_EQ(eval("$grid[1][0]"), V(3));
    // Dot-form array index.
    EXPECT_EQ(eval("$arr.1"), V(20));
    // Out-of-range, non-numeric, and subscript-on-non-array all read as null.
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
    // Strict element equality: "41" is not 41.
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
    // Negative operands divide the same way fmod does.
    EXPECT_EQ(eval(json({{"divisible", json::array({-32, 16})}})), V(true));
    EXPECT_EQ(eval(json({{"divisible", json::array({32, -16})}})), V(true));
    // The tile-fit shape the RFCs use: a product against a kernel constant.
    EXPECT_EQ(eval(json({{"divisible", json::array({{{"*", json::array({"$y", 4})}}, 16})}})),
              V(true));
    // A zero divisor declines, exactly as `%` does.
    EXPECT_EQ(eval(json({{"divisible", json::array({32, 0})}})), V());
    // An unresolved operand yields null rather than reading as divisible.
    EXPECT_EQ(eval(json({{"divisible", json::array({"$nope", 16})}})), V());
    EXPECT_EQ(eval(json({{"divisible", json::array({32, "$nope"})}})), V());
}

TEST(TestJsonExpression, DivisibleMatchesModuloLonghand)
{
    // `divisible` is shorthand for {"==": [{"%": [a, b]}, 0]}. The RFCs use
    // both spellings, so they must agree on every input.
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
    // And against one that does not.
    EXPECT_EQ(ev(json({{"==", json::array({"$x.stride_order", "nchw"})}})), V(false));
    EXPECT_EQ(ev(json({{"!=", json::array({"$x.stride_order", "nchw"})}})), V(true));

    // The alias means exactly its array: both spellings agree.
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
    // "nhwc" is an ordinary string literal anywhere else. Expanding it wherever
    // it appeared would silently rewrite unrelated data.
    const jexpr::JsonDataSource src{json{{"layout", "nhwc"}, {"q", {{"dtype", "BFLOAT16"}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };
    EXPECT_EQ(ev(json({{"==", json::array({"$layout", "nhwc"})}})), V(true));
    EXPECT_EQ(ev(json({{"in", json::array({"$layout", json::array({"nhwc", "nchw"})})}})), V(true));
    EXPECT_EQ(ev("nhwc"), V("nhwc"));
}

TEST(TestJsonExpression, UnknownLayoutAliasRejected)
{
    // A stride_order is an array of integers, so a string opposite one can only
    // be an alias. A typo would otherwise compare unequal forever and decline
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
    // RFC 0018 A.4: every alias has a fixed rank, so an alias compared against
    // a tensor the criteria pin to a different rank is rejected at compile time
    // rather than declining silently at match time.
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

    // A single-operand `and` can take a bare value instead of a one-element
    // array, and the pin inside it is still unconditional.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(
                     json({{"and", json::array({json({{"and", rankFour}}), aliasRank5})}})),
                 jexpr::JsonExpressionCompileError);

    // Integral floating-point rank literals are exact pins too.
    const json rankFourFloat = json({{"==", json::array({"$x.rank", 4.0})}});
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(
                     json({{"and", json::array({rankFourFloat, aliasRank5})}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(json(
        {{"and",
          json::array({rankFourFloat, {{"==", json::array({"$x.stride_order", "nhwc"})}}})}})));

    // A pin on a different tensor does not constrain this alias.
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(
        json({{"and",
               json::array({{{"==", json::array({"$other.rank", 4})}},
                            {{"==", json::array({"$vol.stride_order", "ndhwc"})}}})}})));

    // A conditional pin cannot contradict the alias, so it is not collected.
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(
        json({{"and", json::array({{{"or", json::array({rankFour, false})}}, aliasRank5})}})));
}

TEST(TestJsonExpression, LayoutAliasRankPinAppliesOnlyToItsOwnTensor)
{
    // A pin constrains the tensor it names, and that tensor is the whole path
    // before `.rank`, not the path's first segment. $inputs[0] and $inputs[1]
    // are two tensors living in one array, so keying the pin on the root let
    // element 0's rank veto element 1's layout, rejecting a rule that any
    // 4d-then-5d graph satisfies.
    const json tensor4 = json{{"rank", 4}, {"stride_order", json::array({3, 0, 2, 1})}};
    const json tensor5 = json{{"rank", 5}, {"stride_order", json::array({4, 0, 3, 2, 1})}};
    const jexpr::JsonDataSource src{json{{"inputs", json::array({tensor4, tensor5})},
                                         {"a", json{{"b", json{{"c", tensor4}, {"d", tensor5}}}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    // The pin is on element 0. Element 1 is unconstrained and is in fact rank
    // 5, so the rule compiles and holds.
    EXPECT_EQ(
        ev(json({{"and",
                  json::array({{{"==", json::array({"$inputs[0].rank", 4})}},
                               {{"==", json::array({"$inputs[1].stride_order", "ndhwc"})}}})}})),
        V(true));

    // The same subscript is the same tensor, though, so rank 4 against a
    // rank-5 alias is still unsatisfiable.
    EXPECT_THROW(
        jexpr::compile<jexpr::JsonDataSource>(
            json({{"and",
                   json::array({{{"==", json::array({"$inputs[0].rank", 4})}},
                                {{"==", json::array({"$inputs[0].stride_order", "ndhwc"})}}})}})),
        jexpr::JsonExpressionCompileError);

    // A multi-segment prefix identifies a tensor just as a subscript does.
    // $a.b.c and $a.b.d are distinct, and $a.b.c contradicts itself.
    EXPECT_EQ(ev(json({{"and",
                        json::array({{{"==", json::array({"$a.b.c.rank", 4})}},
                                     {{"==", json::array({"$a.b.d.stride_order", "ndhwc"})}}})}})),
              V(true));
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json(
                     {{"and",
                       json::array({{{"==", json::array({"$a.b.c.rank", 4})}},
                                    {{"==", json::array({"$a.b.c.stride_order", "ndhwc"})}}})}})),
                 jexpr::JsonExpressionCompileError);

    // A pin on the containing array is not a pin on an element of it.
    EXPECT_NO_THROW(jexpr::compile<jexpr::JsonDataSource>(
        json({{"and",
               json::array({{{"==", json::array({"$inputs.rank", 4})}},
                            {{"==", json::array({"$inputs[1].stride_order", "ndhwc"})}}})}})));
}

TEST(TestJsonExpression, LayoutAliasPrePassLeavesVariableReferencesAlone)
{
    // A variable reference is a string too, so the pre-pass keys on the sigil
    // rather than on "is a string". Asking whether two tensors share a layout
    // is an ordinary criterion, and reading the second reference as a
    // misspelled alias made it uncompilable.
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

    // An escaped literal stays a literal here, as it does everywhere else,
    // rather than being read as an unknown alias.
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
    // RFC 0018 section 5: a family that accepts either of two layouts uses an
    // `in` over the set. An alias there must expand too, or it would compare
    // unequal against every array in the haystack and never match.
    const jexpr::JsonDataSource src{json{{"q", {{"rank", 4}, {"stride_order", {3, 1, 2, 0}}}},
                                         {"x", {{"rank", 4}, {"stride_order", {3, 0, 2, 1}}}}}};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    // BHSD or BSHD, the worked example's set. $q is BSHD, so an alias-only set
    // of the two must accept it through the array arm.
    const json bshd = json::array({3, 1, 2, 0});
    EXPECT_EQ(ev(json({{"in", json::array({"$q.stride_order", json::array({"bhsd", bshd})})}})),
              V(true));
    // $x is NHWC and is accepted by a set of aliases.
    EXPECT_EQ(ev(json({{"in", json::array({"$x.stride_order", json::array({"nchw", "nhwc"})})}})),
              V(true));
    // And declined by one that omits its layout.
    EXPECT_EQ(ev(json({{"in", json::array({"$x.stride_order", json::array({"nchw", "bhsd"})})}})),
              V(false));

    // A typo in the set is rejected rather than silently unmatchable.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json(
                     {{"in", json::array({"$x.stride_order", json::array({"nchw", "nhcw"})})}})),
                 jexpr::JsonExpressionCompileError);

    // A rank pin applies to every alias in the set.
    EXPECT_THROW(
        jexpr::compile<jexpr::JsonDataSource>(json(
            {{"and",
              json::array(
                  {{{"==", json::array({"$x.rank", 4})}},
                   {{"in", json::array({"$x.stride_order", json::array({"nhwc", "ndhwc"})})}}})}})),
        jexpr::JsonExpressionCompileError);

    // A string set that is not anchored on a stride_order is left untouched.
    EXPECT_EQ(ev(json({{"in", json::array({"nhwc", json::array({"nhwc", "nchw"})})}})), V(true));
}

TEST(TestJsonExpression, ValueOrDefault)
{
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$x", 99})}})), V(41));
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$nope", 99})}})), V(99));
    // Keys on existence, not truthiness: a present 0 is returned.
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$zero", 99})}})), V(0));
    EXPECT_EQ(eval(json({{"value_or_default", json::array({"$nested.a.b", -1})}})), V(7));
    // The default is itself an expression, evaluated only when needed.
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
    // Presence keys on existence, not truthiness: a present 0 or false counts.
    EXPECT_EQ(eval(json({{"present", json::array({"$zero"})}})), V(true));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$zero"})}})), V(false));
    // Unlike every other operator, an unresolved path yields a real boolean
    // instead of null, so the criterion decides rather than declining.
    EXPECT_TRUE(eval(json({{"present", json::array({"$nope"})}})).isBool());
    EXPECT_TRUE(eval(json({{"not_present", json::array({"$nope"})}})).isBool());
    // n-ary: both combine their arguments with `and`.
    EXPECT_EQ(eval(json({{"present", json::array({"$x", "$y", "$name"})}})), V(true));
    EXPECT_EQ(eval(json({{"present", json::array({"$x", "$nope"})}})), V(false));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$nope", "$missing"})}})), V(true));
    EXPECT_EQ(eval(json({{"not_present", json::array({"$nope", "$x"})}})), V(false));
    // Unary sugar: a bare argument instead of an array.
    EXPECT_EQ(eval(json({{"present", "$x"}})), V(true));
    // At least one argument is required.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"present", json::array()}})),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json({{"not_present", json::array()}})),
                 jexpr::JsonExpressionCompileError);
}

TEST(TestJsonExpression, NullPropagatesThroughEveryOtherOperator)
{
    // An unresolved reference is unknown, not a value. Every operator except
    // the presence pair and value_or_default must yield null instead of
    // coercing, because a coerced null reads as false, 0, or not-equal, which
    // would make a narrowing check pass on data it never saw.
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
    // Two unresolved references are not equal; the question is unanswerable.
    EXPECT_TRUE(eval(json({{"==", json::array({"$nope", "$missing"})}})).isNull());
    EXPECT_TRUE(eval(json({{"!=", json::array({"$nope", "$missing"})}})).isNull());
    // An array literal is unresolved if any element is, however deeply nested.
    // Otherwise equality and membership would answer false from a partial
    // array, and a surrounding negation would then accept.
    const json unresolvedArrayEq
        = json({{"==", json::array({json::array({"$nope"}), json::array()})}});
    EXPECT_TRUE(eval(unresolvedArrayEq).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({unresolvedArrayEq})}})).isNull());
    const json nestedUnresolvedArrayEq = json(
        {{"==",
          json::array({json::array({json::array({"$nope"})}), json::array({json::array({1})})})}});
    EXPECT_TRUE(eval(nestedUnresolvedArrayEq).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({nestedUnresolvedArrayEq})}})).isNull());
    const json unresolvedMembership
        = json({{"in", json::array({99, json::array({1, json::array({"$nope"})})})}});
    EXPECT_TRUE(eval(unresolvedMembership).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({unresolvedMembership})}})).isNull());
}

TEST(TestJsonExpression, UnresolvedArrayFromTheDataSourceDeclinesEverywhere)
{
    // The array literals above are written into the rule. This test covers the
    // other direction, which is the one real data takes: the data source hands
    // back an array with a hole in it. An unsigned value beyond int64_t range
    // reads as null (JsonDataSource::toValue), so `strides` resolves to
    // [1, null], a value that only partly resolved.
    //
    // Every operator must treat that like a bare null. The lazy operators are
    // the ones worth pinning down: they evaluate their own arguments, so they
    // bypass OpNode's guard and need the check themselves.
    const json doc = json{{"strides", json::array({1U, 18446744073709551000ULL})},
                          {"clean", json::array({1, 2})}};
    const jexpr::JsonDataSource src{doc};
    const auto run
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    ASSERT_EQ(src.getData("strides"), V(V::Array{V(1), V()})) << "precondition: the hole exists";

    // Eager path.
    EXPECT_TRUE(run(json({{"==", json::array({"$strides", json::array({1, 2})})}})).isNull());
    EXPECT_TRUE(run(json({{"!!", json::array({"$strides"})}})).isNull());

    // Lazy path: an unresolved condition picks no branch.
    EXPECT_TRUE(run(json({{"if", json::array({"$strides", "taken", "else"})}})).isNull());

    // Three-valued and/or: an unresolved operand is unknown, not truthy.
    EXPECT_TRUE(run(json({{"and", json::array({"$strides", true})}})).isNull());
    EXPECT_TRUE(run(json({{"or", json::array({"$strides", false})}})).isNull());

    // value_or_default must supply the fallback rather than hand back a value
    // with a hole in it.
    EXPECT_EQ(run(json({{"value_or_default", json::array({"$strides", "fallback"})}})),
              V("fallback"));
    EXPECT_EQ(run(json({{"value_or_default", json::array({"$clean", "fallback"})}})),
              V(V::Array{V(1), V(2)}));

    // Presence answers false in both directions here: a partly resolved value
    // is neither wholly supplied nor wholly absent. If `not_present` were true
    // instead, the absent-field guard below would accept input whose field
    // reads never ran.
    EXPECT_EQ(run(json({{"present", json::array({"$strides"})}})), V(false));
    EXPECT_EQ(run(json({{"not_present", json::array({"$strides"})}})), V(false));
    EXPECT_EQ(run(json({{"present", json::array({"$clean"})}})), V(true));
    EXPECT_EQ(run(json({{"not_present", json::array({"$absent"})}})), V(true));

    // The documented absent-field guard, across every shape the field can take.
    // What matters is that it never accepts a value the language could not
    // resolve. Whether it declines with null or answers false is secondary,
    // since a null result is falsy either way.
    const auto guarded = [](const char* var) {
        return json(
            {{"or",
              json::array(
                  {json({{"not_present", json::array({var})}}),
                   json({{"and",
                          json::array(
                              {json({{"present", json::array({var})}}),
                               json({{"==", json::array({var, json::array({1, 2})})}})})}})})}});
    };
    // A value with a hole is not absent, so the first arm is false, and it is
    // not wholly supplied either, so the second arm cannot run. Both presence
    // operators answer false, so the guard rejects.
    EXPECT_EQ(run(guarded("$strides")), V(false))
        << "the absent-field guard must not accept a hole-y value";
    EXPECT_FALSE(run(guarded("$strides")).truthy());
    // The shapes the guard exists to serve still behave.
    EXPECT_EQ(run(guarded("$absent")), V(true));
    EXPECT_EQ(run(guarded("$clean")), V(true));
}

TEST(TestJsonExpression, KleeneAndOrShortCircuitPastUnknown)
{
    // A definite false decides an `and` even beside an unknown, and a definite
    // true decides an `or`. That is what lets "absent, or present and
    // constrained" accept an absent operand whose field checks cannot run.
    EXPECT_EQ(eval(json({{"and", json::array({false, "$nope"})}})), V(false));
    EXPECT_EQ(eval(json({{"or", json::array({true, "$nope"})}})), V(true));
    // Without a decisive argument the result stays unknown rather than
    // collapsing to true or false.
    EXPECT_TRUE(eval(json({{"and", json::array({true, "$nope"})}})).isNull());
    EXPECT_TRUE(eval(json({{"or", json::array({false, "$nope"})}})).isNull());
    // A fully-resolved expression is unaffected.
    EXPECT_EQ(eval(json({{"and", json::array({true, true})}})), V(true));
    EXPECT_EQ(eval(json({{"or", json::array({false, false})}})), V(false));
}

TEST(TestJsonExpression, DivisionAndDomainErrorsFailClosed)
{
    // A zero divisor declines instead of yielding infinity or NaN, so every
    // division operator guards zero the same way.
    EXPECT_TRUE(eval(json({{"/", json::array({"$x", 0})}})).isNull());
    EXPECT_TRUE(eval(json({{"%", json::array({"$x", 0})}})).isNull());
    EXPECT_TRUE(eval(json({{"ceil_div", json::array({"$x", 0})}})).isNull());
    // log2 and rsqrt decline on a non-positive argument rather than returning
    // -infinity or NaN.
    EXPECT_TRUE(eval(json({{"log2", 0}})).isNull());
    EXPECT_TRUE(eval(json({{"rsqrt", 0}})).isNull());
    EXPECT_TRUE(eval(json({{"rsqrt", -4}})).isNull());
    // pow declines on a domain error or an overflow for the same reason: a
    // non-finite result cannot be ordered afterwards, so the arithmetic step
    // stays unresolved instead of handing a predicate data it cannot evaluate.
    EXPECT_TRUE(eval(json({{"pow", json::array({-8, 0.5})}})).isNull());
    EXPECT_TRUE(eval(json({{"pow", json::array({10, 400})}})).isNull());
    // So the surrounding predicate declines rather than passing.
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
    // A NaN can arrive as an operand, not just as a result: Value::toNumber
    // yields NaN for a non-numeric string, and $name is "amd". A domain guard
    // written `n <= 0.0` is false for NaN and would let it through, so every
    // numeric operator rejects a non-finite result, not just the ones with an
    // obvious domain error.
    for(const char* op : {"log2", "rsqrt", "abs"})
    {
        EXPECT_TRUE(eval(json({{op, "$name"}})).isNull()) << op << " admitted a NaN operand";
    }
    for(const char* op : {"+", "-", "*", "/", "%", "ceil_div", "pow"})
    {
        EXPECT_TRUE(eval(json({{op, json::array({"$x", "$name"})}})).isNull())
            << op << " admitted a NaN operand";
    }
    // min/max decline rather than skipping the operand. A NaN sentinel for
    // "nothing chosen yet" is indistinguishable from a NaN argument, so the
    // operator would silently answer from fewer operands than were written.
    for(const char* op : {"min", "max"})
    {
        EXPECT_TRUE(eval(json({{op, json::array({"$y", "$name"})}})).isNull())
            << op << " dropped an unresolvable operand instead of declining";
    }
    // The point of all of this: an unresolvable operand must not let the
    // negation of a narrowing predicate pass. Both sides decline.
    const json narrowing = json({{"<", json::array({json({{"log2", "$name"}}), 8})}});
    EXPECT_TRUE(eval(narrowing).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({narrowing})}})).isNull());
    // Well-behaved arithmetic is untouched.
    EXPECT_EQ(eval(json({{"+", json::array({2, 3})}})), V(5));
    EXPECT_EQ(eval(json({{"min", json::array({3, 9})}})), V(3));
    EXPECT_EQ(eval(json({{"max", json::array({3, 9})}})), V(9));
    EXPECT_EQ(eval(json({{"abs", -5}})), V(5));
}

TEST(TestJsonExpression, OrderingDeclinesOnNonFiniteCoercedOperands)
{
    const json nanCompare = json({{"<", json::array({"$name", 8})}});
    EXPECT_TRUE(eval(nanCompare).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({nanCompare})}})).isNull());

    const json infCompare = json({{"<", json::array({"1e309", 8})}});
    EXPECT_TRUE(eval(infCompare).isNull());
    EXPECT_TRUE(eval(json({{"!", json::array({infCompare})}})).isNull());

    const json chain = json({{"<", json::array({1, "$name", 3})}});
    EXPECT_TRUE(eval(chain).isNull());

    const json falseThenUnordered = json({{"<", json::array({5, 1, "$name"})}});
    EXPECT_TRUE(eval(falseThenUnordered).isNull());
}

TEST(TestJsonExpression, IntegersCompareExactlyAboveTwoToThe53)
{
    // Routed through double, 2^53 and 2^53+1 are the same value. This language
    // gates dispatch on sizes, strides and byte offsets, so that would be a
    // wrong decision rather than a rounding error.
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

    const double roundedBig = static_cast<double>(big);
    EXPECT_NE(V(big + 1), V(roundedBig));
    EXPECT_EQ(V::compare(V(big + 1), V(roundedBig)), V::Ordering::GREATER);
    EXPECT_EQ(V::compare(V(roundedBig), V(big + 1)), V::Ordering::LESS);

    const jexpr::JsonDataSource mixed{json{{"bytes", big + 1}}};
    const auto evalMixed
        = [&mixed](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(mixed); };
    EXPECT_EQ(evalMixed(json({{"==", json::array({"$bytes", roundedBig})}})), V(false));
    EXPECT_EQ(evalMixed(json({{">", json::array({"$bytes", roundedBig})}})), V(true));
    EXPECT_EQ(evalMixed(json({{"<=", json::array({"$bytes", roundedBig})}})), V(false));
    // Cross-kind numeric equality still coerces, as documented.
    EXPECT_EQ(eval(json({{"==", json::array({4, 4.0})}})), V(true));
}

TEST(TestJsonExpression, DeeplyNestedRulesAreRejectedNotFatal)
{
    // Rules are read from descriptor files on disk, and both compilation and
    // evaluation recurse per nesting level, so an over-deep rule must report a
    // bad rule rather than overflow the stack.
    //
    // The limit is tested at its exact boundary rather than loosely around it.
    // Compilation runs three recursive passes (rank pins, alias expansion,
    // lowering) that share one MAX_EXPRESSION_DEPTH. If they counted depth at
    // different rates, the strictest would become the real limit while the
    // error still named the documented one, and a test sized at MAX/2 would
    // not catch it.
    const auto nest = [](std::size_t depth) {
        json rule = json("$x");
        for(std::size_t i = 0; i < depth; ++i)
        {
            rule = json({{"+", json::array({rule, 1})}});
        }
        return rule;
    };
    // Well inside the limit: compiles and evaluates.
    EXPECT_EQ(eval(nest(16)), V(41 + 16));
    // At the limit: still compiles, and still evaluates. Evaluation recurses
    // per level too, and compilation is the only thing bounding it.
    EXPECT_EQ(eval(nest(jexpr::MAX_EXPRESSION_DEPTH - 1)),
              V(41 + static_cast<std::int64_t>(jexpr::MAX_EXPRESSION_DEPTH) - 1));
    // One past the limit: an error, not a crash.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(nest(jexpr::MAX_EXPRESSION_DEPTH + 1)),
                 jexpr::JsonExpressionCompileError);

    // The alias pre-pass runs before lowering and recurses too, so it must
    // enforce the same bound. A looser one would hand an over-deep document to
    // lowering; a tighter one would reject a rule the documented limit allows,
    // while citing that limit.
    const auto aliasNest = [](std::size_t depth) {
        json rule = json({{"==", json::array({"$q.stride_order", "nhwc"})}});
        for(std::size_t i = 0; i < depth; ++i)
        {
            rule = json({{"and", json::array({rule})}});
        }
        return rule;
    };
    // Expanding the alias into its 4-element array adds one tree level, so the
    // deepest accepted alias rule sits one below the plain bound.
    EXPECT_NO_THROW(
        jexpr::compile<jexpr::JsonDataSource>(aliasNest(jexpr::MAX_EXPRESSION_DEPTH - 2)));
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(aliasNest(jexpr::MAX_EXPRESSION_DEPTH + 1)),
                 jexpr::JsonExpressionCompileError);

    // The rank-pin walk is the third pass over the same document. It descends
    // `and` chains, so it must agree on the bound as well.
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

TEST(TestJsonExpression, UnsignedIntegerLiteralsMustFitInInt64)
{
    const auto maxInt = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(eval(json(maxInt)), V(std::numeric_limits<std::int64_t>::max()));
    EXPECT_EQ(
        eval(json({{"+", json::array({json(maxInt), -std::numeric_limits<std::int64_t>::max()})}})),
        V(0));

    const std::uint64_t tooLarge = maxInt + 1U;
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json(tooLarge)),
                 jexpr::JsonExpressionCompileError);
    EXPECT_THROW(
        jexpr::compile<jexpr::JsonDataSource>(json({{"==", json::array({json(tooLarge), 0})}})),
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
    // Inline vars across an operator.
    EXPECT_EQ(vars(json({{"+", json::array({"$x", "$y"})}})), (S{"x", "y"}));
    // A dotted path is kept verbatim.
    EXPECT_EQ(vars(json("$nested.a.b")), (S{"nested.a.b"}));
    // A value_or_default fallback subtree contributes its own variables.
    EXPECT_EQ(vars(json({{"value_or_default", json::array({"$nope", "$y"})}})), (S{"nope", "y"}));
    // Nested composition reaches every leaf variable.
    EXPECT_EQ(
        vars(json({{"if", json::array({{{">", json::array({"$x", "$y"})}}, "$name", "$zero"})}})),
        (S{"x", "y", "name", "zero"}));
    // A literal-only expression references nothing.
    EXPECT_EQ(vars(json({{"+", json::array({1, 2})}})), (S{}));
    EXPECT_TRUE(vars(json(42)).empty());
}

TEST(TestJsonExpression, ReferencesVariableRootMatchesFirstToken)
{
    const auto refs = [](const json& rule, std::string_view root) {
        const auto expr = jexpr::compile<jexpr::JsonDataSource>(rule);
        return jexpr::referencesVariableRoot(expr, root);
    };
    // Matches the first path token, before any '.' or '[' separator.
    EXPECT_TRUE(refs(json({{"<", json::array({"$kernel.tile_m", "$device.lds_size"})}}), "kernel"));
    EXPECT_TRUE(refs(json({{"<", json::array({"$kernel.tile_m", "$device.lds_size"})}}), "device"));
    EXPECT_TRUE(refs(json({{"==", json::array({"$kernel.vec[0]", 1})}}), "kernel"));
    // A bare root, with no field, still matches its own token.
    EXPECT_TRUE(refs(json("$kernel"), "kernel"));
    // No variable has that root.
    EXPECT_FALSE(refs(json({{"==", json::array({"$q.head_size", 128})}}), "kernel"));
    // A root matches a whole token, not a prefix.
    EXPECT_FALSE(refs(json("$kernelish.x"), "kernel"));
    // A literal-only expression references nothing.
    EXPECT_FALSE(refs(json({{"+", json::array({1, 2})}}), "kernel"));
}

TEST(TestJsonExpression, VariablesRangeIsLazyAndKeepsDuplicates)
{
    // A range-for yields every occurrence, duplicates included.
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

    // Empty range: begin() == end(), so the loop body never runs.
    const auto lit = jexpr::compile<jexpr::JsonDataSource>(json(42));
    EXPECT_TRUE(lit.variables().begin() == lit.variables().end());

    // Composes with STL algorithms over the borrowed references.
    const auto r = expr.variables();
    EXPECT_EQ(std::distance(r.begin(), r.end()), 3);
    EXPECT_TRUE(std::any_of(r.begin(), r.end(), [](const std::string& s) { return s == "y"; }));
}

TEST(TestJsonExpression, VariablesIteratorEqualityComparesPositions)
{
    // The range advertises input_iterator_tag, so equality has to compare
    // positions. Treating only "both at end" as equal would make an iterator
    // unequal to itself, which breaks any algorithm that compares two
    // positions, and the end-only cases above would not catch it.
    const auto expr
        = jexpr::compile<jexpr::JsonDataSource>(json({{"+", json::array({"$x", "$y"})}}));
    const auto r = expr.variables();

    auto first = r.begin();
    EXPECT_TRUE(first == first); // reflexive
    EXPECT_FALSE(first != first);

    // A copy sits at the same position as its source, and advancing the copy
    // moves only the copy.
    auto second = first;
    EXPECT_TRUE(second == first);

    ++second;
    EXPECT_FALSE(second == first); // different positions differ
    EXPECT_TRUE(second != first);
    EXPECT_FALSE(second == r.end()); // and neither is at the end yet

    ++second;
    EXPECT_TRUE(second == r.end()); // exhausted compares equal to end
    EXPECT_TRUE(r.end() == r.end());
}

TEST(TestJsonExpression, VariablesRangeSurvivesBeingATemporary)
{
    // variables() returns a VarRange by value, so in `expr.variables().begin()`
    // the range is a temporary that dies at the semicolon. That spelling is
    // safe only because begin() and end() return iterators by value; a
    // reference into the range's own members would dangle. ASAN reports it as
    // stack-use-after-scope, and hipDNN CI runs ASAN.
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

TEST(TestJsonExpression, BareSigilReferenceRejected)
{
    // A bare sigil names no path, and the data-source contract has no location
    // it could address: getData never receives an empty path.
    EXPECT_THROW(jexpr::compile<jexpr::JsonDataSource>(json("$")),
                 jexpr::JsonExpressionCompileError);
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
