// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <ostream>
#include <string>

#include "hip_kernel_provider_common/JsonLogic.hpp"

namespace jl = hip_kernel_provider_common::jsonlogic;

using json = nlohmann::json;
using V = jl::Value;

namespace hip_kernel_provider_common::jsonlogic
{
// GTest value printer so EXPECT_EQ failures render a readable Value.
inline void PrintTo(const Value& v, std::ostream* os)
{
    *os << v.dump();
}
} // namespace hip_kernel_provider_common::jsonlogic

namespace
{
// ---------------------------------------------------------------------------
// A sample data source: wraps an nlohmann::json document and resolves dotted
// paths to jl::Value. Demonstrates the getData(std::string) contract.
// ---------------------------------------------------------------------------
struct JsonData
{
    json doc;

    static V convert(const json& j)
    {
        if(j.is_boolean())
            return V(j.get<bool>());
        if(j.is_number_integer() || j.is_number_unsigned())
            return V(j.get<std::int64_t>());
        if(j.is_number_float())
            return V(j.get<double>());
        if(j.is_string())
            return V(j.get<std::string>());
        if(j.is_array())
        {
            V::Array a;
            a.reserve(j.size());
            for(const auto& e : j)
                a.push_back(convert(e));
            return V(std::move(a));
        }
        return V(); // null or object -> not representable, treated as null
    }

    V getData(const std::string& path) const
    {
        if(path.empty())
            return convert(doc);
        const json* cur = &doc;
        std::size_t start = 0;
        while(start <= path.size())
        {
            const std::size_t dot = path.find('.', start);
            const std::string seg
                = path.substr(start, (dot == std::string::npos ? path.size() : dot) - start);
            if(cur->is_object())
            {
                const auto it = cur->find(seg);
                if(it == cur->end())
                    return V();
                cur = &*it;
            }
            else if(cur->is_array())
            {
                char* end = nullptr;
                const long idx = std::strtol(seg.c_str(), &end, 10);
                if(*end != '\0' || idx < 0 || static_cast<std::size_t>(idx) >= cur->size())
                    return V();
                cur = &(*cur)[static_cast<std::size_t>(idx)];
            }
            else
            {
                return V();
            }
            if(dot == std::string::npos)
                break;
            start = dot + 1;
        }
        return convert(*cur);
    }
};

const JsonData D{json{{"x", 41},
                      {"y", 8},
                      {"name", "amd"},
                      {"nested", {{"a", {{"b", 7}}}}},
                      {"arr", {10, 20, 30}},
                      {"flag", true},
                      {"zero", 0}}};

// Compile a rule and evaluate it against the shared document D.
V eval(const json& rule)
{
    return jl::compile<JsonData>(rule)(D);
}
} // namespace

TEST(JsonLogic, Literals)
{
    EXPECT_EQ(eval(42), V(42));
    EXPECT_EQ(eval("hello"), V("hello"));
    EXPECT_EQ(eval(true), V(true));
    EXPECT_EQ(eval(json::array({1, 2, 3})), V(V::Array{V(1), V(2), V(3)}));
}

TEST(JsonLogic, VarStockForm)
{
    EXPECT_EQ(eval(json({{"var", "x"}})), V(41));
    EXPECT_EQ(eval(json({{"var", "nested.a.b"}})), V(7));
    EXPECT_EQ(eval(json({{"var", "arr.1"}})), V(20));
    EXPECT_EQ(eval(json({{"var", "nope"}})), V());
    EXPECT_EQ(eval(json({{"var", json::array({"nope", 99})}})), V(99));
    EXPECT_EQ(eval(json({{"var", json::array({"x", 99})}})), V(41));
    EXPECT_EQ(eval(json({{"var", "arr"}})), V(V::Array{V(10), V(20), V(30)}));
}

TEST(JsonLogic, InlineVariables)
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

TEST(JsonLogic, Arithmetic)
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

TEST(JsonLogic, ComparisonIsStrict)
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

TEST(JsonLogic, TruthinessAndLogic)
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

TEST(JsonLogic, IfTernary)
{
    EXPECT_EQ(eval(json({{"if", json::array({true, "yes", "no"})}})), V("yes"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "yes", "no"})}})), V("no"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "a", true, "b", "c"})}})), V("b"));
    EXPECT_EQ(eval(json({{"if", json::array({false, "a"})}})), V());
    EXPECT_EQ(eval(json({{"if", json::array({{{">", json::array({"$x", "$y"})}}, "$x", "$y"})}})),
              V(41));
}

TEST(JsonLogic, Composed)
{
    EXPECT_EQ(
        eval(json({{"if",
                    json::array({{{"<", json::array({{{"+", json::array({"$x", "$y"})}}, 50})}},
                                 "small",
                                 "big"})}})),
        V("small"));
}

TEST(JsonLogic, Membership)
{
    EXPECT_EQ(eval(json({{"in", json::array({20, "$arr"})}})), V(true));
    EXPECT_EQ(eval(json({{"in", json::array({99, "$arr"})}})), V(false));
    EXPECT_EQ(eval(json({{"in", json::array({"$x", json::array({40, 41, 42})})}})), V(true));
    // strict element equality: "41" != 41
    EXPECT_EQ(eval(json({{"in", json::array({"41", json::array({40, 41, 42})})}})), V(false));
    EXPECT_EQ(eval(json({{"in", json::array({"m", "$name"})}})), V(true));
    EXPECT_EQ(eval(json({{"in", json::array({"z", "$name"})}})), V(false));
}

TEST(JsonLogic, MathExtensions)
{
    EXPECT_EQ(eval(json({{"ceil_div", json::array({100, 16})}})), V(7));
    EXPECT_EQ(eval(json({{"ceil_div", json::array({32, 16})}})), V(2));
    EXPECT_EQ(eval(json({{"abs", -5}})), V(5));
    EXPECT_EQ(eval(json({{"abs", 5}})), V(5));
    EXPECT_EQ(eval(json({{"pow", json::array({2, 10})}})), V(1024));
    EXPECT_EQ(eval(json({{"log2", 8}})), V(3));
    EXPECT_EQ(eval(json({{"rsqrt", 4}})), V(0.5));
}

TEST(JsonLogic, ValueOrDefault)
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

TEST(JsonLogic, Umd0018ConstraintShapes)
{
    EXPECT_EQ(eval(json({{"in", json::array({"$name", json::array({"amd", "xilinx"})})}})),
              V(true));
    EXPECT_EQ(eval(json({{"==", json::array({"$arr", json::array({10, 20, 30})})}})), V(true));
    // 41 % 8 == 1
    EXPECT_EQ(eval(json({{"==", json::array({{{"%", json::array({"$x", "$y"})}}, 1})}})), V(true));
    // ceil(41/16) == 3
    EXPECT_EQ(eval(json({{"ceil_div", json::array({"$x", 16})}})), V(3));
}

TEST(JsonLogic, CompileOnceReuseAcrossData)
{
    const auto expr = jl::compile<JsonData>(json({{"*", json::array({"$x", 2})}}));
    const JsonData a{json{{"x", 3}}};
    const JsonData b{json{{"x", 10}}};
    EXPECT_EQ(expr(a), V(6));
    EXPECT_EQ(expr(b), V(20));
}

TEST(JsonLogic, MalformedRulesThrowAtCompileTime)
{
    EXPECT_THROW(jl::compile<JsonData>(json({{"nope", json::array({1, 2})}})),
                 jl::JsonLogicCompileError);
    EXPECT_THROW(jl::compile<JsonData>(json({{"/", json::array({1, 2, 3})}})),
                 jl::JsonLogicCompileError);
    EXPECT_THROW(jl::compile<JsonData>(json({{"abs", json::array({1, 2})}})),
                 jl::JsonLogicCompileError);
}
