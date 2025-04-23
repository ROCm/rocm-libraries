/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/value.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>

enum class enum_type
{
    a,
    b,
    c
};

TEST_CASE(value_default_construct)
{
    migraphx::value v;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_null)
{
    migraphx::value v = nullptr;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_assign_null)
{
    migraphx::value v;
    v = nullptr;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int1)
{
    EXPECT(migraphx::value(1).is_int64());
    migraphx::value v(1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int2)
{
    migraphx::value v = 1;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_string)
{
    migraphx::value v = "one";
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "one");
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_key_string_literal_pair)
{
    // Use parens instead {} to construct to test the key-pair constructor
    migraphx::value v("key", "one");
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "one");
    EXPECT(v.get_key() == "key");
}

TEST_CASE(value_construct_float)
{
    migraphx::value v = 1.0;
    EXPECT(v.is_float());
    EXPECT(migraphx::float_equal(v.get_float(), 1.0));
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_bool)
{
    migraphx::value v = true;
    EXPECT(v.is_bool());
    EXPECT(v.get_bool() == true);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_enum1)
{
    migraphx::value v = enum_type::a;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == static_cast<std::uint64_t>(enum_type::a));
    EXPECT(bool{v.to<enum_type>() == enum_type::a});
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_enum2)
{
    migraphx::value v = enum_type::b;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == static_cast<std::uint64_t>(enum_type::b));
    EXPECT(bool{v.to<enum_type>() == enum_type::b});
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_enum3)
{
    migraphx::value v = enum_type::c;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == static_cast<std::uint64_t>(enum_type::c));
    EXPECT(bool{v.to<enum_type>() == enum_type::c});
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_empty_object)
{
    migraphx::value v = migraphx::value::object{};
    EXPECT(v.is_object());
    EXPECT(v.get_object().empty());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_empty_array)
{
    migraphx::value v = migraphx::value::array{};
    EXPECT(v.is_array());
    EXPECT(v.get_array().empty());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_assign_int)
{
    migraphx::value v;
    v = 0;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 0);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_copy_construct)
{
    migraphx::value v1(1);
    migraphx::value v2 = v1; // NOLINT
    EXPECT(v1 == v2);
}

TEST_CASE(value_copy_assign)
{
    migraphx::value v1(1);
    migraphx::value v2;
    v2 = v1;
    EXPECT(v1 == v2);
}

TEST_CASE(value_reassign)
{
    migraphx::value v1(1);
    migraphx::value v2 = v1;
    v1                 = 2;
    EXPECT(v1 != v2);
}

TEST_CASE(value_copy_assign_key)
{
    migraphx::value v1("key", 1);
    migraphx::value v2;
    v2 = v1;
    EXPECT(v2.get_key() == "key");
    EXPECT(v1 == v2);
}

TEST_CASE(value_copy_assign_keyless)
{
    migraphx::value v1(1);
    migraphx::value v2("key", nullptr);
    v2 = v1;
    EXPECT(v2.get_key() == "key");
    EXPECT(v1 != v2);
    EXPECT(v1.without_key() == v2.without_key());
}

TEST_CASE(value_assign_key_string_literal_pair)
{
    migraphx::value v = migraphx::value::object{};
    v["key"]          = "one";
    EXPECT(v["key"].is_string());
    EXPECT(v["key"].get_string() == "one");
    EXPECT(v["key"].get_key() == "key");
}

TEST_CASE(value_construct_array)
{
    migraphx::value v = {1, 2, 3};
    EXPECT(v.is_array());
    EXPECT(v.get_array().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
    EXPECT(test::throws([&] { v.at("???"); }));
    [=] {
        EXPECT(v.data() != nullptr);
        EXPECT(v.front().is_int64());
        EXPECT(v.front() == migraphx::value(1));
        EXPECT(v[1] == migraphx::value(2));
        EXPECT(v.at(1) == migraphx::value(2));
        EXPECT(v.back() == migraphx::value(3));
    }();
}

TEST_CASE(value_insert_array)
{
    migraphx::value v;
    v.insert(v.end(), 1);
    v.insert(v.end(), 2);
    v.insert(v.end(), 3);
    EXPECT(v.is_array());
    EXPECT(v.get_array().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
}

TEST_CASE(value_key_array)
{
    std::vector<migraphx::value> values = {1, 2, 3};
    migraphx::value v("key", values);
    EXPECT(v.is_array());
    EXPECT(v.get_key() == "key");
    EXPECT(v.get_array().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
}

TEST_CASE(value_key_array_empty)
{
    std::vector<migraphx::value> values{};
    migraphx::value v("key", values);
    EXPECT(v.is_array());
    EXPECT(v.get_key() == "key");
    EXPECT(v.get_array().size() == 0);
    EXPECT(v.size() == 0);
    EXPECT(v.empty());
}

TEST_CASE(value_construct_key_int1)
{
    migraphx::value v("one", 1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_key_int2)
{
    migraphx::value v = {"one", 1};
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_key_pair)
{
    migraphx::value v = std::make_pair("one", 1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_object)
{
    migraphx::value v = {{"one", 1}, {"two", migraphx::value(2)}, {"three", 3}};
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_key_object)
{
    std::unordered_map<std::string, migraphx::value> values = {
        {"one", 1}, {"two", migraphx::value(2)}, {"three", 3}};
    migraphx::value v("key", values);
    EXPECT(v.get_key() == "key");
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_key_object_empty)
{
    std::unordered_map<std::string, migraphx::value> values{};
    migraphx::value v("key", values);
    EXPECT(v.get_key() == "key");
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 0);
    EXPECT(v.size() == 0);
    EXPECT(v.empty());
    EXPECT(not v.contains("one"));
}

TEST_CASE(value_bracket_object)
{
    migraphx::value v;
    v["one"]   = 1;
    v["two"]   = migraphx::value(2);
    v["three"] = 3;

    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");
}

TEST_CASE(value_insert_object)
{
    migraphx::value v;
    v.insert({"one", 1});
    v.insert({"two", migraphx::value(2)});
    v.insert({"three", 3});
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_emplace_object)
{
    migraphx::value v;
    v.emplace("one", 1);
    v.emplace("two", migraphx::value(2));
    v.emplace("three", 3);
    EXPECT(v.is_object());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_bracket_convert_throws)
{
    migraphx::value v1;
    EXPECT(test::throws([&] { v1["key"].to<std::string>(); }));
}

TEST_CASE(value_construct_object_string_value)
{
    migraphx::value v = {{"one", "onev"}, {"two", "twov"}};
    EXPECT(v.is_object());
    EXPECT(v.size() == 2);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.at("one").is_string());
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("one").get_string() == "onev");
    EXPECT(v.at("two").is_string());
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("two").get_string() == "twov");
}

TEST_CASE(value_construct_object_string_mixed_value)
{
    migraphx::value v = {{"one", "onev"}, {"two", 2}};
    EXPECT(v.is_object());
    EXPECT(v.size() == 2);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.at("one").is_string());
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("one").get_string() == "onev");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("two").get_int64() == 2);
}

template <class Expression>
static auto compare_predicate(const Expression& e)
{
    bool result = e.value();
    return test::make_predicate(test::as_string(e) + " => " + test::as_string(result),
                                [=] { return result; });
}

TEST_CASE(value_compare)
{
    EXPECT(migraphx::value(1) == migraphx::value(1));
    EXPECT(migraphx::value("key", 1) == migraphx::value("key", 1));
    EXPECT(migraphx::value(1) != migraphx::value(2));
    EXPECT(migraphx::value("key", 1) != migraphx::value("key", 2));
    EXPECT(migraphx::value("key1", 1) != migraphx::value("key2", 1));
    EXPECT(migraphx::value(1) < migraphx::value(2));
    EXPECT(migraphx::value(1) <= migraphx::value(2));
    EXPECT(migraphx::value(1) <= migraphx::value(1));
    EXPECT(migraphx::value(2) > migraphx::value(1));
    EXPECT(migraphx::value(2) >= migraphx::value(1));
    EXPECT(migraphx::value(1) >= migraphx::value(1));
    EXPECT(migraphx::value(1) != migraphx::value("1"));
    EXPECT(migraphx::value(1) != migraphx::value());
}

// NOLINTNEXTLINE
#define MIGRAPHX_VALUE_TEST_COMPARE(...) compare_predicate(TEST_CAPTURE(__VA_ARGS__))

// NOLINTNEXTLINE
#define EXPECT_TOTALLY_ORDERED_IMPL(_, x, y)     \
    EXPECT(_(x <= y) or _(x >= y));              \
    EXPECT(_(x < y) or _(x > y) or _(x == y));   \
    EXPECT((_(x < y) or _(x > y)) == _(x != y)); \
    EXPECT(_(x < y) == _(y > x));                \
    EXPECT(_(x <= y) == _(y >= x));              \
    EXPECT(_(x < y) != _(x >= y));               \
    EXPECT(_(x > y) != _(x <= y));               \
    EXPECT(_(x == y) != _(x != y))

// NOLINTNEXTLINE
#define EXPECT_TOTALLY_ORDERED(x, y)                                \
    EXPECT_TOTALLY_ORDERED_IMPL(MIGRAPHX_VALUE_TEST_COMPARE, x, y); \
    EXPECT_TOTALLY_ORDERED_IMPL(MIGRAPHX_VALUE_TEST_COMPARE, y, x)

// NOLINTNEXTLINE(readability-function-size)
TEST_CASE(value_compare_ordered)
{
    EXPECT_TOTALLY_ORDERED(migraphx::value(), migraphx::value());
    EXPECT_TOTALLY_ORDERED(migraphx::value(1), migraphx::value(1));
    EXPECT_TOTALLY_ORDERED(migraphx::value(1), migraphx::value(2));
    EXPECT_TOTALLY_ORDERED(migraphx::value("key", 1), migraphx::value("key", 1));
    EXPECT_TOTALLY_ORDERED(migraphx::value("key1", 1), migraphx::value("key2", 2));
    EXPECT_TOTALLY_ORDERED(migraphx::value("key", 1), migraphx::value("key", 2));
    EXPECT_TOTALLY_ORDERED(migraphx::value("key1", 1), migraphx::value("key2", 2));
    EXPECT_TOTALLY_ORDERED(migraphx::value("key", 1), migraphx::value("key", "2"));
    EXPECT_TOTALLY_ORDERED(migraphx::value("key1", 1), migraphx::value("key2", "2"));
    EXPECT_TOTALLY_ORDERED(migraphx::value(std::int64_t{1}), migraphx::value(std::uint64_t{1}));
    EXPECT_TOTALLY_ORDERED(migraphx::value(std::int64_t{1}), migraphx::value(std::uint64_t{2}));
    EXPECT_TOTALLY_ORDERED(migraphx::value(std::int64_t{2}), migraphx::value(std::uint64_t{1}));
    EXPECT_TOTALLY_ORDERED(migraphx::value(1), migraphx::value("1"));
    EXPECT_TOTALLY_ORDERED(migraphx::value(1), migraphx::value());
}

TEST_CASE(value_to_from_string)
{
    migraphx::value v = "1";
    EXPECT(v.to<std::string>() == "1");
    EXPECT(v.to<int>() == 1);
    EXPECT(migraphx::float_equal(v.to<float>(), 1.0));
}

TEST_CASE(value_to_from_int)
{
    migraphx::value v = 1;
    EXPECT(v.to<std::string>() == "1");
    EXPECT(v.to<int>() == 1);
    EXPECT(migraphx::float_equal(v.to<float>(), 1.0));
}

TEST_CASE(value_to_from_float)
{
    migraphx::value v = 1.5;
    EXPECT(v.to<std::string>() == "1.5");
    EXPECT(v.to<int>() == 1);
    EXPECT(migraphx::float_equal(v.to<float>(), 1.5));
}

TEST_CASE(value_to_from_pair)
{
    migraphx::value v = {"one", 1};
    EXPECT(bool{v.to<std::pair<std::string, std::string>>() ==
                std::pair<std::string, std::string>("one", "1")});
    EXPECT(bool{v.to<std::pair<std::string, int>>() == std::pair<std::string, int>("one", 1)});
    EXPECT(
        bool{v.to<std::pair<std::string, float>>() == std::pair<std::string, float>("one", 1.0)});
}

TEST_CASE(value_to_struct)
{
    migraphx::value v = 1;
    struct local
    {
        int i   = 0;
        local() = default;
        local(int ii) : i(ii) {}
    };
    EXPECT(v.to<local>().i == 1);
}

TEST_CASE(value_to_error1)
{
    migraphx::value v = {1, 2, 3};
    EXPECT(test::throws([&] { v.to<int>(); }));
}

TEST_CASE(value_to_error2)
{
    migraphx::value v = 1;
    struct local
    {
    };
    EXPECT(test::throws([&] { v.to<local>(); }));
}

TEST_CASE(value_to_error_parse)
{
    migraphx::value v = "abc";
    EXPECT(test::throws([&] { v.to<int>(); }));
}

TEST_CASE(value_to_vector)
{
    migraphx::value v  = {1, 2, 3};
    std::vector<int> a = {1, 2, 3};
    EXPECT(v.to_vector<int>() == a);
}

TEST_CASE(not_array)
{
    migraphx::value v = 1;
    EXPECT(v.size() == 0);
    EXPECT(not v.contains("???"));
    EXPECT(test::throws([&] { v.at(0); }));
    EXPECT(test::throws([&] { v.at("???"); }));
    EXPECT(v.data() == nullptr);
    [=] {
        EXPECT(test::throws([&] { v.at(0); }));
        EXPECT(test::throws([&] { v.at("???"); }));
        EXPECT(v.data() == nullptr);
    }();
}

TEST_CASE(print)
{
    std::stringstream ss;
    migraphx::value v = {1, {{"one", 1}, {"two", 2}}, {1, 2}, {}};
    ss << v;
    EXPECT(ss.str() == "{1, {one: 1, two: 2}, {1, 2}, null}");
}

TEST_CASE(value_clear)
{
    migraphx::value values = {1, 2, 3};
    EXPECT(values.is_array());
    EXPECT(values.size() == 3);
    values.clear();
    EXPECT(values.empty());

    values.push_back(3);
    EXPECT(values.size() == 1);
    EXPECT(values.at(0).to<int>() == 3);
}

TEST_CASE(value_clear_non_array)
{
    migraphx::value values = 1.0;
    EXPECT(test::throws([&] { values.clear(); }));
}

TEST_CASE(value_clear_object)
{
    migraphx::value values = {{"a", 1}, {"b", 2}};
    EXPECT(values.is_object());
    EXPECT(values.size() == 2);
    values.clear();
    EXPECT(values.empty());

    values["c"] = 3;
    EXPECT(values.size() == 1);
    EXPECT(values.at("c").to<int>() == 3);
}

TEST_CASE(value_clear_empty_array)
{
    migraphx::value values = migraphx::value::array{};
    EXPECT(values.empty());
    values.clear();
    EXPECT(values.empty());
}

TEST_CASE(value_clear_empty_object)
{
    migraphx::value values = migraphx::value::object{};
    EXPECT(values.empty());
    values.clear();
    EXPECT(values.empty());
}

TEST_CASE(value_resize)
{
    migraphx::value values = {1, 2, 3};
    EXPECT(values.is_array());
    EXPECT(values.size() == 3);
    values.resize(5);
    EXPECT(values.size() == 5);

    EXPECT(values.at(3).is_null());
    EXPECT(values.at(4).is_null());
}

TEST_CASE(value_resize_with_value)
{
    migraphx::value values = {1, 2, 3};
    EXPECT(values.is_array());
    EXPECT(values.size() == 3);
    values.resize(5, 7);
    EXPECT(values.size() == 5);

    EXPECT(values.at(3).to<int>() == 7);
    EXPECT(values.at(4).to<int>() == 7);
}

TEST_CASE(value_resize_empty_array)
{
    migraphx::value values = migraphx::value::array{};
    EXPECT(values.is_array());
    EXPECT(values.empty());
    values.resize(3);
    EXPECT(values.size() == 3);

    EXPECT(values.at(0).is_null());
    EXPECT(values.at(1).is_null());
    EXPECT(values.at(2).is_null());
}

TEST_CASE(value_resize_object)
{
    migraphx::value values = migraphx::value::object{};
    EXPECT(values.is_object());
    EXPECT(test::throws([&] { values.resize(4); }));
}

TEST_CASE(value_resize_n_object)
{
    migraphx::value values = migraphx::value::object{};
    EXPECT(values.is_object());
    EXPECT(test::throws([&] { values.resize(4, ""); }));
}

TEST_CASE(value_assign_construct_from_vector)
{
    std::vector<int> v     = {1, 2, 3};
    migraphx::value values = v;
    EXPECT(values.to_vector<int>() == v);
}

TEST_CASE(value_construct_from_vector)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::value values(v);
    EXPECT(values.to_vector<int>() == v);
}

TEST_CASE(value_assign_from_vector)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::value values{};
    values = v;
    EXPECT(values.to_vector<int>() == v);
}

TEST_CASE(value_init_from_vector)
{
    std::vector<int> v     = {1, 2, 3};
    migraphx::value values = {{"a", v}};
    EXPECT(values.at("a").to_vector<int>() == v);
}

TEST_CASE(value_binary_default)
{
    migraphx::value v;
    v = migraphx::value::binary{};
    EXPECT(v.is_binary());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_binary)
{
    migraphx::value v;
    std::vector<std::uint8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    v = migraphx::value::binary{data};
    EXPECT(v.is_binary());
    EXPECT(v.get_binary().size() == data.size());
    EXPECT(v.get_binary() == data);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_binary_object)
{
    std::vector<std::uint8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    migraphx::value v = {{"data", migraphx::value::binary{data}}};

    EXPECT(v["data"].is_binary());
    EXPECT(v["data"].get_binary().size() == data.size());
    EXPECT(v["data"].get_binary() == data);
}

TEST_CASE(value_binary_object_conv)
{
    std::vector<std::int8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    migraphx::value v = {{"data", migraphx::value::binary{data}}};

    EXPECT(v["data"].is_binary());
    EXPECT(v["data"].get_binary().size() == data.size());
    EXPECT(migraphx::equal(v["data"].get_binary(), data));
}

template <class T>
static bool is_null_type(const T&)
{
    return false;
}

static bool is_null_type(std::nullptr_t) { return true; }

TEST_CASE(visit_null)
{
    migraphx::value v;
    EXPECT(v.is_null());
    bool visited = false;
    v.visit([&](auto&& x) { visited = is_null_type(x); });
    EXPECT(visited);
}

TEST_CASE(value_or_convert)
{
    migraphx::value v = 1;
    EXPECT(v.is_int64());
    EXPECT(v.value_or(3) == 1);
}

TEST_CASE(value_or_null)
{
    migraphx::value v;
    EXPECT(v.is_null());
    EXPECT(v.value_or(3) == 3);
}

TEST_CASE(value_get_default)
{
    migraphx::value v = {{"key", 1}};
    EXPECT(v.get("key", 3) == 1);
    EXPECT(v.get("missing", 3) == 3);
}

TEST_CASE(value_get_default_vector)
{
    std::vector<int> ints     = {1, 2, 3};
    std::vector<int> fallback = {-1};
    migraphx::value v         = {{"key", ints}};
    EXPECT(v.get("key", fallback) == ints);
    EXPECT(v.get("missing", fallback) == fallback);
    EXPECT(v.get("missing", {-1}) == fallback);
}

TEST_CASE(value_get_default_string_literal)
{
    migraphx::value v = {{"key", "hello"}};
    EXPECT(v.get("key", "none") == "hello");
    EXPECT(v.get("missing", "none") == "none");
}

TEST_CASE(value_get_default_string_literal_vector)
{
    std::vector<std::string> strings  = {"1", "2", "3"};
    std::vector<std::string> fallback = {"none"};
    migraphx::value v                 = {{"key", strings}};
    EXPECT(v.get("key", fallback) == strings);
    EXPECT(v.get("missing", fallback) == fallback);
    EXPECT(v.get("missing", {"none"}) == fallback);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
