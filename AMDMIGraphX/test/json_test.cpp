/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/json.hpp>
#include <test.hpp>

TEST_CASE(null_value)
{
    migraphx::value v;
    auto json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "null");
}

TEST_CASE(null_value_rev)
{
    std::string json_str = "null";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev;
    EXPECT(v == ev);
}

TEST_CASE(null_array)
{
    migraphx::value v;
    migraphx::value arr = {v, v};
    auto json_str       = migraphx::to_json_string(arr);
    EXPECT(json_str == "[null,null]");
}

TEST_CASE(null_array_rev)
{
    std::string json_str = "[null,null]";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value e;
    migraphx::value ev = {e, e};
    EXPECT(ev == v);
}

TEST_CASE(empty_object1)
{
    migraphx::value val = migraphx::from_json_string("{}");
    EXPECT(val == migraphx::value::object{});
    EXPECT(migraphx::to_json_string(migraphx::value::object{}) == "{}");
}

TEST_CASE(empty_array1)
{
    migraphx::value val = migraphx::from_json_string("[]");
    EXPECT(val == migraphx::value::array{});
    EXPECT(migraphx::to_json_string(migraphx::value::array{}) == "[]");
}

TEST_CASE(int_value)
{
    migraphx::value v    = -1;
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "-1");
}

TEST_CASE(int_value_rev)
{
    std::string json_str = "-1";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = -1;
    EXPECT(v == ev);
}

TEST_CASE(unsigned_value)
{
    migraphx::value v    = 1;
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "1");
}

TEST_CASE(unsigned_value_rev)
{
    std::string json_str = "1";
    migraphx::value v    = migraphx::from_json_string(json_str);
    EXPECT(v.is_uint64());
    EXPECT(v.get_uint64() == 1);
}

TEST_CASE(float_value)
{
    migraphx::value v    = 1.5;
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "1.5");
}

TEST_CASE(float_value_rev)
{
    std::string json_str = "1.5";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = 1.5;
    EXPECT(v == ev);
}

TEST_CASE(array_value)
{
    migraphx::value v    = {1, 2};
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "[1,2]");
}

TEST_CASE(array_value_rev)
{
    std::string json_str = "[1,2]";
    migraphx::value v    = migraphx::from_json_string(json_str);
    EXPECT(v.is_array());
    EXPECT(v.size() == 2);
    EXPECT(v[0].get_uint64() == 1);
    EXPECT(v[1].get_uint64() == 2);
}

TEST_CASE(object_value)
{
    migraphx::value v    = {{"a", 1.2}, {"b", true}};
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "{\"a\":1.2,\"b\":true}");
}

TEST_CASE(object_value_rev)
{
    std::string json_str = R"({"a":1.2,"b":true})";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = {{"a", 1.2}, {"b", true}};
    EXPECT(v == ev);
}

TEST_CASE(null_object)
{
    migraphx::value v;
    migraphx::value v1   = {{"a", v}};
    std::string json_str = migraphx::to_json_string(v1);
    EXPECT(json_str == "{\"a\":null}");
}

TEST_CASE(null_object_rev)
{
    std::string json_str = R"({"a":null})";
    migraphx::value eo   = migraphx::from_json_string(json_str);
    migraphx::value v;
    migraphx::value ev = {{"a", v}};
    EXPECT(eo == ev);
}

TEST_CASE(string_value)
{
    migraphx::value v    = "string_test";
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "\"string_test\"");
}

TEST_CASE(string_value_rev)
{
    std::string json_str = "\"string_test\"";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = "string_test";
    EXPECT(v == ev);
}

TEST_CASE(array_of_objects)
{
    migraphx::value obj1 = {"key1", uint64_t{1}};
    migraphx::value obj2 = {"key2", uint64_t{2}};
    migraphx::value arr  = {obj1, obj2};
    std::string json_str = migraphx::to_json_string(arr);
    EXPECT(json_str == "{\"key1\":1,\"key2\":2}");
}

TEST_CASE(array_of_objects_rev)
{
    std::string json_str = R"({"key1":1,"key2":2})";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value obj1 = {"key1", uint64_t{1}};
    migraphx::value obj2 = {"key2", uint64_t{2}};
    migraphx::value arr  = {obj1, obj2};
    EXPECT(arr == v);
}

TEST_CASE(object_of_array)
{
    migraphx::value obj1 = {"key1", 1};
    migraphx::value obj2 = {"key2", 2};
    migraphx::value obj;
    obj["key"]           = {obj1, obj2};
    std::string json_str = migraphx::to_json_string(obj);
    EXPECT(json_str == "{\"key\":{\"key1\":1,\"key2\":2}}");
}

TEST_CASE(object_of_array_rev)
{
    std::string json_str = R"({"key":{"key1":1,"key2":2}})";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value obj1 = {"key1", uint64_t{1}};
    migraphx::value obj2 = {"key2", uint64_t{2}};
    migraphx::value obj;
    obj["key"] = {obj1, obj2};
    EXPECT(v == obj);
}

TEST_CASE(shape_value)
{
    migraphx::shape s{migraphx::shape::int32_type, {2, 3, 4, 5}};
    migraphx::value val     = migraphx::to_value(s);
    std::string json_str    = migraphx::to_json_string(val);
    migraphx::value val_rev = migraphx::from_json_string(json_str);
    migraphx::shape s_rev;
    migraphx::from_value(val_rev, s_rev);

    EXPECT(s == s_rev);
}

TEST_CASE(argument_value)
{
    migraphx::shape s{migraphx::shape::int32_type, {2, 3, 4, 5}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 1);
    migraphx::argument argu = migraphx::argument(s, data.data());

    migraphx::value val     = migraphx::to_value(argu);
    std::string json_str    = migraphx::to_json_string(val);
    migraphx::value val_rev = migraphx::from_json_string(json_str);
    migraphx::argument argu_rev;
    migraphx::from_value(val_rev, argu_rev);

    EXPECT(argu == argu_rev);
}

TEST_CASE(literal_value)
{
    migraphx::shape s{migraphx::shape::int32_type, {2, 3, 4, 5}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 1);
    migraphx::literal l = migraphx::literal(s, data);

    migraphx::value val     = migraphx::to_value(l);
    std::string json_str    = migraphx::to_json_string(val);
    migraphx::value val_rev = migraphx::from_json_string(json_str);
    migraphx::literal l_rev;
    migraphx::from_value(val_rev, l_rev);

    EXPECT(l == l_rev);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
