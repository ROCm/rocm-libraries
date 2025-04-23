/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/serialize.hpp>
#include <migraphx/functional.hpp>
#include <test.hpp>

#include <numeric>
#include <map>
#include <vector>

struct empty_type
{
};
struct reflectable_type
{
    enum simple_enum
    {
        simple1,
        simple2,
        simple3
    };
    enum class class_enum
    {
        class1,
        class2,
        class3
    };
    std::vector<std::size_t> ints = {};
    std::string name              = "";
    float fvalue                  = 0.0;
    empty_type et{};
    simple_enum se = simple1;
    class_enum ce  = class_enum::class1;

    struct nested_type
    {
        int value;
        template <class Self, class F>
        static auto reflect(Self& self, F f)
        {
            return migraphx::pack(f(self.value, "value"));
        }
    };
    std::vector<nested_type> nested_types                 = {};
    std::tuple<int, nested_type, std::string> tuple_items = std::make_tuple(0, nested_type{0}, "");
    migraphx::optional<int> opt_value                     = migraphx::nullopt;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.ints, "ints"),
                              f(self.name, "name"),
                              f(self.fvalue, "fvalue"),
                              f(self.et, "et"),
                              f(self.se, "se"),
                              f(self.ce, "ce"),
                              f(self.nested_types, "nested_types"),
                              f(self.tuple_items, "tuple_items"));
    }
};

TEST_CASE(serialize_reflectable_type)
{
    reflectable_type t1{{1, 2},
                        "hello",
                        1.0,
                        {},
                        reflectable_type::simple1,
                        reflectable_type::class_enum::class2,
                        {{1}, {2}},
                        {5, {4}, "hello"},
                        {migraphx::nullopt}};
    migraphx::value v1  = migraphx::to_value(t1);
    reflectable_type t2 = migraphx::from_value<reflectable_type>(v1);
    migraphx::value v2  = migraphx::to_value(t2);
    migraphx::value v3  = migraphx::to_value(reflectable_type{});

    EXPECT(v1 == v2);
    EXPECT(v1 != v3);
    EXPECT(v2 != v3);
}

TEST_CASE(serialize_empty_array)
{
    std::vector<std::size_t> ints = {};
    migraphx::value v             = migraphx::to_value(ints);
    EXPECT(v.is_array());
    EXPECT(v.empty());
    v.push_back(1);
    EXPECT(v.size() == 1);
    EXPECT(v.front().to<int>() == 1);
}

struct empty_struct
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return migraphx::pack();
    }
};

TEST_CASE(serialize_empty_struct)
{
    empty_struct es{};
    migraphx::value v = migraphx::to_value(es);
    EXPECT(v.is_object());
    EXPECT(v.empty());
    v["a"] = 1;
    EXPECT(v.size() == 1);
    EXPECT(v.at("a").to<int>() == 1);
}

TEST_CASE(serialize_empty_optional)
{
    migraphx::optional<int> x{};
    migraphx::value v = migraphx::to_value(x);
    EXPECT(v.is_null());
}

TEST_CASE(serialize_optional)
{
    migraphx::optional<int> x{2};
    migraphx::value v = migraphx::to_value(x);
    EXPECT(v.is_int64());
    EXPECT(v.to<int>() == 2);
}

TEST_CASE(serialize_map)
{
    std::map<int, int> m = {{1, 1}, {2, 4}, {3, 9}};
    migraphx::value v    = migraphx::to_value(m);
    EXPECT(v.is_array());
    EXPECT(m == migraphx::from_value<std::map<int, int>>(v));
}

TEST_CASE(serialize_invalid_map1)
{
    migraphx::value v = {{1, 1}, {2, 4}, {3, 9, 27}};
    EXPECT(test::throws([&] { migraphx::from_value<std::map<int, int>>(v); }));
}

TEST_CASE(serialize_invalid_map2)
{
    migraphx::value v = 2;
    EXPECT(test::throws([&] { migraphx::from_value<std::map<int, int>>(v); }));
}

TEST_CASE(serialize_struct_to_map)
{
    migraphx::value v = migraphx::to_value(reflectable_type{});
    EXPECT(v.is_object());
    std::map<std::string, migraphx::value> m;
    migraphx::from_value(v, m);
    EXPECT(m.size() == v.size());
}

TEST_CASE(serialize_vector_value)
{
    std::vector<migraphx::value> x = {{1}, {2}};
    migraphx::value v              = migraphx::to_value(x);
    EXPECT(x == migraphx::from_value<std::vector<migraphx::value>>(v));
}

TEST_CASE(from_value_binary)
{
    std::vector<std::uint8_t> data(10);
    std::iota(data.begin(), data.end(), 0);

    migraphx::value v = migraphx::value::binary{data};

    auto out = migraphx::from_value<migraphx::value::binary>(v);
    EXPECT(out == data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
