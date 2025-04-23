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
#include <migraphx/msgpack.hpp>
#include <migraphx/value.hpp>
#include <msgpack.hpp>
#include <map>
#include <numeric>
#include "test.hpp"

template <class T, MIGRAPHX_REQUIRES(not std::is_base_of<std::vector<std::uint8_t>, T>{})>
static void write_msgpack(std::ostream& os, const T& src)
{
    msgpack::pack(os, src);
}
static void write_msgpack(std::ostream& os, const std::vector<std::uint8_t>& src)
{
    const auto limit = std::numeric_limits<uint32_t>::max() - 1;
    std::vector<std::vector<std::uint8_t>> chunks;
    if(src.size() > limit)
    {
        // Only test two chunks
        assert(std::distance(src.begin() + limit, src.end()) < limit);
        chunks.emplace_back(src.begin(), src.begin() + limit);
        chunks.emplace_back(src.begin() + limit, src.end());
    }
    else
    {
        chunks = {src};
    }
    write_msgpack(os, chunks);
}

template <class T>
static std::vector<char> msgpack_buffer(const T& src)
{
    std::stringstream buffer;
    write_msgpack(buffer, src);
    buffer.seekg(0);
    std::string str = buffer.str();
    return std::vector<char>(str.data(), str.data() + str.size()); // NOLINT
}

TEST_CASE(test_msgpack_empty_value)
{
    migraphx::value v;
    auto buffer = migraphx::to_msgpack(v);
    auto mp     = migraphx::from_msgpack(buffer);
    EXPECT(mp == v);
    EXPECT(v.is_null());
    EXPECT(mp.is_null());
}

TEST_CASE(test_msgpack_int)
{
    migraphx::value v = 3;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(3));
    EXPECT(migraphx::from_msgpack(buffer).to<int>() == v.to<int>());
}

TEST_CASE(test_msgpack_int_negative)
{
    migraphx::value v = -3;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(-3));
    EXPECT(migraphx::from_msgpack(buffer).to<int>() == v.to<int>());
}

TEST_CASE(test_msgpack_bool)
{
    migraphx::value v = true;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(true));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_float)
{
    // changed all double values in this code to not end with .0 because on msgpack for Windows if
    // input type is double and ends with .0 it could be converted to uint64_t or int64_t and the
    // goal of these functions is to test double without conversions
    migraphx::value v = 3.01;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(3.01));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_string)
{
    migraphx::value v = "abc";
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer("abc"));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_array)
{
    migraphx::value v = {1, 2, 3};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<int>{1, 2, 3}));
    EXPECT(migraphx::from_msgpack(buffer).to_vector<int>() == v.to_vector<int>());
}

TEST_CASE(test_msgpack_empty_array)
{
    migraphx::value v = migraphx::value::array{};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<int>{}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_object)
{
    migraphx::value v = {{"one", 1.01}, {"three", 3.01}, {"two", 2.01}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::map<std::string, double>{
                         {"one", 1.01}, {"three", 3.01}, {"two", 2.01}}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_empty_object)
{
    migraphx::value v = migraphx::value::object{};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<int>{}));
    auto u = migraphx::from_msgpack(buffer);
    // This is not equal since an empty object becomes an empty array
    EXPECT(u != v);
    EXPECT(u.is_array());
    EXPECT(u.size() == 0);
}

struct foo
{
    double a;
    std::string b;
    MSGPACK_DEFINE_MAP(a, b);
};

TEST_CASE(test_msgpack_object_class)
{
    migraphx::value v = {{"a", 1.01}, {"b", "abc"}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(foo{1.01, "abc"}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_array_class)
{
    migraphx::value v = {{{"a", 1.01}, {"b", "abc"}}, {{"a", 3.01}, {"b", "xyz"}}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<foo>{foo{1.01, "abc"}, foo{3.01, "xyz"}}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_binary)
{
    migraphx::value::binary bin{64};
    std::iota(bin.begin(), bin.end(), 1);
    auto buffer = migraphx::to_msgpack(bin);
    EXPECT(buffer == msgpack_buffer(bin));
    EXPECT(migraphx::from_msgpack(buffer) == bin);
}

#ifndef MIGRAPHX_DISABLE_LARGE_BUFFER_TESTS
TEST_CASE(test_msgpack_large_binary1)
{
    const std::size_t n   = 4LL * 1024 * 1024 * 1024 + 2;
    const char fill_value = 2;
    migraphx::value v;
    {
        std::vector<char> buffer;
        {
            migraphx::value::binary bin{n};
            std::fill(bin.begin(), bin.begin() + n / 2, fill_value);
            std::fill(bin.begin() + n / 2, bin.end(), fill_value + 1);
            buffer = migraphx::to_msgpack(std::move(bin));
        }
        v = migraphx::from_msgpack(buffer);
    }
    EXPECT(v.is_binary());
    EXPECT(v.get_binary().size() == n);
    EXPECT(std::all_of(v.get_binary().begin(), v.get_binary().begin() + n / 2, [](auto c) {
        return c == fill_value;
    }));
    EXPECT(std::all_of(v.get_binary().begin() + n / 2, v.get_binary().end(), [](auto c) {
        return c == fill_value + 1;
    }));
}

TEST_CASE(test_msgpack_binary2)
{
    const std::size_t n = 4LL * 1024 * 1024 * 1024 + 2;
    migraphx::value::binary bin{n};
    std::size_t i = 0;
    std::generate(bin.begin(), bin.end(), [&] {
        i++;
        return i % 256;
    });
    EXPECT(migraphx::to_msgpack(bin) == msgpack_buffer(bin));
}
#endif
int main(int argc, const char* argv[]) { test::run(argc, argv); }
