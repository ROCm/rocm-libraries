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
#include <migraphx/convert_to_json.hpp>
#include <test.hpp>

TEST_CASE(key_int)
{
    std::string str = "{abc:{key:1}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1}}");
}

TEST_CASE(key_negative_int)
{
    std::string str = "{abc:{key:-1}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":-1}}");
}

TEST_CASE(key_float)
{
    std::string str = "{abc:{key:1.0}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1.0}}");
}

TEST_CASE(key_negative_float)
{
    std::string str = "{abc:{key:-1.0}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":-1.0}}");
}

TEST_CASE(key_exp)
{
    std::string str = "{abc:{key:1e+10}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1e+10}}");
}

TEST_CASE(key_exp_1)
{
    std::string str = "{abc:{key:1E-10}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1E-10}}");
}

TEST_CASE(key_null)
{
    std::string str = "{abc:{key:null}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":null}}");
}

TEST_CASE(key_inf)
{
    std::string str = "{abc:{key:inf}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":inf}}");
}

TEST_CASE(key_neg_inf)
{
    std::string str = "{abc:{key:-inf}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":-inf}}");
}

TEST_CASE(key_true)
{
    std::string str = "{abc:{key:true}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":true}}");
}

TEST_CASE(key_false)
{
    std::string str = "{abc:{key:false}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":false}}");
}

TEST_CASE(key_nan)
{
    std::string str = "{abc:{key:nan}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":nan}}");
}

TEST_CASE(quote_key_num)
{
    std::string str = R"({"abc":{"key":1}})";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1}}");
}

TEST_CASE(quote_with_space_key_num)
{
    std::string str = R"({"abc key":{"key":1}})";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc key\":{\"key\":1}}");
}

TEST_CASE(key_value_num_space)
{
    std::string str = "{abc    :    {    key    :    1}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\"    :    {    \"key\"    :    1}}");
}

TEST_CASE(key_value_str)
{
    std::string str = "{abc : {key : value}}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\" : {\"key\" : \"value\"}}");
}

TEST_CASE(key_space_value)
{
    std::string str = "{abc    : [key, value]}";
    auto jstr       = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\"    : [\"key\", \"value\"]}");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
