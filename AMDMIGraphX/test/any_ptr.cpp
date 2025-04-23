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
#include <migraphx/any_ptr.hpp>
#include <test.hpp>

TEST_CASE(test_int_id)
{
    int i               = 1;
    migraphx::any_ptr p = &i;
    EXPECT(p.get<int*>() == &i);
    EXPECT(p.get(migraphx::get_type_name(i)) == &i);
    EXPECT(p.unsafe_get() == &i);
    EXPECT(test::throws([&] { p.get<float*>(); }));
    EXPECT(test::throws([&] { p.get(migraphx::get_type_name(&i)); }));
}

TEST_CASE(test_int_name)
{
    int i    = 1;
    void* vp = &i;
    migraphx::any_ptr p{vp, migraphx::get_type_name(i)};
    EXPECT(p.get<int*>() == &i);
    EXPECT(p.get(migraphx::get_type_name(i)) == &i);
    EXPECT(p.unsafe_get() == &i);
    EXPECT(test::throws([&] { p.get<float*>(); }));
    EXPECT(test::throws([&] { p.get(migraphx::get_type_name(&i)); }));
    EXPECT(test::throws([&] { p.get(migraphx::get_type_name(float{})); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
