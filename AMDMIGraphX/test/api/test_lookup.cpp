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
#include <migraphx/migraphx.hpp>
#include "test.hpp"

template <class T>
static std::false_type has_handle(migraphx::rank<0>, T)
{
    return {};
}

template <class T>
static auto has_handle(migraphx::rank<1>, T*) -> decltype(migraphx::as_handle<T>{},
                                                          std::true_type{})
{
    return {};
}

TEST_CASE(shape)
{
    static_assert(std::is_same<migraphx::as_handle<migraphx_shape>, migraphx::shape>{}, "Failed");
    static_assert(std::is_same<migraphx::as_handle<migraphx_shape_t>, migraphx::shape>{}, "Failed");
    static_assert(std::is_same<migraphx::as_handle<const_migraphx_shape_t>, migraphx::shape>{},
                  "Failed");
}
TEST_CASE(non_handle)
{
    int i = 0;
    EXPECT(bool{has_handle(migraphx::rank<1>{}, migraphx_shape_t{})});
    EXPECT(bool{not has_handle(migraphx::rank<1>{}, &i)});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
