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
#include <test.hpp>
#include <migraphx/gpu/pack_args.hpp>

template <class T>
static std::size_t packed_sizes()
{
    return sizeof(T);
}

template <class T, class U, class... Ts>
static std::size_t packed_sizes()
{
    return sizeof(T) + packed_sizes<U, Ts...>();
}

template <class... Ts>
static std::size_t sizes()
{
    return migraphx::gpu::pack_args({Ts{}...}).size();
}

template <class... Ts>
static std::size_t padding()
{
    EXPECT(sizes<Ts...>() >= packed_sizes<Ts...>());
    return sizes<Ts...>() - packed_sizes<Ts...>();
}

struct float_struct
{
    float x, y;
};

TEST_CASE(alignment_padding)
{
    EXPECT(padding<short, short>() == 0);
    EXPECT(padding<float, float_struct>() == 0);
    EXPECT(padding<short, float_struct>() == 2);
    EXPECT(padding<short, int>() == 2);
    EXPECT(padding<char, short, int, char>() == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
