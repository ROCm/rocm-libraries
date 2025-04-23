/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>
#include <random>

TEST_CASE(randomuniformlike_test)
{
    float high = 10.0;
    float low  = 1.0;
    float seed = 0.0;
    std::vector<int> shape_attr{2, 3, 4};

    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::half_type, shape_attr};
    std::vector<double> rand_vals(s.elements());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> d(low, high);
    std::generate(rand_vals.begin(), rand_vals.end(), [&]() { return d(gen); });

    mm->add_parameter("input", s);
    mm->add_literal(migraphx::literal{s, rand_vals});

    auto prog = optimize_onnx("randomuniformlike_test.onnx");

    EXPECT(p == prog);
}
