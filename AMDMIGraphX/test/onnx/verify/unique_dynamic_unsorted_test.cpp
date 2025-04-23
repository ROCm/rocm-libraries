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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(unique_dynamic_unsorted_test)
{
    migraphx::program p = read_onnx("unique_dynamic_unsorted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x{2, 1, 1, 3, 4, 3};
    std::vector<float> y_gold      = {2, 1, 3, 4};
    std::vector<size_t> y_idx_gold = {0, 1, 3, 4};
    std::vector<size_t> x_idx_gold = {0, 1, 1, 2, 3, 2};
    std::vector<size_t> y_ct_gold  = {1, 2, 2, 1};
    migraphx::shape s{migraphx::shape::float_type, {x.size()}};

    migraphx::parameter_map pm;
    pm["X"]     = migraphx::argument(s, x.data());
    auto result = p.eval(pm);

    std::vector<float> yvec;
    result[0].visit([&](auto out) { yvec.assign(out.begin(), out.end()); });
    EXPECT(yvec == y_gold);

    std::vector<size_t> y_idx_vec;
    result[1].visit([&](auto out) { y_idx_vec.assign(out.begin(), out.end()); });
    EXPECT(y_idx_vec == y_idx_gold);

    std::vector<size_t> x_idx_vec;
    result[2].visit([&](auto out) { x_idx_vec.assign(out.begin(), out.end()); });
    EXPECT(x_idx_vec == x_idx_gold);

    std::vector<size_t> y_ct_vec;
    result[3].visit([&](auto out) { y_ct_vec.assign(out.begin(), out.end()); });
    EXPECT(y_ct_vec == y_ct_gold);
}
