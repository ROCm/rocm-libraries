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
#include <onnx_verify_utils.hpp>

TEST_CASE(unpack_int4_qdq_test)
{
    migraphx::program p = read_onnx("int4_const_identity_qdq_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x2_shape{migraphx::shape::half_type, {4, 4}};
    std::vector<migraphx::half> x2(16, migraphx::half(1.0));

    migraphx::parameter_map pm;
    pm["x2"] = migraphx::argument{x2_shape, x2.data()};

    auto result = p.eval(pm).back();

    std::vector<migraphx::half> rv;
    result.visit([&](auto output) { rv.assign(output.begin(), output.end()); });

    // MatMul output is 4x4: should be all ones:
    // Based on Identity matrix A supplied to AxB. (B aka 'x2' is all ones)
    std::vector<migraphx::half> gold(16, migraphx::half(1.0));
    EXPECT(rv == gold);
}

TEST_CASE(unpack_int4_block_sz_2_qdq_test)
{
    migraphx::program p = read_onnx("int4_const_identity_block_sz_2_qdq_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x2_shape{migraphx::shape::half_type, {2, 4}};
    std::vector<migraphx::half> x2(16, migraphx::half(1.0));

    migraphx::parameter_map pm;
    pm["x2"] = migraphx::argument{x2_shape, x2.data()};

    auto result = p.eval(pm).back();

    std::vector<migraphx::half> rv;
    result.visit([&](auto output) { rv.assign(output.begin(), output.end()); });

    // MatMul output is 4x4: its first half should be all zeros. Rest all ones:
    // Based on partial Identity matrix A supplied to AxB. (B aka 'x2' is all ones).
    std::vector<migraphx::half> gold(16);
    std::fill(gold.begin() + 8, gold.end(), migraphx::half(1.0));

    EXPECT(rv == gold);
}
