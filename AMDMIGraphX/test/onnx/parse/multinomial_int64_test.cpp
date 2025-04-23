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

TEST_CASE(multinomial_int64_test)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    size_t sample_size            = 10;
    float seed                    = 1.0;
    uint32_t batch_size           = 1;
    migraphx::shape::type_t dtype = migraphx::shape::type_t::int64_type;

    auto input = mm->add_parameter("input", migraphx::shape{migraphx::shape::float_type, {1, 10}});
    auto maxes = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);

    auto cdf = add_common_op(*mm, migraphx::make_op("sub"), {input, maxes});
    cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
    cdf      = mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

    migraphx::shape s{migraphx::shape::float_type, {1}};
    std::vector<float> data = {seed};
    auto seed_input         = mm->add_literal(migraphx::literal(s, data));

    // static size
    auto rand_dummy = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {batch_size, sample_size}},
                          std::vector<float>(batch_size * sample_size)});
    auto randoms = mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, rand_dummy);
    mm->add_instruction(migraphx::make_op("multinomial", {{"dtype", dtype}}), cdf, randoms);
    auto prog = optimize_onnx("multinomial_int64_test.onnx");

    EXPECT(p == prog);
}
