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

#include <onnx_test.hpp>

TEST_CASE(multinomial_dyn_test)
{
    // compile-time random seed
    migraphx::program p;
    auto* mm           = p.get_main_module();
    size_t sample_size = 100000;
    size_t categories  = 5;
    float seed         = 1.3f;

    auto input = mm->add_parameter(
        "input",
        migraphx::shape{migraphx::shape::float_type, {{1, categories}, {categories, categories}}});

    auto maxes = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);

    auto cdf = add_common_op(*mm, migraphx::make_op("sub"), {input, maxes});
    cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
    cdf      = mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

    migraphx::shape s{migraphx::shape::float_type, {1}};
    std::vector<float> seed_data = {seed};
    auto seed_input              = mm->add_literal(migraphx::literal(s, seed_data));

    // dynamic input only:  must calculate alloc_shape as (batch_size, sample_size)
    //                read the runtime input dimensions
    auto dim_of = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 2}}), input);
    // make an argument of (1, 0)
    migraphx::shape lit_shape(migraphx::shape::int64_type, {2});
    std::vector<int64_t> data1{1, 0};
    auto l1        = mm->add_literal(lit_shape, data1);
    auto batch_arg = mm->add_instruction(migraphx::make_op("mul"), dim_of, l1);
    std::vector<int64_t> data2(2, 0);
    // make an argument of (0, sample_size)
    data2[1]         = sample_size;
    auto l2          = mm->add_literal(lit_shape, data2);
    auto alloc_shape = mm->add_instruction(migraphx::make_op("add"), batch_arg, l2);
    migraphx::shape compile_shape =
        migraphx::shape(migraphx::shape::float_type,
                        {input->get_shape().dyn_dims().front(), {sample_size, sample_size}});

    auto alloc = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", to_value(compile_shape)}}), alloc_shape);

    auto randoms = mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, alloc);
    auto ret     = mm->add_instruction(
        migraphx::make_op("multinomial", {{"dtype", migraphx::shape::float_type}}), cdf, randoms);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value  = {1, categories};
    options.print_program_on_error = true;
    auto prog                      = read_onnx("multinomial_dyn_test.onnx", options);
    EXPECT(p == prog);
}
