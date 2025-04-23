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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(scatternd_add_reduction_test)
{
    // reduction = add
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {8, 1}};
    migraphx::shape us{dtype, {8}};

    std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ind_vec{4, 3, 1, 7, 4, 3, 1, 7};
    std::vector<float> upd_vec{9, 10, 11, 12, -8, -9, -10, -11};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_add"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 3, 3, 5, 6, 6, 7, 9};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_reduction_dyn_test)
{
    // reduction = add, with dynamic input shapes
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape::dynamic_dimension dd{3, 6};
    migraphx::shape ds{migraphx::shape::float_type, {dd, dd, dd}};
    migraphx::shape is{itype, {2, 1}};
    migraphx::shape us{dtype, {{2, 2}, dd, dd}};

    auto xdata    = mm->add_parameter("X", ds);
    auto xindex   = mm->add_parameter("I", is);
    auto xupdates = mm->add_parameter("U", us);

    auto scatternd_add_op = migraphx::make_op("scatternd_add");
    auto scatternd        = mm->add_instruction(scatternd_add_op, xdata, xindex, xupdates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4, 4, 4}}; // data
    std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6,
                                  7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4,
                                  5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> input_index{0, 2};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {2, 4, 4}}; // updates
    std::vector<float> input_updates{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                     1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

    params["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    params["I"] = migraphx::argument(is, input_index.data());
    params["U"] = migraphx::argument(input_fixed_shape1, input_updates.data());

    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{6, 7, 8, 9, 11, 12, 13, 14, 15, 14, 13, 12, 12, 11, 10, 9,
                            1, 2, 3, 4, 5,  6,  7,  8,  8,  7,  6,  5,  4,  3,  2,  1,
                            9, 8, 7, 6, 6,  5,  4,  3,  4,  5,  6,  7,  9,  10, 11, 12,
                            8, 7, 6, 5, 4,  3,  2,  1,  1,  2,  3,  4,  5,  6,  7,  8};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
