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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <optional>
#include <test.hpp>

namespace {

migraphx::program
create_program(const migraphx::shape& data_shape, int64_t sorted, std::optional<int64_t> axis)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("X", data_shape);
    auto op   = axis ? migraphx::make_op("unique", {{"axis", *axis}, {"sorted", sorted}})
                     : migraphx::make_op("unique", {{"sorted", sorted}});

    auto r = mm->add_instruction(op, data);

    auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
    auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
    auto r2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), r);
    auto r3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 3}}), r);
    mm->add_return({r0, r1, r2, r3});
    return p;
};

template <typename T>
auto run_program(T& data,
                 const migraphx::shape& data_shape,
                 int sorted,
                 std::optional<int64_t> axis = std::nullopt)
{
    auto p = create_program(data_shape, sorted, axis);
    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map pp;
    pp["X"]   = migraphx::argument(data_shape, data.data());
    auto rets = p.eval(pp);
    std::vector<typename std::remove_reference_t<decltype(data)>::value_type> y;
    rets[0].visit([&](auto v) { y.assign(v.begin(), v.end()); });
    std::vector<int64_t> y_idx;
    rets[1].visit([&](auto v) { y_idx.assign(v.begin(), v.end()); });
    std::vector<int64_t> x_rev_idx;
    rets[2].visit([&](auto v) { x_rev_idx.assign(v.begin(), v.end()); });
    std::vector<int64_t> y_ct;
    rets[3].visit([&](auto v) { y_ct.assign(v.begin(), v.end()); });

    return std::make_tuple(y, y_idx, x_rev_idx, y_ct);
}
} // namespace

// sorted. single entry
TEST_CASE(unique_sorted_single_entry_test)
{
    std::vector<int> data    = {2};
    int64_t axis             = 0;
    int64_t sorted           = 1;
    std::vector<size_t> lens = {1};
    migraphx::shape data_shape{migraphx::shape::int32_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<int> gold_val = {2};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {1};
    EXPECT(ct == gold_ct);
}

// unsorted. single entry
TEST_CASE(unique_unsorted_single_entry_test)
{
    std::vector<float> data  = {3.33};
    int64_t axis             = -1;
    int64_t sorted           = 0;
    std::vector<size_t> lens = {1};
    migraphx::shape data_shape{migraphx::shape::float_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<float> gold_val = {3.33};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {1};
    EXPECT(ct == gold_ct);
}

// case 2  sorted. all unique input..
TEST_CASE(unique_sorted_all_unique_test)
{
    std::vector<float> data  = {2.1, 2.3, 2.4, 2.5, 1.9};
    int64_t axis             = 0;
    int64_t sorted           = 1;
    std::vector<size_t> lens = {5};
    migraphx::shape data_shape{migraphx::shape::float_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<float> gold_val = {1.9, 2.1, 2.3, 2.4, 2.5};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {4, 0, 1, 2, 3};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {1, 2, 3, 4, 0};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {1, 1, 1, 1, 1};
    EXPECT(ct == gold_ct);
}

// case 3  unsorted. all unique input
TEST_CASE(unique_unsorted_all_unique_test)
{
    std::vector<float> data  = {2.1, 2.3, 2.4, 2.5, 1.9};
    int64_t axis             = 0;
    int64_t sorted           = 0;
    std::vector<size_t> lens = {5};
    migraphx::shape data_shape{migraphx::shape::float_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<float> gold_val = {2.1, 2.3, 2.4, 2.5, 1.9};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0, 1, 2, 3, 4};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0, 1, 2, 3, 4};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {1, 1, 1, 1, 1};
    EXPECT(ct == gold_ct);
}

// case 4  sorted (with dup entries)
TEST_CASE(unique_sorted_dupes_test)
{
    std::vector<double> data = {2.1, 2.3, 2.4, 2.5, 1.9, 2.5, 2.3, 2.5};
    int64_t axis             = 0;
    int64_t sorted           = 1;
    std::vector<size_t> lens = {8};
    migraphx::shape data_shape{migraphx::shape::double_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<double> gold_val = {1.9, 2.1, 2.3, 2.4, 2.5};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_ct = {1, 1, 2, 1, 3};
    EXPECT(ct == gold_ct);
}

// case 5  unsorted (with dup entries)
TEST_CASE(unique_unsorted_dupes_test)
{
    std::vector<float> data  = {2.1, 2.3, 2.4, 2.5, 1.9, 2.5, 2.3, 2.1};
    int64_t axis             = -1;
    int64_t sorted           = 0;
    std::vector<size_t> lens = {8};
    migraphx::shape data_shape{migraphx::shape::float_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<float> gold_val = {2.1, 2.3, 2.4, 2.5, 1.9};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0, 1, 2, 3, 4};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0, 1, 2, 3, 4, 3, 1, 0};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {2, 2, 1, 2, 1};
    EXPECT(ct == gold_ct);
}

TEST_CASE(unique_3_d_no_axis_test)
{
    // sorted 3D (with dup entries). no axis
    int sorted                  = 1;
    std::vector<double> data_3d = {2.1, 2.3, 2.4, 2.5, 1.9, 2.5, 2.3, 2.5};
    std::vector<size_t> lens    = {2, 2, 2}; // 3D data. type double
    migraphx::shape data_shape{migraphx::shape::double_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data_3d, data_shape, sorted);

    std::vector<double> gold_val = {1.9, 2.1, 2.3, 2.4, 2.5};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_ct = {1, 1, 2, 1, 3};
    EXPECT(ct == gold_ct);
}

TEST_CASE(unique_3_d_no_axis_unsorted_test)
//  unsorted 3D (with dup entries). no axis
{
    int sorted               = 0;
    std::vector<float> data  = {2.1, 2.3, 2.4, 2.5, 1.9, 2.5, 2.3, 2.1};
    std::vector<size_t> lens = {2, 1, 4}; // 3D data. type float
    migraphx::shape data_shape{migraphx::shape::float_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted);

    std::vector<float> gold_val = {2.1, 2.3, 2.4, 2.5, 1.9};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0, 1, 2, 3, 4};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0, 1, 2, 3, 4, 3, 1, 0};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {2, 2, 1, 2, 1};
    EXPECT(ct == gold_ct);
}

//  unique integer sub-tensors: sorted (with dup entries)
TEST_CASE(unique_subtensors_sorted_test)
{
    /*
      input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
      attribute_sorted = 1
      attribute_axis = 0
      output_Y = [[1, 0, 0], [2, 3, 4]]
      output_indices = [0, 2]
      output_inverse_indices = [0, 0, 1]
      output_counts = [2, 1]
    */

    int axis                  = 0;
    int sorted                = 1;
    std::vector<int32_t> data = {1, 0, 0, 1, 0, 0, 2, 3, 4};
    std::vector<size_t> lens  = {3, 3};
    migraphx::shape data_shape{migraphx::shape::int32_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<int32_t> gold_val = {1, 0, 0, 2, 3, 4};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0, 2};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0, 0, 1};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {2, 1};
    EXPECT(ct == gold_ct);
}

//  unique integer sub-tensors: un-sorted (with dup entries)
TEST_CASE(unique_subtensors_neg_axis_test)
{
    /*
      input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
      attribute_sorted = 0
      attribute_axis = 0
      output_Y = [[1, 0, 0], [2, 3, 4]]
      output_indices = [0, 2]
      output_inverse_indices = [0, 0, 1]
      output_counts = [2, 1]
    */

    int axis                  = -2; // == 0
    int sorted                = 0;
    std::vector<int32_t> data = {1, 0, 0, 1, 0, 0, 2, 3, 4};
    std::vector<size_t> lens  = {3, 3};
    migraphx::shape data_shape{migraphx::shape::int32_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<int32_t> gold_val = {1, 0, 0, 2, 3, 4};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0, 2};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0, 0, 1};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {2, 1};
    EXPECT(ct == gold_ct);
}

//  unique float sub-tensors: sorted (with dup entries)  axis = 0
TEST_CASE(unique_subtensors_zero_axis_test)
{
    /*
      input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
      [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
      attribute_sorted = 1
      attribute_axis = 0
    */

    int axis                 = 0;
    int sorted               = 1;
    std::vector<float> data  = {1., 1., 0., 1., 2., 1., 0., 1., 1., 1., 0., 1., 2., 1., 0., 1.};
    std::vector<size_t> lens = {2, 4, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, lens};
    const auto& [y, idx, x_rev, ct] = run_program(data, data_shape, sorted, axis);

    std::vector<float> gold_val = {1., 1., 0., 1., 2., 1., 0., 1.};
    EXPECT(y == gold_val);

    std::vector<int64_t> gold_y_idx = {0};
    EXPECT(idx == gold_y_idx);

    std::vector<int64_t> gold_x_rev = {0, 0};
    EXPECT(x_rev == gold_x_rev);

    std::vector<int64_t> gold_ct = {2};
    EXPECT(ct == gold_ct);
}
