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
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/op/common.hpp>
#include <sstream>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"

template <class... Ts>
static void expect_shape(const migraphx::shape& expected, const migraphx::operation& op, Ts... xs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape> shapes{xs...};
    std::vector<migraphx::instruction_ref> args(shapes.size());
    std::transform(
        shapes.begin(), shapes.end(), args.begin(), [&](auto&& s) { return mm->add_outline(s); });
    mm->add_instruction(op, args);
    if(p.get_output_shapes().back() != expected)
    {
        std::cout << "FAILED: Incorrect shape for " << op << ": ";
        std::cout << expected << " != " << p.get_output_shapes().back() << std::endl;
        for(auto&& s : shapes)
            std::cout << "    " << s << std::endl;
    }
}

template <class... Ts>
static void throws_shape(const migraphx::operation& op, Ts... xs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape> shapes{xs...};
    std::vector<migraphx::instruction_ref> args(shapes.size());
    std::transform(
        shapes.begin(), shapes.end(), args.begin(), [&](auto&& s) { return mm->add_outline(s); });
    bool thrown = test::throws([&] { mm->add_instruction(op, args); });
    if(not thrown)
    {
        std::cout << "FAILED: No error found for " << op.name() << ": ";
        for(auto&& s : shapes)
            std::cout << "    " << s << std::endl;
    }
}

template <class...>
struct always_false : std::false_type
{
};

template <class... Ts>
[[maybe_unused]] static void throws_shape(const migraphx::shape&, Ts...)
{
    static_assert(always_false<Ts...>{},
                  "An expected shape should not be passed to throws_shape function");
}

TEST_CASE(allocate_static)
{
    migraphx::shape out_shape{migraphx::shape::float_type, {2, 3, 4}};
    expect_shape(out_shape, migraphx::make_op("allocate", {{"shape", to_value(out_shape)}}));
}

TEST_CASE(allocate_static_input)
{
    migraphx::shape input{migraphx::shape::int64_type, {3}};
    migraphx::shape out_shape{migraphx::shape::float_type, {2, 3, 4}};
    expect_shape(out_shape, migraphx::make_op("allocate", {{"shape", to_value(out_shape)}}), input);
}

TEST_CASE(allocate_dyn)
{
    migraphx::shape input{migraphx::shape::int64_type, {2}};
    auto max_val = std::numeric_limits<std::size_t>::max();
    std::vector<migraphx::shape::dynamic_dimension> dyn_dims(
        2, migraphx::shape::dynamic_dimension{0, max_val});
    expect_shape(migraphx::shape{migraphx::shape::float_type, dyn_dims},
                 migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}),
                 input);
}

TEST_CASE(allocate_dyn_with_shape_attr)
{
    migraphx::shape input{migraphx::shape::int64_type, {4}};
    migraphx::shape shape_attr{migraphx::shape::float_type,
                               {{1, 4}, {3, 3}, {4, 8, {4, 6}}, {4, 8}, {4, 6}}};
    expect_shape(shape_attr,
                 migraphx::make_op("allocate", {{"shape", migraphx::to_value(shape_attr)}}),
                 input);
}

TEST_CASE(allocate_dyn_no_input)
{
    migraphx::shape shape_attr{migraphx::shape::float_type,
                               {{1, 4}, {3, 3}, {4, 8, {4, 6}}, {4, 8}, {4, 6}}};
    expect_shape(shape_attr,
                 migraphx::make_op("allocate", {{"shape", migraphx::to_value(shape_attr)}}));
}

TEST_CASE(allocate_shape_and_buf_type_error)
{
    migraphx::shape shape_attr{migraphx::shape::float_type,
                               {{1, 4}, {3, 3}, {4, 8, {4, 6}}, {4, 8}, {4, 6}}};
    throws_shape(migraphx::make_op(
        "allocate",
        {{"shape", migraphx::to_value(shape_attr)}, {"buf_type", migraphx::shape::half_type}}));
}

TEST_CASE(allocate_no_attr_error)
{
    migraphx::shape input{migraphx::shape::int64_type, {4}};
    throws_shape(migraphx::make_op("allocate"), input);
}

TEST_CASE(argmax_axis0)
{
    migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                 migraphx::make_op("argmax", {{"axis", 0}}),
                 input);
}

TEST_CASE(argmax_axis1)
{
    migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                 migraphx::make_op("argmax", {{"axis", 1}}),
                 input);
}

TEST_CASE(argmax_axis2)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                 migraphx::make_op("argmax", {{"axis", 2}}),
                 input);
}

TEST_CASE(argmax_axis_neg)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                 migraphx::make_op("argmax", {{"axis", -1}}),
                 input);
}

TEST_CASE(argmax_axis_outofbounds)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
    throws_shape(migraphx::make_op("argmax", {{"axis", 4}}), input);
}

TEST_CASE(argmax_dyn0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {{1, 4}, {1, 1}, {4, 4}, {5, 5}}},
                 migraphx::make_op("argmax", {{"axis", 1}}),
                 input);
}

TEST_CASE(argmax_dyn1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 6}, {4, 6}}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {{1, 4}, {3, 3}, {1, 1}, {4, 6}}},
                 migraphx::make_op("argmax", {{"axis", 2}}),
                 input);
}

TEST_CASE(binary_dyn_static_error)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 4, 4}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 1}, {4, 4, {4}}, {4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("add"), a_shape, b_shape);
}

TEST_CASE(bit_cast_typesize_mismatch)
{
    migraphx::shape a_shape{migraphx::shape::int8_type, {1, 4, 4}};
    throws_shape(migraphx::make_op("bit_cast", {{"target_type", migraphx::shape::int32_type}}),
                 a_shape);
}

TEST_CASE(bit_cast_dyn)
{
    migraphx::shape a_shape{migraphx::shape::int8_type, {{1, 1}, {4, 8}, {4, 8}}};
    expect_shape(migraphx::shape{migraphx::shape::uint8_type, {{1, 1}, {4, 8}, {4, 8}}},
                 migraphx::make_op("bit_cast", {{"target_type", migraphx::shape::uint8_type}}),
                 a_shape);
}

TEST_CASE(bitwise_and_not_integral_error)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {1, 4, 4}};
    throws_shape(migraphx::make_op("bitwise_and"), a_shape, b_shape);
}

TEST_CASE(broadcast)
{
    {
        std::vector<std::size_t> lens{1, 1};
        migraphx::shape input{migraphx::shape::float_type, {1}, {0}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1}, {0, 0}},
                     migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", lens}}),
                     input);
    }

    {
        std::vector<std::size_t> lens{1, 1};
        migraphx::shape input{migraphx::shape::float_type, {2}};
        throws_shape(migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", lens}}), input);
    }

    {
        std::vector<std::size_t> lens{2, 2};
        migraphx::shape input{migraphx::shape::float_type, {1, 2}};
        throws_shape(migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", lens}}), input);
    }

    {
        std::vector<std::size_t> lens{3, 2, 4, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 2, 4, 3}, {0, 0, 3, 1}},
                     migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", lens}}),
                     input);
    }

    {
        std::vector<std::size_t> lens{3, 2, 4, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 4}};
        throws_shape(migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", lens}}), input);
    }
}

TEST_CASE(broadcast_axis_out_of_range_error)
{
    std::vector<std::size_t> lens{1, 1};
    migraphx::shape input{migraphx::shape::float_type, {1}, {0}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 4}, {"out_lens", lens}}), input);
}

TEST_CASE(broadcast_1in_dyn_error)
{
    // broadcast doesn't support single dynamic shape input
    std::vector<std::size_t> lens{3, 2, 4, 3};
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {2, 2}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", lens}}), input);
}

TEST_CASE(broadcast_2in_static_static)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4}, {1}};
    migraphx::shape b_input{migraphx::shape::float_type, {4, 4}, {4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4}, {1, 0}},
                 migraphx::make_op("broadcast", {{"axis", 0}}),
                 a_input,
                 b_input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4}, {0, 1}},
                 migraphx::make_op("broadcast", {{"axis", 1}}),
                 a_input,
                 b_input);
    throws_shape(migraphx::make_op("broadcast", {{"axis", 2}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_not_matching_error)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4}, {1}};
    migraphx::shape b_input{migraphx::shape::float_type, {2, 2}, {2, 1}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 1}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_dynamic_s0_error1)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4, 2}, {2, 1}};
    migraphx::shape b_input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {2, 2}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), b_input, a_input);
}

TEST_CASE(broadcast_2in_dynamic_s0_error2)
{
    std::vector<migraphx::shape::dynamic_dimension> dd{{4, 4}};
    migraphx::shape a_input{migraphx::shape::float_type, dd};
    migraphx::shape b_input{migraphx::shape::float_type, {4, 4}, {4, 1}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_static_dyn)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4}, {1}};
    migraphx::shape b_input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {2, 2}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), a_input, b_input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {2, 2}}},
                 migraphx::make_op("broadcast", {{"axis", 1}}),
                 a_input,
                 b_input);
    throws_shape(migraphx::make_op("broadcast", {{"axis", 2}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_dyn_s0_ndim_greater_than_1_error)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4, 2}};
    migraphx::shape b_input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {2, 2}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), a_input, b_input);
}

TEST_CASE(conv_2d_0)
{
    migraphx::shape output{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("convolution"), input, weights);
    throws_shape(migraphx::make_op("convolution"), input);
    throws_shape(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input);
}

TEST_CASE(conv_2d_1)
{
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape input2{migraphx::shape::float_type, {3, 3}};
    migraphx::shape weights2{migraphx::shape::float_type, {3, 3}};
    throws_shape(migraphx::make_op("convolution"), input2, weights2);
    throws_shape(migraphx::make_op("convolution"), input2, weights);
}

TEST_CASE(conv_1d)
{
    migraphx::shape output_1d{migraphx::shape::float_type, {4, 4, 1}};
    migraphx::shape input_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape weights_1d{migraphx::shape::float_type, {4, 3, 3}};
    expect_shape(
        output_1d,
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input_1d,
        weights_1d);
}

TEST_CASE(conv_channel_mismatch)
{
    migraphx::shape input_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape weights_1d = {migraphx::shape::float_type, {4, 8, 3}};
    throws_shape(migraphx::make_op("convolution"), input_1d, weights_1d);
}

TEST_CASE(conv_3_d)
{
    migraphx::shape output_3d{migraphx::shape::float_type, {4, 4, 1, 1, 1}};
    migraphx::shape input_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    migraphx::shape weights_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    expect_shape(
        output_3d,
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input_3d,
        weights_3d);

    throws_shape(migraphx::make_op("convolution"), input_3d, weights_3d);
}

TEST_CASE(conv_dyn_batch)
{
    migraphx::shape input_dyn_shape{migraphx::shape::float_type,
                                    {{1, 100}, {3, 3}, {5, 5}, {5, 5}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 3, 3, 3}};
    migraphx::shape output_dyn_shape{migraphx::shape::float_type,
                                     {{1, 100}, {1, 1}, {3, 3}, {3, 3}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_dyn_img)
{
    migraphx::shape input_dyn_shape  = {migraphx::shape::float_type,
                                        {{1, 1}, {3, 3}, {5, 20}, {5, 20}}};
    migraphx::shape weights_shape    = {migraphx::shape::float_type, {1, 3, 3, 3}};
    migraphx::shape output_dyn_shape = {migraphx::shape::float_type,
                                        {{1, 1}, {1, 1}, {3, 18}, {3, 18}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_dyn_weights)
{
    migraphx::shape input_dyn_shape = {migraphx::shape::float_type, {1, 3, 10, 10}};
    migraphx::shape weights_shape = {migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 4}, {2, 4}}};
    migraphx::shape output_dyn_shape = {migraphx::shape::float_type,
                                        {{1, 1}, {1, 1}, {7, 9}, {7, 9}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_dyn_img_weights)
{
    migraphx::shape input_dyn_shape = {migraphx::shape::float_type,
                                       {{1, 1}, {3, 3}, {5, 20}, {5, 20}}};
    migraphx::shape weights_shape = {migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 4}, {2, 4}}};
    migraphx::shape output_dyn_shape = {migraphx::shape::float_type,
                                        {{1, 1}, {1, 1}, {2, 19}, {2, 19}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_attr_shape_mismatch)
{
    migraphx::shape input_dyn_shape = {migraphx::shape::float_type,
                                       {{1, 100}, {3, 3}, {5, 5}, {5, 5}, {5, 5}}};
    migraphx::shape weights_shape   = {migraphx::shape::float_type, {1, 3, 3, 3, 3}};
    throws_shape(migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_autopad_dyn_batch)
{
    // auto_pad dynamic batch
    migraphx::shape input_dyn_shape  = {migraphx::shape::float_type,
                                        {{1, 10}, {3, 3}, {5, 5}, {5, 5}}};
    migraphx::shape weights_shape    = {migraphx::shape::float_type, {1, 3, 3, 3}};
    migraphx::shape output_dyn_shape = {migraphx::shape::float_type,
                                        {{1, 10}, {1, 1}, {5, 5}, {5, 5}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"stride", {1, 1}},
                                    {"dilation", {1, 1}},
                                    {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_autopad_dyn_img)
{
    // auto_pad dynamic img
    migraphx::shape input_dyn_shape  = {migraphx::shape::float_type,
                                        {{1, 1}, {3, 3}, {5, 10}, {5, 10}}};
    migraphx::shape weights_shape    = {migraphx::shape::float_type, {1, 3, 3, 3}};
    migraphx::shape output_dyn_shape = {migraphx::shape::float_type,
                                        {{1, 1}, {1, 1}, {5, 10}, {5, 10}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"stride", {1, 1}},
                                    {"dilation", {1, 1}},
                                    {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(conv_autopad_dyn_kernel)
{
    migraphx::shape input_dyn_shape = {migraphx::shape::float_type,
                                       {{1, 1}, {3, 3}, {10, 10}, {10, 10}}};
    migraphx::shape weights_shape = {migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 4}, {2, 4}}};
    migraphx::shape output_dyn_shape = {migraphx::shape::float_type,
                                        {{1, 1}, {1, 1}, {10, 10}, {10, 10}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"stride", {1, 1}},
                                    {"dilation", {1, 1}},
                                    {"padding_mode", migraphx::op::padding_mode_t::same_lower}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(contiguous_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {2, 2}};
    migraphx::shape input{migraphx::shape::float_type, {2, 2}, {1, 2}};
    expect_shape(output, migraphx::make_op("contiguous"), input);
    throws_shape(migraphx::make_op("contiguous"), input, input);

    migraphx::shape single{migraphx::shape::float_type, {2}};
    expect_shape(single, migraphx::make_op("contiguous"), single);
}

TEST_CASE(contiguous_dyn_shape)
{
    migraphx::shape s0{migraphx::shape::float_type, {{1, 4}, {2, 2, {2}}}};
    expect_shape(s0, migraphx::make_op("contiguous"), s0);
}

TEST_CASE(contiguous_shape_scalar)
{
    migraphx::shape output{migraphx::shape::float_type, {1}};
    migraphx::shape input{migraphx::shape::float_type};
    expect_shape(output, migraphx::make_op("contiguous"), input);
}

TEST_CASE(contiguous_shape_singleton_dim)
{
    migraphx::shape output{migraphx::shape::float_type, {5, 1, 8}, {8, 8, 1}};
    migraphx::shape input{migraphx::shape::float_type, {5, 1, 8}, {8, 4, 1}};
    expect_shape(output, migraphx::make_op("contiguous"), input);
}

TEST_CASE(convolution_backwards_1d)
{
    migraphx::shape input_1d{migraphx::shape::float_type, {4, 4, 1}};
    migraphx::shape weights_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape output_1d{migraphx::shape::float_type, {4, 3, 3}};
    expect_shape(output_1d,
                 migraphx::make_op("convolution_backwards",
                                   {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                 input_1d,
                 weights_1d);
}

TEST_CASE(convolution_backwards_2d)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("convolution_backwards"), input, weights);
    throws_shape(migraphx::make_op("convolution_backwards"), input);
    throws_shape(migraphx::make_op("convolution_backwards",
                                   {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                 input);
}

TEST_CASE(convolution_backwards_1padding)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 1, 1}};
    expect_shape(output,
                 migraphx::make_op("convolution_backwards",
                                   {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input,
                 weights);
}

TEST_CASE(convolution_backwards_2stride)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 4, 4}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 9, 9}};
    expect_shape(output,
                 migraphx::make_op("convolution_backwards",
                                   {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
                 input,
                 weights);
}

TEST_CASE(convolution_backwards_2dilation)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 4, 4}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 8, 8}};
    expect_shape(output,
                 migraphx::make_op("convolution_backwards",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {2, 2}}}),
                 input,
                 weights);
}

TEST_CASE(convolution_backwards_2group)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 4, 4}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 2, 4, 4}};
    migraphx::shape output{migraphx::shape::float_type, {4, 4, 7, 7}};
    expect_shape(output,
                 migraphx::make_op(
                     "convolution_backwards",
                     {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}, {"group", 2}}),
                 input,
                 weights);
}

TEST_CASE(convolution_backwards_3d)
{
    migraphx::shape input_3d{migraphx::shape::float_type, {4, 4, 1, 1, 1}};
    migraphx::shape output_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    migraphx::shape weights_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    expect_shape(
        output_3d,
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input_3d,
        weights_3d);
}

TEST_CASE(convolution_backwards_channel_mismatch)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape weights{migraphx::shape::float_type, {3, 3, 3, 3}};
    throws_shape(migraphx::make_op("convolution_backwards"), input, weights);
}

TEST_CASE(convolution_backwards_dyn_batch_2d)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {1, 1}, {1, 1}}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4}, {3, 3}, {3, 3}, {3, 3}}};
    expect_shape(output, migraphx::make_op("convolution_backwards"), input, weights);
}

TEST_CASE(convolution_backwards_dyn_batch_2dgroup)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {4, 4}, {4, 4}}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 2, 4, 4}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4}, {4, 4}, {7, 7}, {7, 7}}};
    expect_shape(
        output, migraphx::make_op("convolution_backwards", {{"group", 2}}), input, weights);
}

TEST_CASE(convolution_backwards_dyn_img_2d)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 1}, {4, 4}, {1, 5}, {1, 5}}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 1}, {3, 3}, {3, 7}, {3, 7}}};
    expect_shape(output, migraphx::make_op("convolution_backwards"), input, weights);
}

TEST_CASE(convolution_backwards_dyn_kernel_2d)
{
    migraphx::shape input{migraphx::shape::float_type, {1, 4, 1, 1}};
    migraphx::shape weights{migraphx::shape::float_type, {{4, 4}, {3, 3}, {2, 6}, {2, 6}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6}, {2, 6}}};
    expect_shape(output, migraphx::make_op("convolution_backwards"), input, weights);
}

TEST_CASE(dimensions_of0)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 2, 1}};
    migraphx::shape output{migraphx::shape::int64_type, {4}};
    expect_shape(output, migraphx::make_op("dimensions_of", {{"end", 4}}), input);
}

TEST_CASE(dimensions_of1)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 2, 1}};
    migraphx::shape output{migraphx::shape::int64_type, {2}};
    expect_shape(output, migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), input);
}

TEST_CASE(dimensions_of2)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4, {2}}, {2, 4}, {2, 4}, {1, 6, {2}}}};
    migraphx::shape output{migraphx::shape::int64_type, {2}};
    expect_shape(output, migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), input);
}

TEST_CASE(dimensions_of_error0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4, {2}}, {2, 4}}};
    throws_shape(migraphx::make_op("dimensions_of", {{"start", 3}, {"end", 3}}), input);
}

TEST_CASE(dimensions_of_error1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4, {2}}, {2, 4}}};
    throws_shape(migraphx::make_op("dimensions_of", {{"start", 3}, {"end", 0}}), input);
}

TEST_CASE(dot_ndim_error0)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_ndim_error1)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5, 2}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_ndim_error2)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_ndim_error3)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {6, 5, 4}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_ndim_error4)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 1, 5, 7}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_mismatch_inner_error0)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {10, 8}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_mismatch_inner_error1)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {4, 6}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_mismatch_inner_error2)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {4, 4}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_mismatch_inner_error3)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 1, 4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 2, 5, 7}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_mismatch_outer_error)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 6}};
    migraphx::shape s_m2{migraphx::shape::float_type, {2, 5, 8}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_2_d_test0)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {4, 8}}, migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_2_d_test1)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5, 4}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {1, 4}}, migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_2_d_test2)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {4, 8}}, migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_2_d_test3)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 1}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 1}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {1, 1}}, migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_3_d_test0)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 8}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_3_d_test_1)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {6, 1, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {6, 5, 4}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {6, 1, 4}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_3_d_test2)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 7}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 7}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_4_d_test)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {1, 6, 1, 5}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 6, 5, 4}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 6, 1, 4}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_static_test0)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 4}, {5, 5}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {8, 8}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_static_test1)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{3, 3}, {5, 5}, {5, 5}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {3, 5, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {5, 5}, {8, 8}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_static_test2)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 4}, {3, 3}, {5, 5}, {5, 5}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {2, 3, 5, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{2, 2}, {3, 3}, {5, 5}, {8, 8}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_test0)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 4}, {5, 5}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {{5, 5}, {6, 8, {8}}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {6, 8, {8}}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_test1)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 4}, {4, 5, {5}}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {{4, 5, {5}}, {6, 8, {8}}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {6, 8, {8}}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_test2)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 20}, {5, 5}, {5, 5}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 1}, {5, 5}, {8, 8}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_test3)
{
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape s_m1{migraphx::shape::float_type, {{4, 4}, {5, 5}, {0, max_val}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {4, 5, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{4, 4}, {5, 5}, {8, 8}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_test4)
{
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape s_m1{migraphx::shape::float_type, {{0, max_val}, {5, 5}, {0, max_val}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {{4, 8}, {5, 5}, {8, 8}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{4, 8}, {5, 5}, {8, 8}}},
                 migraphx::make_op("dot"),
                 s_m1,
                 s_m2);
}

TEST_CASE(dot_dyn_inner_mismatch)
{
    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 4}, {5, 5}, {4, 8}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {{1, 4}, {10, 20}, {8, 8}}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(dot_dyn_test_outer_mismatch)
{

    migraphx::shape s_m1{migraphx::shape::float_type, {{1, 4}, {1, 4}, {5, 5}}};
    migraphx::shape s_m2{migraphx::shape::float_type, {{5, 8}, {5, 5}, {6, 8, {8}}}};
    throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
}

TEST_CASE(broadcast_for_dot_static)
{
    migraphx::shape s0{migraphx::shape::float_type, {481, 356}};
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 356, 254}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 481, 356}, {0, 0, 356, 1}},
                 migraphx::make_op("broadcast_for_dot"),
                 s0,
                 s1);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 356, 254}},
                 migraphx::make_op("broadcast_for_dot"),
                 s1,
                 s0);
}

TEST_CASE(broadcast_for_dot_dyn0)
{
    migraphx::shape s0{migraphx::shape::float_type, {{124, 282}, {254, 484}}};
    migraphx::shape s1{migraphx::shape::float_type,
                       {{1, 4, {1, 2, 4}}, {4, 4}, {254, 484}, {356, 584}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{1, 4, {1, 2, 4}}, {4, 4}, {124, 282}, {254, 484}}},
                 migraphx::make_op("broadcast_for_dot"),
                 s0,
                 s1);
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{1, 4, {1, 2, 4}}, {4, 4}, {254, 484}, {356, 584}}},
                 migraphx::make_op("broadcast_for_dot"),
                 s1,
                 s0);
}

TEST_CASE(broadcast_for_dot_dyn1)
{
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape s0{migraphx::shape::float_type, {{124, 282}, {0, max_val}}};
    migraphx::shape s1{migraphx::shape::float_type,
                       {{1, 4, {1, 2, 4}}, {4, 4}, {254, 484}, {356, 584}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{1, 4, {1, 2, 4}}, {4, 4}, {124, 282}, {0, max_val}}},
                 migraphx::make_op("broadcast_for_dot"),
                 s0,
                 s1);
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{1, 4, {1, 2, 4}}, {4, 4}, {254, 484}, {356, 584}}},
                 migraphx::make_op("broadcast_for_dot"),
                 s1,
                 s0);
}

TEST_CASE(broadcast_for_dot_dyn2)
{
    migraphx::shape s0{migraphx::shape::float_type, {{6, 12}, {4, 4}, {8, 8}}};
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4, {1, 2, 4}}, {2, 10}, {8, 8}, {4, 4}}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {{1, 4, {1, 2, 4}}, {6, 10}, {4, 4}, {8, 8}}},
        migraphx::make_op("broadcast_for_dot"),
        s0,
        s1);
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {{1, 4, {1, 2, 4}}, {6, 10}, {8, 8}, {4, 4}}},
        migraphx::make_op("broadcast_for_dot"),
        s1,
        s0);
}

TEST_CASE(broadcast_with_dims0)
{
    using migraphx::shape;
    shape s0{migraphx::shape::float_type, {2, 4}};
    shape s1{migraphx::shape::int64_type, {4}};
    std::size_t max_int = std::numeric_limits<std::size_t>::max();
    std::vector<shape::dynamic_dimension> dyn_dims(4, shape::dynamic_dimension{0, max_int});
    expect_shape(
        shape{shape::float_type, dyn_dims}, migraphx::make_op("broadcast_with_dims"), s0, s1);
}

TEST_CASE(broadcast_with_dims1)
{
    using migraphx::shape;
    shape s0{migraphx::shape::int32_type, {1, 2, 4}};
    shape s1{migraphx::shape::int64_type, {1}};
    std::size_t max_int = std::numeric_limits<std::size_t>::max();
    std::vector<shape::dynamic_dimension> dyn_dims(3, shape::dynamic_dimension{0, max_int});
    expect_shape(shape{migraphx::shape::int32_type, dyn_dims},
                 migraphx::make_op("broadcast_with_dims"),
                 s0,
                 s1);
}

TEST_CASE(broadcast_with_dims2)
{
    using migraphx::shape;
    shape s0{migraphx::shape::float_type, {{1, 4}, {2, 2}, {4, 4}}};
    shape s1{migraphx::shape::int64_type, {4}};
    std::size_t max_int = std::numeric_limits<std::size_t>::max();
    std::vector<shape::dynamic_dimension> dyn_dims(4, shape::dynamic_dimension{0, max_int});
    expect_shape(shape{migraphx::shape::float_type, dyn_dims},
                 migraphx::make_op("broadcast_with_dims"),
                 s0,
                 s1);
}

TEST_CASE(flatten_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 4, 6, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 2 * 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 0}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 2 * 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", -4}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 1}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", -3}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4, 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 2}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4 * 6, 8}},
                 migraphx::make_op("flatten", {{"axis", 3}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4 * 6 * 8, 1}},
                 migraphx::make_op("flatten", {{"axis", 4}}),
                 input);
    throws_shape(migraphx::make_op("flatten", {{"axis", 5}}), input);
    throws_shape(migraphx::make_op("flatten", {{"axis", -5}}), input);
}

TEST_CASE(flatten_dyn_axis0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {6, 6}, {8, 8}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 1}, {192, 768}}},
                 migraphx::make_op("flatten", {{"axis", 0}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 1}, {192, 768}}},
                 migraphx::make_op("flatten", {{"axis", -4}}),
                 input);
}

TEST_CASE(flatten_dyn_axis1)
{
    migraphx::shape input{migraphx::shape::float_type,
                          {{2, 2, {2}}, {4, 4}, {4, 6, {5}}, {4, 6, {5}}}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {{2, 2, {2}}, {4 * 4 * 4, 4 * 6 * 6}}},
        migraphx::make_op("flatten", {{"axis", 1}}),
        input);
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {{2, 2, {2}}, {4 * 4 * 4, 4 * 6 * 6}}},
        migraphx::make_op("flatten", {{"axis", -3}}),
        input);
}

TEST_CASE(flatten_dyn_axis2)
{
    migraphx::shape input{migraphx::shape::float_type,
                          {{2, 2, {2}}, {4, 4}, {4, 6, {5}}, {4, 6, {5}}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{2 * 4, 2 * 4}, {4 * 4, 6 * 6}}},
                 migraphx::make_op("flatten", {{"axis", 2}}),
                 input);
}

TEST_CASE(flatten_dyn_axis3)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {6, 6}, {8, 8}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1 * 4 * 6, 4 * 4 * 6}, {8, 8}}},
                 migraphx::make_op("flatten", {{"axis", 3}}),
                 input);
}

TEST_CASE(flatten_dyn_axis4)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {6, 6}, {8, 8}}};
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {{1 * 4 * 6 * 8, 4 * 4 * 6 * 8}, {1, 1}}},
        migraphx::make_op("flatten", {{"axis", 4}}),
        input);
}

TEST_CASE(fill_static_int)
{
    migraphx::shape default_value{migraphx::shape::int64_type, {1}, {0}};
    migraphx::shape data{migraphx::shape::int64_type, {3, 4, 4}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type, {3, 4, 4}},
                 migraphx::make_op("fill"),
                 default_value,
                 data);
}

TEST_CASE(fill_static_float)
{
    migraphx::shape default_value{migraphx::shape::float_type, {1}, {0}};
    migraphx::shape data{migraphx::shape::float_type, {4, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}},
                 migraphx::make_op("fill"),
                 default_value,
                 data);
}

TEST_CASE(fill_dyn_int)
{
    migraphx::shape default_value{migraphx::shape::int64_type, {1}, {0}};
    migraphx::shape data{migraphx::shape::int64_type,
                         {{1, 4}, {4, 8, {4, 6, 8}}, {4, 8, {4, 6, 8}}}};
    expect_shape(migraphx::shape{migraphx::shape::int64_type,
                                 {{1, 4}, {4, 8, {4, 6, 8}}, {4, 8, {4, 6, 8}}}},
                 migraphx::make_op("fill"),
                 default_value,
                 data);
}

TEST_CASE(fill_dyn_float)
{
    migraphx::shape default_value{migraphx::shape::float_type, {1}, {0}};
    migraphx::shape data{migraphx::shape::float_type,
                         {{1, 4}, {4, 8, {4, 6, 8}}, {4, 8, {4, 6, 8}}}};
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{1, 4}, {4, 8, {4, 6, 8}}, {4, 8, {4, 6, 8}}}},
                 migraphx::make_op("fill"),
                 default_value,
                 data);
}

TEST_CASE(gather)
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = 1;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {1}};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = 3;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {3}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = 0;
        expect_shape(migraphx::shape{migraphx::shape::float_type},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {3}};
        migraphx::shape indices{migraphx::shape::int32_type, {1}};
        int axis = 0;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = 4;
        throws_shape(migraphx::make_op("gather", {{"axis", axis}}), input, indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = -5;
        throws_shape(migraphx::make_op("gather", {{"axis", axis}}), input, indices);
    }
}

TEST_CASE(gather_dyn0)
{
    // Insert dynamic index into dynamic shape
    migraphx::shape input{migraphx::shape::float_type,
                          {{2, 3, {2}}, {3, 4, {3}}, {6, 9, {7}}, {12, 14, {13}}}};
    migraphx::shape indices{migraphx::shape::int32_type, {{2, 7, {3}}, {3, 3}}};
    int axis = 1;
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{2, 3, {2}}, {2, 7, {3}}, {3, 3}, {6, 9, {7}}, {12, 14, {13}}}},
                 migraphx::make_op("gather", {{"axis", axis}}),
                 input,
                 indices);
}

TEST_CASE(gather_dyn1)
{
    // Insert static index into dynamic shape
    migraphx::shape input{migraphx::shape::float_type,
                          {{2, 3, {2}}, {3, 4, {3}}, {6, 9, {7}}, {12, 14, {13}}}};
    migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
    int axis = 1;
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{2, 3, {2}}, {2, 2}, {3, 3}, {6, 9, {7}}, {12, 14, {13}}}},
                 migraphx::make_op("gather", {{"axis", axis}}),
                 input,
                 indices);
}

TEST_CASE(gather_dyn2)
{
    // Insert scalar (static) index into dynamic shape
    migraphx::shape input{migraphx::shape::float_type,
                          {{2, 3, {2}}, {3, 4, {3}}, {6, 9, {7}}, {12, 14, {13}}}};

    std::vector<std::size_t> mins;
    std::vector<std::size_t> maxes;
    std::vector<std::set<std::size_t>> opts;
    migraphx::shape indices{migraphx::shape::int32_type, mins, maxes, opts};
    int axis = 1;
    expect_shape(
        migraphx::shape{migraphx::shape::float_type, {{2, 3, {2}}, {6, 9, {7}}, {12, 14, {13}}}},
        migraphx::make_op("gather", {{"axis", axis}}),
        input,
        indices);
}

TEST_CASE(gather_dyn3)
{
    // Insert dynamic index into static shape, axis 1
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 6, 12}};
    migraphx::shape indices{migraphx::shape::int32_type, {{2, 3, {2}}, {3, 4, {3}}}};
    int axis = 1;
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{2, 2}, {2, 3, {2}}, {3, 4, {3}}, {6, 6}, {12, 12}}},
                 migraphx::make_op("gather", {{"axis", axis}}),
                 input,
                 indices);
}

TEST_CASE(gather_dyn4)
{
    // Insert dynamic index into static shape, axis 0
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 6, 12}};
    migraphx::shape indices{migraphx::shape::int32_type, {{2, 3, {2}}, {3, 4, {3}}}};
    int axis = 0;
    expect_shape(migraphx::shape{migraphx::shape::float_type,
                                 {{2, 3, {2}}, {3, 4, {3}}, {3, 3}, {6, 6}, {12, 12}}},
                 migraphx::make_op("gather", {{"axis", axis}}),
                 input,
                 indices);
}

TEST_CASE(get_tuple_elem_test)
{
    migraphx::shape s0{migraphx::shape::bool_type, {1, 1}};
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {5, 6}};
    migraphx::shape s_tuple({s0, s1, s2});

    expect_shape(s0, migraphx::make_op("get_tuple_elem", {{"index", 0}}), s_tuple);
    expect_shape(s1, migraphx::make_op("get_tuple_elem", {{"index", 1}}), s_tuple);
    expect_shape(s2, migraphx::make_op("get_tuple_elem", {{"index", 2}}), s_tuple);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 3}}), s_tuple);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 0}}), s0);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 1}}), s1);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 0}}), s2);
}

TEST_CASE(gru)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(inconsistent_attr_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("convolution",
                                   {{"padding", {1, 1}}, {"stride", {2}}, {"dilation", {3, 3, 3}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("convolution_backwards",
                                   {{"padding", {1, 1}}, {"stride", {2}}, {"dilation", {3, 3, 3}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {1}},
                                    {"stride", {0}},
                                    {"lengths", {1, 1}}}),
                 input);
}

static void test_softmax_variations(const std::string& name)
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}},
                     migraphx::make_op(name, {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}},
                     migraphx::make_op(name, {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}},
                     migraphx::make_op(name, {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}},
                     migraphx::make_op(name, {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        int axis = 4;
        throws_shape(migraphx::make_op(name, {{"axis", axis}}), input);
    }
}
TEST_CASE(logsoftmax) { test_softmax_variations("logsoftmax"); }

TEST_CASE(softmax) { test_softmax_variations("softmax"); }

TEST_CASE(lstm)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(multibroadcast)
{
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {2, 1, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 3, 0, 1}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {2, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 1, 0, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {5, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 0, 1, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {1, 0, 0, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 0, 0, 1}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 3, 3, 1}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {1, 1, 1, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 3};
        std::vector<std::size_t> empt = {};
        migraphx::shape input{migraphx::shape::float_type, empt};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{2, 3, 4, 5};
        migraphx::shape input{migraphx::shape::float_type, {3, 4}};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{2, 3, 4, 5};
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4}};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
}

TEST_CASE(multibroadcast_1in_dyn_error_0)
{
    // multibroadcast doesn't support single dynamic shape input
    std::vector<std::size_t> lens{4, 4, 1, 3};
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {4, 4}}};
    throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
}

TEST_CASE(multibroadcast_2in_static_dyn0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 4}, {4, 4, {4}}, {4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {4, 4}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {4, 4}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{8, 8}, {6, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8}, {6, 6}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8}, {6, 6}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{8, 8}, {6, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8}, {6, 6}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(b)}}),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8}, {6, 6}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(b)}}),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_intersection0)
{
    // dynamic_dimension.intersection for first dimension
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 3}, {6, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {6, 6}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {6, 6}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_intersection1)
{
    std::vector<migraphx::shape::dynamic_dimension> a_dds{{5, 10}, {1, 6}};
    migraphx::shape a_shape{migraphx::shape::float_type, a_dds};
    std::vector<migraphx::shape::dynamic_dimension> b_dds{{3, 8}, {3, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, b_dds};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{5, 8}, {3, 6}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{5, 8}, {3, 6}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_intersection2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    auto max_val = std::numeric_limits<std::size_t>::max();
    std::vector<migraphx::shape::dynamic_dimension> b{{0, max_val}, {6, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {6, 6}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {6, 6}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_intersection_error)
{
    // not compatible for first dimension
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 2}, {6, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn0)
{
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4}, {2, 4, {2}}, {2, 4}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, {2}}, {2, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4, {2}}, {2, 4}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4, {2}}, {2, 4}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn1)
{
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4}, {2, 4, {2}}, {2, 4}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, {2}}, {2, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4, {2}}, {2, 4}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(a)}}),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4, {2}}, {2, 4}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(a)}}),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn_within0)
{
    // dynamic_dimension.within_range on second dimension of a
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4}, {2, 4, {2}}, {2, 4}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 5, {2}}, {2, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4}, {2, 4}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4}, {2, 4}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn_within1)
{
    // dynamic_dimension.within_range on second dimension of a, different opt dim
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4}, {2, 4, {2}}, {2, 4}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, {3}}, {2, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4}, {2, 4}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 4}, {2, 4}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, {3, 6}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 6}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 6}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 8}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}, {0, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}, {8, 1}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4, 8}, {0, 0, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4, 8}, {4, 1, 0}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static3)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {16, 4, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {0, 1, 0}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static4)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 1, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {4, 0, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {0, 1, 0}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static_error0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_3in_static)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, {1, 2, 3, 6}};
    migraphx::shape c_shape{migraphx::shape::float_type, {5, 1, 1, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {0, 0, 6, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape,
                 c_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {0, 18, 6, 1}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape,
                 c_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {1, 0, 0, 0}},
                 migraphx::make_op("multibroadcast"),
                 c_shape,
                 a_shape,
                 b_shape);
}

TEST_CASE(multibroadcast_4in_static)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 3, 6}};
    migraphx::shape c_shape{migraphx::shape::float_type, {5, 1, 1, 1}};
    migraphx::shape d_shape{migraphx::shape::float_type, {6}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {0, 0, 6, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape,
                 c_shape,
                 d_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {0, 18, 6, 1}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape,
                 c_shape,
                 d_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {1, 0, 0, 0}},
                 migraphx::make_op("multibroadcast"),
                 c_shape,
                 a_shape,
                 b_shape,
                 d_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {5, 2, 3, 6}, {0, 0, 0, 1}},
                 migraphx::make_op("multibroadcast"),
                 d_shape,
                 a_shape,
                 b_shape,
                 c_shape);
}

TEST_CASE(multibroadcast_3in_dyn_static)
{
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4}, {2, 4, {2}}, {2, 4}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, {2}}, {2, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    migraphx::shape c_shape{migraphx::shape::float_type, {5, 1, 1, 1}};
    migraphx::shape expected_shape{migraphx::shape::float_type,
                                   {{5, 5}, {1, 4}, {2, 4, {2}}, {2, 4}}};
    expect_shape(expected_shape, migraphx::make_op("multibroadcast"), a_shape, b_shape, c_shape);
    expect_shape(expected_shape, migraphx::make_op("multibroadcast"), b_shape, a_shape, c_shape);
    expect_shape(expected_shape, migraphx::make_op("multibroadcast"), c_shape, a_shape, b_shape);
}

TEST_CASE(multibroadcast_3in_dyn_dyn)
{
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4}, {2, 4, {2}}, {2, 4}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, {2}}, {2, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    std::vector<migraphx::shape::dynamic_dimension> c{{1, 5, {1, 5}}, {1, 1}, {2, 4, {2}}, {2, 4}};
    migraphx::shape c_shape{migraphx::shape::float_type, c};
    migraphx::shape expected_shape{migraphx::shape::float_type,
                                   {{1, 5, {1, 5}}, {1, 4}, {2, 4, {2}}, {2, 4}}};
    expect_shape(expected_shape, migraphx::make_op("multibroadcast"), a_shape, b_shape, c_shape);
    expect_shape(expected_shape, migraphx::make_op("multibroadcast"), b_shape, a_shape, c_shape);
    expect_shape(expected_shape, migraphx::make_op("multibroadcast"), c_shape, a_shape, b_shape);
}

TEST_CASE(multinomial_bool_type)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 2}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    int dtype = 0;

    throws_shape(migraphx::make_op("multinomial", {{"dtype", dtype}}), s1, s2);
}

TEST_CASE(multinomial)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 2}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::float_type, {1, 4}};
    int dtype = 2;

    expect_shape(s3, migraphx::make_op("multinomial", {{"dtype", dtype}}), s1, s2);
}

TEST_CASE(multinomial_0size_input)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 2}};
    migraphx::shape s2{migraphx::shape::float_type, {}};
    int dtype = 2;

    throws_shape(migraphx::make_op("multinomial", {{"dtype", dtype}}), s1, s2);
}

TEST_CASE(multinomial_dyn)
{
    migraphx::shape s1{migraphx::shape::int32_type, {{2, 3}, {5, 6}}};
    migraphx::shape s2{migraphx::shape::int32_type, {{7, 8}, {9, 10}}};
    migraphx::shape s3{migraphx::shape::int32_type, {{2, 3}, {9, 10}}};

    expect_shape(
        s3, migraphx::make_op("multinomial", {{"dtype", migraphx::shape::int32_type}}), s1, s2);
}

TEST_CASE(nms_shape)
{
    // use_dyn_output == false
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::shape max_out_s{migraphx::shape::int64_type, {1}};
    migraphx::shape iou_thres_s{migraphx::shape::float_type, {1}};
    migraphx::shape score_thres_s{migraphx::shape::float_type, {1}};
    migraphx::shape output_s{migraphx::shape::int64_type, {6, 3}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", false}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // use_dyn_output == true
    output_s = {migraphx::shape::int64_type, {{0, 6}, {3, 3}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic batches
    boxes_s  = {migraphx::shape::float_type, {{1, 3}, {6, 6}, {4, 4}}};
    scores_s = {migraphx::shape::float_type, {{1, 3}, {1, 1}, {6, 6}}};
    output_s = {migraphx::shape::int64_type, {{0, 18}, {3, 3}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic num boxes
    boxes_s  = {migraphx::shape::float_type, {{1, 1}, {6, 20}, {4, 4}}};
    scores_s = {migraphx::shape::float_type, {{1, 1}, {1, 1}, {6, 20}}};
    output_s = {migraphx::shape::int64_type, {{0, 20}, {3, 3}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // use_dyn_output false with dynamic input shape
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", false}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic classes
    boxes_s  = {migraphx::shape::float_type, {{1, 1}, {6, 6}, {4, 4}}};
    scores_s = {migraphx::shape::float_type, {{1, 1}, {1, 3}, {6, 6}}};
    output_s = {migraphx::shape::int64_type, {{0, 6}, {3, 3}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // fixed mismatch batches
    boxes_s  = {migraphx::shape::float_type, {2, 6, 4}};
    scores_s = {migraphx::shape::float_type, {1, 1, 6}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // fixed mismatch num boxes
    boxes_s  = {migraphx::shape::float_type, {1, 6, 4}};
    scores_s = {migraphx::shape::float_type, {1, 1, 4}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic mismatch batches
    boxes_s  = {migraphx::shape::float_type, {{1, 4}, {6, 6}, {4, 4}}};
    scores_s = {migraphx::shape::float_type, {{2, 8}, {1, 1}, {6, 6}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic mismatch num boxes
    boxes_s  = {migraphx::shape::float_type, {{1, 1}, {6, 8}, {4, 4}}};
    scores_s = {migraphx::shape::float_type, {{1, 1}, {1, 1}, {3, 9}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic number of classes, fixed boxes_s, mismatch batches
    boxes_s  = {migraphx::shape::float_type, {1, 6, 4}};
    scores_s = {migraphx::shape::float_type, {{1, 3}, {1, 3}, {6, 6}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);
    // dynamic number of classes, fixed boxes_s, mismatch num boxes
    boxes_s  = {migraphx::shape::float_type, {1, 6, 4}};
    scores_s = {migraphx::shape::float_type, {{1, 1}, {1, 3}, {4, 8}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);
}

TEST_CASE(onehot_static_2arg0)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 3, 4}};
    expect_shape(output, migraphx::make_op("onehot", {{"depth", 4}}), indices, values);
}

TEST_CASE(onehot_static_2arg1)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 6, 3}};
    expect_shape(output, migraphx::make_op("onehot", {{"axis", 1}, {"depth", 6}}), indices, values);
}

TEST_CASE(onehot_dyn_2arg0)
{
    migraphx::shape indices{migraphx::shape::int64_type, {{1, 4}, {2, 2}, {3, 3}}};
    migraphx::shape values{migraphx::shape::int32_type, {2}};
    migraphx::shape output{migraphx::shape::int32_type, {{1, 4}, {2, 2}, {8, 8}, {3, 3}}};
    expect_shape(output, migraphx::make_op("onehot", {{"axis", 2}, {"depth", 8}}), indices, values);
}

TEST_CASE(onehot_dyn_3arg0)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape output{migraphx::shape::float_type, {{2, 2}, {3, 3}, {0, max_val}}};
    expect_shape(output, migraphx::make_op("onehot"), indices, depth, values);
}

TEST_CASE(onehot_dyn_3arg1)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape output{migraphx::shape::float_type, {{2, 2}, {3, 3}, {0, max_val}}};
    expect_shape(output, migraphx::make_op("onehot", {{"axis", 2}}), indices, depth, values);
}

TEST_CASE(onehot_dyn_3arg2)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape output{migraphx::shape::float_type, {{2, 2}, {0, max_val}, {3, 3}}};
    expect_shape(output, migraphx::make_op("onehot", {{"axis", 1}}), indices, depth, values);
}

TEST_CASE(onehot_dyn_3arg3)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape output{migraphx::shape::float_type, {{0, max_val}, {2, 2}, {3, 3}}};
    expect_shape(output, migraphx::make_op("onehot", {{"axis", -3}}), indices, depth, values);
}

TEST_CASE(onehot_dyn_indices)
{
    migraphx::shape indices{migraphx::shape::int64_type, {{1, 4}, {2, 2}, {3, 3}}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::int32_type, {2}};
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape output{migraphx::shape::int32_type, {{1, 4}, {2, 2}, {0, max_val}, {3, 3}}};
    expect_shape(output, migraphx::make_op("onehot", {{"axis", 2}}), indices, depth, values);
}

TEST_CASE(onehot_axis_error0)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    throws_shape(migraphx::make_op("onehot", {{"axis", 3}}), indices, depth, values);
}

TEST_CASE(onehot_axis_error1)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    throws_shape(migraphx::make_op("onehot", {{"axis", -4}}), indices, depth, values);
}

TEST_CASE(onehot_axis_out_of_range0)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    throws_shape(migraphx::make_op("onehot", {{"axis", 3}}), indices, depth, values);
}

TEST_CASE(onehot_axis_out_of_range1)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape depth{migraphx::shape::int64_type, {1}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    throws_shape(migraphx::make_op("onehot", {{"axis", -4}}), indices, depth, values);
}

TEST_CASE(onehot_neg_depth_attr)
{
    migraphx::shape indices{migraphx::shape::int64_type, {2, 3}};
    migraphx::shape values{migraphx::shape::float_type, {2}};
    throws_shape(migraphx::make_op("onehot", {{"axis", 1}, {"depth", -3}}), indices, values);
}

TEST_CASE(pack_int4)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 4, 16, 8}};
    expect_shape(output, migraphx::make_op("pack_int4"), input);
}

TEST_CASE(pack_int4_axis1)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 2, 16, 16}};
    expect_shape(output, migraphx::make_op("pack_int4", {{"axis", 1}}), input);
}

TEST_CASE(pack_int4_axis2)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 2, 16, 16}};
    expect_shape(output, migraphx::make_op("pack_int4", {{"axis", -3}}), input);
}

TEST_CASE(pack_int4_invalid_axis)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    throws_shape(migraphx::make_op("pack_int4", {{"axis", 4}}), input);
}

TEST_CASE(pack_int4_nonstandard)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 16, 16, 4}, {1024, 16, 1, 256}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 8, 16, 4}};
    expect_shape(output, migraphx::make_op("pack_int4", {{"axis", 1}}), input);
}

TEST_CASE(pack_int4_invalid_dtype)
{
    migraphx::shape input{migraphx::shape::float_type, {1, 4, 16, 16}};
    throws_shape(migraphx::make_op("pack_int4", {{"axis", 0}}), input);
}

TEST_CASE(pack_int4_odd_lengths)
{
    migraphx::shape input{migraphx::shape::uint8_type, {3, 4, 16, 16}};
    throws_shape(migraphx::make_op("pack_int4", {{"axis", 0}}), input);
}

TEST_CASE(unpack_int4)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 4, 16, 8}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    expect_shape(output, migraphx::make_op("unpack_int4"), input);
}

TEST_CASE(unpack_int4_axis1)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 2, 16, 16}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    expect_shape(output, migraphx::make_op("unpack_int4", {{"axis", 1}}), input);
}

TEST_CASE(unpack_int4_axis2)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 2, 16, 16}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    expect_shape(output, migraphx::make_op("unpack_int4", {{"axis", -3}}), input);
}

TEST_CASE(unpack_int4_invalid_axis)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 4, 16, 16}};
    throws_shape(migraphx::make_op("unpack_int4", {{"axis", 4}}), input);
}

TEST_CASE(unpack_int4_nonstandard)
{
    migraphx::shape input{migraphx::shape::uint8_type, {1, 16, 16, 4}, {1024, 16, 1, 256}};
    migraphx::shape output{migraphx::shape::uint8_type, {1, 32, 16, 4}};
    expect_shape(output, migraphx::make_op("unpack_int4", {{"axis", 1}}), input);
}

TEST_CASE(unpack_int4_invalid_dtype)
{
    migraphx::shape input{migraphx::shape::float_type, {1, 4, 16, 16}};
    throws_shape(migraphx::make_op("unpack_int4", {{"axis", 0}}), input);
}

TEST_CASE(unpack_int4_odd_lengths)
{
    migraphx::shape input{migraphx::shape::uint8_type, {3, 4, 16, 16}};
    migraphx::shape output{migraphx::shape::uint8_type, {6, 4, 16, 16}};
    expect_shape(output, migraphx::make_op("unpack_int4", {{"axis", 0}}), input);
}

TEST_CASE(pad_shape0)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {2, 3, 5, 5}};
    expect_shape(output, migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), input);
}

TEST_CASE(pad_shape1)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {2, 3, 6, 6}};
    expect_shape(output, migraphx::make_op("pad", {{"pads", {0, 0, 2, 2, 0, 0, 1, 1}}}), input);
}

TEST_CASE(pad_dyn_shape0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4, {2}}, {3, 3}, {3, 5}, {3, 5}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4, {2}}, {3, 3}, {5, 7}, {5, 7}}};
    expect_shape(output, migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), input);
}

TEST_CASE(pad_dyn_shape1)
{
    migraphx::shape input{migraphx::shape::float_type,
                          {{1, 4, {2}}, {3, 3}, {3, 5, {5}}, {3, 5, {5}}}};
    migraphx::shape output{migraphx::shape::float_type,
                           {{1, 4, {2}}, {3, 3}, {5, 7, {7}}, {5, 7, {7}}}};
    expect_shape(output, migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), input);
}

TEST_CASE(pointwise_no_module)
{
    migraphx::shape input{migraphx::shape::float_type, {0}, {0}};
    throws_shape(migraphx::make_op("pointwise"), input);
}

TEST_CASE(pointwise_no_input)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::module m;
    std::vector<migraphx::instruction_ref> args{};
    auto output = migraphx::shape(migraphx::shape::float_type, {1}, {0});
    auto l      = m.add_literal(migraphx::literal(output, {1}));
    m.add_return({l});
    EXPECT(test::throws([&] { mm->add_instruction(migraphx::make_op("pointwise"), args, {&m}); }));
}

TEST_CASE(pointwise_no_output)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::module m;
    std::vector<migraphx::instruction_ref> args{};
    EXPECT(test::throws([&] { mm->add_instruction(migraphx::make_op("pointwise"), args, {&m}); }));
}

TEST_CASE(pointwise_strict_type)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::module pm;
    {
        auto x = pm.add_parameter("x", s.with_type(migraphx::shape::half_type));
        pm.add_return({x});
    }
    auto x = mm->add_parameter("x", s);
    EXPECT(test::throws([&] { mm->add_instruction(migraphx::make_op("pointwise"), {x}, {&pm}); }));
}

TEST_CASE(pooling_shape0)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {1}},
                                    {"stride", {0}},
                                    {"lengths", {1}},
                                    {"dilations", {1}}}),
                 input);
}

TEST_CASE(pooling_shape1)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 1, 1}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {1, 1}}}),
                 input);
}

TEST_CASE(pooling_shape2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 2, 2}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {1, 1}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(pooling_shape3)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {2, 2}},
                                    {"stride", {3, 3}},
                                    {"lengths", {3, 3}},
                                    {"dilations", {1, 1}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(pooling_shape4)
{
    migraphx::shape tiny_input{migraphx::shape::float_type, {4, 1}};
    throws_shape(migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}),
                 tiny_input);
}

TEST_CASE(pooling_shape5)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 1, 1}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {1, 1}},
                                    {"lengths", {2, 2}},
                                    {"dilations", {2, 2}}}),
                 input);
}

TEST_CASE(pooling_shape6)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 2, 2}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {2, 2}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {2, 2}}}),
                 input);
}

TEST_CASE(pooling_shape7)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 2, 2}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {3, 3}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(pooling_shape8)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {2, 2}},
                                    {"stride", {1, 1}},
                                    {"lengths", {3, 3}},
                                    {"dilations", {2, 2}}}),
                 input);
}

TEST_CASE(pooling_dyn_shape0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {3, 3, {3}}, {3, 3, {3}}, {3, 3}}};
    throws_shape(migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {1}},
                                    {"stride", {0}},
                                    {"lengths", {1}},
                                    {"dilations", {1}}}),
                 input);
}

TEST_CASE(pooling_dyn_shape1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {3, 3, {3}}, {3, 3, {3}}, {3, 3}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4}, {3, 3}, {1, 1}, {1, 1}}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {1, 1}}}),
                 input);
}

TEST_CASE(pooling_dyn_shape2)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {5, 5}, {3, 3, {3}}, {3, 3}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4}, {5, 5}, {2, 2}, {2, 2}}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {1, 1}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(pooling_dyn_shape3)
{
    migraphx::shape input{migraphx::shape::float_type,
                          {{4, 4}, {3, 3}, {4, 12, {8}}, {4, 12, {8}}}};
    migraphx::shape output{migraphx::shape::float_type, {{4, 4}, {3, 3}, {2, 4}, {2, 4}}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"dilations", {1, 1}}}),
                 input);
}

TEST_CASE(pooling_dyn_shape4)
{
    migraphx::shape input{migraphx::shape::float_type,
                          {{4, 4}, {3, 3}, {4, 12, {8}}, {4, 12, {8}}}};
    migraphx::shape output{migraphx::shape::float_type, {{4, 4}, {3, 3}, {3, 6}, {3, 6}}};
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {2, 2}},
                                    {"stride", {3, 3}},
                                    {"lengths", {3, 3}},
                                    {"dilations", {1, 1}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(prefix_scan_sum)
{
    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 3}};
        throws_shape(
            migraphx::make_op("prefix_scan_sum", {{"axis", 3}, {"exclusive", 0}, {"reverse", 0}}),
            s);
    }

    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2}};
        throws_shape(
            migraphx::make_op("prefix_scan_sum", {{"axis", -3}, {"exclusive", 0}, {"reverse", 0}}),
            s);
    }
}

TEST_CASE(prefix_scan_sum_dyn)
{
    {
        std::vector<migraphx::shape::dynamic_dimension> dd{{5, 8}};
        migraphx::shape s{migraphx::shape::float_type, dd};

        expect_shape(
            s,
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", 0}, {"reverse", 0}}),
            s);
    }
}

TEST_CASE(prefix_scan_sum_dyn_2d)
{
    {
        std::vector<migraphx::shape::dynamic_dimension> dd{{5, 8}, {3, 7}};
        migraphx::shape s{migraphx::shape::float_type, dd};

        expect_shape(
            s,
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", 0}, {"reverse", 0}}),
            s);
    }
}

TEST_CASE(random_uniform)
{
    std::vector<migraphx::shape::dynamic_dimension> dd{{5, 8}, {3, 7}};
    migraphx::shape s0{migraphx::shape::uint64_type, {1}};
    migraphx::shape s1{migraphx::shape::float_type, dd};
    expect_shape(s1, migraphx::make_op("random_uniform"), s0, s1);
}

TEST_CASE(random_seed)
{
    migraphx::shape s{migraphx::shape::uint64_type, {1}, {0}};
    expect_shape(s, migraphx::make_op("random_seed"));
}

TEST_CASE(quant_convolution_shape)
{
    migraphx::shape output{migraphx::shape::int32_type, {4, 4, 1, 1}};
    migraphx::shape input{migraphx::shape::int8_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::int8_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("quant_convolution"), input, weights);
    throws_shape(migraphx::make_op("quant_convolution"), input);
    throws_shape(migraphx::make_op("quant_convolution",
                                   {{"padding", {0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("quant_convolution",
                                   {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                 input,
                 weights);

    migraphx::shape input2{migraphx::shape::int32_type, {3, 3}};
    migraphx::shape weights2{migraphx::shape::float_type, {3, 3}};
    throws_shape(migraphx::make_op("quant_convolution"), input2, weights2);
    throws_shape(migraphx::make_op("quant_convolution"), input2, weights);

    migraphx::shape input3{migraphx::shape::int32_type, {4, 3, 3, 3}};
    migraphx::shape weight3{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("quant_convolution"), input3, weights);
    throws_shape(migraphx::make_op("quant_convolution"), input, weight3);
    throws_shape(migraphx::make_op("quant_convolution"), input3, weight3);
}

// quant_dot
TEST_CASE(quant_dot_2args)
{
    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {4, 8}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 8}},
                     migraphx::make_op("quant_dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {3, 8}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {8, 7}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {3, 7}},
                     migraphx::make_op("quant_dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {8, 8}};
        throws_shape(migraphx::make_op("quant_dot"), s_m1, s_m2);
    }
}

TEST_CASE(qlinear)
{
    migraphx::shape scales{migraphx::shape::float_type, {2, 4}};
    migraphx::shape input{migraphx::shape::float_type, {2, 4}};
    migraphx::shape result{migraphx::shape::uint8_type, {2, 4}};
    expect_shape(result, migraphx::make_op("quantizelinear"), input, scales);
}

TEST_CASE(qlinear_zeros)
{
    migraphx::shape zeros{migraphx::shape::int8_type, {2, 4}};
    migraphx::shape scales{migraphx::shape::float_type, {2, 4}};
    migraphx::shape input{migraphx::shape::float_type, {2, 4}};
    migraphx::shape result{migraphx::shape::int8_type, {2, 4}};
    expect_shape(result, migraphx::make_op("quantizelinear"), input, scales, zeros);
}

TEST_CASE(qlinear_fp16)
{
    migraphx::shape scales{migraphx::shape::half_type, {2, 4}};
    migraphx::shape input{migraphx::shape::half_type, {2, 4}};
    migraphx::shape result{migraphx::shape::uint8_type, {2, 4}};
    expect_shape(result, migraphx::make_op("quantizelinear"), input, scales);
}

TEST_CASE(qlinear_output_type_1)
{
    migraphx::shape scales{migraphx::shape::half_type, {2, 4}};
    migraphx::shape input{migraphx::shape::half_type, {2, 4}};
    migraphx::shape result{migraphx::shape::int8_type, {2, 4}};
    expect_shape(
        result, migraphx::make_op("quantizelinear", {{"out_type", result.type()}}), input, scales);
}

TEST_CASE(qlinear_output_type_2)
{
    migraphx::shape scales{migraphx::shape::half_type, {2, 4}};
    migraphx::shape input{migraphx::shape::half_type, {2, 4}};
    migraphx::shape result{migraphx::shape::int8_type, {2, 4}};
    auto op         = migraphx::make_op("quantizelinear");
    auto val        = op.to_value();
    val["out_type"] = migraphx::to_value(migraphx::shape::int8_type);
    expect_shape(result, migraphx::make_op("quantizelinear", val), input, scales);
}

TEST_CASE(qlinear_mismatch_type)
{
    migraphx::shape scales{migraphx::shape::int8_type, {2, 4}};
    migraphx::shape input{migraphx::shape::float_type, {2, 4}};
    throws_shape(migraphx::make_op("quantizelinear"), input, scales);
}

TEST_CASE(dqlinear)
{
    migraphx::shape scales{migraphx::shape::float_type, {2, 4}};
    migraphx::shape input{migraphx::shape::int8_type, {2, 4}};
    migraphx::shape result{migraphx::shape::float_type, {2, 4}};
    expect_shape(result, migraphx::make_op("dequantizelinear"), input, scales);
}

TEST_CASE(dqlinear_fp16)
{
    migraphx::shape scales{migraphx::shape::half_type, {2, 4}};
    migraphx::shape input{migraphx::shape::int8_type, {2, 4}};
    migraphx::shape result{migraphx::shape::half_type, {2, 4}};
    expect_shape(result, migraphx::make_op("dequantizelinear"), input, scales);
}

TEST_CASE(dqlinear_mismatch_type)
{
    migraphx::shape zeros{migraphx::shape::float_type, {2, 4}};
    migraphx::shape scales{migraphx::shape::float_type, {2, 4}};
    migraphx::shape input{migraphx::shape::int8_type, {2, 4}};
    throws_shape(migraphx::make_op("dequantizelinear"), input, scales, zeros);
}

static void test_reduce_ops(const std::string& name)
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}},
                     migraphx::make_op(name, {{"axes", {0, 1, 2, 3}}}),
                     input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 1, 1}},
                     migraphx::make_op(name, {{"axes", {2, 3}}}),
                     input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}},
                     migraphx::make_op(name, {{"axes", {0}}}),
                     input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 1}},
                     migraphx::make_op(name, {{"axes", {-1}}}),
                     input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op(name, {{"axes", {4}}}), input);
    }
}

// dynamic shape
static void test_dyn_reduce_ops(const std::string& name)
{
    {
        migraphx::shape input{migraphx::shape::float_type, {{2, 3, {3}}, {2, 4, {4}}}};
        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            std::vector<migraphx::shape::dynamic_dimension>({{2, 3, {3}}, {1, 1}})},
            migraphx::make_op(name, {{"axes", {-1}}}),
            input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {{2, 3, {3}}, {2, 4, {4}}}};
        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            std::vector<migraphx::shape::dynamic_dimension>({{1, 1}, {2, 4, {4}}})},
            migraphx::make_op(name, {{"axes", {0}}}),
            input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {{2, 3, {3}}, {2, 4, {4}}}};
        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            std::vector<migraphx::shape::dynamic_dimension>({{1, 1}, {1, 1}})},
            migraphx::make_op(name, {{"axes", {0, 1}}}),
            input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {{2, 3, {3}}, {2, 4, {4}}}};
        throws_shape(migraphx::make_op(name, {{"axes", {4}}}), input);
    }
}

static void test_reduce_ops_variable_axes(const std::string& name)
{
    {
        migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
        migraphx::shape axes_shape{migraphx::shape::int64_type, {1}};
        migraphx::shape expected_shape{migraphx::shape::float_type, {{1, 2}, {1, 3}, {1, 4}}};
        expect_shape(expected_shape, migraphx::make_op(name), input_shape, axes_shape);
    }

    {
        migraphx::shape input_shape{migraphx::shape::float_type, {{2, 3}, {3, 4}}};
        migraphx::shape axes_shape{migraphx::shape::int64_type, {1}};
        migraphx::shape expected_shape{migraphx::shape::float_type, {{1, 3}, {1, 4}}};
        expect_shape(expected_shape, migraphx::make_op(name), input_shape, axes_shape);
    }
}

TEST_CASE(reduce_max) { test_reduce_ops("reduce_max"); }
TEST_CASE(reduce_min) { test_reduce_ops("reduce_min"); }
TEST_CASE(reduce_mean) { test_reduce_ops("reduce_mean"); }
TEST_CASE(reduce_prod) { test_reduce_ops("reduce_prod"); }
TEST_CASE(reduce_sum) { test_reduce_ops("reduce_sum"); }

TEST_CASE(reduce_max_dyn) { test_dyn_reduce_ops("reduce_max"); }
TEST_CASE(reduce_min_dyn) { test_dyn_reduce_ops("reduce_min"); }
TEST_CASE(reduce_mean_dyn) { test_dyn_reduce_ops("reduce_mean"); }
TEST_CASE(reduce_prod_dyn) { test_dyn_reduce_ops("reduce_prod"); }
TEST_CASE(reduce_sum_dyn) { test_dyn_reduce_ops("reduce_sum"); }

TEST_CASE(reduce_max_variable_axes) { test_reduce_ops_variable_axes("reduce_max"); }
TEST_CASE(reduce_min_variable_axes) { test_reduce_ops_variable_axes("reduce_min"); }
TEST_CASE(reduce_mean_variable_axes) { test_reduce_ops_variable_axes("reduce_mean"); }
TEST_CASE(reduce_prod_variable_axes) { test_reduce_ops_variable_axes("reduce_prod"); }
TEST_CASE(reduce_sum_variable_axes) { test_reduce_ops_variable_axes("reduce_sum"); }

TEST_CASE(reshape_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {24, 1, 1, 1}};
    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 1, 1}, {1, 3, 4, 2}, {1, 3, 4, 2}})
    {
        std::vector<std::size_t> lens(new_shape.size());
        std::copy(new_shape.begin(), new_shape.end(), lens.begin());
        migraphx::shape output{migraphx::shape::float_type, lens};
        expect_shape(output, migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }
}

TEST_CASE(reshape_shape_invalid)
{
    migraphx::shape input{migraphx::shape::float_type, {24, 1, 1, 1}};
    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 2, 2}, {1, 3, -1, -1}, {3, 0}, {3, 2}})
    {
        throws_shape(migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }
}

TEST_CASE(reshape_shape_minus1_reshapes)
{
    migraphx::shape input{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<std::pair<std::vector<int64_t>, migraphx::shape>> minus1_tests{
        {{2, -1, 3}, {migraphx::shape::float_type, {2, 4, 3}}},
        {{0, -1, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, -1, 0}, {migraphx::shape::float_type, {2, 12, 1}}},
        {{0, 0, -1}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, 0, -1}, {migraphx::shape::float_type, {2, 1, 12}}},
        {{-1, 2, 3}, {migraphx::shape::float_type, {4, 2, 3}}},
        {{-1, 0, 3}, {migraphx::shape::float_type, {8, 1, 3}}},
        {{-1, 0, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{-1, 3, 0}, {migraphx::shape::float_type, {8, 3, 1}}}};

    for(auto& it : minus1_tests)
    {
        expect_shape(it.second, migraphx::make_op("reshape", {{"dims", it.first}}), input);
    }
}

TEST_CASE(reshape_nonstandard)
{
    auto input = migraphx::shape::from_permutation(migraphx::shape::float_type,
                                                   {4, 24, 1, 1, 1},
                                                   migraphx::invert_permutation({1, 0, 2, 3, 4}));
    std::vector<std::vector<std::size_t>> tests{{4, 24},
                                                {4, 24, 1, 1, 1, 1},
                                                {4, 8, 3, 1, 1},
                                                {4, 1, 3, 4, 2},
                                                {4, 1, 4, 3, 2},
                                                {4, 2, 4, 3},
                                                {4, 2, 12, 1},
                                                {4, 2, 1, 12},
                                                {4, 4, 2, 3},
                                                {4, 8, 1, 3},
                                                {4, 8, 3, 1}};

    for(auto dims : tests)
    {
        migraphx::shape output = migraphx::shape{migraphx::shape::float_type, dims};
        expect_shape(output, migraphx::make_op("reshape", {{"dims", dims}}), input);
    }
}

TEST_CASE(reshape_nonstandard_squeeze)
{
    auto input = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {2, 16, 16, 1280}, migraphx::invert_permutation({0, 2, 3, 1}));
    std::vector<std::size_t> lens = {2, 256, 1280};
    migraphx::shape output        = migraphx::shape{migraphx::shape::float_type, lens};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", lens}}), input);
}

TEST_CASE(reshape_nonstandard_error)
{
    auto input = migraphx::shape::from_permutation(migraphx::shape::float_type,
                                                   {4, 24, 1, 1, 1},
                                                   migraphx::invert_permutation({1, 0, 2, 3, 4}));
    for(auto&& new_shape : std::vector<std::vector<int64_t>>{{4, 8, 3, 2, 2},
                                                             {1},
                                                             {4, 8, 4},
                                                             {4, 24, 1, 1, 1, 1, 2},
                                                             {8, 4, 4},
                                                             {4, 1, 3, -1, -1},
                                                             {4, 3, 0},
                                                             {4, 3, 2},
                                                             {3, 0},
                                                             {3, 2}})
    {
        throws_shape(migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }
}

TEST_CASE(reshape_transposed_squeeze)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {1, 4}};
    migraphx::shape output{migraphx::shape::float_type, {64}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_nonpacked_unsqueeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {4, 2, 8}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_nonpacked_unsqueeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 2, 16}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_nonpacked_squeeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {64}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_nonpacked_squeeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {64}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_broadcast_unsqueeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 256, 1280}, {0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 16, 16, 1280}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_broadcast_unsqueeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 256, 1280}, {0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 256, 16, 80}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_broadcast_squeeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 16, 16, 1280}, {0, 0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 256, 1280}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_broadcast_squeeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {64}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_broadcast_squeeze3)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {1, 0}};
    migraphx::shape output{migraphx::shape::float_type, {64}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_broadcast_squeeze_memlayout_change)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 16, 16, 1280}, {0, 0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 16, 256, 80}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_dyn_1in)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    for(auto&& new_shape : std::vector<std::vector<int64_t>>{
            {-1, 1, 1, 24}, {0, 8, 3, 1}, {-1, 3, 4, 2}, {0, 2, 4, 3}, {2, 2, 12, 0}})
    {
        std::vector<migraphx::shape::dynamic_dimension> out_dyn_dims{};
        for(std::size_t i = 0; i < new_shape.size(); ++i)
        {
            if(new_shape[i] == 0 or new_shape[i] == -1)
            {
                out_dyn_dims.push_back(input.dyn_dims().at(i));
            }
            else
            {
                std::size_t d = new_shape[i];
                out_dyn_dims.push_back({d, d});
            }
        }
        migraphx::shape output{migraphx::shape::float_type, out_dyn_dims};
        expect_shape(output, migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }
}

// more -1 dims attribute testing
TEST_CASE(reshape_dyn_1in_negative_1_dims_0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {2, 8}, {2, 8}}};
    std::vector<migraphx::shape::dynamic_dimension> out_dyn_dims = {
        {1, 4}, {12, 12}, {2, 8}, {4, 16}};
    migraphx::shape output{migraphx::shape::float_type, out_dyn_dims};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", {0, 12, 0, -1}}}), input);
}

// output dynamic shape is surprising but that's how the calculation works out
TEST_CASE(reshape_dyn_1in_negative_1_dims_1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {2, 8}, {2, 8}}};
    std::vector<migraphx::shape::dynamic_dimension> out_dyn_dims = {
        {1, 4}, {24, 384}, {2, 2}, {2, 2}};
    migraphx::shape output{migraphx::shape::float_type, out_dyn_dims};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", {0, -1, 2, 2}}}), input);
}

TEST_CASE(reshape_dyn_1in_negative_1_dims_2)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {2, 8}, {2, 8}}};
    std::vector<migraphx::shape::dynamic_dimension> out_dyn_dims = {{1, 4}, {24, 24}, {4, 64}};
    migraphx::shape output{migraphx::shape::float_type, out_dyn_dims};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", {0, 0, -1}}}), input);
}

TEST_CASE(reshape_dyn_1in_negative_1_dims_3)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}}};
    std::vector<migraphx::shape::dynamic_dimension> out_dyn_dims = {{1, 4}, {4, 4}, {3, 3}, {2, 2}};
    migraphx::shape output{migraphx::shape::float_type, out_dyn_dims};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", {0, 4, 3, 2}}}), input);
}

// note how non-fixed dynamic dimension on axis=0 goes to 2 from `dims` attribute
// code assumes that this will work at run-time
TEST_CASE(reshape_dyn_1in_dyn_to_fixed)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    std::vector<int64_t> dims_attr = {2, 1, 1, 24};
    migraphx::shape output{migraphx::shape::float_type, {{2, 2}, {1, 1}, {1, 1}, {24, 24}}};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", dims_attr}}), input);
}

TEST_CASE(reshape_dyn_2in_0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4}, {8, 8}, {3, 3}, {1, 1}}};
    expect_shape(output, migraphx::make_op("reshape"), input, output);
}

TEST_CASE(reshape_dyn_2in_1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    migraphx::shape output{migraphx::shape::float_type, {{12, 12}, {2, 2}, {1, 1}, {1, 4}}};
    expect_shape(output, migraphx::make_op("reshape"), input, output);
}

TEST_CASE(reshape_dyn_2in_2)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 24, 1, 1}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 2}, {6, 12}, {1, 1}, {4, 4}}};
    expect_shape(output, migraphx::make_op("reshape"), input, output);
}

TEST_CASE(reshape_dyn_1in_multiple_non_fixed0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {10, 20}, {1, 1}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 4}, {1, 1}, {10, 20}, {24, 24}}};
    std::vector<int64_t> new_shape = {0, 1, 0, 24};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", new_shape}}), input);
}

TEST_CASE(reshape_dyn_1in_multiple_non_fixed1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 8}, {24, 24}, {10, 20}, {1, 1}}};
    migraphx::shape output{migraphx::shape::float_type, {{1, 8}, {1, 1}, {10, 20}, {24, 24}}};
    std::vector<int64_t> new_shape = {-1, 1, 0, 24};
    expect_shape(output, migraphx::make_op("reshape", {{"dims", new_shape}}), input);
}

TEST_CASE(reshape_lazy_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {24, 1, 1, 1}};
    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 1, 1}, {1, 3, 4, 2}, {1, 3, 4, 2}})
    {
        std::vector<std::size_t> lens(new_shape.size());
        std::copy(new_shape.begin(), new_shape.end(), lens.begin());
        migraphx::shape output{migraphx::shape::float_type, lens};
        expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
    }

    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 2, 2}, {1, 3, -1, -1}, {3, 0}, {3, 2}})
    {
        throws_shape(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
    }

    std::vector<std::pair<std::vector<int64_t>, migraphx::shape>> minus1_tests{
        {{2, -1, 3}, {migraphx::shape::float_type, {2, 4, 3}}},
        {{0, -1, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, -1, 0}, {migraphx::shape::float_type, {2, 12, 1}}},
        {{0, 0, -1}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, 0, -1}, {migraphx::shape::float_type, {2, 1, 12}}},
        {{-1, 2, 3}, {migraphx::shape::float_type, {4, 2, 3}}},
        {{-1, 0, 3}, {migraphx::shape::float_type, {8, 1, 3}}},
        {{-1, 0, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{-1, 3, 0}, {migraphx::shape::float_type, {8, 3, 1}}}};

    for(auto& it : minus1_tests)
    {
        expect_shape(it.second, migraphx::make_op("reshape_lazy", {{"dims", it.first}}), input);
    }
}

// This uses the permutation to compute the reshape_lazy since its simpler than
// trying to calculate strides. As we collapse or expand dimensions, we
// remove the collapsed dimensions or duplicate the expanded dimensions in
// the permutation. Then we renumber the permutation. So for dimensions of 4,
// 24, 1, 1, 1 with a permutation of 1, 0, 2, 3, 4 that reshape_lazys to 4, 1, 3,
// 4, 2, we first remove the collapsed dimensions or duplicate the expanded
// dimensions which gives 1, 0, 0, 0, 0. Then after renumbering we get a
// final permutation of 4, 0, 1, 2, 3.
TEST_CASE(reshape_lazy_nonstandard)
{
    auto input = migraphx::shape::from_permutation(migraphx::shape::float_type,
                                                   {4, 24, 1, 1, 1},
                                                   migraphx::invert_permutation({1, 0, 2, 3, 4}));
    std::vector<std::pair<std::vector<std::size_t>, std::vector<int64_t>>> tests{
        {{4, 24}, {1, 0}},
        {{4, 24, 1, 1, 1, 1}, {1, 0, 2, 3, 4, 5}},
        {{4, 8, 3, 1, 1}, {2, 0, 1, 3, 4}},
        {{4, 1, 3, 4, 2}, {4, 0, 1, 2, 3}},
        {{4, 1, 4, 3, 2}, {4, 0, 1, 2, 3}},
        {{4, 2, 4, 3}, {3, 0, 1, 2}},
        {{4, 2, 12, 1}, {2, 0, 1, 3}},
        {{4, 2, 1, 12}, {3, 0, 1, 2}},
        {{4, 4, 2, 3}, {3, 0, 1, 2}},
        {{4, 8, 1, 3}, {3, 0, 1, 2}},
        {{4, 8, 3, 1}, {2, 0, 1, 3}}};

    for(const auto& [dims, perm] : tests)
    {
        migraphx::shape output = migraphx::shape::from_permutation(
            migraphx::shape::float_type, dims, migraphx::invert_permutation(perm));
        expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", dims}}), input);
    }
}

TEST_CASE(reshape_lazy_nonstandard_squeeze)
{
    auto input = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {2, 16, 16, 1280}, migraphx::invert_permutation({0, 2, 3, 1}));
    std::vector<std::size_t> lens = {2, 256, 1280};
    migraphx::shape output        = migraphx::shape::from_permutation(
        migraphx::shape::float_type, lens, migraphx::invert_permutation({0, 2, 1}));
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", lens}}), input);
}

TEST_CASE(reshape_lazy_nonstandard_error)
{
    auto input = migraphx::shape::from_permutation(migraphx::shape::float_type,
                                                   {4, 24, 1, 1, 1},
                                                   migraphx::invert_permutation({1, 0, 2, 3, 4}));
    for(auto&& new_shape : std::vector<std::vector<int64_t>>{{4, 8, 3, 2, 2},
                                                             {1},
                                                             {4, 8, 4},
                                                             {4, 24, 1, 1, 1, 1, 2},
                                                             {8, 4, 4},
                                                             {4, 1, 3, -1, -1},
                                                             {4, 3, 0},
                                                             {4, 3, 2},
                                                             {3, 0},
                                                             {3, 2}})
    {
        throws_shape(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
    }
}

TEST_CASE(reshape_lazy_transposed_squeeze)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {1, 4}};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", {64}}}), input);
}

TEST_CASE(reshape_lazy_nonpacked_unsqueeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {4, 2, 8}, {32, 16, 2}};
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_lazy_nonpacked_unsqueeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 2, 16}, {64, 32, 2}};
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_lazy_nonpacked_squeeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    migraphx::shape output{migraphx::shape::float_type, {64}, {2}};
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_lazy_nonpacked_squeeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 1}};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", {64}}}), input);
}

TEST_CASE(reshape_lazy_broadcast_unsqueeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 256, 1280}, {0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 16, 16, 1280}, {0, 0, 0, 1}};
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_lazy_broadcast_unsqueeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 256, 1280}, {0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 256, 16, 80}, {0, 0, 80, 1}};
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_lazy_broadcast_squeeze1)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 16, 16, 1280}, {0, 0, 0, 1}};
    migraphx::shape output{migraphx::shape::float_type, {2, 256, 1280}, {0, 0, 1}};
    expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", output.lens()}}), input);
}

TEST_CASE(reshape_lazy_broadcast_squeeze2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {0, 1}};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", {64}}}), input);
}

TEST_CASE(reshape_lazy_broadcast_squeeze3)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {1, 0}};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", {64}}}), input);
}

TEST_CASE(reshape_lazy_broadcast_squeeze_error)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 16, 16, 1280}, {0, 0, 0, 1}};
    std::vector<int64_t> new_shape = {2, 16, 20480};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
}

TEST_CASE(reshape_lazy_dyn_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    for(auto&& new_shape : std::vector<std::vector<int64_t>>{
            {-1, 1, 1, 24}, {0, 8, 3, 1}, {-1, 3, 4, 2}, {0, 2, 4, 3}})
    {
        std::vector<migraphx::shape::dynamic_dimension> out_dyn_dims{};
        for(std::size_t i = 0; i < new_shape.size(); ++i)
        {
            if(new_shape[i] == 0 or new_shape[i] == -1)
            {
                out_dyn_dims.push_back(input.dyn_dims().at(i));
            }
            else
            {
                std::size_t d = new_shape[i];
                out_dyn_dims.push_back({d, d});
            }
        }
        migraphx::shape output{migraphx::shape::float_type, out_dyn_dims};
        expect_shape(output, migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
    }
}

TEST_CASE(reshape_lazy_multiple_non_fixed_error)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {10, 20}, {1, 1}}};
    std::vector<int64_t> new_shape = {0, 1, 0, 24};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
}

TEST_CASE(reshape_lazy_fixed_ele_not_matching_error)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {10, 10}, {1, 1}}};
    std::vector<int64_t> new_shape = {0, 1, 5, 24};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
}

TEST_CASE(reshape_lazy_non_fixed_not_matching_error)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    std::vector<int64_t> new_shape = {2, 1, 1, 24};
    throws_shape(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
}

TEST_CASE(resize_single_input)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    std::vector<size_t> sizes_vec{3, 4};
    migraphx::shape output{migraphx::shape::float_type, {sizes_vec}};
    expect_shape(output,
                 migraphx::make_op("resize",
                                   {{"sizes", {3, 4}},
                                    {"nearest_mode", "floor"},
                                    {"coordinate_transformation_mode", "asymmetric"}}),
                 input);
}

TEST_CASE(resize_single_scale_input)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    std::vector<size_t> sizes_vec{3, 4};
    migraphx::shape output{migraphx::shape::float_type, {sizes_vec}};
    expect_shape(output,
                 migraphx::make_op("resize",
                                   {{"scales", {0.75, 0.25}},
                                    {"nearest_mode", "floor"},
                                    {"coordinate_transformation_mode", "asymmetric"}}),
                 input);
}

TEST_CASE(resize_single_input_err1)
{
    // doesn't have either sizes or scales attribute
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    throws_shape(migraphx::make_op(
                     "resize",
                     {{"nearest_mode", "floor"}, {"coordinate_transformation_mode", "asymmetric"}}),
                 input);
}

TEST_CASE(resize_single_input_err2)
{
    // can't have both sizes and scales attribute
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    throws_shape(migraphx::make_op("resize", {{"scales", {0.75, 0.25}}, {"sizes", {87, 88}}}),
                 input);
}

TEST_CASE(resize_single_dyn_input_err3)
{
    // single dynamic input not supported yet
    std::vector<migraphx::shape::dynamic_dimension> input_dims = {{4, 5}, {15, 16}};
    migraphx::shape input{migraphx::shape::float_type, input_dims};
    throws_shape(migraphx::make_op("resize", {{"scales", {0.75, 0.25}}}), input);
}

TEST_CASE(resize_multi_input)
{
    // resize always outputs a dynamic shape if there are 2 inputs
    migraphx::shape input{migraphx::shape::float_type, {4, 16}, {32, 2}};
    std::size_t max_val = std::numeric_limits<std::size_t>::max();
    migraphx::shape sizes{migraphx::shape::int64_type, {2}};
    migraphx::shape output{migraphx::shape::float_type, {{0, max_val}, {0, max_val}}};
    expect_shape(output,
                 migraphx::make_op("resize",
                                   {{"mode", "nearest"},
                                    {"nearest_mode", "floor"},
                                    {"coordinate_transformation_mode", "asymmetric"}}),
                 input,
                 sizes);
}

TEST_CASE(return_shape_tuple)
{
    using migraphx::shape;
    auto op = migraphx::make_op("@return");
    shape s0{shape::bool_type, {1, 1}};
    shape s1{shape::float_type, {2, 3}};

    std::vector<shape> s{s0, s1};
    auto s_out = op.compute_shape(s);
    EXPECT(s_out.type() == shape::tuple_type);
    EXPECT(s0 == s_out.sub_shapes()[0]);
    EXPECT(s1 == s_out.sub_shapes()[1]);
}

TEST_CASE(return_shape_half)
{
    using migraphx::shape;
    auto op = migraphx::make_op("@return");
    std::vector<shape> s{{shape::half_type}};
    EXPECT(op.compute_shape(s) == shape{shape::half_type});
}

TEST_CASE(return_shape_empty)
{
    using migraphx::shape;
    auto op = migraphx::make_op("@return");
    std::vector<shape> s;
    EXPECT(op.compute_shape(s) == shape{});
}

TEST_CASE(rnn)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(select_module_dyn)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {3, 3}, {255, 255}, {255, 255}}};
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1000, 1000}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    expect_shape(
        out_attr,
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        input);
}

TEST_CASE(slice_static_shape)
{
    migraphx::shape input{migraphx::shape::int32_type, {2, 2, 3}};
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}},
                 migraphx::make_op(
                     "slice", {{"axes", {0, 1, 2}}, {"starts", {0, 0, 1}}, {"ends", {2, 2, 3}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 1}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {10}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 1}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {-1}}, {"ends", {10}}}),
                 input);
}

TEST_CASE(slice_var_inputs_static_shape0)
{
    // attr ends and axes set; inputs are (data, input_starts)
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice", {{"ends", {2, 3}}, {"axes", {1, 2}}}),
                 input,
                 starts);
}

TEST_CASE(slice_var_inputs_static_mismatch_error0)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    throws_shape(
        migraphx::make_op("slice", {{"ends", {2, 3, 4}}, {"axes", {0, 1, 2}}}), input, starts);
}

TEST_CASE(slice_var_inputs_static_shape1)
{
    // attr starts and axes set; inputs are (data, input_ends)
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice", {{"starts", {0, 1}}, {"axes", {1, 2}}}),
                 input,
                 ends);
}

TEST_CASE(slice_var_inputs_static_mismatch_error1)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    throws_shape(
        migraphx::make_op("slice", {{"starts", {0, 1, 2}}, {"axes", {0, 1, 2}}}), input, ends);
}

TEST_CASE(slice_var_inputs_static_shape2)
{
    // attr starts and ends set; inputs are (data, input_axes)
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice", {{"starts", {0, 1}}, {"ends", {1, 2}}}),
                 input,
                 axes);
}

TEST_CASE(slice_var_inputs_static_mismatch_error2)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    throws_shape(
        migraphx::make_op("slice", {{"starts", {0, 1, 2}}, {"ends", {3, 4, 4}}}), input, axes);
}

TEST_CASE(slice_var_inputs_static_shape3)
{
    // attr axes set; inputs are (data, input_starts, input_ends)
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice", {{"axes", {1, 2}}}),
                 input,
                 starts,
                 ends);
}

TEST_CASE(slice_var_inputs_static_mismatch_error3)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    throws_shape(migraphx::make_op("slice", {{"axes", {0, 1, 2}}}), input, starts, ends);
}

TEST_CASE(slice_var_inputs_static_shape4)
{
    // attr ends set; inputs are (data, input_starts, input_axes)
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice", {{"ends", {3, 4}}}),
                 input,
                 starts,
                 axes);
}

TEST_CASE(slice_var_inputs_static_mismatch_error4)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    throws_shape(migraphx::make_op("slice", {{"ends", {3, 3, 3}}}), input, starts, axes);
}

TEST_CASE(slice_var_inputs_static_shape5)
{
    // attr starts set; inputs are (data, input_ends, input_axes)
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice", {{"starts", {0, 2}}}),
                 input,
                 ends,
                 axes);
}

TEST_CASE(slice_var_inputs_static_mismatch_error5)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    throws_shape(migraphx::make_op("slice", {{"starts", {0, 1, 2}}}), input, ends, axes);
}

TEST_CASE(slice_var_inputs_static_shape6)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 3}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice"),
                 input,
                 starts,
                 ends,
                 axes);
}

TEST_CASE(slice_var_inputs_static_mismatch_error6)
{
    migraphx::shape input{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {3}};
    throws_shape(migraphx::make_op("slice"), input, starts, ends, axes);
}

TEST_CASE(slice_var_inputs_dyn_shape0)
{
    // attr ends and axes set; inputs are (data, input_starts)
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 6}, {0, 6}, {0, 6}}},
                 migraphx::make_op("slice", {{"ends", {2, 3}}, {"axes", {1, 2}}}),
                 input,
                 starts);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error0)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    throws_shape(
        migraphx::make_op("slice", {{"ends", {2, 3, 4}}, {"axes", {0, 1, 2}}}), input, starts);
}

TEST_CASE(slice_var_inputs_dyn_shape1)
{
    // attr starts and axes set; inputs are (data, input_ends)
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 6}, {0, 6}, {0, 6}}},
                 migraphx::make_op("slice", {{"starts", {0, 1}}, {"axes", {1, 2}}}),
                 input,
                 ends);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error1)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    throws_shape(
        migraphx::make_op("slice", {{"starts", {0, 1, 2}}, {"axes", {0, 1, 2}}}), input, ends);
}

TEST_CASE(slice_var_inputs_dyn_shape2)
{
    // attr starts and ends set; inputs are (data, input_axes)
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 6}, {0, 6}, {0, 6}}},
                 migraphx::make_op("slice", {{"starts", {0, 1}}, {"ends", {8, 8}}}),
                 input,
                 axes);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error2)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    throws_shape(
        migraphx::make_op("slice", {{"starts", {0, 1, 2}}, {"ends", {3, 4, 4}}}), input, axes);
}

TEST_CASE(slice_var_inputs_dyn_shape3)
{
    // attr axes set; inputs are (data, input_starts, input_ends)
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{3, 6}, {0, 6}, {0, 6}}},
                 migraphx::make_op("slice", {{"axes", {1, 2}}}),
                 input,
                 starts,
                 ends);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error3)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    throws_shape(migraphx::make_op("slice", {{"axes", {0, 1, 2}}}), input, starts, ends);
}

TEST_CASE(slice_var_inputs_dyn_shape4)
{
    // attr ends set; inputs are (data, input_starts, input_axes)
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 6}, {0, 6}, {0, 6}}},
                 migraphx::make_op("slice", {{"ends", {3, 4}}}),
                 input,
                 starts,
                 axes);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error4)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    throws_shape(migraphx::make_op("slice", {{"ends", {3, 3, 3}}}), input, starts, axes);
}

TEST_CASE(slice_var_inputs_dyn_shape5)
{
    // attr starts set; inputs are (data, input_ends, input_axes)
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 6}, {0, 6}, {0, 6}}},
                 migraphx::make_op("slice", {{"starts", {0, 2}}}),
                 input,
                 ends,
                 axes);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error5)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    throws_shape(migraphx::make_op("slice", {{"starts", {0, 1, 2}}}), input, ends, axes);
}

TEST_CASE(slice_var_inputs_dyn_shape6)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {2}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{0, 6}, {0, 4}, {0, 4}}},
                 migraphx::make_op("slice"),
                 input,
                 starts,
                 ends,
                 axes);
}

TEST_CASE(slice_var_inputs_dyn_mismatch_error6)
{
    migraphx::shape input{migraphx::shape::float_type, {{3, 6}, {4, 6}, {4, 6}}};
    migraphx::shape starts{migraphx::shape::int64_type, {2}};
    migraphx::shape ends{migraphx::shape::int64_type, {2}};
    migraphx::shape axes{migraphx::shape::int64_type, {3}};
    throws_shape(migraphx::make_op("slice"), input, starts, ends, axes);
}

TEST_CASE(slice_dyn_shape0)
{
    migraphx::shape input{migraphx::shape::int32_type, {{2, 3}, {7, 7}, {2, 3}}};

    // Slice axis 1 to size 4-1=3
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {{2, 3}, {3, 3}, {2, 3}}},
                 migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {4}}}),
                 input);
}

TEST_CASE(slice_dyn_shape1)
{
    migraphx::shape input{migraphx::shape::int32_type, {{2, 3}, {7, 7}, {2, 3}}};
    // Slice axis 1 with negative index
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {{2, 3}, {2, 2}, {2, 3}}},
                 migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {-4}}}),
                 input);
}

TEST_CASE(slice_dyn_shape2)
{
    migraphx::shape input{migraphx::shape::int32_type, {{2, 3}, {7, 7}, {2, 3}}};
    // Sliced range max bigger than dimension; is clipped
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {{2, 3}, {6, 6}, {2, 3}}},
                 migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {10}}}),
                 input);
}

TEST_CASE(slice_dyn_shape3)
{
    // TODO: When non-fixed dimension slicing is allowed, Slice to a size smaller than min.
    // Until then, this action is an error.
    migraphx::shape input{migraphx::shape::int32_type, {{2, 3}, {7, 8}, {2, 3}}};
    throws_shape(migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}),
                 input);
    // clang-format off
    //     expect_shape(migraphx::shape{migraphx::shape::int32_type, {{2, 3}, {1, 1}, {2, 3}}},
    //                  migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}),
    //                  input);
    // clang-format on
}

TEST_CASE(slice_dyn_shape4)
{
    migraphx::shape input{migraphx::shape::int32_type, {{2, 2}, {7, 7}, {2, 3}}};
    // Slice multiple axes:  axis 0 to size 2-1=1 and axis 1 to size 4-1=3
    expect_shape(
        migraphx::shape{migraphx::shape::int32_type, {{1, 1}, {3, 3}, {2, 3}}},
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 1}}, {"ends", {2, 4}}}),
        input);
}

TEST_CASE(slice_dyn_shape5)
{
    // Axis out of range.
    migraphx::shape input{migraphx::shape::int32_type, {{2, 2}, {7, 7}, {2, 3}}};
    throws_shape(
        migraphx::make_op("slice", {{"axes", {0, 20}}, {"starts", {1, 1}}, {"ends", {2, 4}}}),
        input);
}

TEST_CASE(test_scan_slice1)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape axis_input{migraphx::shape::int64_type};
    migraphx::shape expected{migraphx::shape::float_type, {1, 3, 4}};
    expect_shape(expected,
                 migraphx::make_op("scan_slice", {{"axis", 0}, {"direction", 0}}),
                 input,
                 axis_input);
}

TEST_CASE(test_scan_slice2)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 6, 5}};
    migraphx::shape axis_input{migraphx::shape::int64_type};
    migraphx::shape expected{migraphx::shape::float_type, {4, 1, 5}, {30, 5, 1}};
    expect_shape(expected,
                 migraphx::make_op("scan_slice", {{"axis", 1}, {"direction", 0}}),
                 input,
                 axis_input);
}

TEST_CASE(test_scan_slice3)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 5, 7}};
    migraphx::shape axis_input{migraphx::shape::int64_type};
    migraphx::shape expected{migraphx::shape::float_type, {2, 5, 1}, {35, 7, 1}};
    expect_shape(expected,
                 migraphx::make_op("scan_slice", {{"axis", -1}, {"direction", 0}}),
                 input,
                 axis_input);
}

TEST_CASE(test_scan_slice4)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 5, 7}};
    migraphx::shape axis_input{migraphx::shape::int64_type};
    migraphx::shape expected{migraphx::shape::float_type, {1, 5, 7}, {35, 7, 1}};
    expect_shape(expected,
                 migraphx::make_op("scan_slice", {{"axis", -3}, {"direction", 1}}),
                 input,
                 axis_input);
}

TEST_CASE(softmax_dyn0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}};
    expect_shape(input, migraphx::make_op("softmax", {{"axis", 0}}), input);
}

TEST_CASE(softmax_dyn1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 1}, {3, 3}, {4, 6}, {5, 8, {6}}}};
    expect_shape(input, migraphx::make_op("softmax", {{"axis", 0}}), input);
}

TEST_CASE(test_argmax)
{
    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                     migraphx::make_op("argmax", {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                     migraphx::make_op("argmax", {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                     migraphx::make_op("argmax", {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                     migraphx::make_op("argmax", {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op("argmax", {{"axis", 4}}), input);
    }
}

TEST_CASE(test_argmin)
{
    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                     migraphx::make_op("argmin", {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                     migraphx::make_op("argmin", {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                     migraphx::make_op("argmin", {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                     migraphx::make_op("argmin", {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op("argmin", {{"axis", 4}}), input);
    }
}

TEST_CASE(test_scalar)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}, {1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4, 5}, {0, 0, 0, 0}};
    expect_shape(s2, migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 3, 4, 5}}}), s1);
}

TEST_CASE(test_scalar_nelemnts)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
    throws_shape(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 3, 4, 5}}}), input);
}

TEST_CASE(test_gathernd)
{
    {
        // k > r
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {2, 4}};
        migraphx::shape ds{dtype, {8}};

        int batch_dims(1);
        throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
    }

    {
        // k > r - batch_dims
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {2, 4}};
        migraphx::shape ds{dtype, {2}};

        int batch_dims(1);
        throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
    }

    {
        // batch_dims >= r
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {2, 1}};
        migraphx::shape ds{dtype, {2, 5, 6, 7}};

        int batch_dims(3);
        throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
    }

    {
        // int(q) + r - k - batch_dims - 1 = 0 => returns a scalar
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {1}};
        migraphx::shape ds{dtype, {2}};

        migraphx::shape s0{dtype, {1}};
        expect_shape(s0, migraphx::make_op("gathernd"), ds, is);
    }

    {
        // See Example 4 at https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {2, 2}};
        migraphx::shape ds{dtype, {2, 2}};

        migraphx::shape s0{dtype, {2}};
        expect_shape(s0, migraphx::make_op("gathernd"), ds, is);
    }

    {
        // See Example 5 at https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {2, 1}};
        migraphx::shape ds{dtype, {2, 2, 2}};

        int batch_dims(1);
        migraphx::shape s0{dtype, {2, 2}};
        expect_shape(s0, migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
    }
}

TEST_CASE(test_gathernd_dynamic0)
{
    // k > r
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {2, 4}};
    std::vector<migraphx::shape::dynamic_dimension> b{{8, 8}};
    migraphx::shape ds{dtype, b};

    int batch_dims(1);
    throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic1)
{
    // k > r - batch_dims
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {2, 4}};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 2}};
    migraphx::shape ds{dtype, b};

    int batch_dims(1);
    throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic2)
{
    // batch_dims >= r
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {2, 1}};
    migraphx::shape ds{dtype, {{2, 3, {3}}, {5, 6, {5}}, {6, 9, {7}}, {7, 8, {8}}}};

    int batch_dims(3);
    throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic3)
{
    // int(q) + r - k - batch_dims - 1 = 0 => returns a scalar
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {1}};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 2}};
    migraphx::shape ds{dtype, b};

    migraphx::shape::dynamic_dimension ddout{1, 1};
    migraphx::shape s0{dtype, {ddout}};
    expect_shape(s0, migraphx::make_op("gathernd"), ds, is);
}

TEST_CASE(test_gathernd_dynamic4)
{
    // See Example 1 at https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {2, 2}};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 2}, {2, 2}};
    migraphx::shape ds{dtype, b};

    migraphx::shape::dynamic_dimension ddout{2, 2};
    migraphx::shape s0{dtype, {ddout}};
    expect_shape(s0, migraphx::make_op("gathernd"), ds, is);
}

TEST_CASE(test_gathernd_dynamic5)
{
    // See Example 5 at https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    // index static shape, data dynamic
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {2, 1}};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 2}, {2, 2}, {2, 2}};
    migraphx::shape ds{dtype, b};

    std::vector<migraphx::shape::dynamic_dimension> ddout{{2, 2}, {2, 2}};
    int batch_dims(1);
    migraphx::shape s0{dtype, {ddout}};
    expect_shape(s0, migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic6)
{
    // See Example 5 at https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    // index dynamic shape, data static
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 3}, {1, 1}};
    migraphx::shape is{itype, b};
    migraphx::shape ds{dtype, {2, 2, 2}};

    std::vector<migraphx::shape::dynamic_dimension> ddout{{2, 3}, {2, 2}};
    int batch_dims(1);
    migraphx::shape s0{dtype, {ddout}};
    expect_shape(s0, migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic6a)
{
    // indices with non-fixed dynamic dimension k
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 2}, {1, 3}};
    migraphx::shape is{itype, b};
    migraphx::shape ds{dtype, {2, 2, 2}};

    int batch_dims(1);
    throws_shape(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic7)
{
    // See Example 5 at https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    // index and data both dynamic shapes
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    std::vector<migraphx::shape::dynamic_dimension> idyn{{2, 5}, {1, 1}};
    migraphx::shape is{itype, idyn};
    std::vector<migraphx::shape::dynamic_dimension> bdyn{{1, 2}, {1, 2}, {1, 2}};
    migraphx::shape ds{dtype, bdyn};

    std::vector<migraphx::shape::dynamic_dimension> ddout{{2, 5}, {1, 2}};
    int batch_dims(1);
    migraphx::shape s0{dtype, {ddout}};
    expect_shape(s0, migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_gathernd_dynamic8)
{
    // Same shapes as ref_ops_test gathernd_dynamic
    // index static shape, data dynamic
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {2, 5, 1}};
    std::vector<migraphx::shape::dynamic_dimension> b{{6, 7, {7}}, {3, 3}, {1, 4}};
    migraphx::shape ds{dtype, b};

    std::vector<migraphx::shape::dynamic_dimension> ddout{{2, 2}, {5, 5}, {1, 4}};
    int batch_dims(1);
    migraphx::shape s0{dtype, {ddout}};
    expect_shape(s0, migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), ds, is);
}

TEST_CASE(test_scatternd0)
{
    // good
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape us{dtype, {4}};
    expect_shape(ds, migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd1)
{
    // good, broadcasted
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {4, 1}, {4, 0}};
    migraphx::shape us{dtype, {4}};
    expect_shape(ds, migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd2)
{
    // too many inputs
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape us{dtype, {4}};
    migraphx::shape zs{dtype, {4}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us, zs);
}

TEST_CASE(test_scatternd3)
{
    // q + r - k - 1 matches upd_lens.size(), but k > r
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {5, 4, 2}};
    migraphx::shape us{dtype, {4}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd4)
{
    // q + r - k - 1 != upd_lens.size()
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape us{dtype, {2, 2}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd5)
{
    // dimensions don't match: update.lens != indices.lens[0:q-1]
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8, 3}};
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape us{dtype, {2, 2}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd_dyn0)
{
    // one dynamic input, invalid index
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {4}};
    migraphx::shape is{itype, {4, 13}};
    migraphx::shape::dynamic_dimension dd{4, 4};
    migraphx::shape us{dtype, {dd}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd_dyn1)
{
    // one dynamic input
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape::dynamic_dimension dd{4, 4};
    migraphx::shape us{dtype, {dd}};
    expect_shape(ds, migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd_dyn2)
{
    // one dynamic input and broadcasted data
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 3, 1, 4}, {0, 1, 1, 0}};
    migraphx::shape ds_std{dtype, {2, 3, 1, 4}};
    migraphx::shape is{itype, {4, 4}};
    migraphx::shape::dynamic_dimension dd{4, 4};
    migraphx::shape us{dtype, {dd}};
    expect_shape(ds_std, migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd_dyn3)
{
    // one dynamic input and standard, static data
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 3, 1, 4}};
    migraphx::shape is{itype, {4, 4}};
    migraphx::shape::dynamic_dimension dd{4, 4};
    migraphx::shape us{dtype, {dd}};
    expect_shape(ds, migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd_dyn4)
{
    // index is dynamic with last dimension not fixed
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 3, 1, 4}};
    migraphx::shape::dynamic_dimension dd{4, 5};
    migraphx::shape is{itype, {dd, dd}};
    migraphx::shape us{dtype, {dd}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_scatternd_dyn5)
{
    // dimensions don't match: update.lens != indices.lens[0:q-1]
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 3, 1, 4}};
    migraphx::shape::dynamic_dimension dd{4, 4};
    migraphx::shape::dynamic_dimension dbad{2, 3};
    migraphx::shape is{itype, {dd, dd}};
    migraphx::shape us{dtype, {dbad}};
    throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
}

TEST_CASE(test_squeeze)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {3}}}), s1);
}

TEST_CASE(test_squeeze_all)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}};
    migraphx::shape s2{migraphx::shape::float_type};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_squeeze_dyn)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {1, 1}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {3, 3}}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {3}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type, {{1, 4}, {3, 3}, {3, 3}}};
    expect_shape(s3, migraphx::make_op("squeeze"), s1);

    // allowing to squeeze dynamic_dimension that intersect with {1, 1}
    migraphx::shape s4{migraphx::shape::float_type, {{1, 1}, {3, 3}, {1, 1}, {3, 3}}};
    expect_shape(s4, migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

    throws_shape(migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_dyn_neg_axes)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {1, 1}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {3, 3}}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {-2}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type, {{1, 4}, {3, 3}, {3, 3}}};
    expect_shape(s3, migraphx::make_op("squeeze", {{"axes", {-2, -4}}}), s1);
}

TEST_CASE(test_squeeze_transpose)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 4, 1}, {4, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 4}, {4, 1}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_multibroadcast)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 1, 4}, {0, 1, 1, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4}, {0, 1, 0}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_slice)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 1, 4}, {108, 36, 6, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4}, {108, 36, 1}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_negative_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {-2}}}), s1);
}

TEST_CASE(test_squeeze_wrong_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    throws_shape(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unique_axis_invalid)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {10, 4, 3}};
    throws_shape(migraphx::make_op("unique", {{"axis", -1}}), x_shape);
}

TEST_CASE(test_unique_axis_negative)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {10, 4, 3}};

    std::vector<migraphx::shape::dynamic_dimension> y_dims{{1, 10}, {4, 4}, {3, 3}};
    std::vector<migraphx::shape::dynamic_dimension> idx_dims{{1, 10}};
    std::vector<migraphx::shape> y_dyn_shape{{migraphx::shape::float_type, y_dims},
                                             {migraphx::shape::int64_type, idx_dims},
                                             {migraphx::shape::int64_type, idx_dims},
                                             {migraphx::shape::int64_type, idx_dims}};

    expect_shape(
        migraphx::shape(y_dyn_shape), migraphx::make_op("unique", {{"axis", -3}}), x_shape);
}

TEST_CASE(test_unique_axis_none)
{
    migraphx::shape x_shape{migraphx::shape::half_type, {10, 4, 3}};

    std::vector<migraphx::shape::dynamic_dimension> y_dims{{1, 120}};
    std::vector<migraphx::shape::dynamic_dimension> idx_dims{{1, 120}};
    std::vector<migraphx::shape> y_dyn_shape{{migraphx::shape::half_type, y_dims},
                                             {migraphx::shape::int64_type, idx_dims},
                                             {migraphx::shape::int64_type, idx_dims},
                                             {migraphx::shape::int64_type, idx_dims}};

    expect_shape(migraphx::shape(y_dyn_shape), migraphx::make_op("unique"), x_shape);
}

TEST_CASE(test_unsqueeze)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 1, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_dyn)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4, {3}}, {2, 5}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4, {3}}, {2, 5}, {1, 1}, {3, 3}}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type, {{1, 4, {3}}, {2, 5}, {1, 1}, {3, 3}, {1, 1}}};
    expect_shape(s3, migraphx::make_op("unsqueeze", {{"axes", {2, 4}}}), s1);

    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2, 4}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_dyn_neg_axes)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4, {3}}, {2, 5}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4, {3}}, {2, 5}, {1, 1}, {3, 3}}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type, {{1, 4, {3}}, {2, 5}, {1, 1}, {3, 3}, {1, 1}}};
    expect_shape(s3, migraphx::make_op("unsqueeze", {{"axes", {-1, -3}}}), s1);
}

TEST_CASE(test_unsqueeze_step)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 2, 6}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_step_non_divisable)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_step_zero)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_step_at_end)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {3}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_mismatch_step_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2, 3}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis1)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 1, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis2)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 3, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-1}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis3)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 5, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-3}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}, {0}};
    migraphx::shape s2{migraphx::shape::float_type, {1}, {1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar_tensor1)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}, {0, 0, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}, {0, 0, 1, 0}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar_tensor2)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 1}, {0, 0, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 1, 1, 1}, {0, 0, 0, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-1}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar_step)
{
    migraphx::shape s{migraphx::shape::float_type, {6, 1, 2}, {0, 0, 0}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {0}}, {"steps", {3}}}), s);
}

TEST_CASE(test_unsqueeze_transpose)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 4, 3}, {12, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 4, 1, 3}, {12, 1, 12, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_transpose_step)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 4, 6}, {24, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 4, 2, 3}, {24, 1, 12, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_multibroadcast)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}, {0, 1, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 1, 4}, {0, 1, 0, 0}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_slice)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}, {108, 36, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 1, 4}, {108, 36, 4, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_axis_zero)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2, 3, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_axis_last)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-1}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_1)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2, 3, 4, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0, -1}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_2)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2, 3, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_3)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4, 1, 5, 1, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2, 4, 5}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_4)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4, 1, 5, 1, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {5, 4, 2}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_step)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 4, 10}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4, 2, 5, 1, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2, 4, 5}}, {"steps", {2}}}), s1);
}

TEST_CASE(transpose_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 2}, {1, 2}};
    expect_shape(input, migraphx::make_op("transpose", {{"permutation", {0, 1}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"permutation", {1, 0}}}), input);
    throws_shape(migraphx::make_op("transpose", {{"permutation", {1, 2}}}), input);
}

TEST_CASE(transpose_dyn_shape0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {2, 2}}};
    migraphx::shape output{migraphx::shape::float_type, {{2, 2}, {1, 4}}};
    expect_shape(input, migraphx::make_op("transpose", {{"permutation", {0, 1}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"permutation", {1, 0}}}), input);
}

TEST_CASE(transpose_dyn_shape1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4}, {4, 4}, {4, 4}}};
    migraphx::shape output{migraphx::shape::float_type, {{4, 4}, {4, 4}, {1, 4}}};
    expect_shape(input, migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"permutation", {2, 1, 0}}}), input);
}

TEST_CASE(transpose_axes_error)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 2}};
    throws_shape(migraphx::make_op("transpose", {{"permutation", {1}}}), input);
}

TEST_CASE(step_test)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 2, 4}};
    {
        migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2}, {8, 8, 3}};
        expect_shape(s2, migraphx::make_op("step", {{"axes", {1, 2}}, {"steps", {2, 3}}}), s1);
    }

    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 4}};
        throws_shape(migraphx::make_op("step", {{"axes", {1, 2}}, {"steps", {1}}}), s1);
    }

    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 4}};
        throws_shape(migraphx::make_op("step", {{"axes", {2, 3}}, {"steps", {2, 3}}}), s1);
    }
}

TEST_CASE(unary_scalar_input)
{
    migraphx::shape ss{migraphx::shape::half_type};
    expect_shape(ss, migraphx::make_op("sin"), ss);

    migraphx::shape s{migraphx::shape::float_type, {1}};
    expect_shape(s, migraphx::make_op("sin"), s);
}

TEST_CASE(unary_broadcast_input)
{
    migraphx::shape ss{migraphx::shape::half_type, {2, 3}, {1, 0}};
    migraphx::shape s{migraphx::shape::half_type, {2, 3}};
    expect_shape(s, migraphx::make_op("sin"), ss);
}

TEST_CASE(where_broadcast_input)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 2}, {3, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 2}};
    migraphx::shape s3{migraphx::shape::bool_type, {2, 2}};
    expect_shape(s2, migraphx::make_op("where"), s3, s1, s2);
}

TEST_CASE(where_dyn_input0)
{
    // dynamic shapes not the same
    migraphx::shape s1{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{2, 3}, {2, 3}}};
    migraphx::shape s3{migraphx::shape::bool_type, {2, 2}};
    throws_shape(migraphx::make_op("where"), s3, s1, s2);
}

TEST_CASE(where_dyn_input1)
{
    // mixed static/dynamic inputs (not allowed)
    migraphx::shape s1{migraphx::shape::float_type, {2, 2}, {2, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {{2, 2}, {2, 2}}};
    migraphx::shape s3{migraphx::shape::bool_type, {2, 2}, {2, 1}};
    throws_shape(migraphx::make_op("where"), s3, s1, s2);
}

TEST_CASE(where_dyn_input2)
{
    // dynamic shapes
    migraphx::shape s1{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
    migraphx::shape s3{migraphx::shape::bool_type, {{2, 3}, {3, 3}}};
    expect_shape(s2, migraphx::make_op("where"), s3, s1, s2);
}

TEST_CASE(where_dyn_input3)
{
    // dynamic shapes, predicate shape is different
    migraphx::shape s1{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
    migraphx::shape s2{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
    migraphx::shape s3{migraphx::shape::bool_type, {{2, 3}, {3, 4}}};
    throws_shape(migraphx::make_op("where"), s3, s1, s2);
}

TEST_CASE(roialign_test)
{
    migraphx::shape sx{migraphx::shape::float_type, {3, 4, 5, 6}};
    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}};
    migraphx::shape sout{migraphx::shape::float_type, {2, 4, 1, 1}};

    expect_shape(sout, migraphx::make_op("roialign"), sx, srois, sbi);

    migraphx::shape sbi1{migraphx::shape::int64_type, {2, 3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois, sbi1);

    migraphx::shape sbi2{migraphx::shape::int64_type, {3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois, sbi2);

    migraphx::shape srois1{migraphx::shape::float_type, {2, 4, 3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois1, sbi);

    migraphx::shape srois2{migraphx::shape::float_type, {2, 3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois2, sbi);
}

TEST_CASE(test_concat)
{
    migraphx::shape sx{migraphx::shape::float_type, {3, 4, 5, 6}};
    migraphx::shape sy{migraphx::shape::float_type, {3, 4, 1, 6}};
    migraphx::shape sout{migraphx::shape::float_type, {3, 4, 6, 6}};

    expect_shape(sout, migraphx::make_op("concat", {{"axis", 2}}), sx, sy);

    // axis out of range
    throws_shape(migraphx::make_op("concat", {{"axis", 11}}), sx, sy);

    // 1 input; no-op
    expect_shape(sx, migraphx::make_op("concat", {{"axis", 2}}), sx);

    // rank doesn't match
    migraphx::shape sbi1{migraphx::shape::int64_type, {2, 3}};
    throws_shape(migraphx::make_op("concat", {{"axis", 0}}), sx, sbi1);

    // non-matching dimension 2
    throws_shape(migraphx::make_op("concat", {{"axis", 1}}), sx, sy);

    // no input shapes (at least one is required)
    throws_shape(migraphx::make_op("concat", {{"axis", 0}}));
}

TEST_CASE(test_dyn_concat)
{
    migraphx::shape sx{migraphx::shape::float_type, {{1, 3, {3}}, {4, 4}, {1, 5, {5}}, {6, 6}}};
    migraphx::shape sy{migraphx::shape::float_type, {{1, 3, {3}}, {4, 4}, {1, 4, {4}}, {6, 6}}};
    migraphx::shape sout{migraphx::shape::float_type, {{1, 3, {3}}, {4, 4}, {2, 9}, {6, 6}}};

    expect_shape(sout, migraphx::make_op("concat", {{"axis", 2}}), sx, sy);

    // axis out of range
    throws_shape(migraphx::make_op("concat", {{"axis", 4}}), sx, sy);

    // rank doesn't match
    migraphx::shape srank{migraphx::shape::int64_type, {{1, 3, {3}}, {4, 4}}};
    throws_shape(migraphx::make_op("concat", {{"axis", 0}}), sx, srank);

    // non-matching dimension 2
    throws_shape(migraphx::make_op("concat", {{"axis", 1}}), sx, sy);

    // static and dynamic shapes together
    migraphx::shape sstat{migraphx::shape::float_type, {3, 4, 1, 6}};
    throws_shape(migraphx::make_op("concat", {{"axis", 2}}), sx, sstat);
}

TEST_CASE(test_binary_nonpacked)
{
    auto sx   = migraphx::shape(migraphx::shape::float_type, {4, 3}, {1, 8});
    auto sy   = migraphx::shape(migraphx::shape::float_type, {4, 3}, {1, 16});
    auto sout = migraphx::shape::from_permutation(migraphx::shape::float_type, {4, 3}, {1, 0});

    expect_shape(sout, migraphx::make_op("mul"), sx, sy);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
