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

TEST_CASE(gathernd_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2}};

    std::vector<float> data_vec(2 * 2);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{0, 0, 1, 1};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> gold{0, 3};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 1}};

    std::vector<float> data_vec(2 * 2);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> gold{2, 3, 0, 1};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_test_3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2, 1}};

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_test_4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 3, 2, 3}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2, 2}};

    std::vector<float> data_vec(2 * 3 * 2 * 3);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{0, 0, 0, 1, 0, 0, 0, 1};
    const int batch_dims = 1;

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), data, indices);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> gold{0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_test_5)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1, 3}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 3, 2}};

    std::vector<float> data_vec(2 * 3 * 1 * 3);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{0, 0, 0, 1, 0, 2, 0, 2, 0, 1, 0, 0};
    const int batch_dims = 2;

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), data, indices);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> gold{0, 4, 8, 11, 13, 15};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_test_6)
{
    // k > r - batch_dims
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1, 3}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 3, 3}};

    std::vector<float> data_vec(2 * 3 * 1 * 3);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec(2 * 3 * 3, 0);
    const int batch_dims = 2;

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    EXPECT(test::throws([&] {
        mm->add_instruction(
            migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), data, indices);
    }));
}

TEST_CASE(gathernd_dynamic0)
{
    // dynamic data, all dimensions fixed
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {{2, 2, {2}}, {3, 3}, {1, 1}}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2, 1}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};

    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic1)
{
    // dynamic data, dims not fixed
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {{2, 5, {2}}, {1, 5}, {1, 5}}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2, 1}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic2)
{
    // dynamic both index and data
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {{2, 5, {2}}, {1, 5}, {1, 5}}};
    migraphx::shape is{migraphx::shape::int64_type, {{2, 5, {3}}, {2, 3, {3}}, {1, 1}}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic3)
{
    // dynamic index, static data and a batch_dims input
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::shape is{migraphx::shape::int64_type, {{2, 5, {3}}, {2, 3, {3}}, {1, 1}}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    int batch_dims{1};
    auto gathernd_op = migraphx::make_op("gathernd", {{"batch_dims", batch_dims}});
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{1, 0, 3, 4};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic4)
{
    // int(q) + r - k - batch_dims - 1 = 0 => returns a scalar
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {migraphx::shape::dynamic_dimension({2, 2})}};
    migraphx::shape is{migraphx::shape::int64_type, {1}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {1}}; // index

    std::vector<float> data_vec(2);
    std::iota(data_vec.begin(), data_vec.end(), 4);
    std::vector<int64_t> indices_vec{1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_negative_index_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 1, 1}};

    std::vector<float> data_vec(2 * 2);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{-1, 0};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> gold{2, 3, 0, 1};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(res_data, gold));
}

TEST_CASE(gathernd_negative_index_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 1, 1}};

    std::vector<float> data_vec(2 * 2);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{-3, 0};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

    mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
    p.compile(migraphx::make_target("ref"));

    EXPECT(test::throws([&] { p.eval({}); }));
}
