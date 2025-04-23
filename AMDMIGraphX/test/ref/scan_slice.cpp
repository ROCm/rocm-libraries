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
#include <migraphx/compile_options.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

static migraphx::shape make_shape(const std::vector<size_t>& lens)
{
    return migraphx::shape{migraphx::shape::int32_type, lens};
}

static migraphx::program make_scan_slice_program(int64_t axis, int64_t direction)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape data_sh{migraphx::shape::int32_type, {2, 2, 2}};
    std::vector<int> data(data_sh.elements());
    std::iota(data.begin(), data.end(), 0);
    auto data_lit = mm->add_literal(migraphx::literal{data_sh, data});

    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    auto idx_param = mm->add_parameter("idx", idx_sh);

    mm->add_instruction(migraphx::make_op("scan_slice", {{"axis", axis}, {"direction", direction}}),
                        data_lit,
                        idx_param);

    p.compile(migraphx::make_target("ref"));

    return p;
}

TEST_CASE(scan_slice_test_1)
{
    auto p = make_scan_slice_program(0, 0);

    migraphx::parameter_map pm;
    int64_t idx = 0;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"]   = migraphx::argument{idx_sh, &idx};
    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({1, 2, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{0, 1, 2, 3});

    idx    = 1;
    result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({1, 2, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{4, 5, 6, 7});
}

TEST_CASE(scan_slice_test_2)
{
    auto p = make_scan_slice_program(1, 0);

    migraphx::parameter_map pm;
    int64_t idx = 0;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"]   = migraphx::argument{idx_sh, &idx};
    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({2, 1, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{0, 1, 4, 5});

    idx    = 1;
    result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({2, 1, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{2, 3, 6, 7});
}

TEST_CASE(scan_slice_test_3)
{
    auto p = make_scan_slice_program(2, 0);

    migraphx::parameter_map pm;
    int64_t idx = 0;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"]   = migraphx::argument{idx_sh, &idx};
    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({2, 2, 1}));
    EXPECT(result.to_vector<int>() == std::vector<int>{0, 2, 4, 6});

    idx    = 1;
    result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({2, 2, 1}));
    EXPECT(result.to_vector<int>() == std::vector<int>{1, 3, 5, 7});
}

TEST_CASE(scan_slice_test_4)
{
    auto p = make_scan_slice_program(-3, 0);

    migraphx::parameter_map pm;
    int64_t idx = 0;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"]   = migraphx::argument{idx_sh, &idx};
    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({1, 2, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{0, 1, 2, 3});

    idx    = 1;
    result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({1, 2, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{4, 5, 6, 7});
}

TEST_CASE(scan_slice_test_5)
{
    auto p = make_scan_slice_program(0, 1);

    migraphx::parameter_map pm;
    int64_t idx = 0;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"]   = migraphx::argument{idx_sh, &idx};
    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({1, 2, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{4, 5, 6, 7});

    idx    = 1;
    result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({1, 2, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE(scan_slice_test_6)
{
    auto p = make_scan_slice_program(-2, 1);

    migraphx::parameter_map pm;
    int64_t idx = 0;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"]   = migraphx::argument{idx_sh, &idx};
    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({2, 1, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{2, 3, 6, 7});

    idx    = 1;
    result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape({2, 1, 2}));
    EXPECT(result.to_vector<int>() == std::vector<int>{0, 1, 4, 5});
}

TEST_CASE(scan_slice_test_7)
{
    auto p = make_scan_slice_program(0, 0);

    migraphx::parameter_map pm;
    int64_t idx = 2;
    migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
    pm["idx"] = migraphx::argument{idx_sh, &idx};

    EXPECT(test::throws([&] { p.eval(pm); }));
}
