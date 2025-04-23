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
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/verify.hpp>

static void opt_pooling(migraphx::module& m)
{
    migraphx::rewrite_pooling rp;
    migraphx::dead_code_elimination dce;
    rp.apply(m);
    dce.apply(m);
}

TEST_CASE(rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {0, 0, 0}},
                                                          {"stride", {1, 1, 1}},
                                                          {"lengths", {3, 4, 5}},
                                                          {"dilations", {1, 1, 1}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::operation& reduce_op) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto rdm   = m.add_instruction(reduce_op, input);
        m.add_return({rdm});
        return m;
    };

    auto test_rewrite = [&](const migraphx::op::pooling_mode mode, const migraphx::operation& op) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(op);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    test_rewrite(migraphx::op::pooling_mode::average,
                 migraphx::make_op("reduce_mean", {{"axes", {2, 3, 4}}}));
    test_rewrite(migraphx::op::pooling_mode::max,
                 migraphx::make_op("reduce_max", {{"axes", {2, 3, 4}}}));
}

TEST_CASE(rewrite_pooling_dialtions_test)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 5, 5}};
    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {0, 0}},
                                                          {"stride", {1, 1}},
                                                          {"lengths", {2, 2}},
                                                          {"dilations", {2, 2}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        std::vector<int> indices{0, 2, 1, 3, 2, 4};
        migraphx::shape s_indices{migraphx::shape::int32_type, {indices.size()}};
        auto i1  = m.add_literal(migraphx::literal{s_indices, indices});
        auto g1  = m.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), input, i1);
        auto i2  = m.add_literal(migraphx::literal{s_indices, indices});
        auto g2  = m.add_instruction(migraphx::make_op("gather", {{"axis", 3}}), g1, i2);
        auto ret = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", mode},
                                                        {"padding", {0, 0}},
                                                        {"stride", {2, 2}},
                                                        {"lengths", {2, 2}},
                                                        {"dilations", {1, 1}}}),
                                     g2);
        m.add_return({ret});
        return m;
    };

    auto test_rewrite = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(mode);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    test_rewrite(migraphx::op::pooling_mode::average);
    test_rewrite(migraphx::op::pooling_mode::max);
}

TEST_CASE(rewrite_pooling_dialtions_test2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 5, 5, 5}};
    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {0, 0, 0}},
                                                          {"stride", {1, 1, 1}},
                                                          {"lengths", {2, 2, 2}},
                                                          {"dilations", {2, 2, 2}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        std::vector<int> indices{0, 2, 1, 3, 2, 4};
        migraphx::shape s_indices{migraphx::shape::int32_type, {indices.size()}};
        auto i1  = m.add_literal(migraphx::literal{s_indices, indices});
        auto g1  = m.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), input, i1);
        auto i2  = m.add_literal(migraphx::literal{s_indices, indices});
        auto g2  = m.add_instruction(migraphx::make_op("gather", {{"axis", 3}}), g1, i2);
        auto i3  = m.add_literal(migraphx::literal{s_indices, indices});
        auto g3  = m.add_instruction(migraphx::make_op("gather", {{"axis", 4}}), g2, i3);
        auto ret = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", mode},
                                                        {"padding", {0, 0, 0}},
                                                        {"stride", {2, 2, 2}},
                                                        {"lengths", {2, 2, 2}},
                                                        {"dilations", {1, 1, 1}}}),
                                     g3);
        m.add_return({ret});
        return m;
    };

    auto test_rewrite = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(mode);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    test_rewrite(migraphx::op::pooling_mode::average);
    test_rewrite(migraphx::op::pooling_mode::max);
}

TEST_CASE(rewrite_pooling_dialtions_test3)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret =
            m.add_instruction(migraphx::make_op("pooling",
                                                {{"mode", migraphx::op::pooling_mode::average},
                                                 {"padding", {1}},
                                                 {"stride", {1}},
                                                 {"lengths", {3}},
                                                 {"dilations", {2}}}),
                              input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_pooling_dialtions_test4)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 5, 5}};
    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {1, 0}},
                                                          {"stride", {1, 3}},
                                                          {"lengths", {3, 1}},
                                                          {"dilations", {1, 2}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        std::vector<int> col_indices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        migraphx::shape s_col_indices{migraphx::shape::int32_type, {col_indices.size()}};
        std::vector<int> row_indices{0, 3};
        migraphx::shape s_row_indices{migraphx::shape::int32_type, {row_indices.size()}};
        auto p =
            m.add_instruction(migraphx::make_op("pad",
                                                {{"pads", {0, 0, 1, 0, 0, 0, 1, 0}},
                                                 {"value", std::numeric_limits<float>::lowest()}}),
                              input);
        auto i1  = m.add_literal(migraphx::literal{s_col_indices, col_indices});
        auto g1  = m.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), p, i1);
        auto i2  = m.add_literal(migraphx::literal{s_row_indices, row_indices});
        auto g2  = m.add_instruction(migraphx::make_op("gather", {{"axis", 3}}), g1, i2);
        auto ret = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", mode},
                                                        {"padding", {0, 0}},
                                                        {"stride", {3, 1}},
                                                        {"lengths", {3, 1}},
                                                        {"dilations", {1, 1}}}),
                                     g2);
        m.add_return({ret});
        return m;
    };

    auto test_rewrite = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(mode);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    // Average won't work because of padding
    test_rewrite(migraphx::op::pooling_mode::max);
}

TEST_CASE(rewrite_pooling_dialtions_test5)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 5, 5}};
    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {0, 0}},
                                                          {"stride", {2, 3}},
                                                          {"lengths", {2, 1}},
                                                          {"dilations", {1, 2}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        std::vector<int> col_indices{0, 1, 2, 3};
        migraphx::shape s_col_indices{migraphx::shape::int32_type, {col_indices.size()}};
        std::vector<int> row_indices{0, 3};
        migraphx::shape s_row_indices{migraphx::shape::int32_type, {row_indices.size()}};
        auto i1  = m.add_literal(migraphx::literal{s_col_indices, col_indices});
        auto g1  = m.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), input, i1);
        auto i2  = m.add_literal(migraphx::literal{s_row_indices, row_indices});
        auto g2  = m.add_instruction(migraphx::make_op("gather", {{"axis", 3}}), g1, i2);
        auto ret = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", mode},
                                                        {"padding", {0, 0}},
                                                        {"stride", {2, 1}},
                                                        {"lengths", {2, 1}},
                                                        {"dilations", {1, 1}}}),
                                     g2);
        m.add_return({ret});
        return m;
    };

    auto test_rewrite = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(mode);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    test_rewrite(migraphx::op::pooling_mode::average);
    test_rewrite(migraphx::op::pooling_mode::max);
}

TEST_CASE(rewrite_avgpool_rank3_dil_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {2};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.35, 0.15, 0.85, 0.3, 0.1, 0.65};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_avgpool_rank3_dil_test2)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {3};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.2, 0.45, 0.35};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_avgpool_rank4_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2, 1};
    op.padding   = {0, 0};
    op.stride    = {2, 3};
    op.dilations = {1, 2};

    std::vector<float> data(25);
    std::iota(data.begin(), data.end(), 1);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{3.5, 6.5, 13.5, 16.5};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_maxpool_rank3_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {2};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.2, 0.9, 0.5, 0.1, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_maxpool_rank3_test2)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {1};
    op.stride    = {1};
    op.dilations = {3};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.3, 0.2, 0.9, 0.8, 0.5, 0.1, 0.6, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_maxpool_rank3_test3)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {3};
    op.padding   = {2};
    op.stride    = {2};
    op.dilations = {3};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.2, 0.5, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_maxpool_rank4_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {3, 1};
    op.padding   = {1, 0};
    op.stride    = {1, 3};
    op.dilations = {1, 2};

    std::vector<float> data(25);
    std::iota(data.begin(), data.end(), 1);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{6, 9, 11, 14, 16, 19, 21, 24, 21, 24};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank5_test)
{
    // 3D, input is 5D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2, 2, 2};
    op.padding   = {0, 0, 0};
    op.stride    = {1, 1, 1};
    op.dilations = {2, 2, 2};

    std::vector<float> data{
        -2.8029, 0.5861,  0.7015,  0.1297,  -1.44,   -1.9472, 0.7812,  2.408,   -0.3145, 0.3405,
        -0.9146, 0.0624,  1.5064,  -0.8345, 1.7977,  1.8949,  1.0073,  -0.2102, -0.042,  -0.7146,
        0.6227,  -0.5263, -2.2598, 0.1713,  0.449,   0.5303,  -0.8622, -0.5691, 0.907,   -0.0569,
        -1.5348, -0.4109, -0.1461, -0.5445, 0.4266,  0.2282,  1.3655,  -2.1519, 0.6068,  -0.2001,
        -0.4702, 0.3864,  1.7083,  0.9096,  0.4286,  -1.8866, 0.7034,  0.0293,  1.4587,  0.7672,
        -2.8614, 0.8124,  -0.053,  1.0449,  0.845,   -0.0131, 0.1139,  -0.859,  -1.2681, -0.6337,
        -0.4644, 0.1938,  0.2889,  0.9035,  0.7118,  -0.5767, 0.4577,  -0.0549, 0.2237,  0.5756,
        0.0677,  -0.0223, -0.329,  0.2364,  2.7666,  -0.7417, -1.3196, -0.2655, 0.1698,  -0.1777,
        -0.9427, 2.6859,  -0.7501, 0.5175,  1.0029,  -2.6436, -0.4388, -1.2348, -0.1539, -0.6229,
        -0.4136, 0.5085,  0.4136,  -0.6439, -1.1953, -0.406,  -0.0195, 0.1869,  -0.8664, 1.1364,
        0.5041,  0.0647,  0.1941,  -1.0819, -0.4629, -0.5107, 0.3612,  -0.3583};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.7812, 1.0449, 2.7666, 2.6859};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank5_test2)
{
    // 3D, input is 5D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2, 2, 2};
    op.padding   = {2, 2, 2};
    op.stride    = {2, 2, 2};
    op.dilations = {3, 3, 3};

    std::vector<float> data{
        -2.8029, 0.5861,  0.7015,  0.1297,  -1.44,   -1.9472, 0.7812,  2.408,   -0.3145, 0.3405,
        -0.9146, 0.0624,  1.5064,  -0.8345, 1.7977,  1.8949,  1.0073,  -0.2102, -0.042,  -0.7146,
        0.6227,  -0.5263, -2.2598, 0.1713,  0.449,   0.5303,  -0.8622, -0.5691, 0.907,   -0.0569,
        -1.5348, -0.4109, -0.1461, -0.5445, 0.4266,  0.2282,  1.3655,  -2.1519, 0.6068,  -0.2001,
        -0.4702, 0.3864,  1.7083,  0.9096,  0.4286,  -1.8866, 0.7034,  0.0293,  1.4587,  0.7672,
        -2.8614, 0.8124,  -0.053,  1.0449,  0.845,   -0.0131, 0.1139,  -0.859,  -1.2681, -0.6337,
        -0.4644, 0.1938,  0.2889,  0.9035,  0.7118,  -0.5767, 0.4577,  -0.0549, 0.2237,  0.5756,
        0.0677,  -0.0223, -0.329,  0.2364,  2.7666,  -0.7417, -1.3196, -0.2655, 0.1698,  -0.1777,
        -0.9427, 2.6859,  -0.7501, 0.5175,  1.0029,  -2.6436, -0.4388, -1.2348, -0.1539, -0.6229,
        -0.4136, 0.5085,  0.4136,  -0.6439, -1.1953, -0.406,  -0.0195, 0.1869,  -0.8664, 1.1364,
        0.5041,  0.0647,  0.1941,  -1.0819, -0.4629, -0.5107, 0.3612,  -0.3583};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    opt_pooling(*mm);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-0.8345, 1.5064,  -0.9146, 0.3405,  -1.44,   0.1297,  0.5861,  -2.8029,
                            -0.4702, -0.2001, -2.1519, 1.3655,  -0.4109, -1.5348, 0.907,   -0.5691,
                            -0.0549, 0.4577,  0.7118,  0.9035,  -1.2681, -0.859,  -0.0131, 0.845,
                            -1.1953, -0.6439, 0.5085,  -0.4136, -2.6436, 1.0029,  -0.7501, 2.6859};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_avepooling_na1_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret =
            m.add_instruction(migraphx::make_op("pooling",
                                                {{"mode", migraphx::op::pooling_mode::average},
                                                 {"padding", {0, 1, 0}},
                                                 {"stride", {1, 1, 1}},
                                                 {"lengths", {3, 4, 5}},
                                                 {"dilations", {1, 1, 1}}}),
                              input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_avepooling_na2_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret =
            m.add_instruction(migraphx::make_op("pooling",
                                                {{"mode", migraphx::op::pooling_mode::average},
                                                 {"padding", {0, 0, 0}},
                                                 {"stride", {1, 2, 1}},
                                                 {"lengths", {3, 4, 5}},
                                                 {"dilations", {1, 1, 1}}}),
                              input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_avepooling_na3_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", migraphx::op::pooling_mode::max},
                                                          {"padding", {0, 0, 0}},
                                                          {"stride", {1, 1, 1}},
                                                          {"lengths", {3, 3, 5}},
                                                          {"dilations", {1, 1, 1}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(literal_rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 1.0f);

    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::program p;

        auto* mm   = p.get_main_module();
        auto input = mm->add_literal(migraphx::literal(s, data));
        auto ret   = mm->add_instruction(migraphx::make_op("pooling",
                                                           {{"mode", mode},
                                                            {"padding", {0, 0, 0}},
                                                            {"stride", {1, 1, 1}},
                                                            {"lengths", {3, 4, 5}},
                                                            {"dilations", {1, 1, 1}}}),
                                       input);
        mm->add_return({ret});
        return p;
    };

    auto opt_program = [&](const migraphx::operation& op) {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto input = mm->add_literal(migraphx::literal(s, data));
        auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), input);
        auto rdm   = mm->add_instruction(op, rsp);
        auto ret =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 1, 1, 1}}}), rdm);
        mm->add_return({ret});

        return p;
    };

    auto test_rewrite_pooling = [&](const migraphx::op::pooling_mode mode,
                                    const migraphx::operation& op) {
        migraphx::program p1 = pooling_program(mode);
        migraphx::program p2 = opt_program(op);
        p1.compile(migraphx::make_target("ref"));
        p2.compile(migraphx::make_target("ref"));
        auto result1 = p1.eval({}).back();
        auto result2 = p2.eval({}).back();
        visit_all(result1, result2)(
            [&](auto r1, auto r2) { EXPECT(migraphx::verify::verify_rms_range(r1, r2)); });
    };

    test_rewrite_pooling(migraphx::op::pooling_mode::max,
                         migraphx::make_op("reduce_max", {{"axes", {1}}}));
    test_rewrite_pooling(migraphx::op::pooling_mode::average,
                         migraphx::make_op("reduce_mean", {{"axes", {1}}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
