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

TEST_CASE(scatternd_shapes_test_1)
{
    // broadcasted input
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape us{dtype, {4}};

    std::vector<int64_t> ind_vec{4, 3, 1, 7};
    std::vector<float> upd_vec{9, 10, 11, 12};

    auto data    = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}),
                                    mm->add_literal(migraphx::literal{0.0f}));
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 11, 0, 10, 9, 0, 0, 12};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_shapes_test_2)
{
    // non-standard shape input
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 2}};
    migraphx::shape is{itype, {2, 2}};
    migraphx::shape us{dtype, {2}};

    std::vector<float> data_vec{1, 2, 3, 4};
    std::vector<int64_t> ind_vec{0, 0, 0, 1};
    std::vector<float> upd_vec{5, 6};

    auto data = mm->add_literal(migraphx::literal{ds, data_vec});
    auto td = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), data);
    auto indices   = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates   = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd = mm->add_instruction(migraphx::make_op("scatternd_none"), td, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{5, 6, 2, 4};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_shapes_test_3)
{
    // non-standard updates shape
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 2, 2}};
    migraphx::shape is{itype, {2, 1, 3}};
    migraphx::shape us{dtype, {1, 2}};

    std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ind_vec{0, 0, 0, 1, 1, 1};
    std::vector<float> upd_vec{9, 10};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto tu =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), updates);
    auto scatternd = mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, tu);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{9, 2, 3, 4, 5, 6, 7, 10};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_test_1)
{
    // r=1, q=2, k=1
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {8}};
    migraphx::shape is{itype, {4, 1}};
    migraphx::shape us{dtype, {4}};

    std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ind_vec{4, 3, 1, 7};
    std::vector<float> upd_vec{9, 10, 11, 12};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 11, 3, 10, 9, 6, 7, 12};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_test_2)
{
    // r=2, q=2, k=2
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 2}};
    migraphx::shape is{itype, {2, 2}};
    migraphx::shape us{dtype, {2}};

    std::vector<float> data_vec{1, 2, 3, 4};
    std::vector<int64_t> ind_vec{0, 0, 0, 1};
    std::vector<float> upd_vec{5, 6};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{5, 6, 3, 4};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_test_3)
{
    // r=3, q=3, k=3
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 2, 2}};
    migraphx::shape is{itype, {2, 1, 3}};
    migraphx::shape us{dtype, {2, 1}};

    std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ind_vec{0, 0, 0, 1, 1, 1};
    std::vector<float> upd_vec{9, 10};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{9, 2, 3, 4, 5, 6, 7, 10};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_test_4)
{
    // r=3, q=2, k=1
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {4, 4, 4}};
    migraphx::shape is{itype, {2, 1}};
    migraphx::shape us{dtype, {2, 4, 4}};

    std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6,
                                7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4,
                                5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ind_vec{0, 2};
    std::vector<float> upd_vec{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                               1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 1, 2, 3, 4, 5, 6,
                            7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                            4, 4, 4, 4, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatternd_test_5)
{
    // r=5, q=1, k=1
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape ds{dtype, {2, 2, 2, 2, 2}};
    migraphx::shape is{itype, {1}};
    migraphx::shape us{dtype, {2, 2, 2, 2}};

    std::vector<float> data_vec(32, 1);
    std::vector<int64_t> ind_vec{1};
    std::vector<float> upd_vec(16, 0);

    auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
    auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
    auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(32, 0);
    std::copy(data_vec.begin(), data_vec.begin() + 16, gold.begin());

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
