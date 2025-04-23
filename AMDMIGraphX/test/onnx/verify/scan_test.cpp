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

#include <migraphx/argument.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/module.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

static migraphx::shape make_shape(const std::vector<size_t>& lens)
{
    return migraphx::shape{migraphx::shape::float_type, lens};
}

static auto
scan_test(const std::string& test_file, migraphx::shape scan_ins1_sh, migraphx::shape scan_ins2_sh)
{
    auto prog = read_onnx(test_file);
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(init_state_sh.elements(), 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    std::vector<float> scan_ins1(scan_ins1_sh.elements());
    std::iota(scan_ins1.begin(), scan_ins1.end(), 1);
    pm["scan_ins1"] = migraphx::argument(scan_ins1_sh, scan_ins1.data());

    std::vector<float> scan_ins2(scan_ins2_sh.elements());
    std::iota(scan_ins2.begin(), scan_ins2.end(), 0);
    pm["scan_ins2"] = migraphx::argument(scan_ins2_sh, scan_ins2.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 3);
    return std::make_tuple(result[0], result[1], result[2]);
}

TEST_CASE(scan_test1)
{
    auto [final_state, scan_out1, scan_out2] =
        scan_test("scan_test1.onnx", make_shape({3, 2, 2}), make_shape({3, 1}));

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{18, 21, 24, 27};
    EXPECT(final_state.to_vector<float>() == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{1, 2, 3, 4, 7, 9, 11, 13, 18, 21, 24, 27};
    EXPECT(scan_out1.to_vector<float>() == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{4, 6, 18, 22, 42, 48};
    EXPECT(scan_out2.to_vector<float>() == scan_out2_gold);
}

TEST_CASE(scan_test2)
{
    auto [final_state, scan_out1, scan_out2] =
        scan_test("scan_test2.onnx", make_shape({3, 2, 2}), make_shape({3, 1}));

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{18, 21, 24, 27};
    EXPECT(final_state.to_vector<float>() == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{18, 21, 24, 27, 7, 9, 11, 13, 1, 2, 3, 4};
    EXPECT(scan_out1.to_vector<float>() == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{4, 6, 18, 22, 42, 48};
    EXPECT(scan_out2.to_vector<float>() == scan_out2_gold);
}

TEST_CASE(scan_test3)
{
    auto [final_state, scan_out1, scan_out2] =
        scan_test("scan_test3.onnx", make_shape({3, 2, 2}), make_shape({3, 1}));

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{18, 21, 24, 27};
    EXPECT(final_state.to_vector<float>() == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({2, 3, 2}));
    std::vector<float> scan_out1_gold{1, 2, 7, 9, 18, 21, 3, 4, 11, 13, 24, 27};
    EXPECT(scan_out1.to_vector<float>() == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({2, 3}));
    std::vector<float> scan_out2_gold{4, 18, 42, 6, 22, 48};
    EXPECT(scan_out2.to_vector<float>() == scan_out2_gold);
}

TEST_CASE(scan_test4)
{
    auto [final_state, scan_out1, scan_out2] =
        scan_test("scan_test4.onnx", make_shape({3, 2, 2}), make_shape({3, 1}));

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{18, 21, 24, 27};
    EXPECT(final_state.to_vector<float>() == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{9, 10, 11, 12, 15, 17, 19, 21, 18, 21, 24, 27};
    EXPECT(scan_out1.to_vector<float>() == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{20, 22, 34, 38, 42, 48};
    EXPECT(scan_out2.to_vector<float>() == scan_out2_gold);
}

TEST_CASE(scan_test5)
{
    auto [final_state, scan_out1, scan_out2] =
        scan_test("scan_test5.onnx", make_shape({2, 2, 3}), make_shape({1, 3}));

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{9, 18, 27, 36};
    EXPECT(final_state.to_vector<float>() == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{1, 4, 7, 10, 4, 10, 16, 22, 9, 18, 27, 36};
    EXPECT(scan_out1.to_vector<float>() == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{8, 14, 20, 32, 36, 54};
    EXPECT(scan_out2.to_vector<float>() == scan_out2_gold);
}

TEST_CASE(scan_test6)
{
    auto [final_state, scan_out1, scan_out2] =
        scan_test("scan_test6.onnx", make_shape({2, 3, 2}), make_shape({3, 1}));

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{12, 15, 30, 33};
    EXPECT(final_state.to_vector<float>() == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({2, 2, 3}));
    std::vector<float> scan_out1_gold{12, 7, 3, 15, 9, 4, 30, 19, 9, 33, 21, 10};
    EXPECT(scan_out1.to_vector<float>() == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({2, 3}));
    std::vector<float> scan_out2_gold{42, 26, 12, 48, 30, 14};
    EXPECT(scan_out2.to_vector<float>() == scan_out2_gold);
}

TEST_CASE(scan_test7)
{
    auto prog = read_onnx("scan_test7.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(init_state_sh.elements(), 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {3, 2, 2}};
    std::vector<float> scan_ins(scan_ins_sh.elements());
    std::iota(scan_ins.begin(), scan_ins.end(), 1);
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 2);

    EXPECT(result[0].get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{30, 36, 42, 48};
    EXPECT(result[0].to_vector<float>() == final_state_gold);

    EXPECT(result[1].get_shape() == make_shape({3, 2, 2}));
    EXPECT(result[1].to_vector<float>() == scan_ins);
}
