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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(if_tuple_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = read_onnx("if_tuple_test.onnx");
        p.compile(migraphx::make_target("ref"));
        migraphx::shape xs{migraphx::shape::float_type, {1, 4}};
        migraphx::shape ys{migraphx::shape::float_type, {3, 4}};
        migraphx::shape cond_s{migraphx::shape::bool_type};

        std::vector<float> x_data(xs.elements(), 1.0f);
        std::vector<float> y_data(ys.elements(), 2.0f);
        std::vector<char> cond_data{static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["x"]    = migraphx::argument(xs, x_data.data());
        pp["y"]    = migraphx::argument(ys, y_data.data());
        pp["cond"] = migraphx::argument(cond_s, cond_data.data());

        auto results = p.eval(pp);
        std::vector<std::vector<float>> rets;
        for(const auto& arg : results)
        {
            std::vector<float> vec;
            arg.visit([&](auto output) { vec.assign(output.begin(), output.end()); });
            rets.push_back(vec);
        }

        return rets;
    };

    // then branch
    {
        auto results = run_prog(true);
        std::vector<float> gold0(4, 2.0f);
        std::vector<float> gold1(12, 4.0f);
        EXPECT(migraphx::verify::verify_rms_range(results.at(0), gold0));
        EXPECT(migraphx::verify::verify_rms_range(results.at(1), gold1));
    }

    // else branch
    {
        auto results = run_prog(false);
        std::vector<float> gold0(4, 3.0f);
        std::vector<float> gold1(12, 5.0f);
        EXPECT(migraphx::verify::verify_rms_range(results.at(0), gold0));
        EXPECT(migraphx::verify::verify_rms_range(results.at(1), gold1));
    }
}
