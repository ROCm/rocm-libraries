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
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/fp8_ocp_to_fnuz.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>

#include <test.hpp>
#include <quantize_helpers.hpp>

/**
 * test that before and after the fp8_ocp_to_fnuz pass
 * have equivalent results
 */

static void run_fp8_ocp_to_fnuz(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::fp8_ocp_to_fnuz{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(fp8_ocp_to_fnuz_gemm)
{
    using migraphx::fp8::fp8e4m3fn;
    using migraphx::fp8::fp8e4m3fnuz;
    std::vector<std::size_t> data_lens = {2, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, data_lens};

    migraphx::program p1;
    auto* m1 = p1.get_main_module();
    {
        auto a     = m1->add_parameter("a", data_shape);
        auto b     = m1->add_parameter("b", data_shape);
        auto scale = m1->add_literal(0.5f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            m1->add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(*m1, "quantizelinear", a, scale, zero);
        auto qb = add_quantize_op(*m1, "quantizelinear", b, scale, zero);
        auto da =
            add_quantize_op(*m1, "dequantizelinear", qa, qa->inputs().at(1), qa->inputs().at(2));
        auto db =
            add_quantize_op(*m1, "dequantizelinear", qb, qb->inputs().at(1), qb->inputs().at(2));
        auto dot = m1->add_instruction(migraphx::make_op("dot"), da, db);
        m1->add_return({dot});
    }

    migraphx::program p2 = p1;
    migraphx::module* m2 = p2.get_main_module();
    run_fp8_ocp_to_fnuz(*m2);

    p1.compile(migraphx::make_target("ref"));
    p2.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<float> a_data = {20, -100, 100, 0.25};
    std::vector<float> b_data = {28, 0.125, 2.5, 0.25};
    params["a"]               = migraphx::argument(data_shape, a_data.data());
    params["b"]               = migraphx::argument(data_shape, b_data.data());

    auto result_1 = p1.eval({params}).back();
    auto result_2 = p2.eval({params}).back();
    std::vector<float> results_vector_1(4);
    std::vector<float> results_vector_2(4);
    result_1.visit([&](auto output) { results_vector_1.assign(output.begin(), output.end()); });
    result_2.visit([&](auto output) { results_vector_2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector_1, results_vector_2));
}

TEST_CASE(fp8_ocp_to_fnuz_gemm_multi_scale)
{
    using migraphx::fp8::fp8e4m3fn;
    using migraphx::fp8::fp8e4m3fnuz;
    std::vector<std::size_t> data_lens = {3, 3};
    migraphx::shape data_shape{migraphx::shape::float_type, data_lens};
    migraphx::shape scales_shape{migraphx::shape::float_type, {3}};

    migraphx::program p1;
    auto* m1 = p1.get_main_module();
    {
        auto a      = m1->add_parameter("a", data_shape);
        auto b      = m1->add_parameter("b", data_shape);
        auto scale1 = m1->add_literal(migraphx::generate_literal(scales_shape, 0));
        auto scale2 = m1->add_literal(0.4f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            m1->add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(*m1, "quantizelinear", a, scale1, zero);
        auto qb = add_quantize_op(*m1, "quantizelinear", b, scale2, zero);
        auto da =
            add_quantize_op(*m1, "dequantizelinear", qa, qa->inputs().at(1), qa->inputs().at(2));
        auto db =
            add_quantize_op(*m1, "dequantizelinear", qb, qb->inputs().at(1), qb->inputs().at(2));
        auto dot = m1->add_instruction(migraphx::make_op("dot"), da, db);
        m1->add_return({dot});
    }

    migraphx::program p2 = p1;
    migraphx::module* m2 = p2.get_main_module();
    run_fp8_ocp_to_fnuz(*m2);

    p1.compile(migraphx::make_target("ref"));
    p2.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<float> a_data = {20, -100, 100, 0.25, 0.3, 3.3, 5.0, -8.0, 63.0};
    std::vector<float> b_data = {28, 0.125, 2.5, 0.25, 0.0582, -187, 0.716, 8.12, 1.87};
    params["a"]               = migraphx::argument(data_shape, a_data.data());
    params["b"]               = migraphx::argument(data_shape, b_data.data());

    auto result_1 = p1.eval({params}).back();
    auto result_2 = p2.eval({params}).back();
    std::vector<float> results_vector_1(9);
    std::vector<float> results_vector_2(9);
    result_1.visit([&](auto output) { results_vector_1.assign(output.begin(), output.end()); });
    result_2.visit([&](auto output) { results_vector_2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector_1, results_vector_2));
}

TEST_CASE(fp8_ocp_to_fnuz_conv)
{
    using migraphx::fp8::fp8e4m3fn;
    using migraphx::fp8::fp8e4m3fnuz;
    std::vector<std::size_t> data_lens = {2, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, data_lens};

    migraphx::program p1;
    auto* m1 = p1.get_main_module();
    {
        std::vector<float> a_data = {
            2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,
            0.80927712,  -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929,
            0.67726439,  -0.65290606, 0.02345525,  -0.33579525, 0.38901961,  1.05473483,
            -1.31188095, 1.8963089,   -0.07265259, 0.947339,    0.41949373,  -0.70814759,
            0.25892952,  1.07311416,  1.2571274,   -0.62318051, -0.19951548, -0.94232577,
            -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,  0.13900366,
            1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
            0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559,
            -0.03024297, 1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934,
            0.86956722,  -0.40457946, 0.46691212,  1.29273605,  0.26464137,  0.22073045,
            -1.02178168, 0.22163901,  -1.84387338, 0.75522131,  -0.45775682, -0.42241111,
            -1.50944722, 1.07256448,  -1.95876884, -0.28106022, 0.3341668,   2.13129425,
            -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792, -2.06007552,
            0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
            0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932,
            -0.68230027, -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

        std::vector<float> b_data = {
            2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
            9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
            -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
            3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
            1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
            2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
            -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
            4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
            1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
            1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
            -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

        migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
        auto a = m1->add_literal(migraphx::literal{a_shape, a_data});

        migraphx::shape b_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
        auto b     = m1->add_literal(migraphx::literal{b_shape, b_data});
        auto scale = m1->add_literal(0.5f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            m1->add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(*m1, "quantizelinear", a, scale, zero);
        auto qb = add_quantize_op(*m1, "quantizelinear", b, scale, zero);
        auto da =
            add_quantize_op(*m1, "dequantizelinear", qa, qa->inputs().at(1), qa->inputs().at(2));
        auto db =
            add_quantize_op(*m1, "dequantizelinear", qb, qb->inputs().at(1), qb->inputs().at(2));
        auto conv_ins = m1->add_instruction(migraphx::make_op("convolution"), da, db);
        m1->add_return({conv_ins});
    }

    migraphx::program p2 = p1;
    migraphx::module* m2 = p2.get_main_module();
    run_fp8_ocp_to_fnuz(*m2);

    p1.compile(migraphx::make_target("ref"));
    p2.compile(migraphx::make_target("ref"));

    auto result_1 = p1.eval({}).back();
    auto result_2 = p2.eval({}).back();
    std::vector<float> results_vector_1(16);
    std::vector<float> results_vector_2(16);
    result_1.visit([&](auto output) { results_vector_1.assign(output.begin(), output.end()); });
    result_2.visit([&](auto output) { results_vector_2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector_1, results_vector_2));
}
