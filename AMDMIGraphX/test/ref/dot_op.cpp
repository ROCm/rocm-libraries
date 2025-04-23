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
#include <migraphx/apply_alpha_beta.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include "test.hpp"

template <class T>
static void dot_2d_test()
{
    migraphx::program p;

    auto* mm         = p.get_main_module();
    std::vector<T> a = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                        1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                        -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                        -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<T> b = {6.09568541e-01,
                        -6.10527007e-01,
                        3.66646462e-01,
                        1.18951101e-01,
                        5.58777432e-01,
                        -3.21296298e-01,
                        -5.95997198e-01,
                        -5.01425721e-01,
                        -2.84606807e-01,
                        -5.73673557e-01,
                        -8.99430260e-01,
                        -4.25103093e-01,
                        1.53027987e+00,
                        -3.81407415e-04,
                        -3.29650255e-01};

    migraphx::shape a_shape{migraphx::shape::get_type<T>{}, {4, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::get_type<T>{}, {5, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    mm->add_instruction(migraphx::make_op("dot"), al, bl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<T> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<T> gold = {-1.56327541e+00,
                           -7.09570140e-01,
                           -5.37424982e-01,
                           -2.22994831e-01,
                           -2.15586437e+00,
                           2.09177941e-03,
                           -1.47279677e+00,
                           2.02627040e-01,
                           -6.04527691e-01,
                           -1.29885596e+00,
                           2.16294914e+00,
                           -1.48101497e-01};
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        results_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{9e-6}));
}
TEST_CASE_REGISTER(dot_2d_test<float>)
TEST_CASE_REGISTER(dot_2d_test<double>)

template <class T>
static void dot_4d_test()
{
    migraphx::program p;

    auto* mm            = p.get_main_module();
    std::vector<T> a    = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                           1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                           -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                           -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<T> b    = {6.09568541e-01,
                           -6.10527007e-01,
                           3.66646462e-01,
                           1.18951101e-01,
                           5.58777432e-01,
                           -3.21296298e-01,
                           -5.95997198e-01,
                           -5.01425721e-01,
                           -2.84606807e-01,
                           -5.73673557e-01,
                           -8.99430260e-01,
                           -4.25103093e-01,
                           1.53027987e+00,
                           -3.81407415e-04,
                           -3.29650255e-01};
    std::vector<T> gold = {-1.56327541e+00,
                           -7.09570140e-01,
                           -5.37424982e-01,
                           -2.22994831e-01,
                           -2.15586437e+00,
                           2.09177941e-03,
                           -1.47279677e+00,
                           2.02627040e-01,
                           -6.04527691e-01,
                           -1.29885596e+00,
                           2.16294914e+00,
                           -1.48101497e-01};
    migraphx::shape a_shape{migraphx::shape::get_type<T>{}, {1, 1, 4, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::get_type<T>{}, {1, 1, 5, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    mm->add_instruction(migraphx::make_op("dot"), al, bl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<T> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        results_vector, migraphx::verify::expected{gold}, migraphx::verify::tolerance{9e-6}));
}

TEST_CASE_REGISTER(dot_4d_test<float>)
TEST_CASE_REGISTER(dot_4d_test<double>)

TEST_CASE(dot_3_d_test)
{
    migraphx::program p;

    auto* mm              = p.get_main_module();
    std::vector<float> m1 = {-0.76234141,
                             0.01368910,
                             -0.86343423,
                             -0.99465282,
                             0.76133268,
                             0.96507140,
                             -0.55893585,
                             0.02625652,
                             0.75171776,
                             0.23112578,
                             0.25624787,
                             -1.50442161};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> m2 = {-0.15933632, -0.69594712, -0.06198966, -1.23905184, -0.83672704,
                             -1.06971832, -0.12272917, 1.07094116,  -0.08346820, 1.16820693,
                             -0.95700874, 0.24059691,  0.43326023,  0.78305235,  -0.53506601,
                             -0.69359678, -0.26334436, 1.56292796,  -0.33629175, -1.72693469,
                             0.41435494,  1.52136843,  -0.40699791, -1.59839430};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto l1 = mm->add_literal(migraphx::literal{m1_shape, m1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, m2});

    mm->add_instruction(migraphx::make_op("dot"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.18208394,
                               -0.49276402,
                               0.87189133,
                               0.75150114,
                               -0.55909610,
                               1.00521735,
                               -0.95536130,
                               2.27996211,
                               0.06239879,
                               0.74700068,
                               -0.01570983,
                               -0.85920856,
                               -0.59070835,
                               -1.70729902,
                               0.40245487,
                               1.80182751};

    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_3_d_c_test0)
{
    migraphx::program p;

    auto* mm              = p.get_main_module();
    std::vector<float> m1 = {-0.76234141,
                             0.01368910,
                             -0.86343423,
                             -0.99465282,
                             0.76133268,
                             0.96507140,
                             -0.55893585,
                             0.02625652,
                             0.75171776,
                             0.23112578,
                             0.25624787,
                             -1.50442161};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> m2 = {-0.15933632, -0.69594712, -0.06198966, -1.23905184, -0.83672704,
                             -1.06971832, -0.12272917, 1.07094116,  -0.08346820, 1.16820693,
                             -0.95700874, 0.24059691,  0.43326023,  0.78305235,  -0.53506601,
                             -0.69359678, -0.26334436, 1.56292796,  -0.33629175, -1.72693469,
                             0.41435494,  1.52136843,  -0.40699791, -1.59839430};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> m3 = {0.18208394,
                             -0.49276402,
                             0.87189133,
                             0.75150114,
                             -0.55909610,
                             1.00521735,
                             -0.95536130,
                             2.27996211,
                             0.06239879,
                             0.74700068,
                             -0.01570983,
                             -0.85920856,
                             -0.59070835,
                             -1.70729902,
                             0.40245487,
                             1.80182751};
    migraphx::shape m3_shape{migraphx::shape::float_type, {2, 2, 4}};
    auto l1     = mm->add_literal(migraphx::literal{m1_shape, m1});
    auto l2     = mm->add_literal(migraphx::literal{m2_shape, m2});
    auto l3     = mm->add_literal(migraphx::literal{m3_shape, m3});
    float alpha = 1.0f;
    float beta  = 0.0f;
    migraphx::add_apply_alpha_beta(*mm,
                                   std::vector<migraphx::instruction_ref>{l1, l2, l3},
                                   migraphx::make_op("dot"),
                                   alpha,
                                   beta);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.18208394,
                               -0.49276402,
                               0.87189133,
                               0.75150114,
                               -0.55909610,
                               1.00521735,
                               -0.95536130,
                               2.27996211,
                               0.06239879,
                               0.74700068,
                               -0.01570983,
                               -0.85920856,
                               -0.59070835,
                               -1.70729902,
                               0.40245487,
                               1.80182751};

    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_3_d_c_test1)
{
    migraphx::program p;

    auto* mm              = p.get_main_module();
    std::vector<float> m1 = {
        -0.76234141, 0.01368910, -0.86343423, -0.99465282, 0.76133268, 0.96507140};
    migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
    std::vector<float> m2 = {-0.15933632,
                             -0.69594712,
                             -0.06198966,
                             -1.23905184,
                             -0.83672704,
                             -1.06971832,
                             -0.12272917,
                             1.07094116,
                             -0.08346820,
                             1.16820693,
                             -0.95700874,
                             0.24059691};
    migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};

    migraphx::shape m3_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> m3 = {0.18208394,
                             -0.49276402,
                             0.87189133,
                             0.75150114,
                             -0.55909610,
                             1.00521735,
                             -0.95536130,
                             2.27996211};
    auto l1               = mm->add_literal(migraphx::literal{m1_shape, m1});
    auto l2               = mm->add_literal(migraphx::literal{m2_shape, m2});
    auto l3               = mm->add_literal(migraphx::literal{m3_shape, m3});

    float alpha = 1.0f;
    float beta  = 0.0f;
    migraphx::add_apply_alpha_beta(*mm,
                                   std::vector<migraphx::instruction_ref>{l1, l2, l3},
                                   migraphx::make_op("dot"),
                                   alpha,
                                   beta);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.18208394,
                               -0.49276402,
                               0.87189133,
                               0.75150114,
                               -0.55909610,
                               1.00521735,
                               -0.95536130,
                               2.27996211};

    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_4_d_test1)
{
    migraphx::program p;

    auto* mm              = p.get_main_module();
    std::vector<float> m1 = {
        -1.93300070, 0.33902698,  -0.45173527, -0.72283069, -0.17177134, 1.62199882,
        0.87052847,  0.14989811,  -0.88969184, -0.18131398, 0.72654339,  -0.57123693,
        0.03852506,  -0.72332085, -1.81844083, -0.33465167, -0.71400352, 0.36883161,
        0.08698452,  0.94974586,  0.40087323,  -0.05448534, 0.03220677,  -1.22494296,
        0.97938472,  -1.43714454, -0.80430904, -0.08098728, 0.31520301,  0.49642169,
        -1.63471091, 0.34390096,  2.81292176,  -0.22666528, 1.54559556,  -1.51075762};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
    std::vector<float> m2 = {
        -0.33170529, 2.26325120,  -0.50639461, 0.64802947,  0.44748888,  0.33768068,
        -0.53621075, 0.34341460,  0.58742520,  -1.13995790, -0.99322535, 0.35447353,
        0.01977110,  -0.10155016, -1.02288245, -0.16575791, -1.47870374, 0.29300008,
        -0.39112198, 1.42303608,  -0.02853060, 1.52610164,  0.53540909,  0.75618998,
        -0.26877787, -1.90886366, 0.30622790,  0.59794535,  1.29795331,  -0.37805803,
        -1.58167176, -1.26966832, 0.27435891,  0.89430347,  0.22854926,  -0.50317658};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
    auto l1 = mm->add_literal(migraphx::literal{m1_shape, m1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, m2});

    mm->add_instruction(migraphx::make_op("dot"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.26735861,  -4.30770895, 1.05257728,  -1.19954265, 0.50493170,
                               -0.18729756, 1.09137941,  -1.09298312, 3.42956915,  -0.41681939,
                               0.17833257,  0.26040336,  0.15351280,  1.87632715,  -0.63545406,
                               -0.95467340, -1.74728628, -2.42477030, 0.76262372,  0.15539164,
                               3.32281958,  0.96769613,  0.43727545,  2.43019906};

    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_4_d_alpha_beta_test)
{
    migraphx::program p;

    auto* mm              = p.get_main_module();
    std::vector<float> m1 = {
        1.23636469,  -0.47041261, -0.14375651, -0.48371852, 1.16479301,  -0.89361055,
        -0.18569086, 1.10700457,  -1.02632638, 0.82277012,  0.33525769,  0.52825145,
        -1.00141689, 0.45510090,  -0.02675039, -0.60454439, 0.38551153,  -0.01658514,
        0.93059292,  -0.54595188, -0.04911005, -0.91397221, -0.83127477, -1.57685603,
        -1.36200452, 2.25822236,  -1.23416970, 0.12312496,  0.76232760,  -0.83594234,
        1.67418145,  -0.19412936, 1.05261378,  0.66246074,  -1.15233398, 0.16429736};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
    std::vector<float> m2 = {
        -0.87300530, -0.07112838, 0.19196860,  -1.04986840, 1.20348200,  0.31966893,
        1.04805440,  -2.04777729, -0.67906052, -1.17250760, 0.34305044,  -1.01957785,
        -1.12694862, 0.18431338,  -1.63712290, 0.27566931,  -1.11282021, 1.41738919,
        0.47871283,  -1.01980420, 1.00212436,  -0.78740444, -1.65636133, 1.51466547,
        -0.12470397, 0.70404393,  -0.15244797, 0.74288871,  0.07339926,  -1.45811623,
        0.27185845,  0.08804596,  0.99061977,  -1.61752428, 0.29191159,  0.87271953};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
    std::vector<float> m3 = {-1.07692443, 0.85223457,  -0.37266530, 2.31511577,  0.04227017,
                             1.13229428,  -0.52769242, 0.27307182,  -0.47779843, -0.08023168,
                             -0.22862823, 0.81489871,  1.13139581,  1.13860467,  0.24309065,
                             0.26533729,  0.49106772,  -1.18860493, 0.27842449,  1.03568141,
                             0.49759611,  0.10021662,  0.00592602,  0.90862000};
    migraphx::shape m3_shape{migraphx::shape::float_type, {2, 3, 2, 2}};

    auto l1        = mm->add_literal(migraphx::literal{m1_shape, m1});
    auto l2        = mm->add_literal(migraphx::literal{m2_shape, m2});
    auto l3        = mm->add_literal(migraphx::literal{m3_shape, m3});
    float alpha    = 0.35;
    float beta     = 0.41;
    auto m12_alpha = migraphx::add_apply_alpha_beta(
        *mm, std::vector<migraphx::instruction_ref>{l1, l2}, migraphx::make_op("dot"), alpha);
    auto l_beta = mm->add_literal(beta);
    auto b_beta = mm->add_instruction(
        migraphx::make_op("scalar", {{"scalar_bcst_dims", m12_alpha->get_shape().lens()}}), l_beta);
    auto m3_beta = mm->add_instruction(migraphx::make_op("mul"), b_beta, l3);
    mm->add_instruction(migraphx::make_op("add"), m3_beta, m12_alpha);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-0.91147203, 0.47540785, -0.30313587, 0.43325099,  -0.43711586,
                               0.50928632,  0.06919868, -0.80382802, -0.05125718, -0.06685650,
                               -0.06972163, 0.32407764, 0.45677396,  0.25909489,  0.56911252,
                               -0.17183724, 0.10858734, 0.39406289,  0.04662959,  1.07979824,
                               0.40355016,  0.52410648, -0.31728447, 1.09550845};

    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_4_d_alpha_beta_c_test)
{
    migraphx::program p;

    auto* mm              = p.get_main_module();
    std::vector<float> m1 = {
        1.23636469,  -0.47041261, -0.14375651, -0.48371852, 1.16479301,  -0.89361055,
        -0.18569086, 1.10700457,  -1.02632638, 0.82277012,  0.33525769,  0.52825145,
        -1.00141689, 0.45510090,  -0.02675039, -0.60454439, 0.38551153,  -0.01658514,
        0.93059292,  -0.54595188, -0.04911005, -0.91397221, -0.83127477, -1.57685603,
        -1.36200452, 2.25822236,  -1.23416970, 0.12312496,  0.76232760,  -0.83594234,
        1.67418145,  -0.19412936, 1.05261378,  0.66246074,  -1.15233398, 0.16429736};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
    std::vector<float> m2 = {
        -0.87300530, -0.07112838, 0.19196860,  -1.04986840, 1.20348200,  0.31966893,
        1.04805440,  -2.04777729, -0.67906052, -1.17250760, 0.34305044,  -1.01957785,
        -1.12694862, 0.18431338,  -1.63712290, 0.27566931,  -1.11282021, 1.41738919,
        0.47871283,  -1.01980420, 1.00212436,  -0.78740444, -1.65636133, 1.51466547,
        -0.12470397, 0.70404393,  -0.15244797, 0.74288871,  0.07339926,  -1.45811623,
        0.27185845,  0.08804596,  0.99061977,  -1.61752428, 0.29191159,  0.87271953};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
    std::vector<float> m3 = {-1.07692443, 0.85223457,  -0.37266530, 2.31511577,  0.04227017,
                             1.13229428,  -0.52769242, 0.27307182,  -0.47779843, -0.08023168,
                             -0.22862823, 0.81489871,  1.13139581,  1.13860467,  0.24309065,
                             0.26533729,  0.49106772,  -1.18860493, 0.27842449,  1.03568141,
                             0.49759611,  0.10021662,  0.00592602,  0.90862000};
    migraphx::shape m3_shape{migraphx::shape::float_type, {2, 3, 2, 2}};

    auto l1     = mm->add_literal(migraphx::literal{m1_shape, m1});
    auto l2     = mm->add_literal(migraphx::literal{m2_shape, m2});
    auto l3     = mm->add_literal(migraphx::literal{m3_shape, m3});
    float alpha = 0.35;
    float beta  = 0.41;
    migraphx::add_apply_alpha_beta(*mm,
                                   std::vector<migraphx::instruction_ref>{l1, l2, l3},
                                   migraphx::make_op("dot"),
                                   alpha,
                                   beta);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-0.91147203, 0.47540785, -0.30313587, 0.43325099,  -0.43711586,
                               0.50928632,  0.06919868, -0.80382802, -0.05125718, -0.06685650,
                               -0.06972163, 0.32407764, 0.45677396,  0.25909489,  0.56911252,
                               -0.17183724, 0.10858734, 0.39406289,  0.04662959,  1.07979824,
                               0.40355016,  0.52410648, -0.31728447, 1.09550845};

    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_2_d_c_test0)
{

    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {-0.86217194,
                            -1.04129542,
                            -0.64850364,
                            -0.97078327,
                            -0.40516386,
                            0.83136927,
                            0.37717502,
                            0.42271939,
                            1.10062165,
                            -0.92239359,
                            0.40403076,
                            -0.43935377};
    std::vector<float> b = {0.76084386,
                            1.89201125,
                            1.73218067,
                            0.7148568,
                            -0.55578914,
                            0.05799101,
                            -1.24090721,
                            -0.51151978,
                            1.13255803,
                            0.21540723,
                            -1.10459009,
                            0.45580331};
    std::vector<float> c = {-0.80473623,
                            0.35154171,
                            -2.73077756,
                            -0.09093885,
                            -1.88850472,
                            -0.03375556,
                            -0.41798276,
                            2.87368099,
                            2.11031439};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    migraphx::add_apply_alpha_beta(*mm, {al, bl, cl}, migraphx::make_op("dot"), 1.0f, 1.0f);
    std::vector<float> gold = {
        -1.60947, 0.703083, -5.46156, -0.181878, -3.77701, -0.0675112, -0.835966, 5.74736, 4.22063};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_vv_inner_product_1)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {0.7481789,
                            0.02906279,
                            1.01193836,
                            1.60222907,
                            1.89135978,
                            0.30054158,
                            -0.4892588,
                            -0.27027533};
    std::vector<float> b = {-0.25829116,
                            0.27908929,
                            -1.27888957,
                            0.21152361,
                            0.08593658,
                            0.52163899,
                            1.38343824,
                            -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    auto bl  = mm->add_literal(migraphx::literal{b_shape, b});
    auto ual = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), al);
    auto ubl = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bl);
    mm->add_instruction(migraphx::make_op("dot"), ual, ubl);
    std::vector<float> gold = {-1.43461};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_vv_inner_product_2)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {0.7481789,
                            0.02906279,
                            1.01193836,
                            1.60222907,
                            1.89135978,
                            0.30054158,
                            -0.4892588,
                            -0.27027533};
    std::vector<float> b = {-0.25829116,
                            0.27908929,
                            -1.27888957,
                            0.21152361,
                            0.08593658,
                            0.52163899,
                            1.38343824,
                            -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto al     = mm->add_literal(migraphx::literal{a_shape, a});
    auto bl     = mm->add_literal(migraphx::literal{b_shape, b});
    auto ual    = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), al);
    auto ubl    = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bl);
    float alpha = 0.32f;
    migraphx::add_apply_alpha_beta(
        *mm, std::vector<migraphx::instruction_ref>{ual, ubl}, migraphx::make_op("dot"), alpha);
    std::vector<float> gold = {-0.4590752};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_vm_1)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {1.49530002,
                            -0.07181969,
                            0.44593846,
                            -0.8645019,
                            0.52992304,
                            -0.4910338,
                            -2.12179422,
                            -0.45962977};
    std::vector<float> b = {
        -0.06210242, 0.0187149,   1.47482984,  -1.19590602, -0.45601701, 0.36934488, -0.83913193,
        0.75350964,  0.80707019,  0.35923582,  -2.18480722, -0.85608682, 0.75849199, 0.49103473,
        -0.91329477, -0.36364322, -0.69688937, 0.07165814,  -0.15505523, 0.52221663, -0.98631192,
        -0.37353654, -1.89818706, -0.87209739, -0.33942003, 0.11390353,  0.78181162, -0.18395337,
        -0.34743419, -0.08091231, 1.21119765,  1.23869861,  1.42169414,  0.86412382, 1.05898002,
        -0.31918307, 1.08546695,  1.50682711,  -0.66083538, -0.32683929};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    auto ual = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), al);
    migraphx::shape b_shape{migraphx::shape::float_type, {8, 5}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    mm->add_instruction(migraphx::make_op("dot"), ual, bl);
    std::vector<float> gold = {-3.78111, -3.40007, -2.1972, -3.31448, -3.80326};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_vm_2)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {1.49530002,
                            -0.07181969,
                            0.44593846,
                            -0.8645019,
                            0.52992304,
                            -0.4910338,
                            -2.12179422,
                            -0.45962977};
    std::vector<float> b = {
        -0.06210242, 0.0187149,   1.47482984,  -1.19590602, -0.45601701, 0.36934488, -0.83913193,
        0.75350964,  0.80707019,  0.35923582,  -2.18480722, -0.85608682, 0.75849199, 0.49103473,
        -0.91329477, -0.36364322, -0.69688937, 0.07165814,  -0.15505523, 0.52221663, -0.98631192,
        -0.37353654, -1.89818706, -0.87209739, -0.33942003, 0.11390353,  0.78181162, -0.18395337,
        -0.34743419, -0.08091231, 1.21119765,  1.23869861,  1.42169414,  0.86412382, 1.05898002,
        -0.31918307, 1.08546695,  1.50682711,  -0.66083538, -0.32683929};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    auto ual = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), al);
    migraphx::shape b_shape{migraphx::shape::float_type, {8, 5}};
    auto bl     = mm->add_literal(migraphx::literal{b_shape, b});
    float alpha = 0.5f;
    migraphx::add_apply_alpha_beta(
        *mm, std::vector<migraphx::instruction_ref>{ual, bl}, migraphx::make_op("dot"), alpha);
    std::vector<float> gold = {-1.89056, -1.70003, -1.0986, -1.65724, -1.90163};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_vm_3)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {-1.7468318, -0.38900251, 1.00183915, 0.06016438, 0.08295905, 1.5830535};
    std::vector<float> b = {
        1.2459538,   0.39586199,  -0.77035574, 0.22689828,  0.3289835,   1.02804361,  -0.22941113,
        -0.33940324, 0.80078249,  1.0319152,   0.80034948,  -0.11631159, 0.36899208,  -0.28506697,
        -1.2211584,  -0.55678377, -0.3618498,  0.34857264,  -0.38700147, -0.43434611, 1.73029783,
        -0.71578372, 0.09777723,  0.06616614,  -1.66721186, -0.16046032, -1.64581663, 1.09373609,
        -0.14127692, -0.01938473, -0.67310303, -1.56154787, -1.0665462,  0.68538535,  -1.53920085,
        -0.35710272, 0.06887234,  0.17474616,  1.08194804,  -0.19990148, -0.91149488, 0.95303646,
        0.95448717,  -0.49332393, -1.762213,   -0.56571194, -1.69704968, -0.82798066, 0.65531872,
        1.5007798,   0.99877355,  0.53386114,  -0.88150609, -1.0756985,  0.50962511,  -0.68019002,
        0.1583068,   2.83988407,  -1.10292457, 0.02126969,  0.21129951,  0.25690146,  -1.6490316,
        0.55261771,  -1.70504303, -0.02870394, -0.18205627, 0.29446203,  -1.91360924, 0.46102174,
        0.44977568,  -0.48113321};

    migraphx::shape a_shape{migraphx::shape::float_type, {6}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    auto ual = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), al);
    auto bual =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 1, 6}}}), ual);
    migraphx::shape b_shape{migraphx::shape::float_type, {3, 6, 4}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    mm->add_instruction(migraphx::make_op("dot"), bual, bl);
    std::vector<float> gold = {1.22914,
                               -1.17896,
                               2.28596,
                               -0.345637,
                               -0.962362,
                               0.168508,
                               -0.947471,
                               -3.02458,
                               -3.80131,
                               1.38484,
                               -2.45019,
                               -1.35064};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_vm_4)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {-1.7468318, -0.38900251, 1.00183915, 0.06016438, 0.08295905, 1.5830535};
    std::vector<float> b = {
        1.2459538,   0.39586199,  -0.77035574, 0.22689828,  0.3289835,   1.02804361,  -0.22941113,
        -0.33940324, 0.80078249,  1.0319152,   0.80034948,  -0.11631159, 0.36899208,  -0.28506697,
        -1.2211584,  -0.55678377, -0.3618498,  0.34857264,  -0.38700147, -0.43434611, 1.73029783,
        -0.71578372, 0.09777723,  0.06616614,  -1.66721186, -0.16046032, -1.64581663, 1.09373609,
        -0.14127692, -0.01938473, -0.67310303, -1.56154787, -1.0665462,  0.68538535,  -1.53920085,
        -0.35710272, 0.06887234,  0.17474616,  1.08194804,  -0.19990148, -0.91149488, 0.95303646,
        0.95448717,  -0.49332393, -1.762213,   -0.56571194, -1.69704968, -0.82798066, 0.65531872,
        1.5007798,   0.99877355,  0.53386114,  -0.88150609, -1.0756985,  0.50962511,  -0.68019002,
        0.1583068,   2.83988407,  -1.10292457, 0.02126969,  0.21129951,  0.25690146,  -1.6490316,
        0.55261771,  -1.70504303, -0.02870394, -0.18205627, 0.29446203,  -1.91360924, 0.46102174,
        0.44977568,  -0.48113321};

    migraphx::shape a_shape{migraphx::shape::float_type, {6}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    auto ual = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), al);
    auto bual =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 1, 6}}}), ual);
    migraphx::shape b_shape{migraphx::shape::float_type, {3, 6, 4}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    migraphx::add_apply_alpha_beta(
        *mm, std::vector<migraphx::instruction_ref>{bual, bl}, migraphx::make_op("dot"), 0.21f);
    std::vector<float> gold = {0.25812,
                               -0.247582,
                               0.480051,
                               -0.0725837,
                               -0.202096,
                               0.0353867,
                               -0.198969,
                               -0.635161,
                               -0.798275,
                               0.290817,
                               -0.514539,
                               -0.283635};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mv_1)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {0.1612524,
                            0.61266466,
                            -0.19212896,
                            1.34228825,
                            -1.09746949,
                            0.4680955,
                            -0.431748,
                            -0.89791241,
                            -2.19078702,
                            -0.13767058,
                            -1.66105228,
                            -0.91834613,
                            0.59199744,
                            1.41967261,
                            0.76237423};

    std::vector<float> b = {0.14365572, 0.23401411, -0.8970094, -0.12526676, -1.04703286};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5}};
    auto bl  = mm->add_literal(migraphx::literal{b_shape, b});
    auto ubl = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bl);
    mm->add_instruction(migraphx::make_op("dot"), al, ubl);
    std::vector<float> gold = {1.31982, 1.19022, -1.96062};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mv_2)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {0.1612524,
                            0.61266466,
                            -0.19212896,
                            1.34228825,
                            -1.09746949,
                            0.4680955,
                            -0.431748,
                            -0.89791241,
                            -2.19078702,
                            -0.13767058,
                            -1.66105228,
                            -0.91834613,
                            0.59199744,
                            1.41967261,
                            0.76237423};

    std::vector<float> b = {0.14365572, 0.23401411, -0.8970094, -0.12526676, -1.04703286};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5}};
    auto bl     = mm->add_literal(migraphx::literal{b_shape, b});
    auto ubl    = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bl);
    float alpha = 0.3f;
    migraphx::add_apply_alpha_beta(
        *mm, std::vector<migraphx::instruction_ref>{al, ubl}, migraphx::make_op("dot"), alpha);
    std::vector<float> gold = {0.395946, 0.357067, -0.588187};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mv_3)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.24593227,  -0.84351316, 0.27882229,  -0.42518484, -1.11391528, 0.59141834,  1.34198714,
        2.25884063,  -1.32093452, 0.44766336,  -0.09306479, 0.47526699,  0.25858488,  1.30820392,
        1.17186787,  0.31530864,  -1.19159424, -0.24100903, -1.03857886, 1.54453427,  0.05041654,
        1.67108177,  0.965805,    0.52958924,  -1.61243992, 0.02941846,  0.77523836,  1.97963853,
        -2.51093596, 0.21882645,  -2.60193574, 1.1899952,   1.70883519,  0.94586745,  2.65002512,
        -1.42427102, 1.0143951,   -1.34115312, 1.63833732,  -1.46477355, 0.44014877,  0.58032696,
        -1.63874372, -0.82834423, 1.81131778,  -0.52393379, 1.16721943,  0.39488835,  0.23947128,
        -0.15733194, 0.19451158,  1.21315445,  0.44594897,  0.40809135,  -0.64252994, 0.7541716,
        -0.97203195, 0.69208485,  0.34350988,  0.9836842};
    std::vector<float> b = {0.05013914, 1.39932885, 2.56616476, 1.02225623, -0.03977829};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5}};
    auto bl  = mm->add_literal(migraphx::literal{b_shape, b});
    auto ubl = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bl);
    auto bubl =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 5, 1}}}), ubl);
    mm->add_instruction(migraphx::make_op("dot"), al, bubl);
    std::vector<float> gold = {-0.792717,
                               6.33595,
                               2.61466,
                               -3.39322,
                               5.42485,
                               3.59084,
                               6.78139,
                               -0.360492,
                               -4.28998,
                               2.87146,
                               3.29447,
                               0.765651};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mm1_1)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -0.49450006, -1.07431991, -0.02796692, -0.99631927, 0.20040449,  -1.39709437, -0.15695328,
        0.08208373,  -0.09746386, 0.77923021,  -0.1849151,  0.14419043,  -0.25798175, -0.2504807,
        -1.11134383, -0.71030613, -0.20234025, 0.90229168,  0.62643053,  -0.83512638, 1.66051254,
        0.05941673,  0.73081559,  0.27111867,  0.55060745,  0.34999583,  1.02236619,  0.60178395,
        1.49646162,  1.93255155,  -3.65357913, -1.38059906, -0.46302398, 0.19847152,  0.39785875,
        1.47004861,  -1.24482133, -0.01954702, 0.36073898,  1.56055978,  -0.10344603, -0.34283135,
        -0.56482649, 1.80861249,  -0.92268202, 0.94371182,  -0.02373232, -0.75441145, 0.43325034,
        0.4057425,   -0.48844822, -0.36390512, 0.74110406,  1.25158366,  0.52196654,  1.43461691,
        -0.57530864, -0.66716206, -1.76516289, 0.96582849};
    std::vector<float> b = {0.49899375,
                            -2.20168661,
                            1.08895066,
                            -0.01135643,
                            0.90570669,
                            -1.43550963,
                            -1.73033377,
                            0.21338776,
                            0.96962508,
                            0.38913968,
                            -0.32822861,
                            0.88222863,
                            0.93330718,
                            -1.24265228,
                            -1.62587164};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    auto bbl =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 5, 3}}}), bl);
    mm->add_instruction(migraphx::make_op("dot"), al, bbl);
    std::vector<float> gold = {-0.386828, 0.187735,  -0.22822, -0.148057, 2.015,    -2.56938,
                               -0.782212, 1.9459,    0.927426, -2.44907,  2.40531,  2.30232,
                               0.182745,  -4.21937,  1.77551,  1.50775,   -2.60888, -2.32484,
                               -0.557691, 6.13527,   -2.91743, 2.37836,   -6.42584, 1.14979,
                               0.77227,   0.349659,  2.92759,  2.32384,   -2.90664, 0.0527679,
                               -0.547761, -0.155467, 0.964619, 2.09133,   -4.44281, -1.3864};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mm1_2)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {-0.0309568,
                            -1.57294749,
                            -0.00768606,
                            1.5786921,
                            0.50519718,
                            0.10530702,
                            -0.05302112,
                            -0.06503757,
                            0.4079716,
                            0.0799132,
                            -0.82624962,
                            0.49341502};

    std::vector<float> b = {
        0.3664867,   0.24649534,  1.14728076,  1.09911548,  -1.23711247, -0.49436419, -0.67557879,
        -0.84180575, -1.09754376, 0.07807351,  0.74349043,  -0.92084701, 0.50267885,  0.78709401,
        0.80598159,  -0.51269589, -0.40337193, 0.29457878,  1.25447301,  -1.66251457, -1.54652239,
        -0.35067765, -0.5214464,  -0.7866878,  1.11128573,  0.26927291,  -0.0929818,  0.07523954,
        0.3256776,   -1.08617826, 0.89294253,  -0.91007619, -2.42825765, -1.76805581, 1.08136334,
        -0.14521253, -1.32061148, 0.60663124,  -1.19835255, -0.98803563, -1.06927896, -0.51967419,
        -0.98974639, 1.01287011,  1.34910394,  0.1203349,   0.67387452,  -0.32447465, 1.15187449,
        -0.82253807, 0.22302433,  0.46434695,  0.319647,    1.56459445,  0.15664012,  0.03998102,
        0.62981041,  0.11831296,  0.47824434,  -0.93941882, -0.34674036, 1.17071104,  0.59203806,
        2.75817738,  -0.69300013, 1.30971899,  -0.14231862, -1.90915568, -0.06895489, 0.20160375,
        0.01945916,  0.03586956};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    auto bal =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 3, 4}}}), al);
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 3, 4, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    mm->add_instruction(migraphx::make_op("dot"), bal, bl);
    std::vector<float> gold = {
        -1.61175,  3.11849,  -0.703205, 0.331635,  -0.00946922, 0.645626, 0.834069,  1.06409,
        0.881037,  0.227628, -0.200308, -1.71836,  0.156255,    0.477222, 0.571363,  -1.04543,
        1.40524,   1.24201,  -2.95083,  1.19352,   1.5008,      0.636987, 0.148256,  -0.0231631,
        -1.15079,  1.42139,  1.80996,   1.79259,   2.7192,      0.331902, -0.726565, 0.0963351,
        -0.710558, 0.259424, -0.342345, -1.80522,  -0.580476,   0.277368, -3.95582,  0.614823,
        -0.415107, 0.305138, 0.435993,  -0.107089, -0.767885,   -4.00837, 1.09921,   -2.02129,
        0.109717,  0.618422, 0.438342,  0.29602,   2.00928,     0.420871};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mm2_1)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -0.49450006, -1.07431991, -0.02796692, -0.99631927, 0.20040449,  -1.39709437, -0.15695328,
        0.08208373,  -0.09746386, 0.77923021,  -0.1849151,  0.14419043,  -0.25798175, -0.2504807,
        -1.11134383, -0.71030613, -0.20234025, 0.90229168,  0.62643053,  -0.83512638, 1.66051254,
        0.05941673,  0.73081559,  0.27111867,  0.55060745,  0.34999583,  1.02236619,  0.60178395,
        1.49646162,  1.93255155,  -3.65357913, -1.38059906, -0.46302398, 0.19847152,  0.39785875,
        1.47004861,  -1.24482133, -0.01954702, 0.36073898,  1.56055978,  -0.10344603, -0.34283135,
        -0.56482649, 1.80861249,  -0.92268202, 0.94371182,  -0.02373232, -0.75441145, 0.43325034,
        0.4057425,   -0.48844822, -0.36390512, 0.74110406,  1.25158366,  0.52196654,  1.43461691,
        -0.57530864, -0.66716206, -1.76516289, 0.96582849};
    std::vector<float> b = {-1.12211357, 1.74720423,  0.60382572,  -0.61090125, -0.3315936,
                            0.30924675,  -0.28906435, 0.64039247,  -1.2822253,  0.55899286,
                            2.14013013,  1.00944809,  0.21660017,  -0.75465098, 0.12097934,
                            -1.64006315, 0.43582108,  -0.64348541, 0.43101069,  1.30191386,
                            1.7746011,   0.24935804,  0.42830791,  -0.13593643, 0.38749427,
                            1.39776254,  -0.42911717, -1.3537624,  -0.81999648, -0.1754485};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 1, 5, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    auto bbl =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 5, 3}}}), bl);
    std::vector<float> gold = {
        0.70574512,  -2.80915314, -1.57644969, 1.75415381,  -3.13303087, -1.00150259,
        -0.18675123, -0.23349122, -0.12357225, 0.82911538,  1.37473744,  -1.11709934,
        -1.84001907, 3.51427391,  0.42425673,  0.0638482,   2.40210271,  1.50027643,
        4.81988916,  -3.63687142, -0.19101717, -4.92522092, -1.76377022, -3.58095615,
        1.83096922,  2.5512663,   -1.07926588, -2.12749134, 0.33014536,  -0.80393025,
        0.60740202,  0.95217761,  -1.06087445, -4.75868152, -3.6687713,  -1.26539821};
    mm->add_instruction(migraphx::make_op("dot"), al, bbl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mm2_2)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {-0.19276159, -1.2568421,  -0.321242,   1.21471077,  -0.4927751,
                            0.69446894,  -0.1786371,  -1.00763473, -0.10279314, 3.02931355,
                            1.08359235,  -0.35190132, -0.00639111, 0.78989113,  1.23538029,
                            0.4590747,   0.17304142,  0.42512412,  0.21076913,  -0.01724556,
                            -0.17763898, 0.12852236,  -0.00459301, 1.34498824,  0.02907823,
                            0.1784464,   -0.20790355, -0.52336699, 0.45804085,  1.06025801};

    std::vector<float> b = {-1.12211357, 1.74720423,  0.60382572,  -0.61090125, -0.3315936,
                            0.30924675,  -0.28906435, 0.64039247,  -1.2822253,  0.55899286,
                            2.14013013,  1.00944809,  0.21660017,  -0.75465098, 0.12097934,
                            -1.64006315, 0.43582108,  -0.64348541, 0.43101069,  1.30191386,
                            1.7746011,   0.24935804,  0.42830791,  -0.13593643, 0.38749427,
                            1.39776254,  -0.42911717, -1.3537624,  -0.81999648, -0.1754485};
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 3, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    auto bal =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 5}}}), al);
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 1, 5, 3}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    auto bbl =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 5, 3}}}), bl);
    mm->add_instruction(migraphx::make_op("dot"), bal, bbl);
    std::vector<float> gold = {1.64924590e+00,  2.84575831e+00,  1.07340773e+00,  2.19817080e-01,
                               -1.87873283e+00, 1.91883003e+00,  -2.89962196e-01, 2.76404142e+00,
                               1.50048102e+00,  -6.29650347e-01, 1.48105185e+00,  -3.71716505e-03,
                               8.80281500e-01,  2.50057585e+00,  1.29958508e+00,  5.63751779e-01,
                               2.25703781e-01,  1.30516919e+00,  8.32118386e-01,  2.44050864e-01,
                               -2.49748221e+00, -5.60803176e+00, -2.98919069e+00, -1.11429417e+00,
                               -3.29675989e+00, 1.02442564e-01,  -1.87659303e+00, -4.67302454e-01,
                               9.16189968e-01,  -1.33537175e-01, 8.27398578e-01,  1.94406914e+00,
                               -2.39250915e-01, -1.77062701e+00, -6.46239534e-01, -7.95202750e-01};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mm2_3)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -0.55248691, 0.70275958,  0.56967633,  0.88206033,  -0.85088547, 0.05689149,  -0.20084703,
        0.18024434,  1.0730491,   0.15913531,  0.93621628,  0.35072771,  1.28616952,  1.55384379,
        0.30376261,  -1.12356544, -0.64271552, -2.50703079, -0.23994372, 0.8166084,   0.06542249,
        -0.17472336, -0.37665211, 0.16342699,  0.07645941,  0.65024333,  -1.19883423, -0.40536776,
        -0.31132765, 0.78113691,  -0.16887638, 2.30797418,  -0.36241233, 0.33552153,  -1.05343996,
        -0.16909699, -1.22608815, 1.64165613,  0.96260828,  -0.16733976, 0.84211199,  1.31243813,
        0.89258549,  -0.48250384, -1.06005206, 1.37021342,  -0.35658565, 0.26879188};

    std::vector<float> b = {
        0.17111129,  -0.82134741, -1.58001178, -1.46759447, 0.31522514,  -0.11567352, -0.038978,
        -0.3601414,  -0.84379876, 0.24848939,  -0.37080544, 0.00838631,  1.51316241,  0.42385344,
        2.06043846,  1.82348849,  1.07180434,  0.6567393,   1.41164561,  0.73091185,  -0.33541302,
        -0.98082287, -0.06605479, 0.82219717,  -1.41619634, 0.51326658,  0.26916313,  0.79819769,
        0.85583702,  0.07876046,  -0.42375545, -0.7758751,  1.14334296,  -0.14211708, -1.54520411,
        -0.55244869, -0.48478899, 0.10782164,  -0.20879552, -0.99019754, 1.78783102,  -1.31610052,
        1.73510175,  -0.48360172, 0.62367417,  -1.34180545, -0.37512931, -1.50521357, 0.08383314,
        0.76165608,  -0.4961646,  0.95821311,  -0.68407191, 0.48299435,  -0.24323988, 0.34793412,
        0.37908669,  1.19083454,  1.30218795,  -0.26731035, -0.34544132, -0.09595373, 0.50951334,
        0.48896956,  0.38753818,  -0.4939919,  0.02352126,  0.42013764,  0.07027765,  0.21169851,
        -0.24411376, -1.77793736, -0.88370924, 0.95294025,  -0.08208804, -0.95943892, 0.30280474,
        1.1967013,   -1.17700948, 0.29533973};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 4, 5}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    mm->add_instruction(migraphx::make_op("dot"), al, bl);
    std::vector<float> gold = {
        1.22136035,  1.3765651,   2.0611395,   1.70445494,  1.8189619,   0.2509717,   0.88815736,
        1.13837946,  1.37006127,  -0.53617378, 0.45759693,  -0.503786,   -0.10575749, -0.81715738,
        2.56316255,  0.85812927,  -0.53425671, 1.38147704,  2.57874755,  -1.05591061, -1.42065674,
        -0.25412658, -2.14494165, -2.81045272, 0.27491485,  -0.04229986, 0.10181043,  -0.55680682,
        -0.07633866, 0.313767,    -0.28202571, -1.64696179, -0.50872733, -1.08935912, 0.94291084,
        -0.71792156, 0.82981387,  1.14797592,  3.13989358,  -0.17507726, -0.63429162, -0.72241531,
        -0.61459168, -0.52561056, 0.3309648,   -0.46185697, -1.60586695, -0.98590829, 0.63012062,
        -0.25606052, -0.69419352, -1.78299913, -0.38572706, 1.92249442,  0.3884186,   -0.48153048,
        0.84932351,  0.67234919,  -1.07821322, -0.01208216};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_mm2_4)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -0.55248691, 0.70275958,  0.56967633,  0.88206033,  -0.85088547, 0.05689149,  -0.20084703,
        0.18024434,  1.0730491,   0.15913531,  0.93621628,  0.35072771,  1.28616952,  1.55384379,
        0.30376261,  -1.12356544, -0.64271552, -2.50703079, -0.23994372, 0.8166084,   0.06542249,
        -0.17472336, -0.37665211, 0.16342699,  0.07645941,  0.65024333,  -1.19883423, -0.40536776,
        -0.31132765, 0.78113691,  -0.16887638, 2.30797418,  -0.36241233, 0.33552153,  -1.05343996,
        -0.16909699, -1.22608815, 1.64165613,  0.96260828,  -0.16733976, 0.84211199,  1.31243813,
        0.89258549,  -0.48250384, -1.06005206, 1.37021342,  -0.35658565, 0.26879188};

    std::vector<float> b = {
        -0.33734601, 0.66386073,  0.41425048,  0.40190389,  -0.99645073, -0.10017067, -0.58542118,
        0.48636962,  0.06301405,  1.14669128,  -0.06526677, 0.23172741,  -1.49693143, -0.44464233,
        -0.12775566, -1.32038007, 1.1812471,   1.22362746,  -0.49013843, 0.25339836,  1.31698705,
        1.54256669,  0.11211132,  -0.18005487, 0.36730145,  0.97705953,  -0.18909084, 0.544932,
        0.32891878,  0.64250015,  -0.41381398, 0.47402562,  1.22286761,  1.07573211,  -0.92988077,
        -0.36340925, -1.76152377, -0.96642674, -0.79231929, 0.11517073};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 4, 5}};
    auto bl = mm->add_literal(migraphx::literal{b_shape, b});
    auto bbl =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 4, 5}}}), bl);
    mm->add_instruction(migraphx::make_op("dot"), al, bbl);
    std::vector<float> gold = {
        -1.08585245, 0.39575611,  0.33947977,  -0.86339678, 1.50710753,  0.05646156,  -0.43180359,
        0.19639674,  -0.33742881, 0.98443538,  -0.9021272,  1.25043704,  -0.45038184, -0.14689614,
        -0.91749459, 3.49467934,  3.81336312,  2.4482385,   1.49649707,  1.05889193,  -3.49343731,
        -2.06958956, -2.52082858, -1.61401519, -1.52966956, 0.01191848,  -0.33246613, -0.70641362,
        -0.60391255, 0.28083355,  0.52255496,  -1.08655006, 1.64648546,  0.80344255,  0.71987865,
        -3.00960296, 2.02318221,  3.32785057,  -1.13203844, 1.81235734,  0.38067585,  -0.88086897,
        1.38307367,  0.42677257,  0.83759966,  -0.34827442, -1.45067092, 2.09599671,  1.92882983,
        -0.30996324, 2.19736278,  2.32389426,  2.36741832,  1.62253915,  0.26698225,  -0.00741609,
        -2.53680983, -0.0679954,  0.04499683,  0.85354276};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(dot_dyn_2_d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type, {{1, 4}, {5, 5}}};
    auto ap = mm->add_parameter("a", a_shape);
    migraphx::shape b_shape{migraphx::shape::float_type, {5, 3}};
    auto bp = mm->add_parameter("b", b_shape);
    mm->add_instruction(migraphx::make_op("dot"), ap, bp);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                            1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                            -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                            -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {4, 5}};
    migraphx::parameter_map params;
    params["a"] = migraphx::argument(input_fixed_shape, a.data());
    params["b"] = migraphx::argument(b_shape, b.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1.56327541e+00,
                               -7.09570140e-01,
                               -5.37424982e-01,
                               -2.22994831e-01,
                               -2.15586437e+00,
                               2.09177941e-03,
                               -1.47279677e+00,
                               2.02627040e-01,
                               -6.04527691e-01,
                               -1.29885596e+00,
                               2.16294914e+00,
                               -1.48101497e-01};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(dot_dyn_4_d_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type, {{1, 1}, {1, 1}, {4, 6, {4}}, {5, 5}}};
    auto al = mm->add_parameter("a", a_shape);
    migraphx::shape b_shape{migraphx::shape::float_type, {1, 1, 5, 3}};
    auto bl = mm->add_parameter("b", b_shape);
    mm->add_instruction(migraphx::make_op("dot"), al, bl);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                            1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                            -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                            -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 1, 4, 5}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 1, 5, 3}};
    migraphx::parameter_map params;
    params["a"] = migraphx::argument(input_fixed_shape0, a.data());
    params["b"] = migraphx::argument(input_fixed_shape1, b.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1.56327541e+00,
                               -7.09570140e-01,
                               -5.37424982e-01,
                               -2.22994831e-01,
                               -2.15586437e+00,
                               2.09177941e-03,
                               -1.47279677e+00,
                               2.02627040e-01,
                               -6.04527691e-01,
                               -1.29885596e+00,
                               2.16294914e+00,
                               -1.48101497e-01};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(quant_dot_2args_multi4_1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4, 8}};
    std::vector<int8_t> data1(4 * 4);
    std::vector<int8_t> data2(4 * 8);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1 = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, data2});
    mm->add_instruction(migraphx::make_op("quant_dot"), l1, l2);

    std::vector<int> gold = {112, 118, 124, 130, 136, 142, 148, 154, 304,  326, 348,
                             370, 392, 414, 436, 458, 496, 534, 572, 610,  648, 686,
                             724, 762, 688, 742, 796, 850, 904, 958, 1012, 1066};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_multi4_2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4, 8}};
    std::vector<int8_t> data1(4 * 4);
    std::vector<int8_t> data2(4 * 8);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    mm->add_instruction(migraphx::make_op("quant_dot"), tl1, l2);

    std::vector<int> gold = {448, 472, 496, 520, 544, 568, 592, 616, 496, 524, 552,
                             580, 608, 636, 664, 692, 544, 576, 608, 640, 672, 704,
                             736, 768, 592, 628, 664, 700, 736, 772, 808, 844};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_multi4_3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 4}};
    std::vector<int8_t> data1(4 * 4);
    std::vector<int8_t> data2(4 * 8);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);
    mm->add_instruction(migraphx::make_op("quant_dot"), l1, tl2);

    std::vector<int> gold = {14,  38,   62,  86,  110, 134, 158, 182,  38,   126, 214,
                             302, 390,  478, 566, 654, 62,  214, 366,  518,  670, 822,
                             974, 1126, 86,  302, 518, 734, 950, 1166, 1382, 1598};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_multi4_4)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 4}};
    std::vector<int8_t> data1(4 * 4);
    std::vector<int8_t> data2(4 * 8);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);
    mm->add_instruction(migraphx::make_op("quant_dot"), tl1, tl2);

    std::vector<int> gold = {56,  152, 248, 344, 440, 536, 632, 728, 62,  174, 286,
                             398, 510, 622, 734, 846, 68,  196, 324, 452, 580, 708,
                             836, 964, 74,  218, 362, 506, 650, 794, 938, 1082};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_general_1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4, 5}};
    std::vector<int8_t> data1(3 * 4);
    std::vector<int8_t> data2(4 * 5);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1 = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, data2});
    mm->add_instruction(migraphx::make_op("quant_dot"), l1, l2);

    std::vector<int> gold = {70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_general_2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 3}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4, 5}};
    std::vector<int8_t> data1(4 * 3);
    std::vector<int8_t> data2(4 * 5);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    mm->add_instruction(migraphx::make_op("quant_dot"), tl1, l2);

    std::vector<int> gold = {
        210, 228, 246, 264, 282, 240, 262, 284, 306, 328, 270, 296, 322, 348, 374};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_general_3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {5, 4}};
    std::vector<int8_t> data1(3 * 4);
    std::vector<int8_t> data2(4 * 5);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);

    migraphx::add_apply_alpha_beta(*mm, {l1, tl2}, migraphx::make_op("quant_dot"), 2);

    std::vector<int> gold = {
        28, 76, 124, 172, 220, 76, 252, 428, 604, 780, 124, 428, 732, 1036, 1340};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_2args_general_4)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 3}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {5, 4}};
    std::vector<int8_t> data1(4 * 3);
    std::vector<int8_t> data2(4 * 5);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);
    migraphx::add_apply_alpha_beta(*mm, {tl1, tl2}, migraphx::make_op("quant_dot"), 3);

    std::vector<int> gold = {
        126, 342, 558, 774, 990, 144, 408, 672, 936, 1200, 162, 474, 786, 1098, 1410};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_general_1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 8}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};
    std::vector<int8_t> data1(2 * 8);
    std::vector<int8_t> data2(8 * 7);
    std::vector<int> data3(2 * 7);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 2);

    auto l1 = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto l3 = mm->add_literal(migraphx::literal{m3_shape, data3});
    migraphx::add_apply_alpha_beta(*mm, {l1, l2, l3}, migraphx::make_op("quant_dot"), 1, 1);

    std::vector<int> gold = {
        982, 1011, 1040, 1069, 1098, 1127, 1156, 2557, 2650, 2743, 2836, 2929, 3022, 3115};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_general_2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4, 5}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {3, 5}};
    std::vector<int8_t> data1(3 * 4);
    std::vector<int8_t> data2(4 * 5);
    std::vector<int> data3(3 * 5);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 0);

    auto l1 = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, data2});
    mm->add_instruction(migraphx::make_op("quant_dot"), l1, l2);

    std::vector<int> gold = {70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_general_3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {8, 2}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};
    std::vector<int8_t> data1(2 * 8);
    std::vector<int8_t> data2(8 * 7);
    std::vector<int> data3(2 * 7);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 2);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto l3  = mm->add_literal(migraphx::literal{m3_shape, data3});
    migraphx::add_apply_alpha_beta(*mm, {tl1, l2, l3}, migraphx::make_op("quant_dot"), 1, 3);

    std::vector<int> gold = {
        1966, 2025, 2084, 2143, 2202, 2261, 2320, 2183, 2250, 2317, 2384, 2451, 2518, 2585};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_general_4)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 8}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 8}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};
    std::vector<int8_t> data1(2 * 8);
    std::vector<int8_t> data2(8 * 7);
    std::vector<int> data3(2 * 7);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 2);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);
    auto l3  = mm->add_literal(migraphx::literal{m3_shape, data3});
    migraphx::add_apply_alpha_beta(*mm, {l1, tl2, l3}, migraphx::make_op("quant_dot"), 2, 3);

    std::vector<int> gold = {
        286, 737, 1188, 1639, 2090, 2541, 2992, 755, 2230, 3705, 5180, 6655, 8130, 9605};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_general_5)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {8, 2}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 8}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};
    std::vector<int8_t> data1(2 * 8);
    std::vector<int8_t> data2(8 * 7);
    std::vector<int> data3(2 * 7);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 2);

    auto l1  = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    auto l2  = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);
    auto l3  = mm->add_literal(migraphx::literal{m3_shape, data3});
    migraphx::add_apply_alpha_beta(*mm, {tl1, tl2, l3}, migraphx::make_op("quant_dot"), 3, 2);

    std::vector<int> gold = {
        844, 2190, 3536, 4882, 6228, 7574, 8920, 942, 2480, 4018, 5556, 7094, 8632, 10170};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_batch_1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 2, 2, 4}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {2, 2, 4, 7}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 2, 2, 7}};
    std::vector<int8_t> data1(4 * 2 * 4);
    std::vector<int8_t> data2(4 * 4 * 7);
    std::vector<int> data3(4 * 2 * 7);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 2);

    auto l1 = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto l3 = mm->add_literal(migraphx::literal{m3_shape, data3});
    migraphx::add_apply_alpha_beta(*mm, {l1, l2, l3}, migraphx::make_op("quant_dot"), 1, 2);

    std::vector<int> gold = {102,   110,   118,   126,   134,   142,   150,   284,   308,   332,
                             356,   380,   404,   428,   1530,  1570,  1610,  1650,  1690,  1730,
                             1770,  2160,  2216,  2272,  2328,  2384,  2440,  2496,  4750,  4822,
                             4894,  4966,  5038,  5110,  5182,  5828,  5916,  6004,  6092,  6180,
                             6268,  6356,  9762,  9866,  9970,  10074, 10178, 10282, 10386, 11288,
                             11408, 11528, 11648, 11768, 11888, 12008};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}

TEST_CASE(quant_dot_3args_batch_2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 2, 4, 3}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {2, 2, 6, 4}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 2, 3, 6}};
    std::vector<int8_t> data1(48);
    std::vector<int8_t> data2(96);
    std::vector<int> data3(72);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    std::iota(data3.begin(), data3.end(), 2);

    auto l1 = mm->add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
    auto l2 = mm->add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
    auto l3 = mm->add_literal(migraphx::literal{m3_shape, data3});
    migraphx::add_apply_alpha_beta(*mm, {tl1, tl2, l3}, migraphx::make_op("quant_dot"), 2, 3);

    std::vector<int> gold = {
        90,    237,   384,   531,   678,   825,   120,   299,   478,   657,   836,   1015,
        150,   361,   572,   783,   994,   1205,  3456,  3987,  4518,  5049,  5580,  6111,
        3678,  4241,  4804,  5367,  5930,  6493,  3900,  4495,  5090,  5685,  6280,  6875,
        11430, 12345, 13260, 14175, 15090, 16005, 11844, 12791, 13738, 14685, 15632, 16579,
        12258, 13237, 14216, 15195, 16174, 17153, 24012, 25311, 26610, 27909, 29208, 30507,
        24618, 25949, 27280, 28611, 29942, 31273, 25224, 26587, 27950, 29313, 30676, 32039};

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(m, gold));
}
