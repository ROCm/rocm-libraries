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
#include <migraphx/op/pooling.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(roialign_out_of_bound_test)
{
    auto create_program = [](const std::string& trans_mode = "half_pixel") {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {1, 1, 10, 10}};
        std::vector<float> x_vec = {
            0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
            0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
            0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
            0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
            0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
            0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
            0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
            0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
            0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
            0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

        migraphx::shape roi_s{migraphx::shape::float_type, {3, 4}};
        std::vector<float> roi_vec = {0, 0, 9.99, 9.99, 0, 5, 4, 9, 5, 5, 9.9, 9.9};

        migraphx::shape ind_s{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> ind_vec = {0, 0, 0};

        auto x   = mm->add_literal(migraphx::literal(x_s, x_vec));
        auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r =
            mm->add_instruction(migraphx::make_op("roialign",
                                                  {{"coordinate_transformation_mode", trans_mode},
                                                   {"spatial_scale", 5.0},
                                                   {"output_height", 1},
                                                   {"output_width", 1},
                                                   {"sampling_ratio", 1}}),
                                x,
                                roi,
                                ind);
        mm->add_return({r});
        return p;
    };

    {
        auto p = create_program("half_pixel");
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0.0f, 0.0f, 0.0f};

        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }
}

TEST_CASE(roialign_test)
{
    auto create_program = [](const std::string& trans_mode = "half_pixel",
                             const migraphx::op::pooling_mode pooling_mode =
                                 migraphx::op::pooling_mode::average,
                             int64_t sampling_ratio = 2) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {1, 1, 10, 10}};
        std::vector<float> x_vec = {
            0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
            0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
            0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
            0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
            0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
            0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
            0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
            0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
            0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
            0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

        migraphx::shape roi_s{migraphx::shape::float_type, {3, 4}};
        std::vector<float> roi_vec = {0, 0, 9, 9, 0, 5, 4, 9, 5, 5, 9, 9};

        migraphx::shape ind_s{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> ind_vec = {0, 0, 0};

        auto x   = mm->add_literal(migraphx::literal(x_s, x_vec));
        auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r =
            mm->add_instruction(migraphx::make_op("roialign",
                                                  {{"coordinate_transformation_mode", trans_mode},
                                                   {"spatial_scale", 1.0},
                                                   {"output_height", 5},
                                                   {"output_width", 5},
                                                   {"sampling_ratio", sampling_ratio},
                                                   {"mode", pooling_mode}}),
                                x,
                                roi,
                                ind);
        mm->add_return({r});
        return p;
    };

    {
        auto p = create_program("output_half_pixel");
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {
            0.466421425, 0.446552634, 0.340521216, 0.568848491, 0.606780827, 0.371379346,
            0.429571986, 0.383519977, 0.556241512, 0.351050019, 0.27680251,  0.488286227,
            0.522200167, 0.552770197, 0.417057365, 0.471240699, 0.4844096,   0.690457463,
            0.492039412, 0.877398551, 0.623889625, 0.712461948, 0.628926516, 0.335504025,
            0.349469036, 0.302179992, 0.43046391,  0.469585985, 0.39774403,  0.542259991,
            0.365552008, 0.704923987, 0.516481996, 0.317131996, 0.701444089, 0.291239977,
            0.505897999, 0.647610962, 0.623489916, 0.829879999, 0.591567993, 0.738860011,
            0.704825997, 0.837148011, 0.889315963, 0.622680008, 0.615276039, 0.709713995,
            0.615356028, 0.458524048, 0.238451958, 0.337952018, 0.371693879, 0.609999895,
            0.760059953, 0.376724035, 0.378532052, 0.71468991,  0.924308002, 0.972783983,
            0.574903965, 0.582623959, 0.570936024, 0.761904061, 0.876998067, 0.535508037,
            0.256580025, 0.214098021, 0.279604018, 0.360000014, 0.436488032, 0.350427985,
            0.288755983, 0.366139978, 0.234920025};

        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }

    {
        auto p = create_program("half_pixel");
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {
            0.517783, 0.343411, 0.322905, 0.447362, 0.634375, 0.40308,  0.536647, 0.442791,
            0.486144, 0.402313, 0.251194, 0.400154, 0.515524, 0.695369, 0.346537, 0.33504,
            0.460099, 0.588069, 0.343863, 0.684932, 0.49319,  0.714058, 0.821744, 0.471935,
            0.403946, 0.306955, 0.218678, 0.33369,  0.488001, 0.486962, 0.18709,  0.49142,
            0.55611,  0.419167, 0.368608, 0.143278, 0.460835, 0.597125, 0.53096,  0.498207,
            0.278818, 0.438569, 0.6022,   0.700038, 0.752436, 0.577385, 0.702383, 0.725097,
            0.733754, 0.816304, 0.23933,  0.407514, 0.337893, 0.252521, 0.474335, 0.367075,
            0.270168, 0.41051,  0.64189,  0.830777, 0.55564,  0.454295, 0.55645,  0.75015,
            0.929997, 0.66257,  0.561664, 0.481275, 0.495449, 0.666306, 0.663573, 0.372107,
            0.205603, 0.192776, 0.247849};

        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }

    {
        auto p = create_program("half_pixel", migraphx::op::pooling_mode::max, 0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {
            0.819145, 0.373103, 0.258302,  0.515419, 0.726104, 0.540536, 0.545512,  0.38511,
            0.376545, 0.274635, 0.22341,   0.184511, 0.230843, 0.404869, 0.29546,   0.540409,
            0.265838, 0.409324, 0.213915,  0.708654, 0.687264, 0.580821, 0.461283,  0.462879,
            0.709632, 0.27873,  0.083619,  0.22428,  0.313992, 0.410508, 0.0929099, 0.415373,
            0.296695, 0.231574, 0.136836,  0.0683,   0.296695, 0.211925, 0.245385,  0.28053,
            0.17091,  0.179879, 0.245385,  0.343539, 0.392742, 0.51273,  0.536193,  0.382995,
            0.422793, 0.761886, 0.0839429, 0.276444, 0.19746,  0.126117, 0.378351,  0.254646,
            0.092148, 0.272825, 0.381955,  0.626599, 0.251325, 0.244475, 0.194875,  0.272825,
            0.44757,  0.351855, 0.342265,  0.244475, 0.274841, 0.553644, 0.607176,  0.202392,
            0.07425,  0.066087, 0.126279};

        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }
}
