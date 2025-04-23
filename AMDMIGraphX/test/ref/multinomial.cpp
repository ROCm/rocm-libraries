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
#include <migraphx/onnx.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <numeric>
#include <random>

#include <test.hpp>

TEST_CASE(multinomial_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    size_t sample_size = 100000;
    float seed         = 0.0f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> rand_samples(sample_size);
    std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });
    migraphx::shape rs{migraphx::shape::float_type, {1, sample_size}};
    auto rs_lit = mm->add_literal(migraphx::literal{rs, rand_samples});

    migraphx::shape s{migraphx::shape::float_type, {1, 5}};
    std::vector<int> dist{15, 25, 15, 25, 20};
    std::vector<float> data(5);
    std::vector<float> sum(5);
    // convert to float
    std::transform(dist.begin(), dist.end(), data.begin(), [&](auto d) { return d; });
    // take cumulative sum
    std::partial_sum(data.begin(), data.end(), sum.begin(), std::plus<float>());
    // scale probabilities arbitrarily
    float odd_scale = 10000.;
    std::transform(sum.begin(), sum.end(), data.begin(), [&](auto d) { return d * odd_scale; });

    auto input = mm->add_literal(migraphx::literal(s, data));

    mm->add_instruction(migraphx::make_op("multinomial"), input, rs_lit);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    // result_vec contains an index, or category label, for each random input value
    std::vector<int32_t> result_vec(sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // res_dist is a count, or histogram, of the number of samples in each category.  This is the
    // sampled distribution.
    std::vector<int> res_dist(5, 0);
    for(const auto& r : result_vec)
        res_dist[r]++;

    // To check the result, normalize the original probability distribution dist
    // and the sampling result res_dist; they should be close

    // Total the unnormalized probabilities
    auto dist_sum = std::accumulate(dist.begin(), dist.end(), 0);

    // Total the number of values returned
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);
    std::transform(dist.begin(), dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));
}

TEST_CASE(multinomial_dyn_test)
{
    // Invokes random_uniform and multinomial ops together, to verify the interface
    // Dynamic Batch dimension input of 2 means there are 2 different probability
    // distribution functions contained in Input_2
    migraphx::program p;
    auto* mm = p.get_main_module();

    size_t sample_size = 100000;
    size_t batch_size  = 2;

    //      Shape of the random data
    migraphx::shape rs{migraphx::shape::float_type, {{1, 2}, {2, sample_size + 1}}};
    auto input = mm->add_parameter("Input_1", rs);

    // Runtime randomization seed
    // To seed the random_uniform, we can provide a value by literal or input,
    // or ask the system to auto-seed with random_seed op.
    migraphx::shape seed_shape{migraphx::shape::uint32_type,
                               {migraphx::shape::dynamic_dimension{0, 1}}};
    auto seed_input = mm->add_parameter("Seed", seed_shape);

    // Shape of the probability distribution, which also defines the number of categories
    migraphx::shape s{migraphx::shape::float_type, {{2, 2}, {5, 6}}};

    // Unnormalized distributions for batch size 2:
    //   15, 25, 15, 15, 20
    //   20, 20, 10, 25, 25
    std::vector<int> dist{15, 25, 15, 25, 20, 20, 20, 10, 25, 25};
    // Hard-coded non-normalized, accumulated distribution follows:
    std::vector<float> data{.15f, .40f, .55f, .80f, 1.0f, 20.f, 40.f, 50.f, 75.f, 100.f};

    auto input2 = mm->add_parameter("Input_2", s);

    auto randoms = mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, input);
    mm->add_instruction(migraphx::make_op("multinomial"), input2, randoms);

    p.compile(migraphx::make_target("ref"));

    // Create a dummy input in the shape we want for the random data
    std::vector<float> dummy(sample_size, 0);
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {batch_size, sample_size}};
    migraphx::shape input_fixed_shape2{migraphx::shape::float_type, {batch_size, 5}};
    migraphx::parameter_map params0;
    params0["Input_1"] = migraphx::argument(input_fixed_shape1, dummy.data());

    migraphx::shape seed_fixed_shape{migraphx::shape::uint32_type, {1}};
    std::vector<uint32_t> seed_data = {4};
    params0["Seed"]                 = migraphx::argument(seed_fixed_shape, seed_data.data());

    params0["Input_2"] = migraphx::argument(input_fixed_shape2, data.data());
    auto result        = p.eval(params0).back();

    std::vector<float> result_vec(input_fixed_shape2.elements());
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Make a categorical histogram of output
    std::vector<int> res_dist(5, 0);
    size_t r = 0;
    for(r = 0; r < result_vec.size() / 2; r++)
        res_dist[result_vec[r]]++;

    //  histogram for second set of batch
    std::vector<int> res_dist2(5, 0);
    for(; r < result_vec.size(); r++)
        res_dist2[result_vec[r]]++;

    // Rescale or normalize both the input probability distribution and the output
    // histogram, and compare.  Should be close but not identical.
    auto dist_sum     = std::accumulate(dist.begin(), dist.begin() + 5, 0);
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);

    std::transform(dist.begin(), dist.begin() + 5, norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));

    // Do the same rescaling for the 2nd in batch, which has a different probability distribution
    dist_sum     = std::accumulate(dist.begin() + 5, dist.end(), 0);
    res_dist_sum = std::accumulate(res_dist2.begin(), res_dist2.end(), 0);
    std::transform(dist.begin() + 5, dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist2.begin(), res_dist2.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));
}

TEST_CASE(multinomial_float_dyn_test)
{
    // int data type for random_uniform op and float data type for multinomial.

    migraphx::program p;
    auto* mm = p.get_main_module();

    size_t sample_size = 100000;
    size_t batch_size  = 2;

    //      Shape of the random data
    migraphx::shape rs{migraphx::shape::int32_type, {{1, 2}, {2, sample_size + 1}}};
    auto input = mm->add_parameter("Input_1", rs);

    // Runtime randomization seed
    // To seed the random_uniform, we can provide a value by literal or input,
    // or ask the system to auto-seed with random_seed op.
    migraphx::shape seed_shape{migraphx::shape::uint32_type,
                               {migraphx::shape::dynamic_dimension{0, 1}}};
    auto seed_input = mm->add_parameter("Seed", seed_shape);

    // Shape of the probability distribution, which also defines the number of categories
    migraphx::shape s{migraphx::shape::float_type, {{2, 2}, {5, 6}}};

    // Unnormalized distributions for batch size 2:
    //   15, 25, 15, 15, 20
    //   20, 20, 10, 25, 25
    std::vector<int> dist{15, 25, 15, 25, 20, 20, 20, 10, 25, 25};
    // Hard-coded normalized, accumulated distribution follows:
    std::vector<float> data{.15f, .40f, .55f, .80f, 1.0f, .20f, .40f, .50f, .75f, 1.0f};

    auto input2 = mm->add_parameter("Input_2", s);

    auto randoms = mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, input);
    mm->add_instruction(migraphx::make_op("multinomial", {{"dtype", migraphx::shape::float_type}}),
                        input2,
                        randoms);

    p.compile(migraphx::make_target("ref"));

    // Create a dummy input in the shape we want for the random data
    std::vector<float> dummy(sample_size, 0);
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {batch_size, sample_size}};
    migraphx::shape input_fixed_shape2{migraphx::shape::float_type, {batch_size, 5}};
    migraphx::parameter_map params0;
    params0["Input_1"] = migraphx::argument(input_fixed_shape1, dummy.data());

    migraphx::shape seed_fixed_shape{migraphx::shape::uint32_type, {1}};
    std::vector<uint32_t> seed_data = {4};
    params0["Seed"]                 = migraphx::argument(seed_fixed_shape, seed_data.data());

    params0["Input_2"] = migraphx::argument(input_fixed_shape2, data.data());
    auto result        = p.eval(params0).back();

    std::vector<float> result_vec(input_fixed_shape2.elements());
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Make a categorical histogram of output
    std::vector<int> res_dist(5, 0);
    size_t r = 0;
    for(r = 0; r < result_vec.size() / 2; r++)
        res_dist[result_vec[r]]++;

    //  histogram for second set of batch
    std::vector<int> res_dist2(5, 0);
    for(; r < result_vec.size(); r++)
        res_dist2[result_vec[r]]++;

    // Rescale or normalize both the input probability distribution and the output
    // histogram, and compare.  Should be close but not identical.
    auto dist_sum     = std::accumulate(dist.begin(), dist.begin() + 5, 0);
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);

    std::transform(dist.begin(), dist.begin() + 5, norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));

    // Do the same rescaling for the 2nd in batch, which has a different probability distribution
    dist_sum     = std::accumulate(dist.begin() + 5, dist.end(), 0);
    res_dist_sum = std::accumulate(res_dist2.begin(), res_dist2.end(), 0);
    std::transform(dist.begin() + 5, dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist2.begin(), res_dist2.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));
}
