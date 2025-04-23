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
#include <random>

#include <test.hpp>

/**
 * Reference test for the random_uniform operation.  Also invokes the random_seed operation.
 */

TEST_CASE(random_uniform_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    uint64_t seed(0);
    size_t sample_size(200);

    //      Shape of the random data
    migraphx::shape rs{migraphx::shape::float_type, {1, sample_size}};

    // data tensor must be allocated at this point but does not need to be initialized.
    std::vector<float> data(sample_size);
    auto input = mm->add_literal(migraphx::literal(rs, data));

    // Runtime randomization seed
    migraphx::shape seed_shape{migraphx::shape::uint64_type, {1}};
    std::vector<uint64_t> seed_data{seed};
    auto seed_input = mm->add_literal(migraphx::literal(seed_shape, seed_data));

    mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, input);
    p.compile(migraphx::make_target("ref"));

    // no params_map needed
    auto result = p.eval({}).back();
    std::vector<float> result_vec(sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Compare result with the STL's mt19937 generator
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> rand_samples(sample_size);
    std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });
    EXPECT(migraphx::verify::verify_range_with_tolerance(result_vec,
                                                         migraphx::verify::expected{rand_samples},
                                                         migraphx::verify::tolerance{0.00001}));
}

TEST_CASE(random_uniform_int_test)
{
    // random uniform distribution with an integer type input shape
    migraphx::program p;
    auto* mm = p.get_main_module();
    float seed(0.1);
    size_t sample_size(200);

    //      Shape of the random data
    migraphx::shape rs{migraphx::shape::uint16_type, {1, sample_size}};

    // data tensor must be allocated at this point but does not need to be initialized.
    std::vector<uint16_t> data(sample_size);
    auto input = mm->add_literal(migraphx::literal(rs, data));

    // Runtime randomization seed
    migraphx::shape seed_shape{migraphx::shape::float_type, {1}};
    std::vector<float> seed_data{seed};
    auto seed_input = mm->add_literal(migraphx::literal(seed_shape, seed_data));

    mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params0;
    auto result = p.eval(params0).back();
    std::vector<uint16_t> result_vec(sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Compare result with the STL's mt19937 generator
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint16_t> dis;
    std::vector<uint16_t> gold_rand_samples(sample_size);
    std::generate(gold_rand_samples.begin(), gold_rand_samples.end(), [&]() { return dis(gen); });
    EXPECT(migraphx::verify::verify_rms_range(result_vec, gold_rand_samples));
}

TEST_CASE(random_uniform_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    uint64_t seed(17);
    size_t sample_size(200);

    //      Shape of the random data
    migraphx::shape rs{migraphx::shape::float_type, {{1, 2}, {2, sample_size + 1}}};
    auto input = mm->add_parameter("Input_1", rs);

    // Runtime randomization seed
    migraphx::shape seed_shape{migraphx::shape::uint64_type, {1}};
    auto seed_input = mm->add_parameter("Seed", seed_shape);

    mm->add_instruction(migraphx::make_op("random_uniform", {}), seed_input, input);
    p.compile(migraphx::make_target("ref"));

    // Create a dummy input to hold the random data
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {sample_size}};

    migraphx::parameter_map params0;
    params0["Input_1"] = migraphx::argument(input_fixed_shape1);

    std::vector<uint64_t> seed_data = {seed};
    params0["Seed"]                 = migraphx::argument(seed_shape, seed_data.data());
    auto result                     = p.eval(params0).back();

    std::vector<float> result_vec(sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Compare result with the STL's mt19937 generator
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> gold_rand_samples(sample_size);
    std::generate(gold_rand_samples.begin(), gold_rand_samples.end(), [&]() { return dis(gen); });
    EXPECT(migraphx::verify::verify_rms_range(result_vec, gold_rand_samples));
}

TEST_CASE(random_uniform_and_seed_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    size_t sample_size(20000);

    //      Shape of the random data
    migraphx::shape rs{migraphx::shape::float_type, {{1, 2}, {2, sample_size + 1}}};
    auto input = mm->add_parameter("Input_1", rs);

    // Runtime randomization seed
    auto seed_input = mm->add_instruction(migraphx::make_op("random_seed"));
    mm->add_instruction(migraphx::make_op("random_uniform"), seed_input, input);
    p.compile(migraphx::make_target("ref"));

    // Create a dummy input to hold the random data
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {sample_size}};

    migraphx::parameter_map params0;
    params0["Input_1"] = migraphx::argument(input_fixed_shape1);
    auto result        = p.eval(params0).back();

    result.visit([&](auto output) { EXPECT(output.size() == sample_size); });
    // Do not check the content of the data since it's not repeatable
}
