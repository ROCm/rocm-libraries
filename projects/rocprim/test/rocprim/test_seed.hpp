// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TEST_SEED_HPP_
#define TEST_SEED_HPP_

#include <random>

#include "../../common/utils.hpp"

using engine_type = std::default_random_engine;
using seed_type   = typename engine_type::result_type;

static const char* env_p = common::__get_env("ROCPRIM_TEST_RUNS");
// "env_var" determines the number of iterations.
// If undefined or incorrectly defined, it defaults to 0.
static const size_t env_var = (env_p == nullptr) ? 0ul : std::atoi(env_p);

// Predefined seeds.
static constexpr seed_type seeds[]   = {0, 1997132004};
static constexpr size_t    seed_size = sizeof(seeds) / sizeof(seeds[0]);

// Calculate the number of random seeds.
// Always at least 1, or (env_var - seed_size) if env_var exceeds seed_size.
static const size_t random_seeds_count = (env_var > seed_size) ? (env_var - seed_size) : 1;

// Calculate the total number of runs.
// Default to random_seeds_count + seed_size if env_var is 0, otherwise use env_var.
static const size_t number_of_runs = (env_var == 0) ? (random_seeds_count + seed_size) : env_var;

#endif // TEST_SEED_HPP_
