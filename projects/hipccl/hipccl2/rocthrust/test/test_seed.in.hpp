/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef TEST_SEED_HPP_
#define TEST_SEED_HPP_

#include <random>
#include <initializer_list>

using random_engine = std::minstd_rand;
using seed_type = random_engine::result_type;

static constexpr size_t rng_seed_count = ${RNG_SEED_COUNT};
static const std::initializer_list<uint32_t> prng_seeds = { ${PRNG_SEEDS_INITIALIZER} };
static constexpr seed_type seed_value_addition = 100;

#endif // TEST_SEED_HPP_
