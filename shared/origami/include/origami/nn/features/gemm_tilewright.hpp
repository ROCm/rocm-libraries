/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

#include <cstddef>

namespace origami::nn::features::gemm_tilewright {

constexpr std::size_t query_dim       = 55;
constexpr std::size_t item_dim        = 12;
constexpr std::size_t interaction_dim = 37;

constexpr const char* catalog_id            = "gemm_tilewright";
constexpr const char* feature_names_hash    = "e7fe4b524851e895";

void ORIGAMI_EXPORT build_query(const problem_t& problem, const hardware_t& hardware, float* out);

void ORIGAMI_EXPORT build_item(const config_t& config, float* out);

void ORIGAMI_EXPORT build_interaction(const problem_t& problem,
                       const config_t& config,
                       const hardware_t& hardware,
                       float* out);

}  // namespace origami::nn::features::gemm_tilewright
