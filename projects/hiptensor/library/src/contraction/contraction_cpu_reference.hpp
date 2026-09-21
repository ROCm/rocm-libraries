/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <hip/library_types.h>
#include <vector>

#include <hiptensor/hiptensor.h>

#include "contraction_solution.hpp"

HIPTENSOR_EXPORT hiptensorStatus_t
    hiptensorContractionReference(const hiptensorPlan_t                 plan,
                                  void const*                           alpha,
                                  void const*                           A,
                                  void const*                           B,
                                  void const*                           beta,
                                  void const*                           C,
                                  void*                                 D,
                                  std::vector<size_t> const&            a_ms_ks_lengths,
                                  std::vector<size_t> const&            a_ms_ks_strides,
                                  std::vector<int32_t> const&           a_ms_ks_modes,
                                  std::vector<size_t> const&            b_ks_ns_lengths,
                                  std::vector<size_t> const&            b_ks_ns_strides,
                                  std::vector<int32_t> const&           b_ks_ns_modes,
                                  std::vector<size_t> const&            c_ms_ns_lengths,
                                  std::vector<size_t> const&            c_ms_ns_strides,
                                  std::vector<int32_t> const&           c_ms_ns_modes,
                                  std::vector<size_t> const&            d_ms_ns_lengths,
                                  std::vector<size_t> const&            d_ms_ns_strides,
                                  std::vector<int32_t> const&           d_ms_ns_modes,
                                  hiptensorDataType_t                   typeA,
                                  hiptensorDataType_t                   typeB,
                                  hiptensorDataType_t                   typeC,
                                  hiptensorDataType_t                   typeD,
                                  hiptensor::ContractionUnaryOps const& unaryOps,
                                  void*                                 workspace);
