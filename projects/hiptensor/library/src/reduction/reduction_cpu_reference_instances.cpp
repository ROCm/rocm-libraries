/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "reduction_cpu_reference_instances.hpp"
#include "reduction_cpu_reference_impl.hpp"
#include <hiptensor/internal/types.hpp>

#define REG_CPU_SOLUTION(dim_count, reduced_dim_count, type, computeType) \
    registerSolutions(enumerateReferenceSolutions<type,                   \
                                                  computeType,            \
                                                  type,                   \
                                                  dim_count,              \
                                                  reduced_dim_count,      \
                                                  HIPTENSOR_OP_ADD,       \
                                                  true,                   \
                                                  false>());              \
    registerSolutions(enumerateReferenceSolutions<type,                   \
                                                  computeType,            \
                                                  type,                   \
                                                  dim_count,              \
                                                  reduced_dim_count,      \
                                                  HIPTENSOR_OP_MUL,       \
                                                  true,                   \
                                                  false>());              \
    registerSolutions(enumerateReferenceSolutions<type,                   \
                                                  computeType,            \
                                                  type,                   \
                                                  dim_count,              \
                                                  reduced_dim_count,      \
                                                  HIPTENSOR_OP_MIN,       \
                                                  true,                   \
                                                  false>());              \
    registerSolutions(enumerateReferenceSolutions<type,                   \
                                                  computeType,            \
                                                  type,                   \
                                                  dim_count,              \
                                                  reduced_dim_count,      \
                                                  HIPTENSOR_OP_MAX,       \
                                                  true,                   \
                                                  false>());

namespace hiptensor
{
    ReductionCpuReferenceInstances::ReductionCpuReferenceInstances()
    {
        REG_CPU_SOLUTION(1, 1, ck::half_t, float);
        REG_CPU_SOLUTION(2, 1, ck::half_t, float);
        REG_CPU_SOLUTION(2, 2, ck::half_t, float);
        REG_CPU_SOLUTION(3, 1, ck::half_t, float);
        REG_CPU_SOLUTION(3, 2, ck::half_t, float);
        REG_CPU_SOLUTION(3, 3, ck::half_t, float);
        REG_CPU_SOLUTION(4, 1, ck::half_t, float);
        REG_CPU_SOLUTION(4, 2, ck::half_t, float);
        REG_CPU_SOLUTION(4, 3, ck::half_t, float);
        REG_CPU_SOLUTION(4, 4, ck::half_t, float);
        REG_CPU_SOLUTION(5, 1, ck::half_t, float);
        REG_CPU_SOLUTION(5, 2, ck::half_t, float);
        REG_CPU_SOLUTION(5, 3, ck::half_t, float);
        REG_CPU_SOLUTION(5, 4, ck::half_t, float);
        REG_CPU_SOLUTION(5, 5, ck::half_t, float);
        REG_CPU_SOLUTION(6, 1, ck::half_t, float);
        REG_CPU_SOLUTION(6, 2, ck::half_t, float);
        REG_CPU_SOLUTION(6, 3, ck::half_t, float);
        REG_CPU_SOLUTION(6, 4, ck::half_t, float);
        REG_CPU_SOLUTION(6, 5, ck::half_t, float);
        REG_CPU_SOLUTION(6, 6, ck::half_t, float);

        REG_CPU_SOLUTION(1, 1, ck::bhalf_t, float);
        REG_CPU_SOLUTION(2, 1, ck::bhalf_t, float);
        REG_CPU_SOLUTION(2, 2, ck::bhalf_t, float);
        REG_CPU_SOLUTION(3, 1, ck::bhalf_t, float);
        REG_CPU_SOLUTION(3, 2, ck::bhalf_t, float);
        REG_CPU_SOLUTION(3, 3, ck::bhalf_t, float);
        REG_CPU_SOLUTION(4, 1, ck::bhalf_t, float);
        REG_CPU_SOLUTION(4, 2, ck::bhalf_t, float);
        REG_CPU_SOLUTION(4, 3, ck::bhalf_t, float);
        REG_CPU_SOLUTION(4, 4, ck::bhalf_t, float);
        REG_CPU_SOLUTION(5, 1, ck::bhalf_t, float);
        REG_CPU_SOLUTION(5, 2, ck::bhalf_t, float);
        REG_CPU_SOLUTION(5, 3, ck::bhalf_t, float);
        REG_CPU_SOLUTION(5, 4, ck::bhalf_t, float);
        REG_CPU_SOLUTION(5, 5, ck::bhalf_t, float);
        REG_CPU_SOLUTION(6, 1, ck::bhalf_t, float);
        REG_CPU_SOLUTION(6, 2, ck::bhalf_t, float);
        REG_CPU_SOLUTION(6, 3, ck::bhalf_t, float);
        REG_CPU_SOLUTION(6, 4, ck::bhalf_t, float);
        REG_CPU_SOLUTION(6, 5, ck::bhalf_t, float);
        REG_CPU_SOLUTION(6, 6, ck::bhalf_t, float);

        REG_CPU_SOLUTION(1, 1, float, float);
        REG_CPU_SOLUTION(2, 1, float, float);
        REG_CPU_SOLUTION(2, 2, float, float);
        REG_CPU_SOLUTION(3, 1, float, float);
        REG_CPU_SOLUTION(3, 2, float, float);
        REG_CPU_SOLUTION(3, 3, float, float);
        REG_CPU_SOLUTION(4, 1, float, float);
        REG_CPU_SOLUTION(4, 2, float, float);
        REG_CPU_SOLUTION(4, 3, float, float);
        REG_CPU_SOLUTION(4, 4, float, float);
        REG_CPU_SOLUTION(5, 1, float, float);
        REG_CPU_SOLUTION(5, 2, float, float);
        REG_CPU_SOLUTION(5, 3, float, float);
        REG_CPU_SOLUTION(5, 4, float, float);
        REG_CPU_SOLUTION(5, 5, float, float);
        REG_CPU_SOLUTION(6, 1, float, float);
        REG_CPU_SOLUTION(6, 2, float, float);
        REG_CPU_SOLUTION(6, 3, float, float);
        REG_CPU_SOLUTION(6, 4, float, float);
        REG_CPU_SOLUTION(6, 5, float, float);
        REG_CPU_SOLUTION(6, 6, float, float);

        REG_CPU_SOLUTION(1, 1, double, double);
        REG_CPU_SOLUTION(2, 1, double, double);
        REG_CPU_SOLUTION(2, 2, double, double);
        REG_CPU_SOLUTION(3, 1, double, double);
        REG_CPU_SOLUTION(3, 2, double, double);
        REG_CPU_SOLUTION(3, 3, double, double);
        REG_CPU_SOLUTION(4, 1, double, double);
        REG_CPU_SOLUTION(4, 2, double, double);
        REG_CPU_SOLUTION(4, 3, double, double);
        REG_CPU_SOLUTION(4, 4, double, double);
        REG_CPU_SOLUTION(5, 1, double, double);
        REG_CPU_SOLUTION(5, 2, double, double);
        REG_CPU_SOLUTION(5, 3, double, double);
        REG_CPU_SOLUTION(5, 4, double, double);
        REG_CPU_SOLUTION(5, 5, double, double);
        REG_CPU_SOLUTION(6, 1, double, double);
        REG_CPU_SOLUTION(6, 2, double, double);
        REG_CPU_SOLUTION(6, 3, double, double);
        REG_CPU_SOLUTION(6, 4, double, double);
        REG_CPU_SOLUTION(6, 5, double, double);
        REG_CPU_SOLUTION(6, 6, double, double);
    }
} // namespace hiptensor
