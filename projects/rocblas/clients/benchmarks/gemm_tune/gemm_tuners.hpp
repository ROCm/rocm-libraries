/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#ifndef ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_NO_DEPRECATED_WARNINGS
#endif

#define ROCBLAS_BETA_FEATURES_API

#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_arguments.hpp"
#include "rocblas_data.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_parse_data.hpp"

template <typename Tc>
class GEMMTunerBase
{
public:
    GEMMTunerBase(const Arguments& arg);
    virtual ~GEMMTunerBase() {}

    int get_best_solution();

private:
    // These two methods marshall the GEMM calls depending on function type
    // They are both required for `get_best_solution()` to run
    virtual rocblas_status get_solutions(rocblas_int* solution_list, rocblas_int* size) = 0;
    virtual rocblas_status run_with_solution(int solution_idx)                          = 0;

protected:
    rocblas_local_handle m_handle;
    uint8_t              m_device;

    rocblas_int m_cold_iters;
    rocblas_int m_iters;

    rocblas_operation m_trans_A;
    rocblas_operation m_trans_B;
    rocblas_datatype  m_input_type;
    rocblas_datatype  m_output_type;
    rocblas_datatype  m_compute_type;

    Tc          m_alpha;
    Tc          m_beta;
    rocblas_int m_M;
    rocblas_int m_N;
    rocblas_int m_K;
    rocblas_int m_lda;
    rocblas_int m_ldb;
    rocblas_int m_ldc;
    rocblas_int m_A_row;
    rocblas_int m_A_col;
    rocblas_int m_B_row;
    rocblas_int m_B_col;
};

template <typename Ti, typename To = Ti, typename Tc = To>
class GEMMTunerEx : public GEMMTunerBase<Tc>
{
public:
    GEMMTunerEx(const Arguments& arg);

private:
    rocblas_status get_solutions(rocblas_int* solution_list, rocblas_int* size) override;
    rocblas_status run_with_solution(int solution_idx) override;

    device_matrix<Ti> m_dA;
    device_matrix<Ti> m_dB;
    device_matrix<To> m_dC;
};

template <typename Ti, typename To = Ti, typename Tc = To>
class GEMMTunerBatchedEx : public GEMMTunerBase<Tc>
{
public:
    GEMMTunerBatchedEx(const Arguments& arg);

private:
    rocblas_status get_solutions(rocblas_int* solution_list, rocblas_int* size) override;
    rocblas_status run_with_solution(int solution_idx) override;

    rocblas_int m_batch_count;

    device_batch_matrix<Ti> m_dA;
    device_batch_matrix<Ti> m_dB;
    device_batch_matrix<To> m_dC;
};

template <typename Tc>
class GEMMTunerStridedBase : public GEMMTunerBase<Tc>
{
public:
    GEMMTunerStridedBase(const Arguments& arg);

protected:
    rocblas_stride m_stride_a;
    rocblas_stride m_stride_b;
    rocblas_stride m_stride_c;
};

template <typename Ti, typename To = Ti, typename Tc = To>
class GEMMTunerStridedBatchedEx : public GEMMTunerStridedBase<Tc>
{
public:
    GEMMTunerStridedBatchedEx(const Arguments& arg);

private:
    rocblas_status get_solutions(rocblas_int* solution_list, rocblas_int* size) override;
    rocblas_status run_with_solution(int solution_idx) override;

    rocblas_int m_batch_count;

    device_strided_batch_matrix<Ti> m_dA;
    device_strided_batch_matrix<Ti> m_dB;
    device_strided_batch_matrix<To> m_dC;
};
