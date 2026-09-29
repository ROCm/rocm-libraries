/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "client_utility.hpp"

#include "d_vector.hpp"

#include "device_batch_matrix.hpp"
#include "device_matrix.hpp"
#include "device_multiple_strided_batch_matrix.hpp"
#include "device_strided_batch_matrix.hpp"

#include "host_batch_matrix.hpp"
#include "host_matrix.hpp"
#include "host_multiple_strided_batch_matrix.hpp"
#include "host_strided_batch_matrix.hpp"
#include "rocblas_init.hpp"

//!
//! @brief Initialize a host_strided_batch_matrix.
//! @param hA The host_strided_batch_matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T, bool altInit = false>
inline void rocblas_init_matrix(host_strided_batch_matrix<T>& hA,
                                const Arguments&              arg,
                                rocblas_check_nan_init        nan_init,
                                rocblas_check_matrix_type     matrix_type,
                                bool                          seedReset        = false,
                                bool                          alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int_zero_one)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_zero_one_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::denorm)
    {
        if(altInit)
            rocblas_init_alt_impl_small<T>(hA);
        else
            rocblas_init_alt_impl_big<T>(hA);
    }
    else if(arg.initialization == rocblas_initialization::denorm2)
    {
        if(altInit)
            rocblas_init_non_rep_bf16_vals<T>(hA);
        else
            rocblas_init_identity<T>(hA);
    }
    else if(arg.initialization == rocblas_initialization::zero)
    {
        rocblas_init_matrix_zero<T>(hA);
    }
    else
    {
#ifdef GOOGLE_TEST
        FAIL() << "unknown initialization type";
        return;
#else
        rocblas_cerr << "unknown initialization type" << std::endl;
        rocblas_abort();
#endif
    }
}

//!
//! @brief Initialize a host_batch_matrix.
//! @param hA The host_batch_matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T, bool altInit = false>
inline void rocblas_init_matrix(host_batch_matrix<T>&     hA,
                                const Arguments&          arg,
                                rocblas_check_nan_init    nan_init,
                                rocblas_check_matrix_type matrix_type,
                                bool                      seedReset        = false,
                                bool                      alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int_zero_one)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_zero_one_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::denorm)
    {
        if(altInit)
            rocblas_init_alt_impl_small<T>(hA);
        else
            rocblas_init_alt_impl_big<T>(hA);
    }
    else if(arg.initialization == rocblas_initialization::denorm2)
    {
        if(altInit)
            rocblas_init_non_rep_bf16_vals<T>(hA);
        else
            rocblas_init_identity<T>(hA);
    }
    else if(arg.initialization == rocblas_initialization::zero)
    {
        rocblas_init_matrix_zero<T>(hA);
    }
    else
    {
#ifdef GOOGLE_TEST
        FAIL() << "unknown initialization type";
        return;
#else
        rocblas_cerr << "unknown initialization type" << std::endl;
        rocblas_abort();
#endif
    }
}

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T, bool altInit = false>
inline void rocblas_init_matrix(host_matrix<T>&           hA,
                                const Arguments&          arg,
                                rocblas_check_nan_init    nan_init,
                                rocblas_check_matrix_type matrix_type,
                                bool                      seedReset        = false,
                                bool                      alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int_zero_one)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_zero_one_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::denorm)
    {
        if(altInit)
            rocblas_init_alt_impl_small<T>(hA);
        else
            rocblas_init_alt_impl_big<T>(hA);
    }
    else if(arg.initialization == rocblas_initialization::denorm2)
    {
        if(altInit)
            rocblas_init_non_rep_bf16_vals<T>(hA);
        else
            rocblas_init_identity<T>(hA);
    }
    else if(arg.initialization == rocblas_initialization::zero)
    {
        rocblas_init_matrix_zero<T>(hA);
    }
    else
    {
#ifdef GOOGLE_TEST
        FAIL() << "unknown initialization type";
        return;
#else
        rocblas_cerr << "unknown initialization type" << std::endl;
        rocblas_abort();
#endif
    }
}

//!
//! @brief Largest absolute element value an initialization pattern can produce for type T.
//! @details The rocblas_init_matrix overloads above fill from fixed, known ranges, so the
//! bound is a constant per (init, type) pair. Complex entries take both components from the
//! same range, so the magnitude bound is sqrt(2) * component bound.
//! Keep this in sync with the generators (random_generator, random_hpl_generator,
//! random_zero_one_generator in rocblas_random.hpp; trig sin/cos in rocblas_init.hpp) that
//! rocblas_init_matrix dispatches to above.
//!
template <typename T>
inline double init_abs_bound(rocblas_initialization init)
{
    double comp; // bound on a single (real) component
    switch(init)
    {
    case rocblas_initialization::rand_int:
        // float/double: [1,10]; half/bfloat16: [-2,2]; int8: [1,3]
        if(std::is_same_v<T, rocblas_half> || std::is_same_v<T, rocblas_bfloat16>)
            comp = 2.0;
        else if(std::is_same_v<T, int8_t>)
            comp = 3.0;
        else
            comp = 10.0;
        break;
    case rocblas_initialization::hpl:
        comp = 0.5; // [-0.5, 0.5]
        break;
    case rocblas_initialization::trig_float:
    case rocblas_initialization::rand_int_zero_one:
        comp = 1.0; // sin/cos in [-1,1]; zero_one in [0,1]
        break;
    case rocblas_initialization::zero:
        comp = 0.0;
        break;
    default:
        // denorm and any future pattern: fall back to a safe unit bound.
        comp = 1.0;
        break;
    }
    return rocblas_is_complex<T> ? comp * 1.4142135623730951 : comp;
}

//!
//! @brief Analytical bound on |D|_max for D = alpha*op(A)*op(B) + beta*C, derived from the
//! init ranges of A, B and C.
//! @details TODO: this is a worst-case bound (assumes every product hits max and all K terms
//! add coherently). Real random sums grow ~sqrt(K), so it is generous; fine for a localized
//! tolerance in the solutions test, but could be tightened if reused for tighter checks.
//!
template <typename Ti, typename To, typename Tc>
inline double gemm_result_abs_bound(rocblas_initialization init, int64_t K, Tc alpha, Tc beta)
{
    const double a = init_abs_bound<Ti>(init);
    const double b = init_abs_bound<Ti>(init);
    const double c = init_abs_bound<To>(init);
    return double(rocblas_abs(alpha)) * double(K) * a * b + double(rocblas_abs(beta)) * c;
}

//!
//! @brief Initialize a device matrix.
//! @param dA The device matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T, bool altInit = false>
void rocblas_init_matrix(rocblas_handle                  handle,
                         device_strided_batch_matrix<T>& dA,
                         const Arguments&                arg,
                         rocblas_check_nan_init          nan_init,
                         rocblas_check_matrix_type       matrix_type,
                         bool                            seedReset        = false,
                         bool                            alternating_sign = false);
