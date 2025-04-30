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
