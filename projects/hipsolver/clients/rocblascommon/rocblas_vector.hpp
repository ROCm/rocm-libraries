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
 *
 * ************************************************************************ */
#pragma once

#include "d_vector.hpp"

#include "device_batch_vector.hpp"
#include "device_strided_batch_vector.hpp"

#include "host_batch_vector.hpp"
#include "host_strided_batch_vector.hpp"

//!
//! @brief Random number with type deductions.
//!
template <typename T>
void random_generator(T& n)
{
    n = random_generator<T>();
}

//!
//!
//!
template <typename T>
void random_nan_generator(T& n)
{
    n = T(hipsolver_nan_rng());
}

//!
//! @brief Template for initializing a host
//! (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename U>
void rocblas_init_template(U& that, bool seedReset = false)
{
    if(seedReset)
    {
        hipsolver_seedrand();
    }

    for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto batched_data = that[batch_index];
        auto inc          = std::abs(that.inc());
        auto n            = that.n();
        if(inc < 0)
        {
            batched_data -= (n - 1) * inc;
        }

        for(rocblas_int i = 0; i < n; ++i)
        {
            random_generator(batched_data[i * inc]);
        }
    }
}

//!
//! @brief Template for initializing a host
//! (non_batched|batched|strided_batched)vector with NaNs.
//! @param that That vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename U>
void rocblas_init_nan_template(U& that, bool seedReset = false)
{
    if(seedReset)
    {
        hipsolver_seedrand();
    }

    for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto batched_data = that[batch_index];
        auto inc          = std::abs(that.inc());
        auto n            = that.n();
        if(inc < 0)
        {
            batched_data -= (n - 1) * inc;
        }

        for(rocblas_int i = 0; i < n; ++i)
        {
            random_nan_generator(batched_data[i * inc]);
        }
    }
}

//!
//! @brief Initialize a host_strided_batch_vector.
//! @param that The host strided batch vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
void rocblas_init(host_strided_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, seedReset);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host batch vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
void rocblas_init(host_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, seedReset);
}

//!
//! @brief Initialize a host_strided_batch_vector with NaNs.
//! @param that The host strided batch vector to be initialized.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
void rocblas_init_nan(host_strided_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_nan_template(that, seedReset);
}

//!
//! @brief Initialize a host_strided_batch_vector with NaNs.
//! @param that The host strided batch vector to be initialized.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
void rocblas_init_nan(host_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_nan_template(that, seedReset);
}
