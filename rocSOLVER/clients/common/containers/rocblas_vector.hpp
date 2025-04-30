/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

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
    n = T(rocblas_nan_rng());
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
        rocblas_seedrand();
    }

    for(int64_t batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto batched_data = that[batch_index];
        auto inc = std::abs(that.inc());
        auto n = that.n();
        if(inc < 0)
        {
            batched_data -= (n - 1) * inc;
        }

        for(int64_t i = 0; i < n; ++i)
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
        rocblas_seedrand();
    }

    for(int64_t batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto batched_data = that[batch_index];
        auto inc = std::abs(that.inc());
        auto n = that.n();
        if(inc < 0)
        {
            batched_data -= (n - 1) * inc;
        }

        for(int64_t i = 0; i < n; ++i)
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
