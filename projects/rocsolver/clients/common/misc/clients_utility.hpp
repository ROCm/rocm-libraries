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

#include <cstdio>
#include <iostream>
#include <rocblas/rocblas.h>
#include <string>
#include <type_traits>
#include <vector>

#include "rocblas_test.hpp"

/*!\file
 * \brief provide common utilities
 */

static constexpr char LIMITED_MEMORY_STRING[]
    = "Error: Attempting to allocate more memory than available.";

// TODO: This is dependent on internal gtest behaviour.
// Compared with result.message() when a test ended. Note that "Succeeded\n" is
// added to the beginning of the message automatically by gtest, so this must be
// compared.
static constexpr char LIMITED_MEMORY_STRING_GTEST[]
    = "Succeeded\nError: Attempting to allocate more memory than available.";

// Return the path to the client executable, for finding test matrices on filesystem
std::string rocsolver_exepath();

/* ============================================================================================
 */
/*! \brief  local handle which is automatically created and destroyed  */
class rocblas_local_handle
{
    rocblas_handle m_handle;

public:
    rocblas_local_handle()
    {
        rocblas_create_handle(&m_handle);
    }
    ~rocblas_local_handle()
    {
        rocblas_destroy_handle(m_handle);
    }

    rocblas_local_handle(const rocblas_local_handle&) = delete;
    rocblas_local_handle(rocblas_local_handle&&) = delete;
    rocblas_local_handle& operator=(const rocblas_local_handle&) = delete;
    rocblas_local_handle& operator=(rocblas_local_handle&&) = delete;

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&()
    {
        return m_handle;
    }
    operator const rocblas_handle&() const
    {
        return m_handle;
    }
};

/* ============================================================================================
 */
/*  device query and print out their ID and name */
rocblas_int query_device_property();

/*  set current device to device_id */
void set_device(rocblas_int device_id);

/* ============================================================================================
 */
template <typename T>
void print_strided_batched(const char* name,
                           T* A,
                           rocblas_int n1,
                           rocblas_int n2,
                           rocblas_int n3,
                           rocblas_int s1,
                           rocblas_int s2,
                           rocblas_int s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    std::string s = fmt::format("----------{}----------\n", name);
    int max_size = 8;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                s += fmt::format("{}|", A[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
            }
            s += '\n';
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            s += '\n';
    }
    std::fputs(s.c_str(), stdout);
    std::fflush(stdout);
}
