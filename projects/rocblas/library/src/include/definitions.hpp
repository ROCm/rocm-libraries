/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "macros.hpp"

/*******************************************************************************
 * Constants
 ******************************************************************************/
const int c_rocblas_default_solution   = 0; // either backend -1 is equivalent
const int c_rocblas_source_solution    = -2;
const int c_rocblas_solutions_reserved = 10;
const int c_rocblas_bad_solution_index = 0x7fffffff; // maxint

/*******************************************************************************
 * Definitions
 ******************************************************************************/
#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                      \
    do                                                                                   \
    {                                                                                    \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                        \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                           \
        {                                                                                \
            return rocblas_internal_convert_hip_to_rocblas_status(TMP_STATUS_FOR_CHECK); \
        }                                                                                \
    } while(0)

#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                 \
    do                                                                  \
    {                                                                   \
        rocblas_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)              \
        {                                                               \
            return TMP_STATUS_FOR_CHECK;                                \
        }                                                               \
    } while(0)

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                      \
    do                                                                                  \
    {                                                                                   \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                       \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                          \
        {                                                                               \
            throw rocblas_internal_convert_hip_to_rocblas_status(TMP_STATUS_FOR_CHECK); \
        }                                                                               \
    } while(0)

#define THROW_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                  \
    do                                                                  \
    {                                                                   \
        rocblas_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)              \
        {                                                               \
            throw TMP_STATUS_FOR_CHECK;                                 \
        }                                                               \
    } while(0)

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                                \
    do                                                                                            \
    {                                                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);                               \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                                    \
        {                                                                                         \
            rocblas_cerr << "hip error code: '" << hipGetErrorName(TMP_STATUS_FOR_CHECK)          \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
        }                                                                                         \
    } while(0)

#define PRINT_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                                            \
    do                                                                                            \
    {                                                                                             \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                             \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)                                        \
        {                                                                                         \
            rocblas_cerr << "rocblas error: '" << rocblas_status_to_string(TMP_STATUS_FOR_CHECK)  \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
        }                                                                                         \
    } while(0)

#define PRINT_AND_RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                                 \
    do                                                                                            \
    {                                                                                             \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                             \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)                                        \
        {                                                                                         \
            rocblas_cerr << "rocblas error: '" << rocblas_status_to_string(TMP_STATUS_FOR_CHECK)  \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
            return TMP_STATUS_FOR_CHECK;                                                          \
        }                                                                                         \
    } while(0)

#define PRINT_AND_RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                     \
    do                                                                                            \
    {                                                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);                               \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                                    \
        {                                                                                         \
            rocblas_cerr << "hip error code: '" << hipGetErrorName(TMP_STATUS_FOR_CHECK)          \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
            return rocblas_internal_convert_hip_to_rocblas_status(TMP_STATUS_FOR_CHECK);          \
        }                                                                                         \
    } while(0)
