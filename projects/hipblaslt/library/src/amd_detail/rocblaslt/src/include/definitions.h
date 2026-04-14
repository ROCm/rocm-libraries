/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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
 * ************************************************************************ */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "status.h"

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

//
// @brief Macros for coverage exclusion
//
#define ROCBLASLT_COV_EXCL_START (void)("LCOV_EXCL_START")
#define ROCBLASLT_COV_EXCL_STOP (void)("LCOV_EXCL_STOP")

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                           \
    {                                                                         \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;             \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                \
        {                                                                     \
            return get_rocblaslt_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                     \
    }

#define RETURN_IF_ROCBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocblaslt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblaslt_status_success)            \
        {                                                               \
            return TMP_STATUS_FOR_CHECK;                                \
        }                                                               \
    }

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                           \
    {                                                                        \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;            \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                               \
        {                                                                    \
            throw get_rocblaslt_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                    \
    }

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

#define RETURN_IF_INVALID_HANDLE(HANDLE)            \
    {                                               \
        if(HANDLE == nullptr)                       \
        {                                           \
            return rocblaslt_status_invalid_handle; \
        }                                           \
    }

#define RETURN_IF_NULLPTR(PTR)                       \
    {                                                \
        if(PTR == nullptr)                           \
        {                                            \
            return rocblaslt_status_invalid_pointer; \
        }                                            \
    }

#endif // DEFINITIONS_H
