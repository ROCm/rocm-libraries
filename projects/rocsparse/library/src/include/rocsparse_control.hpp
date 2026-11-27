/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse_argdescr.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_hip.hpp"
#include "rocsparse_message.hpp"
#include <exception>
#include <iostream>

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

//
// @brief Macros for coverage exclusion
//
#define ROCSPARSE_COV_EXCL_START (void)("LCOV_EXCL_START")
#define ROCSPARSE_COV_EXCL_STOP (void)("LCOV_EXCL_STOP")

namespace rocsparse
{
    rocsparse_status exception_to_rocsparse_status(std::exception_ptr e = std::current_exception());
    /*******************************************************************************
     * \brief convert hipError_t to rocsparse_status
     ******************************************************************************/
    rocsparse_status get_rocsparse_status_for_hip_status(hipError_t status);

    /*******************************************************************************
     * \brief log a hip error and return the matching rocsparse_status.
     * The message is "[prefix, ]hip error detected: ...".
     * Kept out of line so the *_IF_HIP_ERROR macros expand to a compare and a call.
     ******************************************************************************/
    rocsparse_status log_hip_error(
        hipError_t status, const char* prefix, const char* function, const char* file, int line);
}

//
//
//
#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                               \
    do                                                                            \
    {                                                                             \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);         \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                    \
        {                                                                         \
            return rocsparse::log_hip_error(                                      \
                TMP_STATUS_FOR_CHECK, nullptr, __FUNCTION__, __FILE__, __LINE__); \
        }                                                                         \
    } while(false)

#define RETURN_WITH_MESSAGE_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)           \
    do                                                                              \
    {                                                                               \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);           \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                      \
        {                                                                           \
            return rocsparse::log_hip_error(                                        \
                TMP_STATUS_FOR_CHECK, (MESSAGE), __FUNCTION__, __FILE__, __LINE__); \
        }                                                                           \
    } while(false)

//
//
//

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                \
    do                                                                            \
    {                                                                             \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);         \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                    \
        {                                                                         \
            throw rocsparse::log_hip_error(                                       \
                TMP_STATUS_FOR_CHECK, nullptr, __FUNCTION__, __FILE__, __LINE__); \
        }                                                                         \
    } while(false)

#define THROW_WITH_MESSAGE_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)            \
    do                                                                              \
    {                                                                               \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);           \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                      \
        {                                                                           \
            throw rocsparse::log_hip_error(                                         \
                TMP_STATUS_FOR_CHECK, (MESSAGE), __FUNCTION__, __FILE__, __LINE__); \
        }                                                                           \
    } while(false)

//
//
//
#define WARNING_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                      \
    do                                                                    \
    {                                                                     \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                            \
        {                                                                 \
            ROCSPARSE_WARNING_MESSAGE("hip error detected");              \
        }                                                                 \
    } while(false)

#define FORWARD_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                              \
    do                                                                            \
    {                                                                             \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);         \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                    \
        {                                                                         \
            rocsparse::log_hip_error(                                             \
                TMP_STATUS_FOR_CHECK, nullptr, __FUNCTION__, __FILE__, __LINE__); \
            return TMP_STATUS_FOR_CHECK;                                          \
        }                                                                         \
    } while(false)

#define FORWARD_WITH_MESSAGE_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)          \
    do                                                                              \
    {                                                                               \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);           \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                      \
        {                                                                           \
            rocsparse::log_hip_error(                                               \
                TMP_STATUS_FOR_CHECK, (MESSAGE), __FUNCTION__, __FILE__, __LINE__); \
            return TMP_STATUS_FOR_CHECK;                                            \
        }                                                                           \
    } while(false)

//
//
//
#define RETURN_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                       \
    do                                                                          \
    {                                                                           \
        const rocsparse_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)                    \
        {                                                                       \
            ROCSPARSE_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK, "none");              \
            return TMP_STATUS_FOR_CHECK;                                        \
        }                                                                       \
    } while(false)

#define WARNING_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                          \
    do                                                                              \
    {                                                                               \
        const rocsparse_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);     \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)                        \
        {                                                                           \
            std::stringstream s;                                                    \
            s << "rocsparse error detected: code '" << TMP_STATUS_FOR_CHECK << "'"; \
            ROCSPARSE_WARNING_MESSAGE(s.str().c_str());                             \
        }                                                                           \
    } while(false)

//
//
//
#define THROW_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                             \
    do                                                                               \
    {                                                                                \
        const rocsparse_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);      \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)                         \
        {                                                                            \
            ROCSPARSE_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK,                            \
                                    "rocSPARSE error detected, throwing exception"); \
            throw TMP_STATUS_FOR_CHECK;                                              \
        }                                                                            \
    } while(false)

//
//
//
#define RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK, MSG)     \
    do                                                                          \
    {                                                                           \
        const rocsparse_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)                    \
        {                                                                       \
            ROCSPARSE_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK, MSG);                 \
            return TMP_STATUS_FOR_CHECK;                                        \
        }                                                                       \
    } while(false)

//
//
//
#define THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)  \
    do                                                                          \
    {                                                                           \
        const rocsparse_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)                    \
        {                                                                       \
            ROCSPARSE_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK, MESSAGE);             \
            throw TMP_STATUS_FOR_CHECK;                                         \
        }                                                                       \
    } while(false)

//
//
//
#define RETURN_ROCSPARSE_EXCEPTION()                                                    \
    do                                                                                  \
    {                                                                                   \
        const rocsparse_status TMP_STATUS = rocsparse::exception_to_rocsparse_status(); \
        ROCSPARSE_ERROR_MESSAGE(TMP_STATUS, "exception detected");                      \
        return TMP_STATUS;                                                              \
    } while(false)

//
//
//
#define THROW_IF_HIPLAUNCHKERNELGGL_ERROR(K, G, T, M, S, ...)                                     \
    do                                                                                            \
    {                                                                                             \
        const bool TMP_DEBUG_KERNEL_LAUNCH = rocsparse_debug_variables.get_debug_kernel_launch(); \
        if(TMP_DEBUG_KERNEL_LAUNCH)                                                               \
        {                                                                                         \
            THROW_WITH_MESSAGE_IF_HIP_ERROR(hipGetLastError(), "prior to hipLaunchKernelGGL");    \
        }                                                                                         \
        rocsparse_hipLaunchKernelGGL((K), (G), (T), (M), (S), __VA_ARGS__);                       \
        if(TMP_DEBUG_KERNEL_LAUNCH)                                                               \
        {                                                                                         \
            THROW_IF_HIP_ERROR(hipGetLastError());                                                \
        }                                                                                         \
    } while(false)

#define RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(K, G, T, M, S, ...)                                    \
    do                                                                                            \
    {                                                                                             \
        const bool TMP_DEBUG_KERNEL_LAUNCH = rocsparse_debug_variables.get_debug_kernel_launch(); \
        if(TMP_DEBUG_KERNEL_LAUNCH)                                                               \
        {                                                                                         \
            RETURN_WITH_MESSAGE_IF_HIP_ERROR(hipGetLastError(), "prior to hipLaunchKernelGGL");   \
        }                                                                                         \
        rocsparse_hipLaunchKernelGGL((K), (G), (T), (M), (S), __VA_ARGS__);                       \
        if(TMP_DEBUG_KERNEL_LAUNCH)                                                               \
        {                                                                                         \
            RETURN_IF_HIP_ERROR(hipGetLastError());                                               \
        }                                                                                         \
    } while(false)

#define RETURN_ROCSPARSE_ERROR_IF(STATUS, INPUT_STATUS_FOR_CHECK)     \
    do                                                                \
    {                                                                 \
        if(INPUT_STATUS_FOR_CHECK)                                    \
        {                                                             \
            ROCSPARSE_ERROR_MESSAGE(STATUS, #INPUT_STATUS_FOR_CHECK); \
            return STATUS;                                            \
        }                                                             \
    } while(false)

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                \
    {                                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                 \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                    \
        {                                                                         \
            rocsparse::log_hip_error(                                             \
                TMP_STATUS_FOR_CHECK, nullptr, __FUNCTION__, __FILE__, __LINE__); \
        }                                                                         \
    }

#define PRINT_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)         \
    {                                                                     \
        rocsparse_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)              \
        {                                                                 \
            ROCSPARSE_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK, MESSAGE);       \
        }                                                                 \
    }
