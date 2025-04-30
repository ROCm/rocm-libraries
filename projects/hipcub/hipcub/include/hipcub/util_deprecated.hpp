/******************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_UTIL_DEPRECATED_HPP_
#define HIPCUB_UTIL_DEPRECATED_HPP_

#include "config.hpp"

#include <iostream>

#ifdef DOXYGEN_SHOULD_SKIP_THIS // Documentation only

    /// \def HIPCUB_IGNORE_DEPRECATED_API
    /// \brief If defined, warnings of deprecated API use are suppressed. If `CUB_IGNORE_DEPRECATED_API` or
    /// `THRUST_IGNORE_DEPRECATED_API` is defined, this is also defined automatically.
    #define HIPCUB_IGNORE_DEPRECATED_API

#endif // DOXYGEN_SHOULD_SKIP_THIS

#if defined(HIPCUB_CUB_API) && defined(HIPCUB_IGNORE_DEPRECATED_API) \
    && !defined(CUB_IGNORE_DEPRECATED_API)
    #define CUB_IGNORE_DEPRECATED_API
#endif

#if !defined(HIPCUB_IGNORE_DEPRECATED_API) \
    && (defined(CUB_IGNORE_DEPRECATED_API) || defined(THRUST_IGNORE_DEPRECATED_API))
    #define HIPCUB_IGNORE_DEPRECATED_API
#endif

#ifdef HIPCUB_IGNORE_DEPRECATED_API
    #define HIPCUB_DEPRECATED
    #define HIPCUB_DEPRECATED_BECAUSE(MSG)
#else
    #define HIPCUB_DEPRECATED [[deprecated]]
    #define HIPCUB_DEPRECATED_BECAUSE(MSG) [[deprecated(MSG)]]
#endif

#define HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS                                                 \
    HIPCUB_DEPRECATED_BECAUSE("The debug_synchronous argument of the hipcub device API functions " \
                              "is deprecated and no longer has any effect.\n"                      \
                              "Use the compile-time definition HIPCUB_DEBUG_SYNC instead.")

#define HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS()                                         \
    do                                                                                        \
    {                                                                                         \
        if(debug_synchronous)                                                                 \
        {                                                                                     \
            HipcubLog("The debug_synchronous argument of the hipcub device API functions is " \
                      "deprecated and no longer has any effect.\n"                            \
                      "Use the compile-time definition HIPCUB_DEBUG_SYNC instead.\n");        \
        }                                                                                     \
    }                                                                                         \
    while(false)

#endif // HIPCUB_UTIL_DEPRECATED_HPP_
