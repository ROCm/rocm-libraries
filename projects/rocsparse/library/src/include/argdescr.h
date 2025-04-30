/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "debug.h"

namespace rocsparse
{
    ///
    /// @brief Log in argument error description.
    /// @param function_file_ name of the file this routine is called from.
    /// @param function_name_ name of the routine this routine is called from.
    /// @param function_line_ line of the file this routine is called from.
    /// @param arg_name_ name of the argument of the routine this routine is called from.
    /// @param arg_index_ index of the argument of the routine this routine is called from.
    /// @param status_  returned status of the routine this routine is called from.
    /// @param msg_ index of the argument of the routine this routine is called from.
    ///
    void argdescr_log(const char*      function_file_,
                      const char*      function_name_,
                      int              function_line_,
                      const char*      arg_name_,
                      int              arg_index_,
                      rocsparse_status status_,
                      const char*      msg_);
}

///
/// @brief Check the argument array, i.e. a non null pointer for a positive size.
/// @param ITH__ index of the argument.
/// @param SIZE__ size of the array.
/// @param POINTER__ pointer of the array.
///
#define ROCSPARSE_CHECKARG_ARRAY(ITH__, SIZE__, POINTER__)                                    \
    do                                                                                        \
    {                                                                                         \
        if((SIZE__) > 0 && (POINTER__) == nullptr)                                            \
        {                                                                                     \
            if(rocsparse_debug_variables.get_debug_arguments())                               \
            {                                                                                 \
                std::stringstream s;                                                          \
                s << "array pointer is " #POINTER__ " null with ('" #SIZE__ " = " << (SIZE__) \
                  << "' > 0)";                                                                \
                rocsparse::argdescr_log(__FILE__,                                             \
                                        __FUNCTION__,                                         \
                                        __LINE__,                                             \
                                        #POINTER__,                                           \
                                        (ITH__),                                              \
                                        rocsparse_status_invalid_pointer,                     \
                                        s.str().c_str());                                     \
            }                                                                                 \
            return rocsparse_status_invalid_pointer;                                          \
        }                                                                                     \
    } while(false)

///
/// @brief Check the argument size, i.e. a non negative size.
/// @param ITH__ index of the argument.
/// @param SIZE__ size.
///
#define ROCSPARSE_CHECKARG_SIZE(ITH__, SIZE__)                         \
    do                                                                 \
    {                                                                  \
        if((SIZE__) < 0)                                               \
        {                                                              \
            if(rocsparse_debug_variables.get_debug_arguments())        \
                rocsparse::argdescr_log(__FILE__,                      \
                                        __FUNCTION__,                  \
                                        __LINE__,                      \
                                        #SIZE__,                       \
                                        (ITH__),                       \
                                        rocsparse_status_invalid_size, \
                                        "size is negative.");          \
            return rocsparse_status_invalid_size;                      \
        }                                                              \
    } while(false)

///
/// @brief Check the argument enum, i.e. a valid value.
/// @param ITH__ index of the argument.
/// @param ENUM__ The enum value.
///
#define ROCSPARSE_CHECKARG_ENUM(ITH__, ENUM__)                          \
    do                                                                  \
    {                                                                   \
        if(rocsparse::enum_utils::is_invalid((ENUM__)))                 \
        {                                                               \
            if(rocsparse_debug_variables.get_debug_arguments())         \
                rocsparse::argdescr_log(__FILE__,                       \
                                        __FUNCTION__,                   \
                                        __LINE__,                       \
                                        #ENUM__,                        \
                                        (ITH__),                        \
                                        rocsparse_status_invalid_value, \
                                        "enum has an invalid value.");  \
            return rocsparse_status_invalid_value;                      \
        }                                                               \
    } while(false)

///
/// @brief Check the argument handle, i.e. a non null pointer.
/// @param ITH__ index of the argument.
/// @param HANDLE__ The handle.
///
#define ROCSPARSE_CHECKARG_HANDLE(ITH__, HANDLE__)                       \
    do                                                                   \
    {                                                                    \
        if((HANDLE__) == nullptr)                                        \
        {                                                                \
            if(rocsparse_debug_variables.get_debug_arguments())          \
                rocsparse::argdescr_log(__FILE__,                        \
                                        __FUNCTION__,                    \
                                        __LINE__,                        \
                                        #HANDLE__,                       \
                                        (ITH__),                         \
                                        rocsparse_status_invalid_handle, \
                                        "handle is null.");              \
            return rocsparse_status_invalid_handle;                      \
        }                                                                \
    } while(false)

///
/// @brief Check the argument pointer, i.e. a non null pointer.
/// @param ITH__ index of the argument.
/// @param POINTER__ The pointer.
///
#define ROCSPARSE_CHECKARG_POINTER(ITH__, POINTER__)                      \
    do                                                                    \
    {                                                                     \
        if((POINTER__) == nullptr)                                        \
        {                                                                 \
            if(rocsparse_debug_variables.get_debug_arguments())           \
                rocsparse::argdescr_log(__FILE__,                         \
                                        __FUNCTION__,                     \
                                        __LINE__,                         \
                                        #POINTER__,                       \
                                        (ITH__),                          \
                                        rocsparse_status_invalid_pointer, \
                                        "pointer is null.");              \
            return rocsparse_status_invalid_pointer;                      \
        }                                                                 \
    } while(false)

///
/// @brief Check the argument pointer, i.e. a non null pointer.
/// @param ITH__ index of the argument.
/// @param ARG__ The argument.
/// @param CONDITION__ condition to log.
/// @param STATUS__ status to return.
///
#define ROCSPARSE_CHECKARG(ITH__, ARG__, CONDITION__, STATUS__)                    \
    do                                                                             \
    {                                                                              \
        if((CONDITION__))                                                          \
        {                                                                          \
            if(rocsparse_debug_variables.get_debug_arguments())                    \
                rocsparse::argdescr_log(__FILE__,                                  \
                                        __FUNCTION__,                              \
                                        __LINE__,                                  \
                                        #ARG__,                                    \
                                        (ITH__),                                   \
                                        (STATUS__),                                \
                                        "failed on condition '" #CONDITION__ "'"); \
            return (STATUS__);                                                     \
        }                                                                          \
    } while(false)
