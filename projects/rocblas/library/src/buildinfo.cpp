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
#include "rocblas-version.h"
#include "rocblas.h"
#include <cstring>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define VERSION_STRING                                                           \
    (TO_STR(ROCBLAS_VERSION_MAJOR) "." TO_STR(ROCBLAS_VERSION_MINOR) "." TO_STR( \
        ROCBLAS_VERSION_PATCH) "." TO_STR(ROCBLAS_VERSION_TWEAK))

/*******************************************************************************
 *! \brief   loads char* buf with the rocblas library version. size_t len
     is the maximum length of char* buf.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_version_string(char* buf, size_t len)
{
    static constexpr char v[] = VERSION_STRING;
    if(!buf)
        return rocblas_status_invalid_pointer;
    if(len < sizeof(v))
        return rocblas_status_invalid_size;
    memcpy(buf, v, sizeof(v));
    return rocblas_status_success;
}

/*******************************************************************************
 *! \brief   Returns size of buffer required for rocblas_get_version_string
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_version_string_size(size_t* len)
{
    if(!len)
        return rocblas_status_invalid_pointer;
    *len = std::strlen(VERSION_STRING) + 1;
    return rocblas_status_success;
}

static constexpr const char* rocblas_tensile_commit_hash[] = {ROCBLAS_TENSILE_COMMIT_ID};

/*******************************************************************************
 *! \brief   loads char* buf with the rocblas library version. size_t len
     is the maximum length of char* buf.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_commit_hash_string(char* buf, size_t len)
{

    if(!buf)
        return rocblas_status_invalid_pointer;
    if(len < sizeof(rocblas_tensile_commit_hash[0]))
        return rocblas_status_invalid_size;
    memcpy(buf, rocblas_tensile_commit_hash[0], strlen(rocblas_tensile_commit_hash[0]));
    return rocblas_status_success;
}

/*******************************************************************************
 *! \brief   Returns size of buffer required for rocblas_get_commit_hash_string
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_commit_hash_string_size(size_t* len)
{
    if(!len)
        return rocblas_status_invalid_pointer;
    *len = std::strlen(rocblas_tensile_commit_hash[0]) + 1;
    return rocblas_status_success;
}
