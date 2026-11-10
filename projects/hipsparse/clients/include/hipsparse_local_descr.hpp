/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef HIPSPARSE_LOCAL_DESCR_HPP
#define HIPSPARSE_LOCAL_DESCR_HPP

#include <hipsparse/hipsparse.h>

#include "utility.hpp"

// Each class is available on the same CUDA versions as its create/destroy API.
// clang-format 19 (math-ci) rewrites "#if(" to "#if (", which clang-format
// 18 (the repo pre-commit hook) reverts; disable so both formatters agree.

// clang-format off
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
// clang-format on
/*! \brief  SpGEMM descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalSpGEMMDescr
{
    hipsparseSpGEMMDescr_t descr{};

public:
    hipsparseLocalSpGEMMDescr()
    {
        const hipsparseStatus_t status = hipsparseSpGEMM_createDescr(&descr);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseSpGEMM_createDescr", status);
        }
    }

    ~hipsparseLocalSpGEMMDescr()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseSpGEMM_destroyDescr(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseSpGEMM_destroyDescr", status);
            }
        }
    }

    hipsparseLocalSpGEMMDescr(const hipsparseLocalSpGEMMDescr&) = delete;
    hipsparseLocalSpGEMMDescr& operator=(const hipsparseLocalSpGEMMDescr&) = delete;

    operator hipsparseSpGEMMDescr_t() const
    {
        return descr;
    }
};

// These beat the implicit conversion to void*. The handle, matrix descriptor,
// HYB, color, sparse-vector, dense-vector, and dense-matrix typedefs are all
// void*, so a leftover destroy of the wrong kind would otherwise compile and
// free this descriptor twice.
hipsparseStatus_t hipsparseDestroy(const hipsparseLocalSpGEMMDescr&)          = delete;
hipsparseStatus_t hipsparseDestroyMatDescr(const hipsparseLocalSpGEMMDescr&)  = delete;
hipsparseStatus_t hipsparseDestroyHybMat(const hipsparseLocalSpGEMMDescr&)    = delete;
hipsparseStatus_t hipsparseDestroyColorInfo(const hipsparseLocalSpGEMMDescr&) = delete;
hipsparseStatus_t hipsparseDestroySpVec(const hipsparseLocalSpGEMMDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnVec(const hipsparseLocalSpGEMMDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnMat(const hipsparseLocalSpGEMMDescr&)     = delete;
#endif

// clang-format off
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
// clang-format on
/*! \brief  SpSV descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalSpSVDescr
{
    hipsparseSpSVDescr_t descr{};

public:
    hipsparseLocalSpSVDescr()
    {
        const hipsparseStatus_t status = hipsparseSpSV_createDescr(&descr);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseSpSV_createDescr", status);
        }
    }

    ~hipsparseLocalSpSVDescr()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseSpSV_destroyDescr(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseSpSV_destroyDescr", status);
            }
        }
    }

    hipsparseLocalSpSVDescr(const hipsparseLocalSpSVDescr&) = delete;
    hipsparseLocalSpSVDescr& operator=(const hipsparseLocalSpSVDescr&) = delete;

    operator hipsparseSpSVDescr_t() const
    {
        return descr;
    }
};

// These beat the implicit conversion to void*. The handle, matrix descriptor,
// HYB, color, sparse-vector, dense-vector, and dense-matrix typedefs are all
// void*, so a leftover destroy of the wrong kind would otherwise compile and
// free this descriptor twice.
hipsparseStatus_t hipsparseDestroy(const hipsparseLocalSpSVDescr&)          = delete;
hipsparseStatus_t hipsparseDestroyMatDescr(const hipsparseLocalSpSVDescr&)  = delete;
hipsparseStatus_t hipsparseDestroyHybMat(const hipsparseLocalSpSVDescr&)    = delete;
hipsparseStatus_t hipsparseDestroyColorInfo(const hipsparseLocalSpSVDescr&) = delete;
hipsparseStatus_t hipsparseDestroySpVec(const hipsparseLocalSpSVDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnVec(const hipsparseLocalSpSVDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnMat(const hipsparseLocalSpSVDescr&)     = delete;
#endif

// clang-format off
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
// clang-format on
/*! \brief  SpSM descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalSpSMDescr
{
    hipsparseSpSMDescr_t descr{};

public:
    hipsparseLocalSpSMDescr()
    {
        const hipsparseStatus_t status = hipsparseSpSM_createDescr(&descr);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseSpSM_createDescr", status);
        }
    }

    ~hipsparseLocalSpSMDescr()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseSpSM_destroyDescr(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseSpSM_destroyDescr", status);
            }
        }
    }

    hipsparseLocalSpSMDescr(const hipsparseLocalSpSMDescr&) = delete;
    hipsparseLocalSpSMDescr& operator=(const hipsparseLocalSpSMDescr&) = delete;

    operator hipsparseSpSMDescr_t() const
    {
        return descr;
    }
};

// These beat the implicit conversion to void*. The handle, matrix descriptor,
// HYB, color, sparse-vector, dense-vector, and dense-matrix typedefs are all
// void*, so a leftover destroy of the wrong kind would otherwise compile and
// free this descriptor twice.
hipsparseStatus_t hipsparseDestroy(const hipsparseLocalSpSMDescr&)          = delete;
hipsparseStatus_t hipsparseDestroyMatDescr(const hipsparseLocalSpSMDescr&)  = delete;
hipsparseStatus_t hipsparseDestroyHybMat(const hipsparseLocalSpSMDescr&)    = delete;
hipsparseStatus_t hipsparseDestroyColorInfo(const hipsparseLocalSpSMDescr&) = delete;
hipsparseStatus_t hipsparseDestroySpVec(const hipsparseLocalSpSMDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnVec(const hipsparseLocalSpSMDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnMat(const hipsparseLocalSpSMDescr&)     = delete;
#endif

// clang-format off
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 13030)
// clang-format on
/*! \brief  SpGEAM descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalSpGEAMDescr
{
    hipsparseSpGEAMDescr_t descr{};

public:
    hipsparseLocalSpGEAMDescr()
    {
        const hipsparseStatus_t status = hipsparseSpGEAM_createDescr(&descr);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseSpGEAM_createDescr", status);
        }
    }

    ~hipsparseLocalSpGEAMDescr()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseSpGEAM_destroyDescr(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseSpGEAM_destroyDescr", status);
            }
        }
    }

    hipsparseLocalSpGEAMDescr(const hipsparseLocalSpGEAMDescr&) = delete;
    hipsparseLocalSpGEAMDescr& operator=(const hipsparseLocalSpGEAMDescr&) = delete;

    operator hipsparseSpGEAMDescr_t() const
    {
        return descr;
    }
};

// These beat the implicit conversion to void*. The handle, matrix descriptor,
// HYB, color, sparse-vector, dense-vector, and dense-matrix typedefs are all
// void*, so a leftover destroy of the wrong kind would otherwise compile and
// free this descriptor twice.
hipsparseStatus_t hipsparseDestroy(const hipsparseLocalSpGEAMDescr&)          = delete;
hipsparseStatus_t hipsparseDestroyMatDescr(const hipsparseLocalSpGEAMDescr&)  = delete;
hipsparseStatus_t hipsparseDestroyHybMat(const hipsparseLocalSpGEAMDescr&)    = delete;
hipsparseStatus_t hipsparseDestroyColorInfo(const hipsparseLocalSpGEAMDescr&) = delete;
hipsparseStatus_t hipsparseDestroySpVec(const hipsparseLocalSpGEAMDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnVec(const hipsparseLocalSpGEAMDescr&)     = delete;
hipsparseStatus_t hipsparseDestroyDnMat(const hipsparseLocalSpGEAMDescr&)     = delete;
#endif

#endif // HIPSPARSE_LOCAL_DESCR_HPP
