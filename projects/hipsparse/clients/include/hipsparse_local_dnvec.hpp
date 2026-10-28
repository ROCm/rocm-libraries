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
#ifndef HIPSPARSE_LOCAL_DNVEC_HPP
#define HIPSPARSE_LOCAL_DNVEC_HPP

#include <hipsparse/hipsparse.h>

#include "arg_check.hpp"

/*! \brief  Dense vector descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalDnVec
{
    hipsparseDnVecDescr_t descr{};

public:
    hipsparseLocalDnVec(int64_t size, void* values, hipDataType valueType)
    {
        const hipsparseStatus_t status = hipsparseCreateDnVec(&descr, size, values, valueType);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "hipsparseCreateDnVec failed: " << hipsparseGetErrorName(status)
                      << std::endl;
            throw(status);
        }
    }

    ~hipsparseLocalDnVec()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseDestroyDnVec(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                verify_hipsparse_status_success(status, "ERROR: hipsparseLocalDnVec destructor");
            }
        }
    }

    hipsparseLocalDnVec(const hipsparseLocalDnVec&) = delete;
    hipsparseLocalDnVec& operator=(const hipsparseLocalDnVec&) = delete;

    operator hipsparseDnVecDescr_t() const
    {
        return descr;
    }
};

// Exact match for the wrapper, so this beats the implicit conversion to
// hipsparseDnVecDescr_t. A leftover hipsparseDestroyDnVec(wrapper) is a compile
// error instead of destroying the descriptor twice.
hipsparseStatus_t hipsparseDestroyDnVec(const hipsparseLocalDnVec&) = delete;

#endif // HIPSPARSE_LOCAL_DNVEC_HPP
