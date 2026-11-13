/*! \file */
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

#include "rocsparse-types.h"

namespace rocsparse
{
    // Indexing vector: an owned device array of indices of a given index type.
    struct idnvec_descr_t
    {
    protected:
        int64_t             m_size{};
        rocsparse_indextype m_indextype{(rocsparse_indextype)-1};
        void*               m_values{};

    public:
        idnvec_descr_t() = default;
        ~idnvec_descr_t();
        idnvec_descr_t(const idnvec_descr_t&) = delete;
        idnvec_descr_t& operator=(const idnvec_descr_t&) = delete;

        rocsparse_status
             create_async(int64_t size, rocsparse_indextype indextype, hipStream_t stream);
        void free();

        // Synchronous deep copy; the device array is reallocated if its size or type differs.
        void copy(const idnvec_descr_t& that);

        int64_t             get_size() const;
        rocsparse_indextype get_indextype() const;
        const void*         get_values() const;
        void*               get_values();
    };
}
