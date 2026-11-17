/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_indextype_utils.hpp"

#include <cstdint>
#include <hip/hip_runtime_api.h>

typedef struct _rocsparse_csrmv_nnzsplit_info
{
protected:
    bool use_starting_block_ids{};

    size_t size{};

    rocsparse_indextype indextype{deprecated_rocsparse_indextype_u16};

    // starting_ids (size + 1 entries), followed by starting_block_ids (size + 1
    // entries) if use_starting_block_ids.
    void* id_buffer{};

public:
    // Launch tuning chosen at analysis time and replayed verbatim at compute
    // time so the two phases always agree (the block layout of starting_ids /
    // starting_block_ids depends on both).
    uint32_t block_size{}; // threads per block (wavefront-relative)
    uint32_t nnz_per_thread{}; // nnz-per-thread granularity (1 / 4 / 8)
    int64_t  max_row_nnz{}; // longest row length (row-skew signal)

    _rocsparse_csrmv_nnzsplit_info()                                      = default;
    _rocsparse_csrmv_nnzsplit_info(const _rocsparse_csrmv_nnzsplit_info&) = delete;
    _rocsparse_csrmv_nnzsplit_info& operator=(const _rocsparse_csrmv_nnzsplit_info&) = delete;

    // Does not free the id buffer: clear() must be called before end of life.
    void clear();

    rocsparse_status allocate(size_t              size,
                              bool                use_starting_block_ids,
                              rocsparse_indextype indextype,
                              hipStream_t         stream);

    size_t              get_size() const;
    bool                get_use_starting_block_ids() const;
    rocsparse_indextype get_indextype() const;
    void*               get_starting_ids() const;
    void*               get_starting_block_ids() const;
    size_t              get_size_in_bytes() const;

    // Copies the id buffer and every field of src. The id buffer of this
    // object is reused when present: it must have the size of the src buffer.
    rocsparse_status copy_from(const _rocsparse_csrmv_nnzsplit_info& src);

    template <typename J>
    J* get_starting_ids() const
    {
        return static_cast<J*>(this->get_starting_ids());
    }

    template <typename J>
    J* get_starting_block_ids() const
    {
        return static_cast<J*>(this->get_starting_block_ids());
    }
} * rocsparse_csrmv_nnzsplit_info;
