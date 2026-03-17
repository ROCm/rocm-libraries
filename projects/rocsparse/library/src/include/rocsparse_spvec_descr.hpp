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

struct _rocsparse_spvec_descr
{
protected:
    bool init{};

    int64_t size{};
    int64_t nnz{};

    void* idx_data{};
    void* val_data{};

    const void* const_idx_data{};
    const void* const_val_data{};

    rocsparse_indextype idx_type{};
    rocsparse_datatype  data_type{};

    rocsparse_index_base idx_base{};

    int64_t batch_count{1};
    int64_t batch_stride{0};

public:
    //
    bool get_init() const;
    void set_init(bool);
    //
    int64_t get_size() const;
    void    set_size(int64_t);
    //
    int64_t get_nnz() const;
    void    set_nnz(int64_t);
    //
    void* get_idx_data() const;
    void  set_idx_data(void*);
    //
    void* get_val_data() const;
    void  set_val_data(void*);
    //
    const void* get_const_idx_data() const;
    void        set_const_idx_data(const void*);
    //
    const void* get_const_val_data() const;
    void        set_const_val_data(const void*);
    //
    rocsparse_indextype get_idx_type() const;
    void                set_idx_type(rocsparse_indextype);
    //
    rocsparse_datatype get_data_type() const;
    void               set_data_type(rocsparse_datatype);
    //
    rocsparse_index_base get_idx_base() const;
    void                 set_idx_base(rocsparse_index_base);
    //
    int64_t get_batch_count() const;
    void    set_batch_count(int64_t);
    //
    void    set_batch_stride(int64_t);
    int64_t get_batch_stride() const;
};
