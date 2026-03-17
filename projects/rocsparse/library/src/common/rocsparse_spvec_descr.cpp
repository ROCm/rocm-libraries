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

#include "rocsparse_spvec_descr.hpp"
#include "rocsparse_utility.hpp"

void _rocsparse_spvec_descr::set_init(bool value)
{
    this->init = value;
}
void _rocsparse_spvec_descr::set_size(int64_t value)
{
    this->size = value;
}
void _rocsparse_spvec_descr::set_nnz(int64_t value)
{
    this->nnz = value;
}
void _rocsparse_spvec_descr::set_idx_data(void* value)
{
    this->idx_data = value;
}
void _rocsparse_spvec_descr::set_val_data(void* value)
{
    this->val_data = value;
}
void _rocsparse_spvec_descr::set_const_idx_data(const void* value)
{
    this->const_idx_data = value;
}
void _rocsparse_spvec_descr::set_const_val_data(const void* value)
{
    this->const_val_data = value;
}
void _rocsparse_spvec_descr::set_idx_type(rocsparse_indextype value)
{
    this->idx_type = value;
}
void _rocsparse_spvec_descr::set_data_type(rocsparse_datatype value)
{
    this->data_type = value;
}
void _rocsparse_spvec_descr::set_idx_base(rocsparse_index_base value)
{
    this->idx_base = value;
}
void _rocsparse_spvec_descr::set_batch_count(int64_t value)
{
    this->batch_count = value;
}
void _rocsparse_spvec_descr::set_batch_stride(int64_t value)
{
    this->batch_stride = value;
}

bool _rocsparse_spvec_descr::get_init() const
{
    return this->init;
}
int64_t _rocsparse_spvec_descr::get_size() const
{
    return this->size;
}
int64_t _rocsparse_spvec_descr::get_nnz() const
{
    return this->nnz;
}
void* _rocsparse_spvec_descr::get_idx_data() const
{
    return this->idx_data;
}
void* _rocsparse_spvec_descr::get_val_data() const
{
    return this->val_data;
}
const void* _rocsparse_spvec_descr::get_const_idx_data() const
{
    return this->idx_data;
}
const void* _rocsparse_spvec_descr::get_const_val_data() const
{
    return this->val_data;
}
rocsparse_indextype _rocsparse_spvec_descr::get_idx_type() const
{
    return this->idx_type;
}
rocsparse_datatype _rocsparse_spvec_descr::get_data_type() const
{
    return this->data_type;
}
rocsparse_index_base _rocsparse_spvec_descr::get_idx_base() const
{
    return this->idx_base;
}
int64_t _rocsparse_spvec_descr::get_batch_count() const
{
    return this->batch_count;
}
int64_t _rocsparse_spvec_descr::get_batch_stride() const
{
    return this->batch_stride;
}
