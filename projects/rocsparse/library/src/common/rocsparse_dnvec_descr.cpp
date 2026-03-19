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

#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_memstat.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status _rocsparse_dnvec_descr::validate()
{
    RETURN_IF_ROCSPARSE_ERROR(((this->m_size > 0) && (this->m_const_values == nullptr))
                                  ? rocsparse_status_internal_error
                                  : rocsparse_status_success);
    RETURN_IF_ROCSPARSE_ERROR(((this->m_size > 0) && (this->m_const_values == nullptr))
                                  ? rocsparse_status_internal_error
                                  : rocsparse_status_success);
    RETURN_IF_ROCSPARSE_ERROR((this->m_batch_count < 1) ? rocsparse_status_internal_error
                                                        : rocsparse_status_success);
    RETURN_IF_ROCSPARSE_ERROR((this->m_values != nullptr && this->m_const_values != this->m_values)
                                  ? rocsparse_status_internal_error
                                  : rocsparse_status_success);
    return rocsparse_status_success;
}

rocsparse_status _rocsparse_dnvec_descr::destroy(hipStream_t stream)
{
    if(this->m_own_values)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->m_values, stream));
    }
    this->m_init          = false;
    this->m_own_values    = false;
    this->m_data_type     = rocsparse_datatype_f32_r;
    this->m_size          = 0;
    this->m_inc           = 0;
    this->m_batch_type    = rocsparse_batchtype_strided;
    this->m_batch_storage = rocsparse_batchstorage_soa;
    this->m_batch_count   = 0;
    this->m_batch_stride  = 0;
    this->m_const_values  = nullptr;
    this->m_values        = nullptr;
    this->m_pointer_mode  = rocsparse_pointer_mode_host;
    return rocsparse_status_success;
}

_rocsparse_dnvec_descr::_rocsparse_dnvec_descr(rocsparse_datatype datatype_,
                                               int64_t            size_,
                                               int64_t            inc_,
                                               const void*        const_values_,
                                               void*              values_)
    : m_const_values(const_values_)
    , m_values(values_)
    , m_size(size_)
    , m_inc(inc_)
    , m_batch_type(rocsparse_batchtype_strided)
    , m_batch_storage(rocsparse_batchstorage_soa)
    , m_batch_count(1)
    , m_batch_stride(0)
    , m_data_type(datatype_)
    , m_pointer_mode(rocsparse_pointer_mode_device)
    , m_init(true)
{
}

_rocsparse_dnvec_descr::_rocsparse_dnvec_descr(rocsparse_datatype     datatype_,
                                               int64_t                size_,
                                               int64_t                inc_,
                                               rocsparse_batchtype    batch_type_,
                                               rocsparse_batchstorage batch_storage_,
                                               int64_t                batch_count_,
                                               int64_t                batch_dist_,
                                               const void*            const_values_,
                                               void*                  values_)
    : m_const_values(const_values_)
    , m_values(values_)
    , m_size(size_)
    , m_inc(inc_)
    , m_batch_type(batch_type_)
    , m_batch_storage(batch_storage_)
    , m_batch_count(batch_count_)
    , m_batch_stride(batch_dist_)
    , m_data_type(datatype_)
    , m_pointer_mode(rocsparse_pointer_mode_device)
    , m_init(true)
{
}

_rocsparse_dnvec_descr::_rocsparse_dnvec_descr(int64_t            batch_count_,
                                               int64_t            nitems_,
                                               rocsparse_datatype datatype_,
                                               const void*        const_values_,
                                               void*              values_,
                                               int64_t            inc_,
                                               int64_t            batch_stride_)
    : m_const_values(const_values_)
    , m_values(values_)
    , m_size(nitems_)
    , m_inc(inc_)
    , m_batch_type(rocsparse_batchtype_strided)
    , m_batch_storage(rocsparse_batchstorage_soa)
    , m_batch_count(batch_count_)
    , m_batch_stride(batch_stride_)
    , m_data_type(datatype_)
    , m_pointer_mode(rocsparse_pointer_mode_device)
    , m_init(true)
{
}

void _rocsparse_dnvec_descr::define(rocsparse_datatype datatype_,
                                    int64_t            size_,
                                    int64_t            inc_,
                                    const void*        const_values_,
                                    void*              values_)
{
    this->m_init          = true;
    this->m_data_type     = datatype_;
    this->m_size          = size_;
    this->m_inc           = inc_;
    this->m_batch_type    = rocsparse_batchtype_strided;
    this->m_batch_storage = rocsparse_batchstorage_soa;
    this->m_batch_count   = 1;
    this->m_batch_stride  = 0;
    this->m_const_values  = const_values_;
    this->m_values        = values_;
    this->m_pointer_mode  = rocsparse_pointer_mode_device;
}

void _rocsparse_dnvec_descr::set_own_values(bool value)
{
    this->m_own_values = value;
}

bool _rocsparse_dnvec_descr::get_own_values() const
{
    return this->m_own_values;
}

void _rocsparse_dnvec_descr::define(rocsparse_datatype     datatype_,
                                    int64_t                size_,
                                    int64_t                inc_,
                                    rocsparse_batchtype    batch_type_,
                                    rocsparse_batchstorage batch_storage_,
                                    int64_t                batch_count_,
                                    int64_t                batch_dist_,
                                    const void*            const_values_,
                                    void*                  values_)
{
    this->m_init          = true;
    this->m_data_type     = datatype_;
    this->m_size          = size_;
    this->m_inc           = inc_;
    this->m_batch_type    = batch_type_;
    this->m_batch_storage = batch_storage_;
    this->m_batch_count   = batch_count_;
    this->m_batch_stride  = batch_dist_;
    this->m_const_values  = const_values_;
    this->m_values        = values_;
    this->m_pointer_mode  = rocsparse_pointer_mode_device;
}

rocsparse_datatype _rocsparse_dnvec_descr::get_datatype() const
{
    return this->m_data_type;
}

rocsparse_datatype _rocsparse_dnvec_descr::get_data_type() const
{
    return this->m_data_type;
}

bool _rocsparse_dnvec_descr::get_init() const
{
    return this->m_init;
}

void _rocsparse_dnvec_descr::set_datatype(rocsparse_datatype value)
{
    this->m_data_type = value;
}

int64_t _rocsparse_dnvec_descr::get_size() const
{
    return this->m_size;
}
void _rocsparse_dnvec_descr::set_size(int64_t value)
{
    this->m_size = value;
}

int64_t _rocsparse_dnvec_descr::get_inc() const
{
    return this->m_inc;
}
void _rocsparse_dnvec_descr::set_inc(int64_t value)
{
    this->m_inc = value;
}

rocsparse_batchtype _rocsparse_dnvec_descr::get_batch_type() const
{
    return this->m_batch_type;
}
void _rocsparse_dnvec_descr::set_batch_type(rocsparse_batchtype value)
{
    this->m_batch_type = value;
}

rocsparse_batchstorage _rocsparse_dnvec_descr::get_batch_storage() const
{
    return this->m_batch_storage;
}
void _rocsparse_dnvec_descr::set_batch_storage(rocsparse_batchstorage value)
{
    this->m_batch_storage = value;
}

int64_t _rocsparse_dnvec_descr::get_batch_count() const
{
    return this->m_batch_count;
}
void _rocsparse_dnvec_descr::set_batch_count(int64_t value)
{
    this->m_batch_count = value;
}

int64_t _rocsparse_dnvec_descr::get_batch_dist() const
{
    return this->m_batch_stride;
}
void _rocsparse_dnvec_descr::set_batch_dist(int64_t value)
{
    this->m_batch_stride = value;
}

int64_t _rocsparse_dnvec_descr::get_batch_stride() const
{
    return this->m_batch_stride;
}
void _rocsparse_dnvec_descr::set_batch_stride(int64_t value)
{
    this->m_batch_stride = value;
}

rocsparse_pointer_mode _rocsparse_dnvec_descr::get_pointer_mode() const
{
    return this->m_pointer_mode;
}

void _rocsparse_dnvec_descr::set_pointer_mode(rocsparse_pointer_mode value)
{
    this->m_pointer_mode = value;
}

const void* _rocsparse_dnvec_descr::const_data() const
{
    return this->m_const_values;
}
const void* _rocsparse_dnvec_descr::const_data()
{
    return this->m_const_values;
}
void _rocsparse_dnvec_descr::set_const_data(const void* value)
{
    this->m_const_values = value;
}

const void* _rocsparse_dnvec_descr::data() const
{
    return this->m_values;
}
void* _rocsparse_dnvec_descr::data()
{
    return this->m_values;
}
void _rocsparse_dnvec_descr::set_data(void* value)
{
    this->m_values = value;
}

const void* _rocsparse_dnvec_descr::get_const_values() const
{
    return this->m_const_values;
}
const void* const* _rocsparse_dnvec_descr::get_ref_const_values() const
{
    return &this->m_const_values;
}
const void* _rocsparse_dnvec_descr::get_const_values()
{
    return this->m_const_values;
}
void _rocsparse_dnvec_descr::set_const_values(const void* value)
{
    this->m_const_values = value;
}

const void* _rocsparse_dnvec_descr::get_values() const
{
    return this->m_values;
}
void* _rocsparse_dnvec_descr::get_values()
{
    return this->m_values;
}
void _rocsparse_dnvec_descr::set_values(void* value)
{
    this->m_values = value;
}
