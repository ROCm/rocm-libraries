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

#include "rocsparse_idvec_descr.hpp"
#include "rocsparse_memstat.hpp"
#include "rocsparse_utility.hpp"

//
//
//
rocsparse_indextype _rocsparse_idvec_descr::get_indextype() const
{
    return this->m_indextype;
};

void _rocsparse_idvec_descr::set_indextype(rocsparse_indextype value)
{
    this->m_indextype = value;
};

//
//
//
rocsparse_index_base _rocsparse_idvec_descr::get_base() const
{
    return this->m_base;
};
void _rocsparse_idvec_descr::set_base(rocsparse_index_base value)
{
    this->m_base = value;
};

//
//
//
int64_t _rocsparse_idvec_descr::get_size() const
{
    return this->m_size;
};
void _rocsparse_idvec_descr::set_size(int64_t value)
{
    this->m_size = value;
};

//
//
//
int64_t _rocsparse_idvec_descr::get_inc() const
{
    return this->m_inc;
};
void _rocsparse_idvec_descr::set_inc(int64_t value)
{
    this->m_inc = value;
};

//
//
//
rocsparse_batchtype _rocsparse_idvec_descr::get_batch_type() const
{
    return this->m_batch_type;
};
void _rocsparse_idvec_descr::set_batch_type(rocsparse_batchtype value)
{
    this->m_batch_type = value;
};

//
//
//
rocsparse_batchstorage _rocsparse_idvec_descr::get_batch_storage() const
{
    return this->m_batch_storage;
};
void _rocsparse_idvec_descr::set_batch_storage(rocsparse_batchstorage value)
{
    this->m_batch_storage = value;
};

//
//
//
int64_t _rocsparse_idvec_descr::get_batch_count() const
{
    return this->m_batch_count;
};
void _rocsparse_idvec_descr::set_batch_count(int64_t value)
{
    this->m_batch_count = value;
};

//
//
//
int64_t _rocsparse_idvec_descr::get_batch_dist() const
{
    return this->m_batch_dist;
};
void _rocsparse_idvec_descr::set_batch_dist(int64_t value)
{
    this->m_batch_dist = value;
};

//
//
//
rocsparse_pointer_mode _rocsparse_idvec_descr::get_pointer_mode() const
{
    return this->m_pointer_mode;
};
void _rocsparse_idvec_descr::set_pointer_mode(rocsparse_pointer_mode value)
{
    this->m_pointer_mode = value;
};

//
//
//
const void* _rocsparse_idvec_descr::const_data() const
{
    return this->m_const_values;
}
const void* _rocsparse_idvec_descr::const_data()
{
    return this->m_const_values;
}
void _rocsparse_idvec_descr::set_const_data(const void* value)
{
    this->m_const_values = value;
}

//
//
//
const void* _rocsparse_idvec_descr::data() const
{
    return this->m_values;
}
void* _rocsparse_idvec_descr::data()
{
    return this->m_values;
}
void _rocsparse_idvec_descr::set_data(void* value)
{
    this->m_values = value;
}

rocsparse_status _rocsparse_idvec_descr::validate()
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

void _rocsparse_idvec_descr::set_own_values(bool value)
{
    this->m_own_values = value;
}

bool _rocsparse_idvec_descr::get_own_values() const
{
    return this->m_own_values;
}

rocsparse_status _rocsparse_idvec_descr::destroy(hipStream_t stream)
{
    if(this->m_own_values)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->m_values, stream));
    }

    this->m_own_values    = false;
    this->m_indextype     = rocsparse_indextype_i32;
    this->m_base          = rocsparse_index_base_zero;
    this->m_size          = 0;
    this->m_inc           = 0;
    this->m_batch_type    = rocsparse_batchtype_strided;
    this->m_batch_storage = rocsparse_batchstorage_soa;
    this->m_batch_count   = 0;
    this->m_batch_dist    = 0;
    this->m_const_values  = nullptr;
    this->m_values        = nullptr;
    this->m_pointer_mode  = rocsparse_pointer_mode_host;
    return rocsparse_status_success;
}

_rocsparse_idvec_descr::_rocsparse_idvec_descr(rocsparse_indextype  indextype_,
                                               rocsparse_index_base base_,
                                               int64_t              size_,
                                               int64_t              inc_,
                                               const void*          const_values_,
                                               void*                values_)
    : m_indextype(indextype_)
    , m_base(base_)
    , m_size(size_)
    , m_inc(inc_)
    , m_batch_type(rocsparse_batchtype_strided)
    , m_batch_storage(rocsparse_batchstorage_soa)
    , m_batch_count(1)
    , m_batch_dist(0)
    , m_const_values(const_values_)
    , m_values(values_)
    , m_pointer_mode(rocsparse_pointer_mode_device)
{
}

_rocsparse_idvec_descr::_rocsparse_idvec_descr(rocsparse_indextype    indextype_,
                                               rocsparse_index_base   base_,
                                               int64_t                size_,
                                               int64_t                inc_,
                                               rocsparse_batchtype    batch_type_,
                                               rocsparse_batchstorage batch_storage_,
                                               int64_t                batch_count_,
                                               int64_t                batch_dist_,
                                               const void*            const_values_,
                                               void*                  values_)
    : m_indextype(indextype_)
    , m_base(base_)
    , m_size(size_)
    , m_inc(inc_)
    , m_batch_type(batch_type_)
    , m_batch_storage(batch_storage_)
    , m_batch_count(batch_count_)
    , m_batch_dist(batch_dist_)
    , m_const_values(const_values_)
    , m_values(values_)
    , m_pointer_mode(rocsparse_pointer_mode_device)
{
}

void _rocsparse_idvec_descr::define(rocsparse_indextype  indextype_,
                                    rocsparse_index_base base_,
                                    int64_t              size_,
                                    int64_t              inc_,
                                    const void*          const_values_,
                                    void*                values_)
{
    this->m_indextype     = indextype_;
    this->m_base          = base_;
    this->m_size          = size_;
    this->m_inc           = inc_;
    this->m_batch_type    = rocsparse_batchtype_strided;
    this->m_batch_storage = rocsparse_batchstorage_soa;
    this->m_batch_count   = 1;
    this->m_batch_dist    = 0;
    this->m_const_values  = const_values_;
    this->m_values        = values_;
    this->m_pointer_mode  = rocsparse_pointer_mode_device;
}

void _rocsparse_idvec_descr::define(rocsparse_indextype    indextype_,
                                    rocsparse_index_base   base_,
                                    int64_t                size_,
                                    int64_t                inc_,
                                    rocsparse_batchtype    batch_type_,
                                    rocsparse_batchstorage batch_storage_,
                                    int64_t                batch_count_,
                                    int64_t                batch_dist_,
                                    const void*            const_values_,
                                    void*                  values_)
{
    this->m_indextype     = indextype_;
    this->m_base          = base_;
    this->m_size          = size_;
    this->m_inc           = inc_;
    this->m_batch_type    = batch_type_;
    this->m_batch_storage = batch_storage_;
    this->m_batch_count   = batch_count_;
    this->m_batch_dist    = batch_dist_;
    this->m_const_values  = const_values_;
    this->m_values        = values_;
    this->m_pointer_mode  = rocsparse_pointer_mode_device;
}
