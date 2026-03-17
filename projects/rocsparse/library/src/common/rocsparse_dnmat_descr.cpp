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

#include "rocsparse_dnmat_descr.hpp"

//
//
//
bool _rocsparse_dnmat_descr::get_init() const
{
    return this->m_init;
}
void _rocsparse_dnmat_descr::set_init(bool value)
{
    this->m_init = value;
}

//
//
//
rocsparse_datatype _rocsparse_dnmat_descr::get_datatype() const
{
    return this->m_data_type;
};
void _rocsparse_dnmat_descr::set_datatype(rocsparse_datatype value)
{
    this->m_data_type = value;
};
rocsparse_datatype _rocsparse_dnmat_descr::get_data_type() const
{
    return this->m_data_type;
};
void _rocsparse_dnmat_descr::set_data_type(rocsparse_datatype value)
{
    this->m_data_type = value;
};

//
//
//
rocsparse_order _rocsparse_dnmat_descr::get_order() const
{
    return this->m_order;
};
void _rocsparse_dnmat_descr::set_order(rocsparse_order value)
{
    this->m_order = value;
};

//
//
//
int64_t _rocsparse_dnmat_descr::get_rows() const
{
    return this->m_rows;
};
void _rocsparse_dnmat_descr::set_rows(int64_t value)
{
    this->m_rows = value;
};

//
//
//
int64_t _rocsparse_dnmat_descr::get_cols() const
{
    return this->m_cols;
};
void _rocsparse_dnmat_descr::set_cols(int64_t value)
{
    this->m_cols = value;
};

//
//
//
int64_t _rocsparse_dnmat_descr::get_ld() const
{
    return this->m_ld;
};
void _rocsparse_dnmat_descr::set_ld(int64_t value)
{
    this->m_ld = value;
};

//
//
//
rocsparse_batchtype _rocsparse_dnmat_descr::get_batch_type() const
{
    return this->m_batch_type;
};
void _rocsparse_dnmat_descr::set_batch_type(rocsparse_batchtype value)
{
    this->m_batch_type = value;
};

//
//
//
rocsparse_batchstorage _rocsparse_dnmat_descr::get_batch_storage() const
{
    return this->m_batch_storage;
};
void _rocsparse_dnmat_descr::set_batch_storage(rocsparse_batchstorage value)
{
    this->m_batch_storage = value;
};

//
//
//
int64_t _rocsparse_dnmat_descr::get_batch_count() const
{
    return this->m_batch_count;
};
void _rocsparse_dnmat_descr::set_batch_count(int64_t value)
{
    this->m_batch_count = value;
};

//
//
//
int64_t _rocsparse_dnmat_descr::get_batch_dist() const
{
    return this->m_batch_stride;
};
void _rocsparse_dnmat_descr::set_batch_dist(int64_t value)
{
    this->m_batch_stride = value;
};

int64_t _rocsparse_dnmat_descr::get_batch_stride() const
{
    return this->m_batch_stride;
};
void _rocsparse_dnmat_descr::set_batch_stride(int64_t value)
{
    this->m_batch_stride = value;
};

const void* _rocsparse_dnmat_descr::get_const_values() const
{
    return this->m_const_values;
}
void _rocsparse_dnmat_descr::set_const_values(const void* value)
{
    this->m_const_values = value;
}

void* _rocsparse_dnmat_descr::get_values()
{
    return this->m_values;
}
void _rocsparse_dnmat_descr::set_values(void* value)
{
    this->m_values = value;
}

//
//
//
const void* _rocsparse_dnmat_descr::const_data() const
{
    return this->m_const_values;
}
const void* _rocsparse_dnmat_descr::const_data()
{
    return this->m_const_values;
}
void _rocsparse_dnmat_descr::set_const_data(const void* value)
{
    this->m_const_values = value;
}

//
//
//
const void* _rocsparse_dnmat_descr::data() const
{
    return this->m_values;
}
void* _rocsparse_dnmat_descr::data()
{
    return this->m_values;
}
void _rocsparse_dnmat_descr::set_data(void* value)
{
    this->m_values = value;
}

rocsparse_status _rocsparse_dnmat_descr::destroy(rocsparse_handle handle)
{
    return rocsparse_status_success;
}

_rocsparse_dnmat_descr::_rocsparse_dnmat_descr(rocsparse_datatype datatype_,
                                               rocsparse_order    order_,
                                               int64_t            rows_,
                                               int64_t            cols_,
                                               int64_t            ld_,
                                               const void*        const_values_,
                                               void*              values_)
    : m_init(true)
    , m_rows(rows_)
    , m_cols(cols_)
    , m_ld(ld_)
    , m_values(values_)
    , m_const_values(const_values_)
    , m_data_type(datatype_)
    , m_order(order_)
    , m_batch_count(1)
    , m_batch_stride(0)
    , m_batch_type(rocsparse_batchtype_strided)
{
}

_rocsparse_dnmat_descr::_rocsparse_dnmat_descr(rocsparse_datatype     datatype_,
                                               rocsparse_order        order_,
                                               int64_t                rows_,
                                               int64_t                cols_,
                                               int64_t                ld_,
                                               rocsparse_batchtype    batchtype_,
                                               rocsparse_batchstorage batch_storage_,
                                               int64_t                batch_count_,
                                               int64_t                batch_dist_,
                                               const void*            const_values_,
                                               void*                  values_)
    : m_init(true)
    , m_rows(rows_)
    , m_cols(cols_)
    , m_ld(ld_)
    , m_values(values_)
    , m_const_values(const_values_)
    , m_data_type(datatype_)
    , m_order(order_)
    , m_batch_count(batch_count_)
    , m_batch_stride(batch_dist_)
    , m_batch_type(batchtype_)
    , m_batch_storage(batch_storage_)
{
}
