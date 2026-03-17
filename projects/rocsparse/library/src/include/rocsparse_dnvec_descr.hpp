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

struct _rocsparse_dnvec_descr
{
protected:
    const void*            m_const_values{};
    void*                  m_values{};
    int64_t                m_size{};
    int64_t                m_inc{};
    rocsparse_batchtype    m_batch_type{};
    rocsparse_batchstorage m_batch_storage{};
    int64_t                m_batch_count;
    int64_t                m_batch_stride;
    rocsparse_datatype     m_data_type{};
    rocsparse_pointer_mode m_pointer_mode{};
    bool                   m_init;
    bool                   m_own_values;

public:
    //
    _rocsparse_dnvec_descr()  = default;
    ~_rocsparse_dnvec_descr() = default;

    rocsparse_status validate();
    rocsparse_status destroy(hipStream_t stream);
    //
    _rocsparse_dnvec_descr(rocsparse_datatype datatype,
                           int64_t            size,
                           int64_t            inc,
                           const void*        const_values,
                           void*              values);
    //
    _rocsparse_dnvec_descr(rocsparse_datatype     datatype,
                           int64_t                size,
                           int64_t                inc,
                           rocsparse_batchtype    batchtype,
                           rocsparse_batchstorage batchstorage,
                           int64_t                batch_count,
                           int64_t                batch_dist,
                           const void*            const_values,
                           void*                  values);

    _rocsparse_dnvec_descr(int64_t            batch_count_,
                           int64_t            nitems_,
                           rocsparse_datatype data_type_,
                           const void*        const_values_,
                           void*              values_,
                           int64_t            inc_,
                           int64_t            batch_stride_);

    void define(rocsparse_datatype datatype,
                int64_t            size,
                int64_t            inc,
                const void*        const_values,
                void*              values);

    void define(rocsparse_datatype     datatype,
                int64_t                size,
                int64_t                inc,
                rocsparse_batchtype    batchtype,
                rocsparse_batchstorage batchstorage,
                int64_t                batch_count,
                int64_t                batch_dist,
                const void*            const_values,
                void*                  values);

    void define(int64_t            batch_count_,
                int64_t            nitems_,
                rocsparse_datatype data_type_,
                const void*        const_values_,
                void*              values_,
                int64_t            inc_,
                int64_t            batch_stride_);
    //
    bool get_own_values() const;
    void set_own_values(bool value);
    //
    bool get_init() const;
    void set_init(bool);
    //
    rocsparse_datatype get_datatype() const;
    void               set_datatype(rocsparse_datatype value);
    //
    rocsparse_datatype get_data_type() const;
    void               set_data_type(rocsparse_datatype value);
    //
    int64_t get_size() const;
    void    set_size(int64_t value);
    //
    int64_t get_inc() const;
    void    set_inc(int64_t value);
    //
    rocsparse_batchtype get_batch_type() const;
    void                set_batch_type(rocsparse_batchtype value);
    //
    rocsparse_batchstorage get_batch_storage() const;
    void                   set_batch_storage(rocsparse_batchstorage value);
    //
    int64_t get_batch_count() const;
    void    set_batch_count(int64_t value);
    //
    int64_t get_batch_dist() const;
    void    set_batch_dist(int64_t value);
    //
    int64_t get_batch_stride() const;
    void    set_batch_stride(int64_t value);
    //
    rocsparse_pointer_mode get_pointer_mode() const;
    void                   set_pointer_mode(rocsparse_pointer_mode value);
    //
    const void* const_data() const;
    const void* const_data();
    void        set_const_data(const void* value);
    //
    const void* data() const;
    void*       data();
    void        set_data(void* value);
    //
    const void* get_const_values() const;
    void        set_const_values(const void* value);
    //
    const void* const* get_ref_const_values() const;
    const void*        get_const_values();
    const void*        get_values() const;
    //
    void* get_values();
    void  set_values(void* value);

    //
    template <typename T>
    inline operator T*()
    {
        return reinterpret_cast<T*>(this->m_values);
    }
    //
    template <typename T>
    inline operator T**()
    {
        return reinterpret_cast<T**>(this->m_values);
    }
    //
    template <typename T>
    inline operator const T*() const
    {
        return reinterpret_cast<const T*>(this->m_const_values);
    }
    //
    template <typename T>
    inline operator const T* const *() const
    {
        return reinterpret_cast<const T* const*>(this->m_const_values);
    }
};
