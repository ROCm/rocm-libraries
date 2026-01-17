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

struct _rocsparse_idvec_descr
{
protected:
    rocsparse_indextype  indextype{};
    rocsparse_index_base base{};
    int64_t              size{};
    int64_t              inc{};

    rocsparse_batchtype    batch_type{};
    rocsparse_batchstorage batch_storage{};
    int64_t                batch_count{};
    int64_t                batch_dist{};

    const void*            const_values{};
    void*                  values{};
    rocsparse_pointer_mode pointer_mode{};

public:
    //
    //
    //
    rocsparse_indextype get_indextype() const
    {
        return this->indextype;
    };

    void set_indextype(rocsparse_indextype value)
    {
        this->indextype = value;
    };

    //
    //
    //
    rocsparse_index_base get_base() const
    {
        return this->base;
    };
    void set_base(rocsparse_index_base value)
    {
        this->base = value;
    };

    //
    //
    //
    int64_t get_size() const
    {
        return this->size;
    };
    void set_size(int64_t value)
    {
        this->size = value;
    };

    //
    //
    //
    int64_t get_inc() const
    {
        return this->inc;
    };
    void set_inc(int64_t value)
    {
        this->inc = value;
    };

    //
    //
    //
    rocsparse_batchtype get_batch_type() const
    {
        return this->batch_type;
    };
    void set_batch_type(rocsparse_batchtype value)
    {
        this->batch_type = value;
    };

    //
    //
    //
    rocsparse_batchstorage get_batch_storage() const
    {
        return this->batch_storage;
    };
    void set_batch_storage(rocsparse_batchstorage value)
    {
        this->batch_storage = value;
    };

    //
    //
    //
    int64_t get_batch_count() const
    {
        return this->batch_count;
    };
    void set_batch_count(int64_t value)
    {
        this->batch_count = value;
    };

    //
    //
    //
    int64_t get_batch_dist() const
    {
        return this->batch_dist;
    };
    void set_batch_dist(int64_t value)
    {
        this->batch_dist = value;
    };

    //
    //
    //
    rocsparse_pointer_mode get_pointer_mode() const
    {
        return this->pointer_mode;
    };
    void set_pointer_mode(rocsparse_pointer_mode value)
    {
        this->pointer_mode = value;
    };

    //
    //
    //
    const void* const_data() const
    {
        return this->const_values;
    }
    const void* const_data()
    {
        return this->const_values;
    }
    void set_const_data(const void* value)
    {
        this->const_values = value;
    }

    //
    //
    //
    const void* data() const
    {
        return this->values;
    }
    void* data()
    {
        return this->values;
    }
    void set_data(void* value)
    {
        this->values = value;
    }

    //
    //
    //
    _rocsparse_idvec_descr() = delete;

    //
    //
    //
    _rocsparse_idvec_descr(rocsparse_indextype  indextype,
                           rocsparse_index_base base,
                           int64_t              size,
                           int64_t              inc,
                           const void*          const_values,
                           void*                values);

    //
    //
    //
    _rocsparse_idvec_descr(rocsparse_indextype    indextype,
                           rocsparse_index_base   base,
                           int64_t                size,
                           int64_t                inc,
                           rocsparse_batchtype    batch_type,
                           rocsparse_batchstorage batch_storage,
                           int64_t                batch_count,
                           int64_t                batch_dist,
                           const void*            const_values,
                           void*                  values);

    //
    //
    //
    ~_rocsparse_idvec_descr() = default;

    //
    //
    //
    rocsparse_status destroy(rocsparse_handle handle);

    //
    // Implicit casts.
    //
    template <typename T>
    inline operator T*()
    {
        return reinterpret_cast<T*>(this->values);
    }

    template <typename T>
    inline operator T**()
    {
        return reinterpret_cast<T**>(this->values);
    }

    template <typename T>
    inline operator const T*() const
    {
        return reinterpret_cast<const T*>(this->const_values);
    }

    template <typename T>
    inline operator const T* const *() const
    {
        return reinterpret_cast<const T* const*>(this->const_values);
    }
};
