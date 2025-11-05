/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

struct _rocsparse_dnmat_descr
{
    bool init{};

    int64_t rows{};
    int64_t cols{};
    int64_t ld{};

    void* values{};

    const void* const_values{};

    rocsparse_datatype data_type{};
    rocsparse_order    order{};

    int64_t                batch_count{};
    int64_t                batch_stride{};
    rocsparse_batchtype    batch_type{};
    rocsparse_batchstorage batch_storage{};

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

    rocsparse_status destroy(rocsparse_handle handle);

    _rocsparse_dnmat_descr() = delete;

    _rocsparse_dnmat_descr(rocsparse_datatype datatype,
                           rocsparse_order    order,
                           int64_t            rows,
                           int64_t            cols,
                           int64_t            ld,
                           const void*        const_values,
                           void*              values);

    _rocsparse_dnmat_descr(rocsparse_datatype     datatype,
                           rocsparse_order        order,
                           int64_t                rows,
                           int64_t                cols,
                           int64_t                ld,
                           rocsparse_batchtype    batchtype,
                           rocsparse_batchstorage batchstorage,
                           int64_t                batch_count,
                           int64_t                batch_dist,
                           const void*            const_values,
                           void*                  values);

    ~_rocsparse_dnmat_descr() = default;

    //
    //
    //
    bool get_init() const
    {
        return this->init;
    }
    void set_init(bool value)
    {
        this->init = value;
    }

    //
    //
    //
    rocsparse_datatype get_datatype() const
    {
        return this->data_type;
    };
    void set_datatype(rocsparse_datatype value)
    {
        this->data_type = value;
    };

    //
    //
    //
    rocsparse_order get_order() const
    {
        return this->order;
    };
    void set_order(rocsparse_order value)
    {
        this->order = value;
    };

    //
    //
    //
    int64_t get_rows() const
    {
        return this->rows;
    };
    void set_rows(int64_t value)
    {
        this->rows = value;
    };

    //
    //
    //
    int64_t get_cols() const
    {
        return this->cols;
    };
    void set_cols(int64_t value)
    {
        this->cols = value;
    };

    //
    //
    //
    int64_t get_ld() const
    {
        return this->ld;
    };
    void set_ld(int64_t value)
    {
        this->ld = value;
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
        return this->batch_stride;
    };
    void set_batch_dist(int64_t value)
    {
        this->batch_stride = value;
    };

    const void* get_const_values() const
    {
        return this->const_values;
    }
    void set_const_values(const void* value)
    {
        this->const_values = value;
    }

    void* get_values()
    {
        return this->values;
    }
    void set_values(void* value)
    {
        this->values = value;
    }

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
};
