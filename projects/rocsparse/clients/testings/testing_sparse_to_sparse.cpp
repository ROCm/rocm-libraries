/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

struct spmat
{
    rocsparse_spmat_descr descr{};
    void*                 row_data{};
    void*                 col_data{};
    void*                 ind_data{};
    void*                 val_data{};

    static void* to_device(hipStream_t stream, size_t nmemb, size_t size_memb, const void* h)
    {
        void* d;
        CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&d, size_memb * nmemb));
        CHECK_HIP_THROW_ERROR(
            hipMemcpyAsync(d, h, size_memb * nmemb, hipMemcpyHostToDevice, stream));
        CHECK_HIP_THROW_ERROR(hipStreamSynchronize(stream));
        return d;
    }

public:
    //
    // coo.
    //
    template <typename T, typename I = rocsparse_int>
    explicit spmat(coo_matrix<memory_mode::host, T, I>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_descr(
            &this->descr,
            h.m,
            h.n,
            h.nnz,
            row_data = to_device(nullptr, h.row_ind.size(), sizeof(I), h.row_ind),
            col_data = to_device(nullptr, h.col_ind.size(), sizeof(I), h.col_ind),
            val_data = to_device(nullptr, h.val.size(), sizeof(T), h.val),
            get_indextype<I>(),
            h.base,
            get_datatype<T>()));
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    explicit spmat(coo_matrix<MODE, T, I>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_descr(&this->descr,
                                                               h.m,
                                                               h.n,
                                                               h.nnz,
                                                               h.row_ind,
                                                               h.col_ind,
                                                               h.val,
                                                               get_indextype<I>(),
                                                               h.base,
                                                               get_datatype<T>()));
    }
    //
    // coo_aos
    //
    template <typename T, typename I = rocsparse_int>
    explicit spmat(coo_aos_matrix<memory_mode::host, T, I>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_aos_descr(
            &this->descr,
            h.m,
            h.n,
            h.nnz,
            ind_data = to_device(nullptr, h.ind.size(), sizeof(I), h.ind),
            val_data = to_device(nullptr, h.val.size(), sizeof(T), h.val),
            get_indextype<I>(),
            h.base,
            get_datatype<T>()));
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    explicit spmat(coo_aos_matrix<MODE, T, I>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_descr(&this->descr,
                                                               h.m,
                                                               h.n,
                                                               h.nnz,
                                                               h.ind,
                                                               h.val,
                                                               get_indextype<I>(),
                                                               h.base,
                                                               get_datatype<T>()));
    }

    template <rocsparse_direction DIRECTION,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    explicit spmat(csx_matrix<memory_mode::host, DIRECTION, T, I, J>& h)
    {
        switch(DIRECTION)
        {
        case rocsparse_direction_row:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_csr_descr(
                &this->descr,
                h.m,
                h.n,
                h.nnz,
                row_data = to_device(nullptr, h.ptr.size(), sizeof(I), h.ptr),
                col_data = to_device(nullptr, h.ind.size(), sizeof(J), h.ind),
                val_data = to_device(nullptr, h.val.size(), sizeof(T), h.val),
                get_indextype<I>(),
                get_indextype<J>(),
                h.base,
                get_datatype<T>()));
            break;
        }
        case rocsparse_direction_column:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_csc_descr(
                &this->descr,
                h.m,
                h.n,
                h.nnz,
                col_data = to_device(nullptr, h.ptr.size(), sizeof(I), h.ptr),
                row_data = to_device(nullptr, h.ind.size(), sizeof(J), h.ind),
                val_data = to_device(nullptr, h.val.size(), sizeof(T), h.val),
                get_indextype<I>(),
                get_indextype<J>(),
                h.base,
                get_datatype<T>()));
            break;
        }
        }
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    explicit spmat(csx_matrix<MODE, DIRECTION, T, I, J>& h)
    {
        switch(DIRECTION)
        {
        case rocsparse_direction_row:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_csr_descr(
                &this->descr, h.m, h.n, h.nnz, h.ptr, h.ind, h.val, h.base, get_datatype<T>()));
            break;
        }
        case rocsparse_direction_column:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_csc_descr(
                &this->descr, h.m, h.n, h.nnz, h.ptr, h.ind, h.val, h.base, get_datatype<T>()));
            break;
        }
        }
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    explicit spmat(gebsx_matrix<MODE, DIRECTION, T, I, J>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_bsr_descr(&this->descr,
                                                               h.mb,
                                                               h.nb,
                                                               h.nnzb,
                                                               h.block_direction,
                                                               h.block_dim,
                                                               h.row_col_ptr,
                                                               h.row_col_ind,
                                                               h.val,
                                                               get_indextype<I>(),
                                                               get_indextype<J>(),
                                                               h.base,
                                                               get_datatype<T>()));
    }

    template <rocsparse_direction DIRECTION,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    explicit spmat(gebsx_matrix<memory_mode::host, DIRECTION, T, I, J>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_bsr_descr(
            &this->descr,
            h.mb,
            h.nb,
            h.nnzb,
            h.block_direction,
            h.row_block_dim,
            row_data = to_device(nullptr, h.ptr.size(), sizeof(I), h.ptr),
            col_data = to_device(nullptr, h.ind.size(), sizeof(I), h.ind),
            val_data = to_device(nullptr, h.val.size(), sizeof(T), h.val),
            get_indextype<I>(),
            get_indextype<J>(),
            h.base,
            get_datatype<T>()));
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    explicit spmat(ell_matrix<MODE, T, I>& d)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_ell_descr(&this->descr,
                                                               d.m,
                                                               d.n,
                                                               d.ind,
                                                               d.val,
                                                               d.width,
                                                               get_indextype<I>(),
                                                               d.base,
                                                               get_datatype<T>()));
    }

    template <typename T, typename I = rocsparse_int>
    explicit spmat(ell_matrix<memory_mode::host, T, I>& h)
    {
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_ell_descr(
            &this->descr,
            h.m,
            h.n,
            ind_data = to_device(nullptr, h.ind.size(), sizeof(I), h.ind),
            val_data = to_device(nullptr, h.val.size(), sizeof(T), h.val),
            h.width,
            get_indextype<I>(),
            h.base,
            get_datatype<T>()));
    }

    ~spmat()
    {
        if(this->descr != nullptr)
        {
            rocsparse_destroy_spmat_descr(this->descr);
        }
        std::ignore = rocsparse_hipFree(this->row_data);
        std::ignore = rocsparse_hipFree(this->col_data);
        std::ignore = rocsparse_hipFree(this->ind_data);
        std::ignore = rocsparse_hipFree(this->val_data);
    }

    // Allow spmat to be used anywhere rocsparse_spmat_descr is expected
    operator rocsparse_spmat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_const_spmat_descr&() const
    {
        return this->descr;
    }

    spmat(rocsparse_format     format,
          int64_t              m,
          int64_t              n,
          int64_t              nnz,
          rocsparse_indextype  row_type,
          rocsparse_indextype  col_type,
          rocsparse_datatype   data_type,
          rocsparse_index_base base)
    {

        switch(format)
        {
        case rocsparse_format_csr:
        {
            const size_t size_row_data_in_bytes = rocsparse_indextype_sizeof(row_type) * (m + 1);
            const size_t size_col_data_in_bytes = rocsparse_indextype_sizeof(col_type) * nnz;
            const size_t size_val_data_in_bytes = rocsparse_datatype_sizeof(data_type) * nnz;
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->row_data, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(hipMemset(this->row_data, 0, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->col_data, size_col_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->val_data, size_val_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_csr_descr(&this->descr,
                                                                   m,
                                                                   n,
                                                                   nnz,
                                                                   this->row_data,
                                                                   this->col_data,
                                                                   this->val_data,
                                                                   row_type,
                                                                   col_type,
                                                                   base,
                                                                   data_type));
            break;
        }

        case rocsparse_format_ell:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_ell_descr(
                &this->descr, m, n, nullptr, nullptr, 0, col_type, base, data_type));
            break;
        }
        case rocsparse_format_csc:
        {
            const size_t size_row_data_in_bytes = rocsparse_indextype_sizeof(row_type) * nnz;
            const size_t size_col_data_in_bytes = rocsparse_indextype_sizeof(col_type) * (n + 1);
            const size_t size_val_data_in_bytes = rocsparse_datatype_sizeof(data_type) * nnz;
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->row_data, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(hipMemset(this->row_data, 0, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->col_data, size_col_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->val_data, size_val_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_csc_descr(&this->descr,
                                                                   m,
                                                                   n,
                                                                   nnz,
                                                                   this->col_data,
                                                                   this->row_data,
                                                                   this->val_data,
                                                                   col_type,
                                                                   row_type,
                                                                   base,
                                                                   data_type));
            break;
        }
        case rocsparse_format_coo:
        {
            const size_t size_row_data_in_bytes = rocsparse_indextype_sizeof(col_type) * nnz;
            const size_t size_col_data_in_bytes = rocsparse_indextype_sizeof(col_type) * nnz;
            const size_t size_val_data_in_bytes = rocsparse_datatype_sizeof(data_type) * nnz;
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->row_data, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->col_data, size_col_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->val_data, size_val_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_descr(&this->descr,
                                                                   m,
                                                                   n,
                                                                   nnz,
                                                                   this->row_data,
                                                                   this->col_data,
                                                                   this->val_data,
                                                                   col_type,
                                                                   base,
                                                                   data_type));
            break;
        }
        case rocsparse_format_bsr:
        {
            const size_t size_row_data_in_bytes = rocsparse_indextype_sizeof(row_type) * (m + 1);
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->row_data, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(hipMemset(this->row_data, 0, size_row_data_in_bytes));
            // will be in invalid state.
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_bsr_descr(&this->descr,
                                                                   m,
                                                                   n,
                                                                   0,
                                                                   rocsparse_direction_row,
                                                                   1,
                                                                   row_data,
                                                                   nullptr,
                                                                   nullptr,
                                                                   row_type,
                                                                   col_type,
                                                                   base,
                                                                   data_type));
            break;
        }
        case rocsparse_format_bell:
        {

            break;
        }
        case rocsparse_format_coo_aos:
        {
            const size_t size_ind_data_in_bytes = rocsparse_indextype_sizeof(col_type) * nnz * 2;
            const size_t size_val_data_in_bytes = rocsparse_datatype_sizeof(data_type) * nnz;
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->ind_data, size_ind_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->val_data, size_val_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_aos_descr(&this->descr,
                                                                       m,
                                                                       n,
                                                                       nnz,
                                                                       this->ind_data,
                                                                       this->val_data,
                                                                       col_type,
                                                                       base,
                                                                       data_type));
            break;
        }
        }
    }

    spmat(rocsparse_format     format,
          int64_t              m,
          int64_t              n,
          rocsparse_indextype  ind_type,
          rocsparse_datatype   val_type,
          rocsparse_index_base base)
    {
        switch(format)
        {
        case rocsparse_format_csr:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_bell:
        {
            throw(rocsparse_status_invalid_value);
        }
        case rocsparse_format_ell:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_ell_descr(
                &this->descr, m, n, nullptr, nullptr, 0, ind_type, base, val_type));
            break;
        }
        case rocsparse_format_coo:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_descr(
                &this->descr, m, n, 0, nullptr, nullptr, nullptr, ind_type, base, val_type));
            break;
        }
        case rocsparse_format_coo_aos:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_aos_descr(
                &this->descr, m, n, 0, nullptr, nullptr, ind_type, base, val_type));
            break;
        }
        }
    }

    spmat(rocsparse_format     format,
          int64_t              m,
          int64_t              n,
          int64_t              nnz,
          rocsparse_indextype  indextype,
          rocsparse_datatype   datatype,
          rocsparse_index_base base)
    {
        switch(format)
        {
        case rocsparse_format_csr:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_bell:
        {
            throw(rocsparse_status_invalid_value);
        }
        case rocsparse_format_ell:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_ell_descr(
                &this->descr, m, n, nullptr, nullptr, 0, indextype, base, datatype));
            break;
        }
        case rocsparse_format_coo:
        {
            const size_t size_ind_data_in_bytes = rocsparse_indextype_sizeof(indextype) * nnz;
            const size_t size_val_data_in_bytes = rocsparse_datatype_sizeof(datatype) * nnz;
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->row_data, size_ind_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->col_data, size_ind_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->val_data, size_val_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_descr(&this->descr,
                                                                   m,
                                                                   n,
                                                                   nnz,
                                                                   this->row_data,
                                                                   this->col_data,
                                                                   this->val_data,
                                                                   indextype,
                                                                   base,
                                                                   datatype));
            break;
        }
        case rocsparse_format_coo_aos:
        {
            const size_t size_ind_data_in_bytes = rocsparse_indextype_sizeof(indextype) * nnz * 2;
            const size_t size_val_data_in_bytes = rocsparse_datatype_sizeof(datatype) * nnz;
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->ind_data, size_ind_data_in_bytes));
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->val_data, size_val_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_coo_aos_descr(&this->descr,
                                                                       m,
                                                                       n,
                                                                       nnz,
                                                                       this->ind_data,
                                                                       this->val_data,
                                                                       indextype,
                                                                       base,
                                                                       datatype));
            break;
        }
        }
    }

    spmat(rocsparse_format     format,
          int64_t              mb,
          int64_t              nb,
          int64_t              nnzb,
          rocsparse_direction  dirb,
          int64_t              dimb,
          rocsparse_indextype  row_type,
          rocsparse_indextype  col_type,
          rocsparse_datatype   val_type,
          rocsparse_index_base base)
    {
        switch(format)
        {
        case rocsparse_format_csr:
        case rocsparse_format_ell:
        case rocsparse_format_csc:
        case rocsparse_format_coo_aos:
        case rocsparse_format_bell:
        case rocsparse_format_coo:
        {
            throw(rocsparse_status_invalid_value);
        }
        case rocsparse_format_bsr:
        {
            const size_t size_row_data_in_bytes = rocsparse_indextype_sizeof(row_type) * (mb + 1);
            CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&this->row_data, size_row_data_in_bytes));
            CHECK_HIP_THROW_ERROR(hipMemset(this->row_data, 0, size_row_data_in_bytes));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_bsr_descr(&this->descr,
                                                                   mb,
                                                                   nb,
                                                                   0,
                                                                   dirb,
                                                                   dimb,
                                                                   row_data,
                                                                   nullptr,
                                                                   nullptr,
                                                                   row_type,
                                                                   col_type,
                                                                   base,
                                                                   val_type));
            break;
        }
        }
    }
};

void spmat_allocate(
    rocsparse_spmat_descr descr, void*& row_data, void*& col_data, void*& ind_data, void*& val_data)
{
    //    ind_data = nullptr;
    //    row_data = nullptr;
    //    col_data = nullptr;
    //    val_data = nullptr;

    rocsparse_format format;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_format(descr, &format));
    switch(format)
    {
    case rocsparse_format_bell:
    {
        break;
    }
    case rocsparse_format_ell:
    {
        int64_t              m;
        int64_t              n;
        void*                ind;
        void*                val;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        rocsparse_indextype  ind_type;
        int64_t              ell_width;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_ell_get(descr, &m, &n, &ind, &val, &ell_width, &ind_type, &base, &val_type));
        const int64_t nnz = ell_width * m;
#if 0
        row_data          = ind;
        col_data          = nullptr;
        val_data          = val;
        ind_data          = nullptr;
#endif

        if(nnz > 0 && val == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&val, rocsparse_datatype_sizeof(val_type) * nnz));
            val_data = val;
        }

        if(nnz > 0 && ind == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&ind, rocsparse_indextype_sizeof(ind_type) * nnz));
            col_data = ind;
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_ell_set_pointers(descr, ind, val));

        break;
    }

    case rocsparse_format_bsr:
    {
        int64_t              mb;
        int64_t              nb;
        int64_t              nnzb;
        rocsparse_direction  dirb;
        int64_t              dimb;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_bsr_get(descr,
                                                &mb,
                                                &nb,
                                                &nnzb,
                                                &dirb,
                                                &dimb,
                                                &ptr,
                                                &ind,
                                                &val,
                                                &ptr_type,
                                                &ind_type,
                                                &base,
                                                &val_type));

#if 0
        row_data = ptr;
        col_data = ind;
        val_data = val;
        ind_data = nullptr;
#endif
        if(mb > 0 && ptr == nullptr)
        {
            CHECK_HIP_ERROR(
                rocsparse_hipMalloc(&ptr, rocsparse_indextype_sizeof(ptr_type) * (mb + 1)));
            row_data = ptr;
        }
        if(nnzb > 0 && ind == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&ind, rocsparse_indextype_sizeof(ind_type) * nnzb));
            col_data = ind;
        }
        if(nnzb > 0 && val == nullptr)
        {
            const int64_t nnz = nnzb * dimb * dimb;
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&val, rocsparse_datatype_sizeof(val_type) * nnz));
            val_data = val;
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_bsr_set_pointers(descr, ptr, ind, val));

        break;
    }
    case rocsparse_format_csr:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  col_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &col_type, &base, &val_type));
#if 0
        row_data = ptr;
        col_data = ind;
        val_data = val;
        ind_data = nullptr;
#endif
        if(m > 0 && ptr == nullptr)
        {
            CHECK_HIP_ERROR(
                rocsparse_hipMalloc(&ptr, rocsparse_indextype_sizeof(ptr_type) * (m + 1)));
            row_data = ptr;
        }
        if(nnz > 0 && val == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&val, rocsparse_datatype_sizeof(val_type) * nnz));
            val_data = val;
        }
        if(nnz > 0 && ind == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&ind, rocsparse_indextype_sizeof(col_type) * nnz));
            col_data = ind;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_pointers(descr, ptr, ind, val));
        break;
    }
    case rocsparse_format_csc:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_csc_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
#if 0
        col_data = ptr;
        row_data = ind;
        val_data = val;
        ind_data = nullptr;
#endif
        if(n > 0 && ptr == nullptr)
        {
            CHECK_HIP_ERROR(
                rocsparse_hipMalloc(&ptr, rocsparse_indextype_sizeof(ptr_type) * (n + 1)));
            col_data = ptr;
        }
        if(nnz > 0 && val == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&val, rocsparse_datatype_sizeof(val_type) * nnz));
            val_data = val;
        }
        if(nnz > 0 && ind == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&ind, rocsparse_indextype_sizeof(ind_type) * nnz));
            row_data = ind;
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_csc_set_pointers(descr, ptr, ind, val));
        break;
    }
    case rocsparse_format_coo:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                row_ind;
        void*                col_ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_coo_get(
            descr, &m, &n, &nnz, &row_ind, &col_ind, &val, &ind_type, &base, &val_type));
        if(nnz > 0 && row_ind == nullptr)
        {
            CHECK_HIP_ERROR(
                rocsparse_hipMalloc(&row_ind, rocsparse_indextype_sizeof(ind_type) * nnz));
            row_data = row_ind;
        }
        if(nnz > 0 && col_ind == nullptr)
        {
            CHECK_HIP_ERROR(
                rocsparse_hipMalloc(&col_ind, rocsparse_indextype_sizeof(ind_type) * nnz));
            col_data = col_ind;
        }
        if(nnz > 0 && val == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&val, rocsparse_datatype_sizeof(val_type) * nnz));
            val_data = val;
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_coo_set_pointers(descr, row_ind, col_ind, val));

        break;
    }
    case rocsparse_format_coo_aos:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_coo_aos_get(descr, &m, &n, &nnz, &ind, &val, &ind_type, &base, &val_type));
#if 0
        val_data = val;
        ind_data = ind;
#endif
        if(nnz > 0 && ind == nullptr)
        {
            CHECK_HIP_ERROR(
                rocsparse_hipMalloc(&ind, rocsparse_indextype_sizeof(ind_type) * nnz * 2));
            ind_data = ind;
        }
        if(nnz > 0 && val == nullptr)
        {
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&val, rocsparse_datatype_sizeof(val_type) * nnz));
            val_data = val;
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_coo_aos_set_pointers(descr, ind, val));
        break;
    }
    }
}

void spmat_get_rawsize(rocsparse_spmat_descr descr, int64_t* m_, int64_t* n_, int64_t* nnz_)
{
    rocsparse_format format;
    rocsparse_spmat_get_format(descr, &format);
    switch(format)
    {
    case rocsparse_format_bell:
    {
        std::cerr << " initialize bell not implemented " << std::endl;
        throw(rocsparse_status_not_implemented);
    }
    case rocsparse_format_ell:
    {
        int64_t              m;
        int64_t              n;
        void*                ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        int64_t              ell_width;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_ell_get(descr, &m, &n, &ind, &val, &ell_width, &ind_type, &base, &val_type));
        m_[0]   = m;
        n_[0]   = n;
        nnz_[0] = 0; // m * ell_width;
        return;
    }

    case rocsparse_format_bsr:
    {
        int64_t              mb;
        int64_t              nb;
        int64_t              nnzb;
        rocsparse_direction  dirb;
        int64_t              dimb;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_bsr_get(descr,
                                                &mb,
                                                &nb,
                                                &nnzb,
                                                &dirb,
                                                &dimb,
                                                &ptr,
                                                &ind,
                                                &val,
                                                &ptr_type,
                                                &ind_type,
                                                &base,
                                                &val_type));
        m_[0]   = mb * dimb;
        n_[0]   = nb * dimb;
        nnz_[0] = nnzb * dimb * dimb;
        return;
    }
    case rocsparse_format_csr:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
        m_[0]   = m;
        n_[0]   = n;
        nnz_[0] = nnz;
        return;
    }
    case rocsparse_format_csc:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_csc_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
        m_[0]   = m;
        n_[0]   = n;
        nnz_[0] = nnz;
        return;
    }
    case rocsparse_format_coo:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                row_ind;
        void*                col_ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_coo_get(
            descr, &m, &n, &nnz, &row_ind, &col_ind, &val, &ind_type, &base, &val_type));
        m_[0]   = m;
        n_[0]   = n;
        nnz_[0] = nnz;
        return;
    }
    case rocsparse_format_coo_aos:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_coo_aos_get(descr, &m, &n, &nnz, &ind, &val, &ind_type, &base, &val_type));
        m_[0]   = m;
        n_[0]   = n;
        nnz_[0] = nnz;
        return;
    }
    }
}

void spmat_get_types(rocsparse_spmat_descr descr,
                     rocsparse_indextype*  row_type,
                     rocsparse_indextype*  col_type,
                     rocsparse_datatype*   data_type,
                     rocsparse_index_base* base_)
{
    rocsparse_format format;
    rocsparse_spmat_get_format(descr, &format);
    switch(format)
    {
    case rocsparse_format_bell:
    {
        std::cerr << " initialize bell not implemented " << std::endl;
        throw(rocsparse_status_not_implemented);
    }
    case rocsparse_format_ell:
    {
        int64_t              m;
        int64_t              n;
        void*                ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        int64_t              ell_width;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_ell_get(descr, &m, &n, &ind, &val, &ell_width, &ind_type, &base, &val_type));
        row_type[0]  = ind_type;
        col_type[0]  = ind_type;
        data_type[0] = val_type;
        base_[0]     = base;
        return;
    }

    case rocsparse_format_bsr:
    {
        int64_t              mb;
        int64_t              nb;
        int64_t              nnzb;
        rocsparse_direction  dirb;
        int64_t              dimb;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_bsr_get(descr,
                                                &mb,
                                                &nb,
                                                &nnzb,
                                                &dirb,
                                                &dimb,
                                                &ptr,
                                                &ind,
                                                &val,
                                                &ptr_type,
                                                &ind_type,
                                                &base,
                                                &val_type));
        row_type[0]  = ptr_type;
        col_type[0]  = ind_type;
        data_type[0] = val_type;
        base_[0]     = base;
        return;
    }
    case rocsparse_format_csr:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
        row_type[0]  = ptr_type;
        col_type[0]  = ind_type;
        data_type[0] = val_type;
        base_[0]     = base;
        return;
    }
    case rocsparse_format_csc:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ptr;
        void*                ind;
        void*                val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_csc_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
        col_type[0]  = ptr_type;
        row_type[0]  = ind_type;
        data_type[0] = val_type;
        base_[0]     = base;
        return;
    }
    case rocsparse_format_coo:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                row_ind;
        void*                col_ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_coo_get(
            descr, &m, &n, &nnz, &row_ind, &col_ind, &val, &ind_type, &base, &val_type));
        row_type[0]  = ind_type;
        col_type[0]  = ind_type;
        data_type[0] = val_type;
        base_[0]     = base;
        return;
    }
    case rocsparse_format_coo_aos:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        void*                ind;
        void*                val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_coo_aos_get(descr, &m, &n, &nnz, &ind, &val, &ind_type, &base, &val_type));
        row_type[0]  = ind_type;
        col_type[0]  = ind_type;
        data_type[0] = val_type;
        base_[0]     = base;
        return;
    }
    }
}

void spmat_bsr_get_block_dim(rocsparse_const_spmat_descr descr,
                             rocsparse_direction*        block_dir,
                             int64_t*                    block_dim)
{
    rocsparse_format format;
    rocsparse_spmat_get_format(descr, &format);
    switch(format)
    {
    case rocsparse_format_bsr:
    {
        int64_t              mb;
        int64_t              nb;
        int64_t              nnzb;
        rocsparse_direction  dirb;
        int64_t              dimb;
        const void*          ptr;
        const void*          ind;
        const void*          val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_bsr_get(descr,
                                                      &mb,
                                                      &nb,
                                                      &nnzb,
                                                      &dirb,
                                                      &dimb,
                                                      &ptr,
                                                      &ind,
                                                      &val,
                                                      &ptr_type,
                                                      &ind_type,
                                                      &base,
                                                      &val_type));
        block_dim[0] = dimb;
        block_dir[0] = dirb;
        return;
    }
    case rocsparse_format_bell:
    case rocsparse_format_ell:
    case rocsparse_format_csr:
    case rocsparse_format_csc:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        std::cerr << " wrong format " << std::endl;
        throw(rocsparse_status_not_implemented);
        return;
    }
    }
}

void spmat_convert(rocsparse_handle               handle, //
                   const spmat&                   A, //
                   spmat&                         B, //
                   rocsparse_sparse_to_sparse_alg alg, //
                   bool                           verbose)
{
    hipStream_t stream;
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    //
    // Create descriptor.
    //
    rocsparse_sparse_to_sparse_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sparse_to_sparse_descr(&descr, A, B, alg));
    CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse_permissive(descr));

    //
    // Analysis
    //
    {
        size_t buffer_size;
        void*  buffer;
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse_buffer_size(
            handle, descr, A, B, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse(
            handle, descr, A, B, rocsparse_sparse_to_sparse_stage_analysis, buffer_size, buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }

    //
    // Allocate.
    //
    spmat_allocate(B, B.row_data, B.col_data, B.ind_data, B.val_data);

    //
    // Compute.
    //
    {
        size_t buffer_size;
        void*  buffer;
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse_buffer_size(
            handle, descr, A, B, rocsparse_sparse_to_sparse_stage_compute, &buffer_size));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse(
            handle, descr, A, B, rocsparse_sparse_to_sparse_stage_compute, buffer_size, buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sparse_to_sparse_descr(descr));
}

//
//
//

spmat spmat_create_B(spmat& A, const Arguments& arg, rocsparse_format format_B)
{
    int64_t M;
    int64_t N;
    int64_t NNZ;
    spmat_get_rawsize(A, &M, &N, &NNZ);

    switch(format_B)
    {
    case rocsparse_format_csr:
    {
        return {arg.formatB,
                M,
                N,
                NNZ,
                arg.B_row_indextype,
                arg.B_col_indextype,
                arg.b_type,
                arg.baseB};
    }

    case rocsparse_format_csc:
    {
        return {arg.formatB,
                M,
                N,
                NNZ,
                arg.B_col_indextype,
                arg.B_row_indextype,
                arg.b_type,
                arg.baseB};
    }

    case rocsparse_format_ell:
    {
        return {arg.formatB, M, N, arg.B_col_indextype, arg.b_type, arg.baseB};
    }

    case rocsparse_format_bsr:
    {
        rocsparse_format format_A;
        rocsparse_spmat_get_format(A, &format_A);

        if(format_A == rocsparse_format_bsr)
        {
            const int64_t mb = (M - 1) / arg.row_block_dimB + 1;
            const int64_t nb = (N - 1) / arg.col_block_dimB + 1;
            return {format_B,
                    mb,
                    nb,
                    NNZ / (arg.row_block_dimB * arg.col_block_dimB),
                    rocsparse_direction_column,
                    arg.row_block_dimB,
                    arg.B_row_indextype,
                    arg.B_col_indextype,
                    arg.b_type,
                    arg.baseB};
        }
        else
        {
            const int64_t mb = (M - 1) / arg.row_block_dimB + 1;
            const int64_t nb = (N - 1) / arg.col_block_dimB + 1;
            return {format_B,
                    mb,
                    nb,
                    0,
                    rocsparse_direction_column,
                    arg.row_block_dimB,
                    arg.B_row_indextype,
                    arg.B_col_indextype,
                    arg.b_type,
                    arg.baseB};
        }
    }

    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        return {format_B, M, N, NNZ, arg.B_row_indextype, arg.b_type, arg.baseB};
    }

    case rocsparse_format_bell:
    {
        std::cerr << "not implemented " << std::endl;
        throw(rocsparse_status_internal_error);
    }
    }
}

spmat spmat_create_C(spmat& A, spmat& B)
{
    int64_t          M;
    int64_t          N;
    int64_t          NNZ;
    rocsparse_format format_A;
    rocsparse_spmat_get_format(A, &format_A);

    spmat_get_rawsize(B, &M, &N, &NNZ);

    rocsparse_indextype  row_type, col_type;
    rocsparse_datatype   data_type;
    rocsparse_index_base baseA;
    spmat_get_types(A, &row_type, &col_type, &data_type, &baseA);

    switch(format_A)
    {
    case rocsparse_format_csr:
    {
        return {format_A, M, N, NNZ, row_type, col_type, data_type, baseA};
    }

    case rocsparse_format_csc:
    {
        return {format_A, M, N, NNZ, row_type, col_type, data_type, baseA};
    }

    case rocsparse_format_ell:
    {
        return {format_A, M, N, col_type, data_type, baseA};
    }

    case rocsparse_format_bsr:
    {
        rocsparse_direction block_dir;
        int64_t             block_dim;
        spmat_bsr_get_block_dim(A, &block_dir, &block_dim);
        const int64_t mb = (M - 1) / block_dim + 1;
        const int64_t nb = (N - 1) / block_dim + 1;
        return {format_A,
                mb,
                nb,
                NNZ / (block_dim * block_dim),
                block_dir,
                block_dim,
                row_type,
                col_type,
                data_type,
                baseA};
    }

    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        return {format_A, M, N, NNZ, col_type, data_type, baseA};
    }
    case rocsparse_format_bell:
    {
        std::cerr << "not implemented " << std::endl;
        throw(rocsparse_status_internal_error);
    }
    }
}

template <typename I, typename J, typename T>
spmat spmat_create_A(const Arguments& arg, rocsparse_format format, rocsparse_index_base base)
{
    switch(format)
    {
    case rocsparse_format_csr:
    {
        host_csr_matrix<T, I, J>          h;
        J                                 M = arg.M;
        J                                 N = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg);
        matrix_factory.init_csr(h, M, N, base);
        return spmat(h);
    }

    case rocsparse_format_ell:
    {
        host_ell_matrix<T, J>             h;
        J                                 M = arg.M;
        J                                 N = arg.N;
        rocsparse_matrix_factory<T, J, J> matrix_factory(arg);

        matrix_factory.init_ell(h, M, N, base);
        return spmat(h);
    }

    case rocsparse_format_csc:
    {
        host_csc_matrix<T, I, J>          h;
        J                                 M = arg.M;
        J                                 N = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg);
        matrix_factory.init_csc(h, M, N, base);
        return spmat(h);
    }

    case rocsparse_format_coo:
    {
        host_coo_matrix<T, J>             h;
        J                                 M = arg.M;
        J                                 N = arg.N;
        rocsparse_matrix_factory<T, J, J> matrix_factory(arg);
        matrix_factory.init_coo(h, M, N, base);
        return spmat(h);
    }

    case rocsparse_format_bsr:
    {
        host_gebsr_matrix<T, I, J>        h;
        J                                 Mb = arg.M;
        J                                 Nb = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg);
        J                                 row_block_dim = arg.block_dim;
        J                                 col_block_dim = arg.block_dim;
        matrix_factory.init_gebsr(h, Mb, Nb, row_block_dim, col_block_dim, arg.baseA);
        return spmat(h);
    }

    case rocsparse_format_bell:
    {
        std::cerr << "not implemented " << std::endl;
        throw(rocsparse_status_internal_error);
    }

    case rocsparse_format_coo_aos:
    {
        host_coo_aos_matrix<T, J>         h;
        J                                 M = arg.M;
        J                                 N = arg.N;
        rocsparse_matrix_factory<T, J, J> matrix_factory(arg);
        matrix_factory.init_coo_aos(h, M, N, base);
        return spmat(h);
    }
    }
}

void testing_sparse_to_sparse_extra(const Arguments& arg)
{
    // 1 0 0 0
    // 1 2 3 4
    // 0 0 3 4
    // 1 0 0 5
    int                  m    = 4;
    int                  n    = 4;
    int                  nnz  = 9;
    rocsparse_index_base base = rocsparse_index_base_zero;

    host_csr_matrix<float, int, int> hcsr(m, n, nnz, base);
    hcsr.ptr = {0, 1, 5, 7, 9};
    hcsr.ind = {0, 0, 1, 2, 3, 2, 3, 0, 3};
    hcsr.val = {1, 1, 2, 3, 4, 3, 4, 1, 5};

    device_csr_matrix<float, int, int> dcsr(hcsr);

    device_csr_matrix<float, int, int> dcsc(m, n, nnz, base);

    rocsparse_handle handle;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));

    // Build Source
    rocsparse_const_spmat_descr source;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_const_csr_descr(&source,
                                                           dcsr.m,
                                                           dcsr.n,
                                                           dcsr.nnz,
                                                           dcsr.ptr,
                                                           dcsr.ind,
                                                           (const float*)dcsr.val,
                                                           rocsparse_indextype_i32,
                                                           rocsparse_indextype_i32,
                                                           dcsr.base,
                                                           rocsparse_datatype_f32_r));

    // Build target
    rocsparse_spmat_descr target;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_csc_descr(&target,
                                                     dcsc.m,
                                                     dcsc.n,
                                                     dcsc.nnz,
                                                     dcsc.ptr,
                                                     dcsc.ind,
                                                     (float*)dcsc.val,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_indextype_i32,
                                                     dcsc.base,
                                                     rocsparse_datatype_f32_r));

    // Create descriptor
    rocsparse_sparse_to_sparse_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sparse_to_sparse_descr(
        &descr, source, target, rocsparse_sparse_to_sparse_alg_default));

    size_t buffer_size;
    void*  buffer;

    // Analysis phase
    CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse_buffer_size(
        handle, descr, source, target, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size));
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size));
    CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse(handle,
                                                     descr,
                                                     source,
                                                     target,
                                                     rocsparse_sparse_to_sparse_stage_analysis,
                                                     buffer_size,
                                                     buffer));
    CHECK_HIP_ERROR(hipFree(buffer));

    // Calculation phase
    CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse_buffer_size(
        handle, descr, source, target, rocsparse_sparse_to_sparse_stage_compute, &buffer_size));
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size));
    CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_sparse(handle,
                                                     descr,
                                                     source,
                                                     target,
                                                     rocsparse_sparse_to_sparse_stage_compute,
                                                     buffer_size,
                                                     buffer));
    CHECK_HIP_ERROR(hipFree(buffer));

    host_csr_matrix<float, int, int> hcsc_solution(dcsc);

    host_csr_matrix<float, int, int> hcsc_gold(m, n, nnz, base);
    hcsc_gold.ptr = {0, 3, 4, 6, 9};
    hcsc_gold.ind = {0, 1, 3, 1, 1, 2, 1, 2, 3};
    hcsc_gold.val = {1, 1, 1, 2, 3, 3, 4, 4, 5};

    unit_check_garray(rocsparse_indextype_i32, n + 1, hcsc_solution.ptr, hcsc_gold.ptr);
    unit_check_garray(rocsparse_indextype_i32, nnz, hcsc_solution.ind, hcsc_gold.ind);
    unit_check_garray(rocsparse_datatype_f32_r, nnz, hcsc_solution.val, hcsc_gold.val);

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(source));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(target));

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
}

template <typename I, typename J, typename T>
void testing_sparse_to_sparse_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle           local_handle;
    rocsparse_spmat_descr            source            = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr            target            = (rocsparse_spmat_descr)0x4;
    rocsparse_handle                 handle            = local_handle;
    rocsparse_sparse_to_sparse_alg   alg               = rocsparse_sparse_to_sparse_alg_default;
    rocsparse_sparse_to_sparse_stage stage             = rocsparse_sparse_to_sparse_stage_analysis;
    size_t                           local_buffer_size = 100;
    void*                            buffer            = (void*)0x4;
    {
        rocsparse_sparse_to_sparse_descr* descr = (rocsparse_sparse_to_sparse_descr*)0x4;
        bad_arg_analysis(rocsparse_create_sparse_to_sparse_descr, descr, source, target, alg);
    }

    {
        rocsparse_sparse_to_sparse_descr descr = (rocsparse_sparse_to_sparse_descr)0x4;
        bad_arg_analysis(rocsparse_sparse_to_sparse_permissive, descr);

        {
            size_t* buffer_size_in_bytes = &local_buffer_size;
            bad_arg_analysis(rocsparse_sparse_to_sparse_buffer_size,
                             handle,
                             descr,
                             source,
                             target,
                             stage,
                             buffer_size_in_bytes);
        }

        {
            size_t               buffer_size_in_bytes              = local_buffer_size;
            static constexpr int nargs_to_exclude                  = 1;
            static constexpr int args_to_exclude[nargs_to_exclude] = {5};

            select_bad_arg_analysis(rocsparse_sparse_to_sparse,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    handle,
                                    descr,
                                    source,
                                    target,
                                    stage,
                                    buffer_size_in_bytes,
                                    buffer);
        }
    }
}

void spmat_check(rocsparse_handle handle, spmat& A)
{
    rocsparse_format format_A;
    rocsparse_spmat_get_format(A, &format_A);
    if(format_A != rocsparse_format_coo_aos && format_A != rocsparse_format_bell)
    {
        size_t                buffer_size;
        rocsparse_data_status data_status;
        CHECK_ROCSPARSE_ERROR(rocsparse_check_spmat(handle,
                                                    A,
                                                    &data_status,
                                                    rocsparse_check_spmat_stage_buffer_size,
                                                    &buffer_size,
                                                    nullptr));
        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_check_spmat(
            handle, A, &data_status, rocsparse_check_spmat_stage_compute, &buffer_size, buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }
}

void spmat_compare(rocsparse_const_spmat_descr A, rocsparse_const_spmat_descr B)
{
    rocsparse_format format_A;
    rocsparse_format format_B;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_format(A, &format_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_format(B, &format_B));
    CHECK_ROCSPARSE_ERROR((format_A != format_B) ? rocsparse_status_internal_error
                                                 : rocsparse_status_success);

    switch(format_A)
    {
    case rocsparse_format_bell:
    {
        break;
    }
    case rocsparse_format_ell:
    {
        int64_t              m_A;
        int64_t              n_A;
        const void*          ind_A;
        const void*          val_A;
        rocsparse_index_base base_A;
        rocsparse_datatype   val_type_A;
        rocsparse_indextype  ind_type_A;
        int64_t              ell_width_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_ell_get(
            A, &m_A, &n_A, &ind_A, &val_A, &ell_width_A, &ind_type_A, &base_A, &val_type_A));

        int64_t              m_B;
        int64_t              n_B;
        const void*          ind_B;
        const void*          val_B;
        rocsparse_index_base base_B;
        rocsparse_datatype   val_type_B;
        rocsparse_indextype  ind_type_B;
        int64_t              ell_width_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_ell_get(
            B, &m_B, &n_B, &ind_B, &val_B, &ell_width_B, &ind_type_B, &base_B, &val_type_B));

        unit_check_scalar(m_A, m_B);
        unit_check_scalar(n_A, n_B);
        unit_check_scalar(ell_width_A, ell_width_B);
        unit_check_enum(base_A, base_B);
        unit_check_enum(ind_type_A, ind_type_B);
        unit_check_enum(val_type_A, val_type_B);
        unit_check_garray(ind_type_A, m_A * ell_width_A, ind_A, ind_B);
        unit_check_garray(val_type_A, m_A * ell_width_A, val_A, val_B);

        break;
    }

    case rocsparse_format_bsr:
    {
        int64_t              mb_A;
        int64_t              nb_A;
        int64_t              nnzb_A;
        rocsparse_direction  dirb_A;
        int64_t              dimb_A;
        const void*          ptr_A;
        const void*          ind_A;
        const void*          val_A;
        rocsparse_indextype  ptr_type_A;
        rocsparse_indextype  ind_type_A;
        rocsparse_index_base base_A;
        rocsparse_datatype   val_type_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_bsr_get(A,
                                                      &mb_A,
                                                      &nb_A,
                                                      &nnzb_A,
                                                      &dirb_A,
                                                      &dimb_A,
                                                      &ptr_A,
                                                      &ind_A,
                                                      &val_A,
                                                      &ptr_type_A,
                                                      &ind_type_A,
                                                      &base_A,
                                                      &val_type_A));
        int64_t              mb_B;
        int64_t              nb_B;
        int64_t              nnzb_B;
        rocsparse_direction  dirb_B;
        int64_t              dimb_B;
        const void*          ptr_B;
        const void*          ind_B;
        const void*          val_B;
        rocsparse_indextype  ptr_type_B;
        rocsparse_indextype  ind_type_B;
        rocsparse_index_base base_B;
        rocsparse_datatype   val_type_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_bsr_get(B,
                                                      &mb_B,
                                                      &nb_B,
                                                      &nnzb_B,
                                                      &dirb_B,
                                                      &dimb_B,
                                                      &ptr_B,
                                                      &ind_B,
                                                      &val_B,
                                                      &ptr_type_B,
                                                      &ind_type_B,
                                                      &base_B,
                                                      &val_type_B));

        unit_check_scalar(mb_A, mb_B);
        unit_check_scalar(nb_A, nb_B);
        unit_check_scalar(nnzb_A, nnzb_B);
        unit_check_scalar(dimb_A, dimb_B);
        unit_check_enum(base_A, base_B);
        unit_check_enum(dirb_A, dirb_B);
        unit_check_enum(ptr_type_A, ptr_type_B);
        unit_check_enum(ind_type_A, ind_type_B);
        unit_check_enum(val_type_A, val_type_B);
        unit_check_garray(ptr_type_A, mb_A + 1, ptr_A, ptr_B);
        unit_check_garray(ind_type_A, nnzb_A, ind_A, ind_B);
        unit_check_garray(val_type_A, nnzb_A * dimb_A * dimb_A, val_A, val_B);
        break;
    }
    case rocsparse_format_csr:
    {
        int64_t              m_A;
        int64_t              n_A;
        int64_t              nnz_A;
        const void*          ptr_A;
        const void*          ind_A;
        const void*          val_A;
        rocsparse_indextype  ptr_type_A;
        rocsparse_indextype  col_type_A;
        rocsparse_index_base base_A;
        rocsparse_datatype   val_type_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_csr_get(A,
                                                      &m_A,
                                                      &n_A,
                                                      &nnz_A,
                                                      &ptr_A,
                                                      &ind_A,
                                                      &val_A,
                                                      &ptr_type_A,
                                                      &col_type_A,
                                                      &base_A,
                                                      &val_type_A));
        int64_t              m_B;
        int64_t              n_B;
        int64_t              nnz_B;
        const void*          ptr_B;
        const void*          ind_B;
        const void*          val_B;
        rocsparse_indextype  ptr_type_B;
        rocsparse_indextype  col_type_B;
        rocsparse_index_base base_B;
        rocsparse_datatype   val_type_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_csr_get(B,
                                                      &m_B,
                                                      &n_B,
                                                      &nnz_B,
                                                      &ptr_B,
                                                      &ind_B,
                                                      &val_B,
                                                      &ptr_type_B,
                                                      &col_type_B,
                                                      &base_B,
                                                      &val_type_B));

        unit_check_scalar(m_A, m_B);
        unit_check_scalar(n_A, n_B);
        unit_check_scalar(nnz_A, nnz_B);
        unit_check_enum(base_A, base_B);
        unit_check_enum(col_type_A, col_type_B);
        unit_check_enum(ptr_type_A, ptr_type_B);
        unit_check_enum(val_type_A, val_type_B);
        unit_check_garray(ptr_type_A, m_A + 1, ptr_A, ptr_B);
        unit_check_garray(col_type_A, nnz_A, ind_A, ind_B);
        unit_check_garray(val_type_A, nnz_A, val_A, val_B);
        break;
    }
    case rocsparse_format_csc:
    {
        int64_t              m_A;
        int64_t              n_A;
        int64_t              nnz_A;
        const void*          ptr_A;
        const void*          ind_A;
        const void*          val_A;
        rocsparse_indextype  ptr_type_A;
        rocsparse_indextype  ind_type_A;
        rocsparse_index_base base_A;
        rocsparse_datatype   val_type_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_csc_get(A,
                                                      &m_A,
                                                      &n_A,
                                                      &nnz_A,
                                                      &ptr_A,
                                                      &ind_A,
                                                      &val_A,
                                                      &ptr_type_A,
                                                      &ind_type_A,
                                                      &base_A,
                                                      &val_type_A));

        int64_t              m_B;
        int64_t              n_B;
        int64_t              nnz_B;
        const void*          ptr_B;
        const void*          ind_B;
        const void*          val_B;
        rocsparse_indextype  ptr_type_B;
        rocsparse_indextype  ind_type_B;
        rocsparse_index_base base_B;
        rocsparse_datatype   val_type_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_csc_get(B,
                                                      &m_B,
                                                      &n_B,
                                                      &nnz_B,
                                                      &ptr_B,
                                                      &ind_B,
                                                      &val_B,
                                                      &ptr_type_B,
                                                      &ind_type_B,
                                                      &base_B,
                                                      &val_type_B));
        unit_check_scalar(m_A, m_B);
        unit_check_scalar(n_A, n_B);
        unit_check_scalar(nnz_A, nnz_B);
        unit_check_enum(base_A, base_B);
        unit_check_enum(ptr_type_A, ptr_type_B);
        unit_check_enum(ind_type_A, ind_type_B);
        unit_check_enum(val_type_A, val_type_B);
        unit_check_garray(ptr_type_A, n_A + 1, ptr_A, ptr_B);
        unit_check_garray(ind_type_A, nnz_A, ind_A, ind_B);
        unit_check_garray(val_type_A, nnz_A, val_A, val_B);
        break;
    }
    case rocsparse_format_coo:
    {
        int64_t              m_A;
        int64_t              n_A;
        int64_t              nnz_A;
        const void*          row_ind_A;
        const void*          col_ind_A;
        const void*          val_A;
        rocsparse_indextype  ind_type_A;
        rocsparse_index_base base_A;
        rocsparse_datatype   val_type_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_coo_get(A,
                                                      &m_A,
                                                      &n_A,
                                                      &nnz_A,
                                                      &row_ind_A,
                                                      &col_ind_A,
                                                      &val_A,
                                                      &ind_type_A,
                                                      &base_A,
                                                      &val_type_A));
        int64_t              m_B;
        int64_t              n_B;
        int64_t              nnz_B;
        const void*          row_ind_B;
        const void*          col_ind_B;
        const void*          val_B;
        rocsparse_indextype  ind_type_B;
        rocsparse_index_base base_B;
        rocsparse_datatype   val_type_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_coo_get(B,
                                                      &m_B,
                                                      &n_B,
                                                      &nnz_B,
                                                      &row_ind_B,
                                                      &col_ind_B,
                                                      &val_B,
                                                      &ind_type_B,
                                                      &base_B,
                                                      &val_type_B));

        unit_check_scalar(m_A, m_B);
        unit_check_scalar(n_A, n_B);
        unit_check_scalar(nnz_A, nnz_B);
        unit_check_enum(base_A, base_B);
        unit_check_enum(ind_type_A, ind_type_B);
        unit_check_enum(val_type_A, val_type_B);
        unit_check_garray(ind_type_B, nnz_A, row_ind_A, row_ind_B);
        unit_check_garray(ind_type_B, nnz_A, col_ind_A, col_ind_B);
        unit_check_garray(val_type_A, nnz_A, val_A, val_B);
        break;
    }
    case rocsparse_format_coo_aos:
    {
        int64_t              m_A;
        int64_t              n_A;
        int64_t              nnz_A;
        const void*          ind_A;
        const void*          val_A;
        rocsparse_indextype  ind_type_A;
        rocsparse_index_base base_A;
        rocsparse_datatype   val_type_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_coo_aos_get(
            A, &m_A, &n_A, &nnz_A, &ind_A, &val_A, &ind_type_A, &base_A, &val_type_A));
        int64_t              m_B;
        int64_t              n_B;
        int64_t              nnz_B;
        const void*          ind_B;
        const void*          val_B;
        rocsparse_indextype  ind_type_B;
        rocsparse_index_base base_B;
        rocsparse_datatype   val_type_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_coo_aos_get(
            B, &m_B, &n_B, &nnz_B, &ind_B, &val_B, &ind_type_B, &base_B, &val_type_B));
        unit_check_scalar(m_A, m_B);
        unit_check_scalar(n_A, n_B);
        unit_check_scalar(nnz_A, nnz_B);
        unit_check_enum(base_A, base_B);
        unit_check_enum(ind_type_A, ind_type_B);
        unit_check_enum(val_type_A, val_type_B);
        unit_check_garray(ind_type_A, nnz_A * 2, ind_A, ind_B);
        unit_check_garray(val_type_A, nnz_A, val_A, val_B);
        break;
    }
    }
}

template <typename I, typename J, typename T>
void testing_sparse_to_sparse_template(const Arguments& arg)
{
    static constexpr bool verbose = false;
    if(verbose)
        std::cout << "// TEST : arg.formatA " << rocsparse_format2string(arg.formatA) << std::endl;
    if(verbose)
        std::cout << "// TEST : arg.formatB " << rocsparse_format2string(arg.formatB) << std::endl;

    rocsparse_local_handle               handle;
    const rocsparse_sparse_to_sparse_alg alg = rocsparse_sparse_to_sparse_alg_default;
    if(verbose)
        std::cout << "// TEST : create A " << std::endl;

    //
    // Create matrix A.
    //
    spmat A = spmat_create_A<I, J, T>(arg, arg.formatA, arg.baseA);
    if(verbose)
        std::cout << "// TEST : check  A " << std::endl;

    //
    // Check matrix A.
    //
    spmat_check(handle, A);

    if(verbose)
        std::cout << "// TEST : create B " << std::endl;
    //
    // Create matrix B.
    //
    spmat B = spmat_create_B(A, arg, arg.formatB);

    if(arg.unit_check)
    {
        if(verbose)
            std::cout << "// TEST : convert A to B " << std::endl;
        //
        // Convert.
        //
        spmat_convert(handle, A, B, alg, verbose);
        if(verbose)
            std::cout << "// TEST : check B " << std::endl;
        //
        // Check matrix B.
        //
        spmat_check(handle, B);

        //
        // Create matrix C
        //
        if(verbose)
            std::cout << "// TEST : create C " << std::endl;
        spmat C = spmat_create_C(A, B);
        if(verbose)
            std::cout << "// TEST : convert B to C " << std::endl;
        //
        //
        //
        spmat_convert(handle, B, C, alg, verbose);
        if(verbose)
            std::cout << "// TEST : check C " << std::endl;
        //
        // Check matrix C.
        //
        spmat_check(handle, C);

        //
        // Compate A and C.
        //
        if(arg.formatA != rocsparse_format_bsr && arg.formatB != rocsparse_format_bsr)
        {
            if(verbose)
                std::cout << "// TEST : check A and C " << std::endl;
            spmat_compare(A, C);
        }
    }

    if(arg.timing)
    {
    }
}

template <typename I, typename J, typename T>
void testing_sparse_to_sparse(const Arguments& arg)
{
    testing_sparse_to_sparse_template<I, J, T>(arg);
}

#define INSTANTIATE(ITYPE, JTYPE, TYPE)                                                       \
    template void testing_sparse_to_sparse_bad_arg<ITYPE, JTYPE, TYPE>(const Arguments& arg); \
    template void testing_sparse_to_sparse<ITYPE, JTYPE, TYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, _Float16);
INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, _Float16);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, _Float16);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
