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

// LCOV_EXCL_START
#include "rocsparse_internal_spmat_print.hpp"
#include "rocsparse_convert_array.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    inline std::ostream& operator<<(std::ostream& out, const _Float16& x)
    {
        out << (float)x;
        return out;
    }

    template <typename T>
    static rocsparse_status internal_dnvec_print(std::ostream& out, int64_t nmemb, const void* h)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const T* p = (const T*)h;
        for(int64_t i = 0; i < nmemb; ++i)
            out << "[" << i << "] = " << p[i] << std::endl;
        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status
        internal_dnmat_print(std::ostream& out, int64_t m, int64_t n, const void* h, int64_t ld)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const T* p = (const T*)h;
        for(int64_t i = 0; i < m; ++i)
        {
            for(int64_t j = 0; j < n; ++j)
                out << " " << p[j * ld + i];
            out << std::endl;
        }
        return rocsparse_status_success;
    }

    static rocsparse_status internal_dnvec_print(std::ostream&       out,
                                                 rocsparse_indextype indextype,
                                                 int64_t             nmemb,
                                                 const void*         dind)

    {
        ROCSPARSE_ROUTINE_TRACE;

        if(dind == nullptr || nmemb == 0)
        {
            return rocsparse_status_success;
        }

        const size_t indextype_sizeof = rocsparse::indextype_sizeof(indextype);
        void*        hind;
        RETURN_IF_HIP_ERROR(rocsparse_hipHostMalloc(&hind, indextype_sizeof * nmemb));
        RETURN_IF_HIP_ERROR(hipMemcpy(hind, dind, indextype_sizeof * nmemb, hipMemcpyDeviceToHost));
        switch(indextype)
        {
        case rocsparse_indextype_i32:
        {
            rocsparse::internal_dnvec_print<int32_t>(out, nmemb, hind);
            break;
        }
        case rocsparse_indextype_i64:
        {
            rocsparse::internal_dnvec_print<int64_t>(out, nmemb, hind);
            break;
        }
        case rocsparse_indextype_u16:
        {
            break;
        }
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hind));
        return rocsparse_status_success;
    }

    static rocsparse_status internal_dnmat_print(
        std::ostream& out, rocsparse_indextype indextype, int64_t m, int64_t n, const void* dind)

    {
        ROCSPARSE_ROUTINE_TRACE;

        if(dind == nullptr || m == 0 || n == 0)
        {
            return rocsparse_status_success;
        }

        const size_t indextype_sizeof = rocsparse::indextype_sizeof(indextype);
        void*        hind;
        RETURN_IF_HIP_ERROR(rocsparse_hipHostMalloc(&hind, indextype_sizeof * m * n));
        RETURN_IF_HIP_ERROR(hipMemcpy(hind, dind, indextype_sizeof * m * n, hipMemcpyDeviceToHost));
        switch(indextype)
        {
        case rocsparse_indextype_i32:
        {
            rocsparse::internal_dnmat_print<int32_t>(out, m, n, hind, m);
            break;
        }
        case rocsparse_indextype_i64:
        {
            rocsparse::internal_dnmat_print<int64_t>(out, m, n, hind, m);
            break;
        }
        case rocsparse_indextype_u16:
        {
            break;
        }
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hind));
        return rocsparse_status_success;
    }

    static rocsparse_status internal_dnvec_print(std::ostream&      out,
                                                 rocsparse_datatype datatype,
                                                 int64_t            nmemb,
                                                 const void*        dind)

    {
        ROCSPARSE_ROUTINE_TRACE;

        if(dind == nullptr || nmemb == 0)
        {
            return rocsparse_status_success;
        }
        const size_t datatype_sizeof = rocsparse::datatype_sizeof(datatype);
        void*        hind;
        RETURN_IF_HIP_ERROR(rocsparse_hipHostMalloc(&hind, datatype_sizeof * nmemb));
        RETURN_IF_HIP_ERROR(hipMemcpy(hind, dind, datatype_sizeof * nmemb, hipMemcpyDeviceToHost));
        switch(datatype)
        {
        case rocsparse_datatype_f16_r:
        {
            rocsparse::internal_dnvec_print<_Float16>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_bf16_r:
        {
            rocsparse::internal_dnvec_print<rocsparse_bfloat16>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_f32_r:
        {
            rocsparse::internal_dnvec_print<float>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_f32_c:
        {
            rocsparse::internal_dnvec_print<rocsparse_float_complex>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_f64_r:
        {
            rocsparse::internal_dnvec_print<double>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_f64_c:
        {
            rocsparse::internal_dnvec_print<rocsparse_double_complex>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_i32_r:
        {
            rocsparse::internal_dnvec_print<int32_t>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_u32_r:
        {
            rocsparse::internal_dnvec_print<uint32_t>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_i8_r:
        {
            rocsparse::internal_dnvec_print<int8_t>(out, nmemb, hind);
            break;
        }
        case rocsparse_datatype_u8_r:
        {
            rocsparse::internal_dnvec_print<uint8_t>(out, nmemb, hind);
            break;
        }
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hind));
        return rocsparse_status_success;
    }

    static rocsparse_status internal_dnmat_print(
        std::ostream& out, rocsparse_datatype datatype, int64_t m, int64_t n, const void* dind)

    {
        ROCSPARSE_ROUTINE_TRACE;

        if(dind == nullptr || m == 0 || n == 0)
        {
            return rocsparse_status_success;
        }
        const size_t datatype_sizeof = rocsparse::datatype_sizeof(datatype);
        void*        hind;
        RETURN_IF_HIP_ERROR(rocsparse_hipHostMalloc(&hind, datatype_sizeof * m * n));
        RETURN_IF_HIP_ERROR(hipMemcpy(hind, dind, datatype_sizeof * m * n, hipMemcpyDeviceToHost));
        switch(datatype)
        {
        case rocsparse_datatype_f16_r:
        {
            rocsparse::internal_dnmat_print<_Float16>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_bf16_r:
        {
            rocsparse::internal_dnmat_print<rocsparse_bfloat16>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_f32_r:
        {
            rocsparse::internal_dnmat_print<float>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_f32_c:
        {
            rocsparse::internal_dnmat_print<rocsparse_float_complex>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_f64_r:
        {
            rocsparse::internal_dnmat_print<double>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_f64_c:
        {
            rocsparse::internal_dnmat_print<rocsparse_double_complex>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_i32_r:
        {
            rocsparse::internal_dnmat_print<int32_t>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_u32_r:
        {
            rocsparse::internal_dnmat_print<uint32_t>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_i8_r:
        {
            rocsparse::internal_dnmat_print<int8_t>(out, m, n, hind, m);
            break;
        }
        case rocsparse_datatype_u8_r:
        {
            rocsparse::internal_dnmat_print<uint8_t>(out, m, n, hind, m);
            break;
        }
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hind));
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::internal_spmat_print(std::ostream&               out,
                                                 rocsparse_const_spmat_descr descr,
                                                 bool                        print_symbolic,
                                                 bool                        print_numeric)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, descr);
    rocsparse_format format;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(descr, &format));
    switch(format)
    {
    case rocsparse_format_bell:
    {
        int64_t              m;
        int64_t              n;
        const void*          ind;
        const void*          val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        int64_t              ell_widthb;
        rocsparse_direction  dirb;
        int64_t              dimb;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_bell_get(
            descr, &m, &n, &dirb, &dimb, &ell_widthb, &ind, &val, &ind_type, &base, &val_type));
        out << "- format     : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- m         : " << m << std::endl;
        out << "- n         : " << n << std::endl;
        out << "- width     : " << ell_widthb << std::endl;
        out << "- ind_type  : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        out << "- dirb       : " << rocsparse::enum_utils::to_string(dirb) << std::endl;
        out << "- dimb       : " << dimb << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::internal_dnmat_print(out, ind_type, m, ell_widthb, ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::internal_dnvec_print(out, val_type, ell_widthb * m * dimb * dimb, val));
        }
        break;
    }
    case rocsparse_format_ell:

    {
        int64_t              m;
        int64_t              n;
        const void*          ind;
        const void*          val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        int64_t              ell_width;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_ell_get(
            descr, &m, &n, &ind, &val, &ell_width, &ind_type, &base, &val_type));
        out << "- format    : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- m         : " << m << std::endl;
        out << "- n         : " << n << std::endl;
        out << "- ell width : " << ell_width << std::endl;
        out << "- ind_type  : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::internal_dnmat_print(out, ind_type, m, ell_width, ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::internal_dnmat_print(out, val_type, m, ell_width, val));
        }
        break;
    }

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
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_bsr_get(descr,
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
        out << "- format     : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- mb         : " << mb << std::endl;
        out << "- nb         : " << nb << std::endl;
        out << "- nnzb       : " << nnzb << std::endl;
        out << "- dirb       : " << rocsparse::enum_utils::to_string(dirb) << std::endl;
        out << "- dimb       : " << dimb << std::endl;
        out << "- m          : " << mb * dimb << std::endl;
        out << "- n          : " << nb * dimb << std::endl;
        out << "- nnz        : " << nnzb * dimb * dimb << std::endl;
        out << "- ptr_type   : " << rocsparse::enum_utils::to_string(ptr_type) << std::endl;
        out << "- ind_type   : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type  : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ptr_type, mb + 1, ptr));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ind_type, nnzb, ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::internal_dnvec_print(out, val_type, nnzb * dimb * dimb, val));
        }
        break;
    }
    case rocsparse_format_csr:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        const void*          ptr;
        const void*          ind;
        const void*          val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_csr_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
        out << "- format     : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- m          : " << m << std::endl;
        out << "- n          : " << n << std::endl;
        out << "- nnz        : " << nnz << std::endl;
        out << "- ptr_type   : " << rocsparse::enum_utils::to_string(ptr_type) << std::endl;
        out << "- ind_type   : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type  : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        out << "- base       : " << base << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ptr_type, m + 1, ptr));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ind_type, nnz, ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, val_type, nnz, val));
        }
        break;
    }
    case rocsparse_format_csc:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        const void*          ptr;
        const void*          ind;
        const void*          val;
        rocsparse_indextype  ptr_type;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_csc_get(
            descr, &m, &n, &nnz, &ptr, &ind, &val, &ptr_type, &ind_type, &base, &val_type));
        out << "- format     : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- m          : " << m << std::endl;
        out << "- n          : " << n << std::endl;
        out << "- nnz        : " << nnz << std::endl;
        out << "- ptr_type   : " << rocsparse::enum_utils::to_string(ptr_type) << std::endl;
        out << "- ind_type   : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type  : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        out << "- base       : " << base << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ptr_type, n + 1, ptr));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ind_type, nnz, ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, val_type, nnz, val));
        }
        break;
    }
    case rocsparse_format_coo:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        const void*          row_ind;
        const void*          col_ind;
        const void*          val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_coo_get(
            descr, &m, &n, &nnz, &row_ind, &col_ind, &val, &ind_type, &base, &val_type));
        out << "- format     : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- m          : " << m << std::endl;
        out << "- n          : " << n << std::endl;
        out << "- nnz        : " << nnz << std::endl;
        out << "- row_type   : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- col_type   : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type  : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        out << "- base       : " << base << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ind_type, nnz, row_ind));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ind_type, nnz, col_ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, val_type, nnz, val));
        }

        break;
    }
    case rocsparse_format_coo_aos:
    {
        int64_t              m;
        int64_t              n;
        int64_t              nnz;
        const void*          ind;
        const void*          val;
        rocsparse_indextype  ind_type;
        rocsparse_index_base base;
        rocsparse_datatype   val_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_coo_aos_get(
            descr, &m, &n, &nnz, &ind, &val, &ind_type, &base, &val_type));
        out << "- format     : " << rocsparse::enum_utils::to_string(format) << std::endl;
        out << "- m          : " << m << std::endl;
        out << "- n          : " << n << std::endl;
        out << "- nnz        : " << nnz << std::endl;
        out << "- ind_type   : " << rocsparse::enum_utils::to_string(ind_type) << std::endl;
        out << "- data_type  : " << rocsparse::enum_utils::to_string(val_type) << std::endl;
        out << "- base       : " << base << std::endl;
        if(print_symbolic)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, ind_type, nnz * 2, ind));
        }
        if(print_numeric)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_dnvec_print(out, val_type, nnz, val));
        }
        break;
    }
    }
    return rocsparse_status_success;
}
// LCOV_EXCL_STOP
