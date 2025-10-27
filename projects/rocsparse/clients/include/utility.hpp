/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief utility.hpp provides common utilities
 */

#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "rocsparse_clients_routine_trace.hpp"
#include "rocsparse_matrix.hpp"
#include "rocsparse_test.hpp"

#include <hip/hip_runtime_api.h>
#include <vector>

// Return index type
template <typename I>
inline rocsparse_indextype get_indextype(void);

// Return data type
template <typename T>
inline rocsparse_datatype get_datatype(void);

/*! \brief  Return \ref rocsparse_indextype */
template <>
inline rocsparse_indextype get_indextype<uint16_t>(void)
{
    return rocsparse_indextype_u16;
}

template <>
inline rocsparse_indextype get_indextype<int32_t>(void)
{
    return rocsparse_indextype_i32;
}

template <>
inline rocsparse_indextype get_indextype<int64_t>(void)
{
    return rocsparse_indextype_i64;
}

/*! \brief  Return \ref rocsparse_datatype */
template <>
inline rocsparse_datatype get_datatype<int8_t>(void)
{
    return rocsparse_datatype_i8_r;
}

template <>
inline rocsparse_datatype get_datatype<uint8_t>(void)
{
    return rocsparse_datatype_u8_r;
}

template <>
inline rocsparse_datatype get_datatype<int32_t>(void)
{
    return rocsparse_datatype_i32_r;
}

template <>
inline rocsparse_datatype get_datatype<uint32_t>(void)
{
    return rocsparse_datatype_u32_r;
}

template <>
inline rocsparse_datatype get_datatype<_Float16>(void)
{
    return rocsparse_datatype_f16_r;
}

template <>
inline rocsparse_datatype get_datatype<rocsparse_bfloat16>(void)
{
    return rocsparse_datatype_bf16_r;
}

template <>
inline rocsparse_datatype get_datatype<float>(void)
{
    return rocsparse_datatype_f32_r;
}

template <>
inline rocsparse_datatype get_datatype<double>(void)
{
    return rocsparse_datatype_f64_r;
}

template <>
inline rocsparse_datatype get_datatype<rocsparse_float_complex>(void)
{
    return rocsparse_datatype_f32_c;
}

template <>
inline rocsparse_datatype get_datatype<rocsparse_double_complex>(void)
{
    return rocsparse_datatype_f64_c;
}

inline constexpr size_t rocsparse_indextype_sizeof(rocsparse_indextype indextype_)
{
    switch(indextype_)
    {
    case rocsparse_indextype_u16:
    {
        return sizeof(uint16_t);
    }
    case rocsparse_indextype_i32:
    {
        return sizeof(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        return sizeof(int64_t);
    }
    }
    return static_cast<size_t>(0);
}

inline constexpr size_t rocsparse_datatype_sizeof(rocsparse_datatype datatype_)
{
    switch(datatype_)
    {
    case rocsparse_datatype_f16_r:
    {
        return sizeof(_Float16);
    }
    case rocsparse_datatype_bf16_r:
    {
        return sizeof(rocsparse_bfloat16);
    }
    case rocsparse_datatype_f32_r:
    {
        return sizeof(float);
    }
    case rocsparse_datatype_i8_r:
    {
        return sizeof(int8_t);
    }
    case rocsparse_datatype_u8_r:
    {
        return sizeof(uint8_t);
    }
    case rocsparse_datatype_i32_r:
    {
        return sizeof(int32_t);
    }
    case rocsparse_datatype_u32_r:
    {
        return sizeof(uint32_t);
    }
    case rocsparse_datatype_f64_r:
    {
        return sizeof(double);
    }
    case rocsparse_datatype_f32_c:
    {
        return sizeof(rocsparse_float_complex);
    }
    case rocsparse_datatype_f64_c:
    {
        return sizeof(rocsparse_double_complex);
    }
    }
    return static_cast<size_t>(0);
}

inline std::ostream& operator<<(std::ostream& os_, const _Float16& that_)
{
    os_ << (float)that_;
    return os_;
}

/*! \brief  local handle which is automatically created and destroyed  */
class rocsparse_local_handle
{
    rocsparse_handle handle{};

public:
    rocsparse_local_handle()
        : capture_started(false)
        , graph_testing(false)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_handle(&this->handle);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }
    rocsparse_local_handle(const Arguments& arg)
        : capture_started(false)
        , graph_testing(arg.graph_test)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_handle(&this->handle);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }
    ~rocsparse_local_handle()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_handle(this->handle);
    }

    // Allow rocsparse_local_handle to be used anywhere rocsparse_handle is expected
    operator rocsparse_handle&()
    {
        return this->handle;
    }
    operator const rocsparse_handle&() const
    {
        return this->handle;
    }

    void rocsparse_stream_begin_capture()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(!(this->graph_testing))
        {
            return;
        }

#ifdef GOOGLE_TEST
        ASSERT_EQ(capture_started, false);
#endif

        CHECK_HIP_ERROR(hipStreamCreate(&this->graph_stream));
        CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(*this, &this->old_stream));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_stream(*this, this->graph_stream));

        // BEGIN GRAPH CAPTURE
        CHECK_HIP_ERROR(hipStreamBeginCapture(this->graph_stream, hipStreamCaptureModeGlobal));

        capture_started = true;
    }

    void rocsparse_stream_end_capture(rocsparse_int runs = 1)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(!(this->graph_testing))
        {
            return;
        }

#ifdef GOOGLE_TEST
        ASSERT_EQ(capture_started, true);
#endif

        hipGraph_t     graph;
        hipGraphExec_t instance;

        // END GRAPH CAPTURE
        CHECK_HIP_ERROR(hipStreamEndCapture(this->graph_stream, &graph));
        CHECK_HIP_ERROR(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

        CHECK_HIP_ERROR(hipGraphDestroy(graph));
        CHECK_HIP_ERROR(hipGraphLaunch(instance, this->graph_stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(this->graph_stream));
        CHECK_HIP_ERROR(hipGraphExecDestroy(instance));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_stream(*this, this->old_stream));
        CHECK_HIP_ERROR(hipStreamDestroy(this->graph_stream));
        this->graph_stream = nullptr;

        capture_started = false;
    }

    hipStream_t get_stream()
    {
        hipStream_t stream;
        rocsparse_get_stream(*this, &stream);
        return stream;
    }

private:
    hipStream_t graph_stream;
    hipStream_t old_stream;
    bool        capture_started;
    bool        graph_testing;
};

/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class rocsparse_local_mat_descr
{
    rocsparse_mat_descr descr{};

public:
    rocsparse_local_mat_descr()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_mat_descr(&this->descr);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    ~rocsparse_local_mat_descr()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_mat_descr(this->descr);
    }

    // Allow rocsparse_local_mat_descr to be used anywhere rocsparse_mat_descr is expected
    operator rocsparse_mat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_mat_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local matrix info which is automatically created and destroyed  */
class rocsparse_local_mat_info
{
    rocsparse_mat_info info{};

public:
    rocsparse_local_mat_info()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_mat_info(&this->info);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }
    ~rocsparse_local_mat_info()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_mat_info(this->info);
    }

    // Sometimes useful to reset local info
    void reset()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_mat_info(this->info);
        const rocsparse_status status = rocsparse_create_mat_info(&this->info);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    // Allow rocsparse_local_mat_info to be used anywhere rocsparse_mat_info is expected
    operator rocsparse_mat_info&()
    {
        return this->info;
    }
    operator const rocsparse_mat_info&() const
    {
        return this->info;
    }
};

/*! \brief  local color info which is automatically created and destroyed  */
class rocsparse_local_color_info
{
    rocsparse_color_info info{};

public:
    rocsparse_local_color_info()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_color_info(&this->info);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }
    ~rocsparse_local_color_info()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_color_info(this->info);
    }

    // Sometimes useful to reset local info
    void reset()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_color_info(this->info);
        const rocsparse_status status = rocsparse_create_color_info(&this->info);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    // Allow rocsparse_local_color_info to be used anywhere rocsparse_color_info is expected
    operator rocsparse_color_info&()
    {
        return this->info;
    }
    operator const rocsparse_color_info&() const
    {
        return this->info;
    }
};

/*! \brief  hyb matrix structure helper to access data for tests  */
struct test_hyb
{
    rocsparse_int           m;
    rocsparse_int           n;
    rocsparse_hyb_partition partition;
    rocsparse_int           ell_nnz;
    rocsparse_int           ell_width;
    rocsparse_int*          ell_col_ind;
    void*                   ell_val;
    rocsparse_int           coo_nnz;
    rocsparse_int*          coo_row_ind;
    rocsparse_int*          coo_col_ind;
    void*                   coo_val;
};

/*! \brief  local hyb matrix structure which is automatically created and destroyed  */
class rocsparse_local_hyb_mat
{
    rocsparse_hyb_mat hyb{};

public:
    rocsparse_local_hyb_mat()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_hyb_mat(&this->hyb);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }
    ~rocsparse_local_hyb_mat()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        rocsparse_destroy_hyb_mat(this->hyb);
    }

    // Allow rocsparse_local_hyb_mat to be used anywhere rocsparse_hyb_mat is expected
    operator rocsparse_hyb_mat&()
    {
        return this->hyb;
    }
    operator const rocsparse_hyb_mat&() const
    {
        return this->hyb;
    }
};

/*! \brief  local dense vector structure which is automatically created and destroyed  */
class rocsparse_local_spvec
{
    rocsparse_spvec_descr descr{};

public:
    rocsparse_local_spvec(int64_t              size,
                          int64_t              nnz,
                          void*                indices,
                          void*                values,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_spvec_descr(
            &this->descr, size, nnz, indices, values, idx_type, idx_base, compute_type);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }
    ~rocsparse_local_spvec()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(this->descr != nullptr)
        {
            rocsparse_destroy_spvec_descr(this->descr);
        }
    }

    // Allow rocsparse_local_spvec to be used anywhere rocsparse_spvec_descr is expected
    operator rocsparse_spvec_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_spvec_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local sparse matrix structure which is automatically created and destroyed  */
class rocsparse_local_spmat
{
    rocsparse_spmat_descr descr{};

public:
    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          int64_t              nnz,
                          void*                coo_row_ind,
                          void*                coo_col_ind,
                          void*                coo_val,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_coo_descr(&this->descr,
                                                                   m,
                                                                   n,
                                                                   nnz,
                                                                   coo_row_ind,
                                                                   coo_col_ind,
                                                                   coo_val,
                                                                   idx_type,
                                                                   idx_base,
                                                                   compute_type);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    explicit rocsparse_local_spmat(coo_matrix<MODE, T, I>& h)
        : rocsparse_local_spmat(h.m,
                                h.n,
                                h.nnz,
                                h.row_ind,
                                h.col_ind,
                                h.val,
                                get_indextype<I>(),
                                h.base,
                                get_datatype<T>())
    {
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          int64_t              nnz,
                          void*                coo_ind,
                          void*                coo_val,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_coo_aos_descr(
            &this->descr, m, n, nnz, coo_ind, coo_val, idx_type, idx_base, compute_type);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    explicit rocsparse_local_spmat(coo_aos_matrix<MODE, T, I>& h)
        : rocsparse_local_spmat(
            h.m, h.n, h.nnz, h.ind, h.val, get_indextype<I>(), h.base, get_datatype<T>())
    {
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          int64_t              nnz,
                          void*                row_col_ptr,
                          void*                row_col_ind,
                          void*                val,
                          rocsparse_indextype  row_col_ptr_type,
                          rocsparse_indextype  row_col_ind_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type,
                          bool                 csc_format = false)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(csc_format == false)
        {
            const rocsparse_status status = rocsparse_create_csr_descr(&this->descr,
                                                                       m,
                                                                       n,
                                                                       nnz,
                                                                       row_col_ptr,
                                                                       row_col_ind,
                                                                       val,
                                                                       row_col_ptr_type,
                                                                       row_col_ind_type,
                                                                       idx_base,
                                                                       compute_type);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }
        else
        {
            const rocsparse_status status = rocsparse_create_csc_descr(&this->descr,
                                                                       m,
                                                                       n,
                                                                       nnz,
                                                                       row_col_ptr,
                                                                       row_col_ind,
                                                                       val,
                                                                       row_col_ptr_type,
                                                                       row_col_ind_type,
                                                                       idx_base,
                                                                       compute_type);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION_,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    explicit rocsparse_local_spmat(csx_matrix<MODE, DIRECTION_, T, I, J>& h)
        : rocsparse_local_spmat(h.m,
                                h.n,
                                h.nnz,
                                h.ptr,
                                h.ind,
                                h.val,
                                get_indextype<I>(),
                                get_indextype<J>(),
                                h.base,
                                get_datatype<T>(),
                                (DIRECTION_ == rocsparse_direction_column))
    {
    }

    rocsparse_local_spmat(int64_t              mb,
                          int64_t              nb,
                          int64_t              nnzb,
                          rocsparse_direction  block_dir,
                          int64_t              block_dim,
                          void*                row_col_ptr,
                          void*                row_col_ind,
                          void*                val,
                          rocsparse_indextype  row_col_ptr_type,
                          rocsparse_indextype  row_col_ind_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type,
                          rocsparse_format     format)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(format == rocsparse_format_bsr)
        {
            const rocsparse_status status = rocsparse_create_bsr_descr(&this->descr,
                                                                       mb,
                                                                       nb,
                                                                       nnzb,
                                                                       block_dir,
                                                                       block_dim,
                                                                       row_col_ptr,
                                                                       row_col_ind,
                                                                       val,
                                                                       row_col_ptr_type,
                                                                       row_col_ind_type,
                                                                       idx_base,
                                                                       compute_type);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION_,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    explicit rocsparse_local_spmat(gebsx_matrix<MODE, DIRECTION_, T, I, J>& h)
        : rocsparse_local_spmat(h.mb,
                                h.nb,
                                h.nnzb,
                                h.block_direction,
                                (h.row_block_dim == h.col_block_dim) ? h.row_block_dim : -1,
                                h.ptr,
                                h.ind,
                                h.val,
                                get_indextype<I>(),
                                get_indextype<J>(),
                                h.base,
                                get_datatype<T>(),
                                rocsparse_format_bsr)
    {
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          rocsparse_direction  block_dir,
                          int64_t              block_size,
                          int64_t              ell_cols,
                          void*                ell_col_ind,
                          void*                ell_val,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_bell_descr(&this->descr,
                                                                    m,
                                                                    n,
                                                                    block_dir,
                                                                    block_size,
                                                                    ell_cols,
                                                                    ell_col_ind,
                                                                    ell_val,
                                                                    idx_type,
                                                                    idx_base,
                                                                    compute_type);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          void*                ell_col_ind,
                          void*                ell_val,
                          int64_t              ell_width,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_ell_descr(
            &this->descr, m, n, ell_col_ind, ell_val, ell_width, idx_type, idx_base, compute_type);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    explicit rocsparse_local_spmat(ell_matrix<MODE, T, I>& h)
        : rocsparse_local_spmat(
            h.m, h.n, h.ind, h.val, h.width, get_indextype<I>(), h.base, get_datatype<T>())
    {
    }
    rocsparse_local_spmat() {}
    ~rocsparse_local_spmat()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(this->descr != nullptr)
            rocsparse_destroy_spmat_descr(this->descr);
    }

    // Allow rocsparse_local_spmat to be used anywhere rocsparse_spmat_descr is expected
    operator rocsparse_spmat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_spmat_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local dense vector structure which is automatically created and destroyed  */
class rocsparse_local_dnvec
{
    rocsparse_dnvec_descr descr{};

public:
    rocsparse_local_dnvec(){};
    rocsparse_local_dnvec(int64_t size, void* values, rocsparse_datatype compute_type)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status
            = rocsparse_create_dnvec_descr(&this->descr, size, values, compute_type);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T>
    explicit rocsparse_local_dnvec(dense_matrix<MODE, T>& h)
        : rocsparse_local_dnvec(h.m, (T*)h, get_datatype<T>())
    {
    }

    template <memory_mode::value_t MODE, typename T>
    explicit rocsparse_local_dnvec(dense_vector<MODE, T>& h)
        : rocsparse_local_dnvec(h.size(), h.data(), get_datatype<T>())
    {
    }

    ~rocsparse_local_dnvec()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(this->descr != nullptr)
            rocsparse_destroy_dnvec_descr(this->descr);
    }

    // Allow rocsparse_local_dnvec to be used anywhere rocsparse_dnvec_descr is expected
    operator rocsparse_dnvec_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_dnvec_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local dense matrix structure which is automatically created and destroyed  */
class rocsparse_local_dnmat
{
    rocsparse_dnmat_descr descr{};

public:
    rocsparse_local_dnmat(int64_t            rows,
                          int64_t            cols,
                          int64_t            ld,
                          void*              values,
                          rocsparse_datatype compute_type,
                          rocsparse_order    order)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse_create_dnmat_descr(
            &this->descr, rows, cols, ld, values, compute_type, order);
        if(status != rocsparse_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T>
    explicit rocsparse_local_dnmat(dense_matrix<MODE, T>& h)
        : rocsparse_local_dnmat(h.m, h.n, h.ld, (T*)h, get_datatype<T>(), h.order)
    {
    }

    ~rocsparse_local_dnmat()
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(this->descr != nullptr)
            rocsparse_destroy_dnmat_descr(this->descr);
    }

    // Allow rocsparse_local_dnmat to be used anywhere rocsparse_dnmat_descr is expected
    operator rocsparse_dnmat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_dnmat_descr&() const
    {
        return this->descr;
    }
};

/*  timing: HIP only provides very limited timers function clock() and not general;
            rocsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return
 *  wall time
 */
double get_time_us(void);

/*! \brief Return path of this executable */
std::string rocsparse_exepath();

/*! \brief Return path of rocsparse_gentest.py file */
std::string rocsparse_gentestpath();

/*! \brief Return path where the test data file (rocsparse_test.data) is located */
std::string rocsparse_datapath();

namespace rocsparse_clients
{

    struct timer
    {
    private:
        hipStream_t m_stream;
        hipEvent_t  m_start, m_stop;

    public:
        timer(hipStream_t stream);
        void  start();
        float stop();
        ~timer();
    };

    /*! \brief  repeated performance runs of a function with a set of arguments that computes
   * the median of the mean of the wall-clock time.
   */
    template <typename T, typename... ARG>
    double run_benchmark(const Arguments&        arguments,
                         T                       func,
                         rocsparse_local_handle& handle,
                         ARG&&... arg)
    {
        if(arguments.iters_inner == 0)
        {
            std::cerr << "error " << __FUNCTION__ << ": arguments.iters_inner is zero, exit program"
                      << std::endl;
            exit(rocsparse_status_invalid_value);
        }

        if(arguments.iters == 0)
        {
            std::cerr << "error " << __FUNCTION__ << ": arguments.iters is zero, exit program."
                      << std::endl;
            exit(rocsparse_status_invalid_value);
        }

        const int32_t n_cold_calls = 2;
        const int32_t n_sub_calls  = arguments.iters_inner;
        const int32_t n_calls      = arguments.iters;

        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);

        for(int32_t iter = 0; iter < n_cold_calls; ++iter)
        {
            const rocsparse_status status = func(handle, std::forward<ARG>(arg)...);
            if(status != rocsparse_status_success)
            {
                std::cerr << "error " << __FUNCTION__ << ": cold call failed." << std::endl;
                exit(rocsparse_status_invalid_value);
            }
        }

        std::vector<double> gpu_time(n_calls);

        rocsparse_clients::timer t(stream);
        for(int32_t iter = 0; iter < n_calls; ++iter)
        {
            t.start();
            for(int32_t sub_iter = 0; sub_iter < n_sub_calls; ++sub_iter)
            {
                std::ignore = func(handle, std::forward<ARG>(arg)...);
            }
            const double t_microseconds = (t.stop() * 1000);
            gpu_time[iter]              = t_microseconds / n_sub_calls;
        }

        std::sort(gpu_time.begin(), gpu_time.end());
        const int32_t mid = n_calls / 2;
        return (n_calls % 2 == 0) ? (gpu_time[mid] + gpu_time[mid - 1]) / 2 : gpu_time[mid];
    }

}

#endif // UTILITY_HPP
