/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_matrix.hpp"

struct rocsparse_matrix_statistics
{

private:
    template <typename I, typename J>
    static void compute_nnz_per_seq_from_offsets(const size_t size,
                                                 const I* __restrict__ ptr,
                                                 J* __restrict__ nnz_per_seq)
    {
        for(size_t i = 0; i < size; ++i)
        {
            nnz_per_seq[i] = ptr[i + 1] - ptr[i];
        }
    }

    template <typename J>
    static void compute_nnz_per_seq_from_indices(const size_t size,
                                                 const J* __restrict__ ind,
                                                 uint32_t             ind_inc,
                                                 rocsparse_index_base ind_base,
                                                 J* __restrict__ nnz_per_seq)
    {
        for(size_t i = 0; i < size; ++i)
        {
            ++nnz_per_seq[ind[i * ind_inc] - ind_base];
        }
    }

    template <typename T>
    static void
        compute_statistics(size_t size, T* values, int64_t& min, int64_t& median, int64_t& max)
    {
        std::sort(values, values + size);
        min    = values[0];
        max    = values[size - 1];
        median = values[size / 2];
        if(size % 2 == 0)
        {
            median = (median + values[size / 2 - 1]) / 2;
        }
    }

    template <typename I, typename J>
    static void host_raw_compute_nnz_statistics(J M,
                                                J N,
                                                I nnz,
                                                const I* __restrict__ ptr,
                                                const J* __restrict__ ind,
                                                uint32_t             ind_inc,
                                                rocsparse_index_base ind_base,
                                                rocsparse_direction  direction,
                                                int64_t&             min_nnz,
                                                int64_t&             median_nnz,
                                                int64_t&             max_nnz)
    {
        const J num_seq = (direction == rocsparse_direction_row) ? M : N;
        J* nnz_per_seq  = (direction == rocsparse_direction_row) ? (J*)malloc(num_seq * sizeof(J))
                                                                 : (J*)calloc(num_seq, sizeof(J));

        if(ptr && !ind)
        {
            compute_nnz_per_seq_from_offsets(num_seq, ptr, nnz_per_seq);
        }
        else if(!ptr && ind)
        {
            compute_nnz_per_seq_from_indices(nnz, ind, ind_inc, ind_base, nnz_per_seq);
        }
        else
        {
            throw(rocsparse_status_internal_error);
        }
        compute_statistics(num_seq, nnz_per_seq, min_nnz, median_nnz, max_nnz);
        free(nnz_per_seq);
    }

    template <typename I>
    static void compute_nnz_per_seq_ell(rocsparse_direction direction,
                                        I                   M,
                                        I                   width,
                                        const I* __restrict__ ind,
                                        rocsparse_index_base ind_base,
                                        I* __restrict__ nnz_per_seq)
    {
        if(direction == rocsparse_direction_row)
        {
            for(I i = 0; i < M; ++i)
            {
                for(I w = 0; w < width; ++w)
                {
                    const I j = ind[i * width + w] - ind_base;
                    if(j >= 0)
                    {
                        ++nnz_per_seq[i];
                    }
                }
            }
        }
        else
        {
            const I nnz = M * width;
            for(I k = 0; k < nnz; ++k)
            {
                const I j = ind[k] - ind_base;
                if(j >= 0)
                {
                    ++nnz_per_seq[j];
                }
            }
        }
    }

    template <typename I>
    static void host_raw_compute_nnz_statistics_ell(I M,
                                                    I N,
                                                    I width,
                                                    const I* __restrict__ ind,
                                                    rocsparse_index_base ind_base,
                                                    rocsparse_direction  direction,
                                                    int64_t&             min_nnz,
                                                    int64_t&             median_nnz,
                                                    int64_t&             max_nnz)
    {
        const I num_seq     = (direction == rocsparse_direction_row) ? M : N;
        I*      nnz_per_seq = (I*)calloc(num_seq, sizeof(I));

        compute_nnz_per_seq_ell(direction, M, width, ind, ind_base, nnz_per_seq);

        compute_statistics(num_seq, nnz_per_seq, min_nnz, median_nnz, max_nnz);

        free(nnz_per_seq);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const host_ell_matrix<T, I>& that,
                                       rocsparse_direction          direction,
                                       int64_t&                     min_nnz,
                                       int64_t&                     median_nnz,
                                       int64_t&                     max_nnz)
    {
        host_raw_compute_nnz_statistics_ell<I>(that.m,
                                               that.n,
                                               that.width,
                                               that.ind.data(),
                                               that.base,
                                               direction,
                                               min_nnz,
                                               median_nnz,
                                               max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const managed_ell_matrix<T, I>& that,
                                       rocsparse_direction             direction,
                                       int64_t&                        min_nnz,
                                       int64_t&                        median_nnz,
                                       int64_t&                        max_nnz)
    {
        host_raw_compute_nnz_statistics_ell<I>(that.m,
                                               that.n,
                                               that.width,
                                               that.ind.data(),
                                               that.base,
                                               direction,
                                               min_nnz,
                                               median_nnz,
                                               max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const device_ell_matrix<T, I>& that,
                                       rocsparse_direction            direction,
                                       int64_t&                       min_nnz,
                                       int64_t&                       median_nnz,
                                       int64_t&                       max_nnz)
    {
        host_dense_vector<I> host_ind(that.ind);
        host_raw_compute_nnz_statistics_ell<I>(that.m,
                                               that.n,
                                               that.width,
                                               host_ind.data(),
                                               that.base,
                                               direction,
                                               min_nnz,
                                               median_nnz,
                                               max_nnz);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    static void compute_nnz_statistics(const host_csx_matrix<DIRECTION, T, I, J>& that,
                                       rocsparse_direction                        direction,
                                       int64_t&                                   min_nnz,
                                       int64_t&                                   median_nnz,
                                       int64_t&                                   max_nnz)
    {
        const I*                   ptr      = (DIRECTION == direction) ? that.ptr.data() : nullptr;
        const J*                   ind      = (DIRECTION == direction) ? nullptr : that.ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, J>(that.m,
                                              that.n,
                                              that.nnz,
                                              ptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    static void compute_nnz_statistics(const managed_csx_matrix<DIRECTION, T, I, J>& that,
                                       rocsparse_direction                           direction,
                                       int64_t&                                      min_nnz,
                                       int64_t&                                      median_nnz,
                                       int64_t&                                      max_nnz)
    {
        const I*                   ptr      = (DIRECTION == direction) ? that.ptr.data() : nullptr;
        const J*                   ind      = (DIRECTION == direction) ? nullptr : that.ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, J>(that.m,
                                              that.n,
                                              that.nnz,
                                              ptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    static void compute_nnz_statistics(const host_gebsx_matrix<DIRECTION, T, I, J>& that,
                                       rocsparse_direction                          direction,
                                       int64_t&                                     min_nnz,
                                       int64_t&                                     median_nnz,
                                       int64_t&                                     max_nnz)
    {
        const I*                   ptr      = (DIRECTION == direction) ? that.ptr.data() : nullptr;
        const J*                   ind      = (DIRECTION == direction) ? nullptr : that.ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, J>(that.mb,
                                              that.nb,
                                              that.nnzb,
                                              ptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    static void compute_nnz_statistics(const managed_gebsx_matrix<DIRECTION, T, I, J>& that,
                                       rocsparse_direction                             direction,
                                       int64_t&                                        min_nnz,
                                       int64_t&                                        median_nnz,
                                       int64_t&                                        max_nnz)
    {
        const I*                   ptr      = (DIRECTION == direction) ? that.ptr.data() : nullptr;
        const J*                   ind      = (DIRECTION == direction) ? nullptr : that.ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, J>(that.mb,
                                              that.nb,
                                              that.nnzb,
                                              ptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const host_coo_aos_matrix<T, I>& that,
                                       rocsparse_direction              direction,
                                       int64_t&                         min_nnz,
                                       int64_t&                         median_nnz,
                                       int64_t&                         max_nnz)
    {
        const I* ind
            = (direction == rocsparse_direction_row) ? that.ind.data() : that.ind.data() + 1;
        static constexpr uint32_t  ind_inc  = 2;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, I>(that.m,
                                              that.n,
                                              that.nnz,
                                              nullptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const managed_coo_aos_matrix<T, I>& that,
                                       rocsparse_direction                 direction,
                                       int64_t&                            min_nnz,
                                       int64_t&                            median_nnz,
                                       int64_t&                            max_nnz)
    {
        const I* ind
            = (direction == rocsparse_direction_row) ? that.ind.data() : that.ind.data() + 1;
        static constexpr uint32_t  ind_inc  = 2;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, I>(that.m,
                                              that.n,
                                              that.nnz,
                                              nullptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const device_coo_aos_matrix<T, I>& that,
                                       rocsparse_direction                direction,
                                       int64_t&                           min_nnz,
                                       int64_t&                           median_nnz,
                                       int64_t&                           max_nnz)
    {
        host_dense_vector<I> host_ind(that.ind);
        const I*             ind
            = (direction == rocsparse_direction_row) ? host_ind.data() : host_ind.data() + 1;
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, I>(that.m,
                                              that.n,
                                              that.nnz,
                                              nullptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const host_coo_matrix<T, I>& that,
                                       rocsparse_direction          direction,
                                       int64_t&                     min_nnz,
                                       int64_t&                     median_nnz,
                                       int64_t&                     max_nnz)
    {
        const I* ind
            = (direction == rocsparse_direction_row) ? that.row_ind.data() : that.col_ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, I>(that.m,
                                              that.n,
                                              that.nnz,
                                              nullptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const managed_coo_matrix<T, I>& that,
                                       rocsparse_direction             direction,
                                       int64_t&                        min_nnz,
                                       int64_t&                        median_nnz,
                                       int64_t&                        max_nnz)
    {
        const I* ind
            = (direction == rocsparse_direction_row) ? that.row_ind.data() : that.col_ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, I>(that.m,
                                              that.n,
                                              that.nnz,
                                              nullptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <typename T, typename I>
    static void compute_nnz_statistics(const device_coo_matrix<T, I>& that,
                                       rocsparse_direction            direction,
                                       int64_t&                       min_nnz,
                                       int64_t&                       median_nnz,
                                       int64_t&                       max_nnz)
    {
        host_dense_vector<I>       host_ind((direction == rocsparse_direction_row) ? that.row_ind
                                                                                   : that.col_ind);
        const I*                   ind      = host_ind.data();
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        host_raw_compute_nnz_statistics<I, I>(that.m,
                                              that.n,
                                              that.nnz,
                                              nullptr,
                                              ind,
                                              ind_inc,
                                              ind_base,
                                              direction,
                                              min_nnz,
                                              median_nnz,
                                              max_nnz);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    static void compute_nnz_statistics(const device_csx_matrix<DIRECTION, T, I, J>& that,
                                       rocsparse_direction                          direction,
                                       int64_t&                                     min_nnz,
                                       int64_t&                                     median_nnz,
                                       int64_t&                                     max_nnz)
    {
        const I*                   ptr      = nullptr;
        const J*                   ind      = nullptr;
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        if(DIRECTION == direction)
        {
            host_dense_vector<I> host_ptr(that.ptr);
            ptr = host_ptr.data();
            host_raw_compute_nnz_statistics<I, J>(that.m,
                                                  that.n,
                                                  that.nnz,
                                                  ptr,
                                                  ind,
                                                  ind_inc,
                                                  ind_base,
                                                  direction,
                                                  min_nnz,
                                                  median_nnz,
                                                  max_nnz);
        }
        else
        {
            host_dense_vector<J> host_ind(that.ind);
            ind = host_ind.data();

            host_raw_compute_nnz_statistics<I, J>(that.m,
                                                  that.n,
                                                  that.nnz,
                                                  ptr,
                                                  ind,
                                                  ind_inc,
                                                  ind_base,
                                                  direction,
                                                  min_nnz,
                                                  median_nnz,
                                                  max_nnz);
        }
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    static void compute_nnz_statistics(const device_gebsx_matrix<DIRECTION, T, I, J>& that,
                                       rocsparse_direction                            direction,
                                       int64_t&                                       min_nnz,
                                       int64_t&                                       median_nnz,
                                       int64_t&                                       max_nnz)
    {
        const I*                   ptr      = nullptr;
        const J*                   ind      = nullptr;
        static constexpr uint32_t  ind_inc  = 1;
        const rocsparse_index_base ind_base = that.base;
        if(DIRECTION == direction)
        {
            host_dense_vector<I> host_ptr(that.ptr);
            ptr = host_ptr.data();
            host_raw_compute_nnz_statistics<I, J>(that.mb,
                                                  that.nb,
                                                  that.nnzb,
                                                  ptr,
                                                  ind,
                                                  ind_inc,
                                                  ind_base,
                                                  direction,
                                                  min_nnz,
                                                  median_nnz,
                                                  max_nnz);
        }
        else
        {
            host_dense_vector<J> host_ind(that.ind);
            ind = host_ind.data();
            host_raw_compute_nnz_statistics<I, J>(that.mb,
                                                  that.nb,
                                                  that.nnzb,
                                                  ptr,
                                                  ind,
                                                  ind_inc,
                                                  ind_base,
                                                  direction,
                                                  min_nnz,
                                                  median_nnz,
                                                  max_nnz);
        }
    }

public:
    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I,
              typename J>
    static void get_nnz_per_row(const csx_matrix<MODE, DIRECTION, T, I, J>& that,
                                int64_t&                                    min_nnz,
                                int64_t&                                    median_nnz,
                                int64_t&                                    max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_row, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I,
              typename J>
    static void get_nnz_per_column(const csx_matrix<MODE, DIRECTION, T, I, J>& that,
                                   int64_t&                                    min_nnz,
                                   int64_t&                                    median_nnz,
                                   int64_t&                                    max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_column, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I,
              typename J>
    static void get_nnz_per_row(const gebsx_matrix<MODE, DIRECTION, T, I, J>& that,
                                int64_t&                                      min_nnz,
                                int64_t&                                      median_nnz,
                                int64_t&                                      max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_row, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I,
              typename J>
    static void get_nnz_per_column(const gebsx_matrix<MODE, DIRECTION, T, I, J>& that,
                                   int64_t&                                      min_nnz,
                                   int64_t&                                      median_nnz,
                                   int64_t&                                      max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_column, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void get_nnz_per_column(const coo_matrix<MODE, T, I>& that,
                                   int64_t&                      min_nnz,
                                   int64_t&                      median_nnz,
                                   int64_t&                      max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_column, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void get_nnz_per_row(const coo_matrix<MODE, T, I>& that,
                                int64_t&                      min_nnz,
                                int64_t&                      median_nnz,
                                int64_t&                      max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_row, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void get_nnz_per_column(const coo_aos_matrix<MODE, T, I>& that,
                                   int64_t&                          min_nnz,
                                   int64_t&                          median_nnz,
                                   int64_t&                          max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_column, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void get_nnz_per_row(const coo_aos_matrix<MODE, T, I>& that,
                                int64_t&                          min_nnz,
                                int64_t&                          median_nnz,
                                int64_t&                          max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_row, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void get_nnz_per_column(const ell_matrix<MODE, T, I>& that,
                                   int64_t&                      min_nnz,
                                   int64_t&                      median_nnz,
                                   int64_t&                      max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_column, min_nnz, median_nnz, max_nnz);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void get_nnz_per_row(const ell_matrix<MODE, T, I>& that,
                                int64_t&                      min_nnz,
                                int64_t&                      median_nnz,
                                int64_t&                      max_nnz)
    {
        compute_nnz_statistics(that, rocsparse_direction_row, min_nnz, median_nnz, max_nnz);
    }
};
