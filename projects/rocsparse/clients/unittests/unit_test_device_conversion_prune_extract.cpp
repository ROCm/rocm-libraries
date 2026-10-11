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

// Device unit tests that drive the public conversion API on small matrices so
// host dispatch in the prune / extract / sparse_to_sparse / csr2gebsr
// translation units takes format, index-type, datatype, and algorithm branches
// the integration suite leaves partial. Wavefront-64 launches are not
// attempted: gfx1201 is wavefront 32.

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&d), rocsparse_status_success);
        }
        ~MatDescr()
        {
            if(d)
            {
                EXPECT_EQ(rocsparse_destroy_mat_descr(d), rocsparse_status_success);
            }
        }
        MatDescr(const MatDescr&)            = delete;
        MatDescr& operator=(const MatDescr&) = delete;
    };

    struct MatInfo
    {
        rocsparse_mat_info i = nullptr;
        MatInfo()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&i), rocsparse_status_success);
        }
        ~MatInfo()
        {
            if(i)
            {
                EXPECT_EQ(rocsparse_destroy_mat_info(i), rocsparse_status_success);
            }
        }
        MatInfo(const MatInfo&)            = delete;
        MatInfo& operator=(const MatInfo&) = delete;
    };

    struct SpMat
    {
        rocsparse_spmat_descr d = nullptr;
        ~SpMat()
        {
            if(d)
            {
                EXPECT_EQ(rocsparse_destroy_spmat_descr(d), rocsparse_status_success);
            }
        }
        SpMat()                              = default;
        SpMat(const SpMat&)                  = delete;
        SpMat& operator=(const SpMat&)       = delete;
        operator rocsparse_spmat_descr() const
        {
            return d;
        }
    };

    // Shrink the handle scratch so the "allocate a private temp buffer" side of
    // `handle->buffer_size >= need` runs, then put the original size back.
    struct ForceHandleAlloc
    {
        rocsparse_handle handle;
        size_t           saved;
        explicit ForceHandleAlloc(rocsparse_handle h)
            : handle(h)
            , saved(h->buffer_size)
        {
            h->buffer_size = 0;
        }
        ~ForceHandleAlloc()
        {
            handle->buffer_size = saved;
        }
    };

    struct CsrPattern
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<float>         val;
        rocsparse_int              m   = 0;
        rocsparse_int              n   = 0;
        rocsparse_int              nnz = 0;
    };

    // nnz_per_row is also the integer mean nnz, which selects the wavefront-32
    // compress segment (<4, <8, <16, <32, otherwise).
    CsrPattern make_pattern(rocsparse_int m, rocsparse_int nnz_per_row, rocsparse_index_base base)
    {
        CsrPattern p;
        p.m         = m;
        p.n         = std::max(nnz_per_row, rocsparse_int{1});
        p.nnz       = m * nnz_per_row;
        const int b = (base == rocsparse_index_base_one) ? 1 : 0;
        p.row_ptr.resize(static_cast<size_t>(m) + 1);
        p.col_ind.resize(static_cast<size_t>(p.nnz));
        p.val.resize(static_cast<size_t>(p.nnz));
        p.row_ptr[0] = b;
        rocsparse_int acc = 0;
        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(rocsparse_int k = 0; k < nnz_per_row; ++k)
            {
                p.col_ind[static_cast<size_t>(acc)] = k + b;
                // Alternate below / above a 0.5 threshold so both keep and drop run.
                p.val[static_cast<size_t>(acc)] = (k % 2 == 0) ? 0.25f : 1.5f;
                ++acc;
            }
            p.row_ptr[static_cast<size_t>(i) + 1] = acc + b;
        }
        return p;
    }

    template <typename T>
    T one_of()
    {
        if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        {
            return rocsparse_float_complex(1.0f, 0.0f);
        }
        else if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_double_complex(1.0, 0.0);
        }
        else
        {
            return static_cast<T>(1);
        }
    }

    template <typename T>
    std::vector<T> cast_val(const std::vector<float>& src)
    {
        std::vector<T> out(src.size());
        for(size_t i = 0; i < src.size(); ++i)
        {
            out[i] = static_cast<T>(src[i]);
        }
        return out;
    }

    template <typename T>
    rocsparse_status prune_buffer_size(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             nnz,
                                       const rocsparse_mat_descr descr_A,
                                       const T*                  val,
                                       const rocsparse_int*      row_ptr,
                                       const rocsparse_int*      col_ind,
                                       const T*                  threshold,
                                       const rocsparse_mat_descr descr_C,
                                       const T*                  val_C,
                                       const rocsparse_int*      row_ptr_C,
                                       const rocsparse_int*      col_ind_C,
                                       size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_sprune_csr2csr_buffer_size(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        descr_A,
                                                        val,
                                                        row_ptr,
                                                        col_ind,
                                                        threshold,
                                                        descr_C,
                                                        val_C,
                                                        row_ptr_C,
                                                        col_ind_C,
                                                        buffer_size);
        }
        else
        {
            return rocsparse_dprune_csr2csr_buffer_size(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        descr_A,
                                                        val,
                                                        row_ptr,
                                                        col_ind,
                                                        threshold,
                                                        descr_C,
                                                        val_C,
                                                        row_ptr_C,
                                                        col_ind_C,
                                                        buffer_size);
        }
    }

    template <typename T>
    rocsparse_status prune_nnz(rocsparse_handle          handle,
                               rocsparse_int             m,
                               rocsparse_int             n,
                               rocsparse_int             nnz,
                               const rocsparse_mat_descr descr_A,
                               const T*                  val,
                               const rocsparse_int*      row_ptr,
                               const rocsparse_int*      col_ind,
                               const T*                  threshold,
                               const rocsparse_mat_descr descr_C,
                               rocsparse_int*            row_ptr_C,
                               rocsparse_int*            nnz_total,
                               void*                     buffer)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_sprune_csr2csr_nnz(handle,
                                                m,
                                                n,
                                                nnz,
                                                descr_A,
                                                val,
                                                row_ptr,
                                                col_ind,
                                                threshold,
                                                descr_C,
                                                row_ptr_C,
                                                nnz_total,
                                                buffer);
        }
        else
        {
            return rocsparse_dprune_csr2csr_nnz(handle,
                                                m,
                                                n,
                                                nnz,
                                                descr_A,
                                                val,
                                                row_ptr,
                                                col_ind,
                                                threshold,
                                                descr_C,
                                                row_ptr_C,
                                                nnz_total,
                                                buffer);
        }
    }

    template <typename T>
    rocsparse_status prune_compute(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr_A,
                                   const T*                  val,
                                   const rocsparse_int*      row_ptr,
                                   const rocsparse_int*      col_ind,
                                   const T*                  threshold,
                                   const rocsparse_mat_descr descr_C,
                                   T*                        val_C,
                                   const rocsparse_int*      row_ptr_C,
                                   rocsparse_int*            col_ind_C,
                                   void*                     buffer)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_sprune_csr2csr(handle,
                                            m,
                                            n,
                                            nnz,
                                            descr_A,
                                            val,
                                            row_ptr,
                                            col_ind,
                                            threshold,
                                            descr_C,
                                            val_C,
                                            row_ptr_C,
                                            col_ind_C,
                                            buffer);
        }
        else
        {
            return rocsparse_dprune_csr2csr(handle,
                                            m,
                                            n,
                                            nnz,
                                            descr_A,
                                            val,
                                            row_ptr,
                                            col_ind,
                                            threshold,
                                            descr_C,
                                            val_C,
                                            row_ptr_C,
                                            col_ind_C,
                                            buffer);
        }
    }

    template <typename T>
    void run_prune_pattern(rocsparse_handle     handle,
                           const CsrPattern&    pattern,
                           rocsparse_index_base base_A,
                           rocsparse_index_base base_C,
                           bool                 device_threshold)
    {
        MatDescr descr_A;
        MatDescr descr_C;
        ASSERT_EQ(rocsparse_set_mat_index_base(descr_A.d, base_A), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, base_C), rocsparse_status_success);

        device_vector<rocsparse_int> row_ptr{pattern.row_ptr};
        device_vector<rocsparse_int> col_ind{pattern.col_ind};
        device_vector<T>             val{cast_val<T>(pattern.val)};
        ASSERT_TRUE(row_ptr.ptr && (pattern.nnz == 0 || col_ind.ptr) && (pattern.nnz == 0 || val.ptr));

        device_vector<rocsparse_int> row_ptr_C{static_cast<size_t>(pattern.m) + 1};
        const size_t                 out_n = static_cast<size_t>(std::max(pattern.nnz, rocsparse_int{1}));
        device_vector<T>             val_C{out_n};
        device_vector<rocsparse_int> col_ind_C{out_n};
        ASSERT_TRUE(row_ptr_C.ptr && val_C.ptr && col_ind_C.ptr);

        const T              host_threshold = static_cast<T>(0.5f);
        device_vector<T>     dev_threshold{std::vector<T>{host_threshold}};
        const T*             threshold = device_threshold ? dev_threshold.ptr : &host_threshold;
        rocsparse_pointer_mode saved   = rocsparse_pointer_mode_host;
        ASSERT_EQ(rocsparse_get_pointer_mode(handle, &saved), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle,
                                            device_threshold ? rocsparse_pointer_mode_device
                                                             : rocsparse_pointer_mode_host),
                  rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(prune_buffer_size<T>(handle,
                                       pattern.m,
                                       pattern.n,
                                       pattern.nnz,
                                       descr_A.d,
                                       val,
                                       row_ptr,
                                       col_ind,
                                       threshold,
                                       descr_C.d,
                                       val_C,
                                       row_ptr_C,
                                       col_ind_C,
                                       &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        // Device pointer mode treats the nnz total as a device pointer.
        device_vector<rocsparse_int> d_nnz_total{std::vector<rocsparse_int>{0}};
        rocsparse_int                 nnz_C   = -1;
        rocsparse_int*                nnz_ptr = device_threshold ? d_nnz_total.ptr : &nnz_C;
        ASSERT_EQ(prune_nnz<T>(handle,
                               pattern.m,
                               pattern.n,
                               pattern.nnz,
                               descr_A.d,
                               val,
                               row_ptr,
                               col_ind,
                               threshold,
                               descr_C.d,
                               row_ptr_C,
                               nnz_ptr,
                               buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(device_threshold)
        {
            nnz_C = to_host(d_nnz_total.ptr, 1)[0];
        }
        EXPECT_GE(nnz_C, 0);
        EXPECT_LE(nnz_C, pattern.nnz);

        ASSERT_EQ(prune_compute<T>(handle,
                                   pattern.m,
                                   pattern.n,
                                   pattern.nnz,
                                   descr_A.d,
                                   val,
                                   row_ptr,
                                   col_ind,
                                   threshold,
                                   descr_C.d,
                                   val_C,
                                   row_ptr_C,
                                   col_ind_C,
                                   buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, saved), rocsparse_status_success);
    }

    template <typename T>
    rocsparse_status pct_buffer_size(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr_A,
                                     const T*                  val,
                                     const rocsparse_int*      row_ptr,
                                     const rocsparse_int*      col_ind,
                                     T                         percentage,
                                     const rocsparse_mat_descr descr_C,
                                     const rocsparse_int*      row_ptr_C,
                                     rocsparse_mat_info        info,
                                     size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle,
                                                                      m,
                                                                      n,
                                                                      nnz,
                                                                      descr_A,
                                                                      val,
                                                                      row_ptr,
                                                                      col_ind,
                                                                      percentage,
                                                                      descr_C,
                                                                      nullptr,
                                                                      row_ptr_C,
                                                                      nullptr,
                                                                      info,
                                                                      buffer_size);
        }
        else
        {
            return rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                                      m,
                                                                      n,
                                                                      nnz,
                                                                      descr_A,
                                                                      val,
                                                                      row_ptr,
                                                                      col_ind,
                                                                      percentage,
                                                                      descr_C,
                                                                      nullptr,
                                                                      row_ptr_C,
                                                                      nullptr,
                                                                      info,
                                                                      buffer_size);
        }
    }

    template <typename T>
    rocsparse_status pct_nnz(rocsparse_handle          handle,
                             rocsparse_int             m,
                             rocsparse_int             n,
                             rocsparse_int             nnz,
                             const rocsparse_mat_descr descr_A,
                             const T*                  val,
                             const rocsparse_int*      row_ptr,
                             const rocsparse_int*      col_ind,
                             T                         percentage,
                             const rocsparse_mat_descr descr_C,
                             rocsparse_int*            row_ptr_C,
                             rocsparse_int*            nnz_total,
                             rocsparse_mat_info        info,
                             void*                     buffer)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz,
                                                             descr_A,
                                                             val,
                                                             row_ptr,
                                                             col_ind,
                                                             percentage,
                                                             descr_C,
                                                             row_ptr_C,
                                                             nnz_total,
                                                             info,
                                                             buffer);
        }
        else
        {
            return rocsparse_dprune_csr2csr_nnz_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz,
                                                             descr_A,
                                                             val,
                                                             row_ptr,
                                                             col_ind,
                                                             percentage,
                                                             descr_C,
                                                             row_ptr_C,
                                                             nnz_total,
                                                             info,
                                                             buffer);
        }
    }

    template <typename T>
    rocsparse_status pct_compute(rocsparse_handle          handle,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const rocsparse_mat_descr descr_A,
                                 const T*                  val,
                                 const rocsparse_int*      row_ptr,
                                 const rocsparse_int*      col_ind,
                                 T                         percentage,
                                 const rocsparse_mat_descr descr_C,
                                 T*                        val_C,
                                 const rocsparse_int*      row_ptr_C,
                                 rocsparse_int*            col_ind_C,
                                 rocsparse_mat_info        info,
                                 void*                     buffer)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_sprune_csr2csr_by_percentage(handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         descr_A,
                                                         val,
                                                         row_ptr,
                                                         col_ind,
                                                         percentage,
                                                         descr_C,
                                                         val_C,
                                                         row_ptr_C,
                                                         col_ind_C,
                                                         info,
                                                         buffer);
        }
        else
        {
            return rocsparse_dprune_csr2csr_by_percentage(handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         descr_A,
                                                         val,
                                                         row_ptr,
                                                         col_ind,
                                                         percentage,
                                                         descr_C,
                                                         val_C,
                                                         row_ptr_C,
                                                         col_ind_C,
                                                         info,
                                                         buffer);
        }
    }

    template <typename T>
    void run_pct_pattern(rocsparse_handle     handle,
                         const CsrPattern&    pattern,
                         T                    percentage,
                         rocsparse_index_base base,
                         bool                 device_mode,
                         bool                 force_alloc)
    {
        MatDescr descr_A;
        MatDescr descr_C;
        MatInfo  info;
        ASSERT_EQ(rocsparse_set_mat_index_base(descr_A.d, base), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, base), rocsparse_status_success);

        device_vector<rocsparse_int> row_ptr{pattern.row_ptr};
        device_vector<rocsparse_int> col_ind{pattern.col_ind.empty()
                                                 ? std::vector<rocsparse_int>{0}
                                                 : pattern.col_ind};
        device_vector<T>             val{cast_val<T>(pattern.val.empty() ? std::vector<float>{0.f}
                                                                         : pattern.val)};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

        device_vector<rocsparse_int> row_ptr_C{static_cast<size_t>(pattern.m) + 1};
        const size_t                 out_n = static_cast<size_t>(std::max(pattern.nnz, rocsparse_int{1}));
        device_vector<T>             val_C{out_n};
        device_vector<rocsparse_int> col_ind_C{out_n};
        ASSERT_TRUE(row_ptr_C.ptr && val_C.ptr && col_ind_C.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(pct_buffer_size<T>(handle,
                                     pattern.m,
                                     pattern.n,
                                     pattern.nnz,
                                     descr_A.d,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     percentage,
                                     descr_C.d,
                                     row_ptr_C,
                                     info.i,
                                     &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{512}};
        ASSERT_TRUE(buffer.ptr);

        std::unique_ptr<ForceHandleAlloc> forced;
        if(force_alloc)
        {
            forced.reset(new ForceHandleAlloc(handle));
        }

        rocsparse_pointer_mode saved = rocsparse_pointer_mode_host;
        ASSERT_EQ(rocsparse_get_pointer_mode(handle, &saved), rocsparse_status_success);
        // nnz writes the threshold into the temp buffer. Compute then reads it
        // from host or from that device slot, depending on pointer mode.
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);

        rocsparse_int nnz_C = -1;
        ASSERT_EQ(pct_nnz<T>(handle,
                             pattern.m,
                             pattern.n,
                             pattern.nnz,
                             descr_A.d,
                             val,
                             row_ptr,
                             col_ind,
                             percentage,
                             descr_C.d,
                             row_ptr_C,
                             &nnz_C,
                             info.i,
                             buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(device_mode)
        {
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
        }
        ASSERT_EQ(pct_compute<T>(handle,
                                 pattern.m,
                                 pattern.n,
                                 pattern.nnz,
                                 descr_A.d,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 percentage,
                                 descr_C.d,
                                 val_C,
                                 row_ptr_C,
                                 col_ind_C,
                                 info.i,
                                 buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, saved), rocsparse_status_success);
    }

    size_t dtype_bytes(rocsparse_datatype dt)
    {
        switch(dt)
        {
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return 2;
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
            return 1;
        case rocsparse_datatype_f32_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return 4;
        case rocsparse_datatype_f64_r:
        case rocsparse_datatype_f32_c:
            return 8;
        case rocsparse_datatype_f64_c:
            return 16;
        }
        return 4;
    }

    void set_attr(rocsparse_spmat_descr descr, rocsparse_spmat_attribute attr, const void* data, size_t n)
    {
        ASSERT_EQ(rocsparse_spmat_set_attribute(descr, attr, data, n), rocsparse_status_success);
    }

    // create_extract_descr lets the default-alg constructor throw rocsparse_status
    // for unsupported formats. Catch it so the switch still counts and the
    // process stays up.
    rocsparse_status create_extract_catch(rocsparse_extract_descr*    descr,
                                          rocsparse_const_spmat_descr source,
                                          rocsparse_spmat_descr       target)
    {
        try
        {
            return rocsparse_create_extract_descr(
                descr, source, target, rocsparse_extract_alg_default);
        }
        catch(const rocsparse_status& st)
        {
            return st;
        }
    }

    void run_extract(rocsparse_handle        handle,
                     rocsparse_spmat_descr   source,
                     rocsparse_spmat_descr   target,
                     rocsparse_fill_mode     fill,
                     rocsparse_diag_type     diag,
                     int64_t                 value_bytes,
                     rocsparse_indextype     col_or_row_out_type,
                     bool                    csc)
    {
        set_attr(target, rocsparse_spmat_fill_mode, &fill, sizeof(fill));
        set_attr(target, rocsparse_spmat_diag_type, &diag, sizeof(diag));

        rocsparse_extract_descr descr = nullptr;
        ASSERT_EQ(create_extract_catch(&descr, source, target), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_extract_buffer_size(
                      handle, descr, source, target, rocsparse_extract_stage_analysis, &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_analysis,
                                    buffer_size,
                                    buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        int64_t target_nnz = 0;
        ASSERT_EQ(rocsparse_extract_nnz(handle, descr, &target_nnz), rocsparse_status_success);
        EXPECT_GE(target_nnz, 0);

        const size_t idx_bytes = (col_or_row_out_type == rocsparse_indextype_i64) ? 8 : 4;
        const size_t nalloc    = static_cast<size_t>(std::max(target_nnz, int64_t{1}));
        device_vector<char> out_ind{nalloc * idx_bytes};
        device_vector<char> out_val{nalloc * static_cast<size_t>(std::max(value_bytes, int64_t{1}))};
        ASSERT_TRUE(out_ind.ptr && out_val.ptr);

        // Row/col pointer was supplied at create time; only the compressed index
        // and values are attached after analysis.
        if(!csc)
        {
            int64_t              rows = 0, cols = 0, nnz = 0;
            void*                row = nullptr;
            void*                col = nullptr;
            void*                val = nullptr;
            rocsparse_indextype  row_type, col_type;
            rocsparse_index_base base;
            rocsparse_datatype   data_type;
            ASSERT_EQ(rocsparse_csr_get(target,
                                        &rows,
                                        &cols,
                                        &nnz,
                                        &row,
                                        &col,
                                        &val,
                                        &row_type,
                                        &col_type,
                                        &base,
                                        &data_type),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_csr_set_pointers(target, row, out_ind.ptr, out_val.ptr),
                      rocsparse_status_success);
        }
        else
        {
            int64_t              rows = 0, cols = 0, nnz = 0;
            void*                cptr = nullptr;
            void*                rind = nullptr;
            void*                val  = nullptr;
            rocsparse_indextype  ptr_type, ind_type;
            rocsparse_index_base base;
            rocsparse_datatype   data_type;
            ASSERT_EQ(rocsparse_csc_get(target,
                                        &rows,
                                        &cols,
                                        &nnz,
                                        &cptr,
                                        &rind,
                                        &val,
                                        &ptr_type,
                                        &ind_type,
                                        &base,
                                        &data_type),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_csc_set_pointers(target, cptr, out_ind.ptr, out_val.ptr),
                      rocsparse_status_success);
        }

        buffer_size = 0;
        ASSERT_EQ(rocsparse_extract_buffer_size(
                      handle, descr, source, target, rocsparse_extract_stage_compute, &buffer_size),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_compute,
                                    buffer_size,
                                    buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(rocsparse_destroy_extract_descr(descr), rocsparse_status_success);
    }

    rocsparse_status run_s2s(rocsparse_handle            handle,
                             rocsparse_const_spmat_descr source,
                             rocsparse_spmat_descr       target,
                             bool                        permissive)
    {
        rocsparse_sparse_to_sparse_descr s2s = nullptr;
        rocsparse_status                 st  = rocsparse_create_sparse_to_sparse_descr(
            &s2s, source, target, rocsparse_sparse_to_sparse_alg_default);
        if(st != rocsparse_status_success)
        {
            return st;
        }
        if(permissive)
        {
            st = rocsparse_sparse_to_sparse_permissive(s2s);
            if(st != rocsparse_status_success)
            {
                rocsparse_destroy_sparse_to_sparse_descr(s2s);
                return st;
            }
        }
        for(rocsparse_sparse_to_sparse_stage stage :
            {rocsparse_sparse_to_sparse_stage_analysis, rocsparse_sparse_to_sparse_stage_compute})
        {
            size_t buffer_size = 0;
            st = rocsparse_sparse_to_sparse_buffer_size(
                handle, s2s, source, target, stage, &buffer_size);
            if(st != rocsparse_status_success)
            {
                rocsparse_destroy_sparse_to_sparse_descr(s2s);
                return st;
            }
            device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
            if(!buffer.ptr)
            {
                rocsparse_destroy_sparse_to_sparse_descr(s2s);
                return rocsparse_status_memory_error;
            }
            st = rocsparse_sparse_to_sparse(
                handle, s2s, source, target, stage, buffer_size, buffer.ptr);
            if(st != rocsparse_status_success)
            {
                rocsparse_destroy_sparse_to_sparse_descr(s2s);
                return st;
            }
        }
        if(hipDeviceSynchronize() != hipSuccess)
        {
            rocsparse_destroy_sparse_to_sparse_descr(s2s);
            return rocsparse_status_internal_error;
        }
        return rocsparse_destroy_sparse_to_sparse_descr(s2s);
    }

    template <typename T>
    rocsparse_status gebsr_buffer(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  const rocsparse_mat_descr csr_descr,
                                  const T*                  csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_int             row_block_dim,
                                  rocsparse_int             col_block_dim,
                                  size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_scsr2gebsr_buffer_size(handle,
                                                    dir,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    row_block_dim,
                                                    col_block_dim,
                                                    buffer_size);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            return rocsparse_dcsr2gebsr_buffer_size(handle,
                                                    dir,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    row_block_dim,
                                                    col_block_dim,
                                                    buffer_size);
        }
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        {
            return rocsparse_ccsr2gebsr_buffer_size(handle,
                                                    dir,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    row_block_dim,
                                                    col_block_dim,
                                                    buffer_size);
        }
        else
        {
            return rocsparse_zcsr2gebsr_buffer_size(handle,
                                                    dir,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    row_block_dim,
                                                    col_block_dim,
                                                    buffer_size);
        }
    }

    template <typename T>
    rocsparse_status gebsr_compute(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr csr_descr,
                                   const T*                  csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   const rocsparse_mat_descr bsr_descr,
                                   T*                        bsr_val,
                                   rocsparse_int*            bsr_row_ptr,
                                   rocsparse_int*            bsr_col_ind,
                                   rocsparse_int             row_block_dim,
                                   rocsparse_int             col_block_dim,
                                   void*                     buffer)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return rocsparse_scsr2gebsr(handle,
                                        dir,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        bsr_descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        row_block_dim,
                                        col_block_dim,
                                        buffer);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            return rocsparse_dcsr2gebsr(handle,
                                        dir,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        bsr_descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        row_block_dim,
                                        col_block_dim,
                                        buffer);
        }
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        {
            return rocsparse_ccsr2gebsr(handle,
                                        dir,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        bsr_descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        row_block_dim,
                                        col_block_dim,
                                        buffer);
        }
        else
        {
            return rocsparse_zcsr2gebsr(handle,
                                        dir,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        bsr_descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        row_block_dim,
                                        col_block_dim,
                                        buffer);
        }
    }

    template <typename T>
    void run_gebsr(rocsparse_handle    handle,
                   rocsparse_direction dir,
                   rocsparse_int       row_block_dim,
                   rocsparse_int       col_block_dim,
                   bool                device_nnz,
                   bool                force_alloc)
    {
        const rocsparse_int m = std::max(row_block_dim, rocsparse_int{2});
        const rocsparse_int n = std::max(col_block_dim, rocsparse_int{1});
        std::vector<rocsparse_int> h_ptr(static_cast<size_t>(m) + 1);
        std::vector<rocsparse_int> h_col(static_cast<size_t>(m));
        std::vector<T>             h_val(static_cast<size_t>(m), one_of<T>());
        for(rocsparse_int i = 0; i < m; ++i)
        {
            h_ptr[static_cast<size_t>(i)] = i;
            h_col[static_cast<size_t>(i)] = 0;
        }
        h_ptr[static_cast<size_t>(m)] = m;

        device_vector<rocsparse_int> row_ptr{h_ptr};
        device_vector<rocsparse_int> col_ind{h_col};
        device_vector<T>             val{h_val};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

        MatDescr csr_descr;
        MatDescr bsr_descr;

        size_t buffer_size = 0;
        ASSERT_EQ(gebsr_buffer<T>(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr.d,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  row_block_dim,
                                  col_block_dim,
                                  &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        const rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;
        device_vector<rocsparse_int> bsr_row_ptr{static_cast<size_t>(mb) + 1};
        ASSERT_TRUE(bsr_row_ptr.ptr);

        std::unique_ptr<ForceHandleAlloc> forced;
        if(force_alloc)
        {
            forced.reset(new ForceHandleAlloc(handle));
        }

        rocsparse_pointer_mode saved = rocsparse_pointer_mode_host;
        ASSERT_EQ(rocsparse_get_pointer_mode(handle, &saved), rocsparse_status_success);
        device_vector<rocsparse_int> d_nnzb{size_t{1}};
        rocsparse_int                 h_nnzb = -1;
        rocsparse_int*                nnzb_ptr = &h_nnzb;
        if(device_nnz)
        {
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
            nnzb_ptr = d_nnzb.ptr;
        }

        ASSERT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          dir,
                                          m,
                                          n,
                                          csr_descr.d,
                                          row_ptr,
                                          col_ind,
                                          bsr_descr.d,
                                          bsr_row_ptr,
                                          row_block_dim,
                                          col_block_dim,
                                          nnzb_ptr,
                                          buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(device_nnz)
        {
            h_nnzb = to_host(d_nnzb.ptr, 1)[0];
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, saved), rocsparse_status_success);
        }
        EXPECT_GE(h_nnzb, 0);

        const size_t nblocks = static_cast<size_t>(std::max(h_nnzb, rocsparse_int{1}));
        const size_t nvals   = nblocks * static_cast<size_t>(row_block_dim)
                             * static_cast<size_t>(col_block_dim);
        device_vector<T>             bsr_val{nvals};
        device_vector<rocsparse_int> bsr_col{nblocks};
        ASSERT_TRUE(bsr_val.ptr && bsr_col.ptr);

        ASSERT_EQ(gebsr_compute<T>(handle,
                                   dir,
                                   m,
                                   n,
                                   csr_descr.d,
                                   val,
                                   row_ptr,
                                   col_ind,
                                   bsr_descr.d,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col,
                                   row_block_dim,
                                   col_block_dim,
                                   buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }
} // namespace

class ConversionPruneExtract : public HandleTest
{
};

TEST_F(ConversionPruneExtract, prune_csr2csr_segments)
{
    // Wavefront-32 segment sizes. Mean nnz/row selects <4, <8, <16, <32, >=32.
    const rocsparse_int means[] = {1, 5, 10, 20, 32};
    for(rocsparse_int mean : means)
    {
        const auto pattern = make_pattern(4, mean, rocsparse_index_base_zero);
        run_prune_pattern<float>(handle,
                                 pattern,
                                 rocsparse_index_base_zero,
                                 rocsparse_index_base_zero,
                                 /*device_threshold=*/false);
        run_prune_pattern<double>(handle,
                                  pattern,
                                  rocsparse_index_base_zero,
                                  rocsparse_index_base_one,
                                  /*device_threshold=*/false);
    }
    // Device-pointer threshold hits the host/device scalar branch of the kernel launch.
    const auto dev_pat = make_pattern(4, 5, rocsparse_index_base_one);
    run_prune_pattern<float>(
        handle, dev_pat, rocsparse_index_base_one, rocsparse_index_base_one, /*device_threshold=*/true);
    run_prune_pattern<double>(
        handle, dev_pat, rocsparse_index_base_one, rocsparse_index_base_zero, /*device_threshold=*/true);
}

TEST_F(ConversionPruneExtract, prune_csr2csr_edges)
{
    MatDescr descr_A;
    MatDescr descr_C;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0}};
    device_vector<rocsparse_int> row_ptr_C{std::vector<rocsparse_int>{0}};
    device_vector<float>         dummy{size_t{1}};
    const float                  threshold = 0.5f;
    size_t                       buffer_size = 0;
    ASSERT_TRUE(row_ptr.ptr && row_ptr_C.ptr && dummy.ptr);

    // m == 0 quick return, host pointer mode.
    rocsparse_int nnz_C = 7;
    ASSERT_EQ(rocsparse_sprune_csr2csr_nnz(handle,
                                           0,
                                           3,
                                           0,
                                           descr_A.d,
                                           nullptr,
                                           row_ptr,
                                           nullptr,
                                           &threshold,
                                           descr_C.d,
                                           row_ptr_C,
                                           &nnz_C,
                                           nullptr),
              rocsparse_status_success);
    EXPECT_EQ(nnz_C, 0);

    // n == 0 and nnz == 0, device pointer mode for the zeroed nnz total.
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    device_vector<rocsparse_int> d_nnz{std::vector<rocsparse_int>{9}};
    device_vector<rocsparse_int> row_ptr_m{std::vector<rocsparse_int>{0, 0, 0}};
    device_vector<rocsparse_int> row_ptr_m_C{size_t{3}};
    ASSERT_EQ(rocsparse_sprune_csr2csr_nnz(handle,
                                           2,
                                           0,
                                           0,
                                           descr_A.d,
                                           nullptr,
                                           row_ptr_m,
                                           nullptr,
                                           nullptr,
                                           descr_C.d,
                                           row_ptr_m_C,
                                           d_nnz.ptr,
                                           nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(to_host(d_nnz.ptr, 1)[0], 0);
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    // Compute quick return on n == 0, and the null-output check once row_ptr_C is live.
    ASSERT_EQ(rocsparse_sprune_csr2csr(handle,
                                       2,
                                       0,
                                       0,
                                       descr_A.d,
                                       nullptr,
                                       row_ptr_m,
                                       nullptr,
                                       &threshold,
                                       descr_C.d,
                                       nullptr,
                                       row_ptr_m_C,
                                       nullptr,
                                       nullptr),
              rocsparse_status_success);

    const auto pattern = make_pattern(4, 1, rocsparse_index_base_zero);
    device_vector<rocsparse_int> rp{pattern.row_ptr};
    device_vector<rocsparse_int> ci{pattern.col_ind};
    device_vector<float>         val{pattern.val};
    device_vector<rocsparse_int> rp_C{size_t{5}};
    ASSERT_EQ(rocsparse_sprune_csr2csr_buffer_size(handle,
                                                   4,
                                                   pattern.n,
                                                   pattern.nnz,
                                                   descr_A.d,
                                                   val,
                                                   rp,
                                                   ci,
                                                   &threshold,
                                                   descr_C.d,
                                                   nullptr,
                                                   rp_C,
                                                   nullptr,
                                                   &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
    ASSERT_EQ(rocsparse_sprune_csr2csr_nnz(handle,
                                           4,
                                           pattern.n,
                                           pattern.nnz,
                                           descr_A.d,
                                           val,
                                           rp,
                                           ci,
                                           &threshold,
                                           descr_C.d,
                                           rp_C,
                                           &nnz_C,
                                           buffer.ptr),
              rocsparse_status_success);
    // Values are 0.25. A zero threshold keeps them, so calculate_nnz sees
    // nnz_C > 0 and the null C arrays return invalid_pointer.
    const float keep = 0.0f;
    ASSERT_EQ(rocsparse_sprune_csr2csr_nnz(handle,
                                           4,
                                           pattern.n,
                                           pattern.nnz,
                                           descr_A.d,
                                           val,
                                           rp,
                                           ci,
                                           &keep,
                                           descr_C.d,
                                           rp_C,
                                           &nnz_C,
                                           buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_GT(nnz_C, 0);
    EXPECT_EQ(rocsparse_sprune_csr2csr(handle,
                                       4,
                                       pattern.n,
                                       pattern.nnz,
                                       descr_A.d,
                                       val,
                                       rp,
                                       ci,
                                       &keep,
                                       descr_C.d,
                                       nullptr,
                                       rp_C,
                                       nullptr,
                                       buffer.ptr),
              rocsparse_status_invalid_pointer);

    // Private temp allocation (handle scratch reported as empty).
    {
        ForceHandleAlloc forced(handle);
        const auto       wide = make_pattern(4, 8, rocsparse_index_base_zero);
        run_prune_pattern<float>(handle,
                                 wide,
                                 rocsparse_index_base_zero,
                                 rocsparse_index_base_zero,
                                 /*device_threshold=*/false);
        run_prune_pattern<double>(handle,
                                  wide,
                                  rocsparse_index_base_zero,
                                  rocsparse_index_base_zero,
                                  /*device_threshold=*/false);
    }

    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_A.d, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sprune_csr2csr_nnz(handle,
                                           4,
                                           pattern.n,
                                           pattern.nnz,
                                           descr_A.d,
                                           val,
                                           rp,
                                           ci,
                                           &threshold,
                                           descr_C.d,
                                           rp_C,
                                           &nnz_C,
                                           buffer.ptr),
              rocsparse_status_requires_sorted_storage);
}

TEST_F(ConversionPruneExtract, prune_csr2csr_by_percentage_segments)
{
    const rocsparse_int means[] = {1, 5, 10, 20, 32};
    const float         pcts[]  = {0.f, 50.f, 100.f};
    for(rocsparse_int mean : means)
    {
        const auto pattern = make_pattern(4, mean, rocsparse_index_base_zero);
        for(float pct : pcts)
        {
            run_pct_pattern<float>(handle,
                                   pattern,
                                   pct,
                                   rocsparse_index_base_zero,
                                   /*device_mode=*/false,
                                   /*force_alloc=*/false);
        }
        run_pct_pattern<double>(handle,
                                pattern,
                                25.0,
                                rocsparse_index_base_one,
                                /*device_mode=*/false,
                                /*force_alloc=*/false);
    }
    const auto dev_pat = make_pattern(4, 10, rocsparse_index_base_zero);
    run_pct_pattern<float>(handle, dev_pat, 40.f, rocsparse_index_base_zero, true, false);
    run_pct_pattern<double>(handle, dev_pat, 40.0, rocsparse_index_base_zero, true, true);
}

TEST_F(ConversionPruneExtract, prune_csr2csr_by_percentage_edges)
{
    MatDescr descr_A;
    MatDescr descr_C;
    MatInfo  info;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0}};
    device_vector<rocsparse_int> row_ptr_C{std::vector<rocsparse_int>{0}};
    device_vector<char>          tmp{size_t{64}};
    ASSERT_TRUE(row_ptr.ptr && row_ptr_C.ptr && tmp.ptr);

    rocsparse_int nnz_C = 3;
    ASSERT_EQ(rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                        0,
                                                        4,
                                                        0,
                                                        descr_A.d,
                                                        nullptr,
                                                        row_ptr,
                                                        nullptr,
                                                        10.f,
                                                        descr_C.d,
                                                        row_ptr_C,
                                                        &nnz_C,
                                                        info.i,
                                                        tmp.ptr),
              rocsparse_status_success);
    EXPECT_EQ(nnz_C, 0);

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    device_vector<rocsparse_int> d_nnz{std::vector<rocsparse_int>{4}};
    device_vector<rocsparse_int> row_m{std::vector<rocsparse_int>{0, 0}};
    device_vector<rocsparse_int> row_m_C{size_t{2}};
    ASSERT_EQ(rocsparse_dprune_csr2csr_nnz_by_percentage(handle,
                                                        1,
                                                        0,
                                                        0,
                                                        descr_A.d,
                                                        nullptr,
                                                        row_m,
                                                        nullptr,
                                                        0.0,
                                                        descr_C.d,
                                                        row_m_C,
                                                        d_nnz.ptr,
                                                        info.i,
                                                        tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(to_host(d_nnz.ptr, 1)[0], 0);
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_sprune_csr2csr_by_percentage(handle,
                                                    2,
                                                    0,
                                                    0,
                                                    descr_A.d,
                                                    nullptr,
                                                    row_m,
                                                    nullptr,
                                                    0.f,
                                                    descr_C.d,
                                                    nullptr,
                                                    row_m_C,
                                                    nullptr,
                                                    info.i,
                                                    tmp.ptr),
              rocsparse_status_success);

    // nnz == 0 lets the null value pointer through so the percentage check runs.
    EXPECT_EQ(rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                        2,
                                                        2,
                                                        0,
                                                        descr_A.d,
                                                        nullptr,
                                                        row_m,
                                                        nullptr,
                                                        -1.f,
                                                        descr_C.d,
                                                        row_m_C,
                                                        &nnz_C,
                                                        info.i,
                                                        tmp.ptr),
              rocsparse_status_invalid_value);

    // percentage > 100 is invalid_value when the other pointers are valid.
    device_vector<double>         dval{std::vector<double>{1.0, 2.0}};
    device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1}};
    size_t                       bs = 0;
    EXPECT_EQ(rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                descr_A.d,
                                                                dval,
                                                                rp,
                                                                ci,
                                                                150.0,
                                                                descr_C.d,
                                                                nullptr,
                                                                rp,
                                                                nullptr,
                                                                info.i,
                                                                &bs),
              rocsparse_status_invalid_value);
}

TEST_F(ConversionPruneExtract, extract_datatype_and_index)
{
    // 3x3 with entries on and off the diagonal so lower/upper and unit/non-unit
    // predicates all see both comparisons.
    const std::vector<int32_t> row32{0, 2, 3, 5};
    const std::vector<int32_t> col32{0, 1, 1, 0, 2};
    const int64_t              rows = 3, cols = 3, nnz = 5;
    device_vector<int32_t>     d_row{row32};
    device_vector<int32_t>     d_col{col32};
    ASSERT_TRUE(d_row.ptr && d_col.ptr);

    const rocsparse_datatype dts[] = {rocsparse_datatype_f16_r,
                                      rocsparse_datatype_bf16_r,
                                      rocsparse_datatype_f32_r,
                                      rocsparse_datatype_f64_r,
                                      rocsparse_datatype_f32_c,
                                      rocsparse_datatype_f64_c,
                                      rocsparse_datatype_i8_r,
                                      rocsparse_datatype_u8_r,
                                      rocsparse_datatype_i32_r,
                                      rocsparse_datatype_u32_r};
    const rocsparse_fill_mode fills[] = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};
    const rocsparse_diag_type diags[] = {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit};

    for(rocsparse_datatype dt : dts)
    {
        device_vector<char> d_val{static_cast<size_t>(nnz) * dtype_bytes(dt)};
        ASSERT_TRUE(d_val.ptr);
        ASSERT_EQ(hipMemset(d_val.ptr, 0x3f, static_cast<size_t>(nnz) * dtype_bytes(dt)), hipSuccess);

        device_vector<int32_t> t_row{size_t{4}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             rows,
                                             cols,
                                             nnz,
                                             d_row.ptr,
                                             d_col.ptr,
                                             d_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             rows,
                                             cols,
                                             0,
                                             t_row.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);
        // Every datatype runs lower/non-unit. f32 also walks the other three
        // fill/diag pairs (the predicate is inside the typed template).
        run_extract(handle,
                    source,
                    target,
                    rocsparse_fill_mode_lower,
                    rocsparse_diag_type_non_unit,
                    static_cast<int64_t>(dtype_bytes(dt)),
                    rocsparse_indextype_i32,
                    /*csc=*/false);
        if(dt == rocsparse_datatype_f32_r)
        {
            for(rocsparse_fill_mode fill : fills)
            {
                for(rocsparse_diag_type diag : diags)
                {
                    if(fill == rocsparse_fill_mode_lower && diag == rocsparse_diag_type_non_unit)
                    {
                        continue;
                    }
                    run_extract(handle,
                                source,
                                target,
                                fill,
                                diag,
                                static_cast<int64_t>(dtype_bytes(dt)),
                                rocsparse_indextype_i32,
                                false);
                }
            }
        }
    }

    // CSC selects the column direction. Mixed and 64-bit index types select the
    // I/J dispatch arms.
    {
        device_vector<float>   val{std::vector<float>{1.f, 1.f, 1.f}};
        device_vector<int32_t> cptr{std::vector<int32_t>{0, 1, 2, 3}};
        device_vector<int32_t> rind{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> tptr{size_t{4}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csc_descr(&source.d,
                                             3,
                                             3,
                                             3,
                                             cptr.ptr,
                                             rind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        // Row index of CSC is the "J" index for column direction; keep storage
        // consistent with the declared type by using i64 row indices below.
    }
    {
        device_vector<float>   val{std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f}};
        device_vector<int64_t> row{std::vector<int64_t>{0, 2, 3, 5}};
        device_vector<int32_t> col{col32};
        device_vector<int64_t> trow{size_t{4}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             rows,
                                             cols,
                                             nnz,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             rows,
                                             cols,
                                             0,
                                             trow.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        run_extract(handle,
                    source,
                    target,
                    rocsparse_fill_mode_upper,
                    rocsparse_diag_type_unit,
                    4,
                    rocsparse_indextype_i32,
                    false);
    }
    {
        device_vector<double>  val{std::vector<double>{1, 2, 3, 4, 5}};
        device_vector<int64_t> row{std::vector<int64_t>{1, 3, 4, 6}};
        device_vector<int64_t> col{std::vector<int64_t>{1, 2, 2, 1, 3}};
        device_vector<int64_t> trow{size_t{4}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             rows,
                                             cols,
                                             nnz,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_one,
                                             rocsparse_datatype_f64_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             rows,
                                             cols,
                                             0,
                                             trow.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_one,
                                             rocsparse_datatype_f64_r),
                  rocsparse_status_success);
        run_extract(handle,
                    source,
                    target,
                    rocsparse_fill_mode_lower,
                    rocsparse_diag_type_unit,
                    8,
                    rocsparse_indextype_i64,
                    false);
    }
    {
        // Column direction, i32 pointer index, i64 compressed index.
        device_vector<char>    val{size_t{3} * 16};
        device_vector<int32_t> cptr{std::vector<int32_t>{0, 1, 2, 3}};
        device_vector<int64_t> rind{std::vector<int64_t>{0, 1, 2}};
        device_vector<int32_t> tptr{size_t{4}};
        ASSERT_EQ(hipMemset(val.ptr, 1, val.n), hipSuccess);
        SpMat source, target;
        ASSERT_EQ(rocsparse_create_csc_descr(&source.d,
                                             3,
                                             3,
                                             3,
                                             cptr.ptr,
                                             rind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_c),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csc_descr(&target.d,
                                             3,
                                             3,
                                             0,
                                             tptr.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_c),
                  rocsparse_status_success);
        run_extract(handle,
                    source,
                    target,
                    rocsparse_fill_mode_upper,
                    rocsparse_diag_type_non_unit,
                    16,
                    rocsparse_indextype_i64,
                    true);
    }

    // Empty matrix: num_seq == 0 skips the count kernel and compute returns early.
    {
        device_vector<int32_t> row{std::vector<int32_t>{0}};
        device_vector<int32_t> trow{std::vector<int32_t>{0}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             0,
                                             4,
                                             0,
                                             row.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             0,
                                             4,
                                             0,
                                             trow.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        run_extract(handle,
                    source,
                    target,
                    rocsparse_fill_mode_lower,
                    rocsparse_diag_type_non_unit,
                    4,
                    rocsparse_indextype_i32,
                    false);
    }
}

TEST_F(ConversionPruneExtract, extract_rejected)
{
    device_vector<int32_t> idx{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<float>   val{std::vector<float>(4, 1.f)};
    ASSERT_TRUE(idx.ptr && val.ptr);

    auto expect_reject = [&](rocsparse_spmat_descr source, rocsparse_spmat_descr target, rocsparse_status want) {
        rocsparse_extract_descr descr = nullptr;
        EXPECT_EQ(create_extract_catch(&descr, source, target), want);
        if(descr)
        {
            rocsparse_destroy_extract_descr(descr);
        }
    };

    {
        SpMat ell_s, ell_t;
        ASSERT_EQ(rocsparse_create_ell_descr(&ell_s.d,
                                             2,
                                             2,
                                             idx.ptr,
                                             val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_ell_descr(&ell_t.d,
                                             2,
                                             2,
                                             idx.ptr,
                                             val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(ell_s, ell_t, rocsparse_status_not_implemented);
    }
    {
        device_vector<int32_t> ind{std::vector<int32_t>{0, 0, 1, 1}};
        SpMat                  s, t;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&s.d,
                                                 2,
                                                 2,
                                                 2,
                                                 ind.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&t.d,
                                                 2,
                                                 2,
                                                 2,
                                                 ind.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(s, t, rocsparse_status_not_implemented);
    }
    {
        SpMat s, t;
        ASSERT_EQ(rocsparse_create_coo_descr(&s.d,
                                             2,
                                             2,
                                             2,
                                             idx.ptr,
                                             idx.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(&t.d,
                                             2,
                                             2,
                                             2,
                                             idx.ptr,
                                             idx.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(s, t, rocsparse_status_not_implemented);
    }
    {
        SpMat s, t;
        ASSERT_EQ(rocsparse_create_bsr_descr(&s.d,
                                             2,
                                             2,
                                             2,
                                             rocsparse_direction_row,
                                             1,
                                             idx.ptr,
                                             idx.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(&t.d,
                                             2,
                                             2,
                                             2,
                                             rocsparse_direction_row,
                                             1,
                                             idx.ptr,
                                             idx.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(s, t, rocsparse_status_not_implemented);
    }
    {
        SpMat s, t;
        ASSERT_EQ(rocsparse_create_bell_descr(&s.d,
                                              2,
                                              2,
                                              rocsparse_direction_row,
                                              1,
                                              1,
                                              idx.ptr,
                                              val.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bell_descr(&t.d,
                                              2,
                                              2,
                                              rocsparse_direction_row,
                                              1,
                                              1,
                                              idx.ptr,
                                              val.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(s, t, rocsparse_status_not_implemented);
    }
    {
        device_vector<int32_t> off{std::vector<int32_t>{0, 1}};
        device_vector<int32_t> col{std::vector<int32_t>{0}};
        device_vector<float>   one{std::vector<float>{1.f}};
        SpMat                  s, t;
        ASSERT_EQ(rocsparse_create_sell_descr(&s.d,
                                              2,
                                              2,
                                              1,
                                              1,
                                              1,
                                              off.ptr,
                                              col.ptr,
                                              one.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_sell_descr(&t.d,
                                              2,
                                              2,
                                              1,
                                              1,
                                              1,
                                              off.ptr,
                                              col.ptr,
                                              one.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(s, t, rocsparse_status_not_implemented);
    }

    // Format mismatch and storage-mode mismatch throw from the constructor.
    {
        device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> col{std::vector<int32_t>{0, 1}};
        SpMat                  csr, csc;
        ASSERT_EQ(rocsparse_create_csr_descr(&csr.d,
                                             2,
                                             2,
                                             2,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csc_descr(&csc.d,
                                             2,
                                             2,
                                             2,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        expect_reject(csr, csc, rocsparse_status_internal_error);
    }
    {
        device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> col{std::vector<int32_t>{0, 1}};
        device_vector<int32_t> trow{size_t{3}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             2,
                                             2,
                                             2,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             2,
                                             2,
                                             0,
                                             trow.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        const rocsparse_storage_mode unsorted = rocsparse_storage_mode_unsorted;
        set_attr(target, rocsparse_spmat_storage_mode, &unsorted, sizeof(unsorted));
        expect_reject(source, target, rocsparse_status_internal_error);
    }

    // Deprecated u16 index type is rejected inside the I and J dispatches.
    const auto u16 = static_cast<rocsparse_indextype>(1);
    {
        device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> col{std::vector<int32_t>{0, 1}};
        device_vector<int32_t> trow{size_t{3}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             2,
                                             2,
                                             2,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             u16,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             2,
                                             2,
                                             0,
                                             trow.ptr,
                                             nullptr,
                                             nullptr,
                                             u16,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_extract_descr descr = nullptr;
        ASSERT_EQ(create_extract_catch(&descr, source, target), rocsparse_status_success);
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_extract_buffer_size(
                      handle, descr, source, target, rocsparse_extract_stage_analysis, &buffer_size),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_analysis,
                                    0,
                                    nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_compute,
                                    0,
                                    nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_destroy_extract_descr(descr), rocsparse_status_success);
    }
    {
        // J (column index) is u16; I (row pointer) stays i32 so buffer-size,
        // which dispatches only on I, succeeds and analysis hits the J case.
        device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> col{std::vector<int32_t>{0, 1}};
        device_vector<int32_t> trow{size_t{3}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             2,
                                             2,
                                             2,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             u16,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             2,
                                             2,
                                             0,
                                             trow.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             u16,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_extract_descr descr = nullptr;
        ASSERT_EQ(create_extract_catch(&descr, source, target), rocsparse_status_success);
        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_extract_buffer_size(
                      handle, descr, source, target, rocsparse_extract_stage_analysis, &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_analysis,
                                    buffer_size,
                                    buffer.ptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_compute,
                                    0,
                                    buffer.ptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_destroy_extract_descr(descr), rocsparse_status_success);
    }

    // Dimensions that do not fit in i32 take the overflow returns. Pointers are
    // non-null dummies; the checks return before any O(dim) work.
    const int64_t huge = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    {
        device_vector<int32_t> dummy{size_t{4}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             huge,
                                             1,
                                             0,
                                             dummy.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             huge,
                                             1,
                                             0,
                                             dummy.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_extract_descr descr = nullptr;
        ASSERT_EQ(create_extract_catch(&descr, source, target), rocsparse_status_success);
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_extract_buffer_size(
                      handle, descr, source, target, rocsparse_extract_stage_analysis, &buffer_size),
                  rocsparse_status_internal_error);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_analysis,
                                    0,
                                    nullptr),
                  rocsparse_status_internal_error);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_compute,
                                    0,
                                    nullptr),
                  rocsparse_status_internal_error);
        EXPECT_EQ(rocsparse_destroy_extract_descr(descr), rocsparse_status_success);
    }
    {
        device_vector<int32_t> dummy{size_t{4}};
        SpMat                  source, target;
        ASSERT_EQ(rocsparse_create_csr_descr(&source.d,
                                             1,
                                             huge,
                                             0,
                                             dummy.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&target.d,
                                             1,
                                             huge,
                                             0,
                                             dummy.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_extract_descr descr = nullptr;
        ASSERT_EQ(create_extract_catch(&descr, source, target), rocsparse_status_success);
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_extract_buffer_size(
                      handle, descr, source, target, rocsparse_extract_stage_analysis, &buffer_size),
                  rocsparse_status_internal_error);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_analysis,
                                    0,
                                    nullptr),
                  rocsparse_status_internal_error);
        EXPECT_EQ(rocsparse_extract(handle,
                                    descr,
                                    source,
                                    target,
                                    rocsparse_extract_stage_compute,
                                    0,
                                    nullptr),
                  rocsparse_status_internal_error);
        EXPECT_EQ(rocsparse_destroy_extract_descr(descr), rocsparse_status_success);
    }
}

TEST_F(ConversionPruneExtract, sparse_to_sparse_formats)
{
    device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.f, 2.f}};
    device_vector<int32_t> coo_row{std::vector<int32_t>{0, 1}};
    device_vector<int32_t> aos{std::vector<int32_t>{0, 0, 1, 1}};
    device_vector<int64_t> row64{std::vector<int64_t>{0, 1, 2}};
    device_vector<int64_t> col64{std::vector<int64_t>{0, 1}};
    device_vector<int64_t> aos64{std::vector<int64_t>{0, 0, 1, 1}};
    ASSERT_TRUE(row.ptr && col.ptr && val.ptr && coo_row.ptr && aos.ptr && row64.ptr && col64.ptr
                && aos64.ptr);

    auto csr = [&](SpMat& m, void* rp, void* ci, rocsparse_indextype itype, rocsparse_index_base base) {
        return rocsparse_create_csr_descr(&m.d,
                                          2,
                                          2,
                                          2,
                                          rp,
                                          ci,
                                          val.ptr,
                                          itype,
                                          itype,
                                          base,
                                          rocsparse_datatype_f32_r);
    };
    auto csc = [&](SpMat& m, void* cp, void* ri, rocsparse_indextype itype) {
        return rocsparse_create_csc_descr(&m.d,
                                          2,
                                          2,
                                          2,
                                          cp,
                                          ri,
                                          val.ptr,
                                          itype,
                                          itype,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r);
    };

    {
        SpMat a, b;
        ASSERT_EQ(csr(a, row.ptr, col.ptr, rocsparse_indextype_i32, rocsparse_index_base_zero),
                  rocsparse_status_success);
        ASSERT_EQ(csr(b, row.ptr, col.ptr, rocsparse_indextype_i32, rocsparse_index_base_zero),
                  rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, a, b, false), rocsparse_status_success);
    }
    {
        SpMat a, b;
        ASSERT_EQ(csc(a, row.ptr, col.ptr, rocsparse_indextype_i32), rocsparse_status_success);
        ASSERT_EQ(csc(b, row.ptr, col.ptr, rocsparse_indextype_i32), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, a, b, false), rocsparse_status_success);
    }
    {
        SpMat a, b;
        ASSERT_EQ(rocsparse_create_coo_descr(&a.d,
                                             2,
                                             2,
                                             2,
                                             coo_row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(&b.d,
                                             2,
                                             2,
                                             2,
                                             coo_row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, a, b, false), rocsparse_status_success);
    }
    {
        SpMat a, b;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&a.d,
                                                 2,
                                                 2,
                                                 2,
                                                 aos.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&b.d,
                                                 2,
                                                 2,
                                                 2,
                                                 aos.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, a, b, false), rocsparse_status_success);
    }
    {
        SpMat a, b;
        ASSERT_EQ(rocsparse_create_ell_descr(&a.d,
                                             2,
                                             2,
                                             col.ptr,
                                             val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_ell_descr(&b.d,
                                             2,
                                             2,
                                             col.ptr,
                                             val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, a, b, false), rocsparse_status_success);
    }
    {
        SpMat a, b;
        ASSERT_EQ(rocsparse_create_bsr_descr(&a.d,
                                             2,
                                             2,
                                             2,
                                             rocsparse_direction_row,
                                             1,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(&b.d,
                                             2,
                                             2,
                                             2,
                                             rocsparse_direction_row,
                                             1,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        // Same format, possibly different block direction: still the bsr->bsr path.
        EXPECT_EQ(run_s2s(handle, a, b, true), rocsparse_status_success);
    }

    // i64 index identity and a coo_aos -> csr pair (mixed formats, 64-bit indices).
    {
        SpMat a, b;
        ASSERT_EQ(csr(a, row64.ptr, col64.ptr, rocsparse_indextype_i64, rocsparse_index_base_one),
                  rocsparse_status_success);
        // 1-based row pointer must match the base. Rebuild.
    }
    {
        device_vector<int64_t> rp{std::vector<int64_t>{1, 2, 3}};
        device_vector<int64_t> ci{std::vector<int64_t>{1, 2}};
        SpMat                  a, b, coo;
        ASSERT_EQ(rocsparse_create_csr_descr(&a.d,
                                             2,
                                             2,
                                             2,
                                             rp.ptr,
                                             ci.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_one,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&b.d,
                                             2,
                                             2,
                                             2,
                                             rp.ptr,
                                             ci.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_index_base_one,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, a, b, false), rocsparse_status_success);

        device_vector<int64_t> aind{std::vector<int64_t>{1, 1, 2, 2}};
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&coo.d,
                                                 2,
                                                 2,
                                                 2,
                                                 aind.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_index_base_one,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, coo, b, true), rocsparse_status_success);
    }

    // Cross-format pairs that select the two-step and direct converters.
    {
        SpMat csr_m, csc_m, coo_m, aos_m, ell_m, bsr_m;
        ASSERT_EQ(csr(csr_m, row.ptr, col.ptr, rocsparse_indextype_i32, rocsparse_index_base_zero),
                  rocsparse_status_success);
        ASSERT_EQ(csc(csc_m, row.ptr, col.ptr, rocsparse_indextype_i32), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(&coo_m.d,
                                             2,
                                             2,
                                             2,
                                             coo_row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&aos_m.d,
                                                 2,
                                                 2,
                                                 2,
                                                 aos.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_ell_descr(&ell_m.d,
                                             2,
                                             2,
                                             col.ptr,
                                             val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(&bsr_m.d,
                                             2,
                                             2,
                                             2,
                                             rocsparse_direction_row,
                                             1,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        EXPECT_EQ(run_s2s(handle, csr_m, ell_m, false), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, ell_m, csr_m, false), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, csr_m, bsr_m, false), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, bsr_m, csr_m, false), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, coo_m, aos_m, false), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, aos_m, coo_m, false), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, coo_m, csc_m, true), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, csc_m, ell_m, true), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, bsr_m, coo_m, true), rocsparse_status_success);
        EXPECT_EQ(run_s2s(handle, ell_m, bsr_m, true), rocsparse_status_success);
    }

    // Blocked ELL is not implemented for every source and target format.
    {
        SpMat bell, csr_m, coo_m, aos_m, csc_m, ell_m, bsr_m;
        ASSERT_EQ(rocsparse_create_bell_descr(&bell.d,
                                              2,
                                              2,
                                              rocsparse_direction_row,
                                              1,
                                              1,
                                              col.ptr,
                                              val.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(csr(csr_m, row.ptr, col.ptr, rocsparse_indextype_i32, rocsparse_index_base_zero),
                  rocsparse_status_success);
        ASSERT_EQ(csc(csc_m, row.ptr, col.ptr, rocsparse_indextype_i32), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(&coo_m.d,
                                             2,
                                             2,
                                             2,
                                             coo_row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&aos_m.d,
                                                 2,
                                                 2,
                                                 2,
                                                 aos.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_ell_descr(&ell_m.d,
                                             2,
                                             2,
                                             col.ptr,
                                             val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(&bsr_m.d,
                                             2,
                                             2,
                                             2,
                                             rocsparse_direction_row,
                                             1,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        rocsparse_spmat_descr sources[] = {coo_m, aos_m, csr_m, csc_m, ell_m, bsr_m, bell};
        for(rocsparse_spmat_descr src : sources)
        {
            EXPECT_EQ(run_s2s(handle, src, bell, true), rocsparse_status_not_implemented);
        }
        for(rocsparse_spmat_descr dst : sources)
        {
            if(dst == bell.d)
            {
                continue;
            }
            EXPECT_EQ(run_s2s(handle, bell, dst, true), rocsparse_status_not_implemented);
        }
    }
}

TEST_F(ConversionPruneExtract, sparse_to_sparse_descr_mismatch)
{
    device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.f, 2.f}};
    ASSERT_TRUE(row.ptr && col.ptr && val.ptr);

    auto make = [&](SpMat& m, rocsparse_index_base base) {
        ASSERT_EQ(rocsparse_create_csr_descr(&m.d,
                                             2,
                                             2,
                                             2,
                                             row.ptr,
                                             col.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             base,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
    };

    auto expect_mismatch = [&](rocsparse_spmat_descr a, rocsparse_spmat_descr b) {
        rocsparse_sparse_to_sparse_descr s2s = nullptr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                      &s2s, a, b, rocsparse_sparse_to_sparse_alg_default),
                  rocsparse_status_success);
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                      handle, s2s, a, b, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
    };

    {
        SpMat a, b;
        make(a, rocsparse_index_base_zero);
        make(b, rocsparse_index_base_zero);
        const rocsparse_matrix_type sym = rocsparse_matrix_type_symmetric;
        set_attr(b, rocsparse_spmat_matrix_type, &sym, sizeof(sym));
        expect_mismatch(a, b);
    }
    {
        SpMat a, b;
        make(a, rocsparse_index_base_zero);
        make(b, rocsparse_index_base_zero);
        const rocsparse_fill_mode upper = rocsparse_fill_mode_upper;
        set_attr(b, rocsparse_spmat_fill_mode, &upper, sizeof(upper));
        expect_mismatch(a, b);
    }
    {
        SpMat a, b;
        make(a, rocsparse_index_base_zero);
        make(b, rocsparse_index_base_zero);
        const rocsparse_diag_type unit = rocsparse_diag_type_unit;
        set_attr(b, rocsparse_spmat_diag_type, &unit, sizeof(unit));
        expect_mismatch(a, b);
    }
    {
        SpMat a, b;
        make(a, rocsparse_index_base_zero);
        make(b, rocsparse_index_base_one);
        expect_mismatch(a, b);
    }
    {
        SpMat a, b;
        make(a, rocsparse_index_base_zero);
        make(b, rocsparse_index_base_zero);
        const rocsparse_storage_mode unsorted = rocsparse_storage_mode_unsorted;
        set_attr(b, rocsparse_spmat_storage_mode, &unsorted, sizeof(unsorted));
        expect_mismatch(a, b);
    }

    // batch_count > 1 is rejected in internal_sparse_to_sparse. Strides stay 0
    // so descriptor creation itself accepts the batch.
    {
        SpMat a, b;
        make(a, rocsparse_index_base_zero);
        make(b, rocsparse_index_base_zero);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(a, 2, 0, 0), rocsparse_status_success);
        rocsparse_sparse_to_sparse_descr s2s = nullptr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                      &s2s, a, b, rocsparse_sparse_to_sparse_alg_default),
                  rocsparse_status_success);
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                      handle, s2s, a, b, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
    }
}

TEST_F(ConversionPruneExtract, csr2gebsr_buckets)
{
    // One launch per host size bucket on wavefront 32, both directions for a
    // small rectangular case, square blocks (csr2bsr), row_block == 1, and the
    // >64 fallback. Double complex with col > 32 takes the non-64x64 fallback.
    struct Item
    {
        rocsparse_int       rbd;
        rocsparse_int       cbd;
        rocsparse_direction dir;
        int                 prec; // 0 float, 1 double, 2 fcomplex, 3 dcomplex
        bool                device_nnz;
        bool                force_alloc;
    };
    const Item items[] = {
        {2, 1, rocsparse_direction_row, 0, true, false},
        {2, 3, rocsparse_direction_column, 0, false, false},
        {2, 6, rocsparse_direction_row, 1, false, false},
        {2, 12, rocsparse_direction_row, 0, false, false},
        {2, 24, rocsparse_direction_row, 2, false, false},
        {2, 40, rocsparse_direction_row, 0, false, false},
        {3, 1, rocsparse_direction_row, 0, false, false},
        {3, 4, rocsparse_direction_column, 0, false, false},
        {3, 6, rocsparse_direction_row, 0, false, false},
        {3, 12, rocsparse_direction_row, 0, false, false},
        {3, 24, rocsparse_direction_row, 0, false, false},
        {3, 40, rocsparse_direction_row, 3, false, false},
        {5, 1, rocsparse_direction_row, 0, false, false},
        {5, 3, rocsparse_direction_row, 0, false, false},
        {5, 6, rocsparse_direction_row, 0, false, false},
        {5, 12, rocsparse_direction_row, 0, false, false},
        {5, 24, rocsparse_direction_row, 0, false, false},
        {5, 40, rocsparse_direction_row, 0, false, false},
        {9, 1, rocsparse_direction_row, 0, false, false},
        {9, 3, rocsparse_direction_row, 0, false, false},
        {9, 6, rocsparse_direction_row, 0, false, false},
        {9, 12, rocsparse_direction_row, 0, false, false},
        {9, 24, rocsparse_direction_column, 0, false, false},
        {9, 40, rocsparse_direction_row, 0, false, false},
        {17, 1, rocsparse_direction_row, 0, false, false},
        {17, 3, rocsparse_direction_row, 0, false, false},
        {17, 6, rocsparse_direction_row, 0, false, false},
        {17, 12, rocsparse_direction_row, 0, false, false},
        {17, 24, rocsparse_direction_row, 1, false, false},
        {17, 40, rocsparse_direction_row, 0, false, false},
        {40, 1, rocsparse_direction_row, 0, false, false},
        {40, 3, rocsparse_direction_row, 0, false, false},
        {40, 6, rocsparse_direction_row, 0, false, false},
        {40, 12, rocsparse_direction_row, 0, false, false},
        {40, 24, rocsparse_direction_row, 0, false, false},
        {48, 40, rocsparse_direction_row, 0, false, false},
        {48, 40, rocsparse_direction_column, 3, false, true},
        {70, 3, rocsparse_direction_row, 0, false, true},
        {4, 4, rocsparse_direction_row, 0, false, false},
        {4, 4, rocsparse_direction_column, 2, false, false},
        {1, 3, rocsparse_direction_row, 0, true, false},
        {1, 5, rocsparse_direction_column, 1, false, false},
    };
    for(const Item& it : items)
    {
        if(it.prec == 0)
        {
            run_gebsr<float>(handle, it.dir, it.rbd, it.cbd, it.device_nnz, it.force_alloc);
        }
        else if(it.prec == 1)
        {
            run_gebsr<double>(handle, it.dir, it.rbd, it.cbd, it.device_nnz, it.force_alloc);
        }
        else if(it.prec == 2)
        {
            run_gebsr<rocsparse_float_complex>(
                handle, it.dir, it.rbd, it.cbd, it.device_nnz, it.force_alloc);
        }
        else
        {
            run_gebsr<rocsparse_double_complex>(
                handle, it.dir, it.rbd, it.cbd, it.device_nnz, it.force_alloc);
        }
    }

    // Null column index with a non-empty row pointer takes the nnz pointer check.
    {
        MatDescr                      csr_descr, bsr_descr;
        device_vector<rocsparse_int>  row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int>  bsr_row{size_t{3}};
        device_vector<float>          val{std::vector<float>{1.f, 1.f}};
        rocsparse_int                 nnzb = 0;
        EXPECT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          rocsparse_direction_row,
                                          2,
                                          4,
                                          csr_descr.d,
                                          row_ptr,
                                          nullptr,
                                          bsr_descr.d,
                                          bsr_row,
                                          2,
                                          3,
                                          &nnzb,
                                          nullptr),
                  rocsparse_status_invalid_pointer);
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_scsr2gebsr_buffer_size(handle,
                                                   rocsparse_direction_row,
                                                   2,
                                                   4,
                                                   csr_descr.d,
                                                   nullptr,
                                                   row_ptr,
                                                   nullptr,
                                                   2,
                                                   3,
                                                   &buffer_size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          static_cast<rocsparse_direction>(99),
                                          2,
                                          4,
                                          csr_descr.d,
                                          row_ptr,
                                          row_ptr,
                                          bsr_descr.d,
                                          bsr_row,
                                          2,
                                          3,
                                          &nnzb,
                                          nullptr),
                  rocsparse_status_invalid_value);
    }

    // m == 0 quick return, both pointer modes.
    {
        MatDescr                     csr_descr, bsr_descr;
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0}};
        device_vector<rocsparse_int> bsr_row{std::vector<rocsparse_int>{0}};
        rocsparse_int                nnzb = 5;
        ASSERT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          rocsparse_direction_row,
                                          0,
                                          4,
                                          csr_descr.d,
                                          row_ptr,
                                          nullptr,
                                          bsr_descr.d,
                                          bsr_row,
                                          2,
                                          3,
                                          &nnzb,
                                          nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(nnzb, 0);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                  rocsparse_status_success);
        device_vector<rocsparse_int> d_nnz{std::vector<rocsparse_int>{8}};
        ASSERT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          rocsparse_direction_column,
                                          3,
                                          0,
                                          csr_descr.d,
                                          row_ptr,
                                          nullptr,
                                          bsr_descr.d,
                                          bsr_row,
                                          2,
                                          3,
                                          d_nnz.ptr,
                                          nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(to_host(d_nnz.ptr, 1)[0], 0);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        size_t buffer_size = 99;
        ASSERT_EQ(rocsparse_scsr2gebsr_buffer_size(handle,
                                                   rocsparse_direction_row,
                                                   0,
                                                   0,
                                                   csr_descr.d,
                                                   nullptr,
                                                   row_ptr,
                                                   nullptr,
                                                   2,
                                                   3,
                                                   &buffer_size),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size, 0u);
    }
}
