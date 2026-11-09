/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_SPARSE_HPP_
#define ROCALUTION_HIP_HIP_SPARSE_HPP_

#include "../../utils/type_traits.hpp"
#include <rocsparse/rocsparse.h>

namespace rocalution
{
    // ValueType to rocsparse_datatype. Value types rocSPARSE has no datatype for
    // (e.g. bool and int64_t, which rocALUTION instantiates its vectors with) only
    // provide is_supported = false, so accessing value for them fails to compile and
    // has to be guarded by if constexpr(is_supported).
    template <typename ValueType>
    struct rocalution_datatype_traits
    {
        static constexpr bool is_supported = false;
    };

    template <>
    struct rocalution_datatype_traits<float>
    {
        static constexpr bool               is_supported = true;
        static constexpr rocsparse_datatype value        = rocsparse_datatype_f32_r;
    };

    template <>
    struct rocalution_datatype_traits<double>
    {
        static constexpr bool               is_supported = true;
        static constexpr rocsparse_datatype value        = rocsparse_datatype_f64_r;
    };

    template <>
    struct rocalution_datatype_traits<std::complex<float>>
    {
        static constexpr bool               is_supported = true;
        static constexpr rocsparse_datatype value        = rocsparse_datatype_f32_c;
    };

    template <>
    struct rocalution_datatype_traits<std::complex<double>>
    {
        static constexpr bool               is_supported = true;
        static constexpr rocsparse_datatype value        = rocsparse_datatype_f64_c;
    };

    template <>
    struct rocalution_datatype_traits<int32_t>
    {
        static constexpr bool               is_supported = true;
        static constexpr rocsparse_datatype value        = rocsparse_datatype_i32_r;
    };

    template <typename IndexType>
    struct rocalution_indextype_traits;

    template <>
    struct rocalution_indextype_traits<int32_t>
    {
        static constexpr rocsparse_indextype value = rocsparse_indextype_i32;
    };

    template <>
    struct rocalution_indextype_traits<int64_t>
    {
        static constexpr rocsparse_indextype value = rocsparse_indextype_i64;
    };

    // rocsparse v2 spmv, the analysis is performed once and shared by all subsequent
    // compute calls. The analysis has to be cleared whenever the sparse matrix
    // descriptor or the sparsity pattern of the matrix changes.
    template <typename ValueType>
    class HIPSpMV
    {
    public:
        HIPSpMV(void);
        ~HIPSpMV(void);

        HIPSpMV(const HIPSpMV&)            = delete;
        HIPSpMV& operator=(const HIPSpMV&) = delete;

        bool IsAnalysed(void) const;

        void Analyse(rocsparse_handle            handle,
                     rocsparse_spmv_alg          alg,
                     rocsparse_const_spmat_descr mat,
                     ValueType                   alpha,
                     rocsparse_const_dnvec_descr x,
                     ValueType                   beta,
                     rocsparse_dnvec_descr       y);
        // Analyse without vectors, e.g. ahead of the first multiplication
        void Analyse(rocsparse_handle            handle,
                     rocsparse_spmv_alg          alg,
                     rocsparse_const_spmat_descr mat);

        void Compute(rocsparse_handle            handle,
                     ValueType                   alpha,
                     rocsparse_const_spmat_descr mat,
                     rocsparse_const_dnvec_descr x,
                     ValueType                   beta,
                     rocsparse_dnvec_descr       y) const;

        void Clear(void);

    private:
        rocsparse_spmv_descr descr_;
        size_t               buffer_size_;
        char*                buffer_;
    };

    // rocsparse sptrsv, solves op(T) * y = alpha * x, where T is the triangle of a sparse
    // matrix that is selected by fill_mode and diag_type. rocsparse supports CSR, COO and
    // ELL matrices. The triangle is kept in its own sparse matrix descriptor on the arrays
    // of the matrix, so the matrix descriptor can be used for other operations meanwhile.
    // Update() has to be called whenever the matrix descriptor is recreated, e.g. after
    // the arrays of the matrix have been reallocated.
    template <typename ValueType>
    class HIPSpTRSV
    {
    public:
        HIPSpTRSV(void);
        ~HIPSpTRSV(void);

        HIPSpTRSV(const HIPSpTRSV&)            = delete;
        HIPSpTRSV& operator=(const HIPSpTRSV&) = delete;

        bool IsAnalysed(void) const;

        void Analyse(rocsparse_handle            handle,
                     rocsparse_const_spmat_descr mat,
                     rocsparse_operation         trans,
                     rocsparse_fill_mode         fill_mode,
                     rocsparse_diag_type         diag_type);

        // Points the triangle to the arrays of mat. The analysis is kept if the sizes of
        // mat did not change, and cleared otherwise.
        void Update(rocsparse_const_spmat_descr mat);

        void Solve(rocsparse_handle            handle,
                   ValueType                   alpha,
                   rocsparse_const_dnvec_descr x,
                   rocsparse_dnvec_descr       y) const;

        void Clear(void);

    private:
        rocsparse_spmat_descr triangle_;
        rocsparse_fill_mode   fill_mode_;
        rocsparse_diag_type   diag_type_;

        rocsparse_sptrsv_descr descr_;
        size_t                 buffer_size_;
        char*                  buffer_;
    };

    // rocsparse spitsv, solves op(T) * y = alpha * x with Jacobi iterations, starting from
    // the initial guess in y, where T is the triangle of a sparse matrix that is selected by
    // fill_mode and diag_type. rocsparse supports CSR matrices only. As for HIPSpTRSV, the
    // triangle is kept in its own sparse matrix descriptor on the arrays of the matrix, and
    // rocsparse stores the analysis in this descriptor. Update() has to be called whenever
    // the matrix descriptor is recreated.
    template <typename ValueType>
    class HIPSpITSV
    {
    public:
        HIPSpITSV(void);
        ~HIPSpITSV(void);

        HIPSpITSV(const HIPSpITSV&)            = delete;
        HIPSpITSV& operator=(const HIPSpITSV&) = delete;

        bool IsAnalysed(void) const;

        void Analyse(rocsparse_handle            handle,
                     rocsparse_const_spmat_descr mat,
                     rocsparse_operation         trans,
                     rocsparse_fill_mode         fill_mode,
                     rocsparse_diag_type         diag_type);

        // Points the triangle to the arrays of mat. The analysis is kept if the sizes of
        // mat did not change, and cleared otherwise.
        void Update(rocsparse_const_spmat_descr mat);

        // Iterates until the residual norm drops below tolerance, or for max_iter
        // iterations if use_tol is false
        void Solve(rocsparse_handle      handle,
                   int                   max_iter,
                   double                tolerance,
                   bool                  use_tol,
                   ValueType             alpha,
                   rocsparse_dnvec_descr x,
                   rocsparse_dnvec_descr y) const;

        void Clear(void);

    private:
        rocsparse_spmat_descr triangle_;
        rocsparse_operation   trans_;

        size_t buffer_size_;
        char*  buffer_;
    };

    // In-place incomplete LU and incomplete Cholesky factorizations with zero fill-in.
    // rocsparse supports CSR and BSR matrices.
    template <typename ValueType>
    void HIPSpILU0(rocsparse_handle handle, rocsparse_spmat_descr mat);

    template <typename ValueType>
    void HIPSpIC0(rocsparse_handle handle, rocsparse_spmat_descr mat);

    // In-place sort of the column indices within each row, values are reordered with them.
    // COO matrices are sorted by row and then by column. rocsparse supports COO, CSR and CSC
    // matrices.
    void HIPSpSort(rocsparse_handle handle, rocsparse_spmat_descr mat);

    // rocsparse bsrsv buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const ValueType*          bsr_val,
                                                 const int*                bsr_row_ptr,
                                                 const int*                bsr_col_ind,
                                                 int                       bsr_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size);

    // rocsparse bsrsv analysis
    template <typename ValueType>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              int                       mb,
                                              int                       nnzb,
                                              const rocsparse_mat_descr descr,
                                              const ValueType*          bsr_val,
                                              const int*                bsr_row_ptr,
                                              const int*                bsr_col_ind,
                                              int                       bsr_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer);

    // rocsparse bsrsv
    template <typename ValueType>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nnzb,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     rocsparse_mat_info        info,
                                     const ValueType*          x,
                                     ValueType*                y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer);

    // rocsparse csritilu0 compute
    template <typename ValueType>
    rocsparse_status rocsparseTcsritilu0_compute_ex(rocsparse_handle            handle,
                                                    rocsparse_itilu0_alg        alg,
                                                    rocsparse_int               option,
                                                    rocsparse_int*              nmaxiter,
                                                    rocsparse_int               nfreeiter,
                                                    numeric_traits_t<ValueType> tol,
                                                    rocsparse_int               m,
                                                    rocsparse_int               nnz,
                                                    const rocsparse_int*        csr_row_ptr,
                                                    const rocsparse_int*        csr_col_ind,
                                                    const ValueType*            csr_val,
                                                    ValueType*                  ilu0,
                                                    rocsparse_index_base        idx_base,
                                                    size_t                      buffer_size,
                                                    void*                       buffer);

    // rocsparse csritilu0 history
    template <typename ValueType>
    rocsparse_status rocsparseTcsritilu0_history(rocsparse_handle             handle,
                                                 rocsparse_itilu0_alg         alg,
                                                 rocsparse_int*               niter,
                                                 numeric_traits_t<ValueType>* data,
                                                 size_t                       buffer_size,
                                                 void*                        buffer);

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_SPARSE_HPP_
