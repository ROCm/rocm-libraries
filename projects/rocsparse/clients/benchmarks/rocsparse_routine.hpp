/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_arguments.hpp"

// clang-format off
#define ROCSPARSE_FOREACH_ROUTINE			\
ROCSPARSE_DO_ROUTINE(axpyi)						\
ROCSPARSE_DO_ROUTINE(bellmm)						\
ROCSPARSE_DO_ROUTINE(bellmm_batched)					\
ROCSPARSE_DO_ROUTINE(bsrgeam)					\
ROCSPARSE_DO_ROUTINE(bsric0)					\
ROCSPARSE_DO_ROUTINE(bsrilu0)					\
ROCSPARSE_DO_ROUTINE(bsrgemm)					\
ROCSPARSE_DO_ROUTINE(bsrmm)					\
ROCSPARSE_DO_ROUTINE(bsrmv)					\
ROCSPARSE_DO_ROUTINE(bsrsm)					\
ROCSPARSE_DO_ROUTINE(bsrsv)					\
ROCSPARSE_DO_ROUTINE(bsrxmv)					\
ROCSPARSE_DO_ROUTINE(bsr2csr)					\
ROCSPARSE_DO_ROUTINE(check_matrix_csr)					\
ROCSPARSE_DO_ROUTINE(check_matrix_csc)					\
ROCSPARSE_DO_ROUTINE(check_matrix_coo)					\
ROCSPARSE_DO_ROUTINE(check_matrix_gebsr)					\
ROCSPARSE_DO_ROUTINE(check_matrix_gebsc)					\
ROCSPARSE_DO_ROUTINE(check_matrix_ell)					\
ROCSPARSE_DO_ROUTINE(check_matrix_hyb)					\
ROCSPARSE_DO_ROUTINE(coomm)					\
ROCSPARSE_DO_ROUTINE(coomm_batched)					\
ROCSPARSE_DO_ROUTINE(coomv)					\
ROCSPARSE_DO_ROUTINE(coosort)					\
ROCSPARSE_DO_ROUTINE(coosv)					\
ROCSPARSE_DO_ROUTINE(coomv_aos)					\
ROCSPARSE_DO_ROUTINE(coosm)					\
ROCSPARSE_DO_ROUTINE(coo2csr)					\
ROCSPARSE_DO_ROUTINE(coo2dense)					\
ROCSPARSE_DO_ROUTINE(cscsort)					\
ROCSPARSE_DO_ROUTINE(csc2dense)					\
ROCSPARSE_DO_ROUTINE(csrcolor)					\
ROCSPARSE_DO_ROUTINE(csric0)					\
ROCSPARSE_DO_ROUTINE(csrilu0)					\
ROCSPARSE_DO_ROUTINE(csritilu0)					\
ROCSPARSE_DO_ROUTINE(csritilu0_ex)				\
ROCSPARSE_DO_ROUTINE(csrgeam)					\
ROCSPARSE_DO_ROUTINE(csrgemm)					\
ROCSPARSE_DO_ROUTINE(csrgemm_reuse)				\
ROCSPARSE_DO_ROUTINE(csrmv)					\
ROCSPARSE_DO_ROUTINE(csrmv_analysis)					\
ROCSPARSE_DO_ROUTINE(csrmv_managed)				\
ROCSPARSE_DO_ROUTINE(cscmv)					\
ROCSPARSE_DO_ROUTINE(csrmm)					\
ROCSPARSE_DO_ROUTINE(csrmm_batched)					\
ROCSPARSE_DO_ROUTINE(cscmm)					\
ROCSPARSE_DO_ROUTINE(cscmm_batched)					\
ROCSPARSE_DO_ROUTINE(csrsm)					\
ROCSPARSE_DO_ROUTINE(csrsort)					\
ROCSPARSE_DO_ROUTINE(csrsv)					\
ROCSPARSE_DO_ROUTINE(csritsv)					\
ROCSPARSE_DO_ROUTINE(spitsv_csr)				\
ROCSPARSE_DO_ROUTINE(csr2dense)					\
ROCSPARSE_DO_ROUTINE(csr2bsr)					\
ROCSPARSE_DO_ROUTINE(csr2coo)					\
ROCSPARSE_DO_ROUTINE(csr2csc)					\
ROCSPARSE_DO_ROUTINE(csr2csr_compress)				\
ROCSPARSE_DO_ROUTINE(csr2ell)					\
ROCSPARSE_DO_ROUTINE(csr2gebsr)					\
ROCSPARSE_DO_ROUTINE(csr2hyb)					\
ROCSPARSE_DO_ROUTINE(dense2coo)					\
ROCSPARSE_DO_ROUTINE(dense2csc)					\
ROCSPARSE_DO_ROUTINE(dense2csr)					\
ROCSPARSE_DO_ROUTINE(dense_to_sparse_coo)			\
ROCSPARSE_DO_ROUTINE(dense_to_sparse_csc)			\
ROCSPARSE_DO_ROUTINE(dense_to_sparse_csr)			\
ROCSPARSE_DO_ROUTINE(doti)					\
ROCSPARSE_DO_ROUTINE(dotci)					\
ROCSPARSE_DO_ROUTINE(ellmv)					\
ROCSPARSE_DO_ROUTINE(ell2csr)					\
ROCSPARSE_DO_ROUTINE(gebsr2csr)					\
ROCSPARSE_DO_ROUTINE(gebsr2gebsr)				\
ROCSPARSE_DO_ROUTINE(gthr)					\
ROCSPARSE_DO_ROUTINE(gthrz)					\
ROCSPARSE_DO_ROUTINE(gebsr2gebsc)				\
ROCSPARSE_DO_ROUTINE(gebsrmv)					\
ROCSPARSE_DO_ROUTINE(gebsrmm)					\
ROCSPARSE_DO_ROUTINE(gemmi)					\
ROCSPARSE_DO_ROUTINE(gemvi)					\
ROCSPARSE_DO_ROUTINE(gpsv_interleaved_batch) \
ROCSPARSE_DO_ROUTINE(gtsv)					\
ROCSPARSE_DO_ROUTINE(gtsv_no_pivot)				\
ROCSPARSE_DO_ROUTINE(gtsv_no_pivot_strided_batch)		\
ROCSPARSE_DO_ROUTINE(gtsv_interleaved_batch)		\
ROCSPARSE_DO_ROUTINE(hybmv)					\
ROCSPARSE_DO_ROUTINE(hyb2csr)					\
ROCSPARSE_DO_ROUTINE(identity)					\
ROCSPARSE_DO_ROUTINE(inverse_permutation)			\
ROCSPARSE_DO_ROUTINE(nnz)					\
ROCSPARSE_DO_ROUTINE(prune_csr2csr)				\
ROCSPARSE_DO_ROUTINE(prune_csr2csr_by_percentage)		\
ROCSPARSE_DO_ROUTINE(prune_dense2csr)				\
ROCSPARSE_DO_ROUTINE(prune_dense2csr_by_percentage)		\
ROCSPARSE_DO_ROUTINE(roti)					\
ROCSPARSE_DO_ROUTINE(sctr)					\
ROCSPARSE_DO_ROUTINE(sddmm)					\
ROCSPARSE_DO_ROUTINE(sparse_to_dense_coo)			\
ROCSPARSE_DO_ROUTINE(sparse_to_dense_csc)			\
ROCSPARSE_DO_ROUTINE(sparse_to_dense_csr)			\
ROCSPARSE_DO_ROUTINE(sparse_to_sparse)				\
ROCSPARSE_DO_ROUTINE(extract)					\
ROCSPARSE_DO_ROUTINE(v2_spmv_bsr)				\
ROCSPARSE_DO_ROUTINE(v2_spmv_csr)				\
ROCSPARSE_DO_ROUTINE(v2_spmv_csc)				\
ROCSPARSE_DO_ROUTINE(v2_spmv_coo)				\
ROCSPARSE_DO_ROUTINE(v2_spmv_coo_aos)				\
ROCSPARSE_DO_ROUTINE(v2_spmv_ell)
// clang-format on

template <std::size_t N, typename T>
static constexpr std::size_t countof(T (&)[N])
{
    return N;
}

struct rocsparse_routine
{
private:
public:
#define ROCSPARSE_DO_ROUTINE(x_) x_,
    typedef enum _ : rocsparse_int
    {
        ROCSPARSE_FOREACH_ROUTINE
    } value_type;
    value_type                  value{};
    static constexpr value_type all_routines[] = {ROCSPARSE_FOREACH_ROUTINE};
#undef ROCSPARSE_DO_ROUTINE

    static constexpr std::size_t num_routines = countof(all_routines);

private:
#define ROCSPARSE_DO_ROUTINE(x_) #x_,
    static constexpr const char* s_routine_names[num_routines]{ROCSPARSE_FOREACH_ROUTINE};
#undef ROCSPARSE_DO_ROUTINE

public:
    rocsparse_routine();
    rocsparse_routine& operator()(const char* function);
    explicit rocsparse_routine(const char* function);
    rocsparse_status
        dispatch(const char precision, const char indextype, const Arguments& arg) const;
    constexpr const char* to_string() const;

private:
    template <rocsparse_routine::value_type FNAME, typename T, typename I, typename J = I>
    static rocsparse_status dispatch_call(const Arguments& arg);

    template <rocsparse_routine::value_type FNAME, typename T>
    static rocsparse_status dispatch_indextype(const char cindextype, const Arguments& arg);

    template <rocsparse_routine::value_type FNAME>
    static rocsparse_status
        dispatch_precision(const char precision, const char indextype, const Arguments& arg);
};
