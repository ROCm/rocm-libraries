/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
template <std::size_t N, typename T>
static constexpr std::size_t countof2(T (&)[N])
{
    return N;
}

// clang-format off
#define ROCSPARSE_FOREACH_TEST_ENUM		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(axpby)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(axpyi)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsr2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrgeam)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrgemm)			    \
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsric0)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrilu0)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrmm)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrmv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrpad_value)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrsm)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrsv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(bsrxmv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_coo)		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_csc)	    \
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_csr)	    \
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_ell)	    \
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_gebsc)		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_gebsr)		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_matrix_hyb)	    \
  TRANSFORM_ROCSPARSE_TEST_ENUM(check_spmat)	        \
  TRANSFORM_ROCSPARSE_TEST_ENUM(const_dnmat_descr)      \
  TRANSFORM_ROCSPARSE_TEST_ENUM(const_dnvec_descr)      \
  TRANSFORM_ROCSPARSE_TEST_ENUM(const_spmat_descr)      \
  TRANSFORM_ROCSPARSE_TEST_ENUM(const_spvec_descr)      \
  TRANSFORM_ROCSPARSE_TEST_ENUM(coo2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(coo2dense)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(coomv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(coosort)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(copy_info)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csc2dense)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(cscsort)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2bsr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2csc)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2csr_compress)		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2dense)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2ell)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2gebsr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csr2hyb)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrcolor)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrgeam)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrgemm)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrgemm_reuse)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csric0)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csricsv)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csritilu0)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csritilu0_ex)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrsldu)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrilu0)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrilusv)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrmm)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrmv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrmv_managed)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrsm)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrsort)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csrsv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(csritsv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dense2coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dense2csc)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dense2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dense_to_sparse_coo)	\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dense_to_sparse_csc)	\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dense_to_sparse_csr)	\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dnmat_descr)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dnvec_descr)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(dotci)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(doti)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(ell2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(ellmv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gather)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gebsr2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gebsr2gebsc)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gebsr2gebsr)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gebsrmm)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gebsrmv)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gemmi)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gemvi)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gpsv_interleaved_batch)	\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gthr)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gthrz)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gtsv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gtsv_no_pivot)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(gtsv_no_pivot_strided_batch) \
  TRANSFORM_ROCSPARSE_TEST_ENUM(gtsv_interleaved_batch)	\
  TRANSFORM_ROCSPARSE_TEST_ENUM(hyb2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(hybmv)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(identity)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(inverse_permutation)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(nnz)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(prune_csr2csr_by_percentage)		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(prune_csr2csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(prune_dense2csr_by_percentage)		\
  TRANSFORM_ROCSPARSE_TEST_ENUM(prune_dense2csr)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(rot)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(roti)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(scatter)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sctr)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sddmm)					\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sparse_to_dense_coo)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sparse_to_dense_csc)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sparse_to_dense_csr)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sparse_to_sparse)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(extract)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgemm_bsr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgemm_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgemm_reuse_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgeam_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgeam_reuse_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgeam_csr_2)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spgeam_reuse_csr_2)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmat_descr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_bell)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_bsr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_csc)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_batched_bell)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_batched_coo)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_batched_csc)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmm_batched_csr)			\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmv_bsr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmv_coo_aos)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmv_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmv_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmv_csc)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spmv_ell)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(v2_spmv_bsr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(v2_spmv_coo_aos)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(v2_spmv_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(v2_spmv_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(v2_spmv_csc)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(v2_spmv_ell)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spsm_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spsm_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spsv_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spsv_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sptrsm_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sptrsm_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sptrsv_coo)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(sptrsv_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spitsv_csr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spvec_descr)				\
  TRANSFORM_ROCSPARSE_TEST_ENUM(spvv)
// clang-format on

struct rocsparse_test_enum
{
private:
public:
    /////
#define TRANSFORM_ROCSPARSE_TEST_ENUM(x_) x_,
    typedef enum _ : int32_t
    {
        ROCSPARSE_FOREACH_TEST_ENUM
    } value_type;
    static constexpr value_type all_test_enum[] = {ROCSPARSE_FOREACH_TEST_ENUM};
#undef TRANSFORM_ROCSPARSE_TEST_ENUM
    /////
    static constexpr std::size_t num_test_enum = countof2(all_test_enum);
    value_type                   value{};

private:
    /////
#define TRANSFORM_ROCSPARSE_TEST_ENUM(x_) #x_,
    static constexpr const char* s_test_enum_names[num_test_enum]{ROCSPARSE_FOREACH_TEST_ENUM};
#undef TRANSFORM_ROCSPARSE_TEST_ENUM
    /////

public:
    static inline const char* to_string(rocsparse_test_enum::value_type value)
    {
        return s_test_enum_names[value];
    }
};
