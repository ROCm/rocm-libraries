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
#include "hipsparse_arguments.hpp"
template <std::size_t N, typename T>
static constexpr std::size_t countof2(T (&)[N])
{
    return N;
}

// clang-format off
#define HIPSPARSE_FOREACH_TEST_ENUM TRANSFORM_HIPSPARSE_TEST_ENUM(axpby) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(axpyi) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsr2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsric02) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsrilu02) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsrmm) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsrmv) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsrsm2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsrsv2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(bsrxmv) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(const_dnmat_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(const_dnvec_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(const_spmat_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(const_spvec_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(coo2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(coosort) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csc2dense) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(cscsort) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2bsr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2csc_ex2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2csr_compress) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2dense) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2gebsr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csr2hyb) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrcolor) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrgeam) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrgeam2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrgemm) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrgemm2_a) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrgemm2_b) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csric02) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrilu02) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrilusv) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrmm) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrmv) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrsm2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrsort) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csrsv2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(csru2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dense_to_sparse_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dense_to_sparse_csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dense_to_sparse_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dense2csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dense2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dnmat_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dnvec_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(dotci) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(doti) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gather) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gebsr2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gebsr2gebsc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gebsr2gebsr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gemmi) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gemvi) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gpsv_interleaved_batch) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gthr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gthrz) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gtsv_interleaved_batch) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gtsv2) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gtsv2_nopivot) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(gtsv2_strided_batch) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(hyb2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(hybmv) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(identity) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(nnz) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(prune_csr2csr_by_percentage) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(prune_csr2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(prune_dense2csr_by_percentage) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(prune_dense2csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(rot) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(roti) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(scatter) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sctr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sddmm_coo_aos) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sddmm_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sddmm_csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sddmm_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sparse_to_dense_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sparse_to_dense_csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(sparse_to_dense_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spgemm_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spgemmreuse_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmat_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_batched_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_batched_csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_batched_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_bell) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_csc) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmm_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmv_coo_aos) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmv_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spmv_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spsm_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spsm_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spsv_coo) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spsv_csr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spvec_descr) \
    TRANSFORM_HIPSPARSE_TEST_ENUM(spvv) \
// clang-format on

struct hipsparse_test_enum
{
private:
public:
    /////
#define TRANSFORM_HIPSPARSE_TEST_ENUM(x_) x_,
    typedef enum _ : int32_t
    {
        HIPSPARSE_FOREACH_TEST_ENUM
    } value_type;
    static constexpr value_type all_test_enum[] = {HIPSPARSE_FOREACH_TEST_ENUM};
#undef TRANSFORM_HIPSPARSE_TEST_ENUM
    /////
    static constexpr std::size_t num_test_enum = countof2(all_test_enum);
    value_type                   value{};

private:
    /////
#define TRANSFORM_HIPSPARSE_TEST_ENUM(x_) #x_,
    static constexpr const char* s_test_enum_names[num_test_enum]{HIPSPARSE_FOREACH_TEST_ENUM};
#undef TRANSFORM_HIPSPARSE_TEST_ENUM
    /////

public:
    static inline const char* to_string(hipsparse_test_enum::value_type value)
    {
        return s_test_enum_names[value];
    }
};