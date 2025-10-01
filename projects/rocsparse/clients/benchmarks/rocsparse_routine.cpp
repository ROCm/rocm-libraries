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
#include "rocsparse_routine.hpp"
#include "rocsparse.h"

//
//
//
rocsparse_routine::rocsparse_routine(const char* function)
{
    for(auto routine : all_routines)
    {
        const char* str = s_routine_names[routine];
        if(!strcmp(function, str))
        {
            this->value = routine;
            return;
        }
    }

    std::cerr << "// function " << function << " is invalid, list of valid function is"
              << std::endl;
    for(auto routine : all_routines)
    {
        const char* str = s_routine_names[routine];
        std::cerr << "//    - " << str << std::endl;
    }

    throw rocsparse_status_invalid_value;
}

//
//
//
rocsparse_routine::rocsparse_routine()
    : value((value_type)-1){};

//
//
//
rocsparse_routine& rocsparse_routine::operator()(const char* function)
{
    for(auto routine : all_routines)
    {
        const char* str = s_routine_names[routine];
        if(!strcmp(function, str))
        {
            this->value = routine;
            return *this;
        }
    }

    std::cerr << "// function " << function << " is invalid, list of valid function is"
              << std::endl;
    for(auto routine : all_routines)
    {
        const char* str = s_routine_names[routine];
        std::cerr << "//    - " << str << std::endl;
    }

    throw rocsparse_status_invalid_value;
}

//
//
//
constexpr rocsparse_routine::value_type rocsparse_routine::all_routines[];

template <rocsparse_routine::value_type FNAME, typename T>
rocsparse_status rocsparse_routine::dispatch_indextype(const char cindextype, const Arguments& arg)
{
    const rocsparse_indextype indextype = (cindextype == 'm')   ? rocsparse_indextype_i64
                                          : (cindextype == 's') ? rocsparse_indextype_i32
                                          : (cindextype == 'd') ? rocsparse_indextype_i64
                                                                : ((rocsparse_indextype)-1);
    const bool                mixed     = (cindextype == 'm');
    switch(indextype)
    {
    case rocsparse_indextype_u16:
    {
        break;
    }
    case rocsparse_indextype_i32:
    {
        return dispatch_call<FNAME, T, int32_t>(arg);
    }
    case rocsparse_indextype_i64:
    {
        if(mixed)
        {
            return dispatch_call<FNAME, T, int64_t, int32_t>(arg);
        }
        else
        {
            return dispatch_call<FNAME, T, int64_t>(arg);
        }
    }
    }
    return rocsparse_status_invalid_value;
}

//
//
//
template <rocsparse_routine::value_type FNAME>
rocsparse_status rocsparse_routine::dispatch_precision(const char       precision,
                                                       const char       indextype,
                                                       const Arguments& arg)
{
    const rocsparse_datatype datatype = (precision == 's')   ? rocsparse_datatype_f32_r
                                        : (precision == 'd') ? rocsparse_datatype_f64_r
                                        : (precision == 'c') ? rocsparse_datatype_f32_c
                                        : (precision == 'z') ? rocsparse_datatype_f64_c
                                                             : ((rocsparse_datatype)-1);
    switch(datatype)
    {
    case rocsparse_datatype_f32_r:
        return dispatch_indextype<FNAME, float>(indextype, arg);
    case rocsparse_datatype_f64_r:
        return dispatch_indextype<FNAME, double>(indextype, arg);
    case rocsparse_datatype_f32_c:
        return dispatch_indextype<FNAME, rocsparse_float_complex>(indextype, arg);
    case rocsparse_datatype_f64_c:
        return dispatch_indextype<FNAME, rocsparse_double_complex>(indextype, arg);
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_bf16_r:
        return rocsparse_status_invalid_value;
    }
    return rocsparse_status_invalid_value;
}

//
//
//
rocsparse_status rocsparse_routine::dispatch(const char       precision,
                                             const char       indextype,
                                             const Arguments& arg) const
{
    switch(this->value)
    {
#define ROCSPARSE_DO_ROUTINE(FNAME) \
    case FNAME:                     \
        return dispatch_precision<FNAME>(precision, indextype, arg);
        ROCSPARSE_FOREACH_ROUTINE;
#undef ROCSPARSE_DO_ROUTINE
    }
    return rocsparse_status_invalid_value;
}

//
//
//
constexpr const char* rocsparse_routine::to_string() const
{
    //
    // switch for checking inconsistency.
    //
    switch(this->value)
    {
#define ROCSPARSE_DO_ROUTINE(x_)                      \
    case x_:                                          \
    {                                                 \
        if(strcmp(#x_, s_routine_names[this->value])) \
            return nullptr;                           \
        break;                                        \
    }

        ROCSPARSE_FOREACH_ROUTINE;
    }

#undef ROCSPARSE_DO_ROUTINE
    return s_routine_names[this->value];
}

// Level1
#include "testing_axpyi.hpp"
#include "testing_dotci.hpp"
#include "testing_doti.hpp"
#include "testing_gthr.hpp"
#include "testing_gthrz.hpp"
#include "testing_roti.hpp"
#include "testing_sctr.hpp"

// Level2
#include "testing_bsrmv.hpp"
#include "testing_bsrsv.hpp"
#include "testing_bsrxmv.hpp"
#include "testing_csritsv.hpp"
#include "testing_csrmv_managed.hpp"
#include "testing_csrsv.hpp"
#include "testing_gebsrmv.hpp"
#include "testing_gemvi.hpp"
#include "testing_hybmv.hpp"
#include "testing_spitsv_csr.hpp"
#include "testing_spmv_bsr.hpp"
#include "testing_spmv_coo.hpp"
#include "testing_spmv_coo_aos.hpp"
#include "testing_spmv_csc.hpp"
#include "testing_spmv_csr.hpp"
#include "testing_spmv_ell.hpp"
#include "testing_spsv_coo.hpp"
#include "testing_spsv_csr.hpp"
#include "testing_v2_spmv_bsr.hpp"
#include "testing_v2_spmv_coo.hpp"
#include "testing_v2_spmv_coo_aos.hpp"
#include "testing_v2_spmv_csc.hpp"
#include "testing_v2_spmv_csr.hpp"
#include "testing_v2_spmv_ell.hpp"

// Level3
#include "testing_bsrmm.hpp"
#include "testing_bsrsm.hpp"
#include "testing_csrmm.hpp"
#include "testing_csrsm.hpp"
#include "testing_gebsrmm.hpp"
#include "testing_gemmi.hpp"
#include "testing_sddmm.hpp"
#include "testing_spmm_batched_bell.hpp"
#include "testing_spmm_batched_coo.hpp"
#include "testing_spmm_batched_csc.hpp"
#include "testing_spmm_batched_csr.hpp"
#include "testing_spmm_bell.hpp"
#include "testing_spmm_bsr.hpp"
#include "testing_spmm_coo.hpp"
#include "testing_spmm_csc.hpp"
#include "testing_spmm_csr.hpp"
#include "testing_spsm_coo.hpp"
#include "testing_spsm_csr.hpp"

// Extra
#include "testing_bsrgeam.hpp"
#include "testing_bsrgemm.hpp"
#include "testing_csrgeam.hpp"
#include "testing_csrgemm.hpp"
#include "testing_csrgemm_reuse.hpp"
#include "testing_spgeam_csr.hpp"
#include "testing_spgeam_csr_2.hpp"
#include "testing_spgeam_reuse_csr.hpp"
#include "testing_spgeam_reuse_csr_2.hpp"
#include "testing_spgemm_bsr.hpp"
#include "testing_spgemm_csr.hpp"
#include "testing_spgemm_reuse_csr.hpp"

// Preconditioner
#include "testing_bsric0.hpp"
#include "testing_bsrilu0.hpp"
#include "testing_csric0.hpp"
#include "testing_csrilu0.hpp"
#include "testing_csritilu0.hpp"
#include "testing_csritilu0_ex.hpp"
#include "testing_gpsv_interleaved_batch.hpp"
#include "testing_gtsv.hpp"
#include "testing_gtsv_interleaved_batch.hpp"
#include "testing_gtsv_no_pivot.hpp"
#include "testing_gtsv_no_pivot_strided_batch.hpp"

// Conversion
#include "testing_bsr2csr.hpp"
#include "testing_coo2csr.hpp"
#include "testing_coo2dense.hpp"
#include "testing_coosort.hpp"
#include "testing_csc2dense.hpp"
#include "testing_cscsort.hpp"
#include "testing_csr2bsr.hpp"
#include "testing_csr2coo.hpp"
#include "testing_csr2csc.hpp"
#include "testing_csr2csr_compress.hpp"
#include "testing_csr2dense.hpp"
#include "testing_csr2ell.hpp"
#include "testing_csr2gebsr.hpp"
#include "testing_csr2hyb.hpp"
#include "testing_csrsort.hpp"
#include "testing_dense2coo.hpp"
#include "testing_dense2csc.hpp"
#include "testing_dense2csr.hpp"
#include "testing_dense_to_sparse_coo.hpp"
#include "testing_dense_to_sparse_csc.hpp"
#include "testing_dense_to_sparse_csr.hpp"
#include "testing_ell2csr.hpp"
#include "testing_extract.hpp"
#include "testing_gebsr2csr.hpp"
#include "testing_gebsr2gebsc.hpp"
#include "testing_gebsr2gebsr.hpp"
#include "testing_hyb2csr.hpp"
#include "testing_identity.hpp"
#include "testing_inverse_permutation.hpp"
#include "testing_nnz.hpp"
#include "testing_prune_csr2csr.hpp"
#include "testing_prune_csr2csr_by_percentage.hpp"
#include "testing_prune_dense2csr.hpp"
#include "testing_prune_dense2csr_by_percentage.hpp"
#include "testing_sparse_to_dense_coo.hpp"
#include "testing_sparse_to_dense_csc.hpp"
#include "testing_sparse_to_dense_csr.hpp"
#include "testing_sparse_to_sparse.hpp"

// Reordering
#include "testing_csrcolor.hpp"

// Util
#include "testing_check_matrix_coo.hpp"
#include "testing_check_matrix_csc.hpp"
#include "testing_check_matrix_csr.hpp"
#include "testing_check_matrix_ell.hpp"
#include "testing_check_matrix_gebsc.hpp"
#include "testing_check_matrix_gebsr.hpp"
#include "testing_check_matrix_hyb.hpp"

template <rocsparse_routine::value_type FNAME, typename T, typename I, typename J>
rocsparse_status rocsparse_routine::dispatch_call(const Arguments& arg)
{
#define DEFINE_CASE_IT_X(value, testingf)     \
    case value:                               \
    {                                         \
        try                                   \
        {                                     \
            testingf<I, T>(arg);              \
            return rocsparse_status_success;  \
        }                                     \
        catch(const rocsparse_status& status) \
        {                                     \
            return status;                    \
        }                                     \
    }

#define DEFINE_CASE_IJT_X(value, testingf)    \
    case value:                               \
    {                                         \
        try                                   \
        {                                     \
            testingf<I, J, T>(arg);           \
            return rocsparse_status_success;  \
        }                                     \
        catch(const rocsparse_status& status) \
        {                                     \
            return status;                    \
        }                                     \
    }

#define DEFINE_CASE_IXYT_X(value, testingf)   \
    case value:                               \
    {                                         \
        try                                   \
        {                                     \
            testingf<I, T, T, T>(arg);        \
            return rocsparse_status_success;  \
        }                                     \
        catch(const rocsparse_status& status) \
        {                                     \
            return status;                    \
        }                                     \
    }

#define DEFINE_CASE_IAXYT_X(value, testingf)  \
    case value:                               \
    {                                         \
        try                                   \
        {                                     \
            testingf<I, T, T, T, T>(arg);     \
            return rocsparse_status_success;  \
        }                                     \
        catch(const rocsparse_status& status) \
        {                                     \
            return status;                    \
        }                                     \
    }
#define DEFINE_CASE_IABCT_X(value, testingf) DEFINE_CASE_IAXYT_X(value, testingf)

#define DEFINE_CASE_IJAXYT_X(value, testingf) \
    case value:                               \
    {                                         \
        try                                   \
        {                                     \
            testingf<I, J, T, T, T, T>(arg);  \
            return rocsparse_status_success;  \
        }                                     \
        catch(const rocsparse_status& status) \
        {                                     \
            return status;                    \
        }                                     \
    }
#define DEFINE_CASE_IJABCT_X(value, testingf) DEFINE_CASE_IJAXYT_X(value, testingf)

#define DEFINE_CASE_IT(value) DEFINE_CASE_IT_X(value, testing_##value)
#define DEFINE_CASE_IJT(value) DEFINE_CASE_IJT_X(value, testing_##value)
#define DEFINE_CASE_IXYT(value) DEFINE_CASE_IXYT_X(value, testing_##value)
#define DEFINE_CASE_IAXYT(value) DEFINE_CASE_IAXYT_X(value, testing_##value)
#define DEFINE_CASE_IJAXYT(value) DEFINE_CASE_IJAXYT_X(value, testing_##value)
#define DEFINE_CASE_IABCT(value) DEFINE_CASE_IABCT_X(value, testing_##value)
#define DEFINE_CASE_IJABCT(value) DEFINE_CASE_IJABCT_X(value, testing_##value)
#define IS_T_REAL (std::is_same<T, double>() || std::is_same<T, float>())
#define IS_T_COMPLEX \
    (std::is_same<T, rocsparse_double_complex>() || std::is_same<T, rocsparse_float_complex>())

#define DEFINE_CASE_T_REAL_ONLY(value)               \
    case value:                                      \
    {                                                \
        if(IS_T_REAL)                                \
        {                                            \
            try                                      \
            {                                        \
                testing_##value<T>(arg);             \
                return rocsparse_status_success;     \
            }                                        \
            catch(const rocsparse_status& status)    \
            {                                        \
                return status;                       \
            }                                        \
        }                                            \
        else                                         \
        {                                            \
            return rocsparse_status_not_implemented; \
        }                                            \
    }

#define DEFINE_CASE_T_FLOAT_ONLY(value)              \
    case value:                                      \
    {                                                \
        if(std::is_same<T, float>())                 \
        {                                            \
            try                                      \
            {                                        \
                testing_##value<T>(arg);             \
                return rocsparse_status_success;     \
            }                                        \
            catch(const rocsparse_status& status)    \
            {                                        \
                return status;                       \
            }                                        \
        }                                            \
        else                                         \
        {                                            \
            return rocsparse_status_not_implemented; \
        }                                            \
    }

#define DEFINE_CASE_T_X(value, testingf)      \
    case value:                               \
    {                                         \
        try                                   \
        {                                     \
            testingf<T>(arg);                 \
            return rocsparse_status_success;  \
        }                                     \
        catch(const rocsparse_status& status) \
        {                                     \
            return status;                    \
        }                                     \
    }

#define DEFINE_CASE_T(value) DEFINE_CASE_T_X(value, testing_##value)

#define DEFINE_CASE_T_REAL_VS_COMPLEX(value, rtestingf, ctestingf) \
    case value:                                                    \
    {                                                              \
        try                                                        \
        {                                                          \
            if(IS_T_REAL)                                          \
            {                                                      \
                rtestingf<T>(arg);                                 \
            }                                                      \
            else if(IS_T_COMPLEX)                                  \
            {                                                      \
                ctestingf<T>(arg);                                 \
            }                                                      \
            else                                                   \
            {                                                      \
                return rocsparse_status_internal_error;            \
            }                                                      \
        }                                                          \
        catch(const rocsparse_status& status)                      \
        {                                                          \
            return status;                                         \
        }                                                          \
    }

    switch(FNAME)
    {
        DEFINE_CASE_T(axpyi);
        DEFINE_CASE_IABCT_X(bellmm, testing_spmm_bell);
        DEFINE_CASE_IT_X(bellmm_batched, testing_spmm_batched_bell);
        DEFINE_CASE_T(bsrgeam);
        DEFINE_CASE_T(bsric0);
        DEFINE_CASE_T(bsrilu0);
        DEFINE_CASE_T(bsrsm);
        DEFINE_CASE_T(bsrsv);
        DEFINE_CASE_T(bsrxmv);
        DEFINE_CASE_T(bsr2csr);
        DEFINE_CASE_T(check_matrix_csr);
        DEFINE_CASE_T(check_matrix_csc);
        DEFINE_CASE_T(check_matrix_coo);
        DEFINE_CASE_T(check_matrix_gebsr);
        DEFINE_CASE_T(check_matrix_gebsc);
        DEFINE_CASE_T(check_matrix_ell);
        DEFINE_CASE_T(check_matrix_hyb);
        DEFINE_CASE_IABCT_X(coomm, testing_spmm_coo);
        DEFINE_CASE_IABCT_X(coomm_batched, testing_spmm_batched_coo);
        DEFINE_CASE_IAXYT_X(coomv, testing_spmv_coo);
        DEFINE_CASE_T_FLOAT_ONLY(coosort);
        DEFINE_CASE_IT_X(coosv, testing_spsv_coo);
        DEFINE_CASE_IAXYT_X(coomv_aos, testing_spmv_coo_aos);
        DEFINE_CASE_IT_X(coosm, testing_spsm_coo);
        DEFINE_CASE_T_FLOAT_ONLY(coo2csr);
        DEFINE_CASE_T(coo2dense);
        DEFINE_CASE_T_FLOAT_ONLY(cscsort);
        DEFINE_CASE_T(csc2dense);
        DEFINE_CASE_T(csrcolor);
        DEFINE_CASE_T(csric0);
        DEFINE_CASE_T(csrilu0);
        DEFINE_CASE_T(csritilu0);
        DEFINE_CASE_T(csritilu0_ex);
        DEFINE_CASE_IJT_X(csrgeam, testing_spgeam_csr);
        DEFINE_CASE_IJT_X(bsrgemm, testing_spgemm_bsr);
        DEFINE_CASE_IJT_X(csrgemm, testing_spgemm_csr);
        DEFINE_CASE_IJT_X(csrgemm_reuse, testing_spgemm_reuse_csr);
        DEFINE_CASE_IJAXYT_X(bsrmv, testing_spmv_bsr);
        DEFINE_CASE_IJAXYT_X(csrmv, testing_spmv_csr);
        DEFINE_CASE_IJAXYT_X(csrmv_analysis, testing_spmv_csr_analysis);
        DEFINE_CASE_T(csrmv_managed);
        DEFINE_CASE_IJABCT_X(bsrmm, testing_spmm_bsr);
        DEFINE_CASE_IJAXYT_X(cscmv, testing_spmv_csc);
        DEFINE_CASE_IJABCT_X(csrmm, testing_spmm_csr);
        DEFINE_CASE_IJABCT_X(csrmm_batched, testing_spmm_batched_csr);
        DEFINE_CASE_IJABCT_X(cscmm, testing_spmm_csc);
        DEFINE_CASE_IJABCT_X(cscmm_batched, testing_spmm_batched_csc);
        DEFINE_CASE_IJT_X(csrsm, testing_spsm_csr);
        DEFINE_CASE_T_FLOAT_ONLY(csrsort);
        DEFINE_CASE_IJT_X(csrsv, testing_spsv_csr);
        DEFINE_CASE_IJT_X(spitsv_csr, testing_spitsv_csr);
        DEFINE_CASE_T(csritsv);
        DEFINE_CASE_T(csr2dense);
        DEFINE_CASE_T(csr2bsr);
        DEFINE_CASE_T_FLOAT_ONLY(csr2coo);
        DEFINE_CASE_T(csr2csc);
        DEFINE_CASE_T(csr2csr_compress);
        DEFINE_CASE_T(csr2ell);
        DEFINE_CASE_T(csr2gebsr);
        DEFINE_CASE_T(csr2hyb);
        DEFINE_CASE_T(dense2coo);
        DEFINE_CASE_T(dense2csc);
        DEFINE_CASE_T(dense2csr);
        DEFINE_CASE_IT(dense_to_sparse_coo);
        DEFINE_CASE_IJT(dense_to_sparse_csc);
        DEFINE_CASE_IJT(dense_to_sparse_csr);
        DEFINE_CASE_T(doti);
        DEFINE_CASE_T_REAL_VS_COMPLEX(dotci, testing_doti, testing_dotci);
        DEFINE_CASE_IAXYT_X(ellmv, testing_spmv_ell);
        DEFINE_CASE_T(ell2csr);
        DEFINE_CASE_T(gebsr2csr);
        DEFINE_CASE_T(gebsr2gebsr);
        DEFINE_CASE_T(gthr);
        DEFINE_CASE_T(gthrz);
        DEFINE_CASE_T(gebsr2gebsc);
        DEFINE_CASE_T(gebsrmv);
        DEFINE_CASE_T(gebsrmm);
        DEFINE_CASE_T(gemmi);
        DEFINE_CASE_T(gemvi);
        DEFINE_CASE_T(gtsv);
        DEFINE_CASE_T(gtsv_no_pivot);
        DEFINE_CASE_T(gtsv_no_pivot_strided_batch);
        DEFINE_CASE_T(gtsv_interleaved_batch);
        DEFINE_CASE_T(gpsv_interleaved_batch);
        DEFINE_CASE_T(hybmv);
        DEFINE_CASE_T(hyb2csr);
        DEFINE_CASE_T_FLOAT_ONLY(identity);
        DEFINE_CASE_T_FLOAT_ONLY(inverse_permutation);
        DEFINE_CASE_T(nnz);
        DEFINE_CASE_T_REAL_ONLY(prune_csr2csr);
        DEFINE_CASE_T_REAL_ONLY(prune_csr2csr_by_percentage);
        DEFINE_CASE_T_REAL_ONLY(prune_dense2csr);
        DEFINE_CASE_T_REAL_ONLY(prune_dense2csr_by_percentage);
        DEFINE_CASE_T_REAL_ONLY(roti);
        DEFINE_CASE_T(sctr);
        DEFINE_CASE_IJABCT(sddmm);
        DEFINE_CASE_IT(sparse_to_dense_coo);
        DEFINE_CASE_IJT(sparse_to_dense_csc);
        DEFINE_CASE_IJT(sparse_to_dense_csr);
        DEFINE_CASE_IJT(sparse_to_sparse);
        DEFINE_CASE_IJT(extract);
        DEFINE_CASE_IJAXYT_X(v2_spmv_bsr, testing_v2_spmv_bsr);
        DEFINE_CASE_IJAXYT_X(v2_spmv_csr, testing_v2_spmv_csr);
        DEFINE_CASE_IJAXYT_X(v2_spmv_csc, testing_v2_spmv_csc);
        DEFINE_CASE_IAXYT_X(v2_spmv_coo, testing_v2_spmv_coo);
        DEFINE_CASE_IAXYT_X(v2_spmv_coo_aos, testing_v2_spmv_coo_aos);
        DEFINE_CASE_IAXYT_X(v2_spmv_ell, testing_v2_spmv_ell);
    }

#undef DEFINE_CASE_IT_X
#undef DEFINE_CASE_IJT_X
#undef DEFINE_CASE_T_REAL_ONLY
#undef DEFINE_CASE_T_FLOAT_ONLY
#undef DEFINE_CASE_T_X
#undef DEFINE_CASE_T
#undef DEFINE_CASE_T_REAL_VS_COMPLEX
#undef IS_T_REAL
#undef IS_T_COMPLEX

    return rocsparse_status_invalid_value;
}
