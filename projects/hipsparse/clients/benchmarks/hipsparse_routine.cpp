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
#include "hipsparse_routine.hpp"

constexpr const char* hipsparse_routine::s_routine_names[hipsparse_routine::num_routines];

hipsparse_routine::hipsparse_routine(const char* function)
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

    throw HIPSPARSE_STATUS_INVALID_VALUE;
}

hipsparse_routine::hipsparse_routine()
    : value((value_type)-1){};

hipsparse_routine& hipsparse_routine::operator()(const char* function)
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

    throw HIPSPARSE_STATUS_INVALID_VALUE;
}

constexpr hipsparse_routine::value_type hipsparse_routine::all_routines[];

template <hipsparse_routine::value_type FNAME, typename T>
hipsparseStatus_t hipsparse_routine::dispatch_indextype(const char cindextype, const Arguments& arg)
{
    const hipsparseIndexType_t indextype = (cindextype == 'm')   ? HIPSPARSE_INDEX_64I
                                           : (cindextype == 's') ? HIPSPARSE_INDEX_32I
                                           : (cindextype == 'd') ? HIPSPARSE_INDEX_64I
                                                                 : ((hipsparseIndexType_t)-1);
    const bool                 mixed     = (cindextype == 'm');

    switch(indextype)
    {
    case HIPSPARSE_INDEX_16U:
    {
        break;
    }
    case HIPSPARSE_INDEX_32I:
    {
        return dispatch_call<FNAME, T, int32_t>(arg);
    }
    case HIPSPARSE_INDEX_64I:
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
    return HIPSPARSE_STATUS_INVALID_VALUE;
}

template <hipsparse_routine::value_type FNAME>
hipsparseStatus_t hipsparse_routine::dispatch_precision(const char       precision,
                                                        const char       indextype,
                                                        const Arguments& arg)
{
    const hipDataType datatype = (precision == 's')   ? HIP_R_32F
                                 : (precision == 'd') ? HIP_R_64F
                                 : (precision == 'c') ? HIP_C_32F
                                 : (precision == 'z') ? HIP_C_64F
                                                      : ((hipDataType)-1);
    switch(datatype)
    {
    case HIP_R_32F:
        return dispatch_indextype<FNAME, float>(indextype, arg);
    case HIP_R_64F:
        return dispatch_indextype<FNAME, double>(indextype, arg);
    case HIP_C_32F:
        return dispatch_indextype<FNAME, hipComplex>(indextype, arg);
    case HIP_C_64F:
        return dispatch_indextype<FNAME, hipDoubleComplex>(indextype, arg);
    default:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
}

hipsparseStatus_t hipsparse_routine::dispatch(const char       precision,
                                              const char       indextype,
                                              const Arguments& arg) const
{
    switch(this->value)
    {
#define HIPSPARSE_DO_ROUTINE(FNAME) \
    case FNAME:                     \
        return dispatch_precision<FNAME>(precision, indextype, arg);

        HIPSPARSE_FOREACH_ROUTINE;
#undef HIPSPARSE_DO_ROUTINE
    }
    return HIPSPARSE_STATUS_INVALID_VALUE;
}

constexpr const char* hipsparse_routine::to_string() const
{
    // switch for checking inconsistency.
    switch(this->value)
    {
#define HIPSPARSE_DO_ROUTINE(x_)                      \
    case x_:                                          \
    {                                                 \
        if(strcmp(#x_, s_routine_names[this->value])) \
            return nullptr;                           \
        break;                                        \
    }

        HIPSPARSE_FOREACH_ROUTINE;
    }

#undef HIPSPARSE_DO_ROUTINE
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
#include "testing_bsrsv2.hpp"
#include "testing_csrsv2.hpp"
#include "testing_gemvi.hpp"
#include "testing_hybmv.hpp"

// Level3
#include "testing_bsrmm.hpp"
#include "testing_bsrsm2.hpp"
#include "testing_gemmi.hpp"

// Extra
#include "testing_csrgeam.hpp"
#include "testing_csrgemm.hpp"

// Precond
#include "testing_bsric02.hpp"
#include "testing_bsrilu02.hpp"
#include "testing_csric02.hpp"
#include "testing_csrilu02.hpp"
#include "testing_gpsv_interleaved_batch.hpp"
#include "testing_gtsv.hpp" // File should be renamed to testing_gtsv2.hpp
#include "testing_gtsv2_nopivot.hpp"
#include "testing_gtsv2_strided_batch.hpp"
#include "testing_gtsv_interleaved_batch.hpp"

// Conversion
#include "testing_bsr2csr.hpp"
#include "testing_coo2csr.hpp"
#include "testing_csr2bsr.hpp"
#include "testing_csr2coo.hpp"
#include "testing_csr2csc.hpp"
#include "testing_csr2csr_compress.hpp"
#include "testing_csr2gebsr.hpp"
#include "testing_csr2hyb.hpp"
#include "testing_gebsr2csr.hpp"
#include "testing_gebsr2gebsc.hpp"
#include "testing_gebsr2gebsr.hpp"
#include "testing_hyb2csr.hpp"

// Generic
#include "testing_dense_to_sparse_coo.hpp"
#include "testing_dense_to_sparse_csc.hpp"
#include "testing_dense_to_sparse_csr.hpp"
#include "testing_sparse_to_dense_coo.hpp"
#include "testing_sparse_to_dense_csc.hpp"
#include "testing_sparse_to_dense_csr.hpp"
#include "testing_spmm_coo.hpp"
#include "testing_spmm_csc.hpp"
#include "testing_spmm_csr.hpp"
#include "testing_spmv_coo.hpp"
#include "testing_spmv_csr.hpp"
#include "testing_spsm_coo.hpp"
#include "testing_spsm_csr.hpp"
#include "testing_spsv_csr.hpp"

bool hipsparse_routine::is_routine_supported(hipsparse_routine::value_type FNAME)
{
    switch(FNAME)
    {
    // Level 1
    case axpyi:
        return routine_support::is_axpyi_supported();
    case doti:
        return routine_support::is_doti_supported();
    case dotci:
        return routine_support::is_dotci_supported();
    case gthr:
        return routine_support::is_gthr_supported();
    case gthrz:
        return routine_support::is_gthrz_supported();
    case roti:
        return routine_support::is_roti_supported();
    case sctr:
        return routine_support::is_sctr_supported();
    // Level 2
    case bsrsv2:
        return routine_support::is_bsrsv2_supported();
    case coomv:
        return routine_support::is_coomv_supported();
    case csrmv:
        return routine_support::is_csrmv_supported();
    case csrsv:
        return routine_support::is_csrsv_supported();
    case gemvi:
        return routine_support::is_gemvi_supported();
    case hybmv:
        return routine_support::is_hybmv_supported();
    // Level 3
    case bsrmm:
        return routine_support::is_bsrmm_supported();
    case bsrsm2:
        return routine_support::is_bsrsm2_supported();
    case coomm:
        return routine_support::is_coomm_supported();
    case cscmm:
        return routine_support::is_cscmm_supported();
    case csrmm:
        return routine_support::is_csrmm_supported();
    case coosm:
        return routine_support::is_coosm_supported();
    case csrsm:
        return routine_support::is_csrsm_supported();
    case gemmi:
        return routine_support::is_gemmi_supported();
    // Extra
    case csrgeam:
        return routine_support::is_csrgeam_supported();
    case csrgemm:
        return routine_support::is_csrgemm_supported();
    // Precond
    case bsric02:
        return routine_support::is_bsric02_supported();
    case bsrilu02:
        return routine_support::is_bsrilu02_supported();
    case csric02:
        return routine_support::is_csric02_supported();
    case csrilu02:
        return routine_support::is_csrilu02_supported();
    case gtsv2:
        return routine_support::is_gtsv2_supported();
    case gtsv2_nopivot:
        return routine_support::is_gtsv2_nopivot_supported();
    case gtsv2_strided_batch:
        return routine_support::is_gtsv2_strided_batch_supported();
    case gtsv_interleaved_batch:
        return routine_support::is_gtsv_interleaved_batch_supported();
    case gpsv_interleaved_batch:
        return routine_support::is_gpsv_interleaved_batch_supported();
    // Conversion
    case bsr2csr:
        return routine_support::is_bsr2csr_supported();
    case csr2coo:
        return routine_support::is_csr2coo_supported();
    case csr2csc:
        return routine_support::is_csr2csc_supported();
    case csr2hyb:
        return routine_support::is_csr2hyb_supported();
    case csr2bsr:
        return routine_support::is_csr2bsr_supported();
    case csr2gebsr:
        return routine_support::is_csr2gebsr_supported();
    case csr2csr_compress:
        return routine_support::is_csr2csr_compress_supported();
    case coo2csr:
        return routine_support::is_coo2csr_supported();
    case hyb2csr:
        return routine_support::is_hyb2csr_supported();
    case csr2dense:
        return routine_support::is_csr2dense_supported();
    case csc2dense:
        return routine_support::is_csc2dense_supported();
    case coo2dense:
        return routine_support::is_coo2dense_supported();
    case dense2csr:
        return routine_support::is_dense2csr_supported();
    case dense2csc:
        return routine_support::is_dense2csc_supported();
    case dense2coo:
        return routine_support::is_dense2coo_supported();
    case gebsr2csr:
        return routine_support::is_gebsr2csr_supported();
    case gebsr2gebsc:
        return routine_support::is_gebsr2gebsc_supported();
    case gebsr2gebsr:
        return routine_support::is_gebsr2gebsr_supported();
    }

    return false;
}

void hipsparse_routine::print_routine_support_info(hipsparse_routine::value_type FNAME)
{
    switch(FNAME)
    {
    // Level 1
    case axpyi:
        routine_support::print_axpyi_support_warning();
        break;
    case doti:
        routine_support::print_doti_support_warning();
        break;
    case dotci:
        routine_support::print_dotci_support_warning();
        break;
    case gthr:
        routine_support::print_gthr_support_warning();
        break;
    case gthrz:
        routine_support::print_gthrz_support_warning();
        break;
    case roti:
        routine_support::print_roti_support_warning();
        break;
    case sctr:
        routine_support::print_sctr_support_warning();
        break;
    // Level 2
    case bsrsv2:
        routine_support::print_bsrsv2_support_warning();
        break;
    case coomv:
        routine_support::print_coomv_support_warning();
        break;
    case csrmv:
        routine_support::print_csrmv_support_warning();
        break;
    case csrsv:
        routine_support::print_csrsv_support_warning();
        break;
    case gemvi:
        routine_support::print_gemvi_support_warning();
        break;
    case hybmv:
        routine_support::print_hybmv_support_warning();
        break;
    // Level 3
    case bsrmm:
        routine_support::print_bsrmm_support_warning();
        break;
    case bsrsm2:
        routine_support::print_bsrsm2_support_warning();
        break;
    case coomm:
        routine_support::print_coomm_support_warning();
        break;
    case cscmm:
        routine_support::print_cscmm_support_warning();
        break;
    case csrmm:
        routine_support::print_csrmm_support_warning();
        break;
    case coosm:
        routine_support::print_coosm_support_warning();
        break;
    case csrsm:
        routine_support::print_csrsm_support_warning();
        break;
    case gemmi:
        routine_support::print_gemmi_support_warning();
        break;
    // Extra
    case csrgeam:
        routine_support::print_csrgeam_support_warning();
        break;
    case csrgemm:
        routine_support::print_csrgemm_support_warning();
        break;
    // Precond
    case bsric02:
        routine_support::print_bsric02_support_warning();
        break;
    case bsrilu02:
        routine_support::print_bsrilu02_support_warning();
        break;
    case csric02:
        routine_support::print_csric02_support_warning();
        break;
    case csrilu02:
        routine_support::print_csrilu02_support_warning();
        break;
    case gtsv2:
        routine_support::print_gtsv2_support_warning();
        break;
    case gtsv2_nopivot:
        routine_support::print_gtsv2_nopivot_support_warning();
        break;
    case gtsv2_strided_batch:
        routine_support::print_gtsv2_strided_batch_support_warning();
        break;
    case gtsv_interleaved_batch:
        routine_support::print_gtsv_interleaved_batch_support_warning();
        break;
    case gpsv_interleaved_batch:
        routine_support::print_gpsv_interleaved_batch_support_warning();
        break;
    // Conversion
    case bsr2csr:
        routine_support::print_bsr2csr_support_warning();
        break;
    case csr2coo:
        routine_support::print_csr2coo_support_warning();
        break;
    case csr2csc:
        routine_support::print_csr2csc_support_warning();
        break;
    case csr2hyb:
        routine_support::print_csr2hyb_support_warning();
        break;
    case csr2bsr:
        routine_support::print_csr2bsr_support_warning();
        break;
    case csr2gebsr:
        routine_support::print_csr2gebsr_support_warning();
        break;
    case csr2csr_compress:
        routine_support::print_csr2csr_compress_support_warning();
        break;
    case coo2csr:
        routine_support::print_coo2csr_support_warning();
        break;
    case hyb2csr:
        routine_support::print_hyb2csr_support_warning();
        break;
    case csr2dense:
        routine_support::print_csr2dense_support_warning();
        break;
    case csc2dense:
        routine_support::print_csc2dense_support_warning();
        break;
    case coo2dense:
        routine_support::print_coo2dense_support_warning();
        break;
    case dense2csr:
        routine_support::print_dense2csr_support_warning();
        break;
    case dense2csc:
        routine_support::print_dense2csc_support_warning();
        break;
    case dense2coo:
        routine_support::print_dense2coo_support_warning();
        break;
    case gebsr2csr:
        routine_support::print_gebsr2csr_support_warning();
        break;
    case gebsr2gebsc:
        routine_support::print_gebsr2gebsc_support_warning();
        break;
    case gebsr2gebsr:
        routine_support::print_gebsr2gebsr_support_warning();
        break;
    }
}

template <hipsparse_routine::value_type FNAME, typename T, typename I, typename J>
hipsparseStatus_t hipsparse_routine::dispatch_call(const Arguments& arg)
{
    if(!is_routine_supported(FNAME))
    {
        print_routine_support_info(FNAME);
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

#define DEFINE_CASE_T_X(value, testingf)       \
    case value:                                \
    {                                          \
        try                                    \
        {                                      \
            testingf<T>(arg);                  \
            return HIPSPARSE_STATUS_SUCCESS;   \
        }                                      \
        catch(const hipsparseStatus_t& status) \
        {                                      \
            return status;                     \
        }                                      \
    }

#define DEFINE_CASE_IT_X(value, testingf)      \
    case value:                                \
    {                                          \
        try                                    \
        {                                      \
            testingf<I, T>(arg);               \
            return HIPSPARSE_STATUS_SUCCESS;   \
        }                                      \
        catch(const hipsparseStatus_t& status) \
        {                                      \
            return status;                     \
        }                                      \
    }

#define DEFINE_CASE_IJT_X(value, testingf)     \
    case value:                                \
    {                                          \
        try                                    \
        {                                      \
            testingf<I, J, T>(arg);            \
            return HIPSPARSE_STATUS_SUCCESS;   \
        }                                      \
        catch(const hipsparseStatus_t& status) \
        {                                      \
            return status;                     \
        }                                      \
    }

#define DEFINE_CASE_T(value) DEFINE_CASE_T_X(value, testing_##value)

#define IS_T_FLOAT (std::is_same<T, float>())
#define IS_T_DOUBLE (std::is_same<T, double>())
#define IS_T_COMPLEX_FLOAT (std::is_same<T, hipComplex>())
#define IS_T_COMPLEX_DOUBLE (std::is_same<T, hipDoubleComplex>())

#define DEFINE_CASE_T_REAL_ONLY(value)              \
    case value:                                     \
    {                                               \
        if(IS_T_FLOAT)                              \
        {                                           \
            try                                     \
            {                                       \
                testing_##value<float>(arg);        \
                return HIPSPARSE_STATUS_SUCCESS;    \
            }                                       \
            catch(const hipsparseStatus_t& status)  \
            {                                       \
                return status;                      \
            }                                       \
        }                                           \
        else if(IS_T_DOUBLE)                        \
        {                                           \
            try                                     \
            {                                       \
                testing_##value<double>(arg);       \
                return HIPSPARSE_STATUS_SUCCESS;    \
            }                                       \
            catch(const hipsparseStatus_t& status)  \
            {                                       \
                return status;                      \
            }                                       \
        }                                           \
        else                                        \
        {                                           \
            return HIPSPARSE_STATUS_INTERNAL_ERROR; \
        }                                           \
    }

#define DEFINE_CASE_T_REAL_VS_COMPLEX(value, rtestingf, ctestingf) \
    case value:                                                    \
    {                                                              \
        try                                                        \
        {                                                          \
            if(IS_T_FLOAT)                                         \
            {                                                      \
                rtestingf<float>(arg);                             \
            }                                                      \
            else if(IS_T_DOUBLE)                                   \
            {                                                      \
                rtestingf<double>(arg);                            \
            }                                                      \
            else if(IS_T_COMPLEX_FLOAT)                            \
            {                                                      \
                ctestingf<hipComplex>(arg);                        \
            }                                                      \
            else if(IS_T_COMPLEX_DOUBLE)                           \
            {                                                      \
                ctestingf<hipDoubleComplex>(arg);                  \
            }                                                      \
            else                                                   \
            {                                                      \
                return HIPSPARSE_STATUS_INTERNAL_ERROR;            \
            }                                                      \
        }                                                          \
        catch(const hipsparseStatus_t& status)                     \
        {                                                          \
            return status;                                         \
        }                                                          \
    }

    switch(FNAME)
    {
        // Level 1
        DEFINE_CASE_T(axpyi);
        DEFINE_CASE_T(doti);
        DEFINE_CASE_T_REAL_VS_COMPLEX(dotci, testing_doti, testing_dotci);
        DEFINE_CASE_T(gthr);
        DEFINE_CASE_T(gthrz);
        DEFINE_CASE_T_REAL_ONLY(roti);
        DEFINE_CASE_T(sctr);

        // Level2
        DEFINE_CASE_T(bsrsv2);
        DEFINE_CASE_IT_X(coomv, testing_spmv_coo);
        DEFINE_CASE_IJT_X(csrmv, testing_spmv_csr);
        DEFINE_CASE_IJT_X(csrsv, testing_spsv_csr);
        DEFINE_CASE_T(gemvi);
        DEFINE_CASE_T(hybmv);

        // Level3
        DEFINE_CASE_T(bsrmm);
        DEFINE_CASE_T(bsrsm2);
        DEFINE_CASE_IT_X(coomm, testing_spmm_coo);
        DEFINE_CASE_IJT_X(cscmm, testing_spmm_csc);
        DEFINE_CASE_IJT_X(csrmm, testing_spmm_csr);
        DEFINE_CASE_IT_X(coosm, testing_spsm_coo);
        DEFINE_CASE_IJT_X(csrsm, testing_spsm_csr);
        DEFINE_CASE_T(gemmi);

        // Extra
        DEFINE_CASE_T(csrgeam);
        DEFINE_CASE_T(csrgemm);

        // Precond
        DEFINE_CASE_T(bsric02);
        DEFINE_CASE_T(bsrilu02);
        DEFINE_CASE_T(csric02);
        DEFINE_CASE_T(csrilu02);
        DEFINE_CASE_T(gtsv2);
        DEFINE_CASE_T(gtsv2_nopivot);
        DEFINE_CASE_T(gtsv2_strided_batch);
        DEFINE_CASE_T(gtsv_interleaved_batch);
        DEFINE_CASE_T(gpsv_interleaved_batch);

        // Conversion
        DEFINE_CASE_T(bsr2csr);
        DEFINE_CASE_T(csr2coo);
        DEFINE_CASE_T(csr2csc);
        DEFINE_CASE_T(csr2hyb);
        DEFINE_CASE_T(csr2bsr);
        DEFINE_CASE_T(csr2gebsr);
        DEFINE_CASE_T(csr2csr_compress);
        DEFINE_CASE_T(coo2csr);
        DEFINE_CASE_T(hyb2csr);
        DEFINE_CASE_IJT_X(csr2dense, testing_sparse_to_dense_csr);
        DEFINE_CASE_IJT_X(csc2dense, testing_sparse_to_dense_csc);
        DEFINE_CASE_IT_X(coo2dense, testing_sparse_to_dense_coo);
        DEFINE_CASE_IJT_X(dense2csr, testing_dense_to_sparse_csr);
        DEFINE_CASE_IJT_X(dense2csc, testing_dense_to_sparse_csc);
        DEFINE_CASE_IT_X(dense2coo, testing_dense_to_sparse_coo);
        DEFINE_CASE_T(gebsr2csr);
        DEFINE_CASE_T(gebsr2gebsc);
        DEFINE_CASE_T(gebsr2gebsr);
    }

#undef DEFINE_CASE_T_X
#undef DEFINE_CASE_IT_X
#undef DEFINE_CASE_IJT_X
#undef DEFINE_CASE_T
#undef IS_T_FLOAT
#undef IS_T_DOUBLE
#undef IS_T_COMPLEX_FLOAT
#undef IS_T_COMPLEX_DOUBLE
#undef DEFINE_CASE_T_REAL_ONLY
#undef DEFINE_CASE_T_REAL_VS_COMPLEX

    return HIPSPARSE_STATUS_INVALID_VALUE;
}
