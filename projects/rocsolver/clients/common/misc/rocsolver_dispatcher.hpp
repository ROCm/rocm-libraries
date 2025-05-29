/* **************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas/rocblas.h"
#include "rocsolver_arguments.hpp"
#include <map>
#include <string>

// auxiliary
#include "common/auxiliary/testing_bdsqr.hpp"
#include "common/auxiliary/testing_bdsvdx.hpp"
#include "common/auxiliary/testing_labrd.hpp"
#include "common/auxiliary/testing_lacgv.hpp"
#include "common/auxiliary/testing_larf.hpp"
#include "common/auxiliary/testing_larfb.hpp"
#include "common/auxiliary/testing_larfg.hpp"
#include "common/auxiliary/testing_larft.hpp"
#include "common/auxiliary/testing_lasr.hpp"
#include "common/auxiliary/testing_laswp.hpp"
#include "common/auxiliary/testing_lasyf.hpp"
#include "common/auxiliary/testing_latrd.hpp"
#include "common/auxiliary/testing_lauum.hpp"
#include "common/auxiliary/testing_orgbr_ungbr.hpp"
#include "common/auxiliary/testing_orglx_unglx.hpp"
#include "common/auxiliary/testing_orgtr_ungtr.hpp"
#include "common/auxiliary/testing_orgxl_ungxl.hpp"
#include "common/auxiliary/testing_orgxr_ungxr.hpp"
#include "common/auxiliary/testing_ormbr_unmbr.hpp"
#include "common/auxiliary/testing_ormlx_unmlx.hpp"
#include "common/auxiliary/testing_ormtr_unmtr.hpp"
#include "common/auxiliary/testing_ormxl_unmxl.hpp"
#include "common/auxiliary/testing_ormxr_unmxr.hpp"
#include "common/auxiliary/testing_stebz.hpp"
#include "common/auxiliary/testing_stedc.hpp"
#include "common/auxiliary/testing_stedcj.hpp"
#include "common/auxiliary/testing_stedcx.hpp"
#include "common/auxiliary/testing_stein.hpp"
#include "common/auxiliary/testing_steqr.hpp"
#include "common/auxiliary/testing_sterf.hpp"

// lapack
#include "common/lapack/testing_gebd2_gebrd.hpp"
#include "common/lapack/testing_geblttrf_npvt.hpp"
#include "common/lapack/testing_geblttrs_npvt.hpp"
#include "common/lapack/testing_gelq2_gelqf.hpp"
#include "common/lapack/testing_gels.hpp"
#include "common/lapack/testing_geql2_geqlf.hpp"
#include "common/lapack/testing_geqr2_geqrf.hpp"
#include "common/lapack/testing_gerq2_gerqf.hpp"
#include "common/lapack/testing_gesdd.hpp"
#include "common/lapack/testing_gesv.hpp"
#include "common/lapack/testing_gesvd.hpp"
#include "common/lapack/testing_gesvdj.hpp"
#include "common/lapack/testing_gesvdx.hpp"
#include "common/lapack/testing_getf2_getrf.hpp"
#include "common/lapack/testing_getf2_getrf_npvt.hpp"
#include "common/lapack/testing_getri.hpp"
#include "common/lapack/testing_getri_npvt.hpp"
#include "common/lapack/testing_getri_npvt_outofplace.hpp"
#include "common/lapack/testing_getri_outofplace.hpp"
#include "common/lapack/testing_getrs.hpp"
#include "common/lapack/testing_posv.hpp"
#include "common/lapack/testing_potf2_potrf.hpp"
#include "common/lapack/testing_potri.hpp"
#include "common/lapack/testing_potrs.hpp"
#include "common/lapack/testing_syev_heev.hpp"
#include "common/lapack/testing_syevd_heevd.hpp"
#include "common/lapack/testing_syevdj_heevdj.hpp"
#include "common/lapack/testing_syevdx_heevdx.hpp"
#include "common/lapack/testing_syevj_heevj.hpp"
#include "common/lapack/testing_syevx_heevx.hpp"
#include "common/lapack/testing_sygsx_hegsx.hpp"
#include "common/lapack/testing_sygv_hegv.hpp"
#include "common/lapack/testing_sygvd_hegvd.hpp"
#include "common/lapack/testing_sygvdj_hegvdj.hpp"
#include "common/lapack/testing_sygvdx_hegvdx.hpp"
#include "common/lapack/testing_sygvj_hegvj.hpp"
#include "common/lapack/testing_sygvx_hegvx.hpp"
#include "common/lapack/testing_sytf2_sytrf.hpp"
#include "common/lapack/testing_sytxx_hetxx.hpp"
#include "common/lapack/testing_trtri.hpp"

// refactorization
#include "common/refact/testing_csrrf_analysis.hpp"
#include "common/refact/testing_csrrf_refactchol.hpp"
#include "common/refact/testing_csrrf_refactlu.hpp"
#include "common/refact/testing_csrrf_solve.hpp"
#include "common/refact/testing_csrrf_splitlu.hpp"
#include "common/refact/testing_csrrf_sumlu.hpp"

// unit
#include "common/unit/testing_gemm.hpp"

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(Arguments&), str_less>;

// Function dispatcher for rocSOLVER tests
class rocsolver_dispatcher
{
    template <typename T>
    static rocblas_status run_function(const char* name, Arguments& argus)
    {
        // Map for functions that support all precisions
        static const func_map map = {
            // auxiliaries
            {"laswp", testing_laswp<T>},
            {"larfg", testing_larfg<T, rocblas_int>},
            {"larfg_64", testing_larfg<T, int64_t>},
            {"larf", testing_larf<T, rocblas_int>},
            {"larf_64", testing_larf<T, int64_t>},
            {"larft", testing_larft<T>},
            {"larfb", testing_larfb<T>},
            {"lasr", testing_lasr<T>},
            {"latrd", testing_latrd<T>},
            {"labrd", testing_labrd<T>},
            {"bdsqr", testing_bdsqr<T>},
            {"steqr", testing_steqr<T>},
            {"stedc", testing_stedc<T>},
            {"stedcj", testing_stedcj<T>},
            {"stedcx", testing_stedcx<T>},
            {"stein", testing_stein<T>},
            {"lasyf", testing_lasyf<T>},
            {"lauum", testing_lauum<T>},
            // potrf
            {"potf2", testing_potf2_potrf<false, false, 0, T, rocblas_int>},
            {"potf2_batched", testing_potf2_potrf<true, true, 0, T, rocblas_int>},
            {"potf2_strided_batched", testing_potf2_potrf<false, true, 0, T, rocblas_int>},
            {"potf2_64", testing_potf2_potrf<false, false, 0, T, int64_t>},
            {"potf2_batched_64", testing_potf2_potrf<true, true, 0, T, int64_t>},
            {"potf2_strided_batched_64", testing_potf2_potrf<false, true, 0, T, int64_t>},
            {"potrf", testing_potf2_potrf<false, false, 1, T, rocblas_int>},
            {"potrf_batched", testing_potf2_potrf<true, true, 1, T, rocblas_int>},
            {"potrf_strided_batched", testing_potf2_potrf<false, true, 1, T, rocblas_int>},
            {"potrf_64", testing_potf2_potrf<false, false, 1, T, int64_t>},
            {"potrf_batched_64", testing_potf2_potrf<true, true, 1, T, int64_t>},
            {"potrf_strided_batched_64", testing_potf2_potrf<false, true, 1, T, int64_t>},
            // potrs
            {"potrs", testing_potrs<false, false, T, rocblas_int>},
            {"potrs_batched", testing_potrs<true, true, T, rocblas_int>},
            {"potrs_strided_batched", testing_potrs<false, true, T, rocblas_int>},
            {"potrs_64", testing_potrs<false, false, T, int64_t>},
            {"potrs_batched_64", testing_potrs<true, true, T, int64_t>},
            {"potrs_strided_batched_64", testing_potrs<false, true, T, int64_t>},
            // posv
            {"posv", testing_posv<false, false, T>},
            {"posv_batched", testing_posv<true, true, T>},
            {"posv_strided_batched", testing_posv<false, true, T>},
            // potri
            {"potri", testing_potri<false, false, T>},
            {"potri_batched", testing_potri<true, true, T>},
            {"potri_strided_batched", testing_potri<false, true, T>},
            // getrf_npvt
            {"getf2_npvt", testing_getf2_getrf_npvt<false, false, 0, T, rocblas_int>},
            {"getf2_npvt_batched", testing_getf2_getrf_npvt<true, true, 0, T, rocblas_int>},
            {"getf2_npvt_strided_batched", testing_getf2_getrf_npvt<false, true, 0, T, rocblas_int>},
            {"getrf_npvt", testing_getf2_getrf_npvt<false, false, 1, T, rocblas_int>},
            {"getrf_npvt_batched", testing_getf2_getrf_npvt<true, true, 1, T, rocblas_int>},
            {"getrf_npvt_strided_batched", testing_getf2_getrf_npvt<false, true, 1, T, rocblas_int>},
            {"getf2_npvt_64", testing_getf2_getrf_npvt<false, false, 0, T, int64_t>},
            {"getf2_npvt_batched_64", testing_getf2_getrf_npvt<true, true, 0, T, int64_t>},
            {"getf2_npvt_strided_batched_64", testing_getf2_getrf_npvt<false, true, 0, T, int64_t>},
            // getrf
            {"getf2", testing_getf2_getrf<false, false, 0, T, rocblas_int>},
            {"getf2_batched", testing_getf2_getrf<true, true, 0, T, rocblas_int>},
            {"getf2_strided_batched", testing_getf2_getrf<false, true, 0, T, rocblas_int>},
            {"getrf", testing_getf2_getrf<false, false, 1, T, rocblas_int>},
            {"getrf_batched", testing_getf2_getrf<true, true, 1, T, rocblas_int>},
            {"getrf_strided_batched", testing_getf2_getrf<false, true, 1, T, rocblas_int>},
            {"getf2_64", testing_getf2_getrf<false, false, 0, T, int64_t>},
            {"getf2_batched_64", testing_getf2_getrf<true, true, 0, T, int64_t>},
            {"getf2_strided_batched_64", testing_getf2_getrf<false, true, 0, T, int64_t>},
            // geqrf
            {"geqr2", testing_geqr2_geqrf<false, false, 0, T, rocblas_int>},
            {"geqr2_batched", testing_geqr2_geqrf<true, true, 0, T, rocblas_int>},
            {"geqr2_strided_batched", testing_geqr2_geqrf<false, true, 0, T, rocblas_int>},
            {"geqr2_64", testing_geqr2_geqrf<false, false, 0, T, int64_t>},
            {"geqr2_batched_64", testing_geqr2_geqrf<true, true, 0, T, int64_t>},
            {"geqr2_strided_batched_64", testing_geqr2_geqrf<false, true, 0, T, int64_t>},
            {"geqrf", testing_geqr2_geqrf<false, false, 1, T, rocblas_int>},
            {"geqrf_batched", testing_geqr2_geqrf<true, true, 1, T, rocblas_int>},
            {"geqrf_strided_batched", testing_geqr2_geqrf<false, true, 1, T, rocblas_int>},
            {"geqrf_ptr_batched", testing_geqr2_geqrf<true, false, 1, T, rocblas_int>},
            {"geqrf_64", testing_geqr2_geqrf<false, false, 1, T, int64_t>},
            {"geqrf_batched_64", testing_geqr2_geqrf<true, true, 1, T, int64_t>},
            {"geqrf_strided_batched_64", testing_geqr2_geqrf<false, true, 1, T, int64_t>},
            {"geqrf_ptr_batched_64", testing_geqr2_geqrf<true, false, 1, T, int64_t>},
            // gerqf
            {"gerq2", testing_gerq2_gerqf<false, false, 0, T>},
            {"gerq2_batched", testing_gerq2_gerqf<true, true, 0, T>},
            {"gerq2_strided_batched", testing_gerq2_gerqf<false, true, 0, T>},
            {"gerqf", testing_gerq2_gerqf<false, false, 1, T>},
            {"gerqf_batched", testing_gerq2_gerqf<true, true, 1, T>},
            {"gerqf_strided_batched", testing_gerq2_gerqf<false, true, 1, T>},
            // geqlf
            {"geql2", testing_geql2_geqlf<false, false, 0, T>},
            {"geql2_batched", testing_geql2_geqlf<true, true, 0, T>},
            {"geql2_strided_batched", testing_geql2_geqlf<false, true, 0, T>},
            {"geqlf", testing_geql2_geqlf<false, false, 1, T>},
            {"geqlf_batched", testing_geql2_geqlf<true, true, 1, T>},
            {"geqlf_strided_batched", testing_geql2_geqlf<false, true, 1, T>},
            // gelqf
            {"gelq2", testing_gelq2_gelqf<false, false, 0, T>},
            {"gelq2_batched", testing_gelq2_gelqf<true, true, 0, T>},
            {"gelq2_strided_batched", testing_gelq2_gelqf<false, true, 0, T>},
            {"gelqf", testing_gelq2_gelqf<false, false, 1, T>},
            {"gelqf_batched", testing_gelq2_gelqf<true, true, 1, T>},
            {"gelqf_strided_batched", testing_gelq2_gelqf<false, true, 1, T>},
            // getrs
            {"getrs", testing_getrs<false, false, T, rocblas_int>},
            {"getrs_batched", testing_getrs<true, true, T, rocblas_int>},
            {"getrs_strided_batched", testing_getrs<false, true, T, rocblas_int>},
            {"getrs_64", testing_getrs<false, false, T, int64_t>},
            {"getrs_batched_64", testing_getrs<true, true, T, int64_t>},
            {"getrs_strided_batched_64", testing_getrs<false, true, T, int64_t>},
            // gesv
            {"gesv", testing_gesv<false, false, T>},
            {"gesv_batched", testing_gesv<true, true, T>},
            {"gesv_strided_batched", testing_gesv<false, true, T>},
            // gesvd
            {"gesvd", testing_gesvd<false, false, T>},
            {"gesvd_batched", testing_gesvd<true, true, T>},
            {"gesvd_strided_batched", testing_gesvd<false, true, T>},
            // gesdd
            {"gesdd", testing_gesdd<false, false, T>},
            {"gesdd_batched", testing_gesdd<true, true, T>},
            {"gesdd_strided_batched", testing_gesdd<false, true, T>},
            // gesvdj
            {"gesvdj", testing_gesvdj<false, false, T>},
            {"gesvdj_batched", testing_gesvdj<true, true, T>},
            {"gesvdj_strided_batched", testing_gesvdj<false, true, T>},
            // gesvdx
            {"gesvdx", testing_gesvdx<false, false, T>},
            {"gesvdx_batched", testing_gesvdx<true, true, T>},
            {"gesvdx_strided_batched", testing_gesvdx<false, true, T>},
            // trtri
            {"trtri", testing_trtri<false, false, T>},
            {"trtri_batched", testing_trtri<true, true, T>},
            {"trtri_strided_batched", testing_trtri<false, true, T>},
            // getri
            {"getri", testing_getri<false, false, T>},
            {"getri_batched", testing_getri<true, true, T>},
            {"getri_strided_batched", testing_getri<false, true, T>},
            // getri_npvt
            {"getri_npvt", testing_getri_npvt<false, false, T>},
            {"getri_npvt_batched", testing_getri_npvt<true, true, T>},
            {"getri_npvt_strided_batched", testing_getri_npvt<false, true, T>},
            // getri_outofplace
            {"getri_outofplace", testing_getri_outofplace<false, false, T>},
            {"getri_outofplace_batched", testing_getri_outofplace<true, true, T>},
            {"getri_outofplace_strided_batched", testing_getri_outofplace<false, true, T>},
            // getri_npvt_outofplace
            {"getri_npvt_outofplace", testing_getri_npvt_outofplace<false, false, T>},
            {"getri_npvt_outofplace_batched", testing_getri_npvt_outofplace<true, true, T>},
            {"getri_npvt_outofplace_strided_batched", testing_getri_npvt_outofplace<false, true, T>},
            // gels
            {"gels", testing_gels<false, false, T>},
            {"gels_batched", testing_gels<true, true, T>},
            {"gels_strided_batched", testing_gels<false, true, T>},
            // gebrd
            {"gebd2", testing_gebd2_gebrd<false, false, 0, T>},
            {"gebd2_batched", testing_gebd2_gebrd<true, true, 0, T>},
            {"gebd2_strided_batched", testing_gebd2_gebrd<false, true, 0, T>},
            {"gebrd", testing_gebd2_gebrd<false, false, 1, T>},
            {"gebrd_batched", testing_gebd2_gebrd<true, true, 1, T>},
            {"gebrd_strided_batched", testing_gebd2_gebrd<false, true, 1, T>},
            // sytrf
            {"sytf2", testing_sytf2_sytrf<false, false, 0, T>},
            {"sytf2_batched", testing_sytf2_sytrf<true, true, 0, T>},
            {"sytf2_strided_batched", testing_sytf2_sytrf<false, true, 0, T>},
            {"sytrf", testing_sytf2_sytrf<false, false, 1, T>},
            {"sytrf_batched", testing_sytf2_sytrf<true, true, 1, T>},
            {"sytrf_strided_batched", testing_sytf2_sytrf<false, true, 1, T>},
            // geblttrf_npvt
            {"geblttrf_npvt", testing_geblttrf_npvt<false, false, T>},
            {"geblttrf_npvt_batched", testing_geblttrf_npvt<true, true, T>},
            {"geblttrf_npvt_strided_batched", testing_geblttrf_npvt<false, true, T>},
            // geblttrs_npvt
            {"geblttrs_npvt", testing_geblttrs_npvt<false, false, T>},
            {"geblttrs_npvt_batched", testing_geblttrs_npvt<true, true, T>},
            {"geblttrs_npvt_strided_batched", testing_geblttrs_npvt<false, true, T>},
            // unit
            {"gemm", testing_gemm<false, false, T, rocblas_int>},
            {"gemm_batched", testing_gemm<true, true, T, rocblas_int>},
            {"gemm_strided_batched", testing_gemm<false, true, T, rocblas_int>},
            {"gemm_64", testing_gemm<false, false, T, int64_t>},
            {"gemm_batched_64", testing_gemm<true, true, T, int64_t>},
            {"gemm_strided_batched_64", testing_gemm<false, true, T, int64_t>},
        };

        // Grab function from the map and execute
        auto match = map.find(name);
        if(match != map.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
            return rocblas_status_invalid_value;
    }

    template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
    static rocblas_status run_function_limited_precision(const char* name, Arguments& argus)
    {
        // Map for functions that support only single and double precisions
        static const func_map map_real = {
            // auxiliaries
            {"sterf", testing_sterf<T>},
            {"stebz", testing_stebz<T>},
            {"bdsvdx", testing_bdsvdx<T>},
            // orgxx
            {"org2r", testing_orgxr_ungxr<T, 0>},
            {"orgqr", testing_orgxr_ungxr<T, 1>},
            {"org2l", testing_orgxl_ungxl<T, 0>},
            {"orgql", testing_orgxl_ungxl<T, 1>},
            {"orgl2", testing_orglx_unglx<T, 0>},
            {"orglq", testing_orglx_unglx<T, 1>},
            {"orgbr", testing_orgbr_ungbr<T>},
            {"orgtr", testing_orgtr_ungtr<T>},
            // ormxx
            {"orm2r", testing_ormxr_unmxr<T, 0>},
            {"ormqr", testing_ormxr_unmxr<T, 1>},
            {"orm2l", testing_ormxl_unmxl<T, 0>},
            {"ormql", testing_ormxl_unmxl<T, 1>},
            {"orml2", testing_ormlx_unmlx<T, 0>},
            {"ormlq", testing_ormlx_unmlx<T, 1>},
            {"ormbr", testing_ormbr_unmbr<T>},
            {"ormtr", testing_ormtr_unmtr<T>},
            // sytrd
            {"sytd2", testing_sytxx_hetxx<false, false, 0, T>},
            {"sytd2_batched", testing_sytxx_hetxx<true, true, 0, T>},
            {"sytd2_strided_batched", testing_sytxx_hetxx<false, true, 0, T>},
            {"sytrd", testing_sytxx_hetxx<false, false, 1, T>},
            {"sytrd_batched", testing_sytxx_hetxx<true, true, 1, T>},
            {"sytrd_strided_batched", testing_sytxx_hetxx<false, true, 1, T>},
            // sygst
            {"sygs2", testing_sygsx_hegsx<false, false, 0, T>},
            {"sygs2_batched", testing_sygsx_hegsx<true, true, 0, T>},
            {"sygs2_strided_batched", testing_sygsx_hegsx<false, true, 0, T>},
            {"sygst", testing_sygsx_hegsx<false, false, 1, T>},
            {"sygst_batched", testing_sygsx_hegsx<true, true, 1, T>},
            {"sygst_strided_batched", testing_sygsx_hegsx<false, true, 1, T>},
            // syev
            {"syev", testing_syev_heev<false, false, T>},
            {"syev_batched", testing_syev_heev<true, true, T>},
            {"syev_strided_batched", testing_syev_heev<false, true, T>},
            // syevd
            {"syevd", testing_syevd_heevd<false, false, T>},
            {"syevd_batched", testing_syevd_heevd<true, true, T>},
            {"syevd_strided_batched", testing_syevd_heevd<false, true, T>},
            // syevdj
            {"syevdj", testing_syevdj_heevdj<false, false, T>},
            {"syevdj_batched", testing_syevdj_heevdj<true, true, T>},
            {"syevdj_strided_batched", testing_syevdj_heevdj<false, true, T>},
            // syevj
            {"syevj", testing_syevj_heevj<false, false, T>},
            {"syevj_batched", testing_syevj_heevj<true, true, T>},
            {"syevj_strided_batched", testing_syevj_heevj<false, true, T>},
            // syevx
            {"syevx", testing_syevx_heevx<false, false, T>},
            {"syevx_batched", testing_syevx_heevx<true, true, T>},
            {"syevx_strided_batched", testing_syevx_heevx<false, true, T>},
            // syevdx
            {"syevdx", testing_syevdx_heevdx<false, false, T>},
            {"syevdx_batched", testing_syevdx_heevdx<true, true, T>},
            {"syevdx_strided_batched", testing_syevdx_heevdx<false, true, T>},
            // sygv
            {"sygv", testing_sygv_hegv<false, false, T>},
            {"sygv_batched", testing_sygv_hegv<true, true, T>},
            {"sygv_strided_batched", testing_sygv_hegv<false, true, T>},
            // sygvd
            {"sygvd", testing_sygvd_hegvd<false, false, T>},
            {"sygvd_batched", testing_sygvd_hegvd<true, true, T>},
            {"sygvd_strided_batched", testing_sygvd_hegvd<false, true, T>},
            // sygvdj
            {"sygvdj", testing_sygvdj_hegvdj<false, false, T>},
            {"sygvdj_batched", testing_sygvdj_hegvdj<true, true, T>},
            {"sygvdj_strided_batched", testing_sygvdj_hegvdj<false, true, T>},
            // sygvj
            {"sygvj", testing_sygvj_hegvj<false, false, T>},
            {"sygvj_batched", testing_sygvj_hegvj<true, true, T>},
            {"sygvj_strided_batched", testing_sygvj_hegvj<false, true, T>},
            // sygvx
            {"sygvx", testing_sygvx_hegvx<false, false, T>},
            {"sygvx_batched", testing_sygvx_hegvx<true, true, T>},
            {"sygvx_strided_batched", testing_sygvx_hegvx<false, true, T>},
            // sygvdx
            {"sygvdx", testing_sygvdx_hegvdx<false, false, T>},
            {"sygvdx_batched", testing_sygvdx_hegvdx<true, true, T>},
            {"sygvdx_strided_batched", testing_sygvdx_hegvdx<false, true, T>},
            // refactorization
            {"csrrf_analysis", testing_csrrf_analysis<T>},
            {"csrrf_sumlu", testing_csrrf_sumlu<T>},
            {"csrrf_splitlu", testing_csrrf_splitlu<T>},
            {"csrrf_refactlu", testing_csrrf_refactlu<T>},
            {"csrrf_refactchol", testing_csrrf_refactchol<T>},
            {"csrrf_solve", testing_csrrf_solve<T>},
        };

        // Grab function from the map and execute
        auto match = map_real.find(name);
        if(match != map_real.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
            return rocblas_status_invalid_value;
    }

    template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
    static rocblas_status run_function_limited_precision(const char* name, Arguments& argus)
    {
        // Map for functions that support only single-complex and double-complex precisions
        static const func_map map_complex = {
            // auxiliaries
            {"lacgv", testing_lacgv<T, rocblas_int>},
            {"lacgv_64", testing_lacgv<T, int64_t>},
            // ungxx
            {"ung2r", testing_orgxr_ungxr<T, 0>},
            {"ungqr", testing_orgxr_ungxr<T, 1>},
            {"ung2l", testing_orgxl_ungxl<T, 0>},
            {"ungql", testing_orgxl_ungxl<T, 1>},
            {"ungl2", testing_orglx_unglx<T, 0>},
            {"unglq", testing_orglx_unglx<T, 1>},
            {"ungbr", testing_orgbr_ungbr<T>},
            {"ungtr", testing_orgtr_ungtr<T>},
            // unmxx
            {"unm2r", testing_ormxr_unmxr<T, 0>},
            {"unmqr", testing_ormxr_unmxr<T, 1>},
            {"unm2l", testing_ormxl_unmxl<T, 0>},
            {"unmql", testing_ormxl_unmxl<T, 1>},
            {"unml2", testing_ormlx_unmlx<T, 0>},
            {"unmlq", testing_ormlx_unmlx<T, 1>},
            {"unmbr", testing_ormbr_unmbr<T>},
            {"unmtr", testing_ormtr_unmtr<T>},
            // hetrd
            {"hetd2", testing_sytxx_hetxx<false, false, 0, T>},
            {"hetd2_batched", testing_sytxx_hetxx<true, true, 0, T>},
            {"hetd2_strided_batched", testing_sytxx_hetxx<false, true, 0, T>},
            {"hetrd", testing_sytxx_hetxx<false, false, 1, T>},
            {"hetrd_batched", testing_sytxx_hetxx<true, true, 1, T>},
            {"hetrd_strided_batched", testing_sytxx_hetxx<false, true, 1, T>},
            // hegst
            {"hegs2", testing_sygsx_hegsx<false, false, 0, T>},
            {"hegs2_batched", testing_sygsx_hegsx<true, true, 0, T>},
            {"hegs2_strided_batched", testing_sygsx_hegsx<false, true, 0, T>},
            {"hegst", testing_sygsx_hegsx<false, false, 1, T>},
            {"hegst_batched", testing_sygsx_hegsx<true, true, 1, T>},
            {"hegst_strided_batched", testing_sygsx_hegsx<false, true, 1, T>},
            // heev
            {"heev", testing_syev_heev<false, false, T>},
            {"heev_batched", testing_syev_heev<true, true, T>},
            {"heev_strided_batched", testing_syev_heev<false, true, T>},
            // heevd
            {"heevd", testing_syevd_heevd<false, false, T>},
            {"heevd_batched", testing_syevd_heevd<true, true, T>},
            {"heevd_strided_batched", testing_syevd_heevd<false, true, T>},
            // heevdj
            {"heevdj", testing_syevdj_heevdj<false, false, T>},
            {"heevdj_batched", testing_syevdj_heevdj<true, true, T>},
            {"heevdj_strided_batched", testing_syevdj_heevdj<false, true, T>},
            // heevj
            {"heevj", testing_syevj_heevj<false, false, T>},
            {"heevj_batched", testing_syevj_heevj<true, true, T>},
            {"heevj_strided_batched", testing_syevj_heevj<false, true, T>},
            // heevx
            {"heevx", testing_syevx_heevx<false, false, T>},
            {"heevx_batched", testing_syevx_heevx<true, true, T>},
            {"heevx_strided_batched", testing_syevx_heevx<false, true, T>},
            // heevdx
            {"heevdx", testing_syevdx_heevdx<false, false, T>},
            {"heevdx_batched", testing_syevdx_heevdx<true, true, T>},
            {"heevdx_strided_batched", testing_syevdx_heevdx<false, true, T>},
            // hegv
            {"hegv", testing_sygv_hegv<false, false, T>},
            {"hegv_batched", testing_sygv_hegv<true, true, T>},
            {"hegv_strided_batched", testing_sygv_hegv<false, true, T>},
            // hegvd
            {"hegvd", testing_sygvd_hegvd<false, false, T>},
            {"hegvd_batched", testing_sygvd_hegvd<true, true, T>},
            {"hegvd_strided_batched", testing_sygvd_hegvd<false, true, T>},
            // hegvdj
            {"hegvdj", testing_sygvdj_hegvdj<false, false, T>},
            {"hegvdj_batched", testing_sygvdj_hegvdj<true, true, T>},
            {"hegvdj_strided_batched", testing_sygvdj_hegvdj<false, true, T>},
            // hegvj
            {"hegvj", testing_sygvj_hegvj<false, false, T>},
            {"hegvj_batched", testing_sygvj_hegvj<true, true, T>},
            {"hegvj_strided_batched", testing_sygvj_hegvj<false, true, T>},
            // hegvx
            {"hegvx", testing_sygvx_hegvx<false, false, T>},
            {"hegvx_batched", testing_sygvx_hegvx<true, true, T>},
            {"hegvx_strided_batched", testing_sygvx_hegvx<false, true, T>},
            // hegvdx
            {"hegvdx", testing_sygvdx_hegvdx<false, false, T>},
            {"hegvdx_batched", testing_sygvdx_hegvdx<true, true, T>},
            {"hegvdx_strided_batched", testing_sygvdx_hegvdx<false, true, T>},
        };

        // Grab function from the map and execute
        auto match = map_complex.find(name);
        if(match != map_complex.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
            return rocblas_status_invalid_value;
    }

public:
    static void invoke(const std::string& name, char precision, Arguments& argus)
    {
        rocblas_status status;

        if(precision == 's')
            status = run_function<float>(name.c_str(), argus);
        else if(precision == 'd')
            status = run_function<double>(name.c_str(), argus);
        else if(precision == 'c')
            status = run_function<rocblas_float_complex>(name.c_str(), argus);
        else if(precision == 'z')
            status = run_function<rocblas_double_complex>(name.c_str(), argus);
        else
            throw std::invalid_argument("Invalid value for --precision");

        if(status == rocblas_status_invalid_value)
        {
            if(precision == 's')
                status = run_function_limited_precision<float>(name.c_str(), argus);
            else if(precision == 'd')
                status = run_function_limited_precision<double>(name.c_str(), argus);
            else if(precision == 'c')
                status = run_function_limited_precision<rocblas_float_complex>(name.c_str(), argus);
            else if(precision == 'z')
                status = run_function_limited_precision<rocblas_double_complex>(name.c_str(), argus);
        }

        if(status == rocblas_status_invalid_value)
        {
            std::string msg = "Invalid combination --function ";
            msg += name;
            msg += " --precision ";
            msg += precision;
            throw std::invalid_argument(msg);
        }
    }
};
