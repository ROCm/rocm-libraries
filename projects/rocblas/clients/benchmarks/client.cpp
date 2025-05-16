/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#define ROCBLAS_BETA_FEATURES_API
#include "program_options.hpp"

#include "client_omp.hpp"
#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_parse_data.hpp"
#include "tensile_host.hpp"
#include "type_dispatch.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "frequency_monitor.hpp"

// aux
#include "testing_set_get_matrix.hpp"
#include "testing_set_get_matrix_async.hpp"
#include "testing_set_get_vector.hpp"
#include "testing_set_get_vector_async.hpp"
// blas1
// new way with prototypes
#include "blas1/common_asum.hpp"
#include "blas1/common_axpy.hpp"
#include "blas1/common_copy.hpp"
#include "blas1/common_dot.hpp"
#include "blas1/common_iamax_iamin.hpp"
#include "blas1/common_nrm2.hpp"
#include "blas1/common_rot.hpp"
#include "blas1/common_scal.hpp"
#include "blas1/common_swap.hpp"
#include "blas_ex/common_axpy_ex.hpp"
#include "blas_ex/common_dot_ex.hpp"
#include "blas_ex/common_nrm2_ex.hpp"
#include "blas_ex/common_rot_ex.hpp"
#include "blas_ex/common_scal_ex.hpp"
// blas2
#include "blas2/common_gbmv.hpp"
#include "blas2/common_gemv.hpp"
#include "blas2/common_ger.hpp"
#include "blas2/common_hbmv.hpp"
#include "blas2/common_hemv.hpp"
#include "blas2/common_her.hpp"
#include "blas2/common_her2.hpp"
#include "blas2/common_hpmv.hpp"
#include "blas2/common_hpr.hpp"
#include "blas2/common_hpr2.hpp"
#include "blas2/common_sbmv.hpp"
#include "blas2/common_spmv.hpp"
#include "blas2/common_spr.hpp"
#include "blas2/common_spr2.hpp"
#include "blas2/common_symv.hpp"
#include "blas2/common_syr.hpp"
#include "blas2/common_syr2.hpp"
#include "blas2/common_tbmv.hpp"
#include "blas2/common_tbsv.hpp"
#include "blas2/common_tpmv.hpp"
#include "blas2/common_tpsv.hpp"
#include "blas2/common_trmv.hpp"
#include "blas2/common_trsv.hpp"
// for blas3 with no tensile, some will use source gemms
#include "blas3/common_dgmm.hpp"
#include "blas3/common_geam.hpp"
#include "blas3/common_gemm.hpp"
#include "blas3/common_her2k.hpp"
#include "blas3/common_herk.hpp"
#include "blas3/common_symm_hemm.hpp"
#include "blas3/common_syr2k.hpp"
#include "blas3/common_syrk.hpp"
#include "blas3/common_trmm.hpp"
#include "blas3/common_trsm.hpp"
#include "blas3/common_trtri.hpp"
#include "blas_ex/common_geam_ex.hpp"
#include "blas_ex/common_gemm_ex.hpp"
#include "blas_ex/common_gemmt.hpp"
#include "blas_ex/common_trsm_ex.hpp"
#include "type_dispatch.hpp"
#undef I

using namespace roc; // For emulated program_options
using namespace std::literals; // For std::string literals of form "str"s

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(const Arguments&), str_less>;

// Run a function by using map to map arg.function to function
void run_function(const func_map& map, const Arguments& arg, const std::string& msg = "")
{
    auto match = map.find(arg.function);
    if(match == map.end())
        throw std::invalid_argument("Invalid combination --function "s + arg.function
                                    + " --a_type "s + rocblas_datatype2string(arg.a_type) + msg);
    match->second(arg);
}

// Template to dispatch testing_gemm_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_ex : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_ex<
    Ti,
    To,
    Tc,
    std::enable_if_t<
        !std::is_same_v<
            Ti,
            void> && !(std::is_same_v<Ti, To> && std::is_same_v<Ti, Tc> && std::is_same_v<Ti, rocblas_bfloat16>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemm_ex", testing_gemm_ex<Ti, To, Tc>},
            {"gemm_batched_ex", testing_gemm_batched_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

// Template to dispatch testing_gemm_strided_batched_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_strided_batched_ex : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_strided_batched_ex<
    Ti,
    To,
    Tc,
    std::enable_if_t<
        !std::is_same_v<
            Ti,
            void> && !(std::is_same_v<Ti, To> && std::is_same_v<Ti, Tc> && std::is_same_v<Ti, rocblas_bfloat16>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemm_strided_batched_ex", testing_gemm_strided_batched_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U = T, typename = void>
struct perf_blas : rocblas_test_invalid
{
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"set_get_vector", testing_set_get_vector<T>},
            {"set_get_vector_async", testing_set_get_vector_async<T>},
            {"set_get_matrix", testing_set_get_matrix<T>},
            {"set_get_matrix_async", testing_set_get_matrix_async<T>},
            // L1
            {"asum", testing_asum<T>},
            {"asum_batched", testing_asum_batched<T>},
            {"asum_strided_batched", testing_asum_strided_batched<T>},
            {"axpy", testing_axpy<T>},
            {"axpy_batched", testing_axpy_batched<T>},
            {"axpy_strided_batched", testing_axpy_strided_batched<T>},
            {"copy", testing_copy<T>},
            {"copy_batched", testing_copy_batched<T>},
            {"copy_strided_batched", testing_copy_strided_batched<T>},
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
            {"iamax", testing_iamax<T>},
            {"iamax_batched", testing_iamax_batched<T>},
            {"iamax_strided_batched", testing_iamax_strided_batched<T>},
            {"iamin", testing_iamin<T>},
            {"iamin_batched", testing_iamin_batched<T>},
            {"iamin_strided_batched", testing_iamin_strided_batched<T>},
            {"nrm2", testing_nrm2<T>},
            {"nrm2_batched", testing_nrm2_batched<T>},
            {"nrm2_strided_batched", testing_nrm2_strided_batched<T>},
            {"rotm", testing_rotm<T>},
            {"rotm_batched", testing_rotm_batched<T>},
            {"rotm_strided_batched", testing_rotm_strided_batched<T>},
            {"rotmg", testing_rotmg<T>},
            {"rotmg_batched", testing_rotmg_batched<T>},
            {"rotmg_strided_batched", testing_rotmg_strided_batched<T>},
            {"swap", testing_swap<T>},
            {"swap_batched", testing_swap_batched<T>},
            {"swap_strided_batched", testing_swap_strided_batched<T>},
            // L2
            {"gbmv", testing_gbmv<T>},
            {"gbmv_batched", testing_gbmv_batched<T>},
            {"gbmv_strided_batched", testing_gbmv_strided_batched<T>},
            {"geam", testing_geam<T>},
            {"geam_batched", testing_geam_batched<T>},
            {"geam_strided_batched", testing_geam_strided_batched<T>},
            {"geam_ex", testing_geam_ex<T>},
            {"gemv", testing_gemv<T>},
            {"ger", testing_ger<T, false>},
            {"ger_batched", testing_ger_batched<T, false>},
            {"ger_strided_batched", testing_ger_strided_batched<T, false>},
            {"spr", testing_spr<T>},
            {"spr_batched", testing_spr_batched<T>},
            {"spr_strided_batched", testing_spr_strided_batched<T>},
            {"spr2", testing_spr2<T>},
            {"spr2_batched", testing_spr2_batched<T>},
            {"spr2_strided_batched", testing_spr2_strided_batched<T>},
            {"syr", testing_syr<T>},
            {"syr_batched", testing_syr_batched<T>},
            {"syr_strided_batched", testing_syr_strided_batched<T>},
            {"syr2", testing_syr2<T>},
            {"syr2_batched", testing_syr2_batched<T>},
            {"syr2_strided_batched", testing_syr2_strided_batched<T>},
            {"sbmv", testing_sbmv<T>},
            {"sbmv_batched", testing_sbmv_batched<T>},
            {"sbmv_strided_batched", testing_sbmv_strided_batched<T>},
            {"spmv", testing_spmv<T>},
            {"spmv_batched", testing_spmv_batched<T>},
            {"spmv_strided_batched", testing_spmv_strided_batched<T>},
            {"symv", testing_symv<T>},
            {"symv_batched", testing_symv_batched<T>},
            {"symv_strided_batched", testing_symv_strided_batched<T>},
            {"tbmv", testing_tbmv<T>},
            {"tbmv_batched", testing_tbmv_batched<T>},
            {"tbmv_strided_batched", testing_tbmv_strided_batched<T>},
            {"tbsv", testing_tbsv<T>},
            {"tbsv_batched", testing_tbsv_batched<T>},
            {"tbsv_strided_batched", testing_tbsv_strided_batched<T>},
            {"tpmv", testing_tpmv<T>},
            {"tpmv_batched", testing_tpmv_batched<T>},
            {"tpmv_strided_batched", testing_tpmv_strided_batched<T>},
            {"tpsv", testing_tpsv<T>},
            {"tpsv_batched", testing_tpsv_batched<T>},
            {"tpsv_strided_batched", testing_tpsv_strided_batched<T>},
            {"trmv", testing_trmv<T>},
            {"trmv_batched", testing_trmv_batched<T>},
            {"trmv_strided_batched", testing_trmv_strided_batched<T>},
            // L3
            {"dgmm", testing_dgmm<T>},
            {"dgmm_batched", testing_dgmm_batched<T>},
            {"dgmm_strided_batched", testing_dgmm_strided_batched<T>},
            {"gemmt", testing_gemmt<T>},
            {"gemmt_batched", testing_gemmt_batched<T>},
            {"gemmt_strided_batched", testing_gemmt_strided_batched<T>},
            {"symm", testing_symm_hemm<T, false>},
            {"symm_batched", testing_symm_hemm_batched<T, false>},
            {"symm_strided_batched", testing_symm_hemm_strided_batched<T, false>},
            {"trmm", testing_trmm<T>},
            {"trmm_batched", testing_trmm_batched<T>},
            {"trmm_strided_batched", testing_trmm_strided_batched<T>},
            {"syrk", testing_syrk<T>},
            {"syrk_batched", testing_syrk_batched<T>},
            {"syrk_strided_batched", testing_syrk_strided_batched<T>},
            {"syr2k", testing_syr2k<T>},
            {"syr2k_batched", testing_syr2k_batched<T>},
            {"syr2k_strided_batched", testing_syr2k_strided_batched<T>},
            {"trsv", testing_trsv<T>},
            {"trsv_batched", testing_trsv_batched<T>},
            {"trsv_strided_batched", testing_trsv_strided_batched<T>},
            // allow profiling with source gemms even without BUILD_WITH_TENSILE
            {"gemm", testing_gemm<T>},
            {"gemm_batched", testing_gemm_batched<T>},
            {"gemm_strided_batched", testing_gemm_strided_batched<T>},
            {"syrkx", testing_syr2k<T, false>},
            {"syrkx_batched", testing_syr2k_batched<T, false>},
            {"syrkx_strided_batched", testing_syr2k_strided_batched<T, false>},
            {"trtri", testing_trtri<T>},
            {"trtri_batched", testing_trtri_batched<T>},
            {"trtri_strided_batched", testing_trtri_strided_batched<T>},
            {"trsm", testing_trsm<T>},
            {"trsm_ex", testing_trsm_ex<T>},
            {"trsm_batched", testing_trsm_batched<T>},
            {"trsm_batched_ex", testing_trsm_batched_ex<T>},
            {"trsm_strided_batched", testing_trsm_strided_batched<T>},
            {"trsm_strided_batched_ex", testing_trsm_strided_batched_ex<T>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same_v<T, rocblas_bfloat16>>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same_v<T, rocblas_half>>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map
            = { {"axpy", testing_axpy<T>},
                {"axpy_batched", testing_axpy_batched<T>},
                {"axpy_strided_batched", testing_axpy_strided_batched<T>},
                {"dot", testing_dot<T>},
                {"dot_batched", testing_dot_batched<T>},
                {"dot_strided_batched", testing_dot_strided_batched<T>},
                {"geam_ex", testing_geam_ex<T>},
#if BUILD_WITH_TENSILE
                {"gemm", testing_gemm<T>},
                {"gemm_batched", testing_gemm_batched<T>},
                {"gemm_strided_batched", testing_gemm_strided_batched<T>},
#endif
              };
        run_function(map, arg);
    }
};

// rocblas-bench dispatch for gemv_batched and gemv_strided_batched tests
template <typename Ti, typename Tex = Ti, typename To = Tex, typename = void>
struct perf_gemv_batched_and_strided_batched : rocblas_test_invalid
{
};

template <typename Ti, typename Tex, typename To>
struct perf_gemv_batched_and_strided_batched<
    Ti,
    Tex,
    To,
    std::enable_if_t<
        (std::is_same<Ti, float>{} && std::is_same<To, Ti>{} && std::is_same<Tex, To>{})
        || (std::is_same<Ti, double>{} && std::is_same<To, Ti>{} && std::is_same<Tex, To>{})
        || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, Ti>{}
            && std::is_same<Tex, To>{})
        || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, Ti>{}
            && std::is_same<Tex, To>{})
        || (std::is_same<Ti, rocblas_half>{} && std::is_same<Tex, float>{}
            && (std::is_same<To, Ti>{} || std::is_same<To, float>{}))
        || (std::is_same<Ti, rocblas_bfloat16>{} && std::is_same<Tex, float>{}
            && (std::is_same<To, Ti>{} || std::is_same<To, float>{}))>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemv_batched", testing_gemv_batched<Ti, Tex, To>},
            {"gemv_strided_batched", testing_gemv_strided_batched<Ti, Tex, To>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<
    T,
    U,
    std::enable_if_t<
        std::is_same_v<T, rocblas_double_complex> || std::is_same_v<T, rocblas_float_complex>>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"set_get_vector", testing_set_get_vector<T>},
            {"set_get_vector_async", testing_set_get_vector_async<T>},
            {"set_get_matrix", testing_set_get_matrix<T>},
            {"set_get_matrix_async", testing_set_get_matrix_async<T>},
            // L1
            {"asum", testing_asum<T>},
            {"asum_batched", testing_asum_batched<T>},
            {"asum_strided_batched", testing_asum_strided_batched<T>},
            {"axpy", testing_axpy<T>},
            {"axpy_batched", testing_axpy_batched<T>},
            {"axpy_strided_batched", testing_axpy_strided_batched<T>},
            {"copy", testing_copy<T>},
            {"copy_batched", testing_copy_batched<T>},
            {"copy_strided_batched", testing_copy_strided_batched<T>},
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
            {"dotc", testing_dotc<T>},
            {"dotc_batched", testing_dotc_batched<T>},
            {"dotc_strided_batched", testing_dotc_strided_batched<T>},
            {"iamax", testing_iamax<T>},
            {"iamax_batched", testing_iamax_batched<T>},
            {"iamax_strided_batched", testing_iamax_strided_batched<T>},
            {"iamin", testing_iamin<T>},
            {"iamin_batched", testing_iamin_batched<T>},
            {"iamin_strided_batched", testing_iamin_strided_batched<T>},
            {"nrm2", testing_nrm2<T>},
            {"nrm2_batched", testing_nrm2_batched<T>},
            {"nrm2_strided_batched", testing_nrm2_strided_batched<T>},
            {"swap", testing_swap<T>},
            {"swap_batched", testing_swap_batched<T>},
            {"swap_strided_batched", testing_swap_strided_batched<T>},
            // L2
            {"gbmv", testing_gbmv<T>},
            {"gbmv_batched", testing_gbmv_batched<T>},
            {"gbmv_strided_batched", testing_gbmv_strided_batched<T>},
            {"gemv", testing_gemv<T>},
            {"geru", testing_ger<T, false>},
            {"geru_batched", testing_ger_batched<T, false>},
            {"geru_strided_batched", testing_ger_strided_batched<T, false>},
            {"gerc", testing_ger<T, true>},
            {"gerc_batched", testing_ger_batched<T, true>},
            {"gerc_strided_batched", testing_ger_strided_batched<T, true>},
            {"hbmv", testing_hbmv<T>},
            {"hbmv_batched", testing_hbmv_batched<T>},
            {"hbmv_strided_batched", testing_hbmv_strided_batched<T>},
            {"hemv", testing_hemv<T>},
            {"hemv_batched", testing_hemv_batched<T>},
            {"hemv_strided_batched", testing_hemv_strided_batched<T>},
            {"her", testing_her<T>},
            {"her_batched", testing_her_batched<T>},
            {"her_strided_batched", testing_her_strided_batched<T>},
            {"her2", testing_her2<T>},
            {"her2_batched", testing_her2_batched<T>},
            {"her2_strided_batched", testing_her2_strided_batched<T>},
            {"hpmv", testing_hpmv<T>},
            {"hpmv_batched", testing_hpmv_batched<T>},
            {"hpmv_strided_batched", testing_hpmv_strided_batched<T>},
            {"hpr", testing_hpr<T>},
            {"hpr_batched", testing_hpr_batched<T>},
            {"hpr_strided_batched", testing_hpr_strided_batched<T>},
            {"hpr2", testing_hpr2<T>},
            {"hpr2_batched", testing_hpr2_batched<T>},
            {"hpr2_strided_batched", testing_hpr2_strided_batched<T>},
            {"spr", testing_spr<T>},
            {"spr_batched", testing_spr_batched<T>},
            {"spr_strided_batched", testing_spr_strided_batched<T>},
            {"syr", testing_syr<T>},
            {"syr_batched", testing_syr_batched<T>},
            {"syr_strided_batched", testing_syr_strided_batched<T>},
            {"syr2", testing_syr2<T>},
            {"syr2_batched", testing_syr2_batched<T>},
            {"syr2_strided_batched", testing_syr2_strided_batched<T>},
            {"tbmv", testing_tbmv<T>},
            {"tbmv_batched", testing_tbmv_batched<T>},
            {"tbmv_strided_batched", testing_tbmv_strided_batched<T>},
            {"tbsv", testing_tbsv<T>},
            {"tbsv_batched", testing_tbsv_batched<T>},
            {"tbsv_strided_batched", testing_tbsv_strided_batched<T>},
            {"tpmv", testing_tpmv<T>},
            {"tpmv_batched", testing_tpmv_batched<T>},
            {"tpmv_strided_batched", testing_tpmv_strided_batched<T>},
            {"tpsv", testing_tpsv<T>},
            {"tpsv_batched", testing_tpsv_batched<T>},
            {"tpsv_strided_batched", testing_tpsv_strided_batched<T>},
            {"symv", testing_symv<T>},
            {"symv_batched", testing_symv_batched<T>},
            {"symv_strided_batched", testing_symv_strided_batched<T>},
            {"trmv", testing_trmv<T>},
            {"trmv_batched", testing_trmv_batched<T>},
            {"trmv_strided_batched", testing_trmv_strided_batched<T>},
            {"trsv", testing_trsv<T>},
            {"trsv_batched", testing_trsv_batched<T>},
            {"trsv_strided_batched", testing_trsv_strided_batched<T>},
            // L3
            {"dgmm", testing_dgmm<T>},
            {"dgmm_batched", testing_dgmm_batched<T>},
            {"dgmm_strided_batched", testing_dgmm_strided_batched<T>},
            {"gemmt", testing_gemmt<T>},
            {"gemmt_batched", testing_gemmt_batched<T>},
            {"gemmt_strided_batched", testing_gemmt_strided_batched<T>},
            {"geam", testing_geam<T>},
            {"geam_batched", testing_geam_batched<T>},
            {"geam_strided_batched", testing_geam_strided_batched<T>},
            {"syrk", testing_syrk<T>},
            {"syrk_batched", testing_syrk_batched<T>},
            {"syrk_strided_batched", testing_syrk_strided_batched<T>},
            {"syr2k", testing_syr2k<T>},
            {"syr2k_batched", testing_syr2k_batched<T>},
            {"syr2k_strided_batched", testing_syr2k_strided_batched<T>},
            {"syrkx", testing_syr2k<T, false>},
            {"syrkx_batched", testing_syr2k_batched<T, false>},
            {"syrkx_strided_batched", testing_syr2k_strided_batched<T, false>},
            {"symm", testing_symm_hemm<T, false>},
            {"symm_batched", testing_symm_hemm_batched<T, false>},
            {"symm_strided_batched", testing_symm_hemm_strided_batched<T, false>},
            {"trmm", testing_trmm<T>},
            {"trmm_batched", testing_trmm_batched<T>},
            {"trmm_strided_batched", testing_trmm_strided_batched<T>},
            {"hemm", testing_symm_hemm<T, true>},
            {"hemm_batched", testing_symm_hemm_batched<T, true>},
            {"hemm_strided_batched", testing_symm_hemm_strided_batched<T, true>},
            {"herk", testing_herk<T>},
            {"herk_batched", testing_herk_batched<T>},
            {"herk_strided_batched", testing_herk_strided_batched<T>},
            {"her2k", testing_her2k<T>},
            {"her2k_batched", testing_her2k_batched<T>},
            {"her2k_strided_batched", testing_her2k_strided_batched<T>},
            {"herkx", testing_her2k<T, false>},
            {"herkx_batched", testing_her2k_batched<T, false>},
            {"herkx_strided_batched", testing_her2k_strided_batched<T, false>},
            {"gemm", testing_gemm<T>},
            {"gemm_batched", testing_gemm_batched<T>},
            {"gemm_strided_batched", testing_gemm_strided_batched<T>},
            {"trsm", testing_trsm<T>},
            {"trsm_ex", testing_trsm_ex<T>},
            {"trsm_batched", testing_trsm_batched<T>},
            {"trsm_batched_ex", testing_trsm_batched_ex<T>},
            {"trsm_strided_batched", testing_trsm_strided_batched<T>},
            {"trsm_strided_batched_ex", testing_trsm_strided_batched_ex<T>},
            {"trtri", testing_trtri<T>},
            {"trtri_batched", testing_trtri_batched<T>},
            {"trtri_strided_batched", testing_trtri_strided_batched<T>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty, typename = void>
struct perf_blas_axpy_ex : rocblas_test_invalid
{
};

template <typename Ta, typename Tx, typename Ty, typename Tex>
struct perf_blas_axpy_ex<
    Ta,
    Tx,
    Ty,
    Tex,
    std::enable_if_t<
        (std::is_same_v<
             Ta,
             float> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                double> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_half> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_float_complex> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_double_complex> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_half> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Ta,
                rocblas_half> && std::is_same_v<Ta, Tx> && std::is_same_v<Ty, Tex> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Ta,
                float> && std::is_same_v<Tx, rocblas_half> && std::is_same_v<Ta, Tex> && std::is_same_v<Tx, Ty>)
        || (std::is_same_v<
                Ta,
                rocblas_bfloat16> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Ta,
                float> && std::is_same_v<Tx, rocblas_bfloat16> && std::is_same_v<Tx, Ty> && std::is_same_v<Ta, Tex>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"axpy_ex", testing_axpy_ex<Ta, Tx, Ty, Tex>},
            {"axpy_batched_ex", testing_axpy_batched_ex<Ta, Tx, Ty, Tex>},
            {"axpy_strided_batched_ex", testing_axpy_strided_batched_ex<Ta, Tx, Ty, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, typename = void>
struct perf_blas_dot_ex : rocblas_test_invalid
{
};

template <typename Tx, typename Ty, typename Tr, typename Tex>
struct perf_blas_dot_ex<
    Tx,
    Ty,
    Tr,
    Tex,
    std::enable_if_t<
        (std::is_same_v<
             Tx,
             float> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                double> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_half> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_float_complex> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_double_complex> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_half> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Tx,
                rocblas_bfloat16> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tr> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Tx,
                float> && std::is_same_v<Tx, Ty> && std::is_same_v<Tr, double> && std::is_same_v<Tr, Tex>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"dot_ex", testing_dot_ex<Tx, Ty, Tr, Tex>},
            {"dot_batched_ex", testing_dot_batched_ex<Tx, Ty, Tr, Tex>},
            {"dot_strided_batched_ex", testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex>},
            {"dotc_ex", testing_dotc_ex<Tx, Ty, Tr, Tex>},
            {"dotc_batched_ex", testing_dotc_batched_ex<Tx, Ty, Tr, Tex>},
            {"dotc_strided_batched_ex", testing_dotc_strided_batched_ex<Tx, Ty, Tr, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Tr = Tx, typename Tex = Tr, typename = void>
struct perf_blas_nrm2_ex : rocblas_test_invalid
{
};

template <typename Tx, typename Tr, typename Tex>
struct perf_blas_nrm2_ex<
    Tx,
    Tr,
    Tex,
    std::enable_if_t<
        (std::is_same_v<Tx, float> && std::is_same_v<Tx, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<Tx, double> && std::is_same_v<Tx, Tr> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_float_complex> && std::is_same_v<Tr, float> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_double_complex> && std::is_same_v<Tr, double> && std::is_same_v<Tr, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_bfloat16> && std::is_same_v<Tx, Tr> && std::is_same_v<Tex, float>)
        || (std::is_same_v<Tx,
                           rocblas_half> && std::is_same_v<Tr, Tx> && std::is_same_v<Tex, float>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"nrm2_ex", testing_nrm2_ex<Tx, Tr>},
            {"nrm2_batched_ex", testing_nrm2_batched_ex<Tx, Tr>},
            {"nrm2_strided_batched_ex", testing_nrm2_strided_batched_ex<Tx, Tr>},
        };
        run_function(map, arg);
    }
};

template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_blas_rot : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_blas_rot<
    Ti,
    To,
    Tc,
    std::enable_if_t<
        (std::is_same_v<Ti, float> && std::is_same_v<Ti, To> && std::is_same_v<To, Tc>)
        || (std::is_same_v<Ti, double> && std::is_same_v<Ti, To> && std::is_same_v<To, Tc>)
        || (std::is_same_v<
                Ti,
                rocblas_float_complex> && std::is_same_v<To, float> && std::is_same_v<Tc, rocblas_float_complex>)
        || (std::is_same_v<
                Ti,
                rocblas_float_complex> && std::is_same_v<To, float> && std::is_same_v<Tc, float>)
        || (std::is_same_v<
                Ti,
                rocblas_double_complex> && std::is_same_v<To, double> && std::is_same_v<Tc, rocblas_double_complex>)
        || (std::is_same_v<
                Ti,
                rocblas_double_complex> && std::is_same_v<To, double> && std::is_same_v<Tc, double>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rot", testing_rot<Ti, To, Tc>},
            {"rot_batched", testing_rot_batched<Ti, To, Tc>},
            {"rot_strided_batched", testing_rot_strided_batched<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs, typename = void>
struct perf_blas_rot_ex : rocblas_test_invalid
{
};

template <typename Tx, typename Ty, typename Tcs, typename Tex>
struct perf_blas_rot_ex<
    Tx,
    Ty,
    Tcs,
    Tex,
    std::enable_if_t<
        (std::is_same_v<
             Tx,
             float> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tcs> && std::is_same_v<Tcs, Tex>)
        || (std::is_same_v<
                Tx,
                double> && std::is_same_v<Ty, Tx> && std::is_same_v<Ty, Tcs> && std::is_same_v<Tex, Tcs>)
        || (std::is_same_v<
                Tx,
                rocblas_float_complex> && std::is_same_v<Ty, Tx> && std::is_same_v<Tcs, Ty> && std::is_same_v<Tcs, Tex>)
        || (std::is_same_v<
                Tx,
                rocblas_double_complex> && std::is_same_v<Tx, Ty> && std::is_same_v<Tcs, Ty> && std::is_same_v<Tex, Tcs>)
        || (std::is_same_v<
                Tx,
                rocblas_float_complex> && std::is_same_v<Ty, Tx> && std::is_same_v<Tcs, float> && std::is_same_v<Tex, rocblas_float_complex>)
        || (std::is_same_v<
                Tx,
                rocblas_double_complex> && std::is_same_v<Tx, Ty> && std::is_same_v<Tcs, double> && std::is_same_v<Tex, rocblas_double_complex>)
        || (std::is_same_v<
                Tx,
                rocblas_half> && std::is_same_v<Ty, Tx> && std::is_same_v<Tcs, Ty> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Tx,
                rocblas_bfloat16> && std::is_same_v<Ty, Tx> && std::is_same_v<Tcs, Ty> && std::is_same_v<Tex, float>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rot_ex", testing_rot_ex<Tx, Ty, Tcs, Tex>},
            {"rot_batched_ex", testing_rot_batched_ex<Tx, Ty, Tcs, Tex>},
            {"rot_strided_batched_ex", testing_rot_strided_batched_ex<Tx, Ty, Tcs, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Ta = Tx, typename = void>
struct perf_blas_scal : rocblas_test_invalid
{
};

template <typename Tx, typename Ta>
struct perf_blas_scal<
    Tx,
    Ta,
    std::enable_if_t<(std::is_same_v<Ta, double> && std::is_same_v<Tx, rocblas_double_complex>)
                     || (std::is_same_v<Ta, float> && std::is_same_v<Tx, rocblas_float_complex>)
                     || (std::is_same_v<Tx, Ta> && std::is_same_v<Tx, float>)
                     || (std::is_same_v<Tx, Ta> && std::is_same_v<Tx, double>)
                     || (std::is_same_v<Tx, Ta> && std::is_same_v<Tx, rocblas_float_complex>)
                     || (std::is_same_v<Tx, Ta> && std::is_same_v<Tx, rocblas_double_complex>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            // Tx, Ta is reversed compared to scal_ex order as alpha is second argument
            {"scal", testing_scal<Tx, Ta>},
            {"scal_batched", testing_scal_batched<Tx, Ta>},
            {"scal_strided_batched", testing_scal_strided_batched<Tx, Ta>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tx = Ta, typename Tex = Tx, typename = void>
struct perf_blas_scal_ex : rocblas_test_invalid
{
};

template <typename Ta, typename Tx, typename Tex>
struct perf_blas_scal_ex<
    Ta,
    Tx,
    Tex,
    std::enable_if_t<
        (std::is_same_v<Ta, float> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Tex>)
        || (std::is_same_v<Ta, double> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Tex>)
        || (std::is_same_v<Ta, rocblas_half> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_float_complex> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_double_complex> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Tex>)
        || (std::is_same_v<Ta,
                           rocblas_half> && std::is_same_v<Ta, Tx> && std::is_same_v<Tex, float>)
        || (std::is_same_v<Ta,
                           float> && std::is_same_v<Tx, rocblas_half> && std::is_same_v<Ta, Tex>)
        || (std::is_same_v<
                Ta,
                rocblas_bfloat16> && std::is_same_v<Ta, Tx> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Ta,
                float> && std::is_same_v<Tx, rocblas_bfloat16> && std::is_same_v<Ta, Tex>)
        || (std::is_same_v<
                Ta,
                float> && std::is_same_v<Tx, rocblas_float_complex> && std::is_same_v<Tx, Tex>)
        || (std::is_same_v<
                Ta,
                double> && std::is_same_v<Tx, rocblas_double_complex> && std::is_same_v<Tx, Tex>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"scal_ex", testing_scal_ex<Ta, Tx, Tex>},
            {"scal_batched_ex", testing_scal_batched_ex<Ta, Tx, Tex>},
            {"scal_strided_batched_ex", testing_scal_strided_batched_ex<Ta, Tx, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename = void>
struct perf_blas_rotg : rocblas_test_invalid
{
};

template <typename Ta, typename Tb, typename Tc>
struct perf_blas_rotg<
    Ta,
    Tb,
    Tc,
    std::enable_if_t<
        (std::is_same_v<
             Ta,
             rocblas_double_complex> && std::is_same_v<Tb, double> && std::is_same_v<Ta, Tc>)
        || (std::is_same_v<
                Ta,
                rocblas_float_complex> && std::is_same_v<Tb, float> && std::is_same_v<Ta, Tc>)
        || (std::is_same_v<Ta, Tb> && std::is_same_v<Ta, float> && std::is_same_v<Ta, Tc>)
        || (std::is_same_v<Ta, Tb> && std::is_same_v<Ta, double> && std::is_same_v<Ta, Tc>)>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rotg", testing_rotg<Ta, Tb>},
            {"rotg_batched", testing_rotg_batched<Ta, Tb>},
            {"rotg_strided_batched", testing_rotg_strided_batched<Ta, Tb>},
        };
        run_function(map, arg, " --b_type "s + rocblas_datatype2string(arg.b_type));
    }
};

//
// globals, functions, and main()

// unit check override
int8_t g_unit_check = 0;

void gemm_arg_adjust(Arguments& arg, bool any_stride)
{
    // adjust dimension for GEMM routines
    int64_t min_lda = arg.transA == 'N' ? arg.M : arg.K;
    int64_t min_ldb = arg.transB == 'N' ? arg.K : arg.N;
    int64_t min_ldc = arg.M;
    int64_t min_ldd = arg.M;

    bool is_ex = strstr(arg.function, "_ex");

    if(arg.lda < min_lda)
    {
        rocblas_cout << "rocblas-bench INFO: lda < min_lda, setting lda = " << min_lda << std::endl;
        arg.lda = min_lda;
    }
    if(arg.ldb < min_ldb)
    {
        rocblas_cout << "rocblas-bench INFO: ldb < min_ldb, setting ldb = " << min_ldb << std::endl;
        arg.ldb = min_ldb;
    }
    if(arg.ldc < min_ldc)
    {
        rocblas_cout << "rocblas-bench INFO: ldc < min_ldc, setting ldc = " << min_ldc << std::endl;
        arg.ldc = min_ldc;
    }
    if(is_ex && arg.ldd < min_ldd)
    {
        rocblas_cout << "rocblas-bench INFO: ldd < min_ldd, setting ldd = " << min_ldd << std::endl;
        arg.ldd = min_ldd;
    }

    if(strstr(arg.function, "strided"))
    {
        rocblas_stride min_stride_c = arg.ldc * arg.N;
        rocblas_stride min_stride_d = arg.ldd * arg.N;
        if(!any_stride && arg.stride_c < min_stride_c)
        {
            rocblas_cout << "rocblas-bench INFO: stride_c < min_stride_c, setting stride_c = "
                         << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }
        if(is_ex && !any_stride && arg.stride_d < min_stride_d)
        {
            rocblas_cout << "rocblas-bench INFO: stride_d < min_stride_d, setting stride_d = "
                         << min_stride_d << std::endl;
            arg.stride_d = min_stride_d;
        }
    }

    if(!strstr(arg.function, "batched") && arg.batch_count > 1)
    {
        rocblas_cout << "rocblas-bench INFO: batch_count can only be 1 for non batched functions"
                     << ", setting batch_count = 1" << std::endl;
        arg.batch_count = 1;
    }

    if(!is_ex && arg.solution_index)
    {
        rocblas_cout << "rocblas-bench INFO: solution_index only supported by gemm_ex, "
                     << "gemm_batched_ex, gemm_strided_batched_ex"
                     << ", setting solution_index = 0" << std::endl;
        arg.solution_index = 0;
    }
}

int run_bench_test(bool               init,
                   Arguments&         arg,
                   const std::string& filter,
                   const std::string& name_filter,
                   bool               any_stride,
                   bool               yaml = false)
{
    if(init)
    {
        static int runOnce = (rocblas_parallel_initialize(1), 0); // Initialize rocBLAS
    }

    rocblas_cout << std::setiosflags(std::ios::fixed)
                 << std::setprecision(7); // Set precision to 7 digits

    // unit_check was forced off by default, so check global flag for enablement
    if(g_unit_check)
        arg.unit_check = g_unit_check;
    else
        arg.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    arg.timing = 1;

    // defeat memory padding for benchmarks
    arg.pad = 0;

    // One stream and one thread (0 indicates to use default behavior)
    arg.streams = 0;
    arg.threads = 0;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    const char*           name     = arg.name;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

    if(yaml && strstr(function, "_bad_arg"))
        return 0;
    if(!filter.empty())
    {
        if(!strstr(function, filter.c_str()))
            return 0;
    }
    if(!name_filter.empty())
    {
        if(!strstr(name, name_filter.c_str()))
            return 0;
    }

    // argument modifications
    if(!strcmp(function, "gemm") || !strcmp(function, "gemm_batched")
       || !strcmp(function, "gemm_strided_batched"))
    {
        gemm_arg_adjust(arg, any_stride);
    }

    // dispatch
    if(!strcmp(function, "gemm_ex") || !strcmp(function, "gemm_batched_ex"))
    {
        gemm_arg_adjust(arg, any_stride);

        rocblas_gemm_dispatch<perf_gemm_ex>(arg);
    }
    else if(!strcmp(function, "gemm_strided_batched_ex"))
    {
        gemm_arg_adjust(arg, any_stride);

        rocblas_gemm_dispatch<perf_gemm_strided_batched_ex>(arg);
    }
    else
    {
        if(!strcmp(function, "scal") || !strcmp(function, "scal_batched")
           || !strcmp(function, "scal_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_scal>(arg);
        else if(!strcmp(function, "rotg") || !strcmp(function, "rotg_batched")
                || !strcmp(function, "rotg_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_rotg>(arg);
        else if(!strcmp(function, "rot") || !strcmp(function, "rot_batched")
                || !strcmp(function, "rot_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_rot>(arg);
        else if(!strcmp(function, "rot_ex") || !strcmp(function, "rot_batched_ex")
                || !strcmp(function, "rot_strided_batched_ex"))
            rocblas_blas1_ex_dispatch<perf_blas_rot_ex>(arg);
        else if(!strcmp(function, "axpy_ex") || !strcmp(function, "axpy_batched_ex")
                || !strcmp(function, "axpy_strided_batched_ex"))
            rocblas_blas1_ex_dispatch<perf_blas_axpy_ex>(arg);
        else if(!strcmp(function, "dot_ex") || !strcmp(function, "dot_batched_ex")
                || !strcmp(function, "dot_strided_batched_ex") || !strcmp(function, "dotc_ex")
                || !strcmp(function, "dotc_batched_ex")
                || !strcmp(function, "dotc_strided_batched_ex"))
            rocblas_blas1_ex_dispatch<perf_blas_dot_ex>(arg);
        else if(!strcmp(function, "nrm2_ex") || !strcmp(function, "nrm2_batched_ex")
                || !strcmp(function, "nrm2_strided_batched_ex"))
            rocblas_blas1_ex_dispatch<perf_blas_nrm2_ex>(arg);
        else if(!strcmp(function, "scal_ex") || !strcmp(function, "scal_batched_ex")
                || !strcmp(function, "scal_strided_batched_ex"))
            rocblas_blas1_ex_dispatch<perf_blas_scal_ex>(arg);
        else if(!strcmp(function, "gemv_batched") || !strcmp(function, "gemv_strided_batched"))
            rocblas_gemv_batched_and_strided_batched_dispatch<
                perf_gemv_batched_and_strided_batched>(arg);
        else
            rocblas_simple_dispatch<perf_blas>(arg);
    }
    return 0;
}

void check_device_matrix_reuse(std::vector<Arguments>& args)
{
    if(args.empty())
    {
        return;
    }
    auto* arg_prev = &args.front();
    for(auto& arg : args)
    {
        if(arg.a_type == arg_prev->a_type && arg.b_type == arg_prev->b_type
           && arg.c_type == arg_prev->c_type && arg.d_type == arg_prev->d_type
           && arg.initialization == arg_prev->initialization)
        {
            arg_prev->cleanup = false;
        }
        arg_prev = &arg;
    }
}

int rocblas_bench_datafile(const std::string& filter,
                           const std::string& name_filter,
                           bool               any_stride)
{
    int                    ret      = 0;
    auto                   arg_iter = RocBLAS_TestData();
    std::vector<Arguments> args{arg_iter.begin(), arg_iter.end()};

    check_device_matrix_reuse(args);

    for(Arguments arg : args)
    {
        ret |= run_bench_test(true, arg, filter, name_filter, any_stride, true);
    }
    test_cleanup::cleanup();
    return ret;
}

void gpu_thread_init_device(int                id,
                            const Arguments&   arg,
                            const std::string& filter,
                            bool               any_stride)
{
    CHECK_HIP_ERROR(hipSetDevice(id));

    Arguments   a(arg);
    std::string name_filter = "";
    a.cold_iters            = 1;
    a.iters                 = 0;
    run_bench_test(false, a, filter, name_filter, any_stride, false);
}

void gpu_thread_run_bench(int id, const Arguments& arg, const std::string& filter, bool any_stride)
{
    CHECK_HIP_ERROR(hipSetDevice(id));

    Arguments   a(arg);
    std::string name_filter = "";
    run_bench_test(false, a, filter, name_filter, any_stride, false);
}

void run_bench_gpu_test(int                parallel_devices,
                        Arguments&         arg,
                        const std::string& filter,
                        bool               any_stride)
{
    int count;
    CHECK_HIP_ERROR(hipGetDeviceCount(&count));

    if(parallel_devices > count || parallel_devices < 1)
        GTEST_ASSERT_TRUE(false);

    // initialization
    rocblas_parallel_initialize(parallel_devices);

    // run cold call on each device
    auto thread_init = std::make_unique<std::thread[]>(parallel_devices);

    for(int id = 0; id < parallel_devices; ++id)
        thread_init[id] = std::thread(::gpu_thread_init_device, id, arg, filter, any_stride);

    for(int id = 0; id < parallel_devices; ++id)
        thread_init[id].join();

    // synchronized launch of cold & hot calls
    auto thread = std::make_unique<std::thread[]>(parallel_devices);

    for(int id = 0; id < parallel_devices; ++id)
        thread[id] = std::thread(::gpu_thread_run_bench, id, arg, filter, any_stride);

    for(int id = 0; id < parallel_devices; ++id)
        thread[id].join();
}

// Replace --batch with --batch_count for backward compatibility
void fix_batch(int argc, char* argv[])
{
    static char b_c[] = "--batch_count";
    for(int i = 1; i < argc; ++i)
        if(!strcmp(argv[i], "--batch"))
        {
            static int once = (rocblas_cerr << argv[0]
                                            << " warning: --batch is deprecated, and --batch_count "
                                               "should be used instead."
                                            << std::endl,
                               0);
            argv[i]         = b_c;
        }
}

int main(int argc, char* argv[])
try
{
    client_omp_manager::limit_by_processor_count();
    rocblas_client_init();

    fix_batch(argc, argv);
    Arguments   arg;
    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string initialization;
    std::string filter;
    std::string name_filter;
    int32_t     device_id           = 0;
    int32_t     parallel_devices    = 0;
    int32_t     flags               = 0;
    int32_t     geam_ex_op          = 0;
    int32_t     api                 = 0;
    bool        datafile            = rocblas_parse_data(argc, argv);
    bool        atomics_allowed     = false;
    bool        atomics_not_allowed = true;
    bool        log_function_name   = false;
    bool        log_datatype        = false;
    bool        any_stride          = false;
    uint32_t    math_mode           = 0;
    uint64_t    flush_batch_count   = 1;
    uint64_t    flush_memory_size   = 0;
    bool        fortran             = false;
    bool        eazy                = false;

    arg.init(); // set all defaults

    options_description desc("rocblas-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<int64_t>(&arg.M)->default_value(128),
         "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows or columns in matrix.")

        ("sizen,n",
         value<int64_t>(&arg.N)->default_value(128),
         "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of rows or columns in matrix")

        ("sizek,k",
         value<int64_t>(&arg.K)->default_value(128),
         "Specific matrix size: BLAS-2: the number of sub or super-diagonals of A. BLAS-3: "
         "the number of columns in A and rows in B.")

        ("kl",
         value<int64_t>(&arg.KL)->default_value(32),
         "Specific matrix size: kl is only applicable to BLAS-2: The number of sub-diagonals "
         "of the banded matrix A.")

        ("ku",
         value<int64_t>(&arg.KU)->default_value(32),
         "Specific matrix size: ku is only applicable to BLAS-2: The number of super-diagonals "
         "of the banded matrix A.")

        ("lda",
         value<int64_t>(&arg.lda)->default_value(128),
         "Leading dimension of matrix A, is only applicable to BLAS-2 & BLAS-3.")

        ("ldb",
         value<int64_t>(&arg.ldb)->default_value(128),
         "Leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3.")

        ("ldc",
         value<int64_t>(&arg.ldc)->default_value(128),
         "Leading dimension of matrix C, is only applicable to BLAS-2 & BLAS-3.")

        ("ldd",
         value<int64_t>(&arg.ldd)->default_value(128),
         "Leading dimension of matrix D, is only applicable to BLAS-EX ")

        ("any_stride",
         value<bool>(&any_stride)->default_value(false),
         "Do not modify input strides based on leading dimensions")

        ("stride_a",
         value<rocblas_stride>(&arg.stride_a)->default_value(128*128),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_b",
         value<rocblas_stride>(&arg.stride_b)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_c",
         value<rocblas_stride>(&arg.stride_c)->default_value(128*128),
         "Specific stride of strided_batched matrix C, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_d",
         value<rocblas_stride>(&arg.stride_d)->default_value(128*128),
         "Specific stride of strided_batched matrix D, is only applicable to strided batched"
         "BLAS_EX: second dimension * leading dimension.")

        ("stride_x",
         value<rocblas_stride>(&arg.stride_x)->default_value(128*128),
         "Specific stride of strided_batched vector x, is only applicable to strided batched"
         "BLAS_2: second dimension.")

        ("stride_y",
         value<rocblas_stride>(&arg.stride_y)->default_value(128*128),
         "Specific stride of strided_batched vector y, is only applicable to strided batched"
         "BLAS_2: leading dimension.")

        ("incx",
         value<int64_t>(&arg.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<int64_t>(&arg.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("alphai",
         value<double>(&arg.alphai)->default_value(0.0), "specifies the imaginary part of the scalar alpha")

        ("beta",
         value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("betai",
         value<double>(&arg.betai)->default_value(0.0), "specifies the imaginary part of the scalar beta")

        ("function,f",
         value<std::string>(&function),
         "BLAS function to test.")

        ("precision,r",
         value<std::string>(&precision)->default_value("f32_r"), "Precision. "
         "Options: h,s,d,c,z, f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("compute_type",
         value<std::string>(&compute_type), "Precision of computation. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("initialization, init",
         value<std::string>(&initialization)->default_value("hpl"),
         "Initialize with random integers, trig functions sin and cos, or hpl-like input. "
         "Options: rand_int, trig_float, hpl")

        ("transposeA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         value<char>(&arg.side)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         value<char>(&arg.uplo)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm xtrsm_ex
                                                                     // xtrmm xtrsv
        ("diag",
         value<char>(&arg.diag)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm xtrsm_ex xtrsv xtrmm

        ("batch_count",
         value<int64_t>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines")

        ("HMM",
         value<bool>(&arg.HMM)->default_value(false),
         "Parameter requesting the use of HipManagedMemory")

        ("test,t",
         value<int8_t>(&g_unit_check)->default_value(0),
         "Equality/tolerance check on GPU vs. CPU results? 0 = No, 1 = Yes (default: No). "
         "Warning: tests may require specific initialization to pass.")

        ("verify,v",
         value<int8_t>(&arg.norm_check)->default_value(0),
         "Norm check on GPU vs. CPU results? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<int32_t>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("cold_iters,j",
         value<int32_t>(&arg.cold_iters)->default_value(2),
         "Cold Iterations to run before entering the timing loop")

        ("algo",
         value<uint32_t>(&arg.algo)->default_value(0),
         "extended precision gemm algorithm")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(0),
         "extended precision gemm solution index")

        ("flags",
         value<int32_t>(&flags)->default_value(rocblas_gemm_flags_none),
         "gemm_ex flags, see rocblas_gemm_flags enum in documentation")

        ("geam_ex_op",
         value<int32_t>(&geam_ex_op)->default_value(rocblas_geam_ex_operation_min_plus),
         "geam_ex_operation, 0: min_plus operation, 1: plus_min operation")

        ("atomics_allowed",
         bool_switch(&atomics_allowed)->default_value(false),
         "Atomic operations with non-determinism in results are allowed (default false)")

        ("atomics_not_allowed",
         bool_switch(&atomics_not_allowed)->default_value(true),
         "Atomic operations with non-determinism in results are not allowed (default true)")

        ("device",
         value<int32_t>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("parallel_devices",
         value<int32_t>(&parallel_devices)->default_value(0),
         "Set number of devices used for parallel runs (device 0 to parallel_devices-1)")

        ("outofplace",
         bool_switch(&arg.outofplace)->default_value(false),
         "for gemm_ex C and D are stored in separate memory, for trmm B and C are stored in separate memory")

        ("fortran",
         bool_switch(&fortran)->default_value(false),
         "Run using Fortran interface")

        ("api",
         value<int32_t>(&api)->default_value(0),
         "Use API, supercedes fortran flag (0==C, 1==C_64, ...)")

        ("workspace",
         value<size_t>(&arg.user_allocated_workspace)->default_value(0),
         "Set fixed workspace memory size instead of using rocblas managed memory")

        ("log_function_name",
         bool_switch(&log_function_name)->default_value(false),
         "Function name precedes other items.")

         ("log_datatype",
         bool_switch(&log_datatype)->default_value(false),
         "Include datatypes used in output.")

        ("use_hipblaslt",
         value<int32_t>(&arg.use_hipblaslt)->default_value(-1),
         "Whether to use hipblaslt (default: -1, always: 1, never: 0)")

        ("function_filter",
         value<std::string>(&filter),
         "Simple strstr filter on function name only without wildcards")

        ("math_mode",
         value<uint32_t>(&arg.math_mode)->default_value(rocblas_default_math),
         "extended precision gemm math mode")

        ("flush_batch_count",
         value<uint64_t>(&arg.flush_batch_count)->default_value(1),
         "number of copies of arrays to allocate for cache flushing in timing code. Functions"
         " are called iters times in a timing loop. If the problem memory footprint is small"
         " enough, then arrays will be cached. flush_batch_count can be used to prevent caching."
         " For example, for sgemm with transA=transB=N:"
         " problem_memory_footprint = (m*k + k*n + m*n) * sizeof(float)."
         " To flush arrays before reuse set:"
         " flush_batch_count >= 1 + cache_size / problem_memory_footprint"
         " Note that in the calculation of flush_batch_count any padding from leading"
         " dimensions is not loaded to cache and not included in the problem_memory_footprint."
         " If you specify flush_batch_count you cannot also specify flush_memory_size")

        ("flush_memory_size",
         value<uint64_t>(&arg.flush_memory_size)->default_value(0),
         "bytes of memory that will be occupied by arrays. Used only in timing code for cache flushing. Set to greater than"
         " cache size so arrays are flushed from cache before they are reused. When the size of arrays (the problem_memory_footprint)"
         " is smaller than flush_memory_size, then flush_batch_count copies of arrays are allocated where:"
         " flush_batch_count = flush_memory_size / problem_memory_footprint."
         " For sgemm with transA=transB=N"
         " problem_memory_footprint = (m*k + k*n + m*n) * sizeof(float). Note that any padding from leading"
         " dimensions is not loaded to cache and not included in the problem_memory_footprint."
         " If you specify flush_memory_size you cannot also specify flush_batch_count")

        ("name_filter",
         value<std::string>(&name_filter),
         "Simple strstr filter on test name only without wildcards, only used with --yaml or --data")

        ("version", "Prints the version number")

        ("rocblas_tensile_commit_hash", "Prints the rocBLAS and Tensile commit-hashes")

        ("z",
         value<bool>(&eazy)->default_value(false),
         "Eazy way to set log_function_name and log_datatype (eye dialect form of easy)")

        ("help,h", "produces this help message");
    // clang-format on

    // parse command line into arg structure and stack variables using desc
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if((argc <= 1 && !datafile) || vm.count("help"))
    {
        rocblas_cout << desc << std::endl;
        rocblas_cout << "Examples : ./rocblas-bench -f gemm -r s -m 4000 -n 4000 -k 4000 --lda "
                        "4000 --ldb 4000 --ldc 4000 --transposeA N --transposeB T"
                     << std::endl
                     << std::endl
                     << "\t   "
                     << "./rocblas-bench -f gemv -r s -m 10240 -n 10240 --lda 10240" << std::endl
                     << std::endl
                     << "\t   "
                     << "./rocblas-bench -f axpy -r d -n 102400000" << std::endl;
        return 0;
    }

    print_rocblas_version_string(); // version and commit hash
    if(vm.find("version") != vm.end() || vm.find("rocblas_tensile_commit_hash") != vm.end())
    {
        return 0;
    }

    // transfer local variable state
    arg.atomics_mode = rocblas_atomics_not_allowed;
    // two command line options so if either changed from default enable atomics
    if(atomics_allowed == true || atomics_not_allowed == false)
        arg.atomics_mode = rocblas_atomics_allowed;

    if(api)
        arg.api = rocblas_client_api(api);
    else if(fortran)
        arg.api = FORTRAN;

    static const char* fp16AltImplEnvStr = std::getenv("ROCBLAS_INTERNAL_FP16_ALT_IMPL");
    static const int   fp16AltImplEnv
        = (fp16AltImplEnvStr == NULL ? -1 : (std::atoi(fp16AltImplEnvStr) == 0 ? 0 : 1));
    if(fp16AltImplEnv != -1)
    {
        if(fp16AltImplEnv == 0)
            flags &= ~rocblas_gemm_flags_fp16_alt_impl;
        else
            flags |= rocblas_gemm_flags_fp16_alt_impl;
    }

    static const char* fp16AltImplRoundEnvStr = std::getenv("ROCBLAS_INTERNAL_FP16_ALT_IMPL_RNZ");
    static const int   fp16AltImplRoundEnv
        = (fp16AltImplRoundEnvStr == NULL ? -1 : (std::atoi(fp16AltImplRoundEnvStr) == 0 ? 0 : 1));
    if(fp16AltImplRoundEnv != -1)
    {
        if(fp16AltImplRoundEnv == 0)
            flags &= ~rocblas_gemm_flags_fp16_alt_impl_rnz;
        else
            flags |= rocblas_gemm_flags_fp16_alt_impl_rnz;
    }

    if(rocblas_internal_get_arch_name() != "gfx90a")
    {
        flags &= ~rocblas_gemm_flags_fp16_alt_impl;
        flags &= ~rocblas_gemm_flags_fp16_alt_impl_rnz;
    }

    arg.flags = rocblas_gemm_flags(flags);

    arg.geam_ex_op = rocblas_geam_ex_operation(geam_ex_op);

    if(eazy)
    {
        log_function_name = true;
        log_datatype      = true;
    }

    ArgumentModel_set_log_function_name(log_function_name);

    ArgumentModel_set_log_datatype(log_datatype);

    // Device Query
    rocblas_int device_count = query_device_property();

    rocblas_cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    if(device_id >= 0)
        set_device(device_id);

    FrequencyMonitor& freq_monitor = getFrequencyMonitor();
    freq_monitor.set_device_id(device_id);

    if(datafile)
        return rocblas_bench_datafile(filter, name_filter, any_stride);

    // single bench run

    // validate arguments

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2rocblas_datatype(precision);
    if(prec == rocblas_datatype_invalid)
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string2rocblas_datatype(a_type);
    if(arg.a_type == rocblas_datatype_invalid)
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string2rocblas_datatype(b_type);
    if(arg.b_type == rocblas_datatype_invalid)
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string2rocblas_datatype(c_type);
    if(arg.c_type == rocblas_datatype_invalid)
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string2rocblas_datatype(d_type);
    if(arg.d_type == rocblas_datatype_invalid)
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    arg.compute_type = compute_type == "" ? prec : string2rocblas_datatype(compute_type);
    if(arg.compute_type == rocblas_datatype_invalid)
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    arg.initialization = string2rocblas_initialization(initialization);
    if(arg.initialization == static_cast<rocblas_initialization>(0)) // zero not in enum
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));
    if(arg.norm_check < 0 || arg.norm_check > 1)
        throw std::invalid_argument("Invalid value for -v or --verify "
                                    + std::to_string(arg.norm_check));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    if(!parallel_devices)
    {
        std::string name_filter = "";
        run_bench_test(true, arg, filter, name_filter, any_stride);
    }
    else
        run_bench_gpu_test(parallel_devices, arg, filter, any_stride);

    freeFrequencyMonitor();

    int status = 0;
    // TODO: query for any failed tests

    rocblas_client_shutdown();

    return status;
}
catch(const std::invalid_argument& exp)
{
    rocblas_cerr << exp.what() << std::endl;
    return -1;
}
