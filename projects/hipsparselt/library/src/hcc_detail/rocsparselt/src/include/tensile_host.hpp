/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
/*********************************************************
 * Declaration of the rocSPARSELt<->Tensile interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: Tensile-specific data types, functions and macros should only be *
 * referenced from tensile_host.cpp. This header file defines the interface  *
 * that the rest of rocSPARSELt uses to access Tensile. If another Tensile       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any Tensile-specific identifiers here.  *
 *****************************************************************************/

#include "activation.hpp"
#include "handle.h"
#include "tuple_helper.hpp"
#include "utility.hpp"
#include <atomic>

/********************************************************************
 * RocsparseltContractionProblem captures the arguments for a GEMM-like *
 * contraction problem, to be passed to runContractionProblem.      *
 ********************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
struct RocsparseltContractionProblem
{
    const _rocsparselt_handle* handle;
    rocsparselt_operation      trans_a;
    rocsparselt_operation      trans_b;
    rocsparselt_order          order;

    // The RocsparseltContractionProblem data members should exactly match
    // Tensile's parameter types, even if rocSPARSELt uses differently
    // sized or signed types. The constructors should convert rocSPARSELt
    // types into the corresponding Tensile types stored in this class.
    size_t m;
    size_t n;
    size_t k;

    const Tc* alpha;

    const Ti*        A;
    const Ti* const* batch_A;
    size_t           row_stride_a;
    size_t           col_stride_a;
    size_t           batch_stride_a;
    size_t           buffer_offset_a;

    const Ti*        B;
    const Ti* const* batch_B;
    size_t           row_stride_b;
    size_t           col_stride_b;
    size_t           batch_stride_b;
    size_t           buffer_offset_b;

    const Tc* beta;

    const To*        C;
    const To* const* batch_C;
    size_t           row_stride_c;
    size_t           col_stride_c;
    size_t           batch_stride_c;
    size_t           buffer_offset_c;

    To*        D;
    To* const* batch_D;
    size_t     row_stride_d;
    size_t     col_stride_d;
    size_t     batch_stride_d;
    size_t     buffer_offset_d;

    size_t batch_count;
    bool   strided_batch;

    bool                 sparseA;
    const unsigned char* metadata;

    hipsparselt_activation_type act_type;
    float                       act_arg0;
    float                       act_arg1;
    const void*                 bias_vector;
    int64_t                     bias_stride;
    hipDataType                 bias_type;
    bool                        alpha_vector_scaling;

    void*  workspace;
    size_t workspaceSize;

    hipStream_t* streams;
    int32_t      numStreams;

    // gemm
    // gemm_strided_batched
    RocsparseltContractionProblem(const _rocsparselt_handle*  handle,
                                  rocsparselt_operation       trans_a,
                                  rocsparselt_operation       trans_b,
                                  rocsparselt_order           order,
                                  int64_t                     m,
                                  int64_t                     n,
                                  int64_t                     k,
                                  const Tc*                   alpha,
                                  const Ti*                   A,
                                  const Ti* const*            batch_A,
                                  int64_t                     ld_a,
                                  int64_t                     batch_stride_a,
                                  int64_t                     offset_a,
                                  const Ti*                   B,
                                  const Ti* const*            batch_B,
                                  int64_t                     ld_b,
                                  int64_t                     batch_stride_b,
                                  int64_t                     offset_b,
                                  const Tc*                   beta,
                                  To*                         C,
                                  To* const*                  batch_C,
                                  int64_t                     ld_c,
                                  int64_t                     batch_stride_c,
                                  int64_t                     offset_c,
                                  int64_t                     batch_count,
                                  bool                        strided_batch,
                                  bool                        sparseA,
                                  const unsigned char*        metadata,
                                  hipsparselt_activation_type act_type,
                                  float                       act_arg0,
                                  float                       act_arg1,
                                  const void*                 bias_vector,
                                  int64_t                     bias_stride,
                                  hipDataType                 bias_type,
                                  bool                        alpha_vector_scaling,
                                  void*                       workspace,
                                  size_t                      workspaceSize,
                                  hipStream_t*                streams,
                                  int32_t                     numStreams)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , order(order)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(C)
        , batch_D(batch_C)
        , row_stride_d(1)
        , col_stride_d(ld_c)
        , batch_stride_d(batch_stride_c)
        , buffer_offset_d(offset_c)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
        , sparseA(sparseA)
        , metadata(metadata)
        , act_type(act_type)
        , act_arg0(act_arg0)
        , act_arg1(act_arg1)
        , bias_vector(bias_vector)
        , bias_stride(bias_stride)
        , bias_type(bias_type)
        , alpha_vector_scaling(alpha_vector_scaling)
        , workspace(workspace)
        , workspaceSize(workspaceSize)
        , streams(streams)
        , numStreams(numStreams)
    {
    }

    // gemm_ex
    // gemm_strided_batched_ex
    RocsparseltContractionProblem(const _rocsparselt_handle*  handle,
                                  rocsparselt_operation       trans_a,
                                  rocsparselt_operation       trans_b,
                                  rocsparselt_order           order,
                                  int64_t                     m,
                                  int64_t                     n,
                                  int64_t                     k,
                                  const Tc*                   alpha,
                                  const Ti*                   A,
                                  const Ti* const*            batch_A,
                                  int64_t                     ld_a,
                                  int64_t                     batch_stride_a,
                                  int64_t                     offset_a,
                                  const Ti*                   B,
                                  const Ti* const*            batch_B,
                                  int64_t                     ld_b,
                                  int64_t                     batch_stride_b,
                                  int64_t                     offset_b,
                                  const Tc*                   beta,
                                  const To*                   C,
                                  const To* const*            batch_C,
                                  int64_t                     ld_c,
                                  int64_t                     batch_stride_c,
                                  int64_t                     offset_c,
                                  To*                         D,
                                  To* const*                  batch_D,
                                  int64_t                     ld_d,
                                  int64_t                     batch_stride_d,
                                  int64_t                     offset_d,
                                  int64_t                     batch_count,
                                  bool                        strided_batch,
                                  bool                        sparseA,
                                  const unsigned char*        metadata,
                                  hipsparselt_activation_type act_type,
                                  float                       act_arg0,
                                  float                       act_arg1,
                                  const void*                 bias_vector,
                                  int64_t                     bias_stride,
                                  hipDataType                 bias_type,
                                  bool                        alpha_vector_scaling,
                                  void*                       workspace,
                                  size_t                      workspaceSize,
                                  hipStream_t*                streams,
                                  int32_t                     numStreams)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , order(order)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(D)
        , batch_D(batch_D)
        , row_stride_d(1)
        , col_stride_d(ld_d)
        , batch_stride_d(batch_stride_d)
        , buffer_offset_d(offset_d)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
        , sparseA(sparseA)
        , metadata(metadata)
        , act_type(act_type)
        , act_arg0(act_arg0)
        , act_arg1(act_arg1)
        , bias_vector(bias_vector)
        , bias_stride(bias_stride)
        , bias_type(bias_type)
        , alpha_vector_scaling(alpha_vector_scaling)
        , workspace(workspace)
        , workspaceSize(workspaceSize)
        , streams(streams)
        , numStreams(numStreams)
    {
    }

    // gemm_ext2
    // gemm_strided_batched_ext2
    RocsparseltContractionProblem(const _rocsparselt_handle*  handle,
                                  int64_t                     m,
                                  int64_t                     n,
                                  int64_t                     k,
                                  const Tc*                   alpha,
                                  const Ti*                   A,
                                  const Ti* const*            batch_A,
                                  int64_t                     row_stride_a,
                                  int64_t                     col_stride_a,
                                  int64_t                     batch_stride_a,
                                  int64_t                     offset_a,
                                  const Ti*                   B,
                                  const Ti* const*            batch_B,
                                  int64_t                     row_stride_b,
                                  int64_t                     col_stride_b,
                                  int64_t                     batch_stride_b,
                                  int64_t                     offset_b,
                                  const Tc*                   beta,
                                  const To*                   C,
                                  const To* const*            batch_C,
                                  int64_t                     row_stride_c,
                                  int64_t                     col_stride_c,
                                  int64_t                     batch_stride_c,
                                  int64_t                     offset_c,
                                  To*                         D,
                                  To* const*                  batch_D,
                                  int64_t                     row_stride_d,
                                  int64_t                     col_stride_d,
                                  int64_t                     batch_stride_d,
                                  int64_t                     offset_d,
                                  int64_t                     batch_count,
                                  bool                        strided_batch,
                                  bool                        sparseA,
                                  const unsigned char*        metadata,
                                  hipsparselt_activation_type act_type,
                                  float                       act_arg0,
                                  float                       act_arg1,
                                  const void*                 bias_vector,
                                  int64_t                     bias_stride,
                                  hipDataType                 bias_type,
                                  bool                        alpha_vector_scaling,
                                  void*                       workspace,
                                  size_t                      workspaceSize,
                                  hipStream_t*                streams,
                                  int32_t                     numStreams)
        : handle(handle)
        , trans_a(rocsparselt_operation_none)
        , trans_b(rocsparselt_operation_none)
        , order(rocsparselt_order_column)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(row_stride_a)
        , col_stride_a(col_stride_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(row_stride_b)
        , col_stride_b(col_stride_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(row_stride_c)
        , col_stride_c(col_stride_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(D)
        , batch_D(batch_D)
        , row_stride_d(row_stride_d)
        , col_stride_d(col_stride_d)
        , batch_stride_d(batch_stride_d)
        , buffer_offset_d(offset_d)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
        , sparseA(sparseA)
        , metadata(metadata)
        , act_type(act_type)
        , act_arg0(act_arg0)
        , act_arg1(act_arg1)
        , bias_vector(bias_vector)
        , bias_stride(bias_stride)
        , bias_type(bias_type)
        , alpha_vector_scaling(alpha_vector_scaling)
        , workspace(workspace)
        , workspaceSize(workspaceSize)
        , streams(streams)
        , numStreams(numStreams)
    {
    }

    /***************************************************
     * Print a RocsparseltContractionProblem for debugging *
     ***************************************************/
    friend hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream&        os,
                                                    const RocsparseltContractionProblem& prob)
    {
        return tuple_helper::print_tuple_pairs(
            os,
            std::make_tuple("a_type",
                            rocsparselt_precision_string<Ti>,
                            "b_type",
                            rocsparselt_precision_string<Ti>,
                            "c_type",
                            rocsparselt_precision_string<To>,
                            "d_type",
                            rocsparselt_precision_string<To>,
                            "compute_type",
                            rocsparselt_precision_string<Tc>,
                            "transA",
                            rocsparselt_transpose_letter(prob.trans_a),
                            "transB",
                            rocsparselt_transpose_letter(prob.trans_b),
                            "M",
                            prob.m,
                            "N",
                            prob.n,
                            "K",
                            prob.k,
                            "alpha",
                            *prob.alpha,
                            "row_stride_a",
                            prob.row_stride_a,
                            "col_stride_a",
                            prob.col_stride_a,
                            "row_stride_b",
                            prob.row_stride_b,
                            "col_stride_b",
                            prob.col_stride_b,
                            "row_stride_c",
                            prob.row_stride_c,
                            "col_stride_c",
                            prob.col_stride_c,
                            "row_stride_d",
                            prob.row_stride_d,
                            "col_stride_d",
                            prob.col_stride_d,
                            "beta",
                            *prob.beta,
                            "batch_count",
                            prob.batch_count,
                            "strided_batch",
                            prob.strided_batch,
                            "stride_a",
                            prob.batch_stride_a,
                            "stride_b",
                            prob.batch_stride_b,
                            "stride_c",
                            prob.batch_stride_c,
                            "stride_d",
                            prob.batch_stride_d,
                            "activation",
                            hipsparselt_activation_type_to_string(prob.act_type),
                            "activation_argument_0",
                            prob.act_arg0,
                            "activation_argument_1",
                            prob.act_arg1,
                            "has_bias",
                            (prob.bias_vector != nullptr) ? true : false,
                            "bias_stride",
                            prob.bias_stride,
                            "bias_type",
                            hip_datatype_to_string(prob.bias_type),
                            "alpha_vector_scaling",
                            prob.alpha_vector_scaling));
    };
};

/*******************************************************************************
 * runContractionProblem() solves a RocsparseltContractionProblem                  *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status runContractionProblem(RocsparseltContractionProblem<Ti, To, Tc> const& problem,
                                         _rocsparselt_matmul_config*                      configs,
                                         int*                                             config_id,
                                         const int config_max_id,
                                         const int search_iterations);

template <typename Ti, typename To, typename Tc>
rocsparselt_status getBestSolutions(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                    int                                              requestConfigs,
                                    _rocsparselt_matmul_config*                      configs,
                                    int*                                             foundConfigs);

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for testing) *
 ***********************************************************************************/
std::atomic_bool& rocsparselt_internal_tensile_is_initialized();

/**********************************************
 * Whether to suppress Tensile error messages *
 **********************************************/
inline bool& rocsparselt_suppress_tensile_error_messages()
{
    thread_local bool t_suppress = false;
    return t_suppress;
}
