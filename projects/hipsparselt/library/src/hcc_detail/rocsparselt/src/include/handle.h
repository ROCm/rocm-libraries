/*! \file */
/* ************************************************************************
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
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
#ifndef HANDLE_H
#define HANDLE_H

#include "rocsparselt.h"

#include <fstream>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp8.h>
#include <iostream>
#include <memory>
#include <vector>

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparselt library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
struct _rocsparselt_handle
{
    // Magic number stored in the inline struct to confirm valid initialization.
    static constexpr uintptr_t IMPL_MAGIC = 0x48414E444C455F4FULL; // "HANDLE_O"

    // constructor
    _rocsparselt_handle() {}
    // destructor
    ~_rocsparselt_handle() {}

    void init();
    void destroy();

    // device id
    int device;

    // device wavefront size
    int wavefront_size = 0;
    // asic revision
    int asic_rev;

    bool has_fp8_ocp  = false;
    bool has_fp8_fnuz = false;

    // pointer mode ; default mode is host
    rocsparselt_pointer_mode pointer_mode = rocsparselt_pointer_mode_host;
    // logging mode
    int  layer_mode;
    bool log_bench = false;

    // device buffer
    size_t    buffer_size;
    void*     buffer;
    uintptr_t is_init     = 0;  // IMPL_MAGIC when valid, 0 when destroyed
    uintptr_t ownership   = 0;  // self-referential for original; zeroed by destroy() to invalidate copies

    // logging streams
    std::ofstream* log_trace_ofs = nullptr;
    std::ofstream* log_bench_ofs = nullptr;
    std::ostream*  log_trace_os  = nullptr;
    std::ostream*  log_bench_os  = nullptr;
};

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparselt matrix
 * content. It must be initialized using rocsparselt_dense_descr_init() or
 * rocsparselt_structured_descr_init()  and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *******************************************************************************/
struct _rocsparselt_mat_descr
{
    static constexpr uintptr_t MAT_MAGIC = 0x4D41544445534352ULL; // "MATDESCR"

    // constructor
    _rocsparselt_mat_descr(const _rocsparselt_handle* handle)
        : handle(handle)
    {
        is_init = MAT_MAGIC;
    };
    _rocsparselt_mat_descr(const _rocsparselt_mat_descr& rhs)
        : handle(rhs.handle)
        , m_type(rhs.m_type)
        , m(rhs.m)
        , n(rhs.n)
        , ld(rhs.ld)
        , alignment(rhs.alignment)
        , type(rhs.type)
        , order(rhs.order)
        , sparsity(rhs.sparsity)
        , num_batches(rhs.num_batches)
        , batch_stride(rhs.batch_stride)
        , c_k(rhs.c_k)
        , c_ld(rhs.c_ld)
        , c_n(rhs.c_n)
    {
        is_init = MAT_MAGIC;
    };

    // destructor
    ~_rocsparselt_mat_descr()
    {
        clear();
    };

    _rocsparselt_mat_descr* clone() const
    {
        return new _rocsparselt_mat_descr(*this);
    };

    void clear()
    {
        // Reset both is_init and m_type for BOTH original and copy.
        // m_type = unknown is the extra guard in check_is_init_mat_descr; always
        // resetting it strengthens protection against false positives from stack residuals.
        // No heap resources → no double-free risk from clearing both.
        m_type  = rocsparselt_matrix_type_unknown;
        is_init = 0;
    }

    friend std::ostream& operator<<(std::ostream& stream, const _rocsparselt_mat_descr& t);

    const _rocsparselt_handle* handle = nullptr;
    // matrix type
    rocsparselt_matrix_type m_type  = rocsparselt_matrix_type_unknown;
    uintptr_t               is_init = 0;

    // num rows
    int64_t m = 0;
    // num cols
    int64_t n = 0;
    // leading dimension
    int64_t ld = 0;
    // memory alignment in bytes
    uint32_t alignment;
    // data type of the matrix
    hipDataType type;
    // memory layout
    rocsparselt_order order;
    // matrix sparsity ratio
    rocsparselt_sparsity sparsity;

    int num_batches = 1;

    int64_t batch_stride = 0;

    // info of compressed matrix, will be auto filled at rocsparselt_matmul_descr_init().
    // numbeer of k after compressed.
    int64_t c_k = -1;
    // leading dimension of compressed matrix.
    int64_t c_ld = -1;

    int64_t c_n = -1;
};

/********************************************************************************
 * \brief rocsparse_matmul_descr holds the description of the matrix multiplication operation.
 * It is initialized and destroyed with rocsparselt_matmul_descr_init()
 * and rocsparselt_matmul_descr_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_descr
{
    static constexpr uintptr_t MATMUL_MAGIC = 0x4D554C5444455343ULL; // "MULTDESC"

    // constructor
    _rocsparselt_matmul_descr(const _rocsparselt_handle* handle)
        : handle(handle)
    {
        is_init = MATMUL_MAGIC;
    };

    _rocsparselt_matmul_descr(const _rocsparselt_matmul_descr& rhs)
        : handle(rhs.handle)
        , op_A(rhs.op_A)
        , op_B(rhs.op_B)
        , compute_type(rhs.compute_type)
        , activation(rhs.activation)
        , activation_relu_upperbound(rhs.activation_relu_upperbound)
        , activation_relu_threshold(rhs.activation_relu_threshold)
        , activation_leakyrelu_alpha(rhs.activation_leakyrelu_alpha)
        , activation_tanh_alpha(rhs.activation_tanh_alpha)
        , activation_tanh_beta(rhs.activation_tanh_beta)
        , activation_gelu_scaling(rhs.activation_gelu_scaling)
        , bias_pointer(rhs.bias_pointer)
        , bias_stride(rhs.bias_stride)
        , bias_type(rhs.bias_type)
        , alpha_vector_scaling(rhs.alpha_vector_scaling)
        , gate_residual_mat_pointer(rhs.gate_residual_mat_pointer)
        , m(rhs.m)
        , n(rhs.n)
        , k(rhs.k)
        , is_sparse_a(rhs.is_sparse_a)
        , _op_A(rhs._op_A)
        , _op_B(rhs._op_B)
        , _m(rhs._m)
        , _n(rhs._n)
        , _k(rhs._k)
        , _lda(rhs._lda)
        , _ldb(rhs._ldb)
        , _is_sparse_a(rhs._is_sparse_a)
        , _swap_ab(rhs._swap_ab)
    {
        matrix_A     = rhs.matrix_A->clone();
        matrix_B     = rhs.matrix_B->clone();
        matrix_C     = rhs.matrix_C->clone();
        matrix_D     = rhs.matrix_D->clone();
        gate_residual_desc = rhs.gate_residual_desc ? rhs.gate_residual_desc->clone() : nullptr;
        is_reference = false;
        is_init      = MATMUL_MAGIC;
    };

    // destructor
    ~_rocsparselt_matmul_descr()
    {
        if(!is_reference)
        {
            delete matrix_A;
            delete matrix_B;
            delete matrix_C;
            delete matrix_D;
        }
        if(gate_residual_desc != nullptr)
            delete gate_residual_desc;
        is_init = 0;
        matrix_A           = nullptr;
        matrix_B           = nullptr;
        matrix_C           = nullptr;
        matrix_D           = nullptr;
        gate_residual_desc = nullptr;
    };

    friend std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_descr& t);

    const _rocsparselt_handle* handle = nullptr;

    // operation applied to the matrix A
    rocsparselt_operation op_A;
    // operation applied to the matrix B
    rocsparselt_operation op_B;
    // matrix description of the matrix A
    _rocsparselt_mat_descr* matrix_A;
    // matrix description of the matrix B
    _rocsparselt_mat_descr* matrix_B;
    // matrix description of the matrix C
    _rocsparselt_mat_descr* matrix_C;
    // matrix description of the matrix D
    _rocsparselt_mat_descr* matrix_D;
    //
    rocsparselt_compute_type compute_type;
    //data of rocsparselt_matmul_descr_attribute
    rocsparselt_matmul_descr_attribute activation = rocsparselt_matmul_activation_none;
    float       activation_relu_upperbound        = std::numeric_limits<float>::infinity();
    float       activation_relu_threshold         = 0.0f;
    float       activation_leakyrelu_alpha        = 1.0f;
    float       activation_tanh_alpha             = 1.0f;
    float       activation_tanh_beta              = 1.0f;
    float       activation_gelu_scaling           = 1.0f;
    float*      bias_pointer                      = nullptr;
    int64_t     bias_stride                       = 0;
    hipDataType bias_type;
    int         alpha_vector_scaling              = 0;
    void*                   gate_residual_mat_pointer = nullptr;
    _rocsparselt_mat_descr* gate_residual_desc  = nullptr;
    int64_t     m                    = 0;
    int64_t     n                    = 0;
    int64_t     k                    = 0;
    bool        is_sparse_a          = true;

    rocsparselt_operation _op_A;
    rocsparselt_operation _op_B;
    int64_t               _m           = 0;
    int64_t               _n           = 0;
    int64_t               _k           = 0;
    int64_t               _lda         = 0;
    int64_t               _ldb         = 0;
    bool                  _is_sparse_a = true;
    bool                  _swap_ab     = false;

    uintptr_t is_init                  = 0;

private:
    bool is_reference = true;
};

struct __attribute__((packed, aligned(8))) _rocsparselt_matmul_config
{
    _rocsparselt_matmul_config() {}
    ~_rocsparselt_matmul_config() {}

    _rocsparselt_matmul_config(const _rocsparselt_matmul_config& rhs)
    {
        this->index               = rhs.index;
        this->max_workspace_bytes = rhs.max_workspace_bytes;
        this->synchronizer_bytes  = rhs.synchronizer_bytes;
    }

    int    index;
    int    use_bias            = 0;
    int    use_scale_alpha_vec = 0;
    size_t max_workspace_bytes = 0;
    size_t synchronizer_bytes  = 0;
};

/********************************************************************************
 * \brief rocsparselt_matmul_alg_selection holds the description of the matrix
 * multiplication algorithm.
 * It is initialized and destroyed with rocsparselt_matmul_alg_selection_init()
 * and rocsparselt_matmul_alg_selection_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_alg_selection
{
    static constexpr int     MAX_CONFIGS = 100;
    // ALG_MAGIC: validity check without pointer dereference.
    // Protects against stack residuals even without data={} zero-initialization.
    static constexpr uintptr_t ALG_MAGIC = 0x414C475345430AULL; // "ALGSEC\n"

    // constructor: heap-allocates the shared configs array.
    // num_configs controls capacity; caller (rocsparselt_auxiliary.cpp) passes requestConfigs.
    _rocsparselt_matmul_alg_selection(const _rocsparselt_handle* handle,
                                      int                        num_configs = MAX_CONFIGS)
        : handle(handle)
        , configs(new _rocsparselt_matmul_config[num_configs]())
        , num_configs(num_configs)
    {
        is_init   = ALG_MAGIC;
        ownership = (uintptr_t)(&ownership); // self-referential: original detection
    };
    // destructor
    ~_rocsparselt_matmul_alg_selection()
    {
        clear();
    }

    void clear()
    {
        // is_original: ownership points to itself (self-referential = original).
        // After byte copy, copy.ownership = &original.ownership (different address = copy).
        //
        // Non-pointer fields (config_id, search_iterations, alg, etc.) are independent
        // per copy — each struct owns its own value after byte copy.
        // configs* is SHARED: only the original frees it; copies just detach.
        const bool is_original = (ownership == (uintptr_t)(&ownership));
        is_init   = 0;
        // Zero ownership: copies reading through their stored address will see 0,
        // making *ptr != copy.ownership → copies become invalid.
        ownership = 0;
        handle    = nullptr;
        if(is_original)
            delete[] configs;
        configs = nullptr;
    };

    friend std::ostream& operator<<(std::ostream&                            stream,
                                    const _rocsparselt_matmul_alg_selection& t);

    const _rocsparselt_handle*  handle    = nullptr;
    _rocsparselt_matmul_config* configs   = nullptr; // heap-allocated, shared on byte copy

    rocsparselt_matmul_alg alg;
    //data of rocsparselt_matmul_alg_attribute
    int       num_configs       = MAX_CONFIGS; // allocated capacity of configs array
    int       config_id         = 0;
    int       config_max_id     = 0;
    int       search_iterations = 10;
    uintptr_t is_init           = 0; // ALG_MAGIC when valid, 0 when destroyed
    uintptr_t ownership         = 0; // self-referential for original; address of original's
                                     // ownership for copies → zeroed by clear() to invalidate
};

/********************************************************************************
 * \brief rocsparselt_matmul_plan holds the matrix multiplication execution plan,
 * namely all the information necessary to execute the rocsparselt_matmul() operation.
 * It is initialized and destroyed with rocsparselt_matmul_plan_init() and
 * rocsparselt_matmul_plan_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_plan
{
    static constexpr uintptr_t IMPL_MAGIC = 0x504C414E494D504CULL; // "PLANIMPL"

    // constructor
    _rocsparselt_matmul_plan(const _rocsparselt_handle* handle)
        : handle(handle)
    {
        is_init   = IMPL_MAGIC;
        ownership = (uintptr_t)(&ownership); // self-referential: original detection
    };
    // destructor
    ~_rocsparselt_matmul_plan()
    {
        clear();
    };

    void clear()
    {
        // matmul_descr is a DEEP COPY owned by the plan (restored from original design).
        // Only the original plan deletes it; byte-copied siblings just detach.
        // alg_selection remains a non-owning reference.
        const bool is_original = (ownership == (uintptr_t)(&ownership));
        if(is_original && matmul_descr != nullptr && is_init == IMPL_MAGIC)
            delete matmul_descr;
        matmul_descr  = nullptr;
        alg_selection = nullptr;
        handle        = nullptr;
        ownership     = 0;
        is_init       = 0;
    }

    friend std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_plan& t);

    const _rocsparselt_handle* handle = nullptr;
    //
    _rocsparselt_matmul_descr* matmul_descr = nullptr;
    //
    _rocsparselt_matmul_alg_selection* alg_selection = nullptr;

    uintptr_t is_init   = 0;
    uintptr_t ownership = 0; // self-referential for original; used in clear() for is_original
};

bool check_is_init_handle(const _rocsparselt_handle* handle);
bool check_is_init_mat_descr(const _rocsparselt_mat_descr* mat);
bool check_is_init_matmul_descr(const _rocsparselt_matmul_descr* matmul);
bool check_is_init_matmul_alg_selection(const _rocsparselt_matmul_alg_selection* alg_selection);
bool check_is_init_plan(const _rocsparselt_matmul_plan* plan);

enum _rocsparselt_matmul_datatype
{
    MATMUL_DATATYPE_H_H_S,
    MATMUL_DATATYPE_B_B_S,
    MATMUL_DATATYPE_I8_I8_S,
    MATMUL_DATATYPE_I8_H_S,
    MATMUL_DATATYPE_I8_B_S,
    MATMUL_DATATYPE_I8_I_S,
    MATMUL_DATATYPE_E4M3_S_S,
    MATMUL_DATATYPE_E5M2_S_S,
    MATMUL_DATATYPE_E4M3_FNUZ_S_S,
    MATMUL_DATATYPE_E5M2_FNUZ_S_S,
    MATMUL_DATATYPE_UNKNOWN,
};

struct _rocsparselt_matmul_type
{
    _rocsparselt_matmul_datatype type;
    hipDataType a;
    hipDataType b;
    hipDataType c;
    hipDataType d;
    rocsparselt_compute_type compute;
};

constexpr _rocsparselt_matmul_type valid_matmul_datatypes[] =
{
    {MATMUL_DATATYPE_H_H_S, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, rocsparselt_compute_f32},
    {MATMUL_DATATYPE_B_B_S, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, rocsparselt_compute_f32},
    {MATMUL_DATATYPE_I8_I8_S, HIP_R_8I, HIP_R_8I, HIP_R_8I, HIP_R_8I, rocsparselt_compute_i32},
    {MATMUL_DATATYPE_I8_H_S, HIP_R_8I, HIP_R_8I, HIP_R_16F, HIP_R_16F, rocsparselt_compute_i32},
    {MATMUL_DATATYPE_I8_B_S, HIP_R_8I, HIP_R_8I, HIP_R_16BF, HIP_R_16BF, rocsparselt_compute_i32},
    {MATMUL_DATATYPE_I8_I_S, HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, rocsparselt_compute_i32},
#if HIP_FP8_TYPE_OCP
    {MATMUL_DATATYPE_E4M3_S_S, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_32F, rocsparselt_compute_f32},
    {MATMUL_DATATYPE_E5M2_S_S, HIP_R_8F_E5M2, HIP_R_8F_E5M2, HIP_R_32F, HIP_R_32F, rocsparselt_compute_f32},
#endif
#if HIP_FP8_TYPE_FNUZ
    {MATMUL_DATATYPE_E4M3_FNUZ_S_S, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_32F, HIP_R_32F, rocsparselt_compute_f32},
    {MATMUL_DATATYPE_E5M2_FNUZ_S_S, HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E5M2_FNUZ, HIP_R_32F, HIP_R_32F, rocsparselt_compute_f32},
#endif
};

_rocsparselt_matmul_datatype is_matmul_datatype_valid(hipDataType a, hipDataType b, hipDataType c, hipDataType d, rocsparselt_compute_type compute);
#endif // HANDLE_H
