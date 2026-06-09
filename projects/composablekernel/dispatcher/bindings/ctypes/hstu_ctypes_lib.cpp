// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HSTU jagged forward dispatcher ctypes library (in-process, not subprocess).
// JIT builds force-include hstu_python_dispatch.hpp defining HSTU_RUN_JAGGED_FWD.
// Prebuilt lib falls back to example hstu_attention_no_group_forward_* entry points.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "example/ck_tile/53_hstu_attention/hstu_attention_api.hpp"
#include "example/ck_tile/53_hstu_attention/hstu_attention_params.hpp"

#ifndef GFX_ARCH
#define GFX_ARCH "gfx950"
#endif

#define HIP_CHECK(call)           \
    do                            \
    {                             \
        hipError_t err_ = (call); \
        if(err_ != hipSuccess)    \
        {                         \
            rc = -1;              \
            goto cleanup;         \
        }                         \
    } while(0)

static bool g_initialized = false;

static int dtype_elem_bytes(const char* dtype)
{
    if(dtype && std::strcmp(dtype, "fp32") == 0)
        return 4;
    return 2;
}

static void set_env_int(const char* name, int value)
{
    setenv(name, std::to_string(value).c_str(), 1);
}

#ifndef HSTU_RUN_JAGGED_FWD
static void run_hstu_jagged_fwd_fallback(HstuAttentionNoGroupFwdParams& params,
                                         hipStream_t stream,
                                         const char* data_type_str,
                                         int force_mtile,
                                         int force_splitkv,
                                         int disable_splitkv)
{
    if(force_mtile > 0)
        set_env_int("HSTU_FORCE_MTILE", force_mtile);
    if(force_splitkv >= 0)
        set_env_int("HSTU_FORCE_SPLITKV", force_splitkv);
    if(disable_splitkv)
        setenv("HSTU_DISABLE_SPLITKV", "1", 1);
    else if(force_splitkv < 0)
        unsetenv("HSTU_DISABLE_SPLITKV");

    if(data_type_str && std::strcmp(data_type_str, "bf16") == 0)
        hstu_attention_no_group_forward_bf16(params, stream);
    else
        hstu_attention_no_group_forward_fp16(params, stream);
}
#endif

extern "C" {

int hstu_dispatcher_initialize(const char* /*arch*/)
{
    g_initialized = true;
    return 0;
}

void hstu_dispatcher_cleanup() { g_initialized = false; }

int hstu_dispatcher_kernel_count() { return g_initialized ? 1 : 0; }

int hstu_dispatcher_run_jagged_fwd(const void* q_host,
                                   const void* k_host,
                                   const void* v_host,
                                   void* o_host,
                                   const int* seq_offsets_host,
                                   const int* num_targets_host,
                                   int batch,
                                   int num_head,
                                   int hdim_qk,
                                   int hdim_v,
                                   int max_seqlen_q,
                                   int total_tokens,
                                   int use_causal,
                                   int window_size,
                                   int contextual_seqlen,
                                   int min_full_attn_seqlen,
                                   int force_mtile,
                                   int force_splitkv,
                                   int disable_splitkv,
                                   float scale_s,
                                   float attn_scale,
                                   const char* data_type_str,
                                   float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    int rc = 0;
    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void* seq_offsets_dev = nullptr;
    void* num_targets_dev = nullptr;
    hipStream_t stream    = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    HstuAttentionNoGroupFwdParams params{};
    float elapsed_ms = 0.f;

    const int elem = dtype_elem_bytes(data_type_str);
    const int64_t q_bytes =
        static_cast<int64_t>(total_tokens) * num_head * hdim_qk * elem;
    const int64_t k_bytes = q_bytes;
    const int64_t v_bytes =
        static_cast<int64_t>(total_tokens) * num_head * hdim_v * elem;
    const int64_t o_bytes = v_bytes;

    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));
    HIP_CHECK(hipMalloc(&seq_offsets_dev, (batch + 1) * sizeof(int)));
    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev, k_host, k_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev, v_host, v_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(seq_offsets_dev, seq_offsets_host, (batch + 1) * sizeof(int),
                        hipMemcpyHostToDevice));

    if(num_targets_host)
    {
        HIP_CHECK(hipMalloc(&num_targets_dev, batch * sizeof(int)));
        HIP_CHECK(hipMemcpy(num_targets_dev, num_targets_host, batch * sizeof(int),
                            hipMemcpyHostToDevice));
    }

    params.is_cross_attention     = false;
    params.is_jagged              = true;
    params.num_batch              = batch;
    params.seq_q_offsets_ptr      = seq_offsets_dev;
    params.seq_kv_offsets_ptr     = seq_offsets_dev;
    params.max_seqlen_q           = max_seqlen_q;
    params.q_ptr                  = q_dev;
    params.k_ptr                  = k_dev;
    params.v_ptr                  = v_dev;
    params.bias_ptr               = nullptr;
    params.o_ptr                  = o_dev;
    params.hdim_qk                = hdim_qk;
    params.hdim_v                 = hdim_v;
    params.num_head               = num_head;
    params.scale_s                = scale_s;
    params.attn_scale             = attn_scale;
    params.seq_stride_q           = hdim_qk * num_head;
    params.seq_stride_k           = hdim_qk * num_head;
    params.seq_stride_v           = hdim_v * num_head;
    params.seq_stride_o           = hdim_v * num_head;
    params.nhead_stride_q         = hdim_qk;
    params.nhead_stride_k         = hdim_qk;
    params.nhead_stride_v         = hdim_v;
    params.nhead_stride_o         = hdim_v;
    params.num_targets_ptr        = num_targets_dev;
    params.use_softmax            = false;
    params.use_causal             = (use_causal != 0);
    params.window_size            = window_size;
    params.contextual_seqlen      = contextual_seqlen;
    params.min_full_attn_seqlen   = min_full_attn_seqlen;

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

#ifdef HSTU_RUN_JAGGED_FWD
    (void)force_mtile;
    (void)force_splitkv;
    (void)disable_splitkv;
    for(int i = 0; i < 2; ++i)
        HSTU_RUN_JAGGED_FWD(params, stream);
    HIP_CHECK(hipEventRecord(start, stream));
    for(int i = 0; i < 10; ++i)
        HSTU_RUN_JAGGED_FWD(params, stream);
#else
    for(int i = 0; i < 2; ++i)
        run_hstu_jagged_fwd_fallback(
            params, stream, data_type_str, force_mtile, force_splitkv, disable_splitkv);
    HIP_CHECK(hipEventRecord(start, stream));
    for(int i = 0; i < 10; ++i)
        run_hstu_jagged_fwd_fallback(
            params, stream, data_type_str, force_mtile, force_splitkv, disable_splitkv);
#endif
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= 10.f;
    if(time_ms_out)
        *time_ms_out = elapsed_ms;

    HIP_CHECK(hipMemcpy(o_host, o_dev, o_bytes, hipMemcpyDeviceToHost));

cleanup:
    if(start)
        hipEventDestroy(start);
    if(stop)
        hipEventDestroy(stop);
    if(stream)
        hipStreamDestroy(stream);
    if(q_dev)
        hipFree(q_dev);
    if(k_dev)
        hipFree(k_dev);
    if(v_dev)
        hipFree(v_dev);
    if(o_dev)
        hipFree(o_dev);
    if(seq_offsets_dev)
        hipFree(seq_offsets_dev);
    if(num_targets_dev)
        hipFree(num_targets_dev);
    return rc;
}

} // extern "C"
