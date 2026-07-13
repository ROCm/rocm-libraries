// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Batched-Contraction Dispatcher ctypes Library (TileEngine -> Dispatcher bridge).
 *
 * Provides a C API for Python ctypes integration. The kernel header is
 * force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE batched_contraction_ctypes_lib.cpp
 *
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * EDataType, AccDataType, CONTRACTION_KEY_NUM_DIM_{G,M,N,K}, NUM_D_TENSORS.
 *
 * Registry bypass: batched contraction's launch takes
 * ck_tile::BatchedContractionHostArgs<NumDTensor> (variable-length dim/stride
 * vectors), which the generic dispatcher backend cannot express. So this lib
 * builds the HostArgs from plain C arrays and calls SelectedKernel::launch()
 * directly -- the same direct-launch pattern used by the batched/multi-D bridges.
 *
 * Memory model: host-pointer in. The lib owns hipMalloc/hipMemcpy/hipFree.
 * Layouts A=[G..,M..,K..], B=[G..,N..,K..], E=[G..,M..,N..], packed row-major
 * strides (matches the Old-TE profiler's HostTensorDescriptor(dims)).
 * v1 scope: NUM_D_TENSORS == 0 (PassThrough epilogue).
 */

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Kernel header force-included via -include. Brings ck_tile core + the
// ck_tile::BatchedContractionHostArgs type and SelectedKernel/KERNEL_NAME.

#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

static bool g_initialized = false;

namespace {

// Packed row-major strides for a dimension list (matches HostTensorDescriptor).
std::vector<ck_tile::index_t> packed_row_major_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> strides(dims.size(), 1);
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * dims[i + 1];
    return strides;
}

int64_t product(const std::vector<ck_tile::index_t>& dims)
{
    int64_t p = 1;
    for(auto d : dims)
        p *= static_cast<int64_t>(d);
    return p;
}

int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if(!v)
        return fallback;
    return std::atoi(v);
}

} // namespace

extern "C" {

int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}
int dispatcher_init() { return dispatcher_initialize(); }

const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }
int dispatcher_get_kernel_count() { return 1; }
int dispatcher_get_num_dim_g() { return CONTRACTION_KEY_NUM_DIM_G; }
int dispatcher_get_num_dim_m() { return CONTRACTION_KEY_NUM_DIM_M; }
int dispatcher_get_num_dim_n() { return CONTRACTION_KEY_NUM_DIM_N; }
int dispatcher_get_num_dim_k() { return CONTRACTION_KEY_NUM_DIM_K; }
int dispatcher_get_num_d_tensors() { return CONTRACTION_KEY_NUM_D_TENSORS; }

void dispatcher_cleanup() { g_initialized = false; }

/**
 * Run batched contraction: E[G..,M..,N..] = sum_K A[G..,M..,K..] * B[G..,N..,K..].
 *
 * A, B, E are host pointers (row-major packed). g/m/n/k_dims give the per-group
 * dimension lengths; their counts must equal the compiled-in NUM_DIM_{G,M,N,K}.
 * Returns 0 ok, -1 HIP/bad-args, -2 kernel reports unsupported args.
 */
int dispatcher_run_batched_contraction(const void* A,
                                       const void* B,
                                       void* E,
                                       const int64_t* g_dims,
                                       const int64_t* m_dims,
                                       const int64_t* n_dims,
                                       const int64_t* k_dims,
                                       int num_dim_g,
                                       int num_dim_m,
                                       int num_dim_n,
                                       int num_dim_k,
                                       int k_batch,
                                       float* time_ms)
{
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_batched_contraction: not initialized\n";
        return -1;
    }
    if(!A || !B || !E)
    {
        std::cerr << "dispatcher_run_batched_contraction: null pointer\n";
        return -1;
    }
    if(CONTRACTION_KEY_NUM_D_TENSORS != 0)
    {
        std::cerr
            << "dispatcher_run_batched_contraction: this ABI supports NUM_D_TENSORS==0 only\n";
        return -1;
    }
    if(k_batch != 1)
    {
        // v1 scope: split-K (k_batch > 1) is not yet numerically correct through this
        // bridge, so reject it rather than return silently-wrong results.
        std::cerr << "dispatcher_run_batched_contraction: only k_batch==1 is supported in v1, got "
                  << k_batch << "\n";
        return -1;
    }
    if(num_dim_g <= 0 || num_dim_m <= 0 || num_dim_n <= 0 || num_dim_k <= 0)
    {
        std::cerr << "dispatcher_run_batched_contraction: num_dim_* must be > 0\n";
        return -1;
    }
    if(num_dim_g != CONTRACTION_KEY_NUM_DIM_G || num_dim_m != CONTRACTION_KEY_NUM_DIM_M ||
       num_dim_n != CONTRACTION_KEY_NUM_DIM_N || num_dim_k != CONTRACTION_KEY_NUM_DIM_K)
    {
        std::cerr << "dispatcher_run_batched_contraction: num_dim mismatch. got (g,m,n,k)=("
                  << num_dim_g << "," << num_dim_m << "," << num_dim_n << "," << num_dim_k
                  << "), compiled (" << CONTRACTION_KEY_NUM_DIM_G << ","
                  << CONTRACTION_KEY_NUM_DIM_M << "," << CONTRACTION_KEY_NUM_DIM_N << ","
                  << CONTRACTION_KEY_NUM_DIM_K << ")\n";
        return -1;
    }

    auto to_vec = [](const int64_t* p, int n) {
        std::vector<ck_tile::index_t> v(n);
        for(int i = 0; i < n; ++i)
            v[i] = static_cast<ck_tile::index_t>(p[i]);
        return v;
    };
    std::vector<ck_tile::index_t> gd = to_vec(g_dims, num_dim_g);
    std::vector<ck_tile::index_t> md = to_vec(m_dims, num_dim_m);
    std::vector<ck_tile::index_t> nd = to_vec(n_dims, num_dim_n);
    std::vector<ck_tile::index_t> kd = to_vec(k_dims, num_dim_k);

    auto concat = [](std::vector<ck_tile::index_t> a,
                     const std::vector<ck_tile::index_t>& b,
                     const std::vector<ck_tile::index_t>& c) {
        a.insert(a.end(), b.begin(), b.end());
        a.insert(a.end(), c.begin(), c.end());
        return a;
    };
    std::vector<ck_tile::index_t> A_dims = concat(gd, md, kd); // [G..,M..,K..]
    std::vector<ck_tile::index_t> B_dims = concat(gd, nd, kd); // [G..,N..,K..]
    std::vector<ck_tile::index_t> E_dims = concat(gd, md, nd); // [G..,M..,N..]

    std::vector<ck_tile::index_t> A_strides = packed_row_major_strides(A_dims);
    std::vector<ck_tile::index_t> B_strides = packed_row_major_strides(B_dims);
    std::vector<ck_tile::index_t> E_strides = packed_row_major_strides(E_dims);

    const int64_t a_elems = product(A_dims);
    const int64_t b_elems = product(B_dims);
    const int64_t e_elems = product(E_dims);
    if(a_elems <= 0 || b_elems <= 0 || e_elems <= 0)
    {
        std::cerr << "dispatcher_run_batched_contraction: non-positive dimension product\n";
        return -1;
    }

    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    EDataType* E_host       = static_cast<EDataType*>(E);

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    EDataType* E_dev = nullptr;
    auto cleanup     = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(E_dev)
            (void)hipFree(E_dev);
    };

    if(hipMalloc(&A_dev, a_elems * sizeof(ADataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev, b_elems * sizeof(BDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&E_dev, e_elems * sizeof(EDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(hipMemcpy(A_dev, A_host, a_elems * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, b_elems * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemset(E_dev, 0, e_elems * sizeof(EDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    ck_tile::BatchedContractionHostArgs<CONTRACTION_KEY_NUM_D_TENSORS> args(
        /*a_ptr*/ A_dev,
        /*b_ptr*/ B_dev,
        /*ds_ptr*/ std::array<const void*, CONTRACTION_KEY_NUM_D_TENSORS>{},
        /*e_ptr*/ E_dev,
        /*k_batch*/ static_cast<ck_tile::index_t>(k_batch),
        /*A_dims*/ A_dims,
        /*B_dims*/ B_dims,
        /*Ds_dims*/ std::array<std::vector<ck_tile::index_t>, CONTRACTION_KEY_NUM_D_TENSORS>{},
        /*E_dims*/ E_dims,
        /*A_strides*/ A_strides,
        /*B_strides*/ B_strides,
        /*Ds_strides*/ std::array<std::vector<ck_tile::index_t>, CONTRACTION_KEY_NUM_D_TENSORS>{},
        /*E_strides*/ E_strides);

    const bool do_time = (time_ms != nullptr);
    const int warmup   = do_time ? env_int("CK_TILE_BENCH_WARMUP", 20) : 0;
    const int repeat   = do_time ? env_int("CK_TILE_BENCH_REPEAT", 50) : 1;
    ck_tile::stream_config stream_cfg{nullptr, do_time, 0, warmup, repeat, false, false, 1};

    float exec_time = 0.0f;
    try
    {
        exec_time = SelectedKernel::launch(args, stream_cfg);
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_batched_contraction: launch threw: " << e.what() << "\n";
        cleanup();
        return -1;
    }
    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_batched_contraction: kernel reports unsupported args\n";
        cleanup();
        return -2;
    }

    if(hipMemcpy(E_host, E_dev, e_elems * sizeof(EDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
}

} // extern "C"
