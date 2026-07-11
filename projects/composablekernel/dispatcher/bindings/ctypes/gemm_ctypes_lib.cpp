// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GEMM Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration.
 * Kernel header included via -include at compile time.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Host-side B-preshuffle utilities. For a weight-preshuffled kernel the device
// expects B already reordered into the pipeline's packed layout; this is the
// SAME transform Old-TE's gemm_preshuffle profiler applies (shuffle_b /
// shuffle_b_permuteN in tensor_shuffle_utils.hpp) so the bridge produces
// byte-for-byte identical B, hence identical results.
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_permute.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"

// Kernel header included via -include compiler flag
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME

// GPU architecture - can be overridden via -DGFX_ARCH="gfx90a" at compile time
#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

#if defined(GEMM_KEY_PRESHUFFLE) && (GEMM_KEY_PRESHUFFLE != 0)
// Adapter exposing the force-included kernel's tile geometry under the field
// names ck_tile::shuffle_b / shuffle_b_permuteN expect. Mirrors Old-TE's
// gemm_preshuffle_benchmark.hpp::KernelConfig so the permutation is identical.
struct BridgePreshuffleConfig
{
    static constexpr ck_tile::index_t M_Tile = SelectedKernel::TileM;
    static constexpr ck_tile::index_t N_Tile = SelectedKernel::TileN;
    static constexpr ck_tile::index_t K_Tile = SelectedKernel::TileK;

    static constexpr ck_tile::index_t M_Warp = SelectedKernel::WarpPerBlock_M;
    static constexpr ck_tile::index_t N_Warp = SelectedKernel::WarpPerBlock_N;
    static constexpr ck_tile::index_t K_Warp = SelectedKernel::WarpPerBlock_K;

    static constexpr ck_tile::index_t M_Warp_Tile = SelectedKernel::WarpTileM;
    static constexpr ck_tile::index_t N_Warp_Tile = SelectedKernel::WarpTileN;
    static constexpr ck_tile::index_t K_Warp_Tile = SelectedKernel::WarpTileK;

    static constexpr bool permuteN = SelectedKernel::PermuteN;
};

// Preshuffle host B into the packed layout the device pipeline reads. Returns a
// contiguous host buffer of the shuffled bytes.
//
// The shuffle utils (shuffle_b / shuffle_b_permuteN) take a rank-2 HostTensor
// with lengths {K, N} whose PHYSICAL buffer is N-outer / K-contiguous -- exactly
// Old-TE's b_k_n, built as host_tensor_descriptor(K, N, stride, is_row_major=
// false) for the rcr kernel's column-major BLayout. The bridge runner hands B in
// this same order for a 'c' B operand (ascontiguousarray(B.T), shape [N, K] row-
// major == column-major [K, N]), so filling the col-major {K, N} tensor's flat
// storage directly reproduces Old-TE's b_k_n byte-for-byte, hence an identical
// permutation and identical results.
template <typename T>
static ck_tile::HostTensor<T> preshuffle_host_b(const T* b_host, int64_t K, int64_t N)
{
    // Build b_k_n with the SAME descriptor Old-TE uses:
    //   host_tensor_descriptor(K, N, stride_b, is_row_major(BLayout))
    // host_tensor_descriptor takes a compile-time bool_constant. BLayout is the
    // force-included kernel's own B layout alias; for the rcr preshuffle kernel
    // it is column-major, giving lengths {K, N} with strides {1, K} (N-outer,
    // K-contiguous) -- the exact physical order shuffle_b / shuffle_b_permuteN
    // expect and that the runner supplies for a 'c' B operand.
    constexpr bool kBRowMajor = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
    const auto stride_b       = ck_tile::get_default_stride(static_cast<ck_tile::index_t>(K),
                                                      static_cast<ck_tile::index_t>(N),
                                                      0,
                                                      ck_tile::bool_constant<kBRowMajor>{});
    ck_tile::HostTensor<T> b_k_n(
        ck_tile::host_tensor_descriptor(static_cast<ck_tile::index_t>(K),
                                        static_cast<ck_tile::index_t>(N),
                                        stride_b,
                                        ck_tile::bool_constant<kBRowMajor>{}));
    std::copy(b_host, b_host + (K * N), b_k_n.begin());
    if constexpr(BridgePreshuffleConfig::permuteN)
    {
        return ck_tile::shuffle_b_permuteN<BridgePreshuffleConfig>(b_k_n);
    }
    else
    {
        return ck_tile::shuffle_b<BridgePreshuffleConfig>(b_k_n);
    }
}

// Cache the shuffled B so the (host) preshuffle + reorder is paid ONCE per
// distinct (B pointer, K, N), not on every dispatcher_run_gemm call. Old-TE
// shuffles B a single time per callable; the A/B perf sweep calls run() in a
// warmup+repeat loop with the same B, so recomputing the shuffle every call
// would (a) add non-kernel host work between iterations and (b) make the sweep
// apples-to-oranges. The kernel-timed region (g_dispatcher->run) never contained
// the shuffle, but hoisting it keeps the per-iteration host path launch-only.
struct ShuffledBCache
{
    const void* b_host = nullptr;
    int64_t K          = 0;
    int64_t N          = 0;
    // Held by shared_ptr so the cache has a trivial (null) default state --
    // HostTensor has no default constructor, and a dummy one would need a valid
    // descriptor. Populated lazily on the first (or a changed) B.
    std::shared_ptr<ck_tile::HostTensor<BDataType>> data;
};
static ShuffledBCache g_shuffled_b_cache;

// Return a pointer to the shuffled bytes for this B, reusing the cache when the
// (pointer, K, N) matches the last call. Not thread-safe (bridge is single-
// threaded), which matches the rest of this translation unit.
static const BDataType* get_shuffled_b(const BDataType* b_host, int64_t K, int64_t N)
{
    if(!(g_shuffled_b_cache.data && g_shuffled_b_cache.b_host == b_host &&
         g_shuffled_b_cache.K == K && g_shuffled_b_cache.N == N))
    {
        g_shuffled_b_cache.data = std::make_shared<ck_tile::HostTensor<BDataType>>(
            preshuffle_host_b<BDataType>(b_host, K, N));
        g_shuffled_b_cache.b_host = b_host;
        g_shuffled_b_cache.K      = K;
        g_shuffled_b_cache.N      = N;
    }
    return g_shuffled_b_cache.data->data();
}
#endif // GEMM_KEY_PRESHUFFLE

// Global dispatcher (initialized once, managed via shared_ptr for safe cleanup)
static std::shared_ptr<Dispatcher> g_dispatcher = nullptr;
static bool g_initialized                       = false;

#define HIP_CHECK(call)        \
    {                          \
        hipError_t err = call; \
        if(err != hipSuccess)  \
        {                      \
            return -1;         \
        }                      \
    }

extern "C" {

/**
 * Initialize dispatcher with a kernel
 * Must be called before run_gemm
 *
 * Returns: 0 on success, -1 on error
 */
int dispatcher_initialize()
{
    if(g_initialized)
    {
        return 0; // Already initialized
    }

    // Create kernel key from the force-included kernel header.
    //
    // The GEMM_KEY_* macros are emitted by the codegen into the force-included
    // header (see unified_gemm_codegen.py, CK_TILE_SINGLE_KERNEL_INCLUDE block).
    // Building the key from them makes the registry entry truthful: it reflects
    // THIS kernel's real dtypes/layouts/tile/traits instead of a hard-coded
    // fp16/rcr/128x128x32 default. Enum fields use the string_to_* helpers from
    // kernel_key.hpp, whose accepted strings match the codegen's emitted values
    // byte-for-byte.
    KernelKey key;
#ifdef GEMM_KEY_DTYPE_A
    key.signature.dtype_a             = string_to_dtype(GEMM_KEY_DTYPE_A);
    key.signature.dtype_b             = string_to_dtype(GEMM_KEY_DTYPE_B);
    key.signature.dtype_c             = string_to_dtype(GEMM_KEY_DTYPE_C);
    key.signature.dtype_acc           = string_to_dtype(GEMM_KEY_DTYPE_ACC);
    key.signature.layout_a            = string_to_layout(GEMM_KEY_LAYOUT_A);
    key.signature.layout_b            = string_to_layout(GEMM_KEY_LAYOUT_B);
    key.signature.layout_c            = string_to_layout(GEMM_KEY_LAYOUT_C);
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = (GEMM_KEY_GROUPED != 0);
    key.signature.split_k             = GEMM_KEY_SPLIT_K;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape      = {GEMM_KEY_TILE_M, GEMM_KEY_TILE_N, GEMM_KEY_TILE_K};
    key.algorithm.wave_shape      = {GEMM_KEY_WAVE_M, GEMM_KEY_WAVE_N, GEMM_KEY_WAVE_K};
    key.algorithm.warp_tile_shape = {
        GEMM_KEY_WARP_TILE_M, GEMM_KEY_WARP_TILE_N, GEMM_KEY_WARP_TILE_K};
    key.algorithm.pipeline        = string_to_pipeline(GEMM_KEY_PIPELINE);
    key.algorithm.scheduler       = string_to_scheduler(GEMM_KEY_SCHEDULER);
    key.algorithm.epilogue        = string_to_epilogue(GEMM_KEY_EPILOGUE);
    key.algorithm.block_size      = GEMM_KEY_BLOCK_SIZE;
    key.algorithm.double_buffer   = (GEMM_KEY_DOUBLE_BUFFER != 0);
    key.algorithm.persistent      = (GEMM_KEY_PERSISTENT != 0);
    key.algorithm.preshuffle      = (GEMM_KEY_PRESHUFFLE != 0);
    key.algorithm.transpose_c     = (GEMM_KEY_TRANSPOSE_C != 0);
    key.algorithm.num_wave_groups = GEMM_KEY_NUM_WAVE_GROUPS;
    // pad_m/n/k participate in both the key's hash/equality and the kernel
    // name, so they must be derived from the codegen macros too -- otherwise a
    // kernel built with padding disabled would register under a key claiming
    // pad=true and disagree with its own name.
    key.algorithm.pad_m = (GEMM_KEY_PAD_M != 0);
    key.algorithm.pad_n = (GEMM_KEY_PAD_N != 0);
    key.algorithm.pad_k = (GEMM_KEY_PAD_K != 0);
    key.gfx_arch        = GFX_ARCH;
#else
    // Fallback default for headers generated before GEMM_KEY_* macros existed
    // (fp16 / rcr / compv4-cshuffle-intrawave, 128x128x32). The macro path
    // above is the source of truth for any freshly generated kernel.
    key.signature.dtype_a   = DataType::FP16;
    key.signature.dtype_b   = DataType::FP16;
    key.signature.dtype_c   = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    // Derive A/B/C layouts from the force-included kernel's own layout types
    // instead of hardcoding rcr. The dispatcher's supports() gate is layout-aware
    // (it only constrains a dimension that an operand's inner axis maps to), so a
    // wrong key layout makes it reject valid problems -- e.g. a crr kernel does not
    // gate K, but with a hardcoded rcr key supports() would apply rcr's K-gate and
    // reject TileK=192 problems that Old-TE runs. ALayout/BLayout/CLayout are the
    // global aliases exported by the kernel header under CK_TILE_SINGLE_KERNEL_INCLUDE.
    using RowMajorLayout = ck_tile::tensor_layout::gemm::RowMajor;
    key.signature.layout_a =
        std::is_same_v<ALayout, RowMajorLayout> ? LayoutTag::RowMajor : LayoutTag::ColMajor;
    key.signature.layout_b =
        std::is_same_v<BLayout, RowMajorLayout> ? LayoutTag::RowMajor : LayoutTag::ColMajor;
    key.signature.layout_c =
        std::is_same_v<CLayout, RowMajorLayout> ? LayoutTag::RowMajor : LayoutTag::ColMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape      = {128, 128, 32};
    key.algorithm.wave_shape      = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline        = Pipeline::CompV4;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = 256;
    key.algorithm.double_buffer   = true;
    key.algorithm.persistent      = false;
    key.algorithm.preshuffle      = false;
    key.algorithm.transpose_c     = false;
    key.algorithm.num_wave_groups = 1;
    key.gfx_arch                  = GFX_ARCH;
#endif // GEMM_KEY_DTYPE_A

    // Register kernel using types from force-included header
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);

    // Create dispatcher (using shared_ptr for safe memory management)
    g_dispatcher  = std::make_shared<Dispatcher>();
    g_initialized = true;

    return 0;
}

/**
 * Get kernel tile configuration
 */
int dispatcher_get_kernel_config(int* tile_m,
                                 int* tile_n,
                                 int* tile_k,
                                 int* warp_tile_m,
                                 int* warp_tile_n,
                                 int* warp_tile_k,
                                 int* warp_m,
                                 int* warp_n,
                                 int* warp_k)
{
    if(!g_initialized)
    {
        return -1;
    }

    auto kernels = Registry::instance().get_all();
    if(kernels.empty())
    {
        return -1;
    }

    // Get configuration from first kernel
    auto& key  = kernels[0]->get_key();
    auto& algo = key.algorithm;

    if(tile_m)
        *tile_m = algo.tile_shape.m;
    if(tile_n)
        *tile_n = algo.tile_shape.n;
    if(tile_k)
        *tile_k = algo.tile_shape.k;
    if(warp_tile_m)
        *warp_tile_m = algo.warp_tile_shape.m;
    if(warp_tile_n)
        *warp_tile_n = algo.warp_tile_shape.n;
    if(warp_tile_k)
        *warp_tile_k = algo.warp_tile_shape.k;
    if(warp_m)
        *warp_m = algo.wave_shape.m;
    if(warp_n)
        *warp_n = algo.wave_shape.n;
    if(warp_k)
        *warp_k = algo.wave_shape.k;

    return 0;
}

/**
 * Get the selected kernel name for a problem
 */
int dispatcher_select_kernel(int64_t M, int64_t N, int64_t K, char* name_buffer, int buffer_size)
{
    if(!g_initialized || !name_buffer || buffer_size <= 0)
    {
        return -1;
    }

    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);

    if(!kernel)
    {
        return -1;
    }

    std::string name = kernel->get_name();
    strncpy(name_buffer, name.c_str(), buffer_size - 1);
    name_buffer[buffer_size - 1] = '\0';

    return 0;
}

/**
 * Check if a problem size is supported by available kernels
 */
int dispatcher_is_supported(int64_t M, int64_t N, int64_t K)
{
    if(!g_initialized)
    {
        return 0;
    }

    if(M <= 0 || N <= 0 || K <= 0)
    {
        return 0;
    }

    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    return kernel != nullptr ? 1 : 0;
}

/**
 * Run GEMM on GPU via dispatcher
 */
int dispatcher_run_gemm(
    const void* A, const void* B, void* C, int64_t M, int64_t N, int64_t K, float* time_ms)
{
    if(!g_initialized || !A || !B || !C)
    {
        return -1;
    }

    // First check if any kernel supports this problem
    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    if(!kernel)
    {
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -2; // No suitable kernel
    }

    // Cast to correct types (from force-included header)
    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    // Allocate GPU memory
    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;

    auto cleanup_gpu_mem = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev, M * K * sizeof(ADataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&B_dev, K * N * sizeof(BDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&C_dev, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Copy input data to GPU
    if(hipMemcpy(A_dev, A_host, M * K * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
#if defined(GEMM_KEY_PRESHUFFLE) && (GEMM_KEY_PRESHUFFLE != 0)
    // Weight-preshuffled kernel: reorder B on the host into the packed layout the
    // device pipeline reads, exactly as Old-TE does before launch. The shuffle is
    // a pure permutation (same element count), so the device buffer size is
    // unchanged. B_host stays the logical (unshuffled) B so the Python-side
    // numpy reference (A @ B) remains valid.
    {
        // Shuffle is cached across calls (see get_shuffled_b): the host reorder
        // runs once per distinct B, so only the H2D copy + kernel launch remain
        // on the repeated benchmark path -- matching Old-TE's one-shuffle model.
        const BDataType* b_shuffled = get_shuffled_b(B_host, K, N);
        if(hipMemcpy(B_dev, b_shuffled, K * N * sizeof(BDataType), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
    }
#else
    if(hipMemcpy(B_dev, B_host, K * N * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
#endif // GEMM_KEY_PRESHUFFLE
    if(hipMemset(C_dev, 0, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Run GEMM via dispatcher
    float exec_time;
    try
    {
        exec_time = g_dispatcher->run(A_dev, B_dev, C_dev, problem);
    }
    catch(const std::exception& e)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Copy result back to host
    if(hipMemcpy(C_host, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    if(time_ms)
    {
        *time_ms = exec_time;
    }

    cleanup_gpu_mem();
    return 0;
}

/**
 * Get kernel information (legacy single-kernel ABI).
 *
 * Returns the compile-time KERNEL_NAME of the force-included kernel header.
 * Kept for backward compatibility with one-kernel-per-.so callers.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Get the name of the kernel at a given registry index (multi-kernel ABI).
 *
 * Mirrors the conv/fmha ctypes libs: copies the index-th registered kernel's
 * name into the caller-provided buffer so one .so can report a whole batch and
 * be selected by name at runtime. Returns 0 on success, -1 on bad args or
 * out-of-range index.
 */
int dispatcher_get_kernel_name_at(int index, char* buffer, int buffer_size)
{
    if(!buffer || buffer_size <= 0)
    {
        return -1;
    }

    auto kernels = Registry::instance().get_all();
    if(index < 0 || index >= static_cast<int>(kernels.size()))
    {
        return -1;
    }

    std::string name = kernels[index]->get_name();
    std::strncpy(buffer, name.c_str(), static_cast<size_t>(buffer_size) - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

/**
 * Initialize dispatcher (alias)
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Get the number of registered kernels
 */
int dispatcher_get_kernel_count() { return static_cast<int>(Registry::instance().size()); }

/**
 * Export registry to JSON string
 */
static std::string g_json_buffer;

const char* dispatcher_export_registry_json()
{
    auto& registry = Registry::instance();

    std::ostringstream json;
    json << "{\n";
    json << "  \"metadata\": {\n";
    json << "    \"timestamp\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    json << "    \"total_kernels\": " << registry.size() << ",\n";
    json << "    \"export_version\": \"1.0\",\n";
    json << "    \"dispatcher_version\": \"1.0.0\"\n";
    json << "  },\n";
    json << "  \"statistics\": {\n";
    json << "    \"by_datatype\": {},\n";
    json << "    \"by_pipeline\": {},\n";
    json << "    \"by_scheduler\": {}\n";
    json << "  },\n";
    json << "  \"kernels\": [\n";

    auto kernels = registry.get_all();
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        auto& kernel     = kernels[i];
        auto& key        = kernel->get_key();
        auto& algo       = key.algorithm;
        std::string name = kernel->get_name();

        json << "    {\n";
        json << "      \"identifier\": \"" << key.encode_identifier() << "\",\n";
        json << "      \"name\": \"" << name << "\",\n";
        json << "      \"algorithm\": {\n";
        json << "        \"tile_shape\": {\"m\": " << algo.tile_shape.m
             << ", \"n\": " << algo.tile_shape.n << ", \"k\": " << algo.tile_shape.k << "},\n";
        json << "        \"wave_shape\": {\"m\": " << unsigned(algo.wave_shape.m)
             << ", \"n\": " << unsigned(algo.wave_shape.n)
             << ", \"k\": " << unsigned(algo.wave_shape.k) << "},\n";
        json << "        \"warp_tile_shape\": {\"m\": " << unsigned(algo.warp_tile_shape.m)
             << ", \"n\": " << unsigned(algo.warp_tile_shape.n)
             << ", \"k\": " << unsigned(algo.warp_tile_shape.k) << "},\n";
        json << "        \"block_size\": " << algo.block_size << ",\n";
        json << "        \"persistent\": " << (algo.persistent ? "true" : "false") << ",\n";
        json << "        \"double_buffer\": " << (algo.double_buffer ? "true" : "false") << ",\n";
        json << "        \"preshuffle\": " << (algo.preshuffle ? "true" : "false") << ",\n";
        json << "        \"transpose_c\": " << (algo.transpose_c ? "true" : "false") << "\n";
        json << "      }\n";
        json << "    }";
        if(i < kernels.size() - 1)
        {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    g_json_buffer = json.str();
    return g_json_buffer.c_str();
}

/**
 * Cleanup dispatcher resources
 */
void dispatcher_cleanup()
{
    g_dispatcher.reset();
    g_initialized = false;
}

} // extern "C"