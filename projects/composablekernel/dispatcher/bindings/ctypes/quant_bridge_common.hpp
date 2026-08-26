// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Shared infrastructure for the block-scale quant GEMM ctypes bridges.
 *
 * The five per-op bridges (tensor_quant, rowcolquant, aquant, abquant, bquant)
 * each compile one kernel per .so, force-including the generated kernel header:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_<op>_ctypes_lib.cpp
 *
 * Because every bridge is its own translation unit, this common layer is
 * header-only: the templates / inline helpers below each get their own copy per
 * .so, so there is no shared library and no ODR concern. It centralizes the
 * infrastructure that used to be copy-pasted into all five sources -- device
 * memory management, GPU-arch validation, the entry guard chain, quant-group
 * validation, kernel-launch timing, init/cleanup, and the exported C
 * boilerplate -- so a fix to any of those happens exactly once.
 *
 * run_scalar_quant_gemm() goes one step further and holds the *entire* run()
 * body for the two bridges (tensor_quant, rowcolquant) that neither reshuffle
 * their operands nor carry a quant group size; those two sources are reduced to
 * an entry point plus their scale extents.
 *
 * The three bridges that do reshuffle (aquant, abquant, bquant) keep their own
 * argument construction here, and their host-side reshuffle steps live in
 * quant_bridge_shuffle.hpp.
 *
 * The generated kernel header (force-included before this one) must already
 * provide ck_tile (numeric_traits, stream_config, QuantGemmHostArgs, index_t)
 * and the KERNEL_NAME macro.
 */

#ifndef CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP
#define CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// GFX_ARCH is injected by CMake (-DGFX_ARCH="gfx942") and by every Python bridge.
// Define a sentinel when it is missing so a hand-rolled compile fails the
// compile-vs-runtime arch check with a readable message instead of dying on an
// undeclared identifier.
#ifndef GFX_ARCH
#define GFX_ARCH "unknown"
#endif

// On a HIP error, print the failing op name + file/line and return -1. RAII
// DeviceBuffers free themselves on the return, so no cleanup call is needed.
// Defined before the namespace because the helpers below use it too.
#define BRIDGE_HIP_CHECK(fn, call)                                                                \
    do                                                                                            \
    {                                                                                             \
        hipError_t _err = (call);                                                                 \
        if(_err != hipSuccess)                                                                    \
        {                                                                                         \
            std::cerr << (fn) << ": HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ \
                      << ":" << __LINE__ << "\n";                                                 \
            return -1;                                                                            \
        }                                                                                         \
    } while(0)

// Report the force-included kernel's name, or the empty string when this .so was
// built without one (the CMake no-kernel fallback path).
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
#define QUANT_BRIDGE_KERNEL_NAME() KERNEL_NAME
#else
#define QUANT_BRIDGE_KERNEL_NAME() ""
#endif

// Emit the C API boilerplate shared by every bridge. Invoke once at the top of
// each op's `extern "C"` block (it declares the file-local reference count that
// bridge_initialized() reads and the op's run() guard checks).
//
// Lifecycle: a reference count rather than a boolean, so several independent
// Python wrappers can share one loaded .so without the first destructor
// invalidating the others. Loads/stores are atomic with acquire/release
// ordering; the entry points themselves are still intended for single-threaded
// use (the Python ctypes harness) and are not otherwise synchronized.
#define QUANT_BRIDGE_C_API()                                                            \
    static std::atomic<int> g_ref_count{0};                                             \
    static bool bridge_initialized()                                                    \
    {                                                                                   \
        return g_ref_count.load(std::memory_order_acquire) > 0;                          \
    }                                                                                   \
    int dispatcher_initialize()                                                         \
    {                                                                                   \
        g_ref_count.fetch_add(1, std::memory_order_release);                            \
        return 0;                                                                       \
    }                                                                                   \
    const char* dispatcher_get_kernel_name() { return QUANT_BRIDGE_KERNEL_NAME(); }     \
    int dispatcher_init() { return dispatcher_initialize(); }                           \
    int dispatcher_get_kernel_count() { return 1; }                                     \
    int dispatcher_set_timing_config(                                                   \
        int flush_cache, int rotating_count, int cold_niters, int nrepeat)              \
    {                                                                                   \
        return quant_bridge::set_timing_config(                                         \
            flush_cache, rotating_count, cold_niters, nrepeat);                         \
    }                                                                                   \
    void dispatcher_cleanup()                                                           \
    {                                                                                   \
        int prev = g_ref_count.load(std::memory_order_relaxed);                         \
        while(prev > 0 && !g_ref_count.compare_exchange_weak(prev,                      \
                                                             prev - 1,                  \
                                                             std::memory_order_release, \
                                                             std::memory_order_relaxed)) \
            ;                                                                           \
    }

namespace quant_bridge {

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE

// RAII owner for a device allocation. Frees on scope exit, which removes the
// hand-written `cleanup` lambda that every bridge used to duplicate: any early
// return (including from BRIDGE_HIP_CHECK) releases every buffer automatically.
template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;

    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if(ptr)
            (void)hipFree(ptr);
    }

    // Allocate `bytes` bytes; returns the hipMalloc status for BRIDGE_HIP_CHECK.
    hipError_t allocate(std::size_t bytes) { return hipMalloc(&ptr, bytes); }

    operator T*() const { return ptr; }
};

// True if `arch` starts with any entry of the supported list. Prefix matching,
// because hipDeviceProp_t::gcnArchName carries feature suffixes (for example
// "gfx942:sramecc+:xnack-") and GFX_ARCH does not.
inline bool is_supported_arch(const std::string& arch, bool allow_gfx90a)
{
    return arch.rfind("gfx950", 0) == 0 || arch.rfind("gfx942", 0) == 0 ||
           (allow_gfx90a && arch.rfind("gfx90a", 0) == 0);
}

// Two checks, both required:
//   1. the arch this .so was *built* for (compile-time GFX_ARCH) is one this
//      bridge supports -- a typo or a newly added CMake target would otherwise
//      only surface as a wrong-answer kernel at runtime;
//   2. the device we are *running* on matches it -- a single-arch .so launched
//      on a different device yields a no-kernel-image failure, or worse, a
//      kernel built with the wrong warp_tile_k that silently returns zeros.
// The device arch is always queried; it is never assumed at compile time.
inline bool validate_supported_arch(const char* fn, bool allow_gfx90a = false)
{
    const std::string compiled_arch(GFX_ARCH);
    if(!is_supported_arch(compiled_arch, allow_gfx90a))
    {
        std::cerr << fn << ": compile-time GFX_ARCH '" << compiled_arch
                  << "' is not a supported architecture (supported: "
                  << (allow_gfx90a ? "gfx90a, " : "") << "gfx942, gfx950)\n";
        return false;
    }

    int dev = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
    {
        std::cerr << fn << ": could not query device architecture\n";
        return false;
    }
    const std::string arch(props.gcnArchName);
    if(!is_supported_arch(arch, allow_gfx90a))
    {
        std::cerr << fn << ": unsupported GPU architecture '" << arch
                  << "' (supported: " << (allow_gfx90a ? "gfx90a, " : "") << "gfx942, gfx950)\n";
        return false;
    }
    if(arch.rfind(compiled_arch, 0) != 0)
    {
        std::cerr << fn << ": runtime device architecture '" << arch
                  << "' does not match compile-time GFX_ARCH '" << compiled_arch
                  << "'; this .so was compiled for a different device\n";
        return false;
    }
    return true;
}

// Timing knobs for the measured launch. Previously hardcoded to
// cold_niters=3 / nrepeat=10 / flush_cache=false / rotating_count=1, which made
// it impossible to configure the bridge to match the Old-TE baseline it is
// benchmarked against (gemm_quant.cpp defaults flush_cache=true,
// rotating_count=1000). They are now settable, from the environment at first use
// and from the exported dispatcher_set_timing_config().
struct timing_config
{
    bool flush_cache    = false;
    int rotating_count  = 1;
    int cold_niters     = 3;
    int nrepeat         = 10;
};

inline int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if(!v || !*v)
        return fallback;
    char* end       = nullptr;
    const long temp = std::strtol(v, &end, 10);
    if(end == v || *end != '\0')
        return fallback;
    return static_cast<int>(temp);
}

inline timing_config& mutable_timing_config()
{
    static timing_config cfg = [] {
        timing_config c;
        c.cold_niters    = env_int("CK_BRIDGE_COLD_NITERS", c.cold_niters);
        c.nrepeat        = env_int("CK_BRIDGE_NREPEAT", c.nrepeat);
        c.rotating_count = env_int("CK_BRIDGE_ROTATING_COUNT", c.rotating_count);
        return c;
    }();
    return cfg;
}

// Backing implementation of the exported dispatcher_set_timing_config(). A
// negative argument leaves that field unchanged.
//
// flush_cache is REFUSED rather than stored: the generated SelectedKernel::launch()
// calls ck_tile::launch_kernel(), which ignores stream_config::flush_cache_ and
// rotating_count_ entirely -- Old-TE implements the rotating-buffer flush in its
// own invoker (run_gemm_quant_example.inc), not in launch_kernel. Storing the flag
// would make the bridge report a cache-flushed measurement it never performed.
inline int
set_timing_config(int flush_cache, int rotating_count, int cold_niters, int nrepeat)
{
    timing_config& cfg = mutable_timing_config();
    if(flush_cache > 0)
    {
        std::cerr << "dispatcher_set_timing_config: flush_cache is not supported by this bridge. "
                     "The generated kernel launch uses ck_tile::launch_kernel, which ignores "
                     "stream_config::flush_cache_/rotating_count_; enabling it requires a "
                     "flush-cache launch overload in the generated header.\n";
        return -2;
    }
    if(rotating_count > 0)
        cfg.rotating_count = rotating_count;
    if(cold_niters >= 0)
        cfg.cold_niters = cold_niters;
    if(nrepeat > 0)
        cfg.nrepeat = nrepeat;
    return 0;
}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE

// Build the stream_config used for a launch. When timing is requested use the
// GPU timer with the configured warmup / repeat counts; otherwise run once.
inline ck_tile::stream_config make_stream_config(bool do_time)
{
    const timing_config& cfg = mutable_timing_config();
    return ck_tile::stream_config{
        nullptr,                       // stream_id_
        do_time,                       // time_kernel_
        0,                             // log_level_
        do_time ? cfg.cold_niters : 0, // cold_niters_
        do_time ? cfg.nrepeat : 1,     // nrepeat_
        do_time,                       // is_gpu_timer_
        cfg.flush_cache,               // flush_cache_
        cfg.rotating_count,            // rotating_count_
    };
}

// Detect whether the force-included kernel exposes a launch() overload taking a
// per-launch preprocess hook. The hook is what lets split-K re-zero C before
// *every* invocation of the timing loop; without it a k_batch>1 timed run would
// accumulate C over cold_niters + nrepeat launches (the epilogue is atomic_add).
// The grouped kernels already provide the overload; the single-problem
// QuantGemmKernel headers do not yet, so the fallback below is still exercised.
template <typename KernelT, typename PreprocessFunc, typename = void>
struct has_preprocess_launch : std::false_type
{
};

template <typename KernelT, typename PreprocessFunc>
struct has_preprocess_launch<
    KernelT,
    PreprocessFunc,
    std::void_t<decltype(KernelT::launch(std::declval<const ck_tile::QuantGemmHostArgs&>(),
                                         std::declval<const ck_tile::stream_config&>(),
                                         std::declval<PreprocessFunc&>()))>> : std::true_type
{
};

// Direct-launch the force-included kernel. Returns the kernel execution time in
// ms, or a negative value if the kernel reports unsupported args (callers treat
// <0 as an error and return -2, matching the previous behavior).
//
// `preprocess` runs before every kernel invocation when the kernel supports the
// hook, and exactly once otherwise. Callers must not rely on it for correctness
// unless launch_supports_preprocess<KernelT, F>() is true.
template <typename KernelT, typename PreprocessFunc>
inline float
launch(const ck_tile::QuantGemmHostArgs& args, bool do_time, PreprocessFunc preprocess)
{
    const ck_tile::stream_config s = make_stream_config(do_time);
    if constexpr(has_preprocess_launch<KernelT, PreprocessFunc>::value)
    {
        return KernelT::launch(args, s, preprocess);
    }
    else
    {
        preprocess();
        return KernelT::launch(args, s);
    }
}

template <typename KernelT>
inline float launch(const ck_tile::QuantGemmHostArgs& args, bool do_time)
{
    return launch<KernelT>(args, do_time, [] {});
}

// The three guard helpers below replace the init / null-pointer / dimension
// checks that every bridge's run() used to inline verbatim. Each prints the same
// diagnostic as before and returns false so the caller can `return -1`. The
// per-op argument lists differ (4 vs 5 pointers; MNK plus op-specific QK/QN
// counts), so the pointer/dimension checks take an initializer_list.
inline bool check_initialized(const char* fn, bool initialized)
{
    if(!initialized)
    {
        std::cerr << fn << ": not initialized\n";
        return false;
    }
    return true;
}

inline bool check_non_null(const char* fn, std::initializer_list<const void*> ptrs)
{
    for(const void* p : ptrs)
    {
        if(!p)
        {
            std::cerr << fn << ": null pointer argument\n";
            return false;
        }
    }
    return true;
}

// Dimensions must be positive AND representable as ck_tile::index_t (int32):
// the C ABI takes int64_t but QuantGemmHostArgs stores index_t, so without the
// upper bound a >2^31 extent would wrap negative and the kernel would read out
// of bounds instead of reporting an error. The per-op stride checks that follow
// pin every stride to one of these dimensions, so they are covered too.
inline bool check_positive_dims(const char* fn, std::initializer_list<int64_t> dims)
{
    constexpr int64_t kIndexMax = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    for(int64_t d : dims)
    {
        if(d <= 0)
        {
            std::cerr << fn << ": invalid dimensions\n";
            return false;
        }
        if(d > kIndexMax)
        {
            std::cerr << fn << ": dimension " << d << " exceeds the " << kIndexMax
                      << " limit of ck_tile::index_t (int32)\n";
            return false;
        }
    }
    return true;
}

// k_batch is the split-K factor and is used as a divisor in the kernel's
// K-per-block computation, so it must be >= 1.
inline bool check_k_batch(const char* fn, int64_t k_batch)
{
    if(k_batch <= 0)
    {
        std::cerr << fn << ": k_batch must be >= 1, got " << k_batch
                  << " (k_batch is used as a divisor in split-K)\n";
        return false;
    }
    return true;
}

// The entry guard every bridge opens with, in the order they all used: init flag,
// null pointers, positive dimensions, then GPU arch. Returns false once anything
// fails (each helper has already printed its own diagnostic) so the caller can
// `return -1`.
//
// check_arch exists for abquant, which must run its compile-time fp4-preshuffle
// reject (return -3) between the argument checks and the arch check; it passes
// false here and calls validate_supported_arch() itself afterwards.
inline bool check_entry_args(const char* fn,
                             bool initialized,
                             std::initializer_list<const void*> ptrs,
                             std::initializer_list<int64_t> dims,
                             bool allow_gfx90a = false,
                             bool check_arch   = true)
{
    return check_initialized(fn, initialized) && check_non_null(fn, ptrs) &&
           check_positive_dims(fn, dims) &&
           (!check_arch || validate_supported_arch(fn, allow_gfx90a));
}

// Verify one caller-supplied scale count against the quant group size baked into
// this .so: `count` must equal ceil(dim / group). A mismatch means the host built
// its scale tensor for a different group size than the compiled kernel reads, so
// the kernel would index past the end of it.
inline bool check_quant_group_count(const char* fn,
                                    const char* count_name,
                                    int64_t count,
                                    const char* dim_name,
                                    int64_t dim,
                                    int64_t group)
{
    const int64_t expected = (dim + group - 1) / group;
    if(count == expected)
        return true;
    std::cerr << fn << ": " << count_name << " mismatch. Got " << count << ", expected " << expected
              << " for " << dim_name << "=" << dim << " with quant group size " << group << "\n";
    return false;
}

// The identical tail every bridge's run() ended with: direct-launch the
// force-included kernel (return -2 if it rejects the args), copy C back to the
// host, publish the optional timing, return 0. C_dev is any type convertible to
// const CT* (e.g. DeviceBuffer<CT>).
// This is also the single place the split-K accumulation invariant is enforced.
// k_batch > 1 selects the atomic_add epilogue, so C must be zero before *every*
// launch, not just the first; a timed run performs cold_niters + nrepeat
// launches. When the kernel exposes the preprocess hook the clear is threaded
// into the timing loop; when it does not, a timed split-K run cannot be made
// correct here and is rejected rather than returning an accumulated C.
template <typename KernelT, typename CT>
inline int launch_and_copyback(const char* fn,
                               const ck_tile::QuantGemmHostArgs& args,
                               void* C_host,
                               CT* C_dev,
                               std::size_t mn_elems,
                               float* time_ms)
{
    if(!check_k_batch(fn, args.k_batch))
        return -1;

    constexpr int64_t kIndexMax = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if(static_cast<int64_t>(mn_elems) > kIndexMax)
    {
        std::cerr << fn << ": M*N (" << mn_elems
                  << ") exceeds the range of ck_tile::index_t (int32)\n";
        return -1;
    }

    const bool do_time        = (time_ms != nullptr);
    const std::size_t c_bytes = elements_to_bytes<CT>(mn_elems);

    hipError_t clear_err = hipSuccess;
    auto clear_c         = [&] {
        if(args.k_batch > 1)
        {
            const hipError_t e = hipMemsetAsync(C_dev, 0, c_bytes, nullptr);
            // Record the first failure rather than aborting: this runs inside the
            // kernel-launch helper, which has no way to propagate an error out.
            if(e != hipSuccess && clear_err == hipSuccess)
                clear_err = e;
        }
    };

    if(do_time && args.k_batch > 1 &&
       !has_preprocess_launch<KernelT, decltype(clear_c)>::value)
    {
        std::cerr << fn << ": timed split-K (k_batch=" << args.k_batch
                  << ") is not supported by this kernel: its launch() has no per-launch "
                     "preprocess hook, so C cannot be re-zeroed between the warmup and repeat "
                     "launches of the atomic_add epilogue\n";
        return -3;
    }

    float exec_time = -1.0f;
    try
    {
        exec_time = launch<KernelT>(args, do_time, clear_c);
    }
    catch(const std::exception& e)
    {
        std::cerr << fn << ": kernel launch threw: " << e.what() << "\n";
        return -3;
    }
    catch(...)
    {
        std::cerr << fn << ": kernel launch threw unknown exception\n";
        return -3;
    }

    if(clear_err != hipSuccess)
    {
        std::cerr << fn << ": failed to clear C between split-K launches: "
                  << hipGetErrorString(clear_err) << "\n";
        return -1;
    }

    if(exec_time < 0.0f)
    {
        std::cerr << fn << ": kernel reported unsupported args\n";
        return -2;
    }

    const hipError_t err = hipMemcpy(C_host, C_dev, c_bytes, hipMemcpyDeviceToHost);
    if(err != hipSuccess)
    {
        std::cerr << fn << ": HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":"
                  << __LINE__ << "\n";
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

// The complete run() body shared by the tensor_quant and rowcolquant bridges.
// Neither reshuffles anything and neither has a quant group size, so both reduce
// to: guard, require packed strides, copy five buffers up, launch, copy C back.
// They differ only in how many scale elements each side carries -- tensor_quant
// passes one scalar per tensor (aq_elems = bq_elems = 1), rowcolquant one per A
// row / B column (M and N) -- which is why those are runtime arguments rather
// than another pair of template parameters.
//
// QK_A/QK_B and stride_AQ/stride_BQ are hardcoded to 1 for both: neither kernel
// has quant groups, and both index their scales by position rather than by a
// scale stride (mirrors the TensorQuant / RowColQuant branches of
// run_gemm_quant_example.inc).
template <typename KernelT, typename AT, typename BT, typename CT, typename QT>
inline int run_scalar_quant_gemm(const char* fn,
                                 bool initialized,
                                 const void* A,
                                 const void* B,
                                 const void* AQ,
                                 const void* BQ,
                                 void* C,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 int64_t stride_A,
                                 int64_t stride_B,
                                 int64_t stride_C,
                                 std::size_t aq_elems,
                                 std::size_t bq_elems,
                                 int k_batch,
                                 float* time_ms)
{
    if(!check_entry_args(fn, initialized, {A, B, AQ, BQ, C}, {M, N, K}))
        return -1;

    // Only packed (contiguous) layouts are supported: A is [M,K] row-major, B is
    // [K,N] column-major (leading dim K), C is [M,N] row-major.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << fn << ": non-packed strides are not supported. Expected stride_A=" << K
                  << " stride_B=" << K << " stride_C=" << N << ", got stride_A=" << stride_A
                  << " stride_B=" << stride_B << " stride_C=" << stride_C << "\n";
        return -1;
    }

    DeviceBuffer<AT> A_dev;
    DeviceBuffer<BT> B_dev;
    DeviceBuffer<QT> AQ_dev;
    DeviceBuffer<QT> BQ_dev;
    DeviceBuffer<CT> C_dev;
    BRIDGE_HIP_CHECK(fn, A_dev.allocate(elements_to_bytes<AT>(M * K)));
    BRIDGE_HIP_CHECK(fn, B_dev.allocate(elements_to_bytes<BT>(K * N)));
    BRIDGE_HIP_CHECK(fn, AQ_dev.allocate(elements_to_bytes<QT>(aq_elems)));
    BRIDGE_HIP_CHECK(fn, BQ_dev.allocate(elements_to_bytes<QT>(bq_elems)));
    BRIDGE_HIP_CHECK(fn, C_dev.allocate(elements_to_bytes<CT>(M * N)));

    BRIDGE_HIP_CHECK(fn, hipMemcpy(A_dev, A, elements_to_bytes<AT>(M * K), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn, hipMemcpy(B_dev, B, elements_to_bytes<BT>(K * N), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn,
                     hipMemcpy(AQ_dev, AQ, elements_to_bytes<QT>(aq_elems), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn,
                     hipMemcpy(BQ_dev, BQ, elements_to_bytes<QT>(bq_elems), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn, hipMemset(C_dev, 0, elements_to_bytes<CT>(M * N)));

    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = 1;
    args.QK_B      = 1;
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 1;
    args.stride_BQ = 1;

    return launch_and_copyback<KernelT, CT>(
        fn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

// The complete run() body shared by the two grouped scalar-quant bridges
// (grouped rowcolquant, grouped tensorquant). They target
// ck_tile::QuantGroupedGemmKernel, whose launch() takes a *vector* of
// QuantGroupedGemmHostArgs plus a device kargs buffer and a per-launch
// preprocess hook -- an entirely different contract from the single-problem
// QuantGemmHostArgs path above, which is why they cannot share
// run_scalar_quant_gemm(). Each call still launches exactly one problem; the
// "grouped" in the name refers to the kernel contract, not to batching.
//
// The two ops differ in three values only, all passed in:
//   expected_qk_a / expected_qk_b -- the required scale counts, which double as
//       the scale-buffer extents (rowcolquant: M and N; tensorquant: 1 and 1);
//   args_scale_stride -- the stride_AQ/stride_BQ written into the host args.
//       rowcolquant writes 0 and tensorquant writes 1, and both sources document
//       the field as unread by the kernel. That contradiction is preserved here
//       rather than resolved: making them agree is a kernel-contract decision,
//       not a refactor.
template <typename KernelT, typename AT, typename BT, typename CT, typename AQT, typename BQT>
inline int run_scalar_quant_grouped_gemm(const char* fn,
                                         bool initialized,
                                         const void* A,
                                         const void* B,
                                         const void* AQ,
                                         const void* BQ,
                                         void* C,
                                         int64_t M,
                                         int64_t N,
                                         int64_t K,
                                         int64_t stride_A,
                                         int64_t stride_B,
                                         int64_t stride_AQ,
                                         int64_t stride_BQ,
                                         int64_t stride_C,
                                         int64_t QK_A,
                                         int64_t QK_B,
                                         int64_t expected_qk_a,
                                         int64_t expected_qk_b,
                                         ck_tile::index_t args_scale_stride,
                                         int k_batch,
                                         float* time_ms)
{
    if(!check_entry_args(fn, initialized, {A, B, AQ, BQ, C}, {M, N, K}))
        return -1;
    if(!check_k_batch(fn, k_batch))
        return -1;

    // The kernel derives the scale counts from the problem shape and indexes the
    // buffers directly, so a smaller count is a device out-of-bounds read.
    if(QK_A != expected_qk_a || QK_B != expected_qk_b)
    {
        std::cerr << fn << ": QK_A/QK_B mismatch. Got (" << QK_A << ", " << QK_B
                  << "), expected (" << expected_qk_a << ", " << expected_qk_b << ")\n";
        return -1;
    }

    // Only packed (contiguous) layouts are supported: A is [M,K] row-major, B is
    // [K,N] column-major (leading dim K, NOT N), C is [M,N] row-major.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << fn << ": non-packed strides are not supported. Expected stride_A=" << K
                  << " stride_B=" << K << " stride_C=" << N << ", got stride_A=" << stride_A
                  << " stride_B=" << stride_B << " stride_C=" << stride_C << "\n";
        return -1;
    }

    // Rejected rather than ignored: the kernel builds its scale views with
    // literal strides, so any other value would silently do nothing at all.
    if(stride_AQ != 1 || stride_BQ != 1)
    {
        std::cerr << fn << ": stride_AQ and stride_BQ must be 1 (the kernel hardwires its scale "
                  << "strides); got stride_AQ=" << stride_AQ << " stride_BQ=" << stride_BQ << "\n";
        return -1;
    }

    constexpr int64_t kIndexMax = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if(M > kIndexMax / N)
    {
        std::cerr << fn << ": M*N (" << M << "*" << N
                  << ") exceeds the range of ck_tile::index_t (int32)\n";
        return -1;
    }

    DeviceBuffer<AT> A_dev;
    DeviceBuffer<BT> B_dev;
    DeviceBuffer<AQT> AQ_dev;
    DeviceBuffer<BQT> BQ_dev;
    DeviceBuffer<CT> C_dev;
    DeviceBuffer<ck_tile::QuantGemmTransKernelArg> kargs_dev;
    const std::size_t c_bytes = elements_to_bytes<CT>(static_cast<std::size_t>(M) * N);
    BRIDGE_HIP_CHECK(fn, A_dev.allocate(elements_to_bytes<AT>(M * K)));
    BRIDGE_HIP_CHECK(fn, B_dev.allocate(elements_to_bytes<BT>(K * N)));
    BRIDGE_HIP_CHECK(fn, AQ_dev.allocate(elements_to_bytes<AQT>(expected_qk_a)));
    BRIDGE_HIP_CHECK(fn, BQ_dev.allocate(elements_to_bytes<BQT>(expected_qk_b)));
    BRIDGE_HIP_CHECK(fn, C_dev.allocate(c_bytes));
    BRIDGE_HIP_CHECK(fn, kargs_dev.allocate(sizeof(ck_tile::QuantGemmTransKernelArg)));

    BRIDGE_HIP_CHECK(fn, hipMemcpy(A_dev, A, elements_to_bytes<AT>(M * K), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn, hipMemcpy(B_dev, B, elements_to_bytes<BT>(K * N), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(
        fn, hipMemcpy(AQ_dev, AQ, elements_to_bytes<AQT>(expected_qk_a), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(
        fn, hipMemcpy(BQ_dev, BQ, elements_to_bytes<BQT>(expected_qk_b), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn, hipMemset(C_dev, 0, c_bytes));

    const std::vector<ck_tile::QuantGroupedGemmHostArgs> gemm_descs = {
        ck_tile::QuantGroupedGemmHostArgs(A_dev,
                                          B_dev,
                                          C_dev,
                                          AQ_dev,
                                          BQ_dev,
                                          static_cast<ck_tile::index_t>(k_batch),
                                          static_cast<ck_tile::index_t>(M),
                                          static_cast<ck_tile::index_t>(N),
                                          static_cast<ck_tile::index_t>(K),
                                          1, // QK_A: one quant group covers the tensor
                                          1, // QK_B
                                          static_cast<ck_tile::index_t>(stride_A),
                                          static_cast<ck_tile::index_t>(stride_B),
                                          static_cast<ck_tile::index_t>(stride_C),
                                          args_scale_stride,
                                          args_scale_stride)};

    const bool do_time             = (time_ms != nullptr);
    const ck_tile::stream_config s = make_stream_config(do_time);

    // Split-K selects the atomic_add epilogue, so C must start at zero before
    // *every* launch -- with timing enabled the kernel runs cold_niters + nrepeat
    // times, and a C zeroed only once would come back holding their sum. The
    // grouped launch() forwards this hook to ck_tile::launch_kernel_time_mask,
    // which calls it before each invocation. For k_batch == 1 the epilogue is
    // `set` and repeated launches are idempotent, so the clear is skipped to keep
    // the timing loop measuring only the kernel.
    hipError_t clear_err = hipSuccess;
    auto clear_c         = [&] {
        if(k_batch > 1)
        {
            const hipError_t e = hipMemsetAsync(C_dev, 0, c_bytes, s.stream_id_);
            if(e != hipSuccess && clear_err == hipSuccess)
                clear_err = e;
        }
    };

    float exec_time = -1.0f;
    try
    {
        exec_time = KernelT::launch(gemm_descs, s, kargs_dev.ptr, clear_c);
    }
    catch(const std::exception& e)
    {
        std::cerr << fn << ": kernel launch threw: " << e.what() << "\n";
        return -3;
    }
    catch(...)
    {
        std::cerr << fn << ": kernel launch threw unknown exception\n";
        return -3;
    }

    if(clear_err != hipSuccess)
    {
        std::cerr << fn << ": failed to clear C between split-K launches: "
                  << hipGetErrorString(clear_err) << "\n";
        return -1;
    }
    if(exec_time < 0.0f)
    {
        std::cerr << fn << ": kernel reported unsupported args\n";
        return -2;
    }

    BRIDGE_HIP_CHECK(fn, hipMemcpy(C, C_dev, c_bytes, hipMemcpyDeviceToHost));

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

#endif // CK_TILE_SINGLE_KERNEL_INCLUDE

} // namespace quant_bridge

#endif // CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP
