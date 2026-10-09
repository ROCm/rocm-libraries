/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
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

#pragma once

#include "benchmark_timing.hpp"
#include "device_vector.hpp"
#include "efficiency_monitor.hpp"
#include "fast_check.hpp"
#include "flops.hpp"
#include "hipBuffer.hpp"
#include "hipblaslt_bench_options.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <hipblaslt/client/MatmulPreparation.hpp>
#include <hipblaslt/client/MatmulProblem.hpp>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hostnumerics/HipblasltDataInitialization.hpp>
#include <hipblaslt/hostnumerics/MatmulValidation.hpp>
#include <hipblaslt/hostnumerics/Types.hpp>
#include <hipblaslt/hostnumerics/near.hpp>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <omp.h>
#include <optional>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <roc/hostnumerics/backends/blas.hpp>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/mx.hpp>
#include <roc/hostnumerics/reduction.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>
#include <set>
#include <span>
#include <vector>

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

template <typename T>
void swizzle_tensor(T*               dst,
                    const T*         src,
                    hipDataType      datatype,
                    const Arguments& arg,
                    size_t           b,
                    size_t           m_n,
                    size_t           k,
                    size_t           ld,
                    bool             colMaj)
{
    if(ld < k)
        throw std::runtime_error("invalid value of ld in swizzle_tensor: ld must be >= k.");

    using roc::hostnumerics::Layout;
    using roc::hostnumerics::ScalarType;
    using roc::hostnumerics::Shape;
    using roc::hostnumerics::Tensor;

    // currently, if A then it means MiM = 16, if B then it means MiN = 16
    constexpr size_t MiM_N = 16;
    const auto parameters  = hipblaslt::client::matmulSwizzleParameters(datatype, arg.compute_type);
    const size_t     MiK   = parameters.innerBlock;
    const size_t     MiKv  = parameters.vectorWidth;
    const size_t     PackK = parameters.packingFactor;
    const size_t     numElements = b * m_n * k;
    const ScalarType tensorType  = datatype == HIP_R_4F_E2M1
                                       ? ScalarType::UInt8
                                       : hipblaslt::hostnumerics::scalarType(datatype);
    std::vector<T>   compact(numElements);

    if(colMaj)
    {
        for(size_t i = 0; i < b * k; i++)
        {
            std::copy(src + (i * ld), src + (i * ld) + m_n, compact.data() + (i * m_n));
        }
    }
    else
    {
        for(size_t i = 0; i < b * m_n; i++)
        {
            std::copy(src + (i * ld), src + (i * ld) + k, compact.data() + (i * k));
        }
    }

    Tensor tmpTensor = Tensor::copyEncodedBackingStorage(
        tensorType,
        Layout::contiguousLastDimensionFastest(
            Shape(colMaj ? std::vector<size_t>{b, k, m_n} : std::vector<size_t>{b, m_n, k})),
        std::as_bytes(std::span<const T>(compact)));
    if(colMaj)
        tmpTensor = tmpTensor.copyWithPermutedDimensions({0, 2, 1});

    auto       MultipleM_N  = MiM_N;
    auto       MultipleK    = MiK * PackK;
    const auto paddedM_N    = (m_n / MultipleM_N + !!(m_n % MultipleM_N)) * MultipleM_N;
    const auto paddedK      = (k / MultipleK + !!(k % MultipleK)) * MultipleK;
    Tensor     paddedTensor = tmpTensor.copyWithZeroPadding(Shape{b, paddedM_N, paddedK});
    Tensor     reshaped     = paddedTensor.reshapeSharingStorage(
        Shape{b, paddedM_N / MiM_N, MiM_N, paddedK / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor       permuted    = reshaped.copyWithPermutedDimensions({0, 1, 3, 4, 2, 5});
    const size_t outputBytes = b * paddedM_N * paddedK * sizeof(T);
    if(permuted.rawEncodedBackingStorage().size() != outputBytes)
        throw std::runtime_error("swizzle_tensor produced an unexpected storage size.");
    std::memcpy(static_cast<void*>(dst), permuted.rawEncodedBackingStorage().data(), outputBytes);
}

void swizzle_tensor_type(HipHostBuffer&       dst,
                         const HipHostBuffer& src,
                         hipDataType          datatype,
                         const Arguments&     arg,
                         size_t               b,
                         size_t               m_n,
                         size_t               k,
                         size_t               ld,
                         bool                 colMaj)
{
    switch(datatype)
    {
    case HIP_R_32F:
        swizzle_tensor<float>(
            dst.as<float>(), src.as<float>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_16F:
        swizzle_tensor<hipblasLtHalf>(
            dst.as<hipblasLtHalf>(), src.as<hipblasLtHalf>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_16BF:
        swizzle_tensor<hip_bfloat16>(
            dst.as<hip_bfloat16>(), src.as<hip_bfloat16>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_8F_E4M3_FNUZ:
        swizzle_tensor<hipblaslt_f8_fnuz>(dst.as<hipblaslt_f8_fnuz>(),
                                          src.as<hipblaslt_f8_fnuz>(),
                                          datatype,
                                          arg,
                                          b,
                                          m_n,
                                          k,
                                          ld,
                                          colMaj);
        return;
    case HIP_R_8F_E5M2_FNUZ:
        swizzle_tensor<hipblaslt_bf8_fnuz>(dst.as<hipblaslt_bf8_fnuz>(),
                                           src.as<hipblaslt_bf8_fnuz>(),
                                           datatype,
                                           arg,
                                           b,
                                           m_n,
                                           k,
                                           ld,
                                           colMaj);
        return;
    case HIP_R_8F_E4M3:
        swizzle_tensor<hipblaslt_f8>(
            dst.as<hipblaslt_f8>(), src.as<hipblaslt_f8>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_8F_E5M2:
        swizzle_tensor<hipblaslt_bf8>(
            dst.as<hipblaslt_bf8>(), src.as<hipblaslt_bf8>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_4F_E2M1:
        // fp4: 2 elements per byte, so k_bytes = k/2, ld_bytes = ld/2
        swizzle_tensor<uint8_t>(
            dst.as<uint8_t>(), src.as<uint8_t>(), datatype, arg, b, m_n, k / 2, ld / 2, colMaj);
        return;
    default:
        throw std::invalid_argument("Unsupported datatype in swizzle_tensor_type().");
    }
}

inline void pre_gpu_time(bool         use_gpu_timer,
                         hipEvent_t&  event_gpu_time_start,
                         double&      gpu_time_used,
                         hipStream_t& stream)
{
    if(use_gpu_timer)
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
    else
        gpu_time_used = get_time_us_sync(stream);
}
inline void post_gpu_time(bool         use_gpu_timer,
                          hipEvent_t&  event_gpu_time_start,
                          hipEvent_t&  event_gpu_time_end,
                          double&      gpu_time_used,
                          hipStream_t& stream)
{
    if(use_gpu_timer)
    {
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
        float gpu_time_ms;
        CHECK_HIP_ERROR(
            hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
        gpu_time_used = gpu_time_ms * 1000; // ms to us
    }
    else
    {
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
    }
}

void testing_matmul_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipblasOperation_t transA = HIPBLAS_OP_T;
    const hipblasOperation_t transB = HIPBLAS_OP_N;

    // allocate memory on device
    HipDeviceBuffer dA(arg.a_type, safe_size / 2, arg.HMM);
    HipDeviceBuffer dB(arg.b_type, safe_size, arg.HMM);
    HipDeviceBuffer dC(arg.c_type, safe_size, arg.HMM);
    HipDeviceBuffer dD(arg.d_type, safe_size, arg.HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout matA(M, K, lda, arg.a_type);
    hipblaslt_local_matrix_layout matB(K, N, ldb, arg.b_type);
    hipblaslt_local_matrix_layout matC(M, N, ldc, arg.c_type);
    hipblaslt_local_matrix_layout matD(M, N, ldc, arg.d_type);
    hipblaslt_local_matmul_descr  matmul(transA,
                                        transB,
                                        arg.compute_type,
                                        arg.scale_type,
                                        arg.compute_input_typeA,
                                        arg.compute_input_typeB);

    size_t                     workspace_size = 0;
    hipblaslt_local_preference pref;

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
}

void copy_gemm_to_host(hipStream_t                   stream,
                       const uint32_t&               problem_count,
                       std::vector<HipHostBuffer>&   hDst,
                       std::vector<HipDeviceBuffer>& dSrc)
{

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
    {
        CHECK_HIP_ERROR(synchronize(hDst[gemmIdx], dSrc[gemmIdx], 0, 0, 0, 0, 1, false, stream));
    }
}

template <typename T>
void dumpBuffer(const char* title, T* buf, size_t M, size_t N)
{
    hipblaslt_cout << "----- DUMP: " << title << " -----" << std::endl;
    for(int n = 0; n < N; n++)
    {
        for(int m = 0; m < M; m++)
        {
            hipblaslt_cout << buf[m + n * M] << " ";
        }
        hipblaslt_cout << std::endl;
    }
}

void dumpBuffer(const char* title, hipDataType To, HipHostBuffer& buf, size_t M, size_t N)
{
    switch(To)
    {
    case HIP_R_32F:
        dumpBuffer(title, buf.as<float>(), M, N);
        break;
    default:
        hipblaslt_cerr << "Unsupported dumpBuffer data type" << std::endl;
        break;
    }

    return;
}

// Returns why fast_check cannot verify this configuration, or an empty string when it can.
inline std::string fast_check_unsupported_reason(const Arguments&     arg,
                                                 hipblasLtBatchMode_t batchMode,
                                                 bool                 do_swizzle,
                                                 hipDataType          TiA,
                                                 hipDataType          TiB,
                                                 hipDataType          TiC,
                                                 hipDataType          To,
                                                 hipDataType          Tc)
{
    if(arg.initialization != hipblaslt_initialization::integer_exact)
        return "fast_check requires initialization: integer_exact";
    if(arg.timing)
        return "fast_check does not support timing runs";
    if(batchMode != HIPBLASLT_BATCH_MODE_STRIDED || arg.grouped_gemm > 0)
        return "fast_check supports single strided-batched GEMMs only";
    if(arg.gradient
       && (arg.activation_type != hipblaslt_activation_type::none || arg.use_e || !arg.bias_vector
           || arg.bias_source == hipblaslt_bias_source::d))
        return "fast_check supports a gradient epilogue only as a bias gradient from A or B "
               "(BGRADA, BGRADB), with no activation and no E input";
    if(arg.gradient && arg.batch_count > 1)
        return "fast_check checks a bias gradient for a single batch only";
    if(!arg.gradient && arg.activation_type != hipblaslt_activation_type::none
       && arg.activation_type != hipblaslt_activation_type::relu
       && arg.activation_type != hipblaslt_activation_type::clamp)
        return "fast_check checks only the relu and clamp activations, which keep integers "
               "exact; GELU, SiLU and sigmoid do not";
    if(!arg.gradient && arg.activation_type != hipblaslt_activation_type::none && !arg.use_e)
        return "fast_check checks an activation through E, the pre-activation output, so it "
               "requires use_e";
    // MX: integer_exact fills fp8 elements with small integers and E8M0 scales with 1, 2 or 4,
    // and fast_check checks against the dequantized values, so both operands must be MX fp8.
    if(isBlockScaling(arg.scaleA) || isBlockScaling(arg.scaleB))
    {
        auto isMxFp8 = [](hipDataType t) { return t == HIP_R_8F_E4M3 || t == HIP_R_8F_E5M2; };
        if(!isBlockScaling(arg.scaleA) || !isBlockScaling(arg.scaleB))
            return "fast_check supports MX scales only on both A and B";
        if(!isMxFp8(TiA) || !isMxFp8(TiB))
            return "fast_check supports MX scales only with fp8 (E4M3, E5M2) A and B";
        // integer_exact writes E8M0 scale codes, which a UE4M3 or UE5M3 format would read as
        // other values.
        if(scaleDataType(arg.scaleA) != HIP_R_8F_UE8M0
           || scaleDataType(arg.scaleB) != HIP_R_8F_UE8M0)
            return "fast_check supports MX scales only in an E8M0 (UE8M0) format";
        // Other orientations have not established agreement between the generator's
        // reference and the device scale layout. In particular, the MX caller currently
        // treats C differently from T even for real inputs. Refuse before allocation.
        if(char_to_hipblas_operation(arg.transA) != HIPBLAS_OP_T
           || char_to_hipblas_operation(arg.transB) != HIPBLAS_OP_N)
            return "fast_check with MX scales requires transA=T and transB=N";
        // Keep the established fast_check MX coverage to complete K blocks.
        if(arg.K[0] % blockSize(arg.scaleA) != 0 || arg.K[0] % blockSize(arg.scaleB) != 0)
            return "fast_check supports MX scales only when K is a multiple of the scale block";
        if(arg.batch_count > 1)
            return "fast_check supports MX scales for a single batch only";
    }
    if(do_swizzle)
        return "fast_check does not support swizzled A or B";
    // Without a gradient, fast_check models the bias as one value per row of D, which is what
    // bias_source a and d allocate. bias_source b allocates one value per column.
    if(!arg.gradient && arg.bias_vector && arg.bias_source == hipblaslt_bias_source::b)
        return "fast_check supports a bias with one value per row of D (bias_source a or d), not "
               "bias_source b";
    // A bias gradient from B has one value per column of D; every other bias, one per row.
    const int64_t bias_length
        = arg.gradient && arg.bias_source == hipblaslt_bias_source::b ? arg.N[0] : arg.M[0];
    if(arg.bias_vector && arg.bias_stride > 0 && arg.bias_stride < bias_length)
        return "fast_check requires bias_stride to be at least the bias vector's length, "
               + std::to_string(bias_length);
    for(hipDataType t : {TiA, TiB, TiC, To, Tc})
    {
        std::string why;
        if(!fast_check_supported_type(t, &why))
            return why;
    }
    return {};
}

// Returns why this process cannot select Stream-K solutions, or an empty string when it can.
// TensileLite reads TENSILE_SOLUTION_SELECTION_METHOD once per process (Debug::Instance). A case's
// tensile_solution_selection_method sets the variable per handle, after that read, so a case that
// needs the Stream-K solutions needs the variable set when the process starts; this is that value.
// On gfx950 the Stream-K library is the only one, and the variable has no effect.
inline const bool kStreamKSelectedAtStartup = [] {
    const char* method = getenv("TENSILE_SOLUTION_SELECTION_METHOD");
    return method && !strcmp(method, "2");
}();

inline std::string streamk_unavailable_reason()
{
    int device = 0;
    if(hipGetDevice(&device) == hipSuccess)
    {
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, device) == hipSuccess
           && !strncmp(props.gcnArchName, "gfx950", 6))
            return {};
    }
    if(kStreamKSelectedAtStartup)
        return {};
    return "this case covers Stream-K solutions, which the library offers only when the process "
           "starts with TENSILE_SOLUTION_SELECTION_METHOD=2";
}

// Skips the test when buffer placement is unavailable on this platform, and fails it when the
// placement request is invalid.
#ifdef GOOGLE_TEST
#define CHECK_PLACEMENT(ok, unsupported, why) \
    do                                        \
    {                                         \
        if(!(ok))                             \
        {                                     \
            if(unsupported)                   \
                GTEST_SKIP() << (why);        \
            else                              \
                FAIL() << (why);              \
        }                                     \
    } while(0)
#else
#define CHECK_PLACEMENT(ok, unsupported, why)     \
    do                                            \
    {                                             \
        if(!(ok))                                 \
        {                                         \
            hipblaslt_cerr << (why) << std::endl; \
            return;                               \
        }                                         \
    } while(0)
#endif

// Seed for the fast_check probe vectors: FNV-1a over the test name, so a failure reproduces.
inline uint64_t fast_check_seed(const Arguments& arg)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for(const char* c = arg.name; *c; c++)
        h = (h ^ uint8_t(*c)) * 0x100000001b3ull;
    return h;
}

// Names one solution for failure messages: its position in this test's list and its library
// index. with_kernel adds the kernel name.
inline std::string solution_description(hipblasLtHandle_t      handle,
                                        hipblasLtMatmulAlgo_t& algo,
                                        size_t                 sol,
                                        size_t                 count,
                                        bool                   with_kernel)
{
    std::ostringstream s;
    s << "solution " << sol << " of " << count << " (library index "
      << hipblaslt_ext::getIndexFromAlgo(algo);
    if(with_kernel)
        s << ", kernel " << hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
    s << ")";
    return s.str();
}

std::vector<roc::hostnumerics::ComparisonTolerance>
    matmulValidationTolerances(const Arguments&                                  arg,
                               std::span<const hipblaslt::client::MatmulProblem> matmulProblems,
                               hipDataType                                       inputTypeA,
                               hipDataType                                       inputTypeB,
                               hipDataType                                       outputType,
                               hipDataType                                       computeType)
{
    std::vector<roc::hostnumerics::ComparisonTolerance> tolerances(matmulProblems.size());
    const bool                                           bfloat16Output = outputType == HIP_R_16BF;

    if(arg.unit_check && hipblaslt_get_arch_major() == 11 && realDataTypeSize(inputTypeA) == 2
       && realDataTypeSize(inputTypeB) == 2)
    {
        for(size_t i = 0; i < matmulProblems.size(); ++i)
        {
            const double coefficient
                = gfx11_low_precision_accumulation_tolerance_coefficient(computeType,
                                                                         matmulProblems[i].k);
            tolerances[i].absolute = coefficient;
            tolerances[i].relative = 2.0 * coefficient;
            if(bfloat16Output)
            {
                const double outputCoefficient
                    = bfloat16_output_rounding_tolerance_coefficient();
                tolerances[i].absolute = std::max(tolerances[i].absolute, outputCoefficient);
                tolerances[i].relative
                    = std::max(tolerances[i].relative, 2.0 * outputCoefficient);
            }
        }
    }

    if(arg.initialization == hipblaslt_initialization::fp16_accumulator_probe
       || arg.initialization == hipblaslt_initialization::norm_dist_one_special)
    {
        for(auto& tolerance : tolerances)
        {
            tolerance          = {};
            tolerance.absolute = 1e-2;
        }
    }
    return tolerances;
}

void testing_matmul_with_bias(const Arguments&                                  arg,
                              std::span<const hipblaslt::client::MatmulProblem> matmulProblems,
                              const hipblaslt::client::MatmulDataTypes&         dataTypes);

void testing_matmul(const Arguments& arg)
{
    const auto dataTypes   = hipblaslt::client::resolveMatmulDataTypes(arg);
    Arguments  arg_revised = arg;
    arg_revised.bias_type  = dataTypes.bias;
    arg_revised.aux_type   = dataTypes.auxiliary;

    const auto matmulProblems = hipblaslt::client::normalizeMatmulProblems(arg_revised);

    if(arg.batch_count == 0)
        return;

    // Set the values of flush, rotating size, cold_iters and hot_iters only for internal use
    hipblasltSetFlushValue(arg.flush);
    hipblasltSetRotatingBufferSizeValue(arg.rotating);
    hipblasltSetColdIterationsValue(arg.cold_iters);
    hipblasltSetHotIterationsValue(arg.iters);

    // integer_exact: skip gfx11 only for 16-bit A (fp16/bf16)—GPU vs CPU exact match unreliable there;
    // f32/f64 (and TF32x1 f32 path) still run on Navi.
    if(arg.initialization == hipblaslt_initialization::integer_exact)
    {
        const bool is_16bit = (arg.a_type == HIP_R_16F || arg.a_type == HIP_R_16BF);
        if(hipblaslt_get_arch_major() == 11 && is_16bit)
        {
            hipblaslt_cout << "Skipping integer_exact on gfx11 for 16-bit float (fp16/bf16 A)"
                           << std::endl;
            return;
        }
        // fast_check models the rounding of large results into 16-bit outputs, so the limit
        // applies only when a host reference comparison runs.
        // A misspelled pattern must fail here rather than be hidden by the skip below.
        hipblaslt::hostnumerics::IntegerExactPattern pattern
            = hipblaslt::hostnumerics::IntegerExactPattern::Standard;
        if(!hipblaslt::hostnumerics::parseIntegerExactPattern(arg.integer_exact_pattern, pattern))
        {
            std::string why = std::string("unknown integer_exact_pattern '")
                              + arg.integer_exact_pattern + "'; use ternary or sparse_k";
#ifdef GOOGLE_TEST
            FAIL() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
        // sparse_k keeps every result below 2 * 17 * 4 + 4 = 140 whatever K is, as long as alpha
        // and beta are at most 2 and nothing else scales the result.
        const bool host_reference = arg.unit_check || arg.norm_check || arg.allclose_check;
        const bool sparse_k_bounded
            = pattern == hipblaslt::hostnumerics::IntegerExactPattern::SparseK
              && std::fabs(arg.alpha) <= 2 && std::fabs(arg.beta) <= 2 && !arg.scaleAlpha_vector
              && !arg.bias_vector && arg.scaleA == hipblaslt_scaling_format::none
              && arg.scaleB == hipblaslt_scaling_format::none && !arg.scaleC && !arg.scaleD;
        if(is_16bit && (host_reference || !arg.fast_check) && !sparse_k_bounded)
        {
            // alpha=2: |2*dot|<=8K; beta=-2 adds 2*C. fp16 exact int ~2048 => K<=256 for both betas used
            const int32_t k_limit
                = (arg.alpha == 2.0f && (arg.beta == 0.0f || arg.beta == -2.0f)) ? 256 : 512;
            for(const auto& problem : matmulProblems)
            {
                if(problem.k > k_limit)
                {
                    hipblaslt_cout << "Skipping integer_exact: 16-bit format with K=" << problem.k
                                   << " > " << k_limit << " (exact representability limit)"
                                   << std::endl;
                    return;
                }
            }
        }
    }

    // FP16 full-matrix accumulator probe: alternate large products so premature
    // Float16 accumulation rounding is visible without overflowing the final sum.
    if(arg.initialization == hipblaslt_initialization::fp16_accumulator_probe)
    {
        if(arg.a_type != HIP_R_16F || arg.b_type != HIP_R_16F || arg.c_type != HIP_R_16F
           || arg.d_type != HIP_R_16F || arg.compute_type != HIPBLAS_COMPUTE_32F)
        {
            hipblaslt_cout
                << "Skipping fp16_accumulator_probe: requires f16 A/B/C/D and HIPBLAS_COMPUTE_32F"
                << std::endl;
            return;
        }
        if(matmulProblems.front().operationA != HIPBLAS_OP_N
           || matmulProblems.front().operationB != HIPBLAS_OP_N)
        {
            hipblaslt_cout << "Skipping fp16_accumulator_probe: only NN transposes supported"
                           << std::endl;
            return;
        }
        if(arg.grouped_gemm > 0)
        {
            hipblaslt_cout << "Skipping fp16_accumulator_probe: grouped_gemm not supported"
                           << std::endl;
            return;
        }
        if(arg.bias_vector || arg.activation_type != hipblaslt_activation_type::none || arg.use_e
           || arg.gradient || arg.scaleA != hipblaslt_scaling_format::none
           || arg.scaleB != hipblaslt_scaling_format::none || arg.scaleC || arg.scaleD || arg.scaleE
           || arg.scaleAlpha_vector || arg.amaxScaleA || arg.amaxScaleB || arg.amaxD)
        {
            hipblaslt_cout
                << "Skipping fp16_accumulator_probe: requires default epilogue (no bias, "
                   "activation, aux, or scaling)"
                << std::endl;
            return;
        }
        if(arg.beta != 0.0f)
        {
            hipblaslt_cout << "Skipping fp16_accumulator_probe: requires beta == 0" << std::endl;
            return;
        }
        for(const auto& problem : matmulProblems)
        {
            if((problem.k & 1) != 0)
            {
                hipblaslt_cout << "Skipping fp16_accumulator_probe: odd K not supported (K="
                               << problem.k << ")" << std::endl;
                return;
            }
        }
    }

    const bool lowPrecisionInput
        = (realDataTypeSize(arg.a_type) == 1 || realDataTypeSize(arg.b_type) == 1)
          && dataTypes.computeScalar != HIP_R_32I;
    if(!lowPrecisionInput && arg.d_type != HIP_R_16F && arg.d_type != HIP_R_16BF
       && arg.d_type != HIP_R_32F && arg.d_type != HIP_R_32I && arg.d_type != HIP_R_8I
       && arg.d_type != HIP_R_64F && arg.d_type != HIP_C_32F && arg.d_type != HIP_C_64F)
    {
        hipblaslt_test_invalid{}(arg);
        return;
    }

    testing_matmul_with_bias(arg_revised, matmulProblems, dataTypes);
}

void testing_matmul_with_bias(const Arguments&                                  arg,
                              std::span<const hipblaslt::client::MatmulProblem> matmulProblems,
                              const hipblaslt::client::MatmulDataTypes&         dataTypes)
{
    const auto& firstProblem = matmulProblems.front();

    const hipDataType          TiA             = firstProblem.a.apiType;
    const hipDataType          TiB             = firstProblem.b.apiType;
    const hipDataType          TiC             = firstProblem.c.apiType;
    const hipDataType          To              = firstProblem.d.apiType;
    const hipDataType          Tc              = dataTypes.computeScalar;
    const hipDataType          TciA            = dataTypes.computeInputA;
    const hipDataType          TciB            = dataTypes.computeInputB;
    const hipDataType          Talpha          = dataTypes.coefficient;
    const hipDataType          Tbias           = dataTypes.biasStorage;
    const hipDataType          Taux            = dataTypes.auxiliary;
    const hipblasOperation_t   transA          = firstProblem.operationA;
    const hipblasOperation_t   transB          = firstProblem.operationB;
    const hipblasLtBatchMode_t batchMode       = firstProblem.batchMode;
    const bool                 do_grouped_gemm = arg.grouped_gemm > 0;
    const int32_t              problem_count   = static_cast<int32_t>(matmulProblems.size());
    const bool do_swizzle_a     = arg.swizzle_a && hipblaslt::client::supportsMatmulSwizzle(TiA);
    const bool do_swizzle_b     = arg.swizzle_b && hipblaslt::client::supportsMatmulSwizzle(TiB);

    hipDeviceProp_t mxProperties{};
    if(isBlockScaling(arg.scaleA) || isBlockScaling(arg.scaleB))
        CHECK_HIP_ERROR(hipGetDeviceProperties(&mxProperties, 0));
    const auto mxScaleLayout = [&](hipblaslt_scaling_format format) {
        return isBlockScaling(format)
                   ? hipblaslt::hostnumerics::mxScaleStorageLayoutForFormat(
                         format, mxProperties.gcnArchName)
                   : roc::hostnumerics::amd_gpu_layout::MxScaleStorageLayout::Natural;
    };

    if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        if(arg.scaleA != hipblaslt_scaling_format::none
           && arg.scaleA != hipblaslt_scaling_format::Scalar)
        {
            hipblaslt_cout << "Only Tensorwide scaling is supported for General Batched GEMM"
                           << std::endl;
            return;
        }
        if(arg.scaleB != hipblaslt_scaling_format::none
           && arg.scaleB != hipblaslt_scaling_format::Scalar)
        {
            hipblaslt_cout << "Only Tensorwide scaling is supported for General Batched GEMM"
                           << std::endl;
            return;
        }
    }

    auto  preparation          = hipblaslt::client::prepareMatmulProblems(arg,
                                                                matmulProblems,
                                                                dataTypes,
                                                                do_swizzle_a,
                                                                do_swizzle_b,
                                                                mxScaleLayout(arg.scaleA),
                                                                mxScaleLayout(arg.scaleB));
    auto& preparedProblems     = preparation.problems;
    auto& firstPreparedProblem = preparedProblems.front();
    for(const auto& preparedProblem : preparedProblems)
    {
        if(preparedProblem.a.replacedUnsupportedBatchStride)
            hipblaslt_cerr << "Warning: swizzle_a does not yet support arbitrary stride_a!"
                           << std::endl;
        if(preparedProblem.b.replacedUnsupportedBatchStride)
            hipblaslt_cerr << "Warning: swizzle_b does not yet support arbitrary stride_b!"
                           << std::endl;
    }

    double gpu_time_used, cpu_time_used, gpu_mem_gbytes;
    gpu_time_used = cpu_time_used = gpu_mem_gbytes = 0.0;
    bool                   HMM                     = arg.HMM;
    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipEvent_t event_gpu_time_start, event_gpu_time_end;
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_end));

    int64_t rotating = arg.rotating * 1024 * 1024;

    struct MatmulRuntimeProblem
    {
        explicit MatmulRuntimeProblem(const hipblaslt::client::MatmulProblem& problem)
            : matrixA(problem.a.rows(),
                      problem.a.columns(),
                      problem.a.leadingDimension(),
                      problem.a.apiType)
            , matrixB(problem.b.rows(),
                      problem.b.columns(),
                      problem.b.leadingDimension(),
                      problem.b.apiType)
            , matrixC(problem.c.rows(),
                      problem.c.columns(),
                      problem.c.leadingDimension(),
                      problem.c.apiType)
            , matrixD(problem.d.rows(),
                      problem.d.columns(),
                      problem.d.leadingDimension(),
                      problem.d.apiType)
        {
        }

        void*                                     alphaPointer = nullptr;
        hipblaslt_local_matrix_layout             matrixA;
        hipblaslt_local_matrix_layout             matrixB;
        hipblaslt_local_matrix_layout             matrixC;
        hipblaslt_local_matrix_layout             matrixD;
        std::vector<hipblaslt_local_matmul_descr> matmulDescriptors;
    };
    std::vector<MatmulRuntimeProblem> runtimeProblems;
    runtimeProblems.reserve(matmulProblems.size());
    for(const auto& problem : matmulProblems)
        runtimeProblems.emplace_back(problem);
    const auto& firstRuntimeProblem = runtimeProblems.front();

    std::vector<HipDeviceBuffer> dA, dB, dC, dD, dE, dBias;
    auto&                        dOutput = firstProblem.cEqualsD ? dC : dD;
    std::vector<HipDeviceBuffer> dScaleAlphaVec, dScaleA, dScaleB, dScaleC, dScaleD, dScaleE,
        dAmaxD;

    std::vector<HipHostBuffer> hE, hE_gold, hBias, hBias_gold;
    std::vector<HipHostBuffer> hA, hB, hC, hD_gold, hD_1;
    // Contiguous host copies of the A, B and C regions, without padding, for fast_check, and the
    // expected probe sums, which depend only on the inputs and are shared by every solution.
    std::vector<std::shared_ptr<char[]>> fcA(problem_count), fcB(problem_count), fcC(problem_count);
    std::vector<hipDataType>             fcTypeA(problem_count), fcTypeB(problem_count);
    std::vector<FastCheckExpected>       fcExpected(problem_count);
    std::vector<HipHostBuffer> hScaleAlphaVec, hScaleA, hScaleB, hScaleC, hScaleD, hScaleE,
        hAmaxD_gold, hAmaxD, hD_gold_epl, hD_gold_ScaleAlpha, hBias_gold_epl;

    // Keep the decoded MX values as Tensors, one per problem and batch. This
    // preserves their compact logical layout without reconstructing it from
    // product allocation strides or converting element offsets to bytes.
    std::vector<std::vector<roc::hostnumerics::Tensor>> refA, refB;

    gpu_mem_gbytes = static_cast<double>(preparation.rotatingBytes) / (1024 * 1024 * 1024);

    // fast_check alone needs compact host inputs and no full host reference.
    const bool fast_check_only
        = arg.fast_check && !(arg.unit_check || arg.norm_check || arg.allclose_check);
    const bool compactHostInputs = fast_check_only && !arg.dump_matrix;
    const auto hostInputElements = [&](const hipblaslt::client::MatmulMatrix& matrix) {
        return compactHostInputs ? matrix.layout.shape().elementCount() : matrix.allocationElements;
    };
    const auto hostInputLayout = [&](const hipblaslt::client::MatmulMatrix& matrix) {
        return compactHostInputs ? roc::hostnumerics::Layout::contiguousFirstDimensionFastest(
                                       matrix.layout.shape())
                                 : matrix.layout;
    };
    if(arg.fast_check)
    {
        std::string why = fast_check_unsupported_reason(
            arg, batchMode, do_swizzle_a || do_swizzle_b, TiA, TiB, TiC, To, Tc);
        for(int i = 0; i < problem_count && why.empty(); i++)
        {
            if(isBlockScaling(arg.scaleA) && (matmulProblems[i].a.leadingDimension() != matmulProblems[i].a.rows() || matmulProblems[i].b.leadingDimension() != matmulProblems[i].b.rows()))
                why = "fast_check with MX scales requires lda and ldb equal to the rows stored";
            else if(matmulProblems[i].a.leadingDimension() < matmulProblems[i].a.rows() || matmulProblems[i].b.leadingDimension() < matmulProblems[i].b.rows() || matmulProblems[i].c.leadingDimension() < matmulProblems[i].m || matmulProblems[i].d.leadingDimension() < matmulProblems[i].m)
                why = "fast_check requires each leading dimension to be at least the number of "
                      "rows stored";
            else if(matmulProblems[i].batchCount > 1
                    && (matmulProblems[i].a.batchStride() < matmulProblems[i].a.leadingDimension() * matmulProblems[i].a.columns() || matmulProblems[i].b.batchStride() < matmulProblems[i].b.leadingDimension() * matmulProblems[i].b.columns()
                        || matmulProblems[i].c.batchStride() < matmulProblems[i].c.leadingDimension() * matmulProblems[i].n || matmulProblems[i].d.batchStride() < matmulProblems[i].d.leadingDimension() * matmulProblems[i].n))
                why = "fast_check requires batch strides that do not overlap";
        }
        if(!why.empty())
        {
#ifdef GOOGLE_TEST
            FAIL() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
    }

    if(arg.placement[0])
    {
        static const char* operands[]
            = {"a", "b", "c", "d", "bias", "scale_alpha_vec", "scale_a", "scale_b", "workspace"};
        std::string why;
        if(!arg.fast_check)
            why = "placement requires fast_check";
        else if(std::none_of(std::begin(operands), std::end(operands), [&](const char* o) {
                    return !strcmp(o, arg.placement);
                }))
            why = std::string("unknown placement operand '") + arg.placement + "'";
        else if(HMM)
            why = "placement does not support HMM";
        else if(arg.c_equal_d && !strcmp(arg.placement, "d"))
            why = "with c_equal_d, D is C: place c instead of d";
        else if(!strcmp(arg.placement, "bias") && !arg.bias_vector)
            why = "placing the bias requires bias_vector";
        else if(!strcmp(arg.placement, "scale_alpha_vec") && !arg.scaleAlpha_vector)
            why = "placing the scaleAlpha vector requires scaleAlpha_vector";
        else if((!strcmp(arg.placement, "scale_a") && !isBlockScaling(arg.scaleA))
                || (!strcmp(arg.placement, "scale_b") && !isBlockScaling(arg.scaleB)))
            why = "placing scale_a or scale_b requires MX block scales";
        if(!why.empty())
        {
#ifdef GOOGLE_TEST
            FAIL() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
    }

    if(arg.fast_check_repeat != 1 || arg.fast_check_inject != -1)
    {
        std::string why;
        if(!arg.fast_check)
            why = "fast_check_repeat and fast_check_inject require fast_check";
        else if(arg.fast_check_repeat < 1)
            why = "fast_check_repeat must be at least 1";
        else if(arg.fast_check_inject < -1)
            why = "fast_check_inject must be -1 (no injection) or an iteration number";
        else if(arg.fast_check_inject >= arg.fast_check_repeat)
            why = "fast_check_inject must name an iteration below fast_check_repeat";
        if(!why.empty())
        {
#ifdef GOOGLE_TEST
            FAIL() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
    }

    hipblaslt::hostnumerics::IntegerExactPattern iePattern
        = hipblaslt::hostnumerics::IntegerExactPattern::Standard;
    if(arg.integer_exact_pattern[0])
    {
        std::string why;
        if(arg.initialization != hipblaslt_initialization::integer_exact)
            why = "integer_exact_pattern requires initialization: integer_exact";
        else if(!hipblaslt::hostnumerics::parseIntegerExactPattern(arg.integer_exact_pattern,
                                                                   iePattern))
            why = std::string("unknown integer_exact_pattern '") + arg.integer_exact_pattern
                  + "'; use ternary or sparse_k";
        if(!why.empty())
        {
#ifdef GOOGLE_TEST
            FAIL() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
    }

    if(arg.requires_streamk)
    {
        std::string why = streamk_unavailable_reason();
        if(!why.empty())
        {
#ifdef GOOGLE_TEST
            GTEST_SKIP() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
    }

    // Calculating block count
    auto    plan        = hipblaslt_bench::compute_rotating_buffer_plan(arg.adaptive,
                                                              arg.max_iters,
                                                              arg.cold_iters,
                                                              arg.iters,
                                                              rotating,
                                                              preparation.rotatingBytes);
    int32_t block_count = plan.block_count;
    if(block_count <= 0)
        throw std::logic_error("Rotating-buffer planning produced no storage blocks.");
    const auto rotatingBlockAddress = [block_count](HipDeviceBuffer& buffer, int32_t block) {
        if(block < 0 || block >= block_count)
            throw std::out_of_range("Rotating-buffer block index is out of range.");
        const size_t blockCount = static_cast<size_t>(block_count);
        if(buffer.getNumBytes() % blockCount != 0)
            throw std::logic_error("Rotating-buffer allocation cannot be divided into blocks.");
        return static_cast<void*>(buffer.as<std::byte>()
                                  + static_cast<size_t>(block) * (buffer.getNumBytes() / blockCount));
    };
    // Scatter compact generated inputs without allocating their device padding on the host.
    // Copying only the logical region also preserves C's poison when C and D share storage.
    const auto copyHostInputToDevice = [&](HipDeviceBuffer&                       device,
                                           const HipHostBuffer&                   host,
                                           const hipblaslt::client::MatmulMatrix& matrix) {
        if(!compactHostInputs)
            return synchronize(device, host, block_count, stream);
        const size_t elementBytes = fast_check_element_size(matrix.apiType);
        const size_t rows         = static_cast<size_t>(matrix.rows());
        const size_t columns      = static_cast<size_t>(matrix.columns());
        if(rows == 0 || columns == 0)
            return hipSuccess;
        for(int32_t block = 0; block < block_count; ++block)
            for(size_t batch = 0; batch < matrix.layout.shape()[2]; ++batch)
            {
                const size_t offset = static_cast<size_t>(block) * matrix.allocationElements
                                      + batch * static_cast<size_t>(matrix.batchStride());
                const auto error = hipMemcpy2DAsync(
                    device.as<char>() + offset * elementBytes,
                    static_cast<size_t>(matrix.leadingDimension()) * elementBytes,
                    host.as<char>() + batch * rows * columns * elementBytes,
                    rows * elementBytes,
                    rows * elementBytes,
                    columns,
                    device.use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice,
                    stream);
                if(error != hipSuccess)
                    return error;
            }
        return hipStreamSynchronize(stream);
    };
    if(rotating > 0)
    {
        hipblaslt_cout << "Rotating buffer " << rotating / (1024 * 1024) << " MiB. "
                       << "Needed Size: " << preparation.rotatingBytes / (1024 * 1024) << " MiB. "
                       << "Needed block count: " << block_count;
        if(plan.capped)
            hipblaslt_cout << " (Capped to max iters: " << plan.iter_cap << ")";
        hipblaslt_cout << std::endl;
    }

    // fast_check runs the large-shape cases, which can need more memory than a runner has. Skip
    // with the amounts rather than fail an allocation or, for host memory, crash on a null buffer.
    // The estimate follows the allocations below, block_count times for the rotating buffers; the
    // workspace is at most user_allocated_workspace.
    if(arg.fast_check)
    {
        const size_t sizeTo = realDataTypeSize(To), sizeC = realDataTypeSize(TiC),
                     sizeAlpha = realDataTypeSize(Talpha);
        size_t       hostBytes = 0, deviceBytes = 0;
        for(int i = 0; i < problem_count; i++)
        {
            deviceBytes
                += (matmulProblems[i].a.allocationElements * realDataTypeSize(TiA)
                    + matmulProblems[i].b.allocationElements * realDataTypeSize(TiB)
                    + matmulProblems[i].c.allocationElements * sizeC
                    + (arg.c_equal_d ? 0 : matmulProblems[i].d.allocationElements) * sizeTo
                    + (matmulProblems[i].auxiliary ? matmulProblems[i].auxiliary->allocationElements
                                                   : 0)
                          * realDataTypeSize(Taux)
                    + preparedProblems[i].biasElements * realDataTypeSize(Tbias)
                    + (preparedProblems[i].scaleAlphaElements
                       + (preparedProblems[i].a.scaleElements + preparedProblems[i].b.scaleElements)
                             * matmulProblems[i].batchCount)
                          * sizeAlpha)
                   * size_t(block_count);
            // MX keeps both a float reference and a float copy for fast_check.
            // Other inputs keep only the compact copy in their original type.
            const bool mxA = isBlockScaling(arg.scaleA), mxB = isBlockScaling(arg.scaleB);
            hostBytes += size_t(matmulProblems[i].a.rows() * matmulProblems[i].a.columns()
                                * matmulProblems[i].batchCount)
                             * (mxA ? 2 * sizeof(float) : realDataTypeSize(TiA))
                         + size_t(matmulProblems[i].b.rows() * matmulProblems[i].b.columns()
                                  * matmulProblems[i].batchCount)
                               * (mxB ? 2 * sizeof(float) : realDataTypeSize(TiB))
                         + size_t(matmulProblems[i].m * matmulProblems[i].n
                                  * matmulProblems[i].batchCount)
                               * sizeC;
            // Generated host inputs remain compact when no full reference or dump is needed.
            hostBytes += hostInputElements(matmulProblems[i].a) * realDataTypeSize(TiA);
            hostBytes += hostInputElements(matmulProblems[i].b) * realDataTypeSize(TiB);
            hostBytes += hostInputElements(matmulProblems[i].c) * sizeC;
            hostBytes += preparedProblems[i].outputCopyElements * (2 * sizeTo + 3 * sizeAlpha)
                         + 2 * preparedProblems[i].biasElements * realDataTypeSize(Tbias)
                         + (matmulProblems[i].auxiliary ? matmulProblems[i].auxiliary->allocationElements : 0) * realDataTypeSize(Taux) * (arg.use_e && !arg.gradient ? 2 : 1)
                         + (preparedProblems[i].scaleAlphaElements
                            + (preparedProblems[i].a.scaleElements + preparedProblems[i].b.scaleElements) * matmulProblems[i].batchCount)
                               * sizeAlpha;
        }
        // scaleC, scaleD and scaleE are one value each, and amaxD two on the host (result and
        // reference); the device count reuses the host's, which is at least as large.
        const size_t sideScalars
            = size_t(arg.scaleC) + size_t(arg.scaleD) + size_t(arg.scaleE) + 2 * size_t(arg.amaxD);
        deviceBytes += size_t(arg.user_allocated_workspace) * size_t(block_count)
                       + size_t(problem_count) * sideScalars * sizeAlpha;
        hostBytes += size_t(problem_count) * sideScalars * sizeAlpha;
        std::string why = fast_check_memory_shortfall(deviceBytes, hostBytes);
        if(!why.empty())
        {
#ifdef GOOGLE_TEST
            GTEST_SKIP() << why;
#else
            hipblaslt_cerr << why << std::endl;
            return;
#endif
        }
    }

    // Calculating block count end
    for(auto& runtimeProblem : runtimeProblems)
        runtimeProblem.matmulDescriptors.reserve(block_count);

    for(int i = 0; i < problem_count; i++)
    {
        const auto&   problem         = matmulProblems[i];
        auto&         preparedProblem = preparedProblems[i];
        auto&         runtimeProblem  = runtimeProblems[i];
        const int64_t batchStrideC    = problem.c.batchStride();
        const int64_t batchStrideD    = problem.d.batchStride();

        CHECK_HIPBLASLT_ERROR(runtimeProblem.matrixA.status());
        CHECK_HIPBLASLT_ERROR(runtimeProblem.matrixB.status());
        CHECK_HIPBLASLT_ERROR(runtimeProblem.matrixC.status());
        CHECK_HIPBLASLT_ERROR(runtimeProblem.matrixD.status());

        if(do_swizzle_a)
        {
            hipblasLtOrder_t orderA = hipblaslt::client::matmulOrderForDataType(TiA);
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                runtimeProblem.matrixA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
        }

        if(do_swizzle_b)
        {
            hipblasLtOrder_t orderB = hipblaslt::client::matmulOrderForDataType(TiB);
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                runtimeProblem.matrixB, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderB, sizeof(orderB)));
        }

        if((problem.batchCount > 1) || batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixA,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &(problem.batchCount),
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixB,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &(problem.batchCount),
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixC,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &(problem.batchCount),
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixD,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &(problem.batchCount),
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixA,
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(preparedProblem.a.batchStride),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixB,
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(preparedProblem.b.batchStride),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixC,
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &batchStrideC,
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixD,
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &batchStrideD,
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixA,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE,
                                                  &batchMode,
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixB,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE,
                                                  &batchMode,
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixC,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE,
                                                  &batchMode,
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(runtimeProblem.matrixD,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE,
                                                  &batchMode,
                                                  sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
        }

        runtimeProblem.matmulDescriptors.emplace_back(arg.compute_type, arg.scale_type);
        auto& primaryMatmul = runtimeProblem.matmulDescriptors.front();
        CHECK_HIPBLASLT_ERROR(primaryMatmul.status());

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                            &TciA,
                                            sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                            &TciB,
                                            sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            primaryMatmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            primaryMatmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));

        // Forward CLI knobs from hipblaslt-bench into the matmul descriptor.
        {
            int32_t sm = hipblaslt_bench_options::sm_count_target();
            if(sm != 0)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul, HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm, sizeof(sm)));
            }
            int32_t dyn = hipblaslt_bench_options::streamk_tile_scheduling_mode();
            if(dyn >= 0)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul,
                    HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                    &dyn,
                    sizeof(dyn)));
            }
            int32_t uso = hipblaslt_bench_options::uniform_summation_order();
            if(uso >= 0)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul,
                    HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT,
                    &uso,
                    sizeof(uso)));
            }
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            // allocate memory on device; the operand named by arg.placement crosses a 4 GiB
            // boundary. allocate() returns false when that placement failed.
            std::string placement_why;
            bool        placement_unsupported = false;
            auto        allocate              = [&](std::vector<HipDeviceBuffer>& v,
                                hipDataType                   type,
                                size_t                        elements,
                                const char*                   operand) {
                if(i == 0 && !strcmp(arg.placement, operand))
                {
                    v.emplace_back(type,
                                   elements,
                                   size_t(arg.placement_offset),
                                   &placement_why,
                                   &placement_unsupported);
                    return v.back().buf() != nullptr;
                }
                v.emplace_back(type, elements, HMM);
                return true;
            };
            CHECK_PLACEMENT(allocate(dA, TiA, preparedProblems[i].a.elements * block_count, "a"),
                            placement_unsupported,
                            placement_why);
            CHECK_DEVICE_ALLOCATION(hipGetLastError());
            CHECK_PLACEMENT(allocate(dB, TiB, preparedProblems[i].b.elements * block_count, "b"),
                            placement_unsupported,
                            placement_why);
            CHECK_DEVICE_ALLOCATION(hipGetLastError());
            CHECK_PLACEMENT(allocate(dC, TiC, matmulProblems[i].c.allocationElements * block_count, "c"),
                            placement_unsupported,
                            placement_why);
            CHECK_DEVICE_ALLOCATION(hipGetLastError());

            if(!firstProblem.cEqualsD)
            {
                CHECK_PLACEMENT(allocate(dD, To, matmulProblems[i].d.allocationElements * block_count, "d"),
                                placement_unsupported,
                                placement_why);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(preparedProblem.biasElements * block_count != 0)
            {
                CHECK_PLACEMENT(allocate(dBias, Tbias, preparedProblems[i].biasElements * block_count, "bias"),
                                placement_unsupported,
                                placement_why);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(arg.scaleAlpha_vector)
            {
                CHECK_PLACEMENT(allocate(dScaleAlphaVec,
                                         Talpha,
                                         preparedProblems[i].scaleAlphaElements * block_count,
                                         "scale_alpha_vec"),
                                placement_unsupported,
                                placement_why);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(arg.use_e)
            {
                dE.emplace_back(Taux, problem.auxiliaryAllocationElements() * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                dScaleA.emplace_back(Talpha, preparedProblem.a.scaleElements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else if(isBlockScaling(arg.scaleA))
            {
                // For MX format, use uint8_t for the scale (E8M0), allocate for all batches. A
                // placed scale buffer is typed int8 so its poison windows hold 0x28, which E8M0
                // reads as 2^-87: a misdirected scale read collapses its block toward zero.
                if(!strcmp(arg.placement, "scale_a"))
                    CHECK_PLACEMENT(allocate(dScaleA,
                                             HIP_R_8I,
                                             preparedProblems[i].a.scaleElements * matmulProblems[i].batchCount * block_count,
                                             "scale_a"),
                                    placement_unsupported,
                                    placement_why);
                else
                    dScaleA.emplace_back(
                        HIP_R_8U, preparedProblems[i].a.scaleElements * matmulProblems[i].batchCount * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                dScaleB.emplace_back(Talpha, preparedProblem.b.scaleElements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else if(isBlockScaling(arg.scaleB))
            {
                // For MX format, use uint8_t for the scale (E8M0), allocate for all batches
                if(!strcmp(arg.placement, "scale_b"))
                    CHECK_PLACEMENT(allocate(dScaleB,
                                             HIP_R_8I,
                                             preparedProblems[i].b.scaleElements * matmulProblems[i].batchCount * block_count,
                                             "scale_b"),
                                    placement_unsupported,
                                    placement_why);
                else
                    dScaleB.emplace_back(
                        HIP_R_8U, preparedProblems[i].b.scaleElements * matmulProblems[i].batchCount * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleC)
            {
                dScaleC.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleD)
            {
                dScaleD.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.amaxD)
            {
                dAmaxD.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleE)
            {
                dScaleE.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
            hA.emplace_back(TiA, hostInputElements(problem.a));
            hB.emplace_back(TiB, hostInputElements(problem.b));
            hC.emplace_back(TiC, hostInputElements(problem.c));
            hD_gold.emplace_back(To, preparedProblems[i].outputCopyElements);
            hD_1.emplace_back(To, preparedProblems[i].outputCopyElements);
            if(preparedProblems[i].biasElements * block_count != 0)
            {
                hBias.emplace_back(Tbias, preparedProblem.biasElements);
                hBias_gold.emplace_back(Tbias, preparedProblem.biasElements);
            }

            hD_gold_epl.emplace_back(Talpha, preparedProblem.outputCopyElements);
            hD_gold_ScaleAlpha.emplace_back(Talpha, preparedProblem.outputCopyElements);
            hBias_gold_epl.emplace_back(
                Talpha, preparedProblem.outputCopyElements); // Reduction for matrix D

            if(arg.scaleAlpha_vector)
                hScaleAlphaVec.emplace_back(Talpha, preparedProblem.scaleAlphaElements);

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                hScaleA.emplace_back(Talpha, preparedProblem.a.scaleElements);
            }
            else if(isBlockScaling(arg.scaleA))
            {
                hScaleA.emplace_back(HIP_R_8U,
                                     preparedProblem.a.scaleElements * problem.batchCount);
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                hScaleB.emplace_back(Talpha, preparedProblem.b.scaleElements);
            }
            else if(isBlockScaling(arg.scaleB))
            {
                hScaleB.emplace_back(HIP_R_8U,
                                     preparedProblem.b.scaleElements * problem.batchCount);
            }
            if(arg.scaleC)
                hScaleC.emplace_back(Talpha, 1);
            if(arg.scaleD)
                hScaleD.emplace_back(Talpha, 1);
            if(arg.amaxD)
            {
                hAmaxD_gold.emplace_back(Talpha, 1);
                hAmaxD.emplace_back(Talpha, 1);
            }
            if(arg.scaleE)
                hScaleE.emplace_back(Talpha, 1);

            if(arg.use_e)
            {
                hE.emplace_back(Taux, problem.auxiliaryAllocationElements());
                if(!arg.gradient)
                {
                    hE_gold.emplace_back(Taux, problem.auxiliaryAllocationElements());
                }
            }
        }
        else
        {
            for(int batchCount = 0; batchCount < arg.batch_count; batchCount++)
            {
                // allocate memory on device
                dA.emplace_back(TiA, preparedProblem.a.elements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
                dB.emplace_back(TiB, preparedProblem.b.elements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
                dC.emplace_back(TiC, problem.c.allocationElements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());

                if(!firstProblem.cEqualsD)
                {
                    dD.emplace_back(To, problem.d.allocationElements * block_count, HMM);
                    CHECK_DEVICE_ALLOCATION(hipGetLastError());
                }

                if(preparedProblem.biasElements * block_count != 0)
                {
                    dBias.emplace_back(Tbias, preparedProblem.biasElements * block_count, HMM);
                    CHECK_DEVICE_ALLOCATION(hipGetLastError());
                }

                if(arg.scaleAlpha_vector)
                {
                    hipblaslt_cout << "General Batched GEMM does not support scaleAlpha_vector."
                                   << std::endl;
                    return;
                }

                if(arg.use_e)
                {
                    hipblaslt_cout << "General Batched GEMM does not support use_e." << std::endl;
                    return;
                }

                // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
                hA.emplace_back(TiA, problem.a.allocationElements);
                hB.emplace_back(TiB, problem.b.allocationElements);
                hC.emplace_back(TiC, problem.c.allocationElements);
                hD_gold.emplace_back(To, preparedProblem.outputCopyElements);
                hD_1.emplace_back(To, preparedProblem.outputCopyElements);
                if(preparedProblem.biasElements * block_count != 0)
                {
                    hBias.emplace_back(Tbias, preparedProblem.biasElements);
                    hBias_gold.emplace_back(Tbias, preparedProblem.biasElements);
                }
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::none)
            {
                dScaleA.emplace_back(Talpha, preparedProblem.a.scaleElements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::none)
            {
                dScaleB.emplace_back(Talpha, preparedProblem.b.scaleElements * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleC)
            {
                dScaleC.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleD)
            {
                dScaleD.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.amaxD)
            {
                hipblaslt_cout << "General Batched GEMM doesn't support Epilogues."
                               << "Only Scaling and Quantization is supported for post processing"
                               << std::endl;
                return;
            }
            if(arg.scaleE)
            {
                hipblaslt_cout << "General Batched GEMM doesn't support Epilogues."
                               << "Only Scaling and Quantization is supported for post processing"
                               << std::endl;
                return;
            }
            if(arg.scaleAlpha_vector)
            {
                hipblaslt_cout << "General Batched GEMM does not support scaleAlpha_vector."
                               << std::endl;
                return;
            }

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::none)
            {
                hScaleA.emplace_back(Talpha, preparedProblem.a.scaleElements);
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::none)
            {
                hScaleB.emplace_back(Talpha, preparedProblem.b.scaleElements);
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleC)
                hScaleC.emplace_back(Talpha, 1);
            if(arg.scaleD)
                hScaleD.emplace_back(Talpha, 1);
        }

        const bool positiveOnlyInitialization
            = arg.ulp_check
              && (arg.initialization == hipblaslt_initialization::hpl
                  || arg.initialization == hipblaslt_initialization::trig_float);

        const uint64_t initializationBaseSeed
            = arg.initialization == hipblaslt_initialization::norm_dist_one_special
                  ? hipblaslt::hostnumerics::oneSpecialInitializationSeed
                  : hipblaslt::hostnumerics::defaultInitializationSeed;
        const auto initializationSeed
            = [&](hipblaslt::hostnumerics::initialization::OperandSequence sequence,
                  size_t                                                    batch = 0,
                  uint64_t baseSeed = hipblaslt::hostnumerics::defaultInitializationSeed) {
                  const uint64_t sequencesPerBatch = static_cast<uint64_t>(
                      hipblaslt::hostnumerics::initialization::OperandSequence::Count);
                  const uint64_t batchesPerProblem
                      = static_cast<uint64_t>(std::max(problem.batchCount, 1));
                  const uint64_t sequenceIndex
                      = (static_cast<uint64_t>(i) * batchesPerProblem + batch) * sequencesPerBatch
                        + static_cast<uint64_t>(sequence);
                  return hipblaslt::hostnumerics::initialization::seedForSequence(baseSeed,
                                                                                   sequenceIndex);
              };
        const auto matrixSeed = [&](hipblaslt::hostnumerics::MatrixRole role, size_t batch = 0) {
            const auto sequence = [&] {
                switch(role)
                {
                case hipblaslt::hostnumerics::MatrixRole::A:
                    return hipblaslt::hostnumerics::initialization::OperandSequence::MatrixA;
                case hipblaslt::hostnumerics::MatrixRole::B:
                    return hipblaslt::hostnumerics::initialization::OperandSequence::MatrixB;
                case hipblaslt::hostnumerics::MatrixRole::C:
                    return hipblaslt::hostnumerics::initialization::OperandSequence::MatrixC;
                case hipblaslt::hostnumerics::MatrixRole::Count:
                    break;
                }
                throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
            }();
            return initializationSeed(sequence, batch, initializationBaseSeed);
        };

        const auto initializeHostTensor = [&](HipHostBuffer&                              buffer,
                                              hipDataType                                 type,
                                              const roc::hostnumerics::Layout&           layout,
                                              const roc::hostnumerics::GenerationRecipe& recipe) {
            auto tensor = buffer.tensor(hipblaslt::hostnumerics::scalarType(type), layout);
            std::ranges::fill(tensor.rawEncodedBackingStorage(), std::byte{0});
            roc::hostnumerics::generate(tensor, recipe);
        };
        const auto randomInitializationRecipe
            = [&](hipDataType                                               type,
                  hipblaslt::hostnumerics::initialization::OperandSequence sequence,
                  bool                                                      small = false) {
                  const auto scalar = hipblaslt::hostnumerics::scalarType(type);
                  const auto seed   = initializationSeed(sequence);
                  if(small)
                  {
                      return roc::hostnumerics::GenerationRecipe::realOnly(
                          roc::hostnumerics::GenerationRecipe::uniformInteger(
                              {.lower = 1, .upper = 10})
                              .withAffineValueMapping({.scale = 0.1}),
                          {.seed = seed});
                  }
                  return hipblaslt::hostnumerics::initializationRecipe(
                      scalar,
                      hipblaslt_initialization::rand_int,
                      seed,
                      hipblaslt::hostnumerics::TrigonometricComponent::Cosine);
              };

        const auto initializeHostMatrix = [&](HipHostBuffer&                         buffer,
                                              const hipblaslt::client::MatmulMatrix& matrix,
                                              hipblaslt::hostnumerics::MatrixRole   role,
                                              bool                                   forceNaN,
                                              bool   separateBatchStorage,
                                              size_t batch) {
            const size_t                     rows    = static_cast<size_t>(matrix.rows());
            const size_t                     columns = static_cast<size_t>(matrix.columns());
            const roc::hostnumerics::Layout  layout
                = separateBatchStorage ? roc::hostnumerics::Layout(
                                             roc::hostnumerics::Shape{rows, columns},
                                             {matrix.layout.stride(0), matrix.layout.stride(1)})
                                       : hostInputLayout(matrix);
            auto tensor
                = buffer.tensor(hipblaslt::hostnumerics::scalarType(matrix.apiType), layout);
            std::ranges::fill(tensor.rawEncodedBackingStorage(), std::byte{0});
            hipblaslt::hostnumerics::initializeMatrix(
                tensor,
                role,
                arg.initialization,
                matrixSeed(role, batch),
                forceNaN,
                std::nullopt,
                positiveOnlyInitialization,
                {.pattern = iePattern, .reductionAxis = transA != HIPBLAS_OP_N ? 0u : 1u});
        };

        auto mxScaleBatchOutput = [](HipHostBuffer& buffer, size_t offset, size_t batchBytes) {
            if(offset > buffer.getNumBytes())
                throw std::invalid_argument("MX output offset exceeds host buffer capacity.");
            size_t capacity = buffer.getNumBytes() - offset;
            if(batchBytes != 0)
                capacity = std::min(capacity, batchBytes);
            return std::span<uint8_t>(buffer.as<uint8_t>() + offset, capacity);
        };
        auto generateMxBatch
            = [&](hipDataType                                                  dataType,
                  hipDataType                                                  scaleType,
                  const roc::hostnumerics::Tensor&                             dataOutput,
                  std::span<uint8_t>                                           scaleOutput,
                  uint64_t                                                     rows,
                  uint64_t                                                     columns,
                  uint64_t                                                     leadingDimension,
                  size_t                                                       blockRows,
                  size_t                                                       blockColumns,
                  const roc::hostnumerics::amd_gpu_layout::MxScaleStoragePlan& scalePlan,
                  uint64_t                                                     seed,
                  hipblaslt::hostnumerics::MatrixRole                          role) {
                  if(blockRows == 0 || blockColumns == 0
                     || blockColumns > std::numeric_limits<size_t>::max() / blockRows)
                      throw std::invalid_argument("Invalid hipBLASLt MX scale block dimensions.");
                  auto generated = hipblaslt::hostnumerics::generateMxData(
                      dataType,
                      scaleType,
                      roc::hostnumerics::Shape{static_cast<size_t>(rows),
                                               static_cast<size_t>(columns)},
                      leadingDimension,
                      blockColumns > 1 ? 1 : 0,
                      blockRows * blockColumns,
                      arg.initialization,
                      seed,
                      role);
                  dataOutput.copyLogicalElementsFrom(generated.data);

                  // gfx950 consumes [MN, K blocks] for either operand orientation.
                  const auto scalesForDevice
                      = scalePlan.layout
                                    == roc::hostnumerics::amd_gpu_layout::MxScaleStorageLayout::
                                        Gfx950
                                && blockColumns > 1
                            ? generated.scales.copyWithPermutedDimensions({1, 0})
                            : generated.scales;
                  const std::vector<std::byte> scaleStorage
                      = roc::hostnumerics::amd_gpu_layout::copyMxScaleStorageToPhysicalLayout(
                          scalesForDevice.rawEncodedBackingStorage().data(),
                          scalesForDevice.rawEncodedBackingStorage().size(),
                          scalePlan);
                  if(scaleOutput.size() < scaleStorage.size())
                      throw std::invalid_argument("hipBLASLt MX scale output is too small.");
                  if(!scaleStorage.empty())
                      std::memcpy(scaleOutput.data(), scaleStorage.data(), scaleStorage.size());

                  return generated.reference;
              };
        size_t scaleA_row = ((transA == HIPBLAS_OP_T) ? blockSize(arg.scaleA) : 1);
        size_t scaleA_col = ((transA == HIPBLAS_OP_T) ? 1 : blockSize(arg.scaleA));
        if(isBlockScaling(arg.scaleA))
        {
            if(arg.initialization != hipblaslt_initialization::hpl
               && arg.initialization != hipblaslt_initialization::trig_float
               && arg.initialization != hipblaslt_initialization::uniform_01
               && arg.initialization != hipblaslt_initialization::zero
               && arg.initialization != hipblaslt_initialization::norm_dist
               && arg.initialization != hipblaslt_initialization::rand_int
               && arg.initialization != hipblaslt_initialization::uniform_low_precision
               && !(arg.initialization == hipblaslt_initialization::integer_exact
                    && arg.fast_check))
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "unsupported MX initialization: "
                             << hipblaslt_initialization2string(arg.initialization);
#else
                hipblaslt_cout
                    << "Initialization of microscaling data only allows hpl, trig_float, "
                       "uniform_01, zero, norm_dist, rand_int or uniform_low_precision, not "
                    << hipblaslt_initialization2string(arg.initialization) << std::endl;
                return;
#endif
            }
            if(arg.algo_method == 1)
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "MX data types do not support algorithm \"all\"";
#else
                hipblaslt_cout << "MX data types do not support algorithm \"all\"" << std::endl;
                return;
#endif
            }
            const auto& scalePlanA = *preparedProblem.a.mxScaleStorage;
            size_t scaleBatchBytesA
                = (problem.batchCount > 1) ? preparedProblem.a.scaleElements : 0;
            std::vector<roc::hostnumerics::Tensor> refAAll;
            refAAll.reserve(static_cast<size_t>(problem.batchCount));
            for(int64_t b = 0; b < problem.batchCount; b++)
            {
                const auto dataOutputA = hA[i].tensor(
                    hipblaslt::hostnumerics::scalarType(TiA),
                    problem.a.logicalBatchLayout(HIPBLAS_OP_N, static_cast<size_t>(b), false));
                auto scaleOutputA
                    = mxScaleBatchOutput(hScaleA[i], b * scaleBatchBytesA, scaleBatchBytesA);
                refAAll.emplace_back(generateMxBatch(
                    TiA,
                    scaleDataType(arg.scaleA),
                    dataOutputA,
                    scaleOutputA,
                    problem.a.rows(),
                    problem.a.columns(),
                    problem.a.leadingDimension(),
                    scaleA_row,
                    scaleA_col,
                    scalePlanA,
                    matrixSeed(hipblaslt::hostnumerics::MatrixRole::A,
                               problem.a.batchStride() == 0 ? 0 : static_cast<size_t>(b)),
                    hipblaslt::hostnumerics::MatrixRole::A));
            }
            refA.emplace_back(std::move(refAAll));
            CHECK_HIP_ERROR(copyHostInputToDevice(dA[i], hA[i], problem.a));
            CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
        }
        else
        {
            if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                initializeHostMatrix(hA[i],
                                     problem.a,
                                     hipblaslt::hostnumerics::MatrixRole::A,
                                     alpha_isnan_type(arg, Talpha),
                                     false,
                                     0);
            }
            else
            {
                for(int batchCount = 0; batchCount < problem.batchCount; batchCount++)
                {
                    initializeHostMatrix(hA[batchCount],
                                         problem.a,
                                         hipblaslt::hostnumerics::MatrixRole::A,
                                         alpha_isnan_type(arg, Talpha),
                                         true,
                                         static_cast<size_t>(batchCount));
                    if(!do_swizzle_a)
                        CHECK_HIP_ERROR(synchronize(dA[batchCount], hA[batchCount], block_count));
                }
            }
        }

        size_t scaleB_row = ((transB == HIPBLAS_OP_T) ? 1 : blockSize(arg.scaleB));
        size_t scaleB_col = ((transB == HIPBLAS_OP_T) ? blockSize(arg.scaleB) : 1);
        if(isBlockScaling(arg.scaleB))
        {
            // MX B always goes through HostNumerics (mirrors the A side above).
            if(arg.initialization != hipblaslt_initialization::hpl
               && arg.initialization != hipblaslt_initialization::trig_float
               && arg.initialization != hipblaslt_initialization::uniform_01
               && arg.initialization != hipblaslt_initialization::zero
               && arg.initialization != hipblaslt_initialization::norm_dist
               && arg.initialization != hipblaslt_initialization::rand_int
               && arg.initialization != hipblaslt_initialization::uniform_low_precision
               && !(arg.initialization == hipblaslt_initialization::integer_exact
                    && arg.fast_check))
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "unsupported MX initialization: "
                             << hipblaslt_initialization2string(arg.initialization);
#else
                hipblaslt_cout
                    << "Initialization of microscaling data only allows hpl, trig_float, "
                       "uniform_01, zero, norm_dist, rand_int or uniform_low_precision, not "
                    << hipblaslt_initialization2string(arg.initialization) << std::endl;
                return;
#endif
            }
            if(arg.algo_method == 1)
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "MX data types do not support algorithm \"all\"";
#else
                hipblaslt_cout << "MX data types do not support algorithm \"all\"" << std::endl;
                return;
#endif
            }
            const auto& scalePlanB = *preparedProblem.b.mxScaleStorage;
            size_t scaleBatchBytesB
                = (problem.batchCount > 1) ? preparedProblem.b.scaleElements : 0;
            std::vector<roc::hostnumerics::Tensor> refBAll;
            refBAll.reserve(static_cast<size_t>(problem.batchCount));
            for(int64_t b = 0; b < problem.batchCount; b++)
            {
                const auto dataOutputB = hB[i].tensor(
                    hipblaslt::hostnumerics::scalarType(TiB),
                    problem.b.logicalBatchLayout(HIPBLAS_OP_N, static_cast<size_t>(b), false));
                auto scaleOutputB
                    = mxScaleBatchOutput(hScaleB[i], b * scaleBatchBytesB, scaleBatchBytesB);
                refBAll.emplace_back(generateMxBatch(
                    TiB,
                    scaleDataType(arg.scaleB),
                    dataOutputB,
                    scaleOutputB,
                    problem.b.rows(),
                    problem.b.columns(),
                    problem.b.leadingDimension(),
                    scaleB_row,
                    scaleB_col,
                    scalePlanB,
                    matrixSeed(hipblaslt::hostnumerics::MatrixRole::B,
                               problem.b.batchStride() == 0 ? 0 : static_cast<size_t>(b)),
                    hipblaslt::hostnumerics::MatrixRole::B));
            }
            refB.emplace_back(std::move(refBAll));
            CHECK_HIP_ERROR(copyHostInputToDevice(dB[i], hB[i], problem.b));
            CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
        }
        else
        {
            if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                initializeHostMatrix(hB[i],
                                     problem.b,
                                     hipblaslt::hostnumerics::MatrixRole::B,
                                     alpha_isnan_type(arg, Talpha),
                                     false,
                                     0);
            }
            else
            {
                for(int batchCount = 0; batchCount < problem.batchCount; batchCount++)
                {
                    initializeHostMatrix(hB[batchCount],
                                         problem.b,
                                         hipblaslt::hostnumerics::MatrixRole::B,
                                         alpha_isnan_type(arg, Talpha),
                                         true,
                                         static_cast<size_t>(batchCount));
                    if(!do_swizzle_b)
                        CHECK_HIP_ERROR(synchronize(dB[batchCount], hB[batchCount], block_count));
                }
            }
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            initializeHostMatrix(hC[i],
                                 problem.c,
                                 hipblaslt::hostnumerics::MatrixRole::C,
                                 beta_isnan_type(arg, Talpha),
                                 false,
                                 0);

            if(!isBlockScaling(arg.scaleA) && !do_swizzle_a)
                CHECK_HIP_ERROR(copyHostInputToDevice(dA[i], hA[i], problem.a));
            if(!isBlockScaling(arg.scaleB) && !do_swizzle_b)
                CHECK_HIP_ERROR(copyHostInputToDevice(dB[i], hB[i], problem.b));
            CHECK_HIP_ERROR(copyHostInputToDevice(dC[i], hC[i], problem.c));

            if(arg.fast_check)
            {
                CHECK_HIP_ERROR(fast_check_poison_padding_device(
                    {dA[i].buf(), TiA, matmulProblems[i].a.rows(), matmulProblems[i].a.columns(), matmulProblems[i].a.leadingDimension(), matmulProblems[i].a.batchStride()},
                    matmulProblems[i].batchCount,
                    matmulProblems[i].a.allocationElements,
                    stream));
                CHECK_HIP_ERROR(fast_check_poison_padding_device(
                    {dB[i].buf(), TiB, matmulProblems[i].b.rows(), matmulProblems[i].b.columns(), matmulProblems[i].b.leadingDimension(), matmulProblems[i].b.batchStride()},
                    matmulProblems[i].batchCount,
                    matmulProblems[i].b.allocationElements,
                    stream));
                CHECK_HIP_ERROR(
                    fast_check_poison_padding_device({dC[i].buf(),
                                                      TiC,
                                                      matmulProblems[i].m,
                                                      matmulProblems[i].n,
                                                      matmulProblems[i].c.leadingDimension(),
                                                      matmulProblems[i].c.batchStride()},
                                                     matmulProblems[i].batchCount,
                                                     matmulProblems[i].c.allocationElements,
                                                     stream));
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));

                const auto copyMxReferences
                    = [](std::shared_ptr<char[]>&                      output,
                         const std::vector<roc::hostnumerics::Tensor>& references) {
                          using namespace roc::hostnumerics;
                          size_t totalBytes = 0;
                          for(const auto& reference : references)
                              totalBytes += reference.shape().elementCount() * sizeof(float);
                          output.reset(new char[totalBytes]);
                          size_t offset = 0;
                          for(const auto& reference : references)
                          {
                              const size_t bytes = reference.shape().elementCount() * sizeof(float);
                              auto         destination = Tensor::shareExternalMutableBackingStorage(
                                  ScalarType::Float32,
                                  Layout::contiguousFirstDimensionFastest(reference.shape()),
                                  std::shared_ptr<void>(output, output.get()),
                                  std::span<std::byte>(
                                      reinterpret_cast<std::byte*>(output.get() + offset), bytes));
                              destination.copyLogicalElementsFrom(reference);
                              offset += bytes;
                          }
                      };
                // MX references are logical Tensors, one per batch. Copy their values in the
                // stored column-major order expected by fast_check, including their block scales.
                if(isBlockScaling(arg.scaleA))
                {
                    copyMxReferences(fcA[i], refA[i]);
                    fcTypeA[i] = HIP_R_32F;
                }
                else
                {
                    fcA[i].reset(
                        new char[problem.a.layout.shape().elementCount() * realDataTypeSize(TiA)]);
                    CHECK_HIP_ERROR(fast_check_copy_region_to_host(fcA[i].get(),
                                                                   {dA[i].buf(),
                                                                    TiA,
                                                                    problem.a.rows(),
                                                                    problem.a.columns(),
                                                                    problem.a.leadingDimension(),
                                                                    problem.a.batchStride()},
                                                                   problem.batchCount,
                                                                   stream));
                    fcTypeA[i] = TiA;
                }
                if(isBlockScaling(arg.scaleB))
                {
                    copyMxReferences(fcB[i], refB[i]);
                    fcTypeB[i] = HIP_R_32F;
                }
                else
                {
                    fcB[i].reset(
                        new char[problem.b.layout.shape().elementCount() * realDataTypeSize(TiB)]);
                    CHECK_HIP_ERROR(fast_check_copy_region_to_host(fcB[i].get(),
                                                                   {dB[i].buf(),
                                                                    TiB,
                                                                    problem.b.rows(),
                                                                    problem.b.columns(),
                                                                    problem.b.leadingDimension(),
                                                                    problem.b.batchStride()},
                                                                   problem.batchCount,
                                                                   stream));
                    fcTypeB[i] = TiB;
                }
                // fast_check reads C only when beta is nonzero.
                if(compute_type_value_as_double(preparedProblems[i].beta, Tc) != 0)
                {
                    fcC[i].reset(new char[size_t(matmulProblems[i].m * matmulProblems[i].n
                                                 * matmulProblems[i].batchCount)
                                          * realDataTypeSize(TiC)]);
                    CHECK_HIP_ERROR(
                        fast_check_copy_region_to_host(fcC[i].get(),
                                                       {dC[i].buf(),
                                                        TiC,
                                                        matmulProblems[i].m,
                                                        matmulProblems[i].n,
                                                        matmulProblems[i].c.leadingDimension(),
                                                        matmulProblems[i].c.batchStride()},
                                                       matmulProblems[i].batchCount,
                                                       stream));
                }
                if(!compactHostInputs && arg.c_equal_d)
                    CHECK_HIP_ERROR(synchronize(hC[i], dC[i], 0, 0, 0, 0, 1, false, stream));
            }

            // The MX path already produced reference values and the
            // kernel-ready scale layout for both A and B.
            if(arg.dump_matrix)
            {
                for(int batchId = 0; batchId < problem.batchCount; batchId++)
                {
                    hipblasltDispatchValuesToFile(transA,
                                                  TiA,
                                                  problem.m,
                                                  problem.k,
                                                  problem.a.leadingDimension(),
                                                  hA[i].buf(),
                                                  "batch_" + std::to_string(batchId) + "_"
                                                      + std::to_string(i) + "_A_input.txt");
                    hipblasltDispatchValuesToFile(transB,
                                                  TiB,
                                                  problem.k,
                                                  problem.n,
                                                  problem.b.leadingDimension(),
                                                  hB[i].buf(),
                                                  "batch_" + std::to_string(batchId) + "_"
                                                      + std::to_string(i) + "_B_input.txt");
                    hipblasltDispatchValuesToFile(HIPBLAS_OP_N,
                                                  TiC,
                                                  problem.m,
                                                  problem.n,
                                                  problem.c.leadingDimension(),
                                                  hC[i].buf(),
                                                  "batch_" + std::to_string(batchId) + "_"
                                                      + std::to_string(i) + "_C_input.txt");
                }
            }

            if(do_swizzle_a)
            {
                HipHostBuffer tmp(TiA, preparedProblem.a.elements);
                swizzle_tensor_type(tmp,
                                    hA[i],
                                    TiA,
                                    arg,
                                    problem.batchCount,
                                    problem.m,
                                    problem.k,
                                    problem.a.leadingDimension(),
                                    false);
                CHECK_HIP_ERROR(synchronize(dA[i], tmp, block_count));
            }

            if(do_swizzle_b)
            {
                HipHostBuffer tmp(TiB, preparedProblem.b.elements);
                swizzle_tensor_type(tmp,
                                    hB[i],
                                    TiB,
                                    arg,
                                    problem.batchCount,
                                    problem.n,
                                    problem.k,
                                    problem.b.leadingDimension(),
                                    false);
                CHECK_HIP_ERROR(synchronize(dB[i], tmp, block_count));
            }

            if(arg.gradient && arg.use_e)
            {
                const auto& auxiliary = *problem.auxiliary;
                initializeHostTensor(
                    hE[i],
                    Taux,
                    auxiliary.layout,
                    randomInitializationRecipe(
                        Taux,
                        hipblaslt::hostnumerics::initialization::OperandSequence::Auxiliary));
            }

            if(arg.bias_vector)
            {
                // Filling up unique bias values for each batch in Strided Batch
                if(arg.bias_stride > 0)
                    initializeHostTensor(
                        hBias[i],
                        Tbias,
                        roc::hostnumerics::Layout(
                            roc::hostnumerics::Shape{static_cast<size_t>(arg.bias_stride),
                                                      1,
                                                      static_cast<size_t>(problem.batchCount)},
                            {1, arg.bias_stride, arg.bias_stride}),
                        randomInitializationRecipe(
                            Tbias,
                            hipblaslt::hostnumerics::initialization::OperandSequence::Bias));
                else
                    initializeHostTensor(
                        hBias[i],
                        Tbias,
                        roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                            roc::hostnumerics::Shape{preparedProblem.biasElements}),
                        randomInitializationRecipe(
                            Tbias,
                            hipblaslt::hostnumerics::initialization::OperandSequence::Bias));
            }

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                if(arg.norm_check)
                    initializeHostTensor(
                        hScaleA[i],
                        Talpha,
                        roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                            roc::hostnumerics::Shape{preparedProblem.a.scaleElements}),
                        randomInitializationRecipe(
                            Talpha,
                            hipblaslt::hostnumerics::initialization::OperandSequence::ScaleA,
                            true));
                else
                    initializeHostTensor(
                        hScaleA[i],
                        Talpha,
                        roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                            roc::hostnumerics::Shape{preparedProblem.a.scaleElements}),
                        randomInitializationRecipe(
                            Talpha,
                            hipblaslt::hostnumerics::initialization::OperandSequence::ScaleA));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                if(arg.norm_check)
                    initializeHostTensor(
                        hScaleB[i],
                        Talpha,
                        roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                            roc::hostnumerics::Shape{preparedProblem.b.scaleElements}),
                        randomInitializationRecipe(
                            Talpha,
                            hipblaslt::hostnumerics::initialization::OperandSequence::ScaleB,
                            true));
                else
                    initializeHostTensor(
                        hScaleB[i],
                        Talpha,
                        roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                            roc::hostnumerics::Shape{preparedProblem.b.scaleElements}),
                        randomInitializationRecipe(
                            Talpha,
                            hipblaslt::hostnumerics::initialization::OperandSequence::ScaleB));
            }

            if(arg.scaleC)
            {
                const bool small = TiC == HIP_R_8F_E4M3_FNUZ || TiC == HIP_R_8F_E5M2_FNUZ;
                initializeHostTensor(
                    hScaleC[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest({1}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::ScaleC,
                        small));
            }

            if(arg.scaleD)
            {
                const bool small = To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ;
                initializeHostTensor(
                    hScaleD[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest({1}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::ScaleD,
                        small));
            }

            if(arg.amaxD)
                initializeHostTensor(
                    hAmaxD_gold[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest({1}),
                    roc::hostnumerics::GenerationRecipe::realOnly(
                        roc::hostnumerics::GenerationRecipe::zero()));

            if(arg.scaleE)
                initializeHostTensor(
                    hScaleE[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest({1}),
                    randomInitializationRecipe(
                        Talpha, hipblaslt::hostnumerics::initialization::OperandSequence::ScaleE));

            if(arg.scaleAlpha_vector)
                initializeHostTensor(
                    hScaleAlphaVec[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                        {static_cast<size_t>(problem.m)}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::AlphaVector));

            if(arg.gradient && arg.use_e)
            {
                CHECK_HIP_ERROR(synchronize(dE[i], hE[i], block_count));
            }
            if(!arg.gradient && arg.bias_vector)
            {
                CHECK_HIP_ERROR(synchronize(dBias[i], hBias[i], block_count));
                // A kernel that reads a bias entry past M, in the gap before the next batch's
                // vector, reads poison and produces a wrong D.
                if(arg.fast_check && arg.bias_stride > matmulProblems[i].m)
                {
                    fast_check_poison_padding_device(
                        {dBias[i].buf(), Tbias, matmulProblems[i].m, 1, arg.bias_stride, arg.bias_stride},
                        matmulProblems[i].batchCount,
                        preparedProblems[i].biasElements,
                        stream);
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                }
            }

            if(arg.scaleAlpha_vector)
            {
                CHECK_HIP_ERROR(synchronize(dScaleAlphaVec[i], hScaleAlphaVec[i], block_count));
                runtimeProblem.alphaPointer = dScaleAlphaVec[i].buf();
            }
            else
                runtimeProblem.alphaPointer = &(preparedProblem.alpha);

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                if(arg.amaxScaleA && (arg.a_type == HIP_R_32F || arg.a_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.a_type,
                                                           HIP_R_32F,
                                                           dScaleA[i].buf(),
                                                           dA[i].buf(),
                                                           problem.a.rows(),
                                                           problem.a.columns(),
                                                           stream));

                    CHECK_HIP_ERROR(synchronize(hScaleA[i], dScaleA[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                if(arg.amaxScaleB && (arg.b_type == HIP_R_32F || arg.b_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.b_type,
                                                           HIP_R_32F,
                                                           dScaleB[i].buf(),
                                                           dB[i].buf(),
                                                           problem.b.rows(),
                                                           problem.b.columns(),
                                                           stream));
                    CHECK_HIP_ERROR(synchronize(hScaleB[i], dScaleB[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
            }

            if(arg.scaleC)
                CHECK_HIP_ERROR(synchronize(dScaleC[i], hScaleC[i]));

            if(arg.scaleD)
                CHECK_HIP_ERROR(synchronize(dScaleD[i], hScaleD[i]));

            if(arg.scaleE)
                CHECK_HIP_ERROR(synchronize(dScaleE[i], hScaleE[i]));

            if(preparedProblem.epilogueEnabled)
            {
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &(preparedProblem.epilogue),
                                                    sizeof(preparedProblem.epilogue)),
                    HIPBLAS_STATUS_SUCCESS);
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
                                                    &arg.activation_arg1,
                                                    sizeof(arg.activation_arg1)));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
                                                    &arg.activation_arg2,
                                                    sizeof(arg.activation_arg2)));
            }

            if(arg.use_e)
            {
                const auto&   auxiliary                 = *problem.auxiliary;
                const int64_t auxiliaryLeadingDimension = auxiliary.leadingDimension();
                const int64_t auxiliaryBatchStride      = auxiliary.batchStride();
                void*         e_addr                    = dE[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                    &e_addr,
                                                    sizeof(void*)));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                                    &arg.aux_type,
                                                    sizeof(hipDataType)));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                    &auxiliaryLeadingDimension,
                                                    sizeof(int64_t)));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                                    &auxiliaryBatchStride,
                                                    sizeof(int64_t)));
            }

            if(arg.bias_vector)
            {
                const void* bias_addr;
                int32_t     bias_stride = arg.bias_stride;
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &arg.bias_type,
                                                    sizeof(hipDataType)),
                    HIPBLAS_STATUS_SUCCESS);
                bias_addr = dBias[i].buf();

                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias_addr,
                                                    sizeof(void*)),
                    HIPBLAS_STATUS_SUCCESS);

                if(bias_stride > 0)
                    EXPECT_HIPBLAS_STATUS(
                        hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                        HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                                                        &bias_stride,
                                                        sizeof(bias_stride)),
                        HIPBLAS_STATUS_SUCCESS);
            }

            if(arg.scaleA != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;

                void* scaleA_addr = (void*)(dScaleA[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul, attr, &scaleA_addr, sizeof(void*)));

                hipblasLtMatmulMatrixScale_t mode = hipblaslt::client::matmulScaleMode(arg.scaleA);

                if(mode != HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
                {
                    auto attr = HIPBLASLT_MATMUL_DESC_A_SCALE_MODE;
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        primaryMatmul, attr, &mode, sizeof(uint32_t)));
                }
            }

            if(arg.scaleB != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;

                void* scaleB_addr = (void*)(dScaleB[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul, attr, &scaleB_addr, sizeof(void*)));

                hipblasLtMatmulMatrixScale_t mode = hipblaslt::client::matmulScaleMode(arg.scaleB);

                if(mode != HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
                {
                    auto attr = HIPBLASLT_MATMUL_DESC_B_SCALE_MODE;
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        primaryMatmul, attr, &mode, sizeof(uint32_t)));
                }
            }

            if(arg.scaleC)
            {
                void* scaleC_addr = dScaleC[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                                    &scaleC_addr,
                                                    sizeof(void*)));
            }

            if(arg.scaleD)
            {
                void* scaleD_addr = dScaleD[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                    &scaleD_addr,
                                                    sizeof(void*)));
            }

            if(arg.amaxD)
            {
                void* amaxD_addr = dAmaxD[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                    &amaxD_addr,
                                                    sizeof(void*)));
            }

            if(arg.scaleE)
            {
                void* scaleE_addr = dScaleE[i].buf();
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul,
                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                    &scaleE_addr,
                    sizeof(void*)));
            }

            if(arg.scaleAlpha_vector)
            {
                hipblasLtPointerMode_t scale_mode
                    = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                    &scale_mode,
                                                    sizeof(scale_mode)),
                    HIPBLAS_STATUS_SUCCESS);
            }
        }
        else
        {
            runtimeProblem.alphaPointer = &(preparedProblem.alpha);
            for(int batchCount = 0; batchCount < problem.batchCount; batchCount++)
            {
                initializeHostMatrix(hC[batchCount],
                                     problem.c,
                                     hipblaslt::hostnumerics::MatrixRole::C,
                                     beta_isnan_type(arg, Talpha),
                                     true,
                                     static_cast<size_t>(batchCount));
                CHECK_HIP_ERROR(synchronize(dC[batchCount], hC[batchCount], block_count));
                if(arg.dump_matrix)
                {
                    hipblasltDispatchValuesToFile(transA,
                                                  TiA,
                                                  problem.m,
                                                  problem.k,
                                                  problem.a.leadingDimension(),
                                                  hA[batchCount].buf(),
                                                  "batch_" + std::to_string(i) + "_A_"
                                                      + std::to_string(batchCount) + "_input.txt");
                    hipblasltDispatchValuesToFile(transB,
                                                  TiB,
                                                  problem.k,
                                                  problem.n,
                                                  problem.b.leadingDimension(),
                                                  hB[batchCount].buf(),
                                                  "batch_" + std::to_string(i) + "_B_"
                                                      + std::to_string(batchCount) + "_input.txt");
                    hipblasltDispatchValuesToFile(HIPBLAS_OP_N,
                                                  TiC,
                                                  problem.m,
                                                  problem.n,
                                                  problem.c.leadingDimension(),
                                                  hC[batchCount].buf(),
                                                  "batch_" + std::to_string(i) + "_C_"
                                                      + std::to_string(batchCount) + "_input.txt");
                }
                if(do_swizzle_a)
                {
                    HipHostBuffer tmp(TiA, preparedProblem.a.elements);
                    swizzle_tensor_type(tmp,
                                        hA[batchCount],
                                        TiA,
                                        arg,
                                        1,
                                        problem.m,
                                        problem.k,
                                        problem.a.leadingDimension(),
                                        false);
                    CHECK_HIP_ERROR(synchronize(dA[batchCount], tmp, block_count));
                }

                if(do_swizzle_b)
                {
                    HipHostBuffer tmp(TiB, preparedProblem.b.elements);
                    swizzle_tensor_type(tmp,
                                        hB[batchCount],
                                        TiB,
                                        arg,
                                        1,
                                        problem.n,
                                        problem.k,
                                        problem.b.leadingDimension(),
                                        false);
                    CHECK_HIP_ERROR(synchronize(dB[batchCount], tmp, block_count));
                }
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            {
                initializeHostTensor(
                    hScaleA[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                        {preparedProblem.a.scaleElements}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::ScaleA,
                        arg.norm_check));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            {
                initializeHostTensor(
                    hScaleB[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest(
                        {preparedProblem.b.scaleElements}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::ScaleB,
                        arg.norm_check));
            }
            if(arg.scaleC)
            {
                const bool small = TiC == HIP_R_8F_E4M3_FNUZ || TiC == HIP_R_8F_E5M2_FNUZ;
                initializeHostTensor(
                    hScaleC[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest({1}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::ScaleC,
                        small));
            }

            if(arg.scaleD)
            {
                const bool small = To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ;
                initializeHostTensor(
                    hScaleD[i],
                    Talpha,
                    roc::hostnumerics::Layout::contiguousLastDimensionFastest({1}),
                    randomInitializationRecipe(
                        Talpha,
                        hipblaslt::hostnumerics::initialization::OperandSequence::ScaleD,
                        small));
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            {
                if(arg.amaxScaleA && (arg.a_type == HIP_R_32F || arg.a_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.a_type,
                                                           HIP_R_32F,
                                                           dScaleA[i].buf(),
                                                           dA[i].buf(),
                                                           problem.a.rows(),
                                                           problem.a.columns(),
                                                           stream));

                    CHECK_HIP_ERROR(synchronize(hScaleA[i], dScaleA[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            {
                if(arg.amaxScaleB && (arg.b_type == HIP_R_32F || arg.b_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.b_type,
                                                           HIP_R_32F,
                                                           dScaleB[i].buf(),
                                                           dB[i].buf(),
                                                           problem.b.rows(),
                                                           problem.b.columns(),
                                                           stream));
                    CHECK_HIP_ERROR(synchronize(hScaleB[i], dScaleB[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
            }

            if(arg.scaleC)
                CHECK_HIP_ERROR(synchronize(dScaleC[i], hScaleC[i]));
            if(arg.scaleD)
                CHECK_HIP_ERROR(synchronize(dScaleD[i], hScaleD[i]));

            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            {
                hipblasLtMatmulDescAttributes_t attr        = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
                void*                           scaleA_addr = (void*)(dScaleA[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul, attr, &scaleA_addr, sizeof(void*)));
                hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                attr                              = HIPBLASLT_MATMUL_DESC_A_SCALE_MODE;
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul, attr, &mode, sizeof(uint32_t)));
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            {
                hipblasLtMatmulDescAttributes_t attr        = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
                void*                           scaleB_addr = (void*)(dScaleB[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    primaryMatmul, attr, &scaleB_addr, sizeof(void*)));
                hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                attr                              = HIPBLASLT_MATMUL_DESC_B_SCALE_MODE;
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul, attr, &mode, sizeof(uint32_t)));
            }
            if(arg.scaleC)
            {
                void* scaleC_addr = dScaleC[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                                    &scaleC_addr,
                                                    sizeof(void*)));
            }
            if(arg.scaleD)
            {
                void* scaleD_addr = dScaleD[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(primaryMatmul,
                                                    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                    &scaleD_addr,
                                                    sizeof(void*)));
            }
        }
        for(int32_t b = 1; b < block_count; b++)
        {
            runtimeProblem.matmulDescriptors.emplace_back(arg.compute_type, arg.scale_type);
            auto& blockMatmul = runtimeProblem.matmulDescriptors.back();
            CHECK_HIPBLASLT_ERROR(blockMatmul.status());
            CHECK_HIPBLASLT_ERROR(hipblaslt_ext::copyMatmul(primaryMatmul, blockMatmul));

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(blockMatmul,
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                                &TciA,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(blockMatmul,
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                                &TciB,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            // Forward CLI knobs from hipblaslt-bench into the matmul descriptor.
            {
                int32_t sm = hipblaslt_bench_options::sm_count_target();
                if(sm != 0)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        blockMatmul, HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm, sizeof(sm)));
                }
                int32_t dyn = hipblaslt_bench_options::streamk_tile_scheduling_mode();
                if(dyn >= 0)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        blockMatmul,
                        HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                        &dyn,
                        sizeof(dyn)));
                }
                int32_t uso = hipblaslt_bench_options::uniform_summation_order();
                if(uso >= 0)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        blockMatmul,
                        HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT,
                        &uso,
                        sizeof(uso)));
                }
            }

            if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
            {
                // Update bias, E
                if(arg.bias_vector)
                {
                    const void* bias_addr = rotatingBlockAddress(dBias[i], b);
                    EXPECT_HIPBLAS_STATUS(
                        hipblasLtMatmulDescSetAttribute(blockMatmul,
                                                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &bias_addr,
                                                        sizeof(void*)),
                        HIPBLAS_STATUS_SUCCESS);
                }
                if(arg.use_e)
                {
                    void* e_addr = rotatingBlockAddress(dE[i], b);
                    CHECK_HIPBLASLT_ERROR(
                        hipblasLtMatmulDescSetAttribute(blockMatmul,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                        &e_addr,
                                                        sizeof(void*)));
                }
            }
            if(arg.scaleA != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
                void* scaleA_addr = rotatingBlockAddress(dScaleA[i], b);
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    blockMatmul, attr, &scaleA_addr, sizeof(void*)));
            }

            if(arg.scaleB != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
                void* scaleB_addr = rotatingBlockAddress(dScaleB[i], b);
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    blockMatmul, attr, &scaleB_addr, sizeof(void*)));
            }
        }
    }

    // set preference
    size_t                     max_workspace_size = arg.user_allocated_workspace;
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)),
        HIPBLAS_STATUS_SUCCESS);

    // set workspace
    std::unique_ptr<device_vector<unsigned char>> dWorkspace;
    size_t                        workspace_size = 0;

    // set user args
    hipblaslt_ext::UserArguments* userArgs   = nullptr;
    hipblaslt_ext::UserArguments* d_userArgs = nullptr;

    // Get Heuristic results
    int32_t requestAlgoCount = arg.requested_solution_num < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM
                                                              : arg.requested_solution_num;
    int     returnedAlgoCount = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    std::vector<size_t>                           heuristicTuningIndex;

    // Cpp API
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    std::vector<hipblaslt_ext::Gemm>                    gemmVec;
    std::vector<hipblaslt_ext::GroupedGemm>             groupedGemmVec;
    std::vector<std::vector<hipblaslt_ext::GemmInputs>> extinputs;

    // Pointer-array GEMM has one logical problem with one pointer binding per batch.
    const int32_t binding_count
        = batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY ? arg.batch_count : problem_count;
    // C to Cpp API for GG
    const auto groupedGemmBlockCount = do_grouped_gemm ? block_count : 0;
    std::vector<std::vector<void*>> da(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<std::vector<void*>> db(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<std::vector<void*>> dc(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<std::vector<void*>> dd(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<uint64_t*> dda, ddb, ddc, ddd;
    if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        for(int i = 0; i < block_count; i++)
        {
            uint64_t* ptr = nullptr;
            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            dda.push_back(ptr);

            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            ddb.push_back(ptr);

            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            ddc.push_back(ptr);

            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            ddd.push_back(ptr);
        }
    }

    for(int32_t b = 0; b < block_count; b++)
    {
        if(!do_grouped_gemm)
            gemmVec.push_back(hipblaslt_ext::Gemm(handle,
                                                  transA,
                                                  transB,
                                                  arg.a_type,
                                                  arg.b_type,
                                                  arg.c_type,
                                                  arg.d_type,
                                                  arg.compute_type));
        else
            groupedGemmVec.push_back(hipblaslt_ext::GroupedGemm(handle,
                                                                transA,
                                                                transB,
                                                                arg.a_type,
                                                                arg.b_type,
                                                                arg.c_type,
                                                                arg.d_type,
                                                                arg.compute_type));
    }

    std::vector<hipblaslt_ext::GemmEpilogue> extepilogue;
    hipblaslt_ext::GemmProblemType           extproblemtype;
    if(arg.use_ext_setproblem && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        extinputs.resize(block_count, std::vector<hipblaslt_ext::GemmInputs>(problem_count));
        extepilogue.resize(problem_count);

        for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
        {
            const auto& problem         = matmulProblems[gemmIdx];
            const auto& preparedProblem = preparedProblems[gemmIdx];
            auto        bias_type       = HIPBLASLT_DATATYPE_INVALID;
            auto        aux_type        = HIPBLASLT_DATATYPE_INVALID;
            void*       bias_addr       = nullptr;
            for(int32_t b = 0; b < block_count; b++)
            {
                if(arg.bias_vector)
                {
                    bias_type = arg.bias_type;
                    bias_addr = rotatingBlockAddress(dBias[gemmIdx], b);
                }
                if(arg.use_e)
                {
                    aux_type = arg.aux_type;
                }
                if(b == 0)
                {
                    extepilogue[gemmIdx].setMode(preparedProblem.epilogue);
                    extepilogue[gemmIdx].setBiasDataType(bias_type);
                    extepilogue[gemmIdx].setAuxDataType(aux_type);
                    if(problem.auxiliary)
                    {
                        extepilogue[gemmIdx].setAuxLeadingDimension(
                            problem.auxiliary->leadingDimension());
                        extepilogue[gemmIdx].setAuxBatchStride(problem.auxiliary->batchStride());
                    }
                    extepilogue[gemmIdx].setScalingAType(
                        hipblaslt::client::matmulScaleMode(arg.scaleA));
                    extepilogue[gemmIdx].setScalingBType(
                        hipblaslt::client::matmulScaleMode(arg.scaleB));
                }
                extinputs[b][gemmIdx].setA(rotatingBlockAddress(dA[gemmIdx], b));
                extinputs[b][gemmIdx].setB(rotatingBlockAddress(dB[gemmIdx], b));
                extinputs[b][gemmIdx].setC(rotatingBlockAddress(dC[gemmIdx], b));
                extinputs[b][gemmIdx].setD(rotatingBlockAddress(dOutput[gemmIdx], b));
                extinputs[b][gemmIdx].setAlpha(&preparedProblem.alpha);
                extinputs[b][gemmIdx].setBeta(&preparedProblem.beta);
                extinputs[b][gemmIdx].setBias(bias_addr);
                extinputs[b][gemmIdx].setScaleA(arg.scaleA != hipblaslt_scaling_format::none
                                                    ? rotatingBlockAddress(dScaleA[gemmIdx], b)
                                                    : nullptr);
                extinputs[b][gemmIdx].setScaleB(arg.scaleB != hipblaslt_scaling_format::none
                                                    ? rotatingBlockAddress(dScaleB[gemmIdx], b)
                                                    : nullptr);
                extinputs[b][gemmIdx].setScaleC(arg.scaleC ? dScaleC[gemmIdx].as<char>() : nullptr);
                extinputs[b][gemmIdx].setScaleD(arg.scaleD ? dScaleD[gemmIdx].as<char>() : nullptr);
                extinputs[b][gemmIdx].setScaleAux(arg.scaleE ? dScaleE[gemmIdx].as<char>()
                                                             : nullptr);
                extinputs[b][gemmIdx].setAmaxD(arg.amaxD ? dAmaxD[gemmIdx].as<char>() : nullptr);
                if(arg.use_e)
                    extinputs[b][gemmIdx].setAux(rotatingBlockAddress(dE[gemmIdx], b));
                if(arg.scaleAlpha_vector)
                    extinputs[b][gemmIdx].setScaleAlphaVec(
                        rotatingBlockAddress(dScaleAlphaVec[gemmIdx], b));
            }
        }
        extproblemtype.setOpA(transA);
        extproblemtype.setOpB(transB);
        extproblemtype.setTypeA(arg.a_type);
        extproblemtype.setTypeB(arg.b_type);
        extproblemtype.setTypeC(arg.c_type);
        extproblemtype.setTypeD(arg.d_type);
        extproblemtype.setTypeCompute(arg.compute_type);

        if(do_swizzle_a)
        {
            hipblasLtOrder_t orderA = hipblaslt::client::matmulOrderForDataType(TiA);
            extproblemtype.setOrderA(orderA);
        }
        if(do_swizzle_b)
        {
            hipblasLtOrder_t orderB = hipblaslt::client::matmulOrderForDataType(TiB);
            extproblemtype.setOrderB(orderB);
        }
    }
    else if(arg.grouped_gemm)
    {
        for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
        {
            const auto& problem         = matmulProblems[gemmIdx];
            const auto& preparedProblem = preparedProblems[gemmIdx];
            for(int32_t b = 0; b < block_count; b++)
            {
                da[b][gemmIdx] = rotatingBlockAddress(dA[gemmIdx], b);
                db[b][gemmIdx] = rotatingBlockAddress(dB[gemmIdx], b);
                dc[b][gemmIdx] = rotatingBlockAddress(dC[gemmIdx], b);
                dd[b][gemmIdx] = rotatingBlockAddress(dOutput[gemmIdx], b);
            }
        }
    }
    else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        std::vector<uint64_t*> da1(binding_count), db1(binding_count), dc1(binding_count),
            dd1(binding_count);
        for(int32_t b = 0; b < block_count; b++)
        {
            for(int gemmIdx = 0; gemmIdx < binding_count; gemmIdx++)
            {
                da1[gemmIdx]
                    = reinterpret_cast<uint64_t*>(rotatingBlockAddress(dA[gemmIdx], b));
                db1[gemmIdx]
                    = reinterpret_cast<uint64_t*>(rotatingBlockAddress(dB[gemmIdx], b));
                dc1[gemmIdx]
                    = reinterpret_cast<uint64_t*>(rotatingBlockAddress(dC[gemmIdx], b));
                dd1[gemmIdx]
                    = reinterpret_cast<uint64_t*>(rotatingBlockAddress(dOutput[gemmIdx], b));
            }
            CHECK_HIP_ERROR(hipMemcpy(
                dda[b], da1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                ddb[b], db1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                ddc[b], dc1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                ddd[b], dd1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
        }
    }

    hipblaslt_ext::GemmType gemmType = do_grouped_gemm
                                           ? hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM
                                           : hipblaslt_ext::GemmType::HIPBLASLT_GEMM;

    // Remove duplicate
    std::vector<uint32_t> gsu_vector;
    std::vector<uint32_t> wgm_vector;
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.gsu_vector[i] == -1)
            break;
        gsu_vector.push_back(arg.gsu_vector[i]);
    }
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.wgm_vector[i] == -1)
            break;
        wgm_vector.push_back(arg.wgm_vector[i]);
    }
    std::set<uint32_t> remove_duplicate(gsu_vector.begin(), gsu_vector.end());
    gsu_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    remove_duplicate = std::set<uint32_t>(wgm_vector.begin(), wgm_vector.end());
    wgm_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    std::vector<hipblaslt_ext::GemmTuning> tuningVec;
    if(arg.use_ext)
    {
        for(size_t wgm = 0; wgm < wgm_vector.size(); wgm++)
            for(size_t gsu = 0; gsu < gsu_vector.size(); gsu++)
            {
                hipblaslt_ext::GemmTuning tuning;
                tuning.setSplitK(gsu_vector[gsu]);
                tuning.setWgm(wgm_vector[wgm]);
                tuningVec.push_back(tuning);
            }
    }
    else
    {
        // C API does not support
        tuningVec.push_back(hipblaslt_ext::GemmTuning());
    }

    hipblaslt_ext::GemmInstance* extProblem = nullptr;
    if(do_grouped_gemm)
    {
        if(arg.use_ext_setproblem)
        {
            auto collectProblemValues = [&](auto projection) {
                std::vector<int64_t> values;
                values.reserve(matmulProblems.size());
                for(const auto& problem : matmulProblems)
                    values.push_back(projection(problem));
                return values;
            };
            auto rows       = collectProblemValues([](const auto& problem) { return problem.m; });
            auto columns    = collectProblemValues([](const auto& problem) { return problem.n; });
            auto reductions = collectProblemValues([](const auto& problem) { return problem.k; });
            std::vector<int64_t> batchCounts;
            batchCounts.reserve(matmulProblems.size());
            for(const auto& problem : matmulProblems)
                batchCounts.push_back(problem.batchCount);
            auto leadingDimensionsA = collectProblemValues(
                [](const auto& problem) { return problem.a.leadingDimension(); });
            auto leadingDimensionsB = collectProblemValues(
                [](const auto& problem) { return problem.b.leadingDimension(); });
            auto leadingDimensionsC = collectProblemValues(
                [](const auto& problem) { return problem.c.leadingDimension(); });
            auto leadingDimensionsD = collectProblemValues(
                [](const auto& problem) { return problem.d.leadingDimension(); });
            auto batchStridesC
                = collectProblemValues([](const auto& problem) { return problem.c.batchStride(); });
            auto batchStridesD
                = collectProblemValues([](const auto& problem) { return problem.d.batchStride(); });
            std::vector<int64_t> deviceBatchStridesA;
            std::vector<int64_t> deviceBatchStridesB;
            deviceBatchStridesA.reserve(preparedProblems.size());
            deviceBatchStridesB.reserve(preparedProblems.size());
            for(const auto& preparedProblem : preparedProblems)
            {
                deviceBatchStridesA.push_back(preparedProblem.a.batchStride);
                deviceBatchStridesB.push_back(preparedProblem.b.batchStride);
            }
            for(int32_t block = 0; block < block_count; ++block)
            {
                CHECK_HIPBLASLT_ERROR(groupedGemmVec[block].setProblem(rows,
                                                                       columns,
                                                                       reductions,
                                                                       batchCounts,
                                                                       leadingDimensionsA,
                                                                       leadingDimensionsB,
                                                                       leadingDimensionsC,
                                                                       leadingDimensionsD,
                                                                       deviceBatchStridesA,
                                                                       deviceBatchStridesB,
                                                                       batchStridesC,
                                                                       batchStridesD,
                                                                       extepilogue,
                                                                       extinputs[block],
                                                                       extproblemtype));
            }
        }
        else
        {
            std::vector<void*>                   alphaPointers;
            std::vector<void*>                   betaPointers;
            std::vector<hipblasLtMatrixLayout_t> matrixLayoutsA;
            std::vector<hipblasLtMatrixLayout_t> matrixLayoutsB;
            std::vector<hipblasLtMatrixLayout_t> matrixLayoutsC;
            std::vector<hipblasLtMatrixLayout_t> matrixLayoutsD;
            alphaPointers.reserve(preparedProblems.size());
            betaPointers.reserve(preparedProblems.size());
            matrixLayoutsA.reserve(preparedProblems.size());
            matrixLayoutsB.reserve(preparedProblems.size());
            matrixLayoutsC.reserve(preparedProblems.size());
            matrixLayoutsD.reserve(preparedProblems.size());
            for(size_t i = 0; i < preparedProblems.size(); ++i)
            {
                auto& preparedProblem = preparedProblems[i];
                auto& runtimeProblem  = runtimeProblems[i];
                alphaPointers.push_back(&preparedProblem.alpha);
                betaPointers.push_back(&preparedProblem.beta);
                matrixLayoutsA.push_back(runtimeProblem.matrixA);
                matrixLayoutsB.push_back(runtimeProblem.matrixB);
                matrixLayoutsC.push_back(runtimeProblem.matrixC);
                matrixLayoutsD.push_back(runtimeProblem.matrixD);
            }
            for(int32_t block = 0; block < block_count; ++block)
            {
                std::vector<hipblasLtMatmulDesc_t> blockDescriptors;
                blockDescriptors.reserve(runtimeProblems.size());
                for(auto& runtimeProblem : runtimeProblems)
                    blockDescriptors.push_back(runtimeProblem.matmulDescriptors[block]);

                CHECK_HIPBLASLT_ERROR(groupedGemmVec[block].setProblem(blockDescriptors,
                                                                       alphaPointers,
                                                                       da[block],
                                                                       matrixLayoutsA,
                                                                       db[block],
                                                                       matrixLayoutsB,
                                                                       betaPointers,
                                                                       dc[block],
                                                                       matrixLayoutsC,
                                                                       dd[block],
                                                                       matrixLayoutsD));
            }
        }
        extProblem = &groupedGemmVec.front();
    }
    else if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        for(int32_t block = 0; block < block_count; ++block)
        {
            if(arg.use_ext_setproblem)
            {
                CHECK_HIPBLASLT_ERROR(gemmVec[block].setProblem(firstProblem.m,
                                                                firstProblem.n,
                                                                firstProblem.k,
                                                                firstProblem.batchCount,
                                                                firstProblem.a.leadingDimension(),
                                                                firstProblem.b.leadingDimension(),
                                                                firstProblem.c.leadingDimension(),
                                                                firstProblem.d.leadingDimension(),
                                                                firstPreparedProblem.a.batchStride,
                                                                firstPreparedProblem.b.batchStride,
                                                                firstProblem.c.batchStride(),
                                                                firstProblem.d.batchStride(),
                                                                extepilogue[0],
                                                                extinputs[block][0],
                                                                extproblemtype));
            }
            else
            {
                CHECK_HIPBLASLT_ERROR(gemmVec[block].setProblem(
                    firstRuntimeProblem.matmulDescriptors[block],
                    arg.scaleAlpha_vector
                        ? rotatingBlockAddress(dScaleAlphaVec[0], block)
                        : firstRuntimeProblem.alphaPointer,
                    rotatingBlockAddress(dA[0], block),
                    firstRuntimeProblem.matrixA,
                    rotatingBlockAddress(dB[0], block),
                    firstRuntimeProblem.matrixB,
                    &firstPreparedProblem.beta,
                    rotatingBlockAddress(dC[0], block),
                    firstRuntimeProblem.matrixC,
                    rotatingBlockAddress(dOutput[0], block),
                    firstRuntimeProblem.matrixD));
            }
        }
        extProblem = &gemmVec.front();
    }

    auto collectSupportedAlgorithms
        = [&](std::span<hipblasLtMatmulHeuristicResult_t> candidates, size_t maximumAlgorithms) {
              size_t acceptedAlgorithms = 0;
              for(auto& candidate : candidates)
              {
                  bool accepted = false;
                  if(extProblem != nullptr)
                  {
                      for(size_t tuningIndex = 0; tuningIndex < tuningVec.size(); ++tuningIndex)
                      {
                          size_t requiredWorkspace = 0;
                          if(extProblem->isAlgoSupported(
                                 candidate.algo, tuningVec[tuningIndex], requiredWorkspace)
                                 != HIPBLAS_STATUS_SUCCESS
                             || requiredWorkspace > max_workspace_size)
                              continue;
                          heuristicResult.push_back(candidate);
                          heuristicTuningIndex.push_back(tuningIndex);
                          workspace_size = std::max(workspace_size, requiredWorkspace);
                          accepted       = true;
                      }
                  }
                  else
                  {
                      size_t requiredWorkspace            = 0;
                      candidate.algo.max_workspace_bytes  = max_workspace_size;
                      const hipblasStatus_t supportStatus = hipblaslt_ext::matmulIsAlgoSupported(
                          handle,
                          firstRuntimeProblem.matmulDescriptors.front(),
                          firstRuntimeProblem.alphaPointer,
                          firstRuntimeProblem.matrixA,
                          firstRuntimeProblem.matrixB,
                          &firstPreparedProblem.beta,
                          firstRuntimeProblem.matrixC,
                          firstRuntimeProblem.matrixD,
                          candidate.algo,
                          requiredWorkspace);
                      if(supportStatus == HIPBLAS_STATUS_SUCCESS
                         && requiredWorkspace <= max_workspace_size)
                      {
                          heuristicResult.push_back(candidate);
                          heuristicTuningIndex.push_back(0);
                          workspace_size = std::max(workspace_size, requiredWorkspace);
                          accepted       = true;
                      }
                  }
                  CHECK_RETURNED_WORKSPACE_SIZE(workspace_size, max_workspace_size);
                  acceptedAlgorithms += accepted;
                  if(acceptedAlgorithms >= maximumAlgorithms)
                      break;
              }
          };

    if(arg.algo_method == 2)
    {
        heuristicResult.clear();
        heuristicTuningIndex.clear();

        const bool       indicesAreDiscovered = (arg.solution_index == -1);
        std::vector<int> indices;
        if(indicesAreDiscovered)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> allAlgos;
            EXPECT_HIPBLAS_STATUS(hipblaslt_ext::getAllAlgos(handle,
                                                             gemmType,
                                                             transA,
                                                             transB,
                                                             arg.a_type,
                                                             arg.b_type,
                                                             arg.c_type,
                                                             arg.d_type,
                                                             arg.compute_type,
                                                             allAlgos),
                                  HIPBLAS_STATUS_SUCCESS);
            indices.reserve(allAlgos.size());
            for(auto& result : allAlgos)
                indices.push_back(hipblaslt_ext::getIndexFromAlgo(result.algo));
        }
        else
        {
            indices.push_back(arg.solution_index);
        }

        constexpr size_t batchSize = 100;
        for(size_t batchStart = 0; batchStart < indices.size(); batchStart += batchSize)
        {
            const size_t     batchEnd = std::min(batchStart + batchSize, indices.size());
            std::vector<int> batch(indices.begin() + batchStart, indices.begin() + batchEnd);
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
            const auto status = hipblaslt_ext::getAlgosFromIndex(handle, batch, candidates);
            if(indicesAreDiscovered)
                EXPECT_HIPBLAS_STATUS(status, HIPBLAS_STATUS_SUCCESS);
            if(candidates.empty())
                break;

            const size_t previousResultCount = heuristicResult.size();
            const size_t maximumAlgorithms
                = indicesAreDiscovered && extProblem == nullptr ? candidates.size() : 1;
            collectSupportedAlgorithms(candidates, maximumAlgorithms);
            if(heuristicResult.size() != previousResultCount)
                break;
        }
    }
    else if(arg.algo_method == 1)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        EXPECT_HIPBLAS_STATUS(hipblaslt_ext::getAllAlgos(handle,
                                                         gemmType,
                                                         transA,
                                                         transB,
                                                         arg.a_type,
                                                         arg.b_type,
                                                         arg.c_type,
                                                         arg.d_type,
                                                         arg.compute_type,
                                                         tmpAlgo),
                              HIPBLAS_STATUS_SUCCESS);
        heuristicResult.clear();
        heuristicTuningIndex.clear();
        collectSupportedAlgorithms(tmpAlgo, requestAlgoCount);
    }
    else
    {
        heuristicResult.clear();
        heuristicTuningIndex.clear();
        if(extProblem != nullptr)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
            CHECK_HIPBLASLT_ERROR(
                extProblem->algoGetHeuristic(requestAlgoCount, gemmPref, candidates));
            collectSupportedAlgorithms(candidates, std::numeric_limits<size_t>::max());
        }
        else
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates(requestAlgoCount);
            EXPECT_HIPBLAS_STATUS(
                (hipblasLtMatmulAlgoGetHeuristic(handle,
                                                 firstRuntimeProblem.matmulDescriptors.front(),
                                                 firstRuntimeProblem.matrixA,
                                                 firstRuntimeProblem.matrixB,
                                                 firstRuntimeProblem.matrixC,
                                                 firstRuntimeProblem.matrixD,
                                                 pref,
                                                 requestAlgoCount,
                                                 candidates.data(),
                                                 &returnedAlgoCount)),
                HIPBLAS_STATUS_SUCCESS);
            candidates.resize(returnedAlgoCount);
            for(auto& candidate : candidates)
            {
                heuristicResult.push_back(candidate);
                heuristicTuningIndex.push_back(0);
                workspace_size = std::max(workspace_size, candidate.workspaceSize);
            }
            CHECK_RETURNED_WORKSPACE_SIZE(workspace_size, max_workspace_size);
        }
    }

    returnedAlgoCount = heuristicResult.size();

    // A size-threshold sweep may reach shapes the library declines; that is a correct answer.
    if(arg.allow_no_solution && returnedAlgoCount == 0)
    {
#ifdef GOOGLE_TEST
        GTEST_SKIP() << "the library offers no solution for this shape";
#else
        hipblaslt_cout << "the library offers no solution for this shape" << std::endl;
        return;
#endif
    }
    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    // A Stream-K case that got no Stream-K kernel (TileProcessingStrategy StreamK, "TPSSK" in the
    // kernel name) would pass while testing none.
    if(arg.requires_streamk)
    {
        bool streamk = false;
        for(auto& r : heuristicResult)
            streamk |= hipblaslt_ext::getKernelNameFromAlgo(handle, r.algo).find("_TPSSK_")
                       != std::string::npos;
        if(!streamk)
        {
            const std::string why = "none of the " + std::to_string(heuristicResult.size())
                                    + " solutions for this case is a Stream-K kernel";
#ifdef GOOGLE_TEST
            GTEST_SKIP() << why;
#else
            hipblaslt_cout << why << std::endl;
            return;
#endif
        }
    }

    // A placed workspace replaces the normal one, which placement runs never use: placement
    // requires fast_check, and fast_check refuses timing runs.
    const bool placeWorkspace = !strcmp(arg.placement, "workspace");
    dWorkspace                = std::make_unique<device_vector<unsigned char>>(
        placeWorkspace ? 0 : workspace_size * block_count, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());

    // The workspace the checked solutions use. A placed workspace moves for each solution, so
    // that solution's own workspace straddles the 4 GiB boundary.
    std::unique_ptr<PlacedRegion> placedWorkspace;
    void*                         workspacePtr   = static_cast<unsigned char*>(*dWorkspace);
    size_t                        workspaceBytes = workspace_size;
    if(placeWorkspace)
    {
        std::string why;
        bool        unsupported = false;
        // Whether any solution needs a workspace depends on the library and GPU, so a run with
        // none has nothing to place: a skip, not a failure.
        if(workspace_size == 0)
        {
            why         = "no solution here uses a workspace, so there is nothing to place";
            unsupported = true;
        }
        else
            // Twice the workspace (plus room for the boundary to land on a mapping granule), so
            // that each solution's start can sit below the boundary with the whole workspace
            // size, which any solution may use, still mapped after it.
            placedWorkspace = PlacedRegion::create(2 * workspace_size + (size_t(4) << 20),
                                                   size_t(arg.placement_offset),
                                                   HIP_R_8I,
                                                   &why,
                                                   &unsupported);
        CHECK_PLACEMENT(placedWorkspace != nullptr, unsupported, why);
        workspacePtr = placedWorkspace->ptr();
    }

    if(arg.use_user_args)
    {
        CHECK_HIP_ERROR(
            hipHostMalloc(&userArgs, problem_count * sizeof(hipblaslt_ext::UserArguments)));
        CHECK_HIP_ERROR(hipMalloc(
            &d_userArgs, block_count * problem_count * sizeof(hipblaslt_ext::UserArguments)));
    }

    auto ptrs = benchmark_allocation();

    if(arg.adaptive)
    {
        hipblaslt_bench::TimingConfig cfg;
        cfg.warmup_time         = arg.warmup_time;
        cfg.sample_time         = arg.sample_time;
        cfg.measure_time        = arg.measure_time;
        cfg.max_measure_time    = arg.max_measure_time;
        cfg.min_iters           = arg.min_iters;
        cfg.max_iters           = arg.max_iters;
        cfg.noise_threshold     = arg.noise_threshold;
        cfg.stability_threshold = arg.stability_threshold;
        cfg.stability_window    = arg.stability_window;
        cfg.stability_interval  = arg.stability_interval;
        hipblaslt_cout << hipblaslt_bench::format_adaptive_timing_summary(cfg) << std::endl;
    }

    if(arg.print_solution_found)
        hipblaslt_cout << "Is supported " << heuristicResult.size()
                       << " / Total solutions: " << returnedAlgoCount * tuningVec.size()
                       << std::endl;

    if(heuristicResult.size() != heuristicTuningIndex.size())
    {
        hipblaslt_cerr << "Internal error, heuristicResult.size() != heuristicTuningIndex.size() "
                       << heuristicResult.size() << " != " << heuristicTuningIndex.size()
                       << std::endl;
        exit(EXIT_FAILURE);
    }

    // get CPU result
    if(arg.unit_check || arg.norm_check || arg.allclose_check)
    {
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
        {
            const auto& problem         = matmulProblems[gemmIdx];
            const auto& preparedProblem = preparedProblems[gemmIdx];

            computeTypeInterface scale{};
            set_compute_type_value_from_double(scale, 1, Talpha, TiA);
            void* scaleCValue = arg.scaleC ? hScaleC[gemmIdx].buf() : (void*)(&scale);
            void* scaleDValue = arg.scaleD ? hScaleD[gemmIdx].buf() : (void*)(&scale);
            void* scaleEValue = arg.scaleE ? hScaleE[gemmIdx].buf() : (void*)(&scale);

            bool const isScaleAMXFormat = isBlockScaling(arg.scaleA);
            bool const isScaleBMXFormat = isBlockScaling(arg.scaleB);
            const bool pointerArrayMode = batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY;

            for(int batchIdx = 0; batchIdx < problem.batchCount; batchIdx++)
            {
                const auto hostBufferTensor = [&](std::vector<HipHostBuffer>&          buffers,
                                                  hipDataType                          type,
                                                  const roc::hostnumerics::Layout& layout,
                                                  bool separateStorage) {
                    const size_t bufferIndex
                        = separateStorage ? static_cast<size_t>(batchIdx)
                                          : static_cast<size_t>(gemmIdx);
                    return buffers.at(bufferIndex).tensor(
                        hipblaslt::hostnumerics::scalarType(type), layout);
                };
                const auto decodedMxTensor = [](const roc::hostnumerics::Tensor& tensor,
                                                hipblasOperation_t operation) {
                    if(operation == HIPBLAS_OP_N)
                        return tensor;
                    return tensor.shareStorageWithLayout(roc::hostnumerics::Layout(
                        roc::hostnumerics::Shape{tensor.shape()[1], tensor.shape()[0]},
                        {tensor.layout().stride(1), tensor.layout().stride(0)},
                        tensor.layout().offset()));
                };

                const auto aLayout = problem.a.logicalBatchLayout(
                    problem.operationA, static_cast<size_t>(batchIdx), pointerArrayMode);
                const auto bLayout = problem.b.logicalBatchLayout(
                    problem.operationB, static_cast<size_t>(batchIdx), pointerArrayMode);
                const auto cLayout = problem.c.logicalBatchLayout(
                    HIPBLAS_OP_N, static_cast<size_t>(batchIdx), pointerArrayMode);
                const auto dLayout = problem.d.logicalBatchLayout(
                    HIPBLAS_OP_N,
                    static_cast<size_t>(batchIdx),
                    pointerArrayMode && !preparedProblem.epilogueEnabled);

                roc::hostnumerics::Tensor referenceA
                    = isScaleAMXFormat
                          ? decodedMxTensor(refA.at(gemmIdx).at(batchIdx), problem.operationA)
                          : hostBufferTensor(hA, TiA, aLayout, pointerArrayMode);
                roc::hostnumerics::Tensor referenceB
                    = isScaleBMXFormat
                          ? decodedMxTensor(refB.at(gemmIdx).at(batchIdx), problem.operationB)
                          : hostBufferTensor(hB, TiB, bLayout, pointerArrayMode);
                roc::hostnumerics::Tensor referenceC
                    = hostBufferTensor(hC, TiC, cLayout, pointerArrayMode);
                roc::hostnumerics::Tensor referenceD
                    = preparedProblem.epilogueEnabled
                          ? hostBufferTensor(hD_gold_epl, Talpha, dLayout, false)
                          : hostBufferTensor(hD_gold, To, dLayout, pointerArrayMode);

                using roc::hostnumerics::ActivationApplication;
                using roc::hostnumerics::ClampActivation;
                using roc::hostnumerics::EpilogueOptions;
                using roc::hostnumerics::EpilogueOutputs;
                using roc::hostnumerics::GeluActivation;
                using roc::hostnumerics::IdentityActivation;
                using roc::hostnumerics::Layout;
                using roc::hostnumerics::MatmulOptions;
                using roc::hostnumerics::OutputConversion;
                using roc::hostnumerics::ReluActivation;
                using roc::hostnumerics::ScalarType;
                using roc::hostnumerics::Shape;
                using roc::hostnumerics::SiluActivation;
                using roc::hostnumerics::Tensor;
                using roc::hostnumerics::matmulIntoWithBlasBackend;
                using roc::hostnumerics::multiply;
                using roc::hostnumerics::referenceEpilogueInto;
                using roc::hostnumerics::referenceSumInto;

                const ScalarType computeTypeA
                    = isScaleAMXFormat
                          ? referenceA.type()
                          : hipblaslt::hostnumerics::referenceComputeType(dataTypes.computeInputA);
                const ScalarType computeTypeB
                    = isScaleBMXFormat
                          ? referenceB.type()
                          : hipblaslt::hostnumerics::referenceComputeType(dataTypes.computeInputB);
                const ScalarType accumulatorType
                    = hipblaslt::hostnumerics::referenceAccumulatorType(dataTypes.coefficient);
                MatmulOptions matmulOptions(accumulatorType);
                matmulOptions.conjugateA = problem.operationA == HIPBLAS_OP_C;
                matmulOptions.conjugateB = problem.operationB == HIPBLAS_OP_C;
                if(computeTypeA != referenceA.type())
                    matmulOptions.computeTypeA = computeTypeA;
                if(computeTypeB != referenceB.type())
                    matmulOptions.computeTypeB = computeTypeB;
                if(arg.scaleA == hipblaslt_scaling_format::Scalar
                   || arg.scaleA == hipblaslt_scaling_format::Vector)
                {
                    matmulOptions.preQuantizationScalesA.push_back(
                        hScaleA.at(gemmIdx)
                            .tensor(hipblaslt::hostnumerics::scalarType(Talpha),
                                    Layout::contiguousLastDimensionFastest(
                                        Shape{preparedProblem.a.scaleElements}))
                            .expandDims(1));
                }
                if(arg.scaleAlpha_vector)
                {
                    matmulOptions.preQuantizationScalesA.push_back(
                        hScaleAlphaVec.at(gemmIdx)
                            .tensor(hipblaslt::hostnumerics::scalarType(Talpha),
                                    Layout::contiguousLastDimensionFastest(
                                        Shape{preparedProblem.scaleAlphaElements}))
                            .expandDims(1));
                }
                if(arg.scaleB == hipblaslt_scaling_format::Scalar
                   || arg.scaleB == hipblaslt_scaling_format::Vector)
                {
                    matmulOptions.preQuantizationScalesB.push_back(
                        hScaleB.at(gemmIdx)
                            .tensor(hipblaslt::hostnumerics::scalarType(Talpha),
                                    Layout::contiguousLastDimensionFastest(
                                        Shape{preparedProblem.b.scaleElements}))
                            .expandDims(0));
                }

                const Tensor alpha = hipblaslt::hostnumerics::scalarValue(preparedProblem.alpha,
                                                                           dataTypes.coefficient);
                const Tensor beta  = hipblaslt::hostnumerics::scalarValue(preparedProblem.beta,
                                                                          dataTypes.coefficient);
                const Tensor scaleC
                    = arg.scaleC
                          ? hipblaslt::hostnumerics::realOnlyScalarValue(scaleCValue, Talpha)
                          : Tensor::scalar(accumulatorType, 1);
                const Tensor outputScale
                    = arg.scaleD && !preparedProblem.epilogueEnabled
                          ? hipblaslt::hostnumerics::scalarValue(scaleDValue, Talpha)
                          : Tensor::scalar(accumulatorType, 1);

                const std::complex<double> alphaValue = alpha.item<std::complex<double>>();
                const std::complex<double> betaValue  = beta.item<std::complex<double>>();
                const std::complex<double> outputScaleValue
                    = outputScale.item<std::complex<double>>();
                const bool hasProduct = alphaValue != std::complex<double>(0.0, 0.0)
                                        && referenceA.shape()[1] != 0;
                const bool productIsFinal
                    = hasProduct && alphaValue == std::complex<double>(1.0, 0.0)
                      && betaValue == std::complex<double>(0.0, 0.0)
                      && outputScaleValue == std::complex<double>(1.0, 0.0)
                      && referenceD.type() != ScalarType::Int8;
                if(productIsFinal)
                {
                    matmulIntoWithBlasBackend(referenceA, referenceB, referenceD, matmulOptions);
                }
                else
                {
                    const Shape referenceShape{referenceA.shape()[0], referenceB.shape()[1]};
                    const Layout referenceLayout
                        = referenceD.layout().stride(0) <= referenceD.layout().stride(1)
                              ? Layout::contiguousFirstDimensionFastest(referenceShape)
                              : Layout::contiguousLastDimensionFastest(referenceShape);
                    Tensor referenceAccumulator
                        = hasProduct
                              ? Tensor::allocateUninitialized(accumulatorType, referenceLayout)
                              : Tensor(accumulatorType, referenceLayout);
                    EpilogueOptions conversion(accumulatorType);
                    if(hasProduct)
                    {
                        matmulIntoWithBlasBackend(
                            referenceA, referenceB, referenceAccumulator, matmulOptions);
                        conversion.inputScale = alpha;
                    }
                    if(betaValue != std::complex<double>(0.0, 0.0))
                    {
                        conversion.addend = referenceC;
                        conversion.addendScale
                            = multiply(beta, scaleC, accumulatorType, accumulatorType);
                    }

                    conversion.outputScale = outputScale;
                    if(referenceD.type() == ScalarType::Int8)
                        conversion.outputConversion = OutputConversion::SaturatingInt8;
                    referenceEpilogueInto(referenceAccumulator, {.output = referenceD}, conversion);
                }

                if(preparedProblem.epilogueEnabled)
                {
                    EpilogueOutputs epilogueOutputs{
                        .output = hostBufferTensor(hD_gold, To, dLayout, pointerArrayMode),
                        .rawOutput
                        = hostBufferTensor(hBias_gold_epl, Talpha, dLayout, pointerArrayMode),
                    };
                    EpilogueOptions epilogueOptions(hipblaslt::hostnumerics::scalarType(Talpha));
                    epilogueOptions.outputScale
                        = hipblaslt::hostnumerics::scalarValue(scaleDValue, Talpha);
                    epilogueOptions.auxiliaryScale
                        = hipblaslt::hostnumerics::scalarValue(scaleEValue, Talpha);

                    if(arg.amaxD)
                    {
                        epilogueOutputs.amax = hAmaxD_gold[gemmIdx].tensor(
                            hipblaslt::hostnumerics::scalarType(Talpha),
                            Layout::contiguousLastDimensionFastest(Shape{1}));
                        epilogueOptions.accumulateAmax = true;
                    }

                    std::vector<HipHostBuffer>& auxiliaryBuffers = arg.gradient ? hE : hE_gold;
                    if(auxiliaryBuffers.size() > static_cast<size_t>(gemmIdx))
                    {
                        Tensor auxiliary
                            = hostBufferTensor(auxiliaryBuffers, Taux, dLayout, pointerArrayMode);
                        if(arg.gradient)
                            epilogueOptions.auxiliaryInput = auxiliary;
                        else
                            epilogueOutputs.auxiliaryOutput = auxiliary;
                    }

                    const bool applyBias = !arg.gradient && arg.bias_vector;
                    if(applyBias)
                    {
                        const ptrdiff_t biasOffset
                            = arg.bias_stride > 0 ? arg.bias_stride * batchIdx : 0;
                        epilogueOptions.bias
                            = hBias[gemmIdx]
                                  .tensor(hipblaslt::hostnumerics::scalarType(Tbias),
                                          Layout(Shape{static_cast<size_t>(problem.m)},
                                                 {1},
                                                 biasOffset))
                                  .expandDims(1);
                    }

                    switch(arg.activation_type)
                    {
                    case hipblaslt_activation_type::gelu:
                        epilogueOptions.activation = GeluActivation{};
                        break;
                    case hipblaslt_activation_type::relu:
                        epilogueOptions.activation = ReluActivation{};
                        break;
                    case hipblaslt_activation_type::swish:
                        // hipBLASLt's historical SWISH_EXT path implements
                        // SiLU and ignores the activation parameter.
                        epilogueOptions.activation = SiluActivation{};
                        break;
                    case hipblaslt_activation_type::clamp:
                        epilogueOptions.activation
                            = ClampActivation{arg.activation_arg1, arg.activation_arg2};
                        break;
                    case hipblaslt_activation_type::none:
                    case hipblaslt_activation_type::sigmoid:
                        epilogueOptions.activation = IdentityActivation{};
                        break;
                    default:
                        throw std::invalid_argument("Invalid hipBLASLt activation type.");
                    }
                    epilogueOptions.activationApplication
                        = arg.gradient && arg.activation_type != hipblaslt_activation_type::none
                              ? ActivationApplication::Gradient
                              : ActivationApplication::Forward;
                    if(hipblaslt::hostnumerics::scalarType(To) == ScalarType::Int8)
                        epilogueOptions.outputConversion = OutputConversion::SaturatingInt8;
                    referenceEpilogueInto(referenceD, epilogueOutputs, epilogueOptions);

                    if(arg.gradient && arg.bias_vector && batchIdx == problem.batchCount - 1)
                    {
                        const ptrdiff_t biasOffset
                            = arg.bias_stride > 0 ? arg.bias_stride * batchIdx : 0;
                        const size_t biasElements = static_cast<size_t>(
                            arg.bias_source == hipblaslt_bias_source::b ? problem.n : problem.m);
                        Tensor biasOutput = hBias_gold[gemmIdx].tensor(
                            hipblaslt::hostnumerics::scalarType(Tbias),
                            Layout(Shape{biasElements}, {1}, biasOffset));
                        const auto reduceBias = [&](const Tensor& input) {
                            referenceSumInto(input, biasOutput, {1}, ScalarType::Float32);
                        };

                        if(arg.bias_source == hipblaslt_bias_source::d)
                            reduceBias(*epilogueOutputs.rawOutput);
                        else if(arg.bias_source == hipblaslt_bias_source::a)
                            reduceBias(referenceA);
                        else if(arg.bias_source == hipblaslt_bias_source::b)
                        {
                            const Layout transposedB(
                                Shape{static_cast<size_t>(preparedProblem.biasElements),
                                      static_cast<size_t>(problem.k)},
                                {referenceB.layout().stride(1), referenceB.layout().stride(0)},
                                referenceB.layout().offset());
                            reduceBias(referenceB.shareStorageWithLayout(transposedB));
                        }
                    }
                }
            }
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }
    }
    void* alpha_ptr = nullptr;
    void* beta_ptr  = nullptr;

    if(problem_count > 0)
    {
        if(TiA == HIP_C_32F || TiA == HIP_C_64F)
        {
            if(TiA == HIP_C_32F)
            {
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(firstPreparedProblem.alpha.cf);
                beta_ptr  = (void*)&(firstPreparedProblem.beta.cf);
            }
            else if(TiA == HIP_C_64F)
            {

                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(firstPreparedProblem.alpha.cd);
                beta_ptr  = (void*)&(firstPreparedProblem.beta.cd);
            }
        }
        else
        {
            switch(Tc)
            {
            case HIP_R_32F:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(firstPreparedProblem.alpha.f32);
                beta_ptr  = (void*)&(firstPreparedProblem.beta.f32);
                break;
            case HIP_R_64F:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(firstPreparedProblem.alpha.f64);
                beta_ptr  = (void*)&(firstPreparedProblem.beta.f64);
                break;
            case HIP_R_16F:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(firstPreparedProblem.alpha.f16);
                beta_ptr  = (void*)&(firstPreparedProblem.beta.f16);
                break;
            case HIP_R_32I:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(firstPreparedProblem.alpha.i32);
                beta_ptr  = (void*)&(firstPreparedProblem.beta.i32);
                break;
            default:
                throw std::invalid_argument(
                    "Unsupported coefficient type in hipblasLtMatmul pointer setup.");
            }
        }
    }

    auto readValidationSideOutputs = [&] {
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
            return;

        for(int gemmIdx = 0; gemmIdx < problem_count; ++gemmIdx)
        {
            if(!arg.gradient && arg.use_e)
            {
                CHECK_HIP_ERROR(
                    synchronize(hE[gemmIdx], dE[gemmIdx], 0, 0, 0, 0, 1, false, stream));
            }
            if(arg.amaxD)
            {
                CHECK_HIP_ERROR(
                    synchronize(hAmaxD[gemmIdx], dAmaxD[gemmIdx], 0, 0, 0, 0, 1, false, stream));
            }
            if(arg.gradient && arg.bias_vector)
            {
                CHECK_HIP_ERROR(
                    synchronize(hBias[gemmIdx], dBias[gemmIdx], 0, 0, 0, 0, 1, false, stream));
            }
        }
    };

    const hipblaslt::hostnumerics::MatmulValidationOptions validationOptions{
        .compareAllClose = bool(arg.unit_check),
        .compareNorm     = bool(arg.norm_check),
        .searchAllClose  = bool(arg.allclose_check),
        .computeUlp      = bool(arg.ulp_check),
        .assertNorm      = arg.norm_check_assert,
        .computeType     = arg.compute_type,
        .inputTypeA      = hipblaslt::hostnumerics::scalarType(arg.a_type),
        .inputTypeB      = hipblaslt::hostnumerics::scalarType(arg.b_type),
    };

    auto makeValidationCases =
        [&](const std::vector<roc::hostnumerics::ComparisonTolerance>& allCloseTolerances) {
            using hipblaslt::hostnumerics::MatmulValidationCase;
            using roc::hostnumerics::Layout;
            using roc::hostnumerics::Shape;

            auto output = [](const HipHostBuffer& expected,
                             const HipHostBuffer& observed,
                             hipDataType          type,
                             const Layout&        layout) {
                const auto scalar = hipblaslt::hostnumerics::scalarType(type);
                return MatmulValidationCase::TensorPair{expected.tensor(scalar, layout),
                                                        observed.tensor(scalar, layout)};
            };

            std::vector<MatmulValidationCase> validationCases;
            if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
            {
                MatmulValidationCase validationCase;
                validationCase.allCloseTolerance = allCloseTolerances.front();
                validationCase.outputs.reserve(matmulProblems.front().batchCount);
                for(int batch = 0; batch < matmulProblems.front().batchCount; ++batch)
                {
                    const auto& matrix = matmulProblems.front().d;
                    validationCase.outputs.push_back(
                        output(hD_gold[batch],
                               hD_1[batch],
                               To,
                               Layout(Shape{static_cast<size_t>(matrix.rows()),
                                            static_cast<size_t>(matrix.columns())},
                                      {matrix.layout.stride(0), matrix.layout.stride(1)})));
                }
                validationCases.push_back(std::move(validationCase));
                return validationCases;
            }

            validationCases.reserve(problem_count);
            for(int gemmIdx = 0; gemmIdx < problem_count; ++gemmIdx)
            {
                const auto&          normalizedProblem = matmulProblems[gemmIdx];
                const auto&          preparedProblem   = preparedProblems[gemmIdx];
                MatmulValidationCase validationCase;
                validationCase.allCloseTolerance = allCloseTolerances[gemmIdx];
                validationCase.outputs.push_back(
                    output(hD_gold[gemmIdx], hD_1[gemmIdx], To, normalizedProblem.d.layout));
                if(arg.amaxD)
                {
                    auto request = output(hAmaxD_gold[gemmIdx],
                                          hAmaxD[gemmIdx],
                                          Talpha,
                                          Layout::contiguousLastDimensionFastest(Shape{1}));
                    validationCase.maximum
                        = MatmulValidationCase::SideOutput{request, request, false};
                }
                if(!arg.gradient && arg.use_e)
                {
                    const auto& auxiliary = *normalizedProblem.auxiliary;
                    auto request = output(hE_gold[gemmIdx], hE[gemmIdx], Taux, auxiliary.layout);
                    validationCase.auxiliary
                        = MatmulValidationCase::SideOutput{request, request, true};
                }
                if(arg.gradient && arg.bias_vector)
                {
                    const auto   selected     = output(hBias_gold[gemmIdx],
                                                 hBias[gemmIdx],
                                                 Tbias,
                                                 Layout::contiguousLastDimensionFastest(
                                                     Shape{preparedProblem.biasElements}));
                    const size_t biasElements = static_cast<size_t>(
                        arg.bias_source == hipblaslt_bias_source::b ? normalizedProblem.n
                                                                    : normalizedProblem.m);
                    const auto norm
                        = output(hBias_gold[gemmIdx],
                                 hBias[gemmIdx],
                                 Tbias,
                                 Layout::contiguousLastDimensionFastest(Shape{biasElements}));
                    validationCase.bias = MatmulValidationCase::SideOutput{selected, norm, false};
                }
                validationCases.push_back(std::move(validationCase));
            }
            return validationCases;
        };

    auto validateOutputs = [&] {
        const auto allCloseTolerances
            = matmulValidationTolerances(arg, matmulProblems, TiA, TiB, To, Tc);
        readValidationSideOutputs();
        const auto validationCases = makeValidationCases(allCloseTolerances);
        return hipblaslt::hostnumerics::validateMatmulOutputs(validationOptions, validationCases);
    };

    if(!arg.timing)
    {
        // fast_check launches and checks each solution fast_check_repeat times, solution by
        // solution, so a defect that fails only some launches is reported with its iteration.
        const int            fcIterations = arg.fast_check ? std::max(1, arg.fast_check_repeat) : 1;
        const bool           fcInjecting  = arg.fast_check && arg.fast_check_inject >= 0;
        FastCheckSolutionLog fcLog;
        // The problem fast_check verifies for GEMM i, with D at d_dev.
        auto fcProblem = [&](int i, const FastCheckMatrix& d_dev) {
            FastCheckProblem fp;
            fp.M           = matmulProblems[i].m;
            fp.N           = matmulProblems[i].n;
            fp.K           = matmulProblems[i].k;
            fp.batch_count = matmulProblems[i].batchCount;
            fp.transA      = transA != HIPBLAS_OP_N;
            fp.transB      = transB != HIPBLAS_OP_N;
            fp.A = {fcA[i].get(), fcTypeA[i], matmulProblems[i].a.rows(), matmulProblems[i].a.columns(), matmulProblems[i].a.rows(), matmulProblems[i].a.rows() * matmulProblems[i].a.columns()};
            fp.B = {fcB[i].get(), fcTypeB[i], matmulProblems[i].b.rows(), matmulProblems[i].b.columns(), matmulProblems[i].b.rows(), matmulProblems[i].b.rows() * matmulProblems[i].b.columns()};
            fp.C            = {fcC[i].get(),
                               TiC,
                               matmulProblems[i].m,
                               matmulProblems[i].n,
                               matmulProblems[i].m,
                               matmulProblems[i].m * matmulProblems[i].n};
            fp.D           = d_dev;
            fp.compute_type = Tc;
            fp.alpha        = compute_type_value_as_double(preparedProblems[i].alpha, Tc);
            fp.beta         = compute_type_value_as_double(preparedProblems[i].beta, Tc);
            if(arg.scaleAlpha_vector)
            {
                fp.scale_alpha_vec      = hScaleAlphaVec[i].buf();
                fp.scale_alpha_vec_type = Talpha;
            }
            if(arg.bias_vector && !arg.gradient)
            {
                fp.bias        = hBias[i].buf();
                fp.bias_type   = Tbias;
                fp.bias_stride = arg.bias_stride > 0 ? arg.bias_stride : 0;
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                fp.scale_a        = hScaleA[i].buf();
                fp.scale_a_vector = arg.scaleA == hipblaslt_scaling_format::Vector;
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                fp.scale_b        = hScaleB[i].buf();
                fp.scale_b_vector = arg.scaleB == hipblaslt_scaling_format::Vector;
            }
            fp.scale_ab_type = Talpha;
            if(arg.scaleC)
                fp.scale_c = fast_check_load(hScaleC[i].buf(), Talpha, 0);
            if(arg.scaleD)
                fp.scale_d = fast_check_load(hScaleD[i].buf(), Talpha, 0);
            fp.seed = fast_check_seed(arg);
            return fp;
        };
        // E is the pre-activation result times scaleE: the same check, with scaleE for scaleD.
        const bool fcCheckE = arg.fast_check && arg.use_e && !arg.gradient;
        const auto fcActivation
            = arg.activation_type == hipblaslt_activation_type::relu    ? FastCheckActivation::relu
              : arg.activation_type == hipblaslt_activation_type::clamp ? FastCheckActivation::clamp
                                                                        : FastCheckActivation::none;
        auto fcProblemE = [&](int i) {
            FastCheckProblem fp
                = fcProblem(i, {dE[i].buf(), Taux, matmulProblems[i].m, matmulProblems[i].n, (matmulProblems[i].auxiliary ? matmulProblems[i].auxiliary->leadingDimension() : 0), (matmulProblems[i].auxiliary ? matmulProblems[i].auxiliary->batchStride() : 0)});
            fp.scale_d = arg.scaleE ? fast_check_load(hScaleE[i].buf(), Talpha, 0) : 1;
            return fp;
        };
        std::vector<FastCheckExpected> fcExpectedE(fcCheckE ? problem_count : 0);
        // The expected sums depend only on the inputs. Computing them before any launch also
        // refuses a configuration whose results the compute type cannot hold exactly.
        if(arg.fast_check)
            for(int i = 0; i < problem_count; i++)
            {
                fcExpected[i] = fast_check_expected(
                    fcProblem(i, {dOutput[i].buf(), To, matmulProblems[i].m, matmulProblems[i].n, matmulProblems[i].d.leadingDimension(), matmulProblems[i].d.batchStride()}));
                if(fcCheckE && fcExpected[i].status.passed)
                    fcExpectedE[i] = fast_check_expected(fcProblemE(i));
                const FastCheckResult& status = fcCheckE && fcExpected[i].status.passed
                                                    ? fcExpectedE[i].status
                                                    : fcExpected[i].status;
                if(!status.passed)
                {
#ifdef GOOGLE_TEST
                    FAIL() << status.message;
#else
                    hipblaslt_cerr << status.message << std::endl;
                    return;
#endif
                }
            }
        for(size_t run = 0; run < heuristicResult.size() * fcIterations; run++)
        {
            const size_t sol  = run / fcIterations;
            const int    iter = int(run % fcIterations);
#ifdef GOOGLE_TEST
            SCOPED_TRACE(solution_description(
                             handle, heuristicResult[sol].algo, sol, heuristicResult.size(), false)
                         + (fcIterations > 1 ? ", iteration " + std::to_string(iter) : ""));
#endif
            if(placedWorkspace)
            {
                workspacePtr
                    = placedWorkspace->straddle(heuristicResult[sol].workspaceSize, workspace_size);
                workspaceBytes = heuristicResult[sol].workspaceSize;
            }
            if(arg.fast_check && !firstProblem.cEqualsD)
            {
                for(int i = 0; i < problem_count; i++)
                    CHECK_HIP_ERROR(
                        fast_check_fill_sentinel_device(dOutput[i].buf(), To, matmulProblems[i].d.allocationElements, stream));
            }
            if(fcCheckE)
                for(int i = 0; i < problem_count; i++)
                    CHECK_HIP_ERROR(
                        fast_check_fill_sentinel_device(dE[i].buf(), Taux, (matmulProblems[i].auxiliary ? matmulProblems[i].auxiliary->allocationElements : 0), stream));
            // A bias gradient is an output, so a store the kernel misses must not find the
            // previous launch's values there.
            if(arg.fast_check && arg.gradient && arg.bias_vector)
                for(int i = 0; i < problem_count; i++)
                    CHECK_HIP_ERROR(fast_check_fill_sentinel_device(
                        dBias[i].buf(), Tbias, preparedProblems[i].biasElements, stream));
            // amaxD too: all ones reads back as NaN (or -1 in int32), which no kernel's amaxD
            // can equal, so a skipped store fails the check.
            if(arg.fast_check && arg.amaxD)
                for(int i = 0; i < problem_count; i++)
                    CHECK_HIP_ERROR(
                        hipMemsetAsync(dAmaxD[i].buf(), 0xFF, realDataTypeSize(Talpha), stream));
            // Return the workspace to the zeros of a fresh allocation, so that a launch cannot
            // pass on the partial sums an earlier launch left there (a Stream-K fixup that reads
            // too early, for example). Zero rather than poison: kernels may rely on their flags
            // starting at zero.
            if(arg.fast_check && workspacePtr && workspace_size)
                CHECK_HIP_ERROR(hipMemsetAsync(workspacePtr, 0, workspace_size, stream));
            if((arg.unit_check || arg.norm_check || arg.allclose_check || arg.fast_check)
               && firstProblem.cEqualsD)
            {
                if(batchMode
                   == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) // Iterate for batch_count for General Batched GEMM
                {
                    for(int i = 0; i < arg.batch_count; i++)
                    {
                        CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                    }
                }
                else
                {
                    for(int i = 0; i < problem_count; i++)
                    {
                        const auto& problem = matmulProblems[i];
                        if(compactHostInputs)
                            CHECK_HIP_ERROR(
                                fast_check_poison_padding_device({dC[i].buf(),
                                                                  TiC,
                                                                  problem.m,
                                                                  problem.n,
                                                                  problem.c.leadingDimension(),
                                                                  problem.c.batchStride()},
                                                                 problem.batchCount,
                                                                 problem.c.allocationElements,
                                                                 stream));
                        CHECK_HIP_ERROR(copyHostInputToDevice(dC[i], hC[i], problem.c));
                    }
                }
            }
            if(!do_grouped_gemm)
            {
                if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    gemmVec[0].setMaxWorkspaceBytes(workspace_size);
                    CHECK_HIPBLASLT_ERROR(
                        gemmVec[0].initialize(heuristicResult[sol].algo,
                                              tuningVec[heuristicTuningIndex[sol]],
                                              workspacePtr));
                    CHECK_HIPBLASLT_ERROR(gemmVec[0].run(stream));
                }
                else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    // Added this logic to mimic the rocblas test quick_gemm_batched_bad_arg_f32_r_bad_arg_F
                    // This rocblas test passes alpha, A and B as 0 but beta as non-zero with valid C and D
                    // To mimic this behavior if --sizek is passed as 0 in hipblaslt-bench for
                    // --batch_mode 1, the prepared A and B element counts are zero since A is MxK
                    // and B is KxN. In this case, we pass the pointer arrays A and B as nullptr.
                    // General batched GEMM as nullptr and introduced an explicit check for AddressA and AddressB != 0
                    // in KernelWriterAssembly.py since the dereference of AddressA and AddressB for
                    // General Batched GEMM happens before the alphaNonZero check.
                    void* ptrA = firstPreparedProblem.a.elements ? dda[0] : nullptr;
                    void* ptrB = firstPreparedProblem.b.elements ? ddb[0] : nullptr;
                    EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                          firstRuntimeProblem.matmulDescriptors.front(),
                                                          firstRuntimeProblem.alphaPointer,
                                                          ptrA,
                                                          firstRuntimeProblem.matrixA,
                                                          ptrB,
                                                          firstRuntimeProblem.matrixB,
                                                          &(firstPreparedProblem.beta),
                                                          ddc[0],
                                                          firstRuntimeProblem.matrixC,
                                                          ddd[0],
                                                          firstRuntimeProblem.matrixD,
                                                          &heuristicResult[sol].algo,
                                                          workspacePtr,
                                                          workspace_size,
                                                          stream),
                                          HIPBLAS_STATUS_SUCCESS);
                }
                else
                {
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                          firstRuntimeProblem.matmulDescriptors.front(),
                                                          alpha_ptr,
                                                          dA[0].buf(),
                                                          firstRuntimeProblem.matrixA,
                                                          dB[0].buf(),
                                                          firstRuntimeProblem.matrixB,
                                                          beta_ptr,
                                                          dC[0].buf(),
                                                          firstRuntimeProblem.matrixC,
                                                          dOutput[0].buf(),
                                                          firstRuntimeProblem.matrixD,
                                                          &heuristicResult[sol].algo,
                                                          workspacePtr,
                                                          workspace_size,
                                                          stream),
                                          HIPBLAS_STATUS_SUCCESS);
                }
            }
            else
            {
                //grouped gemm
                if(arg.use_user_args)
                {
                    groupedGemmVec[0].setMaxWorkspaceBytes(workspace_size);
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemmVec[0].initialize(heuristicResult[sol].algo,
                                                     tuningVec[heuristicTuningIndex[0]],
                                                     workspacePtr));
                    groupedGemmVec[0].getDefaultValueForDeviceUserArguments(userArgs);
                    // Copy them to device memory
                    CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                                              userArgs,
                                              problem_count * sizeof(hipblaslt_ext::UserArguments),
                                              hipMemcpyHostToDevice));

                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(d_userArgs, stream));
                }
                else
                {
                    groupedGemmVec[0].setMaxWorkspaceBytes(workspace_size);
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemmVec[0].initialize(heuristicResult[sol].algo,
                                                     tuningVec[heuristicTuningIndex[0]],
                                                     workspacePtr,
                                                     false,
                                                     stream));

                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(stream));
                }
            }

            double hipblaslt_error   = 0.0;
            double hipblaslt_atol    = 1;
            double hipblaslt_rtol    = 1;
            double hipblaslt_max_ulp = 0.0;
            double hipblaslt_avg_ulp = 0.0;

            if(arg.unit_check || arg.norm_check || arg.allclose_check)
            {
                if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    copy_gemm_to_host(stream, arg.batch_count, hD_1, dOutput);
                }
                else
                {
                    copy_gemm_to_host(stream, problem_count, hD_1, dOutput);
                }
                const auto validation = validateOutputs();
                CHECK_SUCCESS(validation.passed);
                hipblaslt_error   = validation.relativeFrobeniusError;
                hipblaslt_max_ulp = validation.maximumUlp;
                hipblaslt_avg_ulp = validation.averageUlp;
                if(validationOptions.searchAllClose)
                {
                    hipblaslt_atol = validation.absoluteTolerance;
                    hipblaslt_rtol = validation.relativeTolerance;
                }
            }
            if(arg.fast_check)
            {
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));

                // The self-test corrupts one element of the last batch of D after the launch,
                // so only this iteration's result is wrong.
                const int64_t injRow = matmulProblems[0].m / 2, injCol = matmulProblems[0].n / 2, injBatch = matmulProblems[0].batchCount - 1;
                if(iter == arg.fast_check_inject)
                    CHECK_HIP_ERROR(fast_check_corrupt_element_device(
                        {dOutput[0].buf(), To, matmulProblems[0].m, matmulProblems[0].n, matmulProblems[0].d.leadingDimension(), matmulProblems[0].d.batchStride()},
                        injBatch,
                        injRow,
                        injCol,
                        stream));

                // Failures are reported as they are found, except in the self-test, which
                // collects them and checks that they fall on the corrupted iteration.
                bool        iterPassed = true;
                std::string iterReport;
                auto        reportFailure = [&](const std::string& what, const std::string& text) {
                    iterPassed = false;
                    const std::string report
                        = what + ", "
                          + solution_description(
                              handle, heuristicResult[sol].algo, sol, heuristicResult.size(), true)
                          + (fcIterations > 1 ? ", iteration " + std::to_string(iter) + " of "
                                                    + std::to_string(fcIterations)
                                                     : "")
                          + ":\n" + text;
                    iterReport += report;
                    if(fcInjecting)
                        return;
#ifdef GOOGLE_TEST
                    ADD_FAILURE() << report;
#else
                    hipblaslt_cerr << report << std::endl;
#endif
                };

                for(int i = 0; i < problem_count; i++)
                {
                    FastCheckMatrix d_dev{dOutput[i].buf(), To, matmulProblems[i].m, matmulProblems[i].n, matmulProblems[i].d.leadingDimension(), matmulProblems[i].d.batchStride()};
                    FastCheckResult scan = fast_check_scan_padding_device(
                        d_dev, matmulProblems[i].batchCount, matmulProblems[i].d.allocationElements, arg.c_equal_d, stream);

                    FastCheckProblem fp = fcProblem(i, d_dev);
                    // With an activation D is not linear in the inputs; it is checked through E.
                    FastCheckResult res = fcActivation == FastCheckActivation::none
                                              ? fast_check_result_device(fp, fcExpected[i], stream)
                                              : FastCheckResult{};
                    if(fcCheckE)
                    {
                        FastCheckProblem fpE   = fcProblemE(i);
                        FastCheckResult  scanE = fast_check_scan_padding_device(
                            fpE.D, matmulProblems[i].batchCount, (matmulProblems[i].auxiliary ? matmulProblems[i].auxiliary->allocationElements : 0), false, stream);
                        FastCheckResult resE
                            = fast_check_result_device(fpE, fcExpectedE[i], stream);
                        if(!scanE.passed || !resE.passed)
                            reportFailure("fast_check E", scanE.message + resE.message);
                        else if(fcActivation != FastCheckActivation::none)
                        {
                            res = fast_check_activation_device(d_dev,
                                                               fpE.D,
                                                               matmulProblems[i].batchCount,
                                                               fp.scale_d,
                                                               fpE.scale_d,
                                                               fcActivation,
                                                               arg.activation_arg1,
                                                               arg.activation_arg2,
                                                               stream,
                                                               nullptr);
                        }
                    }
                    if(arg.amaxD && scan.passed && res.passed)
                    {
                        double want = 0;
                        if(fcActivation != FastCheckActivation::none)
                            fast_check_activation_device(d_dev,
                                                         fcProblemE(i).D,
                                                         matmulProblems[i].batchCount,
                                                         fp.scale_d,
                                                         fcProblemE(i).scale_d,
                                                         fcActivation,
                                                         arg.activation_arg1,
                                                         arg.activation_arg2,
                                                         stream,
                                                         &want);
                        else
                            want
                                = fast_check_amax_device(d_dev, matmulProblems[i].batchCount, fp.scale_d, stream);
                        double got = 0;
                        CHECK_HIP_ERROR(
                            synchronize(hAmaxD[i], dAmaxD[i], 0, 0, 0, 0, 1, false, stream));
                        got = fast_check_load(hAmaxD[i].buf(), Talpha, 0);
                        if(std::isnan(want))
                            reportFailure("fast_check amaxD",
                                          "amaxD cannot be checked: the verified D or E holds "
                                          "values outside the range its type stores exactly, "
                                          "or is scaled by 0\n");
                        else if(!(got == want))
                            reportFailure("fast_check amaxD",
                                          "amaxD holds " + std::to_string(got)
                                              + ", the verified D gives " + std::to_string(want)
                                              + "\n");
                    }
                    if(arg.gradient && arg.bias_vector)
                    {
                        std::vector<char> hb(preparedProblems[i].biasElements * realDataTypeSize(Tbias));
                        CHECK_HIP_ERROR(
                            hipMemcpy(hb.data(), dBias[i].buf(), hb.size(), hipMemcpyDeviceToHost));
                        const char source = arg.bias_source == hipblaslt_bias_source::a ? 'a' : 'b';
                        FastCheckResult grad
                            = fast_check_bias_gradient(fp, source, hb.data(), Tbias);
                        if(!grad.passed)
                            reportFailure("fast_check bias gradient", grad.message);
                    }
                    if(!scan.passed || !res.passed)
                    {
                        std::vector<FastCheckBuffer> buffers
                            = {{"A",
                                dA[i].buf(),
                                matmulProblems[i].a.allocationElements * realDataTypeSize(TiA)},
                               {"B",
                                dB[i].buf(),
                                matmulProblems[i].b.allocationElements * realDataTypeSize(TiB)},
                               {"C",
                                dC[i].buf(),
                                matmulProblems[i].c.allocationElements * realDataTypeSize(TiC)},
                               {"D",
                                dOutput[i].buf(),
                                matmulProblems[i].d.allocationElements * realDataTypeSize(To)},
                               {"workspace", workspacePtr, workspaceBytes}};
                        if(arg.bias_vector)
                            buffers.push_back(
                                {"bias", dBias[i].buf(), preparedProblems[i].biasElements * realDataTypeSize(Tbias)});
                        if(arg.scaleAlpha_vector)
                            buffers.push_back({"scaleAlpha_vector",
                                               dScaleAlphaVec[i].buf(),
                                               preparedProblems[i].scaleAlphaElements * realDataTypeSize(Talpha)});
                        // MX scales are one byte each, and either can be placed.
                        if(isBlockScaling(arg.scaleA))
                            buffers.push_back(
                                {"scale_a", dScaleA[i].buf(), preparedProblems[i].a.scaleElements * matmulProblems[i].batchCount});
                        if(isBlockScaling(arg.scaleB))
                            buffers.push_back(
                                {"scale_b", dScaleB[i].buf(), preparedProblems[i].b.scaleElements * matmulProblems[i].batchCount});
                        reportFailure("fast_check",
                                      scan.message + res.message
                                          + fast_check_describe_buffers(buffers));
                    }
                }

                // A write that missed a placed operand by exactly 4 GiB lands in its poison.
                const PlacedRegion* placed = placedWorkspace.get();
                for(auto* v : {&dA, &dB, &dC, &dD, &dBias, &dScaleAlphaVec, &dScaleA, &dScaleB})
                    if(!placed && !v->empty() && (*v)[0].placement())
                        placed = (*v)[0].placement();
                // A placement that silently fell back to a normal allocation would pass while
                // testing nothing.
                const bool straddles = placed
                                       && (placed != placedWorkspace.get() || workspaceBytes <= 256
                                           || placed->crosses(workspacePtr, workspaceBytes));
                if(*arg.placement && !straddles)
                {
                    const std::string report
                        = std::string("fast_check placement: ") + arg.placement
                          + " was to cross a 4 GiB boundary but "
                          + (placed ? "this solution's range does not" : "no buffer was placed");
#ifdef GOOGLE_TEST
                    ADD_FAILURE() << report;
#else
                    hipblaslt_cerr << report << std::endl;
#endif
                }
                if(placed)
                {
                    FastCheckResult poison = placed->verify_poison(arg.placement, stream);
                    if(!poison.passed)
                        reportFailure("fast_check placement", poison.message);
                    CHECK_HIP_ERROR(placed->fill_poison(stream));
                }

                fcLog.record(sol,
                             hipblaslt_ext::getIndexFromAlgo(heuristicResult[sol].algo),
                             iterPassed ? std::string()
                                        : hipblaslt_ext::getKernelNameFromAlgo(
                                              handle, heuristicResult[sol].algo),
                             iter,
                             iterPassed);

                if(fcInjecting)
                {
                    // The corrupted iteration must fail and name the corrupted element; every
                    // other iteration must pass.
                    const std::string where = "batch " + std::to_string(injBatch) + ", row "
                                              + std::to_string(injRow) + ", col "
                                              + std::to_string(injCol);
                    std::string selfTestError;
                    if(iter == arg.fast_check_inject && iterPassed)
                        selfTestError = "fast_check_inject: the corrupted iteration "
                                        + std::to_string(iter) + " passed";
                    else if(iter == arg.fast_check_inject
                            && iterReport.find(where) == std::string::npos)
                        selfTestError = "fast_check_inject: the report for iteration "
                                        + std::to_string(iter) + " does not name " + where + ":\n"
                                        + iterReport;
                    else if(iter != arg.fast_check_inject && !iterPassed)
                        selfTestError = "fast_check_inject: iteration " + std::to_string(iter)
                                        + " was not corrupted but failed:\n" + iterReport;
                    if(!selfTestError.empty())
                    {
#ifdef GOOGLE_TEST
                        ADD_FAILURE() << selfTestError;
#else
                        hipblaslt_cerr << selfTestError << std::endl;
#endif
                    }
                }
            }
        }

        // Which solutions failed, and on which iterations, in one place at the end.
        const std::string fcSummary = fcInjecting ? std::string() : fcLog.summary(fcIterations);
        if(!fcSummary.empty())
        {
#ifdef GOOGLE_TEST
            ADD_FAILURE() << fcSummary;
#else
            hipblaslt_cerr << fcSummary << std::endl;
#endif
        }
    }
    else
    {
        // Get device information
        hipDeviceProp_t deviceProps;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
        int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

        size_t      best_sol       = -1;
        double      best_flops     = 0.0;
        double      best_gpu_time  = std::numeric_limits<double>::max();
        double      best_warm_time = std::numeric_limits<double>::max();
        std::string best_s_name    = "";
        std::string best_k_name    = "";
        double      best_norm      = 0.0;
        double      best_atol      = 0.0;
        double      best_rtol      = 0.0;
        double      best_max_ulp   = 0.0;
        double      best_avg_ulp   = 0.0;
        int         number_cold_calls
            = ((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.cold_iters == 0)
                  ? 1
                  : arg.cold_iters;
        // Adaptive timing ignores --cold_iters/--iters: warmup and batch are sized
        // inside run_measurement. Keep one cold call only when a result copy or the
        // skip-slow screen needs it.
        if(arg.adaptive)
            number_cold_calls = (arg.unit_check || arg.norm_check || arg.allclose_check
                                 || arg.skip_slow_solution_ratio != 0.0f)
                                    ? 1
                                    : 0;
        int number_hot_calls = arg.iters;

        // Adaptive timing configuration shared by every solution measured below.
        // Adaptive timing is gated on --adaptive; the per-knob settings apply only then.
        hipblaslt_bench::TimingConfig timingCfg;
        timingCfg.iters         = number_hot_calls;
        timingCfg.use_gpu_timer = arg.use_gpu_timer;
        timingCfg.adaptive      = arg.adaptive;
        if(arg.adaptive)
        {
            timingCfg.warmup_time         = arg.warmup_time;
            timingCfg.sample_time         = arg.sample_time;
            timingCfg.measure_time        = arg.measure_time;
            timingCfg.max_measure_time    = arg.max_measure_time;
            timingCfg.min_iters           = arg.min_iters;
            timingCfg.max_iters           = arg.max_iters;
            timingCfg.noise_threshold     = arg.noise_threshold;
            timingCfg.stability_threshold = arg.stability_threshold;
            timingCfg.stability_window    = arg.stability_window;
            timingCfg.stability_interval  = arg.stability_interval;
        }
        if(arg.adaptive)
        {
            if(const auto err = hipblaslt_bench::validate_adaptive_config(timingCfg); !err.empty())
            {
                hipblaslt_cerr << "error: invalid adaptive timing config: " << err << std::endl;
                return;
            }
        }

        hipblaslt_bench::TimingResult timing;
        // Stop the sample loop if a launch hits a gtest fatal failure.
        auto timingAbort = []() -> bool {
#ifdef GOOGLE_TEST
            return ::testing::Test::HasFatalFailure();
#else
            return false;
#endif
        };
        hipblaslt_bench::TimingResult best_timing;

        int    flush_iter      = 100000;
        double flush_time_used = 0;
        if(arg.flush)
        {
            static std::unordered_map<std::string, double> flush_times_cache;
            static std::mutex                              mtx;
            std::lock_guard<std::mutex>                    lock(mtx);
            std::string                                    device_uuid(deviceProps.uuid.bytes);
            if(!flush_times_cache.count(device_uuid))
            {
                for(int i = 0; i < flush_iter; i++)
                    hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, flush_time_used, stream);
                for(int i = 0; i < flush_iter; i++)
                    hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                post_gpu_time(arg.use_gpu_timer,
                              event_gpu_time_start,
                              event_gpu_time_end,
                              flush_time_used,
                              stream);
                flush_time_used /= flush_iter;
                flush_times_cache[device_uuid] = flush_time_used;
            }
            else
            {
                flush_time_used = flush_times_cache[device_uuid];
            }
        }
        timingCfg.flush_us = flush_time_used;

        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            // Reset per-solution so an aborted/empty measurement can't report the prior
            // solution's stats (run_measurement leaves `out` untouched on early return).
            timing = {};
            if((arg.unit_check || arg.norm_check || arg.allclose_check) && firstProblem.cEqualsD)
            {
                if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    for(int i = 0; i < arg.batch_count; i++)
                    {
                        CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                    }
                }
                else
                {
                    for(int i = 0; i < problem_count; i++)
                    {
                        const auto& problem = matmulProblems[i];
                        if(compactHostInputs)
                            CHECK_HIP_ERROR(
                                fast_check_poison_padding_device({dC[i].buf(),
                                                                  TiC,
                                                                  problem.m,
                                                                  problem.n,
                                                                  problem.c.leadingDimension(),
                                                                  problem.c.batchStride()},
                                                                 problem.batchCount,
                                                                 problem.c.allocationElements,
                                                                 stream));
                        CHECK_HIP_ERROR(copyHostInputToDevice(dC[i], hC[i], problem.c));
                    }
                }
            }
            if(!do_grouped_gemm)
            {
                auto perf_monitor = EfficiencyMonitor::create();
                if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        gemmVec[b].setMaxWorkspaceBytes(workspace_size);
                        CHECK_HIPBLASLT_ERROR(
                            gemmVec[b].initialize(heuristicResult[sol].algo,
                                                  tuningVec[heuristicTuningIndex[sol]],
                                                  *dWorkspace));
                    }
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(gemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, dOutput);
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int b = static_cast<int>(i % block_count);
                            CHECK_HIPBLASLT_ERROR(gemmVec[b].run(stream));
                            if(arg.flush)
                                hipLaunchKernelGGL(
                                    flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                }
                else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        auto& ptr_matmul = firstRuntimeProblem.matmulDescriptors[i % block_count];
                        auto  ptr_alpha  = arg.scaleAlpha_vector
                                               ? rotatingBlockAddress(
                                                     dScaleAlphaVec[0], i % block_count)
                                               : firstRuntimeProblem.alphaPointer;
                        // Added this logic to mimic the rocblas test quick_gemm_batched_bad_arg_f32_r_bad_arg_F
                        // This rocblas test passes alpha, A and B as 0 but beta as non-zero with valid C and D
                        // To mimic this behavior if --sizek is passed as 0 in hipblaslt-bench for
                        // --batch_mode 1, the prepared A and B element counts are zero since A is
                        // MxK and B is KxN. In this case, we pass the pointer arrays A and B as
                        // nullptr.
                        // General batched GEMM as nullptr and introduced an explicit check for AddressA and AddressB != 0
                        // in KernelWriterAssembly.py since the dereference of AddressA and AddressB for
                        // General Batched GEMM happens before the alphaNonZero check.
                        void* ptrA
                            = firstPreparedProblem.a.elements ? dda[i % block_count] : nullptr;
                        void* ptrB
                            = firstPreparedProblem.b.elements ? ddb[i % block_count] : nullptr;
                        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                              ptr_matmul,
                                                              ptr_alpha,
                                                              ptrA,
                                                              firstRuntimeProblem.matrixA,
                                                              ptrB,
                                                              firstRuntimeProblem.matrixB,
                                                              &(firstPreparedProblem.beta),
                                                              ddc[i % block_count],
                                                              firstRuntimeProblem.matrixC,
                                                              ddd[i % block_count],
                                                              firstRuntimeProblem.matrixD,
                                                              &heuristicResult[sol].algo,
                                                              *dWorkspace,
                                                              workspace_size,
                                                              stream),
                                              HIPBLAS_STATUS_SUCCESS);
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, arg.batch_count, hD_1, dOutput);
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int   b          = static_cast<int>(i % block_count);
                            auto& ptr_matmul = firstRuntimeProblem.matmulDescriptors[b];
                            auto  ptr_alpha  = arg.scaleAlpha_vector
                                                   ? rotatingBlockAddress(dScaleAlphaVec[0], b)
                                                   : firstRuntimeProblem.alphaPointer;
                            void* ptrA       = firstPreparedProblem.a.elements ? dda[b] : nullptr;
                            void* ptrB       = firstPreparedProblem.b.elements ? ddb[b] : nullptr;
                            EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                                  ptr_matmul,
                                                                  ptr_alpha,
                                                                  ptrA,
                                                                  firstRuntimeProblem.matrixA,
                                                                  ptrB,
                                                                  firstRuntimeProblem.matrixB,
                                                                  &(firstPreparedProblem.beta),
                                                                  ddc[b],
                                                                  firstRuntimeProblem.matrixC,
                                                                  ddd[b],
                                                                  firstRuntimeProblem.matrixD,
                                                                  &heuristicResult[sol].algo,
                                                                  *dWorkspace,
                                                                  workspace_size,
                                                                  stream),
                                                  HIPBLAS_STATUS_SUCCESS);
                            if(arg.flush)
                                hipLaunchKernelGGL(
                                    flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                }
                else
                {
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        auto& ptr_matmul = firstRuntimeProblem.matmulDescriptors[i % block_count];
                        auto  ptr_alpha  = arg.scaleAlpha_vector
                                               ? rotatingBlockAddress(
                                                     dScaleAlphaVec[0], i % block_count)
                                               : firstRuntimeProblem.alphaPointer;

                        EXPECT_HIPBLAS_STATUS(
                            hipblasLtMatmul(
                                handle,
                                ptr_matmul,
                                ptr_alpha,
                                rotatingBlockAddress(dA[0], i % block_count),
                                firstRuntimeProblem.matrixA,
                                rotatingBlockAddress(dB[0], i % block_count),
                                firstRuntimeProblem.matrixB,
                                beta_ptr,
                                rotatingBlockAddress(dC[0], i % block_count),
                                firstRuntimeProblem.matrixC,
                                rotatingBlockAddress(dOutput[0], i % block_count),
                                firstRuntimeProblem.matrixD,
                                &heuristicResult[sol].algo,
                                *dWorkspace,
                                workspace_size,
                                stream),
                            HIPBLAS_STATUS_SUCCESS);
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, dOutput);
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int   b          = static_cast<int>(i % block_count);
                            auto& ptr_matmul = firstRuntimeProblem.matmulDescriptors[b];
                            auto  ptr_alpha  = arg.scaleAlpha_vector
                                                   ? rotatingBlockAddress(dScaleAlphaVec[0], b)
                                                   : firstRuntimeProblem.alphaPointer;
                            EXPECT_HIPBLAS_STATUS(
                                hipblasLtMatmul(handle,
                                                ptr_matmul,
                                                ptr_alpha,
                                                rotatingBlockAddress(dA[0], b),
                                                firstRuntimeProblem.matrixA,
                                                rotatingBlockAddress(dB[0], b),
                                                firstRuntimeProblem.matrixB,
                                                beta_ptr,
                                                rotatingBlockAddress(dC[0], b),
                                                firstRuntimeProblem.matrixC,
                                                rotatingBlockAddress(dOutput[0], b),
                                                firstRuntimeProblem.matrixD,
                                                &heuristicResult[sol].algo,
                                                *dWorkspace,
                                                workspace_size,
                                                stream),
                                HIPBLAS_STATUS_SUCCESS);
                            if(arg.flush)
                                hipLaunchKernelGGL(
                                    flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                }
                gpu_time_used = timing.median_us;
                perf_monitor->stop();
            }
            else
            {
                auto perf_monitor = EfficiencyMonitor::create();
                if(arg.use_user_args)
                {
                    std::vector<unsigned char*> d_userArgsVec(block_count);
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        groupedGemmVec[b].setMaxWorkspaceBytes(workspace_size);
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size)));
                        groupedGemmVec[b].getDefaultValueForDeviceUserArguments(userArgs);
                        d_userArgsVec[b]
                            = (unsigned char*)d_userArgs
                              + b * problem_count * sizeof(hipblaslt_ext::UserArguments);
                        // Copy them to device memory
                        CHECK_HIP_ERROR(
                            hipMemcpy(d_userArgsVec[b],
                                      userArgs,
                                      problem_count * sizeof(hipblaslt_ext::UserArguments),
                                      hipMemcpyHostToDevice));
                    }
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(
                            d_userArgsVec[i % block_count], stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, dOutput);
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int b = static_cast<int>(i % block_count);
                            CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].run(d_userArgsVec[b], stream));
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                    gpu_time_used = timing.median_us;
                    perf_monitor->stop();
                }
                else
                {
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        groupedGemmVec[b].setMaxWorkspaceBytes(workspace_size);
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size),
                            false,
                            stream));
                    }

                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, dOutput);
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int b = static_cast<int>(i % block_count);
                            CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].run(stream));
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                    gpu_time_used = timing.median_us;
                    perf_monitor->stop();
                }
            }

            double flops = 0;
            for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
            {
                const auto& problem = matmulProblems[gemmIdx];
                flops += gemm_gflop_count(problem.m, problem.n, problem.k, Talpha);
                switch(arg.activation_type)
                {
                case hipblaslt_activation_type::relu:
                    flops += relu_gflop_count(problem.m, problem.n, Talpha);
                    break;
                case hipblaslt_activation_type::gelu:
                    flops += gelu_gflop_count(problem.m, problem.n, Talpha);
                    break;
                case hipblaslt_activation_type::swish:
                    flops += silu_gflop_count(problem.m, problem.n, Talpha);
                    break;
                case hipblaslt_activation_type::clamp:
                    flops += clamp_gflop_count(problem.m, problem.n, Talpha);
                    break;
                case hipblaslt_activation_type::sigmoid:
                    flops += sigmoid_gflop_count(problem.m, problem.n, Talpha);
                    break;
                default:
                    break;
                }
            }

            double hipblaslt_error   = 0.0;
            double hipblaslt_atol    = 1;
            double hipblaslt_rtol    = 1;
            double hipblaslt_max_ulp = 0.0;
            double hipblaslt_avg_ulp = 0.0;
            if(arg.unit_check || arg.norm_check || arg.allclose_check)
            {
                if(arg.dump_matrix)
                {
                    for(int batchId = 0; batchId < firstProblem.batchCount; batchId++)
                    {
                        hipblasltDispatchValuesToFile(
                            HIPBLAS_OP_N,
                            To,
                            firstProblem.m,
                            firstProblem.n,
                            firstProblem.d.leadingDimension(),
                            hD_1[0].as<char>()
                                + batchId * firstProblem.d.batchStride() * realDataTypeSize(To),
                            "batch_" + std::to_string(batchId) + "_D_output.txt");
                        hipblasltDispatchValuesToFile(
                            HIPBLAS_OP_N,
                            To,
                            firstProblem.m,
                            firstProblem.n,
                            firstProblem.d.leadingDimension(),
                            hD_gold[0].as<char>()
                                + batchId * firstProblem.d.batchStride() * realDataTypeSize(To),
                            "batch_" + std::to_string(batchId) + "_D_Gold_output.txt");
                    }
                }
                const auto validation = validateOutputs();
                CHECK_SUCCESS(validation.passed);
                hipblaslt_error   = validation.relativeFrobeniusError;
                hipblaslt_max_ulp = validation.maximumUlp;
                hipblaslt_avg_ulp = validation.averageUlp;
                if(validationOptions.searchAllClose)
                {
                    hipblaslt_atol = validation.absoluteTolerance;
                    hipblaslt_rtol = validation.relativeTolerance;
                }
            }

#define argument_param                                                                            \
    e_transA, e_transB, e_grouped_gemm, e_batch_count, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a, \
        e_beta, e_ldb, e_stride_b, e_ldc, e_stride_c, e_ldd, e_stride_d, e_a_type, e_b_type,      \
        e_c_type, e_d_type, e_compute_type, e_scaleA, e_scaleB, e_scaleC, e_scaleD, e_amaxD,      \
        e_swizzle_a, e_swizzle_b, e_activation_type, e_bias_vector, e_bias_type, e_aux_type

            const char* tuningEnv     = getenv("HIPBLASLT_TUNING_FILE");
            int32_t     solutionIndex = ((tuningEnv && heuristicResult.size() == 1)
                                     || (arg.print_solution_found && arg.print_kernel_info))
                                            ? hipblaslt_ext::getIndexFromAlgo(heuristicResult[sol].algo)
                                            : -1;
            std::string solutionName  = "";
            std::string kernelName    = "";
            std::string archName      = "";
            std::string cuNum         = "";

            // The tuning file has no grouped GEMM key, so a grouped winner would
            // be read back as a single GEMM of the same shape.
            if(tuningEnv && !do_grouped_gemm && heuristicResult.size() == 1)
            {
                archName = deviceProps.gcnArchName;
                cuNum    = std::to_string(deviceProps.multiProcessorCount);
            }

            if(arg.print_solution_found || tuningEnv)
            {
                // A tuning run records the winner's kernel name, which comes
                // from best_k_name below.
                if(arg.print_kernel_info || tuningEnv)
                {
                    if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                    {
                        if(!do_grouped_gemm)
                        {
                            solutionName = gemmVec[0].getSolutionName();
                            kernelName   = gemmVec[0].getKernelName();
                        }
                        else
                        {
                            solutionName = groupedGemmVec[0].getSolutionName();
                            kernelName   = groupedGemmVec[0].getKernelName();
                        }

                        // The ext accessor joins every kernel the solution
                        // launches with "; ". Replay compares against the main
                        // kernel's name alone, which the algo accessor returns.
                        if(tuningEnv)
                            kernelName = hipblaslt_ext::getKernelNameFromAlgo(
                                handle, heuristicResult[sol].algo);
                    }
                    else
                    {
                        solutionName = hipblaslt_ext::getSolutionNameFromAlgo(
                            handle, heuristicResult[sol].algo);
                        kernelName = hipblaslt_ext::getKernelNameFromAlgo(
                            handle, heuristicResult[sol].algo);
                    }
                }
                ArgumentModel<argument_param>{}.log_args(
                    Talpha,
                    hipblaslt_cout,
                    sol,
                    solutionIndex,
                    solutionName,
                    kernelName,
                    archName,
                    cuNum,
                    arg,
                    (uint32_t)tuningVec[heuristicTuningIndex[sol]].getSplitK(),
                    (uint32_t)tuningVec[heuristicTuningIndex[sol]].getWgm(),
                    gpu_time_used,
                    flops,
                    gpu_mem_gbytes,
                    cpu_time_used,
                    hipblaslt_error,
                    hipblaslt_atol,
                    hipblaslt_rtol,
                    hipblaslt_max_ulp,
                    hipblaslt_avg_ulp,
                    timing);
            }
            if(best_gpu_time > gpu_time_used)
            {
                best_sol      = sol;
                best_flops    = flops;
                best_gpu_time = gpu_time_used;
                best_s_name   = solutionName;
                best_k_name   = kernelName;
                best_norm     = hipblaslt_error;
                best_atol     = hipblaslt_atol;
                best_rtol     = hipblaslt_rtol;
                best_max_ulp  = hipblaslt_max_ulp;
                best_avg_ulp  = hipblaslt_avg_ulp;
                best_timing   = timing;
            }
        }

        if(heuristicResult.size() > 1)
        {
            const char* tuningEnv = getenv("HIPBLASLT_TUNING_FILE");
            int32_t     solutionIndex
                = (tuningEnv || arg.print_kernel_info)
                      ? hipblaslt_ext::getIndexFromAlgo(heuristicResult[best_sol].algo)
                      : -1;
            std::string solutionName = "";
            std::string kernelName   = "";
            std::string archName     = "";
            std::string cuNum        = "";
            if(tuningEnv && !do_grouped_gemm)
            {
                archName = deviceProps.gcnArchName;
                cuNum    = std::to_string(deviceProps.multiProcessorCount);
            }

            // Same reason as the per-candidate loop above.
            if(arg.print_kernel_info || tuningEnv)
            {
                solutionName = best_s_name;
                kernelName   = best_k_name;
            }

            hipblaslt_cout << "Winner: " << std::endl;
            ArgumentModel<argument_param>{}.log_args(
                Talpha,
                hipblaslt_cout,
                best_sol,
                solutionIndex,
                solutionName,
                kernelName,
                archName,
                cuNum,
                arg,
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].getSplitK(),
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].getWgm(),
                best_gpu_time,
                best_flops,
                gpu_mem_gbytes,
                cpu_time_used,
                best_norm,
                best_atol,
                best_rtol,
                best_max_ulp,
                best_avg_ulp,
                best_timing);
        }
    }

    for(auto it : ptrs)
    {
        CHECK_HIP_ERROR(hipFree(it));
    }

    //Freeing the device memory allocated for the General Batched GEMM Pointer Arrays
    if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        for(int i = 0; i < block_count; i++)
        {
            CHECK_HIP_ERROR(hipFree(dda[i]));
            CHECK_HIP_ERROR(hipFree(ddb[i]));
            CHECK_HIP_ERROR(hipFree(ddc[i]));
            CHECK_HIP_ERROR(hipFree(ddd[i]));
        }
    }

    if(userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(userArgs));
    if(d_userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(d_userArgs));

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_end));
}
