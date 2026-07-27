// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "reference_device.hpp"
#include "hipblaslt_ostream.hpp"
#include <type_traits>

// Compiled through hipcc/clang (the client common library links hip::device,
// which injects `-x hip`), so the __global__ kernel below builds as device code.

namespace
{
    // Log and return false on HIP error rather than silently reporting a pass.
    inline bool gpu_ref_hip_check(hipError_t err, const char* what)
    {
        if(err != hipSuccess)
        {
            hipblaslt_cerr << "gpu_ref: " << what << " failed: " << hipGetErrorString(err)
                           << std::endl;
            return false;
        }
        return true;
    }

    // Load one input element as float, decoding by runtime type. A single
    // per-element path lets A and B carry the same float-class input type.
    // `idx` is the logical element index into the (batch-offset) matrix. An
    // unhandled type returns NaN so a mis-gated type fails loud (the NaN
    // propagates to the output and the compare kernel flags it).
    __device__ inline float load_input_f32(const void* base, int64_t idx, hipDataType t)
    {
        switch(t)
        {
        case HIP_R_32F:
            return static_cast<const float*>(base)[idx];
        case HIP_R_16BF:
            return float(static_cast<const hip_bfloat16*>(base)[idx]);
        case HIP_R_16F:
            return float(static_cast<const hipblasLtHalf*>(base)[idx]);
        default:
            return nanf("");
        }
    }

    // Input/output types accepted by the reference path: f32/f16/bf16 only.
    bool is_supported_type(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F || t == HIP_R_16BF;
    }

    // float-accumulate reference GEMM with runtime-typed A/B loads. One thread per
    // output element; C/D remain templated. Real inputs only (no conjugation).
    template <typename Tc, typename To>
    __global__ void reference_gemm_kernel_f32(bool        transA_is_n,
                                              bool        transB_is_n,
                                              int64_t     M,
                                              int64_t     N,
                                              int64_t     K,
                                              float       alpha,
                                              float       beta,
                                              const void* A,
                                              hipDataType tA,
                                              int64_t     lda,
                                              int64_t     strideA,
                                              const void* B,
                                              hipDataType tB,
                                              int64_t     ldb,
                                              int64_t     strideB,
                                              const Tc*   C,
                                              int64_t     ldc,
                                              int64_t     strideC,
                                              To*         D,
                                              int64_t     ldd,
                                              int64_t     strideD,
                                              int32_t     batchCount)
    {
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t j = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
        const int64_t b = blockIdx.z;
        if(i >= M || j >= N || b >= batchCount)
            return;

        // Batch offset folded into the element index so the void* A/B advance by
        // the correct per-type element size inside the loader.
        const int64_t aBatch = b * strideA;
        const int64_t bBatch = b * strideB;
        const Tc*     Cb     = C + b * strideC;
        To*           Db     = D + b * strideD;

        // BLAS leaves A/B unreferenced when alpha==0, so skip the loads entirely
        // (also avoids faulting on a null A/B in that case).
        float acc = 0.0f;
        if(alpha != 0.0f)
            for(int64_t l = 0; l < K; ++l)
            {
                const int64_t aIdx = transA_is_n ? (i + l * lda) : (l + i * lda);
                const int64_t bIdx = transB_is_n ? (l + j * ldb) : (j + l * ldb);
                acc += load_input_f32(A, aBatch + aIdx, tA) * load_input_f32(B, bBatch + bIdx, tB);
            }

        // beta==0 leaves C unread even if it holds inf/nan.
        float out = alpha * acc;
        if(beta != 0.0f)
            out += beta * static_cast<float>(Cb[i + j * ldc]);

        Db[i + j * ldd] = static_cast<To>(out);
    }

    template <typename Tc, typename To>
    void launch_reference_gemm_f32(bool        transA_is_n,
                                   bool        transB_is_n,
                                   int64_t     M,
                                   int64_t     N,
                                   int64_t     K,
                                   double      alpha,
                                   double      beta,
                                   const void* dA,
                                   hipDataType tA,
                                   int64_t     lda,
                                   int64_t     strideA,
                                   const void* dB,
                                   hipDataType tB,
                                   int64_t     ldb,
                                   int64_t     strideB,
                                   const void* dC,
                                   int64_t     ldc,
                                   int64_t     strideC,
                                   void*       dDgold,
                                   int64_t     ldd,
                                   int64_t     strideD,
                                   int32_t     batchCount,
                                   hipStream_t stream)
    {
        const dim3 block(16, 16, 1);
        const dim3 grid(uint32_t((M + block.x - 1) / block.x),
                        uint32_t((N + block.y - 1) / block.y),
                        uint32_t(batchCount));
        reference_gemm_kernel_f32<Tc, To><<<grid, block, 0, stream>>>(
            transA_is_n,
            transB_is_n,
            M,
            N,
            K,
            float(alpha),
            float(beta),
            dA,
            tA,
            lda,
            strideA,
            dB,
            tB,
            ldb,
            strideB,
            static_cast<const Tc*>(dC),
            ldc,
            strideC,
            static_cast<To*>(dDgold),
            ldd,
            strideD,
            batchCount);
    }
} // namespace

// Serial-float K accumulation diverges from the library reduction order by more
// than 4 ULP at large K (~71 ULP at K=16384 f32), so the exact (tol==0) unit_check
// is only meaningful at small K. See also the note in gpu_compare.hpp.
bool gpu_ref_supported(const Arguments& arg, std::string& reason)
{
    auto fail = [&](const char* r) {
        reason = r;
        return false;
    };

    if(arg.grouped_gemm != 0)
        return fail("grouped GEMM");
    if(arg.batch_mode != 0)
        return fail("pointer-array (general) batch mode");
    // Batch maps onto a grid dimension (grid.z in the reference launch, grid.y in
    // the compare launch), both capped at 65535.
    if(arg.batch_count > 65535)
        return fail("batch count above the 65535 grid-dimension limit");
    if(arg.a_type != arg.b_type || !is_supported_type(arg.a_type))
        return fail("A/B type other than matching f32/f16/bf16");
    if(!is_supported_type(arg.c_type) || !is_supported_type(arg.d_type))
        return fail("C/D type other than f32/f16/bf16");
    if(arg.compute_type != HIPBLAS_COMPUTE_32F)
        return fail("compute type other than HIPBLAS_COMPUTE_32F");
    if(arg.compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
       || arg.compute_input_typeB != HIPBLASLT_DATATYPE_INVALID)
        return fail("compute-input-type override (e.g. TF32)");
    if(arg.activation_type != hipblaslt_activation_type::none)
        return fail("activation epilogue");
    if(arg.bias_vector)
        return fail("bias epilogue");
    if(arg.gradient)
        return fail("gradient epilogue");
    if(arg.use_e)
        return fail("auxiliary (E) output");
    if(arg.scaleA != hipblaslt_scaling_format::none
       || arg.scaleB != hipblaslt_scaling_format::none)
        return fail("A/B scaling");
    if(arg.scaleC || arg.scaleD)
        return fail("C/D scaling");
    if(arg.scaleAlpha_vector)
        return fail("alpha vector");
    if(arg.scaleE)
        return fail("E scaling");
    if(arg.amaxScaleA || arg.amaxScaleB || arg.amaxD)
        return fail("amax");
    if(arg.swizzle_a || arg.swizzle_b)
        return fail("tensor swizzling");
    if(arg.c_equal_d)
        return fail("in-place C==D");
    if(arg.rotating != 0)
        return fail("rotating buffers");

    reason.clear();
    return true;
}

bool run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               double      alpha,
                               double      beta,
                               const void* dA,
                               hipDataType tA,
                               int64_t     lda,
                               int64_t     strideA,
                               const void* dB,
                               hipDataType tB,
                               int64_t     ldb,
                               int64_t     strideB,
                               const void* dC,
                               hipDataType tC,
                               int64_t     ldc,
                               int64_t     strideC,
                               void*       dDgold,
                               hipDataType tD,
                               int64_t     ldd,
                               int64_t     strideD,
                               int32_t     batchCount,
                               hipStream_t stream)
{
    if(M <= 0 || N <= 0 || batchCount <= 0)
        return true;

    // Float accumulate. A/B input types are runtime args to the loader, so only
    // (C type, D type) are dispatched here (a 3x3 over f32/bf16/f16).
#define GPU_REF_F32_ARGS                                                          \
    transA_is_n, transB_is_n, M, N, K, alpha, beta, dA, tA, lda, strideA, dB, tB, \
        ldb, strideB, dC, ldc, strideC, dDgold, ldd, strideD, batchCount, stream

#define GPU_REF_LAUNCH_F32(TC, TO) launch_reference_gemm_f32<TC, TO>(GPU_REF_F32_ARGS)

#define GPU_REF_DISPATCH_TO(TC)                    \
    do                                             \
    {                                              \
        if(tD == HIP_R_32F)                        \
            GPU_REF_LAUNCH_F32(TC, float);         \
        else if(tD == HIP_R_16BF)                  \
            GPU_REF_LAUNCH_F32(TC, hip_bfloat16);  \
        else                                       \
            GPU_REF_LAUNCH_F32(TC, hipblasLtHalf); \
    } while(0)

    if(tC == HIP_R_32F)
        GPU_REF_DISPATCH_TO(float);
    else if(tC == HIP_R_16BF)
        GPU_REF_DISPATCH_TO(hip_bfloat16);
    else
        GPU_REF_DISPATCH_TO(hipblasLtHalf);

#undef GPU_REF_DISPATCH_TO
#undef GPU_REF_LAUNCH_F32
#undef GPU_REF_F32_ARGS

    return gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
}
