// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "reference_device.hpp"
#include "hipblaslt_ostream.hpp"

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

    // Decode OCP fp8/bf8 via the public HIP conversion API (fp8 -> half -> float).
    // __hip_cvt_fp8_to_halfraw is __device__-callable on all target arches and the
    // interpretation enum carries the format widths; OCP fp8 fits losslessly in half.
    __device__ inline float to_f32(hipblaslt_f8 v)
    {
        return float(__half(__hip_cvt_fp8_to_halfraw(v.__x, __HIP_E4M3)));
    }
    __device__ inline float to_f32(hipblaslt_bf8 v)
    {
        return float(__half(__hip_cvt_fp8_to_halfraw(v.__x, __HIP_E5M2)));
    }

    // Convert an input element to the accumulate type. fp8/bf8 route through the
    // public HIP decode above; the class operator float() is host-only where
    // HIP_FP8_TYPE_OCP is 0 (e.g. gfx942).
    template <typename Tacc, typename Ti>
    __device__ inline Tacc to_acc(Ti v)
    {
        return static_cast<Tacc>(v);
    }
    template <typename Tacc>
    __device__ inline Tacc to_acc(hipblaslt_f8 v)
    {
        return static_cast<Tacc>(to_f32(v));
    }
    template <typename Tacc>
    __device__ inline Tacc to_acc(hipblaslt_bf8 v)
    {
        return static_cast<Tacc>(to_f32(v));
    }

    // Naive reference GEMM: one thread per output element, accumulate in Tacc.
    // Column-major with the same transpose/leading-dim/batch-stride conventions
    // as cblas_gemm(). Correctness and independence from the library kernel are
    // the only goals; not tuned for performance.
    template <typename Ti, typename Tacc, typename Tc, typename To>
    __global__ void reference_gemm_kernel(bool      transA_is_n,
                                          bool      transB_is_n,
                                          int64_t   M,
                                          int64_t   N,
                                          int64_t   K,
                                          Tacc      alpha,
                                          Tacc      beta,
                                          const Ti* A,
                                          int64_t   lda,
                                          int64_t   strideA,
                                          const Ti* B,
                                          int64_t   ldb,
                                          int64_t   strideB,
                                          const Tc* C,
                                          int64_t   ldc,
                                          int64_t   strideC,
                                          To*       D,
                                          int64_t   ldd,
                                          int64_t   strideD,
                                          int32_t   batchCount)
    {
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t j = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
        const int64_t b = blockIdx.z;
        if(i >= M || j >= N || b >= batchCount)
            return;

        const Ti* Ab = A + b * strideA;
        const Ti* Bb = B + b * strideB;
        const Tc* Cb = C + b * strideC;
        To*       Db = D + b * strideD;

        // Read inputs via to_acc; for double, Tacc is the identity.
        Tacc acc = Tacc(0);
        for(int64_t l = 0; l < K; ++l)
        {
            const Tacc a  = transA_is_n ? to_acc<Tacc>(Ab[i + l * lda])
                                        : to_acc<Tacc>(Ab[l + i * lda]);
            const Tacc bv = transB_is_n ? to_acc<Tacc>(Bb[l + j * ldb])
                                        : to_acc<Tacc>(Bb[j + l * ldb]);
            acc += a * bv;
        }

        // alpha==0 drops the A*B product entirely (BLAS convention), so 0*inf
        // does not become nan.
        Tacc out = (alpha == Tacc(0)) ? Tacc(0) : alpha * acc;
        if(beta != Tacc(0)) // beta==0 ignores C even if it holds inf/nan
            out += beta * static_cast<Tacc>(Cb[i + j * ldc]);

        Db[i + j * ldd] = static_cast<To>(out);
    }

    template <typename Ti, typename Tacc, typename Tc, typename To>
    void launch_reference_gemm(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               double      alpha,
                               double      beta,
                               const void* dA,
                               int64_t     lda,
                               int64_t     strideA,
                               const void* dB,
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
        reference_gemm_kernel<Ti, Tacc, Tc, To><<<grid, block, 0, stream>>>(
            transA_is_n,
            transB_is_n,
            M,
            N,
            K,
            static_cast<Tacc>(alpha),
            static_cast<Tacc>(beta),
            static_cast<const Ti*>(dA),
            lda,
            strideA,
            static_cast<const Ti*>(dB),
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

    // Input types accepted by the reference path (a_type/b_type). OCP fp8/bf8 only;
    // f64 is only valid on the compute-64F path and int8 only on the compute-32I
    // path (both enforced by the compute gate).
    bool is_supported_input(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F || t == HIP_R_16BF || t == HIP_R_8F_E4M3
               || t == HIP_R_8F_E5M2 || t == HIP_R_64F || t == HIP_R_8I;
    }

    // C/D types accepted by the reference path (the compare kernel handles these).
    bool is_supported_output(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F || t == HIP_R_16BF || t == HIP_R_64F
               || t == HIP_R_32I;
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
    if(arg.a_type != arg.b_type)
        return fail("mixed A/B input types");
    if(arg.a_type == HIP_R_8F_E4M3_FNUZ || arg.a_type == HIP_R_8F_E5M2_FNUZ
       || arg.b_type == HIP_R_8F_E4M3_FNUZ || arg.b_type == HIP_R_8F_E5M2_FNUZ)
        return fail("FNUZ fp8/bf8 input (only OCP fp8/bf8 supported)");
    // int8 output is unsupported; only int32 output (int8 in / int32 out).
    if(arg.c_type == HIP_R_8I || arg.d_type == HIP_R_8I)
        return fail("int8 output (only int32 output supported)");
    if(!is_supported_input(arg.a_type) || !is_supported_input(arg.b_type))
        return fail("input type other than f32/f16/bf16/fp8/bf8/int8");
    if(!is_supported_output(arg.c_type) || !is_supported_output(arg.d_type))
        return fail("C/D type other than f32/f16/bf16/f64/int32");
    // compute 64F requires all f64; compute 32I requires int8 in / int32 out;
    // compute 32F requires no f64, no int8 in, no int32 out.
    if(arg.compute_type == HIPBLAS_COMPUTE_64F)
    {
        if(arg.a_type != HIP_R_64F || arg.b_type != HIP_R_64F || arg.c_type != HIP_R_64F
           || arg.d_type != HIP_R_64F)
            return fail("compute 64F requires f64 A/B/C/D");
    }
    else if(arg.compute_type == HIPBLAS_COMPUTE_32I)
    {
        if(arg.a_type != HIP_R_8I || arg.b_type != HIP_R_8I || arg.c_type != HIP_R_32I
           || arg.d_type != HIP_R_32I)
            return fail("compute 32I requires int8 A/B and int32 C/D");
    }
    else if(arg.compute_type == HIPBLAS_COMPUTE_32F)
    {
        if(arg.a_type == HIP_R_64F || arg.b_type == HIP_R_64F || arg.c_type == HIP_R_64F
           || arg.d_type == HIP_R_64F)
            return fail("f64 A/B/C/D requires compute 64F");
        if(arg.a_type == HIP_R_8I || arg.b_type == HIP_R_8I || arg.c_type == HIP_R_32I
           || arg.d_type == HIP_R_32I)
            return fail("int8/int32 A/B/C/D requires compute 32I");
    }
    else
    {
        return fail("compute type other than HIPBLAS_COMPUTE_32F/32I/64F");
    }
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
    if(arg.scaleC || arg.scaleD || arg.scaleE)
        return fail("C/D/E scaling");
    if(arg.scaleAlpha_vector)
        return fail("alpha vector");
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

void run_reference_gemm_device(bool        transA_is_n,
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
        return;
    if(tA != tB)
    {
        hipblaslt_cerr << "gpu_ref: mixed A/B types not supported" << std::endl;
        return;
    }

    // f64 is a single all-double instantiation (compute 64F guarantees f64 A/B/C/D).
    if(tD == HIP_R_64F)
    {
        launch_reference_gemm<double, double, double, double>(transA_is_n,
                                                             transB_is_n,
                                                             M,
                                                             N,
                                                             K,
                                                             alpha,
                                                             beta,
                                                             dA,
                                                             lda,
                                                             strideA,
                                                             dB,
                                                             ldb,
                                                             strideB,
                                                             dC,
                                                             ldc,
                                                             strideC,
                                                             dDgold,
                                                             ldd,
                                                             strideD,
                                                             batchCount,
                                                             stream);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }

    // int8 in / int32 out is a single all-int32 instantiation (compute 32I
    // guarantees int8 A/B and int32 C/D). int32 accumulate matches the hardware
    // accumulator, including overflow wrap.
    if(tD == HIP_R_32I)
    {
        launch_reference_gemm<hipblasLtInt8, int32_t, int32_t, int32_t>(transA_is_n,
                                                                        transB_is_n,
                                                                        M,
                                                                        N,
                                                                        K,
                                                                        alpha,
                                                                        beta,
                                                                        dA,
                                                                        lda,
                                                                        strideA,
                                                                        dB,
                                                                        ldb,
                                                                        strideB,
                                                                        dC,
                                                                        ldc,
                                                                        strideC,
                                                                        dDgold,
                                                                        ldd,
                                                                        strideD,
                                                                        batchCount,
                                                                        stream);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }

    // Non-f64: float accumulate. Dispatch on (input type, C type, D type).
#define GPU_REF_LAUNCH(TI, TC, TO)                       \
    launch_reference_gemm<TI, float, TC, TO>(transA_is_n, \
                                             transB_is_n, \
                                             M,           \
                                             N,           \
                                             K,           \
                                             alpha,       \
                                             beta,        \
                                             dA,          \
                                             lda,         \
                                             strideA,     \
                                             dB,          \
                                             ldb,         \
                                             strideB,     \
                                             dC,          \
                                             ldc,         \
                                             strideC,     \
                                             dDgold,      \
                                             ldd,         \
                                             strideD,     \
                                             batchCount,  \
                                             stream)

#define GPU_REF_DISPATCH_TO(TI, TC)                \
    do                                             \
    {                                              \
        if(tD == HIP_R_32F)                        \
            GPU_REF_LAUNCH(TI, TC, float);         \
        else if(tD == HIP_R_16BF)                  \
            GPU_REF_LAUNCH(TI, TC, hip_bfloat16);  \
        else                                       \
            GPU_REF_LAUNCH(TI, TC, hipblasLtHalf); \
    } while(0)

#define GPU_REF_DISPATCH_TC(TI)                    \
    do                                             \
    {                                              \
        if(tC == HIP_R_32F)                        \
            GPU_REF_DISPATCH_TO(TI, float);        \
        else if(tC == HIP_R_16BF)                  \
            GPU_REF_DISPATCH_TO(TI, hip_bfloat16); \
        else                                       \
            GPU_REF_DISPATCH_TO(TI, hipblasLtHalf); \
    } while(0)

    if(tA == HIP_R_32F)
        GPU_REF_DISPATCH_TC(float);
    else if(tA == HIP_R_16BF)
        GPU_REF_DISPATCH_TC(hip_bfloat16);
    else if(tA == HIP_R_8F_E4M3)
        GPU_REF_DISPATCH_TC(hipblaslt_f8);
    else if(tA == HIP_R_8F_E5M2)
        GPU_REF_DISPATCH_TC(hipblaslt_bf8);
    else
        GPU_REF_DISPATCH_TC(hipblasLtHalf);

#undef GPU_REF_DISPATCH_TC
#undef GPU_REF_DISPATCH_TO
#undef GPU_REF_LAUNCH

    gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
}
