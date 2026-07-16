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

    template <typename T>
    __device__ inline float to_float(T v)
    {
        return static_cast<float>(v);
    }

    // Naive reference GEMM: one thread per output element, float accumulate.
    // Column-major with the same transpose/leading-dim/batch-stride conventions
    // as cblas_gemm(). Correctness and independence from the library kernel are
    // the only goals; not tuned for performance.
    template <typename Ti, typename Tc, typename To>
    __global__ void reference_gemm_kernel(bool      transA_is_n,
                                          bool      transB_is_n,
                                          int64_t   M,
                                          int64_t   N,
                                          int64_t   K,
                                          float     alpha,
                                          float     beta,
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

        float acc = 0.0f;
        for(int64_t l = 0; l < K; ++l)
        {
            const float a  = transA_is_n ? to_float(Ab[i + l * lda]) : to_float(Ab[l + i * lda]);
            const float bv = transB_is_n ? to_float(Bb[l + j * ldb]) : to_float(Bb[j + l * ldb]);
            acc += a * bv;
        }

        float out = alpha * acc;
        if(beta != 0.0f) // beta==0 ignores C even if it holds inf/nan
            out += beta * to_float(Cb[i + j * ldc]);

        Db[i + j * ldd] = static_cast<To>(out);
    }

    template <typename Ti, typename Tc, typename To>
    void launch_reference_gemm(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               float       alpha,
                               float       beta,
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
        reference_gemm_kernel<Ti, Tc, To><<<grid, block, 0, stream>>>(transA_is_n,
                                                                      transB_is_n,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      alpha,
                                                                      beta,
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

    bool is_f32_or_f16(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F;
    }
} // namespace

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
    if(!is_f32_or_f16(arg.a_type) || !is_f32_or_f16(arg.b_type))
        return fail("input type other than f32/f16");
    if(!is_f32_or_f16(arg.c_type) || !is_f32_or_f16(arg.d_type))
        return fail("C/D type other than f32/f16");
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
                               float       alpha,
                               float       beta,
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

    // Dispatch on (input type, C type, D type); f32/f16 only.
#define GPU_REF_LAUNCH(TI, TC, TO)                 \
    launch_reference_gemm<TI, TC, TO>(transA_is_n, \
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

#define GPU_REF_DISPATCH_TO(TI, TC)                    \
    do                                                 \
    {                                                  \
        if(tD == HIP_R_32F)                            \
            GPU_REF_LAUNCH(TI, TC, float);             \
        else                                           \
            GPU_REF_LAUNCH(TI, TC, hipblasLtHalf);     \
    } while(0)

#define GPU_REF_DISPATCH_TC(TI)                        \
    do                                                 \
    {                                                  \
        if(tC == HIP_R_32F)                            \
            GPU_REF_DISPATCH_TO(TI, float);            \
        else                                           \
            GPU_REF_DISPATCH_TO(TI, hipblasLtHalf);    \
    } while(0)

    if(tA == HIP_R_32F)
        GPU_REF_DISPATCH_TC(float);
    else
        GPU_REF_DISPATCH_TC(hipblasLtHalf);

#undef GPU_REF_DISPATCH_TC
#undef GPU_REF_DISPATCH_TO
#undef GPU_REF_LAUNCH

    gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
}
