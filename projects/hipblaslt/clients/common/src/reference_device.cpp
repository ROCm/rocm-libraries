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

    // Decode OCP fp8/bf8 via the public HIP conversion API (fp8 -> half -> float).
    // __hip_cvt_fp8_to_halfraw builds as device code on every target arch; the OCP
    // interpretation decodes correctly where OCP fp8 is supported (gfx950+). The
    // interpretation enum carries the format widths; OCP fp8 fits losslessly in half.
    __device__ inline float to_f32(hipblaslt_f8 v)
    {
        return float(__half(__hip_cvt_fp8_to_halfraw(v.__x, __HIP_E4M3)));
    }
    __device__ inline float to_f32(hipblaslt_bf8 v)
    {
        return float(__half(__hip_cvt_fp8_to_halfraw(v.__x, __HIP_E5M2)));
    }

    // Load one input element as float, decoding by runtime type. A single
    // per-element path lets A and B carry different input types (e.g. OCP f8 x
    // bf8). `idx` is the logical element index into the (batch-offset) matrix.
    // An unhandled type returns NaN so a mis-gated type fails loud (the NaN
    // propagates to the output and the compare kernel flags it).
    __device__ inline float load_input_f32(const void* base, int64_t idx, hipDataType t)
    {
        switch(t)
        {
        case HIP_R_32F:
            return static_cast<const float*>(base)[idx];
        case HIP_R_16BF:
            return float(static_cast<const hip_bfloat16*>(base)[idx]);
        case HIP_R_8F_E4M3:
            return to_f32(static_cast<const hipblaslt_f8*>(base)[idx]);
        case HIP_R_8F_E5M2:
            return to_f32(static_cast<const hipblaslt_bf8*>(base)[idx]);
        case HIP_R_16F:
            return float(static_cast<const hipblasLtHalf*>(base)[idx]);
        default:
            return nanf("");
        }
    }

    // Device complex used for the complex GEMM path. Bit-compatible with
    // std::complex<R> ({real, imag}), so device buffers written/read as
    // std::complex<R> reinterpret directly. Supplies the +, *, ==0 the reference
    // kernel needs; complex float accumulates in complex<float>, complex double in
    // complex<double>, matching cblas_cgemm/cblas_zgemm.
    template <typename R>
    struct gpu_ref_complex
    {
        using value_type = R;
        R re;
        R im;
        __host__ __device__ gpu_ref_complex()
            : re(R(0))
            , im(R(0))
        {
        }
        __host__ __device__ gpu_ref_complex(R r)
            : re(r)
            , im(R(0))
        {
        }
        __host__ __device__ gpu_ref_complex(R r, R i)
            : re(r)
            , im(i)
        {
        }
    };

    template <typename R>
    __device__ inline gpu_ref_complex<R> operator+(gpu_ref_complex<R> a, gpu_ref_complex<R> b)
    {
        return {a.re + b.re, a.im + b.im};
    }
    template <typename R>
    __device__ inline gpu_ref_complex<R>& operator+=(gpu_ref_complex<R>& a, gpu_ref_complex<R> b)
    {
        a.re += b.re;
        a.im += b.im;
        return a;
    }
    template <typename R>
    __device__ inline gpu_ref_complex<R> operator*(gpu_ref_complex<R> a, gpu_ref_complex<R> b)
    {
        return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    }
    template <typename R>
    __device__ inline bool operator==(gpu_ref_complex<R> a, gpu_ref_complex<R> b)
    {
        return a.re == b.re && a.im == b.im;
    }
    template <typename R>
    __device__ inline bool operator!=(gpu_ref_complex<R> a, gpu_ref_complex<R> b)
    {
        return !(a == b);
    }

    template <typename T>
    struct is_gpu_ref_complex : std::false_type
    {
    };
    template <typename R>
    struct is_gpu_ref_complex<gpu_ref_complex<R>> : std::true_type
    {
    };

    // op(): conjugate is a no-op for real elements and negates the imaginary part
    // for complex, so transA/transB == HIPBLAS_OP_C applies conjugate-transpose.
    template <typename T>
    __device__ inline T ref_conj(T v)
    {
        return v;
    }
    template <typename R>
    __device__ inline gpu_ref_complex<R> ref_conj(gpu_ref_complex<R> v)
    {
        return {v.re, -v.im};
    }

    // Build the accumulate-type scalar (alpha/beta) from real+imag doubles. The
    // imaginary part is ignored for real Tacc (always 0 there) and carried for
    // complex Tacc.
    template <typename Tacc>
    inline Tacc make_acc(double re, double im)
    {
        if constexpr(is_gpu_ref_complex<Tacc>::value)
            return Tacc(typename Tacc::value_type(re), typename Tacc::value_type(im));
        else
            return static_cast<Tacc>(re);
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

    // Store the accumulated result to D. int8 output saturates to [-128, 127],
    // matching cblas saturate_cast<hipblasLtInt8>; the int path accumulates exactly
    // in int32 so no rounding is needed. All other outputs use a plain cast.
    template <typename To, typename Tacc>
    __device__ inline To store_out(Tacc out)
    {
        if constexpr(std::is_same<To, hipblasLtInt8>::value)
        {
            if(out > Tacc(127))
                out = Tacc(127);
            else if(out < Tacc(-128))
                out = Tacc(-128);
            return static_cast<To>(out);
        }
        else
        {
            return static_cast<To>(out);
        }
    }

    // Naive reference GEMM: one thread per output element, accumulate in Tacc.
    // Column-major with the same transpose/leading-dim/batch-stride conventions
    // as cblas_gemm(). conjA/conjB apply conjugation (HIPBLAS_OP_C) on top of the
    // transpose; they are false on the real paths. Correctness and independence
    // from the library kernel are the only goals; not tuned for performance.
    template <typename Ti, typename Tacc, typename Tc, typename To>
    __global__ void reference_gemm_kernel(bool      transA_is_n,
                                          bool      transB_is_n,
                                          bool      conjA,
                                          bool      conjB,
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
            Tacc a  = transA_is_n ? to_acc<Tacc>(Ab[i + l * lda])
                                  : to_acc<Tacc>(Ab[l + i * lda]);
            Tacc bv = transB_is_n ? to_acc<Tacc>(Bb[l + j * ldb])
                                  : to_acc<Tacc>(Bb[j + l * ldb]);
            if(conjA)
                a = ref_conj(a);
            if(conjB)
                bv = ref_conj(bv);
            acc += a * bv;
        }

        // alpha==0 drops the A*B product entirely (BLAS convention), so 0*inf
        // does not become nan.
        Tacc out = (alpha == Tacc(0)) ? Tacc(0) : alpha * acc;
        if(beta != Tacc(0)) // beta==0 ignores C even if it holds inf/nan
            out += beta * static_cast<Tacc>(Cb[i + j * ldc]);

        Db[i + j * ldd] = store_out<To>(out);
    }

    template <typename Ti, typename Tacc, typename Tc, typename To>
    void launch_reference_gemm(bool        transA_is_n,
                               bool        transB_is_n,
                               bool        conjA,
                               bool        conjB,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               double      alpha,
                               double      alphai,
                               double      beta,
                               double      betai,
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
            conjA,
            conjB,
            M,
            N,
            K,
            make_acc<Tacc>(alpha, alphai),
            make_acc<Tacc>(beta, betai),
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

    // float-accumulate reference GEMM with runtime-typed A/B loads, so A and B
    // may be different input types (e.g. OCP f8 x bf8). One thread per output
    // element; C/D remain templated. Real inputs only (no conjugation).
    //
    // Non-MX scale factors: scaleA/scaleAlphaVec index the M row (i), scaleB the N
    // column (j); scalar mode passes scaleAIsVec/scaleBIsVec == false (index 0).
    // All are constant across the K-loop and batch, so fold outside accumulation.
    // scaleC folds into beta at the caller; scaleD scales the full result at store.
    template <typename Tc, typename To>
    __global__ void reference_gemm_kernel_f32(bool         transA_is_n,
                                              bool         transB_is_n,
                                              int64_t      M,
                                              int64_t      N,
                                              int64_t      K,
                                              float        alpha,
                                              float        beta,
                                              const void*  A,
                                              hipDataType  tA,
                                              int64_t      lda,
                                              int64_t      strideA,
                                              const void*  B,
                                              hipDataType  tB,
                                              int64_t      ldb,
                                              int64_t      strideB,
                                              const Tc*    C,
                                              int64_t      ldc,
                                              int64_t      strideC,
                                              To*          D,
                                              int64_t      ldd,
                                              int64_t      strideD,
                                              int32_t      batchCount,
                                              const float* scaleA,
                                              bool         scaleAIsVec,
                                              const float* scaleB,
                                              bool         scaleBIsVec,
                                              const float* scaleAlphaVec,
                                              float        scaleD)
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

        // Scale factors are constant across K and batch; absent scales are 1.
        const float sa  = scaleA ? scaleA[scaleAIsVec ? i : 0] : 1.0f;
        const float sb  = scaleB ? scaleB[scaleBIsVec ? j : 0] : 1.0f;
        const float sav = scaleAlphaVec ? scaleAlphaVec[i] : 1.0f;

        float acc = 0.0f;
        for(int64_t l = 0; l < K; ++l)
        {
            const int64_t aIdx = transA_is_n ? (i + l * lda) : (l + i * lda);
            const int64_t bIdx = transB_is_n ? (l + j * ldb) : (j + l * ldb);
            acc += load_input_f32(A, aBatch + aIdx, tA) * load_input_f32(B, bBatch + bIdx, tB);
        }

        // alpha==0 drops the A*B product entirely (BLAS convention), so 0*inf
        // does not become nan; beta==0 ignores C even if it holds inf/nan.
        float out = (alpha == 0.0f) ? 0.0f : alpha * sa * sav * sb * acc;
        if(beta != 0.0f) // beta already folded with scaleC by the caller
            out += beta * static_cast<float>(Cb[i + j * ldc]);
        out *= scaleD;

        Db[i + j * ldd] = store_out<To>(out);
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
                                   const void* dScaleA,
                                   bool        scaleAIsVec,
                                   const void* dScaleB,
                                   bool        scaleBIsVec,
                                   const void* dScaleAlphaVec,
                                   double      scaleD,
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
            batchCount,
            static_cast<const float*>(dScaleA),
            scaleAIsVec,
            static_cast<const float*>(dScaleB),
            scaleBIsVec,
            static_cast<const float*>(dScaleAlphaVec),
            float(scaleD));
    }

    // Input types accepted by the reference path (a_type/b_type). OCP fp8/bf8 only;
    // float-class inputs (f32/f16/bf16/fp8/bf8) may differ between A and B. f64 is
    // only valid on the compute-64F path and int8 only on the compute-32I path;
    // complex float/double are all-complex only (all enforced by the gate).
    bool is_supported_input(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F || t == HIP_R_16BF || t == HIP_R_8F_E4M3
               || t == HIP_R_8F_E5M2 || t == HIP_R_64F || t == HIP_R_8I || t == HIP_C_32F
               || t == HIP_C_64F;
    }

    // C/D types accepted by the reference path (the compare kernel handles these).
    // int8 output is only valid on the compute-32I path; complex float/double are
    // all-complex only (both enforced by the gate).
    bool is_supported_output(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F || t == HIP_R_16BF || t == HIP_R_64F
               || t == HIP_R_32I || t == HIP_R_8I || t == HIP_C_32F || t == HIP_C_64F;
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
    if(arg.a_type == HIP_R_8F_E4M3_FNUZ || arg.a_type == HIP_R_8F_E5M2_FNUZ
       || arg.b_type == HIP_R_8F_E4M3_FNUZ || arg.b_type == HIP_R_8F_E5M2_FNUZ)
        return fail("FNUZ fp8/bf8 input (only OCP fp8/bf8 supported)");
    if(!is_supported_input(arg.a_type) || !is_supported_input(arg.b_type))
        return fail("input type other than f32/f16/bf16/fp8/bf8/f64/int8/complex");
    if(!is_supported_output(arg.c_type) || !is_supported_output(arg.d_type))
        return fail("C/D type other than f32/f16/bf16/f64/int32/int8/complex");
    // Complex (HIP_C_32F/HIP_C_64F): require all four A/B/C/D the same complex type
    // (no real/complex mixing), complex float on compute 32F and complex double on
    // compute 64F. The complex path replaces the real compute-type checks below.
    auto is_complex = [](hipDataType t) { return t == HIP_C_32F || t == HIP_C_64F; };
    const bool any_complex = is_complex(arg.a_type) || is_complex(arg.b_type)
                             || is_complex(arg.c_type) || is_complex(arg.d_type);
    if(any_complex)
    {
        if(arg.a_type != arg.b_type || arg.b_type != arg.c_type || arg.c_type != arg.d_type)
            return fail("complex GEMM requires matching complex A/B/C/D (no real/complex mix)");
        if(arg.a_type == HIP_C_32F && arg.compute_type != HIPBLAS_COMPUTE_32F)
            return fail("complex float requires compute 32F");
        if(arg.a_type == HIP_C_64F && arg.compute_type != HIPBLAS_COMPUTE_64F)
            return fail("complex double requires compute 64F");
    }
    // compute 64F requires all f64; compute 32I requires int8 in with int32 or int8
    // out; the f32-class computes (32F and the 16F/fast-16F/fast-16BF/fast-TF32
    // variants, all f32-accumulate in the reference) require no f64, no int8/int32.
    else if(arg.compute_type == HIPBLAS_COMPUTE_64F)
    {
        if(arg.a_type != HIP_R_64F || arg.b_type != HIP_R_64F || arg.c_type != HIP_R_64F
           || arg.d_type != HIP_R_64F)
            return fail("compute 64F requires f64 A/B/C/D");
    }
    else if(arg.compute_type == HIPBLAS_COMPUTE_32I)
    {
        // C and D must share a width: the int dispatch keys the C read-type on D's
        // type, so a mixed int32/int8 C/D would read C as the wrong type.
        const bool d_ok = arg.d_type == HIP_R_32I || arg.d_type == HIP_R_8I;
        if(arg.a_type != HIP_R_8I || arg.b_type != HIP_R_8I || arg.c_type != arg.d_type
           || !d_ok)
            return fail("int8 GEMM requires matching int32 or int8 C/D");
    }
    else if(arg.compute_type == HIPBLAS_COMPUTE_32F
            || arg.compute_type == HIPBLAS_COMPUTE_16F
            || arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_16F
            || arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_16BF
            || arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
    {
        if(arg.a_type == HIP_R_64F || arg.b_type == HIP_R_64F || arg.c_type == HIP_R_64F
           || arg.d_type == HIP_R_64F)
            return fail("f64 A/B/C/D requires compute 64F");
        if(arg.a_type == HIP_R_8I || arg.b_type == HIP_R_8I || arg.c_type == HIP_R_32I
           || arg.d_type == HIP_R_32I || arg.c_type == HIP_R_8I || arg.d_type == HIP_R_8I)
            return fail("int8/int32 A/B/C/D requires compute 32I");
    }
    else
    {
        return fail("compute type other than HIPBLAS_COMPUTE_32F-class/32I/64F");
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
    // Non-MX scalar/vector scaleA/B, scaleAlphaVec, scaleC, scaleD are supported on
    // the float-accumulate, float-scale computes only (32F and the fast-32F
    // variants, where the compute type is float so the scale buffers are float).
    // 16F-compute (half) scaling and MX/block A/B scaling are still deferred.
    const bool scale_capable
        = !any_complex
          && (arg.compute_type == HIPBLAS_COMPUTE_32F
              || arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_16F
              || arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_16BF
              || arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32);
    if(isBlockScaling(arg.scaleA) || isBlockScaling(arg.scaleB))
        return fail("MX/block A/B scaling");
    if((arg.scaleA != hipblaslt_scaling_format::none
        || arg.scaleB != hipblaslt_scaling_format::none)
       && !scale_capable)
        return fail("A/B scaling requires compute 32F-class (float-scale) path");
    if((arg.scaleC || arg.scaleD) && !scale_capable)
        return fail("C/D scaling requires compute 32F-class (float-scale) path");
    if(arg.scaleAlpha_vector && !scale_capable)
        return fail("alpha vector requires compute 32F-class (float-scale) path");
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

void run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               bool        conjA,
                               bool        conjB,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               double      alpha,
                               double      alphai,
                               double      beta,
                               double      betai,
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
                               hipStream_t stream,
                               const void* dScaleA,
                               bool        scaleAIsVec,
                               const void* dScaleB,
                               bool        scaleBIsVec,
                               const void* dScaleAlphaVec,
                               double      scaleD)
{
    if(M <= 0 || N <= 0 || batchCount <= 0)
        return;

    // Shared trailing arguments (dA..stream) for every launch below.
#define GPU_REF_ARGS                                                                       \
    transA_is_n, transB_is_n, conjA, conjB, M, N, K, alpha, alphai, beta, betai, dA, lda,  \
        strideA, dB, ldb, strideB, dC, ldc, strideC, dDgold, ldd, strideD, batchCount,     \
        stream

    // Complex is a single all-complex instantiation per width (the gate guarantees
    // matching complex A/B/C/D). complex float accumulates in complex<float>,
    // complex double in complex<double>, matching cblas_cgemm/cblas_zgemm.
    if(tD == HIP_C_32F)
    {
        launch_reference_gemm<gpu_ref_complex<float>,
                              gpu_ref_complex<float>,
                              gpu_ref_complex<float>,
                              gpu_ref_complex<float>>(GPU_REF_ARGS);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }
    if(tD == HIP_C_64F)
    {
        launch_reference_gemm<gpu_ref_complex<double>,
                              gpu_ref_complex<double>,
                              gpu_ref_complex<double>,
                              gpu_ref_complex<double>>(GPU_REF_ARGS);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }

    // f64 is a single all-double instantiation (compute 64F guarantees f64 A/B/C/D).
    if(tD == HIP_R_64F)
    {
        launch_reference_gemm<double, double, double, double>(GPU_REF_ARGS);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }

    // int8 in / int32 out is a single all-int32 instantiation (compute 32I
    // guarantees int8 A/B and int32 C/D). int32 accumulate matches the hardware
    // accumulator, including overflow wrap.
    if(tD == HIP_R_32I)
    {
        launch_reference_gemm<hipblasLtInt8, int32_t, int32_t, int32_t>(GPU_REF_ARGS);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }

    // int8 in / int8 out (compute 32I guarantees int8 A/B/C/D here). int32
    // accumulate matches the hardware accumulator; store_out saturates the int32
    // result to [-128, 127] on the int8 store, matching cblas saturate_cast.
    if(tD == HIP_R_8I)
    {
        launch_reference_gemm<hipblasLtInt8, int32_t, hipblasLtInt8, hipblasLtInt8>(GPU_REF_ARGS);
        gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
        return;
    }

    // Float accumulate. A/B input types are runtime args to the loader, so only
    // (C type, D type) are dispatched here (a 3x3 over f32/bf16/f16).
#define GPU_REF_F32_ARGS                                                              \
    transA_is_n, transB_is_n, M, N, K, alpha, beta, dA, tA, lda, strideA, dB, tB,     \
        ldb, strideB, dC, ldc, strideC, dDgold, ldd, strideD, batchCount, dScaleA,    \
        scaleAIsVec, dScaleB, scaleBIsVec, dScaleAlphaVec, scaleD, stream

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
#undef GPU_REF_ARGS

    gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
}
