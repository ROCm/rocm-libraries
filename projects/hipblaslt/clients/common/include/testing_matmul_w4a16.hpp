// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// hipblaslt-bench driver for the w4a16 path: int4 weights in A, 16-bit
// activations in B, and one 16-bit float scale (optionally plus an int4
// zero-point) per K group of a row of A.
//
// This is a separate driver rather than a branch inside testing_matmul.hpp
// because the two things that make w4a16 what it is -- A packed two elements
// per byte, and a scale consumed in the main loop instead of the epilogue --
// are exactly what the generic path's element-wise allocation and MX scale
// layouts do not model. Everything that is not w4a16-specific (argument
// parsing, rotating buffers, adaptive timing) is shared.
//
// Supported shape: TN (A stored K-contiguous), single GEMM, batch_count 1,
// D = alpha * op(A) * op(B) + beta * C with no epilogue. Those are the same
// limits the kernels themselves carry.
// ---------------------------------------------------------------------------

#include "benchmark_stats.hpp"
#include "benchmark_timing.hpp"
#include "datatype_interface.hpp"
#include "hipBuffer.hpp"
#include "hipblaslt_arguments.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Defined in testing_matmul.hpp; re-declared so --flush works here without
// depending on include order.
extern "C" __global__ void flush_icache();

namespace w4a16_detail
{
    // Zero-points and the two unsigned weight encodings share this bias: an
    // unsigned nibble q in [0,15] represents q - 8 when no zero-point tensor is
    // supplied.
    constexpr int implicitZeroPoint = 8;

    inline float toStorage(float v, hipDataType t)
    {
        return t == HIP_R_16F ? static_cast<float>(static_cast<hipblasLtHalf>(v))
                              : static_cast<float>(static_cast<hip_bfloat16>(v));
    }

    inline void storeAs(void* base, size_t idx, float v, hipDataType t)
    {
        if(t == HIP_R_16F)
            static_cast<hipblasLtHalf*>(base)[idx] = static_cast<hipblasLtHalf>(v);
        else
            static_cast<hip_bfloat16*>(base)[idx] = static_cast<hip_bfloat16>(v);
    }

    inline float loadAs(const void* base, size_t idx, hipDataType t)
    {
        return t == HIP_R_16F
                   ? static_cast<float>(static_cast<const hipblasLtHalf*>(base)[idx])
                   : static_cast<float>(static_cast<const hip_bfloat16*>(base)[idx]);
    }

    /// Linear element index -> (byte, nibble) for the three weight encodings.
    /// Signed and UnsignedBias8 differ only in how the nibble is read, so they
    /// share an address; UnsignedBias8ExLlama additionally interleaves the eight
    /// nibbles of each dword as [0,2,4,6,1,3,5,7], which puts elements 2k and
    /// 2k+1 in the low and high halves of the dword.
    inline void nibbleAddr(int32_t encoding, size_t idx, size_t& byteIdx, size_t& nibble)
    {
        if(encoding != HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXLLAMA_EXT)
        {
            byteIdx = idx / 2;
            nibble  = idx % 2;
            return;
        }
        // Element j of a dword sits at bit shift (j/2)*4 + (j%2)*16, i.e. byte
        // (j%2)*2 + j/4 of the dword, nibble (j/2)%2.
        const size_t dword = idx / 8;
        const size_t j     = idx % 8;
        byteIdx            = dword * 4 + (j % 2) * 2 + (j / 4);
        nibble             = (j / 2) % 2;
    }

    inline void writeNibble(uint8_t* base, size_t byteIdx, size_t nibble, uint8_t raw)
    {
        if(nibble)
            base[byteIdx] = static_cast<uint8_t>((base[byteIdx] & 0x0F) | (raw << 4));
        else
            base[byteIdx] = static_cast<uint8_t>((base[byteIdx] & 0xF0) | raw);
    }

    /// The problem geometry and every derived size, in one place so the host
    /// generator, the reference and the descriptor setup cannot disagree.
    struct Geometry
    {
        int64_t m = 0, n = 0, k = 0;
        int64_t lda = 0, ldb = 0, ldc = 0, ldd = 0;

        int         groupSize = 0;
        int64_t     kGroups   = 0;
        bool        zeroPoint = false;
        hipDataType scaleType = HIP_R_16BF;
        int32_t     encoding  = 0;

        size_t aBytes         = 0; // packed int4 weights
        size_t zeroPointBytes = 0;
        size_t zeroPointOfs   = 0; // byte offset of the zero-points inside the scale allocation
        size_t scaleAllocBytes = 0;

        /// Nibble index of A(m, k). A is stored K-contiguous with row stride
        /// lda, matching the Tensile "Alik" layout the kernels were built for.
        size_t aIndex(int64_t row, int64_t col) const
        {
            return static_cast<size_t>(row) * static_cast<size_t>(lda) + static_cast<size_t>(col);
        }

        /// The scale tensor is a dense [M][kGroups] with the group innermost.
        size_t scaleIndex(int64_t row, int64_t kGroup) const
        {
            return static_cast<size_t>(row) * static_cast<size_t>(kGroups)
                   + static_cast<size_t>(kGroup);
        }

        /// The zero-points sit on the same [M][kGroups] grid but are packed two
        /// per byte along M: byte = (m/2)*kGroups + g, nibble = m & 1. That is
        /// what keeps a thread's nibble fixed by its row for the whole K walk.
        size_t zeroPointNibbleIndex(int64_t row, int64_t kGroup) const
        {
            return 2 * (static_cast<size_t>(row / 2) * static_cast<size_t>(kGroups)
                        + static_cast<size_t>(kGroup))
                   + static_cast<size_t>(row & 1);
        }
    };

    inline Geometry makeGeometry(const Arguments& arg)
    {
        Geometry g;
        g.m         = arg.M[0];
        g.n         = arg.N[0];
        g.k         = arg.K[0];
        g.lda       = arg.lda[0];
        g.ldb       = arg.ldb[0];
        g.ldc       = arg.ldc[0];
        g.ldd       = arg.ldd[0];
        g.groupSize = w4a16GroupSize(arg.scaleA);
        g.kGroups   = (g.k + g.groupSize - 1) / g.groupSize;
        g.zeroPoint = isW4A16ZeroPoint(arg.scaleA);
        g.scaleType = w4a16ScaleType(arg.scaleA);
        g.encoding  = arg.int4_encoding;

        g.aBytes = static_cast<size_t>(g.m) * static_cast<size_t>(g.lda) / 2;

        const size_t scaleBytes = static_cast<size_t>(g.m) * static_cast<size_t>(g.kGroups) * 2;
        if(g.zeroPoint)
        {
            // The zero-points start at the next 128-byte boundary after the
            // scales, inside the same allocation the user passes as scaleA.
            g.zeroPointOfs    = (scaleBytes + 127) / 128 * 128;
            g.zeroPointBytes  = static_cast<size_t>((g.m + 1) / 2) * static_cast<size_t>(g.kGroups);
            g.scaleAllocBytes = g.zeroPointOfs + g.zeroPointBytes;
        }
        else
        {
            g.scaleAllocBytes = scaleBytes;
        }
        return g;
    }

    /// Generate the packed weights, the scales and (when asymmetric) the
    /// zero-points, and hand back the dequantized weights the reference GEMM
    /// consumes. The dequantized value is rounded through the MAC type because
    /// that is what the kernel does: it converts (q - z) * s to bf16/fp16 on the
    /// way into LDS, so leaving the reference in f32 would make the rounding
    /// error look like a kernel bug.
    inline std::vector<float> generateA(const Geometry& g,
                                        uint8_t*        packedA,
                                        void*           scales,
                                        uint8_t*        zeroPoints,
                                        uint32_t        seed)
    {
        std::mt19937                     rng(seed);
        std::uniform_int_distribution<int> nibbleDist(0, 15);
        // Scales stay positive and O(1) so the dequantized weights land in a
        // range where bf16 has its usual relative precision.
        std::uniform_real_distribution<float> scaleDist(0.02f, 0.08f);

        const bool unsignedEnc = g.encoding != HIPBLASLT_INT4_ENCODING_SIGNED_EXT;

        std::vector<float> zByGroup(static_cast<size_t>(g.m) * g.kGroups, 0.0f);
        for(int64_t row = 0; row < g.m; ++row)
        {
            for(int64_t grp = 0; grp < g.kGroups; ++grp)
            {
                const float s = scaleDist(rng);
                storeAs(scales, g.scaleIndex(row, grp), s, g.scaleType);

                int z = unsignedEnc ? implicitZeroPoint : 0;
                if(g.zeroPoint)
                {
                    const int raw = nibbleDist(rng);
                    // z is read back in the same domain as the weights: raw
                    // [0,15] for the unsigned encodings, two's complement for
                    // the signed one.
                    z = unsignedEnc ? raw : (raw ^ 0x8) - 8;
                    const size_t e = g.zeroPointNibbleIndex(row, grp);
                    writeNibble(zeroPoints, e / 2, e & 1, static_cast<uint8_t>(raw));
                }
                zByGroup[g.scaleIndex(row, grp)] = static_cast<float>(z);
            }
        }

        std::vector<float> deq(static_cast<size_t>(g.m) * static_cast<size_t>(g.k));
        for(int64_t row = 0; row < g.m; ++row)
        {
            for(int64_t col = 0; col < g.k; ++col)
            {
                const int raw = nibbleDist(rng);
                size_t    byteIdx, nibble;
                nibbleAddr(g.encoding, g.aIndex(row, col), byteIdx, nibble);
                writeNibble(packedA, byteIdx, nibble, static_cast<uint8_t>(raw));

                const float q = unsignedEnc ? static_cast<float>(raw)
                                            : static_cast<float>((raw ^ 0x8) - 8);
                const size_t si = g.scaleIndex(row, col / g.groupSize);
                const float  s  = loadAs(scales, si, g.scaleType);
                deq[static_cast<size_t>(row) * static_cast<size_t>(g.k) + col]
                    = toStorage((q - zByGroup[si]) * s, g.scaleType);
            }
        }
        return deq;
    }

    /// D = alpha * Adeq * B + beta * C, accumulating in f32. Adeq is [M][K] and
    /// B is [K][N] with row stride ldb (transB == N).
    inline void referenceGemm(const Geometry&           g,
                              const std::vector<float>& aDeq,
                              const void*               b,
                              const void*               c,
                              hipDataType               bType,
                              float                     alpha,
                              float                     beta,
                              std::vector<float>&       out)
    {
        out.assign(static_cast<size_t>(g.m) * static_cast<size_t>(g.n), 0.0f);
#pragma omp parallel for collapse(2)
        for(int64_t col = 0; col < g.n; ++col)
        {
            for(int64_t row = 0; row < g.m; ++row)
            {
                float acc = 0.0f;
                for(int64_t kk = 0; kk < g.k; ++kk)
                {
                    acc += aDeq[static_cast<size_t>(row) * static_cast<size_t>(g.k) + kk]
                           * loadAs(b, static_cast<size_t>(col) * g.ldb + kk, bType);
                }
                float v = alpha * acc;
                if(beta != 0.0f)
                    v += beta * loadAs(c, static_cast<size_t>(col) * g.ldc + row, bType);
                out[static_cast<size_t>(col) * static_cast<size_t>(g.m) + row] = v;
            }
        }
    }

    /// Relative L2 error of `got` against `want`. Scale-free, so one threshold
    /// covers every problem size.
    inline double relativeL2(const std::vector<float>& want, const void* got, hipDataType t)
    {
        double num = 0.0, den = 0.0;
        for(size_t i = 0; i < want.size(); ++i)
        {
            const double g = loadAs(got, i, t);
            const double w = want[i];
            num += (g - w) * (g - w);
            den += w * w;
        }
        return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    }
} // namespace w4a16_detail

inline void testing_matmul_w4a16(const Arguments& arg)
{
    using namespace w4a16_detail;

    // ---- Shape limits, all of them limits of the kernels themselves --------
    auto reject = [&](const std::string& why) {
        hipblaslt_cerr << "hipblaslt-bench w4a16: " << why << std::endl;
    };

    if(arg.grouped_gemm > 0)
        return reject("grouped GEMM is not supported");
    if(arg.batch_count > 1)
        return reject("batch_count > 1 is not supported (the scale tensor has no batch dimension)");
    if(arg.transA != 'T' || arg.transB != 'N')
        return reject("only TN is supported (--transA T --transB N)");
    if(arg.compute_type != HIPBLAS_COMPUTE_32F)
        return reject("only --compute_type f32_r is supported");
    // The scale element type follows B's, and C/D share B's type here so one
    // 2-byte element size describes every 16-bit buffer below.
    if(arg.c_type != arg.b_type || arg.d_type != arg.b_type)
        return reject("--c_type and --d_type must match --b_type");
    if(arg.bias_vector || arg.activation_type != hipblaslt_activation_type::none || arg.use_e
       || arg.gradient || arg.scaleB != hipblaslt_scaling_format::none || arg.scaleC || arg.scaleD
       || arg.scaleAlpha_vector || arg.amaxD)
        return reject("no epilogue is supported yet (bias, activation, aux, B/C/D scaling)");

    const Geometry g = makeGeometry(arg);

    if(g.lda != g.k)
        return reject("lda must equal K: the packed int4 stream has no room for padding");
    if(g.k % g.groupSize != 0)
        return reject("K must be a multiple of the scale group size");
    // GlobalReadVectorWidthA is 8 int4 (one dword) per load, and the kernels
    // have no tail loop, so a partial dword would read past the end.
    if(g.k % 8 != 0)
        return reject("K must be a multiple of 8 (one dword of int4 per global load)");

    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    hipEvent_t eventStart, eventStop;
    CHECK_HIP_ERROR(hipEventCreate(&eventStart));
    CHECK_HIP_ERROR(hipEventCreate(&eventStop));

    hipDeviceProp_t deviceProps;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));

    const hipDataType bType = arg.b_type;
    const size_t      bElems = static_cast<size_t>(g.n) * static_cast<size_t>(g.ldb);
    const size_t      cElems = static_cast<size_t>(g.n) * static_cast<size_t>(g.ldc);
    const size_t      dElems = static_cast<size_t>(g.n) * static_cast<size_t>(g.ldd);
    const size_t      elemBytes = 2;

    // ---- Rotating buffers ---------------------------------------------------
    // One "block" is a full copy of every input the kernel reads, so rotating
    // over them defeats the cache exactly the way the generic path does.
    const int64_t bytesPerIter = static_cast<int64_t>(g.aBytes + g.scaleAllocBytes
                                                      + (bElems + cElems + dElems) * elemBytes);
    const auto    plan         = hipblaslt_bench::compute_rotating_buffer_plan(
        arg.adaptive,
        arg.max_iters,
        arg.cold_iters,
        arg.iters,
        static_cast<int64_t>(arg.rotating) * 1024 * 1024,
        bytesPerIter);
    const int32_t blockCount = plan.block_count;

    // ---- Host data ----------------------------------------------------------
    std::vector<uint8_t> hA(g.aBytes, 0);
    std::vector<uint8_t> hScale(g.scaleAllocBytes, 0);
    std::vector<uint8_t> hB(bElems * elemBytes, 0);
    std::vector<uint8_t> hC(cElems * elemBytes, 0);
    std::vector<uint8_t> hD(dElems * elemBytes, 0);

    // Fixed seeds: the reference is built from the same draw as the device
    // input, so the run is reproducible and --initialization has no say here.
    const std::vector<float> aDeq
        = generateA(g, hA.data(), hScale.data(), hScale.data() + g.zeroPointOfs, 24681u);

    {
        std::mt19937                          rng(97531u);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(size_t i = 0; i < bElems; ++i)
            storeAs(hB.data(), i, dist(rng), bType);
        for(size_t i = 0; i < cElems; ++i)
            storeAs(hC.data(), i, dist(rng), bType);
    }

    const float alpha = arg.alpha;
    const float beta  = arg.beta;

    // ---- Device data --------------------------------------------------------
    // Byte-typed buffers throughout: HipDeviceBuffer sizes from
    // realDataTypeSize(), which has no entry for packed int4.
    HipDeviceBuffer dA(HIP_R_8U, g.aBytes * blockCount, arg.HMM);
    HipDeviceBuffer dScale(HIP_R_8U, g.scaleAllocBytes * blockCount, arg.HMM);
    HipDeviceBuffer dB(HIP_R_8U, bElems * elemBytes * blockCount, arg.HMM);
    HipDeviceBuffer dC(HIP_R_8U, cElems * elemBytes * blockCount, arg.HMM);
    HipDeviceBuffer dD(HIP_R_8U, dElems * elemBytes * blockCount, arg.HMM);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dScale.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dD.memcheck());

    auto fill = [&](HipDeviceBuffer& dst, const std::vector<uint8_t>& src) {
        for(int32_t b = 0; b < blockCount; ++b)
            CHECK_HIP_ERROR(hipMemcpy(
                dst.as<uint8_t>() + b * src.size(), src.data(), src.size(), hipMemcpyHostToDevice));
    };
    fill(dA, hA);
    fill(dScale, hScale);
    fill(dB, hB);
    fill(dC, hC);
    fill(dD, hD);

    // ---- Descriptors --------------------------------------------------------
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatmulDesc_t   matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(
        &matA, static_cast<hipDataType>(HIP_R_4I_EXT), g.k, g.m, g.lda));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, bType, g.k, g.n, g.ldb));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.c_type, g.m, g.n, g.ldc));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.d_type, g.m, g.n, g.ldd));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, HIP_R_32F));

    const hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // The scale mode number is the public enumerator, which is what --scaleA
    // takes, so no translation table is needed here.
    const int32_t scaleMode = static_cast<int32_t>(arg.scaleA);
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT,
                                        &arg.int4_encoding,
                                        sizeof(arg.int4_encoding)));

    // ---- Algorithm selection ------------------------------------------------
    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
    auto setScalePointer = [&](int32_t block) {
        void* p = dScale.as<uint8_t>() + static_cast<size_t>(block) * g.scaleAllocBytes;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p, sizeof(p)));
    };
    setScalePointer(0);

    if(arg.algo_method == 2 && arg.solution_index >= 0)
    {
        std::vector<int> idx{arg.solution_index};
        CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAlgosFromIndex(handle, idx, algos));
        if(algos.empty())
            return reject("no algorithm for --solution_index "
                          + std::to_string(arg.solution_index));
    }
    else if(arg.algo_method == 1 || (arg.algo_method == 2 && arg.solution_index < 0))
    {
        CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                         opA,
                                                         opB,
                                                         arg.a_type,
                                                         arg.b_type,
                                                         arg.c_type,
                                                         arg.d_type,
                                                         arg.compute_type,
                                                         algos));
    }
    else
    {
        hipblasLtMatmulPreference_t pref;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
        uint64_t workspace = 0;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)));
        const int requested = arg.requested_solution_num > 0 ? arg.requested_solution_num : 1;
        algos.resize(requested);
        int returned = 0;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul, matA, matB, matC, matD, pref, requested, algos.data(), &returned));
        algos.resize(returned);
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    }

    if(algos.empty())
        return reject("no solution found for this problem");

    // ---- Reference ----------------------------------------------------------
    const bool validate = arg.unit_check || arg.norm_check || arg.allclose_check;
    std::vector<float> ref;
    if(validate)
        referenceGemm(g, aDeq, hB.data(), hC.data(), bType, alpha, beta, ref);

    // ---- Flush cost ---------------------------------------------------------
    const int32_t gpuBlocks     = deviceProps.multiProcessorCount * 60;
    double        flushTimeUsed = 0.0;
    if(arg.flush)
    {
        constexpr int flushIter = 100000;
        for(int i = 0; i < flushIter; ++i)
            hipLaunchKernelGGL(flush_icache, dim3(gpuBlocks), dim3(64), 0, stream);
        pre_gpu_time(arg.use_gpu_timer, eventStart, flushTimeUsed, stream);
        for(int i = 0; i < flushIter; ++i)
            hipLaunchKernelGGL(flush_icache, dim3(gpuBlocks), dim3(64), 0, stream);
        post_gpu_time(arg.use_gpu_timer, eventStart, eventStop, flushTimeUsed, stream);
        flushTimeUsed /= flushIter;
    }

    hipblaslt_bench::TimingConfig cfg;
    cfg.iters         = arg.iters;
    cfg.use_gpu_timer = arg.use_gpu_timer;
    cfg.adaptive      = arg.adaptive;
    cfg.flush_us      = flushTimeUsed;
    if(arg.adaptive)
    {
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
        if(const auto err = hipblaslt_bench::validate_adaptive_config(cfg); !err.empty())
            return reject("invalid adaptive timing config: " + err);
    }

    // A w4a16 GEMM reads the packed weights, the scales and B, and writes D;
    // alpha != 0 always, so C is read only when beta != 0.
    const double gemmBytes
        = static_cast<double>(g.aBytes + g.scaleAllocBytes)
          + static_cast<double>(bElems + dElems + (beta != 0.0f ? cElems : 0)) * elemBytes;
    const double gemmFlops = 2.0 * static_cast<double>(g.m) * g.n * g.k;

    hipblaslt_cout << "transA,transB,M,N,K,lda,ldb,ldc,ldd,a_type,b_type,scaleA,group_size,"
                      "zero_point,int4_encoding,alpha,beta,rotating_buffer,flush,solution_index,"
                      "us,tflops,GB/s,rel_l2_err,status"
                   << std::endl;

    for(auto& res : algos)
    {
        const int solIndex = hipblaslt_ext::getIndexFromAlgo(res.algo);
        size_t    wsSize   = 0;
        if(hipblaslt_ext::matmulIsAlgoSupported(
               handle, matmul, &alpha, matA, matB, &beta, matC, matD, res.algo, wsSize)
           != HIPBLAS_STATUS_SUCCESS)
            continue;

        void* workspace = nullptr;
        if(wsSize)
            CHECK_HIP_ERROR(hipMalloc(&workspace, wsSize));

        auto launch = [&](int64_t i) {
            const int32_t b = static_cast<int32_t>(i % blockCount);
            setScalePointer(b);
            if(arg.flush)
                hipLaunchKernelGGL(flush_icache, dim3(gpuBlocks), dim3(64), 0, stream);
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmul(handle,
                                matmul,
                                &alpha,
                                dA.as<uint8_t>() + static_cast<size_t>(b) * g.aBytes,
                                matA,
                                dB.as<uint8_t>() + static_cast<size_t>(b) * bElems * elemBytes,
                                matB,
                                &beta,
                                dC.as<uint8_t>() + static_cast<size_t>(b) * cElems * elemBytes,
                                matC,
                                dD.as<uint8_t>() + static_cast<size_t>(b) * dElems * elemBytes,
                                matD,
                                &res.algo,
                                workspace,
                                wsSize,
                                stream));
        };

        // One cold call feeds both the correctness check and the warm cache.
        launch(0);
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        double      relErr = 0.0;
        const char* status = "-";
        if(validate)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hD.data(), dD.as<uint8_t>(), hD.size(), hipMemcpyDeviceToHost));
            relErr = relativeL2(ref, hD.data(), arg.d_type);
            // bf16 carries 8 significand bits, fp16 11; the dequantized weight
            // is rounded to that width both in the kernel and in the reference,
            // so what is left is accumulation order. One ulp of headroom over
            // the rounding step is enough and still catches a wrong nibble,
            // group or zero-point, which move the result by percent or more.
            const double tol = (bType == HIP_R_16F) ? 1e-3 : 8e-3;
            status           = relErr <= tol ? "PASS" : "FAIL";
        }

        hipblaslt_bench::TimingResult timing;
        if(arg.timing)
            hipblaslt_bench::run_measurement(launch, cfg, eventStart, eventStop, stream, timing);

        const double us     = timing.median_us;
        const double tflops = us > 0.0 ? gemmFlops / us * 1e-6 : 0.0;
        const double gbs    = us > 0.0 ? gemmBytes / us * 1e-3 : 0.0;

        hipblaslt_cout << arg.transA << "," << arg.transB << "," << g.m << "," << g.n << "," << g.k
                       << "," << g.lda << "," << g.ldb << "," << g.ldc << "," << g.ldd << ","
                       << hip_datatype_to_string(arg.a_type) << ","
                       << hip_datatype_to_string(arg.b_type) << ","
                       << static_cast<int>(arg.scaleA) << "," << g.groupSize << ","
                       << (g.zeroPoint ? 1 : 0) << "," << arg.int4_encoding << "," << alpha << ","
                       << beta << "," << arg.rotating << "," << arg.flush << "," << solIndex << ","
                       << us << "," << tflops << "," << gbs << "," << relErr << "," << status
                       << std::endl;

        if(arg.print_solution_found || arg.algo_method != 0)
        {
            hipblaslt_cout << "    --Solution index: " << solIndex << "\n"
                           << "    --Solution name:  "
                           << hipblaslt_ext::getSolutionNameFromAlgo(handle, res.algo) << "\n"
                           << "    --kernel name:    "
                           << hipblaslt_ext::getKernelNameFromAlgo(handle, res.algo) << std::endl;
        }

        if(timing.adaptive)
        {
            const char* conv = !timing.noise_active ? "-"
                               : timing.converged   ? "converged"
                               : timing.stable      ? "stable"
                                                    : "noisy";
            hipblaslt_cout << "    --adaptive: batch=" << timing.batch
                           << " samples=" << timing.samples << " hot_iters=" << timing.hot_iters
                           << " mean_us=" << timing.mean_us << " min_us=" << timing.min_us
                           << " cv=" << timing.cv << " rel_iqr=" << timing.rel_iqr << " " << conv
                           << std::endl;
        }

        if(workspace)
            CHECK_HIP_ERROR(hipFree(workspace));
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIP_ERROR(hipEventDestroy(eventStart));
    CHECK_HIP_ERROR(hipEventDestroy(eventStop));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
