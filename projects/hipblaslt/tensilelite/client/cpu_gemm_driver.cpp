/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "ProgramOptions.hpp"
#include "Reference.hpp"
#include "rocisa/include/enum.hpp"
#include <Tensile/Activation.hpp>

/*
 * CPU GEMM Driver and Validator
 *
 * This tool acts as a test harness for the TensileLite CPU GEMM implementation.
 * It allows for command-line verification of matrix multiplication kernels across
 * different data types (f32, f16, bf16) and geometries. It can also be used for
 * benchmarking different CPU GEMM implementations.
 *
 * The driver performs the following steps:
 * 1. Sets up a contraction problem based on user arguments (M, N, K, Transpose, etc).
 * 2. Initializes input matrices (A, B) with random data.
 * 3. Executes the "Device Under Test" (the optimized CPU solve).
 * 4. Optionally validates the result against a simple, golden reference implementation.
 *
 * Usage Examples:
 * # Standard f32 run
 * ./cpu_gemm_driver --M 1024 --N 1024 --K 1024
 *
 * # BF16 run with validation enabled
 * ./cpu_gemm_driver --type bf16 --M 512 --N 512 --K 256 --validate 1
 *
 * # Benchmark mode (validation disabled)
 * ./cpu_gemm_driver --M 2048 --N 2048 --K 2048 --validate 0 --tryFastPath 1
 *
 * # Help messnage
 * ./cpu_gemm_driver --help
 */

namespace
{
    using namespace TensileLite;
    using namespace TensileLite::Client;

    // Helper traits to map C++ storage types to rocisa data type enums.
    template <typename T>
    struct TypeTraits;

    template <>
    struct TypeTraits<float>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Float;
    };

    template <>
    struct TypeTraits<double>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Double;
    };

    template <>
    struct TypeTraits<TensileLite::Half>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Half;
    };

    template <>
    struct TypeTraits<TensileLite::BFloat16>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::BFloat16;
    };

#ifdef TENSILE_USE_FP8_BF8
    template <>
    struct TypeTraits<TensileLite::Float8>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Float8;
    };

    template <>
    struct TypeTraits<TensileLite::BFloat8>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::BFloat8;
    };

    template <>
    struct TypeTraits<TensileLite::Float8_fnuz>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Float8_fnuz;
    };

    template <>
    struct TypeTraits<TensileLite::BFloat8_fnuz>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::BFloat8_fnuz;
    };
#endif

#ifndef _WIN32
    template <>
    struct TypeTraits<TensileLite::Float4x2>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Float4;
    };
#endif

    // A slow, easy to understand, golden reference implementation of GEMM.
    // Used strictly for validating the correctness of the optimized path.
    // Calculates, for each element (i, j):
    //   D[i,j] = activation( effectiveAlpha * (A * B)[i,j] + beta * C[i,j] + bias[i] )
    // where:
    //   effectiveAlpha = alpha
    //                  * scaleA[i]                              (if scaleAVec     != nullptr)
    //                  * scaleB[j]                              (if scaleBVec     != nullptr)
    //                  * scaleAlphaVec[factorDim == 0 ? i : j]  (if scaleAlphaVec != nullptr)
    //
    // scaleA is always indexed by row (M), scaleB always by col (N).
    // factorDim only affects scaleAlphaVec: 0 = row-dim (length M), 1 = col-dim (length N).
    //
    // Template parameters:
    //   AccumT       — accumulation type (float or double)
    //   MathOpAccumT — math-operation precision type. When different from AccumT,
    //                  each A/B element is cast through MathOpAccumT before multiply
    //                  (e.g. XFloat32 truncates mantissa to 10 bits).
    //
    // When both MX operand descriptors are populated, the K-reduction is
    // block-structured: accumulate min(mxA.block, mxB.block) products, then
    // scale by the MX scale factors before adding to the running sum.

    // Quantize a value through `Narrow`, then return as float. This mirrors
    // what the GPU MFMA path does when storage is wider than the MAC input
    // type (e.g. Half stored, F8 used in MFMA).
    template <typename Narrow>
    inline float quantizeThroughNarrow(float v)
    {
        return static_cast<float>(static_cast<Narrow>(v));
    }

    using QuantizeFn = float (*)(float);

    inline QuantizeFn quantizerFor(rocisa::DataType t)
    {
        switch(t)
        {
        case rocisa::DataType::Float:    return [](float v) { return v; };
        case rocisa::DataType::Half:     return &quantizeThroughNarrow<TensileLite::Half>;
        case rocisa::DataType::BFloat16: return &quantizeThroughNarrow<TensileLite::BFloat16>;
#ifdef TENSILE_USE_FP8_BF8
        case rocisa::DataType::Float8:        return &quantizeThroughNarrow<TensileLite::Float8>;
        case rocisa::DataType::BFloat8:       return &quantizeThroughNarrow<TensileLite::BFloat8>;
        case rocisa::DataType::Float8_fnuz:   return &quantizeThroughNarrow<TensileLite::Float8_fnuz>;
        case rocisa::DataType::BFloat8_fnuz:  return &quantizeThroughNarrow<TensileLite::BFloat8_fnuz>;
#endif
        default:
            throw std::runtime_error("Unsupported compute-input type for golden GEMM quantizer.");
        }
    }

#ifndef _WIN32
    struct MXGemmOperand
    {
        const E8* scale        = nullptr;
        int       block        = 0;
        size_t    strideFree   = 0;
        size_t    strideKBlock = 0;

        bool empty() const
        {
            return scale == nullptr && block == 0 && strideFree == 0
                   && strideKBlock == 0;
        }
    };

    bool validateMXGemmOperands(const MXGemmOperand& mxA,
                                const MXGemmOperand& mxB)
    {
        const bool hasMXA = !mxA.empty();
        const bool hasMXB = !mxB.empty();

        if(hasMXA != hasMXB)
        {
            throw std::runtime_error(
                "CPU reference GEMM requires MX operands for both A and B, "
                "or for neither side.");
        }

        if(!hasMXA)
            return false;

        auto validateSide = [](const char* name, const MXGemmOperand& operand) {
            if(!operand.scale)
                throw std::runtime_error(std::string("CPU reference GEMM MX ")
                                         + name + " operand has no scale data.");
            if(operand.block <= 0)
                throw std::runtime_error(std::string("CPU reference GEMM MX ")
                                         + name + " operand has a non-positive block.");
            if(operand.strideFree == 0 || operand.strideKBlock == 0)
                throw std::runtime_error(std::string("CPU reference GEMM MX ")
                                         + name
                                         + " operand requires non-zero scale strides.");
        };

        validateSide("A", mxA);
        validateSide("B", mxB);
        return true;
    }
#endif

    template <typename AccumT = float, typename MathOpAccumT = AccumT>
    void columnMajorGemm(const AccumT*  a,
                         const AccumT*  b,
                         const AccumT*  c,
                         AccumT*        d,
                         size_t         m,
                         size_t         n,
                         size_t         k,
                         bool           transA,
                         bool           transB,
                         AccumT         alpha,
                         AccumT         beta,
                         const AccumT*  biasVec       = nullptr,
                         const AccumT*  scaleAlphaVec = nullptr,
                         ActivationType activation    = ActivationType::None,
                         const AccumT*  scaleAVec     = nullptr,
                         const AccumT*  scaleBVec     = nullptr,
                         int            factorDim     = 0,
                         QuantizeFn     quantizeA     = nullptr,
                         QuantizeFn     quantizeB     = nullptr
#ifndef _WIN32
                         ,
                         MXGemmOperand  mxA           = {},
                         MXGemmOperand  mxB           = {}
#endif
                         )
    {

        switch(activation)
        {
        case ActivationType::None:
        case ActivationType::Relu:
            break;
        default:
            throw std::runtime_error(
                "Unsupported activation for CPU reference GEMM "
                "(supported: None, Relu).");
        }
        size_t strideAK = transA ? 1 : m;
        size_t strideAM = transA ? k : 1;
        size_t strideBK = transB ? n : 1;
        size_t strideBN = transB ? 1 : k;

#ifndef _WIN32
        const bool hasMX = validateMXGemmOperands(mxA, mxB);
#endif

        for(size_t i = 0; i < m; i++)
        {
            for(size_t j = 0; j < n; j++)
            {
                AccumT sum = AccumT(0);

#ifndef _WIN32
                // MX scale tensor layout follows the ContractionProblem tensor
                // descriptors. Even with padScaleTensor=false, setMXScaleA/B
                // can pad the scale-K dimension, so using k/mxBlock as the
                // free-dimension stride would address the wrong scale element.
                //
                // Both MX operands have positive blocks and non-null scale
                // buffers here. Block shape is validated in runGemm. When the
                // blocks differ, one divides the other; step the inner
                // reduction by the smaller of the two so each inner segment
                // has constant (sa, sb) and can pick the correct scale index
                // for each side.
                if(hasMX)
                {
                    size_t step     = static_cast<size_t>(
                        std::min(mxA.block, mxB.block));
                    for(size_t lBase = 0; lBase < k; lBase += step)
                    {
                        AccumT blockSum = AccumT(0);
                        for(size_t t = 0; t < step; t++)
                        {
                            size_t l    = lBase + t;
                            AccumT aVal = a[i * strideAM + l * strideAK];
                            AccumT bVal = b[l * strideBK + j * strideBN];
                            if(quantizeA) aVal = static_cast<AccumT>(quantizeA(static_cast<float>(aVal)));
                            if(quantizeB) bVal = static_cast<AccumT>(quantizeB(static_cast<float>(bVal)));
                            blockSum += AccumT(MathOpAccumT(aVal))
                                      * AccumT(MathOpAccumT(bVal));
                        }

                        size_t blkA = lBase / static_cast<size_t>(mxA.block);
                        size_t blkB = lBase / static_cast<size_t>(mxB.block);

                        size_t mxsaIdx = i * mxA.strideFree
                                        + blkA * mxA.strideKBlock;
                        size_t mxsbIdx = j * mxB.strideFree
                                        + blkB * mxB.strideKBlock;

                        AccumT mxScale
                            = static_cast<AccumT>(static_cast<float>(mxA.scale[mxsaIdx]))
                            * static_cast<AccumT>(static_cast<float>(mxB.scale[mxsbIdx]));
                        sum += blockSum * mxScale;
                    }
                }
                else
#endif
                {
                    for(size_t l = 0; l < k; l++)
                    {
                        AccumT aVal = a[i * strideAM + l * strideAK];
                        AccumT bVal = b[l * strideBK + j * strideBN];
                        if(quantizeA) aVal = static_cast<AccumT>(quantizeA(static_cast<float>(aVal)));
                        if(quantizeB) bVal = static_cast<AccumT>(quantizeB(static_cast<float>(bVal)));
                        sum += AccumT(MathOpAccumT(aVal)) * AccumT(MathOpAccumT(bVal));
                    }
                }

                AccumT effectiveAlpha = alpha;
                if(scaleAVec)
                    effectiveAlpha *= static_cast<AccumT>(scaleAVec[i]);
                if(scaleBVec)
                    effectiveAlpha *= static_cast<AccumT>(scaleBVec[j]);
                if(scaleAlphaVec)
                    effectiveAlpha *= static_cast<AccumT>(
                        scaleAlphaVec[factorDim == 0 ? i : j]);

                AccumT result = effectiveAlpha * sum + beta * c[i + j * m];

                if(biasVec)
                    result += static_cast<AccumT>(biasVec[i]);

                if(activation == ActivationType::Relu)
                {
                    result = std::max(AccumT(0), result);
                }
                else
                {
                    assert(activation == ActivationType::None);
                }

                d[i + j * m] = result;
            }
        }
    }
}

/*
 * Main templated runner.
 * Handles memory allocation, data initialization, execution, and validation.
 *
 * InputT: The C++ type used for storage of A and B matrices (e.g. float, half).
 * AccumulateT: The type used for accumulation (currently restricted to float).
 */
template <typename InputAT, typename InputBT = InputAT, typename AccumulateT = float>
int runGemm(size_t         m,
            size_t         n,
            size_t         k,
            bool           transA,
            bool           transB,
            float          alpha,
            float          beta,
            bool           validate,
            bool           tryFastPath,
            bool           useBias,
            ActivationType activation,
            bool               useScaleAlphaVec,
            const std::string& useScaleAB,
            int                factorDim,
            rocisa::DataType   computeInputA = rocisa::DataType::None,
            rocisa::DataType   computeInputB = rocisa::DataType::None,
            int                mxBlockA      = 0,
            int                mxBlockB      = 0,
            size_t             batchCount    = 1,
            bool               isTF32        = false)
{
    if(batchCount == 0)
    {
        std::cerr << "Error: batchCount (" << batchCount << ") must be >= 1" << std::endl;
        return 1;
    }
    constexpr rocisa::DataType dtypeEnumA = TypeTraits<InputAT>::value;
    constexpr rocisa::DataType dtypeEnumB = TypeTraits<InputBT>::value;
    if(computeInputA == rocisa::DataType::None) computeInputA = dtypeEnumA;
    if(computeInputB == rocisa::DataType::None) computeInputB = dtypeEnumB;

#ifndef _WIN32
    constexpr bool isInputAFP4 = std::is_same_v<InputAT, Float4x2>;
    constexpr bool isInputBFP4 = std::is_same_v<InputBT, Float4x2>;
    static_assert(isInputAFP4 == isInputBFP4,
                  "FP4 input storage must be used for both A and B, or neither.");
    constexpr bool isFP4 = isInputAFP4;
#else
    constexpr bool isFP4 = false;
#endif

    if constexpr(!isFP4)
    {
        mxBlockA = 0;
        mxBlockB = 0;
    }

    if constexpr(isFP4)
    {
        // One-sided MX (only A or only B scaled) is not supported by either
        // reference path; they would disagree about what one-sided MX means.
        if((mxBlockA > 0) != (mxBlockB > 0))
        {
            std::cerr << "Error: one-sided MX is not supported "
                      << "(mxBlockA=" << mxBlockA << ", mxBlockB=" << mxBlockB
                      << "); set both > 0 or both 0." << std::endl;
            return 1;
        }
        auto checkSide = [&](const char* name, int b) -> int {
            if(b <= 0) return 0;
            if((b & (b - 1)) != 0)
            {
                std::cerr << "Error: " << name << " (" << b << ") must be a power of 2"
                          << std::endl;
                return 1;
            }
            if(k < static_cast<size_t>(b))
            {
                std::cerr << "Error: K (" << k << ") must be >= " << name << " (" << b << ")"
                          << std::endl;
                return 1;
            }
            if(k % static_cast<size_t>(b) != 0)
            {
                std::cerr << "Error: K (" << k << ") must be a multiple of " << name
                          << " (" << b << ")" << std::endl;
                return 1;
            }
            return 0;
        };
        if(int rc = checkSide("mxBlockA", mxBlockA)) return rc;
        if(int rc = checkSide("mxBlockB", mxBlockB)) return rc;

        // Asymmetric MX (mxBlockA != mxBlockB) is only supported on the fast
        // path. The production slow path's MX inner loop uses a single scale
        // per max(mxBlockA, mxBlockB)-sized segment, which collapses the
        // smaller-blocked side's per-segment scales onto the first one and
        // produces wrong results. Reject the combination at the driver rather
        // than ship a known-wrong slow path.
        if(mxBlockA != mxBlockB && !tryFastPath)
        {
            std::cerr << "Error: asymmetric MX (mxBlockA=" << mxBlockA
                      << " != mxBlockB=" << mxBlockB
                      << ") is only supported on the fast path "
                      << "(use --tryFastPath)." << std::endl;
            return 1;
        }
    }

    // Calculate strides assuming standard column-major packed storage
    size_t lda        = transA ? k : m;
    size_t ldb        = transB ? n : k;
    size_t ldc        = m;

    // C/D and alpha/beta types: use AccumulateT's DataType
    constexpr rocisa::DataType accumDtypeEnum = TypeTraits<AccumulateT>::value;

    // Define the contraction problem (geometry, strides, types)
    ContractionProblemGemm contraction
        = ContractionProblemGemm::GEMM_Strides(transA,
                                               transB,
                                               dtypeEnumA,
                                               dtypeEnumB,
                                               accumDtypeEnum,
                                               accumDtypeEnum,
                                               m,
                                               n,
                                               k,
                                               batchCount,
                                               lda,
                                               -1,
                                               ldb,
                                               -1,
                                               ldc,
                                               -1,
                                               ldc,
                                               -1,
                                               static_cast<double>(beta));

    contraction.setComputeInputTypeA(computeInputA);
    contraction.setComputeInputTypeB(computeInputB);
    contraction.setAlphaType(accumDtypeEnum);
    contraction.setBetaType(accumDtypeEnum);

    if(isTF32)
        contraction.setF32XdlMathOp(rocisa::DataType::XFloat32);

    // Allocate host memory for inputs and outputs. Each batch slice is packed.
    size_t numA = m * k;
    size_t numB = k * n;
    size_t numC = m * n;

    size_t storageA, storageB;
#ifndef _WIN32
    if constexpr(isFP4)
    {
        // Packed batch stride: 2 nibbles per byte, packed per batch slice.
        storageA = ((numA + 1) / 2) * batchCount;
        storageB = ((numB + 1) / 2) * batchCount;
    }
    else
#endif
    {
        storageA = numA * batchCount;
        storageB = numB * batchCount;
    }

    std::vector<InputAT>     a(storageA);
    std::vector<InputBT>     b(storageB);
    std::vector<AccumulateT> c(numC * batchCount);
    std::vector<AccumulateT> d(numC * batchCount);

    // Initialize inputs with random values. We use ±1 (binary) for A and B by
    // default because it is exactly representable in every supported storage
    // type (including FP8), so storage-side quantization is a no-op and the
    // comparison stays tight.
    //
    // For mixed-precision MAC validation (storage type wider than compute-input
    // type, e.g. Half storage + F8 compute), the test driver needs values that
    // are NOT on the F8 grid - otherwise the quantization step has nothing to
    // do and the bug being tested for can't be reproduced. We give an operand
    // such values when its storage type is wider than its computeInput type.
    //
    // For FP4 with mxBlockA/B>0 (mxfp4), inputs are drawn from the discrete
    // E2M1-representable value set so the MX-scale logic is exercised.
    size_t                                seed = 42;
    std::mt19937                          gen(seed);
    std::uniform_int_distribution<>       binary_distribution(0, 1);
    std::uniform_real_distribution<float> realDist(-1.0f, 1.0f);

    auto randomGen = [&]() { return binary_distribution(gen) ? 1.0f : -1.0f; };

#ifndef _WIN32
    if constexpr(isFP4)
    {
        // Full E2M1-representable value set: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6.
        // Drawing from the entire grid (not just the powers of two near zero)
        // exercises the MX-scale path with values whose products span more of
        // the FP4 range, while still being exactly representable.
        constexpr float fp4Values[]
            = {-6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
                0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};
        constexpr int fp4ValueCount = sizeof(fp4Values) / sizeof(fp4Values[0]);
        std::uniform_int_distribution<> fp4Dist(0, fp4ValueCount - 1);
        auto randomFp4 = [&]() { return fp4Values[fp4Dist(gen)]; };

        // Pack 2 logical FP4 values per byte (Float4x2). When the logical
        // element count is odd, the second slot of the last byte has no
        // element behind it — it's padding. We must still initialize that
        // slot to a valid FP4 value (we use 0), because the fast-path
        // ShadowBuffer FP4 decoder unconditionally reads both slots of every
        // byte (the guard is on the *write* back, not the read), and reading
        // uninitialized memory would be UB.
        auto initFp4Operand = [&](auto& vec, size_t numLogical) {
            const size_t storage    = vec.size();
            const bool   hasOddTail = (numLogical % 2 != 0);
            for(size_t i = 0; i < storage; ++i)
            {
                const bool isPaddingSlot = hasOddTail && (i == storage - 1);
                float      slot0         = randomFp4();
                float      slot1         = isPaddingSlot ? 0.0f : randomFp4();
                vec[i]                   = Float4x2(slot0, slot1);
            }
        };
        initFp4Operand(a, numA);
        initFp4Operand(b, numB);
    }
    else
#endif
    {
        auto initOperand = [&](auto& vec, bool quantizes) {
            using T = typename std::decay_t<decltype(vec)>::value_type;
#ifndef _WIN32
            if constexpr(std::is_same_v<T, Float4x2>)
            {
                // FP4 mixed-input init unsupported in this branch; the FP4-only
                // path above handles the pure FP4 case. Mixed FP4/non-FP4
                // dispatch is rejected at the dispatcher level.
                throw std::runtime_error(
                    "Mixed FP4 / non-FP4 input is not supported.");
            }
            else
#endif
            if(quantizes)
            {
                // Values representable in storage but not on the compute-input grid -
                // for storage=Half/compute=F8N, values like 0.7 that Half holds
                // exactly but F8N rounds to 0.625 or 0.75.
                std::generate(vec.begin(), vec.end(),
                              [&]() { return static_cast<T>(realDist(gen)); });
            }
            else
            {
                std::generate(vec.begin(), vec.end(),
                              [&]() { return static_cast<T>(randomGen()); });
            }
        };

        bool quantizesA = (sizeof(InputAT) > 1) && (computeInputA != dtypeEnumA);
        bool quantizesB = (sizeof(InputBT) > 1) && (computeInputB != dtypeEnumB);
        initOperand(a, quantizesA);
        initOperand(b, quantizesB);
    }
    std::generate(c.begin(), c.end(), [&]() { return static_cast<AccumulateT>(randomGen()); });

    // Optional feature buffers — typed as AccumulateT so the slow path's
    // GetValue(alphaType, ...) reads the correct byte width.
    std::vector<AccumulateT> biasVec;
    std::vector<AccumulateT> scaleAlphaVecBuf;

    if(useBias)
    {
        biasVec.resize(m);
        std::generate(biasVec.begin(), biasVec.end(), [&]() { return static_cast<AccumulateT>(randomGen()); });
        contraction.setUseBias(1);
        contraction.setBias(accumDtypeEnum, m, m);
    }

    if(useScaleAlphaVec)
    {
        size_t scaleAlphaVecLen = (factorDim == 0) ? m : n;
        scaleAlphaVecBuf.resize(scaleAlphaVecLen);
        std::generate(scaleAlphaVecBuf.begin(), scaleAlphaVecBuf.end(), [&]() { return static_cast<AccumulateT>(randomGen()); });
        contraction.setUseScaleAlphaVec(1);
        contraction.setScaleAlphaVec(accumDtypeEnum, scaleAlphaVecLen, factorDim);
    }

    // Random scale generator: magnitude in (1, 100], integer values to avoid rounding issues, sign random.
    // Excludes 0 and ±1 so missing/incorrect scaling is never masked.
    std::uniform_int_distribution<int> scaleDis(2, 100);
    auto scaleGen = [&]() -> AccumulateT {
        AccumulateT sign = binary_distribution(gen) ? AccumulateT(1) : AccumulateT(-1);
        int         mag  = scaleDis(gen);
        return sign * static_cast<AccumulateT>(mag);
    };

    std::vector<AccumulateT> scaleABuf;
    std::vector<AccumulateT> scaleBBuf;

    if(useScaleAB == "Scalar")
    {
        scaleABuf = {scaleGen()};
        scaleBBuf = {scaleGen()};
        // setUseScaleAB must be called before setScaleA/setScaleB,
        // because setScaleA/B silently skips tensor registration when
        // m_useScaleAB is still empty.
        // See: https://github.com/ROCm/rocm-libraries/issues/6541
        contraction.setUseScaleAB("Scalar");
        contraction.setScaleA(accumDtypeEnum, 1);
        contraction.setScaleB(accumDtypeEnum, 1);
    }
    else if(useScaleAB == "Vector")
    {
        scaleABuf.resize(m);
        scaleBBuf.resize(n);
        std::generate(scaleABuf.begin(), scaleABuf.end(), scaleGen);
        std::generate(scaleBBuf.begin(), scaleBBuf.end(), scaleGen);
        contraction.setUseScaleAB("Vector");
        contraction.setScaleA(accumDtypeEnum, m);
        contraction.setScaleB(accumDtypeEnum, n);
    }

    if(activation != ActivationType::None)
    {
        contraction.setActivationType(activation);
        contraction.setParams().setActivationEnum(activation);
    }

#ifndef _WIN32
    // MX scale setup (FP4 with mxBlockA/B > 0 only)
    [[maybe_unused]] std::vector<E8> mxsa, mxsb;

    if constexpr(isFP4)
    {
        if(mxBlockA > 0 || mxBlockB > 0)
        {
            // Use unpadded MX scale tensors so the columnMajorGemm reference
            // indexing matches: mxsa = {m, k/mxBlockA} with m as leading
            // stride (and analogous for B). Default padScaleTensor=true would
            // round M up to next 32 and K/mxBlockA/B up to next 8, breaking
            // the index math below.
            contraction.setMXScaleA(rocisa::DataType::E8, mxBlockA, /*saStride=*/{}, /*padScaleTensor=*/false);
            contraction.setMXScaleB(rocisa::DataType::E8, mxBlockB, /*sbStride=*/{}, /*padScaleTensor=*/false);

            size_t nmxsa = contraction.mxsa().totalLogicalElements();
            size_t nmxsb = contraction.mxsb().totalLogicalElements();

            if(nmxsa == 0 || nmxsb == 0)
            {
                std::cerr << "Error: MX scale tensor has zero elements (nmxsa=" << nmxsa
                          << ", nmxsb=" << nmxsb << ")" << std::endl;
                return 1;
            }

            mxsa.resize(nmxsa);
            mxsb.resize(nmxsb);

            // Distinct exponents in [0..7] so wrong indexing breaks validation
            std::uniform_int_distribution<> expDist(0, 7);
            for(size_t i = 0; i < nmxsa; i++)
                mxsa[i] = E8(std::ldexp(1.0f, expDist(gen)));
            for(size_t i = 0; i < nmxsb; i++)
                mxsb[i] = E8(std::ldexp(1.0f, expDist(gen)));
        }
    }
#endif

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(),
                             static_cast<AccumulateT>(alpha),
                             static_cast<AccumulateT>(beta));
    inputs.bias          = useBias ? biasVec.data() : nullptr;
    inputs.scaleAlphaVec = useScaleAlphaVec ? scaleAlphaVecBuf.data() : nullptr;
    inputs.scaleA        = (useScaleAB != "none") ? scaleABuf.data() : nullptr;
    inputs.scaleB        = (useScaleAB != "none") ? scaleBBuf.data() : nullptr;

#ifndef _WIN32
    if constexpr(isFP4)
    {
        inputs.mxsa = (mxBlockA > 0) ? mxsa.data() : nullptr;
        inputs.mxsb = (mxBlockB > 0) ? mxsb.data() : nullptr;
    }
#endif

    auto start = std::chrono::high_resolution_clock::now();

    if(tryFastPath && !TensileLite::Client::isFastPathEligible(contraction))
    {
        throw std::runtime_error(
            "--tryFastPath was requested but the problem is not eligible "
            "for the fast CPU GEMM path.");
    }

    // Execute the 'device under test'.
    // passing -1 for elementsToValidate ensures that the 'fast path' which we
    // currently want to test is maybe taken.
    int elementsToValidate = -1;
    TensileLite::Client::SolveGemmCPU(contraction, inputs, elementsToValidate, tryFastPath);

    auto                                      end      = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;

    if(validate)
    {
        std::cout << "Validating..." << std::endl;

        // Convert inputs to AccumulateT for the golden reference comparison.
        // For batched problems, A/B/C are batchCount slices of size numA/numB/numC
        // (column-major packed; batch stride = numA / numB / numC).
        size_t totalA = numA * batchCount;
        size_t totalB = numB * batchCount;
        size_t totalC = numC * batchCount;

        std::vector<AccumulateT> aRef, bRef;

#ifndef _WIN32
        if constexpr(isFP4)
        {
            aRef.resize(totalA);
            for(size_t i = 0; i < totalA; i++)
                aRef[i] = static_cast<AccumulateT>(a[i / 2].getElement(i % 2));
            bRef.resize(totalB);
            for(size_t i = 0; i < totalB; i++)
                bRef[i] = static_cast<AccumulateT>(b[i / 2].getElement(i % 2));
        }
        else if constexpr(std::is_same_v<InputAT, Float4x2> || std::is_same_v<InputBT, Float4x2>)
        {
            throw std::runtime_error("Mixed FP4 / non-FP4 input is not supported.");
        }
        else
#endif
        {
            aRef.resize(totalA);
            for(size_t i = 0; i < totalA; i++)
                aRef[i] = static_cast<AccumulateT>(a[i]);
            bRef.resize(totalB);
            for(size_t i = 0; i < totalB; i++)
                bRef[i] = static_cast<AccumulateT>(b[i]);
        }

        std::vector<AccumulateT> cRef(c.begin(), c.end());
        std::vector<AccumulateT> dRef(d.size());

        // If the storage type is wider than the compute-input type, the GPU
        // (and slow-path validator) quantize the operand down before the MAC.
        // Mirror that here so the golden GEMM reflects the same model.
        QuantizeFn quantA = (computeInputA != dtypeEnumA) ? quantizerFor(computeInputA) : nullptr;
        QuantizeFn quantB = (computeInputB != dtypeEnumB) ? quantizerFor(computeInputB) : nullptr;

#ifndef _WIN32
        size_t mxsaBatchStride = 0, mxsbBatchStride = 0;
        size_t mxsaStrideM = 0, mxsaStrideKBlk = 0;
        size_t mxsbStrideN = 0, mxsbStrideKBlk = 0;
        if constexpr(isFP4)
        {
            if(mxBlockA > 0)
            {
                auto const& mxsaTensor = contraction.mxsa();
                mxsaStrideM            = mxsaTensor.strides()[contraction.freeIndicesA()[0].i];
                mxsaStrideKBlk         = mxsaTensor.strides()[contraction.boundIndices()[0].a];
                mxsaBatchStride        = mxsaTensor.strides()[contraction.batchIndices()[0].a];
            }
            if(mxBlockB > 0)
            {
                auto const& mxsbTensor = contraction.mxsb();
                mxsbStrideN            = mxsbTensor.strides()[contraction.freeIndicesB()[0].i];
                mxsbStrideKBlk         = mxsbTensor.strides()[contraction.boundIndices()[0].b];
                mxsbBatchStride        = mxsbTensor.strides()[contraction.batchIndices()[0].b];
            }
        }
#endif

        // Run the golden reference per-batch.
        // When isTF32, use XFloat32 as MathOpAccumT so the golden ref
        // truncates each A/B element to 10-bit mantissa before multiply.
        auto runGoldenRef = [&](auto mathOpTag) {
            using MathOpT = decltype(mathOpTag);
            for(size_t batch = 0; batch < batchCount; ++batch)
            {
                const AccumulateT* aPtr = aRef.data() + batch * numA;
                const AccumulateT* bPtr = bRef.data() + batch * numB;
                const AccumulateT* cPtr = cRef.data() + batch * numC;
                AccumulateT*       dPtr = dRef.data() + batch * numC;

                columnMajorGemm<AccumulateT, MathOpT>(
                    aPtr,
                    bPtr,
                    cPtr,
                    dPtr,
                    m,
                    n,
                    k,
                    transA,
                    transB,
                    static_cast<AccumulateT>(
                        (useScaleAB == "Scalar") ? alpha * scaleABuf[0] * scaleBBuf[0] : alpha),
                    static_cast<AccumulateT>(beta),
                    useBias ? biasVec.data() : nullptr,
                    useScaleAlphaVec ? scaleAlphaVecBuf.data() : nullptr,
                    activation,
                    (useScaleAB == "Vector") ? scaleABuf.data() : nullptr,
                    (useScaleAB == "Vector") ? scaleBBuf.data() : nullptr,
                    factorDim,
                    quantA,
                    quantB
#ifndef _WIN32
                    ,
                    MXGemmOperand{
                        (isFP4 && mxBlockA > 0) ? mxsa.data() + batch * mxsaBatchStride : nullptr,
                        mxBlockA,
                        mxsaStrideM,
                        mxsaStrideKBlk},
                    MXGemmOperand{
                        (isFP4 && mxBlockB > 0) ? mxsb.data() + batch * mxsbBatchStride : nullptr,
                        mxBlockB,
                        mxsbStrideN,
                        mxsbStrideKBlk}
#endif
                );
            }
        };

        if(isTF32)
            runGoldenRef(XFloat32{});
        else
            runGoldenRef(AccumulateT{});

        // Compare results — reduced-precision types need wider tolerance.
        // TF32 loses 13 of 23 mantissa bits; errors accumulate over K.
        double tolerance = isFP4 ? 0.5 : (isTF32 ? 1.0 : 0.05);

        bool   allClose = true;
        double maxDiff  = 0.0;

        for(size_t i = 0; i < totalC; i++)
        {
            double valDut = static_cast<double>(d[i]);
            double valRef = static_cast<double>(dRef[i]);
            double diff   = std::abs(valDut - valRef);
            maxDiff       = std::max(maxDiff, diff);

            if(diff > tolerance)
            {
                allClose = false;
                if(i < 10)
                {
                    std::cout << "Mismatch at " << i << ": observed=" << valDut
                              << " expected=" << valRef << " diff=" << diff << std::endl;
                }
            }
        }

        if(allClose)
        {
            std::cout << "PASSED! (max diff: " << maxDiff << ")" << std::endl;
        }
        else
        {
            std::cout << "FAILED! (max diff: " << maxDiff << ")" << std::endl;
            return 1;
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{
    using namespace TensileLite;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")(
        "M", po::value<size_t>()->default_value(128), "Matrix M dimension")(
        "N", po::value<size_t>()->default_value(128), "Matrix N dimension")(
        "K", po::value<size_t>()->default_value(128), "Matrix K dimension")(
        "transA", po::value<bool>()->default_value(false), "Transpose A")(
        "transB", po::value<bool>()->default_value(false), "Transpose B")(
        "alpha", po::value<float>()->default_value(1.0f), "Alpha scalar")(
        "beta", po::value<float>()->default_value(0.0f), "Beta scalar")(
        "type", po::value<std::string>()->default_value("f32"), "Data type for A and B (f32, f64, tf32, f16, bf16, f8, bf8, f8fnuz, bf8fnuz, f4)")(
        "typeA", po::value<std::string>()->default_value(""), "Override A storage type (defaults to --type)")(
        "typeB", po::value<std::string>()->default_value(""), "Override B storage type (defaults to --type)")(
        "computeInputA", po::value<std::string>()->default_value(""), "Override A compute-input type for MAC (defaults to --typeA). Set smaller than storage to mimic kernels that quantize A.")(
        "computeInputB", po::value<std::string>()->default_value(""), "Override B compute-input type for MAC (defaults to --typeB). Set smaller than storage to mimic kernels that quantize B.")(
        "validate", po::value<bool>()->default_value(true), "Run validation against ref")(
        "tryFastPath", po::value<bool>()->default_value(false), "Use optimized path")(
        "bias", po::value<bool>()->default_value(false), "Enable bias vector")(
        "activation", po::value<std::string>()->default_value("none"), "Activation (none, relu)")(
        "scaleAlphaVec", po::value<bool>()->default_value(false), "Enable per-row alpha scaling")(
        "factorDim", po::value<int>()->default_value(0), "ScaleAlphaVec dimension: 0=row(M), 1=col(N)")(
        "useScaleAB", po::value<std::string>()->default_value("none"), "ScaleAB mode (none, Scalar, Vector)")(
        "mxBlockA", po::value<int>()->default_value(0), "MX block size for the A side (FP4 only, must be power of 2; both --mxBlockA and --mxBlockB must be set together)")(
        "mxBlockB", po::value<int>()->default_value(0), "MX block size for the B side (FP4 only, must be power of 2; both --mxBlockA and --mxBlockB must be set together)")(
        "batchCount", po::value<size_t>()->default_value(1), "Batch count (default 1)");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error parsing options: " << ex.what() << std::endl;
        return 1;
    }

    if(vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    size_t      m                = vm["M"].as<size_t>();
    size_t      n                = vm["N"].as<size_t>();
    size_t      k                = vm["K"].as<size_t>();
    bool        transA           = vm["transA"].as<bool>();
    bool        transB           = vm["transB"].as<bool>();
    float       alpha            = vm["alpha"].as<float>();
    float       beta             = vm["beta"].as<float>();
    std::string typeStr          = vm["type"].as<std::string>();
    std::string typeAStr         = vm["typeA"].as<std::string>();
    std::string typeBStr         = vm["typeB"].as<std::string>();
    if(typeAStr.empty()) typeAStr = typeStr;
    if(typeBStr.empty()) typeBStr = typeStr;
    std::string computeInputAStr = vm["computeInputA"].as<std::string>();
    std::string computeInputBStr = vm["computeInputB"].as<std::string>();
    if(computeInputAStr.empty()) computeInputAStr = typeAStr;
    if(computeInputBStr.empty()) computeInputBStr = typeBStr;

    auto strToDataType = [](const std::string& s, rocisa::DataType& out) -> bool {
        if(s == "f32")            { out = rocisa::DataType::Float;        return true; }
        if(s == "f64")            { out = rocisa::DataType::Double;       return true; }
        if(s == "tf32")           { out = rocisa::DataType::Float;        return true; }
        if(s == "f16")            { out = rocisa::DataType::Half;         return true; }
        if(s == "bf16")           { out = rocisa::DataType::BFloat16;     return true; }
#ifdef TENSILE_USE_FP8_BF8
        if(s == "f8")             { out = rocisa::DataType::Float8;       return true; }
        if(s == "bf8")            { out = rocisa::DataType::BFloat8;      return true; }
        if(s == "f8fnuz")         { out = rocisa::DataType::Float8_fnuz;  return true; }
        if(s == "bf8fnuz")        { out = rocisa::DataType::BFloat8_fnuz; return true; }
#endif
#ifndef _WIN32
        if(s == "f4")             { out = rocisa::DataType::Float4;       return true; }
#endif
        return false;
    };

    rocisa::DataType computeInputA, computeInputB;
    if(!strToDataType(computeInputAStr, computeInputA)) {
        std::cerr << "Unknown computeInputA: " << computeInputAStr << std::endl;
        return 1;
    }
    if(!strToDataType(computeInputBStr, computeInputB)) {
        std::cerr << "Unknown computeInputB: " << computeInputBStr << std::endl;
        return 1;
    }
    bool        validate         = vm["validate"].as<bool>();
    bool        tryFastPath      = vm["tryFastPath"].as<bool>();
    bool        useBias          = vm["bias"].as<bool>();
    std::string activationStr    = vm["activation"].as<std::string>();
    bool        useScaleAlphaVec = vm["scaleAlphaVec"].as<bool>();
    int         factorDim        = vm["factorDim"].as<int>();
    std::string useScaleAB       = vm["useScaleAB"].as<std::string>();
    int         mxBlockA         = vm["mxBlockA"].as<int>();
    int         mxBlockB         = vm["mxBlockB"].as<int>();
    size_t      batchCount       = vm["batchCount"].as<size_t>();

    if(mxBlockA < 0 || mxBlockB < 0)
    {
        std::cerr << "Error: mxBlockA/mxBlockB must be non-negative" << std::endl;
        return 1;
    }
    // One-sided MX is rejected (see review #1). When either per-side flag is
    // given, require both > 0.
    if((mxBlockA > 0) != (mxBlockB > 0))
    {
        std::cerr << "Error: --mxBlockA and --mxBlockB must both be > 0 "
                  << "(mxBlockA=" << mxBlockA << ", mxBlockB=" << mxBlockB << ")"
                  << std::endl;
        return 1;
    }

    if((mxBlockA > 0 || mxBlockB > 0) && typeStr != "f4")
    {
        std::cerr << "Error: mxBlockA/mxBlockB is only supported for type f4, not "
                  << typeStr << std::endl;
        return 1;
    }

#ifndef _WIN32
    if((typeAStr == "f4") != (typeBStr == "f4"))
    {
        std::cerr << "Error: mixed FP4 / non-FP4 input is not supported." << std::endl;
        return 1;
    }
#endif

    if(useScaleAB != "none" && useScaleAB != "Scalar" && useScaleAB != "Vector")
    {
        std::cerr << "Unknown useScaleAB mode: " << useScaleAB << std::endl;
        return 1;
    }

    if(factorDim != 0 && factorDim != 1)
    {
        std::cerr << "Invalid factorDim: " << factorDim << " (must be 0 or 1)" << std::endl;
        return 1;
    }

    ActivationType activation = ActivationType::None;
    if(activationStr == "relu")
        activation = ActivationType::Relu;
    else if(activationStr != "none")
    {
        std::cerr << "Unknown activation: " << activationStr << std::endl;
        return 1;
    }

    std::cout << "Running GEMM with: M=" << m << " N=" << n << " K=" << k
              << " TypeA=" << typeAStr << " TypeB=" << typeBStr
              << " ComputeInA=" << computeInputAStr << " ComputeInB=" << computeInputBStr
              << " FastPath=" << tryFastPath;
    if(typeAStr == "tf32" || typeBStr == "tf32")
        std::cout << " MathOp=XFloat32";
    std::cout << std::endl;

    // Dispatcher: pick A storage type, then B storage type. Each leaf calls
    // runGemm<A,B>(...). Asymmetric A/B is required to repro mixed-precision
    // bugs in the fast-path validator (e.g. F8N x Half).
    // tf32 = float storage + XFloat32 math-op. Dispatched as float with isTF32 flag.
    bool isTF32 = (typeAStr == "tf32" || typeBStr == "tf32");
    auto resolveAccumStorage = [](std::string& s) {
        if(s == "tf32") s = "f32";
    };
    resolveAccumStorage(typeAStr);
    resolveAccumStorage(typeBStr);

    auto dispatchB = [&](auto aTag) -> int {
        using AT = decltype(aTag);
        auto callB = [&](auto bTag) -> int {
            using BT = decltype(bTag);
#ifndef _WIN32
            constexpr bool isMixedFP4 = std::is_same_v<AT, Float4x2>
                                        != std::is_same_v<BT, Float4x2>;
            if constexpr(isMixedFP4)
            {
                std::cerr << "Error: mixed FP4 / non-FP4 input is not supported."
                          << std::endl;
                return 1;
            }
            else
#endif
            {
                // Promote AccumulateT to double when both A and B storage are double (f64).
                using AccT = std::conditional_t<
                    std::is_same_v<AT, double> && std::is_same_v<BT, double>, double, float>;
                return runGemm<AT, BT, AccT>(
                    m, n, k, transA, transB, alpha, beta, validate, tryFastPath,
                    useBias, activation, useScaleAlphaVec, useScaleAB, factorDim,
                    computeInputA, computeInputB, mxBlockA, mxBlockB, batchCount, isTF32);
            }
        };
        if(typeBStr == "f32")        return callB(float{});
        if(typeBStr == "f64")        return callB(double{});
        if(typeBStr == "f16")        return callB(Half{});
        if(typeBStr == "bf16")       return callB(BFloat16{});
#ifdef TENSILE_USE_FP8_BF8
        if(typeBStr == "f8")         return callB(Float8{});
        if(typeBStr == "bf8")        return callB(BFloat8{});
        if(typeBStr == "f8fnuz")     return callB(Float8_fnuz{});
        if(typeBStr == "bf8fnuz")    return callB(BFloat8_fnuz{});
#endif
#ifndef _WIN32
        if(typeBStr == "f4")         return callB(Float4x2{});
#endif
        std::cerr << "Unknown typeB: " << typeBStr << std::endl;
        return 1;
    };

    try
    {
        if(typeAStr == "f32")        return dispatchB(float{});
        if(typeAStr == "f64")        return dispatchB(double{});
        if(typeAStr == "f16")        return dispatchB(Half{});
        if(typeAStr == "bf16")       return dispatchB(BFloat16{});
#ifdef TENSILE_USE_FP8_BF8
        if(typeAStr == "f8")         return dispatchB(Float8{});
        if(typeAStr == "bf8")        return dispatchB(BFloat8{});
        if(typeAStr == "f8fnuz")     return dispatchB(Float8_fnuz{});
        if(typeAStr == "bf8fnuz")    return dispatchB(BFloat8_fnuz{});
#endif
#ifndef _WIN32
        if(typeAStr == "f4")         return dispatchB(Float4x2{});
#endif
        std::cerr << "Unknown typeA: " << typeAStr << std::endl;
        return 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
