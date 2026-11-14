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
#include <complex>
#include <iostream>
#include <limits>
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
 * # Int8 inputs, Int8 output, float alpha/beta
 * ./cpu_gemm_driver --type i8 --typeCD i8 --typeAlpha f32 --alpha 0.001953125 --validate
 *
 * # Int8 inputs, Int8 output, int32 alpha/beta, A/B in [-3, 3] (mostly unsaturated D)
 * ./cpu_gemm_driver --type i8 --typeCD i8 --int8Min -3 --int8Max 3 --validate --tryFastPath
 *
 * # Float8 x Half inputs (MAC on Half), Float8 output
 * ./cpu_gemm_driver --typeA f8 --typeB f16 --computeInputA f16 --typeCD f8 --validate
 *
 * # Complex float (C_C_C) with conjugated A
 * ./cpu_gemm_driver --type c --transA --conjA --validate --tryFastPath
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
    struct TypeTraits<int8_t>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Int8;
    };

    template <>
    struct TypeTraits<int32_t>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Int32;
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

    template <>
    struct TypeTraits<std::complex<float>>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::ComplexFloat;
    };

    template <>
    struct TypeTraits<std::complex<double>>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::ComplexDouble;
    };

    template <typename T>
    constexpr bool isComplexV = TypeInfo<T>::IsComplex;

    // 8-bit float storage family: OCP (Float8, BFloat8) or fnuz (Float8_fnuz, BFloat8_fnuz).
    enum class Float8Family
    {
        None,
        OCP,
        Fnuz
    };

    template <typename T>
    constexpr Float8Family float8Family()
    {
        if constexpr(std::is_same_v<T, TensileLite::Float8>
                     || std::is_same_v<T, TensileLite::BFloat8>)
            return Float8Family::OCP;
        else if constexpr(std::is_same_v<T, TensileLite::Float8_fnuz>
                          || std::is_same_v<T, TensileLite::BFloat8_fnuz>)
            return Float8Family::Fnuz;
        else
            return Float8Family::None;
    }

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
    //   AccumT         — epilogue type (float, double or int32_t): alpha, beta,
    //                    the scale/bias vectors, C and D are in this type. It is
    //                    also the K-loop accumulation type unless MacT is given.
    //                    Complex inputs use their own type (std::complex<float>
    //                    or std::complex<double>) for everything, as the
    //                    reference's C_C_C/Z_Z_Z do; A or B is conjugated by the
    //                    caller (no scale vectors, bias or activation).
    //   OperandMathOpT — per-operand math type used immediately before multiply.
    //                    XFloat32, for example, truncates float operands to a
    //                    10-bit mantissa.
    //   MacT           — K-loop multiply-accumulate type; A and B are passed in
    //                    it. Int8 inputs use int64_t: the sum is exact, and it
    //                    must fit in int32_t (the GPU's Int8 accumulator) or the
    //                    call throws. The sum is then converted to AccumT once,
    //                    before alpha.
    //
    // An int32_t epilogue is computed in int64_t and every intermediate is
    // checked against the int32_t range: the reference computes it in int32_t,
    // where overflow is undefined, so a test that overflows throws instead of
    // comparing undefined values.
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
        case rocisa::DataType::Float8:        return &quantizeThroughNarrow<TensileLite::Float8>;
        case rocisa::DataType::BFloat8:       return &quantizeThroughNarrow<TensileLite::BFloat8>;
        case rocisa::DataType::Float8_fnuz:   return &quantizeThroughNarrow<TensileLite::Float8_fnuz>;
        case rocisa::DataType::BFloat8_fnuz:  return &quantizeThroughNarrow<TensileLite::BFloat8_fnuz>;
        default:
            throw std::runtime_error("Unsupported compute-input type for golden GEMM quantizer.");
        }
    }

    // A golden operand rounded through its compute-input type (quantize == nullptr:
    // unchanged). Complex operands have no compute-input override.
    template <typename MacT>
    MacT quantizeOperand(QuantizeFn quantize, MacT v)
    {
        if(!quantize)
            return v;
        if constexpr(isComplexV<MacT>)
            throw std::runtime_error("Complex operands cannot be quantized.");
        else
            return static_cast<MacT>(quantize(static_cast<float>(v)));
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

    // Golden arithmetic type for an epilogue type: int32_t epilogues run in
    // int64_t, and each intermediate goes through checkedInt32 (see below).
    template <typename AccumT>
    using GoldenWideT = std::conditional_t<std::is_integral_v<AccumT>, int64_t, AccumT>;

    // Throws if an integer intermediate does not fit in int32_t; floating-point
    // values pass through unchanged.
    template <typename T>
    T checkedInt32(T v)
    {
        if constexpr(std::is_integral_v<T>)
        {
            if(v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
                throw std::runtime_error(
                    "CPU reference GEMM: integer value " + std::to_string(v)
                    + " overflows the int32_t accumulator/epilogue; use smaller inputs.");
        }
        return v;
    }

    template <typename AccumT = float, typename OperandMathOpT = AccumT, typename MacT = AccumT>
    void columnMajorGemm(const MacT*    a,
                         const MacT*    b,
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

        using WideT = GoldenWideT<AccumT>;

#ifndef _WIN32
        const bool hasMX = validateMXGemmOperands(mxA, mxB);
#endif

        for(size_t i = 0; i < m; i++)
        {
            for(size_t j = 0; j < n; j++)
            {
                MacT sum = MacT(0);

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
                        MacT blockSum = MacT(0);
                        for(size_t t = 0; t < step; t++)
                        {
                            size_t l    = lBase + t;
                            MacT   aVal = a[i * strideAM + l * strideAK];
                            MacT   bVal = b[l * strideBK + j * strideBN];
                            aVal        = quantizeOperand(quantizeA, aVal);
                            bVal        = quantizeOperand(quantizeB, bVal);
                            blockSum
                                += static_cast<MacT>(static_cast<OperandMathOpT>(aVal))
                                   * static_cast<MacT>(static_cast<OperandMathOpT>(bVal));
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
                        sum += static_cast<MacT>(blockSum * mxScale);
                    }
                }
                else
#endif
                {
                    for(size_t l = 0; l < k; l++)
                    {
                        MacT aVal = a[i * strideAM + l * strideAK];
                        MacT bVal = b[l * strideBK + j * strideBN];
                        aVal      = quantizeOperand(quantizeA, aVal);
                        bVal      = quantizeOperand(quantizeB, bVal);
                        sum += static_cast<MacT>(static_cast<OperandMathOpT>(aVal))
                               * static_cast<MacT>(static_cast<OperandMathOpT>(bVal));
                    }
                }

                // alpha * sum, beta * C and bias are added in separate steps,
                // each rounded on its own, as in the slow path.
                WideT effectiveAlpha = alpha;
                if(scaleAVec)
                    effectiveAlpha
                        = checkedInt32(effectiveAlpha * static_cast<WideT>(scaleAVec[i]));
                if(scaleBVec)
                    effectiveAlpha
                        = checkedInt32(effectiveAlpha * static_cast<WideT>(scaleBVec[j]));
                if(scaleAlphaVec)
                    effectiveAlpha = checkedInt32(
                        effectiveAlpha * static_cast<WideT>(scaleAlphaVec[factorDim == 0 ? i : j]));

                WideT result = checkedInt32(
                    effectiveAlpha * static_cast<WideT>(static_cast<AccumT>(checkedInt32(sum))));
                result
                    = checkedInt32(result + checkedInt32(static_cast<WideT>(beta) * c[i + j * m]));

                if(biasVec)
                    result = checkedInt32(result + static_cast<WideT>(biasVec[i]));

                if(activation == ActivationType::Relu)
                {
                    if constexpr(!isComplexV<AccumT>)
                        result = std::max(WideT(0), result);
                    else
                        throw std::runtime_error(
                            "CPU reference GEMM: no activation for complex types.");
                }
                else
                {
                    assert(activation == ActivationType::None);
                }

                d[i + j * m] = static_cast<AccumT>(result);
            }
        }
    }

    // Converts a golden epilogue value to the C/D storage type the way the
    // reference's SaturateCast<DType> (client/src/Reference.cpp) does:
    //   int8_t:  from a floating-point epilogue, round to nearest even, then
    //            clamp to [-128, 127]; from an int32_t epilogue, clamp only.
    //   int32_t: plain static_cast (truncation from floating point, no clamp).
    //            Out-of-range conversion is undefined, so it throws here.
    //   others:  plain static_cast (Float8/BFloat8: round to nearest even, saturate
    //            to the largest finite value).
    template <typename OutT, typename AccumT>
    OutT saturateCastLikeReference(AccumT val)
    {
        if constexpr(std::is_same_v<OutT, int8_t>)
        {
            if constexpr(std::is_floating_point_v<AccumT>)
                val = std::nearbyint(val); // round to even
            if(val > static_cast<AccumT>(127))
                val = static_cast<AccumT>(127);
            else if(val < static_cast<AccumT>(-128))
                val = static_cast<AccumT>(-128);
            return static_cast<OutT>(val);
        }
        else if constexpr(std::is_same_v<OutT, int32_t> && std::is_floating_point_v<AccumT>)
        {
            if(!(static_cast<double>(val) > -2147483649.0
                 && static_cast<double>(val) < 2147483648.0))
                throw std::runtime_error("CPU reference GEMM: D value "
                                         + std::to_string(static_cast<double>(val))
                                         + " is out of the int32_t range; use smaller inputs.");
            return static_cast<OutT>(val);
        }
        else
        {
            return static_cast<OutT>(val);
        }
    }
}

/*
 * Main templated runner.
 * Handles memory allocation, data initialization, execution, and validation.
 *
 * InputT: The C++ type used for storage of A and B matrices (e.g. float, half).
 * AccumulateT: alpha/beta type, which is also the reference's epilogue type
 *              (float, double, int32_t for Int8 inputs, or the complex input type).
 * OutputT: C/D storage type (defaults to AccumulateT).
 * conjA/conjB: TensorOp::ComplexConjugate on A/B (complex inputs only).
 */
template <typename InputAT,
          typename InputBT     = InputAT,
          typename AccumulateT = float,
          typename OutputT     = AccumulateT>
int runGemm(size_t         m,
            size_t         n,
            size_t         k,
            bool           transA,
            bool           transB,
            float          alpha,
            float          beta,
            bool           validate,
            bool           injectValidationFailure,
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
            bool               isTF32        = false,
            bool               outputAmaxD   = false,
            int                int8Min       = -128,
            int                int8Max       = 127,
            bool               conjA         = false,
            bool               conjB         = false)
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

    constexpr bool isInt8 = std::is_same_v<InputAT, int8_t>;
    static_assert(isInt8 == std::is_same_v<InputBT, int8_t>,
                  "Int8 input storage must be used for both A and B, or neither.");
    // Golden K-loop type: exact integer accumulation for Int8 (see columnMajorGemm).
    using MacT = std::conditional_t<isInt8, int64_t, AccumulateT>;

    constexpr bool isComplex = isComplexV<InputAT>;
    static_assert(!isComplex
                      || (std::is_same_v<InputAT, InputBT> && std::is_same_v<InputAT, AccumulateT>
                          && std::is_same_v<InputAT, OutputT>),
                  "Complex inputs use one complex type for A, B, C, D and alpha/beta.");

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

    // alpha/beta (and bias, scale vectors, AMAX-D) use AccumulateT; C/D use OutputT.
    constexpr rocisa::DataType accumDtypeEnum  = TypeTraits<AccumulateT>::value;
    constexpr rocisa::DataType outputDtypeEnum = TypeTraits<OutputT>::value;

    // Define the contraction problem (geometry, strides, types)
    ContractionProblemGemm contraction
        = ContractionProblemGemm::GEMM_Strides(transA,
                                               transB,
                                               dtypeEnumA,
                                               dtypeEnumB,
                                               outputDtypeEnum,
                                               outputDtypeEnum,
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

    if(conjA)
        contraction.setAOps({TensorOp::ComplexConjugate()});
    if(conjB)
        contraction.setBOps({TensorOp::ComplexConjugate()});

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
    std::vector<OutputT>     c(numC * batchCount);
    std::vector<OutputT>     d(numC * batchCount);

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
    //
    // Int8 inputs are drawn from [int8Min, int8Max], by default the full range
    // [-128, 127], so the integer accumulation, large sums and saturating Int8
    // output are exercised. A narrower range keeps most Int8 outputs in range.
    //
    // Complex A, B and C get +-1 in both the real and the imaginary part, so a
    // missing or misplaced conjugation changes the result.
    size_t                                seed = 42;
    std::mt19937                          gen(seed);
    std::uniform_int_distribution<>       binary_distribution(0, 1);
    std::uniform_real_distribution<float> realDist(-1.0f, 1.0f);

    auto randomGen = [&]() { return binary_distribution(gen) ? 1.0f : -1.0f; };
    // A complex T with +-1 parts, real part drawn first.
    auto randomComplexGen = [&](auto tag) {
        using T       = decltype(tag);
        const auto re = static_cast<typename T::value_type>(randomGen());
        const auto im = static_cast<typename T::value_type>(randomGen());
        return T(re, im);
    };

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
            if constexpr(std::is_same_v<T, int8_t>)
            {
                std::uniform_int_distribution<int> int8Dist(int8Min, int8Max);
                std::generate(vec.begin(), vec.end(),
                              [&]() { return static_cast<T>(int8Dist(gen)); });
            }
            else if constexpr(isComplexV<T>)
            {
                std::generate(vec.begin(), vec.end(), [&]() { return randomComplexGen(T{}); });
            }
            else if(quantizes)
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
    if constexpr(isComplex)
        std::generate(c.begin(), c.end(), [&]() { return randomComplexGen(OutputT{}); });
    else
        std::generate(c.begin(), c.end(), [&]() { return static_cast<OutputT>(randomGen()); });

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

    // AMAX-D: the reference writes max |D| over every batch as one AccumulateT.
    // Start from -1, which no absolute maximum can be, so an unwritten value
    // fails validation.
    AccumulateT amaxD = static_cast<AccumulateT>(-1);
    if(outputAmaxD)
    {
        contraction.setOutputAmaxD(true);
        contraction.setAmaxD(accumDtypeEnum, /*isOutput=*/true);
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
    inputs.amaxD         = outputAmaxD ? &amaxD : nullptr;

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
    if constexpr(!isComplex)
    {
        if(outputAmaxD)
            std::cout << "AmaxD: " << static_cast<double>(amaxD) << std::endl;
    }

    if(injectValidationFailure)
    {
        if(d.empty())
        {
            std::cerr << "Error: cannot inject validation failure into empty D" << std::endl;
            return 1;
        }
        if constexpr(std::is_integral_v<OutputT>)
            d[0] = static_cast<OutputT>(d[0] > 0 ? d[0] - 16 : d[0] + 16); // stays in range
        else if constexpr(float8Family<OutputT>() != Float8Family::None)
            // d[0] + 16 could saturate back to d[0]; flip the sign instead (16 is exact).
            d[0] = static_cast<OutputT>(static_cast<float>(d[0]) > 0.0f ? -16.0f : 16.0f);
        else
            d[0] = static_cast<OutputT>(static_cast<AccumulateT>(d[0])
                                        + static_cast<AccumulateT>(16.0));
    }

    if(validate)
    {
        std::cout << "Validating..." << std::endl;

        // Convert A/B to the golden K-loop type (MacT) and C to AccumulateT.
        // For batched problems, A/B/C are batchCount slices of size numA/numB/numC
        // (column-major packed; batch stride = numA / numB / numC).
        size_t totalA = numA * batchCount;
        size_t totalB = numB * batchCount;
        size_t totalC = numC * batchCount;

        std::vector<MacT> aRef, bRef;

#ifndef _WIN32
        if constexpr(isFP4)
        {
            aRef.resize(totalA);
            for(size_t i = 0; i < totalA; i++)
                aRef[i] = static_cast<MacT>(a[i / 2].getElement(i % 2));
            bRef.resize(totalB);
            for(size_t i = 0; i < totalB; i++)
                bRef[i] = static_cast<MacT>(b[i / 2].getElement(i % 2));
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
                aRef[i] = static_cast<MacT>(a[i]);
            bRef.resize(totalB);
            for(size_t i = 0; i < totalB; i++)
                bRef[i] = static_cast<MacT>(b[i]);
            if constexpr(isComplex)
            {
                if(conjA)
                    std::transform(aRef.begin(), aRef.end(), aRef.begin(), [](MacT v) {
                        return std::conj(v);
                    });
                if(conjB)
                    std::transform(bRef.begin(), bRef.end(), bRef.begin(), [](MacT v) {
                        return std::conj(v);
                    });
            }
        }

        std::vector<AccumulateT> cRef(c.size());
        for(size_t i = 0; i < c.size(); i++)
            cRef[i] = static_cast<AccumulateT>(c[i]);
        // dRef holds the golden epilogue result before the cast to OutputT
        // (AMAX-D is computed on it); D is compared after the cast.
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

        // Scalar ScaleAB folds into alpha (alpha * scaleA * scaleB, in the
        // reference's order); an int32_t alpha is range-checked like the other
        // int32_t intermediates.
        using GoldenAlphaT     = GoldenWideT<AccumulateT>;
        GoldenAlphaT wideAlpha = static_cast<GoldenAlphaT>(static_cast<AccumulateT>(alpha));
        if(useScaleAB == "Scalar")
        {
            wideAlpha = checkedInt32(wideAlpha * static_cast<GoldenAlphaT>(scaleABuf[0]));
            wideAlpha = checkedInt32(wideAlpha * static_cast<GoldenAlphaT>(scaleBBuf[0]));
        }
        const AccumulateT goldenAlpha = static_cast<AccumulateT>(wideAlpha);

        // Run the golden reference per-batch.
        // When isTF32, use XFloat32 as OperandMathOpT so the golden ref
        // truncates each A/B element to 10-bit mantissa before multiply.
        auto runGoldenRef = [&](auto mathOpTag) {
            using MathOpT = decltype(mathOpTag);
            for(size_t batch = 0; batch < batchCount; ++batch)
            {
                const MacT*        aPtr = aRef.data() + batch * numA;
                const MacT*        bPtr = bRef.data() + batch * numB;
                const AccumulateT* cPtr = cRef.data() + batch * numC;
                AccumulateT*       dPtr = dRef.data() + batch * numC;

                columnMajorGemm<AccumulateT, MathOpT, MacT>(
                    aPtr,
                    bPtr,
                    cPtr,
                    dPtr,
                    m,
                    n,
                    k,
                    transA,
                    transB,
                    goldenAlpha,
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

        // Int8 and complex: no operand math-op type (the MAC type itself).
        if constexpr(isInt8 || isComplex)
            runGoldenRef(MacT{});
        else if(isTF32)
            runGoldenRef(XFloat32{});
        else
            runGoldenRef(AccumulateT{});

        // Compare results — reduced-precision types need wider tolerance.
        // TF32 loses 13 of 23 mantissa bits; errors accumulate over K.
        // Int8 inputs must match exactly: the golden accumulates exactly and
        // applies the epilogue in the reference's epilogue type, rounding each
        // step as the reference does. (It folds scaleAlphaVec into alpha before
        // multiplying by the sum, the reference multiplies after; the driver's
        // scaleAlphaVec values are +-1, so both give the same value.)
        // Float8/BFloat8 outputs must match exactly too: with this driver's data every
        // K-loop sum is exact in float. Products are +-1, or (one operand quantized to
        // e4m3 from (-1, 1), the other +-1) multiples of 2^-9 (fnuz: 2^-10), whose sums
        // are exact while |partial sum| < 2^15 (fnuz: 2^14). C, bias and scaleAlphaVec
        // are +-1 and the ScaleAB scales integers, so the golden and the reference round
        // the same epilogue operations to the same float values before the cast, and a
        // wrong rounding or saturation is off by at least one F8 step.
        constexpr bool isFloat8Output = float8Family<OutputT>() != Float8Family::None;

        double tolerance = [&]() {
            if(isInt8 || isFloat8Output)
                return 0.0;
            if(isFP4)
                return 0.5;
            if(isTF32)
                return 1.0;
            return 0.05;
        }();

        bool   allClose = true;
        double maxDiff  = 0.0;

        // Equal values (also equal infinities) and NaN on both sides match; a NaN on
        // one side only (e.g. an unsaturated Float8 overflow) is an infinite difference,
        // which the tolerance check below would otherwise miss.
        auto valueDiff = [](double valDut, double valRef) {
            const bool bothNaN = std::isnan(valDut) && std::isnan(valRef);
            double     diff    = (valDut == valRef || bothNaN) ? 0.0 : std::abs(valDut - valRef);
            if(std::isnan(diff))
                diff = std::numeric_limits<double>::infinity();
            return diff;
        };

        for(size_t i = 0; i < totalC; i++)
        {
            // Complex: the larger of the real-part and imaginary-part differences.
            const auto valDut = d[i];
            const auto valRef = saturateCastLikeReference<OutputT>(dRef[i]);
            double     diff;
            if constexpr(isComplex)
                diff = std::max(valueDiff(valDut.real(), valRef.real()),
                                valueDiff(valDut.imag(), valRef.imag()));
            else
                diff = valueDiff(static_cast<double>(valDut), static_cast<double>(valRef));
            maxDiff = std::max(maxDiff, diff);

            if(diff > tolerance)
            {
                allClose = false;
                if(i < 10)
                {
                    std::cout << "Mismatch at " << i << ": observed=";
                    if constexpr(isComplex)
                        std::cout << valDut << " expected=" << valRef;
                    else
                        std::cout << static_cast<double>(valDut)
                                  << " expected=" << static_cast<double>(valRef);
                    std::cout << " diff=" << diff << std::endl;
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

        // AMAX-D is checked against the golden D, whichever CPU path computed it.
        if constexpr(!isComplex)
        {
            if(outputAmaxD)
            {
                double goldenAmaxD = 0.0;
                for(size_t i = 0; i < totalC; i++)
                    goldenAmaxD = std::max(goldenAmaxD, std::abs(static_cast<double>(dRef[i])));

                double amaxDiff = std::abs(static_cast<double>(amaxD) - goldenAmaxD);
                if(amaxDiff <= tolerance)
                {
                    std::cout << "AmaxD PASSED! (observed=" << static_cast<double>(amaxD)
                              << " expected=" << goldenAmaxD << ")" << std::endl;
                }
                else
                {
                    std::cout << "AmaxD FAILED! (observed=" << static_cast<double>(amaxD)
                              << " expected=" << goldenAmaxD << " diff=" << amaxDiff << ")"
                              << std::endl;
                    return 1;
                }
            }
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
        "type", po::value<std::string>()->default_value("f32"), "Data type for A and B (f32, f64, tf32, f16, bf16, f8, bf8, f8fnuz, bf8fnuz, f4, i8, c = complex f32, z = complex f64)")(
        "typeA", po::value<std::string>()->default_value(""), "Override A storage type (defaults to --type)")(
        "typeB", po::value<std::string>()->default_value(""), "Override B storage type (defaults to --type)")(
        "typeCD", po::value<std::string>()->default_value(""), "C/D (output) type: f32, f64, f16, bf16, i8, i32, f8, bf8, f8fnuz, bf8fnuz. Defaults to i32 for i8 inputs, f64 for f64 inputs, the input type for c/z inputs, else f32; i8 inputs take any of i8, i32, f16, bf16; 8-bit float inputs of one family (OCP or fnuz) take that family's f8/bf8, f8 x f16 takes f8 (f8fnuz x f16: f8fnuz); other inputs only the default")(
        "typeAlpha", po::value<std::string>()->default_value(""), "Alpha/beta type, also the reference's epilogue type: f32, f64, i32. Defaults to i32 when --typeCD is i8 or i32, f64 for f64 inputs, the input type for c/z inputs (real --alpha/--beta), else f32; i32 needs integer --alpha/--beta")(
        "computeInputA", po::value<std::string>()->default_value(""), "Override A compute-input type for MAC (defaults to --typeA). Set smaller than storage to mimic kernels that quantize A.")(
        "computeInputB", po::value<std::string>()->default_value(""), "Override B compute-input type for MAC (defaults to --typeB). Set smaller than storage to mimic kernels that quantize B.")(
        "validate", po::value<bool>()->default_value(true), "Run validation against ref")(
        "injectValidationFailure", po::value<bool>()->default_value(false), "Perturb D before validation (negative-test hook)")(
        "tryFastPath", po::value<bool>()->default_value(false), "Use optimized path")(
        "bias", po::value<bool>()->default_value(false), "Enable bias vector")(
        "activation", po::value<std::string>()->default_value("none"), "Activation (none, relu)")(
        "scaleAlphaVec", po::value<bool>()->default_value(false), "Enable per-row alpha scaling")(
        "factorDim", po::value<int>()->default_value(0), "ScaleAlphaVec dimension: 0=row(M), 1=col(N)")(
        "useScaleAB", po::value<std::string>()->default_value("none"), "ScaleAB mode (none, Scalar, Vector)")(
        "mxBlockA", po::value<int>()->default_value(0), "MX block size for the A side (FP4 only, must be power of 2; both --mxBlockA and --mxBlockB must be set together)")(
        "mxBlockB", po::value<int>()->default_value(0), "MX block size for the B side (FP4 only, must be power of 2; both --mxBlockA and --mxBlockB must be set together)")(
        "batchCount", po::value<size_t>()->default_value(1), "Batch count (default 1)")(
        "outputAmaxD", po::value<bool>()->default_value(false), "Also output AMAX-D (max |D|) and validate it against the golden GEMM")(
        "int8Min", po::value<int>()->default_value(-128), "Smallest i8 A/B value (i8 inputs only; default -128)")(
        "int8Max", po::value<int>()->default_value(127), "Largest i8 A/B value (i8 inputs only; default 127)")(
        "conjA", po::value<bool>()->default_value(false), "Conjugate A (TensorOp::ComplexConjugate; c/z inputs only)")(
        "conjB", po::value<bool>()->default_value(false), "Conjugate B (TensorOp::ComplexConjugate; c/z inputs only)");

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
        if(s == "f8")             { out = rocisa::DataType::Float8;       return true; }
        if(s == "bf8")            { out = rocisa::DataType::BFloat8;      return true; }
        if(s == "f8fnuz")         { out = rocisa::DataType::Float8_fnuz;  return true; }
        if(s == "bf8fnuz")        { out = rocisa::DataType::BFloat8_fnuz; return true; }
        if(s == "i8")             { out = rocisa::DataType::Int8;         return true; }
        if(s == "c")              { out = rocisa::DataType::ComplexFloat; return true; }
        if(s == "z")              { out = rocisa::DataType::ComplexDouble; return true; }
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
    bool        validate                 = vm["validate"].as<bool>();
    bool        injectValidationFailure  = vm["injectValidationFailure"].as<bool>();
    bool        tryFastPath              = vm["tryFastPath"].as<bool>();
    bool        useBias                  = vm["bias"].as<bool>();
    std::string activationStr            = vm["activation"].as<std::string>();
    bool        useScaleAlphaVec         = vm["scaleAlphaVec"].as<bool>();
    int         factorDim                = vm["factorDim"].as<int>();
    std::string useScaleAB               = vm["useScaleAB"].as<std::string>();
    int         mxBlockA                 = vm["mxBlockA"].as<int>();
    int         mxBlockB                 = vm["mxBlockB"].as<int>();
    size_t      batchCount               = vm["batchCount"].as<size_t>();
    bool        outputAmaxD              = vm["outputAmaxD"].as<bool>();
    int         int8Min                  = vm["int8Min"].as<int>();
    int         int8Max                  = vm["int8Max"].as<int>();
    bool        conjA                    = vm["conjA"].as<bool>();
    bool        conjB                    = vm["conjB"].as<bool>();
    const bool  typeAIsTF32              = (typeAStr == "tf32");
    const bool  typeBIsTF32              = (typeBStr == "tf32");
    const bool  isTF32                   = typeAIsTF32 && typeBIsTF32;

    if(typeAIsTF32 != typeBIsTF32)
    {
        std::cerr << "Error: tf32 is an F32 math-op mode and must be used for both "
                  << "inputs or neither (typeA=" << typeAStr << ", typeB=" << typeBStr
                  << ")" << std::endl;
        return 1;
    }

    // Int8 inputs: A and B both i8, MAC on Int8 (no compute-input override).
    // The supported C/D and alpha/beta types are the reference's Int8 type
    // combinations (TypedGemm_I8_*); other inputs keep C/D and alpha/beta at
    // their defaults.
    const bool typeAIsInt8 = (typeAStr == "i8");
    const bool typeBIsInt8 = (typeBStr == "i8");
    const bool isInt8      = typeAIsInt8 && typeBIsInt8;
    if(typeAIsInt8 != typeBIsInt8)
    {
        std::cerr << "Error: mixed i8 / non-i8 input is not supported (typeA=" << typeAStr
                  << ", typeB=" << typeBStr << ")" << std::endl;
        return 1;
    }
    if(isInt8 && (computeInputAStr != "i8" || computeInputBStr != "i8"))
    {
        std::cerr << "Error: i8 inputs do not support a computeInput override" << std::endl;
        return 1;
    }
    const bool isFullInt8Range = (int8Min == -128 && int8Max == 127);
    if(isInt8 ? !(-128 <= int8Min && int8Min <= int8Max && int8Max <= 127) : !isFullInt8Range)
    {
        std::cerr << "Error: --int8Min/--int8Max need i8 inputs and -128 <= int8Min <= int8Max"
                  << " <= 127 (int8Min=" << int8Min << ", int8Max=" << int8Max << ")" << std::endl;
        return 1;
    }

    // Complex inputs (c, z): A and B of one complex type, which is also the C/D and
    // alpha/beta type (the reference's C_C_C / Z_Z_Z), optionally conjugated
    // (--conjA, --conjB). The reference's fast path takes no bias, ScaleAB,
    // scaleAlphaVec, activation or AMAX-D for them (the slow path reads complex bias
    // and scale values as zero), and the golden has no complex activation.
    const bool typeAIsComplex = (typeAStr == "c" || typeAStr == "z");
    const bool typeBIsComplex = (typeBStr == "c" || typeBStr == "z");
    const bool isComplex      = typeAIsComplex && typeBIsComplex;
    if(typeAIsComplex != typeBIsComplex || (isComplex && typeAStr != typeBStr))
    {
        std::cerr << "Error: complex inputs need the same complex type for A and B (typeA="
                  << typeAStr << ", typeB=" << typeBStr << ")" << std::endl;
        return 1;
    }
    if(isComplex && (computeInputAStr != typeAStr || computeInputBStr != typeBStr))
    {
        std::cerr << "Error: complex inputs do not support a computeInput override" << std::endl;
        return 1;
    }
    if(isComplex
       && (useBias || activationStr != "none" || useScaleAlphaVec || useScaleAB != "none"
           || outputAmaxD))
    {
        std::cerr << "Error: complex inputs do not support --bias, --activation, --scaleAlphaVec,"
                  << " --useScaleAB or --outputAmaxD" << std::endl;
        return 1;
    }
    if(!isComplex && (conjA || conjB))
    {
        std::cerr << "Error: --conjA/--conjB need complex inputs (c or z)" << std::endl;
        return 1;
    }

    const bool        isF64        = (typeAStr == "f64" && typeBStr == "f64");
    std::string       typeCDStr    = vm["typeCD"].as<std::string>();
    std::string       typeAlphaStr = vm["typeAlpha"].as<std::string>();
    // Default C/D and alpha/beta type for non-integer outputs: the input type for f64
    // and complex inputs, else f32.
    const std::string defaultType  = isF64 ? "f64" : (isComplex ? typeAStr : "f32");
    const std::string defaultCD    = isInt8 ? "i32" : defaultType;
    if(typeCDStr.empty())
        typeCDStr = defaultCD;
    const bool        isIntCD      = (typeCDStr == "i8" || typeCDStr == "i32");
    const std::string defaultAlpha = isIntCD ? "i32" : defaultType;
    if(typeAlphaStr.empty())
        typeAlphaStr = defaultAlpha;

    // Float8/BFloat8 C/D: the input types are checked at dispatch (runFloat8Output).
    const bool isFloat8CD = (typeCDStr == "f8" || typeCDStr == "bf8" || typeCDStr == "f8fnuz"
                             || typeCDStr == "bf8fnuz");

    rocisa::DataType typeCD = rocisa::DataType::None;
    if(isFloat8CD)
        strToDataType(typeCDStr, typeCD);

    if(!isInt8 && ((typeCDStr != defaultCD && !isFloat8CD) || typeAlphaStr != defaultAlpha))
    {
        std::cerr << "Error: --typeCD " << typeCDStr << " / --typeAlpha " << typeAlphaStr
                  << " is only supported for i8 inputs, or f8/bf8/f8fnuz/bf8fnuz C/D with f32"
                  << " alpha (defaults here: " << defaultCD << " / " << defaultAlpha << ")"
                  << std::endl;
        return 1;
    }

    if(typeAlphaStr == "i32")
    {
        auto isInt32Value = [](float v) {
            return std::nearbyint(v) == v && v >= -2147483648.0f && v < 2147483648.0f;
        };
        if(!isInt32Value(alpha) || !isInt32Value(beta))
        {
            std::cerr << "Error: --typeAlpha i32 needs integer --alpha and --beta (alpha="
                      << alpha << ", beta=" << beta << ")" << std::endl;
            return 1;
        }
    }

    if(mxBlockA < 0 || mxBlockB < 0)
    {
        std::cerr << "Error: mxBlockA/mxBlockB must be non-negative" << std::endl;
        return 1;
    }

    if(injectValidationFailure && !validate)
    {
        std::cerr << "Error: --injectValidationFailure requires --validate" << std::endl;
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
              << " TypeCD=" << typeCDStr << " TypeAlpha=" << typeAlphaStr
              << " ComputeInA=" << computeInputAStr << " ComputeInB=" << computeInputBStr
              << " FastPath=" << tryFastPath;
    if(isTF32)
        std::cout << " MathOp=XFloat32";
    std::cout << std::endl;

    // Dispatcher: pick A storage type, then B storage type. Each leaf calls
    // runGemm<A,B>(...). Asymmetric A/B is required to repro mixed-precision
    // bugs in the fast-path validator (e.g. F8N x Half).
    // tf32 = float storage + XFloat32 math-op. Dispatched as float with isTF32 flag.
    auto resolveAccumStorage = [](std::string& s) {
        if(s == "tf32") s = "f32";
    };
    resolveAccumStorage(typeAStr);
    resolveAccumStorage(typeBStr);

    auto runTyped = [&](auto aTag, auto bTag, auto accTag, auto outTag) -> int {
        return runGemm<decltype(aTag), decltype(bTag), decltype(accTag), decltype(outTag)>(
            m, n, k, transA, transB, alpha, beta,
            validate, injectValidationFailure, tryFastPath,
            useBias, activation, useScaleAlphaVec, useScaleAB, factorDim,
            computeInputA, computeInputB, mxBlockA, mxBlockB, batchCount, isTF32,
            outputAmaxD, int8Min, int8Max, conjA, conjB);
    };

    // Float8/BFloat8 C/D with float alpha/beta: the reference's Float8-output type ids.
    // A and B are 8-bit floats of one family, with a Float8 or BFloat8 C/D of that
    // family; or Float8 x Half in either order, with a C/D of the Float8 operand's type.
    // The compute-input types must match one of those type ids, too (e.g. f8 x f16
    // with --computeInputA f16 or --computeInputB f8), or the reference rejects them.
    auto runFloat8Output = [&](auto aTag, auto bTag) -> int {
        using AT                       = decltype(aTag);
        using BT                       = decltype(bTag);
        constexpr Float8Family familyA = float8Family<AT>();
        constexpr Float8Family familyB = float8Family<BT>();
        constexpr bool aIsFloat8 = std::is_same_v<AT, Float8> || std::is_same_v<AT, Float8_fnuz>;
        constexpr bool bIsFloat8 = std::is_same_v<BT, Float8> || std::is_same_v<BT, Float8_fnuz>;
        if constexpr(familyA != Float8Family::None && familyA == familyB)
        {
            constexpr bool isOCP = (familyA == Float8Family::OCP);
            using F8T            = std::conditional_t<isOCP, Float8, Float8_fnuz>;
            using BF8T           = std::conditional_t<isOCP, BFloat8, BFloat8_fnuz>;
            if(typeCD == TypeTraits<F8T>::value)
                return runTyped(AT{}, BT{}, float{}, F8T{});
            if(typeCD == TypeTraits<BF8T>::value)
                return runTyped(AT{}, BT{}, float{}, BF8T{});
        }
        else if constexpr(aIsFloat8 && std::is_same_v<BT, Half>)
        {
            if(typeCD == TypeTraits<AT>::value)
                return runTyped(AT{}, BT{}, float{}, AT{});
        }
        else if constexpr(std::is_same_v<AT, Half> && bIsFloat8)
        {
            if(typeCD == TypeTraits<BT>::value)
                return runTyped(AT{}, BT{}, float{}, BT{});
        }
        std::cerr << "Error: unsupported --typeCD " << typeCDStr << " for typeA=" << typeAStr
                  << " typeB=" << typeBStr
                  << " (supported: 8-bit float A and B of one family with that family's f8 or"
                  << " bf8 C/D, or f8 x f16 / f8fnuz x f16 in either order with an f8 / f8fnuz"
                  << " C/D)" << std::endl;
        return 1;
    };

    auto dispatchB = [&](auto aTag) -> int {
        using AT = decltype(aTag);
        auto callB = [&](auto bTag) -> int {
            using BT = decltype(bTag);
            constexpr bool isAInt8 = std::is_same_v<AT, int8_t>;
            constexpr bool isBInt8 = std::is_same_v<BT, int8_t>;
            constexpr bool isAComplex = isComplexV<AT>;
            constexpr bool isBComplex = isComplexV<BT>;
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
            if constexpr(isAInt8 != isBInt8)
            {
                std::cerr << "Error: mixed i8 / non-i8 input is not supported." << std::endl;
                return 1;
            }
            else if constexpr(isAComplex || isBComplex)
            {
                // C_C_C / Z_Z_Z: the complex type for every tensor and alpha/beta.
                if constexpr(std::is_same_v<AT, BT>)
                {
                    return runTyped(AT{}, BT{}, AT{}, AT{});
                }
                else
                {
                    std::cerr << "Error: complex inputs need the same complex type for A and B."
                              << std::endl;
                    return 1;
                }
            }
            else if constexpr(isAInt8)
            {
                // C/D type x alpha/beta type: the reference's TypedGemm_I8_* ids.
                if(typeCDStr == "i8" && typeAlphaStr == "i32")
                    return runTyped(AT{}, BT{}, int32_t{}, int8_t{}); // I8_I8_I32
                if(typeCDStr == "i32" && typeAlphaStr == "i32")
                    return runTyped(AT{}, BT{}, int32_t{}, int32_t{}); // I8_I32_I32
                if(typeCDStr == "i32" && typeAlphaStr == "f32")
                    return runTyped(AT{}, BT{}, float{}, int32_t{}); // I8_I32_S
                if(typeCDStr == "i8" && typeAlphaStr == "f32")
                    return runTyped(AT{}, BT{}, float{}, int8_t{}); // I8_I8_S
                if(typeCDStr == "f16" && typeAlphaStr == "f32")
                    return runTyped(AT{}, BT{}, float{}, Half{}); // I8_H_S
                if(typeCDStr == "bf16" && typeAlphaStr == "f32")
                    return runTyped(AT{}, BT{}, float{}, BFloat16{}); // I8_B_S
                std::cerr << "Error: unsupported i8 type combination --typeCD " << typeCDStr
                          << " --typeAlpha " << typeAlphaStr
                          << " (supported: i8/i32, i32/i32, i32/f32, i8/f32, f16/f32, bf16/f32)"
                          << std::endl;
                return 1;
            }
            else
            {
                if(isFloat8CD)
                    return runFloat8Output(AT{}, BT{});
                // Promote AccumulateT to double when both A and B storage are double (f64).
                using AccT = std::conditional_t<
                    std::is_same_v<AT, double> && std::is_same_v<BT, double>, double, float>;
                return runTyped(AT{}, BT{}, AccT{}, AccT{});
            }
        };
        if(typeBStr == "f32")        return callB(float{});
        if(typeBStr == "f64")        return callB(double{});
        if(typeBStr == "f16")        return callB(Half{});
        if(typeBStr == "bf16")       return callB(BFloat16{});
        if(typeBStr == "f8")         return callB(Float8{});
        if(typeBStr == "bf8")        return callB(BFloat8{});
        if(typeBStr == "f8fnuz")     return callB(Float8_fnuz{});
        if(typeBStr == "bf8fnuz")    return callB(BFloat8_fnuz{});
        if(typeBStr == "i8")         return callB(int8_t{});
        if(typeBStr == "c")          return callB(std::complex<float>{});
        if(typeBStr == "z")          return callB(std::complex<double>{});
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
        if(typeAStr == "f8")         return dispatchB(Float8{});
        if(typeAStr == "bf8")        return dispatchB(BFloat8{});
        if(typeAStr == "f8fnuz")     return dispatchB(Float8_fnuz{});
        if(typeAStr == "bf8fnuz")    return dispatchB(BFloat8_fnuz{});
        if(typeAStr == "i8")         return dispatchB(int8_t{});
        if(typeAStr == "c")          return dispatchB(std::complex<float>{});
        if(typeAStr == "z")          return dispatchB(std::complex<double>{});
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
