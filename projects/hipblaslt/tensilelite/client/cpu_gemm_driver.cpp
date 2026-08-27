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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ProgramOptions.hpp"
#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
#include "rocisa/include/enum.hpp"
#include <Tensile/Activation.hpp>
#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>
#include <roc/host_validation/adapters/tensilelite/TensileDataGeneration.hpp>
#include <roc/host_validation/validation.hpp>

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

    enum class InitializationStream : uint64_t
    {
        OperandA   = 0,
        OperandB   = 1,
        MatrixC    = 2,
        Bias       = 3,
        ScaleAlpha = 4,
        ScaleA     = 5,
        ScaleB     = 6,
    };

    template <typename T>
    void generateValues(std::vector<T>&                                   destination,
                        roc::host_validation::GenerationRecipe::Component component,
                        InitializationStream                              stream)
    {
        using namespace roc::host_validation;
        static_assert(std::is_trivially_copyable_v<T>);

        const auto recipe
            = GenerationRecipe::realOnly(std::move(component),
                                         tensilelite_adapter::dataInitializationSettings(
                                             42, static_cast<uint64_t>(stream)));

        Tensor generated(toHostValidationScalarType(TypeTraits<T>::value),
                         Shape{destination.size()});
        generate(generated, recipe);
        const std::span<std::byte> destinationBytes
            = std::as_writable_bytes(std::span<T>(destination));
        if(generated.rawEncodedBackingStorage().size() != destinationBytes.size())
            throw std::runtime_error("Generated tensor storage does not match CPU driver type.");
        std::memcpy(destinationBytes.data(), generated.rawEncodedBackingStorage().data(), destinationBytes.size());
    }

    template <typename Destination, typename Source>
    std::vector<Destination> convertValues(std::span<const Source> source)
    {
        using namespace roc::host_validation;
        static_assert(std::is_trivially_copyable_v<Source>);
        static_assert(std::is_trivially_copyable_v<Destination>);

        const Tensor sourceTensor(toHostValidationScalarType(TypeTraits<Source>::value),
                                  Layout::contiguousLastDimensionFastest(Shape{source.size()}),
                                  std::as_bytes(source));
        const Tensor converted
            = sourceTensor.copyConvertedTo(toHostValidationScalarType(TypeTraits<Destination>::value));

        std::vector<Destination>   result(source.size());
        const std::span<std::byte> resultBytes
            = std::as_writable_bytes(std::span<Destination>(result));
        if(converted.rawEncodedBackingStorage().size() != resultBytes.size())
            throw std::runtime_error("Converted tensor storage does not match CPU driver type.");
        std::memcpy(resultBytes.data(), converted.rawEncodedBackingStorage().data(), resultBytes.size());
        return result;
    }

#ifndef _WIN32
    template <typename Destination>
    std::vector<Destination> convertPackedFloat4Values(std::span<const Float4x2> source,
                                                       size_t logicalElementsPerBatch,
                                                       size_t batchCount)
    {
        using namespace roc::host_validation;
        static_assert(sizeof(Float4x2) == 1);
        static_assert(std::is_trivially_copyable_v<Destination>);

        const size_t storageBytesPerBatch = (logicalElementsPerBatch + 1) / 2;
        if(source.size_bytes() != storageBytesPerBatch * batchCount)
            throw std::runtime_error("Packed FP4 storage does not match CPU driver batches.");

        std::vector<Destination>         result(logicalElementsPerBatch * batchCount);
        const std::span<const std::byte> sourceBytes = std::as_bytes(source);
        for(size_t batch = 0; batch < batchCount; ++batch)
        {
            const Tensor sourceTensor(
                ScalarType::Float4E2M1,
                Layout::contiguousLastDimensionFastest(Shape{logicalElementsPerBatch}),
                sourceBytes.subspan(batch * storageBytesPerBatch, storageBytesPerBatch));
            const Tensor converted
                = sourceTensor.copyConvertedTo(toHostValidationScalarType(TypeTraits<Destination>::value));
            const std::span<std::byte> destinationBytes
                = std::as_writable_bytes(std::span<Destination>(result).subspan(
                    batch * logicalElementsPerBatch, logicalElementsPerBatch));
            if(converted.rawEncodedBackingStorage().size() != destinationBytes.size())
                throw std::runtime_error(
                    "Converted FP4 tensor storage does not match CPU driver type.");
            std::memcpy(
                destinationBytes.data(), converted.rawEncodedBackingStorage().data(), destinationBytes.size());
        }
        return result;
    }
#endif
}

/*
 * Main templated runner.
 * Handles memory allocation, data initialization, execution, and validation.
 *
 * InputT: The C++ type used for storage of A and B matrices (e.g. float, half).
 * AccumulateT: The type used for accumulation (currently restricted to float).
 */
template <typename InputAT, typename InputBT = InputAT, typename AccumulateT = float>
int runGemm(size_t             m,
            size_t             n,
            size_t             k,
            bool               transA,
            bool               transB,
            float              alpha,
            float              beta,
            bool               validate,
            bool               injectValidationFailure,
            bool               tryFastPath,
            bool               useBias,
            ActivationType     activation,
            bool               useScaleAlphaVec,
            const std::string& useScaleAB,
            int                factorDim,
            rocisa::DataType   computeInputA = rocisa::DataType::None,
            rocisa::DataType   computeInputB = rocisa::DataType::None,
            int                mxBlockA      = 0,
            int                mxBlockB      = 0,
            size_t             batchCount    = 1,
            int                elementsToValidate = -1,
            bool               isTF32        = false)
{
    if(batchCount == 0)
    {
        std::cerr << "Error: batchCount (" << batchCount << ") must be >= 1" << std::endl;
        return 1;
    }
    constexpr rocisa::DataType dtypeEnumA = TypeTraits<InputAT>::value;
    constexpr rocisa::DataType dtypeEnumB = TypeTraits<InputBT>::value;
    if(computeInputA == rocisa::DataType::None)
        computeInputA = dtypeEnumA;
    if(computeInputB == rocisa::DataType::None)
        computeInputB = dtypeEnumB;

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
            if(b <= 0)
                return 0;
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
                std::cerr << "Error: K (" << k << ") must be a multiple of " << name << " (" << b
                          << ")" << std::endl;
                return 1;
            }
            return 0;
        };
        if(int rc = checkSide("mxBlockA", mxBlockA))
            return rc;
        if(int rc = checkSide("mxBlockB", mxBlockB))
            return rc;

        // Asymmetric MX (mxBlockA != mxBlockB) is only supported on the fast
        // path. The production slow path's MX inner loop uses a single scale
        // per max(mxBlockA, mxBlockB)-sized segment, which collapses the
        // smaller-blocked side's per-segment scales onto the first one and
        // produces wrong results. Reject the combination at the driver rather
        // than ship a known-wrong slow path.
        if(mxBlockA != mxBlockB && !tryFastPath)
        {
            std::cerr << "Error: asymmetric MX (mxBlockA=" << mxBlockA
                      << " != mxBlockB=" << mxBlockB << ") is only supported on the fast path "
                      << "(use --tryFastPath)." << std::endl;
            return 1;
        }
    }

    // Calculate strides assuming standard column-major packed storage
    size_t lda = transA ? k : m;
    size_t ldb = transB ? n : k;
    size_t ldc = m;

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

    const bool partialValidation = elementsToValidate > 0
                                   && static_cast<size_t>(elementsToValidate) < d.size();
    std::vector<size_t>              selectedValidationIndices;
    std::vector<std::vector<size_t>> selectedValidationIndicesByBatch;
    if(partialValidation)
    {
        const auto selection = roc::host_validation::OutputSelection::primeStride(
            d.size(), d.size(), static_cast<size_t>(elementsToValidate));
        selectedValidationIndices = selection.indices(d.size());
        selectedValidationIndicesByBatch.resize(batchCount);
        for(const size_t globalIndex : selectedValidationIndices)
        {
            const size_t batch       = globalIndex / numC;
            const size_t batchIndex  = globalIndex % numC;
            const size_t row         = batchIndex % m;
            const size_t column      = batchIndex / m;
            const size_t logicalIndex = row * n + column;
            selectedValidationIndicesByBatch[batch].push_back(logicalIndex);
        }
    }

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

#ifndef _WIN32
    if constexpr(isFP4)
    {
        // Full E2M1-representable value set: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6.
        // Drawing from the entire grid (not just the powers of two near zero)
        // exercises the MX-scale path with values whose products span more of
        // the FP4 range, while still being exactly representable.
        const auto fp4Values = roc::host_validation::GenerationRecipe::candidateSet(
            {.values
             = {-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}});
        // Pack 2 logical FP4 values per byte (Float4x2). When the logical
        // element count is odd, the second slot of the last byte has no
        // element behind it — it's padding. We must still initialize that
        // slot to a valid FP4 value (we use 0), because the fast-path
        // ShadowBuffer FP4 decoder unconditionally reads both slots of every
        // byte (the guard is on the *write* back, not the read), and reading
        // uninitialized memory would be UB.
        auto initFp4Operand = [&](auto&                vec,
                                  size_t               numLogical,
                                  InitializationStream initializationStream) {
            const size_t       storagePerBatch = (numLogical + 1) / 2;
            std::vector<float> logicalValues(numLogical * batchCount);
            generateValues(logicalValues, fp4Values, initializationStream);
            for(size_t batch = 0; batch < batchCount; ++batch)
            {
                auto packed = roc::host_validation::Tensor::copyValuesWithConversion(
                    roc::host_validation::ScalarType::Float4E2M1,
                    roc::host_validation::Shape{numLogical},
                    std::span<const float>(logicalValues).subspan(batch * numLogical, numLogical));
                std::memcpy(reinterpret_cast<std::byte*>(vec.data()) + batch * storagePerBatch,
                            packed.rawEncodedBackingStorage().data(),
                            storagePerBatch);
            }
        };
        initFp4Operand(a, numA, InitializationStream::OperandA);
        initFp4Operand(b, numB, InitializationStream::OperandB);
    }
    else
#endif
    {
        auto initOperand
            = [&](auto& vec, bool quantizes, InitializationStream initializationStream) {
                  using T = typename std::decay_t<decltype(vec)>::value_type;
#ifndef _WIN32
                  if constexpr(std::is_same_v<T, Float4x2>)
                  {
                      // FP4 mixed-input init unsupported in this branch; the FP4-only
                      // path above handles the pure FP4 case. Mixed FP4/non-FP4
                      // dispatch is rejected at the dispatcher level.
                      throw std::runtime_error("Mixed FP4 / non-FP4 input is not supported.");
                  }
                  else
#endif
                      if(quantizes)
                  {
                      // Values representable in storage but not on the compute-input grid -
                      // for storage=Half/compute=F8N, values like 0.7 that Half holds
                      // exactly but F8N rounds to 0.625 or 0.75.
                      generateValues(vec,
                                     roc::host_validation::GenerationRecipe::uniformReal(
                                         {.lower = -1.0, .upper = 1.0}),
                                     initializationStream);
                  }
                  else
                  {
                      generateValues(vec,
                                     roc::host_validation::GenerationRecipe::candidateSet(
                                         {.values = {-1.0, 1.0}}),
                                     initializationStream);
                  }
              };

        bool quantizesA = (sizeof(InputAT) > 1) && (computeInputA != dtypeEnumA);
        bool quantizesB = (sizeof(InputBT) > 1) && (computeInputB != dtypeEnumB);
        initOperand(a, quantizesA, InitializationStream::OperandA);
        initOperand(b, quantizesB, InitializationStream::OperandB);
    }
    const auto binaryValues
        = roc::host_validation::GenerationRecipe::candidateSet({.values = {-1.0, 1.0}});
    generateValues(c, binaryValues, InitializationStream::MatrixC);

    // Optional feature buffers — typed as AccumulateT so the slow path's
    // GetValue(alphaType, ...) reads the correct byte width.
    std::vector<AccumulateT> biasVec;
    std::vector<AccumulateT> scaleAlphaVecBuf;

    if(useBias)
    {
        biasVec.resize(m * batchCount);
        generateValues(biasVec, binaryValues, InitializationStream::Bias);
        contraction.setUseBias(1);
        contraction.setBias(accumDtypeEnum, m, m);
    }

    if(useScaleAlphaVec)
    {
        size_t scaleAlphaVecLen = (factorDim == 0) ? m : n;
        scaleAlphaVecBuf.resize(scaleAlphaVecLen);
        generateValues(scaleAlphaVecBuf, binaryValues, InitializationStream::ScaleAlpha);
        contraction.setUseScaleAlphaVec(1);
        contraction.setScaleAlphaVec(accumDtypeEnum, scaleAlphaVecLen, factorDim);
    }

    std::vector<AccumulateT> scaleABuf;
    std::vector<AccumulateT> scaleBBuf;
    std::vector<double>      scaleCandidates;
    for(int magnitude = 2; magnitude <= 100; ++magnitude)
    {
        scaleCandidates.push_back(-magnitude);
        scaleCandidates.push_back(magnitude);
    }
    const auto scaleValues = roc::host_validation::GenerationRecipe::candidateSet(
        {.values = std::move(scaleCandidates)});

    if(useScaleAB == "Scalar")
    {
        scaleABuf.resize(1);
        scaleBBuf.resize(1);
        generateValues(scaleABuf, scaleValues, InitializationStream::ScaleA);
        generateValues(scaleBBuf, scaleValues, InitializationStream::ScaleB);
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
        generateValues(scaleABuf, scaleValues, InitializationStream::ScaleA);
        generateValues(scaleBBuf, scaleValues, InitializationStream::ScaleB);
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
            // Use unpadded MX scale tensors so the shared reference indexing
            // matches: mxsa = {m, k/mxBlockA} with m as leading
            // stride (and analogous for B). Default padScaleTensor=true would
            // round M up to next 32 and K/mxBlockA/B up to next 8, breaking
            // the index math below.
            contraction.setMXScaleA(
                rocisa::DataType::E8, mxBlockA, /*saStride=*/{}, /*padScaleTensor=*/false);
            contraction.setMXScaleB(
                rocisa::DataType::E8, mxBlockB, /*sbStride=*/{}, /*padScaleTensor=*/false);

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

            // Distinct exponents in [0..7] so wrong indexing breaks validation.
            auto fillScale = [](std::vector<E8>& values, uint64_t stream) {
                roc::host_validation::Tensor generated(roc::host_validation::ScalarType::E8M0,
                                                       roc::host_validation::Shape{values.size()});
                const auto recipe = roc::host_validation::GenerationRecipe::realOnly(
                    roc::host_validation::GenerationRecipe::randomEncodedExponent(
                        {.lowerUnbiasedExponent = 0, .upperUnbiasedExponent = 7}),
                    roc::host_validation::tensilelite_adapter::dataInitializationSettings(
                        42, stream));
                roc::host_validation::generate(generated, recipe);
                std::memcpy(values.data(), generated.rawEncodedBackingStorage().data(), generated.rawEncodedBackingStorage().size());
            };
            fillScale(mxsa, 0);
            fillScale(mxsb, 1);
        }
    }
#endif

    ContractionInputs inputs(a.data(),
                             b.data(),
                             c.data(),
                             d.data(),
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

    const auto execution = tryFastPath
                               ? TensileLite::Client::ReferenceGemmExecution::BlockedRequired
                               : TensileLite::Client::ReferenceGemmExecution::Pointwise;
    const auto runInfo = TensileLite::Client::tryReferenceGemm(
        contraction, inputs, elementsToValidate, execution);
    if(!runInfo)
    {
        if(tryFastPath)
            throw std::runtime_error(
                "--tryFastPath requires execution by the blocked CPU GEMM backend, "
                "but the normalized request is unsupported.");
        throw std::runtime_error("The normalized request is unsupported by the "
                                 "pointwise CPU GEMM backend.");
    }

    auto                                      end      = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;

    if(injectValidationFailure)
    {
        if(d.empty())
        {
            std::cerr << "Error: cannot inject validation failure into empty D" << std::endl;
            return 1;
        }
        d[0] += static_cast<AccumulateT>(16.0);
    }

    if(validate)
    {
        std::cout << "Validating..." << std::endl;

        // Convert inputs to AccumulateT for the golden reference comparison.
        // Batched FP4 storage is converted one slice at a time so odd logical
        // sizes retain their per-batch byte padding.

        std::vector<AccumulateT> aRef, bRef;

#ifndef _WIN32
        if constexpr(isFP4)
        {
            aRef = convertPackedFloat4Values<AccumulateT>(
                std::span<const Float4x2>(a), numA, batchCount);
            bRef = convertPackedFloat4Values<AccumulateT>(
                std::span<const Float4x2>(b), numB, batchCount);
        }
        else if constexpr(std::is_same_v<InputAT, Float4x2> || std::is_same_v<InputBT, Float4x2>)
        {
            throw std::runtime_error("Mixed FP4 / non-FP4 input is not supported.");
        }
        else
#endif
        {
            aRef = convertValues<AccumulateT>(std::span<const InputAT>(a));
            bRef = convertValues<AccumulateT>(std::span<const InputBT>(b));
        }

        std::vector<AccumulateT> cRef = convertValues<AccumulateT>(std::span<const AccumulateT>(c));
        std::vector<AccumulateT> dRef(d.size());

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

        // Run the runtime-typed golden reference per batch. The host buffers
        // remain caller-owned; the validation component sees affine tensor
        // views and product-independent scalar types.
        for(size_t batch = 0; batch < batchCount; ++batch)
        {
            using namespace roc::host_validation;

            const AccumulateT* aPtr = aRef.data() + batch * numA;
            const AccumulateT* bPtr = bRef.data() + batch * numB;
            const AccumulateT* cPtr = cRef.data() + batch * numC;
            AccumulateT*       dPtr = dRef.data() + batch * numC;

            const ptrdiff_t strideARow    = transA ? static_cast<ptrdiff_t>(k) : 1;
            const ptrdiff_t strideAColumn = transA ? 1 : static_cast<ptrdiff_t>(m);
            const ptrdiff_t strideBRow    = transB ? static_cast<ptrdiff_t>(n) : 1;
            const ptrdiff_t strideBColumn = transB ? 1 : static_cast<ptrdiff_t>(k);

            GemmOperand operandA(
                Tensor::copyNativeStorage<AccumulateT>(Layout(Shape{m, k}, {strideARow, strideAColumn}),
                                                std::span<const AccumulateT>(aPtr, numA)));
            GemmOperand operandB(
                Tensor::copyNativeStorage<AccumulateT>(Layout(Shape{k, n}, {strideBRow, strideBColumn}),
                                                std::span<const AccumulateT>(bPtr, numB)));
            if(computeInputA != dtypeEnumA)
                operandA.computeType = toHostValidationScalarType(computeInputA);
            if(computeInputB != dtypeEnumB)
                operandB.computeType = toHostValidationScalarType(computeInputB);

            Tensor outputTensor = Tensor::copyNativeStorage<AccumulateT>(
                Layout(Shape{m, n}, {1, static_cast<ptrdiff_t>(m)}),
                std::span<AccumulateT>(dPtr, numC));
            GemmRequest problem(
                std::move(operandA),
                std::move(operandB),
                Tensor::copyNativeStorage<AccumulateT>(Layout(Shape{m, n}, {1, static_cast<ptrdiff_t>(m)}),
                                                std::span<const AccumulateT>(cPtr, numC)),
                outputTensor,
                nativeScalarType<AccumulateT>);
            if(partialValidation)
                problem.outputSelection =
                    OutputSelection::explicitIndices(selectedValidationIndicesByBatch[batch]);

            problem.epilogue.alpha = static_cast<double>(
                (useScaleAB == "Scalar") ? alpha * scaleABuf[0] * scaleBBuf[0] : alpha);
            problem.epilogue.beta       = static_cast<double>(beta);
            problem.epilogue.activation = toHostValidationActivation(activation);
            problem.mathMode            = isTF32 ? MathMode::XFloat32 : MathMode::Default;

            if(useBias)
            {
                problem.epilogue.bias = VectorBinding{
                    Tensor::copyNativeStorage<AccumulateT>(
                        Layout::contiguousLastDimensionFastest(Shape{m}),
                        std::span<const AccumulateT>(biasVec.data() + batch * m, m)),
                    MatrixAxis::Row};
            }
            if(useScaleAlphaVec)
            {
                problem.epilogue.scaleAlpha
                    = VectorBinding{Tensor::copyNativeStorage<AccumulateT>(
                                        Layout::contiguousLastDimensionFastest(Shape{scaleAlphaVecBuf.size()}),
                                        std::span<const AccumulateT>(scaleAlphaVecBuf)),
                                    factorDim == 0 ? MatrixAxis::Row : MatrixAxis::Column};
            }
            if(useScaleAB == "Vector")
            {
                problem.epilogue.scaleA
                    = Tensor::copyNativeStorage<AccumulateT>(Layout::contiguousLastDimensionFastest(Shape{scaleABuf.size()}),
                                                      std::span<const AccumulateT>(scaleABuf));
                problem.epilogue.scaleB
                    = Tensor::copyNativeStorage<AccumulateT>(Layout::contiguousLastDimensionFastest(Shape{scaleBBuf.size()}),
                                                      std::span<const AccumulateT>(scaleBBuf));
            }

            std::optional<Tensor> runtimeBlockScaleA;
            std::optional<Tensor> runtimeBlockScaleB;
#ifndef _WIN32
            if constexpr(isFP4)
            {
                if(mxBlockA > 0)
                {
                    const size_t scaleABase  = batch * mxsaBatchStride;
                    const size_t scaleBBase  = batch * mxsbBatchStride;
                    const size_t blockCountA = k / static_cast<size_t>(mxBlockA)
                                               + (k % static_cast<size_t>(mxBlockA) != 0 ? 1 : 0);
                    const size_t blockCountB = k / static_cast<size_t>(mxBlockB)
                                               + (k % static_cast<size_t>(mxBlockB) != 0 ? 1 : 0);
                    runtimeBlockScaleA.emplace(ScalarType::Float32,
                                               Layout(Shape{m, blockCountA},
                                                      {static_cast<ptrdiff_t>(mxsaStrideM),
                                                       static_cast<ptrdiff_t>(mxsaStrideKBlk)}));
                    runtimeBlockScaleB.emplace(ScalarType::Float32,
                                               Layout(Shape{n, blockCountB},
                                                      {static_cast<ptrdiff_t>(mxsbStrideN),
                                                       static_cast<ptrdiff_t>(mxsbStrideKBlk)}));
                    for(size_t row = 0; row < m; ++row)
                    {
                        for(size_t block = 0; block < blockCountA; ++block)
                        {
                            const size_t index
                                = scaleABase + row * mxsaStrideM + block * mxsaStrideKBlk;
                            runtimeBlockScaleA->storeFrom({row, block},
                                                          static_cast<float>(mxsa[index]));
                        }
                    }
                    for(size_t column = 0; column < n; ++column)
                    {
                        for(size_t block = 0; block < blockCountB; ++block)
                        {
                            const size_t index
                                = scaleBBase + column * mxsbStrideN + block * mxsbStrideKBlk;
                            runtimeBlockScaleB->storeFrom({column, block},
                                                          static_cast<float>(mxsb[index]));
                        }
                    }
                    problem.a.blockScale
                        = BlockScaleBinding{*runtimeBlockScaleA, static_cast<size_t>(mxBlockA)};
                    problem.b.blockScale
                        = BlockScaleBinding{*runtimeBlockScaleB, static_cast<size_t>(mxBlockB)};
                }
            }
#endif

            referenceGemm(problem);
            std::memcpy(dPtr, outputTensor.rawEncodedBackingStorage().data(), outputTensor.rawEncodedBackingStorage().size());
        }

        // Compare results — reduced-precision types need wider tolerance.
        // TF32 loses 13 of 23 mantissa bits; errors accumulate over K.
        double tolerance = [&]() {
            if(isFP4)
                return 0.5;
            if(isTF32)
                return 1.0;
            return 0.05;
        }();

        const auto comparisonType = toHostValidationScalarType(TypeTraits<AccumulateT>::value);
        const auto comparisonLayout
            = roc::host_validation::Layout::contiguousLastDimensionFastest(roc::host_validation::Shape{d.size()});
        roc::host_validation::ComparisonOptions comparisonOptions{
            .absoluteTolerance     = tolerance,
            .relativeTolerance     = 0.0,
            .maxReportedMismatches = 10,
        };
        if(partialValidation)
        {
            comparisonOptions.selection.stride =
                selectedValidationIndices.size() > 1
                    ? selectedValidationIndices[1] - selectedValidationIndices[0]
                    : 1;
            comparisonOptions.selection.maxElements = selectedValidationIndices.size();
        }
        const auto comparison = roc::host_validation::compare(
            roc::host_validation::Tensor::copyEncodedBackingStorage(
                comparisonType, comparisonLayout, std::as_bytes(std::span<const AccumulateT>(d))),
            roc::host_validation::Tensor::copyEncodedBackingStorage(
                comparisonType, comparisonLayout,
                std::as_bytes(std::span<const AccumulateT>(dRef))),
            comparisonOptions);

        for(const auto& mismatch : comparison.reportedMismatches)
        {
            std::cout << "Mismatch at " << mismatch.index << ": observed=" << mismatch.observed
                      << " expected=" << mismatch.expected
                      << " diff=" << mismatch.absoluteDifference << std::endl;
        }

        if(comparison.passed())
        {
            std::cout << "PASSED! (max diff: " << comparison.maxAbsoluteDifference << ")"
                      << std::endl;
        }
        else
        {
            std::cout << "FAILED! (max diff: " << comparison.maxAbsoluteDifference << ")"
                      << std::endl;
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
        "transA",
        po::value<bool>()->default_value(false)->implicit_value(true),
        "Transpose A")(
        "transB",
        po::value<bool>()->default_value(false)->implicit_value(true),
        "Transpose B")(
        "alpha", po::value<float>()->default_value(1.0f), "Alpha scalar")(
        "beta", po::value<float>()->default_value(0.0f), "Beta scalar")(
        "type",
        po::value<std::string>()->default_value("f32"),
        "Data type for A and B (f32, f64, tf32, f16, bf16, f8, bf8, f8fnuz, bf8fnuz, f4)")(
        "typeA",
        po::value<std::string>()->default_value(""),
        "Override A storage type (defaults to --type)")(
        "typeB",
        po::value<std::string>()->default_value(""),
        "Override B storage type (defaults to --type)")(
        "computeInputA",
        po::value<std::string>()->default_value(""),
        "Override A compute-input type for MAC (defaults to --typeA). Set smaller than storage to "
        "mimic kernels that quantize A.")(
        "computeInputB",
        po::value<std::string>()->default_value(""),
        "Override B compute-input type for MAC (defaults to --typeB). Set smaller than storage to "
        "mimic kernels that quantize B.")(
        "validate",
        po::value<bool>()->default_value(true)->implicit_value(true),
        "Run validation against ref")(
        "injectValidationFailure",
        po::value<bool>()->default_value(false)->implicit_value(true),
        "Perturb D before validation (negative-test hook)")("tryFastPath",
                                                            po::value<bool>()
                                                                ->default_value(false)
                                                                ->implicit_value(true),
                                                            "Require blocked reference execution")(
        "bias",
        po::value<bool>()->default_value(false)->implicit_value(true),
        "Enable bias vector")(
        "activation", po::value<std::string>()->default_value("none"), "Activation (none, relu)")(
        "scaleAlphaVec",
        po::value<bool>()->default_value(false)->implicit_value(true),
        "Enable per-row alpha scaling")(
        "factorDim",
        po::value<int>()->default_value(0),
        "ScaleAlphaVec dimension: 0=row(M), 1=col(N)")(
        "useScaleAB",
        po::value<std::string>()->default_value("none"),
        "ScaleAB mode (none, Scalar, Vector)")(
        "mxBlockA",
        po::value<int>()->default_value(0),
        "MX block size for the A side (FP4 only, must be power of 2; both --mxBlockA and "
        "--mxBlockB must be set together)")(
        "mxBlockB",
        po::value<int>()->default_value(0),
        "MX block size for the B side (FP4 only, must be power of 2; both --mxBlockA and "
        "--mxBlockB must be set together)")(
        "batchCount", po::value<size_t>()->default_value(1), "Batch count (default 1)")(
        "num-elements-to-validate",
        po::value<int>()->default_value(-1),
        "Number of output elements to compute; -1 or 0 computes the complete output");

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

    size_t      m        = vm["M"].as<size_t>();
    size_t      n        = vm["N"].as<size_t>();
    size_t      k        = vm["K"].as<size_t>();
    bool        transA   = vm["transA"].as<bool>();
    bool        transB   = vm["transB"].as<bool>();
    float       alpha    = vm["alpha"].as<float>();
    float       beta     = vm["beta"].as<float>();
    std::string typeStr  = vm["type"].as<std::string>();
    std::string typeAStr = vm["typeA"].as<std::string>();
    std::string typeBStr = vm["typeB"].as<std::string>();
    if(typeAStr.empty())
        typeAStr = typeStr;
    if(typeBStr.empty())
        typeBStr = typeStr;
    std::string computeInputAStr = vm["computeInputA"].as<std::string>();
    std::string computeInputBStr = vm["computeInputB"].as<std::string>();
    if(computeInputAStr.empty())
        computeInputAStr = typeAStr;
    if(computeInputBStr.empty())
        computeInputBStr = typeBStr;

    auto strToDataType = [](const std::string& s, rocisa::DataType& out) -> bool {
        if(s == "f32")
        {
            out = rocisa::DataType::Float;
            return true;
        }
        if(s == "f64")
        {
            out = rocisa::DataType::Double;
            return true;
        }
        if(s == "tf32")
        {
            out = rocisa::DataType::Float;
            return true;
        }
        if(s == "f16")
        {
            out = rocisa::DataType::Half;
            return true;
        }
        if(s == "bf16")
        {
            out = rocisa::DataType::BFloat16;
            return true;
        }
#ifdef TENSILE_USE_FP8_BF8
        if(s == "f8")
        {
            out = rocisa::DataType::Float8;
            return true;
        }
        if(s == "bf8")
        {
            out = rocisa::DataType::BFloat8;
            return true;
        }
        if(s == "f8fnuz")
        {
            out = rocisa::DataType::Float8_fnuz;
            return true;
        }
        if(s == "bf8fnuz")
        {
            out = rocisa::DataType::BFloat8_fnuz;
            return true;
        }
#endif
#ifndef _WIN32
        if(s == "f4")
        {
            out = rocisa::DataType::Float4;
            return true;
        }
#endif
        return false;
    };

    rocisa::DataType computeInputA, computeInputB;
    if(!strToDataType(computeInputAStr, computeInputA))
    {
        std::cerr << "Unknown computeInputA: " << computeInputAStr << std::endl;
        return 1;
    }
    if(!strToDataType(computeInputBStr, computeInputB))
    {
        std::cerr << "Unknown computeInputB: " << computeInputBStr << std::endl;
        return 1;
    }
    bool        validate                = vm["validate"].as<bool>();
    bool        injectValidationFailure = vm["injectValidationFailure"].as<bool>();
    bool        tryFastPath             = vm["tryFastPath"].as<bool>();
    bool        useBias                 = vm["bias"].as<bool>();
    std::string activationStr           = vm["activation"].as<std::string>();
    bool        useScaleAlphaVec        = vm["scaleAlphaVec"].as<bool>();
    int         factorDim               = vm["factorDim"].as<int>();
    std::string useScaleAB              = vm["useScaleAB"].as<std::string>();
    int         mxBlockA                = vm["mxBlockA"].as<int>();
    int         mxBlockB                = vm["mxBlockB"].as<int>();
    size_t      batchCount              = vm["batchCount"].as<size_t>();
    int         elementsToValidate      = vm["num-elements-to-validate"].as<int>();
    const bool  typeAIsTF32             = (typeAStr == "tf32");
    const bool  typeBIsTF32             = (typeBStr == "tf32");
    const bool  isTF32                  = typeAIsTF32 && typeBIsTF32;

    if(typeAIsTF32 != typeBIsTF32)
    {
        std::cerr << "Error: tf32 is an F32 math-op mode and must be used for both "
                  << "inputs or neither (typeA=" << typeAStr << ", typeB=" << typeBStr << ")"
                  << std::endl;
        return 1;
    }

    if(mxBlockA < 0 || mxBlockB < 0)
    {
        std::cerr << "Error: mxBlockA/mxBlockB must be non-negative" << std::endl;
        return 1;
    }
    if(elementsToValidate < -1)
    {
        std::cerr << "Error: num-elements-to-validate must be -1, 0, or positive" << std::endl;
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
                  << "(mxBlockA=" << mxBlockA << ", mxBlockB=" << mxBlockB << ")" << std::endl;
        return 1;
    }

    if((mxBlockA > 0 || mxBlockB > 0) && typeStr != "f4")
    {
        std::cerr << "Error: mxBlockA/mxBlockB is only supported for type f4, not " << typeStr
                  << std::endl;
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

    std::cout << "Running GEMM with: M=" << m << " N=" << n << " K=" << k << " TypeA=" << typeAStr
              << " TypeB=" << typeBStr << " ComputeInA=" << computeInputAStr
              << " ComputeInB=" << computeInputBStr << " FastPath=" << tryFastPath;
    if(isTF32)
        std::cout << " MathOp=XFloat32";
    std::cout << std::endl;

    // Dispatcher: pick A storage type, then B storage type. Each leaf calls
    // runGemm<A,B>(...). Asymmetric A/B is required to repro mixed-precision
    // bugs in the fast-path validator (e.g. F8N x Half).
    // tf32 = float storage + XFloat32 math-op. Dispatched as float with isTF32 flag.
    auto resolveAccumStorage = [](std::string& s) {
        if(s == "tf32")
            s = "f32";
    };
    resolveAccumStorage(typeAStr);
    resolveAccumStorage(typeBStr);

    auto dispatchB = [&](auto aTag) -> int {
        using AT   = decltype(aTag);
        auto callB = [&](auto bTag) -> int {
            using BT = decltype(bTag);
#ifndef _WIN32
            constexpr bool isMixedFP4
                = std::is_same_v<AT, Float4x2> != std::is_same_v<BT, Float4x2>;
            if constexpr(isMixedFP4)
            {
                std::cerr << "Error: mixed FP4 / non-FP4 input is not supported." << std::endl;
                return 1;
            }
            else
#endif
            {
                // Promote AccumulateT to double when both A and B storage are double (f64).
                using AccT
                    = std::conditional_t<std::is_same_v<AT, double> && std::is_same_v<BT, double>,
                                         double,
                                         float>;
                return runGemm<AT, BT, AccT>(m,
                                             n,
                                             k,
                                             transA,
                                             transB,
                                             alpha,
                                             beta,
                                             validate,
                                             injectValidationFailure,
                                             tryFastPath,
                                             useBias,
                                             activation,
                                             useScaleAlphaVec,
                                             useScaleAB,
                                             factorDim,
                                             computeInputA,
                                             computeInputB,
                                             mxBlockA,
                                             mxBlockB,
                                             batchCount,
                                             elementsToValidate,
                                             isTF32);
            }
        };
        if(typeBStr == "f32")
            return callB(float{});
        if(typeBStr == "f64")
            return callB(double{});
        if(typeBStr == "f16")
            return callB(Half{});
        if(typeBStr == "bf16")
            return callB(BFloat16{});
#ifdef TENSILE_USE_FP8_BF8
        if(typeBStr == "f8")
            return callB(Float8{});
        if(typeBStr == "bf8")
            return callB(BFloat8{});
        if(typeBStr == "f8fnuz")
            return callB(Float8_fnuz{});
        if(typeBStr == "bf8fnuz")
            return callB(BFloat8_fnuz{});
#endif
#ifndef _WIN32
        if(typeBStr == "f4")
            return callB(Float4x2{});
#endif
        std::cerr << "Unknown typeB: " << typeBStr << std::endl;
        return 1;
    };

    try
    {
        if(typeAStr == "f32")
            return dispatchB(float{});
        if(typeAStr == "f64")
            return dispatchB(double{});
        if(typeAStr == "f16")
            return dispatchB(Half{});
        if(typeAStr == "bf16")
            return dispatchB(BFloat16{});
#ifdef TENSILE_USE_FP8_BF8
        if(typeAStr == "f8")
            return dispatchB(Float8{});
        if(typeAStr == "bf8")
            return dispatchB(BFloat8{});
        if(typeAStr == "f8fnuz")
            return dispatchB(Float8_fnuz{});
        if(typeAStr == "bf8fnuz")
            return dispatchB(BFloat8_fnuz{});
#endif
#ifndef _WIN32
        if(typeAStr == "f4")
            return dispatchB(Float4x2{});
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
