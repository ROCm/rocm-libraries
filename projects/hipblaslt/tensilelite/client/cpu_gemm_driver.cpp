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
#include <iostream>
#include <random>
#include <string>
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

    void columnMajorGemm(const float*   a,
                         const float*   b,
                         const float*   c,
                         float*         d,
                         size_t         m,
                         size_t         n,
                         size_t         k,
                         bool           transA,
                         bool           transB,
                         float          alpha,
                         float          beta,
                         const float*   biasVec        = nullptr,
                         const float*   scaleAlphaVec  = nullptr,
                         ActivationType activation     = ActivationType::None,
                         const float*   scaleAVec      = nullptr,
                         const float*   scaleBVec      = nullptr,
                         int            factorDim      = 0,
                         QuantizeFn     quantizeA      = nullptr,
                         QuantizeFn     quantizeB      = nullptr)
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

        for(size_t i = 0; i < m; i++)
        {
            for(size_t j = 0; j < n; j++)
            {
                float sum = 0.0f;
                for(size_t l = 0; l < k; l++)
                {
                    float aVal = a[i * strideAM + l * strideAK];
                    float bVal = b[l * strideBK + j * strideBN];
                    if(quantizeA) aVal = quantizeA(aVal);
                    if(quantizeB) bVal = quantizeB(bVal);
                    sum += aVal * bVal;
                }
                float effectiveAlpha = alpha;
                if(scaleAVec)
                    effectiveAlpha *= scaleAVec[i];
                if(scaleBVec)
                    effectiveAlpha *= scaleBVec[j];
                if(scaleAlphaVec)
                    effectiveAlpha *= scaleAlphaVec[factorDim == 0 ? i : j];

                float result = effectiveAlpha * sum + beta * c[i + j * m];

                if(biasVec)
                    result += biasVec[i];

                if(activation == ActivationType::Relu)
                    result = std::max(0.0f, result);

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
            rocisa::DataType   computeInputB = rocisa::DataType::None)
{
    constexpr rocisa::DataType dtypeEnumA = TypeTraits<InputAT>::value;
    constexpr rocisa::DataType dtypeEnumB = TypeTraits<InputBT>::value;
    if(computeInputA == rocisa::DataType::None) computeInputA = dtypeEnumA;
    if(computeInputB == rocisa::DataType::None) computeInputB = dtypeEnumB;
    static_assert(std::is_same<AccumulateT, float>::value,
                  "Currently only float accumulation is supported");

    // Calculate strides assuming standard column-major packed storage
    size_t lda        = transA ? k : m;
    size_t ldb        = transB ? n : k;
    size_t ldc        = m;
    size_t batchCount = 1;

    // Define the contraction problem (geometry, strides, types)
    ContractionProblemGemm contraction
        = ContractionProblemGemm::GEMM_Strides(transA,
                                               transB,
                                               dtypeEnumA,
                                               dtypeEnumB,
                                               rocisa::DataType::Float,
                                               rocisa::DataType::Float, // A, B, C, D types
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
    contraction.setAlphaType(rocisa::DataType::Float);
    contraction.setBetaType(rocisa::DataType::Float);

    // Allocate host memory for inputs and outputs
    std::vector<InputAT> a(m * k);
    std::vector<InputBT> b(k * n);
    std::vector<float>   c(m * n);
    std::vector<float>   d(m * n);

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
    size_t                                seed = 42;
    std::mt19937                          gen(seed);
    std::uniform_int_distribution<>       binary_distribution(0, 1);
    std::uniform_real_distribution<float> realDist(-1.0f, 1.0f);

    auto randomGen = [&]() { return binary_distribution(gen) ? 1.0f : -1.0f; };

    auto initOperand = [&](auto& vec, bool quantizes) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
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
    std::generate(c.begin(), c.end(), [&]() { return static_cast<float>(randomGen()); });

    // Optional feature buffers
    std::vector<float> biasVec;
    std::vector<float> scaleAlphaVecBuf;

    if(useBias)
    {
        biasVec.resize(m);
        std::generate(biasVec.begin(), biasVec.end(), randomGen);
        contraction.setUseBias(1);
        contraction.setBias(rocisa::DataType::Float, m, m);
    }

    if(useScaleAlphaVec)
    {
        size_t scaleAlphaVecLen = (factorDim == 0) ? m : n;
        scaleAlphaVecBuf.resize(scaleAlphaVecLen);
        std::generate(scaleAlphaVecBuf.begin(), scaleAlphaVecBuf.end(), randomGen);
        contraction.setUseScaleAlphaVec(1);
        contraction.setScaleAlphaVec(rocisa::DataType::Float, scaleAlphaVecLen, factorDim);
    }

    // Random scale generator: magnitude in (1, 100], integer values to avoid rounding issues, sign random.
    // Excludes 0 and ±1 so missing/incorrect scaling is never masked.
    std::uniform_int_distribution<int> scaleDis(2, 100);
    auto scaleGen = [&]() {
        float sign = binary_distribution(gen) ? 1.0f : -1.0f;
        int mag    = scaleDis(gen);
        return sign * static_cast<float>(mag);
    };

    std::vector<float> scaleABuf;
    std::vector<float> scaleBBuf;

    if(useScaleAB == "Scalar")
    {
        scaleABuf = {scaleGen()};
        scaleBBuf = {scaleGen()};
        // setUseScaleAB must be called before setScaleA/setScaleB,
        // because setScaleA/B silently skips tensor registration when
        // m_useScaleAB is still empty.
        // See: https://github.com/ROCm/rocm-libraries/issues/6541
        contraction.setUseScaleAB("Scalar");
        contraction.setScaleA(rocisa::DataType::Float, 1);
        contraction.setScaleB(rocisa::DataType::Float, 1);
    }
    else if(useScaleAB == "Vector")
    {
        scaleABuf.resize(m);
        scaleBBuf.resize(n);
        std::generate(scaleABuf.begin(), scaleABuf.end(), scaleGen);
        std::generate(scaleBBuf.begin(), scaleBBuf.end(), scaleGen);
        contraction.setUseScaleAB("Vector");
        contraction.setScaleA(rocisa::DataType::Float, m);
        contraction.setScaleB(rocisa::DataType::Float, n);
    }

    if(activation != ActivationType::None)
    {
        contraction.setActivationType(activation);
        contraction.setParams().setActivationEnum(activation);
    }

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), alpha, beta);
    inputs.bias          = useBias ? biasVec.data() : nullptr;
    inputs.scaleAlphaVec = useScaleAlphaVec ? scaleAlphaVecBuf.data() : nullptr;
    inputs.scaleA        = (useScaleAB != "none") ? scaleABuf.data() : nullptr;
    inputs.scaleB        = (useScaleAB != "none") ? scaleBBuf.data() : nullptr;

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

        // Convert inputs to f32 for the golden reference comparison
        std::vector<float> aF32(a.begin(), a.end());
        std::vector<float> bF32(b.begin(), b.end());
        std::vector<float> cF32(c.begin(), c.end());
        std::vector<float> dRef(d.size());

        // If the storage type is wider than the compute-input type, the GPU
        // (and slow-path validator) quantize the operand down before the MAC.
        // Mirror that here so the golden GEMM reflects the same model.
        QuantizeFn quantA = (computeInputA != dtypeEnumA) ? quantizerFor(computeInputA) : nullptr;
        QuantizeFn quantB = (computeInputB != dtypeEnumB) ? quantizerFor(computeInputB) : nullptr;

        // Run the golden reference
        columnMajorGemm(aF32.data(),
                        bF32.data(),
                        cF32.data(),
                        dRef.data(),
                        m,
                        n,
                        k,
                        transA,
                        transB,
                        (useScaleAB == "Scalar") ? alpha * scaleABuf[0] * scaleBBuf[0] : alpha,
                        beta,
                        useBias ? biasVec.data() : nullptr,
                        useScaleAlphaVec ? scaleAlphaVecBuf.data() : nullptr,
                        activation,
                        (useScaleAB == "Vector") ? scaleABuf.data() : nullptr,
                        (useScaleAB == "Vector") ? scaleBBuf.data() : nullptr,
                        factorDim,
                        quantA,
                        quantB);

        // Compare results
        bool  allClose = true;
        float maxDiff  = 0.0f;

        for(size_t i = 0; i < m * n; i++)
        {
            float valDut = static_cast<float>(d[i]);
            float valRef = dRef[i];
            float diff   = std::abs(valDut - valRef);

            if(diff > 0.05f)
            {
                allClose = false;
                maxDiff  = std::max(maxDiff, diff);
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
        "type", po::value<std::string>()->default_value("f32"), "Data type for A and B (f32, f16, bf16, f8, bf8, f8fnuz, bf8fnuz)")(
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
        "useScaleAB", po::value<std::string>()->default_value("none"), "ScaleAB mode (none, Scalar, Vector)");

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
        if(s == "f16")            { out = rocisa::DataType::Half;         return true; }
        if(s == "bf16")           { out = rocisa::DataType::BFloat16;     return true; }
#ifdef TENSILE_USE_FP8_BF8
        if(s == "f8")             { out = rocisa::DataType::Float8;       return true; }
        if(s == "bf8")            { out = rocisa::DataType::BFloat8;      return true; }
        if(s == "f8fnuz")         { out = rocisa::DataType::Float8_fnuz;  return true; }
        if(s == "bf8fnuz")        { out = rocisa::DataType::BFloat8_fnuz; return true; }
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
              << " FastPath=" << tryFastPath << std::endl;

    // Dispatcher: pick A storage type, then B storage type. Each leaf calls
    // runGemm<A,B>(...). Asymmetric A/B is required to repro mixed-precision
    // bugs in the fast-path validator (e.g. F8N x Half).
    auto dispatchB = [&](auto aTag) -> int {
        using AT = decltype(aTag);
        auto callB = [&](auto bTag) -> int {
            using BT = decltype(bTag);
            return runGemm<AT, BT>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath,
                useBias, activation, useScaleAlphaVec, useScaleAB, factorDim,
                computeInputA, computeInputB);
        };
        if(typeBStr == "f32")        return callB(float{});
        if(typeBStr == "f16")        return callB(Half{});
        if(typeBStr == "bf16")       return callB(BFloat16{});
#ifdef TENSILE_USE_FP8_BF8
        if(typeBStr == "f8")         return callB(Float8{});
        if(typeBStr == "bf8")        return callB(BFloat8{});
        if(typeBStr == "f8fnuz")     return callB(Float8_fnuz{});
        if(typeBStr == "bf8fnuz")    return callB(BFloat8_fnuz{});
#endif
        std::cerr << "Unknown typeB: " << typeBStr << std::endl;
        return 1;
    };

    try
    {
        if(typeAStr == "f32")        return dispatchB(float{});
        if(typeAStr == "f16")        return dispatchB(Half{});
        if(typeAStr == "bf16")       return dispatchB(BFloat16{});
#ifdef TENSILE_USE_FP8_BF8
        if(typeAStr == "f8")         return dispatchB(Float8{});
        if(typeAStr == "bf8")        return dispatchB(BFloat8{});
        if(typeAStr == "f8fnuz")     return dispatchB(Float8_fnuz{});
        if(typeAStr == "bf8fnuz")    return dispatchB(BFloat8_fnuz{});
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
