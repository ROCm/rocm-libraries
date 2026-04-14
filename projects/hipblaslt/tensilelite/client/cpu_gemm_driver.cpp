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
#include <vector>

#include "ProgramOptions.hpp"
#include "Reference.hpp"
#include "rocisa/include/enum.hpp"

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

    // A naive, slow, golden reference implementation of GEMM.
    // Used strictly for validating the correctness of the optimized path.
    // Calculates D = alpha * (A * B) + beta * C
    //
    // We can all the various bells and whistles that tensile supports
    // (activations, etc) as needed.
    void columnMajorGemm(const float* a,
                         const float* b,
                         const float* c,
                         float*       d,
                         size_t       m,
                         size_t       n,
                         size_t       k,
                         bool         transA,
                         bool         transB,
                         float        alpha,
                         float        beta)
    {
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
                    sum += aVal * bVal;
                }
                d[i + j * m] = alpha * sum + beta * c[i + j * m];
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
template <typename InputT, typename AccumulateT = float>
int runGemm(size_t m,
            size_t n,
            size_t k,
            bool   transA,
            bool   transB,
            float  alpha,
            float  beta,
            bool   validate,
            bool   tryFastPath)
{
    constexpr rocisa::DataType dtypeEnum = TypeTraits<InputT>::value;
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
                                               dtypeEnum,
                                               dtypeEnum,
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

    contraction.setComputeInputTypeA(dtypeEnum);
    contraction.setComputeInputTypeB(dtypeEnum);

    // Allocate host memory for inputs and outputs
    std::vector<InputT> a(m * k);
    std::vector<InputT> b(k * n);
    std::vector<float>  c(m * n);
    std::vector<float>  d(m * n);

    // Initialize inputs with random values in {-1.0, 1.0}
    size_t                          seed = 42;
    std::mt19937                    gen(seed);
    std::uniform_int_distribution<> dis(0, 1);

    auto randomGen = [&]() { return dis(gen) ? 1.0f : -1.0f; };

    std::generate(a.begin(), a.end(), [&]() { return static_cast<InputT>(randomGen()); });
    std::generate(b.begin(), b.end(), [&]() { return static_cast<InputT>(randomGen()); });
    std::generate(c.begin(), c.end(), [&]() { return static_cast<float>(randomGen()); });

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), alpha, beta);

    auto start = std::chrono::high_resolution_clock::now();

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
                        alpha,
                        beta);

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
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")(
        "M", po::value<size_t>()->default_value(128), "Matrix M dimension")(
        "N", po::value<size_t>()->default_value(128), "Matrix N dimension")(
        "K", po::value<size_t>()->default_value(128), "Matrix K dimension")(
        "transA", po::value<bool>()->default_value(false), "Transpose A")(
        "transB", po::value<bool>()->default_value(true), "Transpose B")(
        "alpha", po::value<float>()->default_value(1.0f), "Alpha scalar")(
        "beta", po::value<float>()->default_value(0.0f), "Beta scalar")(
        "type", po::value<std::string>()->default_value("f32"), "Data type (f32, f16, bf16)")(
        "validate", po::value<bool>()->default_value(true), "Run validation against ref")(
        "tryFastPath", po::value<bool>()->default_value(true), "Use optimized path");

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

    size_t      m           = vm["M"].as<size_t>();
    size_t      n           = vm["N"].as<size_t>();
    size_t      k           = vm["K"].as<size_t>();
    bool        transA      = vm["transA"].as<bool>();
    bool        transB      = vm["transB"].as<bool>();
    float       alpha       = vm["alpha"].as<float>();
    float       beta        = vm["beta"].as<float>();
    std::string typeStr     = vm["type"].as<std::string>();
    bool        validate    = vm["validate"].as<bool>();
    bool        tryFastPath = vm["tryFastPath"].as<bool>();

    std::cout << "Running GEMM with: M=" << m << " N=" << n << " K=" << k << " Type=" << typeStr
              << " FastPath=" << tryFastPath << std::endl;

    try
    {
        if(typeStr == "f32")
        {
            return runGemm<float>(m, n, k, transA, transB, alpha, beta, validate, tryFastPath);
        }
        else if(typeStr == "bf16")
        {
            return runGemm<TensileLite::BFloat16>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath);
        }
        else if(typeStr == "f16")
        {
            return runGemm<TensileLite::Half>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath);
        }
        else
        {
            std::cerr << "Unknown type: " << typeStr << std::endl;
            return 1;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
