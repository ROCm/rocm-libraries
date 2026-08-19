/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <iostream>
#include <roc/host_validation/validation.hpp>
#include <string>
#include <vector>

void printUsage(char* programName)
{
    std::cout << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-m, --m\t\t\t\tSize of dim 0, default is 1335\n"
              << "\t-n, --n\t\t\t\tSize of dim 1, default is 16\n"
              << "\t-a, --affine\t\t\t\tEnable Gamma and Beta, default is false\n"
              << "\t--initialization \t\tInitialize matrix data. Options: rand_int, trig_float, "
                 "hpl(floating), special, zero. (default is hpl)\n";
}

int parseArgs(
    int argc, char** argv, size_t* m, size_t* n, bool* affine, hipblaslt_initialization* init)
{
    if(argc <= 1)
    {
        return EXIT_SUCCESS;
    }

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
        {
            if((arg == "-h") || (arg == "--help"))
            {
                return EXIT_FAILURE;
            }

            if(arg == "-m" || arg == "--m")
            {
                *m = std::stoul(argv[++i]);
            }
            else if(arg == "-n" || arg == "--n")
            {
                *n = std::stoul(argv[++i]);
            }
            else if(arg == "-a" || arg == "--affine")
            {
                *affine = std::stoul(argv[++i]);
            }
            else if(arg == "--initialization" || arg == "--init")
            {
                const std::string initStr{argv[++i]};

                if(initStr != "rand_int" && initStr != "trig_float" && initStr != "hpl"
                   && initStr != "special" && initStr != "zero")
                {
                    std::cerr << "Invalid initialization type: " << initStr << '\n';
                    return EXIT_FAILURE;
                }

                *init = string2hipblaslt_initialization(initStr);
            }
        }
        else
        {
            std::cerr << "error with " << arg << std::endl;
            std::cerr << "option must start with - or --" << std::endl << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

void reportComparison(const char* title, const roc::host_validation::ComparisonResult& comparison)
{
    std::cout << "----- " << title << " result" << " -----" << std::endl;
    std::cout << "status: " << (comparison.passed() ? "PASS" : "FAIL") << std::endl;
    std::cout << "compared: " << comparison.compared << std::endl;
    std::cout << "mismatches: " << comparison.mismatches << std::endl;
    std::cout << "matched NaNs: " << comparison.matchedNaNs << std::endl;
    std::cout << "matched infinities: " << comparison.matchedInfinities << std::endl;
    std::cout << "non-finite mismatches: " << comparison.nonFiniteMismatches << std::endl;
    std::cout << "max error: " << comparison.maxAbsoluteDifference << std::endl;

    for(const auto& mismatch : comparison.reportedMismatches)
    {
        std::cout << "index " << mismatch.index << ": observed " << mismatch.observed
                  << ", expected " << mismatch.expected << ", absolute difference "
                  << mismatch.absoluteDifference << std::endl;
    }
}

template <typename DType>
void initData(DType* data, std::size_t numElements, hipblaslt_initialization initMethod)
{
    hipblaslt::host_validation::initialize(data, numElements, initMethod);
}

int main(int argc, char** argv)
{
    std::size_t              m{1};
    std::size_t              n{64};
    bool                     affine{false};
    hipblaslt_initialization init{hipblaslt_initialization::hpl};

    if(auto err = parseArgs(argc, argv, &m, &n, &affine, &init))
    {
        printUsage(argv[0]);
        return err;
    }

    std::size_t numElements     = m * n;
    std::size_t elementNumBytes = sizeof(float);

    float* gpuOutput{nullptr};
    float* gpuMean{nullptr};
    float* gpuInvvar{nullptr};
    float* gpuInput{nullptr};
    float* gpuGamma{nullptr};
    float* gpuBeta{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, numElements * elementNumBytes);
    hipErr      = hipMalloc(&gpuMean, m * elementNumBytes);
    hipErr      = hipMalloc(&gpuInvvar, m * elementNumBytes);
    hipErr      = hipMalloc(&gpuInput, numElements * elementNumBytes);
    if(affine)
    {
        hipErr = hipMalloc(&gpuGamma, n * elementNumBytes);
        hipErr = hipMalloc(&gpuBeta, n * elementNumBytes);
    }

    std::vector<float> cpuOutput(numElements, 0.f);
    std::vector<float> cpuMean(m, 0.f);
    std::vector<float> cpuInvvar(m, 0.f);
    std::vector<float> cpuInput(numElements, 0.f);
    std::vector<float> cpuGamma(affine ? n : 0, 1.f);
    std::vector<float> cpuBeta(affine ? n : 0, 0.f);

    initData(cpuInput.data(), cpuInput.size(), init);

    if(affine)
    {
        initData(cpuGamma.data(), cpuGamma.size(), init);
        initData(cpuBeta.data(), cpuBeta.size(), init);
    }

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), numElements * elementNumBytes);
    if(affine)
    {
        hipErr = hipMemcpyHtoD(gpuGamma, cpuGamma.data(), n * elementNumBytes);
        hipErr = hipMemcpyHtoD(gpuBeta, cpuBeta.data(), n * elementNumBytes);
    }

    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    //warmup
    auto hipblasltErr = hipblasltExtLayerNorm(
        HIP_R_32F, gpuOutput, gpuMean, gpuInvvar, gpuInput, m, n, 1e-05, gpuGamma, gpuBeta, stream);

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, numElements * elementNumBytes);
    hipErr = hipMemcpyDtoH(cpuMean.data(), gpuMean, m * elementNumBytes);
    hipErr = hipMemcpyDtoH(cpuInvvar.data(), gpuInvvar, m * elementNumBytes);

    using namespace roc::host_validation;
    using namespace hipblaslt::host_validation;
    const Layout tensorLayout     = Layout::contiguous(Shape{m, n});
    const Layout statisticsLayout = Layout::contiguous(Shape{m});

    LayerNormProblem problem(tensorFromStorage(cpuInput.data(), cpuInput.size(), tensorLayout),
                             ScalarType::Float32,
                             1,
                             ScalarType::Float32);
    problem.meanType            = ScalarType::Float32;
    problem.inverseVarianceType = ScalarType::Float32;
    problem.epsilon             = 1e-5;
    if(affine)
    {
        const Layout affineLayout = Layout::contiguous(Shape{n});
        problem.gamma = tensorFromStorage(cpuGamma.data(), cpuGamma.size(), affineLayout);
        problem.beta  = tensorFromStorage(cpuBeta.data(), cpuBeta.size(), affineLayout);
    }
    const LayerNormResult reference = referenceLayerNorm(problem);

    const ComparisonOptions comparisonOptions = nearComparisonOptions(1e-5);
    reportComparison("Output",
                     roc::host_validation::compare(
                         tensorFromStorage(cpuOutput.data(), cpuOutput.size(), tensorLayout),
                         reference.output,
                         comparisonOptions));
    reportComparison("Mean",
                     roc::host_validation::compare(
                         tensorFromStorage(cpuMean.data(), cpuMean.size(), statisticsLayout),
                         *reference.mean,
                         comparisonOptions));
    reportComparison("Invvar",
                     roc::host_validation::compare(
                         tensorFromStorage(cpuInvvar.data(), cpuInvvar.size(), statisticsLayout),
                         *reference.inverseVariance,
                         comparisonOptions));

    hipEvent_t beg, end;
    hipErr      = hipEventCreate(&beg);
    hipErr      = hipEventCreate(&end);
    int numRuns = 200;
    hipErr      = hipEventRecord(beg, stream);

    for(int i = 0; i < numRuns; ++i)
    {
        hipblasltErr = hipblasltExtLayerNorm(HIP_R_32F,
                                             gpuOutput,
                                             gpuMean,
                                             gpuInvvar,
                                             gpuInput,
                                             m,
                                             n,
                                             1e-05,
                                             gpuGamma,
                                             gpuBeta,
                                             stream);
    }
    hipErr = hipEventRecord(end, stream);
    hipErr = hipEventSynchronize(end);
    hipErr = hipStreamSynchronize(stream);
    float dur{};
    hipErr = hipEventElapsedTime(&dur, beg, end);
    std::cout << "Time elapsed: " << std::to_string(dur / numRuns) << " ms\n";

    hipErr = hipEventDestroy(beg);
    hipErr = hipEventDestroy(end);
    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuMean);
    hipErr = hipFree(gpuInvvar);
    hipErr = hipFree(gpuInput);
    if(gpuGamma)
        hipErr = hipFree(gpuGamma);
    if(gpuBeta)
        hipErr = hipFree(gpuBeta);
    return 0;
}
