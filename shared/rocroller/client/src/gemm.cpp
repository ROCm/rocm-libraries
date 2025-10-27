/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <filesystem>
#include <span>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>
#include <rocRoller/Utilities/Utils.hpp>
#include <rocRoller/Utilities/Version.hpp>

#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>

#include "client/CLI_Utils.hpp"
#include "client/DataParallelGEMMSolution.hpp"
#include "client/GEMMParameters.hpp"
#include "client/GEMMParameters_serialization.hpp"
#include "client/PreSwizzle.hpp"
#include "client/RotatingBuffer.hpp"
#include "client/StreamKGEMMSolution.hpp"

#include <CLI/CLI.hpp>

using namespace rocRoller;

namespace SolutionParams = rocRoller::Parameters::Solution;

enum ReturnCodes : int
{
    OK                         = 0,
    GenerateFailure            = 1,
    CorrectnessFailure         = 2,
    SolutionNotSupportedOnArch = 3
};

namespace rocRoller::Client::GEMMClient
{
    using GEMMSolutionPtr = std::shared_ptr<Client::GEMMClient::GEMMSolution>;

    struct MNKTuple
    {
        int m, n, k;
    };

    template <typename A, typename B, typename C, typename D>
    std::pair<bool, double>
        validate(std::vector<typename PackedTypeOf<A>::type> const&      h_A,
                 std::vector<typename PackedTypeOf<B>::type> const&      h_B,
                 std::vector<C> const&                                   h_C,
                 std::vector<D> const&                                   h_D,
                 std::vector<uint8_t> const&                             h_scaleA,
                 std::vector<uint8_t> const&                             h_scaleB,
                 rocRoller::Client::GEMMClient::ProblemParameters const& problemParams,
                 GPUArchitecture const&                                  arch)
    {
        using namespace rocRoller::Client::GEMMClient;

        // Host result
        std::vector<D> h_result(problemParams.m * problemParams.n, static_cast<D>(0.0));

        if(!h_scaleA.empty() || !h_scaleB.empty())
        {
            rocRoller::ScaledCPUMM(h_result,
                                   h_C,
                                   h_A,
                                   h_B,
                                   h_scaleA,
                                   h_scaleB,
                                   problemParams.m,
                                   problemParams.n,
                                   problemParams.k,
                                   problemParams.alpha,
                                   problemParams.beta,
                                   problemParams.types.transA == TransposeType::T,
                                   problemParams.types.transB == TransposeType::T,
                                   problemParams.types.scaleBlockSize,
                                   problemParams.types.scaleTypeA,
                                   problemParams.types.scaleTypeB);
        }
        else
        {
            CPUMM(h_result,
                  h_C,
                  h_A,
                  h_B,
                  problemParams.m,
                  problemParams.n,
                  problemParams.k,
                  problemParams.alpha,
                  problemParams.beta,
                  problemParams.types.transA == TransposeType::T,
                  problemParams.types.transB == TransposeType::T);
        }

        auto tol = gemmAcceptableError<A, B, D>(
            problemParams.m, problemParams.n, problemParams.k, arch.target());
        auto res = compare(h_D, h_result, tol);

        Log::debug(res.message());

        std::cout << "Result: " << (res.ok ? "Correct" : "Incorrect") << std::endl;
        std::cout << "RNorm: " << res.relativeNormL2 << std::endl;
        if(!res.ok)
        {
            std::cerr << "WARNING: Result incorrect.  " << res.message() << std::endl;
        }
        return {res.ok, res.relativeNormL2};
    }

    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    template <typename A, typename B, typename C, typename D>
    Client::BenchmarkResults GEMM(CommandPtr                 command,
                                  CommandKernelPtr           commandKernel,
                                  GEMMSolutionPtr            gemm,
                                  ProblemParameters const&   problemParams,
                                  RunParameters const&       runParams,
                                  BenchmarkParameters const& benchmarkParams,
                                  GPUArchitecture const&     arch)
    {
        using namespace rocRoller::Client;
        using namespace rocRoller::Client::GEMMClient;

        // Host Data
        std::cout << "Generating input data..." << std::endl;

        TensorDescriptor descA(fromString<DataType>(problemParams.types.typeA),
                               {static_cast<unsigned long>(problemParams.m),
                                static_cast<unsigned long>(problemParams.k)},
                               problemParams.types.transA == TransposeType::T ? "T" : "N");
        TensorDescriptor descB(fromString<DataType>(problemParams.types.typeB),
                               {static_cast<unsigned long>(problemParams.k),
                                static_cast<unsigned long>(problemParams.n)},
                               problemParams.types.transB == TransposeType::T ? "T" : "N");
        TensorDescriptor descC(fromString<DataType>(problemParams.types.typeC),
                               {static_cast<unsigned long>(problemParams.m),
                                static_cast<unsigned long>(problemParams.n)},
                               "N");

        using PackedTypeA = typename PackedTypeOf<A>::type;
        using PackedTypeB = typename PackedTypeOf<B>::type;
        std::vector<PackedTypeA> hostA;
        std::vector<PackedTypeB> hostB;
        std::vector<C>           hostC;
        std::vector<D>           hostD(problemParams.m * problemParams.n, D{});
        std::vector<uint8_t>     hostScaleA, hostScaleB;

        auto seed = 31415u;
        if(problemParams.types.scaleA == Operations::ScaleMode::Separate
           || problemParams.types.scaleB == Operations::ScaleMode::Separate)
        {
            auto scaleBlockSize = problemParams.types.scaleBlockSize;
            AssertFatal(scaleBlockSize > 0, "scaleBlockSize must be set to scale A or B.");
            AssertFatal(arch.isSupportedScaleBlockSize(scaleBlockSize),
                        fmt::format("Architecture {} does not support block scaling (size: {}).",
                                    arch.target().toString(),
                                    scaleBlockSize));
            AssertFatal(problemParams.k % scaleBlockSize == 0,
                        fmt::format("K: {} must be a multiple of the scale block size: {}",
                                    problemParams.k,
                                    scaleBlockSize));
            DGenInput(seed,
                      hostA,
                      descA,
                      hostB,
                      descB,
                      hostC,
                      descC,
                      hostScaleA,
                      hostScaleB,
                      problemParams.types.scaleA == Operations::ScaleMode::Separate,
                      problemParams.types.scaleB == Operations::ScaleMode::Separate,
                      -1.f,
                      1.f,
                      static_cast<uint>(scaleBlockSize));
        }
        else
        {
            DGenInput(seed, hostA, descA, hostB, descB, hostC, descC);
        }

        size_t rotatingSize = benchmarkParams.rotatingBuffSize;

        RotatingBuffer<PackedTypeA> rotatingA(hostA, rotatingSize);
        RotatingBuffer<PackedTypeB> rotatingB(hostB, rotatingSize);
        RotatingBuffer<C>           rotatingC(hostC, rotatingSize);
        auto deviceD = make_shared_device<D>(problemParams.m * problemParams.n, D{});

        std::shared_ptr<uint8_t> deviceScaleA, deviceScaleB;
        AssertFatal(problemParams.types.scaleA == Operations::ScaleMode::None
                        || problemParams.types.scaleA == Operations::ScaleMode::SingleScale
                        || problemParams.types.scaleA == Operations::ScaleMode::Separate,
                    "Scale mode not supported!",
                    ShowValue(problemParams.types.scaleA));
        AssertFatal(problemParams.types.scaleB == Operations::ScaleMode::None
                        || problemParams.types.scaleB == Operations::ScaleMode::SingleScale
                        || problemParams.types.scaleB == Operations::ScaleMode::Separate,
                    "Scale mode not supported!",
                    ShowValue(problemParams.types.scaleB));
        if(problemParams.types.scaleA == Operations::ScaleMode::Separate)
        {
            if(problemParams.types.scaleSkipPermlane)
            {
                AssertFatal(problemParams.types.scaleShuffleTileA.size() == 3);

                auto descScaleA = descA.withNormalizedDimensions();
                {
                    auto sizes = descScaleA.sizes();
                    sizes[0] /= problemParams.types.scaleBlockSize;
                    descScaleA = TensorDescriptor(descScaleA.dataType(), std::move(sizes));
                }

                auto tmpScaleA
                    = preSwizzle(hostScaleA, descScaleA, problemParams.types.scaleShuffleTileA);
                deviceScaleA = make_shared_device(tmpScaleA);
            }
            else
            {
                deviceScaleA = make_shared_device(hostScaleA);
            }
        }
        if(problemParams.types.scaleB == Operations::ScaleMode::Separate)
        {
            if(problemParams.types.scaleSkipPermlane)
            {
                AssertFatal(problemParams.types.scaleShuffleTileB.size() == 3);

                auto descScaleB = descB.withNormalizedDimensions();
                {
                    auto sizes = descScaleB.sizes();
                    sizes[0] /= problemParams.types.scaleBlockSize;
                    descScaleB = TensorDescriptor(descScaleB.dataType(), std::move(sizes));
                }

                auto tmpScaleB
                    = preSwizzle(hostScaleB, descScaleB, problemParams.types.scaleShuffleTileB);
                deviceScaleB = make_shared_device(tmpScaleB);
            }
            else
            {
                deviceScaleB = make_shared_device(hostScaleB);
            }
        }

        std::cout << "Generating launch parameters and runtime arguments..." << std::endl;

        commandKernel->loadKernel();

        auto commandArgs = gemm->commandArguments(command, problemParams, runParams);

        auto [aTag, bTag, cTag, dTag] = gemm->getABCDTags();

        commandArgs.setArgument(dTag, ArgumentType::Value, (D*)deviceD.get());

        if(problemParams.types.scaleA == Operations::ScaleMode::Separate)
        {
            auto dataTypeA  = TypeInfo<A>::Var.dataType;
            auto descAScale = TensorDescriptor(
                dataTypeA,
                {static_cast<size_t>(problemParams.m),
                 static_cast<size_t>(problemParams.k / problemParams.types.scaleBlockSize)},
                problemParams.types.transA == TransposeType::T ? "T" : "N");
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            setCommandTensorArg(commandArgs, aScaleTag.value(), descAScale, deviceScaleA.get());
        }
        else if(problemParams.types.scaleA == Operations::ScaleMode::SingleScale)
        {
            uint8_t scaleValue
                = floatToScale(problemParams.types.scaleTypeA, problemParams.scaleValueA);
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            commandArgs.setArgument(aScaleTag.value(), ArgumentType::Value, scaleValue);

            hostScaleA = {scaleValue};
        }

        if(problemParams.types.scaleB == Operations::ScaleMode::Separate)
        {
            auto dataTypeB  = TypeInfo<A>::Var.dataType;
            auto descBScale = TensorDescriptor(
                dataTypeB,
                {static_cast<size_t>(problemParams.k / problemParams.types.scaleBlockSize),
                 static_cast<size_t>(problemParams.n)},
                problemParams.types.transB == TransposeType::T ? "T" : "N");
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            setCommandTensorArg(commandArgs, bScaleTag.value(), descBScale, deviceScaleB.get());
        }
        else if(problemParams.types.scaleB == Operations::ScaleMode::SingleScale)
        {
            uint8_t scaleValue
                = floatToScale(problemParams.types.scaleTypeB, problemParams.scaleValueB);
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            commandArgs.setArgument(bScaleTag.value(), ArgumentType::Value, scaleValue);

            hostScaleB = {scaleValue};
        }

        gemm->validateRunParameters(
            command, problemParams, runParams, benchmarkParams, commandKernel);

        auto runtimeArgs = commandArgs.runtimeArguments();

        // Note: the lifetime of deviceScratch needs to exceed kernel executions
        std::shared_ptr<uint8_t> deviceScratch;
        {
            auto scratchSpaceRequired = commandKernel->scratchSpaceRequired(runtimeArgs);
            if(scratchSpaceRequired > 0)
            {
                deviceScratch = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
                commandArgs.setArgument(
                    gemm->getScratchTag(), ArgumentType::Value, deviceScratch.get());
            }
        }

        if(benchmarkParams.visualize)
        {
            Client::visualize(command, *commandKernel, commandArgs);
        }

        std::cout << std::endl;
        std::cout << "Problem:" << std::endl;
        std::cout << problemParams << std::endl;

        std::cout << "Launching GPU kernel(s)..." << std::endl;

        BenchmarkResults result;
        result.runParams       = runParams;
        result.benchmarkParams = benchmarkParams;

        // Benchmark runs
        for(int outer = 0; outer < benchmarkParams.numOuter; ++outer)
        {
            // Warmup runs
            for(int i = 0; i < benchmarkParams.numWarmUp; ++i)
            {
                auto spanA = rotatingA.next();
                auto spanB = rotatingB.next();
                auto spanC = rotatingC.next();

                commandArgs.setArgument(
                    aTag, ArgumentType::Value, reinterpret_cast<unsigned char*>(spanA.data()));

                commandArgs.setArgument(
                    bTag, ArgumentType::Value, reinterpret_cast<unsigned char*>(spanB.data()));

                commandArgs.setArgument(
                    cTag, ArgumentType::Value, reinterpret_cast<unsigned char*>(spanC.data()));

                auto runtimeArgs = commandArgs.runtimeArguments();
                commandKernel->launchKernel(runtimeArgs);
            }

            HIP_TIMER(t_kernel, "GEMM", benchmarkParams.numInner);
            for(int inner = 0; inner < benchmarkParams.numInner; ++inner)
            {
                auto spanA = rotatingA.next();
                auto spanB = rotatingB.next();
                auto spanC = rotatingC.next();

                commandArgs.setArgument(
                    aTag, ArgumentType::Value, reinterpret_cast<unsigned char*>(spanA.data()));

                commandArgs.setArgument(
                    bTag, ArgumentType::Value, reinterpret_cast<unsigned char*>(spanB.data()));

                commandArgs.setArgument(
                    cTag, ArgumentType::Value, reinterpret_cast<unsigned char*>(spanC.data()));

                auto runtimeArgs = commandArgs.runtimeArguments();
                commandKernel->launchKernel(runtimeArgs, t_kernel, inner);
            }
            HIP_SYNC(t_kernel);
            t_kernel->sleep(50);
            auto nanoseconds = t_kernel->allNanoseconds();
            result.kernelExecute.insert(
                result.kernelExecute.end(), nanoseconds.begin(), nanoseconds.end());
        }

        double totalTime = 0;
        for(auto ke : result.kernelExecute)
            totalTime += static_cast<double>(ke) / 1.e9;
        double averageTime = totalTime / (benchmarkParams.numInner * benchmarkParams.numOuter);

        std::cout << "Average runtime (s): " << averageTime << std::endl;
        std::cout << "Average GFLOPS:      "
                  << (double)problemParams.m * problemParams.n * problemParams.k * 2.0 / averageTime
                         * 1.e-9
                  << std::endl;
        std::cerr << "Average GFLOPS:      "
                  << (double)problemParams.m * problemParams.n * problemParams.k * 2.0 / averageTime
                         * 1.e-9
                  << std::endl;

        result.kernelAssemble = TimerPool::nanoseconds("Assembler::assembleMachineCode");
        result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

        if(benchmarkParams.check)
        {
            AssertFatal(hipMemcpy(hostD.data(),
                                  deviceD.get(),
                                  problemParams.m * problemParams.n * sizeof(D),
                                  hipMemcpyDeviceToHost)
                        == (hipError_t)HIP_SUCCESS);

            auto [correct, rnorm] = validate<A, B, C, D>(
                hostA, hostB, hostC, hostD, hostScaleA, hostScaleB, problemParams, arch);

            result.checked = true;
            result.correct = correct;
            result.rnorm   = rnorm;
        }

        return result;
    }

    template <typename A, typename C, typename D>
    Client::BenchmarkResults GEMMMixed(CommandPtr                 command,
                                       CommandKernelPtr           commandKernel,
                                       GEMMSolutionPtr            gemm,
                                       ProblemParameters const&   problemParams,
                                       RunParameters const&       runParams,
                                       BenchmarkParameters const& benchmarkParams,
                                       GPUArchitecture const&     arch,
                                       auto                       typeB)
    {
        if(typeB == "fp8")
        {
            return GEMM<A, FP8, C, D>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        else if(typeB == "bf8")
        {
            return GEMM<A, BF8, C, D>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        else if(typeB == "fp6")
        {
            return GEMM<A, FP6, C, D>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        else if(typeB == "bf6")
        {
            return GEMM<A, BF6, C, D>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        else if(typeB == "fp4")
        {
            return GEMM<A, FP4, C, D>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        else
            Throw<FatalError>("Invalid type for Mixed GEMM.");
    }

    template <typename C, typename D>
    Client::BenchmarkResults GEMMMixed(CommandPtr                 command,
                                       CommandKernelPtr           commandKernel,
                                       GEMMSolutionPtr            gemm,
                                       ProblemParameters const&   problemParams,
                                       RunParameters const&       runParams,
                                       BenchmarkParameters const& benchmarkParams,
                                       GPUArchitecture const&     arch,
                                       auto                       typeA,
                                       auto                       typeB)
    {
        if(typeA == "fp8")
        {
            return GEMMMixed<FP8, C, D>(command,
                                        commandKernel,
                                        gemm,
                                        problemParams,
                                        runParams,
                                        benchmarkParams,
                                        arch,
                                        typeB);
        }
        else if(typeA == "bf8")
        {
            return GEMMMixed<BF8, C, D>(command,
                                        commandKernel,
                                        gemm,
                                        problemParams,
                                        runParams,
                                        benchmarkParams,
                                        arch,
                                        typeB);
        }
        else if(typeA == "fp6")
        {
            return GEMMMixed<FP6, C, D>(command,
                                        commandKernel,
                                        gemm,
                                        problemParams,
                                        runParams,
                                        benchmarkParams,
                                        arch,
                                        typeB);
        }
        else if(typeA == "bf6")
        {
            return GEMMMixed<BF6, C, D>(command,
                                        commandKernel,
                                        gemm,
                                        problemParams,
                                        runParams,
                                        benchmarkParams,
                                        arch,
                                        typeB);
        }
        else if(typeA == "fp4")
        {
            return GEMMMixed<FP4, C, D>(command,
                                        commandKernel,
                                        gemm,
                                        problemParams,
                                        runParams,
                                        benchmarkParams,
                                        arch,
                                        typeB);
        }
        else
            Throw<FatalError>("Invalid type for Mixed GEMM.");
    }

    template <typename AB>
    Client::BenchmarkResults GEMMUniform(CommandPtr                 command,
                                         CommandKernelPtr           commandKernel,
                                         GEMMSolutionPtr            gemm,
                                         ProblemParameters const&   problemParams,
                                         RunParameters const&       runParams,
                                         BenchmarkParameters const& benchmarkParams,
                                         GPUArchitecture const&     arch,
                                         auto                       typeCD)
    {
        if(typeCD == "float")
        {
            return GEMM<AB, AB, float, float>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        if(typeCD == "half")
        {
            return GEMM<AB, AB, Half, Half>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        if(typeCD == "bf16")
        {
            return GEMM<AB, AB, BFloat16, BFloat16>(
                command, commandKernel, gemm, problemParams, runParams, benchmarkParams, arch);
        }
        Throw<FatalError>("Invalid CD type for uniform GEMM.");
    }

    Client::BenchmarkResults GEMMUniform(CommandPtr                 command,
                                         CommandKernelPtr           commandKernel,
                                         GEMMSolutionPtr            gemm,
                                         ProblemParameters const&   problemParams,
                                         RunParameters const&       runParams,
                                         BenchmarkParameters const& benchmarkParams,
                                         GPUArchitecture const&     arch,
                                         auto                       typeAB,
                                         auto                       typeCD)
    {
        if(typeAB == "float")
        {
            return GEMMUniform<float>(command,
                                      commandKernel,
                                      gemm,
                                      problemParams,
                                      runParams,
                                      benchmarkParams,
                                      arch,
                                      typeCD);
        }
        if(typeAB == "half")
        {
            return GEMMUniform<Half>(command,
                                     commandKernel,
                                     gemm,
                                     problemParams,
                                     runParams,
                                     benchmarkParams,
                                     arch,
                                     typeCD);
        }
        if(typeAB == "bf16")
        {
            return GEMMUniform<BFloat16>(command,
                                         commandKernel,
                                         gemm,
                                         problemParams,
                                         runParams,
                                         benchmarkParams,
                                         arch,
                                         typeCD);
        }
        if(typeAB == "fp8")
        {
            return GEMMUniform<FP8>(command,
                                    commandKernel,
                                    gemm,
                                    problemParams,
                                    runParams,
                                    benchmarkParams,
                                    arch,
                                    typeCD);
        }
        if(typeAB == "bf8")
        {
            return GEMMUniform<BF8>(command,
                                    commandKernel,
                                    gemm,
                                    problemParams,
                                    runParams,
                                    benchmarkParams,
                                    arch,
                                    typeCD);
        }
        if(typeAB == "fp6")
        {
            return GEMMUniform<FP6>(command,
                                    commandKernel,
                                    gemm,
                                    problemParams,
                                    runParams,
                                    benchmarkParams,
                                    arch,
                                    typeCD);
        }
        if(typeAB == "bf6")
        {
            return GEMMUniform<BF6>(command,
                                    commandKernel,
                                    gemm,
                                    problemParams,
                                    runParams,
                                    benchmarkParams,
                                    arch,
                                    typeCD);
        }
        if(typeAB == "fp4")
        {
            return GEMMUniform<FP4>(command,
                                    commandKernel,
                                    gemm,
                                    problemParams,
                                    runParams,
                                    benchmarkParams,
                                    arch,
                                    typeCD);
        }
        Throw<FatalError>("Invalid AB type for uniform GEMM.");
    }

    /*
     * Generate an instance of GEMMSolution and call generateSolution.
     *
     * The kind of GEMMSolution returned is based on the parameters in
     * solutionParams.
     */
    std::pair<GEMMSolutionPtr, int> createGEMMSolution(rocRoller::ContextPtr     context,
                                                       SolutionParameters const& solution)
    {
        GEMMSolutionPtr gemmSolution;

        auto const& arch = context->targetArchitecture().target();

        if(solution.streamK)
        {
            if(context->targetArchitecture().HasCapability(GPUCapability::ArchAccUnifiedRegs))
            {
                gemmSolution = std::make_shared<Client::GEMMClient::StreamKGEMMSolution>(context);
            }
            else
            {
                std::cout << "Not running StreamK for " << arch.toString() << std::endl;
                return {nullptr, ReturnCodes::SolutionNotSupportedOnArch};
            }
        }
        else
        {
            gemmSolution = std::make_shared<Client::GEMMClient::DataParallelGEMMSolution>(context);
        }

        AssertFatal(gemmSolution, "No solution!");

        return {gemmSolution, ReturnCodes::OK};
    }

    struct ArchitectureParameters
    {
        GPUArchitectureTarget target;
    };

    struct IOParameters
    {
        bool        doSaveAsm, doSaveCO;
        std::string saveAsmPath, loadAsmPath;
        std::string saveCOPath, loadCOPath;
        std::string resultsPath, timersPath;
    };

    void writeFile(std::filesystem::path const& filename, std::vector<char> const& x)
    {
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        std::copy(x.cbegin(), x.cend(), std::ostream_iterator<unsigned char>(file));
    }

    int RunGEMMCLI(bool                   doGenerate,
                   bool                   doValidate,
                   bool                   doBenchmark,
                   bool                   doInfo,
                   ArchitectureParameters architecture,
                   SolutionParameters     solution,
                   ProblemParameters      problem,
                   TypeParameters         types,
                   RunParameters          run,
                   BenchmarkParameters    benchmark,
                   IOParameters           io)
    {
        GEMMSolutionPtr  gemm;
        CommandPtr       command;
        CommandKernelPtr commandKernel;

        if(doInfo)
        {
            std::cout << "Loading kernel from: " << io.loadCOPath << std::endl;
            auto yaml = readMetaDataFromCodeObject(io.loadCOPath);
            std::cout << yaml << std::endl;

            auto kernelFromYAML = AssemblyKernels::fromYAML(yaml).kernels[0];
            std::cout << *kernelFromYAML.command() << std::endl;

            return ReturnCodes::OK;
        }

        // Changing settings has to go before creating the context :(
        if(io.doSaveAsm)
        {
            if(io.saveAsmPath.empty())
                io.saveAsmPath = solution.generateKernelName() + ".s";

            Settings::getInstance()->set(Settings::SaveAssembly, true);
            Settings::getInstance()->set(Settings::AssemblyFile, std::string(io.saveAsmPath));
        }

        // When loading from .s or .co, currently need to load
        // solution from a .yaml file.
        if(!io.loadAsmPath.empty())
        {
            auto yamlPath = std::filesystem::path{io.loadAsmPath};
            yamlPath.replace_extension(".yaml");

            solution = Serialization::readYAMLFile<SolutionParameters>(yamlPath);

            architecture.target = solution.architecture;
            if(solution.version != rocRoller::Version::Git())
            {
                std::cout << "Warning: this version of rocRoller (" << rocRoller::Version::Git()
                          << ") differs from the one that generated the kernel." << std::endl;
            }
        }

        auto const arch = GPUArchitectureLibrary::getInstance()->GetArch(architecture.target);

        if(solution.scheduler != "")
        {
            auto schedulerValue = fromString<Scheduling::SchedulerProcedure>(solution.scheduler);
            Settings::getInstance()->set(Settings::Scheduler, schedulerValue);
        }

        if(solution.schedulerCost != "")
        {
            auto cost = fromString<Scheduling::CostFunction>(solution.schedulerCost);
            Settings::getInstance()->set(Settings::SchedulerCost, cost);
        }

        auto context
            = Context::ForTarget(arch,
                                 solution.generateKernelName(),
                                 {{.scaleSkipPermlane = solution.types.scaleSkipPermlane}});

        bool willRunOnGPU = doValidate || doBenchmark;
        if(willRunOnGPU)
        {
            std::cout << "Setting HIP device to " << benchmark.device << std::endl;
            HIP_CHECK(hipSetDevice(benchmark.device));
        }

        if(willRunOnGPU && solution.streamK)
        {
            if(run.numWGs == 0)
            {
                hipDeviceProp_t deviceProperties;
                AssertFatal(hipGetDeviceProperties(&deviceProperties, 0)
                            == (hipError_t)HIP_SUCCESS);
                run.numWGs = deviceProperties.multiProcessorCount;
            }
            AssertFatal(!solution.streamKTwoTile || solution.streamK);
        }

        if(doGenerate)
        {
            std::cout << "Generating for architecture: "
                      << context->targetArchitecture().target().toString() << std::endl;

            std::cout << std::endl;
            std::cout << "Solution:" << std::endl;
            std::cout << solution << std::endl;

            std::cout << "Generating: " << solution.generateKernelName() << "..." << std::endl;

            int reason;
            std::tie(gemm, reason) = createGEMMSolution(context, solution);
            if(!gemm)
                return reason;
            command       = gemm->makeCommand(solution);
            commandKernel = gemm->generateCommandKernel(command, solution);

            std::string basePath;
            if(io.doSaveAsm)
                basePath = io.saveAsmPath;
            if(io.doSaveCO)
                basePath = io.saveCOPath;

            if(io.doSaveAsm || io.doSaveCO)
            {
                // When saveing ASM, also need code-object so that
                // Command (ie, workgroup size, argument mapping etc)
                // can be de-serialized later.

                auto codeObject = commandKernel->assembleKernel();

                std::filesystem::path codeObjectPath{basePath};
                codeObjectPath.replace_extension(".co");

                {
                    // Output RunParameters into a YAML file
                    std::filesystem::path yamlRunParams{basePath};
                    auto                  stem              = yamlRunParams.stem();
                    std::filesystem::path yamlRunParamsPath = stem.string() + "_runParameters.yaml";
                    if(!std::filesystem::exists(yamlRunParamsPath))
                    {
                        std::ofstream file(yamlRunParamsPath);
                        Serialization::writeYAML(file, run);
                        std::cout << "Wrote: " << yamlRunParamsPath.string() << std::endl;
                    }
                }

                std::filesystem::path yamlPath{basePath};
                yamlPath.replace_extension(".yaml");
                if(!std::filesystem::exists(yamlPath))
                {
                    std::ofstream file(yamlPath);
                    Serialization::writeYAML(file, solution);
                    std::cout << "Wrote: " << yamlPath.string() << std::endl;
                }

                writeFile(codeObjectPath, codeObject);
                std::cout << "Wrote: " << codeObjectPath.string() << std::endl;
            }

            if(io.doSaveAsm)
            {
                std::filesystem::path assemblyPath{basePath};
                assemblyPath.replace_extension(".s");

                // We don't explicitly write here (the code-gen does),
                // but we still emit a message here.
                std::cout << "Wrote: " << assemblyPath.string() << std::endl;
            }
        }
        else
        {
            int reason;
            std::tie(gemm, reason) = createGEMMSolution(context, solution);

            command = gemm->makeCommand(solution);

            if(!io.loadAsmPath.empty())
            {
                // When loading ASM, we also load code-object so that
                // Command (ie, workgroup size, argument mapping etc)
                // can be de-serialized.
                std::filesystem::path codeObjectPath{io.loadAsmPath};
                codeObjectPath.replace_extension(".co");

                std::cout << "Loading kernel meta-data from: " << codeObjectPath.string()
                          << std::endl;
                commandKernel = std::make_shared<CommandKernel>();
                commandKernel->setContext(context);
                auto kernel = commandKernel->loadKernelFromCodeObject(
                    codeObjectPath, solution.generateKernelName());
                command = kernel->command();

                std::cout << "Loading kernel from: " << io.loadAsmPath << std::endl;
                commandKernel
                    = std::make_shared<CommandKernel>(command, solution.generateKernelName());
                commandKernel->setContext(context);
                commandKernel->loadKernelFromAssembly(io.loadAsmPath,
                                                      solution.generateKernelName());
            }
            else if(!io.loadCOPath.empty())
            {
                std::cout << "Loading kernel from: " << io.loadCOPath << std::endl;

                commandKernel = std::make_shared<CommandKernel>();
                commandKernel->setContext(context);
                auto kernel = commandKernel->loadKernelFromCodeObject(
                    io.loadCOPath, solution.generateKernelName());

                command = kernel->command();
            }
        }

        if(!gemm || !command || !commandKernel)
            return ReturnCodes::GenerateFailure;

        if(doValidate || doBenchmark)
        {
            std::cout << "Running..." << std::endl;

            if(doValidate)
            {
                benchmark.check     = true;
                benchmark.numWarmUp = 0;
                benchmark.numOuter  = 1;
                benchmark.numInner  = 1;
            }

            auto isF8F6F4 = [](auto dtype) {
                return (dtype == "fp8" || dtype == "bf8" || dtype == "fp6" || dtype == "bf6"
                        || dtype == "fp4");
            };

            Client::GEMMClient::Result result;

            result.problemParams                    = problem;
            result.solutionParams                   = solution;
            result.benchmarkResults.runParams       = run;
            result.benchmarkResults.benchmarkParams = benchmark;

            if(types.typeA == types.typeB && types.typeC == types.typeD)
            {
                result.benchmarkResults = GEMMUniform(command,
                                                      commandKernel,
                                                      gemm,
                                                      problem,
                                                      run,
                                                      benchmark,
                                                      arch,
                                                      types.typeA,
                                                      types.typeC);
            }
            else if((problem.types.typeA != problem.types.typeB) && isF8F6F4(problem.types.typeA)
                    && isF8F6F4(problem.types.typeB))
            {
                result.benchmarkResults = GEMMMixed<float, float>(command,
                                                                  commandKernel,
                                                                  gemm,
                                                                  problem,
                                                                  run,
                                                                  benchmark,
                                                                  arch,
                                                                  problem.types.typeA,
                                                                  problem.types.typeB);
            }
            else
            {
                Throw<FatalError>("Unsupported combination of datatypes for GEMM");
            }

            if(!io.resultsPath.empty())
            {
                std::ofstream file(io.resultsPath);
                Serialization::writeYAML(file, result);
            }

            if(!result.benchmarkResults.correct)
                return ReturnCodes::CorrectnessFailure;
        }

        // Dump timers
        if(!io.timersPath.empty())
        {
            std::ofstream dfile;
            dfile.open(io.timersPath, std::ofstream::out | std::ofstream::trunc);
            dfile << rocRoller::TimerPool::CSV();
            dfile.close();
        }

        return ReturnCodes::OK;
    }

    void overwriteTypesFromSolution(TypeParameters& types, SolutionParameters const& solution)
    {
        if((types.typeA != solution.types.typeA) || (types.typeB != solution.types.typeB)
           || (types.typeC != solution.types.typeC) || (types.typeD != solution.types.typeD)
           || (types.typeAcc != solution.types.typeAcc))
        {
            std::cout << "NOTE: Types have been superceded by solution." << std::endl;
        }
        if((types.transA != solution.types.transA) || (types.transB != solution.types.transB))
        {
            std::cout << "NOTE: Transposes have been superceded by solution." << std::endl;
        }
        if((types.scaleA != solution.types.scaleA) || (types.scaleA != solution.types.scaleB))
        {
            std::cout << "NOTE: MX Scalings have been superceded by solution." << std::endl;
        }
        if(types.scaleBlockSize != solution.types.scaleBlockSize)
        {
            std::cout << "NOTE: MX scale block size has been superceded by solution." << std::endl;
        }

        types = solution.types;
    }
}

namespace rocRoller::Client::GEMMClient::CLI
{

    constexpr bool PARSE_SUCCESS = true;
    constexpr bool PARSE_FAILURE = false;

    static bool ParseMI(const std::string&                                 arg,
                        rocRoller::Client::GEMMClient::SolutionParameters& solution)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        solution.waveB = 1;

        bool fail = false;
        bool isB  = false;
        try
        {
            std::istringstream iss(arg);
            std::string        token;

            iss.exceptions(std::ios_base::eofbit | std::ios_base::failbit | std::ios_base::badbit);
            std::getline(iss, token, 'x');
            solution.waveM = std::stoi(token);
            std::getline(iss, token, 'x');
            solution.waveN = std::stoi(token);

            iss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
            std::getline(iss, token, 'x');
            solution.waveK = std::stoi(token);

            isB = true;
            std::getline(iss, token, 'x');
            solution.waveB = std::stoi(token);
        }
        catch(const std::invalid_argument&)
        {
            if(!isB)
                fail = true;
        }
        catch(const std::ios_base::failure&)
        {
            if(!isB)
                fail = true;
        }

        fail |= (solution.waveM < 1) || (solution.waveN < 1) || (solution.waveK < 1)
                || (solution.waveB < 1);

        if(fail)
        {
            std::cerr << "Invalid format for Matrix Instruction." << std::endl;
            std::cerr << std::endl;
            std::cerr << "The MI argument should be formatted like:" << std::endl;
            std::cerr << std::endl;
            std::cerr << "    --mi=MxNxKxB" << std::endl;
            std::cerr << std::endl;
            std::cerr << "For example: --mi=32x32x2x1" << std::endl;

            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }

    static bool ParseMNK(const std::string& arg, rocRoller::Client::GEMMClient::MNKTuple& mnk)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        bool fail = false;
        try
        {
            std::istringstream iss(arg);
            std::string        token;

            iss.exceptions(std::ios_base::eofbit | std::ios_base::failbit | std::ios_base::badbit);
            std::getline(iss, token, 'x');
            mnk.m = std::stoi(token);
            std::getline(iss, token, 'x');
            mnk.n = std::stoi(token);
            iss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
            std::getline(iss, token, 'x');
            mnk.k = std::stoi(token);
        }
        catch(const std::invalid_argument&)
        {
            fail = true;
        }
        catch(const std::ios_base::failure&)
        {
            fail = true;
        }

        fail |= (mnk.m < 1) || (mnk.n < 1) || (mnk.k < 1);

        if(fail)
        {
            std::cerr << "Invalid format for M/N/K tuple." << std::endl;
            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }

    constexpr auto SolutionParameterArguments = std::make_tuple(
        std::make_pair("--arch", &SolutionParameters::architecture),
        std::make_pair("--mac_m", &SolutionParameters::macM),
        std::make_pair("--mac_n", &SolutionParameters::macN),
        std::make_pair("--mac_k", &SolutionParameters::macK),
        std::make_pair("--wave_m", &SolutionParameters::waveM),
        std::make_pair("--wave_n", &SolutionParameters::waveN),
        std::make_pair("--wave_k", &SolutionParameters::waveK),
        std::make_pair("--wave_b", &SolutionParameters::waveB),
        std::make_pair("--workgroup_size_x", &SolutionParameters::workgroupSizeX),
        std::make_pair("--workgroup_size_y", &SolutionParameters::workgroupSizeY),
        std::make_pair("--workgroupMappingDim", &SolutionParameters::workgroupMappingDim),
        std::make_pair("--workgroupRemapXCC", &SolutionParameters::workgroupRemapXCC),
        std::make_pair("--workgroupRemapXCCValue", &SolutionParameters::workgroupRemapXCCValue),
        std::make_pair("--loadLDSScale_A", &SolutionParameters::loadLDSScaleA),
        std::make_pair("--loadLDSScale_B", &SolutionParameters::loadLDSScaleB),
        std::make_pair("--swizzleScale", &SolutionParameters::swizzleScale),
        std::make_pair("--prefetchScale", &SolutionParameters::prefetchScale),
        std::make_pair("--load_A", &SolutionParameters::loadPathA),
        std::make_pair("--load_B", &SolutionParameters::loadPathB),
        std::make_pair("--storeLDS_D", &SolutionParameters::storeLDSD),
        std::make_pair("--prefetch", &SolutionParameters::prefetch),
        std::make_pair("--prefetchInFlight", &SolutionParameters::prefetchInFlight),
        std::make_pair("--prefetchLDSFactor", &SolutionParameters::prefetchLDSFactor),
        std::make_pair("--prefetchMixMemOps", &SolutionParameters::prefetchMixMemOps),
        std::make_pair("--betaInFMA", &SolutionParameters::betaInFma),
        std::make_pair("--unroll_x", &SolutionParameters::unrollX),
        std::make_pair("--unroll_y", &SolutionParameters::unrollY),
        std::make_pair("--scheduler", &SolutionParameters::scheduler),
        std::make_pair("--schedulerCost", &SolutionParameters::schedulerCost),
        std::make_pair("--matchMemoryAccess", &SolutionParameters::matchMemoryAccess),
        std::make_pair("--streamK", &SolutionParameters::streamK),
        std::make_pair("--streamKTwoTile", &SolutionParameters::streamKTwoTile),
        std::make_pair("--streamKTwoTileDPFirst", &SolutionParameters::streamKTwoTileDPFirst));

    template <typename T, typename U>
    std::string getSolutionParameterArgumentName(U T::*member_ptr)
    {
        std::optional<std::string> found_name;

        std::apply(
            [&](auto&&... args) {
                (([&] {
                     if constexpr(std::is_same_v<decltype(args.second), decltype(member_ptr)>)
                     {
                         if(args.second == member_ptr)
                         {
                             found_name = args.first;
                         }
                     }
                 }()),
                 ...);
            },
            SolutionParameterArguments);

        AssertFatal(found_name, "Internal error: could not find argument name.");

        return found_name.value();
    }

    void updateSolutionFromArguments(rocRoller::Client::GEMMClient::SolutionParameters& solution,
                                     ::CLI::App const&                                  app)
    {
        using SP = rocRoller::Client::GEMMClient::SolutionParameters;
        auto SN  = [](auto x) {
            return rocRoller::Client::GEMMClient::CLI::getSolutionParameterArgumentName(x);
        };

        auto update = [&](const std::string& optionName, auto& value) {
            if(app.get_option(optionName)->count())
                value = app.get_option(optionName)->as<std::decay_t<decltype(value)>>();
        };

        // Architecture

        if(app.get_option(SN(&SP::architecture))->count())
        {
            auto architectureName = app.get_option(SN(&SP::architecture))->as<std::string>();
            solution.architecture = GPUArchitectureTarget::fromString(architectureName);
        }

        // Workgroup tile size

        if(app.get_option("--wgts")->count())
        {
            rocRoller::Client::GEMMClient::MNKTuple mnk{0, 0, 0};
            if(!ParseMNK(app.get_option("--wgts")->as<std::string>(), mnk))
                Throw<FatalError>("Failed to parse WGTS argument.");
            solution.macM = mnk.m;
            solution.macN = mnk.n;
            solution.macK = mnk.k;
        }

        update(SN(&SP::macM), solution.macM);
        update(SN(&SP::macN), solution.macN);
        update(SN(&SP::macK), solution.macK);

        // Matrix instruction

        if(app.get_option("--mi")->count())
        {
            if(!ParseMI(app.get_option("--mi")->as<std::string>(), solution))
                Throw<FatalError>("Failed to parse MI argument.");
        }

        update(SN(&SP::waveM), solution.waveM);
        update(SN(&SP::waveN), solution.waveN);
        update(SN(&SP::waveK), solution.waveK);
        update(SN(&SP::waveB), solution.waveB);

        // Workgroup size

        update(SN(&SP::workgroupSizeX), solution.workgroupSizeX);
        update(SN(&SP::workgroupSizeY), solution.workgroupSizeY);

        // Workgroup mapping

        update(SN(&SP::workgroupMappingDim), solution.workgroupMappingDim);
        update(SN(&SP::workgroupRemapXCC), solution.workgroupRemapXCC);
        update(SN(&SP::workgroupRemapXCCValue), solution.workgroupRemapXCCValue);

        // LDS

        if(app.get_option("--lds")->count())
        {
            auto arg = app.get_option("--lds")->as<std::string>();

            solution.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
            if(arg.find('A') != std::string::npos)
                solution.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;

            solution.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
            if(arg.find('B') != std::string::npos)
                solution.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

            solution.storeLDSD = false;
            if(arg.find('D') != std::string::npos)
                solution.storeLDSD = true;
        }

        update(SN(&SP::loadPathA), solution.loadPathA);
        update(SN(&SP::loadPathB), solution.loadPathB);
        update(SN(&SP::storeLDSD), solution.storeLDSD);

        if(app.get_option("--d2lds")->count())
        {
            auto arg = app.get_option("--d2lds")->as<std::string>();

            solution.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
            if(arg.find('A') != std::string::npos)
                solution.loadPathA = SolutionParams::LoadPath::BufferToLDS;

            solution.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
            if(arg.find('B') != std::string::npos)
                solution.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        }

        update(SN(&SP::loadPathA), solution.loadPathA);
        update(SN(&SP::loadPathB), solution.loadPathB);

        if(app.get_option("--mxlds")->count())
        {
            auto arg = app.get_option("--mxlds")->as<std::string>();

            solution.loadLDSScaleA = false;
            if(arg.find('A') != std::string::npos)
                solution.loadLDSScaleA = true;

            solution.loadLDSScaleB = false;
            if(arg.find('B') != std::string::npos)
                solution.loadLDSScaleB = true;
        }

        update(SN(&SP::loadLDSScaleA), solution.loadLDSScaleA);
        update(SN(&SP::loadLDSScaleB), solution.loadLDSScaleB);

        // Swizzling

        update(SN(&SP::swizzleScale), solution.swizzleScale);
        update(SN(&SP::prefetchScale), solution.prefetchScale);

        // Prefetching

        update(SN(&SP::prefetch), solution.prefetch);
        update(SN(&SP::prefetchInFlight), solution.prefetchInFlight);
        update(SN(&SP::prefetchLDSFactor), solution.prefetchLDSFactor);
        update(SN(&SP::prefetchMixMemOps), solution.prefetchMixMemOps);

        // StreamK

        update(SN(&SP::streamK), solution.streamK);
        update(SN(&SP::streamKTwoTile), solution.streamKTwoTile);
        update(SN(&SP::streamKTwoTileDPFirst), solution.streamKTwoTileDPFirst);

        // Other

        update(SN(&SP::betaInFma), solution.betaInFma);
        update(SN(&SP::unrollX), solution.unrollX);
        update(SN(&SP::unrollY), solution.unrollY);
        update(SN(&SP::scheduler), solution.scheduler);
        update(SN(&SP::schedulerCost), solution.schedulerCost);
        update(SN(&SP::matchMemoryAccess), solution.matchMemoryAccess);
    }
}

/*
 * Parse the command line and dispatch.
 */
int main(int argc, const char* argv[])
{
    using namespace rocRoller::Client::GEMMClient::CLI;

    CLI::App app{"GEMM Driver: D (MxN) = alpha * A (MxK) * B (KxN) + beta * C (MxN)"};
    app.footer(Settings::getInstance()->help());

    //
    // Parameters
    //
    std::string                                           architectureName;
    rocRoller::Client::GEMMClient::ArchitectureParameters architecture;

    rocRoller::Client::GEMMClient::SolutionParameters solution{
        .macM = 64,
        .macN = 64,
        .macK = -1,

        .waveM = -1,
        .waveN = -1,
        .waveK = -1,
        .waveB = -1,

        .workgroupSizeX         = -1,
        .workgroupSizeY         = 2,
        .workgroupMappingDim    = -1,
        .workgroupRemapXCC      = false,
        .workgroupRemapXCCValue = -1,

        .types = {.scaleA     = Operations::ScaleMode::None,
                  .scaleTypeA = DataType::None,
                  .scaleB     = Operations::ScaleMode::None,
                  .scaleTypeB = DataType::None,

                  .scaleBlockSize = -1},

        .loadLDSScaleA = false,
        .loadLDSScaleB = false,

        .swizzleScale  = false,
        .prefetchScale = false,

        .loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR,
        .loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR,
        .storeLDSD = true,

        .prefetch          = false,
        .prefetchInFlight  = 0,
        .prefetchLDSFactor = 0,
        .prefetchMixMemOps = false,

        .betaInFma = true,

        .unrollX = 0,
        .unrollY = 0,

        .scheduler         = "Priority",
        .matchMemoryAccess = true,

        .streamK               = false,
        .streamKTwoTile        = false,
        .streamKTwoTileDPFirst = false,

        .version = rocRoller::Version::Git(),
    };

    rocRoller::Client::GEMMClient::ProblemParameters problem{
        .m = 3072,
        .n = 4096,
        .k = 4096,

        .alpha = 2.0f,
        .beta  = 0.5f,

        .scaleValueA = 1.0f,
        .scaleValueB = 1.0f,
    };

    rocRoller::Client::GEMMClient::TypeParameters types;

    rocRoller::Client::RunParameters runParams{
        .workgroupMappingValue = -1,
        .numWGs                = 0,
    };

    rocRoller::Client::BenchmarkParameters benchmarkParams{
        .device           = 0,
        .numWarmUp        = 3,
        .numOuter         = 5,
        .numInner         = 2,
        .rotatingBuffSize = 32'000'000ull,
        .check            = true,
        .visualize        = false,
    };

    rocRoller::Client::GEMMClient::IOParameters io{
        .doSaveAsm   = false,
        .doSaveCO    = false,
        .saveAsmPath = "",
        .loadAsmPath = "",
        .saveCOPath  = "",
        .loadCOPath  = "",
        .resultsPath = "",
    };

    //
    // Architecture
    //
    app.option_defaults()->ignore_case()->group("Architecture parameters");
    app.add_option("--arch", architectureName, "GPU architecture name (eg, gfx90a).");

    //
    // Problem definition
    //
    app.option_defaults()->ignore_case()->group("Problem parameters");
    app.add_option("-M,--M", problem.m, "Tensor size M.");
    app.add_option("-N,--N", problem.n, "Tensor size N.");
    app.add_option("-K,--K", problem.k, "Tensor size K.");
    app.add_option("--alpha", problem.alpha, "Alpha scalar.");
    app.add_option("--beta", problem.beta, "Beta scalar.");
    app.add_option("--scaleValue_A", problem.scaleValueA, "Single scale value for A.");
    app.add_option("--scaleValue_B", problem.scaleValueB, "Single scale value for B.");

    //
    // Problem types
    //
    app.option_defaults()->ignore_case()->group("Type parameters");
    app.add_option("--type_A",
                   types.typeA,
                   "Datatype of A matrix [float | half | bf16 | fp8 | bf8 | fp6 | bf6 | fp4].  "
                   "Default: float.");
    app.add_option("--type_B",
                   types.typeB,
                   "Datatype of B matrix [float | half | bf16 | fp8 | bf8 | fp6 | bf6 | fp4].  "
                   "Default: float.");
    app.add_option(
        "--type_C", types.typeC, "Datatype of C matrix [float | half | bf16].  Default: float.");
    app.add_option(
        "--type_D", types.typeD, "Datatype of D matrix [float | half | bf16].  Default: float.");
    app.add_option("--type_acc",
                   types.typeAcc,
                   "Datatype of accumulation [float | half | bf16].  Default: float");
    app.add_option(
        "--trans_A",
        [&types](auto res) -> bool {
            types.transA = fromString<Client::GEMMClient::TransposeType>(res[0]);
            return true;
        },
        "N: A is not to be transposed.  T: A is to be transposed.",
        "N");
    app.add_option(
        "--trans_B",
        [&types](auto res) -> bool {
            types.transB = fromString<Client::GEMMClient::TransposeType>(res[0]);
            return true;
        },
        "N: B is not to be transposed.  T: B is to be transposed.",
        "N");
    app.add_option(
        "--scale_A",
        [&types](auto res) -> bool {
            types.scaleA = fromString<Operations::ScaleMode>(res[0]);
            return true;
        },
        "Enable MX scaling of A matrix [None | Separate | SingleScale].",
        "Default: None.");
    app.add_option(
        "--scaleType_A",
        [&types](auto res) -> bool {
            types.scaleTypeA = fromString<DataType>(res[0]);
            return true;
        },
        "Type for A matrix scales [None | E8M0].",
        "Default: None.");
    app.add_option(
        "--scale_B",
        [&types](auto res) -> bool {
            types.scaleB = fromString<Operations::ScaleMode>(res[0]);
            return true;
        },
        "Enable MX scaling of B matrix [None | Separate | SingleScale].",
        "Default: None.");
    app.add_option(
        "--scaleType_B",
        [&types](auto res) -> bool {
            types.scaleTypeB = fromString<DataType>(res[0]);
            return true;
        },
        "Type for B matrix scales [None | E8M0].",
        "Default: None.");
    app.add_option("--scaleBlockSize",
                   types.scaleBlockSize,
                   "Set MX scaling block size for A and B. (default: 32)");
    app.add_option("--scaleSkipPermlane",
                   types.scaleSkipPermlane,
                   "Experimental: Skip Permlane instructions for scale data for performance.");

    //
    // Solution parameters
    //
    app.option_defaults()->ignore_case()->group("Solution parameters");

    using SP = rocRoller::Client::GEMMClient::SolutionParameters;
    auto SN  = [](auto x) {
        return rocRoller::Client::GEMMClient::CLI::getSolutionParameterArgumentName(x);
    };

    app.add_option(SN(&SP::macM), "(Macro) Tile size M.");
    app.add_option(SN(&SP::macN), "(Macro) Tile size N.");
    app.add_option(SN(&SP::macK), "(Macro) Tile size K.");
    app.add_option("--wgts", "Workgroup tile size (m/n/k tuple).");

    app.add_option(SN(&SP::waveM), "(MI) Tile size M.");
    app.add_option(SN(&SP::waveN), "(MI) Tile size N.");
    app.add_option(SN(&SP::waveK), "(MI) Tile size K.");
    app.add_option(SN(&SP::waveB), "(MI) Tile size B.");
    app.add_option("--mi", "MI (matrix instruction) to use.");

    app.add_option(SN(&SP::workgroupSizeX), "Workgroup size in the x dimension.");
    app.add_option(SN(&SP::workgroupSizeY), "Workgroup size in the y dimension.");

    app.add_option(SN(&SP::workgroupMappingDim),
                   "Workgroup mapping dimension (-1, 0, 1). Default: -1")
        ->check(CLI::IsMember({-1, 0, 1}));
    app.add_flag(SN(&SP::workgroupRemapXCC), "Use an XCC-aware workgroup remapping.");
    app.add_option(SN(&SP::workgroupRemapXCCValue),
                   "Force an XCC-aware workgroup remapping value. (Optional)");
    app.add_option(SN(&SP::unrollX), "Unroll size in X.");
    app.add_option(SN(&SP::unrollY), "Unroll size in Y.");

    app.add_option(
        SN(&SP::loadPathA),
        solution.loadPathA,
        "How to load A (BufferToVGPR, BufferToLDSViaVGPR, BufferToLDS). Default: BufferToLDS");
    app.add_option(
        SN(&SP::loadPathB),
        solution.loadPathB,
        "How to load A (BufferToVGPR, BufferToLDSViaVGPR, BufferToLDS). Default: BufferToLDS");
    app.add_flag(SN(&SP::storeLDSD), "Use LDS when storing D.");
    app.add_option("--lds", "Use LDS for A/B/D.");
    app.add_option("--d2lds", "Use direct-to-LDS for A/B.");

    app.add_flag(SN(&SP::betaInFma), "Use beta in FMA instruction instead of alpha.");
    app.add_option(SN(&SP::scheduler), "Which scheduler to use.");
    app.add_option(SN(&SP::schedulerCost), "Which scheduler cost function to use.");

    app.add_flag(SN(&SP::matchMemoryAccess),
                 "Match memory access to transpose.  Currently decreases performance.");
    app.add_flag(SN(&SP::prefetch), "Enable prefetching (UnrollK=2 implied).");
    app.add_option(SN(&SP::prefetchInFlight), "Number of prefetches in flight at the same time");
    app.add_option(SN(&SP::prefetchLDSFactor),
                   "Prefetch 1/prefetchLDSFactor of MacroTile from LDS");
    app.add_flag(SN(&SP::prefetchMixMemOps),
                 "Mix global and LDS memory operations during prefetching.");
    app.add_flag(SN(&SP::streamK), "Enable StreamK algorithm.");
    app.add_flag(SN(&SP::streamKTwoTile), "Enable two-tile StreamK algorithm.");
    app.add_flag(SN(&SP::streamKTwoTileDPFirst),
                 "Execute data-parallel loop first in the two-tile StreamK algorithm.");

    app.add_flag(SN(&SP::loadLDSScaleA), "Use LDS when loading A scale.");
    app.add_flag(SN(&SP::loadLDSScaleB), "Use LDS when loading B scale.");
    app.add_option("--mxlds", "Use LDS for A/B scales.");

    app.add_flag(SN(&SP::swizzleScale), "Use Swizzle when loading A and B scale.");
    app.add_flag(SN(&SP::prefetchScale), "Prefetch scale values with using Swizzled scales.");

    app.add_option("--workgroupMappingValue",
                   runParams.workgroupMappingValue,
                   "Workgroup mapping value. Default: -1")
        ->check(CLI::IsMember({-1}) | CLI::PositiveNumber);

    //
    // Benchmarking options
    //
    app.option_defaults()->ignore_case()->group("Benchmarking parameters");
    app.add_option("--num_warmup", benchmarkParams.numWarmUp, "Number of warm-up runs.");
    app.add_option("--num_outer", benchmarkParams.numOuter, "Number of outer runs.");
    app.add_option("--num_inner", benchmarkParams.numInner, "Number of inner runs.");
    app.add_option("--device", benchmarkParams.device, "GPU device ordinal");
    app.add_option("--numWGs",
                   runParams.numWGs,
                   "Number of workgroups to use with StreamK algorithm.  Defaults to number of WGs "
                   "present on local device.");
    app.add_option(
        "--rotating_buff_size", benchmarkParams.rotatingBuffSize, "Rotating Buffer Size.");

    //
    // Client params and shortcuts
    //

    app.option_defaults()->ignore_case()->group("Client options and shortcuts");

    bool noCheckResult = false;

    std::string loadPath, examplePath;

    app.add_flag(
        "--hgemm",
        [&types](auto res) -> bool {
            types.typeA   = "half";
            types.typeB   = "half";
            types.typeC   = "half";
            types.typeD   = "half";
            types.typeAcc = "float";
            return true;
        },
        "Overwrite types to: --type_A=half --type_B=half --type_C=half --type_D=half "
        "--type_acc=float.");

    app.add_flag("--visualize",
                 benchmarkParams.visualize,
                 "Dump out volumes describing memory access patterns.");

    app.add_flag("--noCheck", noCheckResult, "Do not verify GEMM results against OpenBLAS.");

    app.add_option("--yaml", io.resultsPath, "Save results to file.");

    app.add_option("--timers", io.timersPath, "Save timers to CSV file.");

    //
    // generate sub-command
    //

    std::string loadConfigPath;

    auto generate = app.add_subcommand("generate", "Generate a GEMM solution.")->fallthrough();

    auto asmOption = generate->add_option("--asm", io.saveAsmPath, "Save yaml+assembly to files.")
                         ->expected(0, 1);
    auto coOption
        = generate->add_option("--co", io.saveCOPath, "Save code-object to file.")->expected(0, 1);
    generate
        ->add_option(
            "--config", loadConfigPath, "Load solution generation parameters from YAML file.")
        ->expected(1, 1);

    //
    // validate sub-command
    //

    auto validate
        = app.add_subcommand("validate",
                             "Run and validate a GEMM solution (only runs the solution once).")
              ->fallthrough();

    validate->add_option(
        "--load", loadPath, "Load solution from code-object (.co) or assembly (.s) file.");

    //
    // benchmark sub-command
    //

    auto benchmark
        = app.add_subcommand("benchmark",
                             "Benchmark a GEMM solution (may run the solution many times).")
              ->fallthrough();

    benchmark->add_option(
        "--load", loadPath, "Load solution from code-object (.co) or assembly (.s) file.");

    std::string loadRunParamsPath;
    benchmark->add_option(
        "--loadRunParams", loadRunParamsPath, "Load run parameters from YAML file.");

    //
    // info sub-command
    //

    auto info = app.add_subcommand("info", "Dump info about a GEMM solution.")->fallthrough();

    info->add_option("load", loadPath, "Load solution from code-object (.co).")->required();

    //
    // example sub-command
    //
    auto example = app.add_subcommand("example", "Save example generation parameters to YAML file.")
                       ->fallthrough();

    example->add_option("save", examplePath, "Example config path.")->required();

    //
    // Parse and update/validate problem definition
    //

    CLI11_PARSE(app, argc, argv);

    updateSolutionFromArguments(solution, app);

    if(architectureName.empty())
        architecture.target = GPUArchitectureLibrary::getInstance()
                                  ->GetDefaultHipDeviceArch(benchmarkParams.device)
                                  .target();
    else
        architecture.target = GPUArchitectureTarget::fromString(architectureName);

    solution.architecture = architecture.target;

    if(!loadConfigPath.empty())
    {
        solution = Serialization::readYAMLFile<rocRoller::Client::GEMMClient::SolutionParameters>(
            loadConfigPath);

        updateSolutionFromArguments(solution, app);
        overwriteTypesFromSolution(types, solution);

        if(solution.architecture.gfx == GPUArchitectureGFX::UNKNOWN)
            solution.architecture = architecture.target;
    }

    if(!loadPath.empty())
    {
        auto path = std::filesystem::path(loadPath);
        if(path.extension() == ".s" || path.extension() == ".yaml")
        {
            io.loadAsmPath = path.string();
        }
        else if(path.extension() == ".co")
        {
            io.loadCOPath = path.string();
        }
        else
        {
            Throw<FatalError>("Extension not supported.  Can not load solution from ", loadPath);
        }
    }

    if(!loadRunParamsPath.empty())
    {
        auto path = std::filesystem::path(loadRunParamsPath);
        path.replace_extension(".yaml");

        // Load RunParameters from a specified YAML file
        runParams = Serialization::readYAMLFile<rocRoller::Client::RunParameters>(path);
    }

    if(!io.loadAsmPath.empty() || !io.loadCOPath.empty())
    {
        std::filesystem::path yamlPath;
        if(!io.loadAsmPath.empty())
            yamlPath = std::filesystem::path{io.loadAsmPath};
        if(!io.loadCOPath.empty())
            yamlPath = std::filesystem::path{io.loadCOPath};
        yamlPath.replace_extension(".yaml");

        // YAML file does not have the workgroupMappingValue used to generate the kernel.
        // Instead, the workgroupMappingValue specified by users in benchmarking will be used.
        solution = Serialization::readYAMLFile<rocRoller::Client::GEMMClient::SolutionParameters>(
            yamlPath);

        overwriteTypesFromSolution(types, solution);
    }

    if(types.scaleA != Operations::ScaleMode::None && types.scaleB == Operations::ScaleMode::None)
    {
        types.scaleB        = Operations::ScaleMode::SingleScale;
        problem.scaleValueB = 1.0f;
        types.scaleTypeB    = types.scaleTypeA;
    }

    if(types.scaleB != Operations::ScaleMode::None && types.scaleA == Operations::ScaleMode::None)
    {
        types.scaleA        = Operations::ScaleMode::SingleScale;
        problem.scaleValueA = 1.0f;
        types.scaleTypeA    = types.scaleTypeB;
    }

    auto const& arch = GPUArchitectureLibrary::getInstance()->GetArch(solution.architecture);
    if(types.scaleBlockSize == -1
       && (types.scaleA == Operations::ScaleMode::Separate
           || types.scaleB == Operations::ScaleMode::Separate))
    {
        AssertFatal(arch.HasCapability(GPUCapability::HasBlockScaling32),
                    fmt::format("Architecture {} does not support block scaling.",
                                arch.target().toString()));
        types.scaleBlockSize = arch.GetCapability(GPUCapability::DefaultScaleBlockSize);
    }

    if(solution.workgroupSizeX == -1)
        solution.workgroupSizeX = 2 * arch.GetCapability(GPUCapability::DefaultWavefrontSize);
    if(solution.workgroupSizeY == -1)
        solution.workgroupSizeY = 2;

    const DataType typeA   = fromString<DataType>(types.typeA);
    const DataType typeB   = fromString<DataType>(types.typeB);
    const DataType typeC   = fromString<DataType>(types.typeC);
    const DataType typeD   = fromString<DataType>(types.typeD);
    const DataType typeAcc = fromString<DataType>(types.typeAcc);

    AssertFatal((typeAcc == DataType::Float) || (typeAcc == DataType::Half)
                || (typeAcc == DataType::BFloat16));

    // TODO: Reevaluate the relationship between problem and solution params.
    problem.workgroupMappingDim = solution.workgroupMappingDim;

    benchmarkParams.check = !noCheckResult;

    io.doSaveAsm = asmOption->count() > 0;
    io.doSaveCO  = coOption->count() > 0;

    // Set default MI and macK sizes
    if(arch.HasCapability(GPUCapability::HasMFMA))
    {
        if(solution.macK == -1)
            solution.macK = 64;
        if(solution.waveB == -1)
            solution.waveB = 1;

        if(typeA == DataType::Float && typeB == DataType::Float && typeC == DataType::Float
           && typeD == DataType::Float)
        {
            if(solution.waveM == -1)
                solution.waveM = 32;
            if(solution.waveN == -1)
                solution.waveN = 32;
            if(solution.waveK == -1)
                solution.waveK = 2;
        }
        else if(typeA == DataType::Half && typeB == DataType::Half)
        {
            if(solution.waveM == -1)
                solution.waveM = 32;
            if(solution.waveN == -1)
                solution.waveN = 32;
            if(solution.waveK == -1)
                solution.waveK = 8;
        }
        else if(typeA == DataType::BFloat16 && typeB == DataType::BFloat16)
        {
            if(solution.waveM == -1)
                solution.waveM = 16;
            if(solution.waveN == -1)
                solution.waveN = 16;
            if(solution.waveK == -1)
                solution.waveK = 8;
        }
        else if((typeA == DataType::FP8 && typeB == DataType::FP8)
                || (typeA == DataType::BF8 && typeB == DataType::BF8))
        {
            if(solution.waveM == -1)
                solution.waveM = 16;
            if(solution.waveN == -1)
                solution.waveN = 16;
            if(solution.waveK == -1)
                solution.waveK = 32;
        }
    }
    else if(arch.HasCapability(GPUCapability::HasWMMA))
    {
        if(solution.waveM == -1)
            solution.waveM = 16;
        if(solution.waveN == -1)
            solution.waveN = 16;
        if(solution.waveB == -1)
            solution.waveB = 1;

        if((typeA == DataType::Half && typeB == DataType::Half)
           || (typeA == DataType::BFloat16 && typeB == DataType::BFloat16))
        {
            if(solution.macK == -1)
                solution.macK = 64;

            if(arch.HasCapability(GPUCapability::HasWMMA_f32_16x16x16_f16))
            {
                if(solution.waveK == -1)
                    solution.waveK = 16;
            }
        }
        else if(isUnpackedF8(fromString<DataType>(solution.types.typeA))
                && isUnpackedF8(fromString<DataType>(solution.types.typeB)))
        {
            if(solution.macK == -1)
                solution.macK = 64;

            if(arch.HasCapability(GPUCapability::HasWMMA_f32_16x16x16_f8))
            {
                if(solution.waveK == -1)
                    solution.waveK = 16;
            }
        }
    }
    else
    {
        Throw<FatalError>("Unsupported arch for GEMM client: ", arch.target().toString());
    }

    if(arch.target().isRDNA4GPU())
    {
        // Override default settings for the `example` and `generate` subcommands.
        if((example->parsed() || generate->parsed()) && typeA == DataType::Float
           && typeB == DataType::Float)
        {
            std::cout << "Warning: A and B types and wave sizes have been overridden for RDNA4."
                      << std::endl;
            types.typeA    = "half";
            types.typeB    = "half";
            types.typeC    = "half";
            types.typeD    = "half";
            solution.waveM = 16;
            solution.waveN = 16;
            solution.waveK = 16;
            solution.waveB = 1;
            solution.macK  = 64;
        }

        if(solution.prefetch)
        {
            std::cout << "Warning: disabling prefetching for RDNA4." << std::endl;
            solution.prefetch = false;
        }
    }

    AssertFatal(solution.waveM > 0 && solution.waveN > 0 && solution.waveK > 0
                    && solution.waveB > 0,
                fmt::format("MI tile sizes must be set greater than zero. "
                            "waveM: {} waveN: {} waveK: {} waveB: {}",
                            solution.waveM,
                            solution.waveN,
                            solution.waveK,
                            solution.waveB));

    AssertFatal(solution.macK >= solution.waveK && solution.macM >= solution.waveM
                    && solution.macN >= solution.waveN,
                fmt::format("Macro tile sizes must be greater than or equal to MI tile sizes. "
                            "macM: {} waveM: {} macN: {} waveN: {} macK: {} waveK: {}",
                            solution.macM,
                            solution.waveM,
                            solution.macN,
                            solution.waveN,
                            solution.macK,
                            solution.waveK));

    if(types.scaleSkipPermlane)
    {
        AssertFatal(types.transA == Client::GEMMClient::TransposeType::T, ShowValue(types));
        AssertFatal(types.scaleA == Operations::ScaleMode::Separate, ShowValue(types));

        size_t kSubtile = solution.waveK / types.scaleBlockSize;

        AssertFatal(kSubtile == 2 || kSubtile == 4,
                    ShowValue(kSubtile),
                    ShowValue(solution.waveK),
                    ShowValue(types.scaleBlockSize));

        types.scaleShuffleTileA = {64, 4, kSubtile};
    }

    if(types.scaleSkipPermlane)
    {
        AssertFatal(types.transB == Client::GEMMClient::TransposeType::N, ShowValue(types));
        AssertFatal(types.scaleB == Operations::ScaleMode::Separate, ShowValue(types));

        size_t kSubtile = solution.waveK / types.scaleBlockSize;

        AssertFatal(kSubtile == 2 || kSubtile == 4,
                    ShowValue(kSubtile),
                    ShowValue(solution.waveK),
                    ShowValue(types.scaleBlockSize));

        types.scaleShuffleTileB = {64, 4, kSubtile};
    }

    problem.types  = types;
    solution.types = types;

    // Set default prefetchMixMemOps
    if(app.get_option("--prefetchMixMemOps")->count() == 0)
    {
        solution.prefetchMixMemOps = false;
        if(solution.prefetchLDSFactor != 0)
            solution.prefetchMixMemOps = true;

        if(types.scaleB == Operations::ScaleMode::Separate && !solution.loadLDSScaleB)
            solution.prefetchMixMemOps = false;

        if(types.scaleA == Operations::ScaleMode::Separate && !solution.loadLDSScaleA)
            solution.prefetchMixMemOps = false;

        // TODO: enable (prefetchMixMemOps == true && prefetchLDSFactor == 2 && direct2LDSA/B = true)
        if(solution.prefetchLDSFactor == 2
           && (IsBufferToLDS(solution.loadPathA) || IsBufferToLDS(solution.loadPathB)))
            solution.prefetchMixMemOps = false;
    }

    //
    // Run!
    //
    if(example->parsed())
    {
        std::ofstream file(examplePath);
        Serialization::writeYAML(file, solution);
        return 0;
    }

    // If no subcommands were given, default behaviour is: generate
    // and benchmark.
    bool generateAndBenchmark
        = !generate->parsed() && !validate->parsed() && !benchmark->parsed() && !info->parsed();

    bool doGenerate  = generateAndBenchmark || generate->parsed();
    bool doValidate  = validate->parsed();
    bool doBenchmark = generateAndBenchmark || benchmark->parsed();
    bool doInfo      = info->parsed();

    return rocRoller::Client::GEMMClient::RunGEMMCLI(doGenerate,
                                                     doValidate,
                                                     doBenchmark,
                                                     doInfo,
                                                     architecture,
                                                     solution,
                                                     problem,
                                                     types,
                                                     runParams,
                                                     benchmarkParams,
                                                     io);
}
